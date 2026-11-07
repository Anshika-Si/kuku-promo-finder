"""Candidate generation + rule scoring + ranking.

Design rule that matters most: timestamps are ONLY ever produced here, by code, from the script.
The optional LLM judge (judge.py) sees candidate IDs and text and may re-score or reword them,
but it can never invent or move a cut point. That removes the classic "hallucinated timecode" failure.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict

from .parser import Episode, Beat, fmt
from .signals import BeatSignal, tag_episode, load_lexicon, dominant_trope, TROPE_LABELS

# ----------------------------------------------------------------------------- config
DEFAULTS = {
    "min_len": 20.0, "max_len": 60.0,
    "tiktok_min": 21.0, "tiktok_max": 34.0,           # TikTok in-feed guidance (21-34s)
    "weights": {"hook": 0.30, "cliff": 0.25, "clarity": 0.20, "escalation": 0.15, "trope": 0.10},
    "starts_per_episode": 24, "ends_per_episode": 24,
    "early_episodes": 12,                             # assumed first paywall zone (ep 8-16 in 80-100 ep shows, scaled)
    "funnel_floor": 0.80,
    # generic "payoff" phrases (the central mystery is resolved -> better for retargeting than cold ads);
    # show-specific ones go in a per-show config file (--show-config), never in code
    "payoff_terms": ["it was always me", "this is my son", "lawful wife", "true heir", "true identity", "here i stand"],
    "speaker_aliases": {},
    "context_terms": ["the last time", "as i told you", "remember when", "like before", "again?"],
    "top_n": 25,
}


@dataclass
class Segment:
    episode: int
    start: float
    end: float

    @property
    def dur(self):
        return self.end - self.start

    def to_dict(self):
        return {"episode": self.episode, "start": fmt(self.start), "end": fmt(self.end),
                "start_s": round(self.start, 2), "end_s": round(self.end, 2), "duration_s": round(self.dur, 2)}


@dataclass
class Promo:
    pid: str
    kind: str                       # continuous | stitched
    segments: list[Segment]
    beats: list[BeatSignal] = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    final: float = 0.0
    hook: dict = field(default_factory=dict)
    cliff: dict = field(default_factory=dict)
    trope: str = ""
    lane: str = "cold_ua"
    flags: list = field(default_factory=list)
    why: str = ""
    title: str = ""
    tiktok_cut: list[Segment] | None = None
    llm: dict | None = None
    cut_end: dict = field(default_factory=dict)
    priority: float = 0.0

    @property
    def duration(self):
        return sum(s.dur for s in self.segments)

    @property
    def episodes(self):
        return sorted({s.episode for s in self.segments})


# ----------------------------------------------------------------------------- helpers
def name_case(s: str) -> str:
    return " ".join(w[:1].upper() + w[1:].lower() for w in s.split())


def _inside_dialogue(t: float, dlg: list[Beat], tol=0.05) -> bool:
    return any(d.start + tol < t < d.end - tol for d in dlg)


def speech_spans(ep: Episode) -> list[Beat]:
    """Dialogue blocks, with sentences the script splits across blocks ('I cannot...' / '...not you')
    merged into one unbreakable span."""
    dlg = sorted((b for b in ep.beats if b.kind == "dialogue"), key=lambda b: b.start)
    spans: list[Beat] = []
    for b in dlg:
        cont = b.text.lstrip().startswith(("...", "\u2026"))
        if spans and cont and spans[-1].text.rstrip().endswith(("...", "\u2026", "-", "\u2014")) and b.start - spans[-1].end < 3.0:
            last = spans[-1]
            spans[-1] = Beat(last.episode, last.idx, last.scene, last.location, "dialogue", last.speaker, last.tags,
                             last.text + " " + b.text, last.start, max(last.end, b.end))
        else:
            spans.append(b)
    return spans


def cut_points(ep: Episode):
    """Legal in/out points: shot or beat boundaries that do not slice through a spoken line or split sentence."""
    dlg = speech_spans(ep)
    starts = sorted({round(b.start, 2) for b in ep.beats if not _inside_dialogue(b.start, dlg)})
    ends = sorted({round(b.end, 2) for b in ep.beats if not _inside_dialogue(b.end, dlg)} | {round(ep.duration, 2)})
    return starts, ends


def _sat(x: float, k: float = 1.2) -> float:
    return 1.0 - math.exp(-max(x, 0.0) / k)


def window_signals(sigs: list[BeatSignal], a: float, b: float) -> list[BeatSignal]:
    return [s for s in sigs if s.beat.start >= a - 0.05 and s.beat.end <= b + 0.05]


# ----------------------------------------------------------------------------- component scores
def hook_score(sigs_in: list[BeatSignal], a: float):
    """Strength of what the viewer sees/hears in the first 3 seconds."""
    first = [s for s in sigs_in if s.beat.start < a + 3.0]
    if not first:
        return 0.0, None, {}
    raw, best = 0.0, None
    for s in first:
        w = 1.0 - 0.1 * max(0.0, s.beat.start - a)          # earlier is better
        v = s.intensity * w
        if s.beat.kind == "dialogue" and s.beat.end <= a + 4.0:
            v += 0.25                                        # a complete line lands inside the swipe zone
        if v > raw:
            raw, best = v, s
    visual = any(s.visual and s.intensity > 0.5 for s in first)
    raw += 0.25 if visual else 0.0
    est = first[0].establishing or all(s.beat.kind == "camera" for s in first)
    raw -= 0.5 if est else 0.0
    # a hot line inside 3-6s still rescues a soft first frame a little (the "second spike")
    second = [s.intensity for s in sigs_in if a + 3.0 <= s.beat.start < a + 6.0]
    raw += 0.15 * max(second, default=0)
    return _sat(raw, 1.1), best, {"sound_off_readable": visual, "establishing_open": est}


def cliff_score(sigs_in: list[BeatSignal], b: float, ep_end: float, after: list[BeatSignal]):
    last = [s for s in sigs_in if s.beat.end > b - 4.5]
    if not last:
        return 0.0, None, "none"
    raw = max(s.unresolved + 0.35 * s.intensity for s in last)
    best = max(last, key=lambda s: s.unresolved + 0.35 * s.intensity)
    kind = "unresolved moment"
    if any(s.question for s in last if s.beat.kind == "dialogue"):
        kind = "an unanswered question"
    if any(s.threat for s in last):
        kind = "a life-or-death threat"
    if any(s.beat.kind in ("action", "camera") and s.unresolved >= 0.8 for s in last):
        kind = "mid-action"
    if b >= ep_end - 1.0:
        raw += 0.6
        kind = "the episode's own cliffhanger" if kind == "unresolved moment" else kind + " (episode cliffhanger)"
    resolution = sum(s.resolution for s in sigs_in if s.beat.end > b - 7.0)
    raw -= min(resolution, 1.2)
    # withheld answer: something important happens right after the cut -> curiosity gap
    nxt = [s for s in after if s.beat.start < b + 6.0]
    if nxt and max(s.intensity for s in nxt) > 0.8:
        raw += 0.25
    return _sat(raw, 1.1), best, kind


def clarity_score(sigs_in: list[BeatSignal], cast_names: set[str], protagonists: set[str], cfg) -> tuple[float, dict]:
    speakers = {s.beat.speaker for s in sigs_in if s.beat.kind == "dialogue"}
    n = len(speakers)
    sp = 1.0 if 2 <= n <= 3 else 0.8 if n in (1, 4) else 0.55 if n >= 5 else 0.4
    present = set()
    for spk in speakers:
        present |= {w.title() for w in re.findall(r"[A-Z']+", spk)}
    for s in sigs_in:
        if s.beat.kind in ("action", "camera"):
            present |= {w for w in re.findall(r"[A-Z][a-z']+", s.beat.text)}
    dangling = set()
    text = " ".join(s.beat.text for s in sigs_in if s.beat.kind == "dialogue")
    for w in re.findall(r"\b[A-Z][a-z]+\b", text):
        if w in cast_names and w not in present and w not in protagonists:
            dangling.add(w)
    early = [s for s in sigs_in if s.beat.start < sigs_in[0].beat.start + 15]
    anchors = sum(s.anchors for s in early)
    ctx = sum(1 for t in cfg["context_terms"] if t in text.lower())
    raw = sp - 0.12 * len(dangling) + 0.06 * min(anchors, 4) - 0.1 * ctx
    return max(0.0, min(1.0, raw)), {"speakers": n, "dangling_names": sorted(dangling), "anchors": anchors}


def escalation_score(sigs_in: list[BeatSignal], dur: float, hot_thr: float = 0.65) -> float:
    """hot_thr is show-relative (80th percentile of beat intensity) so the tool adapts to each writer's style."""
    if not sigs_in:
        return 0.0
    hot = [s for s in sigs_in if s.intensity >= hot_thr]
    density = len(hot) / max(dur / 10.0, 1.0)          # hot beats per 10 seconds
    half = sigs_in[0].beat.start + dur / 2
    first = [s.intensity for s in sigs_in if s.beat.start < half] or [0]
    second = [s.intensity for s in sigs_in if s.beat.start >= half] or [0]
    trend = (sum(second) / len(second)) - (sum(first) / len(first))
    peak_late = max(second) >= 0.85 * max(first + second)
    return max(0.0, min(1.0, 0.15 * density + 0.25 * max(trend, -0.5) + (0.25 if peak_late else 0)))


def trope_score(sigs_in: list[BeatSignal]) -> tuple[float, str]:
    t = dominant_trope(sigs_in)
    mass = sum(min(s.tropes.get(t, 0), 2) for s in sigs_in) if t in TROPE_LABELS else 0
    return _sat(mass, 3.0), t


def dead_air(sigs_in: list[BeatSignal], a: float, b: float) -> float:
    """Longest stretch with no line, action or sound cue (text-only models are blind to this)."""
    ts = sorted((s.beat.start, s.beat.end) for s in sigs_in if s.beat.kind in ("dialogue", "action", "sfx"))
    gap, cur = 0.0, a
    for st, en in ts:
        gap = max(gap, st - cur)
        cur = max(cur, en)
    return max(gap, b - cur)


def length_fit(d: float) -> float:
    return 1.0 if d <= 40 else 0.93 if d <= 50 else 0.86


def funnel_fit(ep: int, cfg) -> float:
    """'Promise distance': a new installer starts at Ep 1, so a promo whose payoff is a few taps away converts
    better (ReelShort's longest-running creative was an Ep-1 clip; duanju editors cut from the first ~20 eps first)."""
    e0 = cfg["early_episodes"]
    if cfg["funnel_floor"] >= 1.0:          # weighting switched off (used in the sensitivity check)
        return 1.0
    if ep <= e0:
        return 1.0 - 0.04 * (ep - 1) / max(e0 - 1, 1)          # 1.00 at Ep 1 -> 0.96 at the paywall
    return max(cfg["funnel_floor"], 0.96 - (ep - e0) * (0.96 - cfg["funnel_floor"]) / 38)


# ----------------------------------------------------------------------------- scoring a promo
def has_phrase(text: str, terms) -> bool:
    """Whole-phrase match: 'duchess of ash' must not fire inside 'Duchess of Ashburgh'."""
    return any(re.search(rf"(?<![a-z0-9]){re.escape(t.lower())}(?![a-z0-9])", text) for t in terms)


class Show:
    def __init__(self, episodes: list[Episode], cfg: dict | None = None, lexicon_path=None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        w = {k: max(0.0, float(v)) for k, v in self.cfg["weights"].items()}
        total = sum(w.values())
        if total <= 0:
            raise ValueError("At least one rubric weight must be above zero.")
        self.cfg["weights"] = {k: v / total for k, v in w.items()}     # always sums to 1 (defaults unchanged)
        self.lex = load_lexicon(lexicon_path)
        self.eps = {e.number: e for e in episodes}
        self.sigs = {e.number: tag_episode(e.beats, self.lex) for e in episodes}
        names = set()
        counts: dict[str, int] = {}
        for e in episodes:
            for b in e.beats:
                if b.kind == "dialogue":
                    for w in re.findall(r"[A-Z']+", b.speaker):
                        names.add(re.sub(r"'S$", "", w).title())
                    counts[b.speaker] = counts.get(b.speaker, 0) + 1
        for e in episodes:
            for b in e.beats:
                if b.kind == "dialogue":
                    for w in re.findall(r"\b[A-Z][a-z]{2,}\b", b.text):
                        if w in names:
                            pass
        generic = {"The", "Lady", "Lord", "Former", "Groom", "Noble", "Matriarch", "Royal", "Guard", "Bearded",
                   "Mysterious", "Stranger", "Man", "Woman", "Elder", "Nobleman", "Noblewoman", "Off", "Screen",
                   "Voice", "Constable", "Police", "High", "Priest", "Judge", "Maid", "Housekeeper", "Informant",
                   "Crown", "Official", "Thug", "With", "Cap", "Bald", "Tall", "Third", "Cart", "Driver",
                   "Loyalist", "Scout", "Fighter", "Peasant", "Mother", "Villager", "Aristocratic", "Townsman",
                   "Dowager", "Duchess", "Father", "Scarred", "In", "Gold", "Gown", "Blond", "Butler", "Footman",
                   "Sentry", "Conspirator", "Old"}
        self.cast = {n for n in names if n not in generic and len(n) > 2}
        vals = sorted(sg.intensity for n in self.sigs for sg in self.sigs[n] if sg.beat.kind != "camera")
        self.hot_thr = vals[int(0.8 * len(vals))] if vals else 0.65
        top = sorted(counts, key=counts.get, reverse=True)[:2]
        self.protagonists = {w.title() for s in top for w in re.findall(r"[A-Z']+", s)}

    def score(self, p: Promo) -> Promo:
        cfg = self.cfg
        seg_sigs = []
        for seg in p.segments:
            seg_sigs.append(window_signals(self.sigs[seg.episode], seg.start, seg.end))
        allsig = [s for ss in seg_sigs for s in ss]
        p.beats = allsig
        if not allsig:
            p.final = 0
            return p
        first, last = p.segments[0], p.segments[-1]
        h, hb, hinfo = hook_score(seg_sigs[0], first.start)
        ep_last = self.eps[last.episode]
        after = [s for s in self.sigs[last.episode] if s.beat.start >= last.end - 0.05]
        c, cb, ckind = cliff_score(seg_sigs[-1], last.end, ep_last.duration, after)
        cl, cinfo = clarity_score(allsig, self.cast, self.protagonists, cfg)
        if len(p.segments) > 1:
            # every jump-cut costs a cold viewer some orientation; unrelated casts across segments cost more
            casts = [{x.beat.speaker for x in ss if x.beat.kind == "dialogue"} for ss in seg_sigs]
            shared = all(c & casts[-1] or not c for c in casts[:-1])
            cl = max(0.0, cl - 0.06 * (len(p.segments) - 1) - (0 if shared else 0.12))
        es = escalation_score(allsig, p.duration, self.hot_thr)
        tr, trope = trope_score(allsig)
        w = cfg["weights"]
        base = w["hook"] * h + w["cliff"] * c + w["clarity"] * cl + w["escalation"] * es + w["trope"] * tr
        gaps = max(dead_air(ss, sg.start, sg.end) for ss, sg in zip(seg_sigs, p.segments))
        da = 1.0 if gaps <= 4.0 else max(0.75, 1.0 - 0.04 * (gaps - 4.0))
        lf = length_fit(p.duration)
        ff = funnel_fit(min(p.episodes), cfg)
        p.scores = {"hook": round(h, 3), "cliff": round(c, 3), "clarity": round(cl, 3),
                    "escalation": round(es, 3), "trope": round(tr, 3), "length_fit": lf,
                    "dead_air_s": round(gaps, 1), "dead_air_mult": round(da, 3), "funnel_fit": round(ff, 3),
                    "rules": round(100 * base * da * lf, 1)}
        p.final = round(100 * base * da * lf * ff, 1)
        p.trope = trope
        text = " ".join(s.beat.text.lower() for s in allsig)
        p.lane = "retarget" if has_phrase(text, cfg["payoff_terms"]) else "cold_ua"
        p.flags = []
        if hinfo.get("sound_off_readable"):
            p.flags.append("sound-off hook")
        else:
            p.flags.append("needs captions for hook")
        if hinfo.get("establishing_open"):
            p.flags.append("soft open (establishing shot)")
        if cinfo["dangling_names"]:
            p.flags.append("unexplained names: " + ", ".join(cinfo["dangling_names"]))
        if gaps > 4.0:
            p.flags.append(f"dead air {gaps:.1f}s")
        from .signals import _hits
        if _hits(text, ["blood", "branding", "gallows", "noose", "explosion", "throat", "naked", "corpse",
                        "poker", "whip", "lacerat", "gash", "sleeper hold"]):
            p.flags.append("ad-policy check: graphic violence")
        if p.lane == "retarget":
            p.flags.append("pays off central mystery -> retargeting")
        p.cut_end = self._cut_ending(last)
        p.hook = self._describe(hb, first.start, "hook")
        p.cliff = self._describe(cb, None, "cliff")
        p.cliff["type"] = ckind
        p.scores["speakers"] = cinfo["speakers"]
        p.why = self._why(p, h, c, ckind)
        p.title = f"{TROPE_LABELS.get(trope, 'Drama')} · Ep {'+'.join(str(e) for e in p.episodes)}"
        return p

    @staticmethod
    def _describe(s: BeatSignal | None, a, role):
        if s is None:
            return {"text": "", "at": None}
        b = s.beat
        if b.kind == "dialogue":
            txt = f'{name_case(b.speaker)} ({b.tags}): "{b.text}"'
        else:
            txt = f"[{b.kind}] {b.text}"
        d = {"text": txt[:220], "episode": b.episode, "time": fmt(b.start)}
        if a is not None:
            d["seconds_in"] = round(max(0.0, b.start - a), 1)
        return d

    def _cut_ending(self, seg: Segment) -> dict:
        """The literal last moments at the out-point (vs. the 'cliffhanger beat', which is the strongest
        unresolved beat in the last ~4.5 s). Reported separately so the two are never confused."""
        beats = self.eps[seg.episode].beats
        inside = [b for b in beats if b.start >= seg.start - 0.05 and b.start < seg.end - 0.05]
        last_line = next((b for b in reversed(inside) if b.kind == "dialogue" and b.end <= seg.end + 0.05), None)
        last_visual = next((b for b in reversed(inside) if b.kind in ("action", "sfx")), None)
        mid_action = [b for b in beats if b.kind == "action" and b.start < seg.end - 0.05 < b.end - 0.05]
        out = {"out_point": fmt(seg.end), "episode": seg.episode}
        if last_line:
            out["last_line"] = f'{name_case(last_line.speaker)}: "{last_line.text}" ({fmt(last_line.start)})'
        if last_visual:
            out["last_action_or_sound"] = f"[{last_visual.kind}] {last_visual.text[:160]} ({fmt(last_visual.start)})"
        if mid_action:
            out["cuts_during_action"] = mid_action[0].text[:160]
        return out

    @staticmethod
    def _why(p: Promo, h, c, ckind):
        hb = p.hook
        lead = "Opens hot" if h >= 0.75 else "Opens on a clear conflict" if h >= 0.55 else "Slower open"
        sec = hb.get("seconds_in", 0)
        trope = TROPE_LABELS.get(p.trope, "drama").lower()
        when = "in the first second" if sec < 1 else f"by {sec:.1f}s"
        return f"{lead} ({trope}) {when}, escalates for {p.duration:.0f}s, cuts on {ckind} before the answer."

    # ------------------------------------------------------------------------- candidates
    def continuous_candidates(self, ep_num: int) -> list[Promo]:
        cfg = self.cfg
        ep = self.eps[ep_num]
        sigs = self.sigs[ep_num]
        starts, ends = cut_points(ep)
        # pre-rank starts by local hook strength and ends by local cliff strength
        st_scored = []
        for a in starts:
            win = window_signals(sigs, a, a + 6.5)
            h, _, _ = hook_score(win, a) if win else (0, None, {})
            st_scored.append((h, a))
        en_scored = []
        for b in ends:
            win = window_signals(sigs, b - 8.0, b)
            after = [s for s in sigs if s.beat.start >= b - 0.05]
            c, _, _ = cliff_score(win, b, ep.duration, after) if win else (0, None, "")
            en_scored.append((c, b))
        st = [a for _, a in sorted(st_scored, reverse=True)[: cfg["starts_per_episode"]]]
        en = [b for _, b in sorted(en_scored, reverse=True)[: cfg["ends_per_episode"]]]
        out = []
        for a in st:
            for b in en:
                if cfg["min_len"] <= b - a <= cfg["max_len"]:
                    out.append(Promo(f"E{ep_num:02d}-{a:06.2f}-{b:06.2f}", "continuous", [Segment(ep_num, a, b)]))
        return out

    def stitched_candidates(self, ep_num: int) -> list[Promo]:
        """SocialPeta 'Formula 1': 3s lure + 2-3 high-impact clips + cliff ending, spliced (can span 2 eps)."""
        cfg = self.cfg
        arc = [n for n in (ep_num - 1, ep_num) if n in self.eps]
        last_ep = self.eps[ep_num]
        sigs_last = self.sigs[ep_num]
        _, ends = cut_points(last_ep)
        # best 3 cliff endpoints in the last episode
        en_scored = []
        for b in ends:
            win = window_signals(sigs_last, b - 8.0, b)
            after = [s for s in sigs_last if s.beat.start >= b - 0.05]
            c, _, _ = cliff_score(win, b, last_ep.duration, after) if win else (0, None, "")
            en_scored.append((c, b))
        out = []
        for _, cend in sorted(en_scored, reverse=True)[:3]:
            # final segment: 8-18s lead-up into the cliff, legal start
            st_last, _ = cut_points(last_ep)
            f_starts = [a for a in st_last if 8 <= cend - a <= 18]
            if not f_starts:
                continue
            fa = min(f_starts, key=lambda a: abs((cend - a) - 12))
            final = Segment(ep_num, fa, cend)
            # pool of short, hot clips earlier in the arc (3-9s, legal cut points)
            pool = []
            for n in arc:
                ep = self.eps[n]
                s_pts, e_pts = cut_points(ep)
                for a in s_pts:
                    if n == ep_num and a >= fa - 1:
                        continue
                    for b in e_pts:
                        if 4.0 <= b - a <= 10.0 and not (n == ep_num and b > fa):
                            win = window_signals(self.sigs[n], a, b)
                            if not win:
                                continue
                            dens = sum(s.intensity for s in win) / (b - a)
                            h, _, _ = hook_score(win, a)
                            pool.append((dens, h, Segment(n, a, b)))
            if not pool:
                continue
            # lure = best hook clip anywhere in pool (may be out of chronological order = "flash-forward")
            best_lure = max(pool, key=lambda x: x[1] + 0.2 * x[0])
            lure_h, lure = best_lure[1] + 0.2 * best_lure[0], best_lure[2]
            if lure_h < 0.75:                         # a stitched cut only earns its jump-cuts with a real lure
                continue
            chosen = [lure]
            total = lure.dur + final.dur
            for dens, h, seg in sorted(pool, key=lambda x: -x[0]):
                if total >= 30 or len(chosen) >= 3:   # lure + max 2 mid clips + final = 4 segments
                    break
                if any(seg.episode == c.episode and not (seg.end <= c.start or seg.start >= c.end) for c in chosen):
                    continue
                if total + seg.dur > 55:
                    continue
                chosen.append(seg)
                total += seg.dur
            mids = sorted(chosen[1:], key=lambda s: (s.episode, s.start))
            segs = [lure] + mids + [final]
            if not (cfg["min_len"] <= sum(s.dur for s in segs) <= cfg["max_len"]) or len(segs) < 3:
                continue
            pid = f"S{ep_num:02d}-{cend:06.2f}-" + "-".join(f"{s.episode}@{s.start:.1f}" for s in segs[:-1])
            out.append(Promo(pid, "stitched", segs))
        return out

    def tiktok_variant(self, p: Promo):
        cfg = self.cfg
        if p.duration <= cfg["tiktok_max"]:
            # only call it a TikTok cut if it really is 21-34 s; never stretch a clip by inventing time
            return list(p.segments) if p.duration >= cfg["tiktok_min"] else None
        if p.kind == "stitched":
            segs = list(p.segments)
            while sum(s.dur for s in segs) > cfg["tiktok_max"] and len(segs) > 2:
                segs.pop(-2)
            return segs if sum(s.dur for s in segs) >= cfg["tiktok_min"] else None
        seg = p.segments[0]
        ep = self.eps[seg.episode]
        starts, _ = cut_points(ep)
        best, bs = None, -1
        for a in starts:
            d = seg.end - a
            if cfg["tiktok_min"] <= d <= cfg["tiktok_max"]:
                win = window_signals(self.sigs[seg.episode], a, seg.end)
                h, _, _ = hook_score(win, a)
                if h > bs:
                    bs, best = h, Segment(seg.episode, a, seg.end)
        return [best] if best else None


# ----------------------------------------------------------------------------- ranking
def overlap(p: Promo, q: Promo) -> float:
    tot = 0.0
    for a in p.segments:
        for b in q.segments:
            if a.episode == b.episode:
                tot += max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return tot / max(min(p.duration, q.duration), 1e-6)


def rank(show: Show, cands: list[Promo], top_n: int | None = None, key=lambda p: p.final) -> list[Promo]:
    top_n = top_n or show.cfg["top_n"]
    pool = sorted(cands, key=key, reverse=True)
    kept: list[Promo] = []
    for p in pool:                                   # non-max suppression on time overlap
        if all(overlap(p, q) < 0.35 for q in kept):
            kept.append(p)
        if len(kept) >= top_n * 3:
            break
    selected: list[Promo] = []                       # diversity re-rank (MMR-style)
    remaining = kept[:]
    while remaining and len(selected) < top_n:
        def adj(p):
            pen = 1.0
            for q in selected:
                if q.trope == p.trope:
                    pen *= 0.96
                if set(q.episodes) & set(p.episodes):
                    pen *= 0.9
            return key(p) * pen
        # quality floor: variety may only choose among clips within 10 points of the best one still
        # available, so a clearly weaker clip can never jump the queue just for being different
        qmax = max(key(p) for p in remaining)
        eligible = [p for p in remaining if key(p) >= qmax - 10.0]
        best = max(eligible, key=adj)
        best.priority = round(adj(best), 1)
        selected.append(best)
        remaining.remove(best)
    return selected


def find_promos(episodes: list[Episode], cfg: dict | None = None, judge=None, progress=None,
                lexicon_path=None) -> tuple[Show, list[Promo], dict]:
    show = Show(episodes, cfg, lexicon_path)
    cands: list[Promo] = []
    for i, n in enumerate(sorted(show.eps)):
        for p in show.continuous_candidates(n) + show.stitched_candidates(n):
            show.score(p)
            if p.final > 0:
                cands.append(p)
        if progress:
            progress((i + 1) / len(show.eps))
    stats = {"episodes": len(show.eps), "beats": sum(len(e.beats) for e in episodes),
             "runtime_min": round(sum(e.duration for e in episodes) / 60, 1), "candidate_windows_scored": len(cands)}
    shortlist = rank(show, cands, top_n=max(show.cfg["top_n"], 30))
    if judge is not None:
        judge.judge(show, shortlist)
        ranked = rank(show, shortlist, key=lambda p: p.final)
    else:
        ranked = shortlist
    ranked = ranked[: show.cfg["top_n"]]
    for p in ranked:
        p.tiktok_cut = show.tiktok_variant(p)
    # separate shortlist for people already watching: clips that pay off the central mystery
    taken = {p.pid for p in ranked}
    retarget = rank(show, [p for p in cands if p.lane == "retarget" and p.pid not in taken], top_n=5)
    for p in retarget:
        p.tiktok_cut = show.tiktok_variant(p)
    show.retarget = retarget
    stats["retarget_candidates"] = sum(p.lane == "retarget" for p in cands)
    return show, ranked, stats


def promo_to_dict(p: Promo, rank_no: int) -> dict:
    d = {
        "rank": rank_no, "id": p.pid, "title": p.title, "type": p.kind, "lane": p.lane,
        "episodes": p.episodes, "total_duration_s": round(p.duration, 1),
        "segments": [s.to_dict() for s in p.segments],
        "hook": p.hook, "cliffhanger_beat": p.cliff, "cut_ending": p.cut_end, "why": p.why,
        "quality_score": p.final, "posting_priority": p.priority, "scores": p.scores, "trope": TROPE_LABELS.get(p.trope, p.trope), "flags": p.flags,
        "tiktok_cut": [s.to_dict() for s in p.tiktok_cut] if p.tiktok_cut else None,
    }
    if p.llm:
        d["llm_judge"] = p.llm
    return d
