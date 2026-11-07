"""Optional LLM judge: re-scores the rule shortlist like a 'cold viewer' + performance editor.

Swap models with one string:  provider:model
  gemini:gemini-3.8-flash        (GEMINI_API_KEY)                 groq:llama-3.3-70b-versatile (GROQ_API_KEY)
  openai:gpt-4o-mini             (OPENAI_API_KEY)                  anthropic:claude-sonnet-4-5  (ANTHROPIC_API_KEY)
  openrouter:<any model>         (OPENROUTER_API_KEY)              ollama:llama3.1              (local, no key)
  custom:<model>                 (LLM_BASE_URL + LLM_API_KEY, any OpenAI-compatible endpoint, e.g. Kuku's own)
  mock                           (offline, deterministic: for tests and demos without a key)

Confidential scripts: only use a local model or an endpoint Kuku has approved. Unpaid API tiers may use
submitted data to improve the provider's products, so they are not suitable for this material.

Guard-rails against hallucination:
  * The LLM receives candidate IDs and transcript text with relative beat offsets (e.g. +3.0s), but no source
    timecodes. Python owns and validates every source start/end point; the LLM only scores the candidates.
  * It is deliberately given NO show context (no title, no logline) so it judges like a stranger scrolling a feed.
  * Output must be JSON matching a schema; unknown IDs, missing fields or out-of-range scores are rejected and the
    batch is retried with the error message; after up to 3 failed attempts the original rule score is kept (the pipeline never breaks).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request

from .parser import fmt
from .promos import name_case

ENDPOINTS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}

SYSTEM = """You are a senior performance-marketing editor for vertical microdrama apps (ReelShort/DramaBox style).
You judge candidate PROMO CLIPS (20-60s, cut from the show itself, posted on TikTok/Reels/Shorts to win NEW viewers).
You have never seen this show. Judge each clip exactly as a stranger scrolling a feed would.

Score each candidate 1-5 on:
- hook_3s: would the first 3 seconds stop a thumb? (slap, shocking line, reveal, visible danger = high; scenery, calm talk = low)
- cliffhanger: does it cut at an unresolved moment so the viewer NEEDS the next second? (resolved / happy / calm ending = low)
- standalone: can a stranger tell who is who and what is at stake, without earlier episodes?
Also return:
- cold_viewer_question: the single question a stranger is left asking (<= 15 words)
- hook_line: the line or image that hooks, copied or closely paraphrased from the transcript (<= 15 words)
- why: one line on why it would work or not (<= 25 words)
- risk: "none" or a short note (confusing, spoils the central twist, ad-policy violence, misleading)
Rules: never invent dialogue or events not in the transcript. Never output timestamps. Use only the given ids.
Return ONLY JSON: {"results":[{"id":"...","hook_3s":1-5,"cliffhanger":1-5,"standalone":1-5,
"cold_viewer_question":"...","hook_line":"...","why":"...","risk":"..."}]}"""


def transcript(p, max_chars=1400, show=None) -> str:
    """Text the judge sees for one promo. Includes every beat that OVERLAPS a segment (camera directions too),
    so an action still running at the out-point (a cut mid-action, i.e. the cliffhanger itself) is not lost.
    Beats clipped by a segment boundary are marked."""
    lines, offset = [], 0.0
    for k, seg in enumerate(p.segments):
        if k:
            lines.append(f"--- CUT to another moment (episode {seg.episode}) ---")
        if show is not None:
            pool = [b for b in show.eps[seg.episode].beats if b.end > seg.start + 0.05 and b.start < seg.end - 0.05]
        else:
            pool = [s.beat for s in p.beats if s.beat.episode == seg.episode
                    and s.beat.end > seg.start + 0.05 and s.beat.start < seg.end - 0.05]
        for b in sorted(pool, key=lambda x: (x.start, x.end)):
            rel = offset + max(0.0, b.start - seg.start)
            note = ""
            if b.start < seg.start - 0.05:
                note = " (already under way when the clip starts)"
            if b.end > seg.end + 0.05:
                note = " (CUT OFF here: the clip ends during this)"
            if b.kind == "dialogue":
                lines.append(f"+{rel:4.1f}s {name_case(b.speaker)} ({b.tags}): \"{b.text}\"{note}")
            else:
                lines.append(f"+{rel:4.1f}s [{b.kind.upper()}] {b.text}{note}")
        offset += seg.dur
    txt = "\n".join(lines)
    if len(txt) <= max_chars:
        return txt
    # keep the opening (hook) AND the ending (cliffhanger); drop the middle
    head, tail = txt[: max_chars // 2], txt[-max_chars // 2:]
    return head + "\n[... middle trimmed ...]\n" + tail


class LLMJudge:
    def __init__(self, provider: str, model: str, alpha: float = 0.5, batch: int = 6, max_cands: int = 30,
                 max_chars: int = 2400, api_key: str | None = None, base_url: str | None = None):
        self.provider, self.model = provider, model
        self.api_key, self.base_url = api_key, base_url      # per-instance; env vars are only a fallback for the CLI
        self.alpha, self.batch, self.max_cands, self.max_chars = alpha, batch, max_cands, max_chars
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "failures": 0, "provider": provider, "model": model}

    @classmethod
    def from_spec(cls, spec: str, **kw):
        if spec == "mock":
            return cls("mock", "mock", **kw)
        provider, _, model = spec.partition(":")
        if provider not in ENDPOINTS and provider not in ("anthropic", "custom"):
            raise ValueError(f"unknown provider '{provider}'")
        return cls(provider, model, **kw)

    # ---------------------------------------------------------------- transport
    def _chat(self, system: str, user: str) -> str:
        if self.provider == "mock":
            return self._mock(user)
        if self.provider == "anthropic":
            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY", "")
            body = {"model": self.model, "max_tokens": 2000, "system": system,
                    "messages": [{"role": "user", "content": user}], "temperature": 0}
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                         headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                                  "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read())
            u = data.get("usage", {})
            self.usage["input_tokens"] += u.get("input_tokens", 0)
            self.usage["output_tokens"] += u.get("output_tokens", 0)
            return "".join(c.get("text", "") for c in data.get("content", []))
        if self.provider == "custom":
            base = self.base_url or os.environ.get("LLM_BASE_URL", "")
            key = self.api_key or os.environ.get("LLM_API_KEY", "")
        else:
            base, env = ENDPOINTS[self.provider]
            key = self.api_key or (os.environ.get(env, "") if env else "") or os.environ.get("LLM_API_KEY", "")
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if "gemini-3" not in self.model.lower():
            # deterministic scoring where the model allows it; Google recommends leaving Gemini 3 models at their
            # default temperature (1.0), since lower values can cause looping or degraded reasoning
            body["temperature"] = 0
        if self.provider in ("openai", "groq", "gemini", "custom"):
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read())
        u = data.get("usage", {}) or {}
        self.usage["input_tokens"] += u.get("prompt_tokens", 0)
        self.usage["output_tokens"] += u.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _mock(user: str) -> str:
        ids = re.findall(r"^### id=(\S+)", user, re.M)
        res = []
        for i in ids:
            block = user.split(f"### id={i}", 1)[1].split("### id=", 1)[0]
            hot = len(re.findall(r"!|\?|Screaming|Furious|Cruel|blood|slap|strike", block))
            v = max(1, min(5, 1 + hot // 3))
            res.append({"id": i, "hook_3s": v, "cliffhanger": v, "standalone": 3,
                        "cold_viewer_question": "What happens next?", "hook_line": "(mock)",
                        "why": "mock judge: counts intensity markers", "risk": "none"})
        return json.dumps({"results": res})

    # ---------------------------------------------------------------- validation
    @staticmethod
    def _parse(raw: str, ids: set[str]) -> dict:
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0) if m else raw)
        out = {}
        for r in data.get("results", []):
            rid = str(r.get("id"))
            if rid not in ids:
                raise ValueError(f"unknown id {rid}; use only {sorted(ids)}")
            for k in ("hook_3s", "cliffhanger", "standalone"):
                v = int(r[k])
                if not 1 <= v <= 5:
                    raise ValueError(f"{k} must be 1-5")
                r[k] = v
            if re.search(r"\d{1,2}:\d{2}", json.dumps(r)):
                r["why"] = re.sub(r"\d{1,2}:\d{2}(\.\d+)?", "", r.get("why", ""))   # strip any stray timecode
            out[rid] = r
        missing = ids - set(out)
        if missing:
            raise ValueError(f"missing ids {sorted(missing)}")
        return out

    # ---------------------------------------------------------------- main
    def judge(self, show, promos):
        cands = promos[: self.max_cands]
        for i in range(0, len(cands), self.batch):
            chunk = cands[i:i + self.batch]
            idmap = {f"c{i + k + 1}": p for k, p in enumerate(chunk)}
            user = "Judge these candidate promos.\n\n" + "\n\n".join(
                f"### id={cid}  ({p.duration:.0f}s, {len(p.segments)} segment(s))\n{transcript(p, self.max_chars, show)}" for cid, p in idmap.items())
            msg, parsed = user, None
            for attempt in range(3):
                try:
                    self.usage["calls"] += 1
                    parsed = self._parse(self._chat(SYSTEM, msg), set(idmap))
                    break
                except Exception as e:                       # noqa: BLE001 - any failure -> retry with feedback
                    self.usage["failures"] += 1
                    msg = user + f"\n\nYour previous answer was invalid ({str(e)[:200]}). Return ONLY valid JSON."
                    time.sleep(1.5 * (attempt + 1))
            if not parsed:
                continue
            for cid, p in idmap.items():
                r = parsed[cid]
                llm01 = (0.40 * (r["hook_3s"] - 1) + 0.35 * (r["cliffhanger"] - 1) + 0.25 * (r["standalone"] - 1)) / 4
                ff = p.scores.get("funnel_fit", 1.0)
                p.scores["llm"] = round(100 * llm01, 1)
                p.scores["rules_final"] = p.final
                p.final = round(self.alpha * p.final + (1 - self.alpha) * 100 * llm01 * ff, 1)
                p.llm = {k: r.get(k) for k in ("hook_3s", "cliffhanger", "standalone", "cold_viewer_question",
                                               "hook_line", "why", "risk")}
                p.llm["note"] = r.get("why", "")
                if r.get("risk") and str(r["risk"]).lower() != "none":
                    p.flags.append("LLM risk: " + str(r["risk"])[:80])
        return promos
