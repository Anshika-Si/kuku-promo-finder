"""Rule layer: turns each beat into numeric signals. Cheap, deterministic, explainable.

Lexicons are grounded in the research brief:
  * Hook openers used by duanju ad editors: slap / wealth-or-status flex / "how can it be you" reveal,
    plus 0-3s images: slap close-up, weapon to the head, secret exposed, kneeling humiliation.
  * Trope groups from 500+ short-drama creatives (AppGrowing/Aggclaw 2026): identity reversal,
    contract/forced marriage, public humiliation & revenge, forbidden romance, rescue.
  * Cliffhanger types from ad editors: "cut on action" (mid-strike) and "cut on plot" (unanswered dilemma).
All lists are editable in config/signals.json without touching code.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .parser import Beat

DEFAULT_LEXICON = {
    "physical": ["slap", "strike", "strikes", "struck", "punch", "shove", "shoves", "grab", "grabs", "seize",
                 "drag", "drags", "hurl", "flings", "throws", "blood", "bloody", "bleeding", "gash", "knife",
                 "blade", "sword", "halberd", "poker", "branding", "iron", "fire", "flames", "blaze", "explosion",
                 "explodes", "fireball", "detonation", "shatter", "crash", "collapses", "whip", "crop", "chains",
                 "chained", "shackle", "noose", "gallows", "throat", "lunges", "tackling", "club", "thorns",
                 "briars", "crawl", "crawls", "kneel", "kneels", "drops to her knees", "fist", "beam"],
    "reveal": ["duke", "heir", "crest", "signet", "true identity", "my son", "who are you", "recogni",
               "reveal", "unmask", "regalia", "your grace", "alive", "presumed dead", "secret", "the truth",
               "poison", "arsenic", "ring", "it's you", "it was always me", "rightful", "imposter", "you already know"],
    "humiliation": ["plain", "worthless", "low value", "beggar", "peasant", "nobody", "penniless", "mud",
                    "dirt", "disowned", "banished", "discarded", "stripped", "mock", "laugh", "sneer", "jeer",
                    "scrub", "servant", "maid", "gutter", "disgrace", "humiliat", "beneath", "tolerate",
                    "crawl", "kneel", "embarrass", "no daughter of mine", "dead to", "witch", "rat"],
    "betrayal": ["marrying me", "marry", "fiancé", "groom", "bride", "sister", "abandon", "betray", "framed",
                 "frame", "false", "lie", "lying", "planted", "stole", "thief", "theft", "changed my mind",
                 "you knew", "you swore", "testimony", "accuse"],
    "threat": ["hang", "hangs", "gallows", "execute", "execution", "executed", "die", "dead", "death", "kill",
               "burn", "arrest", "dungeon", "cell", "gaol", "trial", "sentence", "condemn", "or else",
               "ultimatum", "before the magistrate", "your corpse", "the rope", "two days", "ten days",
               "choose your next words", "wrong answer", "take her", "guards!", "remove her"],
    "romance": ["accept my hand", "marry", "ring", "kiss", "embrace", "my wife", "husband", "hands off her",
                "take your hands", "protect", "vow", "i love", "my duchess", "lawful wife", "chose me"],
}

HOT_TAGS = ["furious", "scream", "shriek", "hysteric", "venom", "menac", "cruel", "shock", "stunn", "horrif",
            "desperate", "sobbing", "gasp", "enraged", "snarl", "triumphant", "mocking", "sneer", "gloat",
            "threaten", "defiant", "booming", "roaring", "thunderous", "outraged", "agoniz", "terrified",
            "panic", "deadly", "chilling", "sinister", "malicious", "vicious", "brutal", "lethal", "devastat",
            "heartbroken", "crying", "weeping", "betrayed", "accusing", "frantic", "wailing", "spiteful",
            "taunting", "callous", "imperious", "commanding", "piercing"]
COOL_TAGS = ["gentle", "soft", "warm", "tender", "calm", "playful", "amused", "teasing", "content", "peaceful",
             "fond", "reflective", "relieved", "smiling", "sweet", "polite", "humble"]
CAMERA_HOT = ["crash zoom", "snap zoom", "whip pan", "extreme close", "push-in", "push in", "slams", "freeze",
              "cut to black", "smash cut", "dutch", "slow motion", "tight insert", "low-angle"]
ESTABLISHING = ["establishing", "aerial", "wide exterior", "landscape", "drone", "exterior of", "slowly pans",
                "morning mist", "sweeping wide", "wide shot of the"]
SFX_HOT = ["slap", "crack", "thud", "smash", "explosion", "detonation", "boom", "scream", "gasp", "sting",
           "ominous", "cliffhanger", "crash", "clang", "slam", "bang", "punch", "whip", "sizzl", "hiss",
           "crescendo", "swell", "pounding", "shatter"]
CLIFF_SFX = ["cliffhanger", "ominous", "sting", "cut to black", "unresolved", "into silence", "abrupt",
             "freeze", "hold", "holds on", "crescendo"]
UNRESOLVED_ACTION = ["raises", "lunges", "swings", "reaches", "aims", "toward", "about to", "freezes",
                     "burst", "bursts", "enters", "arrives", "appears", "steps through", "thrusts", "hoists",
                     "stopping mere", "stops inches", "inches from", "millimeters", "emerge", "turns her head", "eyes snap",
                     "realiz", "drags", "dragged", "twists", "pounding", "door", "latch"]
RESOLUTION = ["smiles", "smiling", "relief", "relieved", "embrace", "embracing", "sleep", "rest", "thank you",
              "safe", "peace", "warmth", "tender", "gentle", "laugh", "together", "fade out", "fades",
              "comforted", "we'll see", "it is enough", "i accept", "i am.", "nestles", "hand-in-hand"]
RELATION_ANCHORS = ["sister", "husband", "wife", "fiancé", "mother", "father", "son", "daughter", "groom",
                    "bride", "duke", "duchess", "stepmother", "maid", "lady", "lord", "soldier", "veteran",
                    "beggar", "cousin", "wedding", "king", "queen", "witch", "family"]

TROPE_LABELS = {
    "physical": "Physical shock",
    "reveal": "Identity / secret reveal",
    "humiliation": "Public humiliation",
    "betrayal": "Betrayal & frame-up",
    "threat": "Life-or-death stakes",
    "romance": "Protector romance",
}


def load_lexicon(path: str | Path | None = None) -> dict:
    lex = {k: list(v) for k, v in DEFAULT_LEXICON.items()}
    if path and Path(path).exists():
        lex.update({k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")})
    return lex


_RX_CACHE: dict = {}


def _rx(words: tuple) -> list:
    if words not in _RX_CACHE:
        # left word-boundary only: stems like "humiliat" still match "humiliated", but "sting" no longer
        # fires inside "inspecting" (a real bug found while checking outputs)
        _RX_CACHE[words] = [re.compile(r"(?<![a-z])" + re.escape(w)) for w in words]
    return _RX_CACHE[words]


def _hits(text: str, words: list[str]) -> int:
    t = text.lower()
    return sum(1 for r in _rx(tuple(words)) if r.search(t))


@dataclass
class BeatSignal:
    beat: Beat
    intensity: float = 0.0          # 0..~3, how scroll-stopping this beat is on its own
    tropes: dict = field(default_factory=dict)
    visual: bool = False            # readable with sound off (action/camera/sfx carries it)
    question: bool = False
    threat: bool = False
    unresolved: float = 0.0         # how much this beat leaves hanging if we cut right after it
    resolution: float = 0.0         # how much it closes things down
    establishing: bool = False
    anchors: int = 0                # relationship words that orient a cold viewer


def score_beat(b: Beat, lex: dict) -> BeatSignal:
    text = b.text
    low = text.lower()
    tags = b.tags.lower()
    s = BeatSignal(b)
    s.tropes = {k: _hits(text, v) for k, v in lex.items()}
    trope_mass = sum(min(v, 2) for v in s.tropes.values())
    hot = _hits(tags, HOT_TAGS)
    cool = _hits(tags, COOL_TAGS)
    if b.kind == "dialogue":
        s.intensity = 0.35 * min(hot, 3) - 0.3 * min(cool, 2) + 0.25 * min(trope_mass, 4)
        s.intensity += 0.35 if "!" in text else 0.0
        s.question = "?" in text
        s.intensity += 0.15 if s.question else 0.0
        s.anchors = _hits(low, RELATION_ANCHORS)
    elif b.kind == "action":
        s.intensity = 0.4 * min(trope_mass, 5) + 0.25 * _hits(low, ["violent", "brutal", "sudden", "abrupt", "forcibly", "shock", "horror", "terror"])
        s.visual = trope_mass > 0
        s.anchors = _hits(low, RELATION_ANCHORS)
    elif b.kind == "sfx":
        s.intensity = 0.45 * min(_hits(low, SFX_HOT), 3)
        s.visual = False
    elif b.kind == "camera":
        s.intensity = 0.2 * min(_hits(low, CAMERA_HOT), 2) + 0.15 * min(trope_mass, 3)
        s.visual = trope_mass > 0
        s.establishing = _hits(low, ESTABLISHING) > 0
    s.threat = s.tropes.get("threat", 0) > 0
    s.unresolved = (0.6 if s.question else 0) + (0.5 if s.threat else 0) \
        + 0.4 * min(_hits(low, UNRESOLVED_ACTION), 2) + 0.6 * min(_hits(low, CLIFF_SFX), 2) * (b.kind in ("sfx", "camera", "action"))
    s.resolution = 0.35 * min(_hits(low + " " + tags, RESOLUTION), 3)
    s.intensity = max(0.0, s.intensity)
    return s


def tag_episode(beats: list[Beat], lex: dict) -> list[BeatSignal]:
    return [score_beat(b, lex) for b in beats]


def dominant_trope(signals: list[BeatSignal]) -> str:
    agg: dict[str, float] = {}
    for s in signals:
        for k, v in s.tropes.items():
            agg[k] = agg.get(k, 0) + min(v, 2) * (1 + s.intensity)
    if not agg or max(agg.values()) == 0:
        return "drama"
    return max(agg, key=agg.get)


WORD_RE = re.compile(r"[A-Za-z']+")
