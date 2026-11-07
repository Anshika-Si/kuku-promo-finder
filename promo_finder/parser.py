"""Parse Kuku-style timestamped microdrama screenplays into a flat list of beats.

Format handled (case-insensitive headings, tolerant of missing scene ends):

    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    SCENE 01
    Scene heading: INT. CATHEDRAL - DAY
    Scene start: 00:00.00
    **CAMERA:** / **ACTION:** / **SOUND FX:** / **NAME (Emotion, Tags):**
    Start: (00:00.00) | End: (00:04.80)
    text...
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

TS = r"(\d{1,2}):(\d{2}(?:\.\d+)?)"
RE_BLOCK = re.compile(r"^\*\*(?P<label>[^*]+?)\s*:\s*\*\*\s*$")
RE_TIME = re.compile(rf"Start:\s*\(\s*{TS}\s*\)\s*\|\s*End:\s*\(\s*{TS}\s*\)", re.I)
RE_EPNUM = re.compile(r"^EPISODE NUMBER:\s*.*?(\d+)", re.I)
RE_LOGLINE = re.compile(r"^EPISODE LOGLINE:\s*(.*)", re.I)
RE_SCENE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
RE_HEADING = re.compile(r"^SCENE HEADING:\s*(.*)", re.I)
RE_SCENE_END = re.compile(rf"^SCENE END:\s*{TS}", re.I)


def to_sec(m: str, s: str) -> float:
    return int(m) * 60 + float(s)


def fmt(t: float) -> str:
    """Seconds -> MM:SS.ss (the screenplay's own format)."""
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Beat:
    episode: int
    idx: int
    scene: int
    location: str
    kind: str            # camera | action | sfx | dialogue | other
    speaker: str         # for dialogue, canonical upper-case name
    tags: str            # parenthetical emotion tags, e.g. "Cold, Commanding"
    text: str
    start: float
    end: float

    def to_dict(self):
        return asdict(self)


@dataclass
class Episode:
    number: int
    logline: str
    beats: list[Beat] = field(default_factory=list)
    duration: float = 0.0
    source: str = ""


SPEAKER_ALIASES: dict[str, str] = {}   # filled per show from --show-config (same character written two ways)


def _canon_speaker(label: str) -> tuple[str, str]:
    m = re.match(r"^(.*?)\s*(?:\((.*)\))?\s*$", label.strip())
    name = (m.group(1) or "").strip().upper()
    tags = (m.group(2) or "").strip()
    return SPEAKER_ALIASES.get(name, name), tags


def parse_episode(text: str, source: str = "") -> Episode:
    lines = [l.rstrip() for l in text.splitlines()]
    number, logline = 0, ""
    scene, location = 0, ""
    beats: list[Beat] = []
    scene_ends: list[float] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if (m := RE_EPNUM.match(line)):
            number = int(m.group(1))
        elif (m := RE_LOGLINE.match(line)):
            logline = m.group(1).strip()
        elif (m := RE_SCENE.match(line)):
            scene = int(m.group(1))
        elif (m := RE_HEADING.match(line)):
            location = m.group(1).strip()
        elif (m := RE_SCENE_END.match(line)):
            scene_ends.append(to_sec(m.group(1), m.group(2)))
        elif (m := RE_BLOCK.match(line)):
            label = m.group("label").strip()
            # find timing line then body text until blank line / next block
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            tm = RE_TIME.search(lines[j]) if j < len(lines) else None
            if tm:
                start = to_sec(tm.group(1), tm.group(2))
                end = to_sec(tm.group(3), tm.group(4))
                body = []
                k = j + 1
                while k < len(lines) and lines[k].strip() and not RE_BLOCK.match(lines[k].strip()):
                    body.append(lines[k].strip())
                    k += 1
                up = label.upper()
                if up == "CAMERA":
                    kind, spk, tags = "camera", "", ""
                elif up == "ACTION":
                    kind, spk, tags = "action", "", ""
                elif up in ("SOUND FX", "SFX", "SOUND"):
                    kind, spk, tags = "sfx", "", ""
                elif up.startswith(("TEXT", "SUPER", "TITLE")):
                    kind, spk, tags = "other", "", ""
                else:
                    kind = "dialogue"
                    spk, tags = _canon_speaker(label)
                txt = " ".join(body).strip()
                if kind == "dialogue":
                    txt = txt.strip().strip('"').strip("“”")
                beats.append(Beat(number, len(beats), scene, location, kind, spk, tags,
                                  txt, start, max(end, start)))
                i = k - 1
        i += 1
    beats.sort(key=lambda b: (b.start, b.end))
    for n, b in enumerate(beats):
        b.idx = n
    dur = max([b.end for b in beats] + scene_ends + [0.0])
    return Episode(number, logline, beats, dur, source)


def load_show(path: str | Path) -> list[Episode]:
    """Load a folder of .txt files, a single .txt, or a .zip containing them."""
    p = Path(path)
    texts: list[tuple[str, str]] = []
    if p.is_dir():
        for f in sorted(p.rglob("*.txt")):
            texts.append((f.name, f.read_text(encoding="utf-8", errors="replace")))
    elif p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            for n in sorted(z.namelist()):
                if n.lower().endswith(".txt") and not n.startswith("__MACOSX"):
                    texts.append((Path(n).name, z.read(n).decode("utf-8", errors="replace")))
    else:
        texts.append((p.name, p.read_text(encoding="utf-8", errors="replace")))
    return load_texts(texts)


def load_texts(texts: list[tuple[str, str]]) -> list[Episode]:
    eps = [parse_episode(t, name) for name, t in texts]
    eps = [e for e in eps if e.beats]
    for n, e in enumerate(sorted(eps, key=lambda e: (e.number or 10**6, e.source)), 1):
        if not e.number:
            e.number = n
            for b in e.beats:
                b.episode = n
    return sorted(eps, key=lambda e: e.number)


def load_zip_bytes(data: bytes) -> list[Episode]:
    texts = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for n in sorted(z.namelist()):
            if n.lower().endswith(".txt") and not n.startswith("__MACOSX"):
                texts.append((Path(n).name, z.read(n).decode("utf-8", errors="replace")))
    return load_texts(texts)
