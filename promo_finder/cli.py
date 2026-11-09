"""Command line: python -m promo_finder.cli --data <folder|zip> --out outputs/ [--llm ollama:llama3.1]"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .parser import load_show, fmt
from .promos import find_promos, promo_to_dict
from .report import write_html, write_markdown


def run(data: str, out: str, llm: str | None = None, top_n: int = 25, show_name: str | None = None,
        lexicon: str | None = None, show_config: str | None = None) -> dict:
    t0 = time.time()
    cfg = {"top_n": top_n}
    if show_config:
        extra = {k: v for k, v in json.loads(Path(show_config).read_text()).items() if not k.startswith("_")}
        if "payoff_terms" in extra:
            from .promos import DEFAULTS
            extra["payoff_terms"] = DEFAULTS["payoff_terms"] + extra["payoff_terms"]
        cfg.update(extra)
        from . import parser as _p
        _p.SPEAKER_ALIASES.update(cfg.get("speaker_aliases", {}))
    eps = load_show(data)
    if not eps:
        raise ValueError(
            f"No episode scripts found in '{data}'. Expected .txt files in the Kuku screenplay format "
            "(blocks with 'Start: (MM:SS.ss) | End: (MM:SS.ss)'). Check that --data points at the input pack, "
            "not at this tool's own code zip.")
    judge = None
    if llm:
        if llm != "mock":
            print("NOTE: the LLM judge sends candidate transcripts to an external model. Only use an endpoint "
                  "approved for confidential scripts (e.g. local Ollama or a Kuku-approved API).")
        from .judge import LLMJudge
        judge = LLMJudge.from_spec(llm)
    show, ranked, stats = find_promos(eps, cfg, judge=judge, lexicon_path=lexicon)
    stats["seconds"] = round(time.time() - t0, 2)   # parse + score only; writing the output files below is not included
    stats["mode"] = f"rules + LLM judge ({llm})" if llm else "rules only (no API key needed)"
    if judge:
        stats["llm_usage"] = judge.usage
    name = show_name or Path(data).stem
    result = {
        "show": name, "stats": stats,
        "episode_durations": {e.number: round(e.duration, 2) for e in eps},
        "promos": [promo_to_dict(p, i + 1) for i, p in enumerate(ranked)],
        "retarget_promos": [promo_to_dict(p, i + 1) for i, p in enumerate(getattr(show, "retarget", []))],
        "retarget_note": "Payoff clips (they resolve the central mystery): for retargeting people who already watch, not cold ads.",
    }
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "promos.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    # flat CSV, one row per segment: what an editor drops into a sheet
    with open(outp / "promos_segments.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "promo_id", "lane", "type", "quality_score", "posting_priority", "segment_no", "episode",
                    "start", "end", "seg_duration_s", "total_duration_s", "hook", "cliffhanger_beat",
                    "cut_ending_last_line", "cut_ending_last_action", "why", "flags"])
        for p in result["promos"]:
            for k, s in enumerate(p["segments"], 1):
                w.writerow([p["rank"], p["id"], p["lane"], p["type"], p["quality_score"], p["posting_priority"], k,
                            s["episode"], s["start"], s["end"], s["duration_s"], p["total_duration_s"],
                            p["hook"]["text"], p["cliffhanger_beat"]["text"], p["cut_ending"].get("last_line", ""),
                            p["cut_ending"].get("last_action_or_sound", ""), p["why"], "; ".join(p["flags"])])
    # CMX3600-style edit decision list per promo (episode = reel), 30 fps nominal
    with open(outp / "promos.edl", "w", encoding="utf-8") as f:
        def tc(sec):
            fr = int(round(sec * 30))
            return f"00:{fr // 1800:02d}:{(fr // 30) % 60:02d}:{fr % 30:02d}"
        for p in result["promos"]:
            f.write(f"TITLE: PROMO {p['rank']:02d} {p['id']}\nFCM: NON-DROP FRAME\n")
            rec = 0.0
            for k, s in enumerate(p["segments"], 1):
                d = s["end_s"] - s["start_s"]
                f.write(f"{k:03d}  EP{s['episode']:03d}  V  C  {tc(s['start_s'])} {tc(s['end_s'])} {tc(rec)} {tc(rec + d)}\n")
                rec += d
            f.write("\n")
    write_markdown(result, outp / "promos.md")
    write_html(result, show, outp / "report.html")
    return result


def main():
    ap = argparse.ArgumentParser(description="Find ranked promo clips in timestamped microdrama screenplays.")
    ap.add_argument("--data", required=True, help="folder of episode .txt files, a .zip, or one .txt")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--llm", default=None,
                    help="optional judge, e.g. ollama:llama3.1 (local), gemini:gemini-3.8-flash, groq:llama-3.3-70b-versatile, "
                         "openai:gpt-4o-mini, anthropic:claude-sonnet-4-5, ollama:llama3.1, mock")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--name", default=None)
    ap.add_argument("--lexicon", default=None, help="optional JSON overriding trope word lists, e.g. config/signals.json")
    ap.add_argument("--show-config", default=None, help="optional per-show JSON: payoff_terms, speaker_aliases, early_episodes")
    a = ap.parse_args()
    try:
        r = run(a.data, a.out, a.llm, a.top, a.name, a.lexicon, a.show_config)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"Error: {e}")
    print(json.dumps(r["stats"], indent=1))
    for p in r["promos"][:10]:
        segs = " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in p["segments"])
        print(f"#{p['rank']:>2} q={p['quality_score']:>5} prio={p['posting_priority']:>5} {p['lane']:<8} {p['total_duration_s']:>5}s  {segs}")


if __name__ == "__main__":
    main()
