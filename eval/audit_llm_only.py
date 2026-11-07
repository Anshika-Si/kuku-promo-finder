"""Audit an LLM-only promo list (timestamps written by the model) against the real screenplay.
Usage: python eval/audit_llm_only.py <scripts folder> <llm_only_promos.json>
The promo list is show-specific (confidential), so it lives with the raw outputs, not in this repo."""
import json, sys
sys.path.insert(0, ".")
from promo_finder.parser import load_show, fmt
from promo_finder.promos import Show, Promo, Segment, cut_points, speech_spans


def main(data, promos_path):
    LLM_ONLY = json.load(open(promos_path))["promos"]
    eps = load_show(data)
    show = Show(eps)
    rows = []
    for pr in LLM_ONLY:
        mid = []
        for ep, a, b in pr["segs"]:
            dl = speech_spans(show.eps[ep])
            for t, side in ((a, "in"), (b, "out")):
                for d in dl:
                    if d.start + 0.05 < t < d.end - 0.05:
                        mid.append(f"Ep{ep} {side}-point {fmt(t)} cuts inside \"{d.text[:40]}\"")
        p = Promo("llm", "stitched" if len(pr["segs"]) > 1 else "continuous", [Segment(*s) for s in pr["segs"]])
        show.score(p)
        real_len = round(p.duration, 1)
        rows.append({"name": pr["name"], "llm_rank": pr["llm_rank"], "segments": len(pr["segs"]), "length_s": real_len,
                     "under_20s": real_len < 20, "mid_line_cuts": mid, "our_score": p.final, "lane": p.lane,
                     "ending": p.cliff.get("text", "")[:90], "ending_type": p.cliff.get("type")})
    print(json.dumps(rows, indent=1))
    return rows

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
