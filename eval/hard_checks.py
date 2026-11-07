"""Hard, binary checks on every promo in an output file: length, legal cut points, no split lines, stamps exist."""
import json, sys
sys.path.insert(0, ".")
from promo_finder.parser import load_show
from promo_finder.promos import cut_points, speech_spans


def main(out_json, data):
    res = json.load(open(out_json))
    eps = {e.number: e for e in load_show(data)}
    rows, ok_all = [], True
    for p in res["promos"]:
        dur_ok = 20.0 - 1e-6 <= p["total_duration_s"] <= 60.0 + 1e-6
        problems = []
        for s in p["segments"]:
            e = eps[s["episode"]]
            starts, ends = cut_points(e)
            if round(s["start_s"], 2) not in starts:
                problems.append(f"Ep{s['episode']} start {s['start']} not a legal in-point")
            if round(s["end_s"], 2) not in ends:
                problems.append(f"Ep{s['episode']} end {s['end']} not a legal out-point")
            for d in speech_spans(e):
                for t in (s["start_s"], s["end_s"]):
                    if d.start + 0.05 < t < d.end - 0.05:
                        problems.append(f"Ep{s['episode']} {t} splits a line")
            if not (0 <= s["start_s"] < s["end_s"] <= e.duration + 1e-6):
                problems.append("timestamp outside episode")
        ok = dur_ok and not problems
        ok_all &= ok
        rows.append({"rank": p["rank"], "duration_ok": dur_ok, "problems": problems, "pass": ok})
    # every candidate window generated (not just the 25 returned) must also avoid splitting a spoken line
    from promo_finder.promos import Show
    show = Show(list(eps.values()))
    n_cand = n_split = 0
    for num in sorted(eps):
        spans = speech_spans(eps[num])
        for c in show.continuous_candidates(num) + show.stitched_candidates(num):
            n_cand += 1
            for seg in c.segments:
                sp = spans if seg.episode == num else speech_spans(eps[seg.episode])
                if any(d.start + 0.05 < t < d.end - 0.05 for d in sp for t in (seg.start, seg.end)):
                    n_split += 1
                    break
    summary = {"promos_checked": len(rows), "all_pass": ok_all, "passed": sum(r["pass"] for r in rows),
               "all_candidates_checked": n_cand, "candidates_splitting_a_line": n_split, "detail": rows}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
