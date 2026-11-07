"""Offline quality check: tool vs. hand-labelled gold moments vs. simple baselines.

Usage: python -m eval.evaluate outputs/promos.json path/to/gold_moments.json
(gold labels are show-specific and confidential, so they live with the raw outputs, not in this repo)
A promo 'hits' a gold moment when, in the same episode, its segments overlap the gold stretch by at
least half of the shorter of (promo length, gold length), or by 12+ seconds
(so a stitched promo that clearly sells the moment through one segment counts).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent


def load_gold(path):
    return json.loads(Path(path).read_text())["moments"]


STRICT = False   # strict rule = only "overlap >= half of the shorter interval" (no 12 s allowance)


def hits(segs, g):
    inter = sum(max(0.0, min(s["end_s"], g["end"]) - max(s["start_s"], g["start"]))
                for s in segs if s["episode"] == g["ep"])
    plen = sum(s["end_s"] - s["start_s"] for s in segs)
    half = inter >= 0.5 * min(plen, g["end"] - g["start"])
    return half if STRICT else (half or inter >= 12.0)


def metrics(promo_lists, gold, ks=(5, 10, 25)):
    """promo_lists: ranked list of segment-lists."""
    out = {}
    for k in ks:
        top = promo_lists[:k]
        prec = sum(any(hits(p, g) for g in gold) for p in top) / max(len(top), 1)
        rec = sum(any(hits(p, g) for p in top) for g in gold) / len(gold)
        tier_a = [g for g in gold if g["tier"] == "A"]
        rec_a = sum(any(hits(p, g) for p in top) for g in tier_a) / len(tier_a)
        out[k] = {"precision": round(prec, 3), "recall": round(rec, 3), "recall_tierA": round(rec_a, 3)}
    return out


def baseline_episode_endings(durations, length=30.0, n=25, seed=0):
    """'Just post episode cliffhangers': the last 30 s of n episodes picked at random from the WHOLE show
    (averaged over 300 seeds), so it is compared top-K for top-K with the tool, not limited to Ep 1-10."""
    rnd = random.Random(seed)
    eps = rnd.sample(sorted(durations), min(n, len(durations)))
    return [[{"episode": ep, "start_s": max(0, durations[ep] - length), "end_s": durations[ep]}] for ep in eps]


def mean_metrics(make, gold, seeds=300):
    runs = [metrics(make(s), gold) for s in range(seeds)]
    return {k: {m: round(sum(r[k][m] for r in runs) / len(runs), 3) for m in runs[0][k]} for k in runs[0]}


def baseline_random(durations, n=25, length=30.0, seed=0):
    rnd = random.Random(seed)
    lists = []
    eps = sorted(durations)
    for _ in range(n):
        ep = rnd.choice(eps)
        a = rnd.uniform(0, max(0.1, durations[ep] - length))
        lists.append([{"episode": ep, "start_s": a, "end_s": a + length}])
    return lists


def main(path, gold_path):
    res = json.loads(Path(path).read_text())
    gold = load_gold(gold_path)
    tool = [p["segments"] for p in res["promos"]]
    durations = {int(k): v for k, v in res["episode_durations"].items()}
    global STRICT
    STRICT = True
    strict = {"tool": metrics(tool, gold),
              "baseline_episode_endings": mean_metrics(lambda s: baseline_episode_endings(durations, seed=s), gold),
              "baseline_random_mean": mean_metrics(lambda s: baseline_random(durations, seed=s), gold)}
    STRICT = False
    report = {"hit_rule": "overlap >= half of the shorter interval, OR >= 12 s (lenient; 12 s clause added after the first run "
                          "so stitched promos get credit)",
              "strict_rule_results": strict,
              "tool": metrics(tool, gold),
              "baseline_episode_endings": mean_metrics(lambda s: baseline_episode_endings(durations, seed=s), gold),
              "baseline_random_mean": mean_metrics(lambda s: baseline_random(durations, seed=s), gold),
              "baselines_note": "both baselines are means over 300 seeds; episode endings are sampled across all 50 episodes"}
    # per-gold coverage detail
    detail = []
    for g in gold:
        ranks = [i + 1 for i, p in enumerate(tool) if hits(p, g)]
        detail.append({"id": g["id"], "ep": g["ep"], "tier": g["tier"], "label": g["label"],
                       "found_at_rank": ranks[0] if ranks else None})
    report["gold_detail"] = detail
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
