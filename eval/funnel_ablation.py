"""Sensitivity check: how much does the 'early episodes first' business assumption change the ranking?
Version A = default (promise-distance weight on). Version B = no episode-based weighting."""
import json, sys
sys.path.insert(0, ".")
from promo_finder.parser import load_show
from promo_finder.promos import find_promos


def top(eps, cfg):
    _, ranked, _ = find_promos(eps, cfg)
    return [{"rank": i + 1, "episodes": p.episodes, "segments": [(s.episode, round(s.start, 2), round(s.end, 2)) for s in p.segments],
             "quality": p.final} for i, p in enumerate(ranked[:10])]


def main(data):
    eps = load_show(data)
    a = top(eps, {})
    b = top(eps, {"early_episodes": 99, "funnel_floor": 1.0})   # flat weight: every episode treated alike
    ka = {tuple(x["segments"]) for x in a}
    kb = {tuple(x["segments"]) for x in b}
    out = {"A_default_top10": a, "B_no_episode_weight_top10": b,
           "shared_in_both_top10": len(ka & kb),
           "entered_top10_without_weighting": [x for x in b if tuple(x["segments"]) not in ka],
           "max_episode_A": max(max(x["episodes"]) for x in a), "max_episode_B": max(max(x["episodes"]) for x in b),
           "episodes_after_15_A": sum(max(x["episodes"]) > 15 for x in a),
           "episodes_after_15_B": sum(max(x["episodes"]) > 15 for x in b)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
