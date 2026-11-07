"""Human-readable outputs: Markdown table + a self-contained HTML report with a show-wide heat map."""
from __future__ import annotations

import html
from pathlib import Path


def write_markdown(result: dict, path: Path):
    L = [f"# Promo shortlist — {result['show']}", "",
         f"Mode: {result['stats']['mode']} · {result['stats']['episodes']} episodes · "
         f"{result['stats']['runtime_min']} min of screen time · {result['stats']['candidate_windows_scored']} candidate windows scored "
         f"in {result['stats']['seconds']}s", "",
         "| # | Quality | Priority | Lane | Len | Segments (episode start–end) | Hook (first 3s) | Cliffhanger beat | Cut ending (last line at out-point) | Why it works | Flags |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in result["promos"]:
        segs = "<br>".join(f"Ep{s['episode']} {s['start']}–{s['end']}" for s in p["segments"])
        L.append(f"| {p['rank']} | {p['quality_score']} | {p['posting_priority']} | {p['lane']} | {p['total_duration_s']}s | {segs} | "
                 f"{p['hook']['text'].replace('|', '/')} | {p['cliffhanger_beat']['text'].replace('|', '/')} "
                 f"({p['cliffhanger_beat'].get('type', '')}) | {p['cut_ending'].get('last_line', '').replace('|', '/')} | "
                 f"{p['why']} | {'; '.join(p['flags'])} |")
    if result.get("retarget_promos"):
        L += ["", "## Retargeting shortlist (payoff clips, for people already watching)", "",
              "| # | Quality | Len | Segments | Ends on |", "|---|---|---|---|---|"]
        for p in result["retarget_promos"]:
            segs = "<br>".join(f"Ep{s['episode']} {s['start']}–{s['end']}" for s in p["segments"])
            L.append(f"| R{p['rank']} | {p['quality_score']} | {p['total_duration_s']}s | {segs} | "
                     f"{(p['cut_ending'].get('last_line') or p['cut_ending'].get('last_action_or_sound', '')).replace('|', '/')} |")
    Path(path).write_text("\n".join(L), encoding="utf-8")


PALETTE = ["#2a6f97", "#c0392b", "#d68910", "#7d3c98", "#1e8449", "#a04000", "#1f618d", "#b03a2e", "#117a65", "#6c3483"]


def heatmap_svg(show, promos: list[dict], width=1100, row_h=11, label_w=46, max_ep_len=None):
    eps = sorted(show.eps)
    max_len = max_ep_len or max(show.eps[n].duration for n in eps)
    W = width - label_w - 10
    H = len(eps) * row_h + 30
    sx = lambda t: label_w + t / max_len * W
    out = [f'<svg viewBox="0 0 {width} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Inter,Arial" font-size="8">']
    for t in range(0, int(max_len) + 1, 30):
        out.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="10" y2="{H-14}" stroke="#e5e5e5"/>'
                   f'<text x="{sx(t):.1f}" y="{H-3}" text-anchor="middle" fill="#888">{t//60}:{t%60:02d}</text>')
    for i, n in enumerate(eps):
        y = 12 + i * row_h
        out.append(f'<text x="{label_w-6}" y="{y+8}" text-anchor="end" fill="#666">Ep {n}</text>')
        out.append(f'<rect x="{sx(0):.1f}" y="{y}" width="{sx(show.eps[n].duration)-sx(0):.1f}" height="{row_h-2}" fill="#f3f1ec"/>')
        for s in show.sigs[n]:
            if s.beat.kind == "camera" or s.intensity < 0.35:
                continue
            a = min(1.0, s.intensity / 2.2)
            out.append(f'<rect x="{sx(s.beat.start):.1f}" y="{y}" width="{max(1.2, sx(s.beat.end)-sx(s.beat.start)):.1f}" '
                       f'height="{row_h-2}" fill="#e4572e" fill-opacity="{a:.2f}"/>')
    for p in promos[:10]:
        col = "#111"
        for s in p["segments"]:
            i = eps.index(s["episode"])
            y = 12 + i * row_h
            out.append(f'<rect x="{sx(s["start_s"]):.1f}" y="{y-1}" width="{sx(s["end_s"])-sx(s["start_s"]):.1f}" '
                       f'height="{row_h}" fill="none" stroke="{col}" stroke-width="1.3" rx="2"/>')
        s0 = p["segments"][-1]
        i = eps.index(s0["episode"])
        out.append(f'<text x="{sx(s0["end_s"])+3:.1f}" y="{12+i*row_h+8}" font-weight="700" fill="#111">#{p["rank"]}</text>')
    out.append("</svg>")
    return "".join(out)


def write_html(result: dict, show, path: Path):
    promos = result["promos"]
    cards = []
    for p in promos:
        segs = "".join(f"<li>Ep {s['episode']} · <b>{s['start']} → {s['end']}</b> ({s['duration_s']}s)</li>" for s in p["segments"])
        bars = "".join(
            f'<div class="bar"><span>{k}</span><i style="width:{min(100, v*100):.0f}%"></i><em>{v:.2f}</em></div>'
            for k, v in p["scores"].items() if k in ("hook", "cliff", "clarity", "escalation", "trope"))
        tt = ""
        if p.get("tiktok_cut"):
            tt = "TikTok 21–34s cut: " + " + ".join(f"Ep{s['episode']} {s['start']}→{s['end']}" for s in p["tiktok_cut"])
        llm = ""
        if p.get("llm_judge"):
            j = p["llm_judge"]
            llm = f"<p class='llm'><b>LLM judge:</b> {html.escape(str(j.get('cold_viewer_question', '')))} — {html.escape(str(j.get('note', '')))}</p>"
        cards.append(f"""
<div class="card {'retarget' if p['lane']=='retarget' else ''}">
 <div class="hd"><span class="rk">#{p['rank']}</span><span class="sc">{p['quality_score']}</span>
 <span class="tt">{html.escape(p['title'])}</span><span class="pill">{p['type']} · {p['total_duration_s']}s · {p['lane']}</span></div>
 <ol>{segs}</ol>
 <p><b>Hook</b> (+{p['hook'].get('seconds_in', 0)}s): {html.escape(p['hook']['text'])}</p>
 <p><b>Cliffhanger beat</b> ({html.escape(p['cliffhanger_beat'].get('type', ''))}): {html.escape(p['cliffhanger_beat']['text'])}</p>
 <p><b>Cut ending</b> (out-point {p['cut_ending'].get('out_point', '')}): {html.escape(p['cut_ending'].get('last_line', '') or p['cut_ending'].get('last_action_or_sound', ''))}</p>
 <p class="why">{html.escape(p['why'])}</p>{llm}
 <div class="bars">{bars}</div>
 <p class="fl">{' · '.join(html.escape(f) for f in p['flags'])}<br>{tt}</p>
</div>""")
    rt = result.get("retarget_promos") or []
    retarget_html = ""
    if rt:
        rows = "".join(f"<li><b>R{p['rank']}</b> · quality {p['quality_score']} · {p['total_duration_s']}s · "
                       + " + ".join(f"Ep {s['episode']} {s['start']}→{s['end']}" for s in p["segments"])
                       + f" — ends on: {html.escape(p['cut_ending'].get('last_line') or p['cut_ending'].get('last_action_or_sound', ''))}</li>"
                       for p in rt)
        retarget_html = f"<h2 style='font-size:16px;margin-top:18px'>Retargeting shortlist: payoff clips for people already watching</h2><ol style='font-size:12.5px'>{rows}</ol>"
    st = result["stats"]
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Promo report — {html.escape(result['show'])}</title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;margin:24px;color:#1b1b1b;background:#fbfaf7}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{color:#666;font-size:13px;margin-bottom:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #e6e2d8;border-radius:10px;padding:12px 14px;font-size:12.5px}}
.card.retarget{{border-left:4px solid #7d3c98}}
.hd{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}} .rk{{font-weight:800;font-size:18px}}
.sc{{background:#e4572e;color:#fff;border-radius:6px;padding:1px 7px;font-weight:700}}
.tt{{font-weight:600}} .pill{{margin-left:auto;color:#555;font-size:11px}}
ol{{margin:6px 0 6px 18px;padding:0}} p{{margin:4px 0}} .why{{color:#0b5d3b;font-style:italic}}
.bars{{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin-top:6px}}
.bar{{font-size:10px;color:#555}} .bar i{{display:block;height:5px;background:#e4572e;border-radius:3px}}
.bar em{{font-style:normal}} .fl{{color:#8a6d00;font-size:11px}} .llm{{color:#2a4d9b}}
.heat{{background:#fff;border:1px solid #e6e2d8;border-radius:10px;padding:8px;margin-bottom:14px}}
</style></head><body>
<h1>Promo shortlist — {html.escape(result['show'])}</h1>
<div class="sub">{st['mode']} · {st['episodes']} episodes · {st['runtime_min']} min · {st['candidate_windows_scored']} candidate windows scored in {st['seconds']}s ·
heat = scroll-stopping intensity of each line/action (darker = hotter); boxes = top-10 promos.
scores.rules is the weighted rubric after length-fit and dead-air adjustments; quality_score additionally applies the early-episode (funnel) multiplier; posting priority applies multiplicative diversity adjustments for repeated tropes (×0.96 each) and shared episodes (×0.90 each), choosing only among candidates within 10 quality points of the best remaining clip, and sets the posting order.</div>
<div class="heat">{heatmap_svg(show, promos)}</div>
<div class="grid">{''.join(cards)}</div>
{retarget_html}
</body></html>"""
    Path(path).write_text(doc, encoding="utf-8")
