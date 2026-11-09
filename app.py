"""Streamlit front-end: upload a show's screenplays (zip or .txt files) -> ranked promos in seconds.

Run locally:  streamlit run app.py
Deploy free:  Streamlit Community Cloud -> point at this repo -> main file app.py
No scripts are bundled with the app (the input pack is confidential); users upload them at run time.
"""
import base64
import html
import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from promo_finder.parser import load_texts, load_zip_bytes
from promo_finder.promos import find_promos, promo_to_dict
from promo_finder.report import heatmap_svg

st.set_page_config(page_title="Promo Finder · microdrama", page_icon="🎬", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.4rem}
.card{border:1px solid #e6e2d8;border-radius:10px;padding:10px 14px;margin-bottom:10px;background:#fff;color:#1a1a1a}
.sc{background:#e4572e;color:#fff;border-radius:6px;padding:1px 8px;font-weight:700}
.why{color:#2f9e6a;font-style:italic}.fl{color:#b8860b;font-size:12px}</style>""", unsafe_allow_html=True)

st.title("🎬 Promo Finder")
st.caption("Reads timestamped microdrama screenplays and returns ranked 20–60s promo cuts: hook in the first 3s, "
           "cliffhanger at the out-point, makes sense to a stranger. Code owns every timestamp; an optional LLM only re-scores.")

with st.sidebar:
    st.header("1 · Input")
    up = st.file_uploader("Episode scripts (.zip or several .txt)", type=["zip", "txt"], accept_multiple_files=True)
    # Reading a server-side path is only safe on your own machine, so it is opt-in:
    #   PROMO_ALLOW_LOCAL_PATH=1 streamlit run app.py
    # On a hosted instance the field is hidden and only uploads are accepted.
    local = None
    if os.environ.get("PROMO_ALLOW_LOCAL_PATH") == "1":
        local = st.text_input("…or a local folder / .zip path (local runs only)", value=os.environ.get("PROMO_DATA", ""))
    st.header("2 · Judge model (optional)")
    st.caption("Off by default. Confidential scripts should only go to an approved endpoint (local Ollama or a Kuku-approved API).")
    mode = st.selectbox("Re-score the shortlist with", ["Rules only (free, instant)", "gemini:gemini-3.8-flash",
                                                        "groq:llama-3.3-70b-versatile", "openai:gpt-4o-mini",
                                                        "anthropic:claude-sonnet-4-5", "ollama:llama3.1 (local)", "custom (OpenAI-compatible URL)", "mock (offline demo)"])
    key = base = model = None
    if mode.startswith(("gemini", "groq", "openai", "anthropic")):
        key = st.text_input("API key (kept in this session only)", type="password")
    if mode.startswith("custom"):
        base = st.text_input("Base URL", placeholder="https://your-gateway/v1")
        model = st.text_input("Model name")
        key = st.text_input("API key", type="password")
    st.header("3 · Settings")
    top_n = st.slider("Promos to return", 5, 40, 20)
    early = st.slider("Episodes before first paywall (assumed)", 4, 20, 12,
                      help="Promos from these episodes get full funnel weight: a new viewer reaches the payoff soon after install.")
    with st.expander("Rubric weights"):
        w_hook = st.slider("Hook (first 3s)", 0.0, 0.6, 0.30, 0.05)
        w_cliff = st.slider("Cliffhanger", 0.0, 0.6, 0.25, 0.05)
        w_cl = st.slider("Stands alone", 0.0, 0.6, 0.20, 0.05)
        w_es = st.slider("Escalation", 0.0, 0.6, 0.15, 0.05)
        w_tr = st.slider("Trope legibility", 0.0, 0.6, 0.10, 0.05)
    go = st.button("Find promos", type="primary", use_container_width=True)

if not up and not local:
    st.warning("Public demo: upload synthetic or public test scripts only. For Kuku's confidential pack, run this app locally "
               "(`streamlit run app.py`) or on hosting Kuku has approved.")
    st.info("Upload a show's episode scripts to start. Format: one .txt per episode with `Start: (MM:SS.ss) | End: (MM:SS.ss)` "
            "blocks for CAMERA / ACTION / SOUND FX / CHARACTER lines (the Kuku screenplay format).")
    st.stop()

if go:
    texts = []
    if local and not up and os.environ.get("PROMO_ALLOW_LOCAL_PATH") == "1":
        from promo_finder.parser import load_show
        eps, texts = load_show(local), None
    for f in (up or []):
        data = f.read()
        if f.name.lower().endswith(".zip"):
            eps = load_zip_bytes(data)
            texts = None
            break
        texts.append((f.name, data.decode("utf-8", errors="replace")))
    if texts is not None:
        eps = load_texts(texts)
    if not eps:
        st.error("No parsable episodes found. Check the file format.")
        st.stop()
    judge = None
    if not mode.startswith("Rules"):
        from promo_finder.judge import LLMJudge
        # the key and URL live only on this session's judge object, never in process-wide environment variables
        if mode.startswith("mock"):
            judge = LLMJudge.from_spec("mock")
        elif mode.startswith("custom"):
            judge = LLMJudge.from_spec(f"custom:{model}", api_key=key or "", base_url=base or "")
        else:
            judge = LLMJudge.from_spec(mode.split(" ")[0], api_key=key or "")  # drop UI labels like "(local)"
    cfg = {"top_n": top_n, "early_episodes": early,
           "weights": {"hook": w_hook, "cliff": w_cliff, "clarity": w_cl, "escalation": w_es, "trope": w_tr}}
    bar = st.progress(0.0, text="Scoring every legal 20–60s cut…")
    show, ranked, stats = find_promos(eps, cfg, judge=judge, progress=lambda f: bar.progress(f, text="Scoring every legal 20–60s cut…"))
    bar.empty()
    st.session_state["res"] = (show, [promo_to_dict(p, i + 1) for i, p in enumerate(ranked)], stats, judge.usage if judge else None)

if "res" in st.session_state:
    show, promos, stats, usage = st.session_state["res"]
    c = st.columns(5)
    c[0].metric("Episodes", stats["episodes"])
    c[1].metric("Screen time", f"{stats['runtime_min']} min")
    c[2].metric("Timed beats parsed", stats["beats"])
    c[3].metric("Candidate windows scored", f"{stats['candidate_windows_scored']:,}")
    c[4].metric("Promos returned", len(promos))
    if usage:
        st.caption(f"LLM judge: {usage}")
    st.subheader("Where the heat is — every line and action across the show (darker = more scroll-stopping), top-10 boxed")
    # shown as an <img> so Streamlit's HTML sanitiser can't strip the SVG; white card so it reads in dark mode too
    svg64 = base64.b64encode(heatmap_svg(show, promos).encode("utf-8")).decode("ascii")
    st.markdown(f"<div style='background:#fff;border-radius:10px;padding:8px'><img alt='heat map' style='width:100%' "
                f"src='data:image/svg+xml;base64,{svg64}'></div>", unsafe_allow_html=True)

    st.subheader("Ranked promos — post from the top")
    lane = st.radio("Lane", ["All", "cold_ua", "retarget"], horizontal=True)
    def e(v):  # script text is user-uploaded: escape it before it goes into any HTML we render
        return html.escape(str(v or ""))

    for p in promos:
        if lane != "All" and p["lane"] != lane:
            continue
        segs = " + ".join(f"Ep {s['episode']} **{s['start']}→{s['end']}**" for s in p["segments"])
        ending = p["cut_ending"].get("last_line") or p["cut_ending"].get("last_action_or_sound", "")
        with st.container():
            st.markdown(f"<div class='card'><span class='sc'>#{p['rank']} · quality {p['quality_score']}</span> &nbsp;<b>{e(p['title'])}</b> "
                        f"· {e(p['type'])} · {p['total_duration_s']}s · {e(p['lane'])}</div>", unsafe_allow_html=True)
            l, r = st.columns([3, 2])
            l.markdown(f"{segs}  \n**Hook** (+{p['hook'].get('seconds_in', 0)}s): {e(p['hook']['text'])}  \n"
                       f"**Cliffhanger beat** ({e(p['cliffhanger_beat'].get('type'))}): {e(p['cliffhanger_beat']['text'])}  \n"
                       f"**Cut ending** ({e(p['cut_ending'].get('out_point'))}): {e(ending)}  \n"
                       f"<span class='why'>{e(p['why'])}</span>", unsafe_allow_html=True)
            sc = p["scores"]
            r.progress(min(1.0, sc["hook"]), text=f"Hook {sc['hook']:.2f}")
            r.progress(min(1.0, sc["cliff"]), text=f"Cliffhanger {sc['cliff']:.2f}")
            r.progress(min(1.0, sc["clarity"]), text=f"Stands alone {sc['clarity']:.2f}")
            if p.get("llm_judge"):
                r.caption(f"LLM: {p['llm_judge'].get('cold_viewer_question')} — {p['llm_judge'].get('why')}")
            if p.get("tiktok_cut"):
                r.caption("TikTok 21–34s: " + " + ".join(f"Ep{s['episode']} {s['start']}→{s['end']}" for s in p["tiktok_cut"]))
            r.caption(" · ".join(p["flags"]))
    st.download_button("Download promos.json", json.dumps({"stats": stats, "promos": promos}, indent=1),
                       file_name="promos.json", mime="application/json")
