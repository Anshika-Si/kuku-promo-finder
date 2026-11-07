# Promo Finder: ranked promo cuts from microdrama screenplays

Reads a show's timestamped episode scripts (Kuku format) and returns **ranked 20–60s promo cuts**. Each one opens on a hook in the first 3 seconds, ends on an unresolved cliffhanger, and makes sense to someone who has never seen the show. Every promo comes with episode, in/out timestamps per segment, the hook, the cliffhanger beat, what actually happens at the out-point (`cut_ending`), a one-line "why", a quality score and posting priority, a 21–34 s TikTok variant when one exists without stretching the clip, and editor flags.

*50 episodes (102.7 min) → 3,727 timed beats → 6,438 candidate windows scored → top 25. Parsing and scoring took 2.49 s on a CPU in the recorded run (`stats.seconds`, see `run_log.txt`); writing the JSON/CSV/EDL/Markdown/HTML outputs adds a little on top, depending on the machine. No API key needed.*

## Quick start
```bash
pip install -r requirements.txt          # the core engine needs only the standard library; this adds Streamlit
python -m promo_finder.cli --data path/to/show_folder_or.zip --out outputs/
# optional per-show settings (payoff phrases, speaker aliases, paywall episode): --show-config show_config.json
streamlit run app.py                      # UI: upload a .zip or the .txt files (PROMO_ALLOW_LOCAL_PATH=1 also shows a local-path field)
```
Outputs in `outputs/`:

| File | What it is |
|---|---|
| `promos.json` | everything: segments, hook, `cliffhanger_beat` (strongest unresolved beat near the end), `cut_ending` (literal last line/action at the out-point), why, `quality_score`, `posting_priority`, sub-scores, flags, TikTok cut |
| `promos_segments.csv` | one row per segment, for an editor's sheet |
| `promos.edl` | CMX3600-style edit list (episode = reel, 30 fps) to drop into an NLE |
| `promos.md` | readable ranked table |
| `report.html` | heat map of the whole show with the top-10 boxed, plus promo cards |

## Confidentiality
The input scripts are confidential. The submitted full-pack run used local, deterministic processing only. The optional LLM judge is **off by default** and sends candidate transcripts to whichever model you choose. Only enable it with an endpoint approved for confidential material: a local model (`--llm ollama:...`) or a Kuku-provided/approved API. Free consumer tiers (e.g. the Gemini free tier) may use submitted content to improve their products. Do not commit scripts or outputs to a public repo (`.gitignore` blocks `data/`, `*.txt` and `outputs/`).

## Swap the model (optional LLM judge)
The rules pick and cut the promos. An LLM can then re-score the top 30 as a "cold viewer" with no show context. Changing the model is one string:
```bash
GEMINI_API_KEY=... python -m promo_finder.cli --data show.zip --llm gemini:gemini-3.8-flash   # paid tier only for confidential scripts
GROQ_API_KEY=...   python -m promo_finder.cli --data show.zip --llm groq:llama-3.3-70b-versatile
ANTHROPIC_API_KEY=... ... --llm anthropic:claude-sonnet-4-5      OPENAI_API_KEY=... ... --llm openai:gpt-4o-mini
LLM_BASE_URL=https://your-gateway/v1 LLM_API_KEY=... ... --llm custom:<model>   # any OpenAI-compatible API
... --llm ollama:llama3.1   (local)          ... --llm mock   (offline test)
```
Guard-rails:
- **The LLM never chooses a cut point.** The LLM receives candidate IDs and transcript text with relative beat offsets (e.g. +3.0s), but no source timecodes. Python owns and validates every source start/end point; the LLM only scores the candidates. It sees every beat that overlaps a clip, including camera directions and any action still running at the out-point (marked "CUT OFF here").
- **Strict JSON output.** Unknown IDs, missing fields and out-of-range scores are rejected and the batch is retried. If the retries fail, the rule score is kept.
- **Cost.** About 16.3k input and 3.3k output tokens per show (30 candidates, camera directions included). At Gemini 3.8 Flash's standard paid rate through 31 Dec 2026 ($0.75 / $3.75 per million input/output tokens) that is about $0.025 a show, or about $3.70 a month at 150 shows. Google lists $1.50 / $7.50 from 1 Jan 2027 (about $0.049 a show), and output pricing includes any thinking tokens, so treat this as an estimate. Long transcripts are trimmed in the middle, so the hook and the ending are always kept.

## How it works
1. **Parse.** Every CAMERA / ACTION / SOUND FX / dialogue block becomes a timed *beat*. Sentences that the script splits across two blocks ("I cannot..." / "...not you") are merged so they can't be cut.
2. **Tag (rules).** Each beat gets an intensity from emotion tags, trope lexicons (physical shock, reveal, humiliation, betrayal, threat, protector romance), SFX stings and camera cues. It also gets *unresolved* vs *resolution* cues for cliffhanger detection. The lexicons are editable in `config/signals.json`.
3. **Generate legal cuts.** In and out points snap to beat boundaries that never slice a spoken line. Continuous 20–60s windows are built between the best hook starts and the best cliff ends. *Stitched* cuts follow SocialPeta's ad formula: a 3s lure, up to 2 hot clips, then the run-up to a cliffhanger, spanning up to 2 episodes.
4. **Score.** The weights come from the research: hook 30%, cliffhanger 25%, stands alone 20%, escalation 15%, trope legibility 10%. The score is then multiplied by:
   - **length fit:** TikTok recommends 21–34s;
   - **dead air:** gaps of more than 4s with no line, action or sound;
   - **promise distance:** promos from episodes before the paywall rank higher.

   Clips that pay off the central mystery are moved to a **retarget** lane.
5. **Rank.** Overlapping cuts are de-duplicated, then re-ranked for variety. The variety step may only choose among clips within 10 quality points of the best remaining one, so a weak clip never jumps the queue. `scores.rules` is the weighted rubric after the length-fit and dead-air adjustments; `quality_score` additionally applies the early-episode (funnel) multiplier; `posting_priority` applies the variety adjustments and sets the final posting order. Rubric weights are normalised to sum to 1, so the sliders in the app can't push scores past 100.
6. *(optional)* **LLM judge** re-scores the top 30 and blends in at 50/50.

## Quality check
```bash
# the show-specific inputs (gold labels, the external LLM list) are confidential and ship with raw_outputs, not this repo
python -m eval.evaluate outputs/promos.json /path/to/raw_outputs/evaluation/gold_moments.json   # vs hand-labelled gold moments + baselines
python eval/audit_llm_only.py /path/to/show.zip /path/to/raw_outputs/evaluation/llm_only_promos.json   # checks an LLM-written promo list against the script
python eval/hard_checks.py outputs/promos.json path/to/show   # length, legal cuts, no split lines
python eval/funnel_ablation.py path/to/show         # top 10 with vs without episode weighting
pip install -r requirements-dev.txt && python -m pytest -q   # tests on a synthetic script (no confidential data)
```
On the 50-episode pack (one title, one annotator, exploratory), precision@10 is **0.80** with the default hit rule and **0.70** with the strict rule (≥ half of the shorter interval only). The baselines, each averaged over 300 seeds, score about 0.34 (the last 30 s of 10 episodes sampled across the whole show) and about 0.17 (random 30 s clips). Recall@25 is 0.68 (default) and 0.50 (strict). `python eval/funnel_ablation.py <pack>` shows how much the early-episode weighting changes the top 10.

## Deploy (free)
Push this repo to GitHub, then in Streamlit Community Cloud choose New app → this repo → `app.py`. **Do not commit the input pack**: it is confidential (`.gitignore` blocks `data/` and `*.txt`). Users upload scripts at run time.

The local folder/zip path field in the app is hidden unless you start it with `PROMO_ALLOW_LOCAL_PATH=1 streamlit run app.py`, so a hosted copy can never be asked to read files from its own server; hosted users can only upload.

**Confidential input restriction.** Streamlit Community Cloud deployment is intended for synthetic or public test inputs only. For Kuku's confidential screenplay pack, run the application locally (`streamlit run app.py` or the CLI) or use a hosting environment explicitly approved by Kuku. Do not upload the confidential input pack to a third-party hosted instance without authorization.

## Layout
```
promo_finder/  parser.py · signals.py · promos.py · judge.py · report.py · cli.py
app.py         Streamlit UI
eval/          evaluate.py · hard_checks.py · audit_llm_only.py · funnel_ablation.py
               (show-specific inputs such as gold labels live with the confidential raw outputs, not here)
tests/         test_pipeline.py
config/        signals.json (optional lexicon overrides)
```

## Known limits
- **Text only.** It scores the *script*, not the rendered video. An AI-generated shot can play weaker or stronger than written, so an editor still watches the top 10.
- **Lexicons have no common sense.** "Strikes flint" reads as violence. The LLM judge, or an editor, catches this.
- **No sympathy check yet.** It cannot tell whether a scene has a lead the viewer roots for, so villain-only scheming scenes can rank too high.
- **Assumed paywall.** It assumes the first paywall is by about episode 12. Set `early_episodes` in a show config from real unlock data.
- **Generic by default.** No show names or plot terms are hard-coded; anything show-specific goes in `--show-config`.

## Sources for the numbers in the write-up
- Kuku TV releases 150+ microdramas a month: [Business Standard, Oct 2026](https://www.business-standard.com/amp/companies/quarterly-results/kuku-revenue-jumps-sixfold-to-1-484-crore-as-ai-fuels-microdrama-expansion-126100300968_1.html)
- ~5,770 creatives per advertiser per month (2025), the "3-second lure + 3–7 high-impact clips + cliffhanger question" ad structure: [SocialPeta × AMO Pictures white paper, 2026](https://investgame.net/news/pdf/2026-04-07-socialpeta_x_amo_pictures_insight_into_global_micro_drama_app_marketing_wp/)
- ~400 creatives to test one drama, ~40 cuts per editor per day (China): [Huxiu, May 2024](https://www.huxiu.com/article/3035241.html)
- 68% of US microdrama ad spend went to social networks, Jan–Sep 2025: [eMarketer / Sensor Tower](https://www.emarketer.com/content/microdramas-bank-on-social-ads-drive-growth)
- TikTok In-Feed ads: 5–60 s allowed, 21–34 s recommended: [TikTok for Business, "7 Top Creative Tips"](https://ads.tiktok.com/business/library/7TopCreativeTips.pdf); the first 3–10 s are where viewers are lost: [TikTok for Business blog](https://ads.tiktok.com/business/es/blog/how-tiktok-is-changing-the-way-brands-approach-videos)
- Five recurring hook types across 500+ short-drama creatives: [AppGrowing, 2026](https://appgrowing.net/blog/en/how-to-find-high-priority-short-drama-app-ad-hooks-from-competitor-data/)
- Median Meta hook rate ~20.9% (479 ads): [Biddy](https://www.biddyco.com/blog-posts/hook-rate-benchmarks-meta)
- First paywall usually around Ep 8–16 in 80–100-episode short dramas: [juduanduan, Oct 2025](https://www.juduanduan.com/2025/1027/3193.shtml)
- Gemini 3.8 Flash standard paid-tier price $0.75 / $3.75 per million input/output tokens through 31 Dec 2026, $1.50 / $7.50 from 1 Jan 2027: [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing). The 2.5 models are now limited to accounts that already used them: [Gemini deprecations](https://ai.google.dev/gemini-api/docs/deprecations)

