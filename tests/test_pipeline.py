"""Smoke tests. Run: python -m pytest -q  (uses a tiny synthetic script, no confidential data needed)."""
from promo_finder.parser import parse_episode, load_texts
from promo_finder.promos import find_promos, cut_points
from promo_finder.judge import LLMJudge

EP = """EPISODE NUMBER: Episode 1
EPISODE LOGLINE: A bride is abandoned at the altar.
--- SCRIPT START ---
SCENE 01
Scene heading: INT. CHURCH - DAY
Scene start: 00:00.00
Scene end: 00:40.00
**CAMERA:**
Start: (00:00.00) | End: (00:02.00)
Wide establishing shot of the church.
**ANNA (Shocked, Desperate):**
Start: (00:01.00) | End: (00:03.00)
"Why is he holding my sister's hand?!"
**ACTION:**
Start: (00:03.00) | End: (00:05.00)
Her sister slaps her across the face.
**SOUND FX:**
Start: (00:03.20) | End: (00:03.60)
A sharp slap.
**SISTER (Cruel, Mocking):**
Start: (00:06.00) | End: (00:09.00)
"Because he is marrying me..."
**SISTER (Gloating):**
Start: (00:09.20) | End: (00:11.00)
"...not you."
**GROOM (Cold, Threatening):**
Start: (00:12.00) | End: (00:16.00)
"Guards! Take her away or she hangs at dawn."
**ANNA (Furious):**
Start: (00:17.00) | End: (00:20.00)
"You will regret this!"
**ACTION:**
Start: (00:20.00) | End: (00:25.00)
The doors burst open and a scarred stranger steps toward the altar.
**STRANGER (Deadly):**
Start: (00:26.00) | End: (00:29.00)
"Who dares touch my wife?"
**CAMERA:**
Start: (00:29.00) | End: (00:30.00)
Cut to black.
--- SCRIPT END ---
"""


def test_parse_and_cuts():
    ep = parse_episode(EP)
    assert ep.number == 1 and len(ep.beats) == 11 and abs(ep.duration - 40.0) < 1e-6
    starts, ends = cut_points(ep)
    assert 9.2 not in starts           # never start on the second half of a split sentence
    assert 9.0 not in ends             # never end on the first half
    assert all(not (6.0 < t < 11.0) for t in starts)


def test_find_promos_rules_and_mock_llm():
    eps = load_texts([("e1.txt", EP), ("e2.txt", EP.replace("Episode 1", "Episode 2"))])
    show, ranked, stats = find_promos(eps, {"top_n": 3})
    assert ranked and all(20 <= p.duration <= 60 for p in ranked)
    show, ranked2, _ = find_promos(eps, {"top_n": 3}, judge=LLMJudge.from_spec("mock"))
    assert all(p.llm for p in ranked2)


def test_tiktok_variant_never_shorter_than_21s():
    eps = load_texts([("e1.txt", EP), ("e2.txt", EP.replace("Episode 1", "Episode 2"))])
    show, ranked, _ = find_promos(eps, {"top_n": 5})
    for p in ranked:
        if p.tiktok_cut:
            assert 21.0 <= sum(s.dur for s in p.tiktok_cut) <= 34.0 + 1e-6


def test_cut_ending_is_the_real_last_line():
    eps = load_texts([("e1.txt", EP)])
    show, ranked, _ = find_promos(eps, {"top_n": 3})
    for p in ranked:
        seg = p.segments[-1]
        lines = [b for b in show.eps[seg.episode].beats if b.kind == "dialogue" and seg.start - 0.05 <= b.start and b.end <= seg.end + 0.05]
        if lines:
            assert lines[-1].text in p.cut_end["last_line"]


def test_llm_transcript_keeps_the_ending():
    from promo_finder.judge import transcript
    eps = load_texts([("e1.txt", EP)])
    show, ranked, _ = find_promos(eps, {"top_n": 1})
    t = transcript(ranked[0], max_chars=200)
    assert "middle trimmed" in t and t.rstrip().endswith(transcript(ranked[0], max_chars=10**6).rstrip()[-60:])


def test_llm_transcript_sees_action_cut_off_at_out_point():
    """A clip that ends mid-action must still show that action (and camera beats) to the LLM judge."""
    from promo_finder.judge import transcript
    from promo_finder.promos import Promo, Segment, Show
    eps = load_texts([("e1.txt", EP)])
    show = Show(eps)
    p = Promo("t", "continuous", [Segment(1, 0.0, 22.0)])   # the 'doors burst open' action runs 20-25 s
    show.score(p)
    t = transcript(p, max_chars=10**6, show=show)
    assert "doors burst open" in t and "CUT OFF here" in t
    assert "[CAMERA] Wide establishing shot" in t


def test_weights_are_normalised_and_zero_rejected():
    import pytest
    from promo_finder.promos import Show
    eps = load_texts([("e1.txt", EP)])
    s = Show(eps, {"weights": {"hook": .6, "cliff": .6, "clarity": .6, "escalation": .6, "trope": .6}})
    assert abs(sum(s.cfg["weights"].values()) - 1.0) < 1e-9
    with pytest.raises(ValueError):
        Show(eps, {"weights": {"hook": 0, "cliff": 0, "clarity": 0, "escalation": 0, "trope": 0}})


def test_api_key_stays_on_the_judge_instance():
    import os
    a = LLMJudge.from_spec("groq:x", api_key="key-A")
    b = LLMJudge.from_spec("groq:x", api_key="key-B")
    assert (a.api_key, b.api_key) == ("key-A", "key-B") and os.environ.get("GROQ_API_KEY") != "key-A"


def test_gemini3_keeps_default_temperature(monkeypatch):
    """Gemini 3 models get no temperature override; other models are pinned to 0 for repeatable scores."""
    import io, json as _json, urllib.request
    from promo_finder.judge import LLMJudge
    sent = []

    def fake_urlopen(req, timeout=0):
        sent.append(_json.loads(req.data))
        return io.BytesIO(_json.dumps({"choices": [{"message": {"content": "{}"}}], "usage": {}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    for spec in ("gemini:gemini-3.8-flash", "openrouter:google/gemini-3.8-flash", "groq:llama-3.3-70b-versatile"):
        LLMJudge.from_spec(spec, api_key="k")._chat("sys", "user")
    assert "temperature" not in sent[0] and "temperature" not in sent[1]
    assert sent[2]["temperature"] == 0


def test_payoff_terms_match_whole_phrases_only():
    """'duchess of ash' must not fire on 'Duchess of Ashburgh' (a false retarget), but must still match on its own."""
    from promo_finder.promos import has_phrase
    terms = ["duchess of ash", "you're the duke"]
    assert not has_phrase("before i become duchess of ashburgh...", terms)
    assert has_phrase("she is the duchess of ash, at last.", terms)
    assert has_phrase("duchess of ash", terms)
    assert has_phrase("wait... you're the duke?", terms)
