from voiceclone.evaluate import select_take


def c(take, sim, cer, ok):
    return {"take": take, "sim": sim, "cer": cer, "ok": ok}


def test_gated_prefers_voice_match_among_clear_takes():
    # the closest voice is garbled (not ok), so a clear take wins
    cands = [c("a", 0.80, 0.95, False), c("b", 0.70, 0.60, True), c("d", 0.65, 0.50, True)]
    assert select_take(cands, "gated")["take"] == "b"


def test_gated_falls_back_to_clearest_when_nothing_passes():
    cands = [c("a", 0.80, 0.95, False), c("b", 0.70, 0.90, False)]
    assert select_take(cands, "gated")["take"] == "b"


def test_voice_mode_ignores_clarity():
    cands = [c("a", 0.80, 0.95, False), c("b", 0.70, 0.60, True)]
    assert select_take(cands, "voice")["take"] == "a"
