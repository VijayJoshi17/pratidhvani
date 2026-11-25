"""`long-form` re-voices a stretch of a recording so it can be heard next to
the original. Each take is placed at its chunk's original start, or right
after the previous take if that one ran long, and generated takes are cached
so an interrupted run resumes. A fake synthesizer stands in for Svara, so
these need no model."""
import numpy as np
import pytest
import soundfile as sf

from voiceclone.audio import TTS_SR
from voiceclone.jsonl import write_jsonl
from voiceclone.longform import long_form

GAP = 0.25


class FakeSynth:
    """Returns a constant-level take of a chosen length per text, and records
    every call, so placement can be read back from cloned.wav."""

    def __init__(self, takes: dict[str, tuple[float, float]]):
        self.takes = takes          # text -> (seconds, level)
        self.calls = []

    def reference(self, ref_wav):
        return "ref-tokens"

    def generate(self, text, ref_tokens, ref_text, seed=0, p=None):
        self.calls.append(text)
        seconds, level = self.takes[text]
        return np.full(int(seconds * TTS_SR), level, dtype=np.float32), {}


def chunks(tmp_path, rows):
    """segments.jsonl and transcripts.jsonl as `prepare` and `transcribe`
    write them, plus a silent source recording as long as the last chunk."""
    d = tmp_path / "chunks"
    d.mkdir()
    write_jsonl(d / "segments.jsonl",
                [{"file": f"c{i}.wav", "start": s, "end": e} for i, (s, e, _) in enumerate(rows)])
    write_jsonl(d / "transcripts.jsonl", [{"file": f"c{i}.wav", "text": t} for i, (_, _, t) in enumerate(rows)])
    src = tmp_path / "source.wav"
    sf.write(src, np.zeros(int(rows[-1][1] * TTS_SR) + TTS_SR, np.float32), TTS_SR)
    return d, src


def run(tmp_path, synth, chunks_dir, src):
    return long_form(synth, chunks_dir, src, tmp_path / "ref.wav", "ref", tmp_path / "out", gap=GAP, log=lambda *a: None)


def level_at(wav, t):
    return float(wav[int(t * TTS_SR)])


def test_a_take_that_runs_long_pushes_the_next_one_back(tmp_path):
    # "one" has a 1 s slot but its take is 2 s; "two" starts at 1.5 s in the original
    d, src = chunks(tmp_path, [(0.0, 1.0, "one"), (1.5, 2.5, "two"), (6.0, 7.0, "three")])
    synth = FakeSynth({"one": (2.0, 0.25), "two": (1.0, 0.5), "three": (0.5, 0.75)})
    out = run(tmp_path, synth, d, src)
    clone, sr = sf.read(out / "cloned.wav", dtype="float32")
    assert sr == TTS_SR
    tol = 1e-3   # cloned.wav is 16-bit
    # "one" is whole: "two" was not written over its tail
    assert level_at(clone, 1.9) == pytest.approx(0.25, abs=tol)
    # "two" starts at the cursor (end of "one" plus the gap), not at its original 1.5 s
    assert level_at(clone, 2.1) == pytest.approx(0.0, abs=tol)
    assert level_at(clone, 2.0 + GAP + 0.01) == pytest.approx(0.5, abs=tol)
    assert level_at(clone, 2.0 + GAP + 0.99) == pytest.approx(0.5, abs=tol)
    # once back in step, a take returns to its original start rather than the cursor
    assert level_at(clone, 5.9) == pytest.approx(0.0, abs=tol)
    assert level_at(clone, 6.01) == pytest.approx(0.75, abs=tol)


def test_takes_never_overlap(tmp_path):
    # every take runs long; each must still be whole in cloned.wav
    rows = [(i * 1.0, i * 1.0 + 0.8, f"line {i}") for i in range(5)]
    d, src = chunks(tmp_path, rows)
    synth = FakeSynth({t: (1.5, 0.1 * (i + 1)) for i, (_, _, t) in enumerate(rows)})
    clone = sf.read(run(tmp_path, synth, d, src) / "cloned.wav", dtype="float32")[0]
    start = 0.0
    for i in range(5):
        take = clone[int(start * TTS_SR) + 1: int((start + 1.5) * TTS_SR) - 1]
        assert np.allclose(take, 0.1 * (i + 1), atol=1e-3), i
        start += 1.5 + GAP


def test_a_cached_take_is_not_generated_again(tmp_path):
    d, src = chunks(tmp_path, [(0.0, 1.0, "one"), (2.0, 3.0, "two")])
    synth = FakeSynth({"one": (0.5, 0.25), "two": (0.5, 0.5)})
    out = run(tmp_path, synth, d, src)
    assert synth.calls == ["one", "two"]
    first = sf.read(out / "cloned.wav", dtype="float32")[0]

    synth.calls.clear()
    run(tmp_path, synth, d, src)
    assert synth.calls == []
    assert np.array_equal(sf.read(out / "cloned.wav", dtype="float32")[0], first)

    # an interrupted run resumes: only the missing take is generated
    (out / "chunks" / "001.wav").unlink()
    run(tmp_path, synth, d, src)
    assert synth.calls == ["two"]


def test_an_empty_transcript_is_skipped(tmp_path):
    d, src = chunks(tmp_path, [(0.0, 1.0, "one"), (1.5, 2.0, " "), (3.0, 4.0, "three")])
    synth = FakeSynth({"one": (0.5, 0.25), "three": (0.5, 0.5)})
    out = run(tmp_path, synth, d, src)
    assert synth.calls == ["one", "three"]
    assert (out / "script.tsv").read_text(encoding="utf-8").count("\n") == 3
