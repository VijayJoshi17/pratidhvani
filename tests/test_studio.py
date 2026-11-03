"""The studio's API and job flow, with a stand-in engine (no models, no ffmpeg).

What matters to a user: a sample is cleaned before anything is cloned from
it; samples outside 5-45 s are refused with a reason; the clone is
conditioned on words that match its reference clip; training only starts
with recordings; once trained, previews use the trained adapter; a server
restart never leaves a job looking busy forever.
"""
import io
import shutil

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from voiceclone.studio.app import create_app  # noqa: E402
from voiceclone.studio.jobs import Studio  # noqa: E402

SR = 24000


def speech_like(sec, sr=SR, seed=0):
    """Bursts of tone separated by pauses, so VAD finds 3-10 s chunks."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * t / 7.0) > -0.6)
    return (x + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


def wav_bytes(sec):
    buf = io.BytesIO()
    sf.write(buf, speech_like(sec), SR, format="WAV")
    return buf.getvalue()


class FakeEngine:
    min_train_minutes = 8.0

    def __init__(self):
        self.cleaned, self.transcribed, self.synth_calls, self.trained = [], [], [], []

    def clean(self, src, dst):
        self.cleaned.append(src)
        shutil.copy(src, dst)
        return dst

    def transcribe(self, wav, lang):
        self.transcribed.append((wav, lang))
        return "ನಮಸ್ಕಾರ ರೀ"

    def synthesize(self, text, ref_wav, ref_text, lora=None, seed=0):
        self.synth_calls.append({"text": text, "ref_text": ref_text, "lora": lora, "seed": seed})
        return np.zeros(SR, np.float32), 84

    def train(self, sample_wav, recordings, work, lang, log=print):
        log("usable speech: 12.5 min ({})")
        log("training the LoRA adapter")
        self.trained.append(list(recordings))
        (work / "adapter").mkdir(parents=True)
        return work / "adapter"


@pytest.fixture
def env(tmp_path):
    engine = FakeEngine()
    studio = Studio(tmp_path, engine, start=False)
    client = TestClient(create_app(studio))

    def drain():
        while not studio.q.empty():
            fn, *args = studio.q.get()
            studio.run_job(fn, *args)
    return client, studio, engine, drain


def create(client, sec=12, name="Narrator", **form):
    return client.post("/api/voices", data={"name": name, "lang": "kn", **form},
                       files={"file": ("sample.wav", wav_bytes(sec), "audio/wav")})


def test_sample_is_cleaned_then_transcribed_then_ready(env):
    client, _, engine, drain = env
    r = create(client)
    assert r.status_code == 201
    vid = r.json()["id"]
    assert r.json()["status"] == "queued"
    drain()
    v = client.get(f"/api/voices/{vid}").json()
    assert v["status"] == "ready", v["error"]
    assert len(engine.cleaned) == 1
    # the reference clip (not the whole sample) is what gets transcribed and conditioned on
    assert engine.transcribed[0][0].name == "reference.wav"
    assert v["ref_text"] == "ನಮಸ್ಕಾರ ರೀ" and v["ref_text_by"] == "asr"
    for kind in ("original", "clean", "reference"):
        assert client.get(f"/api/voices/{vid}/audio/{kind}").status_code == 200


def test_user_transcript_is_used_instead_of_asr(env):
    client, _, engine, drain = env
    vid = create(client, ref_text="ಇವತ್ತು ಸಂಜೆ").json()["id"]
    drain()
    assert client.get(f"/api/voices/{vid}").json()["ref_text"] == "ಇವತ್ತು ಸಂಜೆ"
    assert engine.transcribed == []


@pytest.mark.parametrize("sec", [2, 60])
def test_sample_length_outside_5_to_45_s_fails_with_reason(env, sec):
    client, _, _, drain = env
    vid = create(client, sec=sec).json()["id"]
    drain()
    v = client.get(f"/api/voices/{vid}").json()
    assert v["status"] == "failed"
    assert "5 to 45 s" in v["error"]


def test_upload_checks(env):
    client, studio, _, _ = env
    assert create(client, name="  ").status_code == 400
    r = client.post("/api/voices", data={"name": "x"}, files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    r = client.post("/api/voices", data={"name": "x", "lang": "xx"}, files={"file": ("a.wav", wav_bytes(6), "audio/wav")})
    assert r.status_code == 400
    assert studio.store.list() == []     # refused uploads leave nothing behind


def test_oversize_upload_is_refused_and_removed(env, monkeypatch):
    client, studio, _, _ = env
    monkeypatch.setattr("voiceclone.studio.app.SAMPLE_MAX_MB", 0)
    assert create(client).status_code == 413
    assert studio.store.list() == []


def test_preview_uses_reference_words_and_is_served(env):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    assert client.post(f"/api/voices/{vid}/previews", json={"text": "ನಮಸ್ಕಾರ"}).status_code == 409   # not ready yet
    drain()
    p = client.post(f"/api/voices/{vid}/previews", json={"text": "ಮಕ್ಕಳಿಗೆ ಚಲೋ ವಿದ್ಯೆ ಕೊಡ್ರಿ", "seed": 2}).json()
    drain()
    call = engine.synth_calls[-1]
    assert call == {"text": "ಮಕ್ಕಳಿಗೆ ಚಲೋ ವಿದ್ಯೆ ಕೊಡ್ರಿ", "ref_text": "ನಮಸ್ಕಾರ ರೀ", "lora": None, "seed": 2}
    v = client.get(f"/api/voices/{vid}").json()
    assert v["previews"][0]["status"] == "done"
    assert client.get(f"/api/voices/{vid}/audio/preview", params={"id": p["id"]}).status_code == 200


def test_corrected_reference_words_are_used(env):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    drain()
    client.patch(f"/api/voices/{vid}", json={"ref_text": "ನಮಸ್ಕಾರ ರೀ ಎಲ್ಲಾರೂ"})
    client.post(f"/api/voices/{vid}/previews", json={"text": "ನಮಸ್ಕಾರ"})
    drain()
    assert engine.synth_calls[-1]["ref_text"] == "ನಮಸ್ಕಾರ ರೀ ಎಲ್ಲಾರೂ"


def test_training_needs_recordings_then_previews_use_the_adapter(env):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    drain()
    assert client.post(f"/api/voices/{vid}/train").status_code == 409
    r = client.post(f"/api/voices/{vid}/recordings",
                    files=[("files", ("talk one.wav", wav_bytes(20), "audio/wav")), ("files", ("b.mp3", b"ID3fake", "audio/mpeg"))])
    assert [x["file"] for x in r.json()["recordings"]] == ["000_talk_one.wav", "001_b.mp3"]
    assert client.post(f"/api/voices/{vid}/train").status_code == 202
    assert client.post(f"/api/voices/{vid}/train").status_code == 409   # already queued
    drain()
    v = client.get(f"/api/voices/{vid}").json()
    assert v["train"]["status"] == "trained" and v["train"]["speech_min"] == 12.5
    assert [p.name for p in engine.trained[0]] == ["000_talk_one.wav", "001_b.mp3"]
    client.post(f"/api/voices/{vid}/previews", json={"text": "ನಮಸ್ಕಾರ"})
    drain()
    assert engine.synth_calls[-1]["lora"] == v["train"]["adapter"]


def test_engine_failure_is_shown_not_swallowed(env):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    drain()

    def boom(*a, **k):
        raise RuntimeError("only 3.1 min of usable speech")
    engine.train = boom
    client.post(f"/api/voices/{vid}/recordings", files=[("files", ("a.wav", wav_bytes(6), "audio/wav"))])
    client.post(f"/api/voices/{vid}/train")
    drain()
    t = client.get(f"/api/voices/{vid}").json()["train"]
    assert t["status"] == "failed" and "3.1 min" in t["error"]


def test_restart_marks_interrupted_jobs_failed(tmp_path):
    engine = FakeEngine()
    s1 = Studio(tmp_path, engine, start=False)
    v = s1.store.create("x", "kn")
    s1.store.update(v["id"], lambda v: v.update(status="cleaning"))
    s2 = Studio(tmp_path, engine, start=False)
    assert s2.store.get(v["id"])["status"] == "failed"


def test_audio_route_does_not_escape_the_voice_folder(env):
    client, _, _, drain = env
    vid = create(client).json()["id"]
    drain()
    assert client.get(f"/api/voices/{vid}/audio/preview", params={"id": "../voice"}).status_code == 404
    assert client.get(f"/api/voices/{vid}/audio/voice.json").status_code == 404
    assert client.get("/api/voices/..%2F..%2Fetc/audio/clean").status_code == 404


def test_delete_and_page(env):
    client, studio, _, _ = env
    vid = create(client).json()["id"]
    assert client.get("/").status_code == 200 and "Voice studio" in client.get("/").text
    assert client.delete(f"/api/voices/{vid}").status_code == 204
    assert client.get(f"/api/voices/{vid}").status_code == 404
