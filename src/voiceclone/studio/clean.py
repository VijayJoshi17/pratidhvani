"""Denoise an uploaded file and pick the clip the instant clone is conditioned on.

Cleaning is the same chain `prepare` uses: decode, light ffmpeg restoration
(declip, 60 Hz high-pass, gentle spectral denoise) and, when an environment
with clearvoice is configured, MossFormer2 speech enhancement. The
restoration is deliberately light: heavier denoising shifts formants, and
formants carry the accent.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

from voiceclone.audio import TTS_SR, ffmpeg_decode, load_wav, peak_normalize, restore, write_wav

SAMPLE_MIN_SEC = 5.0     # shorter samples do not hold enough of the voice
SAMPLE_MAX_SEC = 45.0    # longer ones belong in training recordings


def clean_audio(src, dst, enhance_python: str | None = None, enhance_wet: float = 1.0) -> Path:
    """Any audio file -> denoised mono 24 kHz WAV at -3 dBFS peak."""
    dst = Path(dst)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        restored = restore(ffmpeg_decode(src, td / "48k.wav", 48000), td / "in" / "x.wav")
        if enhance_python:
            subprocess.run([enhance_python, "-m", "voiceclone.enhance.clearvoice", str(td / "in"), str(td / "se"),
                            "--steps", "se", "--wet", str(enhance_wet)], check=True)
            restored = td / "se" / "x.wav"
        x = load_wav(restored, TTS_SR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        write_wav(dst, peak_normalize(x), TTS_SR)
    return dst


def duration(path) -> float:
    return sf.info(str(path)).duration


def pick_reference(clean_wav, out_wav, work_dir) -> Path:
    """The longest pause-delimited chunk (3-10 s) of the sample.

    The model is conditioned on this clip and its exact words, so a whole
    45 s sample would be both too long for the prompt and hard to transcribe
    exactly. Without any usable chunk the first 10 s are used.
    """
    from voiceclone.jsonl import read_jsonl
    from voiceclone.segment import vad_segment

    x = load_wav(clean_wav, TTS_SR)
    work = Path(work_dir)
    vad_segment(x, TTS_SR, work, prefix="ref")
    segs = read_jsonl(work / "segments.jsonl")
    if segs:
        best = max(segs, key=lambda s: s["duration"])
        shutil.copy(work / best["file"], out_wav)
    else:
        write_wav(out_wav, x[: 10 * TTS_SR], TTS_SR)
    return Path(out_wav)
