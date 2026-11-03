"""Cleaning with the real ffmpeg chain: the clone is only as good as what it
is conditioned on, so the output must be the model's format (mono 24 kHz),
at a fixed level, with hum and hiss reduced and the voice kept."""
import shutil

import numpy as np
import pytest
import soundfile as sf

from voiceclone.studio.clean import clean_audio, duration, pick_reference

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def band_energy(x, sr, lo, hi):
    f = np.fft.rfftfreq(len(x), 1 / sr)
    s = np.abs(np.fft.rfft(x)) ** 2
    return s[(f >= lo) & (f < hi)].sum()


def test_clean_output_format_and_hum_removed(tmp_path):
    sr = 44100
    t = np.arange(8 * sr) / sr
    voice = 0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * t / 3) > -0.5)
    hum = 0.2 * np.sin(2 * np.pi * 50 * t)
    stereo = np.stack([voice + hum, voice + hum], 1).astype(np.float32)
    src = tmp_path / "in.wav"
    sf.write(src, stereo, sr)
    out = clean_audio(src, tmp_path / "out.wav")
    y, osr = sf.read(out)
    assert osr == 24000 and y.ndim == 1
    assert abs(duration(out) - 8) < 0.1
    assert abs(20 * np.log10(np.abs(y).max()) + 3) < 0.2           # -3 dBFS peak
    x24 = np.interp(np.arange(len(y)) / 24000, t, (voice + hum))
    hum_ratio_before = band_energy(x24, 24000, 40, 60) / band_energy(x24, 24000, 200, 240)
    hum_ratio_after = band_energy(y, 24000, 40, 60) / band_energy(y, 24000, 200, 240)
    assert hum_ratio_after < hum_ratio_before / 2    # the 2-pole 60 Hz high-pass: about 5 dB less 50 Hz hum
    assert band_energy(y, 24000, 200, 240) > 0.2 * band_energy(y, 24000, 0, 12000)   # the voice is still there


def test_reference_is_the_longest_chunk(tmp_path):
    sr = 24000
    parts = []
    for sec in (3.5, 8.0, 4.0):
        t = np.arange(int(sec * sr)) / sr
        parts += [0.3 * np.sin(2 * np.pi * 200 * t), np.zeros(int(0.8 * sr))]
    x = np.concatenate(parts).astype(np.float32) + 0.001 * np.random.default_rng(0).standard_normal(sum(map(len, parts)))
    sf.write(tmp_path / "clean.wav", x.astype(np.float32), sr)
    ref = pick_reference(tmp_path / "clean.wav", tmp_path / "ref.wav", tmp_path / "work")
    assert 7.0 < duration(ref) < 9.0
