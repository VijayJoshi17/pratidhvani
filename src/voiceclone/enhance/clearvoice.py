"""Denoise / restore bandwidth with ClearerVoice-Studio (Apache 2.0).

clearvoice pins numpy<2, so run this module in its own environment:
  python -m voiceclone.enhance.clearvoice IN_DIR OUT_DIR [--steps se,sr]
    se : MossFormer2_SE_48K speech enhancement (noise, some reverb)
    sr : MossFormer2_SR_48K super-resolution (restores high frequencies)
Output is mono 24 kHz, peak -3 dBFS. Imports nothing heavy from voiceclone.
"""
import argparse
import tempfile
from pathlib import Path

MODELS = {"se": ("speech_enhancement", "MossFormer2_SE_48K"),
          "sr": ("speech_super_resolution", "MossFormer2_SR_48K")}


def enhance_dir(inp, out, steps=("se", "sr"), log=print):
    import librosa
    import numpy as np
    import soundfile as sf
    from clearvoice import ClearVoice

    inp, out = Path(inp), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    models = {s: ClearVoice(task=MODELS[s][0], model_names=[MODELS[s][1]]) for s in steps}
    for f in sorted(inp.glob("*.wav")):
        with tempfile.TemporaryDirectory() as td:
            cur = Path(td) / "x.wav"
            sf.write(cur, librosa.load(str(f), sr=48000, mono=True)[0], 48000)
            for st in steps:
                res = models[st](input_path=str(cur), online_write=False)
                res = res[0] if isinstance(res, (list, tuple)) else res
                cur = Path(td) / f"{st}.wav"
                sf.write(cur, np.asarray(res).reshape(-1).astype(np.float32), 48000)
            y, _ = librosa.load(str(cur), sr=24000, mono=True)
            sf.write(out / f.name, y * (10 ** (-3 / 20) / (np.abs(y).max() or 1)), 24000, subtype="PCM_16")
            log(f.name)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m voiceclone.enhance.clearvoice")
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--steps", default="se,sr")
    a = ap.parse_args(argv)
    enhance_dir(a.inp, a.out, tuple(a.steps.split(",")))


if __name__ == "__main__":
    main()
