"""Verbatim transcription with AI4Bharat IndicConformer 600M (MIT) or Whisper fallback."""
from pathlib import Path

from voiceclone.jsonl import read_jsonl

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"


class IndicConformer:
    def __init__(self, model_id: str = MODEL_ID):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    def transcribe(self, x, sr: int, lang: str = "kn", decoder: str = "rnnt") -> str:
        import torch
        import torchaudio

        w = torchaudio.functional.resample(torch.from_numpy(x).view(1, -1), sr, 16000)
        text = self.model(w, lang, decoder)
        text = text[0] if isinstance(text, (list, tuple)) else text
        return text.strip()


class WhisperASR:
    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo"):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        from voiceclone.device import pick_device

        device = pick_device()
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
        )

    def transcribe(self, x, sr: int, lang: str = "kn") -> str:
        lang_name = "kannada" if lang == "kn" else ("marathi" if lang == "mr" else "kannada")
        res = self.pipe(
            {"raw": x, "sampling_rate": sr},
            generate_kwargs={"language": lang_name, "task": "transcribe", "repetition_penalty": 1.3, "no_repeat_ngram_size": 3}
        )
        return res["text"].strip()


def get_default_asr(log=print):
    try:
        return IndicConformer()
    except Exception as e:
        log(f"IndicConformer unavailable ({e}); falling back to openai/whisper-large-v3-turbo")
        return WhisperASR()


def transcribe_dir(vad_dir, lang: str = "kn", limit: int = 0, asr=None, log=print) -> Path:
    """Append {file, text} to vad_dir/transcripts.jsonl for every chunk not yet done (resumable)."""
    import json

    import soundfile as sf

    vad = Path(vad_dir)
    out = vad / "transcripts.jsonl"
    done = {r["file"] for r in read_jsonl(out)} if out.exists() else set()
    segs = read_jsonl(vad / "segments.jsonl")
    if asr is None:
        asr = get_default_asr(log=log)
    with open(out, "a", encoding="utf-8") as f:
        for s in segs[: limit or len(segs)]:
            if s["file"] in done:
                continue
            x, sr = sf.read(str(vad / s["file"]), dtype="float32")
            text = asr.transcribe(x, sr, lang)
            f.write(json.dumps({"file": s["file"], "text": text}, ensure_ascii=False) + "\n"); f.flush()
            log(f"{s['file']} {s['duration']:.1f}s | {text}")
    return out
