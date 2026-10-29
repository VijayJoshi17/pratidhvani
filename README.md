# indic-voice-clone

Clone one speaker of an Indic language, with their regional accent, from a
few archival recordings. Built and tested on northern Kannada lecture tapes
on a 16 GB Apple-silicon laptop, using open-source, commercially usable
models for the voice itself.

The approach: clean the recordings lightly, cut them into short chunks,
transcribe them verbatim in the dialect, keep only clean chunks of the target
speaker, then LoRA-fine-tune the Svara-TTS voice-clone model in the same
"reference-swap" prompt format it uses at inference.

## Install

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[all]"
```

Extras: `tts` (synthesis and training), `align` (forced alignment, CER),
`speaker` (voiceprints), `asr` (IndicConformer). ClearerVoice denoising needs
`numpy<2`, so install `clearvoice` plus this package in a second environment
and pass that interpreter to `voiceclone prepare --enhance-python`.

IndicConformer is gated on Hugging Face: accept its terms and `hf auth login`.

## Pipeline

All steps are subcommands of `voiceclone` (`voiceclone <step> --help`).

| Step | Command | Output |
|---|---|---|
| Restore audio | `restore talk.mp3 --decode-first` | mono 24 kHz WAV, mild denoise |
| Cut at pauses | `vad --audio clean.wav --out data/vad` | 3-10 s chunks + `segments.jsonl` |
| Transcribe | `transcribe --vad data/vad` | `transcripts.jsonl`, verbatim dialect |
| Whole pipeline for one file | `prepare talk.mp3 TAG` | `data/vad_TAG[_se]/` |
| Analogue noise (optional) | `audacity-denoise talk.mp3 TAG --reuse-chunks data/vad_TAG_se` | hiss, hum, clicks removed |
| Rank recordings | `rank DIR --enroll data/clips` | which talks are the speaker, how clean |
| Build training set | `build-clips --vad data/vad --enroll data/clips --out data/clips_v2` | speaker- and alignment-filtered clips |
| Fine-tune | `train --clips data/clips_v2 --out models/lora` | LoRA adapter + `holdout.json` |
| Synthesize | `synth --lora models/lora --ref clip.wav --ref-text "..." --text-file examples/test_sentences_kn.txt` | `sample_XX.wav` + `log.json` |
| Trim run-ons | `trim --samples outputs/synth` | cut after the last word |
| Best take | `best-of --takes A --takes B --clips data/clips_v2 --out best` | best take per sentence |
| Voice similarity | `eval-similarity --clips data/clips_v2 --samples outputs/synth` | ECAPA cosine vs real clips |
| Intelligibility | `eval-intelligibility --clips data/clips_v2 --samples outputs/synth --holdout models/lora/holdout.json` | CER vs real clips |
| Same-words test | `long-form --lora models/lora --chunks data/vad_X --source X.wav --ref ... --out outputs/long` | clone vs original, A/B file |

If you have a timed human transcript (`[SS:MM:00] [Speaker] : text`) instead
of verbatim ASR, use `parse-transcript`, `align` and
`segment --speaker NAME` in place of `vad`, `transcribe` and `build-clips`.
Verbatim ASR worked better: a cleaned-up transcript teaches the model words
the speaker never said.

## Package layout

```
src/voiceclone/
  text.py         Kannada transliteration (keeps the inherent vowel), syllables, CER
  audio.py        loading, normalising, ffmpeg decode and mild restoration
  transcript.py   timed-transcript parser
  segment.py      pause-based and alignment-based chunking
  asr.py          IndicConformer verbatim transcription
  align.py        MMS CTC forced alignment and greedy decoding
  speaker.py      ECAPA voiceprints and centroids
  dataset.py      training-clip filter
  rank.py         recording ranking
  svara/tokens.py prompt layout, SNAC token mapping, generation length budget (pure Python)
  svara/codec.py  SNAC encode/decode
  svara/synth.py  Synthesizer (zero-shot or LoRA)
  train.py        reference-swap LoRA fine-tuning
  evaluate.py     similarity, intelligibility, best-of selection
  postprocess.py  trim to text
  longform.py     same-words re-voicing
  enhance/        ClearerVoice and Audacity 3.x denoisers
  cli.py          the `voiceclone` command
```

## Notes on keeping the accent

- Transcripts stay verbatim in the dialect. Pronunciation is learned from
  the audio, but the text must match what was said.
- Restoration is mild. Aggressive denoising shifts formants, which is where
  an accent lives.
- Training uses the same reference-swap prompt as inference, so the adapter
  learns this speaker and dialect rather than a new speaker-ID token.
- Keep the LoRA unmerged at inference: merging fp32 deltas into bf16 weights
  rounds most of them away.
- Generation passes `use_cache=True`; the base model's saved config disables
  the cache, which makes generation about 10x slower.

## Models and licences

| Component | Licence |
|---|---|
| `kenpath/svara-tts-voiceclone-beta` (3B, 19 Indic languages) | Apache 2.0 |
| `hubertsiuzdak/snac_24khz` codec | MIT |
| AI4Bharat IndicConformer 600M | MIT |
| SpeechBrain ECAPA-TDNN | Apache 2.0 |
| ClearerVoice MossFormer2 | Apache 2.0 |
| `ctc-forced-aligner` code / MMS weights | MIT / CC-BY-NC: use for data prep and evaluation only |
| Audacity 3.x | GPL, used as an external tool |

Only clone a voice you have the right to use. This repository contains code
only: no recordings, transcripts or trained weights.

## Tests

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```
