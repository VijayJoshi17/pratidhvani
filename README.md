# indic-voice-clone

Clone one speaker of an Indic language, regional accent included, from a few
archival recordings. It was built and tested on northern Kannada lecture tapes,
on a 16 GB Apple-silicon laptop. The voice model and codec are open-source and
licensed for commercial use.

How it works: the recordings are cleaned lightly and cut into short chunks.
Each chunk is transcribed verbatim in the dialect, and only clean chunks of
the target speaker are kept. Those chunks LoRA-fine-tune the
[Svara-TTS voice-clone model](https://huggingface.co/kenpath/svara-tts-voiceclone-beta)
in the same "reference-swap" prompt format the model uses at inference.

## Prerequisites

**Hardware**
- An Apple-silicon Mac with 16 GB of memory, or a Linux machine with an
  NVIDIA GPU with 16 GB or more of VRAM. Only the Mac has been tested; the
  code picks CUDA automatically when it is available.
- About 15 GB of free disk. The Svara model alone is 6.2 GB, the alignment
  model is 1.2 GB, and the Python environment is about 1 GB.

**Software**
- Python 3.10 or newer (3.12 is tested).
- [uv](https://docs.astral.sh/uv/), or pip with a virtual environment.
- [ffmpeg](https://ffmpeg.org/) on your `PATH` (`brew install ffmpeg` or
  `apt install ffmpeg`).
- A [Hugging Face](https://huggingface.co/) account. Transcription uses
  AI4Bharat IndicConformer, which is gated: open
  [its model page](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual),
  accept the terms, then run `hf auth login`.

**Optional**
- Audacity 3.x, for removing hiss, mains hum and clicks from tape. Audacity
  4 has no scripting. Enable `mod-script-pipe` under Preferences, Modules,
  then restart Audacity. The `audacity-denoise` step currently assumes
  macOS and `~/Applications/Audacity 3.7.app`; pass `--app` for another path.
  It quits and relaunches Audacity for every file, so it refuses to start
  while Audacity is open unless you pass `--force-restart`.
- A second Python environment for ClearerVoice neural denoising, because
  `clearvoice` needs `numpy<2` (see Installation).

**Languages**
- Text in Kannada, Telugu, Tamil, Malayalam, Devanagari (Hindi, Marathi,
  Sanskrit, Nepali), Bengali/Assamese, Gujarati, Gurmukhi and Odia is
  transliterated for alignment, length limits and scoring. Only Kannada has
  been tested end to end. Text in other scripts, such as Urdu, can still be
  synthesised, but only with the 20-second length cap, and the alignment-based
  steps cannot use it.

**Recordings**
- At least 10 to 20 minutes of the target speaker after cleaning. The
  reference results used about 23 minutes from one 32-minute lecture.
- Any format ffmpeg reads. MP3 at 128 kbps is fine.
- The right to use the voice. Only clone someone who has consented, or whose
  estate has.

## Installation

```bash
git clone https://github.com/svayatta/indic-voice-clone.git
cd indic-voice-clone
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[all]" -c constraints.txt
source .venv/bin/activate
voiceclone --help
```

`constraints.txt` pins the exact versions behind the reference results
(Python 3.12, macOS arm64). Leave it out to take the newest releases, at
the risk of behaviour changes.

The extras are `tts` for synthesis and training, `align` for forced
alignment and character error, `speaker` for voiceprints, `asr` for
IndicConformer, and `all` for everything. Models download on first use:
Svara and IndicConformer go to the Hugging Face cache, and the alignment
model goes to `~/ctc_forced_aligner/`.

Optional ClearerVoice environment:

```bash
uv venv --python 3.12 .venv-enh
uv pip install --python .venv-enh/bin/python clearvoice "numpy<2" -e .
```

## Getting started

This walk-through turns one recording, `talk.mp3`, into a fine-tuned voice
and six test sentences. Run it from the repository root. Everything is
written under `data/`, `models/` and `outputs/`, which git ignores.

**1. Prepare the recording.** This decodes the file, restores it lightly,
cuts it into 3 to 10 second chunks at natural pauses, and transcribes each
chunk verbatim.

```bash
voiceclone prepare talk.mp3 talk --lang kn
```

The chunks and `transcripts.jsonl` land in `data/vad_talk/`. Add
`--enhance-python .venv-enh/bin/python` to denoise the chunks with
ClearerVoice first; the output then lands in `data/vad_talk_se/`.
Transcription takes a while: each chunk is a separate model call.

**2. Pick enrollment clips.** The filter in the next step keeps only chunks
that sound like the target speaker, so it needs a voiceprint to compare
against. Listen to a few chunks and copy 10 or more that are certainly the
target speaker, then index them:

```bash
mkdir -p data/enroll
cp data/vad_talk/talk_0012.wav data/vad_talk/talk_0015.wav ... data/enroll/
python -c "import json,pathlib; d=pathlib.Path('data/enroll'); \
open(d/'metadata.jsonl','w').writelines(json.dumps({'file':p.name})+'\n' for p in sorted(d.glob('*.wav')))"
```

**3. Build the training set.** This drops chunks that are another speaker,
too few words, an implausible speaking rate, or a transcript that does not
fit the audio. It prints what was dropped and why.

```bash
voiceclone build-clips --vad data/vad_talk --enroll data/enroll --n-enroll 10 --out data/clips
```

Repeat `--vad` to combine several prepared recordings.

**4. Fine-tune.** Six clips are held out for evaluation and listed in
`models/lora/holdout.json`. The final adapter is saved in `models/lora/`,
and every 50 optimizer steps a copy is kept in `models/lora/checkpoint-NNNNN/`
(about 100 MB each), so you can go back to an earlier one if it sounds
better. Change the interval with `--save-every`.

```bash
voiceclone train --clips data/clips --out models/lora --epochs 2
```

On an Apple M5 with 16 GB, 23 minutes of clips over 2 epochs took about 70
minutes, at 31 to 42 seconds per optimizer step.

**5. Synthesize.** Choose a clean 4 to 8 second clip from `data/clips` as the
reference, and put its text from `data/clips/metadata.jsonl` in a file.

```bash
voiceclone synth --lora models/lora \
    --ref data/clips/talk_0073.wav --ref-text-file ref.txt \
    --text-file examples/test_sentences_kn.txt --min-sps 2.5 \
    --out outputs/lora
```

Each sentence takes about 30 to 75 seconds on the same laptop. The output is
`outputs/lora/sample_XX.wav` plus `log.json`. Use `--text "..."` for a
single sentence.

**6. Tidy and measure.**

```bash
voiceclone trim --samples outputs/lora
voiceclone eval-similarity --clips data/clips --samples outputs/lora_trimmed
voiceclone eval-intelligibility --clips data/clips --samples outputs/lora_trimmed \
    --holdout models/lora/holdout.json
```

Both reports include the speaker's real clips as a baseline, so compare the
clone against those numbers rather than against perfect scores. In the
reference run, the speaker's real clips scored about 0.75 to 0.78 voice
similarity and the clone scored 0.61 to 0.72. Numbers do not judge accent, though: have native
speakers of the dialect listen.

**7. Improve (optional).** Generate several takes with different `--seed`
values and keep the best one per sentence with `best-of`. It pairs takes by
the sentence recorded in each folder's `log.json`, so folders made from
differently ordered sentence files still compare correctly. Add more
recordings with `prepare`, then retrain or continue training with
`train --init-adapter models/lora --lr 1e-4`.

## Command reference

All steps are subcommands of `voiceclone`; run `voiceclone <step> --help`
for every option.

| Step | Command | Output |
|---|---|---|
| Restore audio | `restore talk.mp3 --decode-first` | mono 24 kHz WAV, mild denoise |
| Cut at pauses | `vad --audio clean.wav --out data/vad` | 3-10 s chunks + `segments.jsonl` |
| Transcribe | `transcribe --vad data/vad` | `transcripts.jsonl`, verbatim, resumable |
| Whole preparation for one file | `prepare talk.mp3 TAG` | `data/vad_TAG[_se]/` |
| Analogue noise (optional) | `audacity-denoise talk.mp3 TAG --reuse-chunks data/vad_TAG_se` | hiss, hum, clicks removed |
| Rank recordings | `rank DIR --enroll data/enroll` | which talks are the speaker, how clean |
| Build training set | `build-clips --vad data/vad --enroll data/enroll --out data/clips` | speaker- and alignment-filtered clips |
| Fine-tune | `train --clips data/clips --out models/lora` | LoRA adapter + `holdout.json` |
| Synthesize | `synth --lora models/lora --ref clip.wav --ref-text-file ref.txt --text "..."` | `sample_XX.wav` + `log.json` |
| Trim run-ons | `trim --samples outputs/synth` | cut after the last word |
| Best take | `best-of --takes A --takes B --clips data/clips --out best` | best take per sentence |
| Voice similarity | `eval-similarity --clips data/clips --samples outputs/synth` | ECAPA cosine vs real clips |
| Intelligibility | `eval-intelligibility --clips data/clips --samples outputs/synth` | character error vs real clips |
| Same-words test | `long-form --lora models/lora --chunks data/vad_X --source X.wav --ref clip.wav --ref-text-file ref.txt --out outputs/long` | clone vs original, A/B file |

If you have a timed human transcript (`[SS:MM:00] [Speaker] : text`) instead
of verbatim ASR, use `parse-transcript`, `align` and `segment --speaker NAME`
in place of `vad`, `transcribe` and `build-clips`. Its clips can also serve
as the enrollment set. Verbatim ASR gave better results, because a cleaned-up
transcript teaches the model words the speaker never said.

## Troubleshooting

- **Out of memory while training.** Lower `--max-tgt-sec` (default 10.5) and
  `--max-ref-sec` (default 4.5), or `--rank`. Close other large apps.
- **The clone rambles past the sentence.** Run `trim`, or raise `--min-sps`
  to tighten the length cap.
- **The clone stops after a syllable.** Lower `--max-sps` to raise the
  minimum length.
- **Few clips survive `build-clips`.** Check the drop reasons it prints.
  `other_speaker` means the enrollment clips do not match, and `score` means
  transcripts do not fit the audio, often because of noise.
- **IndicConformer fails to download.** Accept its terms on Hugging Face
  and run `hf auth login`.
- **Two runs of the same command give slightly different audio.** That is
  expected: the SNAC decoder adds a little random noise, and sampling is
  random unless `--seed` is fixed.

## Notes on keeping the accent

- Transcripts stay verbatim in the dialect. Pronunciation is learned from
  the audio, but the text must match what was said.
- Restoration is mild. Aggressive denoising shifts formants, which is where
  an accent lives.
- Training uses the same reference-swap prompt as inference, so the adapter
  learns this speaker and dialect rather than a new speaker-ID token.
- The LoRA stays unmerged at inference: merging fp32 deltas into bf16
  weights rounds most of them away.
- Generation passes `use_cache=True`. The base model's saved config disables
  the cache, which makes generation about 10x slower.

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

This repository contains code only: no recordings, transcripts or trained
weights.

## Tests

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```

The unit tests cover transliteration, transcript parsing, the Svara token and
prompt layout, pause segmentation and take selection. They need no models.
