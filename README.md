# Indic Dual-Speaker Dialogue Pipeline

Builds speaker-separated two-speaker conversational speech corpora in Indian
languages, for training full-duplex spoken dialogue models.

Full-duplex models need each speaker on a **separate channel**, so overlaps,
backchannels and turn-taking timing stay observable. Public speech corpora are
almost all monaural, and the separated ones are English telephone speech. There
is no equivalent for Indian languages.

## Pipeline

```
YouTube URLs
    |  yt-dlp            audio-only download, 16 kHz mono
    |  Silero VAD        drop music, silence, continuous monologue
    |  Nemotron Omni     <- the verification stage
    |  pyannote / VAD    diarization, turn-taking statistics
    |  DialogueSidon     separation into one channel per speaker
corpus/accepted/<id>/
```

### The verification stage

Search results and platform metadata are unreliable for this task. A query for
Hindi podcasts returns monologues, dubbed content, ad reads, music, and videos
tagged Hindi that are in English. Filtering on titles or tags catches none of
it.

Instead the pipeline samples three windows from the middle 80% of each file and
asks NVIDIA Nemotron Omni — a model that accepts audio directly as input — to
report the spoken language, the number of distinct voices, and whether people
are genuinely talking *with* each other. Results are majority-voted across
windows, so an intro jingle or a solo cold-open does not decide the video.

A clip is kept only if it is exactly two speakers, in an Indic language, and
conversational.

### Quality gates

| Gate | Default | Why |
|---|---|---|
| duration | 2 min - 4 h | shorter has no real turn-taking |
| speech ratio | 0.45 - 0.98 | filters music and read monologue |
| speakers | exactly 2 | DialogueSidon assumes two |
| language | Indic, verified from audio | tags and titles are unreliable |
| conversational | required | rejects narration, dubbing, ads |
| turns/min | >= 2 | few turns means little duplex signal |
| dominance | <= 0.80 | one speaker dominating is monologue |
| overlap ratio | 0.01 - 0.40 | too low means not spontaneous; too high means separation failed |

The overlap floor matters more than it looks. Separately recorded or dubbed
tracks pass every other filter cleanly, then teach a full-duplex model *never
to interrupt* — the opposite of the target behaviour.

## Output

```
corpus/accepted/<video_id>/
  mixture.wav        original two-speaker mixture
  spk0.wav spk1.wav  separated per-speaker channels
  stereo.wav         spk0 = left, spk1 = right
  segments.jsonl     per-segment timing, overlap_frac, clean flag
  diarization.rttm
  meta.json          provenance, verification verdict, turn statistics
```

`stereo.wav` is the shape multi-stream dialogue models expect.

## Install

```bash
git clone https://github.com/Vandan31/indic_duplex_pipeline.git
cd indic_duplex_pipeline
pip install -r requirements.txt
# ffmpeg and ffprobe must be on PATH

cp .env.example .env      # fill in your keys
export $(grep -v '^#' .env | xargs)
```

`NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com) (free tier),
only needed for `discover`/`run` (the verification stage).
`HF_TOKEN` needs the gated `pyannote/speaker-diarization-3.1`,
`pyannote/segmentation-3.0` and `sarulab-speech/DialogueSidon` repos accepted
for `discover`/`run`, and separately the gated
[`bodhan-ai/indic-transcribe-core`](https://huggingface.co/bodhan-ai/indic-transcribe-core)
repo accepted for `transcribe` — request access on its HF page with the same
account this token belongs to; approval is usually quick but isn't instant.
**If you only need `transcribe` (see below), that's the only gated repo you
need approved — you don't need the pyannote/DialogueSidon ones.**

DialogueSidon inference needs the authors' script, which is not redistributed
here — get it from
[sarulab-speech/DialogueSidon-demo](https://huggingface.co/spaces/sarulab-speech/DialogueSidon-demo),
place it alongside the pipeline, and point at it with `--sidon-script`.

## Usage

```bash
# 1. collect candidate URLs
python duplex_corpus.py discover --query "hindi podcast interview" \
    --limit 200 --out urls.txt

# 2. run the pipeline
python duplex_corpus.py run --urls urls.txt --root ./corpus

# 3. inspect
python duplex_corpus.py stats --root ./corpus    # funnel by rejection reason
python duplex_corpus.py check --root ./corpus    # validate separated output
```

## Transcription (word-level timestamps)

A separate pass, independent of `run` — it only needs `corpus/accepted/`
to already exist (from your own run of this pipeline, or any dataset already
laid out in the same shape; see below).

### Quickstart: annotating data you already scraped

If you already have accepted clips (from this pipeline or your own scraping
run) and just need to run transcription/annotation on them:

```bash
git clone https://github.com/Vandan31/indic_duplex_pipeline.git
cd indic_duplex_pipeline
pip install -r requirements.txt        # ffmpeg/ffprobe must be on PATH
cp .env.example .env                   # fill in HF_TOKEN -- the only key
                                        # this stage needs
```

Request access to the gated
[`bodhan-ai/indic-transcribe-core`](https://huggingface.co/bodhan-ai/indic-transcribe-core)
model on your HF account (same account as `HF_TOKEN`) *before* your first
run — approval isn't instant, so do this first. You do **not** need the
pyannote/DialogueSidon gated repos for transcription alone.

Then point `--root` at wherever your data lives — it just needs to match the
layout below (rename/symlink if it doesn't already):

```bash
python duplex_corpus.py transcribe --root /path/to/your/corpus --gpus 0,1,2,3
```

That's it. It's resumable (safe to Ctrl-C and rerun — skips clips that
already have `transcript.json`, pass `--overwrite` to redo) and scales
linearly with however many GPUs you list in `--gpus`, so for 4,000h you'll
want as many as you can get. One worker thread per GPU; each accepted video
is transcribed independently.

**Expected input layout**, per `accepted/<id>/`:

```
accepted/<id>/
  spk0.wav, spk1.wav   per-speaker channels (required)
  meta.json            must have "language_code" (e.g. "hi") and, optionally,
                        "speaker_map" (defaults to {"spk0":"SPEAKER_00",
                        "spk1":"SPEAKER_01"})
  segments.jsonl        optional — used to group words into turns; without it
                        you still get transcript.json's flat word list, just
                        no turns[]
```

If your existing data doesn't have this exact layout, the only hard
requirements are the two speaker-channel wavs and `meta.json`'s
`language_code` — adapt your directory structure (or add a thin wrapper
script) to match rather than reworking the pipeline.

**Pipeline:** ASR is
[`bodhan-ai/indic-transcribe-core`](https://huggingface.co/bodhan-ai/indic-transcribe-core)
(single model, native output across 27 Indic languages — no per-language
model selection needed). Word-level timestamps come from CTC forced alignment
(real [`uroman`](https://pypi.org/project/uroman/) romanization feeding the
`ctc-forced-aligner` package's ONNX aligner — *not* that package's own
`romanize=True` option, which silently uses `unidecode` instead of real
uroman for Devanagari and produces wrong alignments).

**Long clips and crash-safety.** Forced alignment used to crash natively
(SIGABRT/SIGSEGV inside onnxruntime) on long clips, because the alignment DP
ran over the whole clip in one shot, so memory/compute grew with clip length
(crash rate hit ~47% on a stretch of ~28-minute clips). Two fixes are built in:

- **Chunked alignment:** every clip is split into pieces of at most 8 minutes
  (audio by time, transcript proportionally by word position) and each piece
  is aligned independently, so no single alignment call ever sees a long
  clip. After this fix, 0 crashes in 273 consecutive clips, and alignment got
  faster too (e.g. 832s -> 407s on a 25-minute clip). No duration cap is
  needed — clips of any length get real alignment.
- **Capped onnxruntime threads (8 per alignment worker):** left alone,
  onnxruntime sized its thread pool to the node's full core count (~110
  threads per worker), which oversubscribed the CPUs once several GPU workers
  ran at once. Note that alignment runs on CPU in this setup (the installed
  onnxruntime has no CUDA provider), so with many `--gpus` workers make sure
  the machine has roughly 8+ free CPU cores per worker.

Alignment still runs in an isolated subprocess as a backstop: if it ever
crashes or times out for a clip, that one clip automatically falls back to
`faster-whisper` (real word timestamps, just inheriting that model's known
decoder-loop/timestamp-collapse quirks) and, if even that fails, to
evenly-spaced estimated timestamps flagged with `"approx_timestamps": true`
per word so they're easy to filter out downstream. Validated against the old
all-Whisper pipeline on a 200-clip random sample (2026-09-24): ~1% mean
4-gram repetition rate vs ~7% for Whisper, and no stuck/duplicate word
timestamps vs ~78% of Whisper clips affected.

Rough throughput to plan around: about 10 hours of audio per wall-clock hour
on 8 GPUs (measured on long-clip-heavy data), so budget accordingly for a
multi-thousand-hour corpus.

Output, written to `accepted/<id>/transcript.json`:

```json
{
  "id": "...",
  "language_code": "hi",
  "models": {"spk0": "bodhan-ai/indic-transcribe-core", "spk1": "..."},
  "words": [{"speaker": "SPEAKER_00", "word": "...", "start": 1.23, "end": 1.45}, ...],
  "turns": [{"speaker": "SPEAKER_00", "start": 1.2, "end": 4.5, "text": "..."}, ...]
}
```

`models` records which pipeline actually produced each speaker's words —
`bodhan-ai/indic-transcribe-core` for the normal path, a `vasista22/whisper-*`
or `openai/whisper-large-v3` id if that clip hit the Whisper fallback. Check
this field (or the per-word `approx_timestamps` flag) if you need to exclude
fallback clips from anything timestamp-sensitive.

Resumable — rerun the same command after an interruption and it continues.
`stats` breaks rejections down by reason, which is how you tune thresholds: if
most clips die at `not_conversational`, the search queries are wrong, not the
gates.

### Verifying a single file

```bash
python nemo_audio.py recording.wav
```

Reports speaker count, language, tone, background sounds and a summary; writes
`recording_analysis.json` and `.md` beside the input.

## Notes and limitations

- **Separation quality on Indic is unverified.** DialogueSidon was trained
  mainly on English. Listen to `spk0.wav` end to end before scaling up, and
  confirm it is the same voice throughout.
- **DialogueSidon resynthesises.** Output is generated audio, not the original
  recording with one voice removed. Models trained on it learn vocoder
  acoustics.
- **`--separator mask` leaks.** The fallback zeroes the mixture outside each
  speaker's segments; during overlap both voices remain in both channels.
  `segments.jsonl` records `overlap_frac` so those regions can be excluded.
- **Licensing.** Downloading from YouTube is against its Terms of Service, and
  the videos are copyrighted regardless. `--cc-only` restricts to Creative
  Commons. For any public release, distribute URLs and timings rather than
  audio.

## Built on

| Component | Role |
|---|---|
| [DialogueSidon](https://huggingface.co/sarulab-speech/DialogueSidon) | joint separation and restoration of two-speaker mixtures |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | speaker diarization |
| [Silero VAD](https://github.com/snakers4/silero-vad) | voice activity detection |
| NVIDIA Nemotron Omni | audio-level content verification |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | audio retrieval |

Related work: [DuplexChat](https://github.com/sarulab-speech/DuplexChat)
(Nakata et al., arXiv:2607.04941) builds a comparable corpus for English and
Japanese from podcast RSS feeds, at far larger scale.

## Changelog

- **2026-09-26 — Chunking updated in annotation (`transcribe` stage).** Forced
  alignment now splits each clip into pieces of at most 8 minutes and aligns
  them independently, instead of aligning the whole clip in one shot. This
  removes the native alignment crashes on long clips and is faster. If you
  annotated data with an earlier version of this repo, clips that ended up
  with a Whisper model in `transcript.json`'s `models` field were fallbacks
  from those crashes and are worth re-running with `--overwrite` after
  pulling. Also caps onnxruntime at 8 threads per alignment worker.
- **2026-09-24 — Initial `transcribe` stage:** `indic-transcribe-core` ASR +
  CTC forced alignment, with crash-isolated alignment and Whisper fallback.

## License

MIT for this code. Models and datasets carry their own licenses, and crawled
audio remains the property of its rights holders.
