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
`transcribe` also uses
[`ai4bharat/indicwav2vec-hindi`](https://huggingface.co/ai4bharat/indicwav2vec-hindi)
for alignment; that one is auto-approved — just open its page while logged in
with the same account and click to accept the access terms.
**If you only need `transcribe` (see below), those two are the only gated
repos you need — you don't need the pyannote/DialogueSidon ones.**

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
want as many as you can get. With more than one GPU listed, `transcribe`
launches one worker *process* per GPU (each pinned to its GPU and handling the
videos whose id hashes to its shard); each accepted video is transcribed
independently.

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
model selection needed). Word-level timestamps come from CTC forced
alignment with
[`ai4bharat/indicwav2vec-hindi`](https://huggingface.co/ai4bharat/indicwav2vec-hindi)
(Apache-2.0, native Devanagari vocabulary, so no romanization step) via
`torchaudio.functional.forced_align`, on the GPU.

**Long clips.** The model's memory grows linearly with audio length (a 3.4-hour
clip tried to allocate 75 GiB in one tensor), so the *emission* is computed in
overlapping 5-minute windows (10 s of context each side, discarded) and
concatenated, and then the alignment runs **once, globally**, over the whole
clip's emission and full transcript. There is deliberately no per-chunk
transcript splitting: an earlier version (commits `6aecf68` and `33c1cfb`)
split the transcript in proportion to time, which assumes speech is spread
evenly. On separated speaker channels it isn't, and on one 15-minute test clip
about 30% of the words were assigned to the wrong time window. If you ran the
`transcribe` stage with those commits, re-run clips longer than 8 minutes
after pulling.

Measured on 31 clips (2 minutes to 3.4 hours) against the old CPU ONNX aligner:
about 2.6x faster alignment (about 1.8x end-to-end, since ASR becomes about
half of the per-file time), and far fewer words placed on silent audio (2-3%
vs 5-36% on clips over 8 minutes). Quality is judged with ground-truth-free
proxies (word-duration plausibility, agreement with audio energy) — there is
no hand-labelled timestamp data, so spot-check some clips before relying on
sub-100 ms accuracy.

Words the aligner's vocabulary cannot represent (other scripts, digits) are
kept in the transcript with timestamps interpolated between their aligned
neighbours and flagged `"approx_timestamps": true`, rather than dropped.

Alignment still runs in an isolated subprocess as a backstop: if it ever
crashes or fails for a clip, that one clip automatically falls back to
`faster-whisper` (real word timestamps, just inheriting that model's known
decoder-loop/timestamp-collapse quirks) and, if even that fails, to
evenly-spaced estimated timestamps flagged with `"approx_timestamps": true`.
The previous CPU ONNX + uroman aligner is still available with
`ALIGN_BACKEND=onnx` for reference, but it is not recommended for clips longer
than a few minutes. Validated against the old all-Whisper pipeline on a
200-clip random sample (2026-09-24): ~1% mean 4-gram repetition rate vs ~7%
for Whisper, and no stuck/duplicate word timestamps vs ~78% of Whisper clips
affected.

Rough throughput to plan around: about **190 hours of audio per wall-clock
hour on 8 GPUs** (measured on a production run, 60 clips, no failures; short
clips run somewhat slower per audio-hour than long ones), so roughly a day and
a half for 6,500 hours. Earlier versions ran the GPU workers as *threads* of a
single process and topped out around 10 audio-hours per hour no matter how many
GPUs were listed: the ASR decodes token by token in Python, so the threads
serialized on the interpreter lock (node CPU load ~4, GPUs mostly idle).
Worker processes remove that limit.

Output, written to `accepted/<id>/transcript.json`:

```json
{
  "id": "...",
  "language_code": "hi",
  "models": {"spk0": "bodhan-ai/indic-transcribe-core", "spk1": "..."},
  "aligner": "iwv",
  "end_fix": "v1",
  "asr_stats": {"spk0": {"n_letters": 5210, "devanagari_frac": 0.998, "n_asr_words": 1204, "n_approx_words": 3}},
  "suspect_language": false,
  "words": [{"speaker": "SPEAKER_00", "word": "...", "start": 1.23, "end": 1.45}, ...],
  "turns": [{"speaker": "SPEAKER_00", "start": 1.2, "end": 4.5, "text": "..."}, ...]
}
```

`models` records which pipeline actually produced each speaker's words —
`bodhan-ai/indic-transcribe-core` for the normal path, a `vasista22/whisper-*`
or `openai/whisper-large-v3` id if that clip hit the Whisper fallback. Check
this field (or the per-word `approx_timestamps` flag) if you need to exclude
fallback clips from anything timestamp-sensitive. **Word start times are the
reliable field.** CTC alignment marks where each character's sound peaks, so a
raw word span runs only from its first to its last character peak and ends
early (median 0.15 s, ~4% of words under 40 ms). `end_fix: "v1"` means each word's
end has been extended toward the next word's start (fully if the gap is at most
0.25 s, otherwise by 0.10 s into the pause, never across the next word), which
brings the median to ~0.22 s and words under 40 ms to ~0.06%. Ends are still
approximate. `aligner` is `iwv`
(IndicWav2Vec, current) or `onnx` (legacy) — use it to find clips annotated by
an older version. `suspect_language` is true when a speaker's transcript is
under 50% Devanagari letters (with at least 50 letters), which usually means
the audio is not Hindi (a language-verification false positive); filter these
before training on a Hindi-only set.

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

- **2026-09-26 (evening) — Word end times extended.** Raw CTC word spans end
  systematically early (they cover only first-to-last character peak: median
  0.15 s, ~4% of words under 40 ms). Each word's end is now extended toward the
  next word's start (gap up to 0.25 s closed entirely, longer pauses get a 0.10 s
  tail, never across the next word). Parameters were chosen by measurement against
  audio energy on 80 delivered speaker channels: speech not covered by any word
  fell from 37.7% to 14.4%. Transcripts carry `"end_fix": "v1"`; ones written by
  earlier versions lack it, and can be brought up to date from their word lists
  alone with the same rule (`_extend_word_ends`).
- **2026-09-26 (latest) — One worker process per GPU.** `transcribe --gpus
  0,1,2,3` previously ran its GPU workers as threads in one process, which the
  interpreter lock capped at ~10 audio-hours/hour regardless of GPU count. It now
  spawns one pinned worker process per GPU (~190 audio-hours/hour on 8 GPUs
  measured, ~19x faster). Results are identical; a single GPU behaves as before.
- **2026-09-26 (later) — Alignment switched to IndicWav2Vec, global instead of
  chunked.** The proportional chunking below turned out to misplace words on
  long clips (speech is not evenly spread over a clip, so words were forced
  into the wrong chunk; on one clip ~30% of words). Alignment now uses
  `ai4bharat/indicwav2vec-hindi` on GPU with windowed emissions and one global
  alignment per clip: no transcript splitting, no crashes on long clips, about
  2x faster end-to-end. **If you annotated clips longer than 8 minutes with
  commits `6aecf68` / `33c1cfb`, re-run them after pulling.** `transcript.json`
  now also records `aligner`, per-speaker `asr_stats` and a `suspect_language`
  flag; words the aligner cannot place are kept with interpolated timestamps
  and `approx_timestamps: true`. Accept the terms for the new model on its HF
  page (auto-approved) before your first run.
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
