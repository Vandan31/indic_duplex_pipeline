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
git clone https://github.com/USERNAME/indic-duplex-pipeline.git
cd indic-duplex-pipeline
pip install -r requirements.txt
# ffmpeg and ffprobe must be on PATH

cp .env.example .env      # fill in your keys
export $(grep -v '^#' .env | xargs)
```

`NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com) (free tier).
`HF_TOKEN` needs the gated `pyannote/speaker-diarization-3.1`,
`pyannote/segmentation-3.0` and `sarulab-speech/DialogueSidon` repos accepted.

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

## License

MIT for this code. Models and datasets carry their own licenses, and crawled
audio remains the property of its rights holders.
