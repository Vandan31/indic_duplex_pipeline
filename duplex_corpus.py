import argparse
import base64
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import requests
import soundfile as sf


def _preload_cudnn():
    """
    Torch wheels ship cuDNN inside site-packages/nvidia/cudnn/lib, but the dynamic
    loader doesn't look there unless LD_LIBRARY_PATH says so. Loading the .so
    explicitly with RTLD_GLOBAL fixes 'libcudnn.so.9: cannot open shared object
    file' without touching the environment.
    """
    import ctypes, glob, site
    roots = []
    try:
        import nvidia
        roots += [Path(p).parent for p in nvidia.__path__]
    except ImportError:
        pass
    roots += [Path(p) for p in site.getsitepackages()]

    loaded = []
    for root in roots:
        for pat in ("nvidia/cudnn/lib/libcudnn*.so.9",
                    "nvidia/cublas/lib/libcublas*.so.12",
                    "nvidia/cuda_runtime/lib/libcudart*.so.12"):
            for so in sorted(glob.glob(str(root / pat))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(Path(so).name)
                except OSError:
                    pass
        if loaded:
            break
    return loaded


_CUDNN_LOADED = _preload_cudnn()

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
# Hardcoded. Not read from the environment.
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

# Hugging Face token for pyannote 3.1 (gated model). Paste yours here.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

SR = 16000

# Scheduled Eighth Schedule languages + widely-used regional ones.
INDIC = {
    "hi": "Hindi",        "bn": "Bengali",    "ta": "Tamil",     "te": "Telugu",
    "mr": "Marathi",      "gu": "Gujarati",   "kn": "Kannada",   "ml": "Malayalam",
    "pa": "Punjabi",      "or": "Odia",       "as": "Assamese",  "ur": "Urdu",
    "ne": "Nepali",       "si": "Sinhala",    "sd": "Sindhi",    "ks": "Kashmiri",
    "mai": "Maithili",    "sa": "Sanskrit",   "kok": "Konkani",  "doi": "Dogri",
    "mni": "Manipuri",    "brx": "Bodo",      "sat": "Santali",  "bho": "Bhojpuri",
    "raj": "Rajasthani",  "mag": "Magahi",    "tcy": "Tulu",
}

@dataclass
class Thresholds:
    min_duration: float = 120.0        # seconds; shorter clips rarely have real turn-taking
    max_duration: float = 14400.0      # 4h; long-form podcasts routinely exceed 2h
    min_speech_ratio: float = 0.45     # VAD: below this it's music/silence-heavy
    max_speech_ratio: float = 0.98     # above this is often a continuous monologue read
    min_turns: int = 20                # speaker changes across the file
    min_turns_per_min: float = 3.0
    max_speaker_imbalance: float = 0.80  # dominant speaker's share of speech time
    min_overlap_ratio: float = 0.005   # some overlap = real conversation, not stitched VO
    max_overlap_ratio: float = 0.35    # too much = crosstalk mess or bad diarization
    min_segment_dur: float = 0.30      # drop diarization crumbs
    verify_windows: int = 3            # how many clips to send to Nemotron
    verify_window_sec: int = 45

TH = Thresholds()

REJECT_REASONS = [
    "probe_failed", "too_short", "too_long", "license", "download_failed",
    "low_speech", "high_speech", "verify_failed", "not_two_speakers",
    "not_indic", "not_conversational", "diarize_failed", "diar_speaker_count",
    "few_turns", "imbalanced", "overlap_out_of_range", "separate_failed", "emit_failed",
]


# ----------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------

def log(msg, level="INFO"):
    print(f"[{time.strftime('%H:%M:%S')}] {level:5s} {msg}", flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def need(binary):
    if not shutil.which(binary):
        sys.exit(f"ERROR: `{binary}` not found on PATH")


# ----------------------------------------------------------------------------
# state store — resume-safe
# ----------------------------------------------------------------------------

class Store:
    def __init__(self, root: Path):
        self.path = root / "state.db"
        self.db = sqlite3.connect(self.path)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                vid TEXT PRIMARY KEY,
                url TEXT,
                status TEXT,          -- pending | accepted | rejected | error
                reason TEXT,
                stage TEXT,
                meta TEXT,
                updated REAL
            )
        """)
        self.db.commit()

    def get(self, vid):
        r = self.db.execute("SELECT status, reason, stage FROM videos WHERE vid=?", (vid,)).fetchone()
        return r

    def set(self, vid, url, status, reason="", stage="", meta=None):
        self.db.execute(
            "INSERT OR REPLACE INTO videos VALUES (?,?,?,?,?,?,?)",
            (vid, url, status, reason, stage, json.dumps(meta or {}, ensure_ascii=False), time.time()),
        )
        self.db.commit()

    def counts(self):
        rows = self.db.execute(
            "SELECT status, reason, COUNT(*) FROM videos GROUP BY status, reason"
        ).fetchall()
        return rows

    def accepted(self):
        return [r[0] for r in self.db.execute(
            "SELECT vid FROM videos WHERE status='accepted'").fetchall()]


# ----------------------------------------------------------------------------
# stage 1 — discovery
# ----------------------------------------------------------------------------

def discover(args):
    """Collect candidate URLs via yt-dlp search or channel/playlist expansion."""
    need("yt-dlp")
    urls = []

    sources = []
    if args.query:
        # ytsearchN: runs YouTube's own search
        n = args.limit
        if args.cc_only:
            # sp=EgIwAQ%253D%253D is YouTube's Creative Commons filter
            sources.append(f"https://www.youtube.com/results?search_query="
                           f"{requests.utils.quote(args.query)}&sp=EgIwAQ%253D%253D")
        else:
            sources.append(f"ytsearch{n}:{args.query}")
    sources += args.source or []

    for src in sources:
        log(f"discovering: {src}")
        cmd = ["yt-dlp", "--flat-playlist", "--dump-json",
               "--playlist-end", str(args.limit), src]
        try:
            out = run(cmd).stdout
        except subprocess.CalledProcessError as e:
            log(f"discovery failed: {e.stderr[:300]}", "WARN")
            continue
        for line in out.splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = d.get("id")
            if not vid:
                continue
            urls.append(d.get("url") or f"https://www.youtube.com/watch?v={vid}")

    urls = list(dict.fromkeys(urls))
    Path(args.out).write_text("\n".join(urls), encoding="utf-8")
    log(f"wrote {len(urls)} urls -> {args.out}")


# ----------------------------------------------------------------------------
# stage 2 — probe + fetch
# ----------------------------------------------------------------------------

def probe(url):
    cmd = ["yt-dlp", "-J", "--no-warnings", "--skip-download", url]
    d = json.loads(run(cmd).stdout)
    return {
        "id": d.get("id"),
        "title": d.get("title"),
        "duration": d.get("duration") or 0,
        "license": d.get("license") or "",
        "channel": d.get("channel") or d.get("uploader"),
        "channel_id": d.get("channel_id"),
        "upload_date": d.get("upload_date"),
        "webpage_url": d.get("webpage_url") or url,
        "language": d.get("language"),
        "categories": d.get("categories"),
        "tags": (d.get("tags") or [])[:20],
    }


def fetch_audio(url, dest_wav: Path):
    """Audio-only download, normalised to 16 kHz mono wav."""
    with tempfile.TemporaryDirectory() as td:
        tmpl = str(Path(td) / "a.%(ext)s")
        run(["yt-dlp", "-f", "bestaudio/best", "--no-warnings",
             "-x", "--audio-format", "wav", "-o", tmpl, url])
        got = next(Path(td).glob("a.*"))
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(got), "-ar", str(SR), "-ac", "1", str(dest_wav)])
    return dest_wav


# ----------------------------------------------------------------------------
# stage 3 — VAD prefilter
# ----------------------------------------------------------------------------

_vad_cache = {}

def speech_ratio(wav: Path) -> float:
    """Fraction of the file that is speech, per Silero VAD."""
    try:
        import torch
        if "model" not in _vad_cache:
            # VAD is tiny — keep it on CPU so it can never trip over CUDA/cuDNN.
            model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                          trust_repo=True, onnx=False)
            model.cpu()
            _vad_cache["model"] = model
            _vad_cache["get_ts"] = utils[0]
        audio, sr = sf.read(str(wav), dtype="float32")
        t = torch.from_numpy(audio).cpu()
        ts = _vad_cache["get_ts"](t, _vad_cache["model"], sampling_rate=SR)
        speech = sum(s["end"] - s["start"] for s in ts) / SR
        return speech / (len(audio) / SR) if len(audio) else 0.0
    except Exception as e:
        log(f"VAD unavailable ({e}); skipping prefilter", "WARN")
        return 0.6  # neutral value, lets the file through to the model


# ----------------------------------------------------------------------------
# stage 4 — Nemotron verification
# ----------------------------------------------------------------------------

VERIFY_PROMPT = """Listen to this audio excerpt and answer as strict JSON only. No markdown fences, no prose.

{
  "num_speakers": <integer count of distinct human voices>,
  "language": "<primary language name>",
  "language_code": "<ISO 639-1/639-3 code>",
  "is_conversational": <true if two or more people are talking WITH each other, taking turns; false for monologue, lecture to camera, narration, voiceover, or reading>,
  "has_turn_taking": <true if speakers alternate and respond to each other>,
  "is_dubbed_or_voiceover": <true if this sounds like dubbing, TTS, or studio narration laid over content>,
  "music_dominant": <true if music/singing dominates over speech>,
  "background_noise": "<clean|moderate|heavy>",
  "confidence": "<high|medium|low>",
  "notes": "<one sentence>"
}

Be strict. A single host talking to camera is NOT conversational even if lively. A phone interview with two people IS."""


def nemo_verify_window(wav: Path, api_key: str, retries=3):
    b64 = base64.b64encode(wav.read_bytes()).decode()
    payload = {
        "model": NVIDIA_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}},
            {"type": "text", "text": VERIFY_PROMPT},
        ]}],
        "max_tokens": 4096,
        "temperature": 0.1,
        "top_p": 0.95,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "NVCF-POLL-SECONDS": "1800",
    }
    delay = 5
    for attempt in range(1, retries + 1):
        r = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=900)
        if r.status_code == 200:
            txt = r.json()["choices"][0]["message"].get("content", "")
            return parse_json(txt)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(delay); delay *= 2
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return None


def parse_json(text):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(cleaned[s:e + 1])
            except json.JSONDecodeError:
                return None
    return None


def slice_windows(wav: Path, duration: float, workdir: Path, n: int, win: int):
    """Take n evenly spaced windows, avoiding intros and outros."""
    usable_start, usable_end = duration * 0.10, duration * 0.90
    span = max(usable_end - usable_start, win)
    outs = []
    for i in range(n):
        pos = usable_start + span * (i + 0.5) / n - win / 2
        pos = max(0, min(pos, duration - win))
        out = workdir / f"win{i}.wav"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-ss", f"{pos:.2f}", "-t", str(win), "-i", str(wav),
             "-ar", str(SR), "-ac", "1", str(out)])
        outs.append(out)
    return outs


def verify(wav: Path, duration: float, api_key: str):
    """Vote across several windows so one bad excerpt doesn't decide the file."""
    with tempfile.TemporaryDirectory() as td:
        wins = slice_windows(wav, duration, Path(td), TH.verify_windows, TH.verify_window_sec)
        votes = []
        for w in wins:
            try:
                v = nemo_verify_window(w, api_key)
                if v:
                    votes.append(v)
            except Exception as e:
                log(f"  verify window failed: {e}", "WARN")
            time.sleep(1)

    if not votes:
        return None

    def majority(key, default=None):
        vals = [v.get(key) for v in votes if v.get(key) is not None]
        if not vals:
            return default
        return max(set(map(str, vals)), key=lambda x: list(map(str, vals)).count(x))

    speaker_counts = [v.get("num_speakers") for v in votes if isinstance(v.get("num_speakers"), int)]
    return {
        "num_speakers": int(np.median(speaker_counts)) if speaker_counts else 0,
        "speaker_votes": speaker_counts,
        "language": majority("language", ""),
        "language_code": (majority("language_code", "") or "").lower(),
        "is_conversational": majority("is_conversational") == "True",
        "has_turn_taking": majority("has_turn_taking") == "True",
        "is_dubbed_or_voiceover": majority("is_dubbed_or_voiceover") == "True",
        "music_dominant": majority("music_dominant") == "True",
        "background_noise": majority("background_noise", "moderate"),
        "confidence": majority("confidence", "low"),
        "n_windows": len(votes),
        "raw": votes,
    }


def is_indic(code: str, name: str) -> bool:
    code = (code or "").lower().split("-")[0]
    if code in INDIC:
        return True
    name = (name or "").strip().lower()
    return any(name == v.lower() for v in INDIC.values())


# ----------------------------------------------------------------------------
# stage 5 — diarization
# ----------------------------------------------------------------------------

_diar_cache = {}

def diarize_pyannote(wav: Path, num_speakers=2):
    from pyannote.audio import Pipeline
    import torch
    if "pyannote" not in _diar_cache:
        token = HF_TOKEN or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "No HF token. Paste yours into HF_TOKEN at the top of this script "
                "and accept the terms on the pyannote/speaker-diarization-3.1 model page."
            )
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token=token)
        if pipe is None:
            raise RuntimeError(
                "pyannote returned None — the token is valid but you have not accepted "
                "the terms on huggingface.co/pyannote/speaker-diarization-3.1 "
                "AND huggingface.co/pyannote/segmentation-3.0 (both are gated)."
            )
        if torch.cuda.is_available():
            try:
                pipe.to(torch.device("cuda"))
            except Exception as e:
                log(f"  GPU init failed ({str(e)[:80]}), running diarization on CPU", "WARN")
        _diar_cache["pyannote"] = pipe
    ann = _diar_cache["pyannote"](str(wav), num_speakers=num_speakers)
    segs = [{"start": float(t.start), "end": float(t.end), "speaker": str(spk)}
            for t, _, spk in ann.itertracks(yield_label=True)]
    return sorted(segs, key=lambda s: s["start"])


def diarize_nemo(wav: Path, num_speakers=2):
    """NVIDIA Sortformer. Verify the class path against your NeMo version."""
    from nemo.collections.asr.models import SortformerEncLabelModel
    import torch
    if "nemo" not in _diar_cache:
        m = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
        m.eval()
        if torch.cuda.is_available():
            m = m.cuda()
        _diar_cache["nemo"] = m
    preds = _diar_cache["nemo"].diarize(audio=[str(wav)], batch_size=1)
    segs = []
    for entry in preds[0]:
        # Sortformer emits "start end speaker_id" strings
        parts = str(entry).split()
        if len(parts) >= 3:
            segs.append({"start": float(parts[0]), "end": float(parts[1]),
                         "speaker": f"SPEAKER_{parts[2]}"})
    return sorted(segs, key=lambda s: s["start"])


def diarize(wav: Path, backend: str, num_speakers=2):
    segs = (diarize_nemo if backend == "nemo" else diarize_pyannote)(wav, num_speakers)
    return [s for s in segs if s["end"] - s["start"] >= TH.min_segment_dur]


# ----------------------------------------------------------------------------
# separation — DialogueSidon
# ----------------------------------------------------------------------------

_sidon_cache = {}

def load_sidon(script_path: str, device_str: str):
    """
    Import the author's inference script as a module and load the exported
    components once. DialogueSidon SEPARATES a two-speaker mixture into two
    waveforms — it does not diarize and does not label speakers over time.
    """
    if "mod" in _sidon_cache:
        return _sidon_cache["mod"], _sidon_cache["models"], _sidon_cache["device"]

    import importlib.util
    import torch

    p = Path(script_path).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"DialogueSidon script not found at {p} — pass --sidon-script")

    spec = importlib.util.spec_from_file_location("dialogue_sidon_infer", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    log(f"  loading DialogueSidon on {device} (first call only)")
    t0 = time.time()
    models = mod.load_models(device)
    log(f"  DialogueSidon ready in {time.time() - t0:.0f}s")

    _sidon_cache.update({"mod": mod, "models": models, "device": device})
    return mod, models, device


def separate_sidon(wav: Path, script_path: str, device_str: str, num_steps: int):
    """Returns (channels [2, N] float32 numpy, out_sample_rate)."""
    mod, models, device = load_sidon(script_path, device_str)
    audio, sr = mod.load_audio(str(wav))
    t0 = time.time()
    sep, out_sr = mod.separate(audio, sr, num_steps, models, device)
    sep = sep.float().cpu().numpy()
    dur = sep.shape[-1] / out_sr
    log(f"  separated {dur:.0f}s audio in {time.time() - t0:.0f}s "
        f"(RTF {(time.time() - t0) / max(dur, 1e-6):.3f})")
    peak = max(float(np.abs(sep).max()), 1e-6)
    return (sep / peak * 0.9).astype("float32"), out_sr


def vad_segments(signal: np.ndarray, sr: int, label: str):
    """Silero VAD on a single separated channel -> labelled segments."""
    import torch
    if "model" not in _vad_cache:
        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                      trust_repo=True, onnx=False)
        model.cpu()
        _vad_cache["model"] = model
        _vad_cache["get_ts"] = utils[0]

    x = signal
    if sr != SR:
        import scipy.signal as ss
        n = int(round(len(x) * SR / sr))
        x = ss.resample(x, n).astype("float32")

    ts = _vad_cache["get_ts"](torch.from_numpy(x).cpu(), _vad_cache["model"],
                              sampling_rate=SR)
    return [{"start": t["start"] / SR, "end": t["end"] / SR, "speaker": label}
            for t in ts]


def diarize_from_channels(channels: np.ndarray, sr: int):
    """
    Derive a diarization from independently separated channels. Overlap falls
    out naturally as simultaneous activity in both — more reliable than asking
    a diarizer to detect it in a mixture.
    """
    segs = []
    for i in range(channels.shape[0]):
        segs += vad_segments(channels[i], sr, f"SPEAKER_{i:02d}")
    segs = sorted(segs, key=lambda s: s["start"])
    return [s for s in segs if s["end"] - s["start"] >= TH.min_segment_dur]


# ----------------------------------------------------------------------------
# stage 6 — turn-taking qualification
# ----------------------------------------------------------------------------

def conversation_stats(segs, duration):
    if not segs:
        return {}
    speakers = sorted({s["speaker"] for s in segs})
    talk = {sp: sum(s["end"] - s["start"] for s in segs if s["speaker"] == sp)
            for sp in speakers}
    total_talk = sum(talk.values()) or 1e-9

    # speaker changes
    turns = sum(1 for a, b in zip(segs, segs[1:]) if a["speaker"] != b["speaker"])

    # overlap: union of pairwise intersections
    overlap = 0.0
    for i, a in enumerate(segs):
        for b in segs[i + 1:]:
            if b["start"] >= a["end"]:
                break
            if b["speaker"] != a["speaker"]:
                overlap += max(0.0, min(a["end"], b["end"]) - b["start"])

    # backchannels: very short turns sandwiched between the other speaker
    backchannels = sum(
        1 for p, c, n in zip(segs, segs[1:], segs[2:])
        if c["end"] - c["start"] < 1.2 and p["speaker"] == n["speaker"] != c["speaker"]
    )

    gaps = [b["start"] - a["end"] for a, b in zip(segs, segs[1:])
            if a["speaker"] != b["speaker"] and b["start"] > a["end"]]

    return {
        "n_speakers": len(speakers),
        "duration": duration,
        "talk_time": {k: round(v, 2) for k, v in talk.items()},
        "speaker_shares": {k: round(v / total_talk, 3) for k, v in talk.items()},
        "dominance": round(max(talk.values()) / total_talk, 3),
        "n_segments": len(segs),
        "n_turns": turns,
        "turns_per_min": round(turns / (duration / 60), 2) if duration else 0,
        "overlap_sec": round(overlap, 2),
        "overlap_ratio": round(overlap / total_talk, 4),
        "backchannels": backchannels,
        "mean_gap": round(float(np.mean(gaps)), 3) if gaps else 0.0,
        "median_turn_dur": round(float(np.median([s["end"] - s["start"] for s in segs])), 3),
        "speech_ratio": round(total_talk / duration, 3) if duration else 0,
    }


def qualify(stats):
    if stats.get("n_speakers") != 2:
        return "diar_speaker_count"
    if stats["n_turns"] < TH.min_turns or stats["turns_per_min"] < TH.min_turns_per_min:
        return "few_turns"
    if stats["dominance"] > TH.max_speaker_imbalance:
        return "imbalanced"
    if not (TH.min_overlap_ratio <= stats["overlap_ratio"] <= TH.max_overlap_ratio):
        return "overlap_out_of_range"
    return None


# ----------------------------------------------------------------------------
# stage 7 — emit
# ----------------------------------------------------------------------------

def write_channels(channels: np.ndarray, sr: int, outdir: Path):
    """Write separated channels produced by DialogueSidon."""
    n = channels.shape[0]
    for i in range(n):
        sf.write(str(outdir / f"spk{i}.wav"), channels[i], sr)
    if n == 2:
        sf.write(str(outdir / "stereo.wav"), np.stack([channels[0], channels[1]], axis=1), sr)
    return [f"SPEAKER_{i:02d}" for i in range(n)]


def build_channels(wav: Path, segs, outdir: Path):
    """
    Fallback path (--separator mask). Masks the mixture per speaker. During
    overlap both voices remain in both channels — this is masking, not source
    separation. Overlap spans are recorded in segments.jsonl.
    """
    audio, sr = sf.read(str(wav), dtype="float32")
    speakers = sorted({s["speaker"] for s in segs})
    chans = {}
    ramp = int(0.02 * sr)  # 20 ms fade to avoid click artefacts at boundaries
    fade_in = np.linspace(0, 1, ramp, dtype="float32")
    fade_out = fade_in[::-1]

    for sp in speakers:
        buf = np.zeros_like(audio)
        for s in segs:
            if s["speaker"] != sp:
                continue
            a, b = int(s["start"] * sr), min(int(s["end"] * sr), len(audio))
            if b <= a:
                continue
            chunk = audio[a:b].copy()
            if len(chunk) > 2 * ramp:
                chunk[:ramp] *= fade_in
                chunk[-ramp:] *= fade_out
            buf[a:b] = np.maximum(np.abs(buf[a:b]), np.abs(chunk)) * np.sign(chunk + 1e-12)
        chans[sp] = buf

    for i, sp in enumerate(speakers):
        sf.write(str(outdir / f"spk{i}.wav"), chans[sp], sr)

    if len(speakers) == 2:
        stereo = np.stack([chans[speakers[0]], chans[speakers[1]]], axis=1)
        sf.write(str(outdir / "stereo.wav"), stereo, sr)

    return speakers


def write_rttm(segs, vid, path: Path):
    lines = []
    for s in segs:
        lines.append(
            f"SPEAKER {vid} 1 {s['start']:.3f} {s['end'] - s['start']:.3f} "
            f"<NA> <NA> {s['speaker']} <NA> <NA>"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mark_overlaps(segs):
    out = []
    for i, s in enumerate(segs):
        ov = 0.0
        for j, t in enumerate(segs):
            if i == j or t["speaker"] == s["speaker"]:
                continue
            ov += max(0.0, min(s["end"], t["end"]) - max(s["start"], t["start"]))
        d = s["end"] - s["start"]
        out.append({**s, "duration": round(d, 3),
                    "overlap_sec": round(ov, 3),
                    "overlap_frac": round(ov / d, 3) if d else 0.0,
                    "clean": ov / d < 0.05 if d else False})
    return out


def emit(vid, wav: Path, segs, stats, meta, verdict, root: Path,
         channels=None, chan_sr=None, separator="sidon"):
    outdir = root / "accepted" / vid
    outdir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(wav, outdir / "mixture.wav")
    if channels is not None:
        speakers = write_channels(channels, chan_sr, outdir)
    else:
        speakers = build_channels(wav, segs, outdir)
    write_rttm(segs, vid, outdir / "diarization.rttm")

    marked = mark_overlaps(segs)
    with open(outdir / "segments.jsonl", "w", encoding="utf-8") as f:
        for s in marked:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    caveat = (
        ["channels are DialogueSidon source-separated; overlap regions are genuinely "
         "separated, not masked"]
        if channels is not None else
        ["channels are diarization-masked, not source-separated; overlap regions "
         "contain leakage — see overlap_frac per segment"]
    )

    record = {
        "id": vid,
        "source": meta.get("webpage_url"),
        "title": meta.get("title"),
        "channel": meta.get("channel"),
        "license": meta.get("license"),
        "upload_date": meta.get("upload_date"),
        "duration": meta.get("duration"),
        "language": verdict.get("language"),
        "language_code": verdict.get("language_code"),
        "verification": verdict,
        "conversation": stats,
        "speaker_map": {f"spk{i}": sp for i, sp in enumerate(speakers)},
        "separator": separator,
        "channel_sample_rate": chan_sr or SR,
        "files": {
            "mixture": "mixture.wav",
            "stereo": "stereo.wav" if len(speakers) == 2 else None,
            "channels": [f"spk{i}.wav" for i in range(len(speakers))],
            "rttm": "diarization.rttm",
            "segments": "segments.jsonl",
        },
        "caveats": caveat,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (outdir / "meta.json").write_text(json.dumps(record, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    with open(root / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return outdir


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def reject(store, vid, url, reason, stage, meta=None):
    store.set(vid, url, "rejected", reason, stage, meta)
    log(f"  REJECT [{reason}] at {stage}")


def process_one(url, root: Path, store: Store, args):
    vid = None
    try:
        meta = probe(url)
    except Exception as e:
        log(f"  probe failed: {str(e)[:200]}", "WARN")
        store.set(url, url, "rejected", "probe_failed", "probe")
        return
    vid = meta["id"]

    prior = store.get(vid)
    if prior and prior[0] in ("accepted", "rejected") and not args.redo:
        log(f"SKIP {vid} ({prior[0]}/{prior[1]})")
        return

    log(f"START {vid} — {(meta['title'] or '')[:70]}")
    dur = meta["duration"]

    if dur < TH.min_duration:
        return reject(store, vid, url, "too_short", "probe", meta)
    if dur > TH.max_duration:
        return reject(store, vid, url, "too_long", "probe", meta)
    if args.cc_only and "creative commons" not in (meta["license"] or "").lower():
        return reject(store, vid, url, "license", "probe", meta)

    raw = root / "raw" / f"{vid}.wav"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        try:
            log("  downloading audio")
            fetch_audio(url, raw)
        except Exception as e:
            return reject(store, vid, url, "download_failed", "fetch", meta)

    log("  VAD prefilter")
    sr_ratio = speech_ratio(raw)
    if sr_ratio < TH.min_speech_ratio:
        return reject(store, vid, url, "low_speech", "prefilter", {**meta, "speech_ratio": sr_ratio})
    if sr_ratio > TH.max_speech_ratio:
        return reject(store, vid, url, "high_speech", "prefilter", {**meta, "speech_ratio": sr_ratio})

    log(f"  verifying with Nemotron ({TH.verify_windows} windows)")
    v = verify(raw, dur, args.api_key)
    if not v:
        return reject(store, vid, url, "verify_failed", "verify", meta)
    log(f"    speakers={v['num_speakers']} lang={v['language']} "
        f"conv={v['is_conversational']} conf={v['confidence']}")

    if v["num_speakers"] != 2:
        return reject(store, vid, url, "not_two_speakers", "verify", {**meta, "verify": v})
    if not is_indic(v["language_code"], v["language"]):
        return reject(store, vid, url, "not_indic", "verify", {**meta, "verify": v})
    if not v["is_conversational"] or v["music_dominant"] or v["is_dubbed_or_voiceover"]:
        return reject(store, vid, url, "not_conversational", "verify", {**meta, "verify": v})

    channels, chan_sr = None, None

    if args.diarizer == "sidon-vad":
        # Separate first, then derive the timeline from each clean channel.
        # No pyannote, no HF gate — but you pay diffusion cost before the
        # turn-taking gate can reject the file.
        log("  separating with DialogueSidon")
        try:
            channels, chan_sr = separate_sidon(raw, args.sidon_script,
                                               args.sidon_device, args.sidon_steps)
        except Exception as e:
            log(f"    {str(e)[:250]}", "WARN")
            return reject(store, vid, url, "separate_failed", "separate", meta)
        segs = diarize_from_channels(channels, chan_sr)
    else:
        # Cheap gate first: diarize the mixture, reject bad conversations before
        # spending GPU minutes on diffusion.
        log(f"  diarizing ({args.diarizer})")
        try:
            segs = diarize(raw, args.diarizer, num_speakers=2)
        except Exception as e:
            log(f"    {str(e)[:200]}", "WARN")
            return reject(store, vid, url, "diarize_failed", "diarize", meta)

    stats = conversation_stats(segs, dur)
    log(f"    turns={stats.get('n_turns')} tpm={stats.get('turns_per_min')} "
        f"dom={stats.get('dominance')} ovl={stats.get('overlap_ratio')}")

    bad = qualify(stats)
    if bad:
        return reject(store, vid, url, bad, "qualify", {**meta, "verify": v, "stats": stats})

    if channels is None and args.separator == "sidon":
        log("  separating with DialogueSidon")
        try:
            channels, chan_sr = separate_sidon(raw, args.sidon_script,
                                               args.sidon_device, args.sidon_steps)
        except Exception as e:
            log(f"    {str(e)[:250]}", "WARN")
            return reject(store, vid, url, "separate_failed", "separate", meta)

    try:
        outdir = emit(vid, raw, segs, stats, meta, v, root,
                      channels=channels, chan_sr=chan_sr,
                      separator=args.separator if channels is not None else "mask")
    except Exception as e:
        log(f"    {str(e)[:200]}", "WARN")
        return reject(store, vid, url, "emit_failed", "emit", meta)

    store.set(vid, url, "accepted", "", "emit", {**meta, "verify": v, "stats": stats})
    log(f"  ACCEPT -> {outdir}")

    if not args.keep_raw:
        raw.unlink(missing_ok=True)


def cmd_run(args):
    need("yt-dlp"); need("ffmpeg"); need("ffprobe")
    if not args.api_key:
        sys.exit("ERROR: set NVIDIA_API_KEY at the top of this script or pass --api-key")

    if getattr(args, "min_duration", None):
        TH.min_duration = args.min_duration
    if getattr(args, "max_duration", None):
        TH.max_duration = args.max_duration

    log(f"cudnn preload: {', '.join(_CUDNN_LOADED) if _CUDNN_LOADED else 'nothing found'}")
    log(f"duration window: {TH.min_duration:.0f}s – {TH.max_duration:.0f}s")

    root = Path(args.root)
    for sub in ("raw", "accepted", "rejected"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    store = Store(root)

    urls = [u.strip() for u in Path(args.urls).read_text().splitlines() if u.strip()]
    if args.limit:
        urls = urls[:args.limit]
    log(f"{len(urls)} candidate url(s)")

    for i, u in enumerate(urls, 1):
        log(f"--- [{i}/{len(urls)}] ---")
        try:
            process_one(u, root, store, args)
        except KeyboardInterrupt:
            log("interrupted"); break
        except Exception as e:
            log(f"unhandled: {str(e)[:300]}", "ERROR")
        time.sleep(args.delay)

    cmd_stats(args)


def cmd_stats(args):
    root = Path(args.root)
    store = Store(root)
    print("\n" + "=" * 58)
    print(f"  {root}")
    print("=" * 58)
    total = 0
    for status, reason, n in sorted(store.counts(), key=lambda r: -r[2]):
        label = f"{status}" + (f" / {reason}" if reason else "")
        print(f"  {label:38s} {n:6d}")
        total += n
    print("-" * 58)
    print(f"  {'total':38s} {total:6d}")

    hours, n = 0.0, 0
    mf = root / "manifest.jsonl"
    if mf.exists():
        for line in mf.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            hours += (r.get("duration") or 0) / 3600
            n += 1
        print(f"  {'accepted hours':38s} {hours:6.1f}")
    print("=" * 58 + "\n")


def cmd_export(args):
    """Flatten to a segment-level manifest for the pretraining dataloader."""
    root = Path(args.root)
    out = open(args.out, "w", encoding="utf-8")
    n = 0
    for d in sorted((root / "accepted").iterdir()):
        mj = d / "meta.json"
        if not mj.exists():
            continue
        meta = json.loads(mj.read_text(encoding="utf-8"))
        rec = {
            "id": meta["id"],
            "stereo_filepath": str((d / "stereo.wav").resolve()),
            "mixture_filepath": str((d / "mixture.wav").resolve()),
            "channel_filepaths": [str((d / c).resolve()) for c in meta["files"]["channels"]],
            "rttm_filepath": str((d / "diarization.rttm").resolve()),
            "duration": meta["duration"],
            "lang": meta["language_code"],
            "overlap_ratio": meta["conversation"].get("overlap_ratio"),
            "turns_per_min": meta["conversation"].get("turns_per_min"),
            "license": meta.get("license"),
            "source": meta.get("source"),
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
    out.close()
    log(f"exported {n} entries -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="collect candidate urls")
    d.add_argument("--query", help="search phrase, e.g. 'marathi podcast interview'")
    d.add_argument("--source", action="append", help="channel/playlist url (repeatable)")
    d.add_argument("--limit", type=int, default=50)
    d.add_argument("--cc-only", action="store_true", help="Creative Commons results only")
    d.add_argument("--out", default="urls.txt")
    d.set_defaults(func=discover)

    r = sub.add_parser("run", help="run the full pipeline")
    r.add_argument("--urls", required=True)
    r.add_argument("--root", default="./corpus")
    r.add_argument("--diarizer", choices=["pyannote", "nemo", "sidon-vad"],
                   default="pyannote",
                   help="sidon-vad: separate first, derive timeline from channels "
                        "(no pyannote/HF needed, but slower per rejected file)")
    r.add_argument("--separator", choices=["sidon", "mask"], default="sidon",
                   help="sidon: real source separation. mask: cheap fallback with "
                        "overlap leakage.")
    r.add_argument("--sidon-script", default="./dialogue_sidon_infer.py")
    r.add_argument("--sidon-steps", type=int, default=30, help="diffusion steps")
    r.add_argument("--sidon-device", default="auto", choices=["auto", "cuda", "cpu"])
    r.add_argument("--api-key", default=NVIDIA_API_KEY)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--delay", type=float, default=2)
    r.add_argument("--cc-only", action="store_true")
    r.add_argument("--keep-raw", action="store_true", help="keep the source mixture in raw/")
    r.add_argument("--redo", action="store_true", help="reprocess already-decided videos")
    r.add_argument("--min-duration", type=float, help="override min seconds")
    r.add_argument("--max-duration", type=float, help="override max seconds")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("stats", help="show funnel counts")
    s.add_argument("--root", default="./corpus")
    s.set_defaults(func=cmd_stats)

    e = sub.add_parser("export", help="flatten to a training manifest")
    e.add_argument("--root", default="./corpus")
    e.add_argument("--out", default="train_manifest.jsonl")
    e.set_defaults(func=cmd_export)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()