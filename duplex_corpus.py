import argparse
import base64
import contextlib
import functools
import json
import math
import multiprocessing as mp
import os
import queue
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# ASR model for word-level-timestamped transcription (stage 8). Replaced the
# per-language faster-whisper fine-tunes with a single model covering all 27
# Indic languages natively -- validated against the old pipeline on a 200-clip
# stratified random sample (2026-09-24): near-zero decoder-loop repetition
# (1.2% vs 6.8% mean 4-gram repetition rate) and zero stuck/duplicate word
# timestamps (vs 78% of old-pipeline clips affected), at the cost of forced
# alignment (below) being a separate, isolated step.
ASR_MODEL_ID = "bodhan-ai/indic-transcribe-core"


@dataclass
class Thresholds:
    min_duration: float = 120.0        # seconds; shorter clips rarely have real turn-taking
    max_duration: float = 14400.0      # 4h; long-form podcasts routinely exceed 2h
    min_speech_ratio: float = 0.45     # VAD: below this it's music/silence-heavy
    max_speech_ratio: float = 0.98     # above this is often a continuous monologue read
    min_turns: int = 20                # speaker changes across the file
    min_turns_per_min: float = 3.0
    max_speaker_imbalance: float = 0.75  # dominant speaker's share of speech time
    min_overlap_ratio: float = 0.005   # some overlap = real conversation, not stitched VO
    max_overlap_ratio: float = 0.35    # too much = crosstalk mess or bad diarization
    min_segment_dur: float = 0.30      # drop diarization crumbs
    verify_windows: int = 5            # how many clips to send to Nemotron
    verify_window_sec: int = 45
    min_lang_agreement: float = 0.6    # winning (code, name) pair's vote share required
    min_lang_votes: int = 2            # fewer surviving window votes than this -> verify_failed

TH = Thresholds()

REJECT_REASONS = [
    "probe_failed", "too_short", "too_long", "license", "download_failed",
    "low_speech", "high_speech", "verify_failed", "not_two_speakers",
    "not_indic", "not_conversational", "diarize_failed", "diar_speaker_count",
    "few_turns", "imbalanced", "overlap_out_of_range", "separate_failed", "emit_failed",
    "no_qualifying_2spk_segment",
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
    """Shared across light- and heavy-stage worker threads. sqlite3 connections
    aren't safe to use from multiple threads by default, and write volume here
    is tiny (one small row per video) and never the bottleneck, so a single
    connection guarded by one lock is simpler than per-thread connections."""

    def __init__(self, root: Path):
        self.path = root / "state.db"
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
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
        with self._lock:
            return self.db.execute(
                "SELECT status, reason, stage FROM videos WHERE vid=?", (vid,)).fetchone()

    def set(self, vid, url, status, reason="", stage="", meta=None):
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO videos VALUES (?,?,?,?,?,?,?)",
                (vid, url, status, reason, stage, json.dumps(meta or {}, ensure_ascii=False), time.time()),
            )
            self.db.commit()

    def counts(self):
        with self._lock:
            return self.db.execute(
                "SELECT status, reason, COUNT(*) FROM videos GROUP BY status, reason"
            ).fetchall()

    def accepted(self):
        with self._lock:
            return [r[0] for r in self.db.execute(
                "SELECT vid FROM videos WHERE status='accepted'").fetchall()]


# ----------------------------------------------------------------------------
# stage 1 — discovery
# ----------------------------------------------------------------------------

def ytdlp_auth_args(args):
    """--cookies (a Netscape cookies.txt) or --cookies-from-browser (read a
    local browser's live session), passed straight through to yt-dlp. Needed
    once YouTube starts issuing 'Sign in to confirm you're not a bot' to an
    IP doing sustained automated access — authenticated requests are far more
    resistant to that than anonymous ones."""
    if getattr(args, "cookies", None):
        return ["--cookies", args.cookies]
    if getattr(args, "cookies_from_browser", None):
        return ["--cookies-from-browser", args.cookies_from_browser]
    return []


def discover(args):
    """Collect candidate URLs via yt-dlp search or channel/playlist expansion."""
    need("yt-dlp")
    urls = []
    auth_args = ytdlp_auth_args(args)

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
        cmd = ["yt-dlp", "--flat-playlist", "--dump-json", *auth_args,
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

def probe(url, auth_args=()):
    cmd = ["yt-dlp", "-J", "--no-warnings", "--skip-download", *auth_args, url]
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


def fetch_audio(url, dest_wav: Path, auth_args=()):
    """Audio-only download, normalised to 16 kHz mono wav."""
    with tempfile.TemporaryDirectory() as td:
        tmpl = str(Path(td) / "a.%(ext)s")
        run(["yt-dlp", "-f", "bestaudio/best", "--no-warnings", *auth_args,
             "-x", "--audio-format", "wav", "-o", tmpl, url])
        got = next(Path(td).glob("a.*"))
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(got), "-ar", str(SR), "-ac", "1", str(dest_wav)])
    return dest_wav


# ----------------------------------------------------------------------------
# stage 3 — VAD prefilter
# ----------------------------------------------------------------------------

_model_lock = threading.Lock()

def _get_or_load(cache: dict, key, loader):
    """Thread-safe lazy singleton: many worker threads may race to load the
    same (possibly multi-GB) model on first use. Double-checked locking so
    only one thread ever loads it, and later callers just hit the cache."""
    if key in cache:
        return cache[key]
    with _model_lock:
        if key not in cache:
            cache[key] = loader()
    return cache[key]


_vad_cache = {}
# Silero's TorchScript model isn't safe to call concurrently from multiple
# threads on one shared instance — it segfaults the whole interpreter rather
# than raising a catchable Python exception. VAD is cheap, so serializing
# inference (not just the one-time load) costs little and removes the crash.
_vad_infer_lock = threading.Lock()

def _load_vad():
    import torch
    # VAD is tiny — keep it on CPU so it can never trip over CUDA/cuDNN.
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                  trust_repo=True, onnx=False)
    model.cpu()
    return {"model": model, "get_ts": utils[0]}


def speech_ratio(wav: Path) -> float:
    """Fraction of the file that is speech, per Silero VAD."""
    try:
        import torch
        vad = _get_or_load(_vad_cache, "vad", _load_vad)
        audio, sr = sf.read(str(wav), dtype="float32")
        t = torch.from_numpy(audio).cpu()
        with _vad_infer_lock:
            ts = vad["get_ts"](t, vad["model"], sampling_rate=SR)
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


def aggregate_language_votes(votes, min_agreement):
    """Joint (language_code, language) vote across verify()'s window results.

    language and language_code used to be voted on INDEPENDENTLY, which let a
    split window vote produce a code/name pair that disagree with each other
    (e.g. code="te", name="Hindi") -- is_hindi()'s OR check then only needed
    one of the two fields to say Hindi to pass, regardless of what the other
    field's own votes actually said. Voting on the pair jointly, as one unit,
    closes that: the two fields can no longer diverge, and a genuine split
    vote below min_agreement produces an explicit "uncertain" ("", "", ratio)
    result instead of silently picking a winner.

    `votes` is a list of per-window dicts (each with `language`/`language_code`
    keys, as returned by nemo_verify_window). Returns (language_code,
    language, agreement_ratio)."""
    def norm_lang(v):
        code = (v.get("language_code") or "").strip().lower().split("-")[0]
        name = (v.get("language") or "").strip().lower()
        return (code, name)

    lang_pairs = [norm_lang(v) for v in votes]
    if not lang_pairs:
        return "", "", 0.0
    (lang_code, lang_name), lang_win_count = Counter(lang_pairs).most_common(1)[0]
    lang_agreement = lang_win_count / len(lang_pairs)
    if lang_agreement < min_agreement:
        return "", "", lang_agreement  # uncertain -- fails is_hindi() downstream like any other reject
    return lang_code, lang_name, lang_agreement


def verify(wav: Path, duration: float, api_keys, sem: threading.Semaphore = None):
    """Vote across several windows so one bad excerpt doesn't decide the file.

    `sem`, if given, caps how many Nemotron calls are in flight at once across
    ALL concurrent light-stage workers combined — the free-tier NIM endpoint
    is shared, so this is the point to throttle total concurrency, not just
    per-video concurrency.

    `api_keys` is a list — each window call picks one at random, spreading
    load across multiple NIM accounts so no single key's rate limit becomes
    the bottleneck."""
    with tempfile.TemporaryDirectory() as td:
        wins = slice_windows(wav, duration, Path(td), TH.verify_windows, TH.verify_window_sec)
        votes = []
        for w in wins:
            try:
                api_key = random.choice(api_keys)
                if sem is not None:
                    with sem:
                        v = nemo_verify_window(w, api_key)
                else:
                    v = nemo_verify_window(w, api_key)
                if v:
                    votes.append(v)
            except Exception as e:
                log(f"  verify window failed: {e}", "WARN")
            time.sleep(1)

    # Fewer than this many windows actually returned a usable judgment (calls
    # can silently drop -- see the `except` above) -- a lone surviving vote,
    # or none, isn't enough to decide anything on, least of all language.
    if len(votes) < TH.min_lang_votes:
        return None

    def majority(key, default=None):
        vals = [v.get(key) for v in votes if v.get(key) is not None]
        if not vals:
            return default
        return max(set(map(str, vals)), key=lambda x: list(map(str, vals)).count(x))

    lang_code, lang_name, lang_agreement = aggregate_language_votes(votes, TH.min_lang_agreement)

    speaker_counts = [v.get("num_speakers") for v in votes if isinstance(v.get("num_speakers"), int)]
    return {
        "num_speakers": int(np.median(speaker_counts)) if speaker_counts else 0,
        "speaker_votes": speaker_counts,
        "language": lang_name,
        "language_code": lang_code,
        "language_agreement": round(lang_agreement, 2),
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


def is_hindi(code: str, name: str) -> bool:
    """Strict Hindi-only gate — is_indic() is intentionally broad (this
    pipeline can target any Indic language), but the current scraping goal
    is Hindi-only, and the broad gate was letting ~36% non-Hindi Indic
    content (Telugu, Tamil, Marathi, ...) into the accepted set.

    code and name now come from verify()'s single joint (code, name) vote
    (see norm_lang there), so they can no longer disagree with each other --
    `name` is checked only as a fallback for a missing/unrecognized code, not
    as an independent second vote that could pass on its own."""
    code = (code or "").lower().split("-")[0]
    if code:
        return code == "hi"
    return (name or "").strip().lower() == "hindi"


# ----------------------------------------------------------------------------
# stage 5 — diarization
# ----------------------------------------------------------------------------

_diar_cache = {}

@contextlib.contextmanager
def _allow_trusted_checkpoint_unpickling():
    """PyTorch 2.6 flipped torch.load's default from weights_only=False to True,
    which breaks pyannote's pytorch_lightning-saved checkpoints (they pickle
    torch.torch_version.TorchVersion, not on the safe-globals allowlist). A
    plain functools.partial(weights_only=False) preset gets overridden by
    callers (e.g. pytorch_lightning's pl_load) that pass weights_only=None
    explicitly, so force it instead of defaulting it."""
    import torch
    original_load = torch.load

    @functools.wraps(original_load)
    def _load_forcing_weights_only_false(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = _load_forcing_weights_only_false
    try:
        yield
    finally:
        torch.load = original_load


def _patch_hf_hub_use_auth_token():
    """pyannote.audio (core/pipeline.py, core/model.py) still calls
    huggingface_hub.hf_hub_download(..., use_auth_token=...) — removed from
    hf_hub_download's signature in huggingface_hub>=1.0 (renamed to `token`).
    Shim it here rather than pinning huggingface_hub down, since diffusers/
    transformers in this env require huggingface_hub>=1.0 themselves."""
    import huggingface_hub
    if getattr(huggingface_hub.hf_hub_download, "_use_auth_token_shim", False):
        return
    _orig = huggingface_hub.hf_hub_download

    @functools.wraps(_orig)
    def _shimmed(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs.setdefault("token", kwargs.pop("use_auth_token"))
        return _orig(*args, **kwargs)

    _shimmed._use_auth_token_shim = True
    huggingface_hub.hf_hub_download = _shimmed


def _load_pyannote(device_str: str):
    _patch_hf_hub_use_auth_token()
    from pyannote.audio import Pipeline
    import torch
    token = HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "No HF token. Paste yours into HF_TOKEN at the top of this script "
            "and accept the terms on the pyannote/speaker-diarization-3.1 model page."
        )
    with _allow_trusted_checkpoint_unpickling():
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token=token)
    if pipe is None:
        raise RuntimeError(
            "pyannote returned None — the token is valid but you have not accepted "
            "the terms on huggingface.co/pyannote/speaker-diarization-3.1 "
            "AND huggingface.co/pyannote/segmentation-3.0 (both are gated)."
        )
    if device_str != "cpu" and torch.cuda.is_available():
        try:
            pipe.to(torch.device(device_str))
        except Exception as e:
            log(f"  GPU init failed ({str(e)[:80]}), running diarization on CPU", "WARN")
    return pipe


def diarize_pyannote(wav: Path, num_speakers=2, device: str = "cuda"):
    """`device` keys the model cache — each GPU worker thread gets its own
    pipeline instance so concurrent heavy workers never share one Pipeline
    object (and never silently reuse another device's loaded weights).

    `num_speakers=None` runs pyannote's own auto speaker-count estimation
    instead of forcing a count — used when we want the true speaker
    inventory (e.g. to mine 2-speaker segments out of a 3+-speaker file)
    rather than a diarization that's already been coerced to match a
    hypothesis."""
    pipe = _get_or_load(_diar_cache, ("pyannote", device), lambda: _load_pyannote(device))
    kwargs = {} if num_speakers is None else {"num_speakers": num_speakers}
    ann = pipe(str(wav), **kwargs)
    segs = [{"start": float(t.start), "end": float(t.end), "speaker": str(spk)}
            for t, _, spk in ann.itertracks(yield_label=True)]
    return sorted(segs, key=lambda s: s["start"])


def _load_nemo_diar(device_str: str):
    from nemo.collections.asr.models import SortformerEncLabelModel
    m = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
    m.eval()
    if device_str != "cpu":
        m = m.to(device_str)
    return m


def diarize_nemo(wav: Path, num_speakers=2, device: str = "cuda"):
    """NVIDIA Sortformer. Verify the class path against your NeMo version."""
    m = _get_or_load(_diar_cache, ("nemo", device), lambda: _load_nemo_diar(device))
    preds = m.diarize(audio=[str(wav)], batch_size=1)
    segs = []
    for entry in preds[0]:
        # Sortformer emits "start end speaker_id" strings
        parts = str(entry).split()
        if len(parts) >= 3:
            segs.append({"start": float(parts[0]), "end": float(parts[1]),
                         "speaker": f"SPEAKER_{parts[2]}"})
    return sorted(segs, key=lambda s: s["start"])


def diarize(wav: Path, backend: str, num_speakers=2, device: str = "cuda"):
    fn = diarize_nemo if backend == "nemo" else diarize_pyannote
    segs = fn(wav, num_speakers, device=device)
    return [s for s in segs if s["end"] - s["start"] >= TH.min_segment_dur]


def _split_by_silence(segs, gap_seconds):
    """Group consecutive diarization segments; split wherever the gap
    between turns is >= gap_seconds (topic changes, ad breaks, etc. tend to
    land on a real pause, so this is a natural place to cut before even
    looking at speaker count)."""
    if not segs:
        return []
    groups = [[segs[0]]]
    for seg in segs[1:]:
        if seg["start"] - groups[-1][-1]["end"] >= gap_seconds:
            groups.append([seg])
        else:
            groups[-1].append(seg)
    return groups


def _two_speaker_runs(segs):
    """Within one silence-delimited group, find all maximal contiguous
    sub-sequences of turns involving exactly 2 distinct speakers. A 3rd
    speaker's segment closes the current run and starts a fresh one — this
    walks real segment boundaries directly (no grid/window sampling, no
    boundary snapping), so there's no way for a 3rd speaker's audio to leak
    into a run by construction."""
    if not segs:
        return []
    runs = []
    current = [segs[0]]
    speakers = {segs[0]["speaker"]}
    for seg in segs[1:]:
        if seg["speaker"] in speakers or len(speakers) < 2:
            current.append(seg)
            speakers.add(seg["speaker"])
        else:
            if len(speakers) == 2:
                runs.append(current)
            current = [seg]
            speakers = {seg["speaker"]}
    if len(speakers) == 2:
        runs.append(current)
    return runs


def _split_long_run(segs, max_dur, min_dur):
    """Cap an individual dialogue run at max_dur seconds so one very long
    clean 2-speaker stretch becomes several bounded clips instead of one
    giant one; drop a trailing remainder shorter than min_dur."""
    start0 = segs[0]["start"]
    if segs[-1]["end"] - start0 <= max_dur:
        return [segs]
    chunks, current, chunk_start = [], [], start0
    for seg in segs:
        current.append(seg)
        if seg["end"] - chunk_start >= max_dur:
            chunks.append(current)
            current, chunk_start = [], seg["end"]
    if current and current[-1]["end"] - chunk_start >= min_dur:
        chunks.append(current)
    return chunks


def mine_two_speaker_windows(segs, min_chunk_dur, max_chunk_dur=600.0, gap_seconds=5.0):
    """Given a full (possibly 3+-speaker) diarization timeline, return the
    contiguous (start, end, speaker_pair) windows where exactly 2 distinct
    speakers are active — e.g. the core interview in a file whose intro is
    a solo-host monologue and whose middle has a brief 3rd-guest cameo.

    Splits on silence gaps first, then extracts maximal 2-speaker turn
    sequences within each piece, then caps any run longer than
    `max_chunk_dur`. Windows shorter than `min_chunk_dur` are dropped.
    A genuinely 2-speaker file collapses to ~one run spanning the whole
    file, same as before."""
    if not segs:
        return []
    windows = []
    for group in _split_by_silence(segs, gap_seconds):
        for run in _two_speaker_runs(group):
            if run[-1]["end"] - run[0]["start"] < min_chunk_dur:
                continue
            for chunk in _split_long_run(run, max_chunk_dur, min_chunk_dur):
                t0, t1 = chunk[0]["start"], chunk[-1]["end"]
                if t1 - t0 >= min_chunk_dur:
                    windows.append((t0, t1, frozenset(s["speaker"] for s in chunk)))
    return windows


def slice_wav(src: Path, t0: float, t1: float, dest: Path):
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(src), "-ss", f"{max(0.0, t0):.3f}", "-to", f"{t1:.3f}",
         "-ar", str(SR), "-ac", "1", str(dest)])
    return dest


# ----------------------------------------------------------------------------
# separation — DialogueSidon
# ----------------------------------------------------------------------------

_sidon_cache = {}

def _load_sidon_for_device(script_path: str, device_str: str):
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
    if device_str == "cuda":
        # The exported DialogueSidon graph has an explicit "cuda:0" device
        # baked into some ops at trace time; a bare torch.device("cuda")
        # (no index) fails its internal device-equality check even though
        # it resolves to the same physical device.
        device_str = f"cuda:{torch.cuda.current_device()}"
    device = torch.device(device_str)

    log(f"  loading DialogueSidon on {device} (first call for this device)")
    t0 = time.time()
    models = mod.load_models(device)
    log(f"  DialogueSidon ready on {device} in {time.time() - t0:.0f}s")
    return {"mod": mod, "models": models, "device": device}


def load_sidon(script_path: str, device_str: str):
    """
    Import the author's inference script and load its exported components
    once per device. DialogueSidon SEPARATES a two-speaker mixture into two
    waveforms — it does not diarize and does not label speakers over time.

    Cached per `device_str` (not globally) so each GPU heavy-worker thread
    gets its own model instance — a single shared slot would silently make a
    second GPU's worker reuse the first GPU's loaded weights.
    """
    entry = _get_or_load(_sidon_cache, device_str,
                          lambda: _load_sidon_for_device(script_path, device_str))
    return entry["mod"], entry["models"], entry["device"]


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
    vad = _get_or_load(_vad_cache, "vad", _load_vad)

    x = signal
    if sr != SR:
        import scipy.signal as ss
        n = int(round(len(x) * SR / sr))
        x = ss.resample(x, n).astype("float32")

    with _vad_infer_lock:
        ts = vad["get_ts"](torch.from_numpy(x).cpu(), vad["model"], sampling_rate=SR)
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
    # Defense in depth: the language gate normally runs upstream (is_hindi()
    # in stage_light, before diarization/separation ever run), but this is
    # the single choke point every accept path funnels through before a
    # manifest row exists -- refuse here too, so a future out-of-band caller
    # (a backfill script, a --redo, a different driver) can't silently
    # repeat the language-contamination bug even if it forgets the upstream
    # check. Caller (_emit_chunk) already treats emit() exceptions as a
    # reject, so this needs no new handling there.
    if not is_hindi(verdict.get("language_code"), verdict.get("language")):
        raise ValueError(f"emit() refused non-Hindi verdict: "
                          f"code={verdict.get('language_code')!r} name={verdict.get('language')!r}")

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

def reject(store, vid, url, reason, stage, meta=None, raw: Path = None, keep_raw: bool = False):
    store.set(vid, url, "rejected", reason, stage, meta)
    log(f"  REJECT [{vid}] [{reason}] at {stage}")
    if raw is not None and not keep_raw:
        raw.unlink(missing_ok=True)


_ytdlp_backoff_lock = threading.Lock()
_ytdlp_backoff_until = [0.0]
_ytdlp_backoff_streak = [0]          # consecutive triggers with no clean gap between
_ytdlp_backoff_base_secs = 300        # 5min, doubles per streak, capped below
_ytdlp_backoff_max_secs = 2400        # 40min cap

_ytdlp_botwall_streak = [0]           # separate, longer escalation for the
_ytdlp_botwall_base_secs = 900        # harder "sign in to confirm you're not
_ytdlp_botwall_max_secs = 5400        # a bot" wall, which is IP/session-level
                                       # flagging and doesn't clear on the
                                       # same short timescale as the sliding
                                       # window rate limit.


def _note_ytdlp_error(detail: str):
    """YouTube's rate limit is a short sliding window in practice (clears
    within ~1-2 min of quiet), not the 'up to an hour' the error claims —
    but hundreds of threads plowing through it anyway just re-triggers it
    and wastes the whole candidate list. On the first sign of it, make
    every worker pause instead of continuing to hammer. Backoff duration
    escalates (5/10/20/40min) if it keeps re-triggering right as workers
    wake up, and resets once a pause has actually gone by without a hit.

    A harder "sign in to confirm you're not a bot" wall (IP/session-level
    flagging, not a sliding window) gets its own longer escalation
    (15/30/60/90min) since it doesn't self-clear on the same timescale."""
    low = detail.lower()
    is_rate_limit = "rate-limit" in low or "rate limited" in low
    is_bot_wall = "sign in to confirm" in low and "bot" in low
    if not is_rate_limit and not is_bot_wall:
        return
    with _ytdlp_backoff_lock:
        now = time.time()
        # A hit that lands well after the previous backoff should have
        # expired means the streak actually broke clean in between —
        # treat this as a fresh streak rather than continuing to escalate.
        if now > _ytdlp_backoff_until[0] + 60:
            _ytdlp_backoff_streak[0] = 0
            _ytdlp_botwall_streak[0] = 0
        if is_bot_wall:
            _ytdlp_botwall_streak[0] += 1
            secs = min(_ytdlp_botwall_base_secs * (2 ** (_ytdlp_botwall_streak[0] - 1)),
                       _ytdlp_botwall_max_secs)
            log(f"  YouTube bot-check wall detected — pausing all light-stage "
                f"workers for {secs // 60}min (streak {_ytdlp_botwall_streak[0]})", "WARN")
        else:
            _ytdlp_backoff_streak[0] += 1
            secs = min(_ytdlp_backoff_base_secs * (2 ** (_ytdlp_backoff_streak[0] - 1)),
                       _ytdlp_backoff_max_secs)
            log(f"  YouTube rate-limit detected — pausing all light-stage workers "
                f"for {secs // 60}min (streak {_ytdlp_backoff_streak[0]})", "WARN")
        _ytdlp_backoff_until[0] = now + secs


def _wait_out_ytdlp_backoff():
    remaining = _ytdlp_backoff_until[0] - time.time()
    if remaining > 0:
        # Stagger the wake-up across workers instead of everyone resuming
        # in the same instant, which would just re-trigger the limit again.
        time.sleep(remaining + random.uniform(0, 25))


def stage_light(url, root: Path, store: Store, args, verify_sem: threading.Semaphore = None):
    """probe -> download -> VAD prefilter -> Nemotron verify. Network/API-bound,
    safe to run with high thread concurrency. Returns a payload dict for the
    heavy (GPU) stage on pass, or None if rejected (or already decided)."""
    # Sustained scraping (hours, many threads) trips YouTube's rate limiter
    # even with modest --workers — pace every yt-dlp-touching call with a
    # jittered delay so the aggregate request rate stays polite regardless
    # of thread count.
    time.sleep(random.uniform(6.0, 14.0))
    _wait_out_ytdlp_backoff()
    auth_args = ytdlp_auth_args(args)
    try:
        meta = probe(url, auth_args)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or str(e)).strip().splitlines()[-1:] or [str(e)]
        log(f"  probe failed: {detail[0][:300]}", "WARN")
        _note_ytdlp_error(detail[0] if detail else "")
        store.set(url, url, "rejected", "probe_failed", "probe")
        return None
    except Exception as e:
        log(f"  probe failed: {str(e)[:200]}", "WARN")
        _note_ytdlp_error(str(e))
        store.set(url, url, "rejected", "probe_failed", "probe")
        return None
    vid = meta["id"]

    prior = store.get(vid)
    if prior and prior[0] in ("accepted", "rejected") and not args.redo:
        log(f"SKIP {vid} ({prior[0]}/{prior[1]})")
        return None

    log(f"START {vid} — {(meta['title'] or '')[:70]}")
    dur = meta["duration"]

    if dur < TH.min_duration:
        reject(store, vid, url, "too_short", "probe", meta)
        return None
    if dur > TH.max_duration:
        reject(store, vid, url, "too_long", "probe", meta)
        return None
    if args.cc_only and "creative commons" not in (meta["license"] or "").lower():
        reject(store, vid, url, "license", "probe", meta)
        return None

    raw = root / "raw" / f"{vid}.wav"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        try:
            log(f"  [{vid}] downloading audio")
            fetch_audio(url, raw, auth_args)
        except subprocess.CalledProcessError as e:
            detail = (e.stderr or e.stdout or str(e)).strip().splitlines()[-1:] or [str(e)]
            log(f"  download failed: {detail[0][:300]}", "WARN")
            _note_ytdlp_error(detail[0] if detail else "")
            reject(store, vid, url, "download_failed", "fetch", meta, raw, args.keep_raw)
            return None
        except Exception as e:
            log(f"  download failed: {str(e)[:300]}", "WARN")
            _note_ytdlp_error(str(e))
            reject(store, vid, url, "download_failed", "fetch", meta, raw, args.keep_raw)
            return None

    log(f"  [{vid}] VAD prefilter")
    sr_ratio = speech_ratio(raw)
    if sr_ratio < TH.min_speech_ratio:
        reject(store, vid, url, "low_speech", "prefilter",
               {**meta, "speech_ratio": sr_ratio}, raw, args.keep_raw)
        return None
    if sr_ratio > TH.max_speech_ratio:
        reject(store, vid, url, "high_speech", "prefilter",
               {**meta, "speech_ratio": sr_ratio}, raw, args.keep_raw)
        return None

    log(f"  [{vid}] verifying with Nemotron ({TH.verify_windows} windows)")
    v = verify(raw, dur, args.api_keys, sem=verify_sem)
    if not v:
        reject(store, vid, url, "verify_failed", "verify", meta, raw, args.keep_raw)
        return None
    log(f"    [{vid}] speakers={v['num_speakers']} lang={v['language']} "
        f"conv={v['is_conversational']} conf={v['confidence']}")

    if v["num_speakers"] < 2:
        # Whole-file gate only rules out confident monologues here — a file
        # whose sampled windows suggest 3+ speakers still gets a real,
        # unconstrained diarization in stage_heavy, which mines out any
        # contiguous exactly-2-speaker stretches instead of discarding the
        # whole file on a coarse whole-file vote.
        reject(store, vid, url, "not_two_speakers", "verify", {**meta, "verify": v}, raw, args.keep_raw)
        return None
    if not is_hindi(v["language_code"], v["language"]):
        reject(store, vid, url, "not_indic", "verify", {**meta, "verify": v}, raw, args.keep_raw)
        return None
    if not v["is_conversational"] or v["music_dominant"] or v["is_dubbed_or_voiceover"]:
        reject(store, vid, url, "not_conversational", "verify", {**meta, "verify": v}, raw, args.keep_raw)
        return None

    return {"vid": vid, "url": url, "meta": meta, "dur": dur, "raw": raw, "v": v}


def _emit_chunk(seg_vid, chunk_wav, sub_segs, stats, meta, v, root, args, device, url, store):
    """Shared tail of stage_heavy for one candidate window (the whole file,
    or one mined 2-speaker chunk of it): separate -> emit -> record."""
    channels, chan_sr = None, None
    if args.separator == "sidon":
        log(f"  [{seg_vid}] separating with DialogueSidon ({device})")
        try:
            channels, chan_sr = separate_sidon(chunk_wav, args.sidon_script, device, args.sidon_steps)
        except Exception as e:
            log(f"    {str(e)[:250]}", "WARN")
            reject(store, seg_vid, url, "separate_failed", "separate", meta)
            return False

    try:
        outdir = emit(seg_vid, chunk_wav, sub_segs, stats, meta, v, root,
                      channels=channels, chan_sr=chan_sr,
                      separator=args.separator if channels is not None else "mask")
    except Exception as e:
        log(f"    {str(e)[:200]}", "WARN")
        reject(store, seg_vid, url, "emit_failed", "emit", meta)
        return False

    store.set(seg_vid, url, "accepted", "", "emit", {**meta, "verify": v, "stats": stats})
    log(f"  ACCEPT [{seg_vid}] -> {outdir}")
    return True


def stage_heavy(payload, device: str, root: Path, store: Store, args):
    """diarize -> qualify -> (maybe) separate -> emit. GPU-bound; runs on a
    thread pinned to one device (or CPU) for its whole life.

    For the pyannote/nemo backends this diarizes WITHOUT forcing a 2-speaker
    hypothesis. A file that's genuinely 2-speaker collapses to one window
    covering ~the whole thing (unchanged from before). A file with a 3rd
    speaker (a brief guest, an intro monologue before a co-host joins, ad
    reads) gets mined for the contiguous stretches that ARE exactly 2
    speakers — each qualifying stretch is emitted as its own accepted clip
    (`{vid}_seg0`, `{vid}_seg1`, ...) instead of the whole file being thrown
    away for not being 2-speaker end to end."""
    vid, url, meta = payload["vid"], payload["url"], payload["meta"]
    dur, raw, v = payload["dur"], payload["raw"], payload["v"]

    if args.diarizer == "sidon-vad":
        # Separate first, then derive the timeline from each clean channel.
        # Sidon always yields a fixed 2-channel separation, so there's no
        # "true speaker count" to mine here — unchanged from before.
        log(f"  [{vid}] separating with DialogueSidon ({device})")
        try:
            channels, chan_sr = separate_sidon(raw, args.sidon_script, device, args.sidon_steps)
        except Exception as e:
            log(f"    {str(e)[:250]}", "WARN")
            reject(store, vid, url, "separate_failed", "separate", meta, raw, args.keep_raw)
            return
        segs_full = diarize_from_channels(channels, chan_sr)
        windows = [(0.0, dur, None)]
    else:
        log(f"  [{vid}] diarizing ({args.diarizer}, {device})")
        try:
            segs_full = diarize(raw, args.diarizer, num_speakers=None, device=device)
        except Exception as e:
            log(f"    {str(e)[:200]}", "WARN")
            reject(store, vid, url, "diarize_failed", "diarize", meta, raw, args.keep_raw)
            return

        n_speakers_full = len({s["speaker"] for s in segs_full})
        if n_speakers_full == 2:
            windows = [(0.0, dur, None)]
        elif n_speakers_full > 2:
            windows = mine_two_speaker_windows(segs_full, min_chunk_dur=TH.min_duration)
            log(f"    [{vid}] {n_speakers_full} true speakers detected -> "
                f"{len(windows)} candidate 2-speaker window(s)")
        else:
            windows = []

        if not windows:
            reject(store, vid, url, "diar_speaker_count", "diarize",
                   {**meta, "verify": v, "n_speakers_true": n_speakers_full}, raw, args.keep_raw)
            return

    single_whole_file = len(windows) == 1 and windows[0][0] == 0.0 and windows[0][1] == dur
    n_accepted = 0

    for idx, (t0, t1, pair) in enumerate(windows):
        seg_vid = vid if single_whole_file else f"{vid}_seg{idx}"
        sub_dur = t1 - t0
        # Snapping window edges to the nearest real segment boundary (in
        # mine_two_speaker_windows) is boundary-agnostic — it can snap onto
        # a 3rd speaker's segment edge that happens to sit closest to the
        # cut point. Filtering to `pair`'s own speakers (not just the time
        # range) guards against a stray 3rd-speaker sliver leaking into an
        # otherwise-clean 2-speaker chunk regardless of snap precision.
        if single_whole_file:
            sub_segs = list(segs_full)
        else:
            sub_segs = [{**s, "start": max(0.0, s["start"] - t0), "end": min(t1, s["end"]) - t0}
                        for s in segs_full
                        if s["end"] > t0 and s["start"] < t1 and (pair is None or s["speaker"] in pair)]

        stats = conversation_stats(sub_segs, sub_dur)
        log(f"    [{seg_vid}] turns={stats.get('n_turns')} tpm={stats.get('turns_per_min')} "
            f"dom={stats.get('dominance')} ovl={stats.get('overlap_ratio')}")

        bad = qualify(stats)
        if bad:
            if single_whole_file:
                reject(store, vid, url, bad, "qualify", {**meta, "verify": v, "stats": stats}, raw, args.keep_raw)
                return
            reject(store, seg_vid, url, bad, "qualify", {**meta, "verify": v, "stats": stats})
            continue

        seg_meta = meta if single_whole_file else {**meta, "duration": sub_dur, "segment_of": vid,
                                                    "segment_start": t0, "segment_end": t1}
        chunk_wav = raw if single_whole_file else slice_wav(raw, t0, t1, raw.parent / f"{seg_vid}.wav")
        ok = _emit_chunk(seg_vid, chunk_wav, sub_segs, stats, seg_meta, v, root, args, device, url, store)
        if chunk_wav is not raw:
            chunk_wav.unlink(missing_ok=True)
        if ok:
            n_accepted += 1

    if not single_whole_file:
        # Record a status for the ORIGINAL id too, purely so a future rerun's
        # stage_light SKIP check (which looks up the original vid) doesn't
        # reprocess this episode from scratch every time.
        if n_accepted:
            store.set(vid, url, "accepted", f"{n_accepted}_segment(s)", "emit", {**meta, "verify": v})
        else:
            store.set(vid, url, "rejected", "no_qualifying_2spk_segment", "qualify", {**meta, "verify": v})

    if not args.keep_raw:
        raw.unlink(missing_ok=True)


def detect_idle_gpus(threshold_mb=1024):
    """GPU indices currently using less than threshold_mb — a shared machine
    may have other jobs on some devices; only touch the ones sitting idle."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    idle = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        if (total - free) / 1e6 < threshold_mb:
            idle.append(i)
    return idle


def resolve_devices(gpus_arg: str):
    """--gpus 'auto' | 'cpu' | '0,1' -> list of device strings, one per heavy
    worker group. 'auto' only picks GPUs with ~nothing already on them."""
    if gpus_arg == "cpu":
        return ["cpu"]
    if gpus_arg == "auto":
        idle = detect_idle_gpus()
        if not idle:
            log("no idle GPU found (auto-detect) — heavy stage falls back to CPU", "WARN")
            return ["cpu"]
        log(f"auto-detected idle GPU(s): {idle}")
        return [f"cuda:{i}" for i in idle]
    ids = [x.strip() for x in gpus_arg.split(",") if x.strip()]
    if not ids:
        sys.exit("ERROR: --gpus given but empty")
    return [f"cuda:{i}" for i in ids]


def cmd_run(args):
    need("yt-dlp"); need("ffmpeg"); need("ffprobe")
    args.api_keys = [k.strip() for k in (args.api_key or "").split(",") if k.strip()]
    if not args.api_keys:
        sys.exit("ERROR: set NVIDIA_API_KEY at the top of this script or pass --api-key "
                  "(comma-separate multiple keys to spread load across them)")

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

    devices = resolve_devices(args.gpus)
    heavy_per_group = max(1, args.heavy_workers_per_gpu)
    n_heavy = len(devices) * heavy_per_group
    queue_size = args.queue_size if args.queue_size > 0 else 2 * n_heavy
    heavy_queue = queue.Queue(maxsize=queue_size)
    verify_sem = threading.Semaphore(max(1, args.verify_concurrency))
    log(f"light workers: {args.workers}   heavy workers: {n_heavy} on {devices}   "
        f"verify concurrency: {args.verify_concurrency}   queue size: {queue_size}")

    def light_job(url):
        try:
            payload = stage_light(url, root, store, args, verify_sem)
        except Exception as e:
            log(f"unhandled in light stage ({url}): {str(e)[:300]}", "ERROR")
            return
        if payload:
            heavy_queue.put(payload)  # blocks if full -> backpressure on light stage

    def heavy_worker(device):
        while True:
            item = heavy_queue.get()
            if item is None:
                heavy_queue.task_done()
                return
            try:
                stage_heavy(item, device, root, store, args)
            except Exception as e:
                log(f"unhandled in heavy stage [{item['vid']}]: {str(e)[:300]}", "ERROR")
            heavy_queue.task_done()

    heavy_threads = []
    for device in devices:
        for _ in range(heavy_per_group):
            t = threading.Thread(target=heavy_worker, args=(device,), daemon=True)
            t.start()
            heavy_threads.append(t)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(light_job, u) for u in urls]
        done = 0
        try:
            for fut in as_completed(futures):
                fut.result()  # light_job never raises; exceptions are caught inside
                done += 1
                if done % 10 == 0 or done == len(futures):
                    log(f"--- light stage: {done}/{len(futures)} done ---")
        except KeyboardInterrupt:
            log(f"interrupted — cancelling not-yet-started light-stage work "
                f"({done}/{len(futures)} done, rest will drain)", "WARN")
            for f in futures:
                f.cancel()

    # signal heavy workers to stop once the queue drains, then wait for them
    for _ in heavy_threads:
        heavy_queue.put(None)
    for t in heavy_threads:
        t.join()

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


# ----------------------------------------------------------------------------
# stage 8 — transcription (word-level timestamps)
# ----------------------------------------------------------------------------

_indic_asr_cache = {}
_alignment_workers = {}
_alignment_workers_lock = threading.Lock()

# Very long / complex clips have been observed (200-clip validation run,
# 2026-09-24) to crash the CTC forced-alignment step with a native SIGABRT/
# SIGSEGV -- inside onnxruntime/torch, under memory pressure, before any
# Python try/except can catch it. Earlier attempt used a fixed duration cutoff
# to dodge this, but that would have silently dropped word-level timestamps
# for ~314h (23%) of the corpus's long-tail clips. Isolating the alignment
# call in a dedicated subprocess (below) removes the need to guess a safe
# duration at all: every clip gets a real alignment attempt, and only the
# ones that actually crash or hang fall back to approximate timestamps --
# and only that one clip is affected, since the crash can't take down the
# worker thread or any other in-flight clip.
ALIGN_TIMEOUT_SEC = 1800  # 30 min wall-clock ceiling per clip's alignment call


def _onnx_alignment_loop(device_str, req_q, res_q):
    """Legacy backend (ALIGN_BACKEND=onnx): CPU ONNX aligner + uroman, chunked
    by proportional transcript splitting. Kept for reference/fallback only --
    the proportional split misplaces words on clips whose speech is unevenly
    spread over time (e.g. separated speaker channels with long silences),
    so IndicWav2Vec (below) is the default.

    Runs in an isolated child process. Loads the aligner once, then serves
    align requests until killed. A native crash here only takes down this
    process -- the parent (AlignmentWorker.align) detects it and respawns."""
    # AlignmentSingleton's onnxruntime session has no device kwarg -- it
    # picks up whichever GPU CUDA_VISIBLE_DEVICES exposes, so pin it here
    # (before onnxruntime/ctc_forced_aligner import) to match the ASR
    # model's device for this same worker slot, not always GPU 0.
    if device_str.startswith("cuda"):
        idx = device_str.split(":")[-1] if ":" in device_str else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = idx

    # onnxruntime here is CPU-only (no CUDAExecutionProvider in this venv --
    # see requirements.txt), and left to its own defaults it sizes its
    # intra-op thread pool off the NODE's total core count (observed: ~110
    # threads, matching this cluster's 112-core sockets), not the cgroup CPU
    # allocation SLURM actually gave this job. With several concurrent
    # alignment workers each doing this, the node's cores get oversubscribed
    # several times over (observed load average 287 on 224 cores running
    # 5 workers), each individual alignment call slows down from contention,
    # and it also produces the (harmless but noisy) pthread_setaffinity_np
    # warnings from threads trying to pin to cores outside the cgroup's mask.
    # AlignmentSingleton doesn't expose SessionOptions to control this, so
    # patch onnxruntime.InferenceSession to inject a capped thread count
    # before it's ever called -- scoped to this subprocess only.
    import onnxruntime
    _orig_inference_session = onnxruntime.InferenceSession

    def _capped_inference_session(*args, **kwargs):
        if "sess_options" not in kwargs:
            so = onnxruntime.SessionOptions()
            so.intra_op_num_threads = 8
            so.inter_op_num_threads = 1
            kwargs["sess_options"] = so
        return _orig_inference_session(*args, **kwargs)

    onnxruntime.InferenceSession = _capped_inference_session

    import uroman as uroman_lib
    from ctc_forced_aligner import (
        AlignmentSingleton, generate_emissions, get_alignments, get_spans,
        postprocess_results, load_audio,
    )
    ur = uroman_lib.Uroman()
    aligner = AlignmentSingleton()

    def real_preprocess(text, language="hin"):
        tokens_starred, text_starred = [], []
        for w in (w for w in text.split() if w.strip()):
            roman = re.sub(r"[^a-z]", "", ur.romanize_string(w, lcode=language).strip().lower())
            if not roman:
                continue
            tokens_starred += ["<star>", " ".join(list(roman))]
            text_starred += ["<star>", w]
        return tokens_starred, text_starred

    # Root cause of the native SIGABRT/SIGSEGV crashes on long clips: the
    # forced-alignment DP (get_alignments) runs over the FULL emission x
    # token sequence in one shot, so its memory/compute footprint grows with
    # the *entire* clip's length -- generate_emissions' own internal windowing
    # (window_length=30s) doesn't help here, since it only chunks the
    # emission generation, not the alignment DP consuming its output. A
    # 200-clip validation (2026-09-24) and a live production run both showed
    # crash rate rising sharply with duration (all 5 originally-found crash
    # cases were 26-37 min clips; a later window averaging ~28 min/clip hit a
    # ~47% crash rate). Fix: split any clip into <=ALIGN_CHUNK_SEC pieces
    # (audio by time, transcript proportionally by word position, assuming
    # roughly uniform speech rate within a clip) and align each piece
    # independently -- no single alignment DP call ever sees more than a few
    # minutes of audio, regardless of the original clip's length. Verified
    # sizes (<=10 min) were reliably crash-free throughout all testing so
    # far.
    ALIGN_CHUNK_SEC = 480  # 8 min

    def _align_chunk(audio_chunk, text_chunk):
        tokens_starred, text_starred = real_preprocess(text_chunk)
        if not tokens_starred:
            return []
        emissions, stride = generate_emissions(aligner.alignment_model, audio_chunk)
        segments, scores, blank_label = get_alignments(
            emissions, tokens_starred, aligner.alignment_tokenizer)
        spans = get_spans(tokens_starred, segments, blank_label)
        return postprocess_results(text_starred, spans, stride, scores)

    def chunked_align(wav_path, text):
        full_audio = load_audio(wav_path, ret_type="np")
        total_samples = len(full_audio)
        sr = 16000  # ctc_forced_aligner.SAMPLING_FREQ
        total_dur = total_samples / sr
        words = [w for w in text.split() if w.strip()]
        if not words or total_dur <= 0:
            return []

        n_chunks = max(1, math.ceil(total_dur / ALIGN_CHUNK_SEC))
        chunk_dur = total_dur / n_chunks
        results = []
        for i in range(n_chunks):
            t0 = i * chunk_dur
            t1 = total_dur if i == n_chunks - 1 else (i + 1) * chunk_dur
            s0, s1 = int(t0 * sr), int(t1 * sr)
            audio_chunk = full_audio[s0:s1]
            w0 = int(len(words) * t0 / total_dur)
            w1 = len(words) if i == n_chunks - 1 else int(len(words) * t1 / total_dur)
            chunk_words = words[w0:w1]
            if not chunk_words or len(audio_chunk) == 0:
                continue
            word_ts = _align_chunk(audio_chunk, " ".join(chunk_words))
            for w in word_ts:
                results.append({
                    "word": w["text"],
                    "start": round(float(w["start"]) + t0, 3),
                    "end": round(float(w["end"]) + t0, 3),
                })
        return results

    while True:
        wav_path, text = req_q.get()
        try:
            word_ts = chunked_align(wav_path, text)
            res_q.put(("ok", word_ts))
        except Exception as e:
            res_q.put(("error", f"{type(e).__name__}: {str(e)[:300]}"))


def _interpolate_unaligned(all_words, aligned_by_idx, total_dur):
    """Full word list in transcript order. Words the aligner could not place
    (no characters in its vocabulary: other scripts, digits, punctuation-only)
    are kept, not dropped, with timestamps spread evenly across the gap between
    their aligned neighbours and flagged 'approx_timestamps': True."""
    n = len(all_words)
    res = [None] * n
    for i, (s, e) in aligned_by_idx.items():
        res[i] = {"word": all_words[i], "start": s, "end": e}
    i = 0
    while i < n:
        if res[i] is not None:
            i += 1
            continue
        j = i
        while j < n and res[j] is None:
            j += 1
        t0 = res[i - 1]["end"] if i > 0 else 0.0
        t1 = max(res[j]["start"] if j < n else total_dur, t0)
        step = (t1 - t0) / (j - i)
        for k in range(i, j):
            res[k] = {"word": all_words[k], "start": round(t0 + (k - i) * step, 3),
                      "end": round(t0 + (k - i + 1) * step, 3), "approx_timestamps": True}
        i = j
    return res


# Chosen by measurement on 80 delivered speaker channels against audio energy:
# speech not covered by any word 37.7% -> 14.4%, median word 0.15 -> 0.23 s,
# words <40 ms 3.6% -> 0.1%, lowest combined over-extension/uncovered cost.
# Longer settings mostly add silence.
END_GAP_FILL_SEC = 0.25   # gap to the next word up to this: close it entirely
END_TAIL_SEC = 0.10       # larger gap (a pause): extend only this far past the CTC end


def _extend_word_ends(words, total_dur=None, gap_fill=END_GAP_FILL_SEC, tail=END_TAIL_SEC):
    """CTC forced alignment marks where each character's sound peaks, so a
    word's span covers only first-to-last character peak: its END is
    systematically early (median word ~0.15 s; ~4% under 40 ms) while its START
    is reliable. Extend each word's end toward the next word's start: fully if
    the gap is <= gap_fill, otherwise by `tail` (the decay of the last sound)
    into the pause. Never crosses the next word, so words never overlap.
    Interpolated words (approx_timestamps) already span their gaps and are
    left alone. `words` is one speaker channel in time order; returns a new
    list, input untouched."""
    out = [dict(w) for w in words]
    n = len(out)
    for i, w in enumerate(out):
        if w.get("approx_timestamps"):
            continue
        nxt = out[i + 1]["start"] if i + 1 < n else (total_dur if total_dur is not None else w["end"] + tail)
        gap = nxt - w["end"]
        if gap <= 0:
            continue
        w["end"] = round(nxt if gap <= gap_fill else min(w["end"] + tail, nxt), 3)
    return out


IWV_MODEL_ID = "ai4bharat/indicwav2vec-hindi"
IWV_WINDOW_SEC = 300     # model forward pass runs on windows of this length...
IWV_CONTEXT_SEC = 10     # ...with this much extra audio each side, then discarded
IWV_GPU_TRELLIS_CELLS = 70e9  # T*(2L+1) above this -> run forced_align on CPU


def _iwv_alignment_loop(device_str, req_q, res_q):
    """Default backend: AI4Bharat's indicwav2vec-hindi (Hindi CTC model with a
    native Devanagari vocabulary -- no romanization) + torchaudio forced_align,
    on GPU, in an isolated child process.

    Long clips: the model's CNN front-end and attention memory grow linearly
    with audio length (a 3.4h clip tried to allocate 75 GiB in one tensor), so
    the *emission* is computed in overlapping windows and concatenated, but
    the alignment itself is ONE global forced_align over the whole clip's
    emission and full transcript -- no per-chunk transcript splitting, so a
    word can never be assigned to the wrong time window (the failure mode of
    proportional splitting when speech is unevenly spread over the clip).
    If the alignment trellis (T x (2L+1)) would not fit on the GPU it falls
    back to CPU for that clip."""
    import unicodedata
    import librosa
    import torch
    import torchaudio.functional as F
    from transformers import AutoModelForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

    # Use the explicit device index rather than CUDA_VISIBLE_DEVICES: under
    # multiprocessing 'spawn' this module's top-level code (and torch's CUDA
    # init) has already run by the time we get here, so changing the env var
    # now would not re-pin the process.
    dev = torch.device(device_str if device_str.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        # torchaudio's forced_align is a custom CUDA kernel that launches on the
        # *current* device, not the device its tensors live on; with several GPUs
        # visible and dev != cuda:0 it hits "illegal memory access" unless the
        # current device is set explicitly.
        torch.cuda.set_device(dev)
    tok = Wav2Vec2CTCTokenizer.from_pretrained(IWV_MODEL_ID)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(IWV_MODEL_ID)
    model = AutoModelForCTC.from_pretrained(IWV_MODEL_ID).to(dev).eval()
    vocab = tok.get_vocab()
    blank_id = vocab.get(tok.pad_token, 0)
    delim_id = vocab.get("|")
    use_delim = os.environ.get("IWV_USE_DELIM", "1") == "1" and delim_id is not None
    sr, hop = 16000, 320  # 20 ms frames; window/context are multiples of hop
    win, ctx = IWV_WINDOW_SEC * sr, IWV_CONTEXT_SEC * sr

    def emission_for(wav_path):
        wave = librosa.load(wav_path, sr=sr, mono=True)[0].astype("float32")
        n = len(wave)
        inp = fe(wave, sampling_rate=sr, return_tensors="pt").input_values[0]  # normalized over the whole clip
        outs, s = [], 0
        while s < n:
            e = min(s + win, n)
            a, b = max(0, s - ctx), min(n, e + ctx)
            with torch.inference_mode():
                lg = model(inp[a:b].unsqueeze(0).to(dev)).logits[0].float()
            lg = torch.log_softmax(lg, dim=-1)
            f0 = (s - a) // hop
            n_core = lg.size(0) - f0 if e >= n else (e - s) // hop
            outs.append(lg[f0:f0 + n_core])
            s = e
        return torch.cat(outs, dim=0), n

    def align_words(wav_path, text):
        emission, n_samples = emission_for(wav_path)
        T = emission.size(0)
        all_words = text.split()
        words, ids_per_word, kept_idx = [], [], []
        for wi, w in enumerate(all_words):
            ids = [vocab[c] for c in unicodedata.normalize("NFC", w)
                   if c in vocab and vocab[c] != blank_id and c != "|"]
            if ids:
                words.append(w)
                ids_per_word.append(ids)
                kept_idx.append(wi)
        if not words:
            # nothing alignable (e.g. transcript entirely in another script)
            return _interpolate_unaligned(all_words, {}, n_samples / sr)
        targets = []
        for i, ids in enumerate(ids_per_word):
            if use_delim and i > 0:
                targets.append(delim_id)
            targets.extend(ids)
        L = len(targets)
        tgt = torch.tensor([targets], dtype=torch.int32)
        aligned = None
        if emission.is_cuda and T * (2 * L + 1) <= IWV_GPU_TRELLIS_CELLS:
            try:
                aligned, scores = F.forced_align(emission.unsqueeze(0), tgt.to(emission.device), blank=blank_id)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                aligned = None
        if aligned is None:
            aligned, scores = F.forced_align(emission.unsqueeze(0).cpu(), tgt, blank=blank_id)
        spans = F.merge_tokens(aligned[0], scores[0])
        if len(spans) != L:
            raise RuntimeError(f"alignment produced {len(spans)} token spans for {L} targets")
        ratio = n_samples / T
        aligned_by_idx, k = {}, 0
        for i, ids in enumerate(ids_per_word):
            if use_delim and i > 0:
                k += 1
            sp = spans[k:k + len(ids)]
            k += len(ids)
            aligned_by_idx[kept_idx[i]] = (round(int(ratio * sp[0].start) / sr, 3),
                                           round(int(ratio * sp[-1].end) / sr, 3))
        return _extend_word_ends(_interpolate_unaligned(all_words, aligned_by_idx, n_samples / sr),
                                 total_dur=n_samples / sr)

    while True:
        wav_path, text = req_q.get()
        try:
            res_q.put(("ok", align_words(wav_path, text)))
        except Exception as e:
            res_q.put(("error", f"{type(e).__name__}: {str(e)[:300]}"))
        finally:
            if dev.type == "cuda":
                torch.cuda.empty_cache()


def _alignment_worker_main(device_str, req_q, res_q):
    if os.environ.get("ALIGN_BACKEND", "iwv") == "onnx":
        _onnx_alignment_loop(device_str, req_q, res_q)
    else:
        _iwv_alignment_loop(device_str, req_q, res_q)


class AlignmentWorker:
    """One long-lived, isolated subprocess per device. Models are loaded once
    (expensive) and reused across clips; a crash or hang on one clip respawns
    the process for the next one instead of poisoning the caller."""

    # 'spawn', not the Linux default 'fork': by the time this worker is
    # created, the parent already has CUDA initialized (ASR/diarization
    # models loaded) — forking a CUDA-initialized process is a well-known
    # hang/crash source, since the child inherits a CUDA context it can't
    # actually use. spawn re-imports cleanly instead.
    _ctx = mp.get_context("spawn")

    def __init__(self, device_str):
        self.device_str = device_str
        self.req_q = None
        self.res_q = None
        self.proc = None
        self._lock = threading.Lock()
        self._start()

    def _start(self):
        self.req_q = self._ctx.Queue()
        self.res_q = self._ctx.Queue()
        self.proc = self._ctx.Process(
            target=_alignment_worker_main,
            args=(self.device_str, self.req_q, self.res_q),
            daemon=True,
        )
        self.proc.start()

    def align(self, wav_path: str, text: str):
        """Returns (words_or_None, error_or_None). Serialized per worker --
        the subprocess handles one clip at a time by design (matches how the
        old code used one model instance per device)."""
        with self._lock:
            if not self.proc.is_alive():
                self._start()
            self.req_q.put((wav_path, text))
            deadline = time.time() + ALIGN_TIMEOUT_SEC
            while time.time() < deadline:
                if not self.proc.is_alive():
                    self._start()
                    return None, "WORKER_CRASHED (native abort/segfault)"
                try:
                    status, payload = self.res_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                if status == "ok":
                    return payload, None
                return None, payload
            self.proc.terminate()
            self.proc.join(timeout=5)
            self._start()
            return None, f"TIMEOUT (>{ALIGN_TIMEOUT_SEC}s)"


def get_alignment_worker(device_str: str) -> AlignmentWorker:
    with _alignment_workers_lock:
        if device_str not in _alignment_workers:
            _alignment_workers[device_str] = AlignmentWorker(device_str)
        return _alignment_workers[device_str]


def _load_indic_asr(device_str: str):
    from huggingface_hub import snapshot_download
    log(f"  loading ASR model {ASR_MODEL_ID} on {device_str} (first call for this key)")
    t0 = time.time()
    model_dir = snapshot_download(ASR_MODEL_ID)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from indic_transcribe import IndicTranscribe
    model = IndicTranscribe.from_pretrained(model_dir, device=device_str)
    log(f"  ASR model {ASR_MODEL_ID} ready on {device_str} in {time.time() - t0:.0f}s")
    return model


def get_asr_pipeline(device_str: str):
    """Cached per device — same lazy-singleton pattern as the diarization/
    separation model caches, so concurrent GPU workers each get their own
    model instance without racing on first load."""
    return _get_or_load(_indic_asr_cache, device_str, lambda: _load_indic_asr(device_str))


# ---- legacy Whisper fallback -----------------------------------------------
# Only used for the small minority of clips (~2.5% in the 200-clip validation
# run, 2026-09-24) whose new-pipeline forced alignment crashes/times out even
# with subprocess isolation. Gives real word-level timestamps for that one
# clip instead of linear interpolation -- inherits the old pipeline's known
# failure modes (decoder-loop repetition, stuck timestamps) but only for this
# rare fallback path, not the default one.
_whisper_cache = {}
CT2_MODEL_DIR = Path(__file__).resolve().parent / ".ct2_models"
WHISPER_LANG_MODEL = {
    "hi": "vasista22/whisper-hindi-large-v2",
    "ta": "vasista22/whisper-tamil-large-v2",
    "te": "vasista22/whisper-telugu-large-v2",
}
WHISPER_DEFAULT_MODEL = "openai/whisper-large-v3"


def _ct2_model_path(model_id: str) -> str:
    if model_id == "openai/whisper-large-v3":
        return "large-v3"
    out_dir = CT2_MODEL_DIR / model_id.replace("/", "__")
    if not (out_dir / "model.bin").exists():
        log(f"  converting {model_id} to CTranslate2 format (one-time, cached at {out_dir})")
        out_dir.mkdir(parents=True, exist_ok=True)
        converter = Path(sys.executable).parent / "ct2-transformers-converter"
        run([str(converter) if converter.exists() else "ct2-transformers-converter",
             "--model", model_id,
             "--output_dir", str(out_dir), "--quantization", "float16", "--force"])
    return str(out_dir)


def _load_whisper_pipeline(model_id: str, device_str: str):
    from faster_whisper import WhisperModel
    log(f"  loading fallback ASR model {model_id} on {device_str}")
    ct2_path = _ct2_model_path(model_id)
    if device_str.startswith("cuda"):
        device, device_index, compute_type = "cuda", int(device_str.split(":")[-1]), "float16"
    else:
        device, device_index, compute_type = "cpu", 0, "int8"
    return WhisperModel(ct2_path, device=device, device_index=device_index, compute_type=compute_type)


def get_whisper_pipeline(lang_code: str, device_str: str):
    model_id = WHISPER_LANG_MODEL.get(lang_code, WHISPER_DEFAULT_MODEL)
    return model_id, _get_or_load(_whisper_cache, (model_id, device_str),
                                   lambda: _load_whisper_pipeline(model_id, device_str))


def _whisper_fallback_transcribe(wav_path: Path, lang_code: str, device_str: str):
    model_id, model = get_whisper_pipeline(lang_code, device_str)
    segments, _info = model.transcribe(
        str(wav_path), word_timestamps=True, beam_size=1, language=lang_code or None)
    words = [
        {"word": w.word.strip(), "start": round(float(w.start), 3), "end": round(float(w.end), 3)}
        for seg in segments for w in (seg.words or [])
    ]
    return model_id, words


def _script_stats(text: str) -> dict:
    """Share of Devanagari among the letters of an ASR transcript. A clip
    labeled Hindi whose transcript is mostly Gujarati/Bengali/... script is
    probably not Hindi audio (language-verification false positive)."""
    dev = other = 0
    for ch in text:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            dev += 1
        elif 0x0980 <= o <= 0x0DFF or 0x0600 <= o <= 0x06FF or ch.isascii() and ch.isalpha():
            other += 1
    return {"n_letters": dev + other, "devanagari_frac": round(dev / (dev + other), 4) if dev + other else None}


def transcribe_file(wav_path: Path, lang_code: str, device_str: str, stats_out: dict = None):
    """Returns (model_id, [{'word', 'start', 'end'}, ...]).

    If `stats_out` is given it is filled with per-file script statistics
    (see _script_stats) and word counts.

    ASR: bodhan-ai/indic-transcribe-core (native Devanagari, no chunking
    needed on our side -- long_form.transcribe_long already handles long
    audio internally). Word timestamps: CTC forced alignment
    (ai4bharat/indicwav2vec-hindi by default, see _iwv_alignment_loop), run
    in an isolated subprocess (see AlignmentWorker above) so a crash there
    costs only this clip's timestamps, never the whole transcription run.

    On crash/timeout, falls back to the legacy faster-whisper pipeline for
    real word-level timestamps on just this clip (rather than the new
    pipeline's own text with fake evenly-spaced timing). If even that fails,
    last resort is evenly-spaced estimated timestamps, flagged via the
    'approx_timestamps' key so any downstream consumer that needs real word
    timing can filter these out. The returned model_id reflects whichever
    path actually produced the words, so fallback clips stay traceable."""
    asr = get_asr_pipeline(device_str)  # side effect: puts long_form.py's dir on sys.path
    from long_form import transcribe_long
    text = transcribe_long(asr, str(wav_path), lang=lang_code or "hi", mode="native")
    if stats_out is not None:
        stats_out.update(_script_stats(text))
        stats_out["n_asr_words"] = len(text.split())

    words_out, err = get_alignment_worker(device_str).align(str(wav_path), text)
    if words_out is not None:
        if stats_out is not None:
            stats_out["n_approx_words"] = sum(1 for w in words_out if w.get("approx_timestamps"))
        return ASR_MODEL_ID, words_out

    log(f"  [{wav_path.name}] alignment failed ({err}) -- falling back to legacy Whisper ASR+timestamps for this clip", "WARN")
    try:
        return _whisper_fallback_transcribe(wav_path, lang_code, device_str)
    except Exception as e:
        log(f"  [{wav_path.name}] Whisper fallback ALSO failed ({str(e)[:200]}) -- using evenly-spaced estimated timestamps as last resort", "WARN")
        toks = [w for w in text.split() if w.strip()]
        dur = sf.info(str(wav_path)).duration if toks else 0.0
        step = dur / len(toks) if toks else 0.0
        words_out = [
            {"word": w, "start": round(i * step, 3), "end": round((i + 1) * step, 3), "approx_timestamps": True}
            for i, w in enumerate(toks)
        ]
        return ASR_MODEL_ID, words_out


def transcribe_one(vid_dir: Path, device_str: str, overwrite: bool = False):
    """Transcribes both speaker channels, then merges them into a single
    time-ordered transcript matching the actual conversation timeline (not
    two isolated per-speaker word lists) — words are tagged with their real
    speaker id and sorted by start time, and grouped into turns using
    segments.jsonl's diarized boundaries."""
    out_path = vid_dir / "transcript.json"
    if out_path.exists() and not overwrite:
        return "skip"

    mj = vid_dir / "meta.json"
    if not mj.exists():
        return "no_meta"
    meta = json.loads(mj.read_text(encoding="utf-8"))
    lang_code = (meta.get("language_code") or "").lower()
    speaker_map = meta.get("speaker_map") or {"spk0": "SPEAKER_00", "spk1": "SPEAKER_01"}

    models = {}
    asr_stats = {}
    all_words = []
    for spk in ("spk0", "spk1"):
        wav = vid_dir / f"{spk}.wav"
        if not wav.exists():
            continue
        try:
            asr_stats[spk] = {}
            model_id, words = transcribe_file(wav, lang_code, device_str, stats_out=asr_stats[spk])
        except Exception as e:
            log(f"  [{vid_dir.name}] transcribe {spk} failed: {str(e)[:200]}", "WARN")
            asr_stats.pop(spk, None)
            continue
        models[spk] = model_id
        speaker = speaker_map.get(spk, spk)
        for w in words:
            all_words.append({"speaker": speaker, **w})

    all_words.sort(key=lambda w: w["start"])

    turns = []
    seg_path = vid_dir / "segments.jsonl"
    if seg_path.exists():
        wi = 0
        for line in seg_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            seg = json.loads(line)
            seg_start, seg_end, speaker = seg["start"], seg["end"], seg["speaker"]
            turn_words = [w for w in all_words
                          if w["speaker"] == speaker and seg_start <= w["start"] < seg_end]
            if not turn_words:
                continue
            turns.append({
                "speaker": speaker, "start": seg_start, "end": seg_end,
                "text": " ".join(w["word"] for w in turn_words),
            })

    # A speaker channel needs enough text to judge; a near-silent channel
    # proves nothing either way.
    suspect_language = any(
        s.get("n_letters", 0) >= 50 and (s.get("devanagari_frac") or 0) < 0.5
        for s in asr_stats.values())
    record = {
        "id": meta.get("id"),
        "language_code": lang_code,
        "models": models,
        "aligner": os.environ.get("ALIGN_BACKEND", "iwv"),
        # word ends extended toward the next word (see _extend_word_ends); only
        # the IndicWav2Vec path does this, and Whisper-fallback words already
        # carry their own end times
        "end_fix": "v1" if (os.environ.get("ALIGN_BACKEND", "iwv") == "iwv"
                            and not any("whisper" in m for m in models.values())) else None,
        "asr_stats": asr_stats,
        "suspect_language": suspect_language,
        "words": all_words,
        "turns": turns,
    }
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return "ok"


def _transcribe_multiprocess(args, devices):
    """One worker PROCESS per GPU. The ASR decodes token by token in Python,
    so worker threads in a single process serialize on the GIL: N GPUs then
    give roughly one GPU's throughput (measured ~10 audio-hours per hour on 8
    GPUs, node CPU load ~4). Separate processes scale close to linearly.
    Each child is pinned to one GPU and handles the videos whose id hashes to
    its shard, so results are identical to a single-process run."""
    vis = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    procs = []
    for i, dev in enumerate(devices):
        idx = dev.split(":")[-1]
        phys = vis[int(idx)] if vis else idx
        cmd = [sys.executable, os.path.abspath(__file__), "transcribe",
               "--root", str(args.root), "--gpus", "0", "--shard", f"{i}/{len(devices)}"]
        if args.overwrite:
            cmd.append("--overwrite")
        procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=phys)))
    log(f"launched {len(procs)} worker processes, one per GPU: {devices}")
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit(f"worker exit codes: {codes}")
    log("all workers finished")


def cmd_transcribe(args):
    """Word-level-timestamped transcription over already-accepted videos —
    a separate pass over corpus/accepted/, independent of `run`."""
    import zlib
    root = Path(args.root)
    accepted = sorted(d for d in (root / "accepted").iterdir() if d.is_dir())
    if not accepted:
        sys.exit(f"no accepted entries under {root/'accepted'}")

    devices = resolve_devices(args.gpus)
    if getattr(args, "shard", None):
        k, n = (int(x) for x in args.shard.split("/"))
        # stable hash of the id, not list position: the directory can grow
        # while a long run is in progress
        accepted = [d for d in accepted if zlib.crc32(d.name.encode("utf-8")) % n == k]
    elif len(devices) > 1 and devices != ["cpu"]:
        return _transcribe_multiprocess(args, devices)
    log(f"transcribing {len(accepted)} video(s) on {devices}")

    work_q = queue.Queue()
    for d in accepted:
        work_q.put(d)
    for _ in devices:
        work_q.put(None)

    counts = {}
    counts_lock = threading.Lock()

    def worker(device_str):
        while True:
            d = work_q.get()
            if d is None:
                work_q.task_done()
                return
            try:
                status = transcribe_one(d, device_str, args.overwrite)
            except Exception as e:
                log(f"  [{d.name}] unhandled: {str(e)[:250]}", "ERROR")
                status = "error"
            with counts_lock:
                counts[status] = counts.get(status, 0) + 1
            log(f"  [{d.name}] transcribe: {status}")
            work_q.task_done()

    threads = [threading.Thread(target=worker, args=(dev,), daemon=True) for dev in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    log(f"done: {counts}")


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
    d.add_argument("--cookies", help="path to a Netscape-format cookies.txt for yt-dlp auth")
    d.add_argument("--cookies-from-browser",
                   help="browser to read a live YouTube session from, e.g. 'chrome', 'firefox'")
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
    r.add_argument("--api-key", default=NVIDIA_API_KEY,
                   help="Nemotron NIM API key(s). Comma-separate multiple keys to "
                        "spread verify calls across accounts — raise --verify-concurrency "
                        "roughly proportionally when you do.")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--cc-only", action="store_true")
    r.add_argument("--keep-raw", action="store_true", help="keep the source mixture in raw/")
    r.add_argument("--redo", action="store_true", help="reprocess already-decided videos")
    r.add_argument("--min-duration", type=float, help="override min seconds")
    r.add_argument("--max-duration", type=float, help="override max seconds")
    r.add_argument("--cookies", help="path to a Netscape-format cookies.txt for yt-dlp auth")
    r.add_argument("--cookies-from-browser",
                   help="browser to read a live YouTube session from, e.g. 'chrome', 'firefox'")
    r.add_argument("--workers", type=int, default=8,
                   help="light-stage thread count (probe/download/VAD/Nemotron verify)")
    r.add_argument("--gpus", default="auto",
                   help="'auto' (idle GPUs only), comma list e.g. '0,1', or 'cpu'. "
                        "Drives both diarization and DialogueSidon device placement "
                        "for the heavy (GPU) stage.")
    r.add_argument("--heavy-workers-per-gpu", type=int, default=1,
                   help="workers per GPU for diarize+separate; >1 is experimental, "
                        "pyannote Pipelines aren't meant to be called concurrently "
                        "from multiple threads on one instance")
    r.add_argument("--verify-concurrency", type=int, default=3,
                   help="max concurrent Nemotron API calls across all light workers "
                        "combined — the free-tier NIM endpoint is shared, throttle here")
    r.add_argument("--queue-size", type=int, default=0,
                   help="light->heavy backpressure buffer; 0 = 2x heavy worker count")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("stats", help="show funnel counts")
    s.add_argument("--root", default="./corpus")
    s.set_defaults(func=cmd_stats)

    e = sub.add_parser("export", help="flatten to a training manifest")
    e.add_argument("--root", default="./corpus")
    e.add_argument("--out", default="train_manifest.jsonl")
    e.set_defaults(func=cmd_export)

    t = sub.add_parser("transcribe", help="word-level-timestamped transcription of accepted videos")
    t.add_argument("--root", default="./corpus")
    t.add_argument("--gpus", default="auto",
                   help="'auto' (idle GPUs only), comma list e.g. '0,1', or 'cpu'")
    t.add_argument("--overwrite", action="store_true",
                   help="redo videos that already have transcript.json")
    t.add_argument("--shard", default=None, help=argparse.SUPPRESS)  # "k/n", set by the launcher
    t.set_defaults(func=cmd_transcribe)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()