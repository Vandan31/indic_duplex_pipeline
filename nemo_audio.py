#!/usr/bin/env python3
"""
nemo_audio.py — analyze audio with NVIDIA Nemotron 3 Nano Omni via NVIDIA NIM.

Reports number of speakers, language, and a content summary.
Writes results next to the input file:
    <name>_analysis.json   structured fields
    <name>_analysis.md     human-readable report

Usage:
    python3 nemo_audio.py hi.wav
    NVIDIA_API_KEY="nvapi-..." python3 nemo_audio.py hi.wav   # override embedded key
    python3 nemo_audio.py /data3/vandan.raval/s2st/clips/     # whole folder
    python3 nemo_audio.py hi.wav --convert                    # force 16k mono wav first
    python3 nemo_audio.py hi.wav --no-think                   # skip reasoning trace (faster)

Requires: requests           (pip install requests)
Optional: ffmpeg             (only for --convert / non-wav inputs)
"""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

# Hardcoded fallback. NVIDIA_API_KEY in the environment overrides this if set.
# Don't commit this file to git as-is.
API_KEY = os.environ.get("NVIDIA_API_KEY", "")

MIME_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
    ".aac": "audio/aac",
}

PROMPT = """Listen to this audio and analyze it. Respond with ONLY a JSON object, no markdown fences, no commentary before or after.

Use exactly this schema:

{
  "num_speakers": <integer, your best count of distinct voices; use 0 if no speech>,
  "speakers": [
    {"id": "S1", "apparent_gender": "male|female|unclear", "notes": "<accent, age impression, role>"}
  ],
  "language": "<primary language name, e.g. Hindi, English, Marathi>",
  "language_code": "<ISO 639-1 code, e.g. hi, en, mr>",
  "other_languages": ["<any additional languages or code-switching observed>"],
  "summary": "<3-5 sentence description of what is happening in the audio>",
  "topic": "<short topic label>",
  "tone": "<formal, casual, excited, angry, neutral, etc.>",
  "setting": "<interview, lecture, phone call, podcast, news, field recording, etc.>",
  "background_sounds": ["<non-speech sounds: music, traffic, silence, hum, etc.>"],
  "audio_quality": "<clear / noisy / muffled / clipped, plus any issues>",
  "key_moments": ["<notable points or quotes, in order>"],
  "transcript": "<transcription of the speech; empty string if no speech>",
  "confidence": "<high|medium|low — your confidence in the speaker count and language>",
  "unclear": ["<anything you could not make out>"]
}

Be honest: if you cannot tell how many speakers there are, set confidence to "low" and explain in unclear. Do not invent content."""


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_wav(src: Path, workdir: Path) -> Path:
    """Convert to 16 kHz mono WAV — the format the model handles most reliably."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found but conversion requested")
    dst = workdir / (src.stem + ".wav")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        check=True,
    )
    return dst


def duration_seconds(path: Path):
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def build_data_uri(path: Path) -> str:
    mime = MIME_MAP.get(path.suffix.lower(), "audio/wav")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}", len(b64)


def call_model(api_key: str, data_uri: str, think: bool, max_retries: int = 4) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                # NVIDIA NIM uses audio_url with a data URI — NOT OpenRouter's input_audio.
                {"type": "audio_url", "audio_url": {"url": data_uri}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        "max_tokens": 16384,
        "temperature": 0.2,
        "top_p": 0.95,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    if think:
        payload["reasoning_budget"] = 8192

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        # Keeps the serverless endpoint from dropping long audio requests.
        "NVCF-POLL-SECONDS": "1800",
    }

    delay = 5
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=1800)
        except requests.RequestException as e:
            if attempt == max_retries:
                raise
            log(f"  network error ({e.__class__.__name__}), retry in {delay}s")
            time.sleep(delay); delay *= 2
            continue

        if r.status_code == 200:
            return r.json()

        if r.status_code in (408, 429, 500, 502, 503, 504) and attempt < max_retries:
            wait = int(r.headers.get("Retry-After", delay))
            log(f"  HTTP {r.status_code}, retry in {wait}s ({attempt}/{max_retries})")
            time.sleep(wait); delay *= 2
            continue

        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:600]}")

    raise RuntimeError("exhausted retries")


def extract_json(text: str):
    """Pull a JSON object out of the reply, tolerating fences or stray prose."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # fall back to the outermost {...}
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def render_markdown(a: dict, meta: dict) -> str:
    def get(k, default="—"):
        v = a.get(k)
        if v in (None, "", []):
            return default
        if isinstance(v, list):
            return "\n".join(f"- {i}" for i in v)
        return v

    lines = [
        f"# Audio analysis — {meta['file']}",
        "",
        f"- Source: `{meta['path']}`",
        f"- Model: `{MODEL}`",
        f"- Generated: {meta['when']}",
    ]
    if meta.get("duration"):
        lines.append(f"- Duration: {meta['duration']:.1f}s")
    lines += [
        f"- Processing time: {meta['elapsed']:.0f}s",
        f"- Tokens: {meta.get('tokens', 0)}",
        "",
        "---",
        "",
        "## At a glance",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Speakers | **{a.get('num_speakers', '—')}** |",
        f"| Language | **{a.get('language', '—')}** (`{a.get('language_code', '—')}`) |",
        f"| Other languages | {', '.join(a.get('other_languages') or []) or '—'} |",
        f"| Topic | {a.get('topic', '—')} |",
        f"| Tone | {a.get('tone', '—')} |",
        f"| Setting | {a.get('setting', '—')} |",
        f"| Audio quality | {a.get('audio_quality', '—')} |",
        f"| Confidence | {a.get('confidence', '—')} |",
        "",
        "## Summary",
        "",
        get("summary"),
        "",
        "## Speakers",
        "",
    ]

    speakers = a.get("speakers") or []
    if speakers:
        for s in speakers:
            lines.append(
                f"- **{s.get('id', '?')}** — {s.get('apparent_gender', 'unclear')}"
                f" — {s.get('notes', '')}"
            )
    else:
        lines.append("—")

    lines += [
        "",
        "## Background sounds",
        "",
        get("background_sounds"),
        "",
        "## Key moments",
        "",
        get("key_moments"),
        "",
        "## Unclear / low confidence",
        "",
        get("unclear", "None flagged."),
        "",
        "## Transcript",
        "",
        "```",
        a.get("transcript") or "(none)",
        "```",
        "",
    ]
    return "\n".join(lines)


def process(api_key: str, path: Path, args):
    json_out = path.with_name(path.stem + "_analysis.json")
    md_out = path.with_name(path.stem + "_analysis.md")

    if json_out.exists() and not args.overwrite:
        log(f"SKIP {path.name} (already analyzed; --overwrite to redo)")
        return

    log(f"START {path.name}")
    t0 = time.time()
    dur = duration_seconds(path)
    if dur:
        log(f"  duration {dur:.1f}s")

    with tempfile.TemporaryDirectory() as td:
        send_path = path
        if args.convert or path.suffix.lower() not in MIME_MAP:
            log("  converting to 16 kHz mono wav")
            send_path = to_wav(path, Path(td))

        data_uri, b64_len = build_data_uri(send_path)
        log(f"  payload ~{b64_len / 1_048_576:.1f} MB base64, sending")
        resp = call_model(api_key, data_uri, think=not args.no_think)

    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    tokens = resp.get("usage", {}).get("total_tokens", 0)
    elapsed = time.time() - t0

    analysis = extract_json(content)
    if analysis is None:
        log("  WARN model did not return valid JSON — saving raw text")
        analysis = {"summary": content, "parse_error": True}

    meta = {
        "file": path.name,
        "path": str(path.resolve()),
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": dur,
        "elapsed": elapsed,
        "tokens": tokens,
    }

    record = {"meta": meta, "analysis": analysis}
    if args.keep_reasoning and reasoning:
        record["reasoning"] = reasoning
    if analysis.get("parse_error"):
        record["raw_content"] = content

    json_out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    md_out.write_text(render_markdown(analysis, meta), encoding="utf-8")

    # console summary
    print("\n" + "=" * 60)
    print(f"  {path.name}")
    print("=" * 60)
    print(f"  Speakers : {analysis.get('num_speakers', '?')}")
    print(f"  Language : {analysis.get('language', '?')} ({analysis.get('language_code', '?')})")
    print(f"  Confidence: {analysis.get('confidence', '?')}")
    print("-" * 60)
    print(f"  {analysis.get('summary', '')}")
    print("=" * 60 + "\n")

    log(f"DONE  -> {json_out.name} , {md_out.name}  ({elapsed:.0f}s, {tokens} tok)")


def collect(inputs):
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += [f for f in sorted(p.iterdir()) if f.suffix.lower() in MIME_MAP]
        elif p.is_file():
            files.append(p)
        else:
            log(f"WARN not found: {item}")
    return files


def main():
    ap = argparse.ArgumentParser(description="Analyze audio with Nemotron 3 Nano Omni (NVIDIA NIM).")
    ap.add_argument("inputs", nargs="+", help="audio file(s) or folder(s)")
    ap.add_argument("--convert", action="store_true", help="force ffmpeg conversion to 16k mono wav")
    ap.add_argument("--no-think", action="store_true", help="disable the reasoning trace")
    ap.add_argument("--keep-reasoning", action="store_true", help="store the reasoning trace in the json")
    ap.add_argument("--overwrite", action="store_true", help="redo files that already have output")
    ap.add_argument("--delay", type=float, default=2, help="pause between files (seconds)")
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY") or API_KEY
    if not api_key:
        sys.exit("ERROR: no API key — set NVIDIA_API_KEY or fill in API_KEY at the top.")

    files = collect(args.inputs)
    if not files:
        sys.exit("ERROR: no audio files found")

    log(f"{len(files)} file(s) queued")
    failed = []
    for f in files:
        try:
            process(api_key, f, args)
        except Exception as e:
            log(f"FAIL  {f.name}: {e}")
            failed.append(f.name)
        time.sleep(args.delay)

    log(f"Finished. {len(files) - len(failed)} ok, {len(failed)} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()