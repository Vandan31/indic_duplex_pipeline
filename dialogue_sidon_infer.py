#!/usr/bin/env python3
"""Local DialogueSidon inference: split a two-speaker mixture into speaker tracks.

Adapted from the authors' Gradio Space (sarulab-speech/DialogueSidon-demo) with
the Gradio/CUDA assumptions removed so it can run on Apple Silicon.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from diffusers import DPMSolverMultistepScheduler
from huggingface_hub import hf_hub_download
from torch.export.passes import move_to_device_pass

REPO_ID = "sarulab-speech/DialogueSidon"
SAMPLE_RATE_IN = 16_000
CHUNK_SECONDS = 120.0
OVERLAP_SECONDS = 10.0


def _pad_batch(features, pad_to_multiple_of=2, padding_value=0.0):
    target_length = max(f.shape[0] for f in features)
    if pad_to_multiple_of:
        target_length = (
            (target_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
        )
    batch_size = len(features)
    feature_dim = features[0].shape[1]
    device = features[0].device
    padded = torch.full(
        (batch_size, target_length, feature_dim),
        padding_value,
        dtype=torch.float32,
        device=device,
    )
    mask = torch.zeros((batch_size, target_length), dtype=torch.int64, device=device)
    for i, feat in enumerate(features):
        padded[i, : feat.shape[0]] = feat
        mask[i, : feat.shape[0]] = 1
    return padded, mask


def extract_fbank_features(waveforms, device, num_mel_bins=80, stride=2):
    features = []
    for wav in waveforms:
        if wav.ndim > 1:
            wav = wav[0]
        feat = torchaudio.compliance.kaldi.fbank(
            wav.unsqueeze(0).cpu(),
            sample_frequency=SAMPLE_RATE_IN,
            num_mel_bins=num_mel_bins,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            preemphasis_coefficient=0.97,
            remove_dc_offset=True,
            window_type="povey",
            use_energy=False,
            energy_floor=1.192092955078125e-07,
        )
        mean = feat.mean(0, keepdim=True)
        var = feat.var(0, keepdim=True)
        feat = (feat - mean) / torch.sqrt(var + 1e-5)
        features.append(feat.to(device))

    input_features, attention_mask = _pad_batch(features)
    b, t, c = input_features.shape
    t = (t // stride) * stride
    input_features = input_features[:, :t, :]
    attention_mask = attention_mask[:, :t]
    input_features = input_features.reshape(b, t // stride, c * stride)
    attention_mask = attention_mask[:, 1::stride]
    return {"input_features": input_features, "attention_mask": attention_mask}


@contextlib.contextmanager
def _torch_load_mapped_to(device: torch.device):
    """The published .pt2 files carry CUDA storage tags, and torch.export.load
    exposes no map_location. Force one for the duration of the load."""
    original = torch.load

    def patched(*args, **kwargs):
        kwargs["map_location"] = device
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


def load_models(device: torch.device) -> dict:
    files = ["ssl_encoder.pt2", "diffusion_head.pt2", "vae_decoder.pt2", "metadata.json"]
    paths = {f: hf_hub_download(repo_id=REPO_ID, filename=f) for f in files}

    with open(paths["metadata.json"]) as fp:
        meta = json.load(fp)

    def load_component(name: str):
        t0 = time.time()
        # Storages load onto `device`; move_to_device_pass additionally rewrites
        # device kwargs that were baked into the exported graph at trace time.
        with _torch_load_mapped_to(device):
            ep = torch.export.load(paths[name])
        # no_grad is required: the pass calls .to() on CUDA fake tensors in node
        # metadata, and building autograd metadata for them aborts the process
        # on a build without CUDA support.
        with torch.no_grad():
            module = move_to_device_pass(ep, str(device)).module().to(device)
        n = sum(p.numel() for p in module.parameters()) + sum(
            b.numel() for b in module.buffers()
        )
        print(f"  {name:20s} {n/1e6:8.1f}M tensors  ({time.time()-t0:.1f}s)")
        return module, n

    print("Loading exported components:")
    ssl_encoder, n1 = load_component("ssl_encoder.pt2")
    diffusion_head, n2 = load_component("diffusion_head.pt2")
    vae_decoder, n3 = load_component("vae_decoder.pt2")
    print(f"  {'TOTAL':20s} {(n1+n2+n3)/1e6:8.1f}M")

    scheduler = DPMSolverMultistepScheduler.from_config(
        meta["ddpm_config"], algorithm_type="dpmsolver++", timestep_spacing="linspace"
    )

    return {
        "ssl_encoder": ssl_encoder,
        "diffusion_head": diffusion_head,
        "vae_decoder": vae_decoder,
        "latent_norm_mean": torch.tensor(
            meta["latent_norm_mean"], dtype=torch.float32, device=device
        ).view(1, 1, -1),
        "latent_norm_std": torch.tensor(
            meta["latent_norm_std"], dtype=torch.float32, device=device
        ).view(1, 1, -1),
        "latent_norm_initialized": meta["latent_norm_initialized"],
        "scheduler": scheduler,
        "latent_dim": meta["latent_dim"],
        "sample_rate": meta["sample_rate"],
    }


def _normalize(latents, models):
    if not models["latent_norm_initialized"]:
        return latents
    return (
        (latents.float() - models["latent_norm_mean"]) / models["latent_norm_std"]
    ).to(latents.dtype)


def _denormalize(latents, models):
    if not models["latent_norm_initialized"]:
        return latents
    return (latents.float() * models["latent_norm_std"] + models["latent_norm_mean"]).to(
        latents.dtype
    )


@torch.inference_mode()
def _separate_chunk(wav, num_steps, models, device):
    latent_dim = models["latent_dim"]

    t0 = time.time()
    noisy_ssl = extract_fbank_features([wav.view(-1)], device)
    features, pred0, pred1 = models["ssl_encoder"](
        noisy_ssl["input_features"], noisy_ssl["attention_mask"]
    )
    print(f"    SSL encoder: {time.time()-t0:.1f}s  frames={features.shape[1]}")

    predicted_latents = torch.cat([pred0, pred1], dim=-1)
    conditioning = torch.cat([_normalize(predicted_latents, models), features], dim=-1)

    seq_len = conditioning.shape[1]
    scheduler = models["scheduler"]
    scheduler.set_timesteps(num_steps, device=device)
    latents = torch.randn(
        (1, seq_len, latent_dim * 2), device=device, dtype=conditioning.dtype
    )
    t0 = time.time()
    for i, t in enumerate(scheduler.timesteps):
        t_batch = torch.full((1,), int(t.item()), device=device, dtype=torch.long)
        latents = scheduler.step(
            models["diffusion_head"](latents, t_batch, conditioning), t, latents
        ).prev_sample
        if i == 0:
            print(f"    diffusion step 0: {time.time()-t0:.2f}s")
    print(f"    diffusion ({num_steps} steps): {time.time()-t0:.1f}s")

    latents = _denormalize(latents, models)
    t0 = time.time()
    spk1 = models["vae_decoder"](latents[:, :, :latent_dim].transpose(1, 2)).squeeze(0)
    spk2 = models["vae_decoder"](latents[:, :, latent_dim:].transpose(1, 2)).squeeze(0)
    print(f"    VAE decode x2: {time.time()-t0:.1f}s")
    return torch.cat([spk1, spk2], dim=0)


def _channel_similarity(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    # (a * b).sum() instead of torch.dot(a, b): torch.dot routes to cublasSdot,
    # which raises CUBLAS_STATUS_NOT_SUPPORTED on this driver/GPU combination
    # (V100, sm_70). Elementwise-multiply-then-reduce is mathematically
    # identical and avoids that cuBLAS call entirely.
    return float((a * b).sum() / denom) if float(denom) > 1e-8 else 0.0


def _maybe_swap(prev_overlap, curr_chunk, overlap_samples):
    if overlap_samples <= 0:
        return curr_chunk
    curr_ov = curr_chunk[:, :overlap_samples]
    direct = _channel_similarity(prev_overlap[0], curr_ov[0]) + _channel_similarity(
        prev_overlap[1], curr_ov[1]
    )
    swapped = _channel_similarity(prev_overlap[0], curr_ov[1]) + _channel_similarity(
        prev_overlap[1], curr_ov[0]
    )
    return curr_chunk[[1, 0], :] if swapped > direct else curr_chunk


def separate(wav, sample_rate, num_steps, models, device):
    out_sr = models["sample_rate"]
    if sample_rate != SAMPLE_RATE_IN:
        wav_16k = torchaudio.functional.resample(wav, sample_rate, SAMPLE_RATE_IN)
    else:
        wav_16k = wav
    wav_16k = wav_16k.to(device)

    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE_IN)
    total_samples = wav_16k.shape[-1]

    if total_samples <= chunk_samples:
        print(f"  single-shot inference ({total_samples/SAMPLE_RATE_IN:.1f}s)")
        max_val = wav_16k.abs().max().clamp_min(1e-6)
        wav_norm = torch.nn.functional.pad(0.9 * wav_16k / max_val, (160, 160))
        return _separate_chunk(wav_norm, num_steps, models, device), out_sr

    overlap_samples_in = int(OVERLAP_SECONDS * SAMPLE_RATE_IN)
    hop_samples = chunk_samples - overlap_samples_in
    stitched, prev_end_in = None, 0

    for start in range(0, total_samples, hop_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = wav_16k[:, start:end]
        max_val = chunk.abs().max().clamp_min(1e-6)
        chunk_norm = torch.nn.functional.pad(0.9 * chunk / max_val, (160, 160))
        pred = _separate_chunk(chunk_norm, num_steps, models, device)

        target_out = max(1, round((end - start) * out_sr / SAMPLE_RATE_IN))
        if pred.shape[-1] > target_out:
            pred = pred[:, :target_out]
        elif pred.shape[-1] < target_out:
            pred = torch.cat(
                [pred, torch.zeros(2, target_out - pred.shape[-1], device=device)], dim=-1
            )

        if stitched is None:
            stitched, prev_end_in = pred, end
            continue

        overlap_in = max(0, prev_end_in - start)
        overlap_out = max(
            0,
            min(
                round(overlap_in * out_sr / SAMPLE_RATE_IN),
                stitched.shape[-1],
                pred.shape[-1],
            ),
        )
        if overlap_out > 0:
            pred = _maybe_swap(stitched[:, -overlap_out:], pred, overlap_out)
            fade = torch.linspace(0.0, 1.0, overlap_out, device=device).unsqueeze(0)
            blended = stitched[:, -overlap_out:] * (1 - fade) + pred[:, :overlap_out] * fade
            stitched = torch.cat(
                [stitched[:, :-overlap_out], blended, pred[:, overlap_out:]], dim=-1
            )
        else:
            stitched = torch.cat([stitched, pred], dim=-1)
        prev_end_in = end

    return stitched, out_sr


def load_audio(path: str):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--output-dir", default="separated")
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    if args.device == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    else:
        dev = args.device
    device = torch.device(dev)
    print(f"device={device}  torch={torch.__version__}  threads={torch.get_num_threads()}")

    wav, sr = load_audio(args.input)
    print(f"input: {args.input}  {wav.shape[-1]/sr:.2f}s @ {sr} Hz -> mono")

    models = load_models(device)

    t0 = time.time()
    separated, out_sr = separate(wav, sr, args.num_steps, models, device)
    elapsed = time.time() - t0
    audio_seconds = separated.shape[-1] / out_sr
    print(f"\ninference: {elapsed:.1f}s for {audio_seconds:.1f}s audio -> RTF {elapsed/audio_seconds:.3f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem.replace(" ", "_")

    separated = separated.float().cpu()
    peak = separated.abs().max().clamp_min(1e-6)
    written = []
    for i in range(2):
        track = (separated[i] / peak * 0.9).numpy()
        p = out_dir / f"{stem}_speaker{i+1}.wav"
        sf.write(p, track, out_sr)
        written.append(p)
        rms = float(np.sqrt((track**2).mean()))
        print(f"wrote {p}  rms={rms:.4f}  peak={float(np.abs(track).max()):.3f}")

    stereo = np.stack([(separated[0] / peak * 0.9).numpy(), (separated[1] / peak * 0.9).numpy()], axis=1)
    p = out_dir / f"{stem}_stereo_spk1L_spk2R.wav"
    sf.write(p, stereo, out_sr)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
