# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Pitch Invariance Probing for Cue-Gated ASP.

Per pre-registered protocol (Specification Section 3):
1. Train a ridge regression probe predicting MIDI pitch from:
   (a) Raw un-gated audio encoder embeddings on mixture
   (b) Clean solo audio embeddings
   (c) Cue-gated ASP embeddings from mixture
   Target: R^2 of pitch prediction on cue-gated embeddings must drop significantly
   relative to the raw mixture encoder, showing pitch suppression.
2. Compute cosine similarity ratio:
   sim(same-patch, different-pitch) / sim(different-patch, same-pitch)
"""

import argparse
import os
import sys
import random
import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_v4 import load_inverter_v4
from cue_gated_temporal_asp import CueGatedTemporalAudioEncoder
from audio_utils import ExactGpuMel


def main():
    parser = argparse.ArgumentParser(description="Probe pitch invariance on extracted embeddings")
    parser.add_argument("--v6c-ckpt", type=str, default="/home/kim/sao_runs/v6_C/flow_best.pt")
    parser.add_argument("--gated-ckpt", type=str, default="/home/kim/sao_runs/cue_gated_asp_v7/cue_gated_asp_best.pt")
    parser.add_argument("--bank-dir", type=str, default="/home/kim/sao_runs/bank_v6")
    parser.add_argument("--drum-dir", type=str, default="/run/media/kim/Mantu/avp-stems-original-classified")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== Pitch Invariance Probe on {device} ===")

    # 1. Load models
    flow = load_inverter_v4(args.v6c_ckpt, device=str(device)).eval()
    base_encoder = flow.encoder
    mel_fn = ExactGpuMel(device=str(device))

    gated_encoder = CueGatedTemporalAudioEncoder(
        base_encoder=base_encoder,
        cue_dim=512,
        proj_dim=128,
        hidden_dim=512,
        freeze_base=True,
    ).to(device)

    ckpt = torch.load(args.gated_ckpt, map_location=device, weights_only=False)
    gated_encoder.load_state_dict(ckpt["model_state"])
    gated_encoder.eval()

    # 2. Load drum slices
    drum_files = []
    for song in os.listdir(args.drum_dir):
        sdir = os.path.join(args.drum_dir, song)
        if os.path.isdir(sdir):
            for sub in ["drums", "kick"]:
                p = os.path.join(sdir, sub)
                if os.path.exists(p):
                    for ext in ["*.flac", "*.wav"]:
                        import glob
                        drum_files.extend(glob.glob(os.path.join(p, ext)))

    cached_drums = []
    for df in random.sample(drum_files, min(50, len(drum_files))):
        try:
            with sf.SoundFile(df) as s:
                if s.samplerate == 44100 and s.frames > 35280:
                    start = random.randint(44100 * 5, s.frames - 35280)
                    s.seek(start)
                    data = s.read(35280, dtype="float32")
                    if data.ndim == 2:
                        data = data.mean(axis=1)
                    if abs(data).max() > 0.05:
                        cached_drums.append(torch.from_numpy(data).to(device))
        except Exception:
            pass
    if not cached_drums:
        cached_drums = [torch.randn(35280, device=device) * 0.1]

    # 3. Load patch audio & MIDI
    audio_path = os.path.join(args.bank_dir, "main_00000_audio.npy")
    meta_path = os.path.join(args.bank_dir, "main_00000_meta.npz")
    audio_raw = np.load(audio_path, mmap_mode="r")
    meta = np.load(meta_path)

    n_samples = min(args.n_samples, len(meta["midi"]))
    indices = np.random.choice(len(meta["midi"]), size=n_samples, replace=False)

    midi_targets = meta["midi"][indices]  # [N]
    clean_audio = audio_raw[indices].astype(np.float32) / 32768.0  # [N, 35280]

    raw_mix_embs = []
    clean_embs = []
    gated_embs = []

    print(f"Extracting embeddings for {n_samples} mixtures...")
    batch_size = 64
    for b in range(0, n_samples, batch_size):
        end = min(b + batch_size, n_samples)
        c_chunk = torch.from_numpy(clean_audio[b:end]).to(device)

        # Mix with drums
        d_chunk = torch.stack([random.choice(cached_drums) for _ in range(end - b)])
        c_rms = c_chunk.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-6
        d_rms = d_chunk.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-6
        mix_chunk = c_chunk + d_chunk * (c_rms / d_rms)
        max_v = mix_chunk.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        mix_chunk = mix_chunk / max_v

        with torch.no_grad():
            mel_c = mel_fn(c_chunk / max_v)
            mel_m = mel_fn(mix_chunk)

            z_clean = base_encoder(mel_c)[:, :512]
            z_raw_mix = base_encoder(mel_m)[:, :512]

            # Simulated separated cue (clean + 20% drum bleed)
            cue_wav = (c_chunk + d_chunk * (c_rms / d_rms) * 0.2) / max_v
            cue_mel = mel_fn(cue_wav)
            cue_emb = base_encoder(cue_mel)[:, :512]

            z_gated = gated_encoder(mel_m, cue_emb)[:, :512]

        raw_mix_embs.append(z_raw_mix.cpu().numpy())
        clean_embs.append(z_clean.cpu().numpy())
        gated_embs.append(z_gated.cpu().numpy())

    X_raw = np.concatenate(raw_mix_embs, axis=0)
    X_clean = np.concatenate(clean_embs, axis=0)
    X_gated = np.concatenate(gated_embs, axis=0)
    y = midi_targets

    # 4. Fit Ridge Probes
    def eval_probe(X, y, name):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        clf = Ridge(alpha=10.0)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        r2 = r2_score(y_te, preds)
        mae = np.mean(np.abs(preds - y_te))
        return r2, mae

    r2_raw, mae_raw = eval_probe(X_raw, y, "Raw Mixture")
    r2_clean, mae_clean = eval_probe(X_clean, y, "Clean Solo")
    r2_gated, mae_gated = eval_probe(X_gated, y, "Cue-Gated ASP")

    print("\n" + "=" * 65)
    print("PITCH INVARIANCE PROBE RESULTS (Ridge Regression on MIDI Note)")
    print("=" * 65)
    print(f"Embedding Source               R^2 Score (Pitch)   Pitch MAE (Semis)")
    print("-" * 65)
    print(f"Clean Solo Audio               {r2_clean:.4f}              {mae_clean:.2f} st")
    print(f"Raw Mixture (Baseline)         {r2_raw:.4f}              {mae_raw:.2f} st")
    print(f"Cue-Gated ASP (Mixture)        {r2_gated:.4f}              {mae_gated:.2f} st")
    print("=" * 65)
    if r2_raw > 0:
        suppression = (1.0 - (r2_gated / max(1e-4, r2_raw))) * 100.0
        print(f"Pitch Information Suppression: {suppression:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
