"""Model -> Surge patch(es), shared by the evaluation scripts."""
import time

import numpy as np
import torch

import param_codec as codec
from models import FlowMatchingResMLP, load_inverter  # noqa: F401  (load_inverter re-exported)
from surge_spec import vector_to_patch


def parse_ckpt_specs(specs):
    """['name=/path/a.pt', ...] -> [(name, path), ...]"""
    out = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--ckpt expects NAME=PATH, got {s!r}")
        name, path = s.split("=", 1)
        out.append((name, path))
    return out


def is_flow(model) -> bool:
    return isinstance(model, FlowMatchingResMLP)


@torch.no_grad()
def predict_vectors(model, mel, n_draws=1, steps=20, seed=0):
    """mel [B, 128, 81] tensor -> np.array [n_draws, B, 23] (point models: n_draws = 1).
    Returns (vectors, latency_ms per input for one draw)."""
    device = next(model.parameters()).device
    mel = mel.to(device)
    t0 = time.time()
    if not is_flow(model):
        vecs = model.predict_params(mel).unsqueeze(0)
    else:
        gen = torch.Generator(device=device).manual_seed(seed)
        vecs = torch.stack([model.predict_params(mel, num_steps=steps, generator=gen) for _ in range(n_draws)])
    lat_ms = (time.time() - t0) * 1000.0 / max(1, vecs.shape[0]) / mel.shape[0]
    return vecs.cpu().numpy(), lat_ms


def vectors_to_patches(vecs_one_input):
    """[n_draws, 23] -> list of patch dicts."""
    return [vector_to_patch(v) for v in np.atleast_2d(vecs_one_input)]


def predict_vectors_pinned(model, mel, midi_note, n_draws=20, steps=25, seed=42):
    """Predict vectors with coordinate 0 (midi_note) pinned to target_midi along the ODE path."""
    from surge_spec import NOTE_LOW, NOTE_HIGH
    device = next(model.parameters()).device
    mel = mel.to(device)
    target_norm_midi = float(np.clip((midi_note - NOTE_LOW) / (NOTE_HIGH - NOTE_LOW), 0.0, 1.0))
    with torch.no_grad():
        audio_emb = model.encoder(mel)
        dt = 1.0 / steps
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(n_draws, model.param_dim, device=device, generator=gen)
        for i in range(steps):
            t_val = i / steps
            t = torch.full((n_draws,), t_val, device=device, dtype=torch.float32)
            x[:, 0] = (1.0 - t_val) * x[:, 0] + t_val * target_norm_midi
            v = model(x, t, audio_emb=audio_emb)
            x = x + v * dt
        x[:, 0] = target_norm_midi
        x = torch.clamp(x, 0.0, 1.0)
        if getattr(model, "encoding", None) != codec.ENCODING_V1:
            decoded = codec.decode(x)
        else:
            decoded = x
    return decoded.cpu().numpy()



def compute_bin_energy_contour_loss(cand_audio, target_audio, sr=44100, n_mels=48, hop_length=128):
    """Measures both the per-frame resonance envelope profile and per-bin temporal decay contours."""
    import librosa
    mc = librosa.feature.melspectrogram(y=cand_audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mt = librosa.feature.melspectrogram(y=target_audio, sr=sr, n_mels=n_mels, hop_length=hop_length)

    # 1. Per-frame spectral resonance profile (cosine distance)
    pc = mc / (np.linalg.norm(mc, axis=0, keepdims=True) + 1e-7)
    pt = mt / (np.linalg.norm(mt, axis=0, keepdims=True) + 1e-7)
    shape_err = float(1.0 - np.mean(np.sum(pc * pt, axis=0)))

    # 2. Per-bin temporal energy contour correlation
    tc = mc - np.mean(mc, axis=1, keepdims=True)
    tt = mt - np.mean(mt, axis=1, keepdims=True)
    norm_c = np.linalg.norm(tc, axis=1) + 1e-7
    norm_t = np.linalg.norm(tt, axis=1) + 1e-7
    bin_corrs = np.sum(tc * tt, axis=1) / (norm_c * norm_t)
    temp_err = float(1.0 - np.mean(bin_corrs))

    return shape_err, temp_err


def predict_and_rerank_candidates(
    model, synth, target_audio, mel, midi_note, note_dur,
    n_candidates=20, steps=25, seed=42,
    weight_stft=1.0, weight_pitch=3.0, weight_sub=4.0, weight_env=2.0, weight_richness=1.0,
    weight_stereo=1.5, weight_mod=0.1, weight_contour=1.5, pin_pitch=True
):
    """Generate n_candidates with the Flow model (with pitch-locking), render each in Surge XT,
    extract acoustic descriptors (F0, sub-octave power, bin energy contours, envelope duration, centroid, harmonics, stereo width, beating speed),
    and pick the best candidate that matches pitch, timbre, resonance contour, envelope dynamics, and detuning/chorus speed.
    """
    from audio_utils import MultiScaleSTFTLoss, extract_pitch_and_envelope
    from surge_spec import render_patch, SAMPLE_RATE

    loss_fn = MultiScaleSTFTLoss()
    target_desc = extract_pitch_and_envelope(target_audio, SAMPLE_RATE)

    # Sub-octave power below 0.75 * nominal F0 (detects false sub-oscillator injection)
    nominal_f0 = 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))
    sub_thresh = 0.75 * nominal_f0

    def get_sub_ratio(sig):
        fft = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), 1.0 / SAMPLE_RATE)
        tot = np.sum(fft**2) + 1e-9
        return float(np.sum(fft[freqs < sub_thresh]**2) / tot)

    t_sub = get_sub_ratio(target_audio)

    if pin_pitch and is_flow(model):
        vecs = predict_vectors_pinned(model, mel, midi_note=midi_note, n_draws=n_candidates, steps=steps, seed=seed)
        candidates = vectors_to_patches(vecs)
    else:
        vecs, _ = predict_vectors(model, mel, n_draws=n_candidates, steps=steps, seed=seed)
        candidates = vectors_to_patches(vecs[:, 0])

    scored_candidates = []
    for idx, cand_patch in enumerate(candidates):
        cand_audio = render_patch(synth, cand_patch, midi_note, note_dur, duration=0.8)
        cand_desc = extract_pitch_and_envelope(cand_audio, SAMPLE_RATE)
        c_sub = get_sub_ratio(cand_audio)

        # 1. MultiScale STFT loss
        stft_l = loss_fn(cand_audio, target_audio)

        # 2. Musical pitch discrepancy penalty (semitones with octave-jump disqualification)
        f0_diff = abs(target_desc["f0_hz"] - cand_desc["f0_hz"]) if (target_desc["f0_hz"] > 0 and cand_desc["f0_hz"] > 0) else 0.0
        if target_desc["f0_hz"] > 20.0 and cand_desc["f0_hz"] > 20.0:
            semi_err = 12.0 * abs(np.log2(cand_desc["f0_hz"] / target_desc["f0_hz"]))
            pitch_penalty = semi_err + (5.0 * max(0.0, semi_err - 1.5))
        else:
            semi_err = 0.0
            pitch_penalty = 0.05 * f0_diff

        # 3. Sub-octave leakage penalty (rejects unwanted sub_mix octave drops)
        sub_diff = abs(t_sub - c_sub)

        # 4. Envelope length discrepancy penalty (prevents plucks from turning into drones)
        env_diff = abs(target_desc["active_dur_s"] - cand_desc["active_dur_s"])
        center_diff = abs(target_desc["env_center_s"] - cand_desc["env_center_s"])

        # 5. Harmonic richness mismatch
        rich_diff = abs(target_desc["harmonic_richness"] - cand_desc["harmonic_richness"])

        # 6. Stereo detuning and chorus modulation rate
        stereo_diff = abs(target_desc["stereo_width"] - cand_desc["stereo_width"])
        mod_diff = abs(target_desc["mod_speed_hz"] - cand_desc["mod_speed_hz"])

        # 7. Resonant Bin Energy Contour Matching (tracks filter sweep profile & individual band decays)
        shape_err, temp_err = compute_bin_energy_contour_loss(cand_audio, target_audio, SAMPLE_RATE)
        contour_loss = shape_err + 0.5 * temp_err

        # 8. Extreme resonance suppression (unless target is screaming)
        reso_penalty = 2.0 * max(0.0, cand_patch["resonance"] - 0.85)

        total_score = (
            weight_stft * stft_l
            + weight_pitch * pitch_penalty
            + weight_sub * sub_diff
            + weight_env * (env_diff + 0.5 * center_diff)
            + weight_richness * rich_diff
            + weight_stereo * stereo_diff
            + weight_mod * mod_diff
            + weight_contour * contour_loss
            + reso_penalty
        )

        scored_candidates.append({
            "candidate_index": idx,
            "patch": cand_patch,
            "audio": cand_audio,
            "descriptors": cand_desc,
            "stft_loss": stft_l,
            "f0_err_hz": f0_diff,
            "semi_err": semi_err,
            "sub_err": sub_diff,
            "env_err_s": env_diff,
            "stereo_err": stereo_diff,
            "mod_err_hz": mod_diff,
            "contour_loss": contour_loss,
            "res_shape_err": shape_err,
            "res_temp_err": temp_err,
            "total_score": total_score
        })

    scored_candidates.sort(key=lambda x: x["total_score"])
    best = scored_candidates[0]
    return best, target_desc, scored_candidates



