"""Training data from pre-encoded SA3 latents + grid-aligned mir control features.

Each item is a SAME-L latent (256, 4096), control features already at T=4096 (from
the per-crop .TIMESERIES.npz — no resampling), the text prompt, and an audio
reference latent (a different crop of the same track) for the riffer branch.
No audio decode/encode, no on-the-fly feature extraction.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

# Raw control channels pulled straight from the .TIMESERIES.npz (each already T=4096).
CONTROL_FIELDS = {
    "dynamics": ["rms_energy_bass_ts", "rms_energy_body_ts", "rms_energy_mid_ts", "rms_energy_air_ts"],
    "rhythm": ["beat_activation_ts", "downbeat_activation_ts", "onset_envelope_ts"],
    "melody": ["hpcp_ts"],
    "density_ts": ["onset_envelope_ts"],   # 1-ch onset-density curve (time-varying density control)
    "chroma384": ["same_chroma_ts"],       # SAME 384-d chroma (3 bands x 128), stored (T,384); data-gen TODO
}
CONTROL_DIMS = {"dynamics": 4, "rhythm": 3, "melody": 12, "density_ts": 1, "chroma384": 384, "audio": 256}


def _to_ct(arr: np.ndarray) -> np.ndarray:
    """(T,) -> (1, T); (T, C) -> (C, T). float32."""
    arr = np.asarray(arr, dtype=np.float32)
    return arr[None, :] if arr.ndim == 1 else arr.T


def track_key(meta: dict, path: str) -> str:
    """Group crops by their source track. All crops of a track share `path`
    (the source full_mix.flac); `relpath`'s number is a GLOBAL crop id, so its
    dirname (`<Artist - Title>`) is the per-track fallback."""
    p = meta.get("path")
    if p:
        return p
    rel = meta.get("relpath")
    if rel:
        return os.path.dirname(rel)
    return path


class LatentControlDataset(Dataset):
    """(latent, controls, prompt, ref_latent) from a latents_sa3 directory."""

    def __init__(self, root, controls=("dynamics", "rhythm", "melody"),
                 audio_ref="same_track", seed=0, subset_tracks=None,
                 scalar_field=None, scalar_norm=(0.0, 1.0), random_crop_frames=None):
        self.root = root
        self.controls = [c for c in controls if c in CONTROL_FIELDS]
        self.audio_ref = audio_ref
        self.scalar_field = scalar_field            # e.g. "onset_density" — a per-crop .json scalar control
        self.scalar_mean, self.scalar_std = scalar_norm
        self.random_crop_frames = random_crop_frames   # int N = return a RANDOM beat-aligned N-frame window
        self.paths = sorted(glob.glob(os.path.join(root, "*.npy")))
        self.meta = {}
        self.by_track = defaultdict(list)
        for p in self.paths:
            j = p[:-4] + ".json"
            m = json.load(open(j)) if os.path.exists(j) else {}
            self.meta[p] = m
            self.by_track[track_key(m, p)].append(p)
        if subset_tracks is not None:           # keep a random subset of TRACKS (not crops),
            keys = sorted(self.by_track)         # so each kept track keeps all its crops and the
            np.random.default_rng(seed).shuffle(keys)   # audio-reference pairing still works
            n = int(subset_tracks * len(keys)) if subset_tracks <= 1 else int(subset_tracks)
            keep = set(keys[:max(1, n)])
            self.by_track = defaultdict(list, {k: v for k, v in self.by_track.items() if k in keep})
            self.paths = [p for p in self.paths if track_key(self.meta[p], p) in keep]
        if scalar_field is not None:            # drop crops with no scalar label (e.g. junk 'silence.npy')
            before = len(self.paths)
            self.paths = [p for p in self.paths if scalar_field in self.meta.get(p, {})]
            if len(self.paths) < before:
                print(f"[dataset] dropped {before - len(self.paths)} crops missing '{scalar_field}'", flush=True)
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.paths)

    def _load_controls(self, stem: str) -> dict:
        if not self.controls:                       # scalar/no-timeseries mode: don't touch the npz
            return {}                               # (some crops, e.g. 'silence', have no .TIMESERIES.npz)
        z = np.load(stem + ".TIMESERIES.npz")
        chroma_z = None                             # lazy: the .CHROMA.npz sidecar (e.g. same_chroma_ts)
        out = {}
        for name in self.controls:
            chans = []
            for f in CONTROL_FIELDS[name]:
                if f in z:
                    chans.append(_to_ct(z[f]))
                else:                               # field lives in the parallel <crop>.CHROMA.npz sidecar
                    if chroma_z is None:
                        chroma_z = np.load(stem + ".CHROMA.npz")
                    chans.append(_to_ct(chroma_z[f]))
            out[name] = torch.from_numpy(np.concatenate(chans, axis=0))  # (C, 4096)
        return out

    def _pick_reference(self, path: str) -> torch.Tensor:
        peers = self.by_track.get(track_key(self.meta[path], path), [path])
        cands = [q for q in peers if q != path] or [path]
        ref = cands[int(self._rng.integers(len(cands)))]
        return torch.from_numpy(np.load(ref).astype(np.float32))  # (256, 4096)

    def _beat_aligned_start(self, stem: str, t_total: int) -> int:
        """Random window start snapped to a beat (peak of beat_activation_ts); plain-random fallback.
        Gives crop variety across epochs so training stops seeing only the first window of each track."""
        tw = self.random_crop_frames
        hi = t_total - tw
        if hi <= 0:
            return 0
        try:
            a = np.asarray(np.load(stem + ".TIMESERIES.npz")["beat_activation_ts"],
                           dtype=np.float32).reshape(-1)[:t_total]
            thr = a.mean() + a.std()
            pk = np.where((a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:]) & (a[1:-1] > thr))[0] + 1
            pk = pk[pk <= hi]
            if len(pk):
                return int(pk[self._rng.integers(len(pk))])
        except Exception:
            pass
        return int(self._rng.integers(hi + 1))

    def __getitem__(self, i: int) -> dict:
        p = self.paths[i]
        stem = p[:-4]
        m = self.meta[p]
        lat = np.load(p).astype(np.float32)
        while lat.ndim > 2 and lat.shape[0] == 1:   # squeeze stray batch dims, e.g. (1,256,4096) -> (256,4096)
            lat = lat[0]
        controls = self._load_controls(stem)
        if self.random_crop_frames and lat.shape[-1] > self.random_crop_frames:  # beat-aligned random window
            s = self._beat_aligned_start(stem, lat.shape[-1]); tw = self.random_crop_frames
            lat = np.ascontiguousarray(lat[:, s:s + tw])
            controls = {k: v[:, s:s + tw].contiguous() for k, v in controls.items()}
        item = {
            "latent": torch.from_numpy(lat),                             # (256, 4096) or (256, crop) if random
            "prompt": m.get("prompt", ""),
            "controls": controls,
            "stem": os.path.basename(stem),
            "seconds_total": float(m.get("seconds_total", 0.0)),
        }
        if self.audio_ref == "same_track":
            item["ref_latent"] = self._pick_reference(p)
        if self.scalar_field is not None:           # normalised per-crop scalar control (e.g. onset_density)
            raw = float(m.get(self.scalar_field, self.scalar_mean))
            item["scalar"] = torch.tensor((raw - self.scalar_mean) / (self.scalar_std + 1e-9), dtype=torch.float32)
        pm = m.get("padding_mask")
        if pm is not None:
            item["padding_mask"] = torch.tensor(np.asarray(pm, dtype=np.float32))
        return item

    def track_stats(self) -> dict:
        sizes = [len(v) for v in self.by_track.values()]
        return {"crops": len(self.paths), "tracks": len(self.by_track),
                "crops_per_track_min": min(sizes), "crops_per_track_max": max(sizes),
                "crops_per_track_mean": round(sum(sizes) / len(sizes), 2),
                "singletons": sum(1 for s in sizes if s == 1)}
