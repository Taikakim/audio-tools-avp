# Owner: KUANG (created 2026-10-13)
"""Paired audio-space metrics for synth inversion: target clip vs rendered candidate (V5_PROPOSAL_REVIEW.md, sec. 3).

Every metric here is a DISTANCE on one (target, prediction) pair: 0 for identical audio, larger = worse. Inputs are
mono float arrays at 44.1 kHz (the renderer's rate); the two clips are cut to the shorter length.

Families (the `family` field in METRICS):
  ours        log-mel L1 on the model's own mel (the paper's best-of-k selector, best human alignment in
              timbre_alignment_check.py), Mel-SSIM on the same mel (plain and silence-gated), a bass-oriented
              octave-band profile, a modulation-spectrum distance and a silence-gated spectral flatness.
  paper       reference_metrics.mss / wmfcc / rms_env_cos (exact ports of the Synth-JEPA paper's metrics).
  sonicmaster faithful ports of SonicMaster's evaluation code (evaluation/evaluate_control_multiple_degs_mass.py and
              extract_kl_ssim_mass.py at ab32b2d), kept ONLY so the ladder gate (ladder_metric_check_v5.py) can test
              them next to ours. Their quirks are reproduced on purpose and listed per function.

Why the "ours" variants differ from SonicMaster's:
  * Mel-SSIM: SonicMaster scales each dB mel min-max per clip, so silent gaps and release tails (a large part of our
    two-note phrase) weigh as much as the notes. `mel_ssim_gated_dist` averages the SSIM map only over pixels within
    `gate_db` of the clip maximum in either clip.
  * Flatness: digital silence has flatness 1, so an ungated clip mean mostly measures how much of the clip is silent.
    `flatness_gated_err` averages only frames within `gate_db` of the loudest frame.
  * Modulation spectrum: mel bands instead of 513 STFT bins, a 256-sample hop, a Hann window over time and the DC bin
    dropped, so unison beating (a few Hz, random phase) shows as a stable peak rather than as leakage of the decay.

Status: unit-tested on synthetic signals (identity, a slow-loop SSIM reference, monotonicity on stand-in sweeps,
beating, silence gating). NOT validated against listeners or on Surge renders: that is what the ladder gate and the
blind A/B answers are for. CLAP is in `clap_distances` (optional dependency laion_clap, a checkpoint path required).
"""
import numpy as np

SR = 44100
MEL_KW = dict(n_mels=128, n_fft=1024, hop_length=441, fmin=20, fmax=16000)   # audio_utils.MEL_KW (the model's input)
OCTAVE_EDGES_HZ = (22.0, 44.0, 88.0, 177.0, 355.0, 710.0, 1420.0, 2840.0, 5680.0, 11360.0)
SM_BANDS_HZ = ((20, 60), (60, 250), (250, 500), (500, 2000), (2000, 4000), (4000, 6000), (6000, 10000),
               (10000, 16000), (16000, 20000))


# ------------------------------------------------------------------------------------------------------- helpers
def _mono(y):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=0) if y.shape[0] <= 2 else y.mean(axis=1)
    return y


def _pair(target, pred):
    t, p = _mono(target), _mono(pred)
    n = min(len(t), len(p))
    if n == 0:
        raise ValueError("empty clip")
    return t[:n], p[:n]


def logmel(y, sr=SR):
    """The model's input mel without the 0.8 s length check: dB re the clip maximum, floor -80 dB, mapped to [-1, 1]
    ((dB + 80) / 40 - 1). One unit = 40 dB."""
    import librosa
    spec = librosa.feature.melspectrogram(y=_mono(y), sr=sr, **MEL_KW)
    db = librosa.power_to_db(spec, ref=np.max)
    return np.clip((db + 80.0) / 40.0 - 1.0, -1.0, 1.0).astype(np.float32)


def ssim_map(x, y, win=7, data_range=1.0, k1=0.01, k2=0.03):
    """SSIM map with skimage.metrics.structural_similarity's defaults (uniform win x win window, sample covariance,
    reflect padding). Returns the full map; average the interior (crop (win - 1) // 2) for skimage's number."""
    from scipy.ndimage import uniform_filter
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
    if min(x.shape) < win:
        raise ValueError(f"input {x.shape} smaller than the {win} x {win} window")
    npix = win ** x.ndim
    cov_norm = npix / (npix - 1.0)
    ux, uy = uniform_filter(x, win), uniform_filter(y, win)
    uxx, uyy, uxy = uniform_filter(x * x, win), uniform_filter(y * y, win), uniform_filter(x * y, win)
    vx, vy, vxy = cov_norm * (uxx - ux * ux), cov_norm * (uyy - uy * uy), cov_norm * (uxy - ux * uy)
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    return ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))


def _crop(a, win):
    p = (win - 1) // 2
    return a[p:a.shape[0] - p, p:a.shape[1] - p]


# ------------------------------------------------------------------------------------------------------ ours
def logmel_l1(target, pred, sr=SR):
    t, p = _pair(target, pred)
    return float(np.mean(np.abs(logmel(t, sr) - logmel(p, sr))))


def mel_ssim_dist(target, pred, sr=SR, win=7):
    """1 - SSIM between the two model mels (scaled to [0, 1], data_range 1), interior mean as in skimage."""
    t, p = _pair(target, pred)
    mt, mp = (logmel(t, sr) + 1.0) / 2.0, (logmel(p, sr) + 1.0) / 2.0
    return float(1.0 - _crop(ssim_map(mt, mp, win=win), win).mean())


def mel_ssim_gated_dist(target, pred, sr=SR, win=7, gate_db=40.0):
    """1 - SSIM averaged only over interior pixels where either clip is within `gate_db` of its own maximum.
    NaN when no pixel passes (both clips silent)."""
    t, p = _pair(target, pred)
    lt, lp = logmel(t, sr), logmel(p, sr)
    thr = (80.0 - gate_db) / 40.0 - 1.0
    s = _crop(ssim_map((lt + 1.0) / 2.0, (lp + 1.0) / 2.0, win=win), win)
    active = _crop((lt > thr) | (lp > thr), win)
    if not active.any():
        return float("nan")
    return float(1.0 - s[active].mean())


def _octave_profile_db(y, sr=SR, floor_db=-60.0):
    import librosa
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=4096)
    e = np.array([S[(f >= lo) & (f < hi)].sum() for lo, hi in zip(OCTAVE_EDGES_HZ[:-1], OCTAVE_EDGES_HZ[1:])])
    db = 10.0 * np.log10(e / max(e.max(), 1e-20) + 1e-20)
    return np.maximum(db, floor_db)


def octave_db_l1(target, pred, sr=SR):
    """Mean |dB difference| of the octave-band energy profiles (31.5 Hz .. 8 kHz centres, each re its loudest band,
    floor -60 dB), in dB. A spectral-tilt diagnostic with resolution where bass energy sits."""
    t, p = _pair(target, pred)
    return float(np.mean(np.abs(_octave_profile_db(t, sr) - _octave_profile_db(p, sr))))


def _mod_matrix(y, sr=SR, n_bands=32, hop=256, n_mod=20):
    import librosa
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=hop, n_mels=n_bands, fmin=20, fmax=8000)
    env = np.sqrt(mel)                                             # amplitude envelope per band
    env = (env - env.mean(axis=1, keepdims=True)) * np.hanning(env.shape[1])[None, :]
    m = np.abs(np.fft.rfft(env, axis=1))[:, 1:n_mod + 1]           # drop DC; bins are sr / hop / n_frames apart
    return m / (np.linalg.norm(m) + 1e-10)


def modspec_dist(target, pred, sr=SR):
    """Euclidean distance between L2-normalised modulation spectra (32 mel bands, 1..20 modulation bins; at 0.8 s and
    a 256-sample hop a bin is ~1.25 Hz, so ~1.25..25 Hz). Magnitude only, so it tolerates oscillator phase once a beat
    completes a few cycles in the clip: on a two-voice saw at 110 Hz it separates 40 from 80 cents ~3x better than a
    re-render does, where log-mel L1 cannot separate them at all (test_v5.py). Beats slower than ~2 Hz (small detune
    at low notes) stay phase-dependent in 0.8 s."""
    t, p = _pair(target, pred)
    return float(np.linalg.norm(_mod_matrix(t, sr) - _mod_matrix(p, sr)))


def _flatness_frames(y, n_fft=2048, hop=512):
    import librosa
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    return librosa.feature.spectral_flatness(S=S)[0], S.sum(axis=0)


def flatness_gated_err(target, pred, sr=SR, gate_db=30.0):
    """|mean spectral flatness| difference over frames within `gate_db` of each clip's loudest frame."""
    t, p = _pair(target, pred)

    def gated(y):
        fl, e = _flatness_frames(y)
        keep = e >= e.max() * 10.0 ** (-gate_db / 10.0)
        return float(fl[keep].mean()) if keep.any() else float("nan")
    return float(abs(gated(t) - gated(p)))


# ----------------------------------------------------------------------------------------------------- paper
def mss(target, pred, sr=SR):
    import reference_metrics as rm
    return rm.mss(target, pred, sr)


def wmfcc(target, pred, sr=SR):
    import reference_metrics as rm
    return rm.wmfcc(target, pred, sr)


def env_dist(target, pred, sr=SR):
    import reference_metrics as rm
    return float(1.0 - rm.rms_env_cos(target, pred, sr))


# ----------------------------------------------------------------------------------------------- sonicmaster
def _sm_mel_db(y, sr=SR):
    import librosa
    return librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128),
                               ref=np.max)


def sm_mel_kl(target, pred, sr=SR):
    """SonicMaster's "KL on mel": KL(pred || target) over the time-mean of the dB mel normalised to sum 1. The dB values
    are <= 0, so after p / sum(p) the QUIETEST bins carry the most mass (reproduced, not fixed)."""
    t, p = _pair(target, pred)
    a, b = np.mean(_sm_mel_db(p, sr), axis=1), np.mean(_sm_mel_db(t, sr), axis=1)
    a, b = a / np.sum(a), b / np.sum(b)
    return float(np.sum(a * np.log((a + 1e-10) / (b + 1e-10))))


def sm_ssim_dist(target, pred, sr=SR):
    """SonicMaster's mel SSIM: dB mel (n_fft 2048, hop 512), min-max scaled per clip, 7 x 7 window; as 1 - SSIM."""
    t, p = _pair(target, pred)

    def mm(m):
        return (m - m.min()) / (m.max() - m.min() + 1e-12)
    a, b = mm(_sm_mel_db(t, sr)), mm(_sm_mel_db(p, sr))
    n = min(a.shape[1], b.shape[1])
    return float(1.0 - _crop(ssim_map(a[:, :n], b[:, :n], win=7), 7).mean())


def _sm_band_profile(y, sr=SR):
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    e = np.array([S[(f >= lo) & (f < hi)].sum() for lo, hi in SM_BANDS_HZ])
    return e / (e.sum() + 1e-10)


def sm_band9_cos(target, pred, sr=SR):
    """SonicMaster's 9-band spectral balance, cosine distance of LINEAR energy shares (time-averaged)."""
    from scipy.spatial.distance import cosine
    t, p = _pair(target, pred)
    return float(cosine(_sm_band_profile(t, sr), _sm_band_profile(p, sr)))


def sm_band9_euc(target, pred, sr=SR):
    t, p = _pair(target, pred)
    return float(np.linalg.norm(_sm_band_profile(t, sr) - _sm_band_profile(p, sr)))


def sm_punch_err(target, pred, sr=SR):
    """SonicMaster's "transient punch": |difference of mean librosa onset strength| (librosa defaults)."""
    import librosa
    t, p = _pair(target, pred)
    return float(abs(np.mean(librosa.onset.onset_strength(y=t, sr=sr)) -
                     np.mean(librosa.onset.onset_strength(y=p, sr=sr))))


def sm_dr_err(target, pred, sr=SR):
    """SonicMaster's dynamic range: |difference of the std of frame RMS| (2048-sample frames, 1024 hop, no padding)."""
    import librosa
    t, p = _pair(target, pred)

    def dr(y):
        fr = librosa.util.frame(y, frame_length=2048, hop_length=1024)
        return float(np.std(np.sqrt(np.mean(fr ** 2, axis=0))))
    return float(abs(dr(t) - dr(p)))


def sm_flatness_err(target, pred, sr=SR):
    """SonicMaster's spectral flatness error: |difference of the mean over ALL frames| (silence counts, flatness ~1)."""
    t, p = _pair(target, pred)
    return float(abs(_flatness_frames(t)[0].mean() - _flatness_frames(p)[0].mean()))


def sm_modspec_dist(target, pred, sr=SR):
    """SonicMaster's modulation spectrum distance: scipy STFT (nperseg 1024, 512 overlap), per-bin envelope minus its
    mean, |FFT| over time, first 20 bins (DC included), whole matrix L2-normalised, Euclidean distance."""
    from scipy.signal import stft
    t, p = _pair(target, pred)

    def mod(y):
        _, _, Z = stft(y, fs=sr, nperseg=1024, noverlap=512)
        env = np.abs(Z)
        env = env - env.mean(axis=1, keepdims=True)
        m = np.abs(np.fft.fft(env, axis=1))[:, :20]
        return m / (np.linalg.norm(m) + 1e-10)
    a, b = mod(t), mod(p)
    n = min(a.shape[1], b.shape[1])
    return float(np.linalg.norm(a[:, :n] - b[:, :n]))


# ------------------------------------------------------------------------------------------------------ registry
METRICS = {
    # name: (function, family, what it is for)
    "logmel_l1": (logmel_l1, "ours", "primary; best-of-k selector"),
    "mss": (mss, "paper", "primary; paper metric"),
    "wmfcc": (wmfcc, "paper", "secondary; DTW-tolerant timbre"),
    "env_dist": (env_dist, "paper", "secondary; amplitude envelope (1 - rms_env_cos)"),
    "mel_ssim_dist": (mel_ssim_dist, "ours", "candidate; structure on the model mel"),
    "mel_ssim_gated_dist": (mel_ssim_gated_dist, "ours", "candidate; structure on non-silent pixels only"),
    "octave_db_l1": (octave_db_l1, "ours", "diagnostic; spectral tilt (cutoff, shape)"),
    "modspec_dist": (modspec_dist, "ours", "diagnostic; beating and envelope rates (detune)"),
    "flatness_gated_err": (flatness_gated_err, "ours", "diagnostic; noisiness (drive, noise, FM)"),
    "sm_mel_kl": (sm_mel_kl, "sonicmaster", "reference port; quiet bins dominate"),
    "sm_ssim_dist": (sm_ssim_dist, "sonicmaster", "reference port"),
    "sm_band9_cos": (sm_band9_cos, "sonicmaster", "reference port"),
    "sm_band9_euc": (sm_band9_euc, "sonicmaster", "reference port"),
    "sm_punch_err": (sm_punch_err, "sonicmaster", "reference port"),
    "sm_dr_err": (sm_dr_err, "sonicmaster", "reference port"),
    "sm_flatness_err": (sm_flatness_err, "sonicmaster", "reference port; silence counts"),
    "sm_modspec_dist": (sm_modspec_dist, "sonicmaster", "reference port"),
}
DEFAULT_EVAL = ("logmel_l1", "mss", "wmfcc", "env_dist", "mel_ssim_dist", "mel_ssim_gated_dist",
                "octave_db_l1", "modspec_dist", "flatness_gated_err")


def score_pair(target, pred, names=DEFAULT_EVAL, sr=SR):
    return {n: METRICS[n][0](target, pred, sr) for n in names}


def _score_chunk(args):
    targets, preds, names, sr = args
    return [score_pair(t, p, names, sr) for t, p in zip(targets, preds)]


def score_many(targets, preds, names=DEFAULT_EVAL, sr=SR, workers=0):
    """targets, preds: sequences of clips (same length). -> {name: np.array [n]} (NaN where a metric is undefined)."""
    if len(targets) != len(preds):
        raise ValueError("targets and preds differ in length")
    names = tuple(names)
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise KeyError(f"unknown metrics {unknown}; known: {sorted(METRICS)}")
    if workers and workers > 1 and len(targets) > 1:
        import multiprocessing as mp
        k = int(np.ceil(len(targets) / workers))
        jobs = [(targets[i:i + k], preds[i:i + k], names, sr) for i in range(0, len(targets), k)]
        with mp.get_context("spawn").Pool(workers) as pool:
            rows = [r for part in pool.map(_score_chunk, jobs) for r in part]
    else:
        rows = _score_chunk((targets, preds, names, sr))
    return {n: np.array([r[n] for r in rows], dtype=np.float64) for n in names}


# ---------------------------------------------------------------------------------------------------------- CLAP
def clap_distances(targets, preds, ckpt, fusion=True, sr=SR, device=None, batch=32):
    """1 - cosine similarity of LAION-CLAP audio embeddings (audio-to-audio; stable-audio-metrics only offers
    text-to-audio). laion_clap repeats a short clip to fill its 10 s window. Needs `pip install laion_clap` and a
    checkpoint file (e.g. 630k-audioset-fusion-best.pt for fusion=True). Returns np.array [n]."""
    import librosa
    import torch
    try:
        import laion_clap
    except ImportError as e:
        raise SystemExit("clap_distances needs laion_clap (pip install laion_clap)") from e
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = laion_clap.CLAP_Module(enable_fusion=fusion, device=device)
    model.load_ckpt(ckpt)

    def prep(c):
        x = librosa.resample(_mono(c), orig_sr=sr, target_sr=48000)
        return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).astype(np.float32) / 32767.0  # int16 round trip, as SAM

    def emb(clips):
        x = [prep(c) for c in clips]
        out = [None] * len(x)
        by_len = {}
        for i, c in enumerate(x):                       # batch equal lengths only: padding would change CLAP's
            by_len.setdefault(len(c), []).append(i)     # own repeat-to-10-s behaviour
        for idx in by_len.values():
            for s in range(0, len(idx), batch):
                sel = idx[s:s + batch]
                with torch.no_grad():
                    e = model.get_audio_embedding_from_data(
                        x=torch.from_numpy(np.stack([x[i] for i in sel])).to(device), use_tensor=True)
                e = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
                for j, i in enumerate(sel):
                    out[i] = e[j]
        return np.stack(out)
    a, b = emb(list(targets)), emb(list(preds))
    return 1.0 - np.sum(a * b, axis=1)
