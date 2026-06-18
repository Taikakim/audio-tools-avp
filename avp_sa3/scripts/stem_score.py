"""
stem_score.py — ground-truth scoring for SA3 separation/riffer param bracketing.

You have per-generator project stems. Sum the relevant ones into a role submix
(drums = kick+snare+perc, bass = desertbass+bb, lead = flangerlead+acid...), then
rank a sweep of candidate outputs by how close they are to that target role. Use it
to pick params (anchor_eta, eta, cfg, n_max) against GROUND TRUTH instead of the
env-corr-to-mix proxy.

Generative outputs are RE-SYNTHESIZED, not phase-locked to the input, so we lead
with alignment-tolerant metrics:
  - logmel_l1  : L1 of log-mel spectrograms (perceptual closeness; LOWER = better)
  - lsd        : log-spectral distance (LOWER = better)
  - env_corr   : onset-envelope correlation (rhythm/timing; HIGHER = better)
  - centroid   : spectral centroid of candidate vs reference (timbre sanity)
SI-SDR is reported (best-lag aligned) only as a loose reference — it stays low even
for good generative matches (ZeroSep itself uses FAD/CLAP, not SDR, for this reason).

Run with the mir venv (has librosa):
    MIR=/home/kim/Projects/mir/mir/bin/python

Build a role reference from stems, then score a candidate sweep:
    $MIR stem_score.py --role drums --stems-dir "Stems/Acid Alien 136" \
        --ref-start 400 --ref-sec 10 --ref-out drums_ref.wav
    $MIR stem_score.py --reference drums_ref.wav \
        --candidates flowsep_results/<run>/*drum*aeta*.wav --sort-by logmel_l1
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import soundfile as sf

# Name->role patterns for auto-summing project stems. Synth-VST names (Tritium,
# GakStoar, V-Station...) are ambiguous -> leave to --sum or your renamed stems.
ROLE_PATTERNS = {
    "drums": r"kick|snare|perc|\bhat|\bhh\b|clap|\btom|ride|crash|\bdrum|cymbal",
    "bass": r"\bbass|\bbb\b|\bsub\b|desertbass|reese|\b303\b",
    "lead": r"\blead|\bacid|\barp|pluck|\bseq\b|flanger",
    "pad": r"\bpad|string|drone|atmos|\bfx\b|delay|reverb",
}
AUDIO_EXT = (".flac", ".wav", ".aiff", ".aif", ".mp3", ".ogg")


def load_stereo(path, sr, start=0.0, sec=None):
    y, s = sf.read(path, always_2d=True)
    y = y.T.astype(np.float64)                         # (C, T)
    if s != sr:
        import librosa
        y = librosa.resample(y, orig_sr=s, target_sr=sr)
    a = int(start * sr)
    b = a + int(sec * sr) if sec else y.shape[-1]
    return y[:, a:b]


def to_stereo(y):
    return np.vstack([y, y]) if y.shape[0] == 1 else y[:2]


def build_reference(files, sr, start, sec, out):
    chans = [to_stereo(load_stereo(p, sr, start, sec)) for p in files]
    T = max(y.shape[1] for y in chans)
    mix = np.zeros((2, T))
    for y in chans:
        mix[:, :y.shape[1]] += y
    peak = float(np.abs(mix).max())
    if peak > 0:
        mix = mix / peak * 0.99
    sf.write(out, mix.T, sr)
    return mix.mean(0)                                  # mono for scoring


# ── metrics (mono float64) ───────────────────────────────────────────────────

def _stft_mag(y, n_fft=2048, hop=512):
    import librosa
    return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))


def lsd(est, ref):
    Se, Sr = np.log(_stft_mag(est) + 1e-6), np.log(_stft_mag(ref) + 1e-6)
    m = min(Se.shape[1], Sr.shape[1])
    return float(np.mean(np.sqrt(np.mean((Se[:, :m] - Sr[:, :m]) ** 2, axis=0))))


def logmel_l1(est, ref, sr):
    import librosa
    Me = np.log(librosa.feature.melspectrogram(y=est, sr=sr, n_mels=64) + 1e-6)
    Mr = np.log(librosa.feature.melspectrogram(y=ref, sr=sr, n_mels=64) + 1e-6)
    m = min(Me.shape[1], Mr.shape[1])
    return float(np.mean(np.abs(Me[:, :m] - Mr[:, :m])))


def env_corr(est, ref, sr):
    import librosa
    ee = librosa.onset.onset_strength(y=est, sr=sr)
    er = librosa.onset.onset_strength(y=ref, sr=sr)
    m = min(len(ee), len(er))
    return float(np.corrcoef(ee[:m], er[:m])[0, 1]) if m > 1 else float("nan")


def centroid(y, sr):
    import librosa
    return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))


def si_sdr(est, ref):
    n = min(len(est), len(ref))
    est, ref = est[:n].copy(), ref[:n].copy()
    if n > 1:                                            # best-lag align (not phase-locked)
        lag = int(np.argmax(np.correlate(est, ref, mode="full")) - (n - 1))
        if lag > 0:
            est = np.r_[np.zeros(lag), est][:n]
        elif lag < 0:
            est = np.r_[est[-lag:], np.zeros(-lag)][:n]
    proj = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12) * ref
    noise = est - proj
    return float(10 * np.log10((np.sum(proj ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12)))


def score(cand, ref, sr):
    return {
        "logmel_l1": round(logmel_l1(cand, ref, sr), 3),
        "lsd": round(lsd(cand, ref), 3),
        "env_corr": round(env_corr(cand, ref, sr), 3),
        "si_sdr_db": round(si_sdr(cand, ref), 2),
        "cen_cand": round(centroid(cand, sr)),
        "cen_ref": round(centroid(ref, sr)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--reference", help="reference role submix wav (or build with --sum/--role)")
    ap.add_argument("--candidates", nargs="*", default=[], help="candidate wavs (globs ok)")
    ap.add_argument("--sum", nargs="*", default=[], help="explicit stem files to sum into a reference")
    ap.add_argument("--role", choices=list(ROLE_PATTERNS), help="auto-pick stems matching this role")
    ap.add_argument("--stems-dir", help="folder of project stems (used with --role)")
    ap.add_argument("--ref-out", default="reference.wav")
    ap.add_argument("--ref-start", type=float, default=0.0)
    ap.add_argument("--ref-sec", type=float, default=None)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--sort-by", default="logmel_l1",
                    choices=["logmel_l1", "lsd", "env_corr", "si_sdr_db"])
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    # 1. Resolve / build the reference.
    files = list(args.sum)
    if args.role and args.stems_dir:
        pat = re.compile(ROLE_PATTERNS[args.role], re.I)
        files += [p for p in glob.glob(os.path.join(args.stems_dir, "*"))
                  if p.lower().endswith(AUDIO_EXT) and pat.search(os.path.basename(p))]
    if files:
        print(f"building reference from {len(files)} stems -> {args.ref_out}")
        for p in files:
            print("   +", os.path.basename(p))
        ref = build_reference(files, args.sr, args.ref_start, args.ref_sec, args.ref_out)
        ref_path = args.ref_out
    elif args.reference:
        y = load_stereo(args.reference, args.sr, args.ref_start, args.ref_sec)
        ref = y.mean(0)
        ref_path = args.reference
    else:
        ap.error("need --reference, or --sum/--role+--stems-dir to build one")

    # 2. Score candidates (full window; candidates assumed already the right length).
    cand_paths = []
    for c in args.candidates:
        cand_paths += sorted(glob.glob(c)) if any(ch in c for ch in "*?[") else [c]
    if not cand_paths:
        print(f"\nreference built: {ref_path} ({len(ref)/args.sr:.1f}s). No --candidates to score.")
        return

    rows = []
    for p in cand_paths:
        cand = load_stereo(p, args.sr).mean(0)
        rows.append({"file": os.path.basename(p), **score(cand, ref, args.sr)})

    reverse = args.sort_by in ("env_corr", "si_sdr_db")   # higher is better
    rows.sort(key=lambda r: r[args.sort_by], reverse=reverse)

    print(f"\nreference: {os.path.basename(ref_path)}   (ranked by {args.sort_by}, best first)\n")
    hdr = ["logmel_l1", "lsd", "env_corr", "si_sdr_db", "cen_cand"]
    print(f"{'candidate':52s} " + " ".join(f"{h:>9s}" for h in hdr))
    for r in rows:
        print(f"{r['file'][:52]:52s} " + " ".join(f"{r[h]:>9}" for h in hdr))
    print(f"\n(reference centroid {rows[0]['cen_ref']} Hz)")

    if args.out_json:
        json.dump({"reference": ref_path, "rows": rows}, open(args.out_json, "w"), indent=2)
        print(f"-> {args.out_json}")


if __name__ == "__main__":
    main()
