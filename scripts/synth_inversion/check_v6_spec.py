# Owner: KUANG (created 2026-10-19)
"""Desktop check of the two v6 spec assumptions that only Surge can confirm (surge_spec_v6 docstring):
  1. osc 2 fine pitch: stored 0 / 1 -> -0.2 / +0.2 semitones relative to osc 1 (relies on surge_spec_v5's
     +-1 semitone mapping, raw = 0.5 + (p - 0.5) * 2 / 14);
  2. relative octaves: osc 2 at -2..+2 sounds 2^rel times osc 1's pitch, and osc 1 sounds at the MIDI note.
Renders one oscillator at a time (others muted, filter open, no waveshaper / unison / FX) on a sustained note and
measures the frequency of a harmonic peak with parabolic interpolation.

  python check_v6_spec.py --plugin "~/.vst3/Surge XT.vst3" --note 45
Prints measured vs expected and exits 1 if any check is off by more than --tol_cents (default 3 cents).

--shape_scan (V6.1): renders osc 1 alone at shape 0, 0.125, ..., 1 (width 0.5) and prints the harmonic parity
(even / odd harmonic power; ~0 = odd harmonics only), the harmonic richness and the odd-harmonic slope. KUANG
2026-10-20: parity ~0 does not mean square: a triangle is odd-only too. Antigravity's scan found parity 0 at shape 0
and 0.289 at 0.5 (a saw's value), which fits Surge's classic oscillator going triangle (-100 %) -> saw (0 %) ->
pulse (+100 %), i.e. shape 0 a triangle and the square at 1.0. The slope tells them apart (square amplitudes 1/k:
slope ~-2 in log power; triangle 1/k^2: ~-4). The scan picks the odd-only shape with a square's slope and prints
"use --square_shape X" only if there is one (render_pairs_dpo_v6.py / eval_real_stems_consensus_v6.py take it).
"""
import argparse
import sys

import numpy as np

import surge_spec_v6 as S

SR = S.SAMPLE_RATE


def peak_hz(y, sr, f_expect, rel_window=0.06):
    """Frequency of the largest spectral peak within +-rel_window of f_expect (Hann window, parabolic interpolation)."""
    y = np.asarray(y, np.float64)
    w = np.hanning(len(y))
    spec = np.abs(np.fft.rfft(y * w))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    band = np.where((freqs > f_expect * (1 - rel_window)) & (freqs < f_expect * (1 + rel_window)))[0]
    k = band[np.argmax(spec[band])]
    a, b, c = np.log(spec[k - 1] + 1e-12), np.log(spec[k] + 1e-12), np.log(spec[k + 1] + 1e-12)
    off = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return float((k + off) * sr / len(y))


def cents(f, f_ref):
    return float(1200.0 * np.log2(f / f_ref))


def base_vector(note):
    I = S.PARAM_INDEX
    v = np.full(S.NUM_PARAMS, 0.5, np.float32)
    v[I["midi_note"]] = (note - S.NOTE_LOW) / (S.NOTE_HIGH - S.NOTE_LOW)
    for n in ("waveshaper_type", "drive", "unison", "unison_detune", "sub_mix", "sync", "fm_depth", "fm_routing",
              "ring_1x2_volume", "ring_2x3_volume", "noise_volume", "drift", "chorus_mix", "delay_mix", "delay_fb",
              "filter_configuration", "resonance", "feg_amount", "filter_2_resonance", "osc_3_volume"):
        v[I[n]] = 0.0
    v[I["shape"]] = 0.5                                        # a saw (all harmonics; shape 0 has no even ones)
    v[I["filter_feedback"]] = 0.5
    v[I["cutoff"]] = v[I["filter_2_cutoff"]] = 0.95
    v[I["aeg_sustain"]] = 0.85
    v[I["osc_2_pitch"]] = v[I["osc_3_pitch"]] = 0.5
    v[I["osc_2_octave"]] = v[I["osc_3_octave"]] = 0.5
    return v


def render_sustained(plugin, vec, note, seconds=2.0):
    import mido
    S.flush_synth(plugin)
    S.apply_patch(plugin, S.vector_to_patch(vec))
    plugin.reset()
    ev = [mido.Message("note_on", note=int(note), velocity=105, time=0.0),
          mido.Message("note_off", note=int(note), velocity=0, time=seconds)]
    out = plugin.process(ev, duration=seconds, sample_rate=SR, num_channels=2)
    y = np.mean(out, axis=0)
    return y[int(0.25 * SR):int((seconds - 0.05) * SR)]        # skip the attack


def run(render, note, harmonic=7, tol=3.0, log=print):
    """render(vec) -> mono audio. -> list of (name, measured cents, expected cents, ok)."""
    I = S.PARAM_INDEX
    f0 = 440.0 * 2 ** ((note - 69) / 12)
    solo1 = base_vector(note)
    solo1[I["osc_1_volume"]], solo1[I["osc_2_volume"]] = 1.0, 0.0
    f1 = peak_hz(render(solo1), SR, harmonic * f0)
    rows = [("osc 1 at the MIDI note", cents(f1, harmonic * f0), 0.0)]
    for p, want in ((0.0, -20.0), (1.0, 20.0)):
        v = base_vector(note)
        v[I["osc_1_volume"]], v[I["osc_2_volume"]], v[I["osc_2_pitch"]] = 0.0, 1.0, p
        rows.append((f"osc 2 fine pitch stored {p:.0f}", cents(peak_hz(render(v), SR, harmonic * f0), f1), want))
    for rel in (-2, -1, 1, 2):
        v = base_vector(note)
        v[I["osc_1_volume"]], v[I["osc_2_volume"]], v[I["osc_2_octave"]] = 0.0, 1.0, (rel + 2) / 4.0
        rows.append((f"osc 2 relative octave {rel:+d}", cents(peak_hz(render(v), SR, harmonic * f0 * 2.0 ** rel), f1),
                     1200.0 * rel))
    out = []
    for name, got, want in rows:
        ok = abs(got - want) <= tol
        out.append((name, got, want, ok))
        log(f"{name:28s} measured {got:+9.2f} cents | expected {want:+9.2f} | {'ok' if ok else 'OFF'}")
    return out


def classify_wave(parity, slope):
    """odd-only with a 1/k spectrum -> "square"; odd-only steeper (1/k^2) -> "triangle"; else "saw/mixed"."""
    if not parity < 0.12:
        return "saw/mixed"
    return "square" if slope > -3.0 else "triangle"


def shape_scan(render, note, shapes=tuple(np.linspace(0.0, 1.0, 9)), log=print):
    """-> [(shape, parity, odd slope, label)] for osc 1 alone, width 0.5, and the square's shape (or None)."""
    import audio_metrics_v6 as am6
    I = S.PARAM_INDEX
    f0 = 440.0 * 2 ** ((note - 69) / 12)
    out = []
    for sh in shapes:
        v = base_vector(note)
        v[I["osc_1_volume"]], v[I["osc_2_volume"]] = 1.0, 0.0
        v[I["shape"]], v[I["width"]] = float(sh), 0.5
        y = render(v)
        par, slope = am6.compute_harmonic_parity(y, f0, SR), am6.odd_harmonic_slope(y, f0, SR)
        lab = classify_wave(par, slope)
        out.append((float(sh), float(par), float(slope), lab))
        log(f"shape {sh:5.3f}: parity {par:.4f} | odd slope {slope:+.2f} | {lab}")
    squares = [r for r in out if r[3] == "square"]
    if squares:
        best = min(squares, key=lambda r: r[1])
        log(f"symmetric square at shape {best[0]:.3f} -> use --square_shape {best[0]:.3f}")
        return out, best[0]
    log("no shape gives an odd-only 1/k spectrum at width 0.5: no --square_shape (the parity rules stay off)")
    return out, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plugin", default=S.DEFAULT_PLUGIN_PATH)
    ap.add_argument("--note", type=int, default=45)
    ap.add_argument("--harmonic", type=int, default=8, help="measure this harmonic (finer cents resolution)")
    ap.add_argument("--tol_cents", type=float, default=3.0)
    ap.add_argument("--shape_scan", action="store_true", help="also report parity against osc shape (V6.1)")
    a = ap.parse_args(argv)
    plugin = S.init_synth(a.plugin)
    res = run(lambda v: render_sustained(plugin, v, a.note), a.note, a.harmonic, a.tol_cents)
    if a.shape_scan:
        shape_scan(lambda v: render_sustained(plugin, v, a.note), a.note)
    return 0 if all(r[3] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
