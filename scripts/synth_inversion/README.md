# Surge XT Neural Synth Inversion Toolchain

This package implements reverse-engineering and neural inversion of synthesizer patches from isolated audio stems and notes into **Surge XT 1.3.4 VST3** presets.

---

## 1. Architecture & Capabilities

- **23-Dimensional Sound Design Space**:
  - **Oscillator**: Shape (Saw <-> Morph <-> Pulse), Pulse Width, Sub-oscillator mix, Hard Sync, FM/PM Depth.
  - **Filter Topology**: 10 Lowpass circuits (`LP 12dB`, `LP 24dB`, `Legacy Ladder`, `Vintage Ladder`, `OB-Xd 12dB`, `OB-Xd 24dB`, `K35`, `Diode Ladder 303`, `Cutoff Warp`, `Res Warp`), Master Cutoff, Resonance, Keyboard Tracking.
  - **Envelopes**: Filter EG depth, decay, sustain; Amp EG decay, sustain, release. Domain-specific physical coupling: FEG_release >= AMP_release.
  - **Distortion & Drive**: 6 Analog waveshaper types (`Off`, `Soft`, `Hard`, `Asymmetric`, `Sine`, `Fuzz`) with variable drive.
  - **Time & Spatial FX**: Chorus wet mix, ping-pong delay wet mix & feedback.

- **Models**:
  - `ResMLPInverter` (Deep ResMLP with residual linear blocks, 512-dim, 6 layers). Instantaneous inference (<2 ms).
  - `FlowMatchingResMLP` (Conditional Flow Matching optimal transport ODE velocity predictor, 512-dim, 6 layers).

- **Optimizers & Stability Safeguards**:
  - **NorMuon** (2D weight matrices with Newton-Schulz quintic orthogonalization and **Radial Brake** soft-limiting).
  - **Schedule-Free AdamW** (1D biases, norms, embeddings with 200-step burn-in safeguard and dynamic test/train iterate swap).
  - **Decoupled AdamW Baseline** (with cosine annealing).

---

## 2. Directory Layout & Scripts

```
scripts/synth_inversion/
|-- generate_bass_dataset.py   # Multi-threaded (Pedalboard VST3) 200k dataset generator
|-- models.py                  # PyTorch ResMLP & Conditional Flow Matching models
|-- optimizer.py               # NorMuon + Schedule-Free optimizer implementations
|-- train_bracket.py           # Training harness with domain loss weighting & safeguards
|-- evaluate_200k_inversion.py # Real-stem evaluation and VST3 preset exporter
|-- match_untitled_note.py     # Single-note closed-loop Differential Evolution & neural match
`-- run_overnight_suite.sh     # 200-epoch SF + 100-epoch AdamW runner
```
