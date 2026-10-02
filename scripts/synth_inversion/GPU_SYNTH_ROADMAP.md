# Conceptual & Technical Roadmap: Native HIP/GPU Synthesizer Engine Proxying Surge XT

> [!NOTE]
> **Phase 0 Immediate Baseline (Online Surge on CPU + Exact GPU Mel)**:
> Before building a custom GPU synth kernel, online training can be driven directly by real headless Surge XT C++ instances (`SurgeOnlineDataset`, seeds $\ge 10^6$ to prevent validation leakage) paired with `ExactGpuMel` (bit-exact Librosa Slaney Mel spectrograms executed on GPU in $8.3\ \mu\text{s}$ per note). Because 16–20 CPU workers on an AMD 9900X produce $\sim 1,000$ real Surge notes/sec—exceeding the GPU's training consumption of $\sim 533$ notes/sec—online training can run with 100% genuine Surge audio today without proxy gap errors. The GPU synth below remains the ultimate scaling path for $100\text{k}+$ sounds/sec renderer-in-the-loop search.

**Target Hardware**: AMD Radeon RX 9070 XT (`gfx1201` via ROCm / HIP)  
**Target DAW Environment**: Surge XT 1.3+ VST3 / CLAP across Bitwig, Ableton, FL Studio, Reaper  
**Goal**: Build a GPU-native, massively parallel synthesizer proxy in HIP/PyTorch that matches Surge XT's core DSP topology, renders 100k+ sounds/second directly in VRAM for neural model training and search, and exports 1:1 identical `.vstpreset` files for real-world DAW workflows.

---

## 1. Problem Statement & Motivation

1. **The Inversion Problem**: Inverse synthesis models (like Synth-JEPA, Flow Matching, and ResMLP) need either massive pre-rendered datasets or online audio generation.
2. **The CPU Bottleneck**: Surge XT's native C++ codebase, while fast on CPU SIMD (AVX/SSE), is bound by CPU core limits, Python GIL context switching, and PCIe host-to-device memory transfers.
3. **The Solution**: Clone the essential sound-producing DSP architecture of Surge XT as a pure **SIMT GPU kernel (HIP C++ / PyTorch ROCm)**. 
   - Render entire batches (thousands of voices) in parallel in GPU registers and VRAM.
   - Use the GPU engine for limitless online training and fast renderer-free / renderer-in-the-loop search.
   - Once optimal parameters are found, emit a standard Surge XT `.vstpreset` container for the music producer.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           GPU Inversion Engine                               │
│                                                                              │
│   Parameter Space [B, 23]                                                    │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │                 Native HIP / GPU Voice Kernel                        │   │
│   │                                                                      │   │
│   │  [PolyBLEP Osc + Sub + FM] ──► [Waveshaper] ──► [4-Pole SVF/Ladder]  │   │
│   │               ▲                                        ▲             │   │
│   │               │                                        │             │   │
│   │          [Amp ADSR]                               [Filter ADSR]      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│          │                                                                   │
│          ▼                                                                   │
│   Audio Buffer [B, 35280] in VRAM                                            │
│          │                                                                   │
│          ▼                                                                   │
│   [torch.stft] ──► Log-Mel Spectrograms [B, 128, 81]                         │
│          │                                                                   │
│          ▼                                                                   │
│   Model Training / Loss (Synth-JEPA / SIGReg / Flow Matching)                │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       │ Target Sound Inverted
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         DAW Interoperability Layer                           │
│                                                                              │
│   Canonical 23-d Parameter Vector                                            │
│          │                                                                   │
│          ▼                                                                   │
│   [vstpreset / Surge XML Exporter]                                           │
│          │                                                                   │
│          ▼                                                                   │
│   `~/Documents/Surge XT/Patches/AI Inversions/<PatchName>.vstpreset`         │
│          │                                                                   │
│          ▼                                                                   │
│   DAW Producer drags preset onto Surge XT (Bitwig, Ableton, FL Studio)       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Synthesis Modules to Clone (1:1 with Surge XT)

The target model is Surge XT's **Classic / Modern Bass & Lead topology** (defined across the 23-parameter space in `surge_spec.py`):

### 3.1. Oscillator Block
- **Primary Oscillator**: PolyBLEP band-limited continuous waveform:
  - Shape parameter morphs between: Sawtooth $\to$ Square/Pulse with variable pulse width.
  - PolyBLEP residual correction applied at discontinuities ($t=0, t=\text{width}$) to eliminate Nyquist aliasing.
- **Sub-Oscillator**: Hardwired 1 octave down (Square/Triangle) mixed via `sub_mix`.
- **Hard Sync**: Master-to-slave phase accumulator reset.
- **Linear Phase Modulation (FM)**: Modulator sine modulating primary oscillator phase with depth $D \in [0, 1]$.
- **Unison Engine**: 2-voice detuned stack with stereo phase spreading.

### 3.2. Filter Block
Surge XT's filters determine its timbral character. Two primary circuits must be implemented:
1. **Cascade 4-Pole Low-Pass Ladder (24 dB/oct)**:
   - 4-stage non-linear integrator with unit delay feedback:
     $$y_n = \tanh\left(x_n - 4k \cdot y_{4}[n-1]\right)$$
   - Internal thermal clipping saturation per stage.
2. **OB-Xd / State Variable Filter (SVF, 12 dB & 24 dB)**:
   - Differentiable Chamberlin / trapezoidal integrated SVF yielding simultaneous low-pass and band-pass outputs without high-frequency cramping.
3. **Surge Tuning Calibration**:
   - Surge cutoff is calibrated exponentially in semitones:
     $$f_c = 440.0 \times 2^{\frac{\text{cutoff\_pitch} - 69.0}{12.0}}$$

### 3.3. Envelope Generators (ADSR)
- **Filter Envelope (FEG)** & **Amp Envelope (AEG)**:
  - Analog-style curved envelopes:
    - Attack: Convex polynomial curve.
    - Decay & Release: Concave exponential decay ($e^{-t / \tau}$).
    - Sustain: Linear plateau ($0 \le S \le 1$).

### 3.4. Non-Linear Waveshaper / Drive
- Pre-filter or post-filter waveshaper modes:
  - **Soft**: Cubic saturator $f(x) = x - \frac{1}{3}x^3$ clamped to $[-1, 1]$.
  - **Hard / Asymmetric**: Diode-ladder style asymmetric transfer curve.
  - **Sine**: Phase folding $\sin(\pi \cdot x \cdot \text{drive})$.

---

## 4. Execution Phases & Milestones

### Phase 1: Mathematical Specification & Reference Extraction
- **Objective**: Extract exact DSP coefficient tables, saturation curves, and filter transfer functions from Surge XT's open-source C++ core (`src/common/dsp/`).
- **Deliverables**:
  - Exact transfer function equations documented for the 10 filter types and 6 waveshapers.
  - Envelope time constant lookup tables ($\tau_{\text{decay}}$ vs normalized knob value $[0, 1]$).

### Phase 2: PyTorch Vectorized Prototype (`surge_gpu_torch.py`)
- **Objective**: Implement the complete synth as a vectorized PyTorch module operating over batch dimension $B$ and time samples $T = 35280$.
- **Why**: Allows instant debugging, gradient checking, and unit testing on ROCm without C++ compilation delays.
- **Performance Expectation**: ~15,000–30,000 notes/sec on RX 9070 XT.

### Phase 3: Bit-Exact Spectral Calibration Suite
- **Objective**: Ensure the GPU proxy sounds indistinguishable from Surge XT.
- **Methodology**:
  - Sweep every parameter across its legal range (`surge_spec.CONT_BOUNDS`).
  - Render audio from both Surge XT C++ (via Pedalboard) and GPU Proxy.
  - Compute Multi-Scale STFT Loss (MSS) and warped MFCC (wMFCC):
    $$\mathcal{L}_{\text{MSS}}(y_{\text{surge}}, y_{\text{gpu}}) < \epsilon \quad (\text{target } \epsilon < 0.05)$$
  - Adjust resonance damping and envelope decay curvature until MSS convergence.

### Phase 4: Native Compiled HIP Kernel (`surge_synth_kernel.hip`)
- **Objective**: Maximize throughput by writing a single-pass fused SIMT kernel.
- **Kernel Layout**:
  - `blockDim.x = 256` threads; each thread synthesizes **one complete voice** (35,280 samples) sequentially in register space.
  - Avoids saving intermediate time buffers to global VRAM; only the final audio waveform or STFT frame is stored.
  - Zero host-to-device PCIe overhead.
- **Performance Expectation**: **100,000+ voices/second** on AMD RX 9070 XT.

### Phase 5: DAW Interoperability & Preset Generation
- **Objective**: Turn AI-inverted parameter vectors into fully editable DAW presets.
- **Deliverable**:
  - Automatic serialization into standard `.vstpreset` format (already implemented in `audio_utils.save_patch`).
  - Automatic installation into `~/Documents/Surge XT/Patches/AI Inversions/`.
  - Verification: User opens Surge XT inside Bitwig/Ableton $\to$ selects patch from browser $\to$ sound matches reference audio.

---

## 5. Technical Stack & Environment

- **Compiler**: `/opt/rocm/bin/hipcc` (ROCm 6.x)
- **Target GPU Architecture**: `gfx1201` (AMD Radeon RX 9070 XT, RDNA4)
- **Host Languages**: C++20, Python 3.12 / 3.13
- **Deep Learning Framework**: PyTorch with ROCm HIP backend (`torch.cuda.is_available() == True`)
- **Audio Verification Framework**: `pedalboard` (Surge XT C++ headless wrapper), `librosa`, `soundfile`

---

## 6. Key Pitfalls & Guidance for the Implementing Agent

1. **Avoid SIMD Intrinsics**: Do NOT attempt to port Surge's `_mm_load_ps` / `_mm_add_ps` SSE intrinsics to HIP. On GPU, vectorize across the **batch / voice dimension**, not across stereo channels or 4-sample blocks.
2. **Filter Feedback Stability**: Non-linear feedback filters running at 44.1 kHz can become unstable at extreme cutoff/resonance settings. Implement $2\times$ oversampling inside the inner loop or clamp internal integrator state using `fmaxf`/`fminf`.
3. **Keep the Interface Canonical**: Maintain strict compatibility with `surge_spec.py` (`CONT_BOUNDS`, `PARAM_NAMES`, `LP_FILTERS`, `WAVESHAPER_TYPES`) so that models trained on the GPU proxy can immediately emit valid Surge XT parameters.
