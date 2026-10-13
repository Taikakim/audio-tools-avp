# V5 Training Pipeline: Stage 0 Render Bank on OS NVMe & Staged Roadmap

Execute **Stage 0** of the Synth Inversion V5 plan ([V5_PROPOSAL_REVIEW.md](file:///home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion/V5_PROPOSAL_REVIEW.md)): establish metric validity, build the pre-rendered training bank on OS NVMe, benchmark throughput to verify GPU saturation, and prepare the Stage 1–2 model search.

---

## User Review Required

> [!IMPORTANT]
> ### 1. Bank Storage & OS NVMe Allocation
> - **Target Location:** `/home/kim/sao_runs/bank_v5_s0` (on `/dev/nvme0n1p2`, OS NVMe).
> - **Capacity Status:** `/dev/nvme0n1p2` currently has **85 GB free** (91% used).
> - **Sizing Options:**
>   - **Option A (Full Standard Bank):** 786,432 main items + 12,288 ladders = **~69.4 GB**. This leaves ~15.6 GB free on `/`. Safe if Kim makes extra room soon.
>   - **Option B (Stage 2 Sized Bank):** 576,000 main items + 12,000 ladders = **~51.0 GB** (exact requirement for a 12,000-step run at batch 64 with 0 repeats). Leaves ~34 GB free on `/`.
>
> *Which sizing option do you prefer to start with? (Default recommendation: Option B if manual cleanup has not happened yet, or Option A if space is being cleared now).*

> [!NOTE]
> ### 2. RAM Preloading vs. OS NVMe Memory Mapping (`mmap`)
> - **Memory Status:** System RAM is 91 GiB total, **43 GiB used, 48 GiB available**.
> - **Preloading Constraint:** Attempting to force-load the full ~69.4 GB uncompressed audio bank into Python memory or a `tmpfs` RAM disk will exceed the 48 GiB available RAM and trigger the Linux Out-Of-Memory (OOM) killer.
> - **Built-in `mmap` Solution:** [`render_bank_v5.py`](file:///home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion/render_bank_v5.py#L161-L177) is already built with `np.load(..., mmap_mode="r")`:
>   - The Linux kernel transparently page-caches frequently read shards into available RAM (~45 GB) on demand.
>   - All DataLoader worker processes share the exact same physical memory pages with zero IPC overhead.
>   - Cache misses read directly from OS NVMe at 3–7 GB/s (sub-millisecond latency).
>   - **Conclusion:** Memory mapping directly from OS NVMe gives the exact speed benefit of RAM caching without the hazard of an OOM crash.

---

## Proposed Roadmap & Execution Steps

### Step 0.1: Environment & Manifold Setup
* Manifold target: Use the updated 149-preset manifold ([`real_bass_manifold_v3.npz`](file:///home/kim/sao_runs/real_bass_manifold_v3.npz), 127 train / 22 val) now staged in `/home/kim/sao_runs/`.
* Plugin target: Use native Surge XT VST3 at `~/.vst3/Surge XT.vst3`.
* Verify CPU test suite passes (`pytest scripts/synth_inversion/test_v5.py`).

### Step 0.2: Metric Ladder Validation (`ladder_metric_check_v5.py`)
* Run [`ladder_metric_check_v5.py`](file:///home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion/ladder_metric_check_v5.py) across 9 synthesis axes (cutoff, sustain, FEG, decay, detune, drive, resonance, FM depth, noise volume) on 22 held-out presets (~700 renders, ~2 min on CPU).
* Confirm that candidate metrics rise monotonically along parameter sweeps ($\text{Spearman } \rho \ge 0.8$ on $\ge 70\%$ of ladders).
* Validate the ratio of single-step distance to Surge render noise ($\text{step}_1 / \text{noise} > 1.0$).
* Decide whether **gated Mel-SSIM** (`mel_ssim_gated_dist`) or standard Mel-SSIM advances as the primary structural spectral metric alongside `logmel_l1`.

### Step 0.3: Live Baseline Throughput Benchmark (500 steps)
* Run a quick 500-step baseline of [`train_realistic_bass_v3.py`](file:///home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion/train_realistic_bass_v3.py) on GPU 0 (`gfx1201`) with live pedalboard rendering under `gpu_guard`.
* Measure and record:
  * Baseline `samples/s`.
  * Baseline `data wait %` (share of step time spent waiting on DataLoader Surge synthesis).

### Step 0.4: Build Render Bank on OS NVMe
* Run `render_bank_v5.py build`:
  * Output directory: `/home/kim/sao_runs/bank_v5_s0`.
  * Manifold: `/home/kim/sao_runs/real_bass_manifold_v3.npz`.
  * Workers: 14 parallel CPU render workers.
  * Sizing: Selected Option A (786k items + 12k ladders) or Option B (576k items + 12k ladders).
* Monitor disk space periodically during generation.

### Step 0.5: Verify Bank Integrity
* Run `render_bank_v5.py verify --bank /home/kim/sao_runs/bank_v5_s0 --n 32`:
  * Re-renders 32 sample items from their deterministic seeds.
  * Asserts 100% bit-exact parameter vector equality.
  * Asserts audio median `logmel_l1` distance to stored bank audio is $\le 0.06$ (within Surge's phase-randomization noise floor).

### Step 0.6: Banked Throughput Benchmark (500 steps)
* Run a 500-step benchmark with `--bank /home/kim/sao_runs/bank_v5_s0`:
  * Verify `data wait %` drops to $< 5\%$.
  * Record net training speedup ($X\times$ samples/s).

---

## Downstream Stages (Gated on Stage 0)

* **Stage 1 (Audio Baseline, No Training):** Run [`audio_eval_v5.py`](file:///home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion/audio_eval_v5.py) on existing A0 and A3′ checkpoints to measure DPO headroom (draw 0 vs best-of-8) and assemble the 16-item listening pack for Kim.
* **Stage 2 (Small Model Search, 12k steps):** Launch S0 (Temporal ASP) vs S1 (Temporal ASP + Cross-Attention) at `SIZE=small` on the OS NVMe render bank.

---

## Verification Plan

### Automated Checks
1. `pytest scripts/synth_inversion/test_v5.py` exits 0 (32 tests passing).
2. `ladder_metric_check_v5.py` generates `ladder_gate_v5/summary.json` with valid monotonic scores.
3. `render_bank_v5.py verify` exits with `"vectors_identical": true` and `logmel_l1_median <= 0.06`.

### Manual & System Verification
1. `df -h /` verified before and after bank build to ensure headroom remains above 10 GB.
2. `free -h` verified during training to ensure Linux page cache operates cleanly without swapping.
