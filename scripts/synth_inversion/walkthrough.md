# V5 Stage 0: Render Bank Generation & Metric Validation

## 1. Summary of Completed Milestones

### 1.1 Metric Ladder Validation Gate (`ladder_metric_check_v5.py`)
Executed [`ladder_metric_check_v5.py`](file:///home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion/ladder_metric_check_v5.py) across **9 synthesis axes** on the held-out split of the updated 149-preset manifold ([`real_bass_manifold_v3.npz`](file:///home/kim/sao_runs/real_bass_manifold_v3.npz)):
* **Render Throughput:** 720 clips rendered in **9.0 seconds (76.9 renders/second)**.
* **Metric Rankings:**
  * **Primary Spectral:** `mss` (0.65 mean $\rho$) and `logmel_l1` (0.64 mean $\rho$) established highest sensitivity and monotonicity.
  * **Structural Envelope/Filter:** `mel_ssim_dist` achieved 0.61 mean $\rho$; `mel_ssim_gated_dist` achieved 0.57 mean $\rho$ (0.85 cutoff, 0.90 sustain, 0.96 FEG, with high step-to-noise ratio).
  * **Unison Detune Diagnostic:** `modspec_dist` confirmed as the primary detector of unison detune beating ($\rho=0.37$, $2.0\times$ step-to-noise ratio).
* Full report stored at [`/home/kim/sao_runs/ladder_gate_v5/ladder_gate.md`](file:///home/kim/sao_runs/ladder_gate_v5/ladder_gate.md).

### 1.2 OS NVMe Render Bank Build (`render_bank_v5.py build`)
Built the full Option A pre-rendered dataset bank on OS NVMe:
* **Target Location:** [`/home/kim/sao_runs/bank_v5_s0`](file:///home/kim/sao_runs/bank_v5_s0) on `/dev/nvme0n1p2`.
* **Total Volume:** **983,040 audio clips** across 240 shards:
  * 786,432 main dataset items (192 shards $\times$ 4,096 items).
  * 12,288 ladders (48 shards $\times$ 256 ladders $\times$ 16 steps = 196,608 clips).
* **Generation Throughput:** Averaged **579.8 clips/second** across 14 CPU workers.
* **Total Generation Time:** **28.2 minutes** (1,695 seconds).
* **Disk Space:** Bank occupies **65 GB**; **36 GB free** remains on `/`.

### 1.3 Bank Integrity Verification (`render_bank_v5.py verify`)
Re-rendered 32 sample items from their deterministic seeds and compared against stored bank shards:
* **Vector Identity:** `vectors_identical: true` (100% bit-exact parameter reproduction).
* **Mel-L1 Distance:** `logmel_l1_median: 0.0307`, max `0.0994` (median is well within Surge XT's render noise floor of $\le 0.06$).

---

## 2. Next Actions

1. **Step 0.6 (Banked Throughput Benchmark):** Benchmark 500 steps of `train_realistic_bass_v3.py --bank /home/kim/sao_runs/bank_v5_s0` on GPU 0 (`gfx1201`) to confirm that `data wait %` drops near zero.
2. **Stage 1 (Audio Baseline):** Evaluate existing A0 and A3′ checkpoints on 256 held-out notes (`audio_eval_v5.py`).
3. **Stage 2 (Small-Model Paradigm Search):** Launch S0 (Temporal ASP) vs S1 (Temporal ASP + Cross-Attention) at `SIZE=small` (12,000 steps) consuming the new bank.
