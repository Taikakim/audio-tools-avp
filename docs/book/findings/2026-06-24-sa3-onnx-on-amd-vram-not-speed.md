# Stable Audio 3 on AMD via ONNX — a VRAM win, not a speed win

*Finding — 2026-06-24. Source: our own work — exporting the SA3 SAME autoencoder **and** the medium-base DiT to ONNX, GPU-verifying them on the RX 9070 XT (ROCm + MIGraphX EP), and a like-for-like ONNX-vs-native-torch benchmark. A **runtime/deployment** finding, not a control method; it sits beside the Sourcebook rather than advancing a chapter. Full tooling + the long-form gotcha list live in `stable-audio-3/docs/onnx-amd-inference.md` and `SAO/MASTER.md` §5. House rule kept: this arc corrected **three** over-optimistic claims of mine in flight (§2, §3, §4) — the corrected versions are what's below.*

---

## 1. Why bother: low-VRAM inference that coexists with training

The 9070 XT has 16 GB. A torch SA3 decode holds ~5 GB and the full `StableAudioModel` ~9.6 GB, so you can't run inference and a training job on the same card. The goal was a **torch-free, low-VRAM** path: export the model to ONNX and run it under ONNX Runtime + the MIGraphX execution provider — no torch/ROCm-torch stack at inference time. The cgisky `stable-audio-3-rs` MNN port proved the SA3-Small AE quantizes; this is the same idea on our stack, for the full medium pipeline.

It works end-to-end. The interesting part is what it actually costs (§4), which is **not** what I assumed going in.

## 2. The export recipe, and the bias that almost shipped

Both the AE (decoder/encoder) and the DiT export with the *same* recipe (details in the SA3 doc):

- **Disable FlexAttention, not just flash.** `SA3_DISABLE_FLASH_ATTN=1` is necessary but not sufficient — with flash off, the sliding-window layers fall to FlexAttention (a `torch.compile` HOP the dynamo ONNX exporter can't translate). Also set `transformer.flex_attention_available=False; flex_attention_compiled=None` → math-equivalent masked SDPA, exports clean. **opset ≥ 18.**
- **AE = fixed-chunk + host overlap-add** (it folds the sequence length-dependently); **DiT = a ladder of fixed lengths** {256,512,1024,2048,4096} (full-sequence attention can't be chunked, one compiled graph per length).

**The correction (house rule):** my first DiT export fed `local_add_cond=None` and validated at **cos 1.0** vs torch — and I believed it. It was comparing `_forward(None)` against `_forward(None)`. Real generation feeds a **257-channel** `local_add_cond` (= `inpaint_mask` ⊕ `inpaint_masked_input`); for text-to-audio it's all-zeros, but the DiT **projects it with a bias**, so `None ≠ zeros` (measured cos 0.98, max|Δ| 0.74). The export now takes `local_add_cond` as an input and the runner feeds zeros. Lesson that generalizes: **a cos-1.0 against your own reference proves nothing if both sides share the same omission** — validate against the real generation path, not a convenient stub. With it fixed, the full 8-step real-prompt generation reproduces torch to **z0 cos 0.9999**.

## 3. fp16: the "fix" that OOM'd harder

The DiT (5.8 GB fp32) + decoder won't co-reside while compiling on 16 GB — the decoder's MIGraphX compile with the DiT resident pushes VRAM to ~16.5 GB and **thrashes (31 min vs 9 min standalone)**, or OOMs outright.

**The correction (house rule):** the obvious fix looked like the **`migraphx_fp16_enable` EP option** — run the existing fp32 graph in fp16, no re-export. It does the opposite of helping. That EP path **loads the fp32 weights and quantizes at session-init**, so peak memory is *higher*, and DiT+decoder co-residency **OOM'd harder** (HIP out-of-memory, hit it twice). The real fix is a **true fp16 *export*** — fp16 weights on disk that load directly: **DiT 2.9 GB + decoder 0.9 GB = 3.8 GB resident**, both fit with room. (>2 GB fp16 models need `convert_float_to_float16_model_path` + external-data save — the in-memory converter overflows the 2 GB protobuf limit.) fp16 export vs fp32: **cos 0.999992**. So: *fp16-exported files* for low VRAM; *not* the runtime-fp16 EP flag.

## 4. The benchmark: I expected speed, it's VRAM

Like-for-like, L256 / 8 steps / CFG 6 / identical seed+conditioning. The fair metric is the **DiT sampling loop alone** (`dit_loop_s`) — the full pipeline isn't comparable because the shared ONNX decoder runs on CPU-ORT in the torch (SA3) venv vs MIGraphX in the onnx (mir) venv, so the full `gen_s` reflects a CPU decode for torch. Time the loop, cuda-synced:

| | torch (cuda fp16, eager/SDPA) | ONNX fp16 (MIGraphX) |
|---|---|---|
| **DiT loop, 16 calls** | **0.707 s** (44 ms/call, RTF 33.6×) | **2.314 s** (144 ms/call, RTF 10.3×) |
| Resident VRAM | 3.1 GB working (full model load ~9.6 GB) | **3.8 GB, no torch stack** |
| z0 vs torch | — | **cos 0.999308** |
| AOT compile | none | ~14 min DiT + ~24 min decoder, one-time/session |

**The correction (house rule):** I assumed compiling to ONNX/MIGraphX would be *faster*. It is **~3.3× slower** per DiT call. MIGraphX's compiled graph does not beat torch's tuned rocBLAS/MIOpen kernels on this transformer (and torch with CK flash-attn — which we run everywhere else — would widen the gap further). Quality is identical (z0 cos 0.9993). fp16 MIGraphX is ~25 % faster than fp32 (144 vs 191 ms/call), but still well behind eager torch.

So the value of the port is **not** throughput. It's **3.8 GB resident in a lightweight ORT process with zero torch/ROCm-torch dependency** — exactly the §1 goal: inference that coexists with a training run, as one portable graph. The decoder alone was already the strong case (cos 0.999998, RTF ~39× — decode is cheaper than the DiT loop).

## 5. Use it for what it's good at

- **Reach for ONNX/MIGraphX when VRAM is the constraint** — running inference next to a training job, or shipping a torch-free graph. The gen server (`latent_server_dit_onnx.py`) compiles DiT+decoder once at boot and serves `/generate` from precached conditioning; the decode-only `latent_server_onnx.py` is the lighter cousin.
- **Reach for torch when speed is the constraint** — eager cuda fp16 (more so with CK flash-attn) is ~3× faster and needs no AOT compile.
- **One-time costs that bite:** ~13–24 min MIGraphX AOT compile per graph per session (compiled-model caching is *not* exposed in this `onnxruntime_migraphx` 1.23.2 build), so amortize it behind a long-lived server. Text conditioning (T5-Gemma) is precached offline so the runtime stays torch-free.

**Open question:** the speed gap is the whole story now. A `batch=2` export (one DiT call/step for CFG, already exported at cos 1.0) would halve the call count; a newer ORT-ROCm build with compiled-model caching would kill the compile tax; and it's worth checking whether MIGraphX's gap closes at the longer ladder rungs (T=2048/4096, where the attention GEMMs dominate) or widens. Until then the rule stands: **ONNX for low-VRAM coexistence, torch for raw speed.**
