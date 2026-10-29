#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Phase 0: Benchmark Musical Dataset V8 Throughput on Discrete GPU.

Measures whether MusicalDatasetV8 and DataLoader workers keep GPU supplied:
1. Measures raw samples/second and steps/second for 200 batches.
2. Checks GPU VRAM allocation.
3. Target: >= 50 samples/s throughput to guarantee GPU does not starve.
"""

import os
import sys
import time
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from musical_dataset_v8 import MusicalDatasetV8


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    num_workers = 4
    num_batches = 200

    print("=" * 80)
    print("Phase 0: Musical Dataset V8 Dataloader Benchmark")
    print(f"Device: {device} | Batch Size: {batch_size} | Workers: {num_workers} | Steps: {num_batches}")
    print("=" * 80)

    dataset = MusicalDatasetV8(split="train", epoch_length=num_batches * batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
    )

    print(f"Starting 200-step throughput benchmark...")
    start_t = time.time()
    total_samples = 0

    for i, batch in enumerate(loader):
        mix = batch["mix"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        cue = batch["cue_pos"].to(device, non_blocking=True)

        total_samples += mix.shape[0]

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_t
            rate = total_samples / elapsed
            print(f"Step {i+1:03d}/{num_batches} | Elapsed: {elapsed:.1f}s | Throughput: {rate:5.1f} samples/s ({rate/batch_size:4.2f} steps/s)")

        if i + 1 >= num_batches:
            break

    total_time = time.time() - start_t
    final_rate = total_samples / total_time

    print("=" * 80)
    print(f"FINAL THROUGHPUT: {final_rate:5.1f} samples/sec ({final_rate/batch_size:4.2f} batches/sec)")
    if device == "cuda":
        mem_alloc = torch.cuda.memory_allocated() / (1024**2)
        mem_res = torch.cuda.memory_reserved() / (1024**2)
        print(f"GPU Memory: {mem_alloc:.1f} MB allocated, {mem_res:.1f} MB reserved")
    print("=" * 80)

    gate_pass = final_rate >= 50.0
    print(f"Gate Status (Target >= 50 samples/s): {'PASSED' if gate_pass else 'FAILED / BOTTLENECK'}")


if __name__ == "__main__":
    main()
