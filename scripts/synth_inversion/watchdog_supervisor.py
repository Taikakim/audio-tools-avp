"""Automated Training Watchdog & Adaptive Hyperparameter Supervisor.

Monitors overnight training metrics in real-time. If loss increases across consecutive
evaluation windows (divergence/plateau/instability), it automatically:
  1. Gracefully terminates the diverging process.
  2. Rolls back model weights to the best recorded checkpoint.
  3. Adapts hyperparameters (increases batch size to next power of 2, halves LR, increases damping).
  4. Relaunches training and logs the remediation action to watchdog.log.
"""

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import List, Tuple

LOG_DIR = "/run/media/kim/Mantu/surge_200k_models/overnight_realistic_bass"
RUN_LOG = os.path.join(LOG_DIR, "run.log")
WATCHDOG_LOG = os.path.join(LOG_DIR, "watchdog.log")
PYTHON_BIN = "/home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python"
TRAIN_SCRIPT = "/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion/train_realistic_bass_overnight.py"

CHECK_INTERVAL_S = 45  # Check every 45s
CONSECUTIVE_INCREASE_LIMIT = 4  # Trigger after 4 consecutive loss increases
DIVERGENCE_RATIO_LIMIT = 1.30   # Trigger if loss spikes > 30% above best seen

# Initial hyperparameter state
current_batch_size = 32
current_lr_jepa = 9e-5
current_lr_flow = 6e-5

def log_watchdog(msg: str):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} [WATCHDOG] {msg}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a") as f:
        f.write(line + "\n")
        f.flush()

def parse_recent_log_metrics() -> List[Tuple[int, float, float, float]]:
    """Returns list of (step, jepa_loss, flow_loss, combined_loss) from run.log."""
    if not os.path.exists(RUN_LOG):
        return []
    
    entries = []
    # Pattern: Step   1050 | ... | JEPA Loss: 0.6369 | ... | Flow Loss: 0.2898 | ...
    pattern = re.compile(r"Step\s+(\d+)\s+\|\s+Elapsed:\s+[\d\.]+h\s+\|\s+ETA:\s+[\d\.]+h\s+\|\s+JEPA Loss:\s+([\d\.]+)\s+.*?\|\s+Flow Loss:\s+([\d\.]+)")
    with open(RUN_LOG, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                jepa_loss = float(m.group(2))
                flow_loss = float(m.group(3))
                combined = jepa_loss + flow_loss
                entries.append((step, jepa_loss, flow_loss, combined))
    return entries

def find_active_train_pid():
    try:
        out = subprocess.check_output(["pgrep", "-f", "train_realistic_bass_overnight.py"]).decode()
        pids = [int(p.strip()) for p in out.strip().split() if p.strip()]
        return pids
    except:
        return []

def apply_remediation(reason: str, best_step: int):
    global current_batch_size, current_lr_jepa, current_lr_flow
    log_watchdog(f"ALERT: Divergence condition detected! Reason: {reason}")
    log_watchdog("Initiating Automated Remediation Sequence...")

    # 1. Terminate running training process
    pids = find_active_train_pid()
    if pids:
        log_watchdog(f"Stopping active training processes: {pids}")
        for p in pids:
            try:
                os.kill(p, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(6)  # Allow clean checkpoint write

    # 2. Revert checkpoint to best known state
    best_ckpt = os.path.join(LOG_DIR, "checkpoint_best.pt")
    latest_ckpt = os.path.join(LOG_DIR, "checkpoint_latest.pt")
    if os.path.exists(best_ckpt):
        shutil.copy2(best_ckpt, latest_ckpt)
        log_watchdog(f"Reverted checkpoint_latest.pt to checkpoint_best.pt (best step: {best_step})")
    else:
        log_watchdog("checkpoint_best.pt not found; keeping current latest checkpoint.")

    # 3. Adapt hyperparameters
    old_bs = current_batch_size
    current_batch_size = min(128, current_batch_size * 2)  # Power of 2 escalation: 32 -> 64 -> 128
    current_lr_jepa *= 0.50                                # Halve learning rates
    current_lr_flow *= 0.50

    log_watchdog(f"Adapted Hyperparameters:")
    log_watchdog(f"  Batch size: {old_bs} -> {current_batch_size} (power of 2 escalation)")
    log_watchdog(f"  LR JEPA:    {current_lr_jepa:.2e} (damped 50%)")
    log_watchdog(f"  LR Flow:    {current_lr_flow:.2e} (damped 50%)")

    # 4. Relaunch training process
    cmd = [
        PYTHON_BIN, TRAIN_SCRIPT,
        "--out_dir", LOG_DIR,
        "--hours", "8.0",
        "--batch_size", str(current_batch_size),
        "--num_workers", "4",
        "--lr_jepa", str(current_lr_jepa),
        "--lr_flow", str(current_lr_flow),
        "--checkpoint_interval_steps", "1000",
        "--log_interval_steps", "50",
        "--device", "cuda:0",
    ]
    log_watchdog(f"Relaunching training command: {' '.join(cmd)}")
    with open(RUN_LOG, "a") as f_out:
        proc = subprocess.Popen(cmd, stdout=f_out, stderr=subprocess.STDOUT)
    log_watchdog(f"Training resumed under PID {proc.pid}. Watchdog resumed.")

def main():
    log_watchdog("Watchdog Supervisor started. Monitoring run.log every 45s...")
    last_processed_step = 0
    all_time_best_loss = float("inf")
    best_step = 0

    while True:
        time.sleep(CHECK_INTERVAL_S)
        entries = parse_recent_log_metrics()
        if not entries:
            continue

        latest_step, jepa_loss, flow_loss, combined_loss = entries[-1]
        
        # Track all-time best
        for s, j_l, f_l, comb in entries:
            if comb < all_time_best_loss:
                all_time_best_loss = comb
                best_step = s

        # Check if new steps have occurred
        if latest_step <= last_processed_step:
            # Check if training process crashed / died unexpectedly
            pids = find_active_train_pid()
            if not pids:
                log_watchdog("WARNING: Training process is not running! Relaunching from latest checkpoint...")
                apply_remediation("Process not found / unexpected termination", best_step)
            continue

        last_processed_step = latest_step

        # Analyze trend over the last N log entries
        if len(entries) >= CONSECUTIVE_INCREASE_LIMIT + 1:
            recent_losses = [e[3] for e in entries[-CONSECUTIVE_INCREASE_LIMIT-1:]]
            
            # Check for monotonic increase
            is_increasing = all(recent_losses[i] < recent_losses[i+1] for i in range(len(recent_losses)-1))
            
            # Check for sudden severe spike
            is_spike = combined_loss > all_time_best_loss * DIVERGENCE_RATIO_LIMIT

            if is_increasing:
                apply_remediation(f"Loss monotonically increasing over last {CONSECUTIVE_INCREASE_LIMIT} checks ({recent_losses})", best_step)
                time.sleep(60) # Allow new process to spin up
            elif is_spike and latest_step > 2000:
                apply_remediation(f"Loss spiked to {combined_loss:.4f} (> {DIVERGENCE_RATIO_LIMIT}x best {all_time_best_loss:.4f})", best_step)
                time.sleep(60)
            else:
                log_watchdog(f"Status OK @ step {latest_step:6d} | Combined Loss: {combined_loss:.4f} (Best: {all_time_best_loss:.4f} @ {best_step})")

if __name__ == "__main__":
    main()
