"""Crash-only supervisor for train_realistic_bass_overnight.py (v2, 2026-10-02 review).

Launches the trainer itself and restarts it (the trainer resumes from checkpoint_latest.pt) when:
  * it exits with a non-zero code (crash, OOM, or code 3 = repeated non-finite steps), or
  * training.log stops growing for --stall_minutes while the process is alive (GPU hang, stuck
    worker): SIGTERM, then SIGKILL after 90 s, then restart.
It gives up after --max_restarts restarts within --restart_window_minutes (a crash loop is a bug to
read, not to retry). A zero exit code (time budget reached, or stopped by Ctrl-C/SIGTERM) ends it.

What it deliberately does NOT do any more: change batch size or learning rate. v1 halved the LR and
doubled the batch whenever five consecutive 50-step loss averages rose, which happens by chance
about once per 120 log lines on a noisy online loss — an LR decay driven by noise. Its rollback
target (checkpoint_best.pt) was never written by the v1 trainer, and it SIGTERMed first, so the
trainer saved the state it was meant to roll back from. LR decay belongs in the trainer's schedule.

Usage (everything after -- is the trainer command, run as is):
  python watchdog_supervisor.py --out_dir DIR -- /path/python train_realistic_bass_overnight.py --out_dir DIR ...
"""
import argparse
import os
import signal
import subprocess
import sys
import time

_stop = False


def _on_signal(sig, _frame):
    global _stop
    _stop = True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True, help="the trainer's --out_dir (logs live there)")
    ap.add_argument("--stall_minutes", type=float, default=20.0)
    ap.add_argument("--max_restarts", type=int, default=5)
    ap.add_argument("--restart_window_minutes", type=float, default=60.0)
    ap.add_argument("--poll_s", type=float, default=30.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- trainer command")
    args = ap.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("give the trainer command after --")

    os.makedirs(args.out_dir, exist_ok=True)
    train_log = os.path.join(args.out_dir, "training.log")
    wd_log = open(os.path.join(args.out_dir, "watchdog.log"), "a", buffering=1)

    def log(msg):
        line = f"{time.strftime('[%Y-%m-%d %H:%M:%S]')} {msg}"
        print(line, flush=True)
        wd_log.write(line + "\n")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    restarts = []

    def launch():
        log("Launching: " + " ".join(cmd))
        out = open(os.path.join(args.out_dir, "run.log"), "a")
        return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT), time.time()

    def stop(proc):
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            log("Trainer did not stop within 90 s; SIGKILL")
            proc.kill()
            proc.wait()

    proc, started = launch()
    while True:
        time.sleep(args.poll_s)
        if _stop:
            log("Watchdog stopping; forwarding SIGTERM to the trainer (it saves a checkpoint)")
            stop(proc)
            return 0
        rc = proc.poll()
        reason = None
        if rc is not None:
            if rc == 0:
                log("Trainer finished (exit 0).")
                return 0
            reason = f"trainer exited with code {rc}"
        else:
            last = os.path.getmtime(train_log) if os.path.exists(train_log) else started
            idle_min = (time.time() - max(last, started)) / 60.0
            if idle_min > args.stall_minutes:
                reason = f"training.log idle for {idle_min:.1f} min"
                stop(proc)
        if reason is None:
            continue
        now = time.time()
        restarts = [t for t in restarts if now - t < args.restart_window_minutes * 60] + [now]
        if len(restarts) > args.max_restarts:
            log(f"{reason}; {len(restarts) - 1} restarts within {args.restart_window_minutes:.0f} min already. "
                "Giving up: read run.log.")
            return 1
        log(f"{reason}; restarting from checkpoint_latest.pt (restart {len(restarts)}/{args.max_restarts} in window)")
        proc, started = launch()


if __name__ == "__main__":
    sys.exit(main())
