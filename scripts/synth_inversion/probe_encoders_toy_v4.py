"""TOY probe: can the temporal encoders learn attack and decay controls that the pooled v3 encoder has to squeeze?

Trains each encoder (the v3 conv + global-average-pool `AudioEncoder`, and the three v4 pools) with the same small head to
regress four controls of the stand-in synth from a 128 x 81 log-mel, then reports held-out R^2 per control.

Controls (log10 scale, standardised): attack time, amplitude decay time, filter-envelope decay time, resonance Q.
Fixed: filter envelope depth 2 oct. Varied: note (MIDI 28 to 50), cutoff, oscillator phases.

What this is NOT: Surge audio, real stems, the flow model, or the full trainer. It is a capacity and
optimisation check on a clean task, with one seed per encoder by default. A result in which all four encoders reach a high
R^2 means the toy is too easy to separate them; a gap means the pooling matters on this task. Neither says what happens on
real data. Treat it as a reason to run the real ablation (Stage 2), not as evidence for it.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from frontend_v4 import _MelBranch
from models_v4 import make_encoder_v4
from standin_synth_v4 import StandinPatch, midi_to_hz, render

SR = 44100
NAMES = ("log10 attack_s", "log10 amp_decay_s", "log10 fc_decay_s", "log10 q")


def make_dataset(n, seed):
    rng = np.random.default_rng(seed)
    audio, y = [], []
    for _ in range(n):
        att, dec, fcd, q = (10 ** rng.uniform(np.log10(0.002), np.log10(0.04)), 10 ** rng.uniform(np.log10(0.05), np.log10(0.8)),
                            10 ** rng.uniform(np.log10(0.02), np.log10(0.4)), 10 ** rng.uniform(np.log10(0.7), np.log10(20.0)))
        p = StandinPatch(f0=midi_to_hz(int(rng.integers(28, 51))), fc_hz=float(10 ** rng.uniform(np.log10(300), np.log10(3000))),
                         fc_env_oct=2.0, fc_decay_s=fcd, q=q, attack_s=att, amp_decay_s=dec, amp_sustain=0.2,
                         phase_seed=int(rng.integers(0, 10 ** 6)))
        audio.append(render(p, dur=0.8, note_dur=0.5, max_harm=80))
        y.append([np.log10(att), np.log10(dec), np.log10(fcd), np.log10(q)])
    return np.stack(audio), np.asarray(y, np.float32)


def to_mel(audio):
    branch = _MelBranch(SR, 1024, 441, 128, 20.0, 16000.0)
    with torch.no_grad():
        return branch(torch.as_tensor(audio)).unsqueeze(1)          # [N, 1, 128, 81]


class Net(nn.Module):
    def __init__(self, kind, embed=128):
        super().__init__()
        self.enc = make_encoder_v4(kind, embed)
        self.head = nn.Sequential(nn.Linear(embed, 128), nn.GELU(), nn.Linear(128, 4))

    def forward(self, m):
        return self.head(self.enc(m))


def r2(pred, y):
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return (1.0 - ss_res / ss_tot).numpy()


def train_one(kind, mel_tr, y_tr, mel_va, y_va, steps, batch, lr, seed):
    torch.manual_seed(seed)
    net = Net(kind)
    n_params = sum(p.numel() for p in net.enc.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, len(mel_tr), (batch,), generator=g)
        loss = (net(mel_tr[idx]) - y_tr[idx]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(mel_va[i:i + 100]) for i in range(0, len(mel_va), 100)])
    return n_params, r2(pred, y_va), time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--encoders", default="conv,temporal_asp,temporal_segments,temporal_queries")
    ap.add_argument("--out", default="probe_encoders_toy_v4.md")
    a = ap.parse_args(argv)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    t0 = time.time()
    audio, y = make_dataset(a.n_train + a.n_val, seed=0)
    print(f"rendered {len(audio)} stand-in clips in {time.time() - t0:.0f} s; cpu threads {torch.get_num_threads()}", flush=True)
    mel = to_mel(audio)
    mu, sd = y[:a.n_train].mean(0), y[:a.n_train].std(0)
    yt = torch.as_tensor((y - mu) / sd)
    mel_tr, mel_va, y_tr, y_va = mel[:a.n_train], mel[a.n_train:], yt[:a.n_train], yt[a.n_train:]
    rows = {}
    for kind in a.encoders.split(","):
        rs = []
        for s in range(a.seeds):
            n_params, r, dt = train_one(kind, mel_tr, y_tr, mel_va, y_va, a.steps, a.batch, a.lr, seed=s)
            rs.append(r)
            print(f"{kind} seed {s}: R2 " + " ".join(f"{v:.3f}" for v in r) + f" ({dt:.0f} s)", flush=True)
        rows[kind] = (n_params, np.mean(rs, axis=0), np.std(rs, axis=0) if a.seeds > 1 else None)
    L = ["# Toy encoder probe (stand-in synth, NOT Surge; held-out R-squared)", "",
         f"{a.n_train} train / {a.n_val} held-out clips, {a.steps} steps, batch {a.batch}, AdamW {a.lr}, {a.seeds} seed(s) per encoder.", "",
         "| encoder | encoder params | " + " | ".join(NAMES) + " | mean |", "|---|---|" + "---|" * (len(NAMES) + 1)]
    for k, (n, r, s) in rows.items():
        cells = " | ".join(f"{v:.3f}" + (f" ±{e:.3f}" if s is not None else "") for v, e in zip(r, s if s is not None else r))
        L.append(f"| {k} | {n:,} | {cells} | {r.mean():.3f} |")
    L += ["", "Toy result on a clean stand-in task: it says whether pooling matters here, not on Surge or real stems."]
    rep = "\n".join(L)
    print(rep)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(rep + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
