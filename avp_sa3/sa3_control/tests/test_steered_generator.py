import torch
from sa3_control.density_schedule import ControlSchedule
from sa3_control.steered_longform import SteeredGenerator

class FakeEnc:
    def __call__(self, s): return s.view(1, 1, 1)

class FakeInner:
    def __init__(self): self.seen = []
    def generate(self, prompt, prefix_latents, prefix_frames, n_frames, seed):
        self.seen.append((prefix_frames, n_frames))
        return torch.zeros(1, 4, n_frames)

def test_window_time_and_scalar_tracking():
    sched = ControlSchedule("linear_descending", 100.0, 2.0, 14.0)
    inner = FakeInner()
    g = SteeredGenerator(inner, sched, FakeEnc(), mean=0.0, std=1.0, gain=1.0,
                         fps=10.0, cfg_scale=1.0, device="cpu", dtype=torch.float32)
    # window=300 frames (30s @10fps), overlap=50; window0 prefix=0, then prefix=50
    g.generate("p", None, 0, 300, 0)        # t=0   -> 14.0
    g.generate("p", torch.zeros(1,4,50), 50, 300, 1)  # t=300/10=30 -> resolve(30)
    g.generate("p", torch.zeros(1,4,50), 50, 300, 2)  # t=(300+250)/10=55 -> resolve(55)
    ts = [t for t, _ in g.applied]
    assert ts == [0.0, 30.0, 55.0]
    assert abs(g.applied[0][1] - 14.0) < 1e-6
    assert abs(g.applied[1][1] - sched.resolve(30.0)) < 1e-6
    assert abs(g.applied[2][1] - sched.resolve(55.0)) < 1e-6
