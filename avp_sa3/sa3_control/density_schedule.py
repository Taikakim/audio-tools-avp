# sa3_control/density_schedule.py
"""Time-varying control schedules for steered long-form generation.
Pure: (shape, duration, lo, hi) -> raw onset-density value at time t. No model deps."""
import math

SHAPES = ("linear_descending", "triangular", "bimodal", "sinewave")

class ControlSchedule:
    def __init__(self, shape, duration, lo, hi):
        if shape not in SHAPES:
            raise ValueError(f"unknown shape {shape!r}; expected one of {SHAPES}")
        self.shape = shape
        self.duration = float(duration)
        self.lo = float(lo)
        self.hi = float(hi)

    def resolve(self, t):
        u = min(max(t / self.duration, 0.0), 1.0)   # normalized 0..1
        if self.shape == "linear_descending":
            f = 1.0 - u
        elif self.shape == "triangular":
            f = 1.0 - abs(2.0 * u - 1.0)
        elif self.shape == "bimodal":
            f = 0.5 * (1.0 - math.cos(4.0 * math.pi * u))   # peaks at u=.25,.75
        else:  # sinewave: 3 smooth oscillations
            f = 0.5 * (1.0 - math.cos(6.0 * math.pi * u))
        return self.lo + (self.hi - self.lo) * f


def ridge_gain(density):
    """Inverse gain-per-density operating ridge (riffer-disentanglement-recipe memory):
    low density needs more authority (~2.75 at d=2), high density needs almost none
    (~1.0 at d=12). A FIXED gain over-steers the busy end and fries CE; the operating
    recipe is a gain-PER-density curve. Linear, clamped to [0.7, 3.0]."""
    g = 2.75 - 0.175 * (float(density) - 2.0)
    return max(0.7, min(3.0, g))
