# tests/test_density_schedule.py
from sa3_control.density_schedule import ControlSchedule, SHAPES

def test_linear_descending_endpoints():
    s = ControlSchedule("linear_descending", 100.0, 2.0, 14.0)
    assert abs(s.resolve(0.0) - 14.0) < 1e-6
    assert abs(s.resolve(100.0) - 2.0) < 1e-6

def test_triangular_peaks_at_mid():
    s = ControlSchedule("triangular", 100.0, 2.0, 14.0)
    assert abs(s.resolve(0.0) - 2.0) < 1e-6
    assert abs(s.resolve(50.0) - 14.0) < 1e-6

def test_bimodal_two_peaks():
    s = ControlSchedule("bimodal", 100.0, 2.0, 14.0)
    assert abs(s.resolve(25.0) - 14.0) < 1e-6   # first peak
    assert abs(s.resolve(75.0) - 14.0) < 1e-6   # second peak
    assert abs(s.resolve(50.0) - 2.0) < 1e-6    # dip between

def test_clamps_outside_duration_and_validates_shape():
    s = ControlSchedule("linear_descending", 100.0, 2.0, 14.0)
    assert abs(s.resolve(500.0) - 2.0) < 1e-6
    assert set(SHAPES) == {"linear_descending","triangular","bimodal","sinewave"}
    try:
        ControlSchedule("bogus", 1.0, 0.0, 1.0); assert False
    except ValueError:
        pass
