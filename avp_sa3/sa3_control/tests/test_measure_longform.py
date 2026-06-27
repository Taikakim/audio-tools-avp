# tests/test_measure_longform.py
import numpy as np, soundfile as sf, tempfile, os
from sa3_control.measure_longform import windowed_metrics

def test_density_of_click_train(tmp_path):
    sr = 44100; dur = 20.0; rate = 4.0   # 4 clicks/sec
    y = np.zeros(int(sr*dur), np.float32)
    for i in range(int(dur*rate)):
        y[int(i*sr/rate)] = 1.0
    p = str(tmp_path/"clicks.wav"); sf.write(p, y, sr)
    rows = windowed_metrics(p, win_sec=10.0, hop_sec=10.0)
    assert len(rows) >= 1
    assert abs(rows[0]["onset_density"] - rate) < 1.0   # within 1/s of 4
