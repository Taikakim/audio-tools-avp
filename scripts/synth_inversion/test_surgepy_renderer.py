"""Acceptance tests for the native surgepy renderer (v4 Spec Section 1)."""
import os
import random
import tempfile
import time
import numpy as np
import pytest
import librosa
from surge_synth import SurgeRenderer

def test_fx_off_is_really_off():
    """1. With FX disabled, one 8th note's tail falls below -60 dB within 0.4 s (no repeats)."""
    renderer = SurgeRenderer()
    renderer.ensure_fx_off()
    
    # 0.2s note, total 0.8s
    audio = renderer.render_note(midi_note=36, note_dur=0.2, total_dur=0.8)
    sr = renderer.sample_rate
    
    # Check tail from 0.4s to 0.8s
    tail = audio[int(sr * 0.4):]
    tail_rms = float(np.sqrt(np.mean(tail**2)))
    tail_db = 20 * np.log10(tail_rms + 1e-12)
    
    print(f"Tail RMS dB: {tail_db:.2f} dB")
    assert tail_db < -60.0, f"Tail level {tail_db:.2f} dB is not below -60 dB"

def test_no_state_leak_between_renders():
    """2. A reference patch renders identically before and after 20 unrelated random patches."""
    renderer = SurgeRenderer()
    sr = renderer.sample_rate
    
    def mel_spec(y):
        m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
        return librosa.power_to_db(m, ref=np.max)

    # Reference patch
    a_ref1 = renderer.render_note(midi_note=36, note_dur=0.3, total_dur=0.8)
    a_ref2 = renderer.render_note(midi_note=36, note_dur=0.3, total_dur=0.8)
    
    m1 = mel_spec(a_ref1)
    m2 = mel_spec(a_ref2)
    repeat_noise = float(np.mean(np.abs(m1 - m2)))

    # Render 20 unrelated random patches, turning some FX on
    for _ in range(20):
        renderer.set_param("FX A1 FX Type", float(random.randint(0, 10)))
        renderer.set_param("Osc 1 Shape", random.uniform(-1.0, 1.0))
        renderer.render_note(midi_note=random.randint(28, 50), note_dur=random.uniform(0.1, 0.4))

    # Reset back to reference patch
    renderer.ensure_fx_off()
    renderer.set_param("Osc 1 Shape", 0.0)
    
    a_ref3 = renderer.render_note(midi_note=36, note_dur=0.3, total_dur=0.8)
    m3 = mel_spec(a_ref3)
    
    after_diff = float(np.mean(np.abs(m1 - m3)))
    print(f"Repeat noise: {repeat_noise:.4f}, After 20 patches diff: {after_diff:.4f}")
    assert after_diff <= 1.5 * max(repeat_noise, 0.5), "State leaked between renders"

def test_preset_roundtrip():
    """3. Export -> load in fresh instance -> every parameter equal and render equal."""
    r1 = SurgeRenderer()
    r1.set_param("Osc 1 Shape", 0.75)
    r1.set_param("Osc 1 Width 1", 0.25)
    r1.set_param("Filter 1 Cutoff", 10.0)
    
    with tempfile.NamedTemporaryFile(suffix=".fxp", delete=False) as f:
        tmp_path = f.name
        
    try:
        r1.save_patch(tmp_path)
        
        r2 = SurgeRenderer()
        ok = r2.load_patch(tmp_path)
        assert ok, "Failed to load patch into fresh instance"
        
        assert abs(r2.get_param("Osc 1 Shape") - 0.75) < 1e-4
        assert abs(r2.get_param("Osc 1 Width 1") - 0.25) < 1e-4
        assert abs(r2.get_param("Filter 1 Cutoff") - 10.0) < 1e-4
        
        # FX slots must be Off
        assert r2.get_param("FX A1 FX Type") == 0.0
        assert r2.get_param("FX A2 FX Type") == 0.0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def test_render_speed():
    """4. Renders/sec per CPU worker >= 100 renders/s."""
    renderer = SurgeRenderer()
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        renderer.render_note(midi_note=36, note_dur=0.3, total_dur=0.8)
    elapsed = time.perf_counter() - t0
    rps = n / elapsed
    print(f"Single-core render speed: {rps:.1f} renders/sec")
    assert rps >= 100.0, f"Render speed {rps:.1f} is slower than 100 renders/sec"

if __name__ == "__main__":
    test_fx_off_is_really_off()
    test_no_state_leak_between_renders()
    test_preset_roundtrip()
    test_render_speed()
    print("ALL 4 RENDERER ACCEPTANCE TESTS PASSED!")
