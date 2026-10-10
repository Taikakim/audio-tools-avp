import numpy as np
from inharmonicity_target import InharmonicityExtractor

def make_sine_wave(freq, duration, sr=44100):
    t = np.arange(int(duration * sr)) / sr
    return np.sin(2 * np.pi * freq * t)

def test_waveform():
    sr = 44100
    extractor = InharmonicityExtractor(sample_rate=sr)
    
    # 1. Harmonic signal (100, 200, 300, 400Hz)
    audio_harmonic = (make_sine_wave(100, 0.1) + 
                     0.5 * make_sine_wave(200, 0.1) + 
                     0.3 * make_sine_wave(300, 0.1) + 
                     0.2 * make_sine_wave(400, 0.1))
    
    # MIDI note for 100Hz is roughly 43.8, let's just pass exact note for 100Hz
    # f0 = 440 * 2**((m-69)/12) => m = 69 + 12*log2(f0/440)
    m_100 = 69 + 12 * np.log2(100.0 / 440.0)
    
    res1 = extractor.process_frame(audio_harmonic, m_100)
    print("Harmonic Waveform:", res1['value'], res1['valid'], res1['rejection_reason'])
    
    # 2. Inharmonic signal (100, 210, 300, 400Hz)
    audio_inharmonic = (make_sine_wave(100, 0.1) + 
                     0.5 * make_sine_wave(210, 0.1) + 
                     0.3 * make_sine_wave(300, 0.1) + 
                     0.2 * make_sine_wave(400, 0.1))
                     
    res2 = extractor.process_frame(audio_inharmonic, m_100)
    print("Inharmonic Waveform:", res2['value'], res2['valid'], res2['rejection_reason'])
    
    # 3. Highly displaced signal (100, 250, 350, 450Hz) - drops out of tolerance!
    audio_chaos = (make_sine_wave(100, 0.1) + 
                     0.5 * make_sine_wave(250, 0.1) + 
                     0.3 * make_sine_wave(350, 0.1) + 
                     0.2 * make_sine_wave(450, 0.1))
                     
    res3 = extractor.process_frame(audio_chaos, m_100)
    print("Chaos Waveform:", res3['value'], res3['valid'], res3['rejection_reason'], res3['coverage_diagnostics'])
    
test_waveform()
