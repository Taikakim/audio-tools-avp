import os
import sys
import time
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import concurrent.futures
import numpy as np
import h5py
import librosa
import pedalboard
import mido
from tqdm import tqdm

LP_FILTERS = [
    ("LP 12 dB", 0.0303),
    ("LP 24 dB", 0.0606),
    ("LP Legacy Ladder", 0.0909),
    ("LP Vintage Ladder", 0.3030),
    ("LP OB-Xd 12 dB", 0.3333),
    ("LP OB-Xd 24 dB", 0.3636),
    ("LP K35", 0.3939),
    ("LP Diode Ladder", 0.4545),
    ("LP Cutoff Warp", 0.4848),
    ("LP Res Warp", 0.8485),
]

WAVESHAPER_TYPES = [
    ("Off", 0.0),
    ("Soft", 0.025),
    ("Hard", 0.050),
    ("Asymmetric", 0.075),
    ("Sine", 0.100),
    ("Fuzz", 0.575),
]

_plugin = None

def init_worker():
    global _plugin
    plugin_path = os.path.expanduser("~/.vst3/Surge XT.vst3")
    _plugin = pedalboard.load_plugin(plugin_path)
    _plugin.parameters["active_scene"].raw_value = 0.0
    _plugin.parameters["a_osc_1_type"].raw_value = 0.0
    _plugin.parameters["a_osc_1_octave"].raw_value = 0.5
    _plugin.parameters["fx_a1_fx_type"].raw_value = 0.3103 # Chorus
    _plugin.parameters["fx_a2_fx_type"].raw_value = 0.0345 # Delay
    _plugin.process([], 0.05, 44100, 2)

def make_mel_spec(audio, sr=44100, n_mels=128, n_fft=1024, hop_length=441):
    spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=20, fmax=16000
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_norm = np.clip((spec_db + 80.0) / 40.0 - 1.0, -1.0, 1.0)
    return spec_norm.astype(np.float32)

def render_sample(seed, duration=0.8, sample_rate=44100):
    global _plugin
    np.random.seed(seed)
    p = _plugin
    
    # 1. Filter selection across all 10 LP circuits
    filter_choice = int(np.random.randint(0, len(LP_FILTERS)))
    p.parameters["a_filter_1_type"].raw_value = LP_FILTERS[filter_choice][1]
    
    # 2. Pitch & Osc parameters
    midi_note = int(np.random.randint(28, 51)) # E1 to D3
    shape = float(np.random.uniform(0.0, 1.0))
    width = float(np.random.uniform(0.0, 1.0))
    sub_mix = float(np.random.uniform(0.0, 0.85))
    sync = float(np.random.uniform(0.0, 0.40)) if np.random.rand() < 0.3 else 0.0
    fm_depth = float(np.random.uniform(0.0, 0.45)) if np.random.rand() < 0.4 else 0.0
    
    # Unison spread (1 voice vs 2 voices)
    use_unison = 1.0 if np.random.rand() < 0.25 else 0.0
    unison_detune = float(np.random.uniform(0.05, 0.35)) if use_unison > 0.0 else 0.0
    p.parameters["a_osc_1_unison_voices"].raw_value = 0.05 if use_unison > 0.0 else 0.0
    p.parameters["a_osc_1_unison_detune"].raw_value = unison_detune

    # Filter Cutoff, Resonance, Keytracking
    cutoff = float(np.random.uniform(0.08, 0.92))
    resonance = float(np.random.uniform(0.0, 0.85))
    keytrack_raw = float(np.random.uniform(0.5, 1.0)) # 0% to 100%
    feg_amount = float(np.random.uniform(0.2, 0.95))
    
    feg_decay = float(np.random.uniform(0.03, 0.65))
    feg_sustain = float(np.random.uniform(0.0, 0.60))
    aeg_decay = float(np.random.uniform(0.05, 0.65))
    aeg_sustain = float(np.random.uniform(0.0, 0.80))
    aeg_release = float(np.random.uniform(0.01, 0.40))
    
    # Waveshaper / Drive
    if np.random.rand() < 0.45:
        ws_idx = int(np.random.randint(1, len(WAVESHAPER_TYPES)))
        ws_raw = WAVESHAPER_TYPES[ws_idx][1]
        drive_raw = float(np.random.uniform(0.50, 0.82)) # 0 dB to +15 dB
    else:
        ws_idx = 0
        ws_raw = 0.0
        drive_raw = 0.50

    p.parameters["a_waveshaper_type"].raw_value = ws_raw
    p.parameters["a_waveshaper_drive"].raw_value = drive_raw

    # FX: Chorus & Delay
    chorus_mix = float(np.random.uniform(0.1, 0.6)) if np.random.rand() < 0.35 else 0.0
    delay_mix = float(np.random.uniform(0.1, 0.45)) if np.random.rand() < 0.25 else 0.0
    delay_fb = float(np.random.uniform(0.1, 0.5)) if delay_mix > 0.0 else 0.0

    p.parameters["fx_a1_output_mix"].raw_value = chorus_mix
    p.parameters["fx_a2_output_mix"].raw_value = delay_mix
    p.parameters["fx_a2_feedback_eq_feedback"].raw_value = delay_fb

    p.parameters["a_osc_1_shape"].raw_value = shape
    p.parameters["a_osc_1_width_1"].raw_value = width
    p.parameters["a_osc_1_sub_mix"].raw_value = sub_mix
    p.parameters["a_osc_1_sync"].raw_value = sync
    p.parameters["a_fm_depth"].raw_value = fm_depth
    
    p.parameters["a_filter_1_cutoff"].raw_value = cutoff
    p.parameters["a_filter_1_resonance"].raw_value = resonance
    p.parameters["a_filter_1_keytrack"].raw_value = keytrack_raw
    p.parameters["a_filter_1_feg_mod_amount"].raw_value = feg_amount
    
    p.parameters["a_filter_eg_decay"].raw_value = feg_decay
    p.parameters["a_filter_eg_sustain"].raw_value = feg_sustain
    p.parameters["a_amp_eg_decay"].raw_value = aeg_decay
    p.parameters["a_amp_eg_sustain"].raw_value = aeg_sustain
    p.parameters["a_amp_eg_release"].raw_value = aeg_release
    
    note_dur = float(np.random.uniform(0.18, 0.45))
    midi_events = [
        mido.Message("note_on", note=midi_note, velocity=105, time=0.0),
        mido.Message("note_off", note=midi_note, velocity=0, time=note_dur),
    ]
    
    p.reset()
    audio = p.process(midi_events, duration=duration, sample_rate=sample_rate, num_channels=2)
    mono = np.mean(audio, axis=0).astype(np.float32)
    
    peak = np.max(np.abs(mono)) + 1e-7
    mono = mono / peak
    
    mel = make_mel_spec(mono, sr=sample_rate)
    
    # 23-dimensional normalized parameter vector
    param_vec = np.array([
        (midi_note - 28.0) / (50.0 - 28.0), # 0: MIDI note
        filter_choice / (len(LP_FILTERS) - 1), # 1: Filter circuit
        shape,                               # 2: Shape
        width,                               # 3: Width
        sub_mix,                             # 4: Sub-osc mix
        sync,                                # 5: Sync
        fm_depth,                            # 6: FM depth
        use_unison,                          # 7: Unison voices (0 or 1)
        unison_detune / 0.35,                # 8: Unison detune
        cutoff,                              # 9: Cutoff
        resonance,                           # 10: Resonance
        (keytrack_raw - 0.5) / 0.5,          # 11: Filter keytracking [0, 1]
        feg_amount,                          # 12: Filter EG mod amount
        feg_decay,                           # 13: Filter EG decay
        feg_sustain,                         # 14: Filter EG sustain
        aeg_decay,                           # 15: Amp EG decay
        aeg_sustain,                         # 16: Amp EG sustain
        aeg_release,                         # 17: Amp EG release
        ws_idx / (len(WAVESHAPER_TYPES) - 1),# 18: Waveshaper type
        (drive_raw - 0.50) / 0.32,           # 19: Drive amount [0, 1]
        chorus_mix,                          # 20: Chorus mix
        delay_mix,                           # 21: Delay mix
        delay_fb,                            # 22: Delay feedback
    ], dtype=np.float32)
    
    return mono, mel, param_vec

def generate_dataset(output_path, num_samples=200000, n_workers=12, batch_size=1000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"=== Generating {num_samples} Surge XT Bass Samples ===", flush=True)
    print(f"Destination: {output_path}", flush=True)
    print(f"Workers: {n_workers} CPU processes", flush=True)
    
    t0 = time.time()
    audio_len = 35280
    mel_shape = (128, 81)
    num_params = 23
    
    print(f"Sample shapes: Audio={audio_len} samples, Mel={mel_shape}, Params={num_params} dimensions", flush=True)
    
    with h5py.File(output_path, "w") as h5f:
        d_audio = h5f.create_dataset("audio", shape=(num_samples, audio_len), dtype=np.float32)
        d_mel = h5f.create_dataset("mel", shape=(num_samples, *mel_shape), dtype=np.float32)
        d_params = h5f.create_dataset("params", shape=(num_samples, num_params), dtype=np.float32)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as executor:
            for start_idx in range(0, num_samples, batch_size):
                b_t0 = time.time()
                end_idx = min(start_idx + batch_size, num_samples)
                cur_batch_size = end_idx - start_idx
                
                seeds = list(range(start_idx + 5000, end_idx + 5000))
                results = list(executor.map(render_sample, seeds))
                
                batch_audio = np.stack([r[0] for r in results], axis=0)
                batch_mel = np.stack([r[1] for r in results], axis=0)
                batch_params = np.stack([r[2] for r in results], axis=0)
                
                d_audio[start_idx:end_idx] = batch_audio
                d_mel[start_idx:end_idx] = batch_mel
                d_params[start_idx:end_idx] = batch_params
                
                b_dt = time.time() - b_t0
                total_done = end_idx
                pct = (total_done / num_samples) * 100.0
                rate = cur_batch_size / b_dt
                eta_s = (num_samples - total_done) / (rate + 1e-7)
                print(f"Batch {end_idx // batch_size:3d}/{(num_samples + batch_size - 1)//batch_size:3d}: {total_done:6d}/{num_samples} ({pct:5.1f}%) | {rate:.1f} samp/s | Elapsed: {time.time()-t0:.1f}s | ETA: {eta_s/60:.1f}m", flush=True)
                
    dt = time.time() - t0
    print(f"\nSuccessfully generated {num_samples} samples in {dt:.1f}s ({num_samples/dt:.1f} samples/sec)!", flush=True)
    print(f"Dataset saved at: {output_path}", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200000, help="Number of samples to generate")
    parser.add_argument("--out", type=str, default="/run/media/kim/Mantu/surge_dataset/surge_bass_200k.h5", help="Path to output HDF5")
    parser.add_argument("--workers", type=int, default=12, help="Number of CPU worker processes")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch write size")
    args = parser.parse_args()

    generate_dataset(args.out, num_samples=args.samples, n_workers=args.workers, batch_size=args.batch_size)
