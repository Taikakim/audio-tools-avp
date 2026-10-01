import os
import sys
import time
import json
import numpy as np
import torch
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pedalboard
import mido
from scipy.optimize import differential_evolution

from models import ResMLPInverter, FlowMatchingResMLP
from generate_bass_dataset import make_mel_spec, LP_FILTERS, WAVESHAPER_TYPES

class MultiScaleSTFTLoss(torch.nn.Module):
    def __init__(self, fft_sizes=[1024, 512, 256], hop_sizes=[256, 128, 64], win_lengths=[1024, 512, 256]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)
            
        min_len = min(x.shape[1], y.shape[1])
        x = x[:, :min_len]
        y = y[:, :min_len]
            
        loss = 0.0
        for n_fft, hop_length, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.hann_window(win_length).to(x.device)
            X = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
            Y = torch.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
            
            X_mag = torch.abs(X) + 1e-7
            Y_mag = torch.abs(Y) + 1e-7
            
            sc_loss = torch.norm(Y_mag - X_mag, p="fro") / (torch.norm(Y_mag, p="fro") + 1e-7)
            log_loss = torch.mean(torch.abs(torch.log(Y_mag) - torch.log(X_mag)))
            loss += (sc_loss + log_loss).item()
            
        return loss / len(self.fft_sizes)

def match_single_note():
    target_path = "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results/untitled.wav"
    out_dir = "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results/untitled_note_match"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading single note target: {target_path}")
    y_target, sr = sf.read(target_path)
    if y_target.ndim > 1:
        y_target = np.mean(y_target, axis=1)
        
    duration = len(y_target) / sr
    print(f"Target duration: {duration*1000:.1f}ms ({len(y_target)} samples), SR: {sr}")
    
    # Normalize target amplitude
    y_target = y_target / (np.max(np.abs(y_target)) + 1e-7)
    target_tensor = torch.from_numpy(y_target.astype(np.float32))
    
    midi_note = 36 # C2
    note_dur = min(0.08, duration * 0.8)
    
    loss_fn = MultiScaleSTFTLoss()
    
    # 1. Neural Inversion with 200k Deep Flow & ResMLP
    device = torch.device("cpu")
    print("Running 200k Neural Models on single note...")
    
    resmlp = ResMLPInverter(param_dim=23, hidden_dim=512, num_layers=6).to(device)
    resmlp.load_state_dict(torch.load("/run/media/kim/Mantu/surge_200k_models/G01_resmlp_deep_200k_best.pt", map_location=device)["model_state"])
    resmlp.eval()

    flow = FlowMatchingResMLP(param_dim=23, hidden_dim=512, num_layers=6).to(device)
    flow.load_state_dict(torch.load("/run/media/kim/Mantu/surge_200k_models/G02_deepflow_200k_best.pt", map_location=device)["model_state"])
    flow.eval()
    
    # Pad to 35280 for mel input
    y_pad = np.pad(y_target, (0, max(0, 35280 - len(y_target))))[:35280]
    mel = torch.from_numpy(make_mel_spec(y_pad, sr=sr)).unsqueeze(0)
    
    with torch.no_grad():
        p_res = resmlp(mel).squeeze(0).numpy()
        p_flow = flow.sample(mel, num_steps=20).squeeze(0).numpy()

    # Initialize Surge XT
    plugin_path = os.path.expanduser("~/.vst3/Surge XT.vst3")
    synth = pedalboard.load_plugin(plugin_path)
    synth.parameters["active_scene"].raw_value = 0.0
    synth.parameters["a_osc_1_type"].raw_value = 0.0
    synth.parameters["a_osc_1_octave"].raw_value = 0.5
    synth.parameters["fx_a1_fx_type"].raw_value = 0.3103
    synth.parameters["fx_a2_fx_type"].raw_value = 0.0345
    synth.process([], 0.05, sr, 2)
    
    def render_params(p, dur):
        filter_idx = int(np.clip(round(p[1] * (len(LP_FILTERS) - 1)), 0, len(LP_FILTERS) - 1))
        filter_name, filter_raw = LP_FILTERS[filter_idx]
        
        synth.parameters["a_filter_1_type"].raw_value = filter_raw
        synth.parameters["a_osc_1_shape"].raw_value = float(p[2])
        synth.parameters["a_osc_1_width_1"].raw_value = float(p[3])
        synth.parameters["a_osc_1_sub_mix"].raw_value = float(p[4])
        synth.parameters["a_osc_1_sync"].raw_value = float(p[5])
        synth.parameters["a_fm_depth"].raw_value = float(p[6])
        
        use_unison = p[7] > 0.5
        synth.parameters["a_osc_1_unison_voices"].raw_value = 0.05 if use_unison else 0.0
        synth.parameters["a_osc_1_unison_detune"].raw_value = float(p[8] * 0.35) if use_unison else 0.0
        
        synth.parameters["a_filter_1_cutoff"].raw_value = float(p[9])
        synth.parameters["a_filter_1_resonance"].raw_value = float(p[10])
        synth.parameters["a_filter_1_keytrack"].raw_value = float(0.5 + p[11] * 0.5)
        
        synth.parameters["a_filter_1_feg_mod_amount"].raw_value = float(p[12])
        synth.parameters["a_filter_eg_decay"].raw_value = float(p[13])
        synth.parameters["a_filter_eg_sustain"].raw_value = float(p[14])
        
        synth.parameters["a_amp_eg_decay"].raw_value = float(p[15])
        synth.parameters["a_amp_eg_sustain"].raw_value = float(p[16])
        synth.parameters["a_amp_eg_release"].raw_value = float(p[17])
        
        ws_idx = int(np.clip(round(p[18] * (len(WAVESHAPER_TYPES) - 1)), 0, len(WAVESHAPER_TYPES) - 1))
        synth.parameters["a_waveshaper_type"].raw_value = WAVESHAPER_TYPES[ws_idx][1]
        synth.parameters["a_waveshaper_drive"].raw_value = float(0.50 + p[19] * 0.32)
        
        synth.parameters["fx_a1_output_mix"].raw_value = float(p[20])
        synth.parameters["fx_a2_output_mix"].raw_value = float(p[21])
        synth.parameters["fx_a2_feedback_eq_feedback"].raw_value = float(p[22])
        
        synth.reset()
        events = [
            mido.Message("note_on", note=midi_note, velocity=105, time=0.0),
            mido.Message("note_off", note=midi_note, velocity=0, time=note_dur),
        ]
        audio = synth.process(events, duration=dur, sample_rate=sr, num_channels=2)
        mono = np.mean(audio, axis=0)
        mono = mono / (np.max(np.abs(mono)) + 1e-7)
        return mono, filter_name, synth.raw_state
        
    flow_audio, flow_filter, flow_vst = render_params(p_flow, duration)
    res_audio, res_filter, res_vst = render_params(p_res, duration)
    
    stft_flow = loss_fn(flow_audio, y_target)
    stft_res = loss_fn(res_audio, y_target)
    print(f"Deep Flow Inversion | Filter: {flow_filter:18s} | STFT Loss: {stft_flow:.3f}")
    print(f"ResMLP Inversion    | Filter: {res_filter:18s} | STFT Loss: {stft_res:.3f}")
    
    # 2. Run Closed-Loop Differential Evolution on untitled.wav (Numerical Ground Truth)
    print("\nStarting Closed-Loop Differential Evolution search on untitled.wav...")
    t_de_start = time.time()
    
    # Candidate filter circuits to evaluate in DE
    # Top candidates: LP OB-Xd 12dB (0.3333), LP Vintage Ladder (0.3030), LP Legacy Ladder (0.0909), LP 24dB (0.0606)
    candidate_filters = [
        ("LP OB-Xd 12 dB", 0.3333),
        ("LP Vintage Ladder", 0.3030),
        ("LP Legacy Ladder", 0.0909),
    ]
    
    best_de_loss = float("inf")
    best_de_params = None
    best_de_filter = None
    best_de_audio = None
    best_de_vst = None
    
    for f_name, f_raw in candidate_filters:
        print(f"  Testing circuit: {f_name}...")
        synth.parameters["a_filter_1_type"].raw_value = f_raw
        synth.parameters["a_waveshaper_type"].raw_value = 0.0 # Clean waveshaper
        synth.parameters["a_waveshaper_drive"].raw_value = 0.50
        synth.parameters["fx_a1_output_mix"].raw_value = 0.0
        synth.parameters["fx_a2_output_mix"].raw_value = 0.0
        synth.parameters["a_filter_1_keytrack"].raw_value = 0.77
        
        # 8 continuous parameters to optimize
        # 0: shape (0.0=saw to 1.0=sqr)
        # 1: sub_mix (0 to 0.5)
        # 2: cutoff (0.1 to 0.8)
        # 3: resonance (0.05 to 0.75)
        # 4: feg_mod_amount (0.5 to 0.95)
        # 5: feg_decay (0.05 to 0.6)
        # 6: amp_decay (0.1 to 0.7)
        # 7: amp_release (0.0 to 0.5)
        bounds = [
            (0.0, 1.0),
            (0.0, 0.5),
            (0.1, 0.75),
            (0.05, 0.75),
            (0.5, 0.95),
            (0.05, 0.50),
            (0.15, 0.60),
            (0.0, 0.40),
        ]
        
        def obj_fn(x):
            synth.parameters["a_osc_1_shape"].raw_value = float(x[0])
            synth.parameters["a_osc_1_sub_mix"].raw_value = float(x[1])
            synth.parameters["a_filter_1_cutoff"].raw_value = float(x[2])
            synth.parameters["a_filter_1_resonance"].raw_value = float(x[3])
            synth.parameters["a_filter_1_feg_mod_amount"].raw_value = float(x[4])
            synth.parameters["a_filter_eg_decay"].raw_value = float(x[5])
            synth.parameters["a_amp_eg_decay"].raw_value = float(x[6])
            synth.parameters["a_amp_eg_release"].raw_value = float(x[7])
            
            synth.reset()
            events = [
                mido.Message("note_on", note=midi_note, velocity=105, time=0.0),
                mido.Message("note_off", note=midi_note, velocity=0, time=note_dur),
            ]
            audio = synth.process(events, duration=duration, sample_rate=sr, num_channels=2)
            mono = np.mean(audio, axis=0)
            mono = mono / (np.max(np.abs(mono)) + 1e-7)
            return loss_fn(mono, y_target)
            
        res = differential_evolution(obj_fn, bounds=bounds, maxiter=8, popsize=6, seed=42)
        print(f"    Circuit {f_name}: Best STFT Loss = {res.fun:.3f}")
        
        if res.fun < best_de_loss:
            best_de_loss = res.fun
            best_de_params = res.x
            best_de_filter = f_name
            # Re-render best audio
            synth.parameters["a_osc_1_shape"].raw_value = float(res.x[0])
            synth.parameters["a_osc_1_sub_mix"].raw_value = float(res.x[1])
            synth.parameters["a_filter_1_cutoff"].raw_value = float(res.x[2])
            synth.parameters["a_filter_1_resonance"].raw_value = float(res.x[3])
            synth.parameters["a_filter_1_feg_mod_amount"].raw_value = float(res.x[4])
            synth.parameters["a_filter_eg_decay"].raw_value = float(res.x[5])
            synth.parameters["a_amp_eg_decay"].raw_value = float(res.x[6])
            synth.parameters["a_amp_eg_release"].raw_value = float(res.x[7])
            synth.reset()
            events = [
                mido.Message("note_on", note=midi_note, velocity=105, time=0.0),
                mido.Message("note_off", note=midi_note, velocity=0, time=note_dur),
            ]
            audio = synth.process(events, duration=duration, sample_rate=sr, num_channels=2)
            mono = np.mean(audio, axis=0)
            best_de_audio = mono / (np.max(np.abs(mono)) + 1e-7)
            best_de_vst = synth.raw_state

    de_time = time.time() - t_de_start
    print(f"\nDifferential Evolution finished in {de_time:.1f}s!")
    print(f"Champion Circuit: {best_de_filter} | Best STFT Loss: {best_de_loss:.3f}")
    print(f"Optimized Parameters: shape={best_de_params[0]:.3f}, sub={best_de_params[1]:.3f}, cutoff={best_de_params[2]:.3f}, res={best_de_params[3]:.3f}, feg_mod={best_de_params[4]:.3f}, feg_decay={best_de_params[5]:.3f}, amp_decay={best_de_params[6]:.3f}, amp_rel={best_de_params[7]:.3f}")

    # Save audio files
    sf.write(f"{out_dir}/untitled_target.wav", y_target, sr)
    sf.write(f"{out_dir}/untitled_de_match.wav", best_de_audio, sr)
    sf.write(f"{out_dir}/untitled_deepflow_match.wav", flow_audio, sr)
    sf.write(f"{out_dir}/untitled_resmlp_match.wav", res_audio, sr)

    # Save VST presets
    with open(f"{out_dir}/untitled_de_match.vstpreset", "wb") as f:
        f.write(best_de_vst)
    with open(f"{out_dir}/untitled_deepflow.vstpreset", "wb") as f:
        f.write(flow_vst)

    # Save Comparison Spectrogram & Waveform Plot
    fig, axs = plt.subplots(4, 2, figsize=(14, 10))
    t_axis = np.linspace(0, duration*1000, len(y_target))

    plot_rows = [
        ("Target (untitled.wav)", y_target, "royalblue", 0.0),
        (f"DE Match ({best_de_filter}, STFT={best_de_loss:.2f})", best_de_audio, "crimson", best_de_loss),
        (f"Deep Flow 200k ({flow_filter}, STFT={stft_flow:.2f})", flow_audio, "forestgreen", stft_flow),
        (f"ResMLP 200k ({res_filter}, STFT={stft_res:.2f})", res_audio, "darkorange", stft_res),
    ]

    for idx, (title, sig, col, loss_val) in enumerate(plot_rows):
        axs[idx, 0].plot(t_axis, sig[:len(t_axis)], color=col, alpha=0.85)
        axs[idx, 0].set_title(f"{title} - Waveform", fontsize=10)
        axs[idx, 0].set_ylim(-1.05, 1.05)
        axs[idx, 0].set_xlabel("Time (ms)")
        axs[idx, 0].grid(True, alpha=0.3)

        S = librosa.amplitude_to_db(np.abs(librosa.stft(sig[:len(y_target)], n_fft=512, hop_length=64)), ref=np.max)
        librosa.display.specshow(S, sr=sr, hop_length=64, x_axis='time', y_axis='hz', ax=axs[idx, 1], cmap='magma', vmin=-50, vmax=0)
        axs[idx, 1].set_title(f"{title} - Spectrogram", fontsize=10)
        axs[idx, 1].set_ylim(0, 4500)

    plt.tight_layout()
    plot_p = f"{out_dir}/untitled_match_comparison.png"
    plt.savefig(plot_p, dpi=150)
    plt.close()
    print(f"\nSaved comparison plot to {plot_p}!")
    print(f"All files saved to {out_dir}/")

if __name__ == "__main__":
    match_single_note()
