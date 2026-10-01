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

from models import ResMLPInverter, FlowMatchingResMLP
from generate_bass_dataset import make_mel_spec, LP_FILTERS, WAVESHAPER_TYPES

class MultiScaleSTFTLoss(torch.nn.Module):
    def __init__(self, fft_sizes=[2048, 1024, 512, 256], hop_sizes=[512, 256, 128, 64], win_lengths=[2048, 1024, 512, 256]):
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

def evaluate_200k():
    device = torch.device("cpu")
    print(f"Loading 200k models on {device}...")
    
    # 1. 200k ResMLP
    g01_ckpt = "/run/media/kim/Mantu/surge_200k_models/G01_resmlp_deep_200k_best.pt"
    m_200k_resmlp = ResMLPInverter(param_dim=23, hidden_dim=512, num_layers=6).to(device)
    m_200k_resmlp.load_state_dict(torch.load(g01_ckpt, map_location=device)["model_state"])
    m_200k_resmlp.eval()

    # 2. 200k Deep Flow Champion
    g02_ckpt = "/run/media/kim/Mantu/surge_200k_models/G02_deepflow_200k_best.pt"
    m_200k_flow = FlowMatchingResMLP(param_dim=23, hidden_dim=512, num_layers=6).to(device)
    m_200k_flow.load_state_dict(torch.load(g02_ckpt, map_location=device)["model_state"])
    m_200k_flow.eval()
    
    print("All 200k models successfully loaded into memory.")
    
    # Surge setup
    plugin_path = os.path.expanduser("~/.vst3/Surge XT.vst3")
    synth = pedalboard.load_plugin(plugin_path)
    synth.parameters["active_scene"].raw_value = 0.0
    synth.parameters["a_osc_1_type"].raw_value = 0.0
    synth.parameters["a_osc_1_octave"].raw_value = 0.5
    synth.parameters["fx_a1_fx_type"].raw_value = 0.3103 # Chorus
    synth.parameters["fx_a2_fx_type"].raw_value = 0.0345 # Delay
    synth.process([], 0.05, 44100, 2)
    
    test_stems = [
        ("acid_303_pluck", "/run/media/kim/Mantu/surge_stem_inversion/targets/target_acid_pluck_a2.wav", 45, 0.22), # A2
        ("goa_disco_bass", "/run/media/kim/Mantu/surge_stem_inversion/targets/target_disco_bass_f2.wav", 41, 0.28), # F2
    ]
    
    out_dir = "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results"
    os.makedirs(out_dir, exist_ok=True)
    stft_loss_fn = MultiScaleSTFTLoss()
    
    summary = []
    
    for stem_name, wav_path, midi_note, note_dur in test_stems:
        print(f"\n{'='*70}\nEvaluating 200k Models on: {stem_name} (MIDI {midi_note})\n{'='*70}")
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        target_len = int(0.8 * sr)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        y = y / (np.max(np.abs(y)) + 1e-7)
        
        mel = make_mel_spec(y, sr=sr)
        mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)
        
        models_to_test = [
            ("200k_resmlp", m_200k_resmlp, False),
            ("200k_deepflow", m_200k_flow, True),
        ]
        
        rendered_outputs = {}
        for m_name, model, is_flow in models_to_test:
            t0 = time.time()
            with torch.no_grad():
                if is_flow:
                    pred_params = model.sample(mel_t, num_steps=20).squeeze(0).cpu().numpy()
                else:
                    pred_params = model(mel_t).squeeze(0).cpu().numpy()
            lat_ms = (time.time() - t0) * 1000.0
            
            # Map 23 params
            filter_idx = int(np.clip(round(pred_params[1] * (len(LP_FILTERS) - 1)), 0, len(LP_FILTERS) - 1))
            filter_name, filter_raw = LP_FILTERS[filter_idx]
            
            shape = float(pred_params[2])
            width = float(pred_params[3])
            sub_mix = float(pred_params[4])
            sync = float(pred_params[5])
            fm_depth = float(pred_params[6])
            
            use_unison = pred_params[7] > 0.5
            unison_detune = float(pred_params[8] * 0.35) if use_unison else 0.0
            
            cutoff = float(pred_params[9])
            resonance = float(pred_params[10])
            keytrack_raw = float(0.5 + pred_params[11] * 0.5)
            
            feg_amount = float(pred_params[12])
            feg_decay = float(pred_params[13])
            feg_sustain = float(pred_params[14])
            aeg_decay = float(pred_params[15])
            aeg_sustain = float(pred_params[16])
            aeg_release = float(pred_params[17])
            
            ws_idx = int(np.clip(round(pred_params[18] * (len(WAVESHAPER_TYPES) - 1)), 0, len(WAVESHAPER_TYPES) - 1))
            ws_name, ws_raw = WAVESHAPER_TYPES[ws_idx]
            drive_raw = float(0.50 + pred_params[19] * 0.32)
            
            chorus_mix = float(pred_params[20])
            delay_mix = float(pred_params[21])
            delay_fb = float(pred_params[22])
            
            synth.parameters["a_filter_1_type"].raw_value = filter_raw
            synth.parameters["a_osc_1_shape"].raw_value = shape
            synth.parameters["a_osc_1_width_1"].raw_value = width
            synth.parameters["a_osc_1_sub_mix"].raw_value = sub_mix
            synth.parameters["a_osc_1_sync"].raw_value = sync
            synth.parameters["a_fm_depth"].raw_value = fm_depth
            
            synth.parameters["a_osc_1_unison_voices"].raw_value = 0.05 if use_unison else 0.0
            synth.parameters["a_osc_1_unison_detune"].raw_value = unison_detune
            
            synth.parameters["a_filter_1_cutoff"].raw_value = cutoff
            synth.parameters["a_filter_1_resonance"].raw_value = resonance
            synth.parameters["a_filter_1_keytrack"].raw_value = keytrack_raw
            synth.parameters["a_filter_1_feg_mod_amount"].raw_value = feg_amount
            
            synth.parameters["a_filter_eg_decay"].raw_value = feg_decay
            synth.parameters["a_filter_eg_sustain"].raw_value = feg_sustain
            synth.parameters["a_amp_eg_decay"].raw_value = aeg_decay
            synth.parameters["a_amp_eg_sustain"].raw_value = aeg_sustain
            synth.parameters["a_amp_eg_release"].raw_value = aeg_release
            
            synth.parameters["a_waveshaper_type"].raw_value = ws_raw
            synth.parameters["a_waveshaper_drive"].raw_value = drive_raw
            
            synth.parameters["fx_a1_output_mix"].raw_value = chorus_mix
            synth.parameters["fx_a2_output_mix"].raw_value = delay_mix
            synth.parameters["fx_a2_feedback_eq_feedback"].raw_value = delay_fb
            
            synth.reset()
            events = [
                mido.Message("note_on", note=midi_note, velocity=105, time=0.0),
                mido.Message("note_off", note=midi_note, velocity=0, time=note_dur),
            ]
            audio = synth.process(events, duration=0.8, sample_rate=sr, num_channels=2)
            mono_out = np.mean(audio, axis=0)
            mono_out = mono_out / (np.max(np.abs(mono_out)) + 1e-7)
            
            out_wav = f"{out_dir}/{stem_name}_{m_name}.wav"
            sf.write(out_wav, mono_out, sr)
            
            # Save preset dict
            preset_dict = {
                "stem": stem_name,
                "model": m_name,
                "latency_ms": lat_ms,
                "filter_circuit": filter_name,
                "waveshaper": ws_name,
                "drive_raw": drive_raw,
                "use_unison": bool(use_unison),
                "unison_detune": unison_detune,
                "parameters": {
                    "shape": shape,
                    "width": width,
                    "sub_mix": sub_mix,
                    "sync": sync,
                    "fm_depth": fm_depth,
                    "cutoff": cutoff,
                    "resonance": resonance,
                    "keytrack": keytrack_raw,
                    "feg_amount": feg_amount,
                    "feg_decay": feg_decay,
                    "feg_sustain": feg_sustain,
                    "aeg_decay": aeg_decay,
                    "aeg_sustain": aeg_sustain,
                    "aeg_release": aeg_release,
                    "chorus_mix": chorus_mix,
                    "delay_mix": delay_mix,
                    "delay_fb": delay_fb,
                }
            }
            with open(f"{out_dir}/{stem_name}_{m_name}_preset.json", "w") as f:
                json.dump(preset_dict, f, indent=2)
            
            stft_l = stft_loss_fn(mono_out, y)
            sc = float(np.mean(librosa.feature.spectral_centroid(y=mono_out, sr=sr)))
            
            rendered_outputs[m_name] = {
                "audio": mono_out,
                "filter": filter_name,
                "waveshaper": ws_name,
                "stft_loss": stft_l,
                "centroid": sc,
                "latency_ms": lat_ms,
            }
            print(f"{m_name:15s} | Filter: {filter_name:18s} | Drive: {ws_name:10s} | STFT Loss: {stft_l:.3f} | Centroid: {sc:6.1f} Hz | Latency: {lat_ms:.1f}ms")
            
        sc_target = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        print(f"Target Centroid: {sc_target:.1f} Hz")
        
        # Save comparison plot
        fig, axs = plt.subplots(3, 2, figsize=(14, 8))
        t_axis = np.linspace(0, len(y)/sr, len(y))
        
        plot_items = [("Target Stem", y, "royalblue", 0.0)] + [
            (f"{k} ({v['filter']}, {v['waveshaper']})", v["audio"], "crimson", v["stft_loss"]) for k, v in rendered_outputs.items()
        ]
        
        for idx, (title, sig, col, loss) in enumerate(plot_items):
            axs[idx, 0].plot(t_axis, sig, color=col, alpha=0.8)
            axs[idx, 0].set_title(f"{title} - Waveform (STFT={loss:.2f})", fontsize=9)
            axs[idx, 0].set_ylim(-1.05, 1.05)
            axs[idx, 0].grid(True, alpha=0.3)
            
            S = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=1024, hop_length=128)), ref=np.max)
            librosa.display.specshow(S, sr=sr, hop_length=128, x_axis='time', y_axis='hz', ax=axs[idx, 1], cmap='magma', vmin=-60, vmax=0)
            axs[idx, 1].set_title(f"{title} - Spectrogram", fontsize=9)
            axs[idx, 1].set_ylim(0, 5000)
            
        plt.tight_layout()
        plot_p = f"{out_dir}/{stem_name}_200k_comparison.png"
        plt.savefig(plot_p, dpi=150)
        plt.close()
        print(f"Saved comparison figure: {plot_p}")

    print("\n200k Evaluation Complete!")

if __name__ == "__main__":
    evaluate_200k()
