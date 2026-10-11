import os, json, time, torch
import numpy as np
from pathlib import Path
import surge_spec_v3
from realistic_bass_prior_v3 import RealisticBassPrior
from synth_jepa_model import SynthJEPA
from synth_jepa_search_v3 import WelfordNormalizer, SynthJEPASearcher
from audio_utils import make_mel_spec
import soundfile as sf

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path("a1_clips")
    out_dir.mkdir(exist_ok=True)
    
    with open("scripts/synth_inversion/label_stats.json", "r") as f:
        stats = json.load(f)
    train_labels = [l["value"] for l in stats["train"]["labels"]]
    p95 = float(np.percentile(train_labels, 95))
    print(f"95th percentile inharmonicity: {p95:.5f}")
    with open("scripts/synth_inversion/inharmonic_quantile.json", "w") as f:
        json.dump({"p95": p95}, f)
        
    model = SynthJEPA(inharmonicity_head=True, ff_dim=1152).to(device)
    ckpt = torch.load("/run/media/kim/Mantu2/surge_200k_models/real_bass_v3_inharm/checkpoint_latest.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["jepa_state_dict"])
    model.eval()
    
    normalizer = WelfordNormalizer(128)
    normalizer.frozen_mean = ckpt["normalizer"]["mean"].cpu()
    normalizer.frozen_std = ckpt["normalizer"]["std"].cpu()
    normalizer.frozen = True
    
    plugin = surge_spec_v3.init_synth()
    prior = RealisticBassPrior("/run/media/kim/Mantu2/surge_200k_models/real_bass_manifold_v3.npz", split="val", jitter=0.0)
    
    searcher = SynthJEPASearcher(model, normalizer=normalizer, device=device)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    for i in range(32):
        patch, vec, note, dur = prior.sample_patch_and_midi()
        audio = surge_spec_v3.render_patch(plugin, patch, note, dur)
        mel = torch.tensor(make_mel_spec(audio)).unsqueeze(0).to(device)
        
        t0 = time.time()
        res_zero = searcher.search(mel, midi_note=note, inharm_target=0.0, inharm_weight=1.0)
        p_zero = res_zero["patch"]
        p_zero["midi_note"] = note
        a_zero = surge_spec_v3.render_patch(plugin, p_zero, note, dur)
        sf.write(out_dir / f"{i:02d}_target0.wav", a_zero, surge_spec_v3.SAMPLE_RATE)
        
        res_p95 = searcher.search(mel, midi_note=note, inharm_target=p95, inharm_weight=1.0)
        p_p95 = res_p95["patch"]
        p_p95["midi_note"] = note
        a_p95 = surge_spec_v3.render_patch(plugin, p_p95, note, dur)
        sf.write(out_dir / f"{i:02d}_target95.wav", a_p95, surge_spec_v3.SAMPLE_RATE)
        
        print(f"[{i:02d}/32] rendered {note} in {time.time()-t0:.1f}s")
        
    print("Done! Clips in a1_clips/")

if __name__ == "__main__":
    main()
