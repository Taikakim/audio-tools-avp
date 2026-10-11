import argparse
import json
import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from realistic_bass_prior_v3 import MANIFOLD_PATH, RealisticBassPrior
import surge_spec_v3
from synth_jepa_model import SynthJEPA
from audio_utils import make_mel_spec

def get_za(audio, model, device):
    mel = torch.tensor(make_mel_spec(audio)).unsqueeze(0).to(device)
    with torch.no_grad():
        za = model.encode_audio(mel).cpu().numpy()[0]
    return za

def build_dataset(prior, n_samples, labels_list, plugin, model, device):
    labels_dict = {l["index"]: l for l in labels_list}
    X, y = [], []
    for i in range(n_samples):
        patch, _, midi_note, note_dur = prior.sample_patch_and_midi()
        if i in labels_dict:
            # Re-render to ensure same audio, or we could have just extracted za
            # Actually, `sample_patch_and_midi` produces identical patches because of the seed
            # But the midi_note and note_dur are exactly the same
            audio = surge_spec_v3.render_patch(plugin, patch, midi_note, note_dur)
            za = get_za(audio, model, device)
            X.append(za)
            y.append(labels_dict[i]["value"])
    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifold", default=MANIFOLD_PATH)
    parser.add_argument("--stats", default="label_stats.json")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--ff_dim", type=int, default=1152)
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with open(args.stats, "r") as f:
        stats = json.load(f)
        
    model = SynthJEPA(inharmonicity_head=True, ff_dim=args.ff_dim).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["jepa_state_dict"], strict=False)
    model.eval()
    
    plugin = surge_spec_v3.init_synth()
    
    np.random.seed(args.seed)
    prior_train = RealisticBassPrior(args.manifold, split="train", jitter=0.0)
    X_train, y_train = build_dataset(prior_train, args.n, stats["train"]["labels"], plugin, model, device)
    
    prior_val = RealisticBassPrior(args.manifold, split="val", jitter=0.0)
    X_val, y_val = build_dataset(prior_val, len(prior_val.params), stats["val"]["labels"], plugin, model, device)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    
    y_pred_train = ridge.predict(X_train)
    y_pred_val = ridge.predict(X_val)
    
    print(f"Train MSE: {mean_squared_error(y_train, y_pred_train):.5f}")
    print(f"Train R^2: {r2_score(y_train, y_pred_train):.5f}")
    print(f"Val MSE: {mean_squared_error(y_val, y_pred_val):.5f}")
    print(f"Val R^2: {r2_score(y_val, y_pred_val):.5f}")

if __name__ == "__main__":
    main()
