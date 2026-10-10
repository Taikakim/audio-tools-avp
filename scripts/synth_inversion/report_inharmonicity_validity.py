import argparse
import pprint
from train_realistic_bass_v3 import RealisticOnlineDataset, DEFAULT_PLUGIN_PATH
from inharmonicity_target import InharmonicityConfig, RejectionStats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifold", default="/run/media/kim/Mantu2/surge_200k_models/real_bass_manifold_v3.npz")
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN_PATH)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    cfg = InharmonicityConfig.from_json(args.config)
    ds = RealisticOnlineDataset(args.manifold, args.plugin, length=args.n, random_patch_frac=0.0, inharm_cfg=cfg)
    
    stats = RejectionStats()
    for i in range(args.n):
        # We need to feed RejectionStats, but __getitem__ only returns torch tensors and valid.
        # We'll use the extractor directly here to feed stats.
        patch, vec, midi_note, note_dur = ds._prior.sample_patch_and_midi() if ds._prior else ds[i] # Hack to init ds
        if ds._synth is None:
            _ = ds[0] # trigger init
        
        patch, vec, midi_note, note_dur = ds._prior.sample_patch_and_midi()
        import surge_spec_v3
        audio = surge_spec_v3.render_patch(ds._synth, patch, midi_note, note_dur)
        
        res = ds._extractor.process_phrase(audio, midi_note)
        stats.add(res, midi_note)
        
        # also run process_window directly to see what failed
        spans = surge_spec_v3.phrase_note_spans(midi_note)
        from inharmonicity_target import note_windows
        wins = note_windows(spans, cfg)
        for w in wins:
            if w is not None:
                wr = ds._extractor.process_window(audio[w["start_sample"]:w["end_sample"]], w["f0_hz"])
                stats.reasons[f"win_{wr.get('rejection_reason')}"] = stats.reasons.get(f"win_{wr.get('rejection_reason')}", 0) + 1
                if wr["valid"]:
                    print(f"val={wr['value']:.5f}")
        
    pprint.pprint(stats.summary())

if __name__ == "__main__":
    main()
