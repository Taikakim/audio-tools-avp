#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Test for V8 DAW Export Bundle."""

import os
import sys
from pathlib import Path
import numpy as np
import pedalboard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from daw_export_v8 import create_daw_automation_midi, export_v8_daw_bundle
import surge_spec_v5 as S5

def main():
    print("Testing create_daw_automation_midi...")
    lanes = np.random.uniform(0.1, 0.9, (4, 43)).astype(np.float32)
    mid = create_daw_automation_midi(lanes, bpm=140.0)
    assert len(mid.tracks[0]) > 40, f"Expected >40 messages, got {len(mid.tracks[0])}"
    print(f"MIDI generation passed! Total messages: {len(mid.tracks[0])}")

    test_out = Path("/tmp/test_v8_bundle/lead_01")
    test_out.parent.mkdir(parents=True, exist_ok=True)

    print("Testing export_v8_daw_bundle with Surge XT...")
    synth = pedalboard.load_plugin(os.path.expanduser(S5.DEFAULT_PLUGIN_PATH))
    static_params = np.full(59, 0.5, dtype=np.float32)
    bundle = export_v8_daw_bundle(synth, static_params, lanes, test_out)

    for k, p in bundle.items():
        assert os.path.exists(p), f"Artifact {k} missing: {p}"
        print(f"  - {k}: {p} (size {os.path.getsize(p)} bytes)")

    print("All DAW export tests passed!")

if __name__ == "__main__":
    main()
