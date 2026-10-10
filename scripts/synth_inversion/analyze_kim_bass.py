import os, glob
import pedalboard
import numpy as np

plugin = pedalboard.load_plugin(os.path.expanduser("~/.vst3/Surge XT.vst3"))

os.system("mkdir -p kim_bass_patches && unzip -o '/home/kim/Documents/Surge XT/Patches/AI Inversions/kim_bass_.zip' -d kim_bass_patches > /dev/null")
fxps = glob.glob("kim_bass_patches/*.fxp")

all_params = {}
for fxp in fxps:
    try:
        plugin.load_preset(fxp)
    except Exception as e:
        print(f"Failed to load {fxp}: {e}")
        continue
        
    for k, p in plugin.parameters.items():
        if k not in all_params:
            all_params[k] = []
        all_params[k].append(p.raw_value)

plugin2 = pedalboard.load_plugin(os.path.expanduser("~/.vst3/Surge XT.vst3"))
plugin2.parameters["active_scene"].raw_value = 0.0
plugin2.parameters["a_osc_1_type"].raw_value = 0.0

varied = []
for k, vals in all_params.items():
    if k not in plugin2.parameters: continue
    init_val = plugin2.parameters[k].raw_value
    min_v = np.min(vals)
    max_v = np.max(vals)
    if min_v != max_v or min_v != init_val:
        if k.startswith("a_") or k.startswith("fx_"):
            varied.append((k, min_v, max_v, init_val))

for k, min_v, max_v, init_v in sorted(varied):
    print(f"{k}: min={min_v:.4f}, max={max_v:.4f}, init={init_v:.4f}")
