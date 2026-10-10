import os
import pedalboard

plugin = pedalboard.load_plugin(os.path.expanduser("~/.vst3/Surge XT.vst3"))

keys = list(plugin.parameters.keys())
for k in keys:
    if "osc_" in k or "filter" in k or "mixer" in k or "waveshaper" in k or "fm" in k or "noise" in k:
        print(k)
