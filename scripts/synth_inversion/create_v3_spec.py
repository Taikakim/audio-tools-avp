import re
with open("surge_spec.py", "r") as f:
    code = f.read()

# Modify PARAM_NAMES
new_params = """
    "osc_1_width_2",
    "osc_2_shape",
    "osc_2_width_1",
    "osc_2_width_2",
    "osc_3_shape",
    "osc_3_width_1",
    "osc_3_width_2",
    "osc_1_octave",
    "osc_2_octave",
    "osc_3_octave",
    "noise_color",
    "osc_1_volume",
    "osc_2_volume",
    "osc_3_volume",
    "ring_1x2_volume",
    "ring_2x3_volume",
    "noise_volume",
    "filter_balance",
    "filter_configuration",
    "fm_routing",
    "filter_2_type",
    "filter_2_cutoff",
    "filter_2_resonance",
    "filter_2_keytrack",
    "filter_2_feg_amount",
"""
code = code.replace('"delay_fb",         # 22\n]', '"delay_fb",         # 22\n' + new_params + ']')

# Modify DOMAIN_WEIGHTS
code = code.replace('"aeg_release": 1.5,\n}', '"aeg_release": 1.5,\n    "filter_2_cutoff": 1.5,\n}')

# Update patch_to_vector
ptv = """
        patch["osc_1_width_2"],
        patch["osc_2_shape"],
        patch["osc_2_width_1"],
        patch["osc_2_width_2"],
        patch["osc_3_shape"],
        patch["osc_3_width_1"],
        patch["osc_3_width_2"],
        patch["osc_1_octave"],
        patch["osc_2_octave"],
        patch["osc_3_octave"],
        patch["noise_color"],
        patch["osc_1_volume"],
        patch["osc_2_volume"],
        patch["osc_3_volume"],
        patch["ring_1x2_volume"],
        patch["ring_2x3_volume"],
        patch["noise_volume"],
        patch["filter_balance"],
        patch["filter_configuration"],
        patch["fm_routing"],
        patch["filter_2_type"],
        patch["filter_2_cutoff"],
        patch["filter_2_resonance"],
        (patch["filter_2_keytrack"] - 0.5) / 0.5,
        patch["filter_2_feg_amount"],
    ], dtype=np.float32)
"""
code = code.replace('        patch["delay_fb"],\n    ], dtype=np.float32)', '        patch["delay_fb"],\n' + ptv)

# Categorical mapping. 
# In V2: CATEGORICAL = {"filter_type": len(LP_FILTERS), "unison": 2, "waveshaper_type": len(WAVESHAPER_TYPES)}
# In V3, we add filter_2_type, filter_configuration, fm_routing. We need sizes for them.
# filter_configuration has 7 types. fm_routing has 4 types. filter_2_type has len(LP_FILTERS).
code = code.replace(
    'CATEGORICAL = {"filter_type": len(LP_FILTERS), "unison": 2, "waveshaper_type": len(WAVESHAPER_TYPES)}',
    'CATEGORICAL = {"filter_type": len(LP_FILTERS), "unison": 2, "waveshaper_type": len(WAVESHAPER_TYPES), "filter_2_type": len(LP_FILTERS), "filter_configuration": 7, "fm_routing": 4}'
)

with open("surge_spec_v3.py", "w") as f:
    f.write(code)
print("Created surge_spec_v3.py")
