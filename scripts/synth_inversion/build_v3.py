import os
import numpy as np

def build_surge_spec_v3():
    with open('/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion/surge_spec.py', 'r') as f:
        code = f.read()

    # Replace PARAM_NAMES list
    code = code.replace(
        '    "delay_fb",         # 22\n]',
        '    "delay_fb",         # 22\n' +
        '    "osc_1_width_2", "osc_2_shape", "osc_2_width_1", "osc_2_width_2", "osc_3_shape", "osc_3_width_1", "osc_3_width_2",\n' +
        '    "osc_1_octave", "osc_2_octave", "osc_3_octave", "noise_color",\n' +
        '    "osc_1_volume", "osc_2_volume", "osc_3_volume", "ring_1x2_volume", "ring_2x3_volume", "noise_volume",\n' +
        '    "filter_balance", "filter_configuration", "fm_routing",\n' +
        '    "filter_2_type", "filter_2_cutoff", "filter_2_resonance", "filter_2_keytrack", "filter_2_feg_amount"\n]'
    )

    # Replace CATEGORICAL
    code = code.replace(
        'CATEGORICAL = {"filter_type": len(LP_FILTERS), "unison": 2, "waveshaper_type": len(WAVESHAPER_TYPES)}',
        'CATEGORICAL = {"filter_type": len(LP_FILTERS), "unison": 2, "waveshaper_type": len(WAVESHAPER_TYPES), "filter_configuration": 7, "fm_routing": 4, "filter_2_type": len(LP_FILTERS)}'
    )

    # Replace CONT_BOUNDS
    code = code.replace(
        '    "drive": (0.0, 1.0), "chorus_mix": (0.0, 0.60), "delay_mix": (0.0, 0.45), "delay_fb": (0.0, 0.50),\n}',
        '    "drive": (0.0, 1.0), "chorus_mix": (0.0, 0.60), "delay_mix": (0.0, 0.45), "delay_fb": (0.0, 0.50),\n' +
        '    "osc_1_width_2": (0.0, 1.0), "osc_2_shape": (0.0, 1.0), "osc_2_width_1": (0.0, 1.0), "osc_2_width_2": (0.0, 1.0),\n' +
        '    "osc_3_shape": (0.0, 1.0), "osc_3_width_1": (0.0, 1.0), "osc_3_width_2": (0.0, 1.0),\n' +
        '    "osc_1_octave": (0.0, 1.0), "osc_2_octave": (0.0, 1.0), "osc_3_octave": (0.0, 1.0), "noise_color": (0.0, 1.0),\n' +
        '    "osc_1_volume": (0.0, 1.0), "osc_2_volume": (0.0, 1.0), "osc_3_volume": (0.0, 1.0),\n' +
        '    "ring_1x2_volume": (0.0, 1.0), "ring_2x3_volume": (0.0, 1.0), "noise_volume": (0.0, 0.25),\n' +
        '    "filter_balance": (0.0, 1.0), "filter_2_cutoff": (0.0, 1.0), "filter_2_resonance": (0.0, 1.0),\n' +
        '    "filter_2_keytrack": (0.0, 1.0), "filter_2_feg_amount": (0.0, 1.0)\n}'
    )

    # Update draw_patch
    code = code.replace(
        '        delay_fb=delay_fb,\n        note_dur=note_dur,\n    )',
        '''        delay_fb=delay_fb,
        note_dur=note_dur,
        osc_1_width_2=float(np.random.uniform(0.0, 1.0)),
        osc_2_shape=float(np.random.uniform(0.0, 1.0)),
        osc_2_width_1=float(np.random.uniform(0.0, 1.0)),
        osc_2_width_2=float(np.random.uniform(0.0, 1.0)),
        osc_3_shape=float(np.random.uniform(0.0, 1.0)),
        osc_3_width_1=float(np.random.uniform(0.0, 1.0)),
        osc_3_width_2=float(np.random.uniform(0.0, 1.0)),
        osc_1_octave=float(np.random.uniform(0.0, 1.0)),
        osc_2_octave=float(np.random.uniform(0.0, 1.0)),
        osc_3_octave=float(np.random.uniform(0.0, 1.0)),
        noise_color=float(np.random.uniform(0.0, 1.0)),
        osc_1_volume=float(np.random.uniform(0.0, 1.0)),
        osc_2_volume=float(np.random.uniform(0.0, 1.0)),
        osc_3_volume=float(np.random.uniform(0.0, 1.0)),
        ring_1x2_volume=float(np.random.uniform(0.0, 1.0)),
        ring_2x3_volume=float(np.random.uniform(0.0, 1.0)),
        noise_volume=float(np.random.uniform(0.0, 0.25)),
        filter_balance=float(np.random.uniform(0.0, 1.0)),
        filter_configuration=int(np.random.randint(0, 7)),
        fm_routing=int(np.random.randint(0, 4)),
        filter_2_type=int(np.random.randint(0, len(LP_FILTERS))),
        filter_2_cutoff=float(np.random.uniform(0.0, 1.0)),
        filter_2_resonance=float(np.random.uniform(0.0, 1.0)),
        filter_2_keytrack=float(np.random.uniform(0.0, 1.0)),
        filter_2_feg_amount=float(np.random.uniform(0.0, 1.0)),
    )'''
    )

    # patch_to_vector append
    code = code.replace(
        '        patch["delay_fb"],\n    ], dtype=np.float32)',
        '''        patch["delay_fb"],
        patch["osc_1_width_2"], patch["osc_2_shape"], patch["osc_2_width_1"], patch["osc_2_width_2"],
        patch["osc_3_shape"], patch["osc_3_width_1"], patch["osc_3_width_2"],
        patch["osc_1_octave"], patch["osc_2_octave"], patch["osc_3_octave"], patch["noise_color"],
        patch["osc_1_volume"], patch["osc_2_volume"], patch["osc_3_volume"],
        patch["ring_1x2_volume"], patch["ring_2x3_volume"], patch["noise_volume"],
        patch["filter_balance"], patch["filter_configuration"] / 6.0, patch["fm_routing"] / 3.0,
        patch["filter_2_type"] / (len(LP_FILTERS) - 1), patch["filter_2_cutoff"], patch["filter_2_resonance"],
        patch["filter_2_keytrack"], patch["filter_2_feg_amount"]
    ], dtype=np.float32)'''
    )

    # vector_to_patch append
    code = code.replace(
        '        delay_fb=float(p[22]),\n    )',
        '''        delay_fb=float(p[22]),
        osc_1_width_2=float(p[23]),
        osc_2_shape=float(p[24]),
        osc_2_width_1=float(p[25]),
        osc_2_width_2=float(p[26]),
        osc_3_shape=float(p[27]),
        osc_3_width_1=float(p[28]),
        osc_3_width_2=float(p[29]),
        osc_1_octave=float(p[30]),
        osc_2_octave=float(p[31]),
        osc_3_octave=float(p[32]),
        noise_color=float(p[33]),
        osc_1_volume=float(p[34]),
        osc_2_volume=float(p[35]),
        osc_3_volume=float(p[36]),
        ring_1x2_volume=float(p[37]),
        ring_2x3_volume=float(p[38]),
        noise_volume=float(p[39]),
        filter_balance=float(p[40]),
        filter_configuration=ordinal_to_class(p[41], 7),
        fm_routing=ordinal_to_class(p[42], 4),
        filter_2_type=ordinal_to_class(p[43], len(LP_FILTERS)),
        filter_2_cutoff=float(p[44]),
        filter_2_resonance=float(p[45]),
        filter_2_keytrack=float(p[46]),
        filter_2_feg_amount=float(p[47]),
    )'''
    )

    # apply_patch append
    code = code.replace(
        '    P["a_waveshaper_drive"].raw_value = patch["drive_raw"]',
        '''    P["a_waveshaper_drive"].raw_value = patch["drive_raw"]
    P["a_osc_1_width_2"].raw_value = patch["osc_1_width_2"]
    P["a_osc_2_shape"].raw_value = patch["osc_2_shape"]
    P["a_osc_2_width_1"].raw_value = patch["osc_2_width_1"]
    P["a_osc_2_width_2"].raw_value = patch["osc_2_width_2"]
    P["a_osc_3_shape"].raw_value = patch["osc_3_shape"]
    P["a_osc_3_width_1"].raw_value = patch["osc_3_width_1"]
    P["a_osc_3_width_2"].raw_value = patch["osc_3_width_2"]
    P["a_osc_1_octave"].raw_value = patch["osc_1_octave"]
    P["a_osc_2_octave"].raw_value = patch["osc_2_octave"]
    P["a_osc_3_octave"].raw_value = patch["osc_3_octave"]
    P["a_noise_color"].raw_value = patch["noise_color"]
    P["a_osc_1_level"].raw_value = patch["osc_1_volume"]
    P["a_osc_2_level"].raw_value = patch["osc_2_volume"]
    P["a_osc_3_level"].raw_value = patch["osc_3_volume"]
    P["a_ring_1x2_level"].raw_value = patch["ring_1x2_volume"]
    P["a_ring_2x3_level"].raw_value = patch["ring_2x3_volume"]
    P["a_noise_level"].raw_value = patch["noise_volume"]
    P["a_filter_balance"].raw_value = patch["filter_balance"]
    P["a_filter_configuration"].raw_value = patch["filter_configuration"] / 6.0
    P["a_fm_routing"].raw_value = patch["fm_routing"] / 3.0
    P["a_filter_2_type"].raw_value = LP_FILTERS[patch["filter_2_type"]][1]
    P["a_filter_2_cutoff"].raw_value = patch["filter_2_cutoff"]
    P["a_filter_2_resonance"].raw_value = patch["filter_2_resonance"]
    P["a_filter_2_keytrack"].raw_value = patch["filter_2_keytrack"]
    P["a_filter_2_feg_mod_amount"].raw_value = patch["filter_2_feg_amount"]'''
    )

    with open('/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion/surge_spec_v3.py', 'w') as f:
        f.write(code)

def build_extract_v3():
    with open('/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion/extract_real_bass_manifold.py', 'r') as f:
        ext = f.read()

    ext = ext.replace('from surge_spec import', 'from surge_spec_v3 import')
    ext = ext.replace('surge_spec.patch_to_vector conventions', 'surge_spec_v3.patch_to_vector conventions')
    ext = ext.replace('surge_spec.LP_FILTERS', 'surge_spec_v3.LP_FILTERS')
    ext = ext.replace('surge_spec.vector_to_patch', 'surge_spec_v3.vector_to_patch')
    ext = ext.replace('surge_spec.init_synth', 'surge_spec_v3.init_synth')

    ext = ext.replace(
        'OUTPUT_PATH = "/run/media/kim/Mantu/surge_200k_models/real_bass_manifold.npz"',
        'OUTPUT_PATH = "/run/media/kim/Mantu2/surge_200k_models/real_bass_manifold_v3.npz"'
    )
    ext = ext.replace(
        'OUTPUT_PATH = "/run/media/kim/Mantu2/surge_200k_models/real_bass_manifold.npz"',
        'OUTPUT_PATH = "/run/media/kim/Mantu2/surge_200k_models/real_bass_manifold_v3.npz"'
    )


    ext = ext.replace(
        '    "drive":         ("a_ws_drive",          "a_waveshaper_drive",       -24.0, 24.0, "db"),\n}',
        '''    "drive":         ("a_ws_drive",          "a_waveshaper_drive",       -24.0, 24.0, "db"),
    "osc_1_width_2": ("a_osc1_param2", "a_osc_1_width_2", 0.0, 1.0, "pct"),
    "osc_2_shape": ("a_osc2_param0", "a_osc_2_shape", -1.0, 1.0, "pct"),
    "osc_2_width_1": ("a_osc2_param1", "a_osc_2_width_1", 0.0, 1.0, "pct"),
    "osc_2_width_2": ("a_osc2_param2", "a_osc_2_width_2", 0.0, 1.0, "pct"),
    "osc_3_shape": ("a_osc3_param0", "a_osc_3_shape", -1.0, 1.0, "pct"),
    "osc_3_width_1": ("a_osc3_param1", "a_osc_3_width_1", 0.0, 1.0, "pct"),
    "osc_3_width_2": ("a_osc3_param2", "a_osc_3_width_2", 0.0, 1.0, "pct"),
    "osc_1_octave": ("a_osc1_octave", "a_osc_1_octave", -4.0, 4.0, "semitones"),
    "osc_2_octave": ("a_osc2_octave", "a_osc_2_octave", -4.0, 4.0, "semitones"),
    "osc_3_octave": ("a_osc3_octave", "a_osc_3_octave", -4.0, 4.0, "semitones"),
    "noise_color": ("a_noise_color", "a_noise_color", -1.0, 1.0, "pct"),
    "osc_1_volume": ("a_level_o1", "a_osc_1_level", 0.0, 1.0, "pct"),
    "osc_2_volume": ("a_level_o2", "a_osc_2_level", 0.0, 1.0, "pct"),
    "osc_3_volume": ("a_level_o3", "a_osc_3_level", 0.0, 1.0, "pct"),
    "ring_1x2_volume": ("a_level_ring12", "a_ring_1x2_level", 0.0, 1.0, "pct"),
    "ring_2x3_volume": ("a_level_ring23", "a_ring_2x3_level", 0.0, 1.0, "pct"),
    "noise_volume": ("a_level_noise", "a_noise_level", 0.0, 1.0, "pct"),
    "filter_balance": ("a_f_balance", "a_filter_balance", -1.0, 1.0, "pct"),
    "filter_2_cutoff": ("a_filter2_cutoff", "a_filter_2_cutoff", -60.0, 70.0, "hz"),
    "filter_2_resonance": ("a_filter2_resonance", "a_filter_2_resonance", 0.0, 1.0, "pct"),
    "filter_2_keytrack": ("a_filter2_keytrack", "a_filter_2_keytrack", -1.0, 1.0, "pct"),
    "filter_2_feg_amount": ("a_filter2_envmod", "a_filter_2_feg_mod_amount", -96.0, 96.0, "semitones"),
}'''
    )

    ext = ext.replace(
        '             "ws_type": "a_ws_type", "fm_switch": "a_fm_switch"}',
        '             "ws_type": "a_ws_type", "fm_switch": "a_fm_switch",\n' +
        '             "filter_configuration": "a_fb_config", "fm_routing": "a_fm_routing", "filter_2_type": "a_filter2_type"}'
    )

    ext = ext.replace(
        '    ws_class = WS_POS_TO_CLASS[ws_pos]\n    vec[PARAM_INDEX["waveshaper_type"]] = ws_class / (len(WAVESHAPER_TYPES) - 1)',
        '''    ws_class = WS_POS_TO_CLASS[ws_pos]
    vec[PARAM_INDEX["waveshaper_type"]] = ws_class / (len(WAVESHAPER_TYPES) - 1)
    
    vec[PARAM_INDEX["filter_configuration"]] = min(max((xml_p.get(ENUM_TAGS["filter_configuration"], 4.0) - 4.0) / 3.0, 0.0), 1.0)
    vec[PARAM_INDEX["fm_routing"]] = min(max(xml_p.get(ENUM_TAGS["fm_routing"], 0) / 3.0, 0.0), 1.0)
    f2_pos = int(round(xml_p.get(ENUM_TAGS["filter_2_type"], f_pos)))
    vec[PARAM_INDEX["filter_2_type"]] = FILTER_POS_TO_CLASS.get(f2_pos, 0) / (len(LP_FILTERS) - 1)'''
    )

    with open('/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion/extract_v3_manifold.py', 'w') as f:
        f.write(ext)

if __name__ == "__main__":
    build_surge_spec_v3()
    build_extract_v3()
    print("Files created successfully.")
