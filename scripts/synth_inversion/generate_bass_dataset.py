"""Multi-process Surge XT bass dataset generator (pedalboard VST3 host).

Sample i is drawn with surge_spec.draw_patch(i + SEED_OFFSET); the draw order there is
frozen, so this script regenerates surge_bass_200k.h5 bit-for-bit in its params.
v2 additionally stores each sample's note_dur (needed to re-render targets for audio-domain
evaluation) and records the generator settings as h5 attributes.
"""
import multiprocessing
import os
import time

multiprocessing.set_start_method("spawn", force=True)
import concurrent.futures

import h5py
import numpy as np

from audio_utils import MEL_SHAPE, make_mel_spec
from surge_spec import (AUDIO_LEN, DEFAULT_PLUGIN_PATH, NOTE_DUR_RANGE, NUM_PARAMS, PARAM_NAMES, SAMPLE_RATE,
                        SEED_OFFSET, draw_patch, init_synth, patch_to_vector, render_patch)

# Re-exported for older imports (evaluate scripts used to import these from here).
from surge_spec import LP_FILTERS, WAVESHAPER_TYPES  # noqa: F401

GENERATOR_VERSION = 2

_plugin = None
_note_dur_range = NOTE_DUR_RANGE


def init_worker(plugin_path=DEFAULT_PLUGIN_PATH, note_dur_range=NOTE_DUR_RANGE):
    global _plugin, _note_dur_range
    _plugin = init_synth(plugin_path)  # verifies the enum raw values against the plugin
    _note_dur_range = tuple(note_dur_range)


def render_sample(seed):
    patch = draw_patch(seed, _note_dur_range)
    mono = render_patch(_plugin, patch, patch["midi_note"], patch["note_dur"])
    mel = make_mel_spec(mono, sr=SAMPLE_RATE)
    return mono, mel, patch_to_vector(patch), patch["note_dur"]


def generate_dataset(output_path, num_samples=200000, n_workers=12, batch_size=1000,
                     plugin_path=DEFAULT_PLUGIN_PATH, note_dur_range=NOTE_DUR_RANGE):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"=== Generating {num_samples} Surge XT Bass Samples ===", flush=True)
    print(f"Destination: {output_path}", flush=True)
    print(f"Workers: {n_workers} CPU processes | note_dur range: {note_dur_range}", flush=True)

    t0 = time.time()
    print(f"Sample shapes: Audio={AUDIO_LEN} samples, Mel={MEL_SHAPE}, Params={NUM_PARAMS} dimensions", flush=True)

    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["generator_version"] = GENERATOR_VERSION
        h5f.attrs["seed_offset"] = SEED_OFFSET
        h5f.attrs["sample_rate"] = SAMPLE_RATE
        h5f.attrs["note_dur_range"] = np.asarray(note_dur_range, dtype=np.float64)
        h5f.attrs["param_names"] = np.asarray(PARAM_NAMES, dtype="S")
        h5f.attrs["param_storage"] = "ordinal_v1 (categoricals stored as class/(n-1); see surge_spec.py)"

        d_audio = h5f.create_dataset("audio", shape=(num_samples, AUDIO_LEN), dtype=np.float32)
        d_mel = h5f.create_dataset("mel", shape=(num_samples, *MEL_SHAPE), dtype=np.float32)
        d_params = h5f.create_dataset("params", shape=(num_samples, NUM_PARAMS), dtype=np.float32)
        d_note_dur = h5f.create_dataset("note_dur", shape=(num_samples,), dtype=np.float32)

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker,
                                                    initargs=(plugin_path, note_dur_range)) as executor:
            for start_idx in range(0, num_samples, batch_size):
                b_t0 = time.time()
                end_idx = min(start_idx + batch_size, num_samples)
                cur_batch_size = end_idx - start_idx

                seeds = list(range(start_idx + SEED_OFFSET, end_idx + SEED_OFFSET))
                results = list(executor.map(render_sample, seeds))

                d_audio[start_idx:end_idx] = np.stack([r[0] for r in results], axis=0)
                d_mel[start_idx:end_idx] = np.stack([r[1] for r in results], axis=0)
                d_params[start_idx:end_idx] = np.stack([r[2] for r in results], axis=0)
                d_note_dur[start_idx:end_idx] = np.array([r[3] for r in results], dtype=np.float32)

                b_dt = time.time() - b_t0
                total_done = end_idx
                pct = (total_done / num_samples) * 100.0
                rate = cur_batch_size / b_dt
                eta_s = (num_samples - total_done) / (rate + 1e-7)
                print(f"Batch {end_idx // batch_size:3d}/{(num_samples + batch_size - 1)//batch_size:3d}: {total_done:6d}/{num_samples} ({pct:5.1f}%) | {rate:.1f} samp/s | Elapsed: {time.time()-t0:.1f}s | ETA: {eta_s/60:.1f}m", flush=True)

    dt = time.time() - t0
    print(f"\nSuccessfully generated {num_samples} samples in {dt:.1f}s ({num_samples/dt:.1f} samples/sec)!", flush=True)
    print(f"Dataset saved at: {output_path}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200000, help="Number of samples to generate")
    parser.add_argument("--out", type=str, default="/run/media/kim/Mantu/surge_dataset/surge_bass_200k.h5", help="Path to output HDF5")
    parser.add_argument("--workers", type=int, default=12, help="Number of CPU worker processes")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch write size")
    parser.add_argument("--plugin", type=str, default=DEFAULT_PLUGIN_PATH, help="Surge XT VST3 path")
    parser.add_argument("--note_dur_min", type=float, default=NOTE_DUR_RANGE[0],
                        help="Shortest note (s). Lower it (e.g. 0.06) to cover 16th-note stabs; keeps the RNG stream")
    parser.add_argument("--note_dur_max", type=float, default=NOTE_DUR_RANGE[1], help="Longest note (s)")
    args = parser.parse_args()

    generate_dataset(args.out, num_samples=args.samples, n_workers=args.workers, batch_size=args.batch_size,
                     plugin_path=args.plugin, note_dur_range=(args.note_dur_min, args.note_dur_max))
