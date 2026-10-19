"""Batch rendering of 60-d parameter vectors through real Surge XT for v4 banks.

Includes anti-glitch gate: redraws degenerate sterile locked sines or silences.
"""
import numpy as np

_W = {}


def _init_worker(plugin, inharm_config):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from surge_spec_v4 import init_synth
    _W["synth"] = init_synth(plugin, verify=False)
    _W["ext"] = None
    if inharm_config:
        from inharmonicity_target import InharmonicityConfig, InharmonicityExtractor
        _W["ext"] = InharmonicityExtractor(InharmonicityConfig.from_json(inharm_config))


def _render_chunk(args):
    vecs, midi, labels = args
    from surge_spec_v4 import render_patch, vector_to_patch, check_glitch
    from realistic_prior_v4 import sample_unconstrained_random_patch
    import zlib
    out, inh, val, final_vecs, final_midi = [], [], [], [], []
    for v, m in zip(vecs, midi):
        a = render_patch(_W["synth"], vector_to_patch(v), int(m), 0.3)
        tries = 0
        cur_v, cur_m = v.copy(), int(m)
        # The redraw seed comes from the item's own vector (KUANG 2026-10-19). It was a function of the MIDI note
        # only, so every glitching item on the same note got the same replacement patch: at most ~5 distinct
        # replacements per note in the whole bank.
        base_seed = zlib.crc32(np.ascontiguousarray(v, dtype=np.float32).tobytes())
        while check_glitch(a) and tries < 5:
            # Degenerate whistle or silence detected: redraw clean unconstrained random patch
            rng = np.random.RandomState((base_seed + 7919 * (tries + 1)) % (2 ** 32))
            cur_v, cur_m = sample_unconstrained_random_patch(rng)
            a = render_patch(_W["synth"], vector_to_patch(cur_v), int(cur_m), 0.3)
            tries += 1
        out.append(a.astype(np.float32))
        final_vecs.append(cur_v)
        final_midi.append(int(cur_m))
        if labels:
            r = _W["ext"].process_phrase(a, int(cur_m))
            inh.append(float(r["value"]) if r["valid"] else 0.0)
            val.append(1.0 if r["valid"] else 0.0)
    return (np.stack(out), np.array(inh, np.float32), np.array(val, np.float32), np.stack(final_vecs),
            np.array(final_midi, np.int16))


class RenderPoolV4:
    def __init__(self, plugin=None, workers=0, inharm_config=None, chunk=16):
        self.plugin, self.workers, self.inharm_config = plugin, int(workers), inharm_config
        self.chunk = int(chunk)
        self._pool = None

    def __enter__(self):
        if self.plugin is None:
            from surge_spec_v4 import DEFAULT_PLUGIN_PATH
            self.plugin = DEFAULT_PLUGIN_PATH
        if self.workers > 0:
            import multiprocessing as mp
            self._pool = mp.get_context("spawn").Pool(self.workers, initializer=_init_worker,
                                                      initargs=(self.plugin, self.inharm_config))
        else:
            _init_worker(self.plugin, self.inharm_config)
        return self

    def __exit__(self, *exc):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        return False

    def render(self, vecs, midi, labels=False):
        vecs = np.asarray(vecs, dtype=np.float32)
        midi = np.asarray(midi).astype(int)
        if len(vecs) != len(midi):
            raise ValueError("vecs and midi differ in length")
        jobs = [(vecs[i:i + self.chunk], midi[i:i + self.chunk], labels) for i in range(0, len(vecs), self.chunk)]
        parts = self._pool.map(_render_chunk, jobs) if self._pool is not None else [_render_chunk(j) for j in jobs]
        audio = np.concatenate([p[0] for p in parts])
        final_vecs = np.concatenate([p[3] for p in parts])
        self.last_midi = np.concatenate([p[4] for p in parts])      # the note each clip was rendered at (redraws)
        if labels:
            return audio, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]), final_vecs
        return audio, final_vecs


def to_int16(audio):
    return np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
