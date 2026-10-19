# Owner: KUANG (created 2026-10-13)
"""Batch rendering of parameter vectors through real Surge XT, in parallel worker processes (one persistent plugin
per worker), exactly as the trainer renders: surge_spec_vN.render_patch(synth, vector_to_patch(vec), midi, dur).
`spec` picks the layout ("v3" default, "v4", "v5"; layout_v6.py maps a flow to its layout). It must match the
vectors: a v5 vector rendered through the v3 spec is a different patch.

Used by render_bank_v5.py, ladder_metric_check_v5.py, dpo_pairs_v5.py and audio_eval_v5.py.

  with RenderPool(plugin, workers=8, inharm_config=None) as pool:
      audio = pool.render(vecs, midi)                 # float32 [n, 35280], peak-normalised mono, like training
      audio, inh, valid = pool.render(vecs, midi, labels=True)   # + inharmonicity labels (needs inharm_config)

`render_fn` replaces Surge entirely (tests, stand-ins): render_fn(vec, midi) -> np.ndarray.
The main process verifies the enum mapping once (init_synth(verify=True)); workers skip that check, as the
trainer's DataLoader workers do. Spawned, not forked: a forked copy of a loaded plugin can deadlock.

Note: render_patch ignores note_dur (the phrase timing is fixed in phrase_events), so a clip is a function of the
vector and the MIDI note, plus Surge's own render-to-render noise.
"""
import numpy as np

_W = {}


def _spec(name):
    import importlib
    return importlib.import_module(f"surge_spec_{name}")


def _init_worker(plugin, inharm_config, spec="v3"):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _W["spec"] = _spec(spec)
    _W["synth"] = _W["spec"].init_synth(plugin, verify=False)
    _W["ext"] = None
    if inharm_config:
        from inharmonicity_target import InharmonicityConfig, InharmonicityExtractor
        _W["ext"] = InharmonicityExtractor(InharmonicityConfig.from_json(inharm_config))


def _render_chunk(args):
    vecs, midi, labels = args
    sp = _W["spec"]
    out, inh, val = [], [], []
    for v, m in zip(vecs, midi):
        a = sp.render_patch(_W["synth"], sp.vector_to_patch(v), int(m), 0.3)
        out.append(a.astype(np.float32))
        if labels:
            r = _W["ext"].process_phrase(a, int(m))
            inh.append(float(r["value"]) if r["valid"] else 0.0)
            val.append(1.0 if r["valid"] else 0.0)
    return np.stack(out), np.array(inh, np.float32), np.array(val, np.float32)


class RenderPool:
    def __init__(self, plugin=None, workers=0, inharm_config=None, render_fn=None, chunk=16, verify=True, spec="v3"):
        self.plugin, self.workers, self.inharm_config = plugin, int(workers), inharm_config
        self.spec = spec
        self.render_fn, self.chunk, self.verify = render_fn, int(chunk), verify
        self._pool = None

    def __enter__(self):
        if self.render_fn is None:
            sp = _spec(self.spec)
            if self.plugin is None:
                self.plugin = sp.DEFAULT_PLUGIN_PATH
            if self.verify:
                del_synth = sp.init_synth(self.plugin, verify=True)  # one enum-mapping check, then drop it
                del del_synth
            if self.workers > 0:
                import multiprocessing as mp
                self._pool = mp.get_context("spawn").Pool(self.workers, initializer=_init_worker,
                                                          initargs=(self.plugin, self.inharm_config, self.spec))
            else:
                _init_worker(self.plugin, self.inharm_config, self.spec)
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
        if labels and self.render_fn is None and not self.inharm_config:
            raise ValueError("labels=True needs inharm_config")
        if self.render_fn is not None:
            audio = np.stack([np.asarray(self.render_fn(v, int(m)), dtype=np.float32) for v, m in zip(vecs, midi)])
            if labels:
                raise ValueError("labels=True is not available with render_fn")
            return audio
        jobs = [(vecs[i:i + self.chunk], midi[i:i + self.chunk], labels) for i in range(0, len(vecs), self.chunk)]
        parts = self._pool.map(_render_chunk, jobs) if self._pool is not None else [_render_chunk(j) for j in jobs]
        audio = np.concatenate([p[0] for p in parts])
        if labels:
            return audio, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])
        return audio


def to_int16(audio):
    return np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def from_int16(a):
    return np.asarray(a, dtype=np.float32) / 32767.0
