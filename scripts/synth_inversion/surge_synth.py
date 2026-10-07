"""Native surgepy audio renderer for Surge XT neural inversion (v4).

Provides direct in-memory synthesis via surgepy without VST3/Pedalboard host overhead.
Supports note triggering, multi-block rendering, patch state get/set, modulation routing,
and guaranteed FX isolation.
"""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import surgepy

# Reference parameters for Surge XT
SAMPLE_RATE = 44100
DEFAULT_NOTE_DUR = 0.3
DEFAULT_TOTAL_DUR = 0.8

class SurgeRenderer:
    """High-speed native Surge XT renderer using PyBind11 surgepy bindings."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.synth = surgepy.createSurge(sample_rate)
        self.block_size = self.synth.getBlockSize()
        self._param_map = self._build_param_map()
        self._mod_sources = self._build_mod_sources()
        self.ensure_fx_off()

    def _build_param_map(self) -> Dict[str, Any]:
        """Maps parameter names to SurgeNamedParamId handles."""
        params = {}
        for cg_idx in range(12):
            try:
                cg = self.synth.getControlGroup(cg_idx)
                for entry in cg.getEntries():
                    # Support Global (0) and Scene A (1)
                    if entry.getScene() in (0, 1):
                        for p in entry.getParams():
                            params[p.getName()] = p
            except Exception:
                pass
        return params

    def _build_mod_sources(self) -> Dict[str, Any]:
        """Maps mod source names to SurgeModSource objects."""
        sources = {}
        for ms_idx in range(50):
            try:
                ms = self.synth.getModSource(ms_idx)
                sources[ms.getName()] = ms
            except Exception:
                pass
        return sources

    def ensure_fx_off(self) -> None:
        """Explicitly set all FX slot types to Off (0.0)."""
        cg_fx = self.synth.getControlGroup(7) # cg_FX
        for entry in cg_fx.getEntries():
            for p in entry.getParams():
                if "FX Type" in p.getName():
                    self.synth.setParamVal(p, 0.0)

    def set_param(self, name: str, value: float) -> bool:
        """Sets a parameter by exact name."""
        if name in self._param_map:
            p = self._param_map[name]
            p_min = self.synth.getParamMin(p)
            p_max = self.synth.getParamMax(p)
            clamped = float(np.clip(value, p_min, p_max))
            self.synth.setParamVal(p, clamped)
            return True
        return False

    def get_param(self, name: str) -> Optional[float]:
        """Gets a parameter value by exact name."""
        if name in self._param_map:
            return float(self.synth.getParamVal(self._param_map[name]))
        return None

    def get_param_display(self, name: str) -> Optional[str]:
        """Gets display string of a parameter."""
        if name in self._param_map:
            return str(self.synth.getParamDisplay(self._param_map[name]))
        return None

    def set_mod_routing(self, source_name: str, dest_name: str, depth: float) -> bool:
        """Sets modulation routing depth (e.g. Filter EG -> Osc 1 Width 1)."""
        if dest_name in self._param_map and source_name in self._mod_sources:
            dest_p = self._param_map[dest_name]
            src_m = self._mod_sources[source_name]
            # scene 0 is scene A in surgepy modulation calls
            self.synth.setModDepth01(dest_p, src_m, float(depth), 0, 0)
            return True
        return False

    def render_note(
        self,
        midi_note: int = 36,
        velocity: int = 100,
        note_dur: float = DEFAULT_NOTE_DUR,
        total_dur: float = DEFAULT_TOTAL_DUR,
        stereo: bool = False,
    ) -> np.ndarray:
        """Renders a single MIDI note and returns audio as a numpy float32 array."""
        total_samples = int(self.sample_rate * total_dur)
        note_samples = int(self.sample_rate * note_dur)
        
        total_blocks = (total_samples + self.block_size - 1) // self.block_size
        note_blocks = min(note_samples // self.block_size, total_blocks)
        tail_blocks = total_blocks - note_blocks

        buf = self.synth.createMultiBlock(total_blocks)
        
        self.synth.allNotesOff()
        self.synth.playNote(0, int(midi_note), int(velocity), 0)
        
        if note_blocks > 0:
            self.synth.processMultiBlock(buf, 0, note_blocks)
            
        self.synth.releaseNote(0, int(midi_note), int(velocity))
        
        if tail_blocks > 0:
            self.synth.processMultiBlock(buf, note_blocks, tail_blocks)
            
        self.synth.allNotesOff()

        audio = buf[:, :total_samples]
        if not stereo:
            return 0.5 * (audio[0] + audio[1])
        return audio

    def save_patch(self, path: str) -> None:
        """Saves current state to an .fxp patch file."""
        self.synth.savePatch(path)

    def load_patch(self, path: str) -> bool:
        """Loads an .fxp patch file from disk."""
        ok = self.synth.loadPatch(path)
        return ok
