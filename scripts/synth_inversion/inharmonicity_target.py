import hashlib
import json
import functools
import numpy as np
import essentia
import essentia.standard as es
from dataclasses import dataclass, asdict
from typing import Optional

EXTRACTOR_VERSION = "inharm-2"

class Reason:
    NONFINITE_INPUT = "nonfinite input"
    SIZE_MISMATCH = "frequency and magnitude arrays differ in length"
    NO_PEAKS = "no spectral peaks"
    BAD_FUNDAMENTAL = "fundamental is not positive"
    UNSORTED_OR_DUPLICATE = "peak frequencies not strictly ascending and positive"
    ESSENTIA_ERROR = "essentia raised"
    NO_FUNDAMENTAL_PEAK = "no spectral peak near the nominal fundamental"
    TOO_FEW_PARTIALS = "too few selected partials"
    EXCLUDED_ENERGY = "excluded energy fraction above limit"
    NONFINITE_RESULT = "nonfinite result"
    SILENCE = "silence"
    WINDOW_LENGTH = "window length differs from frame_size"
    LEVEL_UNSTABLE = "level change above limit"
    TOO_FEW_VALID_WINDOWS = "too few valid note windows"
    ALL = frozenset()

Reason.ALL = frozenset({
    Reason.NONFINITE_INPUT, Reason.SIZE_MISMATCH, Reason.NO_PEAKS,
    Reason.BAD_FUNDAMENTAL, Reason.UNSORTED_OR_DUPLICATE, Reason.ESSENTIA_ERROR,
    Reason.NO_FUNDAMENTAL_PEAK, Reason.TOO_FEW_PARTIALS, Reason.EXCLUDED_ENERGY,
    Reason.NONFINITE_RESULT, Reason.SILENCE, Reason.WINDOW_LENGTH,
    Reason.LEVEL_UNSTABLE, Reason.TOO_FEW_VALID_WINDOWS
})

@dataclass(frozen=True)
class InharmonicityConfig:
    sample_rate: int
    frame_size: int
    max_harmonics: int
    tolerance: float
    min_selected_partials: int
    max_excluded_energy_fraction: float
    require_fundamental_peak: bool
    silence_peak_abs: float
    max_level_change_db: Optional[float]
    attack_guard_s: float
    min_valid_windows: int
    min_partial_relative_db: float

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        try:
            return cls(**data)
        except TypeError as e:
            raise ValueError(str(e))

    def config_id(self):
        d = {**asdict(self), "extractor_version": EXTRACTOR_VERSION, "essentia_version": essentia.__version__}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

@functools.lru_cache(maxsize=16)
def _get_harmonic_peaks_algo(max_harmonics, tolerance):
    return es.HarmonicPeaks(maxHarmonics=max_harmonics, tolerance=tolerance)

def _get_inharmonicity_algo():
    return es.Inharmonicity()

def inharmonicity_from_peaks(freqs, mags, f0_hz, cfg: InharmonicityConfig, fundamental_source="midi_note"):
    freqs = np.asarray(freqs, dtype=np.float32)
    mags = np.asarray(mags, dtype=np.float32)
    
    def _invalid(reasons, diags=None):
        r1 = reasons[0] if reasons else None
        return {
            "value": float("nan"),
            "valid": False,
            "rejection_reason": r1,
            "all_rejection_reasons": reasons,
            "fundamental_hz": float(f0_hz) if np.isfinite(f0_hz) else float("nan"),
            "fundamental_source": fundamental_source,
            "analysis_configuration_id": cfg.config_id(),
            "coverage_diagnostics": diags or {
                "input_peak_count": None, "selected_peak_count": None, "total_peak_energy": None,
                "selected_peak_energy": None, "selected_energy_fraction": None,
                "excluded_energy_fraction": None, "tolerance": cfg.tolerance,
                "max_harmonics": cfg.max_harmonics, "first_selected_hz": None, "level_change_db": None
            }
        }
        
    if not (np.isfinite(freqs).all() and np.isfinite(mags).all() and np.isfinite(f0_hz)):
        return _invalid([Reason.NONFINITE_INPUT])
    if len(freqs) != len(mags):
        return _invalid([Reason.SIZE_MISMATCH])
    if len(freqs) == 0:
        return _invalid([Reason.NO_PEAKS])
    if f0_hz <= 0:
        return _invalid([Reason.BAD_FUNDAMENTAL])
    if not (np.all(freqs > 0) and np.all(np.diff(freqs) > 0)):
        return _invalid([Reason.UNSORTED_OR_DUPLICATE])

    algo = _get_harmonic_peaks_algo(cfg.max_harmonics, cfg.tolerance)
    hf, hm = algo(freqs, mags, float(f0_hz))

    total_peak_energy = float(np.sum(np.float64(mags)**2))
    selected_peak_energy = float(np.sum(np.float64(hm)**2))
    
    input_peak_count = len(freqs)
    if len(mags) > 0 and np.max(mags) > 0:
        threshold = np.max(mags) * (10.0 ** (-cfg.min_partial_relative_db / 20.0))
        selected_peak_count = int(np.sum(hm > threshold))
    else:
        selected_peak_count = 0

    sel_frac = float(selected_peak_energy / total_peak_energy) if total_peak_energy > 0 else None
    exc_frac = float(1.0 - sel_frac) if sel_frac is not None else None
    first_sel = float(hf[0]) if len(hf) > 0 else None

    diags = {
        "input_peak_count": input_peak_count,
        "selected_peak_count": selected_peak_count,
        "total_peak_energy": total_peak_energy,
        "selected_peak_energy": selected_peak_energy,
        "selected_energy_fraction": sel_frac,
        "excluded_energy_fraction": exc_frac,
        "tolerance": cfg.tolerance,
        "max_harmonics": cfg.max_harmonics,
        "first_selected_hz": first_sel,
        "level_change_db": None
    }

    reasons = []
    if cfg.require_fundamental_peak and (len(hm) == 0 or hm[0] <= 0):
        reasons.append(Reason.NO_FUNDAMENTAL_PEAK)
    if selected_peak_count < cfg.min_selected_partials:
        reasons.append(Reason.TOO_FEW_PARTIALS)
    if exc_frac is not None and exc_frac > cfg.max_excluded_energy_fraction:
        reasons.append(Reason.EXCLUDED_ENERGY)
    elif total_peak_energy == 0:
        pass 

    if reasons:
        return _invalid(reasons, diags)

    inh_algo = _get_inharmonicity_algo()
    try:
        val = inh_algo(hf, hm)
    except RuntimeError:
        return _invalid([Reason.ESSENTIA_ERROR], diags)

    if not np.isfinite(val):
        return _invalid([Reason.NONFINITE_RESULT], diags)

    return {
        "value": float(val),
        "valid": True,
        "rejection_reason": None,
        "all_rejection_reasons": [],
        "fundamental_hz": float(f0_hz),
        "fundamental_source": fundamental_source,
        "analysis_configuration_id": cfg.config_id(),
        "coverage_diagnostics": diags
    }

class InharmonicityExtractor:
    def __init__(self, cfg: InharmonicityConfig):
        self.cfg = cfg
        self.window = es.Windowing(type='hann')
        self.spectrum = es.Spectrum()
        self.spectral_peaks = es.SpectralPeaks(
            sampleRate=float(cfg.sample_rate),
            orderBy='frequency',
            magnitudeThreshold=1e-5,
            minFrequency=20.0,
            maxFrequency=float(cfg.sample_rate/2.0),
            maxPeaks=100
        )

    def process_window(self, window, f0_hz, fundamental_source="midi_note"):
        window = np.asarray(window, dtype=np.float32)
        
        def _invalid(reasons, diags=None):
            r1 = reasons[0] if reasons else None
            return {
                "value": float("nan"),
                "valid": False,
                "rejection_reason": r1,
                "all_rejection_reasons": reasons,
                "fundamental_hz": float(f0_hz) if np.isfinite(f0_hz) else float("nan"),
                "fundamental_source": fundamental_source,
                "analysis_configuration_id": self.cfg.config_id(),
                "coverage_diagnostics": diags or {
                    "input_peak_count": None, "selected_peak_count": None, "total_peak_energy": None,
                    "selected_peak_energy": None, "selected_energy_fraction": None,
                    "excluded_energy_fraction": None, "tolerance": self.cfg.tolerance,
                    "max_harmonics": self.cfg.max_harmonics, "first_selected_hz": None, "level_change_db": None
                }
            }

        if not np.isfinite(window).all():
            return _invalid([Reason.NONFINITE_INPUT])
        if len(window) != self.cfg.frame_size:
            return _invalid([Reason.WINDOW_LENGTH])
        if np.max(np.abs(window)) < self.cfg.silence_peak_abs:
            return _invalid([Reason.SILENCE])

        w = self.window(window)
        spec = self.spectrum(w)
        freqs, mags = self.spectral_peaks(spec)

        half = len(window) // 2
        rms1 = np.sqrt(np.mean(np.float64(window[:half])**2))
        rms2 = np.sqrt(np.mean(np.float64(window[half:])**2))
        level_change_db = 20 * np.log10((rms2 + 1e-12) / (rms1 + 1e-12))
        
        res = inharmonicity_from_peaks(freqs, mags, f0_hz, self.cfg, fundamental_source)
        res["coverage_diagnostics"]["level_change_db"] = float(level_change_db)
        
        if self.cfg.max_level_change_db is not None and abs(level_change_db) > self.cfg.max_level_change_db:
            if res["valid"]:
                res["valid"] = False
                res["value"] = float("nan")
            res["all_rejection_reasons"].append(Reason.LEVEL_UNSTABLE)
            if res["rejection_reason"] is None:
                res["rejection_reason"] = Reason.LEVEL_UNSTABLE

        return res

def process_phrase(): pass
def note_windows(): pass
def pool_note_results(): pass
def to_training_pair(): pass
class RejectionStats: pass
