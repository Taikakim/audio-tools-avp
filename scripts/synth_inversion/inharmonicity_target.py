import essentia.standard as es
import numpy as np

class InharmonicityExtractor:
    def __init__(self, sample_rate=44100, max_harmonics=20, tolerance=0.2):
        self.sample_rate = sample_rate
        self.max_harmonics = max_harmonics
        self.tolerance = tolerance
        
        # We need a peak picker. SpectralPeaks takes a spectrum.
        # Windowing and FFT
        self.window = es.Windowing(type='hann')
        self.spectrum = es.Spectrum()
        self.spectral_peaks = es.SpectralPeaks(
            orderBy='frequency',
            magnitudeThreshold=1e-5,
            minFrequency=20.0,
            maxFrequency=sample_rate/2.0,
            maxPeaks=100
        )
        self.harmonic_peaks = es.HarmonicPeaks(
            maxHarmonics=max_harmonics,
            tolerance=tolerance
        )
        self.inharmonicity = es.Inharmonicity()

    def process_frame(self, frame_audio, midi_note=None):
        """
        Processes a single audio frame (1D array).
        Returns a dict:
        {
            'value': float,
            'valid': bool,
            'rejection_reason': str,
            'fundamental_hz': float,
            'coverage_diagnostics': dict
        }
        """
        # Validity checks
        if not np.isfinite(frame_audio).all():
            return self._invalid("nonfinite input")
            
        if np.max(np.abs(frame_audio)) < 1e-4:
            return self._invalid("silence")
            
        if midi_note is None:
            return self._invalid("unknown fundamental")
            
        # 1. Compute spectrum
        frame_size = 4096
        if len(frame_audio) < frame_size:
            return self._invalid("audio too short")
        start = len(frame_audio) // 2 - frame_size // 2
        frame = frame_audio[start:start+frame_size]
        w = self.window(np.ascontiguousarray(frame, dtype=np.float32))
        spec = self.spectrum(w)
        
        # 2. Extract spectral peaks
        freqs, mags = self.spectral_peaks(spec)
        
        if len(freqs) == 0:
            return self._invalid("no spectral peaks found")
            
        # Compute ground truth f0 from midi note
        f0 = 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))
        
        # 3. Extract harmonic peaks
        hf, hm = self.harmonic_peaks(freqs, mags, f0)
        
        # Calculate coverage diagnostics
        total_peak_energy = np.sum(mags ** 2)
        selected_peak_energy = np.sum(hm ** 2)
        
        # Prevent division by zero
        if total_peak_energy < 1e-12:
            return self._invalid("insufficient peak energy")
            
        coverage = selected_peak_energy / total_peak_energy
        
        # 4. Compute inharmonicity
        val = self.inharmonicity(hf, hm)
        
        # Define validity policy
        valid = True
        rejection_reason = None
        
        # If less than 2 valid harmonics, the score is meaningless
        if np.sum(hm > 0) < 2:
            valid = False
            rejection_reason = "insufficient informative partials"
            
        # The exact requirement from spec: "If substantial displaced energy is excluded, do not interpret 
        # a low descriptor value as evidence that the entire sound is harmonic."
        # We enforce a coverage threshold. If it's too low, we mark it invalid.
        if coverage < 0.5:
            valid = False
            rejection_reason = f"low harmonic coverage ({coverage:.2f})"
            
        return {
            'value': float(val),
            'valid': valid,
            'rejection_reason': rejection_reason,
            'fundamental_hz': f0,
            'fundamental_source': 'midi_note',
            'coverage_diagnostics': {
                'input_peak_count': len(freqs),
                'selected_peak_count': int(np.sum(hm > 0)),
                'coverage': float(coverage),
                'total_peak_energy': float(total_peak_energy),
                'selected_peak_energy': float(selected_peak_energy)
            }
        }
        
    def _invalid(self, reason):
        return {
            'value': 0.0,
            'valid': False,
            'rejection_reason': reason,
            'fundamental_hz': 0.0,
            'fundamental_source': 'none',
            'coverage_diagnostics': {}
        }
