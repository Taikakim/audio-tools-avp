# sa3_control/measure_longform.py
"""Windowed output metrics for steered long-form audio (mir venv: librosa + essentia)."""
import numpy as np, librosa, librosa.feature.rhythm

def windowed_metrics(wav_path, win_sec=10.0, hop_sec=5.0):
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    win = int(win_sec * sr); hop = int(hop_sec * sr); rows = []
    for start in range(0, max(1, len(y) - win + 1), hop):
        seg = y[start:start + win]
        if len(seg) < win // 2: break
        on = librosa.onset.onset_detect(y=seg, sr=sr, units="time")
        dens = len(on) / (len(seg) / sr)
        tempo = librosa.feature.rhythm.tempo(y=seg, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        rows.append({"t": start / sr, "onset_density": dens, "bpm": bpm})
    return rows

def main():
    import sys, json
    rows = windowed_metrics(sys.argv[1])
    json.dump(rows, open(sys.argv[1] + ".metrics.json", "w"), indent=1)
    print(f"[measure] {len(rows)} windows -> {sys.argv[1]}.metrics.json")

if __name__ == "__main__":
    main()
