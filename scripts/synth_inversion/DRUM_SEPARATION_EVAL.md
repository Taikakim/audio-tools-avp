# IDM 44.1 kHz 9-Class Drum Separation & Transcription Benchmark

**Owner:** KUANG / KIM / ANTIGRAVITY  
**Date:** 2026-10-21  
**Status:** Complete & Audition Page Generated  
**Page URI:** [`idm_stems_eval/listen/index.html`](file:///home/kim/sao_runs/idm_stems_eval/listen/index.html)  
**Summary JSON:** [`idm_stems_eval/idm_eval_summary.json`](file:///home/kim/sao_runs/idm_stems_eval/idm_eval_summary.json)

---

## 1. Executive Summary

We evaluated the **Inverse Drum Machine (IDM)** (Torres et al., IEEE TASLP 2025; `idm-44-train-kits`) across the complete evaluation corpus:
- **Set 1: Real Drum Kits (23 Tracks):** Matched full drum mixes (Kick + Percussion downmixes) sliced to the exact evaluation phrase windows.
- **Set 2: JEPA Bass Stems (24 Tracks):** Real bass stems evaluated during synth inversion benchmarks (`stem_01_real.wav` to `stem_24_real.wav`).

### Key Performance Metrics
| Metric | Value | Note |
| :--- | :--- | :--- |
| **Total Audio Sets Decomposed** | **47 sets** (23 Drum Mixes + 24 JEPA Stems) | 423 stems total (9 per set) |
| **Total Onsets Transcribed** | **770 onsets** (463 in drums, 307 in stems) | Sub-frame timestamp accuracy |
| **Average Separation Latency** | **1.47 seconds** per 7.0s clip | **~0.21x real-time on CPU** (5x faster than real-time) |
| **Compute Constraints** | **CPU Only** (4 threads) | GPU 0 remained 100% free for Wintermute training |
| **Sample Rate & Format** | **44.1 kHz 32-bit Float Stereo** | Preserved native fidelity with Wiener masking |

---

## 2. Key Findings & Diagnostic Insights

### 2.1 Demucs Bass Stems Carry Heavy Kick Bleed (Stems 21–24)
In all 4 Demucs-separated tracks, IDM immediately detected and cleanly extracted significant **Kick Drum (`KD`)** leakage:
- **Stem 21 (`astral_mahadeva`):** 23 Kick onsets detected (**64.9%** of percussive energy).
- **Stem 22 (`mfg_shape_future`):** 10 Kick onsets detected (**78.0%** of percussive energy).
- **Stem 23 (`cosmosis_alien_disco`):** 17 Kick onsets detected (**82.4%** of percussive energy).
- **Stem 24 (`final_mission`):** 5 Kick onsets detected (**32.5%** of percussive energy).

*Implication for Synth Inversion:* Because Demucs leaves substantial kick drum bleed in bass stems, earlier spectral distance metrics were partially penalizing models for failing to reproduce the kick transient. Using IDM pre-filtering can cleanly isolate and subtract the kick transient before parameter estimation.

### 2.2 Transients in Pure Melodic Stems (Click/Pluck vs Drum)
- **Filtered Saws (Stems 01, 04, 05, 08):** Zero to 1 onset detected. IDM does not hallucinate drums on smooth or moderately filtered analog basslines.
- **Aggressive Plucks & Squelch (Stem 02 Acid Alien, Stem 14 TB-303):** IDM picked up 21 and 32 closed hi-hat (`HH_CHH`) onsets corresponding to the ultra-sharp filter attack transient and square-wave clicks.
- **Moroder Octaves & Group Mixes (Stems 09, 10, 16, 18):** Studio tracks where kick and bass were routed together in stems showed exact four-on-the-floor kick onsets isolated into the `KD` stem.

### 2.3 Real Drum Kit Decomposition (23 Tracks)
- In the authentic drum mixes, IDM isolated complex Goa/Psytrance patterns:
  - 16th-note offbeat closed hi-hats (`HH_CHH`) were transcribed with high confidence (>0.75).
  - Snares (`SD`), open hats (`HH_OHH`), and cymbal crashes were separated into distinct channels.
  - The recombined stem audio (`recombined.wav`) reconstructs the drum mix with minimal phase cancellation thanks to Wiener masking ($\alpha = 1.0$).

---

## 3. Interactive Web Audition Page

The evaluation page is located at:
```text
/home/kim/sao_runs/idm_stems_eval/listen/index.html
```

### Audition Page Features:
1. **Interactive Multi-Track Console:**
   - **Play 9 Stems in Sync:** Simultaneously triggers all 9 kit pieces with sample-accurate playback.
   - **Solo (S) & Mute (M):** Audition any combination of kit pieces (e.g., Solo Kick Drum, or mute Hi-Hats to inspect ghost snares).
   - **Master Comparison:** Instant toggle between Original Audio and Recombined 9-Stem Mix.
2. **Interactive Onset Timeline:**
   - Click anywhere along the visual progress bar to scrub playback across all stems simultaneously.
   - Hover over individual hit pips to view exact timestamp, velocity, confidence, and instrument class.
3. **Kit Energy Share Meters:**
   - Real-time proportional energy bars showing the balance of Kick vs Hats vs Snares vs Toms.
4. **Tab Switcher:**
   - Easily alternate between **Real Drum Kits** and **JEPA Bass Stems & Bleed**.
   - Search/filter box to instantly jump to specific tracks or musical styles.

---

## 4. Serving the Page to the Team

To share the evaluation page with the team:
```bash
# Serve locally on port 8080:
cd /home/kim/sao_runs/idm_stems_eval
python3 -m http.server 8080

# Or bundle for transfer:
tar -czf /home/kim/idm_stems_eval_pack.tar.gz -C /home/kim/sao_runs idm_stems_eval
```
All audio links within `index.html` use relative paths (`../audio/...`), making the package fully self-contained and ready to host or rsync.
