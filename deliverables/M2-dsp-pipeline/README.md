# M2 Deliverable — DSP Pipeline (Dataset Generator + Classical Baselines)

**Contributor:** Maitreyee Kumbhojkar
**Milestone:** M2 (dataset pipeline + classical DSP baseline)
**Status:** As-delivered, **unmodified**. Under review. Integration into
`src/scan_engine/` and any fixes are pending team discussion — do not
treat the reported numbers as final.

These files are kept verbatim so review comments can reference exact
line numbers and so the original delivery stays traceable in git history.

## Contents

| File | Role | What it is |
|---|---|---|
| `GSM_Dataset_gen.py` | **Dataset** (not a baseline) | Self-contained synthetic 5-bit I/Q generator at 104 MS/s. Real GSM 51-multiframe bursts (FCCH/SCH/BCCH/CCCH) with Fire + R=1/2 K=5 convolutional coding, from-scratch GMSK (BT=0.3, SPS=4), ITU TDL fading with Jakes Doppler, 6 independent (multi-label) impairment flags, 5-bit quantization. Emits per-sample `.npy` (2×65536 int8) + label dict, plus `metadata.csv`. numpy + scipy only. |
| `FFT_energy_detector.py` | **Baseline #1** | 7-step FFT energy detector: decimate 104→5 MS/s, Hann window, 1024-pt FFT (4.88 kHz/bin, ~41 bins per 200 kHz channel), median-CFAR noise floor, threshold sweep for ROC. |
| `pfb_channeliser.py` | **Baseline #2** | 25-channel polyphase filter bank, 400-tap Kaiser prototype (16 taps/branch), same median-CFAR occupancy logic. |
| `dsp_pipeline.pptx` | Slide deck | Walkthrough of the three scripts + headline results. |

The TinyML model, feature extraction, training, and the ML-vs-DSP
benchmark harness (M3+/D4/D5) are **not** part of this delivery.

## How to run (from the deck)

```bash
pip install numpy scipy

# Generate a dataset
python GSM_Dataset_gen.py --n-samples 1000 --out-dir ./gsm_dataset --seed 42

# FFT baseline
python FFT_energy_detector.py --dataset ./gsm_dataset --pd-snr --roc

# PFB baseline
python pfb_channeliser.py --dataset ./gsm_dataset --pd-snr --roc
```

## Reported results (from the deck, as-delivered — under review)

| Detector | P_d | P_fa | F1 | Notes |
|---|---|---|---|---|
| FFT channeliser | 0.333 | 0.011 | 0.467 | k=3.0×, median CFAR |
| PFB channeliser | 0.056 | 0.045 | 0.078 | short-window framing |
| ML target | ≥ 0.90 | ≤ 0.05 | ≥ 0.85 | not yet built |

## Open items under review (to be filed as issues / PR comments)

These are flagged for discussion; **no changes have been made here.**

1. **FFT detector window/FFT-length mismatch.** `apply_window()` sizes the
   Hann to the full decimated window (3151 samples) but `compute_fft()`
   calls `np.fft.fft(..., n=1024)`, which truncates to the first 1024
   samples (~32% of the window) and applies only the rising half of the
   window before a hard cut. Likely the main cause of the low FFT P_d;
   a too-weak baseline makes the eventual ML comparison unfair.
2. **PFB P_d = 0.056** is suspiciously low (worse than the broken FFT);
   the "short blocks" explanation is likely a misdiagnosis — investigate
   scaling/normalization before accepting.
3. **SNR convention.** AWGN uses Es/N0 with SPS≈384, so a label of
   "0 dB" is not the in-band SNR a detector sees. Must be reconciled with
   the success-criteria definition before targets are locked.
4. **Storage at scale.** Generator persists IQ (128 KB/sample → ~128 GB
   at 1M), contrary to the "regenerate on-the-fly, ~50 MB params" plan.
5. **Per-window vs per-slot impairment labels.** `impairment_flags` is one
   global `(6,)` vector; the revised approach calls for masked *per-slot*
   Stage-2 impairment labels.
6. **Integration.** Decide whether to fold this into `src/scan_engine/`
   (keeping our config/CLI/test/results hygiene + on-the-fly regeneration)
   or keep the standalone scripts as the canonical path.
