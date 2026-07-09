# TinyML GSM Channel Scan — Occupancy Detection

Compact ML models for 25-channel GSM occupancy detection on low-bit wideband IF,
benchmarked against a classical CFAR baseline. R&D internship project — see
`Intern_RnD_Project_Brief_TinyML_GSM_Channel_Scan.md` for full scope.

**Status: M4 (robustness benchmarking) complete, M5 (embedded feasibility) in progress.**

---

## Project structure

```
├── gsm_dataset_gen_new.py     # synthetic dataset generator
├── feature_extraction.py      # hand-engineered features (F1-F6) + matched-filter bank
├── dataset_new.py             # GSMDataset loader, train/val/test split, augmentation
│
├── model.py                   # MLP architecture (deprecated — see Model Notes)
├── model_attn.py               # Slot-attention architecture (ATTN)
├── model_tcn.py                # Raw-IQ temporal conv network (TCN)
├── sweep.py                   # DeepSweep-style parallel chunk classifier (deprecated)
│
├── train_new.py               # training loop, shared across all ML architectures
├── fft_cfar_baseline.py       # classical CFAR energy-detector baseline (R3)
│
├── eval_breakdown.py          # per-SNR / blocker / adj-channel / carrier-count breakdown
├── compare_checkpoints.py     # side-by-side checkpoint comparison (non-iso-FAR)
├── iso_far_comparison.py      # ★ matched-false-alarm-rate comparison (the real benchmark)
├── make_plots.py              # generates all report figures from collected results
│
├── plot_*.png                  # generated figures
└── requirements.txt
```

---

## Pipeline — reproduce from scratch

**1. Generate dataset** (~50k samples, stratified toward low-SNR):
```bash
python gsm_dataset_gen_new.py --n-samples 50000 --mode synthetic \
  --out-dir ./gsm_dataset_50k --seed 42
```

**2. Train ML models:**
```bash
python train_new.py --dataset-dir ./gsm_dataset_50k --model attn \
  --batch-size 64 --epochs 40 --learning-rate 3e-4 --patience 8 \
  --loss focal --focal-alpha 0.5 --focal-gamma 2.0 --dropout 0.3 \
  --checkpoint-path best_attn_v2.pth

python train_new.py --dataset-dir ./gsm_dataset_50k --model tcn \
  --batch-size 64 --epochs 60 --learning-rate 1e-3 --patience 10 \
  --loss focal --focal-alpha 0.5 --focal-gamma 2.0 --dropout 0.3 \
  --checkpoint-path best_tcn.pth
```

**3. Run the classical baseline** (tunes CFAR alpha on validation, evaluates on test):
```bash
python fft_cfar_baseline.py --dataset-dir ./gsm_dataset_50k --scenario-col snr_db
```

**4. Run the fair (iso-FAR) comparison** — this is the benchmark that matters, not
   step 3/earlier eval scripts alone. Matches all three models to the same
   false-alarm rate before comparing detection rate:
```bash
python iso_far_comparison.py \
  --dataset-dir ./gsm_dataset_50k \
  --attn-checkpoint best_attn_v2.pth \
  --tcn-checkpoint best_tcn.pth \
  --target-far 0.05
```

**5. Generate report figures:**
```bash
python make_plots.py
```
(`make_plots.py` uses a hardcoded `RESULTS` dict populated from step 4's output —
update it manually with fresh numbers if you retrain.)

---

## Model notes

| Model | Input | Params | MACs | Status |
|---|---|---|---|---|
| CFAR (baseline) | raw IQ → energy | 0 (no learned params) | ~400k | Required baseline (R3) |
| **Attn** | hand features (125-dim) | 38,085 | 514,400 | Primary candidate — cheap, strong at high SNR |
| **TCN** | raw IQ | 211,943 | 1,023,950 | Primary candidate — strong at low SNR, CFO-sensitive |
| MLP | hand features | ~56k | ~57k | Early architecture, kept for record — overfit on 14k samples in earlier iterations (v2: train F1=0.42, val F1=0.28); superseded by Attn as primary compact candidate but retained as a documented negative/comparative finding |
| Sweep | hand features, ±1 slot context | few thousand | small | Deprecated — structurally capped receptive field, can't see wideband blockers |

---

## Key results (iso-FAR comparison, target P_fa ≈ 0.05)

All three models compared at the **same matched false-alarm rate** — earlier
comparisons in this repo's history used each model's own self-optimal threshold,
which is not a fair comparison; iso_far_comparison.py fixes this.

| Model | P_d | P_fa | F1 |
|---|---|---|---|
| CFAR | 0.232 | 0.050 | 0.277 |
| Attn | 0.299 | 0.026 | 0.391 |
| TCN | 0.350 | 0.049 | 0.392 |

**Both ML models beat CFAR at every SNR bin tested.** Aggregate F1 between Attn
and TCN is a near-tie; the meaningful difference is regime-specific:

- **TCN wins at low SNR (0-10dB)** — the range confirmed as the real operating
  condition by an OTA capture (`cfile_window0000_power_spectrum.png`).
- **Attn wins at high SNR (20-41dB).**
- **TCN is near-perfectly robust to wideband blockers** (F1 drop ~0.1% blocker
  vs no-blocker); Attn shows a moderate ~11% drop; CFAR the largest (~18%).
- **TCN is sharply sensitive to carrier frequency offset** — F1 collapses
  ~65-75% relative beyond ±3kHz CFO, likely due to its physically-initialized
  matched-filter frontend. Attn is comparatively CFO-tolerant. This is a real
  limitation to weigh against TCN's low-SNR advantage.

See `plot_f1_vs_snr.png`, `plot_cfo_robustness.png`, `plot_blocker_comparison.png`,
`plot_complexity.png` for the figures.

---

## Known limitations / caveats (read before citing these numbers)

- **Single training run per model.** No repeat-seed variance estimate exists.
  Trends above are plausible but not statistically confirmed across seeds.
- **CFO tail bins are thin** (n=352-509 vs n=5737 in the center bin) — treat
  small differences there with caution; the large TCN center-vs-tail collapse
  is well-supported, smaller cross-model gaps in the tails are not.
- **Synthetic wideband-blocker model is visually, not statistically, validated**
  against the one available OTA capture.
- **Test-set SNR composition is heavily skewed low** (~66% of test samples are
  below 10dB SNR, by design — stratified sampling toward the real operating
  regime). This is why aggregate F1 numbers look modest; per-SNR-bin numbers
  are the ones that matter for interpretation.
- **Split-leakage (RISK-2):** verified by code inspection — `gsm_dataset_gen_new.py`
  draws each sample's full scenario independently from a single RNG stream, no
  scenario-reuse structure exists, so a random stratified split is appropriate.
  Not empirically verified via duplicate-detection on the generated files.

---

## Recommendation (current, pending M5/M6)

Both ML models justify continued investment over the CFAR baseline. Primary
open decision: TCN's low-SNR advantage (matches the real deployment regime)
vs its CFO fragility, against Attn's ~5x lower parameter count and CFO
robustness. Leaning toward **TCN as primary candidate** if the target hardware
has disciplined oscillator control (CFO within ±3kHz); otherwise **Attn** or a
CFO-compensated TCN frontend should be considered. Final recommendation to be
confirmed in the M6 report pending embedded feasibility analysis (M5).
