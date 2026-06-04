# GSM Channel Scan Engine — ML Pipeline

TinyML-assisted GSM channel scan engine for 5-bit I/Q wideband IF receivers at 104 MS/s, scanning 25 × 200 kHz GSM slots across 5 MHz.

---

## Project Structure

```
gsm_dataset_gen.py      Dataset generator
fft_channeliser.py      FFT DSP baseline (Welch)
pfb_channeliser.py      PFB DSP baseline
phase1.py       Data pipeline and feature extraction
phase2.py         Branch 3 MLP — scalar features only
phase3.py         Branch 3 MLP — all four output heads
phase4.py         Branch 1 CNN + Branch 3 MLP fusion
phase5.py         Full model — all three branches
phase6.py          Ablation study and final evaluation
```

---

## Requirements

```bash
pip install numpy scipy torch
```

---

## Step 0 — Generate Dataset

Generates synthetic 5-bit IQ samples with labelled GSM carriers and impairments.

```bash
# Generate 5000 samples
python gsm_dataset_gen.py --n-samples 5000 --out-dir ./gsm_dataset --seed 42

# Resume an interrupted run from sample 4700
python gsm_dataset_gen.py --n-samples 5000 --start-idx 4700 --out-dir ./gsm_dataset --seed 42
```

Each sample saved as `sample_XXXXXXX.npy` containing:
- `iq` — (2, 65536) int8 raw IQ at 104 MS/s
- `occupancy` — (25,) uint8 ground truth per slot
- `power_class` — (25,) int8 signal strength class (0–3, -1 for empty)
- `impairment_flags` — (6,) uint8 CFO / fading / CW / adj channel / blocker
- `snr_db` — float32
- `cfo_hz` — float32

---

## DSP Baselines

### FFT Channeliser (Welch)

Welch-averaged energy detector with median CFAR threshold.

```bash
# Evaluate on dataset
python fft_channeliser.py --dataset ./gsm_dataset

# P_d vs SNR table
python fft_channeliser.py --dataset ./gsm_dataset --pd-snr

# ROC curve
python fft_channeliser.py --dataset ./gsm_dataset --roc

# Step-by-step on one sample
python fft_channeliser.py --sample ./gsm_dataset/samples/sample_0000042.npy
```

**Results:** P_d=0.562  P_fa=0.028  F1=0.620  (threshold k=1.38)

### PFB Channeliser

Polyphase filter bank channeliser.

```bash
python pfb_channeliser.py --dataset ./gsm_dataset
python pfb_channeliser.py --dataset ./gsm_dataset --pd-snr --roc
python pfb_channeliser.py --sample ./gsm_dataset/samples/sample_0000042.npy
```

**Results:** P_d=0.554  P_fa=0.025  F1=0.622

---

## Phase 1 — Data Pipeline

Builds the PyTorch Dataset, verifies shapes and labels, measures DataLoader throughput.

```bash
python phase1_dataset.py --dataset ./gsm_dataset
python phase1_dataset.py --dataset ./gsm_dataset --n-samples 1000 --num-workers 4
```

**What it checks:**
- Tensor shapes correct for all samples
- No NaN or Inf in features
- Occupancy labels binary (0 or 1)
- Power class -1 for empty slots, 0–3 for occupied
- Impairment flags binary
- DataLoader throughput in samples/sec

**Expected output:**
```
Dataset split:
  Train : 700 samples
  Val   : 150 samples
  Test  : 150 samples
Occupied slots : 126 / 1250 (10.1%)
SNR range      : -9.9 to 24.4 dB (mean 7.5 dB)
DataLoader     : 860 samples/sec — OK
```

---

## Phase 2 — Scalar MLP (Branch 3 only)

MLP on 156 scalar features (F1–F4, F7–F11, F14). Occupancy head only.

```bash
python phase2_train.py --dataset ./gsm_dataset --epochs 100 --num-workers 4 --seed 42
```

**Architecture:**
```
Input(156) → FC(256) → BN → ReLU → Dropout(0.3)
           → FC(128) → BN → ReLU → Dropout(0.3)
           → FC(64)  → BN → ReLU
           → FC(25)  → occupancy logits
```

Parameters: 83,865

**Key flags:**
- `--epochs` — max training epochs (default 100)
- `--lr` — learning rate (default 1e-3)
- `--batch-size` — batch size (default 32)
- `--checkpoint` — directory to save best model (default ./checkpoints)
- `--seed` — random seed for reproducibility (default 42)

**Results:** P_d=0.530  P_fa=0.033  F1=0.583  (threshold=0.65)

---

## Phase 3 — Multi-head MLP

Same trunk as Phase 2 with three additional output heads: power class, GMSK confidence, and impairment flags. Masked loss for slot-level heads.

```bash
python phase3_train.py --dataset ./gsm_dataset --epochs 100 --num-workers 4 --seed 42 --checkpoint ./checkpoints_p3
```

Parameters: 92,380

**Additional output columns:**
- `PcAcc` — power class accuracy on occupied slots (target ≥ 0.75)
- `ImpF1` — impairment macro-F1 across 6 flags (target ≥ 0.70)

**Results:** P_d=0.561  P_fa=0.053  F1=0.553  PcAcc=0.745  ImpF1=0.006

Note: auxiliary heads did not improve occupancy F1 over Phase 2 at this dataset size. The trunk capacity is the bottleneck with 5000 samples.

---

## Phase 4 — Spectrogram CNN + Scalar MLP (Branch 1 + Branch 3)

Adds Branch 1: a 2D CNN on the per-slot log-power spectrogram (25, 41, 5). Fused with Branch 3 via per-slot late fusion.

```bash
python phase4_train.py --dataset ./gsm_dataset --epochs 100 --num-workers 4 --seed 42 --checkpoint ./checkpoints_p4
```

Parameters: 112,620

**Branch 1 architecture:**
```
Input (batch*25, 1, 41, 5)
→ Conv2d(1→16, 3x3) → BN → ReLU
→ Conv2d(16→32, 3x3) → BN → ReLU
→ AdaptiveAvgPool2d(4,2)
→ Flatten → FC(256→64)
→ Output (batch, 25, 64)
```

**Fusion:**
```
cat(scalar_global(64), cnn_slot(64)) → FC(128→64) → heads
```

**Results:** P_d=0.617  P_fa=0.010  F1=0.722  PcAcc=0.808

This is the biggest jump in the pipeline — +0.139 F1 over Phase 2. The spectrogram shape features distinguish weak carriers from noise and leakage.

---

## Phase 5 — Full Fusion Model (Branch 1 + Branch 2 + Branch 3)

Adds Branch 2: a 1D CNN on F12 (temporal variance) and F13 (inter-frame correlation) stacked as (2, 1024).

```bash
python phase5_train.py --dataset ./gsm_dataset --epochs 100 --num-workers 4 --seed 42 --checkpoint ./checkpoints_p5
```

Parameters: 137,116

**Branch 2 architecture:**
```
Input (batch, 2, 1024)
→ Conv1d(2→16, kernel=7) → BN → ReLU
→ Conv1d(16→32, kernel=7) → BN → ReLU
→ AdaptiveAvgPool1d(8)
→ Flatten → FC(256→64)
→ Output (batch, 64)
```

**Fusion:**
```
cat(scalar_global(64), cnn_slot(64), fading_global(64)) → FC(192→64) → heads
```

**Results:** P_d=0.634  P_fa=0.014  F1=0.724  PcAcc=0.799  ImpF1=0.330

Branch 2 added only +0.002 F1 over Phase 4. With only 5 time frames, F12 and F13 are noisy estimates and the spectrogram already captures most of the same information.

---

## Phase 6 — Ablation Study and Final Evaluation

### Step 1 — Train Config C (Branch 1 only, ~15 min)

```bash
python phase6_eval.py --train-c --dataset ./gsm_dataset \
    --epochs 100 --num-workers 4 --ckpt-c-dir ./checkpoints_c
```

### Step 2 — Full evaluation

```bash
python phase6_eval.py --eval-all --dataset ./gsm_dataset \
    --ckpt-b  ./checkpoints/best_model.pt \
    --ckpt-c  ./checkpoints_c/best_model.pt \
    --ckpt-d  ./checkpoints_p5/best_model.pt \
    --ckpt-e  ./checkpoints_p4/best_model.pt \
    --norm-b  ./checkpoints/normaliser.npz \
    --norm-de ./checkpoints_p5/normaliser.npz
```

---

## Ablation Results

| Config | Description | P_d | P_fa | F1 | Params |
|--------|-------------|-----|------|----|--------|
| A | FFT Channeliser (DSP baseline) | 0.549 | 0.024 | 0.625 | N/A |
| B | Scalar MLP (Branch 3) | 0.530 | 0.033 | 0.583 | 83,865 |
| C | Spectrogram CNN only (Branch 1) | 0.612 | 0.012 | 0.712 | 22,124 |
| D | Full model (Branch 1+2+3) | 0.634 | 0.014 | 0.724 | 137,116 |
| E | Spectral-only (Branch 1+3) | 0.617 | 0.010 | 0.722 | 112,620 |
| F | Int8 quantised (from D) | 0.632 | 0.014 | 0.722 | 137,116 |

---

## P_d vs SNR

| Config | -12 to -4 dB | -4 to 0 dB | 0 to 6 dB | 6 to 12 dB | 12 to 20 dB | 20 to 28 dB |
|--------|-------------|------------|-----------|------------|-------------|-------------|
| A FFT  | ~0.00 | ~0.00 | ~0.10 | ~0.45 | ~0.85 | ~0.96 |
| B MLP  | 0.094 | 0.240 | 0.496 | 0.674 | 0.781 | 0.816 |
| C CNN  | 0.119 | 0.332 | 0.595 | 0.736 | 0.880 | 0.939 |
| D Full | 0.187 | 0.367 | 0.608 | 0.750 | 0.888 | 0.935 |
| F Int8 | 0.190 | 0.367 | 0.603 | 0.740 | 0.888 | 0.935 |

---

## Model Size and Inference Time

| Model | Size | Inference |
|-------|------|-----------|
| Float32 (Config D) | 559.8 KB | 5.06 ms |
| Int8 (Config F) | 195.7 KB | 3.45 ms |
| Reduction | 2.9x smaller | 1.5x faster |

---

## Project Targets vs Current Results (Config D)

| Target | Status | Actual |
|--------|--------|--------|
| P_fa ≤ 0.05 | MET | 0.014 |
| Power class accuracy ≥ 0.75 | MET | 0.799 |
| Impairment macro-F1 ≥ 0.70 | Not yet | 0.330 |
| Occupancy F1 ≥ 0.85 | Not yet | 0.724 |
| P_d ≥ 0.90 at SNR ≥ 0 dB | Not yet | 0.608 |

The three unmet targets are expected to close at Stage 2 scale (50k samples). The consistent improvement trend across phases supports this expectation.

---

## Checkpoints

Each phase saves its best checkpoint to its checkpoint directory:

| Phase | Directory | Files |
|-------|-----------|-------|
| Phase 2 | ./checkpoints | best_model.pt, normaliser.npz |
| Phase 3 | ./checkpoints_p3 | best_model.pt, normaliser.npz |
| Phase 4 | ./checkpoints_p4 | best_model.pt, normaliser.npz |
| Phase 5 | ./checkpoints_p5 | best_model.pt, normaliser.npz |
| Phase 6 Config C | ./checkpoints_c | best_model.pt |
