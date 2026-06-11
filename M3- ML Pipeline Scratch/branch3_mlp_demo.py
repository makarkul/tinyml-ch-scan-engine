"""
branch3_mlp_demo.py
===================
Standalone demo of Branch 3 — the MLP that takes 156 scalar features
and predicts GSM channel occupancy.

This file contains only Branch 3. No CNN, no spectrogram, no fading vectors.
It shows exactly:
  1. How the 156 features are extracted from raw IQ
  2. How they are normalised
  3. How the MLP processes them
  4. What the output looks like

This is the Config B model from the ablation study (F1=0.583).

Architecture
------------
  Input  : (156,) scalar features
  Layer 1: Linear(156 → 256) + BatchNorm + ReLU + Dropout
  Layer 2: Linear(256 → 128) + BatchNorm + ReLU + Dropout
  Layer 3: Linear(128 → 64)  + BatchNorm + ReLU
  Output : Linear(64  → 25)  → sigmoid → occupancy probabilities

Usage
-----
  # Run on one sample using the trained checkpoint
  python branch3_mlp_demo.py \\
      --sample  ./gsm_dataset/samples/sample_0000005.npy \\
      --ckpt    ./checkpoints/best_model.pt \\
      --norm    ./checkpoints/normaliser.npz

  # Run on multiple samples and show a summary
  python branch3_mlp_demo.py \\
      --dataset ./gsm_dataset \\
      --ckpt    ./checkpoints/best_model.pt \\
      --norm    ./checkpoints/normaliser.npz \\
      --n-samples 20
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.signal import resample as fft_resample

# ── Signal constants ──────────────────────────────────────────────────
TARGET_RATE = 104_000_000
SCAN_RATE   =   5_000_000
N_WINDOW    =  65_536
N_FFT       =   1_024
WELCH_HOP   =     512
N_SLOTS     =      25
BIN_HZ      = SCAN_RATE / N_FFT
N_FEATURES  = 156

SLOT_BINS = []
for k in range(N_SLOTS):
    c = int(round((-2_400_000 + k*200_000 + SCAN_RATE/2) / BIN_HZ))
    SLOT_BINS.append((max(0, c-20), min(N_FFT, c+21)))


# ─────────────────────────────────────────────────────────────────────
#  STEP 1 — FEATURE EXTRACTION
#  Takes raw (2, 65536) int8 IQ and produces (156,) float32 features
# ─────────────────────────────────────────────────────────────────────

def extract_features(iq_int8: np.ndarray) -> np.ndarray:
    """
    Extract all 156 scalar features from one IQ window.

    Feature groups
    --------------
    F1  (2)  : variance of I and Q channels
    F2  (1)  : envelope variance — low for GMSK constant-envelope
    F3  (1)  : ACF at GSM symbol lag — non-zero when carrier present
    F4  (1)  : kurtosis — ~1.0 for GMSK, ~2.0 for noise
    F7  (1)  : noise floor estimate (median slot energy)
    F8  (25) : spectral flatness per slot — low = carrier present
    F9  (25) : in-band flatness per slot — low = fading notch
    F10 (50) : spectral centroid and spread per slot
    F11 (25) : PAPR per slot — high = CW tone or strong carrier
    F14 (25) : guard-band leakage ratio — high = adjacent channel
    ───────────────────────────────────────────────────────
    Total    : 156 features
    """
    I = iq_int8[0].astype(np.float32)
    Q = iq_int8[1].astype(np.float32)
    x = I + 1j * Q

    # F1: I and Q variance (2 features)
    f1 = np.array([np.var(I), np.var(Q)], dtype=np.float32)

    # F2: envelope variance (1 feature)
    f2 = np.array([np.var(np.sqrt(I**2 + Q**2))], dtype=np.float32)

    # Decimate from 104 MS/s to 5 MS/s
    n_out = int(round(N_WINDOW * SCAN_RATE / TARGET_RATE))
    dec   = fft_resample(x, n_out).astype(np.complex64)

    # F3: ACF at GSM symbol lag (1 feature)
    sym_lag = int(round(SCAN_RATE / 270_833))   # 18 samples
    acf_val = float(np.mean(np.conj(dec[:-sym_lag]) * dec[sym_lag:]).real)
    pwr_val = float(np.mean(np.abs(dec)**2))
    f3 = np.array([acf_val / max(pwr_val, 1e-30)], dtype=np.float32)

    # F4: kurtosis (1 feature)
    m2 = float(np.mean(np.abs(dec)**2))
    m4 = float(np.mean(np.abs(dec)**4))
    f4 = np.array([m4 / max(m2**2, 1e-30)], dtype=np.float32)

    # Welch power spectrum — average of 5 overlapping FFT frames
    window = np.hanning(N_FFT)
    n_segs = (len(dec) - N_FFT) // WELCH_HOP + 1
    acc    = np.zeros(N_FFT, dtype=np.float64)
    for i in range(n_segs):
        seg  = dec[i*WELCH_HOP : i*WELCH_HOP+N_FFT]
        spec = np.fft.fftshift(np.fft.fft(seg*window, n=N_FFT))
        acc += np.abs(spec)**2 / N_FFT
    power = (acc / n_segs).astype(np.float64)

    slot_energy = np.array(
        [np.sum(power[lo:hi]) for lo, hi in SLOT_BINS],
        dtype=np.float64
    )

    # F7: noise floor (1 feature)
    f7 = np.array([float(np.median(slot_energy))], dtype=np.float32)

    # F8: spectral flatness per slot (25 features)
    f8 = np.zeros(N_SLOTS, dtype=np.float32)
    for k, (lo, hi) in enumerate(SLOT_BINS):
        bins = power[lo:hi] + 1e-30
        f8[k] = float(np.exp(np.mean(np.log(bins)))) / \
                max(float(np.mean(bins)), 1e-30)

    # F9: in-band flatness per slot (25 features)
    f9 = np.zeros(N_SLOTS, dtype=np.float32)
    for k, (lo, hi) in enumerate(SLOT_BINS):
        bins  = power[lo:hi]
        f9[k] = float(bins.min()) / max(float(bins.mean()), 1e-30)

    # F10: spectral centroid and spread per slot (50 features)
    f10 = np.zeros(N_SLOTS * 2, dtype=np.float32)
    for k, (lo, hi) in enumerate(SLOT_BINS):
        bins     = power[lo:hi]
        freqs    = np.arange(lo, hi) * BIN_HZ - SCAN_RATE / 2
        total    = float(bins.sum()) + 1e-30
        centroid = float(np.sum(freqs * bins)) / total
        spread   = float(np.sqrt(np.sum((freqs-centroid)**2 * bins) / total))
        f10[k*2]   = centroid / 200_000.0
        f10[k*2+1] = spread   / 200_000.0

    # F11: PAPR per slot (25 features)
    f11 = np.zeros(N_SLOTS, dtype=np.float32)
    for k, (lo, hi) in enumerate(SLOT_BINS):
        bins    = power[lo:hi]
        f11[k]  = float(bins.max()) / max(float(bins.mean()), 1e-30)

    # F14: guard-band leakage per slot (25 features)
    f14 = np.zeros(N_SLOTS, dtype=np.float32)
    for k, (lo, hi) in enumerate(SLOT_BINS):
        in_slot  = float(np.sum(power[lo:hi]))
        guard_lo = float(power[lo-1]) if lo > 0     else 0.0
        guard_hi = float(power[hi])   if hi < N_FFT else 0.0
        f14[k]   = (guard_lo + guard_hi) / max(in_slot, 1e-30)

    return np.concatenate([f1, f2, f3, f4, f7, f8, f9, f10, f11, f14]
                          ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
#  STEP 2 — NORMALISATION
#  Standardise features to zero mean, unit variance using
#  statistics computed from the training set
# ─────────────────────────────────────────────────────────────────────

class FeatureNormaliser:
    """
    Standardise: (feature - mean) / std
    Mean and std are computed from the training set and saved to .npz.
    The same statistics must be applied at inference time.
    """

    def __init__(self):
        self.mean = None
        self.std  = None

    def load(self, path: str):
        d         = np.load(path)
        self.mean = d["mean"].astype(np.float32)
        self.std  = d["std"].astype(np.float32)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.std).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
#  STEP 3 — BRANCH 3 MLP
#  Three fully connected layers with BatchNorm, ReLU, Dropout
#  Input: (batch, 156) normalised features
#  Output: (batch, 25) occupancy logits
# ─────────────────────────────────────────────────────────────────────

class Branch3MLP(nn.Module):
    """
    Branch 3: MLP on 156 scalar features → 25 slot occupancy logits.

    Why three layers
    ----------------
    One layer can only learn linear decision boundaries — straight
    hyperplanes separating occupied from empty in 156-dimensional space.
    Three layers with ReLU activations can learn curved, non-linear
    boundaries. For example: "high energy AND high flatness AND low PAPR
    → empty" is a non-linear rule that requires depth.

    Why BatchNorm
    -------------
    The 156 features span very different scales — F1 variance can be
    0-200, F3 ACF is -1 to +1, F7 noise floor is 0-1000. Even after
    normalisation, internal activations can drift during training.
    BatchNorm keeps activations stable, which stabilises gradients.

    Why Dropout
    -----------
    With 156 input features and 256 hidden units, the MLP has 83k
    parameters trained on only 7000 samples. Without regularisation
    it would memorise the training data. Dropout randomly zeroes 30%
    of neurons during training, forcing the network to learn redundant
    representations that generalise better.
    """

    def __init__(self,
                 n_features : int   = N_FEATURES,
                 n_slots    : int   = N_SLOTS,
                 dropout    : float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1: 156 → 256
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Output head: one logit per slot
        # BCEWithLogitsLoss applies sigmoid internally during training
        # At inference we apply sigmoid manually to get probabilities
        self.occupancy_head = nn.Linear(64, n_slots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 156) normalised scalar features

        Returns
        -------
        (batch, 25) raw logits — apply sigmoid for probabilities
        """
        return self.occupancy_head(self.net(x))


# ─────────────────────────────────────────────────────────────────────
#  DEMO — run on one or more samples and print results
# ─────────────────────────────────────────────────────────────────────

def run_single_sample(sample_path, model, normaliser, threshold=0.65):
    """Run Branch 3 MLP on one sample and print detailed output."""
    d   = np.load(sample_path, allow_pickle=True).item()
    iq  = d["iq"]
    occ = d["occupancy"]
    snr = float(d["snr_db"])

    # Step 1: extract features
    raw_feats  = extract_features(iq)           # (156,) raw

    # Step 2: normalise
    norm_feats = normaliser.transform(raw_feats) # (156,) normalised

    # Step 3: MLP forward pass
    model.eval()
    with torch.no_grad():
        x      = torch.from_numpy(norm_feats).unsqueeze(0)  # (1, 156)
        logits = model(x)                                    # (1, 25)
        probs  = torch.sigmoid(logits).squeeze().numpy()     # (25,)

    # Threshold to binary prediction
    preds   = (probs > threshold).astype(int)
    truth   = occ.astype(int)
    correct = int(np.sum(preds == truth))

    print(f"\nSample: {Path(sample_path).name}")
    print(f"SNR   : {snr:.1f} dB")
    print(f"Truth : occupied slots = {[k for k in range(25) if truth[k]]}")
    print(f"Pred  : occupied slots = {[k for k in range(25) if preds[k]]}")
    print(f"Correct: {correct}/25 slots")
    print()
    print(f"{'Slot':>5}  {'Prob':>8}  {'Pred':>8}  {'Truth':>7}  {'OK?':>4}")
    print("-" * 40)
    for k in range(25):
        ok     = "YES" if preds[k] == truth[k] else "NO"
        marker = " ← carrier" if truth[k] else ""
        print(f"  {k:3d}  {probs[k]:>8.4f}  "
              f"{'carrier' if preds[k] else 'empty':>8}  "
              f"{'carrier' if truth[k] else 'empty':>7}  "
              f"{ok:>4}{marker}")

    print()
    print("Feature summary (first value of each group):")
    groups = {
        "F1 I-variance"    : raw_feats[0],
        "F2 env-variance"  : raw_feats[2],
        "F3 ACF"           : raw_feats[3],
        "F4 kurtosis"      : raw_feats[4],
        "F7 noise-floor"   : raw_feats[5],
        "F8 flatness[0]"   : raw_feats[6],
        "F9 inband-flat[0]": raw_feats[31],
        "F10 centroid[0]"  : raw_feats[56],
        "F11 PAPR[0]"      : raw_feats[106],
        "F14 leakage[0]"   : raw_feats[131],
    }
    for name, val in groups.items():
        print(f"  {name:<22}: {val:.6f}")


def run_dataset(dataset_dir, model, normaliser, n_samples, threshold=0.65):
    """Run on multiple samples and show summary statistics."""
    import glob
    files = sorted(glob.glob(
        str(Path(dataset_dir) / "samples" / "sample_*.npy")
    ))[:n_samples]

    total_correct = 0
    total_slots   = 0
    tp = fp = tn = fn = 0

    for fpath in files:
        d          = np.load(fpath, allow_pickle=True).item()
        raw_feats  = extract_features(d["iq"])
        norm_feats = normaliser.transform(raw_feats)
        truth      = d["occupancy"].astype(int)

        model.eval()
        with torch.no_grad():
            x      = torch.from_numpy(norm_feats).unsqueeze(0)
            logits = model(x)
            probs  = torch.sigmoid(logits).squeeze().numpy()
        preds = (probs > threshold).astype(int)

        for k in range(25):
            total_slots   += 1
            total_correct += (preds[k] == truth[k])
            if preds[k] == 1 and truth[k] == 1: tp += 1
            if preds[k] == 1 and truth[k] == 0: fp += 1
            if preds[k] == 0 and truth[k] == 0: tn += 1
            if preds[k] == 0 and truth[k] == 1: fn += 1

    p_d  = tp / max(tp+fn, 1)
    p_fa = fp / max(fp+tn, 1)
    prec = tp / max(tp+fp, 1)
    f1   = 2*prec*p_d / max(prec+p_d, 1e-10)

    print(f"\nBranch 3 MLP results on {len(files)} samples:")
    print(f"  Threshold : {threshold}")
    print(f"  P_d       : {p_d:.4f}  ({tp} TP / {tp+fn} occupied)")
    print(f"  P_fa      : {p_fa:.4f}  ({fp} FP / {fp+tn} empty)")
    print(f"  F1        : {f1:.4f}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Branch 3 MLP demo — 156 scalar features → occupancy"
    )
    ap.add_argument("--sample",    type=str, default=None,
                    help="Path to one .npy sample (single sample mode)")
    ap.add_argument("--dataset",   type=str, default=None,
                    help="Path to dataset folder (multi-sample mode)")
    ap.add_argument("--n-samples", type=int, default=20,
                    help="Number of samples to evaluate (multi-sample mode)")
    ap.add_argument("--ckpt",      type=str,
                    default="./checkpoints/best_model.pt",
                    help="Path to Branch 3 checkpoint")
    ap.add_argument("--norm",      type=str,
                    default="./checkpoints/normaliser.npz",
                    help="Path to normaliser statistics")
    ap.add_argument("--threshold", type=float, default=0.65,
                    help="Decision threshold (default 0.65)")
    args = ap.parse_args()

    # Load normaliser
    print(f"Loading normaliser from {args.norm}...")
    normaliser = FeatureNormaliser()
    normaliser.load(args.norm)

    # Load model
    print(f"Loading model from {args.ckpt}...")
    model = Branch3MLP()
    ckpt  = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}")
    print(f"  Val F1     : {ckpt.get('val_F1', 'N/A')}")

    print()
    print("Architecture:")
    print("  Input  : (156,) scalar features")
    print("  Layer 1: Linear(156→256) + BatchNorm + ReLU + Dropout(0.3)")
    print("  Layer 2: Linear(256→128) + BatchNorm + ReLU + Dropout(0.3)")
    print("  Layer 3: Linear(128→64)  + BatchNorm + ReLU")
    print("  Output : Linear(64→25)   → sigmoid → probabilities")

    if args.sample:
        run_single_sample(args.sample, model, normaliser, args.threshold)
    elif args.dataset:
        run_dataset(args.dataset, model, normaliser,
                    args.n_samples, args.threshold)
    else:
        print("\nProvide --sample or --dataset to run inference.")
        print("Example:")
        print("  python branch3_mlp_demo.py \\")
        print("    --sample ./gsm_dataset/samples/sample_0000005.npy \\")
        print("    --ckpt   ./checkpoints/best_model.pt \\")
        print("    --norm   ./checkpoints/normaliser.npz")


if __name__ == "__main__":
    main()