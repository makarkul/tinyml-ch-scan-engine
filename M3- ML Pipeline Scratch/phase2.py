import argparse
import time
from pathlib import Path
from xml.parsers.expat import model

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Phase 1 dataset and feature extraction
from phase1 import (
    GSMScanDataset,
    split_dataset,
    N_SLOTS,
    N_IMP_FLAGS,
)


# ─────────────────────────────────────────────────────────────────────
#  SCALAR FEATURE EXTRACTION  (F1-F4, F7-F11, F14)
# ─────────────────────────────────────────────────────────────────────

from scipy.signal import resample as fft_resample

TARGET_RATE = 104_000_000
SCAN_RATE   =   5_000_000
N_WINDOW    = 65_536
N_FFT       = 1024
WELCH_HOP   = N_FFT // 2
BIN_HZ      = SCAN_RATE / N_FFT

SLOT_BINS = []
for _k in range(N_SLOTS):
    _c  = int(round((-2_400_000 + _k * 200_000 + SCAN_RATE / 2) / BIN_HZ))
    SLOT_BINS.append((max(0, _c - 20), min(N_FFT, _c + 21)))

N_SCALAR_FEATURES = 156   # F1(2)+F2(1)+F3(1)+F4(1)+F7(1)+F8(25)+F9(25)+F10(50)+F11(25)+F14(25)


def extract_scalar_features(iq_int8: np.ndarray) -> np.ndarray:
    """
    Extract all 156 scalar features (F1-F4, F7-F11, F14) from one IQ window.

    Steps
    -----
    1. Compute Welch power spectrum (reuses the FFT channeliser pipeline)
    2. Extract wideband features (F1-F4) from raw IQ and decimated signal
    3. Extract per-slot features (F7-F11, F14) from the power spectrum
    4. Normalise frequency-domain features to be scale-invariant

    Returns
    -------
    (156,) float32 — concatenated scalar feature vector
    """
    # ── Raw IQ features ──────────────────────────────────────────────
    I = iq_int8[0].astype(np.float32)
    Q = iq_int8[1].astype(np.float32)
    x = I + 1j * Q

    # F1: variance of I and Q channels separately
    f1 = np.array([np.var(I), np.var(Q)], dtype=np.float32)

    # F2: variance of the envelope sqrt(I^2 + Q^2)
    # Near-zero for constant-envelope signals (GMSK) — strong GMSK indicator
    envelope = np.sqrt(I ** 2 + Q ** 2)
    f2 = np.array([np.var(envelope)], dtype=np.float32)

    # ── Decimated signal features ─────────────────────────────────────
    n_out = int(round(N_WINDOW * SCAN_RATE / TARGET_RATE))
    dec   = fft_resample(x, n_out).astype(np.complex64)

    # F3: normalised ACF at GSM symbol lag
    # GSM symbol rate = 270,833 Hz. At 5 MS/s: lag = round(5e6/270833) = 18 samples
    # Near-zero for noise (in expectation), non-zero for GMSK
    sym_lag = int(round(SCAN_RATE / 270833))
    acf_val = float(np.mean(np.conj(dec[:-sym_lag]) * dec[sym_lag:]).real)
    pwr_val = float(np.mean(np.abs(dec) ** 2))
    f3 = np.array([acf_val / max(pwr_val, 1e-30)], dtype=np.float32)

    # F4: M42 normalised 4th moment (kurtosis)
    # GMSK (constant envelope): ~1.0  |  AWGN: ~2.0  |  Faded GMSK: between
    m2 = float(np.mean(np.abs(dec) ** 2))
    m4 = float(np.mean(np.abs(dec) ** 4))
    f4 = np.array([m4 / max(m2 ** 2, 1e-30)], dtype=np.float32)

    # ── Welch power spectrum ──────────────────────────────────────────
    window  = np.hanning(N_FFT)
    n_segs  = (len(dec) - N_FFT) // WELCH_HOP + 1
    acc     = np.zeros(N_FFT, dtype=np.float64)
    for i in range(n_segs):
        seg  = dec[i * WELCH_HOP : i * WELCH_HOP + N_FFT]
        spec = np.fft.fftshift(np.fft.fft(seg * window, n=N_FFT))
        acc += np.abs(spec) ** 2 / N_FFT
    power = (acc / n_segs).astype(np.float64)

    # Per-slot energies
    slot_energy = np.array(
        [np.sum(power[lo:hi]) for lo, hi in SLOT_BINS], dtype=np.float64
    )

    # F7: noise floor — median of all slot energies
    noise_floor = float(np.median(slot_energy))
    f7 = np.array([noise_floor], dtype=np.float32)

    # F8: spectral flatness per slot — geometric mean / arithmetic mean
    # 1.0 = perfectly flat (noise-like), < 1.0 = peaked (carrier present)
    f8 = np.zeros(N_SLOTS, dtype=np.float32)
    for k, (lo, hi) in enumerate(SLOT_BINS):
        bins      = power[lo:hi] + 1e-30
        geo_mean  = float(np.exp(np.mean(np.log(bins))))
        arith_mean = float(np.mean(bins))
        f8[k]     = geo_mean / max(arith_mean, 1e-30)

    # F9: in-band flatness — min / mean ratio per slot
    # Low value = deep fading notch present within the slot
    f9 = np.zeros(N_SLOTS, dtype=np.float32)
    for k, (lo, hi) in enumerate(SLOT_BINS):
        bins  = power[lo:hi]
        f9[k] = float(bins.min()) / max(float(bins.mean()), 1e-30)

    # F10: spectral centroid and spread per slot
    # Centroid: CFO shifts this away from slot centre
    # Spread: measures how concentrated the energy is
    # Both normalised by slot bandwidth (200 kHz) to be scale-invariant
    f10 = np.zeros(N_SLOTS * 2, dtype=np.float32)
    for k, (lo, hi) in enumerate(SLOT_BINS):
        bins     = power[lo:hi]
        freqs    = np.arange(lo, hi) * BIN_HZ - SCAN_RATE / 2
        total    = float(bins.sum()) + 1e-30
        centroid = float(np.sum(freqs * bins)) / total
        spread   = float(np.sqrt(np.sum((freqs - centroid) ** 2 * bins) / total))
        slot_bw  = 200_000.0
        f10[k * 2]     = centroid / slot_bw   # normalised centroid
        f10[k * 2 + 1] = spread   / slot_bw   # normalised spread

    # F11: PAPR per slot — peak / mean power
    # High PAPR = CW tone or strong carrier present in slot
    f11 = np.zeros(N_SLOTS, dtype=np.float32)
    for k, (lo, hi) in enumerate(SLOT_BINS):
        bins    = power[lo:hi]
        f11[k]  = float(bins.max()) / max(float(bins.mean()), 1e-30)

    # F14: guard-band leakage ratio per slot
    # High value = energy bleeding from adjacent occupied channel
    f14 = np.zeros(N_SLOTS, dtype=np.float32)
    for k, (lo, hi) in enumerate(SLOT_BINS):
        in_slot  = float(np.sum(power[lo:hi]))
        guard_lo = float(power[lo - 1]) if lo > 0      else 0.0
        guard_hi = float(power[hi])     if hi < N_FFT  else 0.0
        f14[k]   = (guard_lo + guard_hi) / max(in_slot, 1e-30)

    # ── Concatenate all features ──────────────────────────────────────
    features = np.concatenate([f1, f2, f3, f4, f7, f8, f9, f10, f11, f14])
    return features.astype(np.float32)   # (156,)


# ─────────────────────────────────────────────────────────────────────
#  FEATURE NORMALISATION
# ─────────────────────────────────────────────────────────────────────

class FeatureNormaliser:
    """
    Standardise features to zero mean and unit variance.

    Computed from the training set and applied to val/test sets.
    Saved alongside the model checkpoint so inference uses the same
    statistics as training.

    Why normalise
    -------------
    The 156 features span very different scales:
      F1 (variance): 0–200
      F3 (ACF):     -1 to +1
      F7 (noise):   0–1000
      F10 (centroid/BW): -0.5 to +0.5
    Without normalisation, the gradient for large-scale features
    dominates and the MLP struggles to learn from small-scale features.
    """

    def __init__(self):
        self.mean = None   # (156,) float32
        self.std  = None   # (156,) float32

    def fit(self, dataset: GSMScanDataset, n_samples: int = None) -> None:
        """
        Compute mean and std from a dataset.
        Uses a subset of the training set for speed.
        """
        n = min(n_samples or len(dataset), len(dataset))
        feats = np.zeros((n, N_SCALAR_FEATURES), dtype=np.float32)

        print(f"  Computing normalisation statistics from {n} samples...")
        for i in range(n):
            # Load raw IQ directly (not through __getitem__ which gives energy ratios)
            d = np.load(str(dataset.files[i]), allow_pickle=True).item()
            feats[i] = extract_scalar_features(d["iq"])

        self.mean = feats.mean(axis=0)
        self.std  = feats.std(axis=0)
        # Avoid division by zero for constant features
        self.std  = np.where(self.std < 1e-8, 1.0, self.std)
        print(f"  Done. Feature mean range: [{self.mean.min():.3f}, {self.mean.max():.3f}]")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply normalisation: (x - mean) / std."""
        return ((features - self.mean) / self.std).astype(np.float32)

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path: str) -> None:
        d = np.load(path)
        self.mean = d["mean"]
        self.std  = d["std"]


# ─────────────────────────────────────────────────────────────────────
#  DATASET WITH FULL SCALAR FEATURES
# ─────────────────────────────────────────────────────────────────────

class ScalarFeatureDataset(torch.utils.data.Dataset):
    """
    Dataset variant that extracts all 156 scalar features instead of
    the 25-element energy ratios from Phase 1.

    The normaliser is applied at load time so all returned tensors
    are already standardised.
    """

    def __init__(self, sample_files: list, normaliser: FeatureNormaliser = None):
        self.files      = [Path(f) for f in sample_files]
        self.normaliser = normaliser

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple:
        d = np.load(str(self.files[idx]), allow_pickle=True).item()

        features = extract_scalar_features(d["iq"])   # (156,)

        if self.normaliser is not None:
            features = self.normaliser.transform(features)

        features_t  = torch.from_numpy(features)
        occupancy_t = torch.from_numpy(d["occupancy"].astype(np.float32))
        snr_t       = torch.tensor(float(d["snr_db"]), dtype=torch.float32)

        return features_t, occupancy_t, snr_t


# ─────────────────────────────────────────────────────────────────────
#  MODEL — BRANCH 3 MLP
# ─────────────────────────────────────────────────────────────────────

class Branch3MLP(nn.Module):

    def __init__(self, n_features: int = N_SCALAR_FEATURES,
                 n_slots: int = N_SLOTS, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Occupancy head: one logit per slot
        self.occupancy_head = nn.Linear(64, n_slots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.occupancy_head(self.net(x))


# ─────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(logits: torch.Tensor,
                    targets: torch.Tensor,
                    threshold: float = 0.6) -> dict:

    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    TP = int(((preds == 1) & (targets == 1)).sum())
    FP = int(((preds == 1) & (targets == 0)).sum())
    TN = int(((preds == 0) & (targets == 0)).sum())
    FN = int(((preds == 0) & (targets == 1)).sum())

    P_d  = TP / max(TP + FN, 1)
    P_fa = FP / max(FP + TN, 1)
    prec = TP / max(TP + FP, 1)
    F1   = 2 * prec * P_d / max(prec + P_d, 1e-10)

    return dict(P_d=P_d, P_fa=P_fa, F1=F1, TP=TP, FP=FP, TN=TN, FN=FN)


# ─────────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimiser: torch.optim.Optimizer) -> dict:
    """Run one full pass over the training set."""
    model.train()
    total_loss = 0.0
    all_logits  = []
    all_targets = []

    for features, occupancy, _ in loader:
        optimiser.zero_grad()
        logits = model(features)            # (batch, 25)
        loss   = criterion(logits, occupancy)
        loss.backward()
        optimiser.step()

        total_loss  += loss.item() * len(features)
        all_logits .append(logits.detach())
        all_targets.append(occupancy)

    all_logits  = torch.cat(all_logits,  dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    m = compute_metrics(all_logits, all_targets)
    m["loss"] = total_loss / len(loader.dataset)
    return m


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             threshold: float = 0.6) -> dict:
    """Evaluate on val or test set."""
    model.eval()
    total_loss  = 0.0
    all_logits  = []
    all_targets = []
    all_snr     = []

    for features, occupancy, snr in loader:
        logits = model(features)
        loss   = criterion(logits, occupancy)

        total_loss  += loss.item() * len(features)
        all_logits .append(logits)
        all_targets.append(occupancy)
        all_snr    .append(snr)

    all_logits  = torch.cat(all_logits,  dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_snr     = torch.cat(all_snr,     dim=0)

    m = compute_metrics(all_logits, all_targets, threshold=threshold)
    m["loss"] = total_loss / len(loader.dataset)

    # P_d vs SNR breakdown
    snr_bins = [(-12,-4), (-4, 0), (0, 6), (6, 12), (12, 20), (20, 28)]
    for lo, hi in snr_bins:
        mask = (all_snr >= lo) & (all_snr < hi)
        if mask.sum() > 0:
            mi = compute_metrics(all_logits[mask], all_targets[mask])
            m[f"P_d_snr_{lo}_{hi}"] = mi["P_d"]

    return m


# ─────────────────────────────────────────────────────────────────────
#  MAIN TRAINING SCRIPT
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phase 2 — Branch 3 MLP training")
    ap.add_argument("--dataset",    type=str, required=True)
    ap.add_argument("--n-samples",  type=int, default=None)
    ap.add_argument("--epochs",     type=int, default=50)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers",type=int, default=2)
    ap.add_argument("--dropout",    type=float, default=0.3)
    ap.add_argument("--checkpoint", type=str, default="./checkpoints")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset split ─────────────────────────────────────────────────
    print("Building dataset splits...")
    train_base, val_base, test_base = split_dataset(
        args.dataset, n_samples=args.n_samples, seed=args.seed
    )
    print(f"  Train: {len(train_base)}  Val: {len(val_base)}  "
          f"Test: {len(test_base)}")

    # ── Normalisation ─────────────────────────────────────────────────
    print("\nFitting feature normaliser on training set...")
    normaliser = FeatureNormaliser()
    normaliser.fit(train_base, n_samples=None)
    normaliser.save(str(checkpoint_dir / "normaliser.npz"))

    # ── DataLoaders ───────────────────────────────────────────────────
    train_ds = ScalarFeatureDataset(train_base.files, normaliser)
    val_ds   = ScalarFeatureDataset(val_base.files,   normaliser)
    test_ds  = ScalarFeatureDataset(test_base.files,  normaliser)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    # ── Model ─────────────────────────────────────────────────────────
    model = Branch3MLP(dropout=args.dropout)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: Branch3MLP")
    print(f"  Parameters: {n_params:,}")

    # ── Loss — BCEWithLogitsLoss with class-balance weight ────────────
    # pos_weight = n_negative / n_positive ≈ 8.9 for 10.1% occupied
    pos_weight = torch.tensor([4.0])
    criterion  = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.expand(N_SLOTS)
    )

    # ── Optimiser and scheduler ───────────────────────────────────────
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", patience=5, factor=0.5
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_val_f1    = -1.0
    early_stop_cnt = 0
    early_stop_pat = 10

    print(f"\nTraining for up to {args.epochs} epochs "
          f"(early stop patience={early_stop_pat})...")
    print(f"{'Epoch':>6}  {'TrLoss':>8}  {'TrF1':>7}  {'TrPd':>7}  "
          f"{'TrPfa':>7}  {'VaLoss':>8}  {'VaF1':>7}  {'VaPd':>7}  "
          f"{'VaPfa':>7}  {'LR':>9}")
    print("─" * 90)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_m = train_one_epoch(model, train_loader, criterion, optimiser)
        val_m   = evaluate(model, val_loader, criterion)

        scheduler.step(val_m["F1"])

        lr_now = optimiser.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0

        print(f"{epoch:>6}  {train_m['loss']:>8.4f}  "
              f"{train_m['F1']:>7.4f}  {train_m['P_d']:>7.4f}  "
              f"{train_m['P_fa']:>7.4f}  {val_m['loss']:>8.4f}  "
              f"{val_m['F1']:>7.4f}  {val_m['P_d']:>7.4f}  "
              f"{val_m['P_fa']:>7.4f}  {lr_now:>9.2e}  "
              f"({elapsed:.1f}s)")

        # Save best checkpoint
        if val_m["F1"] > best_val_f1:
            best_val_f1 = val_m["F1"]
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_F1"     : best_val_f1,
                "val_P_d"    : val_m["P_d"],
                "val_P_fa"   : val_m["P_fa"],
            }, str(checkpoint_dir / "best_model.pt"))
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= early_stop_pat:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {early_stop_pat} epochs)")
                break
    
    # Tune threshold on validation set
    print("\nTuning decision threshold on validation set...")
    print(f"  {'Threshold':>10}  {'P_d':>7}  {'P_fa':>7}  {'F1':>7}")
    print("  " + "-"*36)
    best_threshold = 0.5
    best_val_f1    = -1.0
    for thresh in np.arange(0.3, 0.81, 0.05):
        vm = evaluate(model, val_loader, criterion, threshold=float(thresh))
        print(f"  {thresh:>10.2f}  {vm['P_d']:>7.4f}  {vm['P_fa']:>7.4f}  {vm['F1']:>7.4f}"
          + ("  *" if vm['P_fa'] <= 0.05 else ""))
        if vm['P_fa'] <= 0.05 and vm['F1'] > best_val_f1:
            best_val_f1    = vm['F1']
            best_threshold = float(thresh)
    print(f"\n  Best threshold: {best_threshold:.2f}  "
      f"(val F1={best_val_f1:.4f}, P_fa<=0.05)")

# Evaluate test set at the tuned threshold
    test_m = evaluate(model, test_loader, criterion, threshold=best_threshold)

    # ── Final evaluation on test set ──────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(str(checkpoint_dir / "best_model.pt"))
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_loader, criterion)

    print(f"\n{'='*55}")
    print(f"Phase 2 Test Results  (best val F1 at epoch {ckpt['epoch']})")
    print(f"{'='*55}")
    print(f"  P_d        : {test_m['P_d']:.4f}  "
          f"({test_m['TP']} TP / {test_m['TP']+test_m['FN']} occupied)")
    print(f"  P_fa       : {test_m['P_fa']:.4f}  "
          f"({test_m['FP']} FP / {test_m['FP']+test_m['TN']} empty)")
    print(f"  F1         : {test_m['F1']:.4f}")
    print(f"  Precision  : {test_m['TP']/max(test_m['TP']+test_m['FP'],1):.4f}")
    print()
    print("  P_d vs SNR:")
    for k, v in sorted(test_m.items()):
        if k.startswith("P_d_snr"):
            parts = k.split("_")
            lo, hi = parts[-2], parts[-1]
            print(f"    SNR [{lo:>4}, {hi:>3}) dB : {v:.4f}")

    print()
    print(f"  Baseline comparison:")
    print(f"    FFT Channeliser (Welch, k=1.38) : "
          f"P_d=0.562  P_fa=0.028  F1=0.620")
    print(f"    This model                      : "
          f"P_d={test_m['P_d']:.3f}  "
          f"P_fa={test_m['P_fa']:.3f}  "
          f"F1={test_m['F1']:.3f}")
    delta = test_m['F1'] - 0.620
    sign  = "+" if delta >= 0 else ""
    print(f"    Delta F1                        : {sign}{delta:.3f}")


if __name__ == "__main__":
    main()