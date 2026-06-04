"""
phase4_train.py
===============
Phase 4 — Branch 1 (2D CNN on spectrogram) fused with Branch 3 (MLP).

What is new in Phase 4
----------------------
Branch 1: a 2D CNN that processes a per-slot log-power spectrogram.
Each slot gets its own (41, 5) spectrogram — 41 frequency bins across
5 time frames from the Welch segments. The CNN learns spectral shape
features: the GMSK hump profile, fading notches, CW tone spikes, and
adjacent channel leakage patterns.

The CNN output (64-dim per slot) is concatenated with the Branch 3
trunk output (64-dim global) and passed through a per-slot fusion
layer before the occupancy head. This gives each slot's prediction
access to both its local spectral shape and the global signal context.

Why this should help
--------------------
The scalar features (Phase 2/3) captured aggregate statistics per slot
but could not distinguish between:
  - A weak GMSK carrier and adjacent channel leakage (both raise F8)
  - Static fading and an empty slot (both have low energy)
  - A CW tone and a strong GMSK carrier (both have high PAPR)

The spectrogram carries the actual shape of the energy distribution
across 41 bins and 5 time frames. A GMSK carrier has a characteristic
raised-cosine spectral hump. Leakage looks different. Fading leaves
notches at specific frequencies. The CNN learns to distinguish these.

Architecture
------------
  Input A: scalar features (156,)  → Branch3 trunk → (64,) global
  Input B: spectrogram (25, 41, 5) → Branch1 CNN   → (25, 64) per-slot

  Fusion per slot k:
    cat([global(64), cnn_k(64)]) → (128,) → FC(64) → ReLU → logit

  Output heads (from fused 64-dim per slot):
    occupancy   : (25,)   BCEWithLogitsLoss, pos_weight=4.0
    power_class : (25, 4) CrossEntropyLoss  (masked)
    gmsk        : (25,)   BCEWithLogitsLoss (masked)
    impairment  : (6,)    BCEWithLogitsLoss (from global only)

  Total parameters: ~120,420

F6 spectrogram extraction
--------------------------
  1. Decimate IQ to 5 MS/s (3151 samples)
  2. Compute 5 Welch segments (N_FFT=1024, hop=512)
  3. Take log10 of power per bin → dB values
  4. Extract 41-bin window per slot → (25, 41, 5)
  5. Subtract per-slot median → relative dB (noise-floor invariant)

Usage
-----
  python phase4_train.py --dataset ./gsm_dataset
  python phase4_train.py --dataset ./gsm_dataset --epochs 100 --num-workers 4

Dependencies: numpy, scipy, torch, phase1_dataset.py, phase2_train.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phase1 import split_dataset, N_SLOTS, N_IMP_FLAGS
from phase2  import (
    extract_scalar_features,
    FeatureNormaliser,
    N_SCALAR_FEATURES,
)
from phase3   import (
    MaskedMultiHeadLoss,
    compute_occupancy_metrics,
    compute_power_class_accuracy,
    compute_impairment_f1,
)

from scipy.signal import resample as fft_resample

N_POWER_CLASSES = 4

# ── Signal constants ──────────────────────────────────────────────────
TARGET_RATE = 104_000_000
SCAN_RATE   =   5_000_000
N_WINDOW    = 65_536
N_FFT       = 1024
WELCH_HOP   = N_FFT // 2
BIN_HZ      = SCAN_RATE / N_FFT
N_FRAMES    = (int(round(N_WINDOW * SCAN_RATE / TARGET_RATE)) - N_FFT) // WELCH_HOP + 1  # 5

SLOT_BINS = []
for _k in range(N_SLOTS):
    _c  = int(round((-2_400_000 + _k * 200_000 + SCAN_RATE / 2) / BIN_HZ))
    SLOT_BINS.append((max(0, _c - 20), min(N_FFT, _c + 21)))


# ─────────────────────────────────────────────────────────────────────
#  F6 SPECTROGRAM EXTRACTION
# ─────────────────────────────────────────────────────────────────────

def extract_spectrogram(iq_int8: np.ndarray) -> np.ndarray:
    """
    Extract the F6 per-slot log-power spectrogram.

    Steps
    -----
    1. Decimate 104 MS/s → 5 MS/s  (3151 samples)
    2. Compute 5 Welch segments (N_FFT=1024, hop=512, Hann window)
       → power spectrum per segment: (1024, 5)
    3. Convert to log-power: 10 * log10(power + 1e-10)  [dB]
    4. Extract per-slot 41-bin windows → (25, 41, 5)
    5. Subtract per-slot median → relative dB
       This makes the feature invariant to the absolute noise floor.
       A carrier stands out as positive dB above its slot's median.

    Returns
    -------
    (25, 41, 5) float32 — relative log-power per slot, freq bin, frame
    """
    x     = (iq_int8[0].astype(np.float32) +
              1j * iq_int8[1].astype(np.float32))
    n_out = int(round(N_WINDOW * SCAN_RATE / TARGET_RATE))
    dec   = fft_resample(x, n_out).astype(np.complex64)

    # Build per-frame power spectrum: (N_FFT, N_FRAMES)
    window   = np.hanning(N_FFT)
    n_frames = (len(dec) - N_FFT) // WELCH_HOP + 1
    spec_db  = np.zeros((N_FFT, n_frames), dtype=np.float32)

    for i in range(n_frames):
        seg           = dec[i * WELCH_HOP : i * WELCH_HOP + N_FFT]
        s             = np.fft.fftshift(np.fft.fft(seg * window, n=N_FFT))
        power         = np.abs(s) ** 2 / N_FFT
        spec_db[:, i] = (10.0 * np.log10(power + 1e-10)).astype(np.float32)

    # Extract per-slot spectrogram: (25, 41, n_frames)
    slot_spec = np.zeros((N_SLOTS, 41, n_frames), dtype=np.float32)
    for k, (lo, hi) in enumerate(SLOT_BINS):
        n_bins = hi - lo
        slot_spec[k, :n_bins, :] = spec_db[lo:hi, :]

    # Subtract per-slot median — makes feature noise-floor invariant
    for k in range(N_SLOTS):
        slot_spec[k] -= np.median(slot_spec[k])

    return slot_spec   # (25, 41, 5)


# ─────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────

class Phase4Dataset(torch.utils.data.Dataset):
    """
    Returns scalar features, per-slot spectrogram, and all labels.

    Items:
      scalars     : (156,)      float32 — normalised scalar features
      spectrogram : (25, 41, 5) float32 — relative log-power per slot
      occupancy   : (25,)       float32 — binary ground truth
      power_class : (25,)       int64   — 0-3 occupied, -1 empty
      imp_flags   : (6,)        float32 — multi-label impairment binary
      snr         : ()          float32
    """

    def __init__(self, sample_files: list,
                 normaliser: FeatureNormaliser = None):
        self.files      = [Path(f) for f in sample_files]
        self.normaliser = normaliser

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple:
        d        = np.load(str(self.files[idx]), allow_pickle=True).item()
        scalars  = extract_scalar_features(d["iq"])
        spec     = extract_spectrogram(d["iq"])

        if self.normaliser is not None:
            scalars = self.normaliser.transform(scalars)

        return (
            torch.from_numpy(scalars),
            torch.from_numpy(spec),
            torch.from_numpy(d["occupancy"].astype(np.float32)),
            torch.from_numpy(d["power_class"].astype(np.int64)),
            torch.from_numpy(d["impairment_flags"].astype(np.float32)),
            torch.tensor(float(d["snr_db"]), dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────
#  BRANCH 1 — 2D CNN ON SPECTROGRAM
# ─────────────────────────────────────────────────────────────────────

class Branch1CNN(nn.Module):
    """
    2D CNN that processes per-slot log-power spectrograms.

    Input:  (batch, 25, 41, 5) spectrogram
    Output: (batch, 25, 64)    per-slot feature vectors

    Processing steps
    ----------------
    1. Reshape to (batch*25, 1, 41, 5) — treat each slot independently
       The CNN processes all 25 slots in parallel using the batch dim.

    2. Two conv layers with 3x3 kernels and padding=1:
       - Padding preserves spatial dimensions through convolutions
       - The CNN can see the full 41-bin spectral profile per slot
       - It can also see correlations across the 5 time frames

    3. AdaptiveAvgPool2d((4, 2)):
       - Reduces (41, 5) → (4, 2) regardless of exact input size
       - Averages over spatial regions, retaining the most important
         patterns while reducing parameters in the FC layer

    4. Flatten and FC: 32*4*2=256 → 64

    5. Reshape back to (batch, 25, 64)

    Why 3x3 kernels
    ---------------
    A 3x3 kernel over (freq, time) learns local spectral patterns:
    - The GMSK spectral hump spans ~5 consecutive freq bins
    - A fading notch affects 1-3 consecutive bins
    - A CW tone creates a single-bin spike
    - Adjacent channel leakage creates an asymmetric pattern
    These are all local patterns well-suited to 3x3 convolutions.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            # Conv layer 1: 1 → 16 channels, 3x3, pad=1
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Conv layer 2: 16 → 32 channels, 3x3, pad=1
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Adaptive pooling: any (H, W) → (4, 2)
            nn.AdaptiveAvgPool2d((4, 2)),
        )

        # FC: 32*4*2 → 64
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        spec : (batch, 25, 41, 5) float32

        Returns
        -------
        (batch, 25, 64) float32 — per-slot CNN features
        """
        B = spec.shape[0]

        # Reshape: (batch, 25, 41, 5) → (batch*25, 1, 41, 5)
        x = spec.view(B * N_SLOTS, 1, 41, N_FRAMES)

        # CNN: (batch*25, 1, 41, 5) → (batch*25, 32, 4, 2)
        x = self.cnn(x)

        # Flatten: (batch*25, 32*4*2)
        x = x.view(B * N_SLOTS, -1)

        # FC: (batch*25, 256) → (batch*25, 64)
        x = self.fc(x)

        # Reshape back: (batch*25, 64) → (batch, 25, 64)
        return x.view(B, N_SLOTS, 64)


# ─────────────────────────────────────────────────────────────────────
#  FULL MODEL — BRANCH 1 + BRANCH 3 FUSED
# ─────────────────────────────────────────────────────────────────────

class FusedModel(nn.Module):
    """
    Branch 1 (CNN) + Branch 3 (MLP) with per-slot late fusion.

    Fusion strategy
    ---------------
    Branch 3 trunk produces a single 64-dim vector that summarises
    global signal properties (noise floor, wideband kurtosis, ACF).
    This global context is the same for all 25 slots.

    Branch 1 CNN produces 25 slot-specific 64-dim vectors capturing
    the spectral shape of each slot independently.

    To fuse: expand the global vector to (batch, 25, 64), concatenate
    with the per-slot CNN vectors → (batch, 25, 128), then apply a
    shared per-slot FC layer to combine both information sources.

    This design means the occupancy decision for each slot uses:
    - The slot's own spectral shape (from Branch 1)
    - The overall signal context (from Branch 3)

    The impairment head uses the global Branch 3 representation only
    since impairments are sample-level, not per-slot.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        # Branch 3: shared trunk on scalar features
        self.branch3_trunk = nn.Sequential(
            nn.Linear(N_SCALAR_FEATURES, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Branch 1: per-slot CNN
        self.branch1_cnn = Branch1CNN(dropout=dropout)

        # Per-slot fusion: cat(global_64, slot_64) → 64
        self.fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Output heads — all from fused 64-dim per-slot representation
        self.occupancy_head   = nn.Linear(64, 1)          # → (batch, 25)
        self.power_class_head = nn.Linear(64, N_POWER_CLASSES)  # → (batch, 25, 4)
        self.gmsk_head        = nn.Linear(64, 1)          # → (batch, 25)

        # Impairment head uses global representation (not per-slot)
        self.impairment_head  = nn.Linear(64, N_IMP_FLAGS)

    def forward(self, scalars: torch.Tensor,
                spec: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        scalars : (batch, 156) float32
        spec    : (batch, 25, 41, 5) float32

        Returns
        -------
        dict of raw logits
        """
        B = scalars.shape[0]

        # Branch 3: global features → (batch, 64)
        global_feat = self.branch3_trunk(scalars)

        # Branch 1: per-slot spectrogram → (batch, 25, 64)
        slot_feat = self.branch1_cnn(spec)

        # Expand global: (batch, 64) → (batch, 25, 64)
        global_exp = global_feat.unsqueeze(1).expand(-1, N_SLOTS, -1)

        # Fuse: (batch, 25, 128) → (batch, 25, 64)
        fused = self.fusion(
            torch.cat([global_exp, slot_feat], dim=-1)
        )

        # Per-slot heads: reshape to (batch*25, 64) for linear layers
        fused_flat = fused.view(B * N_SLOTS, 64)

        occ_logits = self.occupancy_head(fused_flat).view(B, N_SLOTS)
        pc_logits  = self.power_class_head(fused_flat).view(
                         B, N_SLOTS, N_POWER_CLASSES)
        gmsk_logits = self.gmsk_head(fused_flat).view(B, N_SLOTS)

        # Impairment: from global representation
        imp_logits  = self.impairment_head(global_feat)

        return {
            "occupancy"  : occ_logits,
            "power_class": pc_logits,
            "gmsk"       : gmsk_logits,
            "impairment" : imp_logits,
        }


# ─────────────────────────────────────────────────────────────────────
#  TRAINING AND EVALUATION LOOPS
# ─────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimiser) -> dict:
    model.train()
    total_loss = 0.0
    all_occ_l = []; all_occ_t = []
    all_pc_l  = []; all_pc_t  = []
    all_imp_l = []; all_imp_t = []

    for scalars, spec, occ, pc, imp, _ in loader:
        optimiser.zero_grad()
        preds = model(scalars, spec)
        loss, _ = criterion(preds, occ, pc, imp)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(scalars)
        all_occ_l.append(preds["occupancy"].detach())
        all_occ_t.append(occ)
        all_pc_l .append(preds["power_class"].detach())
        all_pc_t .append(pc)
        all_imp_l.append(preds["impairment"].detach())
        all_imp_t.append(imp)

    occ_l = torch.cat(all_occ_l); occ_t = torch.cat(all_occ_t)
    pc_l  = torch.cat(all_pc_l);  pc_t  = torch.cat(all_pc_t)
    imp_l = torch.cat(all_imp_l); imp_t = torch.cat(all_imp_t)

    occ_m  = compute_occupancy_metrics(occ_l, occ_t)
    pc_acc = compute_power_class_accuracy(pc_l, pc_t, occ_t.bool())
    imp_f1 = compute_impairment_f1(imp_l, imp_t)

    return {
        "loss"   : total_loss / len(loader.dataset),
        "F1"     : occ_m["F1"],
        "P_d"    : occ_m["P_d"],
        "P_fa"   : occ_m["P_fa"],
        "pc_acc" : pc_acc,
        "imp_f1" : imp_f1,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, threshold=0.5) -> dict:
    model.eval()
    total_loss = 0.0
    all_occ_l = []; all_occ_t = []
    all_pc_l  = []; all_pc_t  = []
    all_imp_l = []; all_imp_t = []
    all_snr   = []

    for scalars, spec, occ, pc, imp, snr in loader:
        preds = model(scalars, spec)
        loss, _ = criterion(preds, occ, pc, imp)
        total_loss += loss.item() * len(scalars)
        all_occ_l.append(preds["occupancy"]); all_occ_t.append(occ)
        all_pc_l .append(preds["power_class"]); all_pc_t.append(pc)
        all_imp_l.append(preds["impairment"]); all_imp_t.append(imp)
        all_snr.append(snr)

    occ_l = torch.cat(all_occ_l); occ_t = torch.cat(all_occ_t)
    pc_l  = torch.cat(all_pc_l);  pc_t  = torch.cat(all_pc_t)
    imp_l = torch.cat(all_imp_l); imp_t = torch.cat(all_imp_t)
    snr_t = torch.cat(all_snr)

    occ_m  = compute_occupancy_metrics(occ_l, occ_t, threshold)
    pc_acc = compute_power_class_accuracy(pc_l, pc_t, occ_t.bool())
    imp_f1 = compute_impairment_f1(imp_l, imp_t)

    m = {
        "loss"   : total_loss / len(loader.dataset),
        "F1"     : occ_m["F1"],
        "P_d"    : occ_m["P_d"],
        "P_fa"   : occ_m["P_fa"],
        "pc_acc" : pc_acc,
        "imp_f1" : imp_f1,
        "TP": occ_m["TP"], "FP": occ_m["FP"],
        "TN": occ_m["TN"], "FN": occ_m["FN"],
    }

    for lo, hi in [(-12,-4),(-4,0),(0,6),(6,12),(12,20),(20,28)]:
        mask = (snr_t >= lo) & (snr_t < hi)
        if mask.sum() > 0:
            mi = compute_occupancy_metrics(occ_l[mask], occ_t[mask], threshold)
            m[f"P_d_snr_{lo}_{hi}"] = mi["P_d"]

    return m


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Phase 4 — Branch 1 CNN + Branch 3 MLP fusion"
    )
    ap.add_argument("--dataset",     type=str,   required=True)
    ap.add_argument("--n-samples",   type=int,   default=None)
    ap.add_argument("--epochs",      type=int,   default=100)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--batch-size",  type=int,   default=32)
    ap.add_argument("--num-workers", type=int,   default=2)
    ap.add_argument("--dropout",     type=float, default=0.3)
    ap.add_argument("--checkpoint",  type=str,   default="./checkpoints_p4")
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────
    print("Building dataset splits...")
    train_base, val_base, test_base = split_dataset(
        args.dataset, n_samples=args.n_samples, seed=args.seed
    )
    print(f"  Train: {len(train_base)}  Val: {len(val_base)}  "
          f"Test: {len(test_base)}")

    # ── Normaliser ────────────────────────────────────────────────────
    print("\nFitting feature normaliser on full training set...")
    normaliser = FeatureNormaliser()
    normaliser.fit(train_base, n_samples=None)
    normaliser.save(str(ckpt_dir / "normaliser.npz"))

    # ── DataLoaders ───────────────────────────────────────────────────
    train_ds = Phase4Dataset(train_base.files, normaliser)
    val_ds   = Phase4Dataset(val_base.files,   normaliser)
    test_ds  = Phase4Dataset(test_base.files,  normaliser)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    # ── Model ─────────────────────────────────────────────────────────
    model    = FusedModel(dropout=args.dropout)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: FusedModel (Branch1 CNN + Branch3 MLP)")
    print(f"  Parameters: {n_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────
    criterion = MaskedMultiHeadLoss(w_occ=1.0, w_pc=0.5,
                                    w_gmsk=0.3, w_imp=0.5)

    # ── Optimiser ─────────────────────────────────────────────────────
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", patience=5, factor=0.5
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_val_f1 = -1.0
    es_count    = 0
    es_patience = 10

    print(f"\nTraining for up to {args.epochs} epochs "
          f"(early stop patience={es_patience})...")
    print(f"{'Ep':>4}  {'TrLoss':>7}  {'TrF1':>6}  {'TrPd':>6}  "
          f"{'TrPfa':>6}  {'VaLoss':>7}  {'VaF1':>6}  {'VaPd':>6}  "
          f"{'VaPfa':>6}  {'PcAcc':>6}  {'ImpF1':>6}  {'LR':>8}")
    print("─" * 100)

    for epoch in range(1, args.epochs + 1):
        t0      = time.perf_counter()
        train_m = train_one_epoch(model, train_loader, criterion, optimiser)
        val_m   = evaluate(model, val_loader, criterion)
        scheduler.step(val_m["F1"])
        elapsed = time.perf_counter() - t0
        lr_now  = optimiser.param_groups[0]["lr"]

        print(f"{epoch:>4}  {train_m['loss']:>7.4f}  {train_m['F1']:>6.4f}  "
              f"{train_m['P_d']:>6.4f}  {train_m['P_fa']:>6.4f}  "
              f"{val_m['loss']:>7.4f}  {val_m['F1']:>6.4f}  "
              f"{val_m['P_d']:>6.4f}  {val_m['P_fa']:>6.4f}  "
              f"{val_m['pc_acc']:>6.4f}  {val_m['imp_f1']:>6.4f}  "
              f"{lr_now:>8.2e}  ({elapsed:.1f}s)")

        if val_m["F1"] > best_val_f1:
            best_val_f1 = val_m["F1"]
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_F1"     : best_val_f1,
                "val_P_d"    : val_m["P_d"],
                "val_P_fa"   : val_m["P_fa"],
            }, str(ckpt_dir / "best_model.pt"))
            es_count = 0
        else:
            es_count += 1
            if es_count >= es_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {es_patience} epochs)")
                break

    # ── Threshold tuning ──────────────────────────────────────────────
    print("\nTuning decision threshold on validation set...")
    ckpt = torch.load(str(ckpt_dir / "best_model.pt"))
    model.load_state_dict(ckpt["model_state"])

    print(f"  {'Threshold':>10}  {'P_d':>7}  {'P_fa':>7}  {'F1':>7}")
    print("  " + "─" * 36)
    best_thresh = 0.5
    best_vf1    = -1.0
    for thresh in np.arange(0.30, 0.81, 0.05):
        vm = evaluate(model, val_loader, criterion, threshold=float(thresh))
        marker = "  *" if vm["P_fa"] <= 0.05 else ""
        print(f"  {thresh:>10.2f}  {vm['P_d']:>7.4f}  "
              f"{vm['P_fa']:>7.4f}  {vm['F1']:>7.4f}{marker}")
        if vm["P_fa"] <= 0.05 and vm["F1"] > best_vf1:
            best_vf1    = vm["F1"]
            best_thresh = float(thresh)

    print(f"\n  Best threshold: {best_thresh:.2f}  "
          f"(val F1={best_vf1:.4f}, P_fa≤0.05)")

    # ── Final test evaluation ─────────────────────────────────────────
    test_m = evaluate(model, test_loader, criterion,
                      threshold=best_thresh)

    print(f"\n{'='*60}")
    print(f"Phase 4 Test Results  (epoch {ckpt['epoch']}, "
          f"threshold={best_thresh:.2f})")
    print(f"{'='*60}")
    print(f"  Occupancy")
    print(f"    P_d       : {test_m['P_d']:.4f}  "
          f"({test_m['TP']} TP / {test_m['TP']+test_m['FN']} occupied)")
    print(f"    P_fa      : {test_m['P_fa']:.4f}  "
          f"({test_m['FP']} FP / {test_m['FP']+test_m['TN']} empty)")
    print(f"    F1        : {test_m['F1']:.4f}")
    print(f"  Power class accuracy  : {test_m['pc_acc']:.4f}")
    print(f"  Impairment macro-F1   : {test_m['imp_f1']:.4f}")
    print()
    print("  P_d vs SNR:")
    for k, v in sorted(test_m.items()):
        if k.startswith("P_d_snr"):
            parts = k.split("_")
            lo, hi = parts[-2], parts[-1]
            print(f"    SNR [{lo:>4}, {hi:>3}) dB : {v:.4f}")
    print()
    print(f"  Comparison:")
    print(f"    FFT Channeliser : P_d=0.562  P_fa=0.028  F1=0.620")
    print(f"    Phase 2 MLP     : P_d=0.530  P_fa=0.033  F1=0.583")
    print(f"    Phase 4 (this)  : "
          f"P_d={test_m['P_d']:.3f}  "
          f"P_fa={test_m['P_fa']:.3f}  "
          f"F1={test_m['F1']:.3f}")
    delta = test_m["F1"] - 0.583
    sign  = "+" if delta >= 0 else ""
    print(f"    Delta vs Ph2    : {sign}{delta:.3f}")


if __name__ == "__main__":
    main()