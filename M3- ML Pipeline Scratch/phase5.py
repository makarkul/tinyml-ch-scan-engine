"""
phase5_train.py
===============
Phase 5 — Branch 2 (1D CNN on fading vectors) added to Phase 4.

What is new in Phase 5
----------------------
Branch 2: a 1D CNN that processes F12 and F13 — two (1024,) vectors
stacked as a (2, 1024) input. The CNN slides along the frequency axis
with kernel_size=7 to learn frequency-selective fading patterns.

F12 = per-bin temporal variance across the 5 Welch frames.
      High variance in a specific frequency region → fading notch.

F13 = per-bin inter-frame power correlation.
      Low correlation in a specific frequency region → rapid fading.

These two vectors together encode WHERE in the spectrum the channel
is behaving non-stationarily — information the scalar features and
spectrogram cannot capture as directly.

Architecture
------------
  Input A: scalars (156,)       → Branch3 trunk   → global (64,)
  Input B: spectrogram (25,41,5)→ Branch1 CNN      → per-slot (25, 64)
  Input C: fading (2, 1024)     → Branch2 1D CNN   → global (64,)

  Fusion per slot k:
    cat([global_scalar(64), cnn_slot_k(64), fading_global(64)]) → (192,)
    → FC(64) → ReLU → logit

  Output heads same as Phase 4.

  Total parameters: ~137,000

Branch 2 architecture
----------------------
  Conv1d(2, 16, kernel=7, padding=3)   → (batch, 16, 1024)
  BN1d, ReLU
  Conv1d(16, 32, kernel=7, padding=3)  → (batch, 32, 1024)
  BN1d, ReLU
  AdaptiveAvgPool1d(8)                 → (batch, 32, 8)
  Flatten                              → (batch, 256)
  FC(256, 64)                          → (batch, 64)

Usage
-----
  python phase5_train.py --dataset ./gsm_dataset
  python phase5_train.py --dataset ./gsm_dataset --epochs 100 --num-workers 4

Dependencies: numpy, scipy, torch, phase1-4 files
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phase1 import split_dataset, N_SLOTS, N_IMP_FLAGS
from phase2   import (
    extract_scalar_features, FeatureNormaliser, N_SCALAR_FEATURES,
)
from phase3   import (
    MaskedMultiHeadLoss, compute_occupancy_metrics,
    compute_power_class_accuracy, compute_impairment_f1,
)
from phase4   import (
    extract_spectrogram, Branch1CNN,
    TARGET_RATE, SCAN_RATE, N_WINDOW, N_FFT, WELCH_HOP, N_FRAMES, SLOT_BINS,
)
from scipy.signal import resample as fft_resample

N_POWER_CLASSES = 4


# ─────────────────────────────────────────────────────────────────────
#  F12 + F13 FADING FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────

def extract_fading_features(iq_int8: np.ndarray) -> np.ndarray:
    """
    Extract F12 (temporal variance) and F13 (inter-frame correlation).

    Both are (1024,) vectors — one value per FFT bin after fftshift.
    Stacked as (2, 1024) for Branch 2 input.

    F12: per-bin temporal variance across the 5 Welch frames.
         var(power_bin_k over 5 frames)
         High F12 in a bin → power fluctuating rapidly → fading notch.

    F13: per-bin normalised inter-frame power product.
         mean(power[k,t] × power[k,t+1]) / mean(power[k])²
         Low F13 → consecutive frames decorrelated → time-varying fading.
         F13 ≈ 1.0 for stationary signal (clean carrier or stable noise).

    The (2, 1024) output is normalised channel-wise to zero mean and
    unit variance before being fed to the CNN.

    Returns
    -------
    (2, 1024) float32 — [F12, F13] stacked
    """
    x     = (iq_int8[0].astype(np.float32) +
              1j * iq_int8[1].astype(np.float32))
    n_out = int(round(N_WINDOW * SCAN_RATE / TARGET_RATE))
    dec   = fft_resample(x, n_out).astype(np.complex64)

    # Build per-frame power spectra: (N_FFT, n_frames)
    window   = np.hanning(N_FFT)
    n_frames = (len(dec) - N_FFT) // WELCH_HOP + 1
    frames   = np.zeros((N_FFT, n_frames), dtype=np.float32)

    for i in range(n_frames):
        seg           = dec[i * WELCH_HOP : i * WELCH_HOP + N_FFT]
        s             = np.fft.fftshift(np.fft.fft(seg * window, n=N_FFT))
        frames[:, i]  = (np.abs(s) ** 2 / N_FFT).astype(np.float32)

    # F12: per-bin temporal variance
    f12 = np.var(frames, axis=1).astype(np.float32)           # (1024,)

    # F13: per-bin normalised inter-frame power product
    f13 = np.zeros(N_FFT, dtype=np.float32)
    for i in range(n_frames - 1):
        f13 += frames[:, i] * frames[:, i + 1]
    f13 /= max(n_frames - 1, 1)
    mean_pwr    = frames.mean(axis=1)
    mean_pwr_sq = np.where(mean_pwr ** 2 < 1e-20, 1e-20, mean_pwr ** 2)
    f13         = (f13 / mean_pwr_sq).astype(np.float32)
    # Stack and normalise each channel independently
    out = np.stack([f12, f13], axis=0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    for ch in range(2):
        mu  = out[ch].mean()
        std = out[ch].std() + 1e-8
        out[ch] = (out[ch] - mu) / std

    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────

class Phase5Dataset(torch.utils.data.Dataset):
    """
    Returns scalars, spectrogram, fading vectors, and all labels.

    Items:
      scalars     : (156,)       float32
      spectrogram : (25, 41, 5)  float32
      fading      : (2, 1024)    float32
      occupancy   : (25,)        float32
      power_class : (25,)        int64
      imp_flags   : (6,)         float32
      snr         : ()           float32
    """

    def __init__(self, sample_files: list,
                 normaliser: FeatureNormaliser = None):
        self.files      = [Path(f) for f in sample_files]
        self.normaliser = normaliser

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple:
        d       = np.load(str(self.files[idx]), allow_pickle=True).item()
        scalars = extract_scalar_features(d["iq"])
        spec    = extract_spectrogram(d["iq"])
        fading  = extract_fading_features(d["iq"])

        if self.normaliser is not None:
            scalars = self.normaliser.transform(scalars)

        return (
            torch.from_numpy(scalars),
            torch.from_numpy(spec),
            torch.from_numpy(fading),
            torch.from_numpy(d["occupancy"].astype(np.float32)),
            torch.from_numpy(d["power_class"].astype(np.int64)),
            torch.from_numpy(d["impairment_flags"].astype(np.float32)),
            torch.tensor(float(d["snr_db"]), dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────
#  BRANCH 2 — 1D CNN ON FADING VECTORS
# ─────────────────────────────────────────────────────────────────────

class Branch2CNN(nn.Module):
    """
    1D CNN that processes F12 and F13 fading vectors.

    Input:  (batch, 2, 1024) — F12 and F13 as 2 channels
    Output: (batch, 64)      — global fading feature vector

    The CNN slides along the frequency axis with kernel_size=7.
    A window of 7 consecutive bins covers about 34 kHz — enough to
    detect a fading notch (typically 5–20 kHz wide) or a consistent
    high-variance region from a strong carrier.

    AdaptiveAvgPool1d(8) reduces 1024 bins to 8 positions,
    retaining the coarse frequency-domain structure (which octave of
    the band is most affected) without overfitting to specific bins.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(8),   # (batch, 32, 8)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 8, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, fading: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        fading : (batch, 2, 1024) float32

        Returns
        -------
        (batch, 64) float32
        """
        x = self.cnn(fading)          # (batch, 32, 8)
        x = x.view(x.shape[0], -1)   # (batch, 256)
        return self.fc(x)             # (batch, 64)


# ─────────────────────────────────────────────────────────────────────
#  FULL MODEL — ALL THREE BRANCHES FUSED
# ─────────────────────────────────────────────────────────────────────

class FullFusedModel(nn.Module):
    """
    Branch 1 (spectrogram CNN) + Branch 2 (fading 1D CNN)
    + Branch 3 (scalar MLP) with per-slot late fusion.

    Fusion per slot k:
      cat([scalar_global(64), cnn_slot_k(64), fading_global(64)])
      → (192,) → FC(64) → ReLU → heads

    The fading vector contributes the same global 64-dim context to
    every slot's decision (like the scalar Branch 3), because fading
    is a wideband phenomenon — not slot-specific in F12/F13.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        # Branch 3 trunk
        self.branch3_trunk = nn.Sequential(
            nn.Linear(N_SCALAR_FEATURES, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(),
        )

        # Branch 1: per-slot spectrogram CNN
        self.branch1_cnn = Branch1CNN(dropout=dropout)

        # Branch 2: global fading 1D CNN
        self.branch2_cnn = Branch2CNN(dropout=dropout)

        # Per-slot fusion: cat(64+64+64) → 64
        self.fusion = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
        )

        # Output heads
        self.occupancy_head   = nn.Linear(64, 1)
        self.power_class_head = nn.Linear(64, N_POWER_CLASSES)
        self.gmsk_head        = nn.Linear(64, 1)
        self.impairment_head  = nn.Linear(64, N_IMP_FLAGS)

    def forward(self, scalars: torch.Tensor,
                spec: torch.Tensor,
                fading: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        scalars : (batch, 156)
        spec    : (batch, 25, 41, 5)
        fading  : (batch, 2, 1024)
        """
        B = scalars.shape[0]

        # Branch 3: global scalar features
        global_scalar = self.branch3_trunk(scalars)       # (batch, 64)

        # Branch 1: per-slot spectrogram features
        slot_feat     = self.branch1_cnn(spec)            # (batch, 25, 64)

        # Branch 2: global fading features
        global_fading = self.branch2_cnn(fading)          # (batch, 64)

        # Expand globals to per-slot: (batch, 64) → (batch, 25, 64)
        scalar_exp = global_scalar.unsqueeze(1).expand(-1, N_SLOTS, -1)
        fading_exp = global_fading.unsqueeze(1).expand(-1, N_SLOTS, -1)

        # Fuse: (batch, 25, 192) → (batch, 25, 64)
        fused = self.fusion(
            torch.cat([scalar_exp, slot_feat, fading_exp], dim=-1)
        )

        # Heads: flatten to (batch*25, 64)
        fused_flat = fused.view(B * N_SLOTS, 64)

        occ_logits  = self.occupancy_head(fused_flat).view(B, N_SLOTS)
        pc_logits   = self.power_class_head(fused_flat).view(
                          B, N_SLOTS, N_POWER_CLASSES)
        gmsk_logits = self.gmsk_head(fused_flat).view(B, N_SLOTS)
        imp_logits  = self.impairment_head(global_scalar)  # global only

        return {
            "occupancy"  : occ_logits,
            "power_class": pc_logits,
            "gmsk"       : gmsk_logits,
            "impairment" : imp_logits,
        }


# ─────────────────────────────────────────────────────────────────────
#  TRAINING AND EVALUATION
# ─────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimiser) -> dict:
    model.train()
    total_loss = 0.0
    all_occ_l=[]; all_occ_t=[]
    all_pc_l=[];  all_pc_t=[]
    all_imp_l=[]; all_imp_t=[]

    for scalars, spec, fading, occ, pc, imp, _ in loader:
        optimiser.zero_grad()
        preds = model(scalars, spec, fading)
        loss, _ = criterion(preds, occ, pc, imp)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(scalars)
        all_occ_l.append(preds["occupancy"].detach()); all_occ_t.append(occ)
        all_pc_l .append(preds["power_class"].detach()); all_pc_t.append(pc)
        all_imp_l.append(preds["impairment"].detach()); all_imp_t.append(imp)

    occ_l=torch.cat(all_occ_l); occ_t=torch.cat(all_occ_t)
    pc_l =torch.cat(all_pc_l);  pc_t =torch.cat(all_pc_t)
    imp_l=torch.cat(all_imp_l); imp_t=torch.cat(all_imp_t)

    occ_m  = compute_occupancy_metrics(occ_l, occ_t)
    pc_acc = compute_power_class_accuracy(pc_l, pc_t, occ_t.bool())
    imp_f1 = compute_impairment_f1(imp_l, imp_t)

    return {
        "loss"   : total_loss / len(loader.dataset),
        "F1"     : occ_m["F1"], "P_d": occ_m["P_d"], "P_fa": occ_m["P_fa"],
        "pc_acc" : pc_acc, "imp_f1": imp_f1,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, threshold=0.5) -> dict:
    model.eval()
    total_loss=0.0
    all_occ_l=[]; all_occ_t=[]
    all_pc_l=[];  all_pc_t=[]
    all_imp_l=[]; all_imp_t=[]
    all_snr=[]

    for scalars, spec, fading, occ, pc, imp, snr in loader:
        preds = model(scalars, spec, fading)
        loss, _ = criterion(preds, occ, pc, imp)
        total_loss += loss.item() * len(scalars)
        all_occ_l.append(preds["occupancy"]); all_occ_t.append(occ)
        all_pc_l .append(preds["power_class"]); all_pc_t.append(pc)
        all_imp_l.append(preds["impairment"]); all_imp_t.append(imp)
        all_snr.append(snr)

    occ_l=torch.cat(all_occ_l); occ_t=torch.cat(all_occ_t)
    pc_l =torch.cat(all_pc_l);  pc_t =torch.cat(all_pc_t)
    imp_l=torch.cat(all_imp_l); imp_t=torch.cat(all_imp_t)
    snr_t=torch.cat(all_snr)

    occ_m  = compute_occupancy_metrics(occ_l, occ_t, threshold)
    pc_acc = compute_power_class_accuracy(pc_l, pc_t, occ_t.bool())
    imp_f1 = compute_impairment_f1(imp_l, imp_t)

    m = {
        "loss"  : total_loss/len(loader.dataset),
        "F1"    : occ_m["F1"], "P_d": occ_m["P_d"], "P_fa": occ_m["P_fa"],
        "pc_acc": pc_acc, "imp_f1": imp_f1,
        "TP": occ_m["TP"], "FP": occ_m["FP"],
        "TN": occ_m["TN"], "FN": occ_m["FN"],
    }
    for lo, hi in [(-12,-4),(-4,0),(0,6),(6,12),(12,20),(20,28)]:
        mask = (snr_t>=lo)&(snr_t<hi)
        if mask.sum()>0:
            mi = compute_occupancy_metrics(occ_l[mask], occ_t[mask], threshold)
            m[f"P_d_snr_{lo}_{hi}"] = mi["P_d"]
    return m


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Phase 5 — All three branches fused"
    )
    ap.add_argument("--dataset",     type=str,   required=True)
    ap.add_argument("--n-samples",   type=int,   default=None)
    ap.add_argument("--epochs",      type=int,   default=100)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--batch-size",  type=int,   default=32)
    ap.add_argument("--num-workers", type=int,   default=2)
    ap.add_argument("--dropout",     type=float, default=0.3)
    ap.add_argument("--checkpoint",  type=str,   default="./checkpoints_p5")
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataset splits...")
    train_base, val_base, test_base = split_dataset(
        args.dataset, n_samples=args.n_samples, seed=args.seed
    )
    print(f"  Train: {len(train_base)}  Val: {len(val_base)}  "
          f"Test: {len(test_base)}")

    print("\nFitting normaliser on full training set...")
    normaliser = FeatureNormaliser()
    normaliser.fit(train_base, n_samples=None)
    normaliser.save(str(ckpt_dir / "normaliser.npz"))

    train_ds = Phase5Dataset(train_base.files, normaliser)
    val_ds   = Phase5Dataset(val_base.files,   normaliser)
    test_ds  = Phase5Dataset(test_base.files,  normaliser)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    model    = FullFusedModel(dropout=args.dropout)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: FullFusedModel (Branch 1+2+3)")
    print(f"  Parameters: {n_params:,}")

    criterion = MaskedMultiHeadLoss(w_occ=1.0, w_pc=0.5,
                                    w_gmsk=0.3, w_imp=0.5)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", patience=5, factor=0.5
    )

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
                "epoch": epoch, "model_state": model.state_dict(),
                "val_F1": best_val_f1, "val_P_d": val_m["P_d"],
                "val_P_fa": val_m["P_fa"],
            }, str(ckpt_dir / "best_model.pt"))
            es_count = 0
        else:
            es_count += 1
            if es_count >= es_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {es_patience} epochs)")
                break

    # Threshold tuning
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

    test_m = evaluate(model, test_loader, criterion, threshold=best_thresh)

    print(f"\n{'='*60}")
    print(f"Phase 5 Test Results  (epoch {ckpt['epoch']}, "
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
    print(f"    Phase 4 (B1+B3) : P_d=0.617  P_fa=0.010  F1=0.722")
    print(f"    Phase 5 (this)  : "
          f"P_d={test_m['P_d']:.3f}  "
          f"P_fa={test_m['P_fa']:.3f}  "
          f"F1={test_m['F1']:.3f}")
    delta = test_m["F1"] - 0.722
    sign  = "+" if delta >= 0 else ""
    print(f"    Delta vs Ph4    : {sign}{delta:.3f}")


if __name__ == "__main__":
    main()