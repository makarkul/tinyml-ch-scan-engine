"""
phase3_train.py
===============
Phase 3 — Branch 3 MLP with all four output heads and masked loss.

What is new in Phase 3
----------------------
Three new output heads added to the same trunk from Phase 2:
  - Power class head   : 4-way classification per occupied slot
  - GMSK confidence    : binary per occupied slot
  - Impairment flags   : 6-way multi-label at sample level

The trunk (three FC layers) and occupancy head are unchanged.
The combined loss forces the trunk to build a richer shared
representation, which typically improves the occupancy head as well.

Architecture
------------
  Input(156) → FC(256) → BN → ReLU → Dropout(0.3)
             → FC(128) → BN → ReLU → Dropout(0.3)
             → FC(64)  → BN → ReLU → trunk (64-dim)
                ├→ occupancy_head  : FC(64→25)      BCEWithLogits
                ├→ power_class_head: FC(64→100)→(25,4) CrossEntropy (masked)
                ├→ gmsk_head       : FC(64→25)      BCEWithLogits (masked)
                └→ impairment_head : FC(64→6)       BCEWithLogits

  Total parameters: 92,380

Combined loss
-------------
  L = 1.0×L_occ + 0.5×L_pc + 0.3×L_gmsk + 0.3×L_imp

  Occupancy has weight 1.0 — it is the primary task.
  Power class and GMSK have lower weights — they are auxiliary.
  Impairment flags are sample-level and relatively easy to learn.

Masked loss
-----------
  Power class and GMSK losses are only computed for occupied slots.
  Empty slots have power_class = -1 and no meaningful GMSK target.
  The mask is the occupancy vector: loss[k] = 0 if occupancy[k] == 0.

Power class encoding
--------------------
  0 = weak        SNR < -10 dB
  1 = medium     -10 ≤ SNR < 0 dB
  2 = strong       0 ≤ SNR < 10 dB
  3 = very_strong SNR ≥ 10 dB
  -1 = empty slot  (masked out — no loss contribution)

Usage
-----
  python phase3_train.py --dataset ./gsm_dataset
  python phase3_train.py --dataset ./gsm_dataset --epochs 100 --num-workers 4

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
from phase2   import (
    extract_scalar_features,
    FeatureNormaliser,
    N_SCALAR_FEATURES,
)

N_POWER_CLASSES = 4   # weak / medium / strong / very_strong


# ─────────────────────────────────────────────────────────────────────
#  DATASET — same as Phase 2 but returns power_class and imp_flags too
# ─────────────────────────────────────────────────────────────────────

class Phase3Dataset(torch.utils.data.Dataset):
    """
    Returns all labels needed for the four-head model:
      features    : (156,)  float32  — scalar features
      occupancy   : (25,)   float32  — binary 0/1
      power_class : (25,)   int64    — 0-3 for occupied, -1 for empty
      imp_flags   : (6,)    float32  — multi-label binary
      snr         : ()      float32
    """

    def __init__(self, sample_files: list,
                 normaliser: FeatureNormaliser = None):
        self.files      = [Path(f) for f in sample_files]
        self.normaliser = normaliser

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple:
        d = np.load(str(self.files[idx]), allow_pickle=True).item()

        features = extract_scalar_features(d["iq"])
        if self.normaliser is not None:
            features = self.normaliser.transform(features)

        return (
            torch.from_numpy(features),
            torch.from_numpy(d["occupancy"].astype(np.float32)),
            torch.from_numpy(d["power_class"].astype(np.int64)),
            torch.from_numpy(d["impairment_flags"].astype(np.float32)),
            torch.tensor(float(d["snr_db"]), dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────
#  MODEL — BRANCH 3 MLP WITH ALL FOUR HEADS
# ─────────────────────────────────────────────────────────────────────

class Branch3MLPv2(nn.Module):
    """
    Branch 3 MLP with all four output heads.

    The trunk is identical to Phase 2. Three new heads are added.
    All heads branch from the same 64-dimensional trunk output so
    the trunk must learn a representation useful for all four tasks.

    Forward returns a dict with keys:
      occupancy   : (batch, 25)    raw logits
      power_class : (batch, 25, 4) raw logits per slot per class
      gmsk        : (batch, 25)    raw logits
      impairment  : (batch, 6)     raw logits
    """

    def __init__(self, n_features: int = N_SCALAR_FEATURES,
                 n_slots: int = N_SLOTS,
                 n_imp: int = N_IMP_FLAGS,
                 n_pc: int = N_POWER_CLASSES,
                 dropout: float = 0.3):
        super().__init__()

        # Shared trunk — unchanged from Phase 2
        self.trunk = nn.Sequential(
            nn.Linear(n_features, 256),
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

        # Head 1 — occupancy (unchanged from Phase 2)
        self.occupancy_head = nn.Linear(64, n_slots)

        # Head 2 — power class
        # Outputs n_slots × n_pc logits, reshaped to (batch, 25, 4)
        self.power_class_head = nn.Linear(64, n_slots * n_pc)
        self.n_slots = n_slots
        self.n_pc    = n_pc

        # Head 3 — GMSK confidence per slot
        self.gmsk_head = nn.Linear(64, n_slots)

        # Head 4 — impairment flags (sample-level, not per-slot)
        self.impairment_head = nn.Linear(64, n_imp)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x : (batch, 156) float32

        Returns
        -------
        dict of raw logits (no sigmoid/softmax — losses handle that)
        """
        z = self.trunk(x)   # (batch, 64)

        return {
            "occupancy"  : self.occupancy_head(z),                            # (batch, 25)
            "power_class": self.power_class_head(z).view(
                               -1, self.n_slots, self.n_pc),                  # (batch, 25, 4)
            "gmsk"       : self.gmsk_head(z),                                 # (batch, 25)
            "impairment" : self.impairment_head(z),                           # (batch, 6)
        }


# ─────────────────────────────────────────────────────────────────────
#  MASKED LOSS
# ─────────────────────────────────────────────────────────────────────

class MaskedMultiHeadLoss(nn.Module):
    """
    Combined loss for all four heads with masking for slot-level heads.

    Occupancy loss    — all 25 slots, BCEWithLogitsLoss, pos_weight=4.0
    Power class loss  — occupied slots only, CrossEntropyLoss
    GMSK conf loss    — occupied slots only, BCEWithLogitsLoss
    Impairment loss   — sample level (no mask), BCEWithLogitsLoss

    Combined:
      L = w_occ × L_occ
        + w_pc   × L_pc   (0 if no occupied slots in batch)
        + w_gmsk × L_gmsk (0 if no occupied slots in batch)
        + w_imp  × L_imp

    Why these weights
    -----------------
    Occupancy (1.0) is the primary task — it drives the trunk.
    Power class (0.5) is a meaningful auxiliary task that requires
      understanding signal strength, helping the trunk learn power-
      related features that also improve occupancy detection.
    GMSK and impairment (0.3 each) are lower-weight auxiliaries —
      useful regularisers but not the primary focus.
    """

    def __init__(self,
                 w_occ:  float = 1.0,
                 w_pc:   float = 0.5,
                 w_gmsk: float = 0.3,
                 w_imp:  float = 0.5,
                 pos_weight_occ: float = 4.0):
        super().__init__()

        self.w_occ  = w_occ
        self.w_pc   = w_pc
        self.w_gmsk = w_gmsk
        self.w_imp  = w_imp

        # Occupancy: BCEWithLogitsLoss with class balance weight
        self.occ_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_occ]).expand(N_SLOTS)
        )

        # Power class: CrossEntropyLoss (expects class index targets)
        self.pc_loss = nn.CrossEntropyLoss(reduction="none")

        # GMSK confidence: BCEWithLogitsLoss
        self.gmsk_loss = nn.BCEWithLogitsLoss(reduction="none")

        # Impairment flags: BCEWithLogitsLoss (multi-label)
        self.imp_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.ones(N_IMP_FLAGS) * 4.0
        )

    def forward(self, preds: dict, occupancy: torch.Tensor,
                power_class: torch.Tensor,
                imp_flags: torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        preds       : dict from Branch3MLPv2.forward()
        occupancy   : (batch, 25) float32  — binary ground truth
        power_class : (batch, 25) int64    — 0-3 occupied, -1 empty
        imp_flags   : (batch, 6)  float32  — multi-label binary

        Returns
        -------
        total_loss  : scalar tensor
        loss_dict   : dict of individual loss values for logging
        """
        batch = occupancy.shape[0]

        # ── Occupancy loss ────────────────────────────────────────────
        L_occ = self.occ_loss(preds["occupancy"], occupancy)

        # ── Power class loss (masked) ─────────────────────────────────
        # Mask: only compute loss where occupancy == 1
        # power_class shape: (batch, 25, 4) logits
        # Flatten to (batch*25, 4) for CrossEntropyLoss
        occ_mask = occupancy.bool()   # (batch, 25)
        n_occ    = occ_mask.sum()

        if n_occ > 0:
            # Select logits and targets for occupied slots only
            pc_logits  = preds["power_class"][occ_mask]    # (n_occ, 4)
            pc_targets = power_class[occ_mask]              # (n_occ,) int64
            L_pc = self.pc_loss(pc_logits, pc_targets).mean()
        else:
            L_pc = torch.tensor(0.0, requires_grad=False)

        # ── GMSK confidence loss (masked) ─────────────────────────────
        # Target: 1.0 for occupied slots (all carriers in dataset are GMSK)
        #         masked out for empty slots
        if n_occ > 0:
            gmsk_logits  = preds["gmsk"][occ_mask]          # (n_occ,)
            gmsk_targets = occupancy[occ_mask]               # (n_occ,) all 1.0
            L_gmsk = self.gmsk_loss(gmsk_logits, gmsk_targets).mean()
        else:
            L_gmsk = torch.tensor(0.0, requires_grad=False)

        # ── Impairment loss (no mask — sample level) ──────────────────
        L_imp = self.imp_loss(preds["impairment"], imp_flags)

        # ── Combined loss ─────────────────────────────────────────────
        L_total = (self.w_occ  * L_occ
                 + self.w_pc   * L_pc
                 + self.w_gmsk * L_gmsk
                 + self.w_imp  * L_imp)

        return L_total, {
            "loss_occ" : L_occ.item(),
            "loss_pc"  : L_pc.item()  if n_occ > 0 else 0.0,
            "loss_gmsk": L_gmsk.item() if n_occ > 0 else 0.0,
            "loss_imp" : L_imp.item(),
            "n_occ"    : int(n_occ),
        }


# ─────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────

def compute_occupancy_metrics(logits: torch.Tensor,
                              targets: torch.Tensor,
                              threshold: float = 0.55) -> dict:
    """P_d, P_fa, F1 for occupancy head."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    TP = int(((preds==1)&(targets==1)).sum())
    FP = int(((preds==1)&(targets==0)).sum())
    TN = int(((preds==0)&(targets==0)).sum())
    FN = int(((preds==0)&(targets==1)).sum())
    P_d  = TP / max(TP+FN, 1)
    P_fa = FP / max(FP+TN, 1)
    prec = TP / max(TP+FP, 1)
    F1   = 2*prec*P_d / max(prec+P_d, 1e-10)
    return dict(P_d=P_d, P_fa=P_fa, F1=F1, TP=TP, FP=FP, TN=TN, FN=FN)


def compute_power_class_accuracy(logits: torch.Tensor,
                                  targets: torch.Tensor,
                                  occ_mask: torch.Tensor) -> float:
    """Accuracy of power class prediction on occupied slots only."""
    n_occ = occ_mask.sum()
    if n_occ == 0:
        return 0.0
    preds   = logits[occ_mask].argmax(dim=-1)
    correct = (preds == targets[occ_mask]).sum()
    return float(correct) / float(n_occ)


def compute_impairment_f1(logits: torch.Tensor,
                           targets: torch.Tensor) -> float:
    """Macro-averaged F1 across all 6 impairment flags."""
    preds = (torch.sigmoid(logits) >= 0.5).float()
    f1s   = []
    for i in range(N_IMP_FLAGS):
        tp = float(((preds[:,i]==1)&(targets[:,i]==1)).sum())
        fp = float(((preds[:,i]==1)&(targets[:,i]==0)).sum())
        fn = float(((preds[:,i]==0)&(targets[:,i]==1)).sum())
        p  = tp / max(tp+fp, 1)
        r  = tp / max(tp+fn, 1)
        f1s.append(2*p*r / max(p+r, 1e-10))
    return float(np.mean(f1s))


# ─────────────────────────────────────────────────────────────────────
#  TRAINING AND EVALUATION LOOPS
# ─────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimiser) -> dict:
    model.train()
    total_loss = 0.0
    all_occ_logits = []; all_occ_targets = []
    all_pc_logits  = []; all_pc_targets  = []
    all_imp_logits = []; all_imp_targets = []

    for features, occupancy, power_class, imp_flags, _ in loader:
        optimiser.zero_grad()
        preds = model(features)
        loss, _ = criterion(preds, occupancy, power_class, imp_flags)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(features)
        all_occ_logits.append(preds["occupancy"].detach())
        all_occ_targets.append(occupancy)
        all_pc_logits .append(preds["power_class"].detach())
        all_pc_targets .append(power_class)
        all_imp_logits.append(preds["impairment"].detach())
        all_imp_targets.append(imp_flags)

    occ_l = torch.cat(all_occ_logits);  occ_t = torch.cat(all_occ_targets)
    pc_l  = torch.cat(all_pc_logits);   pc_t  = torch.cat(all_pc_targets)
    imp_l = torch.cat(all_imp_logits);  imp_t = torch.cat(all_imp_targets)

    occ_m = compute_occupancy_metrics(occ_l, occ_t)
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

    for features, occupancy, power_class, imp_flags, snr in loader:
        preds = model(features)
        loss, _ = criterion(preds, occupancy, power_class, imp_flags)
        total_loss += loss.item() * len(features)
        all_occ_l.append(preds["occupancy"]); all_occ_t.append(occupancy)
        all_pc_l .append(preds["power_class"]); all_pc_t.append(power_class)
        all_imp_l.append(preds["impairment"]); all_imp_t.append(imp_flags)
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

    # P_d vs SNR
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
        description="Phase 3 — Branch 3 MLP with all four heads"
    )
    ap.add_argument("--dataset",     type=str,   required=True)
    ap.add_argument("--n-samples",   type=int,   default=None)
    ap.add_argument("--epochs",      type=int,   default=100)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--batch-size",  type=int,   default=32)
    ap.add_argument("--num-workers", type=int,   default=2)
    ap.add_argument("--dropout",     type=float, default=0.3)
    ap.add_argument("--checkpoint",  type=str,   default="./checkpoints_p3")
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
    train_ds = Phase3Dataset(train_base.files, normaliser)
    val_ds   = Phase3Dataset(val_base.files,   normaliser)
    test_ds  = Phase3Dataset(test_base.files,  normaliser)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    # ── Model ─────────────────────────────────────────────────────────
    model    = Branch3MLPv2(dropout=args.dropout)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: Branch3MLPv2")
    print(f"  Parameters: {n_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────
    criterion = MaskedMultiHeadLoss()

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
                "val_pc_acc" : val_m["pc_acc"],
                "val_imp_f1" : val_m["imp_f1"],
            }, str(ckpt_dir / "best_model.pt"))
            es_count = 0
        else:
            es_count += 1
            if es_count >= es_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {es_patience} epochs)")
                break

    # ── Threshold tuning on val set ───────────────────────────────────
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

    print(f"\n{'='*58}")
    print(f"Phase 3 Test Results  (epoch {ckpt['epoch']}, "
          f"threshold={best_thresh:.2f})")
    print(f"{'='*58}")
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
    print(f"  Baseline comparison:")
    print(f"    FFT Channeliser      : P_d=0.562  P_fa=0.028  F1=0.620")
    print(f"    Phase 2 MLP          : P_d=0.530  P_fa=0.033  F1=0.583")
    print(f"    Phase 3 MLP (this)   : "
          f"P_d={test_m['P_d']:.3f}  "
          f"P_fa={test_m['P_fa']:.3f}  "
          f"F1={test_m['F1']:.3f}")
    delta = test_m['F1'] - 0.583
    sign  = "+" if delta >= 0 else ""
    print(f"    Delta vs Phase 2     : {sign}{delta:.3f}")


if __name__ == "__main__":
    main()