from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import logging
from pathlib import Path
import random
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset_new import DatasetSplitConfig, GSMDataset, DatasetMode, N_DEC
from feature_extraction import FEATURE_DIM, N_SLOTS, FeatureExtractionConfig

LOGGER = logging.getLogger(__name__)

ModelChoice = Literal["mlp", "tcn", "attn", "sweep"]


# ─────────────────────────────────────────────────────────────────────
#  LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

class FocalLossWithLogits(nn.Module):
    """Multi-label focal loss (Lin et al., "Focal Loss for Dense Object
    Detection"), used by the reference paper's SigDetNet confidence head
    in place of plain cross-entropy for the same class-imbalance reason
    we currently use BCEWithLogitsLoss(pos_weight=...).

    L_FL = -alpha * (1-p)^gamma * log(p)         for positive targets
           -(1-alpha) * p^gamma * log(1-p)       for negative targets

    gamma down-weights easy (already well-classified) examples so the
    loss focuses on hard, misclassified slots -- e.g. weak carriers,
    blocker-corrupted slots, adjacent-channel leakage. alpha plays a
    similar role to pos_weight: it rebalances the ~90% empty / ~10%
    occupied split.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p     = torch.sigmoid(logits)
        p     = torch.clamp(p, min=1e-6, max=1.0 - 1e-6)
        ce_pos = -torch.log(p)
        ce_neg = -torch.log(1.0 - p)

        loss_pos = self.alpha       * (1.0 - p) ** self.gamma * ce_pos
        loss_neg = (1.0 - self.alpha) * p        ** self.gamma * ce_neg

        loss = targets * loss_pos + (1.0 - targets) * loss_neg

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainConfig:
    dataset_dir     : Path        = Path("dataset")
    model           : ModelChoice = "mlp"       # "mlp", "tcn", "attn", or "sweep"
    batch_size      : int         = 64
    epochs          : int         = 100
    learning_rate   : float       = 1e-3
    patience        : int         = 10
    pos_weight      : float       = 1.0         # ~9.93% occupied → (1-p)/p ≈ 9.07
    dropout         : float       = 0.3
    threshold       : float       = 0.5         # decision threshold after sigmoid (fallback)
    random_seed     : int         = 42
    checkpoint_path : Path        = Path("best_model.pth")
    num_workers     : int         = 0
    loss            : Literal["bce", "focal"] = "focal"
    focal_alpha     : float       = 0.75
    focal_gamma     : float       = 2.0
    freeze_tcn_frontend : bool    = False


# ─────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _build_model(config: TrainConfig) -> nn.Module:
    """Instantiate the right model class based on config.model."""
    if config.model == "tcn":
        from model_tcn import GSMOccupancyTCN
        return GSMOccupancyTCN(
            n_slots=N_SLOTS, dropout=config.dropout,
            freeze_frontend=config.freeze_tcn_frontend,
        )
    elif config.model == "attn":
        from model_attn import GSMOccupancySlotAttention
        return GSMOccupancySlotAttention(
            input_dim=FEATURE_DIM, n_slots=N_SLOTS, dropout=config.dropout
        )
    elif config.model == "sweep":
        from sweep import GSMOccupancySweep
        return GSMOccupancySweep(
            input_dim=FEATURE_DIM, n_slots=N_SLOTS, dropout=config.dropout
        )
    else:
        from model import GSMOccupancyMLP
        return GSMOccupancyMLP(
            input_dim=FEATURE_DIM, n_slots=N_SLOTS, dropout=config.dropout
        )


def _dataset_mode(config: TrainConfig) -> DatasetMode:
    """Map model choice to the correct dataset output mode."""
    return "iq" if config.model == "tcn" else "features"


def _checkpoint_meta(
    config        : TrainConfig,
    model         : nn.Module,
    best_val_f1   : float,
    epoch         : int,
    feature_config: FeatureExtractionConfig,
) -> dict:
    """Build the checkpoint dict — includes enough metadata to reload correctly."""
    base = {
        "model_state_dict" : model.state_dict(),
        "model_choice"     : config.model,
        "n_slots"          : N_SLOTS,
        "val_F1"           : best_val_f1,
        "epoch"            : epoch,
        "threshold"        : config.threshold,
    }
    if config.model == "tcn":
        base["n_dec"] = N_DEC          # raw-IQ input length
    else:
        base["input_dim"]      = FEATURE_DIM
        base["feature_config"] = feature_config.__dict__
    return base


# ─────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(
    logits   : torch.Tensor,
    occupancy: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute P_d, P_fa, and F1 from logits and binary labels."""
    preds = (torch.sigmoid(logits) > threshold).float()
    occ   = occupancy

    TP = float(((preds == 1) & (occ == 1)).sum())
    FP = float(((preds == 1) & (occ == 0)).sum())
    TN = float(((preds == 0) & (occ == 0)).sum())
    FN = float(((preds == 0) & (occ == 1)).sum())

    P_d  = TP / max(TP + FN, 1)
    P_fa = FP / max(FP + TN, 1)
    prec = TP / max(TP + FP, 1)
    F1   = 2 * prec * P_d / max(prec + P_d, 1e-10)

    return {"P_d": P_d, "P_fa": P_fa, "F1": F1}


# ─────────────────────────────────────────────────────────────────────
#  EPOCH RUNNER  (shared by all architectures)
# ─────────────────────────────────────────────────────────────────────

def run_epoch(
    model    : nn.Module,
    loader   : DataLoader,
    criterion: nn.Module,
    device   : torch.device,
    optimiser: torch.optim.Optimizer | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Run one train or validation epoch. Returns loss and detection metrics.

    The loop is deliberately model-agnostic: it receives whatever tensor
    the DataLoader yields as `inputs` and forwards it straight into the
    model. No conditional branching needed here.
    """
    is_train = optimiser is not None
    model.train(is_train)

    total_loss    = 0.0
    all_logits    = []
    all_occupancy = []

    for inputs, occupancy in loader:
        inputs    = inputs.to(device)
        occupancy = occupancy.to(device)

        if is_train:
            optimiser.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss   = criterion(logits, occupancy)

        if is_train:
            loss.backward()
            # Gradient clipping: stabilises early TCN training when the
            # front-end conv weights are still far from convergence.
            # Has negligible effect on the MLP but doesn't hurt it.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

        total_loss    += float(loss.item()) * inputs.size(0)
        all_logits.append(logits.detach().cpu())
        all_occupancy.append(occupancy.cpu())

    all_logits    = torch.cat(all_logits)
    all_occupancy = torch.cat(all_occupancy)
    metrics       = compute_metrics(all_logits, all_occupancy, threshold)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


# ─────────────────────────────────────────────────────────────────────
#  MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────

def train(config: TrainConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Model: %s", config.model.upper())

    feature_config = FeatureExtractionConfig(normalize=False)
    split_config   = DatasetSplitConfig(random_seed=config.random_seed)
    ds_mode        = _dataset_mode(config)

    LOGGER.info("Dataset mode: %s", ds_mode)

    # Dataset construction is identical for all models — only `mode` differs
    train_ds = GSMDataset(config.dataset_dir, "train",
                          split_config, feature_config, mode=ds_mode)
    val_ds   = GSMDataset(config.dataset_dir, "validation",
                          split_config, feature_config, mode=ds_mode)
    test_ds  = GSMDataset(config.dataset_dir, "test",
                          split_config, feature_config, mode=ds_mode)

    # Feature-space augmentation (phase/energy/jitter) — only defined for
    # "features" mode, since it operates on the (25, 5) feature grid.
    from dataset_new import AugmentedGSMDataset, make_weighted_sampler
    train_source = (AugmentedGSMDataset(train_ds, augment=True)
                    if ds_mode == "features" else train_ds)

    # Weighted sampler: upsample occupied samples to reduce class imbalance.
    sampler      = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_source, batch_size=config.batch_size,
                              sampler=sampler, num_workers=config.num_workers,
                              pin_memory=device.type == "cuda")

    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size,
                              shuffle=False, num_workers=config.num_workers,
                              pin_memory=device.type == "cuda")
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size,
                              shuffle=False, num_workers=config.num_workers,
                              pin_memory=device.type == "cuda")

    model = _build_model(config).to(device)

    # Print parameter / complexity summary via the model's own helper if available
    if hasattr(model, "log_model_size"):
        model.log_model_size()
    elif config.model in ("attn", "sweep"):
        if config.model == "attn":
            from model_attn import log_model_size
        else:
            from sweep import log_model_size
        log_model_size(model)

    if config.loss == "focal":
        criterion = FocalLossWithLogits(alpha=config.focal_alpha, gamma=config.focal_gamma)
        LOGGER.info("Loss: FocalLoss(alpha=%.3f, gamma=%.2f)",
                config.focal_alpha, config.focal_gamma)
    else:
        if config.pos_weight > 1.0:
            LOGGER.warning(
                "pos_weight=%.2f used together with WeightedRandomSampler — "
            "this double-corrects class imbalance. Consider --pos-weight 1.0 "
            "when sampler is active.",
                config.pos_weight,
            )
        pw        = torch.tensor([config.pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        LOGGER.info("Loss: BCEWithLogitsLoss(pos_weight=%.2f)", config.pos_weight)

    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", patience=5, factor=0.5, min_lr=1e-6,
    )

    best_val_f1           = -1.0
    epochs_no_improvement = 0

    print(f"\nModel: {config.model.upper()}  |  "
          f"Dataset: {config.dataset_dir}  |  "
          f"Device: {device}")
    print(f"\n{'Ep':>4}  {'TrLoss':>8}  {'TrPd':>7}  {'TrPfa':>7}  "
          f"{'TrF1':>7}  {'VaLoss':>8}  {'VaPd':>7}  {'VaPfa':>7}  "
          f"{'VaF1':>7}  {'LR':>8}")
    print("─" * 90)

    for epoch in range(1, config.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device,
                       optimiser, config.threshold)
        va = run_epoch(model, val_loader,   criterion, device,
                       threshold=config.threshold)

        scheduler.step(va["F1"])
        lr = optimiser.param_groups[0]["lr"]

        print(f"{epoch:>4}  {tr['loss']:>8.4f}  {tr['P_d']:>7.4f}  "
              f"{tr['P_fa']:>7.4f}  {tr['F1']:>7.4f}  "
              f"{va['loss']:>8.4f}  {va['P_d']:>7.4f}  "
              f"{va['P_fa']:>7.4f}  {va['F1']:>7.4f}  {lr:>8.2e}")

        if va["F1"] > best_val_f1:
            best_val_f1           = va["F1"]
            epochs_no_improvement = 0
            ckpt = _checkpoint_meta(
                config, model, best_val_f1, epoch, feature_config
            )
            torch.save(ckpt, config.checkpoint_path)
            LOGGER.info("Saved checkpoint (val F1=%.4f) at epoch %d → %s",
                        best_val_f1, epoch, config.checkpoint_path)
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= config.patience:
                LOGGER.info(
                    "Early stopping at epoch %d "
                    "(no F1 improvement for %d epochs)",
                    epoch, config.patience,
                )
                break

    # ── Final test evaluation ─────────────────────────────────────────
    ckpt  = torch.load(config.checkpoint_path, map_location=device,
                       weights_only=False)
    model = _build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Threshold search on VAL set (fix: 0.5 not always best) ────────
    # Search here, not on test — tuning on test would leak test info into
    # the decision boundary. FocalLoss(alpha=0.75) biases the model
    # differently than plain BCE, so 0.5 is just a guess, not a given.
    model.eval()
    val_logits_list, val_occ_list = [], []
    with torch.no_grad():
        for inputs, occupancy in val_loader:
            inputs = inputs.to(device)
            val_logits_list.append(model(inputs).detach().cpu())
            val_occ_list.append(occupancy)
    val_logits    = torch.cat(val_logits_list)
    val_occupancy = torch.cat(val_occ_list)

    best_thresh, best_thresh_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.02):
        f1_t = compute_metrics(val_logits, val_occupancy, threshold=float(t))["F1"]
        if f1_t > best_thresh_f1:
            best_thresh, best_thresh_f1 = float(t), f1_t

    LOGGER.info(
        "Threshold search: best_thresh=%.2f (val F1=%.4f) vs "
        "checkpoint default thresh=%.2f",
        best_thresh, best_thresh_f1, float(ckpt.get("threshold", config.threshold)),
    )

    test_threshold = best_thresh
    test_m = run_epoch(model, test_loader, criterion, device,
                       threshold=test_threshold)

    print(f"\n{'='*50}")
    print(f"Test results  [{config.model.upper()}]  "
          f"best epoch={ckpt['epoch']}  threshold={test_threshold:.2f} (val-tuned)")
    print(f"{'='*50}")
    print(f"  P_d  : {test_m['P_d']:.4f}")
    print(f"  P_fa : {test_m['P_fa']:.4f}")
    print(f"  F1   : {test_m['F1']:.4f}")
    print(f"  Loss : {test_m['loss']:.4f}")


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-dir",     type=Path,  default=Path("dataset"),
                        help="Root directory of the generated dataset")
    parser.add_argument("--model",           type=str,   default="mlp",
                        choices=["mlp", "tcn", "attn", "sweep"],
                        help="Architecture to train: 'mlp', 'tcn', 'attn', or 'sweep'")
    parser.add_argument("--batch-size",      type=int,   default=64)
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--learning-rate",   type=float, default=1e-3)
    parser.add_argument("--patience",        type=int,   default=10)
    parser.add_argument("--pos-weight",      type=float, default=1.0)
    parser.add_argument("--dropout",         type=float, default=0.3)
    parser.add_argument("--threshold",       type=float, default=0.5)
    parser.add_argument("--random-seed",     type=int,   default=42)
    parser.add_argument("--checkpoint-path", type=Path,  default=Path("best_model.pth"),
                        help="Where to save the best checkpoint")
    parser.add_argument("--num-workers",     type=int,   default=0)
    parser.add_argument("--loss",            type=str,   default="focal",
                        choices=["bce", "focal"],
                        help="Loss function: 'bce' (BCEWithLogitsLoss+pos_weight) or 'focal'")
    parser.add_argument("--focal-alpha",     type=float, default=0.75)
    parser.add_argument("--focal-gamma",     type=float, default=2.0)
    parser.add_argument("--freeze-tcn-frontend", action="store_true",
                        help="Freeze TCN's demod mixing weights at the exact matched-filter "
                             "values instead of letting them train.")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())