"""Train the per-slot GSM occupancy MLP."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import DatasetSplitConfig, GSMDataset
from feature_extraction import FEATURE_DIM, N_SLOTS, FeatureExtractionConfig
from model import GSMOccupancyMLP, log_model_size

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    dataset_dir     : Path  = Path("dataset")
    sample_rate_hz  : float = 5_000_000.0
    fft_size        : int   = 1024
    batch_size      : int   = 64
    epochs          : int   = 100
    learning_rate   : float = 1e-3
    patience        : int   = 10
    pos_weight      : float = 4.0      # handles 90% empty / 10% occupied imbalance
    threshold       : float = 0.5      # decision threshold after sigmoid
    random_seed     : int   = 42
    checkpoint_path : Path  = Path("best_model.pth")
    num_workers     : int   = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


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


def run_epoch(
    model    : nn.Module,
    loader   : DataLoader,
    criterion: nn.Module,
    device   : torch.device,
    optimiser: torch.optim.Optimizer | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Run one train or validation epoch. Returns loss and detection metrics."""

    is_train = optimiser is not None
    model.train(is_train)

    total_loss   = 0.0
    all_logits   = []
    all_occupancy= []

    for features, occupancy in loader:
        features  = features.to(device)
        occupancy = occupancy.to(device)

        if is_train:
            optimiser.zero_grad(set_to_none=True)

        logits = model(features)
        loss   = criterion(logits, occupancy)

        if is_train:
            loss.backward()
            optimiser.step()

        total_loss    += float(loss.item()) * features.size(0)
        all_logits.append(logits.detach().cpu())
        all_occupancy.append(occupancy.cpu())

    all_logits    = torch.cat(all_logits)
    all_occupancy = torch.cat(all_occupancy)
    metrics       = compute_metrics(all_logits, all_occupancy, threshold)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)

    return metrics


def train(config: TrainConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    feature_config = FeatureExtractionConfig(
        sample_rate_hz=config.sample_rate_hz,
        fft_size=config.fft_size,
        normalize=True,
    )
    split_config = DatasetSplitConfig(random_seed=config.random_seed)

    train_ds = GSMDataset(config.dataset_dir, "train",
                          split_config, feature_config)
    val_ds   = GSMDataset(config.dataset_dir, "validation",
                          split_config, feature_config)
    test_ds  = GSMDataset(config.dataset_dir, "test",
                          split_config, feature_config)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True,  num_workers=config.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size,
                              shuffle=False, num_workers=config.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size,
                              shuffle=False, num_workers=config.num_workers)

    model = GSMOccupancyMLP(input_dim=FEATURE_DIM, n_slots=N_SLOTS).to(device)
    log_model_size(model)

    # BCEWithLogitsLoss applies sigmoid internally and handles class imbalance
    # pos_weight=4.0 makes false negatives 4× more costly than false positives
    # matching the ~90% empty / ~10% occupied class distribution
    pw        = torch.tensor([config.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", patience=5, factor=0.5
    )

    best_val_f1            = -1.0
    epochs_no_improvement  = 0

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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim"       : FEATURE_DIM,
                    "n_slots"         : N_SLOTS,
                    "val_F1"          : best_val_f1,
                    "epoch"           : epoch,
                    "feature_config"  : feature_config.__dict__,
                },
                config.checkpoint_path,
            )
            LOGGER.info("Saved checkpoint (val F1=%.4f) to %s",
                        best_val_f1, config.checkpoint_path)
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
    ckpt = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = run_epoch(model, test_loader, criterion, device,
                       threshold=config.threshold)

    print(f"\n{'='*50}")
    print(f"Test results  (best epoch={ckpt['epoch']}, "
          f"threshold={config.threshold})")
    print(f"{'='*50}")
    print(f"  P_d  : {test_m['P_d']:.4f}")
    print(f"  P_fa : {test_m['P_fa']:.4f}")
    print(f"  F1   : {test_m['F1']:.4f}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir",     type=Path,  default=Path("dataset"))
    parser.add_argument("--sample-rate-hz",  type=float, default=5_000_000.0)
    parser.add_argument("--fft-size",        type=int,   default=1024)
    parser.add_argument("--batch-size",      type=int,   default=64)
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--learning-rate",   type=float, default=1e-3)
    parser.add_argument("--patience",        type=int,   default=10)
    parser.add_argument("--pos-weight",      type=float, default=4.0)
    parser.add_argument("--threshold",       type=float, default=0.5)
    parser.add_argument("--random-seed",     type=int,   default=42)
    parser.add_argument("--checkpoint-path", type=Path,  default=Path("best_model.pth"))
    parser.add_argument("--num-workers",     type=int,   default=0)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
