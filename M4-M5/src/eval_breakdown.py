"""Per-SNR-bin and per-blocker-flag diagnostic breakdown.

Run this instead of guessing from aggregate F1/P_d/P_fa. Answers:
  - Is the model actually failing at low SNR (as suspected), or uniformly weak?
  - Does the wideband blocker (imp_wb_blocker=1) tank performance specifically?

Usage:
    python eval_breakdown.py --checkpoint best_attn.pth --dataset-dir ./gsm_dataset_50k
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset_new import DatasetSplitConfig, GSMDataset
from feature_extraction import FeatureExtractionConfig
from train_new import compute_metrics, _build_model, TrainConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--random-seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config = TrainConfig(model=ckpt["model_choice"], dropout=0.0)
    model = _build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    threshold = float(ckpt.get("threshold", 0.5))
    ds_mode = "iq" if ckpt["model_choice"] == "tcn" else "features"

    split_config = DatasetSplitConfig(random_seed=args.random_seed)
    feature_config = FeatureExtractionConfig(normalize=False)
    test_ds = GSMDataset(args.dataset_dir, "test", split_config, feature_config, mode=ds_mode)

    all_logits, all_occ, all_snr, all_wb, all_n_active, all_adj = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            inputs, occupancy = test_ds[i]
            record = test_ds.records[i]
            logits = model(inputs.unsqueeze(0).to(device)).cpu()
            all_logits.append(logits.squeeze(0))
            all_occ.append(occupancy)
            all_snr.append(record.snr_db if record.snr_db is not None else float("nan"))
            all_wb.append(record.wb_blocker if record.wb_blocker is not None else 0)
            all_n_active.append(record.n_active_slots if record.n_active_slots is not None else 0)
            all_adj.append(record.adj_ch if record.adj_ch is not None else 0)

    all_logits   = torch.stack(all_logits)
    all_occ      = torch.stack(all_occ)
    all_snr      = np.array(all_snr)
    all_wb       = np.array(all_wb)
    all_n_active = np.array(all_n_active)
    all_adj      = np.array(all_adj)

    print(f"\n{'='*60}\nOVERALL (n={len(test_ds)})\n{'='*60}")
    m = compute_metrics(all_logits, all_occ, threshold)
    print(f"  P_d={m['P_d']:.4f}  P_fa={m['P_fa']:.4f}  F1={m['F1']:.4f}")

    print(f"\n{'='*60}\nPER SNR BIN\n{'='*60}")
    bins = [(-10, 0), (0, 10), (10, 20), (20, 30), (30, 41)]
    for lo, hi in bins:
        mask = (all_snr >= lo) & (all_snr < hi)
        n = int(mask.sum())
        if n == 0:
            print(f"  [{lo:>4},{hi:>4}) dB   n=0")
            continue
        m = compute_metrics(all_logits[mask], all_occ[mask], threshold)
        print(f"  [{lo:>4},{hi:>4}) dB   n={n:>5}   "
              f"P_d={m['P_d']:.4f}  P_fa={m['P_fa']:.4f}  F1={m['F1']:.4f}")

    print(f"\n{'='*60}\nWIDEBAND BLOCKER PRESENT vs ABSENT\n{'='*60}")
    for flag, label in [(0, "no wb blocker"), (1, "wb blocker present")]:
        mask = all_wb == flag
        n = int(mask.sum())
        if n == 0:
            print(f"  {label:<20} n=0")
            continue
        m = compute_metrics(all_logits[mask], all_occ[mask], threshold)
        print(f"  {label:<20} n={n:>5}   "
              f"P_d={m['P_d']:.4f}  P_fa={m['P_fa']:.4f}  F1={m['F1']:.4f}")

    print(f"\n{'='*60}\nADJACENT-CHANNEL FLAG\n{'='*60}")
    for flag, label in [(0, "no adj-channel"), (1, "adj-channel present")]:
        mask = all_adj == flag
        n = int(mask.sum())
        if n == 0:
            print(f"  {label:<20} n=0")
            continue
        m = compute_metrics(all_logits[mask], all_occ[mask], threshold)
        print(f"  {label:<20} n={n:>5}   "
              f"P_d={m['P_d']:.4f}  P_fa={m['P_fa']:.4f}  F1={m['F1']:.4f}")

    print(f"\n{'='*60}\nCARRIER COUNT (multicarrier)\n{'='*60}")
    for count in sorted(set(all_n_active.tolist())):
        mask = all_n_active == count
        n = int(mask.sum())
        if n == 0:
            continue
        m = compute_metrics(all_logits[mask], all_occ[mask], threshold)
        label = f"{count} carrier{'s' if count != 1 else ''}"
        print(f"  {label:<20} n={n:>5}   "
              f"P_d={m['P_d']:.4f}  P_fa={m['P_fa']:.4f}  F1={m['F1']:.4f}")


if __name__ == "__main__":
    main()