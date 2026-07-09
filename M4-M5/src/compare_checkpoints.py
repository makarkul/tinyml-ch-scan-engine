"""Side-by-side attn vs tcn breakdown — same SNR bins, same blocker split.

Usage:
    python compare_checkpoints.py \
        --checkpoints best_attn.pth best_tcn.pth \
        --dataset-dir ./gsm_dataset_50k
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset_new import DatasetSplitConfig, GSMDataset
from feature_extraction import FeatureExtractionConfig
from train_new import compute_metrics, _build_model, TrainConfig

SNR_BINS = [(-10, 0), (0, 10), (10, 20), (20, 30), (30, 41)]


def evaluate(checkpoint_path: Path, dataset_dir: Path, random_seed: int, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = TrainConfig(model=ckpt["model_choice"], dropout=0.0)
    model = _build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    threshold = float(ckpt.get("threshold", 0.5))
    ds_mode = "iq" if ckpt["model_choice"] == "tcn" else "features"

    split_config = DatasetSplitConfig(random_seed=random_seed)
    feature_config = FeatureExtractionConfig(normalize=False)
    test_ds = GSMDataset(dataset_dir, "test", split_config, feature_config, mode=ds_mode)

    all_logits, all_occ, all_snr, all_wb = [], [], [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            inputs, occupancy = test_ds[i]
            record = test_ds.records[i]
            logits = model(inputs.unsqueeze(0).to(device)).cpu()
            all_logits.append(logits.squeeze(0))
            all_occ.append(occupancy)
            all_snr.append(record.snr_db if record.snr_db is not None else float("nan"))
            all_wb.append(record.wb_blocker if record.wb_blocker is not None else 0)

    all_logits = torch.stack(all_logits)
    all_occ    = torch.stack(all_occ)
    all_snr    = np.array(all_snr)
    all_wb     = np.array(all_wb)

    results = {"model": ckpt["model_choice"], "overall": compute_metrics(all_logits, all_occ, threshold)}
    results["snr_bins"] = {}
    for lo, hi in SNR_BINS:
        mask = (all_snr >= lo) & (all_snr < hi)
        n = int(mask.sum())
        results["snr_bins"][(lo, hi)] = (n, compute_metrics(all_logits[mask], all_occ[mask], threshold) if n else None)
    results["wb"] = {}
    for flag in (0, 1):
        mask = all_wb == flag
        n = int(mask.sum())
        results["wb"][flag] = (n, compute_metrics(all_logits[mask], all_occ[mask], threshold) if n else None)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--random-seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = [evaluate(cp, args.dataset_dir, args.random_seed, device) for cp in args.checkpoints]

    names = [r["model"].upper() for r in all_results]
    col_w = 22

    print(f"\n{'='*60}\nOVERALL\n{'='*60}")
    header = f"{'':<14}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    for key in ("P_d", "P_fa", "F1"):
        row = f"{key:<14}" + "".join(f"{r['overall'][key]:>{col_w}.4f}" for r in all_results)
        print(row)

    print(f"\n{'='*60}\nPER SNR BIN (the target regime is 0-10dB)\n{'='*60}")
    print(header)
    for lo, hi in SNR_BINS:
        n0 = all_results[0]["snr_bins"][(lo, hi)][0]
        print(f"\n[{lo:>4},{hi:>4}) dB  n={n0}")
        for key in ("P_d", "P_fa", "F1"):
            row = f"  {key:<12}" + "".join(
                f"{(r['snr_bins'][(lo, hi)][1][key] if r['snr_bins'][(lo, hi)][1] else float('nan')):>{col_w}.4f}"
                for r in all_results
            )
            print(row)

    print(f"\n{'='*60}\nWIDEBAND BLOCKER\n{'='*60}")
    print(header)
    for flag, label in [(0, "no blocker"), (1, "blocker present")]:
        n0 = all_results[0]["wb"][flag][0]
        print(f"\n{label}  n={n0}")
        for key in ("P_d", "P_fa", "F1"):
            row = f"  {key:<12}" + "".join(
                f"{(r['wb'][flag][1][key] if r['wb'][flag][1] else float('nan')):>{col_w}.4f}"
                for r in all_results
            )
            print(row)


if __name__ == "__main__":
    main()
