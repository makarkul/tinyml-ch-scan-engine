"""Iso-FAR (matched false-alarm-rate) comparison: CFAR vs Attn vs TCN.

Fixes the structural flaw flagged in review: each model was being compared
at its OWN self-chosen operating point (ML models: max-F1 threshold on val;
CFAR: threshold hit to a target P_fa). That is not a controlled comparison —
R5 requires identical test conditions, not each model doing its best at
whatever point it likes.

This script:
  1. Sweeps threshold (ML) / alpha (CFAR) on the VALIDATION set to find the
     operating point that best hits a fixed TARGET_FAR (e.g. 5%), same
     selection rule for all three models.
  2. Applies that single matched threshold/alpha to the TEST set.
  3. Reports P_d / F1 at that matched point — overall AND per SNR bin.
  4. Also reconciles bin counts explicitly (idle/NaN samples get their own
     bucket instead of being silently dropped — fixes review item #1).

Usage:
    python iso_far_comparison.py \
        --dataset-dir ./gsm_dataset_50k \
        --attn-checkpoint best_attn_v2.pth \
        --tcn-checkpoint best_tcn.pth \
        --target-far 0.05
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset_new import DatasetSplitConfig, GSMDataset
from feature_extraction import FeatureExtractionConfig
from train_new import compute_metrics, _build_model, TrainConfig

from fft_cfar_baseline import (
    CFARConfig, _per_slot_energy, _batch_cfar, _get_split_records,
    _compute_metrics as _cfar_compute_metrics,
)
from dataset_new import load_iq_samples

SNR_EDGES = [-100, 0, 10, 20, 30, 100]
SNR_BINS  = ["[-10,0)", "[0,10)", "[10,20)", "[20,30)", "[30,41)"]

CFO_EDGES = [-100_000, -10_000, -3_000, 3_000, 10_000, 100_000]
CFO_BINS  = ["<-10kHz", "[-10,-3)kHz", "[-3,3)kHz", "[3,10)kHz", ">10kHz"]


# ─────────────────────────────────────────────────────────────────────
#  ML MODEL: collect logits/labels/metadata for a split
# ─────────────────────────────────────────────────────────────────────

def _collect_ml(checkpoint_path: Path, dataset_dir: Path, split: str,
                 random_seed: int, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = TrainConfig(model=ckpt["model_choice"], dropout=0.0)
    model = _build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds_mode = "iq" if ckpt["model_choice"] == "tcn" else "features"
    split_config = DatasetSplitConfig(random_seed=random_seed)
    feature_config = FeatureExtractionConfig(normalize=False)
    ds = GSMDataset(dataset_dir, split, split_config, feature_config, mode=ds_mode)

    all_logits, all_occ, all_snr, all_cfo, all_wb = [], [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            inputs, occupancy = ds[i]
            record = ds.records[i]
            logits = model(inputs.unsqueeze(0).to(device)).cpu()
            all_logits.append(logits.squeeze(0))
            all_occ.append(occupancy)
            all_snr.append(record.snr_db if record.snr_db is not None else float("nan"))
            all_cfo.append(record.cfo_hz if record.cfo_hz is not None else float("nan"))
            all_wb.append(record.wb_blocker if record.wb_blocker is not None else 0)

    return (ckpt["model_choice"], torch.stack(all_logits), torch.stack(all_occ),
            np.array(all_snr), np.array(all_cfo), np.array(all_wb))


def _sweep_ml_threshold(logits, occ, target_far: float) -> float:
    """Find threshold on (val) logits/labels closest to target_far, same
    selection rule as CFAR's tune_threshold: max F1 among thresholds with
    P_fa <= target_far; fall back to closest P_fa if none qualify."""
    grid = np.arange(0.02, 0.98, 0.01)
    best_t, best_f1 = 0.5, -1.0
    closest_t, closest_gap = 0.5, float("inf")

    for t in grid:
        m = compute_metrics(logits, occ, threshold=float(t))
        gap = abs(m["P_fa"] - target_far)
        if gap < closest_gap:
            closest_gap, closest_t = gap, float(t)
        if m["P_fa"] <= target_far and m["F1"] > best_f1:
            best_f1, best_t = m["F1"], float(t)

    return best_t if best_f1 >= 0 else closest_t


# ─────────────────────────────────────────────────────────────────────
#  CFAR: collect energies for a split
# ─────────────────────────────────────────────────────────────────────

def _collect_cfar(dataset_dir: Path, split: str, random_seed: int):
    records = _get_split_records(dataset_dir, split, random_seed, max_samples=None)
    energies, labels, snrs, cfos, wbs = [], [], [], [], []
    for record in records:
        iq  = load_iq_samples(record.iq_path)
        eng = _per_slot_energy(iq)
        energies.append(eng)
        labels.append(record.occupancy)
        snrs.append(record.snr_db if record.snr_db is not None else float("nan"))
        cfos.append(record.cfo_hz if record.cfo_hz is not None else float("nan"))
        wbs.append(record.wb_blocker if record.wb_blocker is not None else 0)
    return (np.stack(energies), np.stack(labels), np.array(snrs), np.array(cfos), np.array(wbs))


def _sweep_cfar_alpha(energies, labels, target_far: float,
                       guard_cells=1, reference_cells=4) -> float:
    alpha_grid = np.linspace(0.3, 8.0, 100)
    best_alpha, best_f1 = alpha_grid[0], -1.0
    closest_alpha, closest_gap = alpha_grid[0], float("inf")

    for alpha in alpha_grid:
        cfg = CFARConfig(alpha=float(alpha), guard_cells=guard_cells,
                         reference_cells=reference_cells)
        preds = _batch_cfar(energies, cfg)
        pd, pfa, f1 = _cfar_compute_metrics(preds, labels)
        gap = abs(pfa - target_far)
        if gap < closest_gap:
            closest_gap, closest_alpha = gap, float(alpha)
        if pfa <= target_far and f1 > best_f1:
            best_f1, best_alpha = f1, float(alpha)

    return best_alpha if best_f1 >= 0 else closest_alpha


# ─────────────────────────────────────────────────────────────────────
#  Reconciled per-SNR-bin metrics (explicit N/A bucket, fixes item #1)
# ─────────────────────────────────────────────────────────────────────

def _breakdown_flag_ml(logits, occ, flags, threshold, labels=("no blocker", "blocker present")):
    rows = []
    accounted = 0
    for val, label in zip((0, 1), labels):
        mask = flags == val
        n = int(mask.sum())
        accounted += n
        if n == 0:
            rows.append((label, 0, None))
            continue
        m = compute_metrics(logits[mask], occ[mask], threshold=threshold)
        rows.append((label, n, m))
    rows.append(("TOTAL (check)", accounted, None))
    return rows, len(flags)


def _breakdown_flag_cfar(energies, labels_arr, flags, alpha, guard_cells=1, reference_cells=4,
                          labels=("no blocker", "blocker present")):
    cfg = CFARConfig(alpha=alpha, guard_cells=guard_cells, reference_cells=reference_cells)
    preds_all = _batch_cfar(energies, cfg)
    rows = []
    accounted = 0
    for val, label in zip((0, 1), labels):
        mask = flags == val
        n = int(mask.sum())
        accounted += n
        if n == 0:
            rows.append((label, 0, None))
            continue
        pd, pfa, f1 = _cfar_compute_metrics(preds_all[mask], labels_arr[mask])
        rows.append((label, n, {"P_d": pd, "P_fa": pfa, "F1": f1}))
    rows.append(("TOTAL (check)", accounted, None))
    return rows, len(flags)


def _breakdown_ml(logits, occ, values, threshold, edges, labels):
    rows = []
    accounted = 0
    for lo, hi, label in zip(edges[:-1], edges[1:], labels):
        mask = (values >= lo) & (values < hi)
        n = int(mask.sum())
        accounted += n
        if n == 0:
            rows.append((label, 0, None))
            continue
        m = compute_metrics(logits[mask], occ[mask], threshold=threshold)
        rows.append((label, n, m))
    na_mask = np.isnan(values)
    n_na = int(na_mask.sum())
    accounted += n_na
    if n_na > 0:
        m = compute_metrics(logits[na_mask], occ[na_mask], threshold=threshold)
        rows.append(("N/A", n_na, m))
    total = len(values)
    rows.append(("TOTAL (check)", accounted, None))
    return rows, total


def _breakdown_cfar(energies, labels_arr, values, alpha, edges, bin_labels,
                     guard_cells=1, reference_cells=4):
    cfg = CFARConfig(alpha=alpha, guard_cells=guard_cells, reference_cells=reference_cells)
    preds_all = _batch_cfar(energies, cfg)
    rows = []
    accounted = 0
    for lo, hi, label in zip(edges[:-1], edges[1:], bin_labels):
        mask = (values >= lo) & (values < hi)
        n = int(mask.sum())
        accounted += n
        if n == 0:
            rows.append((label, 0, None))
            continue
        pd, pfa, f1 = _cfar_compute_metrics(preds_all[mask], labels_arr[mask])
        rows.append((label, n, {"P_d": pd, "P_fa": pfa, "F1": f1}))
    na_mask = np.isnan(values)
    n_na = int(na_mask.sum())
    accounted += n_na
    if n_na > 0:
        pd, pfa, f1 = _cfar_compute_metrics(preds_all[na_mask], labels_arr[na_mask])
        rows.append(("N/A", n_na, {"P_d": pd, "P_fa": pfa, "F1": f1}))
    total = len(values)
    rows.append(("TOTAL (check)", accounted, None))
    return rows, total


def _print_rows(name, rows, total):
    print(f"\n  {name}")
    print(f"  {'Bin':<20} {'n':>6}  {'P_d':>7}  {'P_fa':>7}  {'F1':>7}")
    print("  " + "─" * 52)
    for label, n, m in rows:
        if m is None:
            marker = " ← should equal test n" if "TOTAL" in label else ""
            print(f"  {label:<20} {n:>6}  {'--':>7}  {'--':>7}  {'--':>7}{marker}")
        else:
            print(f"  {label:<20} {n:>6}  {m['P_d']:>7.4f}  {m['P_fa']:>7.4f}  {m['F1']:>7.4f}")
    print(f"  (total test n = {total})")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--attn-checkpoint", type=Path, required=True)
    ap.add_argument("--tcn-checkpoint", type=Path, required=True)
    ap.add_argument("--target-far", type=float, default=0.05)
    ap.add_argument("--random-seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*70}\nISO-FAR COMPARISON — target P_fa = {args.target_far:.3f}\n{'='*70}")
    print("Selection rule (same for all 3 models): on VALIDATION, pick the\n"
          "threshold/alpha with max F1 among those achieving P_fa <= target;\n"
          "if none qualify, fall back to the closest P_fa to target.\n")

    overall_summary = []

    # ── Attn ──────────────────────────────────────────────────────────
    name, val_logits, val_occ, val_snr, val_cfo, val_wb = _collect_ml(
        args.attn_checkpoint, args.dataset_dir, "validation", args.random_seed, device)
    attn_thresh = _sweep_ml_threshold(val_logits, val_occ, args.target_far)
    _, test_logits, test_occ, test_snr, test_cfo, test_wb = _collect_ml(
        args.attn_checkpoint, args.dataset_dir, "test", args.random_seed, device)
    attn_overall = compute_metrics(test_logits, test_occ, threshold=attn_thresh)
    print(f"\n[ATTN] matched threshold = {attn_thresh:.3f}")
    print(f"  Overall (test): P_d={attn_overall['P_d']:.4f}  "
          f"P_fa={attn_overall['P_fa']:.4f}  F1={attn_overall['F1']:.4f}")
    rows, total = _breakdown_ml(test_logits, test_occ, test_snr, attn_thresh, SNR_EDGES, SNR_BINS)
    _print_rows("ATTN per-SNR (iso-FAR)", rows, total)
    rows, total = _breakdown_ml(test_logits, test_occ, test_cfo, attn_thresh, CFO_EDGES, CFO_BINS)
    _print_rows("ATTN per-CFO (iso-FAR)", rows, total)
    rows, total = _breakdown_flag_ml(test_logits, test_occ, test_wb, attn_thresh)
    _print_rows("ATTN per-blocker (iso-FAR)", rows, total)
    overall_summary.append(("Attn", attn_thresh, attn_overall))

    # ── TCN ───────────────────────────────────────────────────────────
    name, val_logits, val_occ, val_snr, val_cfo, val_wb = _collect_ml(
        args.tcn_checkpoint, args.dataset_dir, "validation", args.random_seed, device)
    tcn_thresh = _sweep_ml_threshold(val_logits, val_occ, args.target_far)
    _, test_logits, test_occ, test_snr, test_cfo, test_wb = _collect_ml(
        args.tcn_checkpoint, args.dataset_dir, "test", args.random_seed, device)
    tcn_overall = compute_metrics(test_logits, test_occ, threshold=tcn_thresh)
    print(f"\n[TCN] matched threshold = {tcn_thresh:.3f}")
    print(f"  Overall (test): P_d={tcn_overall['P_d']:.4f}  "
          f"P_fa={tcn_overall['P_fa']:.4f}  F1={tcn_overall['F1']:.4f}")
    rows, total = _breakdown_ml(test_logits, test_occ, test_snr, tcn_thresh, SNR_EDGES, SNR_BINS)
    _print_rows("TCN per-SNR (iso-FAR)", rows, total)
    rows, total = _breakdown_ml(test_logits, test_occ, test_cfo, tcn_thresh, CFO_EDGES, CFO_BINS)
    _print_rows("TCN per-CFO (iso-FAR)", rows, total)
    rows, total = _breakdown_flag_ml(test_logits, test_occ, test_wb, tcn_thresh)
    _print_rows("TCN per-blocker (iso-FAR)", rows, total)
    overall_summary.append(("TCN", tcn_thresh, tcn_overall))

    # ── CFAR ──────────────────────────────────────────────────────────
    val_e, val_l, val_snr, val_cfo, val_wb = _collect_cfar(args.dataset_dir, "validation", args.random_seed)
    cfar_alpha = _sweep_cfar_alpha(val_e, val_l, args.target_far)
    test_e, test_l, test_snr, test_cfo, test_wb = _collect_cfar(args.dataset_dir, "test", args.random_seed)
    cfg = CFARConfig(alpha=cfar_alpha)
    preds = _batch_cfar(test_e, cfg)
    pd, pfa, f1 = _cfar_compute_metrics(preds, test_l)
    cfar_overall = {"P_d": pd, "P_fa": pfa, "F1": f1}
    print(f"\n[CFAR] matched alpha = {cfar_alpha:.3f}")
    print(f"  Overall (test): P_d={pd:.4f}  P_fa={pfa:.4f}  F1={f1:.4f}")
    rows, total = _breakdown_cfar(test_e, test_l, test_snr, cfar_alpha, SNR_EDGES, SNR_BINS)
    _print_rows("CFAR per-SNR (iso-FAR)", rows, total)
    rows, total = _breakdown_cfar(test_e, test_l, test_cfo, cfar_alpha, CFO_EDGES, CFO_BINS)
    _print_rows("CFAR per-CFO (iso-FAR)", rows, total)
    rows, total = _breakdown_flag_cfar(test_e, test_l, test_wb, cfar_alpha)
    _print_rows("CFAR per-blocker (iso-FAR)", rows, total)
    overall_summary.append(("CFAR", cfar_alpha, cfar_overall))

    # ── Final summary table ──────────────────────────────────────────
    print(f"\n{'='*70}\nSUMMARY — all 3 models at matched P_fa ≈ {args.target_far:.3f}\n{'='*70}")
    print(f"  {'Model':<8} {'Thresh/Alpha':>14} {'P_d':>8} {'P_fa':>8} {'F1':>8}")
    print("  " + "─" * 50)
    for name, t, m in overall_summary:
        print(f"  {name:<8} {t:>14.3f} {m['P_d']:>8.4f} {m['P_fa']:>8.4f} {m['F1']:>8.4f}")


if __name__ == "__main__":
    main()