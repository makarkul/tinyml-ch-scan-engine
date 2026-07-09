from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from dataset_new import GSMDataset, DatasetSplitConfig, load_iq_samples



import numpy as np

from feature_extraction import (
    N_SLOTS,
    SLOT_FREQ_HZ,
    _decimate,
    _matched_filter_bank,
)

LOGGER = logging.getLogger(__name__)

SLOT_BW_HZ = 200_000   # GSM channel spacing


# ─────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CFARConfig:
    """Tunable parameters for the CFAR detector.

    Attributes
    ----------
    alpha : float
        CFAR scaling factor — threshold = alpha × mean(reference cells).
        Tuned on validation set via tune_threshold(). Default 1.5 is a
        starting point; typical values range 1.2–4.0 depending on FAR target.
    guard_cells : int
        Number of slots on each side of the CUT excluded from the
        reference window (to prevent carrier energy leaking into the
        noise estimate). Default 1 = ±1 slot guard.
    reference_cells : int
        Number of slots on each side (beyond guard) used to estimate
        the noise floor. Default 4 → 8 reference cells total.
    target_far : float
        Target false-alarm rate used when calling tune_threshold().
        Default 0.05 = 5%.
    """
    alpha           : float = 1.5
    guard_cells     : int   = 1
    reference_cells : int   = 4
    target_far      : float = 0.05


# ─────────────────────────────────────────────────────────────────────
#  CORE SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────

def _per_slot_energy(iq: np.ndarray) -> np.ndarray:
    """
    Compute matched-filter energy for each of the 25 GSM slots.

    Reuses the same decimation and matched-filter bank as the MLP
    feature extractor so both paths see identical per-slot energies.

    Parameters
    ----------
    iq : (2, 65536) int8  or  (N,) complex64

    Returns
    -------
    (25,) float32 — |corr_k|² for each slot
    """
    signal      = _decimate(iq)                        # (3151,) complex64
    corr        = _matched_filter_bank(signal)         # (25,)   complex64
    return (np.abs(corr) ** 2).astype(np.float32)     # (25,)   float32


# ─────────────────────────────────────────────────────────────────────
#  CFAR DETECTOR
# ─────────────────────────────────────────────────────────────────────

def _cfar_threshold(
    energies      : np.ndarray,   # (25,) float32
    alpha         : float,
    guard_cells   : int,
    reference_cells: int,
) -> np.ndarray:
    """
    Compute cell-averaging CFAR thresholds for all 25 slots.

    For each cell-under-test (CUT) k:
        reference = slots outside the guard window [k-G..k+G]
                    but within [k-G-R..k+G+R]  (clipped at band edges)
        threshold_k = alpha × mean(reference energies)

    Parameters
    ----------
    energies        : (25,) per-slot energies
    alpha           : scaling factor
    guard_cells     : one-sided guard band width in slots
    reference_cells : one-sided reference window width in slots

    Returns
    -------
    (25,) float32 — per-slot CFAR thresholds
    """
    N          = len(energies)
    thresholds = np.zeros(N, dtype=np.float32)

    for k in range(N):
        ref_indices = []
        for offset in range(1, guard_cells + reference_cells + 1):
            if offset <= guard_cells:
                continue   # inside guard band — skip
            for sign in (-1, 1):
                idx = k + sign * offset
                if 0 <= idx < N:
                    ref_indices.append(idx)

        if ref_indices:
            thresholds[k] = alpha * float(np.mean(energies[ref_indices]))
        else:
            # Edge case: no reference cells available (very narrow band)
            thresholds[k] = alpha * float(np.mean(energies))

    return thresholds


def detect(
    iq    : np.ndarray,
    config: CFARConfig,
) -> np.ndarray:
    """
    Run the full CFAR detector on one IQ window.

    Parameters
    ----------
    iq     : (2, 65536) int8  or  (N,) complex64
    config : CFARConfig with tuned alpha

    Returns
    -------
    (25,) uint8 — binary occupancy prediction per slot (1 = occupied)
    """
    energies   = _per_slot_energy(iq)
    thresholds = _cfar_threshold(
        energies, config.alpha, config.guard_cells, config.reference_cells
    )
    return (energies > thresholds).astype(np.uint8)


def detect_with_scores(
    iq    : np.ndarray,
    config: CFARConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run detector and return predictions, raw energies, and thresholds.

    Useful for plotting and threshold sweep analysis.

    Returns
    -------
    predictions : (25,) uint8
    energies    : (25,) float32
    thresholds  : (25,) float32
    """
    energies   = _per_slot_energy(iq)
    thresholds = _cfar_threshold(
        energies, config.alpha, config.guard_cells, config.reference_cells
    )
    predictions = (energies > thresholds).astype(np.uint8)
    return predictions, energies, thresholds


# ─────────────────────────────────────────────────────────────────────
#  THRESHOLD TUNING (on validation set)
# ──────────────────────────────────────────────────────────────
def _get_split_records(
    dataset_dir : Path,
    split       : str,
    random_seed : int = 42,
    max_samples : int | None = None,
):
    """
    Load the exact same split GSMDataset/train_new.py uses — guarantees
    the baseline is evaluated on identical samples to the ML models.
    """
    split_config = DatasetSplitConfig(random_seed=random_seed)
    ds = GSMDataset(dataset_dir, split, split_config, feature_config=None, mode="iq")
    records = ds.records
    if max_samples is not None:
        records = records[:max_samples]
    return records   # each record: .iq_path, .occupancy, .snr_db, .wb_blocker

def _batch_cfar(
    energies: np.ndarray,   # (N, 25)
    config  : CFARConfig,
) -> np.ndarray:             # (N, 25) uint8
    """Apply CFAR to a batch of energy vectors."""
    N    = energies.shape[0]
    preds = np.zeros((N, 25), dtype=np.uint8)
    for i in range(N):
        thresh     = _cfar_threshold(energies[i], config.alpha,
                                     config.guard_cells, config.reference_cells)
        preds[i]   = (energies[i] > thresh).astype(np.uint8)
    return preds


def tune_threshold(
    dataset_dir     : str | Path,
    split           : Literal["validation", "train"] = "validation",
    target_far      : float = 0.05,
    alpha_grid      : np.ndarray | None = None,
    guard_cells     : int = 1,
    reference_cells : int = 4,
    max_samples     : int = 2000,
    random_seed     : int = 42,
) -> CFARConfig:
    """
    Sweep alpha on the validation set and return the config that
    achieves the target false-alarm rate.

    Uses GSMDataset's own split (not a reimplemented one) so the
    baseline is evaluated on exactly the same samples as the ML models.
    """
    if alpha_grid is None:
        alpha_grid = np.linspace(0.5, 6.0, 50)

    records = _get_split_records(dataset_dir, split, random_seed, max_samples)

    LOGGER.info("Tuning CFAR threshold on %d %s samples (target FAR=%.3f)",
                len(records), split, target_far)

    all_energies, all_labels = [], []
    for record in records:
        try:
            iq  = load_iq_samples(record.iq_path)
            eng = _per_slot_energy(iq)
            all_energies.append(eng)
            all_labels.append(record.occupancy)
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", record.iq_path, exc)

    all_energies = np.stack(all_energies)   # (N, 25)
    all_labels   = np.stack(all_labels)     # (N, 25)

    best_alpha = alpha_grid[0]
    best_f1    = -1.0
    results    = []

    for alpha in alpha_grid:
        cfg   = CFARConfig(alpha=float(alpha), guard_cells=guard_cells,
                           reference_cells=reference_cells)
        preds = _batch_cfar(all_energies, cfg)   # (N, 25)

        pd, pfa, f1 = _compute_metrics(preds, all_labels)
        results.append((float(alpha), pd, pfa, f1))

        if pfa <= target_far and f1 > best_f1:
            best_f1    = f1
            best_alpha = float(alpha)

    if best_f1 < 0:
        LOGGER.warning(
            "No alpha achieved target FAR=%.3f — using best F1 instead", target_far
        )
        best_alpha = max(results, key=lambda r: r[3])[0]

    print(f"\n{'Alpha':>7}  {'P_d':>7}  {'P_fa':>7}  {'F1':>7}")
    print("─" * 34)
    for alpha, pd, pfa, f1 in results[::5]:
        marker = " ←" if abs(alpha - best_alpha) < 1e-6 else ""
        print(f"{alpha:>7.3f}  {pd:>7.4f}  {pfa:>7.4f}  {f1:>7.4f}{marker}")
    print(f"\nChosen alpha = {best_alpha:.3f}  (val F1={best_f1:.4f})")

    return CFARConfig(
        alpha=best_alpha,
        guard_cells=guard_cells,
        reference_cells=reference_cells,
        target_far=target_far,
    )


def evaluate(
    dataset_dir  : str | Path,
    config       : CFARConfig,
    split        : Literal["test", "validation", "train"] = "test",
    max_samples  : int | None = None,
    scenario_col : str | None = None,
    random_seed  : int = 42,
) -> dict:
    """
    Evaluate the CFAR detector on a dataset split, using the exact same
    GSMDataset split as the ML models (RISK-2 mitigation).

    scenario_col : "snr_db" or "wb_blocker" — attribute name on SampleRecord,
                   not a raw CSV column anymore. "snr_db" is auto-binned
                   into 5 ranges; anything else is grouped by exact value.
    """
    records = _get_split_records(dataset_dir, split, random_seed, max_samples)

    all_energies, all_labels, all_meta = [], [], []
    for record in records:
        try:
            iq  = load_iq_samples(record.iq_path)
            eng = _per_slot_energy(iq)
            all_energies.append(eng)
            all_labels.append(record.occupancy)
            all_meta.append(record)
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", record.iq_path, exc)

    all_energies = np.stack(all_energies)
    all_labels   = np.stack(all_labels)
    preds        = _batch_cfar(all_energies, config)

    pd, pfa, f1  = _compute_metrics(preds, all_labels)
    prec         = _precision(preds, all_labels)

    result = {
        "P_d"      : pd,
        "P_fa"     : pfa,
        "F1"       : f1,
        "precision": prec,
        "n_samples": len(all_labels),
        "alpha"    : config.alpha,
        "split"    : split,
    }

    if scenario_col is not None:
        def _snr_bin(v: float) -> str:
            edges = [-10, 0, 10, 20, 30, 41]
            for lo, hi in zip(edges[:-1], edges[1:]):
                if lo <= v < hi:
                    return f"[{lo},{hi})"
            return "other"

        if scenario_col == "snr_db":
            meta_vals = [_snr_bin(getattr(r, "snr_db")) for r in all_meta]
        else:
            meta_vals = [getattr(r, scenario_col, "unknown") for r in all_meta]

        breakdown = {}
        for val in sorted(set(meta_vals), key=lambda x: (x is None, str(x))):
            mask = np.array([v == val for v in meta_vals])
            if mask.sum() == 0:
                continue
            s_pd, s_pfa, s_f1 = _compute_metrics(preds[mask], all_labels[mask])
            breakdown[val] = {"P_d": s_pd, "P_fa": s_pfa, "F1": s_f1,
                              "n": int(mask.sum())}
        result["per_scenario"] = breakdown

    return result


def print_results(result: dict, title: str = "CFAR Baseline") -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f"{title}  [{result['split']}  n={result['n_samples']}]")
    print(f"{'='*50}")
    print(f"  Alpha     : {result['alpha']:.3f}")
    print(f"  P_d       : {result['P_d']:.4f}")
    print(f"  P_fa      : {result['P_fa']:.4f}")
    print(f"  Precision : {result['precision']:.4f}")
    print(f"  F1        : {result['F1']:.4f}")

    if "per_scenario" in result:
        print(f"\n  Scenario breakdown:")
        print(f"  {'Value':>12}  {'n':>6}  {'P_d':>7}  {'P_fa':>7}  {'F1':>7}")
        print("  " + "─" * 44)
        for val, m in result["per_scenario"].items():
            print(f"  {str(val):>12}  {m['n']:>6}  "
                  f"{m['P_d']:>7.4f}  {m['P_fa']:>7.4f}  {m['F1']:>7.4f}")


# ─────────────────────────────────────────────────────────────────────
#  INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────

def _load_metadata(dataset_dir: Path) -> dict[int, dict]:
    """Load metadata CSV, skipping duplicate header rows."""
    rows = {}
    with open(dataset_dir / "metadata.csv", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows[int(row["sample_idx"])] = row
            except (ValueError, KeyError):
                continue   # skip duplicate headers
    return rows



def _compute_metrics(
    preds : np.ndarray,   # (N, 25) uint8
    labels: np.ndarray,   # (N, 25) uint8
) -> tuple[float, float, float]:
    """Return (P_d, P_fa, F1) over all slot decisions."""
    TP = float(((preds == 1) & (labels == 1)).sum())
    FP = float(((preds == 1) & (labels == 0)).sum())
    TN = float(((preds == 0) & (labels == 0)).sum())
    FN = float(((preds == 0) & (labels == 1)).sum())

    pd   = TP / max(TP + FN, 1)
    pfa  = FP / max(FP + TN, 1)
    prec = TP / max(TP + FP, 1)
    f1   = 2 * prec * pd / max(prec + pd, 1e-10)
    return pd, pfa, f1


def _precision(preds: np.ndarray, labels: np.ndarray) -> float:
    TP = float(((preds == 1) & (labels == 1)).sum())
    FP = float(((preds == 1) & (labels == 0)).sum())
    return TP / max(TP + FP, 1)


# ─────────────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir",     type=Path, default=Path("gsm_dataset"))
    parser.add_argument("--target-far",      type=float, default=0.05)
    parser.add_argument("--guard-cells",     type=int,   default=1)
    parser.add_argument("--reference-cells", type=int,   default=4)
    parser.add_argument("--alpha",           type=float, default=None,
                        help="Skip tuning and use this alpha directly")
    parser.add_argument("--scenario-col",    type=str,   default="wb_blocker",
                        help="Metadata column for scenario breakdown")
    args = parser.parse_args()

    if args.alpha is not None:
        cfg = CFARConfig(alpha=args.alpha, guard_cells=args.guard_cells,
                         reference_cells=args.reference_cells,
                         target_far=args.target_far)
        print(f"Using fixed alpha={args.alpha:.3f} (skipping tuning)")
    else:
        cfg = tune_threshold(
            dataset_dir    = args.dataset_dir,
            split          = "validation",
            target_far     = args.target_far,
            guard_cells    = args.guard_cells,
            reference_cells= args.reference_cells,
        )

    # Evaluate on test set with scenario breakdown
    result = evaluate(
        dataset_dir  = args.dataset_dir,
        config       = cfg,
        split        = "test",
        scenario_col = args.scenario_col,
    )
    print_results(result, title="FFT-CFAR Baseline")

    # Also report validation numbers for reference
    val_result = evaluate(
        dataset_dir = args.dataset_dir,
        config      = cfg,
        split       = "validation",
    )
    print_results(val_result, title="FFT-CFAR Baseline (validation)")