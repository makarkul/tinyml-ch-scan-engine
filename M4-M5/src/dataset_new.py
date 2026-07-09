"""PyTorch dataset for GSM per-slot occupancy detection.

Returns per-sample tensors and occupancy labels of shape (25,) —
one binary value per GSM channel.

Two dataset modes:
  "features"  →  (125,) float32 matched-filter feature vector  [MLP]
  "iq"        →  (2, 3151) float32 decimated I/Q tensor        [TCN]

Supports two dataset layouts:
  1. Real captures: *.cfile + *.json metadata (carrier_present + slot info)
  2. Simulator:     metadata.csv + samples/*.npy
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    train_test_split = None

from feature_extraction import (
    FEATURE_DIM,
    N_SLOTS,
    FEATURES_PER_SLOT,
    FeatureExtractionConfig,
    extract_spectral_features,
    load_complex64_cfile,
    _decimate,
)

LOGGER = logging.getLogger(__name__)
SplitName = Literal["train", "validation", "test"]
DatasetMode = Literal["features", "iq"]
N_DEC = 3151


@dataclass(frozen=True)
class DatasetSplitConfig:
    validation_size: float = 0.15
    test_size: float       = 0.15
    random_seed: int       = 42



@dataclass(frozen=True)
class SampleRecord:
    iq_path      : Path
    metadata_path: Path | None
    occupancy    : np.ndarray
    snr_db        : float | None = None
    wb_blocker    : int   | None = None
    n_active_slots: int   | None = None
    adj_ch        : int   | None = None
    cfo_hz        : float | None = None

# ─────────────────────────────────────────────────────────────────────
#  PATH RESOLUTION
# ─────────────────────────────────────────────────────────────────────

def _resolve_dataset_root(root_dir: str | Path) -> Path:
    root = Path(root_dir)
    if root.exists():
        return root
    if root == Path("dataset"):
        for candidate in (
            Path("gsm_dataset_10k_new"),
            Path("gsm_dataset_10k"),
            Path("gsm_dataset"),
        ):
            if candidate.exists():
                LOGGER.warning(
                    "Dataset directory %s not found; using %s instead",
                    root, candidate,
                )
                return candidate
    raise FileNotFoundError(
        f"Dataset directory does not exist: {root}. "
        "Pass --dataset-dir, for example --dataset-dir gsm_dataset."
    )


# ─────────────────────────────────────────────────────────────────────
#  LABEL READING
# ─────────────────────────────────────────────────────────────────────

def _occupancy_from_json(metadata_path: Path) -> np.ndarray | None:
    """Read 25-slot occupancy from a .json metadata file.

    The JSON can provide occupancy in two ways:
      - 'occupancy': [0,0,1,...] list of 25 binary values  (preferred)
      - 'carrier_present': 0 or 1  (fallback — assumes carrier is in slot 12
        if present, all empty otherwise)
    """
    try:
        with metadata_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Skipping malformed metadata %s: %s", metadata_path, exc)
        return None

    occ = meta.get("occupancy")
    if isinstance(occ, list) and len(occ) == N_SLOTS:
        return np.array(occ, dtype=np.int32)

    # Fallback: binary label — treat as slot 12 occupied for real captures
    carrier_present = meta.get("carrier_present")
    if carrier_present in (0, 1, False, True):
        arr = np.zeros(N_SLOTS, dtype=np.int32)
        if int(carrier_present):
            arr[12] = 1   # assume slot 12 for real Vodafone captures
        return arr

    LOGGER.warning(
        "Skipping %s: no usable occupancy label", metadata_path
    )
    return None


def _occupancy_from_csv_row(row: dict[str, str]) -> np.ndarray | None:
    """Read 25-slot occupancy from a simulator metadata.csv row."""

    # Preferred: occupancy_bitmap column
    bitmap_str = row.get("occupancy_bitmap")
    if bitmap_str:
        try:
            bitmap = ast.literal_eval(bitmap_str)
            if isinstance(bitmap, list) and len(bitmap) == N_SLOTS:
                return np.array([int(v) for v in bitmap], dtype=np.int32)
        except (ValueError, SyntaxError) as exc:
            LOGGER.warning("Malformed occupancy_bitmap %s: %s", bitmap_str, exc)

    # Fallback: n_active_slots / carrier_present (binary)
    carrier_present = row.get("carrier_present")
    if carrier_present in {"0", "1"}:
        arr = np.zeros(N_SLOTS, dtype=np.int32)
        # Cannot know which slots without bitmap — mark as unknown
        # Return None so the sample is skipped rather than mislabelled
        LOGGER.warning(
            "Row has carrier_present but no occupancy_bitmap — skipping"
        )
        return None

    return None


# ─────────────────────────────────────────────────────────────────────
#  SAMPLE DISCOVERY
# ─────────────────────────────────────────────────────────────────────

def _discover_cfile_json_samples(root: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for iq_path in sorted(root.glob("*.cfile")):
        meta_path = iq_path.with_suffix(".json")
        if not meta_path.exists():
            LOGGER.warning(
                "Skipping %s: missing metadata %s", iq_path, meta_path
            )
            continue
        occ = _occupancy_from_json(meta_path)
        if occ is None:
            continue
        records.append(SampleRecord(
            iq_path=iq_path, metadata_path=meta_path, occupancy=occ
        ))
    return records


def _discover_csv_npy_samples(root: Path) -> list[SampleRecord]:
    metadata_path = root / "metadata.csv"
    samples_dir   = root / "samples"
    if not metadata_path.exists() or not samples_dir.exists():
        return []

    records: list[SampleRecord] = []
    seen: set[Path] = set()

    try:
        with metadata_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                filename = row.get("filename")
                if not filename or filename == "filename":
                    continue
                iq_path = samples_dir / filename
                if not iq_path.exists() or iq_path in seen:
                    continue
                seen.add(iq_path)

                occ = _occupancy_from_csv_row(row)
                if occ is None:
                    # Try loading .npy directly for occupancy
                    try:
                        d   = np.load(str(iq_path), allow_pickle=True).item()
                        occ = d["occupancy"].astype(np.int32)
                    except Exception:
                        LOGGER.warning(
                            "Skipping %s: could not get occupancy", iq_path
                        )
                        continue

                records.append(SampleRecord(
                iq_path=iq_path,
                metadata_path=metadata_path,
                occupancy=occ,
                snr_db=float(row.get("snr_db", "nan")),
                wb_blocker=int(row.get("imp_wb_blocker", 0)),
                n_active_slots=int(row.get("n_active_slots", 0)),
                adj_ch=int(row.get("imp_adj_ch", 0)),
                cfo_hz=float(row.get("cfo_hz", "nan")),
                ))
    except OSError as exc:
        LOGGER.warning("Could not read %s: %s", metadata_path, exc)

    return records


def discover_samples(root_dir: str | Path) -> list[SampleRecord]:
    root    = _resolve_dataset_root(root_dir)
    records = _discover_cfile_json_samples(root)
    if not records:
        records = _discover_csv_npy_samples(root)
    if not records:
        raise RuntimeError(
            f"No valid GSM samples found in {root}."
        )
    return records


# ─────────────────────────────────────────────────────────────────────
#  IQ LOADING
# ─────────────────────────────────────────────────────────────────────

def _extract_iq_from_npy(payload) -> np.ndarray:
    """Extract complex IQ from a simulator .npy payload."""

    if isinstance(payload, np.ndarray) and payload.dtype == object:
        payload = payload.item()

    if isinstance(payload, dict):
        iq = payload.get("iq")
        if iq is not None:
            iq = np.asarray(iq)
            # int8 (2, N) simulator format
            if iq.dtype == np.int8 and iq.ndim == 2 and iq.shape[0] == 2:
                return (iq[0].astype(np.float32) +
                        1j * iq[1].astype(np.float32)).astype(np.complex64)
            return iq.astype(np.complex64, copy=False).reshape(-1)

    iq = np.asarray(payload)
    if np.iscomplexobj(iq):
        return iq.astype(np.complex64, copy=False).reshape(-1)
    if iq.ndim >= 2 and iq.shape[0] == 2:
        return (iq[0] + 1j * iq[1]).astype(np.complex64).reshape(-1)

    LOGGER.warning("Unsupported .npy IQ shape: %s", iq.shape)
    return np.empty(0, dtype=np.complex64)


def load_iq_samples(path: str | Path) -> np.ndarray:
    iq_path = Path(path)
    if iq_path.suffix == ".cfile":
        return load_complex64_cfile(iq_path)
    if iq_path.suffix == ".npy":
        try:
            payload = np.load(iq_path, allow_pickle=True)
        except (OSError, ValueError) as exc:
            LOGGER.warning("Could not read %s: %s", iq_path, exc)
            return np.empty(0, dtype=np.complex64)
        return _extract_iq_from_npy(payload)
    LOGGER.warning("Unsupported extension: %s", iq_path)
    return np.empty(0, dtype=np.complex64)


# ─────────────────────────────────────────────────────────────────────
#  TRAIN / VAL / TEST SPLITTING
# ─────────────────────────────────────────────────────────────────────

def split_records(
    records: list[SampleRecord],
    split: SplitName,
    config: DatasetSplitConfig | None = None,
) -> list[SampleRecord]:
    cfg = config or DatasetSplitConfig()

    # Stratify on whether ANY slot is occupied
    labels   = np.array([int(r.occupancy.any()) for r in records])
    stratify = labels if len(np.unique(labels)) > 1 else None

    if train_test_split is None:
        train_val, test = _fallback_split(
            records, cfg.test_size, cfg.random_seed, stratify is not None
        )
    else:
        train_val, test = train_test_split(
            records,
            test_size=cfg.test_size,
            random_state=cfg.random_seed,
            stratify=stratify,
        )

    tv_labels   = np.array([int(r.occupancy.any()) for r in train_val])
    tv_stratify = tv_labels if len(np.unique(tv_labels)) > 1 else None
    val_frac    = cfg.validation_size / max(1.0 - cfg.test_size, 1e-6)

    if train_test_split is None:
        train, val = _fallback_split(
            train_val, val_frac, cfg.random_seed, tv_stratify is not None
        )
    else:
        train, val = train_test_split(
            train_val,
            test_size=val_frac,
            random_state=cfg.random_seed,
            stratify=tv_stratify,
        )

    return {"train": list(train), "validation": list(val),
            "test": list(test)}[split]


def _fallback_split(records, test_size, seed, stratify):
    rng = np.random.default_rng(seed)
    if not stratify:
        idx = np.arange(len(records)); rng.shuffle(idx)
        n   = max(1, int(round(len(records) * test_size)))
        ts  = set(idx[:n].tolist())
        return ([r for i,r in enumerate(records) if i not in ts],
                [r for i,r in enumerate(records) if i in ts])
    train, test = [], []
    for lbl in sorted({int(r.occupancy.any()) for r in records}):
        cls = [r for r in records if int(r.occupancy.any()) == lbl]
        idx = np.arange(len(cls)); rng.shuffle(idx)
        n   = max(1, int(round(len(cls) * test_size)))
        ts  = set(idx[:n].tolist())
        train.extend(r for i,r in enumerate(cls) if i not in ts)
        test.extend( r for i,r in enumerate(cls) if i in ts)
    rng.shuffle(train); rng.shuffle(test)
    return train, test


# ─────────────────────────────────────────────────────────────────────
#  PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────

class GSMDataset(Dataset):
    """Dataset returning (features, occupancy) pairs.

    Each item (mode="features"):
        features  : float32 tensor  (125,)  -- 5 per-slot features × 25 slots
        occupancy : float32 tensor  (25,)   -- binary label per slot

    Each item (mode="iq"):
        features  : float32 tensor  (2, 3151) -- decimated I/Q
        occupancy : float32 tensor  (25,)      -- binary label per slot
    """

    def __init__(
        self,
        root_dir      : str | Path           = "dataset",
        split         : SplitName            = "train",
        split_config  : DatasetSplitConfig | None = None,
        feature_config: FeatureExtractionConfig | None = None,
        mode          : DatasetMode               = "features",
    ) -> None:
        self.root_dir       = _resolve_dataset_root(root_dir)
        self.split          = split
        self.feature_config = feature_config or FeatureExtractionConfig()
        self.mode           = mode
        all_records         = discover_samples(self.root_dir)
        self.records        = split_records(
            all_records, split=split, config=split_config
        )
        LOGGER.info(
            "Loaded %d %s samples from %s",
            len(self.records), split, self.root_dir,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        iq     = load_iq_samples(record.iq_path)

        # occupancy: (25,) float32 for BCEWithLogitsLoss
        occupancy = torch.from_numpy(record.occupancy.astype(np.float32))

        if self.mode == "iq":
            # TCN path: return decimated I/Q as (2, N_DEC) float32
            if iq.size == 0:
                LOGGER.warning(
                    "Using zero IQ for unreadable sample %s",
                    record.iq_path,
                )
                iq_dec = np.zeros(N_DEC, dtype=np.complex64)
            else:
                try:
                    iq_dec = _decimate(iq)          # (N_DEC,) complex64
                except Exception as exc:
                    LOGGER.warning(
                        "Decimation failed for %s: %s", record.iq_path, exc,
                    )
                    iq_dec = np.zeros(N_DEC, dtype=np.complex64)

            # Stack real/imag -> (2, N_DEC) float32
            iq_tensor = torch.from_numpy(
                np.stack([iq_dec.real, iq_dec.imag], axis=0).astype(np.float32)
            )
            return iq_tensor, occupancy

        # "features" path (default / MLP): return (FEATURE_DIM,) float32
        if iq.size == 0:
            LOGGER.warning(
                "Using zero features for unreadable sample %s",
                record.iq_path,
            )
            features = np.zeros(FEATURE_DIM, dtype=np.float32)
        else:
            try:
                features = extract_spectral_features(iq, self.feature_config)
            except Exception as exc:
                LOGGER.warning(
                    "Feature extraction failed for %s: %s",
                    record.iq_path, exc,
                )
                features = np.zeros(FEATURE_DIM, dtype=np.float32)

        return torch.from_numpy(features), occupancy

# ─────────────────────────────────────────────────────────────────────
#  WEIGHTED SAMPLER  (fix class imbalance without pos_weight alone)
# ─────────────────────────────────────────────────────────────────────

def make_weighted_sampler(dataset: GSMDataset) -> torch.utils.data.WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that upsamples occupied samples.

    Samples with more active slots get higher sampling probability,
    so the model sees more examples of carrier detection during training.
    Empty samples (0 carriers) get weight 1.0; samples with k carriers
    get weight (1 + k * carrier_boost).

    This complements pos_weight in the loss — pos_weight penalises
    missed detections, weighted sampling ensures the gradient sees more
    occupied-slot examples per epoch.

    Usage
    -----
        from dataset_new import GSMDataset, make_weighted_sampler
        from torch.utils.data import DataLoader

        ds      = GSMDataset(dataset_dir, "train", ...)
        sampler = make_weighted_sampler(ds)
        loader  = DataLoader(ds, batch_size=64, sampler=sampler)
        # Note: sampler and shuffle=True are mutually exclusive
    """
    carrier_boost = 4.0   # each active slot multiplies sample weight by this

    weights = []
    for record in dataset.records:
        n_active = int(record.occupancy.sum())
        w        = 1.0 + n_active * carrier_boost
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights        = weights,
        num_samples    = len(weights),
        replacement    = True,
    )
    LOGGER.info(
        "WeightedRandomSampler: mean_weight=%.2f  max_weight=%.2f",
        float(weights.mean()), float(weights.max()),
    )
    return sampler


# ─────────────────────────────────────────────────────────────────────
#  AUGMENTED DATASET  (wrap GSMDataset with on-the-fly augmentation)
# ─────────────────────────────────────────────────────────────────────

class AugmentedGSMDataset(torch.utils.data.Dataset):
    """
    Wraps GSMDataset with on-the-fly feature-space augmentation.

    Three augmentations applied randomly during training:

    1. Phase rotation (always safe)
       Multiply the complex IQ by exp(jφ) before feature extraction.
       Preserves all energy features (F1, F4, F5) — they depend on |corr|².
       Randomises F3 (phase coherence) slightly, teaching robustness.
       Implementation: rotate features in pairs (F3 is already magnitude).

    2. Energy scaling (±3 dB)
       Multiply all slot energies by a random scale factor in [0.5, 2.0].
       F1 (normalised) is unchanged. F4/F5 (ratio features) are unchanged.
       F2 (temporal stability) is unchanged.
       This augmentation is effectively a no-op for the normalised features
       but adds minor diversity to the training distribution.

    3. Slot energy jitter (±10%)
       Add small multiplicative noise to each slot's energy independently.
       Simulates minor calibration errors and teaches the model that
       small energy differences between slots may not be meaningful.

    Parameters
    ----------
    base_dataset : GSMDataset in "features" mode
    augment      : whether to apply augmentation (set False for val/test)
    jitter_std   : std of multiplicative noise on slot energies (default 0.05)
    """

    def __init__(
        self,
        base_dataset: GSMDataset,
        augment     : bool  = True,
        jitter_std  : float = 0.05,
    ) -> None:
        self.ds         = base_dataset
        self.augment    = augment
        self.jitter_std = jitter_std

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, occupancy = self.ds[index]

        if not self.augment or not self.training_mode():
            return features, occupancy

        feat = features.numpy().copy()[:FEATURE_DIM].reshape(N_SLOTS, FEATURES_PER_SLOT)   # (25, 6)   # (25, 5)

        # Augmentation 1: slot energy jitter on F1, F4, F5 (indices 0, 3, 4)
        # F2 (stability) and F3 (coherence) are left unchanged
        if np.random.random() < 0.5:
            noise = 1.0 + np.random.randn(25) * self.jitter_std
            noise = np.clip(noise, 0.5, 2.0).astype(np.float32)
            feat[:, 0] *= noise   # F1
            feat[:, 3] *= noise   # F4 (cross-slot ratio — relative, mostly invariant)
            feat[:, 4] *= noise   # F5 (noise floor ratio — relative, mostly invariant)
            # Re-clip log features to valid range
            feat[:, 3] = np.clip(feat[:, 3], 0.0, 8.0)
            feat[:, 4] = np.clip(feat[:, 4], 0.0, 8.0)
            # Re-normalise F1 so max = 1
            max_f1 = feat[:, 0].max()
            if max_f1 > 0:
                feat[:, 0] /= max_f1

        return torch.from_numpy(feat.reshape(-1)), occupancy

    def training_mode(self) -> bool:
        """Check if we are in training mode (augment only during training)."""
        return self.ds.split == "train"