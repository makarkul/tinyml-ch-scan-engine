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
    FeatureExtractionConfig,
    extract_spectral_features,
    load_complex64_cfile,
)

LOGGER = logging.getLogger(__name__)
SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class DatasetSplitConfig:
    validation_size: float = 0.15
    test_size: float       = 0.15
    random_seed: int       = 42


@dataclass(frozen=True)
class SampleRecord:
    """A validated IQ file path and its 25-slot occupancy label."""
    iq_path      : Path
    metadata_path: Path | None
    # occupancy is a (25,) int array — 1 = carrier present, 0 = empty
    occupancy    : np.ndarray


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


def _occupancy_from_json(metadata_path: Path) -> np.ndarray | None:

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


class GSMDataset(Dataset):


    def __init__(
        self,
        root_dir      : str | Path           = "dataset",
        split         : SplitName            = "train",
        split_config  : DatasetSplitConfig | None = None,
        feature_config: FeatureExtractionConfig | None = None,
    ) -> None:
        self.root_dir       = _resolve_dataset_root(root_dir)
        self.split          = split
        self.feature_config = feature_config or FeatureExtractionConfig()
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

        # occupancy: (25,) float32 for BCEWithLogitsLoss
        occupancy = torch.from_numpy(
            record.occupancy.astype(np.float32)
        )
        return torch.from_numpy(features), occupancy
