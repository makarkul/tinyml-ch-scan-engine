import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample as fft_resample


# ─────────────────────────────────────────────────────────────────────
#  SIGNAL CONSTANTS  (must match gsm_dataset_gen.py)
# ─────────────────────────────────────────────────────────────────────

TARGET_RATE = 104_000_000
SCAN_RATE   =   5_000_000
N_WINDOW    = 65_536
N_FFT       = 1024
N_SLOTS     = 25
WELCH_HOP   = N_FFT // 2    # 512 — 50% overlap, same as fft_channeliser.py
BIN_HZ      = SCAN_RATE / N_FFT

# Pre-compute slot bin ranges (identical to fft_channeliser.py)
SLOT_BINS = []
for _k in range(N_SLOTS):
    _c  = int(round((-2_400_000 + _k * 200_000 + SCAN_RATE / 2) / BIN_HZ))
    _lo = max(0,     _c - 20)
    _hi = min(N_FFT, _c + 21)
    SLOT_BINS.append((_lo, _hi))

# Feature and label sizes
N_ENERGY_FEATURES = N_SLOTS   # 25 — one energy ratio per slot
N_IMP_FLAGS       = 6
N_POWER_CLASSES   = 4         # 0=weak, 1=medium, 2=strong, 3=very_strong


# ─────────────────────────────────────────────────────────────────────
#  PHASE 1 FEATURE: PER-SLOT ENERGY RATIOS
# ─────────────────────────────────────────────────────────────────────

def extract_slot_energies(iq_int8: np.ndarray) -> np.ndarray:
    """
    Extract the minimal Phase 1 feature: per-slot energy ratios.

    This reuses the Welch power spectrum from the FFT channeliser baseline.
    For each of the 25 channels the output is:

        energy_ratio[k] = slot_energy[k] / median_noise_floor

    A value of 1.0 means the slot has the same energy as a typical empty
    slot. A value of 10.0 means it is 10x above the noise floor — a strong
    indicator of carrier presence.

    Normalising by the noise floor makes the feature scale-invariant.
    A signal at SNR +5 dB has the same ratio on a quiet radio environment
    as on a noisy one, because both signal and noise scale together.

    Parameters
    ----------
    iq_int8 : (2, 65536) int8 — row 0=I, row 1=Q

    Returns
    -------
    (25,) float32 — per-slot energy ratio, each value >= 0
    """
    # Reconstruct complex signal
    iq = (iq_int8[0].astype(np.float32) +
          1j * iq_int8[1].astype(np.float32))

    # Decimate 104 MS/s → 5 MS/s
    n_out     = int(round(N_WINDOW * SCAN_RATE / TARGET_RATE))
    decimated = fft_resample(iq, n_out)

    # Welch power spectrum — 5 segments, Hann(1024) each, 50% overlap
    window  = np.hanning(N_FFT)
    n_segs  = (len(decimated) - N_FFT) // WELCH_HOP + 1
    acc     = np.zeros(N_FFT, dtype=np.float64)
    for i in range(n_segs):
        seg  = decimated[i * WELCH_HOP : i * WELCH_HOP + N_FFT]
        spec = np.fft.fftshift(np.fft.fft(seg * window, n=N_FFT))
        acc += np.abs(spec) ** 2 / N_FFT
    power = acc / n_segs

    # Per-slot integrated energy
    slot_energy = np.array(
        [np.sum(power[lo:hi]) for lo, hi in SLOT_BINS],
        dtype=np.float32,
    )

    # Normalise by noise floor (median of all slot energies)
    # Robust: unaffected by occupied slots as long as fewer than 13 of
    # 25 slots are occupied — always true in this dataset
    noise_floor = float(np.median(slot_energy))
    if noise_floor > 1e-30:
        slot_energy = slot_energy / noise_floor

    return slot_energy   # (25,) float32


# ─────────────────────────────────────────────────────────────────────
#  PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────

class GSMScanDataset(Dataset):
    """
    PyTorch Dataset for the GSM channel scan project — Phase 1.

    Loads pre-generated .npy sample files. Each sample contains the raw
    IQ array and ground-truth labels. Feature extraction runs at load
    time inside the DataLoader worker.

    In Phase 1 the only feature returned is the 25-element per-slot
    energy ratio vector. Later phases will add the full F1-F14 set.

    Parameters
    ----------
    sample_files : list of Path or str — paths to .npy sample files
    transform    : optional callable applied to the feature tensor

    Returns per sample (from __getitem__)
    --------------------------------------
    features   : (25,) float32 tensor — per-slot energy ratios
    occupancy  : (25,) float32 tensor — binary ground truth (0 or 1)
    power_class: (25,) int64  tensor  — 0-3 for occupied, -1 for empty
    imp_flags  : (6,)  float32 tensor — multi-label binary impairments
    snr_db     : ()    float32 scalar
    """

    def __init__(self, sample_files: list, transform=None):
        self.files     = [Path(f) for f in sample_files]
        self.transform = transform

        if len(self.files) == 0:
            raise ValueError("sample_files is empty — check your dataset path")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple:
        # Load sample dict
        d = np.load(str(self.files[idx]), allow_pickle=True).item()

        # Extract minimal Phase 1 feature
        features = extract_slot_energies(d["iq"])   # (25,) float32

        # Apply optional transform (e.g. normalisation)
        if self.transform is not None:
            features = self.transform(features)

        # Convert to tensors
        features_t    = torch.from_numpy(features)
        occupancy_t   = torch.from_numpy(
            d["occupancy"].astype(np.float32)          # (25,) float32
        )
        power_class_t = torch.from_numpy(
            d["power_class"].astype(np.int64)          # (25,) int64
        )
        imp_flags_t   = torch.from_numpy(
            d["impairment_flags"].astype(np.float32)   # (6,)  float32
        )
        snr_t         = torch.tensor(float(d["snr_db"]), dtype=torch.float32)

        return features_t, occupancy_t, power_class_t, imp_flags_t, snr_t

    @staticmethod
    def from_directory(dataset_dir: str, n_samples: int = None) -> "GSMScanDataset":
        """
        Convenience constructor. Finds all sample_*.npy files under
        dataset_dir/samples/ and returns a Dataset over them.

        Parameters
        ----------
        dataset_dir : root dataset directory (contains samples/ subfolder)
        n_samples   : if set, limit to the first n_samples files
        """
        sample_dir   = Path(dataset_dir) / "samples"
        sample_files = sorted(sample_dir.glob("sample_*.npy"))

        if len(sample_files) == 0:
            raise FileNotFoundError(
                f"No sample_*.npy files found in {sample_dir}"
            )

        if n_samples is not None:
            sample_files = sample_files[:n_samples]

        return GSMScanDataset(sample_files)


# ─────────────────────────────────────────────────────────────────────
#  TRAIN / VALIDATION / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────

def split_dataset(dataset_dir: str,
                  train_frac: float = 0.70,
                  val_frac:   float = 0.15,
                  n_samples:  int   = None,
                  seed:       int   = 42
                  ) -> tuple:
    """
    Split sample files into train / validation / test sets.

    Split is done at the file level (scenario-level separation) to
    prevent any leakage between sets. The split is deterministic given
    the same seed.

    Returns (train_dataset, val_dataset, test_dataset).
    """
    sample_dir   = Path(dataset_dir) / "samples"
    all_files    = sorted(sample_dir.glob("sample_*.npy"))

    if n_samples is not None:
        all_files = all_files[:n_samples]

    N     = len(all_files)
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(N)

    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)

    train_files = [all_files[i] for i in idx[:n_train]]
    val_files   = [all_files[i] for i in idx[n_train : n_train + n_val]]
    test_files  = [all_files[i] for i in idx[n_train + n_val:]]

    return (
        GSMScanDataset(train_files),
        GSMScanDataset(val_files),
        GSMScanDataset(test_files),
    )


# ─────────────────────────────────────────────────────────────────────
#  VERIFICATION
# ─────────────────────────────────────────────────────────────────────

def verify_dataset(dataset: GSMScanDataset, n_check: int = 50) -> bool:
    """
    Run sanity checks on the Dataset. Prints a report and returns True
    if all checks pass.

    Checks
    ------
    1. Tensor shapes are correct for all n_check samples
    2. No NaN or Inf values in features
    3. Occupancy labels are binary (0 or 1 only)
    4. Power class is -1 for empty slots, 0-3 for occupied slots
    5. Impairment flags are binary (0 or 1 only)
    6. Class balance: occupied vs empty slot ratio
    7. SNR distribution looks reasonable
    """
    print(f"\nVerifying dataset ({min(n_check, len(dataset))} samples)...")
    print(f"  Total samples: {len(dataset)}")

    n_check   = min(n_check, len(dataset))
    all_pass  = True
    occ_total = 0
    emp_total = 0
    snr_vals  = []

    for i in range(n_check):
        features, occupancy, power_class, imp_flags, snr = dataset[i]

        # ── Shape checks ──────────────────────────────────────────────
        assert features.shape    == (N_SLOTS,),   \
            f"Sample {i}: features shape {features.shape} != ({N_SLOTS},)"
        assert occupancy.shape   == (N_SLOTS,),   \
            f"Sample {i}: occupancy shape {occupancy.shape} != ({N_SLOTS},)"
        assert power_class.shape == (N_SLOTS,),   \
            f"Sample {i}: power_class shape {power_class.shape} != ({N_SLOTS},)"
        assert imp_flags.shape   == (N_IMP_FLAGS,), \
            f"Sample {i}: imp_flags shape {imp_flags.shape} != ({N_IMP_FLAGS},)"

        # ── dtype checks ──────────────────────────────────────────────
        assert features.dtype    == torch.float32, \
            f"Sample {i}: features dtype {features.dtype} != float32"
        assert occupancy.dtype   == torch.float32, \
            f"Sample {i}: occupancy dtype {occupancy.dtype} != float32"
        assert power_class.dtype == torch.int64,   \
            f"Sample {i}: power_class dtype {power_class.dtype} != int64"

        # ── NaN / Inf check ───────────────────────────────────────────
        assert not torch.isnan(features).any(), \
            f"Sample {i}: NaN in features"
        assert not torch.isinf(features).any(), \
            f"Sample {i}: Inf in features"

        # ── Label validity ────────────────────────────────────────────
        assert occupancy.min() >= 0 and occupancy.max() <= 1, \
            f"Sample {i}: occupancy out of [0,1]"
        assert imp_flags.min() >= 0 and imp_flags.max() <= 1, \
            f"Sample {i}: imp_flags out of [0,1]"
        assert power_class.min() >= -1 and power_class.max() <= 3, \
            f"Sample {i}: power_class out of [-1,3]"

        # ── Mask consistency: power_class == -1 iff slot is empty ─────
        occ_mask = occupancy.bool()
        assert (power_class[~occ_mask] == -1).all(), \
            f"Sample {i}: power_class != -1 for empty slot"
        assert (power_class[occ_mask] >= 0).all(), \
            f"Sample {i}: power_class < 0 for occupied slot"

        # Accumulate stats
        occ_total += int(occupancy.sum())
        emp_total += int((1 - occupancy).sum())
        snr_vals.append(float(snr))

    snr_arr = np.array(snr_vals)
    occ_rate = occ_total / (occ_total + emp_total) * 100

    print(f"  Shape checks     : PASS")
    print(f"  NaN/Inf checks   : PASS")
    print(f"  Label validity   : PASS")
    print(f"  Mask consistency : PASS")
    print(f"  Occupied slots   : {occ_total} / {occ_total+emp_total}  "
          f"({occ_rate:.1f}%)")
    print(f"  SNR range        : {snr_arr.min():.1f} to {snr_arr.max():.1f} dB  "
          f"(mean {snr_arr.mean():.1f} dB)")
    print(f"  All checks PASSED\n")

    return all_pass


# ─────────────────────────────────────────────────────────────────────
#  CLI — run verification and print one sample
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Phase 1 — Dataset verification"
    )
    ap.add_argument("--dataset",   type=str, required=True,
                    help="Path to dataset root directory")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Limit to first N samples")
    ap.add_argument("--batch-size",type=int, default=32)
    ap.add_argument("--num-workers",type=int, default=2)
    args = ap.parse_args()

    # Build and split dataset
    train_ds, val_ds, test_ds = split_dataset(
        args.dataset, n_samples=args.n_samples
    )
    print(f"Dataset split:")
    print(f"  Train : {len(train_ds)} samples")
    print(f"  Val   : {len(val_ds)}   samples")
    print(f"  Test  : {len(test_ds)}  samples")

    # Verify train set
    verify_dataset(train_ds)

    # Print one sample in detail
    features, occupancy, power_class, imp_flags, snr = train_ds[0]
    print("Sample 0 detail:")
    print(f"  SNR         : {snr:.1f} dB")
    print(f"  Occupied    : {occupancy.nonzero(as_tuple=True)[0].tolist()}")
    print(f"  Imp flags   : {imp_flags.int().tolist()}")
    print(f"  Features    : shape={list(features.shape)}  "
          f"min={features.min():.3f}  max={features.max():.3f}")
    print()
    print("  Per-slot energy ratios (occupied slots highlighted):")
    for k in range(N_SLOTS):
        ratio = float(features[k])
        occ   = int(occupancy[k])
        bar   = "█" * int(min(ratio, 20))
        if ratio > 1.5 or occ:
            marker = " <-- occupied" if occ else ""
            print(f"    slot {k:2d}: {ratio:6.2f}x  {bar}{marker}")

    # Test DataLoader throughput
    import time
    loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = False,
    )
    print(f"\nDataLoader throughput test")
    print(f"  batch_size={args.batch_size}  num_workers={args.num_workers}")
    t0 = time.perf_counter()
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 10:
            break
    elapsed = time.perf_counter() - t0
    samples_per_sec = min(10 * args.batch_size, len(train_ds)) / elapsed
    print(f"  10 batches in {elapsed:.2f}s  "
          f"= {samples_per_sec:.0f} samples/sec")
    print(f"  {'OK — sufficient for training' if samples_per_sec > 50 else 'SLOW — consider fewer workers or caching features'}")


if __name__ == "__main__":
    main()