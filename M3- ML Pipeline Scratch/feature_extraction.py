from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

EPSILON           = 1e-12
N_SLOTS           = 25
FEATURES_PER_SLOT = 5
FEATURE_DIM       = N_SLOTS * FEATURES_PER_SLOT   # 125

TARGET_RATE = 104_000_000
SCAN_RATE   =   5_000_000
N_WINDOW    =  65_536
N_DEC       = int(round(N_WINDOW * SCAN_RATE / TARGET_RATE))   # 3151
N_SEGMENTS  = 5    # temporal segments for stability and coherence

# GSM slot centre frequencies in Hz
# Slot k: centre = -2400 kHz + k × 200 kHz
SLOT_FREQ_HZ = np.array(
    [-2_400_000 + k * 200_000 for k in range(N_SLOTS)],
    dtype=np.float64,
)

# Pre-compute reference sinusoids for all 25 slots
# Shape: (25, N_DEC) complex128
# ref[k, t] = exp(-j2π f_k t / SCAN_RATE)
# Correlating the signal with ref[k] gives energy at slot k's frequency
_t   = np.arange(N_DEC, dtype=np.float64)
_REF = np.exp(
    -1j * 2 * np.pi * SLOT_FREQ_HZ[:, None] * _t[None, :] / SCAN_RATE
).astype(np.complex64)   # (25, 3151)

# Segment boundaries for temporal analysis
_SEG_BOUNDARIES = [
    (i * N_DEC // N_SEGMENTS, (i + 1) * N_DEC // N_SEGMENTS)
    for i in range(N_SEGMENTS)
]


@dataclass(frozen=True)
class FeatureExtractionConfig:
    normalize : bool = True
    n_segments: int  = 5


def load_complex64_cfile(path: str | Path) -> np.ndarray:
    """Load GNU Radio-style complex64 IQ samples from a .cfile."""
    file_path = Path(path)
    try:
        iq = np.fromfile(file_path, dtype=np.complex64)
    except OSError as exc:
        LOGGER.warning("Could not read IQ file %s: %s", file_path, exc)
        return np.empty(0, dtype=np.complex64)
    if iq.size == 0:
        LOGGER.warning("IQ file %s is empty", file_path)
    return iq.astype(np.complex64, copy=False)


def _decimate(iq: np.ndarray) -> np.ndarray:
    from scipy.signal import resample as fft_resample

    if iq.dtype == np.int8 and iq.ndim == 2 and iq.shape[0] == 2:
        iq = (iq[0].astype(np.float32) +
              1j * iq[1].astype(np.float32)).astype(np.complex64)

    iq = np.asarray(iq, dtype=np.complex64).reshape(-1)

    if len(iq) < N_DEC:
        # Zero-pad short captures
        padded        = np.zeros(N_DEC, dtype=np.complex64)
        padded[:len(iq)] = iq
        return padded

    if len(iq) == N_DEC:
        return iq

    return fft_resample(iq, N_DEC).astype(np.complex64)


def _matched_filter_bank(signal: np.ndarray) -> np.ndarray:

    # Matrix multiply: (25, N_DEC) × (N_DEC,) → (25,)
    # conj(_REF) because correlation = sum(signal × conj(ref))
    return (np.conj(_REF) @ signal.astype(np.complex64)) / N_DEC


def _segment_energies(signal: np.ndarray, k: int) -> np.ndarray:

    energies = np.zeros(N_SEGMENTS, dtype=np.float32)
    for i, (lo, hi) in enumerate(_SEG_BOUNDARIES):
        seg          = signal[lo:hi]
        ref_seg      = _REF[k, lo:hi]
        corr         = np.dot(np.conj(ref_seg), seg.astype(np.complex64))
        energies[i]  = float(np.abs(corr) ** 2) / max((hi - lo) ** 2, 1)
    return energies


# ─────────────────────────────────────────────────────────────────────
#  STEP 3 — PER-SLOT FEATURES
# ─────────────────────────────────────────────────────────────────────

def _per_slot_features(
    signal    : np.ndarray,
    corr_full : np.ndarray,
    slot_energies: np.ndarray,
    k         : int,
) -> np.ndarray:

    # Normalised by the maximum energy across all slots so features
    # are scale-invariant across different SNR levels.
    max_e = float(np.max(slot_energies)) + EPSILON
    f1    = float(slot_energies[k]) / max_e

    # Variance of energy across 5 time segments, normalised by mean.
    # A stable carrier has low relative variance → high stability score.
    # Random noise has high relative variance → low stability score.
    seg_e = _segment_energies(signal, k)
    mean_e = float(np.mean(seg_e)) + EPSILON
    std_e  = float(np.std(seg_e))
    f2     = max(0.0, 1.0 - std_e / mean_e)   # 1 = perfectly stable

    # A real carrier has a consistent phase rotation across time.
    # Compute the correlation as a complex number in each segment,
    # normalise each to unit magnitude, then measure how well they
    # point in the same direction (vector magnitude of their mean).
    # Value close to 1.0 = coherent carrier. Close to 0 = noise.
    phase_vectors = np.zeros(N_SEGMENTS, dtype=np.complex64)
    for i, (lo, hi) in enumerate(_SEG_BOUNDARIES):
        seg   = signal[lo:hi]
        ref_s = _REF[k, lo:hi]
        corr  = np.dot(np.conj(ref_s), seg.astype(np.complex64))
        mag   = float(np.abs(corr)) + EPSILON
        phase_vectors[i] = corr / mag   # unit complex number

    f3 = float(np.abs(np.mean(phase_vectors)))   # 0 to 1

    # Is this slot a local energy maximum?
    # A real carrier creates a peak. Leakage from an adjacent carrier
    # decays as you move away — it is not a local maximum.
    left_e  = float(slot_energies[k-1]) if k > 0             else 0.0
    right_e = float(slot_energies[k+1]) if k < N_SLOTS-1     else 0.0
    neighbour_mean = (left_e + right_e) / 2.0 + EPSILON
    f4 = float(slot_energies[k]) / neighbour_mean

    # Energy relative to the median across all 25 slots.
    # This is the CFAR-style normalisation used by the FFT channeliser.
    noise_floor = float(np.median(slot_energies)) + EPSILON
    f5 = float(slot_energies[k]) / noise_floor

    return np.array([f1, f2, f3, f4, f5], dtype=np.float32)




def extract_spectral_features(
    iq    : np.ndarray,
    config: FeatureExtractionConfig | None = None,
) -> np.ndarray:

    cfg    = config or FeatureExtractionConfig()
    signal = _decimate(iq)                          # (3151,) complex64

    # Compute full-window correlations and energies for all slots at once
    corr_full     = _matched_filter_bank(signal)    # (25,) complex
    slot_energies = np.abs(corr_full) ** 2          # (25,) float32

    # Build per-slot features
    all_features = np.zeros((N_SLOTS, FEATURES_PER_SLOT), dtype=np.float32)
    for k in range(N_SLOTS):
        all_features[k] = _per_slot_features(
            signal, corr_full, slot_energies.astype(np.float32), k
        )

    features = all_features.reshape(-1)             # (125,) float32

    if features.shape[0] != FEATURE_DIM:
        raise RuntimeError(
            f"Expected {FEATURE_DIM} features, got {features.shape[0]}"
        )

    if cfg.normalize:
        mu      = float(np.mean(features))
        std     = float(np.std(features))
        features = ((features - mu) / max(std, EPSILON)).astype(np.float32)

    return np.nan_to_num(
        features, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
