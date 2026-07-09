"""Matched-filter feature extraction for TinyML GSM occupancy detection.

Extracts 6 features per slot × 25 slots = 150 features total.
No FFT, no spectrogram, no CNN. Pure time-domain signal processing.

Feature groups per slot
-----------------------
  F1  carrier energy       -- |corr_k|² / max_energy across all slots  [0, 1]
  F2  temporal stability   -- 1 - CV of segment energies               [0, 1]
  F3  phase coherence      -- |mean of unit phase vectors|             [0, 1]
  F4  cross-slot ratio     -- log1p(energy_k / neighbour_mean)         [0, ∞) soft
  F5  noise-floor ratio    -- log1p(energy_k / median_energy)          [0, ∞) soft
  F6  absolute energy      -- log1p(energy_k / NOISE_FLOOR_REF)        [0, ∞) soft

Normalization policy
--------------------
The original code applied a single global z-score across all 150 values
per sample. This corrupts the features because:
  - F1/F2/F3 are already bounded [0, 1] and need no scaling.
  - F4/F5/F6 are ratios that can be 0.01–100+ and dominate the global mean.
  - The global std changes wildly depending on how many carriers are present,
    making the same physical situation produce different normalized values.

Fix: remove per-sample global normalization entirely.
  - F1, F2, F3 are already in [0, 1] by construction.
  - F4, F5, F6 are log-compressed to [0, ~5] for typical carrier/noise ratios,
    then soft-clipped at 5.0 to prevent outliers from dominating.
  - The MLP receives a BatchNorm1d(150 ) as its first layer, which learns
    per-feature-index mean/std from the training distribution. This is the
    correct place for normalization.

Performance change
------------------
Removing the broken global z-score and adding log compression to F4/F5
is the primary fix for the 0.73 P_d ceiling. Expect a meaningful jump.

Why matched filters instead of FFT
-----------------------------------
A full 1024-bin FFT computes correlation with 1024 sinusoids.
You only need 25. Computing 25 matched filters is:
  - 40× fewer multiplications than a full FFT
  - No windowing, no fftshift, no bin extraction
  - Directly interpretable: each feature answers one question
  - Naturally gives temporal information by splitting into segments
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
EPSILON           = 1e-12
N_SLOTS           = 25
FEATURES_PER_SLOT = 6
FEATURE_DIM       = N_SLOTS * FEATURES_PER_SLOT   # 150

TARGET_RATE = 104_000_000
SCAN_RATE   =   5_000_000
N_WINDOW    =  65_536
N_DEC       = int(round(N_WINDOW * SCAN_RATE / TARGET_RATE))   # 3151
N_SEGMENTS  = 5    # temporal segments for stability and coherence

# Exact rational resampling ratio SCAN_RATE/TARGET_RATE = 5e6/104e6 = 5/104.
# Used by resample_poly for proper polyphase decimation (see _decimate).
_RESAMPLE_UP   = 5
_RESAMPLE_DOWN = 104

# Maximum value for log-compressed ratio features (F4, F5).
# Raised from 5.0 -> 8.0: diagnostics showed moderate-SNR carrier/noise
# ratios (roughly 2x-20x) were being compressed into too narrow a band by
# the tighter clip, hurting separability. log1p(2980) ~= 8.0, so this only
# clips truly extreme outliers, not the typical-SNR range that matters most.
_LOG_CLIP = 8.0

# GSM slot centre frequencies in Hz
# Slot k: centre = -2400 kHz + k × 200 kHz
SLOT_FREQ_HZ = np.array(
    [-2_400_000 + k * 200_000 for k in range(N_SLOTS)],
    dtype=np.float64,
)

# Pre-compute reference sinusoids for all 25 slots
# Shape: (25, N_DEC) complex64
# ref[k, t] = exp(+j2π f_k t / SCAN_RATE)
#
# BUG FIX: this previously used exp(-j2π f_k t / SCAN_RATE) (negative sign).
# _matched_filter_bank() computes conj(_REF) @ signal. For a real tone at
# +f0, conj(REF[k]) @ signal stays coherent (sums constructively) only when
# the sinusoid embedded in conj(REF[k]) rotates at -f0 relative to the
# tone -- i.e. when REF[k] itself rotates at +f_k. The old minus sign meant
# REF[k] rotated at -f_k, so conj(REF[k]) rotated at +f_k, and the product
# conj(REF[k])*signal = exp(j2*pi*f_k*t/fs)*exp(j2*pi*f0*t/fs) only stays
# coherent when f_k = -f0 -- i.e. every slot was matched to its MIRROR
# frequency. Confirmed empirically: real-data peak_slot == 24 - true_slot
# in 100% of clean single-carrier test cases (diagnose_slot_alignment.py).
# Removing the minus sign here fixes the sign convention so
# conj(REF[k]) @ signal peaks when f_k == f0, as intended.
_t   = np.arange(N_DEC, dtype=np.float64)
_REF = np.exp(
    1j * 2 * np.pi * SLOT_FREQ_HZ[:, None] * _t[None, :] / SCAN_RATE
).astype(np.complex64)   # (25, 3151)

# Segment boundaries for temporal analysis — precomputed once
_SEG_BOUNDARIES: list[tuple[int, int]] = [
    (i * N_DEC // N_SEGMENTS, (i + 1) * N_DEC // N_SEGMENTS)
    for i in range(N_SEGMENTS)
]

# Precomputed reference segments for each (slot, segment) pair
# Shape: (N_SEGMENTS, N_SLOTS, seg_len) — jagged so stored as list of arrays
_REF_SEGS: list[np.ndarray] = [
    _REF[:, lo:hi]   # (25, seg_len)
    for lo, hi in _SEG_BOUNDARIES
]


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Configuration for matched-filter feature extraction.

    Attributes:
        normalize: Kept for API compatibility. Now always False — per-sample
            global z-score normalization has been removed because it destroys
            the relative structure between features. Use BatchNorm1d(150) as
            the first layer of the MLP instead.
        n_segments: Number of time segments for temporal features.
            Default 5 matches the Welch frame count used elsewhere.
    """
    normalize : bool = False   # was True — see module docstring for why changed
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


# ─────────────────────────────────────────────────────────────────────
#  STEP 1 — DECIMATE
# ─────────────────────────────────────────────────────────────────────

def _decimate(iq: np.ndarray) -> np.ndarray:
    """
    Decimate IQ to SCAN_RATE (5 MS/s) and return (N_DEC,) complex64.

    Accepts two input formats:
      - complex64 (N,)       from .cfile
      - int8 (2, N_WINDOW)   from simulator .npy files

    The decimation uses scipy resample (polyphase filter bank internally).
    After decimation the signal covers exactly SCAN_RATE / 2 = 2.5 MHz
    on each side of DC — the full 5 MHz GSM scan band.
    """
    from scipy.signal import resample_poly

    if iq.dtype == np.int8 and iq.ndim == 2 and iq.shape[0] == 2:
        iq = (iq[0].astype(np.float32) +
              1j * iq[1].astype(np.float32)).astype(np.complex64)

    iq = np.asarray(iq, dtype=np.complex64).reshape(-1)

    if len(iq) < N_WINDOW:
        # Zero-pad short captures up to the expected window length first,
        # so the up/down ratio below still applies cleanly.
        padded            = np.zeros(N_WINDOW, dtype=np.complex64)
        padded[:len(iq)]  = iq
        iq = padded

    if len(iq) == N_DEC:
        return iq

    # IMPORTANT: scipy.signal.resample (FFT-based) was used previously and
    # is the wrong tool for this ratio. TARGET_RATE/SCAN_RATE = 104e6/5e6 =
    # 20.8 is NOT an integer, so resample() truncates/pads in the frequency
    # domain at a boundary that does not align with the true bandwidth edge.
    # This introduces frequency-dependent phase distortion and edge artifacts
    # (the input window is a non-periodic burst, but FFT resampling assumes
    # periodicity), which was scattering carrier energy across slot bins
    # almost at random -- confirmed via diagnose_slot_alignment.py showing
    # 96.7% slot mismatches with no consistent offset.
    #
    # resample_poly performs proper polyphase decimation with an anti-alias
    # FIR filter at the *exact* rational ratio (up=5, down=104, since
    # SCAN_RATE/TARGET_RATE = 5/104 exactly). This produces the same output
    # length (3151 for N_WINDOW=65536) without the FFT-truncation artifacts.
    return resample_poly(iq, up=_RESAMPLE_UP, down=_RESAMPLE_DOWN).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────
#  STEP 2 — MATCHED FILTER BANK (full window + all segments)
# ─────────────────────────────────────────────────────────────────────

def _matched_filter_bank(signal: np.ndarray) -> np.ndarray:
    """
    Correlate signal with reference sinusoids for all 25 slots.

    corr_k = (conj(_REF[k]) · signal) / N_DEC

    Returns
    -------
    (25,) complex64 — one complex correlation value per slot
    """
    return (_REF.conj() @ signal) / N_DEC


def _segment_correlations(signal: np.ndarray) -> np.ndarray:
    """
    Compute per-segment matched-filter correlations for all slots at once.

    Vectorized: processes all 25 slots and all 5 segments in one shot
    rather than looping over (slot, segment) pairs.

    Returns
    -------
    (N_SEGMENTS, 25) complex64 — corr[seg, slot]
    """
    seg_corrs = np.empty((N_SEGMENTS, N_SLOTS), dtype=np.complex64)
    for i, (lo, hi) in enumerate(_SEG_BOUNDARIES):
        seg_len        = hi - lo
        seg            = signal[lo:hi]                      # (seg_len,)
        ref_seg        = _REF_SEGS[i]                       # (25, seg_len)
        # (25, seg_len) @ (seg_len,) → (25,)
        seg_corrs[i]   = (ref_seg.conj() @ seg) / seg_len
    return seg_corrs   # (N_SEGMENTS, 25)


# ─────────────────────────────────────────────────────────────────────
#  STEP 3 — VECTORIZED FEATURE COMPUTATION
# ─────────────────────────────────────────────────────────────────────

def _compute_all_features(
    slot_energies: np.ndarray,   # (25,) float32
    seg_corrs    : np.ndarray,   # (N_SEGMENTS, 25) complex64
) -> np.ndarray:
    """
    Compute all 5 features for all 25 slots in one vectorized pass.

    No Python loop over slots — fully NumPy.

    Parameters
    ----------
    slot_energies : (25,) float32  — |corr_full|² per slot
    seg_corrs     : (N_SEGMENTS, 25) complex64

    Returns
    -------
    (25, 5) float32  — features[k, :] = [F1, F2, F3, F4, F5] for slot k
    """

    # ── F1: Carrier energy ────────────────────────────────────────────
    # Normalized by the max across all slots.
    # Range: [0, 1]. The strongest carrier always scores 1.0.
    max_e = float(np.max(slot_energies)) + EPSILON
    f1    = slot_energies / max_e                          # (25,)

    # ── F2: Temporal stability ────────────────────────────────────────
    # seg_energies[seg, k] = |seg_corrs[seg, k]|²
    seg_energies = np.abs(seg_corrs) ** 2                  # (N_SEGMENTS, 25)
    mean_e = np.mean(seg_energies, axis=0) + EPSILON       # (25,)
    std_e  = np.std(seg_energies,  axis=0)                 # (25,)
    # CV = std/mean. Low CV → stable carrier. Clamp to [0, 1].
    f2 = np.clip(1.0 - std_e / mean_e, 0.0, 1.0)          # (25,)

    # ── F3: Phase coherence ───────────────────────────────────────────
    # Normalize each segment's correlation to unit magnitude, then
    # measure the vector mean magnitude across segments.
    # Range: [0, 1]. 1.0 = perfectly coherent phase across time.
    seg_mag            = np.abs(seg_corrs) + EPSILON       # (N_SEGMENTS, 25)
    unit_phase         = seg_corrs / seg_mag               # (N_SEGMENTS, 25)
    mean_phase_vec     = np.mean(unit_phase, axis=0)       # (25,)
    f3 = np.abs(mean_phase_vec).astype(np.float32)         # (25,)

    # ── F4: Cross-slot ratio (log-compressed) ─────────────────────────
    # energy_k vs. mean of immediate neighbours.
    # Using log1p to compress the range: a 10× peak → log1p(10) ≈ 2.4,
    # a 100× peak → log1p(100) ≈ 4.6. Soft-clipped at _LOG_CLIP.
    left_e  = np.concatenate([[0.0],        slot_energies[:-1]])  # (25,)
    right_e = np.concatenate([slot_energies[1:], [0.0]])          # (25,)
    neighbour_mean = (left_e + right_e) / 2.0 + EPSILON
    f4 = np.clip(
        np.log1p(slot_energies / neighbour_mean),
        0.0, _LOG_CLIP,
    ).astype(np.float32)                                    # (25,)

    # ── F5: Noise-floor ratio (log-compressed) ─────────────────────────
    # Energy relative to the median across all slots (CFAR-style).
    # Log-compressed for the same reason as F4.
    noise_floor = float(np.median(slot_energies)) + EPSILON
    f5 = np.clip(
        np.log1p(slot_energies / noise_floor),
        0.0, _LOG_CLIP,
    ).astype(np.float32)                                    # (25,)

    # feature_extraction.py — add a 6th feature, absolute energy vs fixed noise floor
    NOISE_FLOOR_REF = 1e-6  # calibrate from known-empty captures, not per-window

    f6 = np.clip(
    np.log1p(slot_energies / NOISE_FLOOR_REF),
    0.0, _LOG_CLIP,
    ).astype(np.float32)

    # Stack: (25, 6) → (25, 5) float32  
    return np.stack([f1, f2, f3, f4, f5, f6], axis=1).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────
#  MAIN EXTRACTION FUNCTION
# ─────────────────────────────────────────────────────────────────────

def extract_spectral_features(
    iq    : np.ndarray,
    config: FeatureExtractionConfig | None = None,
) -> np.ndarray:
    """
    Extract 125 matched-filter features from one IQ window.

    Steps
    -----
    1. Decimate to 5 MS/s              →  (3151,) complex64
    2. Full-window matched filter bank →  (25,)   complex64
    3. Per-segment matched filters     →  (N_SEGMENTS, 25) complex64
    4. Vectorized feature computation  →  (25, 5) float32
    5. Flatten                         →  (125,)  float32
    6. NaN/Inf guard

    Normalization
    -------------
    Per-sample global z-score has been removed (see module docstring).
    Use BatchNorm1d(125) as the first layer of your MLP.
    The `config.normalize` flag is accepted but ignored.

    Parameters
    ----------
    iq : (2, 65536) int8  or  (N,) complex64
    config : optional extraction settings (normalize flag is no-op)

    Returns
    -------
    (125,) float32  --  5 features × 25 slots, concatenated slot-major:
        [F1_0, F2_0, F3_0, F4_0, F5_0,  F1_1, ..., F5_24]
    """
    signal = _decimate(iq)                              # (3151,) complex64

    # Full-window correlation → per-slot energy
    corr_full     = _matched_filter_bank(signal)        # (25,) complex64
    slot_energies = np.abs(corr_full).astype(np.float32) ** 2  # (25,) float32

    # Per-segment correlations for F2 and F3
    seg_corrs = _segment_correlations(signal)           # (N_SEGMENTS, 25) complex64

    # All features in one vectorized call
    all_features = _compute_all_features(slot_energies, seg_corrs)  # (25, 5)

    features = all_features.reshape(-1)                 # (125,) float32

    if features.shape[0] != FEATURE_DIM:
        raise RuntimeError(
            f"Expected {FEATURE_DIM} features, got {features.shape[0]}"
        )

    return np.nan_to_num(
        features, nan=0.0, posinf=0.0, neginf=0.0,
    ).astype(np.float32)