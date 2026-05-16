"""FFT band-energy occupancy baseline.

Computes per-200-kHz-channel energy via averaged-periodogram (Welch-like)
and thresholds against the per-window noise-floor estimate.
"""
from __future__ import annotations

import numpy as np

from ..signals.gsm_carrier import GSM_CHANNEL_SPACING


def channel_energies(iq: np.ndarray, fs: float, n_channels: int, nfft: int = 1024) -> np.ndarray:
    """Return shape-(n_channels,) array of band energies."""
    # Welch averaging with 50% overlap, Hann window.
    win = np.hanning(nfft)
    hop = nfft // 2
    if iq.size < nfft:
        iq = np.pad(iq, (0, nfft - iq.size))
    frames = 1 + (iq.size - nfft) // hop
    psd_acc = np.zeros(nfft)
    for k in range(frames):
        seg = iq[k * hop : k * hop + nfft] * win
        spec = np.fft.fftshift(np.fft.fft(seg, nfft))
        psd_acc += (np.abs(spec) ** 2)
    psd = psd_acc / frames
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / fs))
    centers = (np.arange(n_channels) - (n_channels - 1) / 2.0) * GSM_CHANNEL_SPACING
    half = GSM_CHANNEL_SPACING / 2.0
    energies = np.zeros(n_channels)
    for i, c in enumerate(centers):
        mask = (freqs >= c - half) & (freqs < c + half)
        energies[i] = psd[mask].sum()
    return energies


def detect_occupancy(iq: np.ndarray, fs: float, n_channels: int, threshold_db: float = 6.0) -> np.ndarray:
    """Per-channel occupancy decision: energy > median + threshold_db."""
    e = channel_energies(iq, fs, n_channels)
    e_db = 10 * np.log10(e + 1e-12)
    noise_floor = np.median(e_db)
    return e_db > (noise_floor + threshold_db)
