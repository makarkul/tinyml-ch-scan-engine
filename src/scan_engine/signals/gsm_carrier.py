"""GMSK-like GSM carrier generator.

Produces complex baseband samples for a single GSM-like carrier at a
chosen center frequency offset. Uses a Gaussian-filtered FSK
approximation of GMSK (BT≈0.3, 270.833 kHz symbol rate). Continuous
modulation only — burst structure is out of scope for occupancy
detection.
"""
from __future__ import annotations

import numpy as np

GSM_SYMBOL_RATE = 270_833.0  # Hz
GSM_CHANNEL_SPACING = 200_000.0  # Hz
GSM_BT = 0.3


def _gaussian_pulse(sps: int, bt: float = GSM_BT, span: int = 4) -> np.ndarray:
    n = np.arange(-span * sps, span * sps + 1)
    t = n / sps
    alpha = np.sqrt(np.log(2) / 2) / bt
    h = np.exp(-((t / alpha) ** 2) * np.pi**2 / np.log(2)) / (alpha * np.sqrt(np.pi / np.log(2)))
    h = h / h.sum()
    return h


def gmsk_carrier(
    n_samples: int,
    fs: float,
    f_offset: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return n_samples of complex baseband GMSK at f_offset Hz from DC."""
    rng = rng if rng is not None else np.random.default_rng()
    sps = max(2, int(round(fs / GSM_SYMBOL_RATE)))
    n_symbols = int(np.ceil(n_samples / sps)) + 16
    bits = rng.integers(0, 2, size=n_symbols) * 2 - 1  # ±1
    upsampled = np.zeros(n_symbols * sps)
    upsampled[::sps] = bits
    h = _gaussian_pulse(sps)
    shaped = np.convolve(upsampled, h, mode="same")
    # MSK-style integration: phase = pi/2 * cumulative shaped symbols / sps
    phase = (np.pi / 2.0) * np.cumsum(shaped) / sps
    iq = np.exp(1j * phase).astype(np.complex64)
    iq = iq[:n_samples]
    if f_offset != 0.0:
        t = np.arange(n_samples) / fs
        iq = iq * np.exp(1j * 2 * np.pi * f_offset * t).astype(np.complex64)
    return iq
