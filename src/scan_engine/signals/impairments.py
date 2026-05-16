"""Channel/receiver impairments applied to complex baseband samples."""
from __future__ import annotations

import numpy as np


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng if rng is not None else np.random.default_rng()
    sig_power = float(np.mean(np.abs(x) ** 2)) + 1e-12
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power / 2.0)
    noise = (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)) * sigma
    return (x + noise).astype(np.complex64)


def apply_cfo(x: np.ndarray, fs: float, cfo_hz: float) -> np.ndarray:
    t = np.arange(x.size) / fs
    return (x * np.exp(1j * 2 * np.pi * cfo_hz * t)).astype(np.complex64)
