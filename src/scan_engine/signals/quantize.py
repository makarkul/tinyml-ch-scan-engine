"""Low-bit I/Q quantization. Reference: 5-bit I, 5-bit Q."""
from __future__ import annotations

import numpy as np


def quantize_iq(x: np.ndarray, bits: int = 5, full_scale: float | None = None) -> np.ndarray:
    """Mid-rise uniform quantization of I and Q to `bits` bits each.

    full_scale defaults to the per-window 99.5th percentile so headroom
    behaves like an AGC-tuned receiver. Returns complex64 in the
    original scale (not packed ints) so downstream code stays simple.
    """
    if full_scale is None:
        mag = np.maximum(np.abs(x.real), np.abs(x.imag))
        full_scale = float(np.quantile(mag, 0.995)) if mag.size else 1.0
        full_scale = max(full_scale, 1e-9)
    levels = 2 ** (bits - 1)
    step = full_scale / levels
    i = np.clip(np.round(x.real / step), -levels, levels - 1) * step
    q = np.clip(np.round(x.imag / step), -levels, levels - 1) * step
    return (i + 1j * q).astype(np.complex64)
