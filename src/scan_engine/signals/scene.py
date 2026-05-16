"""Compose multi-carrier scenes for a scan span."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gsm_carrier import GSM_CHANNEL_SPACING, gmsk_carrier
from .impairments import add_awgn, apply_cfo
from .quantize import quantize_iq


@dataclass
class CarrierSpec:
    channel_index: int  # 0..n_channels-1
    snr_db: float
    cfo_hz: float = 0.0


@dataclass
class SceneSpec:
    n_samples: int
    fs: float
    scan_bw: float
    n_channels: int
    carriers: list[CarrierSpec]
    quant_bits: int = 5
    seed: int = 0

    def channel_centers(self) -> np.ndarray:
        spacing = GSM_CHANNEL_SPACING
        start = -((self.n_channels - 1) / 2.0) * spacing
        return start + np.arange(self.n_channels) * spacing


def render_scene(spec: SceneSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return (iq_samples_complex64, occupancy_bool_array_of_len_n_channels)."""
    rng = np.random.default_rng(spec.seed)
    centers = spec.channel_centers()
    occupancy = np.zeros(spec.n_channels, dtype=bool)
    composite = np.zeros(spec.n_samples, dtype=np.complex64)
    for c in spec.carriers:
        occupancy[c.channel_index] = True
        f0 = centers[c.channel_index] + c.cfo_hz
        carrier = gmsk_carrier(spec.n_samples, spec.fs, f_offset=f0, rng=rng)
        # Scale to encode per-carrier SNR before adding noise once at the end.
        scale = 10.0 ** (c.snr_db / 20.0)
        composite = composite + (scale * carrier).astype(np.complex64)
    # Noise reference: unit power, so per-carrier amplitude already encodes SNR.
    noise = (rng.standard_normal(spec.n_samples) + 1j * rng.standard_normal(spec.n_samples)) / np.sqrt(2)
    iq = (composite + noise).astype(np.complex64)
    iq = quantize_iq(iq, bits=spec.quant_bits)
    return iq, occupancy
