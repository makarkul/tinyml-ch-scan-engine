"""M2 sanity gates (per docs/PLAN.md):
- Zero-carrier scene -> baseline FAR near zero.
- High-SNR single-carrier scene -> baseline detects the occupied channel.
"""
import numpy as np

from scan_engine.baselines.fft_energy import detect_occupancy
from scan_engine.signals.scene import CarrierSpec, SceneSpec, render_scene


FS = 10_400_000.0
N_CHANNELS = 25
N_SAMPLES = 4096


def _spec(carriers, seed=0):
    return SceneSpec(
        n_samples=N_SAMPLES, fs=FS, scan_bw=5_000_000.0,
        n_channels=N_CHANNELS, carriers=carriers, quant_bits=5, seed=seed,
    )


def test_zero_carrier_low_far():
    fa = 0
    trials = 16
    for s in range(trials):
        iq, occ = render_scene(_spec([], seed=s))
        pred = detect_occupancy(iq, fs=FS, n_channels=N_CHANNELS, threshold_db=6.0)
        assert occ.sum() == 0
        fa += int(pred.sum())
    # Expect well under 1 false alarm per scene on average.
    assert fa / trials < 1.0, f"FAR too high: {fa}/{trials} false channels"


def test_single_strong_carrier_detected():
    target_ch = 12
    iq, occ = render_scene(_spec([CarrierSpec(channel_index=target_ch, snr_db=20.0)], seed=1))
    pred = detect_occupancy(iq, fs=FS, n_channels=N_CHANNELS, threshold_db=6.0)
    assert pred[target_ch], "strong on-channel carrier should be detected"
