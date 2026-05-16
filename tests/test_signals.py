import numpy as np

from scan_engine.signals.gsm_carrier import gmsk_carrier
from scan_engine.signals.quantize import quantize_iq


def test_gmsk_carrier_is_constant_envelope():
    x = gmsk_carrier(n_samples=4096, fs=10_400_000, f_offset=0.0,
                     rng=np.random.default_rng(0))
    assert x.shape == (4096,)
    mag = np.abs(x)
    # GMSK is constant-envelope. Filter-edge transients aside, std should be tiny.
    assert mag.std() / mag.mean() < 0.05


def test_quantize_levels():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(8192) + 1j * rng.standard_normal(8192)).astype(np.complex64)
    q = quantize_iq(x, bits=5)
    # 5 bits per axis -> at most 2^5 = 32 distinct real values.
    assert len(np.unique(q.real)) <= 32
    assert len(np.unique(q.imag)) <= 32
