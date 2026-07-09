# ══════════════════════════════════════════════════════════════════════
#  CONFIG  ←  change these numbers, nothing else needs to touch
# ══════════════════════════════════════════════════════════════════════
N_SAMPLES   = 1_000     # total samples to generate
OUT_DIR     = "./gsm_dataset"
SEED        = 42        # master random seed — same seed = same dataset
START_IDX   = 0         # set > 0 to resume an interrupted run
# ══════════════════════════════════════════════════════════════════════

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from scipy.signal import resample as fft_resample
from scipy.ndimage import gaussian_filter1d

# ─────────────────────────────────────────────────────────────────────
#  SECTION 1 — GSM PHYSICAL-LAYER CONSTANTS
# ─────────────────────────────────────────────────────────────────────

NATIVE_RATE   = 1_083_333        # Sa/s  — native GSM baseband rate (4 SPS)
SYMB_RATE     = NATIVE_RATE / 4.0  # 270,833 baud
SPS           = 4.0              # samples per symbol at native rate
FRAME_SAMPLES = int(round(8 * 156.25 / SYMB_RATE * NATIVE_RATE))  # 5000

# SCH 64-bit sync sequence — TS 45.002 Table 5
SCH_SYNC64 = np.array([
    1,0,1,1,1,0,0,1,0,1,1,0,0,0,1,0,
    0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,
    0,0,1,0,1,1,0,1,0,1,0,0,0,1,0,1,
    0,1,1,1,0,1,1,0,0,0,0,1,1,0,1,1,
], dtype=np.int8)

# GSM dummy-burst 142-bit payload — TS 45.002
DUMMY_142 = np.array([
    1,1,1,1,1,0,1,1,0,1,1,1,0,1,1,0,0,0,0,0,1,0,1,0,0,1,0,0,1,1,1,0,
    0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,
    0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,0,1,0,1,0,0,1,0,0,1,1,1,
    0,0,1,1,0,0,1,1,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,
], dtype=np.int8)

DUMMY_148 = np.concatenate([
    np.zeros(3, dtype=np.int8), DUMMY_142, np.zeros(3, dtype=np.int8)
])

# Training sequences (TSC) — TS 45.002 Table 3
_TSC = np.array([
    [0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1],
    [0,0,1,0,1,1,0,1,1,1,0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,1],
    [0,1,0,0,0,0,1,1,1,0,1,1,1,0,1,0,0,1,0,0,0,0,1,1,1,0],
    [0,1,0,0,0,1,1,1,1,0,1,1,0,1,0,0,0,1,0,0,0,1,1,1,1,0],
    [0,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,0,0,0,1,1,0,1,0,1,1],
    [0,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,0,1,0,0,1,1,1,0,1,0],
    [1,0,1,0,0,1,1,1,1,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,1,1],
    [1,1,1,0,1,1,1,1,0,0,0,1,0,0,1,0,1,1,1,0,1,1,1,1,0,0],
], dtype=np.int8)

# 51-multiframe layout — TS 45.002 Table 3
FCCH_FRAMES     = {0, 10, 20, 30, 40}
SCH_FRAMES      = {1, 11, 21, 31, 41}
BCCH_FRAMES     = {2, 3, 4, 5}
CCCH_BLOCKS_51  = [
    {6,7,8,9}, {12,13,14,15}, {16,17,18,19},
    {22,23,24,25}, {26,27,28,29}, {32,33,34,35},
    {36,37,38,39}, {42,43,44,45}, {46,47,48,49},
]
CCCH_FRAMES_ALL = set().union(*CCCH_BLOCKS_51)

# SI3 system information — 23-byte L2
SI3_L2_BYTES = [
    0x49, 0x06, 0x1B, 0x00, 0x01, 0x00, 0xF1, 0x10,
    0x00, 0x01, 0x1C, 0x00, 0x05, 0x08, 0x00, 0x48,
    0xC0, 0x01, 0x08, 0x00, 0x00, 0x00, 0x2B,
]

# Paging L2 — 23 bytes
PAGING_L2_BYTES = [
    0x31, 0x06, 0x21, 0x20, 0x08, 0x29, 0x26, 0x20,
    0x99, 0x05, 0x70, 0x26, 0x71, 0x8B, 0x2B, 0x2B,
    0x2B, 0x2B, 0x2B, 0x2B, 0x2B, 0x2B, 0x2B,
]


# ─────────────────────────────────────────────────────────────────────
#  SECTION 2 — GSM ENCODING  (no external imports)
# ─────────────────────────────────────────────────────────────────────

def _conv_encode(bits):
    """
    Convolutional encoder R=1/2, K=5.
    Polynomials: G0 = 1+D^3+D^4 (0b10011), G1 = 1+D+D^3+D^4 (0b11011).
    Each input bit produces two output bits.
    """
    reg = 0   # 4-bit shift register
    out = []
    for b in bits:
        b = int(b) & 1
        g0 = b ^ ((reg >> 2) & 1) ^ ((reg >> 3) & 1)
        g1 = b ^ ((reg >> 0) & 1) ^ ((reg >> 2) & 1) ^ ((reg >> 3) & 1)
        out.append(g0 & 1)
        out.append(g1 & 1)
        reg = ((reg >> 1) | (b << 3)) & 0xF
    return out


def _fire_encode_228(info184):
    """
    GSM control-channel coding: 184-bit L2 → 228-bit pre-convolution word.
    Fire code (40-bit CRC) + 4 tail bits.
    Generator: G(x) = (x^23 + 1)(x^17 + x^3 + 1) = degree-40 polynomial.
    """
    FIRE_POLY = 0x10_0000_0901   # degree-40, matches 3GPP TS 05.03
    crc = 0
    for b in info184:
        bit = (int(b) ^ ((crc >> 39) & 1)) & 1
        crc = ((crc << 1) & 0xFF_FFFF_FFFF) ^ (FIRE_POLY if bit else 0)
    # Append inverted CRC (40 bits) + 4 tail bits
    crc_bits = [((crc >> (39 - i)) & 1) ^ 1 for i in range(40)]
    return list(info184) + crc_bits + [0, 0, 0, 0]   # 184+40+4 = 228


def _encode_control_channel(l2_bytes):
    """
    23-byte L2 → four 114-bit burst payloads (GSM TS 05.03 control coding).
    Returns list of 4 numpy int8 arrays, each 114 elements.
    """
    # Pack bytes LSB-first into 184 bits
    info184 = []
    for byte_val in l2_bytes:
        v = int(byte_val) & 0xFF
        for k in range(8):
            info184.append((v >> k) & 1)

    pre228  = _fire_encode_228(info184)       # 228 bits
    coded456 = _conv_encode(pre228)            # 456 bits
    assert len(coded456) == 456

    payloads = [np.zeros(114, dtype=np.int8) for _ in range(4)]
    for k in range(456):
        b = k % 4
        j = 2 * ((49 * k) % 57) + ((k // 4) % 2)
        payloads[b][j] = coded456[k]
    return payloads


def _map_normal_burst(left57, right57, tsc_idx=1):
    """
    Build a 148-bit GSM normal burst.
    Structure: 3 tail | 57 data | 1 steal | 26 TSC | 1 steal | 57 data | 3 tail
    """
    tsc = _TSC[tsc_idx % 8]
    burst = np.concatenate([
        np.zeros(3,   dtype=np.int8),
        np.asarray(left57,  dtype=np.int8),
        np.zeros(1,   dtype=np.int8),   # stealing flag
        tsc,
        np.zeros(1,   dtype=np.int8),   # stealing flag
        np.asarray(right57, dtype=np.int8),
        np.zeros(3,   dtype=np.int8),
    ])
    assert len(burst) == 148
    return burst


def _map_sync_burst(e1_39, e2_39):
    """
    Build a 148-bit GSM sync burst.
    Structure: 3 tail | 39 bits | 64 sync seq | 39 bits | 3 tail
    """
    burst = np.concatenate([
        np.zeros(3,  dtype=np.int8),
        np.asarray(e1_39, dtype=np.int8),
        SCH_SYNC64,
        np.asarray(e2_39, dtype=np.int8),
        np.zeros(3,  dtype=np.int8),
    ])
    assert len(burst) == 148
    return burst


def _make_burst_from_payload(payload_114, tsc_idx=1):
    return _map_normal_burst(payload_114[:57], payload_114[57:], tsc_idx)


def _make_bcch_bursts(tsc_idx=1):
    payloads = _encode_control_channel(SI3_L2_BYTES)
    return [_make_burst_from_payload(p, tsc_idx) for p in payloads]


def _make_ccch_bursts(tsc_idx=1):
    payloads = _encode_control_channel(PAGING_L2_BYTES)
    return [_make_burst_from_payload(p, tsc_idx) for p in payloads]


def _sch_info39(fn, bsic):
    """Pack frame number + BSIC into 25-bit SCH info, compute CRC10, return 39 bits."""
    bsic &= 0x3F
    t1 = fn // (26 * 51)
    t2 = fn % 26
    t3 = fn % 51
    t3p = t3 // 10
    b0 = ((bsic & 0x3F) << 2) | ((t1 >> 9) & 0x03)
    b1 = (t1 >> 1) & 0xFF
    b2 = ((t1 & 1) << 7) | ((t2 & 0x1F) << 2) | ((t3p >> 1) & 0x03)
    b3 = (t3p & 1) << 7
    packed = np.array([b0, b1, b2, b3], dtype=np.uint8)
    info25 = np.array([(int(packed[i//8]) >> (7-(i%8))) & 1 for i in range(25)], dtype=np.int8)
    # CRC10 with polynomial 0x175
    reg = 0
    for bit in info25:
        fb = ((reg >> 9) & 1) ^ int(bit)
        reg = (reg << 1) & 0x3FF
        if fb:
            reg ^= 0x175
    reg ^= 0x3FF
    crc10 = np.array([(reg >> (9-i)) & 1 for i in range(10)], dtype=np.int8)
    # Conv-encode the 39-bit (25+10+4_tail) sequence
    raw39 = np.concatenate([info25, crc10, np.zeros(4, dtype=np.int8)])
    coded78 = np.array(_conv_encode(raw39.tolist()), dtype=np.int8)
    return coded78[:39], coded78[39:]


# ─────────────────────────────────────────────────────────────────────
#  SECTION 3 — GMSK MODULATOR  (no external imports)
# ─────────────────────────────────────────────────────────────────────

def _gmsk_mod(bits, phase_start=0.0, sps=4, BT=0.3):
    """
    GMSK modulator: bits → complex64 IQ at sps samples/symbol.

    Steps:
      1. NRZ encode: bit 0 → -1,  bit 1 → +1
      2. Upsample by sps (repeat each symbol)
      3. Apply Gaussian filter (bandwidth-time product BT=0.3)
      4. Integrate → continuous phase
      5. Output exp(j * phase)

    Returns (iq_array, final_phase).
    """
    bits   = np.asarray(bits, dtype=np.float64)
    nrz    = 2.0 * bits - 1.0
    up     = np.repeat(nrz, int(sps))
    sigma  = sps / (2.0 * np.pi * BT)
    filt   = gaussian_filter1d(up, sigma=sigma, mode='nearest')
    phase  = phase_start + np.cumsum((np.pi / 2.0 / sps) * filt)
    iq     = np.exp(1j * phase).astype(np.complex64)
    return iq, float(phase[-1])


def _build_frame(ts0_bits, phase_in, ts1_bits=None):
    """
    Modulate one 8-timeslot GSM frame at NATIVE_RATE.
    ts0_bits: 148-bit array for timeslot 0
    ts1_bits: 148-bit array for timeslot 1 (None → dummy burst)
    Returns (frame_complex64, phase_out).
    """
    slot_samples = 156.25 * SPS
    frame = np.zeros(FRAME_SAMPLES, dtype=np.complex64)
    phase = phase_in

    for ts in range(8):
        if   ts == 0:                          bits = ts0_bits
        elif ts == 1 and ts1_bits is not None: bits = ts1_bits
        else:                                  bits = DUMMY_148

        iq_ts, phase = _gmsk_mod(bits, phase_start=phase, sps=SPS)
        start = int(round(ts * slot_samples))
        end   = min(start + len(iq_ts), FRAME_SAMPLES)
        frame[start:end] = iq_ts[:(end - start)]

    return frame, phase


# ─────────────────────────────────────────────────────────────────────
#  SECTION 4 — ITU CHANNEL PROFILES  (no external imports)
# ─────────────────────────────────────────────────────────────────────

# ITU/GSM 05.05 channel profiles
_CHANNEL_PROFILES = {
    "TU3":   {"name": "Typical Urban 3 km/h",     "delays_us": [0, 0.2, 0.5, 1.6, 2.3, 5.0],   "powers_db": [0, -1, -2, -3, -5, -7],  "velocity": 3},
    "TU50":  {"name": "Typical Urban 50 km/h",    "delays_us": [0, 0.2, 0.5, 1.6, 2.3, 5.0],   "powers_db": [0, -1, -2, -3, -5, -7],  "velocity": 50},
    "RA130": {"name": "Rural Area 130 km/h",       "delays_us": [0, 0.1, 0.2, 0.3, 0.5, 0.7],   "powers_db": [0, -2, -4, -6, -8, -10], "velocity": 130},
    "RA250": {"name": "Rural Area 250 km/h",       "delays_us": [0, 0.1, 0.2, 0.3, 0.5, 0.7],   "powers_db": [0, -2, -4, -6, -8, -10], "velocity": 250},
    "HT100": {"name": "Hilly Terrain 100 km/h",    "delays_us": [0, 0.2, 0.4, 0.6, 15.0, 17.2], "powers_db": [0, -1, -2, -3, -4, -5],  "velocity": 100},
    "HT200": {"name": "Hilly Terrain 200 km/h",    "delays_us": [0, 0.2, 0.4, 0.6, 15.0, 17.2], "powers_db": [0, -1, -2, -3, -4, -5],  "velocity": 200},
}

_SAMPLE_PERIOD_US = (1.0 / SYMB_RATE) * 1e6 / SPS   # µs per sample at native rate


def _build_static_channel(profile_name, seed=1):
    """
    Build a static Rayleigh TDL channel tap vector from an ITU profile.
    Returns complex64 array h of length max(delays)+1.
    """
    prof = _CHANNEL_PROFILES[profile_name]
    rng  = np.random.default_rng(seed)
    delays_samp = [int(round(d / _SAMPLE_PERIOD_US)) for d in prof["delays_us"]]
    n_taps = max(delays_samp) + 1
    h = np.zeros(n_taps, dtype=np.complex64)
    for d, p_db in zip(delays_samp, prof["powers_db"]):
        amp = float(10.0 ** (p_db / 20.0))
        tap = amp * (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2.0)
        h[d] += np.complex64(tap)
    return h


def _apply_static_channel(wave, h):
    """Convolve wave with channel tap vector h."""
    out = fftconvolve(wave, h)[:len(wave)]
    return out.astype(np.complex64)


def _build_timevarying_channel(n_samples, profile_name, velocity_kmh,
                                fc_hz=935e6, seed=1):
    """
    Build a time-varying Rayleigh TDL channel matrix H of shape (n_samples, n_taps).
    Each row is the channel impulse response at that sample instant.
    Uses Jake's model (sum-of-sinusoids) for Doppler.
    """
    prof = _CHANNEL_PROFILES[profile_name]
    rng  = np.random.default_rng(seed)
    delays_samp = [int(round(d / _SAMPLE_PERIOD_US)) for d in prof["delays_us"]]
    n_taps = max(delays_samp) + 1

    fd_hz  = (velocity_kmh / 3.6) * (fc_hz / 3.0e8)
    fd_norm = fd_hz / NATIVE_RATE  # normalised Doppler

    H = np.zeros((n_samples, n_taps), dtype=np.complex64)
    t = np.arange(n_samples, dtype=np.float64)

    N_SINUSOIDS = 16  # Jake's model approximation
    for d, p_db in zip(delays_samp, prof["powers_db"]):
        amp = float(10.0 ** (p_db / 20.0))
        # Sum of N_SINUSOIDS complex sinusoids at random Doppler frequencies
        angles = rng.uniform(0, 2 * np.pi, N_SINUSOIDS)
        freqs  = fd_norm * np.cos(rng.uniform(0, 2 * np.pi, N_SINUSOIDS))
        phases = rng.uniform(0, 2 * np.pi, N_SINUSOIDS)
        tap_t  = (amp / np.sqrt(N_SINUSOIDS)) * np.sum(
            np.exp(1j * (2 * np.pi * freqs[:, None] * t[None, :] + phases[:, None])),
            axis=0
        )
        H[:, d] += tap_t.astype(np.complex64)
    return H


def _apply_timevarying_channel(wave, H):
    """Apply time-varying channel H (shape n_samples × n_taps) to wave."""
    n_samples, n_taps = H.shape
    out = np.zeros(n_samples, dtype=np.complex64)
    for tap_idx in range(n_taps):
        if tap_idx < n_samples:
            taps = H[:, tap_idx]
            delayed = np.concatenate([np.zeros(tap_idx, dtype=np.complex64),
                                       wave[:n_samples - tap_idx]])
            out += taps * delayed
    return out


# ─────────────────────────────────────────────────────────────────────
#  SECTION 5 — GSM WAVE BUILDER
# ─────────────────────────────────────────────────────────────────────

_BCCH_BURSTS_CACHE  = {}   # tsc_idx -> list of 4 burst arrays
_CCCH_BURSTS_CACHE  = {}   # tsc_idx -> list of 4 burst arrays


def _get_burst_cache(tsc_idx):
    """Pre-compute BCCH and CCCH burst bit patterns once per TSC value."""
    if tsc_idx not in _BCCH_BURSTS_CACHE:
        _BCCH_BURSTS_CACHE[tsc_idx] = _make_bcch_bursts(tsc_idx)
        _CCCH_BURSTS_CACHE[tsc_idx] = _make_ccch_bursts(tsc_idx)
    return _BCCH_BURSTS_CACHE[tsc_idx], _CCCH_BURSTS_CACHE[tsc_idx]


def generate_gsm_wave(
    n_frames     : int   = 2,
    bsic         : int   = 25,
    amp          : float = 0.8,
    channel      : str   = "NONE",
    channel_seed : int   = 1,
    velocity_kmh : float = 0.0,
    fc_hz        : float = 935.0e6,
) -> np.ndarray:
    """
    Generate a GSM baseband waveform at NATIVE_RATE (1,083,333 Sa/s).
    Returns complex64 array of length n_frames * FRAME_SAMPLES.

    Parameters
    ----------
    n_frames     : number of GSM frames (each = 5000 samples).
                   2 frames → 10,000 samples >> the 1,066 we actually need
                   (N_RESAMPLE=683 + max timing_offset=383).
    bsic         : Base Station Identity Code (0–63)
    amp          : target RMS amplitude
    channel      : ITU profile name or "NONE"
    channel_seed : seed for channel tap randomisation
    velocity_kmh : Doppler velocity (0 = static channel)
    fc_hz        : carrier frequency for Doppler calculation
    """
    tsc_idx = bsic & 7
    bcch_bursts, ccch_bursts = _get_burst_cache(tsc_idx)

    frames = []
    fn     = 0
    phase  = 0.0

    frames_left = n_frames
    mf = 0
    while frames_left > 0:
        for fr in range(51):
            if frames_left <= 0:
                break

            # ── TS0: 51-multiframe control channel ─────────────────
            if fr in FCCH_FRAMES:
                ts0 = np.zeros(148, dtype=np.int8)          # all-zero → +90°/sym
            elif fr in SCH_FRAMES:
                e1, e2 = _sch_info39(fn, bsic)
                ts0 = _map_sync_burst(e1, e2)
            elif fr in BCCH_FRAMES:
                ts0 = bcch_bursts[fr - 2]
            elif fr in CCCH_FRAMES_ALL:
                for blk_set in CCCH_BLOCKS_51:
                    if fr in blk_set:
                        ts0 = ccch_bursts[sorted(blk_set).index(fr)]
                        break
            else:
                ts0 = DUMMY_148

            frame, phase = _build_frame(ts0, phase)
            frames.append(frame)
            fn          += 1
            frames_left -= 1
        mf += 1

    wave = np.concatenate(frames).astype(np.complex64)

    # Scale to target amplitude
    pwr = float(np.mean(np.abs(wave) ** 2))
    if pwr > 1e-30:
        wave *= np.float32(np.sqrt(amp ** 2 / pwr))

    # Apply channel
    ch = str(channel).upper()
    if ch != "NONE" and ch in _CHANNEL_PROFILES:
        if velocity_kmh > 0.0:
            H    = _build_timevarying_channel(len(wave), ch, velocity_kmh,
                                               fc_hz=fc_hz, seed=channel_seed)
            wave = _apply_timevarying_channel(wave, H)
        else:
            h    = _build_static_channel(ch, seed=channel_seed)
            wave = _apply_static_channel(wave, h)

    return wave.astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────
#  SECTION 6 — DATASET CONSTANTS
# ─────────────────────────────────────────────────────────────────────

TARGET_RATE    = 104_000_000
RESAMPLE_RATIO = TARGET_RATE / NATIVE_RATE    # ≈ 96.0

N_SLOTS     = 25
SLOT_BW_HZ  = 200_000
SLOT_CENTRES_HZ = np.array(
    [-2_400_000 + k * SLOT_BW_HZ for k in range(N_SLOTS)],
    dtype=np.float64,
)

# Window: 65,536 samples at 104 MS/s ≈ 630 µs  (covers one GSM burst = 577 µs)
N_WINDOW = 65_536

# CRITICAL: the number of native-rate samples that, when FFT-resampled,
# produce exactly N_WINDOW samples at TARGET_RATE.
# Formula: round(N_WINDOW * NATIVE_RATE / TARGET_RATE) = round(65536 * 1083333 / 104000000) = 683
# This is the ONLY value that should be passed to fft_resample().
# Passing more samples stretches the duration and compresses all frequencies,
# causing every carrier to land in the wrong FFT bin.
N_RESAMPLE = int(round(N_WINDOW * NATIVE_RATE / TARGET_RATE))   # 683

# Native buffer size: generate enough samples to allow timing offset slicing.
# timing_offset is in [0, 383] samples. We need N_RESAMPLE samples AFTER
# the offset, so the buffer must be at least N_RESAMPLE + 383 samples.
N_NATIVE_BUFFER = N_RESAMPLE + 400   # ≈ 1083 — for slicing only, NOT for resampling

# 5-bit signed quantisation
QMAX, QMIN = 15, -16

# SNR grid
SNR_GRID_DB = np.arange(-10, 42, 2, dtype=float)   # 18 points

# Impairment flags (multi-label, 6 independent binary flags)
IMP_NAMES = ["cfo", "static_fading", "timevarying_fading", "cw_tone", "adjacent_channel", "wideband_blocker"]
N_IMP     = len(IMP_NAMES)  # 6

# Power class bins
PWR_EDGES  = [-np.inf, -10.0, 0.0, 10.0, np.inf]
PWR_LABELS = ["weak", "medium", "strong", "very_strong"]

# Channel profiles available for fading impairments
STATIC_PROFILES    = ["TU3", "RA130", "HT100"]
TIMEVARYING_PROFILES = [("TU50", 50), ("RA250", 130), ("HT200", 200)]


# ─────────────────────────────────────────────────────────────────────
#  SECTION 7 — DSP HELPERS
# ─────────────────────────────────────────────────────────────────────

def _resample_to_104mhz(iq: np.ndarray) -> np.ndarray:
    """FFT-resample native-rate IQ to TARGET_RATE."""
    return fft_resample(iq, N_WINDOW).astype(np.complex64)


def _freq_shift(iq: np.ndarray, freq_hz: float) -> np.ndarray:
    """Shift spectrum of iq by freq_hz at TARGET_RATE."""
    if abs(freq_hz) < 0.5:
        return iq
    n = np.arange(len(iq), dtype=np.float64)
    return (iq * np.exp(1j * 2 * np.pi * freq_hz * n / TARGET_RATE).astype(np.complex64)).astype(np.complex64)


def _set_power(iq: np.ndarray, target: float) -> np.ndarray:
    p = float(np.mean(np.abs(iq) ** 2))
    return iq if p < 1e-30 else (iq * np.sqrt(target / p)).astype(np.complex64)


def _add_awgn(iq: np.ndarray, snr_db: float, rng: np.random.Generator,
              ref_pwr: float | None = None) -> np.ndarray:
    """Add AWGN at Es/N0 = snr_db. SPS_104 = TARGET_RATE/SYMB_RATE ≈ 384."""
    sig_pwr = ref_pwr if ref_pwr is not None else float(np.mean(np.abs(iq) ** 2))
    sigma   = float(np.sqrt(sig_pwr * (TARGET_RATE / SYMB_RATE) / (10.0 ** (snr_db / 10.0)) / 2.0))
    noise   = (rng.normal(0, sigma, len(iq)) + 1j * rng.normal(0, sigma, len(iq))).astype(np.complex64)
    return (iq + noise).astype(np.complex64)

def _inject_cw_tone(iq, freq_hz, power_rel_db, rng):
    sig_pwr = float(np.mean(np.abs(iq) ** 2))
    amp     = float(np.sqrt(sig_pwr * 10.0 ** (power_rel_db / 10.0)))
    phase0  = rng.uniform(0, 2 * np.pi)
    n       = np.arange(len(iq), dtype=np.float64)
    tone    = (amp * np.exp(1j * (2 * np.pi * freq_hz * n / TARGET_RATE + phase0))).astype(np.complex64)
    return (iq + tone).astype(np.complex64)


def _inject_wideband_blocker(iq, power_rel_db, rng):
    """
    Inject structured wideband interference.

    Four types chosen randomly (Gap 2 fix — added type 3):
      0 - flat AWGN (original behaviour)
      1 - narrowband cluster: energy concentrated in a random
          200-500 kHz band, simulating LTE subcarriers or UMTS
      2 - shaped noise: Gaussian spectral envelope centred at a
          random offset with std 300-800 kHz
      3 - wide LTE/UMTS hump (NEW): broad Gaussian envelope
          1.5-3 MHz wide, centred near DC (±500 kHz), matching
          the dominant blocker type observed in OTA captures where
          a co-existing LTE carrier fills the centre of the scan band.
          Type 3 is drawn with 2× weight so it occurs ~40% of blocker
          events rather than the naive 25%, reflecting OTA prevalence.

    All types are scaled to the same total power so the SNR impact
    is consistent regardless of type.
    """
    sig_pwr    = float(np.mean(np.abs(iq) ** 2))
    target_pwr = sig_pwr * 10.0 ** (power_rel_db / 10.0)
    n          = len(iq)

    # Type 3 gets 2× weight: draw from [0,1,2,3,3] → P(3)=0.4
    itype = int(rng.choice([0, 1, 2, 3, 3]))

    if itype == 0:
        # Flat AWGN -- original behaviour
        sigma = float(np.sqrt(target_pwr / 2.0))
        wb    = (rng.normal(0, sigma, n) +
                 1j * rng.normal(0, sigma, n)).astype(np.complex64)

    elif itype == 1:
        # Narrowband cluster -- energy in a random 200-500 kHz band
        # Simulate LTE subcarrier group or UMTS channel
        bw_hz     = float(rng.uniform(200_000, 500_000))
        centre_hz = float(rng.uniform(-2_000_000, 2_000_000))
        freqs     = np.fft.fftfreq(n, d=1.0/104_000_000)
        mask      = np.abs(freqs - centre_hz) < bw_hz / 2
        spectrum  = np.zeros(n, dtype=np.complex128)
        n_bins    = mask.sum()
        if n_bins > 0:
            spectrum[mask] = (rng.normal(0, 1, n_bins) +
                              1j * rng.normal(0, 1, n_bins))
        wb      = np.fft.ifft(spectrum).astype(np.complex64)
        cur_pwr = float(np.mean(np.abs(wb) ** 2)) + 1e-30
        wb      = (wb * np.sqrt(target_pwr / cur_pwr)).astype(np.complex64)

    elif itype == 2:
        # Shaped noise -- Gaussian spectral envelope, moderate width
        # Simulate out-of-band emission rolling off with distance
        centre_hz = float(rng.uniform(-1_800_000, 1_800_000))
        std_hz    = float(rng.uniform(300_000, 800_000))
        freqs     = np.fft.fftfreq(n, d=1.0/104_000_000)
        envelope  = np.exp(-0.5 * ((freqs - centre_hz) / std_hz) ** 2)
        noise_f   = ((rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)) *
                     envelope)
        wb      = np.fft.ifft(noise_f).astype(np.complex64)
        cur_pwr = float(np.mean(np.abs(wb) ** 2)) + 1e-30
        wb      = (wb * np.sqrt(target_pwr / cur_pwr)).astype(np.complex64)

    else:
        # Wide LTE/UMTS hump (NEW — Gap 2 fix)
        # Broad Gaussian envelope 1.5-3 MHz wide, centred near DC.
        # This matches the dominant blocker morphology seen in OTA captures:
        # a smooth hump spanning most of the scan band with a peak near centre.
        # std_hz drawn from 600k-1.5 MHz → FWHM ≈ 1.4-3.5 MHz at -3dB points.
        centre_hz = float(rng.uniform(-500_000, 500_000))
        std_hz    = float(rng.uniform(600_000, 1_500_000))
        freqs     = np.fft.fftfreq(n, d=1.0/104_000_000)
        envelope  = np.exp(-0.5 * ((freqs - centre_hz) / std_hz) ** 2)
        noise_f   = ((rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)) *
                     envelope)
        wb      = np.fft.ifft(noise_f).astype(np.complex64)
        cur_pwr = float(np.mean(np.abs(wb) ** 2)) + 1e-30
        wb      = (wb * np.sqrt(target_pwr / cur_pwr)).astype(np.complex64)

    return (iq + wb).astype(np.complex64)

def _inject_adjacent_carrier(iq, blocker_slot, power_rel_db, rng, main_carrier_pwr):
    carrier = generate_gsm_wave(n_frames=2, bsic=int(rng.integers(0, 64)))
    carrier = carrier[:N_RESAMPLE]
    carrier = _resample_to_104mhz(carrier)
    carrier = _freq_shift(carrier, float(SLOT_CENTRES_HZ[blocker_slot]))
    carrier = _set_power(carrier, main_carrier_pwr * 10.0 ** (power_rel_db / 10.0))
    return (iq + carrier).astype(np.complex64)

def _apply_spectral_tilt(iq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply a linear spectral tilt to simulate receiver passband roll-off.

    Multiplies the signal spectrum by a linear amplitude ramp so that
    one edge of the band is boosted and the other is attenuated.  The
    total end-to-end power variation is drawn from 0–6 dB uniformly,
    and the direction (rising or falling across the band) is randomised.

    Implemented in the frequency domain so the effect is purely spectral
    (no time-domain ringing or edge artefacts from a time-domain window).

    Parameters
    ----------
    iq  : (N,) complex64 — full-rate IQ at TARGET_RATE (104 MS/s)
    rng : numpy Generator

    Returns
    -------
    (N,) complex64 — IQ with spectral tilt applied, same total length.
    """
    n             = len(iq)
    tilt_db       = float(rng.uniform(0.0, 6.0))    # 0–6 dB end-to-end variation
    tilt_linear   = 10.0 ** (tilt_db / 20.0)        # amplitude ratio edge-to-edge
    # Linear ramp from 1/sqrt(tilt) to sqrt(tilt) so geometric mean = 1
    # (total power is preserved on average across many draws)
    lo            = 1.0 / np.sqrt(tilt_linear)
    hi            = np.sqrt(tilt_linear)
    ramp          = np.linspace(lo, hi, n, dtype=np.float64)
    if rng.random() < 0.5:
        ramp = ramp[::-1]                            # randomise direction

    spectrum      = np.fft.fft(iq)
    spectrum     *= ramp
    return np.fft.ifft(spectrum).astype(np.complex64)


def _quantise_5bit(iq: np.ndarray) -> np.ndarray:
    """Scale to 5-bit range and quantise. Returns (2, N_WINDOW) int8."""
    peak = float(np.max(np.abs(iq)))
    if peak < 1e-30:
        return np.zeros((2, len(iq)), dtype=np.int8)
    scale = QMAX / (peak * 1.10)   # 10% headroom
    I_q   = np.clip(np.round(iq.real * scale), QMIN, QMAX).astype(np.int8)
    Q_q   = np.clip(np.round(iq.imag * scale), QMIN, QMAX).astype(np.int8)
    return np.stack([I_q, Q_q], axis=0)


def _snr_to_power_class(snr_db):
    for i in range(len(PWR_EDGES) - 1):
        if PWR_EDGES[i] <= snr_db < PWR_EDGES[i + 1]:
            return i
    return len(PWR_LABELS) - 1


# ─────────────────────────────────────────────────────────────────────
#  SECTION 8 — SCENARIO SAMPLER
# ─────────────────────────────────────────────────────────────────────

# ── Measured statistics from Vodafone BCCH capture ────────────────
# Carrier slot distribution: slot 13 (77.5%), slot 12 (17.5%), slot 15 (5%)
# SNR: mean=27.7 dB, std=0.7 dB, range 25.6-32.3 dB
# Always one carrier (BCCH beacon transmits continuously)
_RW_SLOTS  = [13, 12, 15]
_RW_PROBS  = [0.775, 0.175, 0.05]
_RW_SNR_MEAN = 27.7
_RW_SNR_STD  = 0.7

def _sample_realworld_scenario(rng: np.random.Generator) -> dict:
    """
    Sample a scenario mimicking the real Vodafone BCCH capture.

    Key differences from synthetic mode:
      - Always exactly 1 carrier (BCCH beacon never silent)
      - Carrier drawn from slot distribution [13, 12, 15]
      - SNR drawn from N(27.7, 0.7) matching measured beacon power
      - Adjacent carrier always present at high power (leakage simulation)
      - Structured wideband blocker always active
      - CFO small (base station has accurate frequency reference)
    """
    # Always one carrier
    slot_idx   = int(rng.choice(_RW_SLOTS, p=_RW_PROBS))
    snr_db     = float(np.clip(
        rng.normal(_RW_SNR_MEAN, _RW_SNR_STD), 24.0, 34.0
    ))

    # Intra-slot frequency offset (hopping within slot)
    freq_hop_offset_hz = float(rng.uniform(-80_000, 80_000))

    # Adjacent carriers at distance 1-3 slots (leakage)
    adj_candidates = [s for s in [slot_idx-3, slot_idx-2, slot_idx-1,
                                   slot_idx+1, slot_idx+2, slot_idx+3]
                      if 0 <= s < N_SLOTS and s != slot_idx]
    adj_slot   = int(rng.choice(adj_candidates))
    adj_pwr_db = float(rng.uniform(-10, 0.0))   # always strong

    # Small CFO — base station has good frequency reference
    cfo_hz = float(rng.uniform(-5_000, 5_000))

    # Impairment flags:
    # - No fading (strong urban macro, close to tower)
    # - Adjacent channel always on (leakage is always present)
    # - Wideband blocker always on (interference observed in capture)
    imp_flags = np.array([
        1 if abs(cfo_hz) > 500 else 0,  # cfo
        0,   # static fading — off
        0,   # tv fading    — off
        0,   # cw tone      — off
        1,   # adjacent channel — always on
        1,   # wideband blocker — always on
    ], dtype=np.uint8)

    return dict(
        n_carriers       = 1,
        active_slots     = [slot_idx],
        snr_db           = snr_db,
        power_offsets    = [0.0],
        timing_offset    = int(rng.integers(0, 384)),
        imp_flags        = imp_flags,
        cfo_hz           = cfo_hz,
        channel_name     = 'NONE',
        velocity_kmh     = 0.0,
        channel_seed     = int(rng.integers(0, 100_000)),
        cw_freq_hz       = 0.0,
        cw_pwr_db        = 0.0,
        adj_slot         = adj_slot,
        adj_pwr_db       = adj_pwr_db,
        wb_pwr_db        = float(rng.uniform(-10.0, 0.0)),   # was uniform(5, 15) — blocker was always louder than the signal
        use_freq_hop     = True,
        freq_hop_offset_hz = freq_hop_offset_hz,
    )

def sample_scenario(rng: np.random.Generator, mode: str = 'synthetic') -> dict:
    """
    Draw all random parameters for one sample.
    Each of the 6 impairment flags is drawn independently (multi-label).
    """
    if mode == 'realworld':
        return _sample_realworld_scenario(rng)
    # Active carriers
    n_carriers   = int(rng.integers(0, 6))
    active_slots = sorted(
        rng.choice(N_SLOTS, size=n_carriers, replace=False).tolist()
    ) if n_carriers > 0 else []

    # weight low-SNR bins 3x — that's where P_d/P_fa stall
    snr_weights = np.where(SNR_GRID_DB < 10, 3.0, 1.0)
    snr_weights /= snr_weights.sum()
    snr_db = float(rng.choice(SNR_GRID_DB, p=snr_weights)) + float(rng.uniform(-0.5, 0.5))
    power_offsets = [float(rng.uniform(-3.0, 3.0)) for _ in active_slots]
    timing_offset = int(rng.integers(0, 384))

    # ── Impairment flags — drawn independently ────────────────────────
    # Gap 1 fix: wideband blocker raised from 0.10 → 0.50.
    # OTA captures show a co-existing LTE/UMTS blocker is the norm in
    # urban GSM bands, not a rare edge case. At 10% the TCN's stage-2
    # spatial suppression layer never sees enough blocker examples to
    # learn the wide-plateau vs narrow-peak distinction.
    #
    # Gap 5 fix: CFO when flag is SET narrowed from ±100 kHz → ±15 kHz.
    # Real base stations hold frequency to within a few kHz of nominal.
    # ±100 kHz CFO shifts carriers by half a slot width, making correct
    # slot assignment nearly impossible and teaching the front-end conv
    # to tolerate misalignment it will never see in practice.
    IMP_PROB = [0.3, 0.25, 0.20, 0.15, 0.35, 0.50]   # [cfo, st_fad, tv_fad, cw, adj, wb]
    imp_flags = np.array(
        [1 if rng.random() < p else 0 for p in IMP_PROB],
        dtype=np.uint8
    )   # shape (6,)  — multi-label binary

    # CFO magnitude (Gap 5 fix: tightened upper bound from ±100k → ±15k Hz)
    cfo_hz = (
        float(rng.uniform(-15_000, 15_000))   # was ±100,000
        if imp_flags[0] else
        float(rng.uniform(-500, 500))
    )

    # Fading channel selection
    has_static = bool(imp_flags[1])
    has_tv     = bool(imp_flags[2])
    if has_tv:
        ch_name, vel = TIMEVARYING_PROFILES[int(rng.integers(0, len(TIMEVARYING_PROFILES)))]
    elif has_static:
        ch_name, vel = str(rng.choice(STATIC_PROFILES)), 0.0
    else:
        ch_name, vel = "NONE", 0.0
    channel_seed = int(rng.integers(0, 100_000))

    # CW tone parameters
    cw_slot   = int(rng.choice(active_slots)) if active_slots else int(rng.integers(0, N_SLOTS))
    cw_freq   = float(SLOT_CENTRES_HZ[cw_slot]) + float(rng.uniform(-80_000, 80_000))
    cw_pwr_db = float(rng.uniform(-5, 15))

    # Adjacent channel parameters (Gap 4 fix)
    # Expanded search radius to ±3 slots (was ±1) so the model sees
    # energy leaking from further neighbours, not just immediate ones.
    # Also raise the minimum power from 5 dB to 10 dB: weak adjacent
    # carriers are indistinguishable from noise at this step and teach
    # nothing about spatial suppression — strong ones do.
    adj_candidates = []
    for s in active_slots:
        for delta in range(1, 4):   # ±1, ±2, ±3 slots
            for nb in (s - delta, s + delta):
                if 0 <= nb < N_SLOTS and nb not in active_slots:
                    adj_candidates.append(nb)
    # Also add random non-adjacent slots occasionally (15% of the time)
    # to train suppression of isolated interference not near any carrier.
    if not adj_candidates or rng.random() < 0.15:
        adj_candidates = [s for s in range(N_SLOTS)
                          if s not in active_slots] or list(range(N_SLOTS))
    adj_slot   = int(rng.choice(list(set(adj_candidates))))
    adj_pwr_db = float(rng.uniform(-15, 0.0))   # raised floor: was 5

    # Wideband blocker
    wb_pwr_db = float(rng.uniform(-15.0, 0.0))   # was uniform(0, 15) — blocker was always louder than the signal

    return dict(
        n_carriers    = n_carriers,
        active_slots  = active_slots,
        snr_db        = snr_db,
        power_offsets = power_offsets,
        timing_offset = timing_offset,
        imp_flags     = imp_flags,
        cfo_hz        = cfo_hz,
        channel_name  = ch_name,
        velocity_kmh  = float(vel),
        channel_seed  = channel_seed,
        cw_freq_hz    = cw_freq,
        cw_pwr_db     = cw_pwr_db,
        adj_slot      = adj_slot,
        adj_pwr_db    = adj_pwr_db,
        wb_pwr_db     = wb_pwr_db,
    )


# ─────────────────────────────────────────────────────────────────────
#  SECTION 9 — SAMPLE SYNTHESIS
# ─────────────────────────────────────────────────────────────────────

def synthesise_sample(sc: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Build one wideband 5-bit IQ window from a scenario dict.
    Returns int8 array of shape (2, N_WINDOW).
    """
    active_slots  = sc["active_slots"]
    imp_flags     = sc["imp_flags"]

    # ── Step 1: Build and mix per-carrier waveforms ───────────────────
    # Seed with unit-power thermal noise — power-relative impairments
    # (AWGN, blocker) scale to zero on a silent buffer when active_slots=[]
    _sigma   = 1.0 / np.sqrt(2.0)
    wideband = (rng.normal(0, _sigma, N_WINDOW) + 1j * rng.normal(0, _sigma, N_WINDOW)).astype(np.complex64)

    for i, slot_idx in enumerate(active_slots):
        native = generate_gsm_wave(
            n_frames     = 2,
            bsic         = 25,
            amp          = 0.8,
            channel      = sc["channel_name"],
            channel_seed = sc["channel_seed"] + i,
            velocity_kmh = sc["velocity_kmh"],
        )
        # Apply timing offset: skip timing_offset samples from the start.
        # CRITICAL: feed exactly N_RESAMPLE (683) samples to fft_resample.
        # Feeding more samples stretches the signal duration and compresses
        # all carrier frequencies — causing them to land in wrong FFT bins.
        start = sc["timing_offset"]
        end   = start + N_RESAMPLE   # exactly 683 samples
        while end > len(native):
            native = np.concatenate([native, native])
        carrier = _resample_to_104mhz(native[start:end])
        carrier = _freq_shift(carrier, float(SLOT_CENTRES_HZ[slot_idx]))
        carrier = _set_power(carrier, 10.0 ** (sc["power_offsets"][i] / 10.0))
        wideband = wideband + carrier

    if active_slots:
        wideband = _set_power(wideband, 1.0)

  
    carrier_ref_pwr = float(np.mean(np.abs(wideband) ** 2))
    # ── Step 2: Impairment injections ────────────────────────────────
    # CW tone
    if imp_flags[3] and active_slots:
        wideband = _inject_cw_tone(wideband, sc["cw_freq_hz"], sc["cw_pwr_db"], rng)

    # Adjacent channel carrier
    if imp_flags[4]:
        wideband = _inject_adjacent_carrier(wideband, sc["adj_slot"], sc["adj_pwr_db"], rng, carrier_ref_pwr)

    # Wideband blocker
    if imp_flags[5]:
        wideband = _inject_wideband_blocker(wideband, sc["wb_pwr_db"], rng)

    # ── Step 3: CFO ───────────────────────────────────────────────────
    if abs(sc["cfo_hz"]) > 0.5:
        wideband = _freq_shift(wideband, sc["cfo_hz"])

    # ── Step 4: AWGN ──────────────────────────────────────────────────
    wideband = _add_awgn(wideband, sc["snr_db"], rng, ref_pwr=carrier_ref_pwr)

    # ── Step 4b: Spectral tilt (Gap 3 fix) ───────────────────────────
    # Real receivers have a non-flat passband: the IF filter response
    # creates a power slope across the 5 MHz scan band (typically rising
    # left-to-right or right-to-left depending on the IF centre offset).
    # Without this, the TCN front-end conv sees a flat noise floor in
    # training but a tilted one on real hardware — the per-slot energy
    # levels shift systematically, confusing slots near the band edges.
    #
    # Implementation: apply a linear amplitude ramp in the frequency
    # domain (i.e. multiply the spectrum by a smooth slope). The ramp
    # direction is randomised; magnitude drawn from ±(0–6 dB) end-to-end.
    # Applied with 50% probability so the model sees both flat and tilted.
    if rng.random() < 0.50:
        wideband = _apply_spectral_tilt(wideband, rng)

    # ── Step 5: Quantise ─────────────────────────────────────────────
    return _quantise_5bit(wideband)


# ─────────────────────────────────────────────────────────────────────
#  SECTION 10 — LABEL BUILDER
# ─────────────────────────────────────────────────────────────────────

def build_labels(sc: dict, iq: np.ndarray, idx: int):
    """
    Returns:
      npy_dict  — saved inside the .npy file alongside IQ
      csv_row   — human-readable metadata row
    """
    active = sc["active_slots"]
    occ    = np.array([1 if k in active else 0 for k in range(N_SLOTS)], dtype=np.uint8)

    pwr_idx = np.full(N_SLOTS, -1, dtype=np.int8)
    for k in range(N_SLOTS):
        if k in active:
            i = active.index(k)
            pwr_idx[k] = _snr_to_power_class(sc["snr_db"] + sc["power_offsets"][i])

    npy_dict = {
        "iq"              : iq,                                 # (2, 65536) int8
        "occupancy"       : occ,                               # (25,)  uint8
        "power_class"     : pwr_idx,                           # (25,)  int8, -1=empty
        "impairment_flags": sc["imp_flags"],                   # (6,)   uint8 multi-label
        "snr_db"          : np.float32(sc["snr_db"]),
        "cfo_hz"          : np.float32(sc["cfo_hz"]),
    }

    pwr_strs = []
    for k in range(N_SLOTS):
        pwr_strs.append(PWR_LABELS[pwr_idx[k]] if pwr_idx[k] >= 0 else "none")

    csv_row = {
        "sample_idx"      : idx,
        "filename"        : f"sample_{idx:07d}.npy",
        "occupancy_bitmap": json.dumps(occ.tolist()),
        "n_active_slots"  : sc["n_carriers"],
        "active_slots"    : json.dumps(active),
        "power_class"     : json.dumps(pwr_strs),
        "snr_db"          : round(float(sc["snr_db"]), 4),
        "cfo_hz"          : round(float(sc["cfo_hz"]), 2),
        "imp_cfo"         : int(sc["imp_flags"][0]),
        "imp_static_fad"  : int(sc["imp_flags"][1]),
        "imp_tv_fad"      : int(sc["imp_flags"][2]),
        "imp_cw"          : int(sc["imp_flags"][3]),
        "imp_adj_ch"      : int(sc["imp_flags"][4]),
        "imp_wb_blocker"  : int(sc["imp_flags"][5]),
        "channel_name"    : sc["channel_name"],
        "velocity_kmh"    : round(float(sc["velocity_kmh"]), 1),
        "channel_seed"    : sc["channel_seed"],
        "timing_offset"   : sc["timing_offset"],
    }
    return npy_dict, csv_row


CSV_FIELDS = [
    "sample_idx", "filename",
    "occupancy_bitmap", "n_active_slots", "active_slots", "power_class",
    "snr_db", "cfo_hz",
    "imp_cfo", "imp_static_fad", "imp_tv_fad", "imp_cw", "imp_adj_ch", "imp_wb_blocker",
    "channel_name", "velocity_kmh", "channel_seed", "timing_offset",
]


# ─────────────────────────────────────────────────────────────────────
#  SECTION 11 — MAIN GENERATION LOOP
# ─────────────────────────────────────────────────────────────────────

def generate_dataset(
    out_dir   : str,
    n_samples : int,
    seed      : int,
    start_idx : int  = 0,
    verbose   : bool = True,
    mode      : str = 'realworld',
) -> None:
    out_dir     = Path(out_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    csv_path     = out_dir / "metadata.csv"
    write_header = (not csv_path.exists()) or (start_idx == 0)

    rng = np.random.default_rng(seed + start_idx)

    n_ok = n_err = 0

    with open(csv_path, "a", newline="") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for i in range(n_samples - start_idx):
            idx = start_idx + i
            try:
                sc = sample_scenario(rng, mode=mode)
                iq          = synthesise_sample(sc, rng)
                npy_d, row  = build_labels(sc, iq, idx)

                np.save(
                    str(samples_dir / f"sample_{idx:07d}.npy"),
                    npy_d, allow_pickle=True,
                )
                writer.writerow(row)
                csv_fh.flush()
                n_ok += 1

                if verbose and (n_ok == 1 or n_ok % 100 == 0):
                    flags = "".join(str(f) for f in sc["imp_flags"])
                    print(
                        f"  [{n_ok:>6}/{n_samples - start_idx}]  "
                        f"idx={idx}  slots={sc['active_slots']}  "
                        f"snr={sc['snr_db']:+.1f} dB  "
                        f"cfo={sc['cfo_hz']:+.0f} Hz  "
                        f"flags={flags}  ch={sc['channel_name']}"
                    )

            except Exception as exc:
                n_err += 1
                print(f"  [ERROR] idx={idx}: {exc}", file=sys.stderr)
                if n_err > 20:
                    print("Too many errors — aborting.", file=sys.stderr)
                    raise

    print(f"\nDone.  Generated={n_ok}  Errors={n_err}")
    print(f"  Output : {out_dir}")
    print(f"  CSV    : {csv_path}")
    print(f"  Shape  : (2, {N_WINDOW}) int8 + label dict per sample")
    if n_ok:
        print(f"  Size   : ~{n_ok * 2 * N_WINDOW / (1024**3):.2f} GB (IQ arrays only)")


# ─────────────────────────────────────────────────────────────────────
#  SECTION 12 — CLI  (override CONFIG values from the command line)
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=(
            "GSM channel-scan dataset generator — 5-bit IQ at 104 MS/s.\n"
            "No external dependencies beyond numpy and scipy.\n"
            "Edit the CONFIG block at the top of this file, or use the flags below."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n-samples", type=int,   default=N_SAMPLES,
                    help=f"Number of samples to generate (default: {N_SAMPLES})")
    ap.add_argument('--mode', type=str, default='synthetic',
                choices=['synthetic', 'realworld'],
                help='synthetic = broad random, realworld = Vodafone BCCH mimic')
    ap.add_argument("--out-dir",   type=str,   default=OUT_DIR,
                    help=f"Output root directory (default: {OUT_DIR})")
    ap.add_argument("--seed",      type=int,   default=SEED,
                    help=f"Master random seed (default: {SEED})")
    ap.add_argument("--start-idx", type=int,   default=START_IDX,
                    help="Resume from this sample index (default: 0)")
    ap.add_argument("--quiet",     action="store_true",
                    help="Suppress per-sample progress output")
    ap.add_argument("--list-channels", action="store_true",
                    help="Print available ITU channel profiles and exit")
    args = ap.parse_args()

    if args.list_channels:
        print("Available ITU channel profiles:")
        print("  NONE  (ideal, no multipath)")
        for k, v in _CHANNEL_PROFILES.items():
            print(f"  {k:<8}  {v['name']}")
        return

    print("GSM Channel Scan — Dataset Generator (self-contained)")
    print(f"  Output     : {args.out_dir}")
    print(f"  N samples  : {args.n_samples}")
    print(f"  Seed       : {args.seed}")
    print(f"  Resume from: {args.start_idx}")
    print(f"  Window     : {N_WINDOW} samples @ {TARGET_RATE/1e6:.0f} MS/s"
          f" = {N_WINDOW/TARGET_RATE*1e6:.0f} µs")
    print(f"  Slots      : {N_SLOTS} × {SLOT_BW_HZ/1e3:.0f} kHz = "
          f"{N_SLOTS*SLOT_BW_HZ/1e6:.0f} MHz")
    print(f"  Impairments: {IMP_NAMES}  (multi-label, drawn independently)")
    print()

    generate_dataset(
        out_dir   = args.out_dir,
        n_samples = args.n_samples,
        seed      = args.seed,
        start_idx = args.start_idx,
        verbose   = not args.quiet,
        mode      = args.mode,
    )


if __name__ == "__main__":
    main()