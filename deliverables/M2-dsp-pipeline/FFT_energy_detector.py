import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample as fft_resample

# ─────────────────────────────────────────────────────────────────────
#  SIGNAL PARAMETERS  (fixed — do not change without regenerating data)
# ─────────────────────────────────────────────────────────────────────

TARGET_RATE   = 104_000_000   # ADC sample rate (Sa/s)
SCAN_RATE     =   5_000_000   # scan bandwidth sample rate (Sa/s)
N_WINDOW      = 65_536        # samples per input window at TARGET_RATE
N_FFT         = 1024          # FFT size after decimation
N_CH          = 25            # number of 200 kHz channels
CH_BW_HZ      = 200_000       # channel bandwidth (Hz)
BIN_HZ        = SCAN_RATE / N_FFT   # 4882.8 Hz per bin

# Each channel k is centred at:  f_k = -2.4 MHz + k × 200 kHz
# After fftshift the centre bin is:
#   bin_k = round( (f_k + SCAN_RATE/2) / BIN_HZ )
# Channel k uses bins [bin_k - 20 : bin_k + 21)  — 41 bins, ~200 kHz
CH_BINS = []
for _k in range(N_CH):
    _f    = -2_400_000 + _k * CH_BW_HZ
    _cbin = int(round((_f + SCAN_RATE / 2) / BIN_HZ))
    _lo   = max(0,     _cbin - 20)
    _hi   = min(N_FFT, _cbin + 21)
    CH_BINS.append((_lo, _hi))

DEFAULT_THRESHOLD = 1.38  # re-tuned for Welch; optimal F1 on validation set
                          # (was 3.0 in the buggy single-FFT version)
WELCH_HOP = N_FFT // 2   # 512 samples — 50% overlap between segments

# Threshold sweep values for ROC curve
THRESHOLD_SWEEP = np.concatenate([
    np.linspace(0.5,  5.0, 60),
    np.linspace(5.0, 30.0, 20),
])

IMP_NAMES = [
    "cfo", "static_fading", "timevarying_fading",
    "cw_tone", "adjacent_channel", "wideband_blocker",
]


# ─────────────────────────────────────────────────────────────────────
#  STEP 1 — FORM COMPLEX BASEBAND SIGNAL
# ─────────────────────────────────────────────────────────────────────

def form_complex(iq_int8: np.ndarray) -> np.ndarray:
    """
    Combine I and Q channels into one complex float array.

    Input:  (2, 65536) int8   — row 0 = I, row 1 = Q
    Output: (65536,) complex64

    Each pair (I[n], Q[n]) becomes the complex number I[n] + j·Q[n].
    This is the standard baseband representation of a radio signal.
    The complex form is required for the FFT to correctly resolve
    positive and negative frequencies separately.
    """
    return (iq_int8[0].astype(np.float32) +
            1j * iq_int8[1].astype(np.float32))


# ─────────────────────────────────────────────────────────────────────
#  STEP 2 — DECIMATE TO SCAN BANDWIDTH
# ─────────────────────────────────────────────────────────────────────

def decimate(signal: np.ndarray) -> np.ndarray:
    """
    Decimate the signal from TARGET_RATE to SCAN_RATE using FFT resampling.

    Input:  (65536,) complex64 at 104 MS/s
    Output: (3151,)  complex64 at   5 MS/s

    Both arrays represent the same 630 µs duration. The output has
    lower sample rate because we only care about the central 5 MHz
    of the 104 MHz input bandwidth.

    A 1024-point FFT at 104 MS/s gives 101 kHz per bin — only 2 bins
    per 200 kHz channel, which is too coarse to separate channels.
    After decimation to 5 MS/s the same FFT gives 4.88 kHz per bin —
    41 bins per channel, which is enough resolution.
    """
    n_out = int(round(len(signal) * SCAN_RATE / TARGET_RATE))
    return fft_resample(signal, n_out).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────
#  STEP 3 — WELCH POWER SPECTRUM ESTIMATE
#  Replaces the buggy apply_window() + compute_fft() pair.
# ─────────────────────────────────────────────────────────────────────

def welch_power_spectrum(decimated: np.ndarray) -> np.ndarray:
    """
    Compute an averaged power spectrum using the Welch method.

    The Welch method is the standard solution to the mismatch between
    the signal length (3151 samples) and the FFT size (1024 points).

    What was wrong before
    ---------------------
    The previous code called:
        window = np.hanning(3151)          # 3151-point Hann
        spec   = np.fft.fft(sig * window, n=1024)
    When the input array is longer than n=1024, NumPy silently crops
    to the first 1024 samples. The effective window on those 1024 samples
    ran from 0.000 → 0.726 and never reached 1.0. This meant:
      - 67.5% of the signal energy (samples 1024–3151) was discarded
      - The truncation at sample 1024 created a hard discontinuity —
        exactly the spectral leakage the window was supposed to remove

    What Welch does instead
    -----------------------
    Split the 3151 decimated samples into overlapping segments of exactly
    N_FFT=1024 samples each, with hop size WELCH_HOP=512 (50% overlap).
    From 3151 samples this gives 5 segments:
        seg 0: samples [   0, 1024)
        seg 1: samples [ 512, 1536)
        seg 2: samples [1024, 2048)
        seg 3: samples [1536, 2560)
        seg 4: samples [2048, 3072)   (covers samples up to 3072 of 3151)

    For each segment:
      1. Multiply by Hann(1024) — a window exactly 1024 samples long
         that goes 0 → 1 → 0 with no truncation and no discontinuity
      2. Compute 1024-pt FFT + fftshift
      3. Compute squared magnitude per bin: |X[k]|² / N_FFT

    After all 5 segments, average the 5 power spectra.
    Averaging reduces noise variance by factor 5, equivalent to
    approximately 3.5 dB improvement in SNR for energy detection.

    Parameters
    ----------
    decimated : (3151,) complex64 at SCAN_RATE

    Returns
    -------
    (N_FFT,) float64 — averaged power spectrum, one value per bin
    """
    n       = len(decimated)
    window  = np.hanning(N_FFT)          # correct: exactly N_FFT long
    n_segs  = (n - N_FFT) // WELCH_HOP + 1
    acc     = np.zeros(N_FFT, dtype=np.float64)

    for i in range(n_segs):
        start = i * WELCH_HOP
        seg   = decimated[start : start + N_FFT]  # exactly 1024 samples
        spec  = np.fft.fftshift(np.fft.fft(seg * window, n=N_FFT))
        acc  += np.abs(spec) ** 2 / N_FFT

    return acc / n_segs     # averaged power spectrum


# ─────────────────────────────────────────────────────────────────────
#  STEP 5 — COMPUTE PER-CHANNEL POWER
# ─────────────────────────────────────────────────────────────────────

def channel_power(power_spectrum: np.ndarray) -> np.ndarray:
    """
    Sum the FFT bin powers within each channel's frequency window.

    Input:  (1024,) float64 power spectrum
    Output: (25,)   float64 power per channel

    Channel k uses bins [CH_BINS[k][0] : CH_BINS[k][1]).
    Summing 41 bins per channel integrates all the signal energy
    within that 200 kHz window into one number.

    If the channel is empty, this sum is approximately:
        41 × σ²_noise / N_FFT  (noise energy in 41 bins)
    If the channel is occupied by a GSM carrier, this sum is much
    larger because the GMSK signal adds extra energy across the 41 bins.
    """
    return np.array(
        [np.sum(power_spectrum[lo:hi]) for lo, hi in CH_BINS],
        dtype=np.float64,
    )


# ─────────────────────────────────────────────────────────────────────
#  STEP 6 — ESTIMATE NOISE FLOOR
# ─────────────────────────────────────────────────────────────────────

def noise_floor(ch_power: np.ndarray) -> float:
    """
    Estimate the background noise floor from the 25 channel powers.

    Returns the median of all 25 channel power values.

    The median is used instead of the mean because it is robust to
    outliers. If 3 of the 25 channels are occupied and have high power,
    the mean would be pulled upward, making the noise estimate too high
    and the threshold too conservative. The median is unaffected by
    outliers as long as fewer than half the channels are occupied —
    which is always the case in this dataset (at most 5 of 25 occupied).

    This is the standard CFAR (Constant False Alarm Rate) technique
    used in radar and spectrum sensing literature.
    """
    return float(np.median(ch_power))


# ─────────────────────────────────────────────────────────────────────
#  STEP 7 — APPLY THRESHOLD AND DECIDE
# ─────────────────────────────────────────────────────────────────────

def threshold_decision(ch_power: np.ndarray,
                       noise: float,
                       threshold_factor: float) -> np.ndarray:
    """
    Compare each channel's power to the detection threshold.

    threshold = threshold_factor × noise_floor

    channel k is declared OCCUPIED  if  ch_power[k] > threshold
    channel k is declared EMPTY     if  ch_power[k] ≤ threshold

    Returns: (25,) uint8 — 1 = occupied, 0 = empty

    The threshold_factor controls the trade-off between detection
    probability (P_d) and false alarm probability (P_fa):
        low  factor → low threshold → high P_d, high P_fa
        high factor → high threshold → low P_fa,  low P_d
    Sweep the factor from 0.5 to 30 to build the full ROC curve.
    """
    threshold = threshold_factor * max(noise, 1e-30)
    return (ch_power > threshold).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
#  FULL CHANNELISER — assembles all 7 steps
# ─────────────────────────────────────────────────────────────────────

def channelise(iq_int8: np.ndarray,
               threshold_factor: float = DEFAULT_THRESHOLD) -> tuple:
    """
    Run the complete FFT channeliser on one IQ sample.

    Steps 1 → 2 → 3 → 4 → 5 → 6.

    Parameters
    ----------
    iq_int8          : (2, 65536) int8
    threshold_factor : noise floor multiplier for occupancy decision

    Returns
    -------
    occupancy  : (25,) uint8   — 1 = occupied, 0 = empty
    ch_pwr     : (25,) float64 — integrated power per channel
    noise      : float         — median channel power (noise estimate)
    threshold  : float         — detection threshold used
    """
    x         = form_complex(iq_int8)              # step 1
    x_dec     = decimate(x)                        # step 2
    pwr_spec  = welch_power_spectrum(x_dec)        # step 3 (Welch)
    ch_pwr    = channel_power(pwr_spec)            # step 4
    noise     = noise_floor(ch_pwr)               # step 5
    occupancy = threshold_decision(               # step 6
        ch_pwr, noise, threshold_factor
    )
    threshold = threshold_factor * max(noise, 1e-30)
    return occupancy, ch_pwr, noise, threshold


# ─────────────────────────────────────────────────────────────────────
#  BATCH EVALUATION
# ─────────────────────────────────────────────────────────────────────

def run_on_dataset(sample_files: list,
                   threshold_factor: float = DEFAULT_THRESHOLD) -> tuple:
    """Run channeliser on a list of .npy sample files."""
    pred_list  = []
    truth_list = []
    snr_list   = []
    flags_list = []

    for fpath in sample_files:
        d = np.load(str(fpath), allow_pickle=True).item()
        occ, _, _, _ = channelise(d["iq"], threshold_factor)
        pred_list .append(occ)
        truth_list.append(d["occupancy"].astype(np.int32))
        snr_list  .append(float(d["snr_db"]))
        flags_list.append(d["impairment_flags"].astype(np.int32))

    return (
        np.array(pred_list,  dtype=np.int32),
        np.array(truth_list, dtype=np.int32),
        np.array(snr_list,   dtype=np.float32),
        np.array(flags_list, dtype=np.int32),
    )


# ─────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Binary detection metrics from prediction and ground-truth arrays."""
    p  = pred .flatten().astype(int)
    t  = truth.flatten().astype(int)
    TP = int(np.sum((p == 1) & (t == 1)))
    FP = int(np.sum((p == 1) & (t == 0)))
    TN = int(np.sum((p == 0) & (t == 0)))
    FN = int(np.sum((p == 0) & (t == 1)))
    P_d  = TP / max(TP + FN, 1)
    P_fa = FP / max(FP + TN, 1)
    prec = TP / max(TP + FP, 1)
    F1   = 2 * prec * P_d / max(prec + P_d, 1e-10)
    return dict(TP=TP, FP=FP, TN=TN, FN=FN,
                P_d=P_d, P_fa=P_fa, precision=prec, F1=F1)


def pd_vs_snr(pred_arr, truth_arr, snr_arr, bw=2.0) -> dict:
    """P_d and P_fa broken down by SNR bin."""
    edges = np.arange(
        np.floor(snr_arr.min() / bw) * bw,
        np.ceil (snr_arr.max() / bw) * bw + bw,
        bw,
    )
    out = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask   = (snr_arr >= lo) & (snr_arr < hi)
        if mask.sum() == 0:
            continue
        m = compute_metrics(pred_arr[mask], truth_arr[mask])
        m["n"] = int(mask.sum())
        out[round((lo + hi) / 2, 1)] = m
    return out


# ─────────────────────────────────────────────────────────────────────
#  REPORTING
# ─────────────────────────────────────────────────────────────────────

def print_metrics(m: dict, label: str = ""):
    tag = f"[{label}]  " if label else ""
    print(f"\n{tag}FFT Channeliser — Metrics")
    print(f"  P_d   : {m['P_d']:.4f}  "
          f"({m['TP']} TP / {m['TP']+m['FN']} occupied)")
    print(f"  P_fa  : {m['P_fa']:.4f}  "
          f"({m['FP']} FP / {m['FP']+m['TN']} empty)")
    print(f"  Prec  : {m['precision']:.4f}")
    print(f"  F1    : {m['F1']:.4f}")
    print(f"  TP={m['TP']}  FP={m['FP']}  TN={m['TN']}  FN={m['FN']}")


def print_pd_snr(table: dict):
    print(f"\n  P_d vs SNR  (each █ ≈ 0.05)")
    print(f"  {'SNR':>7}  {'P_d':>7}  {'P_fa':>7}  {'F1':>7}  {'N':>5}")
    print("  " + "─" * 50)
    for snr_c in sorted(table.keys()):
        m   = table[snr_c]
        bar = "█" * int(m["P_d"] * 20)
        print(f"  {snr_c:>7.1f}  {m['P_d']:>7.4f}  {m['P_fa']:>7.4f}"
              f"  {m['F1']:>7.4f}  {m['n']:>5}  {bar}")


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="FFT Channeliser — DSP baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset",   type=str,   default=None,
                    help="Path to dataset root folder (contains samples/)")
    ap.add_argument("--sample",    type=str,   default=None,
                    help="Path to a single .npy file for step-by-step output")
    ap.add_argument("--n-samples", type=int,   default=None,
                    help="Limit evaluation to first N samples")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Noise floor multiplier  (default: {DEFAULT_THRESHOLD})")
    ap.add_argument("--pd-snr",  action="store_true",
                    help="Print P_d vs SNR table")
    ap.add_argument("--roc",     action="store_true",
                    help="Sweep threshold and print ROC curve")
    args = ap.parse_args()

    # ── Single sample — step-by-step output ───────────────────────────
    if args.sample:
        fpath = Path(args.sample)
        d     = np.load(str(fpath), allow_pickle=True).item()
        truth = d["occupancy"]
        imp_on = [IMP_NAMES[i] for i, v in enumerate(d["impairment_flags"]) if v]

        print(f"\n{'='*62}")
        print(f"FFT Channeliser — Step-by-Step")
        print(f"{'='*62}")
        print(f"  Sample     : {fpath.name}")
        print(f"  SNR        : {float(d['snr_db']):.1f} dB")
        print(f"  Impairments: {imp_on if imp_on else ['none']}")
        print(f"  Occupied   : {[i for i,v in enumerate(truth) if v]}")

        # Step 1
        x = form_complex(d["iq"])
        print(f"\nStep 1 — Form complex signal")
        print(f"  x[n] = I[n] + j·Q[n]")
        print(f"  Length  : {len(x)} samples at {TARGET_RATE/1e6:.0f} MS/s")
        print(f"  Range   : I ∈ [{d['iq'][0].min()},{d['iq'][0].max()}]  "
              f"Q ∈ [{d['iq'][1].min()},{d['iq'][1].max()}]")
        print(f"  RMS pwr : {float(np.mean(np.abs(x)**2)):.4f}")

        # Step 2
        x_dec = decimate(x)
        print(f"\nStep 2 — Decimate to scan bandwidth")
        print(f"  {len(x)} samples at {TARGET_RATE/1e6:.0f} MS/s")
        print(f"  → {len(x_dec)} samples at {SCAN_RATE/1e6:.0f} MS/s")
        print(f"  Same duration: {len(x)/TARGET_RATE*1e6:.1f} µs → "
              f"{len(x_dec)/SCAN_RATE*1e6:.1f} µs")
        print(f"  FFT bin width will be: {BIN_HZ:.1f} Hz  "
              f"({int(CH_BW_HZ/BIN_HZ)} bins per channel)")

        # Step 3 — Welch
        n_segs = (len(x_dec) - N_FFT) // WELCH_HOP + 1
        print(f"\nStep 3 — Welch power spectrum  (replaces single windowed FFT)")
        print(f"  Signal length : {len(x_dec)} samples")
        print(f"  Segment size  : {N_FFT} samples  (each gets its own Hann({N_FFT}))")
        print(f"  Hop size      : {WELCH_HOP} samples  (50% overlap)")
        print(f"  Segments      : {n_segs}")
        hann_check = np.hanning(N_FFT)
        print(f"  Hann({N_FFT}): w[0]={hann_check[0]:.4f}  "
              f"w[512]={hann_check[512]:.4f}  w[1023]={hann_check[-1]:.4f}")
        print(f"  (correctly reaches 1.0 at centre — no truncation)")
        pwr_spec = welch_power_spectrum(x_dec)
        peak_bin  = int(np.argmax(pwr_spec))
        peak_freq = (peak_bin - N_FFT / 2) * BIN_HZ
        print(f"  Averaged spectrum: {N_FFT} bins of {BIN_HZ:.1f} Hz each")
        print(f"  Total power : {pwr_spec.sum():.4f}")
        print(f"  Peak bin    : {peak_bin} = {peak_freq/1e3:+.1f} kHz  "
              f"(power = {pwr_spec[peak_bin]:.4f})")

        # Step 4
        ch_pwr = channel_power(pwr_spec)
        print(f"\nStep 4 — Per-channel power  (41 bins × 4882.8 Hz = 200 kHz each)")
        print(f"  {'Ch':>4}  {'Centre':>10}  {'Bins':>12}  "
              f"{'Power':>10}  {'Truth':>8}")
        print(f"  {'─'*52}")
        for k in range(N_CH):
            lo, hi  = CH_BINS[k]
            f_c     = (-2_400_000 + k * CH_BW_HZ) / 1e3
            t_str   = "OCCUPIED" if truth[k] else ""
            if truth[k] or ch_pwr[k] > np.median(ch_pwr) * 1.3:
                print(f"  {k:>4}  {f_c:>+9.0f} kHz  [{lo:4d},{hi:4d})  "
                      f"{ch_pwr[k]:>10.4f}  {t_str}")

        # Step 5
        noise = noise_floor(ch_pwr)
        print(f"\nStep 5 — Estimate noise floor")
        print(f"  Method   : median of all {N_CH} channel powers")
        print(f"  Sorted powers (lowest 5): "
              f"{np.sort(ch_pwr)[:5].round(4).tolist()}")
        print(f"  Median (rank 13 of 25) : {noise:.4f}")

        # Step 6
        occupancy = threshold_decision(ch_pwr, noise, args.threshold)
        threshold = args.threshold * max(noise, 1e-30)
        print(f"\nStep 6 — Threshold decision")
        print(f"  Threshold = {args.threshold} × {noise:.4f} = {threshold:.4f}")
        print(f"  Channel is OCCUPIED if power > {threshold:.4f}")
        print()
        print(f"  {'Ch':>4}  {'Power':>10}  {'Ratio':>8}  "
              f"{'Pred':>6}  {'Truth':>6}  {'Result':>8}")
        print(f"  {'─'*52}")
        for k in range(N_CH):
            if truth[k] or occupancy[k]:
                ratio = ch_pwr[k] / max(noise, 1e-30)
                if   truth[k]==1 and occupancy[k]==1: res = "✓ TP"
                elif truth[k]==0 and occupancy[k]==1: res = "✗ FP"
                elif truth[k]==1 and occupancy[k]==0: res = "✗ FN"
                else:                                  res = "─ TN"
                print(f"  {k:>4}  {ch_pwr[k]:>10.4f}  {ratio:>8.2f}×  "
                      f"{occupancy[k]:>6}  {truth[k]:>6}  {res}")

        m = compute_metrics(occupancy.reshape(1,-1), truth.reshape(1,-1))
        print(f"\n  Result: {occupancy.sum()} predicted occupied  /  "
              f"{int(truth.sum())} actually occupied")
        print(f"  P_d={m['P_d']:.3f}  P_fa={m['P_fa']:.3f}  F1={m['F1']:.3f}")
        return

    # ── Dataset evaluation ─────────────────────────────────────────────
    if args.dataset is None:
        ap.print_help()
        sys.exit(1)

    dataset_dir  = Path(args.dataset)
    sample_files = sorted((dataset_dir / "samples").glob("sample_*.npy"))
    if not sample_files:
        print(f"ERROR: no samples in {dataset_dir}/samples", file=sys.stderr)
        sys.exit(1)
    if args.n_samples:
        sample_files = sample_files[: args.n_samples]

    print(f"\nFFT Channeliser — DSP Baseline  (Welch averaged)")
    print(f"  Dataset      : {dataset_dir}")
    print(f"  Samples      : {len(sample_files)}")
    print(f"  Threshold    : {args.threshold} × median noise floor")
    print(f"  FFT size     : {N_FFT}  ({SCAN_RATE/1e6:.0f} MS/s after decimation)")
    print(f"  Bin width    : {BIN_HZ:.1f} Hz")
    print(f"  Bins/channel : ~41  (~{CH_BW_HZ/1e3:.0f} kHz per channel)")
    print(f"  Welch segs   : {(int(round(N_WINDOW*SCAN_RATE/TARGET_RATE))-N_FFT)//WELCH_HOP+1}  "
          f"(hop={WELCH_HOP}, 50% overlap)")
    print(f"  Window       : Hann({N_FFT}) per segment  (correct: 0→1→0)")
    print(f"  Noise floor  : median of {N_CH} channel powers")

    print("\nRunning...", end="", flush=True)
    pred_arr, truth_arr, snr_arr, flags_arr = run_on_dataset(
        sample_files, args.threshold
    )
    print(" done.")

    m = compute_metrics(pred_arr, truth_arr)
    print_metrics(m, label=f"threshold={args.threshold}×")

    if args.pd_snr:
        table = pd_vs_snr(pred_arr, truth_arr, snr_arr)
        print_pd_snr(table)

    if args.roc:
        print(f"\n  ROC curve — sweeping threshold factor")
        print(f"  {'k':>8}  {'P_fa':>8}  {'P_d':>8}  {'F1':>8}")
        print("  " + "─" * 38)
        best = {"F1": -1}
        for k in THRESHOLD_SWEEP:
            p, t, _, _ = run_on_dataset(sample_files, k)
            mi = compute_metrics(p, t)
            print(f"  {k:>8.2f}  {mi['P_fa']:>8.4f}  "
                  f"{mi['P_d']:>8.4f}  {mi['F1']:>8.4f}")
            if mi["F1"] > best["F1"]:
                best = {"k": k, **mi}
        print(f"\n  Best F1 operating point:")
        print(f"    threshold={best['k']:.2f}×  "
              f"P_fa={best['P_fa']:.4f}  "
              f"P_d={best['P_d']:.4f}  "
              f"F1={best['F1']:.4f}")

    print(f"\n  Per-impairment breakdown:")
    print(f"  {'Condition':25s}  {'P_d':>7}  {'P_fa':>7}  {'F1':>7}  {'N':>5}")
    print("  " + "─" * 54)
    clean = flags_arr.sum(axis=1) == 0
    if clean.sum() > 0:
        mc = compute_metrics(pred_arr[clean], truth_arr[clean])
        print(f"  {'clean':25s}  {mc['P_d']:>7.4f}  {mc['P_fa']:>7.4f}"
              f"  {mc['F1']:>7.4f}  {clean.sum():>5}")
    for fi, name in enumerate(IMP_NAMES):
        mask = flags_arr[:, fi] == 1
        if mask.sum() == 0:
            continue
        mi = compute_metrics(pred_arr[mask], truth_arr[mask])
        print(f"  {name:25s}  {mi['P_d']:>7.4f}  {mi['P_fa']:>7.4f}"
              f"  {mi['F1']:>7.4f}  {mask.sum():>5}")


if __name__ == "__main__":
    main()
