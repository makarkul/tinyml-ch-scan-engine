import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import firwin, resample as fft_resample

# ─────────────────────────────────────────────────────────────────────
#  SIGNAL AND FILTER PARAMETERS
# ─────────────────────────────────────────────────────────────────────

TARGET_RATE = 104_000_000   # ADC sample rate (Sa/s)
SCAN_RATE   =   5_000_000   # rate after decimation (Sa/s)
N_WINDOW    = 65_536        # samples per input window at TARGET_RATE
N_CH        = 25            # number of channels
CH_BW_HZ    = 200_000       # channel bandwidth (Hz)
L           = 16            # filter taps per polyphase branch
N_FILTER    = L * N_CH      # total prototype filter length = 400 taps

DEFAULT_THRESHOLD = 1.42  # re-tuned after fixing the FFT sign convention bug
                          # (was 3.0 before the fix; optimal F1 on validation set)

THRESHOLD_SWEEP = np.concatenate([
    np.linspace(0.5,  5.0, 60),
    np.linspace(5.0, 30.0, 20),
])

IMP_NAMES = [
    "cfo", "static_fading", "timevarying_fading",
    "cw_tone", "adjacent_channel", "wideband_blocker",
]


# ─────────────────────────────────────────────────────────────────────
#  STEP 3+4 — PROTOTYPE FILTER DESIGN AND POLYPHASE DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────

def design_prototype_filter(n_ch: int, l: int, fs: float) -> np.ndarray:
    """
    Design a Kaiser-windowed sinc prototype lowpass filter.

    The prototype filter is the fundamental building block of the PFB.
    It is a lowpass filter that passes one channel bandwidth (100 kHz)
    and rejects all higher frequencies.

    Parameters
    ----------
    n_ch : number of channels (N = 25)
    l    : taps per polyphase branch (L = 16)
    fs   : sample rate in Hz (5 MS/s)

    Returns
    -------
    h : (n_ch*l,) float64 — prototype filter coefficients

    Design parameters:
      Cutoff     : fs / (2*N) = 100 kHz  (half channel bandwidth)
      Length     : L*N = 400 taps
      Window     : Kaiser with beta=8 → ~30 dB stopband attenuation
      Normalised : sum(h) = 1.0  (unity DC gain)
    """
    fc_norm = (fs / n_ch) / fs   # cutoff as fraction of sample rate = 1/N = 0.04
    h       = firwin(n_ch * l, fc_norm, window=('kaiser', 8.0))
    return h.astype(np.float64)


def polyphase_decompose(h: np.ndarray, n_ch: int, l: int) -> np.ndarray:
    """
    Decompose the prototype filter h into the polyphase matrix G.

    The prototype filter h[n] of length N*L is split into N short
    filters of length L, one per channel.

    Polyphase component p contains every N-th tap of h starting at p:
        g_p[l] = h[p + l*N]  for l = 0, 1, ..., L-1

    Mathematically: h.reshape(L, N) gives a matrix where row l
    contains the taps h[l*N], h[l*N+1], ..., h[l*N+N-1].
    Transposing gives G[p, l] = h[p + l*N] — exactly what we want.

    Parameters
    ----------
    h    : (N*L,) prototype filter
    n_ch : N = 25
    l    : L = 16

    Returns
    -------
    G : (N, L) float64 — polyphase filter matrix
        G[p, :] = the p-th polyphase component filter of length L
    """
    return h.reshape(l, n_ch).T.astype(np.float64)   # shape (N, L)


# Pre-compute the polyphase matrix once at import time
_H = design_prototype_filter(N_CH, L, SCAN_RATE)
_G = polyphase_decompose(_H, N_CH, L)   # (25, 16)


# ─────────────────────────────────────────────────────────────────────
#  STEPS 1+2 — FORM COMPLEX SIGNAL AND DECIMATE
# ─────────────────────────────────────────────────────────────────────

def prepare_signal(iq_int8: np.ndarray) -> np.ndarray:
    """
    Form complex signal and decimate to SCAN_RATE.

    Input:  (2, 65536) int8
    Output: (3151,)   complex64 at SCAN_RATE (5 MS/s)
    """
    iq    = (iq_int8[0].astype(np.float32) +
             1j * iq_int8[1].astype(np.float32))
    n_out = int(round(N_WINDOW * SCAN_RATE / TARGET_RATE))
    return fft_resample(iq, n_out).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────
#  STEP 5 — PFB PROCESSING
# ─────────────────────────────────────────────────────────────────────

def pfb_process(x: np.ndarray) -> np.ndarray:
    """
    Apply the PFB to the decimated signal and return channel powers.

    For each block m of N consecutive input samples:
      1. Build the input buffer of L*N samples ending at sample (m+1)*N.
      2. Reverse the buffer for convolution ordering, reshape to (L, N).T
         giving the polyphase input matrix X of shape (N_CH, L).
      3. Compute branch outputs:
            v[p] = sum_l  G[p, l] * X[p, l]  = (G * X).sum(axis=1)
      4. Compute channel outputs:
            y_k[m] = IDFT direction of DFT{v}[k]   via fft(conj(v))
         Apply fftshift so channel k maps to frequency:
            f_k = -2.4 MHz + k × 200 kHz

    Sign convention note (root cause of the original bug)
    ------------------------------------------------------
    The buffer reversal buf[::-1] in step 2 means branch p receives
    samples indexed at time (m+1)*N - 1 - p, not (m+1)*N - 1 + p.
    Under this reversed commutation, branch p accumulates phase:

        phi[p] ≈ phi_0 - 2π f p / SCAN_RATE   (decreasing with p)

    The standard np.fft.fft computes:
        Y[k] = Σ_p v[p] · e^{-j2π k p / N}

    Peaking when  f/SCAN_RATE + k/N = 0  →  k = -f·N/SCAN_RATE.

    For a tone at f = -1 MHz:  k_peak = +1e6·25/5e6 = +5 (before fftshift)
    → channel 5 + 12 = 17 after fftshift.  But slot 7 is at -1 MHz.

    The fix: conjugate v before the FFT.  conj(v[p]) has phase:
        -phi[p] ≈ -phi_0 + 2π f p / SCAN_RATE   (increasing with p)

    Now np.fft.fft(conj(v)) peaks when f/SCAN_RATE - k/N = 0
    → k = f·N/SCAN_RATE.
    For f = -1 MHz: k_peak = -5 mod 25 = 20 (before fftshift)
    → channel 20 - 13 = 7 after fftshift.  Correct.

    After processing all blocks, return the mean |y_k|² per channel.

    Parameters
    ----------
    x : (M,) complex64 signal at SCAN_RATE

    Returns
    -------
    ch_power : (N_CH,) float64 — mean power per channel
    """
    M        = len(x)
    n_blocks = M // N_CH
    ch_acc   = np.zeros(N_CH, dtype=np.float64)

    for m in range(n_blocks):
        end       = (m + 1) * N_CH
        buf_start = end - N_FILTER
        if buf_start >= 0:
            buf = x[buf_start : end]
        else:
            buf = np.concatenate([
                np.zeros(-buf_start, dtype=np.complex64),
                x[0 : end],
            ])

        buf_rev = buf[::-1]
        X       = buf_rev.reshape(L, N_CH).T        # (N_CH, L)
        v       = (_G * X).sum(axis=1)              # (N_CH,)

        # FIX: conjugate v before FFT to correct the reversed-commutation
        # sign convention.  Without conj(), a tone at -1 MHz (slot 7) appears
        # at channel 17 instead of channel 7.  With conj() it appears at
        # channel 7 — correct for all 25 slots.
        y = np.fft.fftshift(np.fft.fft(np.conj(v)))   # (N_CH,)

        ch_acc += np.abs(y) ** 2

    return ch_acc / max(n_blocks, 1)


# ─────────────────────────────────────────────────────────────────────
#  STEPS 6+7+8 — NOISE ESTIMATION AND THRESHOLD DECISION
# ─────────────────────────────────────────────────────────────────────

def detect_channels(ch_power: np.ndarray,
                    threshold_factor: float) -> tuple:
    """
    Estimate noise floor and apply threshold decision.

    Noise floor = median of 25 channel powers (robust to occupied channels).
    Threshold   = threshold_factor × noise_floor.
    Decision    = occupied if ch_power[k] > threshold.

    Returns (occupancy, noise_floor, threshold).
    """
    noise     = float(np.median(ch_power))
    threshold = threshold_factor * max(noise, 1e-30)
    occupancy = (ch_power > threshold).astype(np.uint8)
    return occupancy, noise, threshold


# ─────────────────────────────────────────────────────────────────────
#  FULL CHANNELISER
# ─────────────────────────────────────────────────────────────────────

def channelise(iq_int8: np.ndarray,
               threshold_factor: float = DEFAULT_THRESHOLD) -> tuple:
    """
    Run the complete PFB channeliser on one IQ sample.

    Steps: 1→2→3→4→5→6→7→8

    Returns
    -------
    occupancy   : (25,) uint8
    ch_power    : (25,) float64
    noise_floor : float
    threshold   : float
    """
    x         = prepare_signal(iq_int8)             # steps 1+2
    ch_power  = pfb_process(x)                      # step 5  (filter designed at import)
    occ, nf, th = detect_channels(ch_power, threshold_factor)  # steps 6+7+8
    return occ, ch_power, nf, th


# ─────────────────────────────────────────────────────────────────────
#  BATCH EVALUATION
# ─────────────────────────────────────────────────────────────────────

def run_on_dataset(sample_files: list,
                   threshold_factor: float = DEFAULT_THRESHOLD) -> tuple:
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
    p  = pred .flatten().astype(int)
    t  = truth.flatten().astype(int)
    TP = int(np.sum((p==1)&(t==1))); FP = int(np.sum((p==1)&(t==0)))
    TN = int(np.sum((p==0)&(t==0))); FN = int(np.sum((p==0)&(t==1)))
    P_d  = TP / max(TP+FN, 1)
    P_fa = FP / max(FP+TN, 1)
    prec = TP / max(TP+FP, 1)
    F1   = 2*prec*P_d / max(prec+P_d, 1e-10)
    return dict(TP=TP,FP=FP,TN=TN,FN=FN,P_d=P_d,P_fa=P_fa,precision=prec,F1=F1)


def pd_vs_snr(pred_arr, truth_arr, snr_arr, bw=2.0) -> dict:
    edges = np.arange(np.floor(snr_arr.min()/bw)*bw,
                      np.ceil (snr_arr.max()/bw)*bw+bw, bw)
    out = {}
    for i in range(len(edges)-1):
        lo,hi = edges[i],edges[i+1]
        mask  = (snr_arr>=lo)&(snr_arr<hi)
        if mask.sum()==0: continue
        m=compute_metrics(pred_arr[mask],truth_arr[mask])
        m["n"]=int(mask.sum())
        out[round((lo+hi)/2,1)]=m
    return out


# ─────────────────────────────────────────────────────────────────────
#  REPORTING
# ─────────────────────────────────────────────────────────────────────

def print_metrics(m: dict, label: str = ""):
    tag = f"[{label}]  " if label else ""
    print(f"\n{tag}PFB Channeliser — Metrics")
    print(f"  P_d   : {m['P_d']:.4f}  ({m['TP']} TP / {m['TP']+m['FN']} occupied)")
    print(f"  P_fa  : {m['P_fa']:.4f}  ({m['FP']} FP / {m['FP']+m['TN']} empty)")
    print(f"  Prec  : {m['precision']:.4f}")
    print(f"  F1    : {m['F1']:.4f}")
    print(f"  TP={m['TP']}  FP={m['FP']}  TN={m['TN']}  FN={m['FN']}")


def print_pd_snr(table: dict):
    print(f"\n  P_d vs SNR  (each █ ≈ 0.05)")
    print(f"  {'SNR':>7}  {'P_d':>7}  {'P_fa':>7}  {'F1':>7}  {'N':>5}")
    print("  " + "─"*50)
    for snr_c in sorted(table.keys()):
        m=table[snr_c]; bar="█"*int(m["P_d"]*20)
        print(f"  {snr_c:>7.1f}  {m['P_d']:>7.4f}  {m['P_fa']:>7.4f}"
              f"  {m['F1']:>7.4f}  {m['n']:>5}  {bar}")


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="PFB Channeliser — DSP baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset",   type=str,   default=None)
    ap.add_argument("--sample",    type=str,   default=None)
    ap.add_argument("--n-samples", type=int,   default=None)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--pd-snr",  action="store_true")
    ap.add_argument("--roc",     action="store_true")
    ap.add_argument("--filter-info", action="store_true",
                    help="Print prototype filter design details and exit")
    args = ap.parse_args()

    if args.filter_info:
        print(f"\nPrototype filter design:")
        print(f"  N channels    : {N_CH}")
        print(f"  L taps/branch : {L}")
        print(f"  Total taps    : {N_FILTER}")
        print(f"  Cutoff        : {SCAN_RATE/N_CH/2/1e3:.0f} kHz  "
              f"(= fs / (2N) = {SCAN_RATE/1e6:.0f} MS/s / {2*N_CH})")
        print(f"  Window        : Kaiser beta=8")
        print(f"  Polyphase G   : ({N_CH}, {L}) matrix")
        print(f"  G row 0 (first branch) : {_G[0].round(6)}")
        print(f"  G row 12 (centre)      : {_G[12].round(6)}")
        return

    # ── Single sample ──────────────────────────────────────────────────
    if args.sample:
        fpath = Path(args.sample)
        d     = np.load(str(fpath), allow_pickle=True).item()
        truth = d["occupancy"]
        imp_on = [IMP_NAMES[i] for i,v in enumerate(d["impairment_flags"]) if v]

        print(f"\n{'='*62}")
        print(f"PFB Channeliser — Step-by-Step")
        print(f"{'='*62}")
        print(f"  Sample     : {fpath.name}")
        print(f"  SNR        : {float(d['snr_db']):.1f} dB")
        print(f"  Impairments: {imp_on if imp_on else ['none']}")
        print(f"  Occupied   : {[i for i,v in enumerate(truth) if v]}")

        print(f"\nStep 1+2 — Form complex and decimate")
        x = prepare_signal(d["iq"])
        print(f"  {N_WINDOW} samples at {TARGET_RATE/1e6:.0f} MS/s → "
              f"{len(x)} samples at {SCAN_RATE/1e6:.0f} MS/s")

        print(f"\nStep 3+4 — Prototype filter (pre-computed at import)")
        print(f"  Filter length  : {N_FILTER} taps  ({L} per branch × {N_CH} channels)")
        print(f"  Cutoff         : {SCAN_RATE/N_CH/2/1e3:.0f} kHz")
        print(f"  Polyphase G    : ({N_CH}, {L})  — one row per channel")

        print(f"\nStep 5 — PFB processing")
        n_blocks = len(x) // N_CH
        print(f"  Input length   : {len(x)} samples")
        print(f"  Block size     : {N_CH} samples per block")
        print(f"  Blocks         : {n_blocks}")
        print(f"  Per block      : filter {N_CH} branches → FFT → {N_CH} channel outputs")
        print(f"  Output         : {n_blocks} × {N_CH} complex values")
        ch_pwr = pfb_process(x)
        print(f"  Mean |y_k|²   : averaged over {n_blocks} blocks")

        print(f"\nStep 6+7+8 — Noise floor and threshold")
        occ, nf, th = detect_channels(ch_pwr, args.threshold)
        print(f"  Noise floor    : {nf:.6f}  (median of {N_CH} channel powers)")
        print(f"  Threshold      : {args.threshold} × {nf:.6f} = {th:.6f}")
        print()
        print(f"  {'Ch':>4}  {'Centre':>10}  {'Power':>12}  {'Ratio':>8}  "
              f"{'Pred':>5}  {'Truth':>5}  Result")
        print(f"  {'─'*60}")
        for k in range(N_CH):
            if truth[k] or occ[k] or ch_pwr[k] > nf*1.3:
                f_c = (-2_400_000 + k*CH_BW_HZ)/1e3
                ratio = ch_pwr[k] / max(nf, 1e-30)
                if   truth[k]==1 and occ[k]==1: res="✓ TP"
                elif truth[k]==0 and occ[k]==1: res="✗ FP"
                elif truth[k]==1 and occ[k]==0: res="✗ FN"
                else:                            res="─ TN"
                print(f"  {k:>4}  {f_c:>+9.0f} kHz  {ch_pwr[k]:>12.6f}  "
                      f"{ratio:>8.2f}×  {occ[k]:>5}  {truth[k]:>5}  {res}")

        m = compute_metrics(occ.reshape(1,-1), truth.reshape(1,-1))
        print(f"\n  Predicted {occ.sum()} occupied / {int(truth.sum())} actual")
        print(f"  P_d={m['P_d']:.3f}  P_fa={m['P_fa']:.3f}  F1={m['F1']:.3f}")
        return

    # ── Dataset evaluation ──────────────────────────────────────────────
    if args.dataset is None:
        ap.print_help(); sys.exit(1)

    dataset_dir  = Path(args.dataset)
    sample_files = sorted((dataset_dir/"samples").glob("sample_*.npy"))
    if not sample_files:
        print(f"ERROR: no samples in {dataset_dir}/samples", file=sys.stderr)
        sys.exit(1)
    if args.n_samples:
        sample_files = sample_files[:args.n_samples]

    print(f"\nPFB Channeliser — DSP Baseline")
    print(f"  Dataset       : {dataset_dir}")
    print(f"  Samples       : {len(sample_files)}")
    print(f"  Threshold     : {args.threshold} × median noise floor")
    print(f"  Channels      : {N_CH}  ×  {CH_BW_HZ/1e3:.0f} kHz = "
          f"{N_CH*CH_BW_HZ/1e6:.0f} MHz")
    print(f"  Filter        : Kaiser-windowed sinc, {N_FILTER} taps, "
          f"{L} per branch")
    print(f"  Isolation     : ~30 dB inter-channel (L={L})")

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
        print(f"\n  ROC curve:")
        print(f"  {'k':>8}  {'P_fa':>8}  {'P_d':>8}  {'F1':>8}")
        print("  "+"─"*38)
        best = {"F1":-1}
        for k in THRESHOLD_SWEEP:
            p,t,_,_ = run_on_dataset(sample_files, k)
            mi = compute_metrics(p,t)
            print(f"  {k:>8.2f}  {mi['P_fa']:>8.4f}  {mi['P_d']:>8.4f}  {mi['F1']:>8.4f}")
            if mi["F1"] > best["F1"]: best={"k":k,**mi}
        print(f"\n  Best F1: threshold={best['k']:.2f}×  "
              f"P_fa={best['P_fa']:.4f}  P_d={best['P_d']:.4f}  F1={best['F1']:.4f}")

    print(f"\n  Per-impairment breakdown:")
    print(f"  {'Condition':25s}  {'P_d':>7}  {'P_fa':>7}  {'F1':>7}  {'N':>5}")
    print("  "+"─"*54)
    clean = flags_arr.sum(axis=1)==0
    if clean.sum()>0:
        mc=compute_metrics(pred_arr[clean],truth_arr[clean])
        print(f"  {'clean':25s}  {mc['P_d']:>7.4f}  {mc['P_fa']:>7.4f}"
              f"  {mc['F1']:>7.4f}  {clean.sum():>5}")
    for fi,name in enumerate(IMP_NAMES):
        mask=flags_arr[:,fi]==1
        if mask.sum()==0: continue
        mi=compute_metrics(pred_arr[mask],truth_arr[mask])
        print(f"  {name:25s}  {mi['P_d']:>7.4f}  {mi['P_fa']:>7.4f}"
              f"  {mi['F1']:>7.4f}  {mask.sum():>5}")


if __name__ == "__main__":
    main()
