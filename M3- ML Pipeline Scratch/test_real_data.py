"""
test_real_data.py
=================
Test the trained Phase 5 model on real captured GSM data.

The .cfile was captured at 4 MS/s. The model expects input at 104 MS/s.
Preprocessing: upsample 4 MS/s -> 104 MS/s, scale to int8, then let
the model pipeline decimate back to 5 MS/s internally.

Ground truth (from FFT analysis):
  Slot 12 = real carrier (1268x above noise floor)
  Slots 9-11, 13-16 = leakage from slot 12
  All other visible slots = empty
"""

import sys, numpy as np, torch
from pathlib import Path
from scipy.signal import resample as fft_resample

sys.path.insert(0, '/home/maitreyee/Work')
from phase2  import extract_scalar_features, FeatureNormaliser, N_SCALAR_FEATURES
from phase4  import extract_spectrogram, N_SLOTS, N_FRAMES
from phase5  import extract_fading_features, FullFusedModel

# ── Config ────────────────────────────────────────────────────────
CFILE = '/home/maitreyee/Work/vf_call6_a725_d174_g5_Kc1EF00BAB3BAC7002.cfile'
CKPT_D  = '/home/maitreyee/Work/checkpoints_p5/best_model.pt'
NORM_DE = '/home/maitreyee/Work/checkpoints_p5/normaliser.npz'
REAL_SR    = 4_000_000
MODEL_SR   = 104_000_000
N_WINDOW   = 65536
N_IN       = int(round(N_WINDOW * REAL_SR / MODEL_SR))   # 2521 samples
THRESHOLD  = 0.50

# ── Load model ────────────────────────────────────────────────────
print("Loading model and normaliser...")
normaliser = FeatureNormaliser()
normaliser.load(NORM_DE)

model = FullFusedModel()
ckpt  = torch.load(CKPT_D, map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters : {n_params:,}")
print(f"  Best epoch : {ckpt['epoch']}")
print(f"  Val F1     : {ckpt['val_F1']:.4f}")

# ── Load real data ────────────────────────────────────────────────
print(f"\nLoading real data...")
data = np.fromfile(CFILE, dtype=np.complex64)
print(f"  Total samples : {len(data):,}")
print(f"  Windows of {N_IN} samples: {len(data)//N_IN}")

# ── Preprocess one window -> (2, 65536) int8 ─────────────────────
def preprocess_window(chunk_complex64):
    """
    Convert one window of real 4 MS/s float32 IQ data
    into the (2, 65536) int8 format the model expects.

    Steps:
    1. Upsample 4 MS/s -> 104 MS/s (model pipeline decimates back down)
    2. Scale to int8 range using 95th percentile
    3. Clip and cast to int8
    """
    resampled = fft_resample(chunk_complex64.astype(np.complex64), N_WINDOW)
    p95       = float(np.percentile(np.abs(resampled), 95))
    scale     = 10.0 / max(p95, 1e-10)
    scaled    = resampled * scale
    I = np.clip(scaled.real, -16, 15).astype(np.int8)
    Q = np.clip(scaled.imag, -16, 15).astype(np.int8)
    return np.stack([I, Q])   # (2, 65536)

# ── Run model on multiple windows ─────────────────────────────────
print(f"\nRunning model on real data (threshold={THRESHOLD})...")
print(f"\n{'Win':>4}  {'Model detects':30s}  {'FFT detects':30s}  {'Match?':>6}")
print("-" * 80)

N_WINDOWS_TO_TEST = 20

for w in range(N_WINDOWS_TO_TEST):
    chunk  = data[w*N_IN : (w+1)*N_IN]
    iq     = preprocess_window(chunk)

    # Extract all three feature types
    scalars = extract_scalar_features(iq)
    scalars = normaliser.transform(scalars)
    spec    = extract_spectrogram(iq)
    fading  = extract_fading_features(iq)

    # Model prediction
    with torch.no_grad():
        s_t  = torch.from_numpy(scalars).unsqueeze(0)
        sp_t = torch.from_numpy(spec).unsqueeze(0)
        f_t  = torch.from_numpy(fading).unsqueeze(0)
        out    = model(s_t, sp_t, f_t)
        logits = out["occupancy"]
        probs  = torch.sigmoid(logits).squeeze().numpy()

    model_detected = [k for k in range(25) if probs[k] > THRESHOLD]

    # FFT channeliser on same window
    x_float = iq[0].astype(np.float32) + 1j*iq[1].astype(np.float32)
    n_dec   = int(round(N_WINDOW * 5_000_000 / MODEL_SR))
    x_dec   = fft_resample(x_float, n_dec)
    win_h   = np.hanning(1024)
    acc     = np.zeros(1024)
    for i in range(5):
        seg = x_dec[i*512:i*512+1024]
        s   = np.fft.fftshift(np.fft.fft(seg*win_h, n=1024))
        acc += np.abs(s)**2/1024
    pwr_w   = acc/5
    BIN_HZ  = 5_000_000/1024
    slot_e  = {}
    for k in range(25):
        c  = int(round((-2_400_000+k*200_000+5_000_000/2)/BIN_HZ))
        lo = max(0,c-20); hi=min(1024,c+21)
        slot_e[k] = float(np.sum(pwr_w[lo:hi]))
    nf      = float(np.median(list(slot_e.values())))
    fft_det = [k for k,e in slot_e.items() if e > 1.38*nf]

    match = "YES" if set(model_detected) == set(fft_det) else "---"
    print(f"  {w:3d}  {str(model_detected):30s}  {str(fft_det):30s}  {match:>6}")

# ── Detailed probability breakdown for window 0 ───────────────────
print(f"\n{'='*60}")
print(f"Detailed slot probabilities — window 0")
print(f"{'='*60}")
chunk  = data[:N_IN]
iq     = preprocess_window(chunk)
scalars = normaliser.transform(extract_scalar_features(iq))
spec    = extract_spectrogram(iq)
fading  = extract_fading_features(iq)

with torch.no_grad():
    out    = model(
    torch.from_numpy(scalars).unsqueeze(0),
    torch.from_numpy(spec).unsqueeze(0),
    torch.from_numpy(fading).unsqueeze(0),
)
probs = torch.sigmoid(out["occupancy"]).squeeze().numpy()

print(f"\n{'Slot':>5}  {'Prob':>8}  {'Pred':>8}  {'Note'}")
print("-" * 45)
for k in range(25):
    pred = "CARRIER" if probs[k] > THRESHOLD else "empty"
    note = ""
    if k == 12:   note = "<-- real carrier"
    elif 9 <= k <= 16 and k != 12: note = "<-- leakage zone"
    print(f"  {k:3d}  {probs[k]:>8.4f}  {pred:>8}  {note}")

print(f"\nKey question: does the model give slot 12 the highest probability?")
top3 = sorted(range(25), key=lambda k: probs[k], reverse=True)[:3]
print(f"Top 3 slots by probability: {top3}")
print(f"Slot 12 probability: {probs[12]:.4f}")

# ── Inspect suspicious slots ──────────────────────────────────────
print(f"\nSpectral shape comparison (window 0):")
print(f"{'Bin':>4}  {'Slot 2':>8}  {'Slot 12':>8}  {'Slot 22':>8}")
print("-" * 35)

slot2_spec  = extract_spectrogram(iq)[2]    # (41, 5)
slot12_spec = extract_spectrogram(iq)[12]
slot22_spec = extract_spectrogram(iq)[22]

for b in range(41):
    v2  = slot2_spec[b].mean()
    v12 = slot12_spec[b].mean()
    v22 = slot22_spec[b].mean()
    print(f"  {b:2d}  {v2:>8.2f}  {v12:>8.2f}  {v22:>8.2f}")