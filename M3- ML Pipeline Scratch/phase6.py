"""
phase6_eval.py
==============
Phase 6 — Ablation study and final evaluation.

Ablation configurations
-----------------------
  A  FFT Channeliser (DSP baseline)      fft_channeliser.py
  B  Scalar MLP only  (Branch 3)         phase2 checkpoint
  C  Spectrogram CNN only (Branch 1)     phase6_configC checkpoint
  D  Full model (Branch 1+2+3)           phase5 checkpoint
  E  Spectral-only (Branch 1+3)          phase4 checkpoint
  F  Full model int8 quantised           derived from D

Outputs
-------
  - Ablation table: P_d / P_fa / F1 / PcAcc / ImpF1 per config
  - P_d vs SNR table for all configs
  - Per-impairment breakdown for configs A and D
  - Model size and inference time for all configs

Usage
-----
  # First train Config C (Branch 1 only)
  python phase6_eval.py --train-c --dataset ./gsm_dataset

  # Then run full evaluation
  python phase6_eval.py --eval-all --dataset ./gsm_dataset \\
      --ckpt-b ./checkpoints/best_model.pt \\
      --ckpt-c ./checkpoints_c/best_model.pt \\
      --ckpt-d ./checkpoints_p5/best_model.pt \\
      --ckpt-e ./checkpoints_p4/best_model.pt \\
      --norm-b ./checkpoints/normaliser.npz \\
      --norm-de ./checkpoints_p5/normaliser.npz

Dependencies: numpy, scipy, torch, phase1-5 files
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phase1  import split_dataset, N_SLOTS, N_IMP_FLAGS
from phase2    import (
    extract_scalar_features, FeatureNormaliser,
    N_SCALAR_FEATURES, Branch3MLP, ScalarFeatureDataset,
)
from phase3    import (
    MaskedMultiHeadLoss, compute_occupancy_metrics,
    compute_power_class_accuracy, compute_impairment_f1,
)
from phase4    import (
    extract_spectrogram, Branch1CNN, N_FRAMES,
    TARGET_RATE, SCAN_RATE, N_WINDOW, N_FFT, WELCH_HOP, SLOT_BINS,
)
from phase5    import (
    extract_fading_features, Branch2CNN,
    Phase5Dataset, FullFusedModel,
)
from scipy.signal import resample as fft_resample

N_POWER_CLASSES = 4

IMP_NAMES = [
    "clean", "cfo", "static_fading", "tv_fading",
    "cw_tone", "adj_channel", "wb_blocker",
]


# ─────────────────────────────────────────────────────────────────────
#  CONFIG C — BRANCH 1 ONLY MODEL
# ─────────────────────────────────────────────────────────────────────

class Branch1OnlyModel(nn.Module):
    """
    Config C: spectrogram CNN only — no scalar features, no fading.

    Input:  (batch, 25, 41, 5) spectrogram
    Output: per-slot occupancy + power class + gmsk + impairment

    The impairment head uses global average pooling over slots
    since there are no global scalar features in this config.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.branch1 = Branch1CNN(dropout=dropout)
        self.occupancy_head   = nn.Linear(64, 1)
        self.power_class_head = nn.Linear(64, N_POWER_CLASSES)
        self.gmsk_head        = nn.Linear(64, 1)
        # Impairment: average slot features → global
        self.impairment_head  = nn.Linear(64, N_IMP_FLAGS)

    def forward(self, spec: torch.Tensor) -> dict:
        B         = spec.shape[0]
        slot_feat = self.branch1(spec)            # (batch, 25, 64)
        flat      = slot_feat.view(B * N_SLOTS, 64)

        occ_logits  = self.occupancy_head(flat).view(B, N_SLOTS)
        pc_logits   = self.power_class_head(flat).view(B, N_SLOTS, N_POWER_CLASSES)
        gmsk_logits = self.gmsk_head(flat).view(B, N_SLOTS)

        # Global impairment from mean-pooled slot features
        global_feat = slot_feat.mean(dim=1)       # (batch, 64)
        imp_logits  = self.impairment_head(global_feat)

        return {
            "occupancy"  : occ_logits,
            "power_class": pc_logits,
            "gmsk"       : gmsk_logits,
            "impairment" : imp_logits,
        }


class SpectrogramOnlyDataset(torch.utils.data.Dataset):
    """Dataset for Config C — spectrogram only, no scalar features."""

    def __init__(self, sample_files: list):
        self.files = [Path(f) for f in sample_files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple:
        d    = np.load(str(self.files[idx]), allow_pickle=True).item()
        spec = extract_spectrogram(d["iq"])
        return (
            torch.from_numpy(spec),
            torch.from_numpy(d["occupancy"].astype(np.float32)),
            torch.from_numpy(d["power_class"].astype(np.int64)),
            torch.from_numpy(d["impairment_flags"].astype(np.float32)),
            torch.tensor(float(d["snr_db"]), dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────
#  TRAIN CONFIG C
# ─────────────────────────────────────────────────────────────────────

def train_config_c(dataset_dir, n_samples, epochs, lr,
                   batch_size, num_workers, dropout,
                   ckpt_dir, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    train_base, val_base, test_base = split_dataset(
        dataset_dir, n_samples=n_samples, seed=seed
    )
    print(f"Config C — Branch 1 only")
    print(f"  Train:{len(train_base)}  Val:{len(val_base)}  Test:{len(test_base)}")

    train_ds = SpectrogramOnlyDataset(train_base.files)
    val_ds   = SpectrogramOnlyDataset(val_base.files)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    model     = Branch1OnlyModel(dropout=dropout)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion = MaskedMultiHeadLoss(w_occ=1.0, w_pc=0.5, w_gmsk=0.3, w_imp=0.5)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", patience=5, factor=0.5
    )

    best_f1 = -1.0; es_count = 0

    print(f"\n{'Ep':>4}  {'TrLoss':>7}  {'TrF1':>6}  {'VaLoss':>7}  "
          f"{'VaF1':>6}  {'VaPd':>6}  {'VaPfa':>6}  {'LR':>8}")
    print("─" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train(); total_loss = 0.0
        all_ol=[]; all_ot=[]

        for spec, occ, pc, imp, _ in train_loader:
            optimiser.zero_grad()
            preds = model(spec)
            loss, _ = criterion(preds, occ, pc, imp)
            loss.backward(); optimiser.step()
            total_loss += loss.item() * len(spec)
            all_ol.append(preds["occupancy"].detach()); all_ot.append(occ)

        occ_l = torch.cat(all_ol); occ_t = torch.cat(all_ot)
        tr_m  = compute_occupancy_metrics(occ_l, occ_t)
        tr_m["loss"] = total_loss / len(train_loader.dataset)

        # Validation
        model.eval(); vl = 0.0; vol=[]; vot=[]
        with torch.no_grad():
            for spec, occ, pc, imp, _ in val_loader:
                preds = model(spec)
                loss, _ = criterion(preds, occ, pc, imp)
                vl += loss.item()*len(spec)
                vol.append(preds["occupancy"]); vot.append(occ)
        va_l=torch.cat(vol); va_t=torch.cat(vot)
        va_m = compute_occupancy_metrics(va_l, va_t)
        va_m["loss"] = vl / len(val_loader.dataset)
        scheduler.step(va_m["F1"])

        elapsed = time.perf_counter() - t0
        lr_now  = optimiser.param_groups[0]["lr"]
        print(f"{epoch:>4}  {tr_m['loss']:>7.4f}  {tr_m['F1']:>6.4f}  "
              f"{va_m['loss']:>7.4f}  {va_m['F1']:>6.4f}  "
              f"{va_m['P_d']:>6.4f}  {va_m['P_fa']:>6.4f}  "
              f"{lr_now:>8.2e}  ({elapsed:.1f}s)")

        if va_m["F1"] > best_f1:
            best_f1 = va_m["F1"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_F1": best_f1},
                       str(Path(ckpt_dir) / "best_model.pt"))
            es_count = 0
        else:
            es_count += 1
            if es_count >= 10:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nConfig C checkpoint saved to {ckpt_dir}/best_model.pt")


# ─────────────────────────────────────────────────────────────────────
#  EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_model_generic(model, loader, model_type, threshold=0.5) -> dict:
    """
    Evaluate any model variant. model_type controls which inputs to pass.
    Types: 'scalar', 'spectrogram', 'phase4', 'phase5'
    """
    model.eval()
    all_occ_l=[]; all_occ_t=[]
    all_pc_l=[];  all_pc_t=[]
    all_imp_l=[]; all_imp_t=[]
    all_snr=[]; all_flags=[]

    for batch in loader:
        if model_type == 'scalar':
            scalars, occ, snr = batch
            preds = {"occupancy": model(scalars),
                     "power_class": torch.zeros(scalars.shape[0], N_SLOTS, N_POWER_CLASSES),
                     "gmsk": torch.zeros(scalars.shape[0], N_SLOTS),
                     "impairment": torch.zeros(scalars.shape[0], N_IMP_FLAGS)}
            pc = torch.zeros(scalars.shape[0], N_SLOTS, dtype=torch.int64)
            imp = torch.zeros(scalars.shape[0], N_IMP_FLAGS)
        elif model_type == 'spectrogram':
            spec, occ, pc, imp, snr = batch
            preds = model(spec)
        elif model_type == 'phase4':
            scalars, spec, occ, pc, imp, snr = batch
            preds = model(scalars, spec)
        else:  # phase5
            scalars, spec, fading, occ, pc, imp, snr = batch
            preds = model(scalars, spec, fading)

        all_occ_l.append(preds["occupancy"]); all_occ_t.append(occ)
        all_pc_l.append(preds["power_class"]); all_pc_t.append(pc)
        all_imp_l.append(preds["impairment"]); all_imp_t.append(imp)
        all_snr.append(snr)
        all_flags.append(imp)

    occ_l = torch.cat(all_occ_l); occ_t = torch.cat(all_occ_t)
    pc_l  = torch.cat(all_pc_l);  pc_t  = torch.cat(all_pc_t)
    imp_l = torch.cat(all_imp_l); imp_t = torch.cat(all_imp_t)
    snr_t = torch.cat(all_snr)
    flags = torch.cat(all_flags)

    occ_m  = compute_occupancy_metrics(occ_l, occ_t, threshold)
    pc_acc = compute_power_class_accuracy(pc_l, pc_t, occ_t.bool())
    imp_f1 = compute_impairment_f1(imp_l, imp_t)

    m = {**occ_m, "pc_acc": pc_acc, "imp_f1": imp_f1}

    # P_d vs SNR
    for lo, hi in [(-12,-4),(-4,0),(0,6),(6,12),(12,20),(20,28)]:
        mask = (snr_t>=lo)&(snr_t<hi)
        if mask.sum()>0:
            mi = compute_occupancy_metrics(occ_l[mask], occ_t[mask], threshold)
            m[f"pd_{lo}_{hi}"] = mi["P_d"]

    # Per-impairment (flag index 0=cfo,1=static,2=tv,3=cw,4=adj,5=wb)
    # flags here is imp_t (the ground truth impairment flags)
    clean_mask = (imp_t.sum(dim=1)==0)
    if clean_mask.sum()>0:
        mc = compute_occupancy_metrics(occ_l[clean_mask], occ_t[clean_mask], threshold)
        m["imp_clean_pd"] = mc["P_d"]; m["imp_clean_pfa"] = mc["P_fa"]
    for fi, name in enumerate(IMP_NAMES[1:]):
        mask = imp_t[:,fi]==1
        if mask.sum()>0:
            mi = compute_occupancy_metrics(occ_l[mask], occ_t[mask], threshold)
            m[f"imp_{name}_pd"]  = mi["P_d"]
            m[f"imp_{name}_pfa"] = mi["P_fa"]

    return m


def tune_threshold(model, loader, model_type):
    """Find best threshold on val set with P_fa <= 0.05."""
    best_thresh = 0.5; best_f1 = -1.0
    for thresh in np.arange(0.30, 0.81, 0.05):
        m = eval_model_generic(model, loader, model_type, float(thresh))
        if m["P_fa"] <= 0.05 and m["F1"] > best_f1:
            best_f1 = m["F1"]; best_thresh = float(thresh)
    return best_thresh


# ─────────────────────────────────────────────────────────────────────
#  FFT CHANNELISER EVALUATION (Config A)
# ─────────────────────────────────────────────────────────────────────

def eval_fft_baseline(test_files, snr_arr, flags_arr) -> dict:
    """Re-evaluate FFT channeliser on the same test split."""
    from fft_channeliser import channelise, compute_metrics as fft_metrics

    pred_list=[]; truth_list=[]
    for fpath in test_files:
        d = np.load(str(fpath), allow_pickle=True).item()
        occ, _, _, _ = channelise(d["iq"], threshold_factor=1.38)
        pred_list.append(occ)
        truth_list.append(d["occupancy"].astype(np.int32))

    pred_arr  = np.array(pred_list,  dtype=np.int32)
    truth_arr = np.array(truth_list, dtype=np.int32)

    m = fft_metrics(pred_arr, truth_arr)

    result = {
        "P_d": m["P_d"], "P_fa": m["P_fa"], "F1": m["F1"],
        "TP": m["TP"], "FP": m["FP"], "TN": m["TN"], "FN": m["FN"],
        "pc_acc": 0.0, "imp_f1": 0.0,
    }

    snr_t = torch.tensor(snr_arr)
    pred_t  = torch.tensor(pred_arr.astype(np.float32))
    truth_t = torch.tensor(truth_arr.astype(np.float32))

    for lo, hi in [(-12,-4),(-4,0),(0,6),(6,12),(12,20),(20,28)]:
        mask = (snr_t>=lo)&(snr_t<hi)
        if mask.sum()>0:
            mi = compute_occupancy_metrics(pred_t[mask], truth_t[mask], 0.5)
            result[f"pd_{lo}_{hi}"] = mi["P_d"]

    flags_t = torch.tensor(flags_arr.astype(np.float32))
    clean = (flags_t.sum(dim=1)==0)
    if clean.sum()>0:
        mc = compute_occupancy_metrics(pred_t[clean], truth_t[clean], 0.5)
        result["imp_clean_pd"] = mc["P_d"]
    for fi, name in enumerate(IMP_NAMES[1:]):
        mask = flags_t[:,fi]==1
        if mask.sum()>0:
            mi = compute_occupancy_metrics(pred_t[mask], truth_t[mask], 0.5)
            result[f"imp_{name}_pd"] = mi["P_d"]

    return result


# ─────────────────────────────────────────────────────────────────────
#  INT8 QUANTISATION (Config F)
# ─────────────────────────────────────────────────────────────────────

def quantise_model(model: nn.Module) -> nn.Module:
    """
    Apply dynamic int8 quantisation to the model.

    Dynamic quantisation converts Linear, Conv1d, Conv2d weights
    to int8. Activations are quantised at runtime. No retraining
    or calibration data needed — this is post-training quantisation.

    The quantised model is smaller and faster at inference while
    typically losing <1 pp F1.
    """
    quantised = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.Conv1d, nn.Conv2d},
        dtype=torch.qint8,
    )
    return quantised


def model_size_kb(model: nn.Module) -> float:
    """Estimate model size by saving to a buffer and measuring bytes."""
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024.0


# ─────────────────────────────────────────────────────────────────────
#  REPORTING
# ─────────────────────────────────────────────────────────────────────

def print_ablation_table(results: dict):
    print("\n" + "="*80)
    print("ABLATION STUDY — ALL CONFIGURATIONS")
    print("="*80)
    print(f"\n{'Config':>8}  {'Name':30s}  {'P_d':>7}  {'P_fa':>7}  "
          f"{'F1':>7}  {'PcAcc':>7}  {'ImpF1':>7}  {'Params':>8}")
    print("─"*80)

    for cfg, (name, m, params) in results.items():
        print(f"  {cfg:>6}  {name:30s}  {m['P_d']:>7.4f}  {m['P_fa']:>7.4f}  "
              f"{m['F1']:>7.4f}  {m['pc_acc']:>7.4f}  {m['imp_f1']:>7.4f}  "
              f"{params:>8}")


def print_pd_vs_snr_table(results: dict):
    snr_keys = [(-12,-4),(-4,0),(0,6),(6,12),(12,20),(20,28)]
    print("\n" + "="*80)
    print("P_d vs SNR — ALL CONFIGURATIONS")
    print("="*80)
    header = f"{'Config':>8}"
    for lo, hi in snr_keys:
        header += f"  [{lo:>4},{hi:>3})"
    print(header)
    print("─"*80)

    for cfg, (name, m, _) in results.items():
        row = f"  {cfg:>6}"
        for lo, hi in snr_keys:
            val = m.get(f"pd_{lo}_{hi}", float("nan"))
            row += f"  {val:>8.4f}"
        print(row)


def print_per_impairment(results: dict, configs=("A", "D")):
    print("\n" + "="*80)
    print("PER-IMPAIRMENT P_d BREAKDOWN")
    print("="*80)
    print(f"\n{'Condition':20s}", end="")
    for cfg in configs:
        name = results[cfg][0][:12]
        print(f"  {name:>15}", end="")
    print()
    print("─"*60)

    conditions = ["clean"] + IMP_NAMES[1:]
    for cond in conditions:
        key = f"imp_{cond}_pd"
        print(f"  {cond:18s}", end="")
        for cfg in configs:
            val = results[cfg][1].get(key, float("nan"))
            print(f"  {val:>15.4f}", end="")
        print()


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phase 6 — Ablation evaluation")
    ap.add_argument("--dataset",     type=str, required=True)
    ap.add_argument("--n-samples",   type=int, default=None)
    ap.add_argument("--seed",        type=int, default=42)

    # Mode
    ap.add_argument("--train-c",  action="store_true",
                    help="Train Config C (Branch 1 only) and exit")
    ap.add_argument("--eval-all", action="store_true",
                    help="Run full ablation evaluation")

    # Checkpoints
    ap.add_argument("--ckpt-b",    type=str, default="./checkpoints/best_model.pt")
    ap.add_argument("--ckpt-c",    type=str, default="./checkpoints_c/best_model.pt")
    ap.add_argument("--ckpt-d",    type=str, default="./checkpoints_p5/best_model.pt")
    ap.add_argument("--ckpt-e",    type=str, default="./checkpoints_p4/best_model.pt")
    ap.add_argument("--norm-b",    type=str, default="./checkpoints/normaliser.npz")
    ap.add_argument("--norm-de",   type=str, default="./checkpoints_p5/normaliser.npz")

    # Training params for Config C
    ap.add_argument("--ckpt-c-dir",  type=str,   default="./checkpoints_c")
    ap.add_argument("--epochs",      type=int,   default=100)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--batch-size",  type=int,   default=32)
    ap.add_argument("--num-workers", type=int,   default=2)
    ap.add_argument("--dropout",     type=float, default=0.3)

    args = ap.parse_args()

    # ── Train Config C ─────────────────────────────────────────────────
    if args.train_c:
        train_config_c(
            args.dataset, args.n_samples,
            args.epochs, args.lr, args.batch_size,
            args.num_workers, args.dropout,
            args.ckpt_c_dir, args.seed,
        )
        return

    # ── Full evaluation ────────────────────────────────────────────────
    if not args.eval_all:
        ap.print_help(); return

    # Build test split (same seed → same split as training)
    _, val_base, test_base = split_dataset(
        args.dataset, n_samples=args.n_samples, seed=args.seed
    )
    test_files = test_base.files

    # Collect SNR and flags arrays for FFT evaluation
    snr_arr   = np.array([float(np.load(str(f), allow_pickle=True).item()["snr_db"])
                          for f in test_files])
    flags_arr = np.array([np.load(str(f), allow_pickle=True).item()["impairment_flags"]
                          for f in test_files])

    results = {}   # cfg -> (name, metrics_dict, params_str)

    # ── Config A: FFT Channeliser ──────────────────────────────────────
    print("\nEvaluating Config A (FFT Channeliser)...")
    m_a = eval_fft_baseline(test_files, snr_arr, flags_arr)
    results["A"] = ("FFT Channeliser", m_a, "N/A")

    # ── Config B: Scalar MLP ───────────────────────────────────────────
    print("Evaluating Config B (Scalar MLP)...")
    norm_b = FeatureNormaliser(); norm_b.load(args.ckpt_b.replace("best_model.pt","normaliser.npz"))
    model_b = Branch3MLP(); model_b.load_state_dict(torch.load(args.ckpt_b, map_location="cpu")["model_state"])
    model_b.eval()
    test_ds_b  = ScalarFeatureDataset(test_files, norm_b)
    loader_b   = DataLoader(test_ds_b, batch_size=32, shuffle=False, num_workers=2)
    thresh_b   = tune_threshold(model_b, DataLoader(
        ScalarFeatureDataset(val_base.files, norm_b), batch_size=32), "scalar")
    m_b = eval_model_generic(model_b, loader_b, "scalar", thresh_b)
    params_b = f"{sum(p.numel() for p in model_b.parameters()):,}"
    results["B"] = ("Scalar MLP (B3)", m_b, params_b)

    # ── Config C: Spectrogram CNN only ─────────────────────────────────
    print("Evaluating Config C (Spectrogram CNN only)...")
    model_c = Branch1OnlyModel()
    model_c.load_state_dict(torch.load(args.ckpt_c, map_location="cpu")["model_state"])
    model_c.eval()
    test_ds_c = SpectrogramOnlyDataset(test_files)
    loader_c  = DataLoader(test_ds_c, batch_size=32, shuffle=False, num_workers=2)
    thresh_c  = tune_threshold(model_c, DataLoader(
        SpectrogramOnlyDataset(val_base.files), batch_size=32), "spectrogram")
    m_c = eval_model_generic(model_c, loader_c, "spectrogram", thresh_c)
    params_c = f"{sum(p.numel() for p in model_c.parameters()):,}"
    results["C"] = ("Spectrogram CNN (B1)", m_c, params_c)

    # ── Config D: Full model (Phase 5) ─────────────────────────────────
    print("Evaluating Config D (Full model B1+B2+B3)...")
    norm_de = FeatureNormaliser(); norm_de.load(args.norm_de)
    model_d = FullFusedModel()
    model_d.load_state_dict(torch.load(args.ckpt_d, map_location="cpu")["model_state"])
    model_d.eval()
    test_ds_d = Phase5Dataset(test_files, norm_de)
    loader_d  = DataLoader(test_ds_d, batch_size=32, shuffle=False, num_workers=2)
    thresh_d  = tune_threshold(model_d, DataLoader(
        Phase5Dataset(val_base.files, norm_de), batch_size=32), "phase5")
    m_d = eval_model_generic(model_d, loader_d, "phase5", thresh_d)
    params_d = f"{sum(p.numel() for p in model_d.parameters()):,}"
    results["D"] = ("Full model (B1+B2+B3)", m_d, params_d)

    # ── Config E: Spectral-only (Phase 4) ──────────────────────────────
    print("Evaluating Config E (Spectral-only B1+B3)...")
    from phase4 import FusedModel, Phase4Dataset
    model_e = FusedModel()
    model_e.load_state_dict(torch.load(args.ckpt_e, map_location="cpu")["model_state"])
    model_e.eval()
    norm_e = FeatureNormaliser(); norm_e.load(args.ckpt_e.replace("best_model.pt","normaliser.npz"))
    test_ds_e = Phase4Dataset(test_files, norm_e)
    loader_e  = DataLoader(test_ds_e, batch_size=32, shuffle=False, num_workers=2)
    thresh_e  = tune_threshold(model_e, DataLoader(
        Phase4Dataset(val_base.files, norm_e), batch_size=32), "phase4")
    m_e = eval_model_generic(model_e, loader_e, "phase4", thresh_e)
    params_e = f"{sum(p.numel() for p in model_e.parameters()):,}"
    results["E"] = ("Spectral B1+B3", m_e, params_e)

    # ── Config F: Int8 quantised ───────────────────────────────────────
    print("Evaluating Config F (int8 quantised)...")
    model_f    = quantise_model(FullFusedModel())
    # Load the float32 weights first, then quantise
    model_f_fp = FullFusedModel()
    model_f_fp.load_state_dict(torch.load(args.ckpt_d, map_location="cpu")["model_state"])
    model_f    = quantise_model(model_f_fp)
    model_f.eval()
    size_fp32  = model_size_kb(model_f_fp)
    size_int8  = model_size_kb(model_f)

    # Inference time comparison
    dummy_s = torch.zeros(1, N_SCALAR_FEATURES)
    dummy_sp= torch.zeros(1, N_SLOTS, 41, N_FRAMES)
    dummy_f = torch.zeros(1, 2, N_FFT)
    model_f_fp.eval()
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad(): model_f_fp(dummy_s, dummy_sp, dummy_f)
    time_fp32 = (time.perf_counter()-t0)/100*1000

    model_f.eval()
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad(): model_f(dummy_s, dummy_sp, dummy_f)
    time_int8 = (time.perf_counter()-t0)/100*1000

    m_f = eval_model_generic(model_f, loader_d, "phase5", thresh_d)
    results["F"] = (f"Int8 quantised", m_f, f"{sum(p.numel() for p in model_f_fp.parameters()):,}")

    # ── Print results ──────────────────────────────────────────────────
    print_ablation_table(results)
    print_pd_vs_snr_table(results)
    print_per_impairment(results, configs=["A","D"])

    print("\n" + "="*80)
    print("MODEL SIZE AND INFERENCE TIME")
    print("="*80)
    print(f"  Float32 model : {size_fp32:.1f} KB  |  {time_fp32:.2f} ms/inference")
    print(f"  Int8 model    : {size_int8:.1f} KB  |  {time_int8:.2f} ms/inference")
    print(f"  Size reduction: {size_fp32/size_int8:.1f}x")
    print(f"  Speed change  : {time_fp32/time_int8:.1f}x")

    print("\n" + "="*80)
    print("PROJECT TARGETS vs BEST MODEL (Config D)")
    print("="*80)
    md = results["D"][1]
    targets = [
        ("P_fa <= 0.05",       md["P_fa"]   <= 0.05,   f"{md['P_fa']:.4f}"),
        ("Power class >= 0.75",md["pc_acc"] >= 0.75,   f"{md['pc_acc']:.4f}"),
        ("Imp macro-F1 >= 0.70",md["imp_f1"]>= 0.70,   f"{md['imp_f1']:.4f}"),
        ("F1 >= 0.85",         md["F1"]     >= 0.85,   f"{md['F1']:.4f}"),
        ("P_d >= 0.90 @SNR>=0",md.get("pd_0_6",0)>=0.90, f"{md.get('pd_0_6',0):.4f}"),
    ]
    for name, met, val in targets:
        tick = "MET" if met else "NOT YET"
        print(f"  [{tick:>7}]  {name:30s}  actual={val}")


if __name__ == "__main__":
    main()