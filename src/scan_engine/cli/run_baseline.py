from __future__ import annotations

import numpy as np

from ..baselines.fft_energy import detect_occupancy
from ..data.generator import iter_dataset
from ..eval.metrics import detection_stats
from ._config import load_config


def main() -> None:
    cfg = load_config()
    fs = cfg["signal"]["fs"]
    n_channels = cfg["signal"]["n_channels"]
    thresh_db = cfg["baseline"]["threshold_db"]
    y_true, y_pred = [], []
    for iq, occ in iter_dataset(cfg, "test"):
        pred = detect_occupancy(iq, fs=fs, n_channels=n_channels, threshold_db=thresh_db)
        y_true.append(occ)
        y_pred.append(pred)
    stats = detection_stats(np.stack(y_true), np.stack(y_pred))
    print(f"[run_baseline] fft_energy thr={thresh_db}dB: " +
          " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
