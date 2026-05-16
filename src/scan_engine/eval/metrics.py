"""Detection metrics for per-channel occupancy."""
from __future__ import annotations

import numpy as np


def detection_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = y_true.astype(bool).ravel()
    yp = y_pred.astype(bool).ravel()
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    tn = int(np.sum(~yt & ~yp))
    pd = tp / max(tp + fn, 1)
    far = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = pd
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"pd": pd, "far": far, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}
