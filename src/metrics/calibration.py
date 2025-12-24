from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    return float(np.mean((y_prob - y_true) ** 2))


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += float(np.abs(acc - conf)) * (float(np.sum(mask)) / float(len(y_prob)))
    return float(ece)


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    acc = np.zeros_like(bin_centers)
    conf = np.zeros_like(bin_centers)
    counts = np.zeros_like(bin_centers)
    for i in range(len(bin_centers)):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            acc[i] = float(np.mean(y_true[mask]))
            conf[i] = float(np.mean(y_prob[mask]))
    return bin_centers, acc, conf


def plot_reliability(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10, out_png: str
) -> None:
    bin_centers, acc, conf = reliability_curve(y_true, y_prob, n_bins=n_bins)
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
    plt.plot(conf, acc, marker="o", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
