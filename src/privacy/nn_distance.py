from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import TrajectoryBatch


def _masked_stats(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = mask.astype(np.float32)
    denom = mask.sum(axis=1, keepdims=True)
    denom = np.maximum(denom, 1.0)
    mean = (x * mask[..., None]).sum(axis=1) / denom
    var = ((x - mean[:, None, :]) ** 2 * mask[..., None]).sum(axis=1) / denom
    return mean, np.sqrt(var)


def _normalize_embed_space(embed_space: str) -> str:
    space = str(embed_space or "y_only").strip().lower()
    if space in {"y_only", "yonly", "y"}:
        return "y_only"
    if space in {"all", "full"}:
        return "all"
    raise ValueError(f"Unknown embed_space={embed_space}")


def _embed(batch: TrajectoryBatch, *, embed_space: str = "y_only") -> np.ndarray:
    X = batch.X.detach().cpu().numpy().astype(np.float32)
    A = batch.A.detach().cpu().numpy()
    T = batch.T.detach().cpu().numpy()
    Y = batch.Y.detach().cpu().numpy()
    M = batch.mask.detach().cpu().numpy()

    space = _normalize_embed_space(embed_space)
    denom = np.maximum(1.0, M.sum(axis=1, keepdims=True))

    y_mean = (Y * M).sum(axis=1, keepdims=True) / denom
    y_std = np.sqrt(((Y - y_mean) ** 2 * M).sum(axis=1, keepdims=True) / denom)
    lengths = M.sum(axis=1, keepdims=True) / max(1.0, M.shape[1])

    if space == "y_only":
        lengths_int = np.maximum(M.sum(axis=1).astype(np.int64), 1)
        last_idx = lengths_int - 1
        y_last = Y[np.arange(Y.shape[0]), last_idx][:, None]
        return np.concatenate([y_mean, y_std, y_last, lengths], axis=1).astype(np.float32)

    if A.ndim == 2:
        a_feat = (A * M).sum(axis=1, keepdims=True) / denom
    else:
        a_mean, a_std = _masked_stats(A, M)
        a_feat = np.concatenate([a_mean, a_std], axis=1)

    t_feat = (T * M).sum(axis=1, keepdims=True) / denom
    return np.concatenate([X, a_feat, t_feat, y_mean, y_std, lengths], axis=1).astype(np.float32)


def compute_nn_distance(
    *,
    real: TrajectoryBatch,
    synth: TrajectoryBatch,
    metric: str = "embedding_l2",
    encoder: callable | None = None,
    embed_space: str = "y_only",
) -> pd.DataFrame:
    if metric not in {"embedding_l2", "dtw"}:
        raise ValueError(f"Unknown metric={metric}")

    if metric == "dtw":
        real_y = real.Y.detach().cpu().numpy()
        synth_y = synth.Y.detach().cpu().numpy()
        real_m = real.mask.detach().cpu().numpy()
        synth_m = synth.mask.detach().cpu().numpy()

        def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
            n, m = a.shape[0], b.shape[0]
            dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
            dtw[0, 0] = 0.0
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = abs(a[i - 1] - b[j - 1])
                    dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
            return float(dtw[n, m])

        rows = []
        for i in range(synth_y.shape[0]):
            sy = synth_y[i][synth_m[i] > 0.5]
            best_j = -1
            best_d = np.inf
            for j in range(real_y.shape[0]):
                ry = real_y[j][real_m[j] > 0.5]
                d = dtw_distance(sy, ry)
                if d < best_d:
                    best_d = d
                    best_j = j
            rows.append(
                {
                    "model": "unknown",
                    "dataset": "unknown",
                    "metric": metric,
                    "synth_id": int(i),
                    "nn_real_id": int(best_j),
                    "distance": float(best_d),
                }
            )
        return pd.DataFrame(rows)

    if encoder is None:
        embed = lambda batch: _embed(batch, embed_space=embed_space)
    else:
        embed = encoder

    real_emb = embed(real)
    synth_emb = embed(synth)

    rows = []
    for i in range(synth_emb.shape[0]):
        diff = real_emb - synth_emb[i][None, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        j = int(np.argmin(dist))
        rows.append(
            {
                "model": "unknown",
                "dataset": "unknown",
                "metric": metric,
                "synth_id": int(i),
                "nn_real_id": int(j),
                "distance": float(dist[j]),
            }
        )
    return pd.DataFrame(rows)
