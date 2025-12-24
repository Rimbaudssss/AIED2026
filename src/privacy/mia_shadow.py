from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from src.baselines import BaseSeqModel
from src.data import TrajectoryBatch
from src.privacy.nn_distance import _embed


def _recon_error(gen_model: BaseSeqModel, batch: TrajectoryBatch) -> np.ndarray:
    ro = gen_model.rollout(batch, do_t=None, policy=None, horizon=None, t0=0, teacher_forcing=False)
    y_prob = ro["Y_prob"].detach().cpu().numpy()
    y_true = batch.Y.detach().cpu().numpy()
    m = batch.mask.detach().cpu().numpy()
    eps = 1e-6
    loss = -(y_true * np.log(y_prob + eps) + (1.0 - y_true) * np.log(1.0 - y_prob + eps))
    per_seq = (loss * m).sum(axis=1) / np.maximum(1.0, m.sum(axis=1))
    return per_seq.astype(np.float32)


def _avg_confidence(gen_model: BaseSeqModel, batch: TrajectoryBatch) -> np.ndarray:
    ro = gen_model.rollout(batch, do_t=None, policy=None, horizon=None, t0=0, teacher_forcing=False)
    y_prob = ro["Y_prob"].detach().cpu().numpy()
    m = batch.mask.detach().cpu().numpy()
    conf = np.abs(y_prob - 0.5)
    per_seq = (conf * m).sum(axis=1) / np.maximum(1.0, m.sum(axis=1))
    return per_seq.astype(np.float32)


def _nn_distance_feature(
    train: TrajectoryBatch,
    batch: TrajectoryBatch,
    *,
    leave_one_out: bool,
    embed_space: str,
) -> np.ndarray:
    train_emb = _embed(train, embed_space=embed_space)
    batch_emb = _embed(batch, embed_space=embed_space)
    if train_emb.shape[0] == 0 or batch_emb.shape[0] == 0:
        return np.zeros((batch_emb.shape[0],), dtype=np.float32)

    n_neighbors = 2 if leave_one_out else 1
    if train_emb.shape[0] < n_neighbors:
        n_neighbors = train_emb.shape[0]
    if n_neighbors == 0:
        return np.zeros((batch_emb.shape[0],), dtype=np.float32)

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(train_emb)
    distances, _ = nn.kneighbors(batch_emb, return_distance=True)
    if leave_one_out and n_neighbors > 1:
        dist = distances[:, 1]
    else:
        dist = distances[:, 0]
    return dist.astype(np.float32)


def _fit_attack_classifier(X_in: np.ndarray, X_out: np.ndarray, *, seed: int) -> dict:
    if X_in.size == 0 or X_out.size == 0:
        return {"attack_auc": float("nan"), "attack_acc": float("nan")}

    X = np.concatenate([X_in, X_out], axis=0)
    y = np.concatenate([np.ones(X_in.shape[0]), np.zeros(X_out.shape[0])], axis=0)
    if X.shape[0] < 2:
        return {"attack_auc": float("nan"), "attack_acc": float("nan")}

    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(X.shape[0])
    split = int(0.7 * len(idx))
    if split <= 0:
        split = 1
    if split >= len(idx):
        split = len(idx) - 1
    train_idx = idx[:split]
    test_idx = idx[split:]

    try:
        clf = LogisticRegression(max_iter=200)
        clf.fit(X[train_idx], y[train_idx])
        prob = clf.predict_proba(X[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(y[test_idx], prob))
        except Exception:
            auc = float("nan")
        acc = float(accuracy_score(y[test_idx], pred))
    except Exception:
        auc = float("nan")
        acc = float("nan")

    return {"attack_auc": auc, "attack_acc": acc}


def run_membership_inference(
    *,
    gen_model: BaseSeqModel,
    real_train: TrajectoryBatch,
    real_holdout: TrajectoryBatch,
    attack_features: List[str],
    embed_space: str = "y_only",
    seed: int = 0,
) -> dict:
    feats_in = []
    feats_out = []

    if "recon_error" in attack_features:
        feats_in.append(_recon_error(gen_model, real_train))
        feats_out.append(_recon_error(gen_model, real_holdout))
    if "avg_confidence" in attack_features:
        feats_in.append(_avg_confidence(gen_model, real_train))
        feats_out.append(_avg_confidence(gen_model, real_holdout))
    if "nn_distance" in attack_features:
        feats_in.append(_nn_distance_feature(real_train, real_train, leave_one_out=True, embed_space=embed_space))
        feats_out.append(_nn_distance_feature(real_train, real_holdout, leave_one_out=False, embed_space=embed_space))

    if not feats_in:
        return {"attack_auc": float("nan"), "attack_acc": float("nan"), "features": attack_features}

    X_in = np.stack(feats_in, axis=1)
    X_out = np.stack(feats_out, axis=1)
    result = _fit_attack_classifier(X_in, X_out, seed=int(seed))
    result["features"] = attack_features
    return result


def run_synth_membership_inference(
    *,
    real_train: TrajectoryBatch,
    real_holdout: TrajectoryBatch,
    synth: TrajectoryBatch,
    embed_space: str = "y_only",
    seed: int = 0,
    attack_features: List[str] | None = None,
) -> dict:
    feats_in = _nn_distance_feature(synth, real_train, leave_one_out=False, embed_space=embed_space)
    feats_out = _nn_distance_feature(synth, real_holdout, leave_one_out=False, embed_space=embed_space)

    X_in = feats_in.reshape(-1, 1)
    X_out = feats_out.reshape(-1, 1)
    result = _fit_attack_classifier(X_in, X_out, seed=int(seed))
    result["features"] = attack_features or ["synth_nn_distance"]
    return result


def mia_to_dataframe(result: dict, *, model: str, dataset: str, attack_features: List[str], n_in: int, n_out: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model,
                "dataset": dataset,
                "attack_features": ",".join(attack_features),
                "attack_auc": float(result.get("attack_auc", np.nan)),
                "attack_acc": float(result.get("attack_acc", np.nan)),
                "n_in": int(n_in),
                "n_out": int(n_out),
            }
        ]
    )
