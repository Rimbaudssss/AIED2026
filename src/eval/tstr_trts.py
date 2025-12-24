from __future__ import annotations

from typing import Tuple

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data import TrajectoryBatch
from src.metrics.calibration import brier_score, ece_score


A_HASH_VOCAB = 4096
A_EMB_DIM = 16
T_HASH_VOCAB = 128
T_EMB_DIM = 4


def _hash_np(values: np.ndarray, vocab: int) -> np.ndarray:
    return np.mod(values, int(vocab)).astype(np.int64)


def _hash_torch(values: torch.Tensor, vocab: int) -> torch.Tensor:
    return torch.remainder(values, int(vocab))


def _infer_vocab_size_torch(values: torch.Tensor, *, default: int = 1) -> int:
    if values.numel() == 0:
        return int(default)
    return max(int(default), int(values.max().item()) + 1)


def _flatten_batch_logreg(
    batch: TrajectoryBatch,
    *,
    a_hash_vocab: int,
    t_vocab: int,
    t_hash_vocab: int,
) -> Tuple[sp.csr_matrix | np.ndarray, np.ndarray]:
    X = batch.X.detach().cpu().numpy()
    A = batch.A.detach().cpu().numpy()
    T = batch.T.detach().cpu().numpy()
    Y = batch.Y.detach().cpu().numpy()
    M = batch.mask.detach().cpu().numpy()

    n, seq_len = T.shape
    if n == 0 or seq_len == 0:
        return np.zeros((0, X.shape[1] + 1), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    mask_flat = (M > 0.5).reshape(-1)
    if not mask_flat.any():
        return np.zeros((0, X.shape[1] + 1), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    X_rep = np.repeat(X, seq_len, axis=0)
    X_flat = X_rep[mask_flat].astype(np.float32)

    if A.ndim == 2:
        a_flat = A.reshape(-1)[mask_flat].astype(np.int64)
    else:
        a_flat = A.reshape(-1, A.shape[2])[mask_flat].astype(np.float32)

    if T.ndim == 2:
        t_flat = T.reshape(-1)[mask_flat].astype(np.int64)
    else:
        t_flat = T.reshape(-1, T.shape[2])[mask_flat].astype(np.float32)

    y_prev = np.zeros_like(Y)
    y_prev[:, 1:] = Y[:, :-1]
    y_prev_flat = y_prev.reshape(-1)[mask_flat].astype(np.float32)
    y_flat = Y.reshape(-1)[mask_flat].astype(np.float32)

    dense_parts = [X_flat]
    if A.ndim == 3:
        dense_parts.append(a_flat)
    if T.ndim == 3:
        dense_parts.append(t_flat)
    dense_parts.append(y_prev_flat[:, None])
    dense = np.concatenate(dense_parts, axis=1).astype(np.float32)

    sparse_blocks = []
    n_rows = int(y_flat.shape[0])
    rows = np.arange(n_rows)

    if A.ndim == 2:
        a_ids = _hash_np(a_flat, a_hash_vocab)
        a_sparse = sp.csr_matrix((np.ones(n_rows, dtype=np.float32), (rows, a_ids)), shape=(n_rows, a_hash_vocab))
        sparse_blocks.append(a_sparse)

    if T.ndim == 2:
        t_vocab = max(1, int(t_vocab))
        t_use_hash = t_vocab > int(t_hash_vocab)
        t_dim = int(t_hash_vocab) if t_use_hash else t_vocab
        t_ids = _hash_np(t_flat, t_dim) if t_use_hash else np.mod(t_flat, t_dim).astype(np.int64)
        t_sparse = sp.csr_matrix((np.ones(n_rows, dtype=np.float32), (rows, t_ids)), shape=(n_rows, t_dim))
        sparse_blocks.append(t_sparse)

    if sparse_blocks:
        dense_sparse = sp.csr_matrix(dense)
        out = sp.hstack([*sparse_blocks, dense_sparse], format="csr")
        return out, y_flat

    return dense, y_flat


def _flatten_batch_tensors(
    batch: TrajectoryBatch,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X = batch.X.to(device)
    A = batch.A.to(device)
    T = batch.T.to(device)
    Y = batch.Y.to(device)
    M = batch.mask.to(device)

    n, seq_len = T.shape
    if n == 0 or seq_len == 0:
        empty = torch.empty((0,), device=device)
        return (
            empty.reshape(0, X.shape[1]),
            empty,
            empty,
            empty.reshape(0, 1),
            empty,
        )

    mask_flat = (M > 0.5).reshape(-1)
    x_rep = X.float()[:, None, :].expand(-1, seq_len, -1).reshape(-1, X.shape[1])
    x_flat = x_rep[mask_flat]

    if A.ndim == 2:
        a_flat = A.long().reshape(-1)[mask_flat]
    else:
        a_flat = A.float().reshape(-1, A.shape[2])[mask_flat]

    if T.ndim == 2:
        t_flat = T.long().reshape(-1)[mask_flat]
    else:
        t_flat = T.float().reshape(-1, T.shape[2])[mask_flat]

    y_prev = torch.zeros_like(Y)
    y_prev[:, 1:] = Y[:, :-1]
    y_prev_flat = y_prev.float().reshape(-1)[mask_flat].unsqueeze(-1)
    y_flat = Y.float().reshape(-1)[mask_flat]

    return x_flat, a_flat, t_flat, y_prev_flat, y_flat


class _MLP(nn.Module):
    def __init__(self, d_in: int, d_h: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.GELU(),
            nn.Linear(d_h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


class _MLPPredictor(nn.Module):
    def __init__(
        self,
        *,
        d_x: int,
        a_is_dense: bool,
        d_a: int,
        a_hash_vocab: int,
        a_emb_dim: int,
        t_is_dense: bool,
        d_t: int,
        t_vocab: int,
        t_hash_vocab: int,
        t_emb_dim: int,
    ):
        super().__init__()
        self.a_is_dense = bool(a_is_dense)
        self.t_is_dense = bool(t_is_dense)

        if self.a_is_dense:
            self.a_dim = int(d_a)
            self.a_emb = None
            self.a_hash_vocab = None
        else:
            self.a_hash_vocab = int(a_hash_vocab)
            self.a_emb = nn.Embedding(self.a_hash_vocab, int(a_emb_dim))
            self.a_dim = int(a_emb_dim)

        if self.t_is_dense:
            self.t_dim = int(d_t)
            self.t_emb = None
            self.t_vocab = None
            self.t_use_hash = False
        else:
            t_vocab = max(1, int(t_vocab))
            t_hash_vocab = int(t_hash_vocab)
            self.t_use_hash = t_vocab > t_hash_vocab
            self.t_vocab = t_hash_vocab if self.t_use_hash else t_vocab
            self.t_emb = nn.Embedding(self.t_vocab, int(t_emb_dim))
            self.t_dim = int(t_emb_dim)

        d_in = int(d_x) + self.a_dim + self.t_dim + 1
        self.mlp = _MLP(d_in=d_in)

    def forward(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        y_prev: torch.Tensor,
    ) -> torch.Tensor:
        parts = [x]

        if self.a_is_dense:
            a_feat = a.float()
        else:
            a_id = _hash_torch(a.long(), int(self.a_hash_vocab))
            a_feat = self.a_emb(a_id)
        parts.append(a_feat)

        if self.t_is_dense:
            t_feat = t.float()
        else:
            t_id = t.long()
            if self.t_use_hash:
                t_id = _hash_torch(t_id, int(self.t_vocab))
            t_feat = self.t_emb(t_id)
        parts.append(t_feat)

        parts.append(y_prev)
        inp = torch.cat(parts, dim=-1)
        return self.mlp(inp)


class _SAKT(nn.Module):
    def __init__(self, d_in: int, d_model: int, n_heads: int, n_layers: int, seq_len: int):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = x.shape[0], x.shape[1]
        pos = torch.arange(seq_len, device=x.device)
        h = self.inp(x) + self.pos(pos)[None, :, :]
        causal = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        key_padding = ~mask.bool()
        h = self.encoder(h, mask=causal, src_key_padding_mask=key_padding)
        return self.out(h).squeeze(-1)


class _SAKTPredictor(nn.Module):
    def __init__(
        self,
        *,
        d_x: int,
        a_is_dense: bool,
        d_a: int,
        a_hash_vocab: int,
        a_emb_dim: int,
        t_is_dense: bool,
        d_t: int,
        t_vocab: int,
        t_hash_vocab: int,
        t_emb_dim: int,
        seq_len: int,
    ):
        super().__init__()
        self.a_is_dense = bool(a_is_dense)
        self.t_is_dense = bool(t_is_dense)

        if self.a_is_dense:
            self.a_dim = int(d_a)
            self.a_emb = None
            self.a_hash_vocab = None
        else:
            self.a_hash_vocab = int(a_hash_vocab)
            self.a_emb = nn.Embedding(self.a_hash_vocab, int(a_emb_dim))
            self.a_dim = int(a_emb_dim)

        if self.t_is_dense:
            self.t_dim = int(d_t)
            self.t_emb = None
            self.t_vocab = None
            self.t_use_hash = False
        else:
            t_vocab = max(1, int(t_vocab))
            t_hash_vocab = int(t_hash_vocab)
            self.t_use_hash = t_vocab > t_hash_vocab
            self.t_vocab = t_hash_vocab if self.t_use_hash else t_vocab
            self.t_emb = nn.Embedding(self.t_vocab, int(t_emb_dim))
            self.t_dim = int(t_emb_dim)

        d_in = int(d_x) + self.a_dim + self.t_dim + 1
        self.sakt = _SAKT(d_in=d_in, d_model=64, n_heads=4, n_layers=2, seq_len=int(seq_len))

    def _build_inputs(self, x: torch.Tensor, a: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[1]
        x_rep = x.float()[:, None, :].expand(-1, seq_len, -1)

        if self.a_is_dense:
            a_feat = a.float()
        else:
            a_id = _hash_torch(a.long(), int(self.a_hash_vocab))
            a_feat = self.a_emb(a_id)

        if self.t_is_dense:
            t_feat = t.float()
        else:
            t_id = t.long()
            if self.t_use_hash:
                t_id = _hash_torch(t_id, int(self.t_vocab))
            t_feat = self.t_emb(t_id)

        y_prev = torch.zeros_like(y)
        y_prev[:, 1:] = y[:, :-1]
        y_prev = y_prev.float().unsqueeze(-1)
        return torch.cat([x_rep, a_feat, t_feat, y_prev], dim=-1)

    def forward(self, x: torch.Tensor, a: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inp = self._build_inputs(x, a, t, y)
        return self.sakt(inp, mask)


def _train_logreg(x_train: sp.csr_matrix | np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=200, solver="saga")
    clf.fit(x_train, y_train.astype(int))
    return clf


def _train_mlp(batch: TrajectoryBatch, *, t_vocab: int) -> _MLPPredictor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_flat, a_flat, t_flat, y_prev, y_flat = _flatten_batch_tensors(batch, device=device)
    model = _MLPPredictor(
        d_x=int(batch.X.shape[1]),
        a_is_dense=batch.A.ndim == 3,
        d_a=int(batch.A.shape[2]) if batch.A.ndim == 3 else 1,
        a_hash_vocab=A_HASH_VOCAB,
        a_emb_dim=A_EMB_DIM,
        t_is_dense=batch.T.ndim == 3,
        d_t=int(batch.T.shape[2]) if batch.T.ndim == 3 else 1,
        t_vocab=int(t_vocab),
        t_hash_vocab=T_HASH_VOCAB,
        t_emb_dim=T_EMB_DIM,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        logits = model(x_flat, a_flat, t_flat, y_prev)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_flat)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def _predict_mlp(model: _MLPPredictor, batch: TrajectoryBatch) -> Tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    x_flat, a_flat, t_flat, y_prev, y_flat = _flatten_batch_tensors(batch, device=device)
    with torch.no_grad():
        logits = model(x_flat, a_flat, t_flat, y_prev)
        y_prob = torch.sigmoid(logits)
    return y_prob.detach().cpu().numpy(), y_flat.detach().cpu().numpy()


def _train_sakt(batch: TrajectoryBatch, *, t_vocab: int) -> _SAKTPredictor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = batch.X.to(device)
    A = batch.A.to(device)
    T = batch.T.to(device)
    Y = batch.Y.to(device)
    M = batch.mask.to(device)

    seq_len = T.shape[1]
    model = _SAKTPredictor(
        d_x=int(X.shape[1]),
        a_is_dense=A.ndim == 3,
        d_a=int(A.shape[2]) if A.ndim == 3 else 1,
        a_hash_vocab=A_HASH_VOCAB,
        a_emb_dim=A_EMB_DIM,
        t_is_dense=T.ndim == 3,
        d_t=int(T.shape[2]) if T.ndim == 3 else 1,
        t_vocab=int(t_vocab),
        t_hash_vocab=T_HASH_VOCAB,
        t_emb_dim=T_EMB_DIM,
        seq_len=int(seq_len),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(5):
        logits = model(X, A, T, Y, M)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, Y.float(), reduction="none")
        loss = (loss * M).sum() / M.sum().clamp(min=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def _predict_sakt(model: _SAKTPredictor, batch: TrajectoryBatch) -> np.ndarray:
    device = next(model.parameters()).device
    X = batch.X.to(device)
    A = batch.A.to(device)
    T = batch.T.to(device)
    Y = batch.Y.to(device)
    M = batch.mask.to(device)
    with torch.no_grad():
        logits = model(X, A, T, Y, M)
        prob = torch.sigmoid(logits)
    return prob.detach().cpu().numpy()


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = y_true.astype(np.float64)
    y_prob = y_prob.astype(np.float64)
    out = {}
    try:
        out["auc"] = float(roc_auc_score(y_true.astype(int), y_prob))
    except Exception:
        out["auc"] = float("nan")
    out["rmse"] = float(np.sqrt(np.mean((y_true - y_prob) ** 2)))
    out["brier"] = brier_score(y_true, y_prob)
    out["ece"] = ece_score(y_true, y_prob, n_bins=10)
    return out


def run_tstr_trts(
    *,
    gen_model,
    real_train: TrajectoryBatch,
    real_test: TrajectoryBatch,
    n_synth: int,
    predictor: str,
    seed: int = 0,
    save_calibration: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    n_synth = int(n_synth)
    idx = rng.integers(0, real_train.X.shape[0], size=n_synth)
    synth = TrajectoryBatch(
        X=real_train.X[idx],
        A=real_train.A[idx],
        T=real_train.T[idx],
        Y=real_train.Y[idx],
        mask=real_train.mask[idx],
        lengths=real_train.lengths[idx],
    )

    ro = gen_model.rollout(synth, do_t=None, policy=None, horizon=None, t0=0, teacher_forcing=False)
    y_prob = ro["Y_prob"].detach().cpu().numpy()
    y_sample = rng.binomial(n=1, p=np.clip(y_prob, 1e-4, 1.0 - 1e-4)).astype(np.float32)
    synth = TrajectoryBatch(
        X=synth.X,
        A=synth.A,
        T=synth.T,
        Y=torch.as_tensor(y_sample),
        mask=synth.mask,
        lengths=synth.lengths,
    )

    t_vocab = max(
        _infer_vocab_size_torch(real_train.T, default=1),
        _infer_vocab_size_torch(real_test.T, default=1),
    )
    rows = []
    for setting, train_batch in [("TRTS", real_train), ("TSTR", synth)]:
        if predictor == "logreg":
            x_train, y_train = _flatten_batch_logreg(
                train_batch, a_hash_vocab=A_HASH_VOCAB, t_vocab=t_vocab, t_hash_vocab=T_HASH_VOCAB
            )
            x_test, y_test = _flatten_batch_logreg(
                real_test, a_hash_vocab=A_HASH_VOCAB, t_vocab=t_vocab, t_hash_vocab=T_HASH_VOCAB
            )
            clf = _train_logreg(x_train, y_train)
            y_prob = clf.predict_proba(x_test)[:, 1]
        elif predictor == "mlp":
            model = _train_mlp(train_batch, t_vocab=t_vocab)
            y_prob, y_test = _predict_mlp(model, real_test)
        elif predictor == "sakt":
            model = _train_sakt(train_batch, t_vocab=t_vocab)
            y_prob_full = _predict_sakt(model, real_test)
            y_test_full = real_test.Y.detach().cpu().numpy()
            m = real_test.mask.detach().cpu().numpy()
            y_prob = y_prob_full[m > 0.5]
            y_test = y_test_full[m > 0.5]
        else:
            raise ValueError(f"Unknown predictor={predictor}")

        if save_calibration:
            calib_dir = Path("results") / "calibration"
            calib_dir.mkdir(parents=True, exist_ok=True)
            calib_path = calib_dir / f"calibration_{gen_model.name}_{predictor}_{setting}.npz"
            np.savez_compressed(calib_path, y_true=y_test.astype(np.float32), y_prob=y_prob.astype(np.float32))
        metrics = _compute_metrics(y_test, y_prob)
        rows.append(
            {
                "model": gen_model.name,
                "dataset": "unknown",
                "predictor": predictor,
                "setting": setting,
                "auc": metrics["auc"],
                "rmse": metrics["rmse"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
                "n_train": int(train_batch.X.shape[0]),
                "n_test": int(real_test.X.shape[0]),
            }
        )

    return pd.DataFrame(rows)
