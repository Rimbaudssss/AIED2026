from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SequenceBatch:
    X: torch.Tensor  # [B, d_x]
    A: torch.Tensor  # [B, T, ...] or [B, T] if discrete
    T: torch.Tensor  # [B, T, ...] or [B, T] if discrete
    Y: torch.Tensor  # [B, T, ...] or [B, T]
    mask: torch.Tensor  # [B, T] bool/float


@dataclass(frozen=True)
class TrajectoryBatch:
    """Unified trajectory batch structure used across baselines and estimators."""

    X: torch.Tensor  # [B, d_x]
    A: torch.Tensor  # [B, T, d_a] or [B, T] if discrete
    T: torch.Tensor  # [B, T] discrete action ids
    Y: torch.Tensor  # [B, T] outcome (0/1 for binary)
    mask: torch.Tensor  # [B, T] bool/float
    lengths: torch.Tensor  # [B] int64

    def to(self, device: torch.device) -> "TrajectoryBatch":
        return TrajectoryBatch(
            X=_as_tensor(self.X, device=device),
            A=_as_tensor(self.A, device=device),
            T=_as_tensor(self.T, device=device),
            Y=_as_tensor(self.Y, device=device),
            mask=_as_tensor(self.mask, device=device),
            lengths=_as_tensor(self.lengths, device=device),
        )


def compute_lengths(mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    return m.sum(dim=1).clamp(min=1.0).long()


def _as_tensor(x: Any, *, device: Optional[torch.device] = None) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if device is not None:
        t = t.to(device)
    return t


class NPZSequenceDataset(Dataset):
    """Loads a padded dataset from an .npz file.

    Expected keys (minimal):
      - X: [N, d_x]
      - A: [N, T, d_a] (float) or [N, T] (int indices)
      - T: [N, T, d_t] (float) or [N, T] (int indices)
      - Y: [N, T, d_y] (float) or [N, T] (float/int)
      - M: [N, T] (0/1) mask (optional; defaults to all-ones)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        data = np.load(self.path, allow_pickle=False)

        self.X = data["X"]
        self.A = data["A"]
        self.T = data["T"]
        self.Y = data["Y"]
        self.M = data["M"] if "M" in data.files else np.ones(self.Y.shape[:2], dtype=np.float32)

        if self.X.ndim != 2:
            raise ValueError(f"X must be [N,d_x], got {self.X.shape}")
        if self.M.ndim != 2:
            raise ValueError(f"M must be [N,T], got {self.M.shape}")

        self.n = int(self.X.shape[0])
        self.seq_len = int(self.M.shape[1])

        self.d_x = int(self.X.shape[1])
        self.a_is_discrete = self.A.ndim == 2
        self.t_is_discrete = self.T.ndim == 2
        self.y_is_discrete = self.Y.ndim == 2 and np.issubdtype(self.Y.dtype, np.integer)

        self.d_a = 1 if self.a_is_discrete else int(self.A.shape[2])
        self.d_t = 1 if self.t_is_discrete else int(self.T.shape[2])
        self.d_y = 1 if self.Y.ndim == 2 else int(self.Y.shape[2])

        self.a_vocab_size = int(self.A.max() + 1) if self.a_is_discrete else None
        self.t_vocab_size = int(self.T.max() + 1) if self.t_is_discrete else None

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.as_tensor(self.X[idx]).float()
        a = torch.as_tensor(self.A[idx])
        t = torch.as_tensor(self.T[idx])
        y = torch.as_tensor(self.Y[idx]).float()
        m = torch.as_tensor(self.M[idx]).float()
        return {"X": x, "A": a, "T": t, "Y": y, "mask": m}


class SyntheticEduDataset(Dataset):
    """A tiny synthetic generator for smoke tests (not for research claims)."""

    def __init__(
        self,
        n: int = 2048,
        seq_len: int = 30,
        d_x: int = 8,
        a_vocab_size: int = 50,
        t_vocab_size: int = 3,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.n = int(n)
        self.seq_len = int(seq_len)
        self.d_x = int(d_x)
        self.a_vocab_size = int(a_vocab_size)
        self.t_vocab_size = int(t_vocab_size)

        self.X = rng.normal(size=(self.n, self.d_x)).astype(np.float32)
        self.A = rng.integers(low=0, high=self.a_vocab_size, size=(self.n, self.seq_len), dtype=np.int64)
        self.T = rng.integers(low=0, high=self.t_vocab_size, size=(self.n, self.seq_len), dtype=np.int64)

        # A simple outcome mechanism with time trend.
        base = self.X[:, :1] * 0.3 + (self.T[..., None] == 1).astype(np.float32) * 0.2
        trend = np.linspace(-0.2, 0.2, self.seq_len, dtype=np.float32)[None, :, None]
        logits = base[:, None, :] + trend
        prob = 1.0 / (1.0 + np.exp(-logits))
        self.Y = rng.binomial(n=1, p=prob).astype(np.float32).squeeze(-1)  # [N,T]

        self.M = np.ones((self.n, self.seq_len), dtype=np.float32)

        self.a_is_discrete = True
        self.t_is_discrete = True
        self.y_is_discrete = True
        self.d_a = 1
        self.d_t = 1
        self.d_y = 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "X": torch.as_tensor(self.X[idx]).float(),
            "A": torch.as_tensor(self.A[idx]).long(),
            "T": torch.as_tensor(self.T[idx]).long(),
            "Y": torch.as_tensor(self.Y[idx]).float(),
            "mask": torch.as_tensor(self.M[idx]).float(),
        }


class IRTSyntheticDataset(Dataset):
    """IRT-like simulator with evolving latent ability (theta).

    This is a better AIED-style simulator than a plain logistic baseline:
      P(Y_t=1) = sigmoid(theta_t - beta_{A_t} + gamma * 1[T_t == 1])
      theta_{t+1} = theta_t + lr * (Y_t - P(Y_t=1)) + delta * 1[T_t == 1] + noise

    Notes:
    - Action id 1 is treated as a "help/hint" intervention in the simulator.
    - `theta` is stored as `self.theta` with shape [N, T+1] for optional evaluation.
    """

    def __init__(
        self,
        n: int = 4096,
        seq_len: int = 50,
        d_x: int = 8,
        a_vocab_size: int = 200,
        t_vocab_size: int = 3,
        gamma: float = 0.8,
        lr: float = 0.05,
        delta: float = 0.02,
        confounding: float = 1.0,
        noise_std: float = 0.02,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.n = int(n)
        self.seq_len = int(seq_len)
        self.d_x = int(d_x)
        self.a_vocab_size = int(a_vocab_size)
        self.t_vocab_size = int(t_vocab_size)

        self.X = rng.normal(size=(self.n, self.d_x)).astype(np.float32)

        beta = rng.normal(loc=0.0, scale=1.0, size=(self.a_vocab_size,)).astype(np.float32)
        self.A = rng.integers(low=0, high=self.a_vocab_size, size=(self.n, self.seq_len), dtype=np.int64)

        theta = (0.6 * self.X[:, 0] + 0.2 * self.X[:, 1] + rng.normal(scale=0.4, size=self.n)).astype(np.float32)
        theta_seq = np.zeros((self.n, self.seq_len + 1), dtype=np.float32)
        theta_seq[:, 0] = theta

        self.T = np.zeros((self.n, self.seq_len), dtype=np.int64)
        self.Y = np.zeros((self.n, self.seq_len), dtype=np.float32)

        def sigmoid(z: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-z))

        for t in range(self.seq_len):
            a_t = self.A[:, t]
            diff = beta[a_t]

            # Selection bias / confounding: weaker students receive more help (action 1).
            p_help = sigmoid(-confounding * theta)
            if self.t_vocab_size == 2:
                t_t = rng.binomial(n=1, p=p_help).astype(np.int64)
            else:
                # Split remaining probability mass across other actions.
                remaining = 1.0 - p_help
                p0 = remaining * 0.7
                p_other = remaining * 0.3
                probs = np.zeros((self.n, self.t_vocab_size), dtype=np.float32)
                probs[:, 0] = p0
                probs[:, 1] = p_help
                if self.t_vocab_size > 2:
                    probs[:, 2:] = (p_other[:, None] / max(1, self.t_vocab_size - 2))
                probs = probs / probs.sum(axis=1, keepdims=True)
                t_t = np.array([rng.choice(self.t_vocab_size, p=probs[i]) for i in range(self.n)], dtype=np.int64)

            help_flag = (t_t == 1).astype(np.float32)
            logit = theta - diff + gamma * help_flag
            p = sigmoid(logit)
            y_t = rng.binomial(n=1, p=p).astype(np.float32)

            # Latent update (knowledge evolves).
            theta = theta + lr * (y_t - p).astype(np.float32) + delta * help_flag + rng.normal(scale=noise_std, size=self.n)
            theta = theta.astype(np.float32)

            self.T[:, t] = t_t
            self.Y[:, t] = y_t
            theta_seq[:, t + 1] = theta

        self.theta = theta_seq
        self.M = np.ones((self.n, self.seq_len), dtype=np.float32)

        self.a_is_discrete = True
        self.t_is_discrete = True
        self.y_is_discrete = True
        self.d_a = 1
        self.d_t = 1
        self.d_y = 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "X": torch.as_tensor(self.X[idx]).float(),
            "A": torch.as_tensor(self.A[idx]).long(),
            "T": torch.as_tensor(self.T[idx]).long(),
            "Y": torch.as_tensor(self.Y[idx]).float(),
            "mask": torch.as_tensor(self.M[idx]).float(),
        }


class TrajectoryDataset(Dataset):
    """Dataset wrapper that yields TrajectoryBatch-style samples."""

    def __init__(self, *, X: torch.Tensor, A: torch.Tensor, T: torch.Tensor, Y: torch.Tensor, mask: torch.Tensor):
        if X.ndim != 2:
            raise ValueError(f"X must be [N,d_x], got {tuple(X.shape)}")
        if A.ndim not in (2, 3):
            raise ValueError(f"A must be [N,T] or [N,T,d_a], got {tuple(A.shape)}")
        if T.ndim != 2:
            raise ValueError(f"T must be [N,T], got {tuple(T.shape)}")
        if Y.ndim != 2:
            raise ValueError(f"Y must be [N,T], got {tuple(Y.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"mask must be [N,T], got {tuple(mask.shape)}")
        self.X = X
        self.A = A
        self.T = T
        self.Y = Y
        self.mask = mask
        self.n = int(X.shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "X": self.X[int(idx)],
            "A": self.A[int(idx)],
            "T": self.T[int(idx)],
            "Y": self.Y[int(idx)],
            "mask": self.mask[int(idx)],
        }


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    def collate(examples: list[Dict[str, torch.Tensor]]) -> SequenceBatch:
        x = torch.stack([e["X"] for e in examples], dim=0)
        a = torch.stack([e["A"] for e in examples], dim=0)
        t = torch.stack([e["T"] for e in examples], dim=0)
        y = torch.stack([e["Y"] for e in examples], dim=0)
        m = torch.stack([e["mask"] for e in examples], dim=0)
        return SequenceBatch(X=x, A=a, T=t, Y=y, mask=m)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=True,
    )


def make_trajectory_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    def collate(examples: list[Dict[str, torch.Tensor]]) -> TrajectoryBatch:
        x = torch.stack([e["X"] for e in examples], dim=0)
        a = torch.stack([e["A"] for e in examples], dim=0)
        t = torch.stack([e["T"] for e in examples], dim=0)
        y = torch.stack([e["Y"] for e in examples], dim=0)
        m = torch.stack([e["mask"] for e in examples], dim=0)
        lengths = compute_lengths(m)
        return TrajectoryBatch(X=x, A=a, T=t, Y=y, mask=m, lengths=lengths)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=drop_last,
    )


def move_batch(batch: SequenceBatch, device: torch.device) -> SequenceBatch:
    return SequenceBatch(
        X=_as_tensor(batch.X, device=device),
        A=_as_tensor(batch.A, device=device),
        T=_as_tensor(batch.T, device=device),
        Y=_as_tensor(batch.Y, device=device),
        mask=_as_tensor(batch.mask, device=device),
    )


def move_trajectory_batch(batch: TrajectoryBatch, device: torch.device) -> TrajectoryBatch:
    return batch.to(device)
