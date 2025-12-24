from __future__ import annotations

from typing import Optional

import numpy as np

from src.data import TrajectoryBatch
from src.policy import Policy


class CausalEstimator:
    name: str = "base"

    def fit(self, train: TrajectoryBatch, valid: Optional[TrajectoryBatch] = None, **kwargs) -> None:
        raise NotImplementedError

    def estimate_do(
        self,
        data: TrajectoryBatch,
        *,
        t0: int,
        horizon: int,
        action: int,
        subgroup: Optional[dict] = None,
        n_boot: int = 200,
        seed: int = 0,
    ) -> dict:
        raise NotImplementedError

    def estimate_policy_value(
        self,
        data: TrajectoryBatch,
        *,
        policy: Policy,
        horizon: Optional[int] = None,
        n_boot: int = 200,
        seed: int = 0,
    ) -> dict:
        raise NotImplementedError


def _bootstrap_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    if samples.ndim == 1:
        samples = samples[:, None]
    low = np.quantile(samples, alpha / 2.0, axis=0)
    high = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return low, high
