from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.causal_estimators.base import CausalEstimator, _bootstrap_ci
from src.data import TrajectoryBatch
from src.policy import Policy


class OracleIRT(CausalEstimator):
    """Oracle estimator for the IRT synthetic simulator (expected outcomes)."""

    name = "oracle_irt"
    is_oracle = True

    def __init__(
        self,
        *,
        simulator: Optional[object] = None,
        beta: Optional[np.ndarray] = None,
        gamma: Optional[float] = None,
        lr: Optional[float] = None,
        delta: Optional[float] = None,
        noise_std: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        if simulator is not None:
            beta = getattr(simulator, "beta", beta)
            gamma = getattr(simulator, "gamma", gamma)
            lr = getattr(simulator, "lr", lr)
            delta = getattr(simulator, "delta", delta)
            noise_std = getattr(simulator, "noise_std", noise_std)
            seed = getattr(simulator, "seed", seed)
            theta_seq = getattr(simulator, "theta", None)
            if theta_seq is not None:
                self._theta0 = np.asarray(theta_seq, dtype=np.float32)[:, 0]
            else:
                self._theta0 = None
        else:
            self._theta0 = None

        if beta is None:
            raise ValueError("OracleIRT requires beta or a simulator with beta.")
        self.beta = np.asarray(beta, dtype=np.float32)
        self.gamma = float(0.0 if gamma is None else gamma)
        self.lr = float(0.0 if lr is None else lr)
        self.delta = float(0.0 if delta is None else delta)
        self.noise_std = float(0.0 if noise_std is None else noise_std)
        self.seed = int(0 if seed is None else seed)
        self._beta_torch = torch.as_tensor(self.beta, dtype=torch.float32)

    def fit(self, train: TrajectoryBatch, valid: Optional[TrajectoryBatch] = None, **kwargs) -> None:
        _ = (train, valid, kwargs)
        return None

    def _resolve_theta0(self, data: TrajectoryBatch) -> np.ndarray:
        if self._theta0 is not None and data.ids is not None:
            ids = data.ids.detach().cpu().numpy().astype(np.int64)
            if ids.size == 0:
                return np.zeros((0,), dtype=np.float32)
            if int(ids.max()) >= self._theta0.shape[0]:
                raise ValueError("OracleIRT: batch ids exceed simulator size.")
            return self._theta0[ids].astype(np.float32)

        X = data.X.detach().cpu().numpy().astype(np.float32)
        if X.shape[1] >= 2:
            return (0.6 * X[:, 0] + 0.2 * X[:, 1]).astype(np.float32)
        if X.shape[1] == 1:
            return (0.6 * X[:, 0]).astype(np.float32)
        return np.zeros((X.shape[0],), dtype=np.float32)

    def _masked_mean(self, y: np.ndarray, m: np.ndarray, *, axis: int) -> np.ndarray:
        denom = m.sum(axis=axis)
        denom = np.where(denom == 0.0, np.nan, denom)
        return (y * m).sum(axis=axis) / denom

    def expected_outcomes_do(
        self,
        data: TrajectoryBatch,
        *,
        t0: int,
        horizon: int,
        action: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if data.A.ndim != 2 or data.T.ndim != 2:
            raise ValueError("OracleIRT expects discrete A/T with shape [N,T].")

        A = data.A.detach().cpu().numpy().astype(np.int64)
        T_obs = data.T.detach().cpu().numpy().astype(np.int64)
        mask = data.mask.detach().cpu().numpy().astype(np.float32)

        n, seq_len = T_obs.shape
        t0 = int(t0)
        horizon = int(horizon)
        if not (0 <= t0 < seq_len):
            raise ValueError(f"t0 out of range: {t0} for seq_len={seq_len}")

        steps = min(horizon, seq_len - t0 - 1)
        H = max(0, int(steps))

        T_used = T_obs.copy()
        T_used[:, t0] = int(action)

        theta0 = self._resolve_theta0(data)
        help_flag = (T_used == 1).astype(np.float32)
        cum_help = np.cumsum(help_flag, axis=1)
        theta_t = theta0[:, None] + self.delta * np.concatenate(
            [np.zeros((n, 1), dtype=np.float32), cum_help[:, :-1]], axis=1
        )

        logits = theta_t - self.beta[A] + self.gamma * help_flag
        y_prob = 1.0 / (1.0 + np.exp(-logits))

        sl = slice(t0, t0 + H + 1)
        return y_prob[:, sl].astype(np.float32), mask[:, sl].astype(np.float32)

    def expected_outcomes_policy(
        self,
        data: TrajectoryBatch,
        *,
        policy: Policy,
        horizon: Optional[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        if data.A.ndim != 2:
            raise ValueError("OracleIRT expects discrete A with shape [N,T].")

        X = data.X.detach().cpu()
        A = data.A.detach().cpu().long()
        mask = data.mask.detach().cpu()

        n, seq_len = A.shape
        if horizon is None:
            horizon = seq_len - 1
        horizon = int(min(int(horizon), seq_len - 1))

        theta = torch.as_tensor(self._resolve_theta0(data), dtype=torch.float32)
        beta = self._beta_torch

        y_prob = torch.zeros((n, horizon + 1), dtype=torch.float32)
        T_hist = torch.zeros((n, horizon + 1), dtype=torch.long)

        for t in range(horizon + 1):
            act = policy.act(
                X=X,
                A_hist=A[:, :t],
                T_hist=T_hist[:, :t],
                Y_hist=y_prob[:, :t],
                t=t,
                mask=mask[:, :t],
            )
            act = act.to(torch.long).view(-1)
            T_hist[:, t] = act
            help_flag = (act == 1).float()

            a_t = A[:, t]
            logits = theta - beta[a_t] + self.gamma * help_flag
            p = torch.sigmoid(logits)
            y_prob[:, t] = p
            theta = theta + self.delta * help_flag

        m_slice = mask[:, : horizon + 1].float()
        return y_prob.numpy(), m_slice.numpy()

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
        y_slice, m_slice = self.expected_outcomes_do(data, t0=t0, horizon=horizon, action=action)

        if subgroup is not None and "mask" in subgroup:
            sel = np.asarray(subgroup["mask"]).astype(bool)
            y_slice = y_slice[sel]
            m_slice = m_slice[sel]

        n_eff = int(y_slice.shape[0])
        if n_eff == 0:
            mu = np.full(int(horizon) + 1, np.nan, dtype=np.float64)
            ci_low = mu.copy()
            ci_high = mu.copy()
            return {"mu": mu, "ci_low": ci_low, "ci_high": ci_high, "n": 0}

        mu = self._masked_mean(y_slice, m_slice, axis=0).astype(np.float64)
        if int(n_boot) <= 1 or n_eff <= 1:
            return {"mu": mu, "ci_low": mu.copy(), "ci_high": mu.copy(), "n": n_eff}

        rng = np.random.default_rng(int(seed))
        samples = []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n_eff, size=n_eff)
            samples.append(self._masked_mean(y_slice[idx], m_slice[idx], axis=0))
        samples = np.stack(samples, axis=0)
        ci_low, ci_high = _bootstrap_ci(samples, alpha=0.05)
        return {
            "mu": mu,
            "ci_low": ci_low.astype(np.float64),
            "ci_high": ci_high.astype(np.float64),
            "n": n_eff,
        }

    def estimate_policy_value(
        self,
        data: TrajectoryBatch,
        *,
        policy: Policy,
        horizon: Optional[int] = None,
        n_boot: int = 200,
        seed: int = 0,
    ) -> dict:
        y_slice, m_slice = self.expected_outcomes_policy(data, policy=policy, horizon=horizon)
        denom = m_slice.sum(axis=1)
        denom = np.where(denom == 0.0, np.nan, denom)
        per_seq = (y_slice * m_slice).sum(axis=1) / denom
        n_eff = int(per_seq.shape[0])
        value = float(np.nanmean(per_seq)) if n_eff > 0 else float("nan")

        if int(n_boot) <= 1 or n_eff <= 1:
            return {
                "value": value,
                "ci_low": value,
                "ci_high": value,
                "n": n_eff,
                "policy_name": getattr(policy, "name", "policy"),
            }

        rng = np.random.default_rng(int(seed))
        samples = []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n_eff, size=n_eff)
            samples.append(np.nanmean(per_seq[idx]))
        samples = np.asarray(samples, dtype=np.float64)
        ci_low, ci_high = _bootstrap_ci(samples.reshape(-1, 1), alpha=0.05)
        return {
            "value": float(np.nanmean(samples)),
            "ci_low": float(ci_low.reshape(-1)[0]),
            "ci_high": float(ci_high.reshape(-1)[0]),
            "n": n_eff,
            "policy_name": getattr(policy, "name", "policy"),
        }
