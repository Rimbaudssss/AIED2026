from __future__ import annotations

from typing import Optional
import zlib

import numpy as np
import torch

from src.causal_estimators.base import CausalEstimator, _bootstrap_ci
from src.data import TrajectoryBatch
from src.policy import Policy


def _stable_hash(text: str) -> int:
    return zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF


class OracleIRT(CausalEstimator):
    """MC oracle estimator for the IRT synthetic simulator (expected outcomes)."""

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
        mc_samples: int = 200,
        mc_seed: int = 0,
        mc_batch: int = 256,
        return_prob: bool = False,
        debug: bool = False,
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

        self.mc_samples = int(mc_samples)
        self.mc_seed = int(mc_seed)
        self.mc_batch = int(mc_batch)
        self.return_prob = bool(return_prob)
        self.debug = bool(debug)

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

    def _rng_for(self, *, k: int, t0: Optional[int], horizon: Optional[int], policy: Optional[Policy]) -> np.random.Generator:
        seed = int(self.mc_seed)
        if t0 is not None:
            seed += 1009 * int(t0)
        if horizon is not None:
            seed += 9176 * int(horizon)
        if policy is not None:
            name = getattr(policy, "name", None)
            seed += 7919 * _stable_hash(str(name or policy))
        seed += 104729 * int(k)
        return np.random.default_rng(int(seed) & 0xFFFFFFFF)

    def _mc_simulate(
        self,
        batch: TrajectoryBatch,
        *,
        t0: Optional[int] = None,
        do_value: Optional[int] = None,
        policy: Optional[Policy] = None,
        horizon: Optional[int] = None,
    ) -> np.ndarray:
        if batch.A.ndim != 2 or batch.T.ndim != 2:
            raise ValueError("OracleIRT expects discrete A/T with shape [N,T].")
        if (do_value is None) == (policy is None):
            raise ValueError("Provide either do_value or policy.")

        A = batch.A.detach().cpu().numpy().astype(np.int64)
        T_obs = batch.T.detach().cpu().numpy().astype(np.int64)
        mask = batch.mask.detach().cpu().numpy().astype(np.float32)

        bsz, seq_len = T_obs.shape
        if policy is not None:
            if horizon is None:
                horizon = seq_len - 1
            H = min(int(horizon), seq_len - 1)
            start = 0
            steps = H + 1
        else:
            if t0 is None:
                raise ValueError("t0 is required for do() simulation.")
            t0 = int(t0)
            if not (0 <= t0 < seq_len):
                raise ValueError(f"t0 out of range: {t0} for seq_len={seq_len}")
            if horizon is None:
                horizon = seq_len - t0 - 1
            H = min(int(horizon), seq_len - t0 - 1)
            start = t0
            steps = t0 + H + 1

        H = max(0, int(H))
        steps = max(0, int(steps))
        if steps == 0:
            return np.zeros((bsz, 0), dtype=np.float32)

        y_sum = np.zeros((bsz, H + 1), dtype=np.float64)
        theta0 = self._resolve_theta0(batch).astype(np.float32)
        beta = self.beta
        gamma = float(self.gamma)
        lr = float(self.lr)
        delta = float(self.delta)
        noise_std = float(self.noise_std)
        use_prob = bool(self.return_prob)

        if policy is not None:
            X_t = batch.X.detach().cpu()
            A_t = batch.A.detach().cpu()
            mask_t = batch.mask.detach().cpu()

        chunk = max(1, int(self.mc_batch))
        n_mc = max(1, int(self.mc_samples))
        for k in range(n_mc):
            rng = self._rng_for(k=k, t0=t0, horizon=H, policy=policy)
            theta = theta0.copy()

            if policy is not None:
                y_hist = torch.zeros((bsz, steps), dtype=torch.float32)
                t_hist = torch.zeros((bsz, steps), dtype=torch.long)

            for t in range(steps):
                if policy is not None:
                    act = policy.act(
                        X=X_t,
                        A_hist=A_t[:, :t],
                        T_hist=t_hist[:, :t],
                        Y_hist=y_hist[:, :t],
                        t=t,
                        mask=mask_t[:, :t],
                    )
                    act = act.to(torch.long)
                    t_hist[:, t] = act
                    t_np = act.detach().cpu().numpy().astype(np.int64)
                else:
                    if t == t0:
                        t_np = np.full((bsz,), int(do_value), dtype=np.int64)
                    else:
                        t_np = T_obs[:, t]

                u_all = rng.random(bsz).astype(np.float32)
                if noise_std != 0.0:
                    noise_all = rng.normal(0.0, noise_std, size=bsz).astype(np.float32)
                else:
                    noise_all = np.zeros((bsz,), dtype=np.float32)

                for s in range(0, bsz, chunk):
                    e = min(bsz, s + chunk)
                    sl = slice(s, e)
                    a_t = A[sl, t]
                    help_flag = (t_np[sl] == 1).astype(np.float32)
                    logits = theta[sl] - beta[a_t] + gamma * help_flag
                    p = 1.0 / (1.0 + np.exp(-logits))
                    y_sampled = (u_all[sl] < p).astype(np.float32)

                    if policy is not None:
                        y_hist[sl, t] = torch.from_numpy(y_sampled)

                    y_out = p if use_prob else y_sampled
                    if start <= t <= start + H:
                        y_sum[sl, t - start] += y_out

                    m_t = mask[sl, t]
                    update = lr * (y_sampled - p) + delta * help_flag + noise_all[sl]
                    theta[sl] = theta[sl] + update * m_t

        y_mean = y_sum / float(n_mc)
        return y_mean.astype(np.float32)

    def _analytic_expected_outcomes_do(
        self,
        data: TrajectoryBatch,
        *,
        t0: int,
        horizon: int,
        action: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        A = data.A.detach().cpu().numpy().astype(np.int64)
        T_obs = data.T.detach().cpu().numpy().astype(np.int64)
        mask = data.mask.detach().cpu().numpy().astype(np.float32)
        n, seq_len = T_obs.shape
        t0 = int(t0)
        horizon = int(horizon)
        steps = min(horizon, seq_len - t0 - 1)
        H = max(0, int(steps))

        T_used = T_obs.copy()
        T_used[:, t0] = int(action)
        help_flag = (T_used == 1).astype(np.float32)
        cum_help = np.cumsum(help_flag, axis=1)
        theta0 = self._resolve_theta0(data).astype(np.float32)
        theta_t = theta0[:, None] + self.delta * np.concatenate(
            [np.zeros((n, 1), dtype=np.float32), cum_help[:, :-1]], axis=1
        )
        logits = theta_t - self.beta[A] + self.gamma * help_flag
        y_prob = 1.0 / (1.0 + np.exp(-logits))
        sl = slice(t0, t0 + H + 1)
        return y_prob[:, sl].astype(np.float32), mask[:, sl].astype(np.float32)

    def debug_sanity_check(self, batch: TrajectoryBatch) -> None:
        bsz = int(batch.X.shape[0])
        if bsz <= 0:
            return
        sub_n = min(128, bsz)
        sub = TrajectoryBatch(
            X=batch.X[:sub_n],
            A=batch.A[:sub_n],
            T=batch.T[:sub_n],
            Y=batch.Y[:sub_n],
            mask=batch.mask[:sub_n],
            lengths=batch.lengths[:sub_n],
            ids=batch.ids[:sub_n] if batch.ids is not None else None,
        )
        seq_len = int(sub.T.shape[1])
        if seq_len <= 0:
            return
        t0 = 0
        horizon = max(0, min(4, seq_len - 1))

        if self.lr == 0.0 and self.noise_std == 0.0:
            mc = self._mc_simulate(sub, t0=t0, do_value=1, horizon=horizon)
            analytic, _ = self._analytic_expected_outcomes_do(sub, t0=t0, horizon=horizon, action=1)
            diff = float(np.nanmean(np.abs(mc - analytic)))
            tol = max(0.02, 3.0 / np.sqrt(max(1, self.mc_samples)))
            if diff > tol:
                print(f"[OracleIRT] sanity check: MC vs analytic diff={diff:.4f} (tol={tol:.4f})")
            else:
                print(f"[OracleIRT] sanity check OK: diff={diff:.4f} (tol={tol:.4f})")
        else:
            mc1 = self._mc_simulate(sub, t0=t0, do_value=1, horizon=horizon)
            mc2 = self._mc_simulate(sub, t0=t0, do_value=1, horizon=horizon)
            max_diff = float(np.nanmax(np.abs(mc1 - mc2)))
            if max_diff > 0.0:
                print(f"[OracleIRT] sanity check warning: reproducibility diff={max_diff:.6f}")
            else:
                print("[OracleIRT] sanity check OK: reproducible MC oracle.")

    def expected_outcomes_do(
        self,
        data: TrajectoryBatch,
        *,
        t0: int,
        horizon: int,
        action: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        seq_len = int(data.T.shape[1])
        t0 = int(t0)
        horizon = int(horizon)
        steps = min(horizon, seq_len - t0 - 1)
        H = max(0, int(steps))
        y_slice = self._mc_simulate(data, t0=t0, do_value=int(action), horizon=H)
        m_slice = data.mask.detach().cpu().numpy().astype(np.float32)[:, t0 : t0 + H + 1]
        return y_slice.astype(np.float32), m_slice.astype(np.float32)

    def expected_outcomes_policy(
        self,
        data: TrajectoryBatch,
        *,
        policy: Policy,
        horizon: Optional[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        seq_len = int(data.T.shape[1])
        if horizon is None:
            horizon = seq_len - 1
        horizon = int(min(int(horizon), seq_len - 1))
        y_slice = self._mc_simulate(data, policy=policy, horizon=horizon)
        m_slice = data.mask.detach().cpu().numpy().astype(np.float32)[:, : horizon + 1]
        return y_slice.astype(np.float32), m_slice.astype(np.float32)

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

    def estimate_tau_per_sample(
        self,
        data: TrajectoryBatch,
        *,
        t0: int,
        horizon: int,
    ) -> np.ndarray:
        seq_len = int(data.T.shape[1])
        t0 = int(t0)
        horizon = int(horizon)
        steps = min(horizon, seq_len - t0 - 1)
        H = max(0, int(steps))
        y1 = self._mc_simulate(data, t0=t0, do_value=1, horizon=H)
        y0 = self._mc_simulate(data, t0=t0, do_value=0, horizon=H)
        diff = y1 - y0
        m_slice = data.mask.detach().cpu().numpy().astype(np.float32)[:, t0 : t0 + H + 1]
        denom = m_slice.sum(axis=1)
        denom = np.where(denom == 0.0, np.nan, denom)
        tau = (diff * m_slice).sum(axis=1) / denom
        return tau.astype(np.float32)
