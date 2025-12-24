from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.baselines import BaseSeqModel
from src.causal_estimators.base import CausalEstimator
from src.data import TrajectoryBatch
from src.policy import Policy, get_default_policies


def _mean_curve(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.float64)
    denom = mask.sum(axis=0)
    denom[denom == 0.0] = np.nan
    return (y * mask).sum(axis=0) / denom


def compute_causal_bias(
    *,
    gen_model: BaseSeqModel,
    ref_estimator: CausalEstimator,
    data: TrajectoryBatch,
    t0_list: List[int],
    horizon: int,
    actions: List[int],
    subgroups: List[dict],
    n_gen: int = 50,
    seed: int = 0,
    policy_set: str = "fixed",
) -> Tuple[pd.DataFrame, Dict]:
    rng = np.random.default_rng(int(seed))
    rows = []
    seq_len = int(data.T.shape[1])
    horizon = int(horizon)

    for t0 in t0_list:
        t0 = int(t0)
        if not (0 <= t0 < seq_len):
            continue
        for subgroup in subgroups:
            subgroup_name = str(subgroup.get("name", "all"))
            if "mask" in subgroup:
                sel = np.asarray(subgroup["mask"]).astype(bool)
            else:
                sel = np.ones(data.X.shape[0], dtype=bool)
            for action in actions:
                ref = ref_estimator.estimate_do(
                    data,
                    t0=t0,
                    horizon=horizon,
                    action=int(action),
                    subgroup=subgroup,
                    n_boot=200,
                    seed=seed,
                )
                ref_mu = np.asarray(ref["mu"], dtype=np.float64)
                ref_ci_low = np.asarray(ref["ci_low"], dtype=np.float64)
                ref_ci_high = np.asarray(ref["ci_high"], dtype=np.float64)

                gen_samples = []
                for _ in range(int(n_gen)):
                    do_t = {t0: int(action)}
                    ro = gen_model.rollout(data, do_t=do_t, horizon=horizon, t0=t0, teacher_forcing=False)
                    y_prob = ro["Y_prob"].detach().cpu().numpy()
                    m = ro["mask"].detach().cpu().numpy()
                    y_slice = y_prob[:, t0 : t0 + horizon + 1]
                    m_slice = m[:, t0 : t0 + horizon + 1]
                    gen_samples.append(_mean_curve(y_slice[sel], m_slice[sel]))
                gen_samples = np.stack(gen_samples, axis=0)
                gen_mu = np.nanmean(gen_samples, axis=0)
                gen_std = np.nanstd(gen_samples, axis=0)

                row = {
                    "model": gen_model.name,
                    "ref_estimator": ref_estimator.name,
                    "dataset": subgroup.get("dataset", "unknown"),
                    "t0": int(t0),
                    "horizon": int(horizon),
                    "subgroup": subgroup_name,
                    "action": int(action),
                    "n_effective": int(ref.get("n", 0)),
                }
                for h in range(horizon + 1):
                    row[f"ref_mu_{h}"] = float(ref_mu[h]) if h < len(ref_mu) else np.nan
                    row[f"ref_ci_low_{h}"] = float(ref_ci_low[h]) if h < len(ref_ci_low) else np.nan
                    row[f"ref_ci_high_{h}"] = float(ref_ci_high[h]) if h < len(ref_ci_high) else np.nan
                    row[f"gen_mu_{h}"] = float(gen_mu[h]) if h < len(gen_mu) else np.nan
                    row[f"gen_std_{h}"] = float(gen_std[h]) if h < len(gen_std) else np.nan
                rows.append(row)

    df = pd.DataFrame(rows)
    summary: Dict[str, float] = {
        "model": gen_model.name,
        "ref_estimator": ref_estimator.name,
        "dataset": subgroups[0].get("dataset", "unknown") if subgroups else "unknown",
        "horizon": int(horizon),
    }

    def _get_curve(df_sub: pd.DataFrame, action_val: int, prefix: str) -> np.ndarray:
        cols = [f"{prefix}{h}" for h in range(horizon + 1)]
        sub = df_sub[df_sub["action"] == int(action_val)]
        return sub[cols].to_numpy(dtype=np.float64)

    if not df.empty:
        ate_errors = []
        cate_errors = []
        for t0 in df["t0"].unique():
            df_t0 = df[df["t0"] == t0]
            for subgroup_name in df_t0["subgroup"].unique():
                df_s = df_t0[df_t0["subgroup"] == subgroup_name]
                ref_0 = _get_curve(df_s, 0, "ref_mu_")
                ref_1 = _get_curve(df_s, 1, "ref_mu_")
                gen_0 = _get_curve(df_s, 0, "gen_mu_")
                gen_1 = _get_curve(df_s, 1, "gen_mu_")
                if ref_0.size == 0 or ref_1.size == 0 or gen_0.size == 0 or gen_1.size == 0:
                    continue
                ate_ref = ref_1 - ref_0
                ate_gen = gen_1 - gen_0
                err = np.abs(ate_gen - ate_ref)
                if subgroup_name == "all":
                    ate_errors.append(err)
                else:
                    cate_errors.append(err)

        def _agg(err_list: List[np.ndarray]) -> tuple[float, float]:
            if not err_list:
                return float("nan"), float("nan")
            err = np.concatenate(err_list, axis=None)
            return float(np.nanmean(err)), float(np.sqrt(np.nanmean(err ** 2)))

        ate_mae, ate_rmse = _agg(ate_errors)
        cate_mae, _ = _agg(cate_errors)
        summary["ate_mae_mean"] = ate_mae
        summary["ate_rmse_mean"] = ate_rmse
        summary["cate_mae_mean"] = cate_mae

    policies: List[Policy] = get_default_policies(seq_len, actions, policy_set=policy_set)
    ref_vals = []
    gen_vals = []
    policy_supported = True
    policy_skip_reason = ""
    for policy in policies:
        ref_out = ref_estimator.estimate_policy_value(data, policy=policy, horizon=horizon, n_boot=200, seed=seed)
        ref_vals.append(float(ref_out.get("value", np.nan)))

        try:
            ro = gen_model.rollout(data, policy=policy, horizon=horizon, t0=0, teacher_forcing=True)
            y = ro["Y_prob"].detach().cpu().numpy()
            m = ro["mask"].detach().cpu().numpy()
            y_slice = y[:, : horizon + 1]
            m_slice = m[:, : horizon + 1]
            val = float(np.nanmean((y_slice * m_slice).sum(axis=1) / np.maximum(1.0, m_slice.sum(axis=1))))
        except Exception as e:
            policy_supported = False
            if not policy_skip_reason:
                policy_skip_reason = str(e)
            val = float("nan")
        gen_vals.append(val)

    ref_arr = np.asarray(ref_vals, dtype=np.float64)
    gen_arr = np.asarray(gen_vals, dtype=np.float64)
    summary["value_abs_error"] = float(np.nanmean(np.abs(ref_arr - gen_arr)))
    if np.any(np.isfinite(ref_arr)) and np.any(np.isfinite(gen_arr)):
        summary["regret_error"] = float(np.abs(np.nanmax(ref_arr) - np.nanmax(gen_arr)))
    else:
        summary["regret_error"] = float("nan")
    summary["n_t0"] = int(len(t0_list))
    summary["n_subgroups"] = int(len(subgroups))
    summary["policy_supported"] = 1.0 if policy_supported else 0.0
    if not policy_supported and policy_skip_reason:
        summary["policy_skip_reason"] = policy_skip_reason
    return df, summary


def compute_policy_values(
    *,
    gen_model: BaseSeqModel,
    ref_estimator: CausalEstimator,
    data: TrajectoryBatch,
    actions: List[int],
    horizon: int,
    dataset: str,
    n_boot: int = 200,
    seed: int = 0,
    policy_set: str = "fixed",
) -> Tuple[pd.DataFrame, bool, str]:
    seq_len = int(data.T.shape[1])
    policies: List[Policy] = get_default_policies(seq_len, actions, policy_set=policy_set)
    rows = []
    policy_supported = True
    policy_skip_reason = ""

    for policy in policies:
        ref_out = ref_estimator.estimate_policy_value(data, policy=policy, horizon=horizon, n_boot=n_boot, seed=seed)
        ref_value = float(ref_out.get("value", np.nan))
        ref_ci_low = float(ref_out.get("ci_low", np.nan))
        ref_ci_high = float(ref_out.get("ci_high", np.nan))

        gen_value = float("nan")
        skip_reason = ""
        supported = 1.0
        try:
            ro = gen_model.rollout(data, policy=policy, horizon=horizon, t0=0, teacher_forcing=True)
            y = ro["Y_prob"].detach().cpu().numpy()
            m = ro["mask"].detach().cpu().numpy()
            y_slice = y[:, : horizon + 1]
            m_slice = m[:, : horizon + 1]
            gen_value = float(np.nanmean((y_slice * m_slice).sum(axis=1) / np.maximum(1.0, m_slice.sum(axis=1))))
        except Exception as e:
            supported = 0.0
            policy_supported = False
            skip_reason = str(e)
            if not policy_skip_reason:
                policy_skip_reason = skip_reason

        rows.append(
            {
                "model": gen_model.name,
                "ref_estimator": ref_estimator.name,
                "dataset": dataset,
                "policy": getattr(policy, "name", "policy"),
                "horizon": int(horizon),
                "ref_value": ref_value,
                "ref_ci_low": ref_ci_low,
                "ref_ci_high": ref_ci_high,
                "gen_value": gen_value,
                "supported": supported,
                "skip_reason": skip_reason,
            }
        )

    df = pd.DataFrame(rows)
    return df, policy_supported, policy_skip_reason
