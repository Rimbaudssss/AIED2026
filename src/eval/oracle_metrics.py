from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.data import TrajectoryBatch
from src.policy import Policy, get_default_policies


@dataclass(frozen=True)
class OraclePack:
    """Container for oracle counterfactuals in a semi-synthetic dataset.

    Expected shapes:
      - y_cf: [N, T, 2, Hmax+1] where y_cf[:, t0, a, s] is the oracle
        *probability* outcome at time (t0+s) under a single-step do(T_{t0}=a),
        with all other T fixed to the factual sequence.
      - y_policy: [N, P, T] where P corresponds to policy_names ordering.
    """

    y_cf: np.ndarray
    hmax: int
    y_policy: Optional[np.ndarray] = None
    policy_names: Optional[list[str]] = None


def _masked_mean(x: torch.Tensor, m: torch.Tensor, dim: int):
    denom = m.sum(dim=dim).clamp(min=1.0)
    return (x * m).sum(dim=dim) / denom


@torch.no_grad()
def compute_oracle_ite_metrics(
    *,
    gen_model,
    test_batch: TrajectoryBatch,
    oracle: OraclePack,
    dataset: str,
    model_name: str,
    t0_list: list[int],
    horizon: int,
    actions: tuple[int, int] = (0, 1),
    n_gen: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute ITE/ATE metrics against oracle counterfactuals (PEHE, ATE-RMSE).

    This is *only* used for semi-synthetic datasets that contain oracle truth.
    """

    a0, a1 = int(actions[0]), int(actions[1])
    if a0 != 0 or a1 != 1:
        # Oracle format assumes 2 arms indexed [0,1].
        raise ValueError("oracle y_cf assumes actions=(0,1)")

    y_cf = oracle.y_cf
    if y_cf.ndim != 4 or y_cf.shape[2] != 2:
        raise ValueError(f"Expected y_cf [N,T,2,Hmax+1], got {tuple(y_cf.shape)}")
    hmax = int(oracle.hmax)
    horizon = int(min(int(horizon), hmax))

    device = next(gen_model.model.parameters()).device  # type: ignore[attr-defined]
    B, T = int(test_batch.A.shape[0]), int(test_batch.A.shape[1])
    if y_cf.shape[0] != B:
        raise ValueError(f"oracle N mismatch: y_cf has {y_cf.shape[0]}, batch has {B}")
    if y_cf.shape[1] < T:
        raise ValueError(f"oracle T mismatch: y_cf has {y_cf.shape[1]}, batch has {T}")

    rows = []
    for t0 in [int(x) for x in t0_list]:
        if not (0 <= t0 < T):
            continue
        steps = min(horizon, T - t0 - 1)
        H = int(steps)
        sl = slice(t0, t0 + H + 1)
        m = test_batch.mask[:, sl].to(device).float()

        # Oracle truth (probabilities)
        true0 = torch.as_tensor(y_cf[:, t0, 0, : H + 1], device=device).float()
        true1 = torch.as_tensor(y_cf[:, t0, 1, : H + 1], device=device).float()
        tau_true = true1 - true0  # [B, H+1]

        # Generator predictions: average over n_gen rollouts to marginalize latent noise.
        def predict(action: int) -> torch.Tensor:
            acc = 0.0
            y_sum = torch.zeros((B, H + 1), device=device)
            for r in range(int(max(1, n_gen))):
                torch.manual_seed(int(seed) + 10007 * r + 13 * t0 + 3 * int(action))
                ro = gen_model.rollout(test_batch, do_t={t0: int(action)}, horizon=H, t0=t0, teacher_forcing=False)
                y = ro["Y_prob"][:, sl].to(device).float()
                y_sum += y
                acc += 1.0
            return y_sum / max(1.0, acc)

        pred0 = predict(a0)
        pred1 = predict(a1)
        tau_hat = pred1 - pred0

        # PEHE / ITE-RMSE
        mse_ite = _masked_mean((tau_hat - tau_true) ** 2, m, dim=1).mean().item()
        pehe = float(np.sqrt(max(0.0, mse_ite)))

        # ATE curve and ATE-RMSE across horizon
        ate_true = _masked_mean(tau_true, m, dim=0)
        ate_hat = _masked_mean(tau_hat, m, dim=0)
        ate_rmse = float(torch.sqrt(torch.mean((ate_hat - ate_true) ** 2)).item())
        ate_mae = float(torch.mean(torch.abs(ate_hat - ate_true)).item())

        rows.append(
            {
                "model": model_name,
                "dataset": dataset,
                "t0": t0,
                "horizon": H,
                "pehe": pehe,
                "ate_rmse": ate_rmse,
                "ate_mae": ate_mae,
                "n": int(B),
                "n_gen": int(max(1, n_gen)),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["model", "dataset", "t0", "horizon", "pehe", "ate_rmse", "ate_mae", "n", "n_gen"]
        )

    df = pd.DataFrame(rows)
    # Add a summary row per model/dataset (mean over t0)
    summary = {
        "model": model_name,
        "dataset": dataset,
        "t0": "avg",
        "horizon": int(df["horizon"].max()),
        "pehe": float(df["pehe"].mean()),
        "ate_rmse": float(df["ate_rmse"].mean()),
        "ate_mae": float(df["ate_mae"].mean()),
        "n": int(df["n"].iloc[0]),
        "n_gen": int(df["n_gen"].iloc[0]),
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    return df


@torch.no_grad()
def compute_oracle_policy_metrics(
    *,
    gen_model,
    test_batch: TrajectoryBatch,
    oracle: OraclePack,
    dataset: str,
    model_name: str,
    actions: list[int],
    horizon: int,
    policy_set: str = "fixed",
    seed: int = 0,
) -> pd.DataFrame:
    """Compute policy value errors and regret against oracle policy rollouts."""

    if oracle.y_policy is None or not oracle.policy_names:
        return pd.DataFrame(
            columns=[
                "model",
                "dataset",
                "policy",
                "horizon",
                "oracle_value",
                "gen_value",
                "abs_error",
                "supported",
                "skip_reason",
            ]
        )

    y_policy = oracle.y_policy
    names = list(oracle.policy_names)
    name2idx = {n: i for i, n in enumerate(names)}

    device = next(gen_model.model.parameters()).device  # type: ignore[attr-defined]
    B, T = int(test_batch.A.shape[0]), int(test_batch.A.shape[1])
    if y_policy.shape[0] != B:
        raise ValueError(f"oracle N mismatch: y_policy has {y_policy.shape[0]}, batch has {B}")

    horizon = int(min(int(horizon), T - 1))
    sl = slice(0, horizon + 1)
    m = test_batch.mask[:, sl].to(device).float()

    policies: list[Policy] = get_default_policies(T, action_space=actions, policy_set=policy_set)

    rows = []
    for pol in policies:
        pol_name = getattr(pol, "name", "policy")
        if pol_name not in name2idx:
            rows.append(
                {
                    "model": model_name,
                    "dataset": dataset,
                    "policy": pol_name,
                    "horizon": horizon,
                    "oracle_value": np.nan,
                    "gen_value": np.nan,
                    "abs_error": np.nan,
                    "supported": False,
                    "skip_reason": "oracle_missing_policy",
                }
            )
            continue

        oracle_y = torch.as_tensor(y_policy[:, name2idx[pol_name], :], device=device).float()[:, sl]
        oracle_value = float(_masked_mean(oracle_y, m, dim=1).mean().item())

        gen_value = np.nan
        supported = True
        skip_reason = ""
        try:
            torch.manual_seed(int(seed) + 997)
            ro = gen_model.rollout(test_batch, policy=pol, horizon=horizon, t0=0, teacher_forcing=False)
            y = ro["Y_prob"][:, sl].to(device).float()
            gen_value = float(_masked_mean(y, m, dim=1).mean().item())
        except NotImplementedError as e:
            supported = False
            skip_reason = str(e)

        abs_error = float(abs(gen_value - oracle_value)) if (supported and np.isfinite(gen_value)) else np.nan
        rows.append(
            {
                "model": model_name,
                "dataset": dataset,
                "policy": pol_name,
                "horizon": horizon,
                "oracle_value": oracle_value,
                "gen_value": gen_value,
                "abs_error": abs_error,
                "supported": supported,
                "skip_reason": skip_reason,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Summary: mean absolute value error and regret error within supported policies.
    df_supported = df[df["supported"] == True].copy()
    if not df_supported.empty:
        oracle_best = float(df_supported["oracle_value"].max())
        gen_best = float(df_supported["gen_value"].max())
        regret_error = float(abs(gen_best - oracle_best))
        value_abs_error = float(df_supported["abs_error"].mean())
    else:
        regret_error = np.nan
        value_abs_error = np.nan

    summary = {
        "model": model_name,
        "dataset": dataset,
        "policy": "__summary__",
        "horizon": horizon,
        "oracle_value": np.nan,
        "gen_value": np.nan,
        "abs_error": value_abs_error,
        "supported": bool(not df_supported.empty),
        "skip_reason": "",
        "regret_error": regret_error,
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    return df


def _masked_mean_np(y: np.ndarray, m: np.ndarray, *, axis: int) -> np.ndarray:
    denom = m.sum(axis=axis)
    denom = np.where(denom == 0.0, np.nan, denom)
    return (y * m).sum(axis=axis) / denom


def _oracle_skip_frames(
    *,
    model_name: str,
    dataset: str,
    t0_list: list[int],
    horizon: int,
    actions: list[int],
    subgroups: list[dict],
    policy_set: str,
    seq_len: int,
    skip_reason: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ite_rows = []
    t0_vals = t0_list if t0_list else [0]
    subgroup_vals = subgroups if subgroups else [{"name": "all"}]
    for t0 in t0_vals:
        for subgroup in subgroup_vals:
            ite_rows.append(
                {
                    "model": model_name,
                    "dataset": dataset,
                    "t0": int(t0),
                    "subgroup": str(subgroup.get("name", "all")),
                    "horizon": int(horizon),
                    "pehe": np.nan,
                    "ite_rmse": np.nan,
                    "ate_true": np.nan,
                    "ate_hat": np.nan,
                    "ate_abs_error": np.nan,
                    "ate_rmse": np.nan,
                    "n_effective": 0,
                    "tau_true_mean": np.nan,
                    "tau_hat_mean": np.nan,
                    "n_gen": 0,
                    "supported": False,
                    "skip_reason": skip_reason,
                }
            )

    policies: list[Policy] = get_default_policies(seq_len, action_space=actions, policy_set=policy_set)
    policy_rows = []
    for pol in policies:
        policy_rows.append(
            {
                "model": model_name,
                "dataset": dataset,
                "policy": getattr(pol, "name", "policy"),
                "horizon": int(horizon),
                "oracle_value": np.nan,
                "gen_value": np.nan,
                "value_abs_error": np.nan,
                "regret_oracle": np.nan,
                "supported": False,
                "skip_reason": skip_reason,
            }
        )

    return pd.DataFrame(ite_rows), pd.DataFrame(policy_rows)


@torch.no_grad()
def compute_oracle_metrics(
    *,
    gen_model,
    oracle_estimator,
    data: TrajectoryBatch,
    dataset_name: str,
    t0_list: list[int],
    horizon: int,
    actions: list[int],
    subgroups: list[dict],
    policy_set: str,
    seed: int,
    n_gen: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = str(dataset_name)
    model_name = getattr(gen_model, "name", "model")
    seq_len = int(data.T.shape[1])
    bsz = int(data.X.shape[0])
    print(
        f"[oracle_metrics] start model={model_name} dataset={dataset} "
        f"batch={bsz} horizon={int(horizon)} t0_list={t0_list} actions={actions} n_gen={int(n_gen)}"
    )

    if oracle_estimator is None or not getattr(oracle_estimator, "is_oracle", False):
        return _oracle_skip_frames(
            model_name=model_name,
            dataset=dataset,
            t0_list=t0_list,
            horizon=horizon,
            actions=actions,
            subgroups=subgroups,
            policy_set=policy_set,
            seq_len=seq_len,
            skip_reason="oracle_not_available",
        )

    if bsz <= 0:
        return _oracle_skip_frames(
            model_name=model_name,
            dataset=dataset,
            t0_list=t0_list,
            horizon=horizon,
            actions=actions,
            subgroups=subgroups,
            policy_set=policy_set,
            seq_len=seq_len,
            skip_reason="empty_oracle_batch",
        )

    if len(actions) < 2:
        return _oracle_skip_frames(
            model_name=model_name,
            dataset=dataset,
            t0_list=t0_list,
            horizon=horizon,
            actions=actions,
            subgroups=subgroups,
            policy_set=policy_set,
            seq_len=seq_len,
            skip_reason="actions_requires_two",
        )

    a0, a1 = int(actions[0]), int(actions[1])
    n_gen_eff = int(max(1, n_gen))
    ite_rows = []

    subgroup_vals = subgroups if subgroups else [{"name": "all"}]
    for t0 in [int(x) for x in t0_list]:
        if not (0 <= t0 < seq_len):
            for subgroup in subgroup_vals:
                ite_rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset,
                        "t0": int(t0),
                        "subgroup": str(subgroup.get("name", "all")),
                        "horizon": int(horizon),
                        "pehe": np.nan,
                        "ite_rmse": np.nan,
                        "ate_true": np.nan,
                        "ate_hat": np.nan,
                        "ate_abs_error": np.nan,
                        "ate_rmse": np.nan,
                        "n_effective": 0,
                        "tau_true_mean": np.nan,
                        "tau_hat_mean": np.nan,
                        "n_gen": n_gen_eff,
                        "supported": False,
                        "skip_reason": "t0_out_of_range",
                    }
                )
            continue
        steps = min(int(horizon), seq_len - t0 - 1)
        H = max(0, int(steps))
        sl = slice(t0, t0 + H + 1)
        print(f"[oracle_metrics] t0={int(t0)} horizon_eff={int(H)}")

        try:
            tau_true_all = oracle_estimator.estimate_tau_per_sample(data, t0=t0, horizon=H)
            y_true0, m_slice = oracle_estimator.expected_outcomes_do(data, t0=t0, horizon=H, action=a0)
            y_true1, _ = oracle_estimator.expected_outcomes_do(data, t0=t0, horizon=H, action=a1)
        except Exception as e:
            for subgroup in subgroup_vals:
                ite_rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset,
                        "t0": int(t0),
                        "subgroup": str(subgroup.get("name", "all")),
                        "horizon": int(H),
                        "pehe": np.nan,
                        "ite_rmse": np.nan,
                        "ate_true": np.nan,
                        "ate_hat": np.nan,
                        "ate_abs_error": np.nan,
                        "ate_rmse": np.nan,
                        "n_effective": 0,
                        "tau_true_mean": np.nan,
                        "tau_hat_mean": np.nan,
                        "n_gen": n_gen_eff,
                        "supported": False,
                        "skip_reason": str(e),
                    }
                )
            continue

        def _gen_mean(action: int) -> np.ndarray:
            print(f"[oracle_metrics]   action={int(action)} rollouts={int(n_gen_eff)}")
            acc = None
            progress_every = max(1, int(n_gen_eff // 4))
            for r in range(n_gen_eff):
                torch.manual_seed(int(seed) + 10007 * r + 13 * t0 + 3 * int(action))
                ro = gen_model.rollout(data, do_t={t0: int(action)}, horizon=H, t0=t0, teacher_forcing=False)
                y = ro["Y_prob"][:, sl].detach().cpu().numpy().astype(np.float32)
                if acc is None:
                    acc = y
                else:
                    acc = acc + y
                if (r + 1) % progress_every == 0 or r == n_gen_eff - 1:
                    print(f"[oracle_metrics]     rollout {r + 1}/{int(n_gen_eff)} done")
            if acc is None:
                return np.zeros_like(y_true0)
            return acc / float(n_gen_eff)

        y_hat0 = _gen_mean(a0)
        y_hat1 = _gen_mean(a1)

        diff_true = y_true1 - y_true0
        diff_hat = y_hat1 - y_hat0
        tau_hat_all = _masked_mean_np(diff_hat, m_slice, axis=1)

        for subgroup in subgroup_vals:
            name = str(subgroup.get("name", "all"))
            if "mask" in subgroup:
                sel = np.asarray(subgroup["mask"]).astype(bool)
            else:
                sel = np.ones((diff_true.shape[0],), dtype=bool)

            n_eff = int(sel.sum())
            if n_eff == 0:
                ite_rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset,
                        "t0": int(t0),
                        "subgroup": name,
                        "horizon": int(H),
                        "pehe": np.nan,
                        "ite_rmse": np.nan,
                        "ate_true": np.nan,
                        "ate_hat": np.nan,
                        "ate_abs_error": np.nan,
                        "ate_rmse": np.nan,
                        "n_effective": 0,
                        "tau_true_mean": np.nan,
                        "tau_hat_mean": np.nan,
                        "n_gen": n_gen_eff,
                        "skip_reason": "empty_subgroup",
                    }
                )
                continue

            diff_true_s = diff_true[sel]
            diff_hat_s = diff_hat[sel]
            m_slice_s = m_slice[sel]

            tau_true = tau_true_all[sel]
            tau_hat = tau_hat_all[sel]
            ate_true_curve = _masked_mean_np(diff_true_s, m_slice_s, axis=0)
            ate_hat_curve = _masked_mean_np(diff_hat_s, m_slice_s, axis=0)

            ite_mse = np.nanmean((tau_hat - tau_true) ** 2)
            pehe = float(np.sqrt(max(0.0, ite_mse)))

            ate_true = float(np.nanmean(tau_true))
            ate_hat = float(np.nanmean(tau_hat))
            ate_abs_error = float(abs(ate_hat - ate_true))
        ate_rmse = float(np.sqrt(np.nanmean((ate_hat_curve - ate_true_curve) ** 2)))

        ite_rows.append(
            {
                "model": model_name,
                    "dataset": dataset,
                    "t0": int(t0),
                    "subgroup": name,
                    "horizon": int(H),
                    "pehe": pehe,
                    "ite_rmse": pehe,
                    "ate_true": ate_true,
                    "ate_hat": ate_hat,
                    "ate_abs_error": ate_abs_error,
                "ate_rmse": ate_rmse,
                "n_effective": n_eff,
                "tau_true_mean": float(np.nanmean(tau_true)),
                "tau_hat_mean": float(np.nanmean(tau_hat)),
                "n_gen": n_gen_eff,
                "supported": True,
                "skip_reason": "",
            }
        )

    if not ite_rows:
        return _oracle_skip_frames(
            model_name=model_name,
            dataset=dataset,
            t0_list=t0_list,
            horizon=horizon,
            actions=actions,
            subgroups=subgroups,
            policy_set=policy_set,
            seq_len=seq_len,
            skip_reason="oracle_t0_out_of_range",
        )

    df_ite = pd.DataFrame(ite_rows)

    policy_rows = []
    policies: list[Policy] = get_default_policies(seq_len, action_space=actions, policy_set=policy_set)
    for pol in policies:
        pol_name = getattr(pol, "name", "policy")
        print(f"[oracle_metrics] policy_eval policy={pol_name}")
        supported = True
        skip_reason = ""
        try:
            oracle_out = oracle_estimator.estimate_policy_value(
                data, policy=pol, horizon=int(horizon), n_boot=0, seed=seed
            )
            oracle_value = float(oracle_out.get("value", np.nan))
        except Exception as e:
            oracle_value = np.nan
            supported = False
            skip_reason = str(e)

        gen_value = np.nan
        try:
            ro = gen_model.rollout(data, policy=pol, horizon=int(horizon), t0=0, teacher_forcing=False)
            y = ro["Y_prob"].detach().cpu().numpy()
            m = ro["mask"].detach().cpu().numpy()
            y_slice = y[:, : int(horizon) + 1]
            m_slice = m[:, : int(horizon) + 1]
            gen_value = float(np.nanmean((y_slice * m_slice).sum(axis=1) / np.maximum(1.0, m_slice.sum(axis=1))))
        except Exception as e:
            supported = False
            skip_reason = str(e)

        value_abs_error = float(abs(gen_value - oracle_value)) if supported and np.isfinite(gen_value) else np.nan
        policy_rows.append(
            {
                "model": model_name,
                "dataset": dataset,
                "policy": pol_name,
                "horizon": int(horizon),
                "oracle_value": oracle_value,
                "gen_value": gen_value,
                "value_abs_error": value_abs_error,
                "regret_oracle": np.nan,
                "supported": bool(supported),
                "skip_reason": skip_reason,
            }
        )

    df_policy = pd.DataFrame(policy_rows)
    if not df_policy.empty:
        oracle_vals = df_policy["oracle_value"].to_numpy(dtype=np.float64)
        oracle_best = np.nanmax(oracle_vals) if np.isfinite(oracle_vals).any() else np.nan

        gen_sel = df_policy[df_policy["supported"] == True]
        gen_sel = gen_sel[np.isfinite(gen_sel["gen_value"].to_numpy(dtype=np.float64))]
        if not gen_sel.empty and np.isfinite(oracle_best):
            best_idx = gen_sel["gen_value"].idxmax()
            oracle_selected = float(df_policy.loc[best_idx, "oracle_value"])
            regret = float(oracle_best - oracle_selected)
        else:
            regret = np.nan
        df_policy["regret_oracle"] = regret

    return df_ite, df_policy
