from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.baselines import BaseSeqModel
from src.causal_estimators.iptw_msm import IPTWMSM
from src.data import TrajectoryBatch
from src.policy import Policy, get_default_policies


def _squeeze_y(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 3 and y.shape[-1] == 1:
        return y.squeeze(-1)
    return y


def _predict_factual(
    gen_model: BaseSeqModel, batch: TrajectoryBatch, *, horizon: int
) -> torch.Tensor:
    """Predict Y under observed actions (factual) as probabilities using teacher forcing."""
    model = getattr(gen_model, "model", None)
    model_name = str(getattr(gen_model, "name", ""))
    if model is None:
        raise RuntimeError("factual prediction requires a torch model.")

    X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask
    with torch.no_grad():
        if model_name.startswith("scm"):
            out = model.teacher_forcing(x=X, a=A, t=T, y=Y, mask=M, eps=None, eps_mode="zero")
            y_prob = torch.sigmoid(out["y_logits"])
        elif model_name == "rcgan":
            d_eps = int(getattr(getattr(model, "cfg", None), "d_eps", 16))
            eps = torch.zeros(A.shape[0], A.shape[1], d_eps, device=X.device)
            out = model.teacher_forcing(x=X, a=A, t=T, y=Y, mask=M, eps=eps, stochastic_y=False)
            y_prob = torch.sigmoid(out["y_logits"])
        elif model_name == "crn":
            out = model.forward(x=X, a=A, t=T, y=Y, mask=M)
            y_prob = torch.sigmoid(out["y_logits"])
        elif model_name == "vae":
            mu, _logvar = model.encode(x=X, a=A, t=T, y=Y, mask=M)
            out = model.decode(x=X, a=A, t=T, mask=M, z=mu, y=Y, teacher_forcing=True, stochastic_y=False)
            y_prob = torch.sigmoid(out["y_logits"])
        elif model_name == "timegan":
            z_dim = int(getattr(getattr(model, "cfg", None), "z_dim", 16))
            z = torch.zeros(A.shape[0], A.shape[1], z_dim, device=X.device)
            out = model.teacher_forcing(x=X, a=A, t=T, y=None, mask=M, z=z, stochastic_y=False)
            y_prob = torch.sigmoid(out["y_logits"])
        else:
            raise RuntimeError(f"unsupported model for factual DR: {model_name}")

    return _squeeze_y(y_prob)[:, : horizon + 1]


def _policy_actions(
    policy: Policy, batch: TrajectoryBatch, *, horizon: int
) -> torch.Tensor:
    """Return policy actions [B, H+1] using observed history."""
    X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask
    bsz, seq_len = T.shape
    end = min(seq_len, int(horizon) + 1)
    actions = torch.zeros((bsz, end), device=X.device, dtype=torch.long)
    for t in range(end):
        act = policy.act(
            X=X,
            A_hist=A[:, :t],
            T_hist=T[:, :t],
            Y_hist=Y[:, :t],
            t=t,
            mask=M[:, :t],
        )
        actions[:, t] = act.long()
    return actions


def _make_do_t(actions: torch.Tensor) -> Optional[dict]:
    """Return do_t dict if all samples share the same action per time."""
    bsz, seq_len = actions.shape
    if bsz == 0:
        return None
    per_t = []
    for t in range(seq_len):
        col = actions[:, t]
        if not torch.all(col == col[0]):
            return None
        per_t.append(int(col[0].item()))
    do_t = {int(t): int(a) for t, a in enumerate(per_t)}
    return do_t


def _dr_correction(
    *,
    logp_beh: torch.Tensor,
    match: torch.Tensor,
    y_obs: torch.Tensor,
    y_hat_obs: torch.Tensor,
    mask: torch.Tensor,
    weight_clip: float,
    eps: float,
) -> tuple[float, dict]:
    logp_beh = torch.clamp(logp_beh, min=float(np.log(eps)))
    neg_inf = torch.full_like(logp_beh, -1e9)
    log_w_step = torch.where(match, -logp_beh, neg_inf)
    log_w_step = torch.where(mask > 0.5, log_w_step, torch.zeros_like(log_w_step))
    log_w_cum = torch.cumsum(log_w_step, dim=1)
    w_cum = torch.exp(log_w_cum)
    if weight_clip > 0.0:
        w_cum = torch.clamp(w_cum, max=float(weight_clip))
    denom = mask.sum().clamp(min=1.0)
    correction = (w_cum * (y_obs - y_hat_obs) * mask).sum() / denom
    stats = {
        "w_mean": float(w_cum[mask > 0.5].mean().item()) if torch.any(mask > 0.5) else float("nan"),
        "w_max": float(w_cum.max().item()) if w_cum.numel() else float("nan"),
    }
    return float(correction.item()), stats


def compute_dr_policy_values(
    *,
    gen_model: BaseSeqModel,
    propensity_estimator: IPTWMSM,
    data: TrajectoryBatch,
    actions: list[int],
    horizon: int,
    dataset: str,
    seed: int = 0,
    policy_set: str = "fixed",
    weight_clip: float = 10.0,
    weight_mean_threshold: float = 5.0,
    eps: float = 1e-6,
) -> tuple[pd.DataFrame, dict]:
    """Compute doubly robust policy values for a set of policies."""
    _ = seed
    device = next(getattr(gen_model, "model").parameters()).device  # type: ignore[call-arg]
    batch = data.to(device)
    X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask
    seq_len = int(T.shape[1])
    horizon = int(min(int(horizon), seq_len - 1))
    end = min(seq_len, horizon + 1)

    logp_beh = propensity_estimator.behavior_log_probs(data)
    logp_beh = torch.as_tensor(logp_beh, device=device).float()[:, :end]

    try:
        y_hat_obs = _predict_factual(gen_model, batch, horizon=horizon)
        y_hat_obs = _squeeze_y(y_hat_obs)
    except Exception as e:
        policies = get_default_policies(end, actions, policy_set=policy_set)
        rows = []
        for policy in policies:
            rows.append(
                {
                    "model": gen_model.name,
                    "dataset": dataset,
                    "policy": getattr(policy, "name", "policy"),
                    "horizon": int(horizon),
                    "dr_value": np.nan,
                    "dm_value": np.nan,
                    "correction": np.nan,
                    "w_mean": np.nan,
                    "w_max": np.nan,
                    "supported": 0.0,
                    "skip_reason": f"factual_pred_failed: {e}",
                }
            )
        df = pd.DataFrame(rows)
        summary = {
            "model": gen_model.name,
            "dataset": dataset,
            "horizon": int(horizon),
            "dr_value_mean": float("nan"),
            "dm_value_mean": float("nan"),
            "correction_mean": float("nan"),
            "policy_supported": 0.0,
            "policy_skip_reason": f"factual_pred_failed: {e}",
        }
        return df, summary
    y_obs = _squeeze_y(Y)[:, :end]
    mask = M[:, :end].float()

    policies = get_default_policies(end, actions, policy_set=policy_set)
    rows = []
    policy_supported = True
    policy_skip_reason = ""

    for policy in policies:
        dm_value = float("nan")
        dr_value = float("nan")
        correction = float("nan")
        w_mean = float("nan")
        w_max = float("nan")
        supported = 1.0
        skip_reason = ""

        pi_actions = _policy_actions(policy, batch, horizon=horizon)
        match = (T[:, :end].long() == pi_actions.long())

        try:
            ro = gen_model.rollout(batch, policy=policy, horizon=horizon, t0=0, teacher_forcing=True)
            y_policy = _squeeze_y(ro["Y_prob"])[:, :end]
        except Exception as e:
            do_t = _make_do_t(pi_actions)
            if do_t is None:
                supported = 0.0
                policy_supported = False
                skip_reason = str(e)
                if not policy_skip_reason:
                    policy_skip_reason = skip_reason
                rows.append(
                    {
                        "model": gen_model.name,
                        "dataset": dataset,
                        "policy": getattr(policy, "name", "policy"),
                        "horizon": int(horizon),
                        "dr_value": dr_value,
                        "dm_value": dm_value,
                        "correction": correction,
                        "w_mean": w_mean,
                        "w_max": w_max,
                        "supported": supported,
                        "skip_reason": skip_reason,
                    }
                )
                continue
            ro = gen_model.rollout(batch, do_t=do_t, horizon=horizon, t0=0, teacher_forcing=True)
            y_policy = _squeeze_y(ro["Y_prob"])[:, :end]

        denom = mask.sum().clamp(min=1.0)
        dm_value = float(((y_policy * mask).sum() / denom).item())
        correction, stats = _dr_correction(
            logp_beh=logp_beh,
            match=match,
            y_obs=y_obs,
            y_hat_obs=y_hat_obs,
            mask=mask,
            weight_clip=float(weight_clip),
            eps=float(eps),
        )
        w_mean = stats.get("w_mean", float("nan"))
        w_max = stats.get("w_max", float("nan"))
        dr_value = float(dm_value + correction)
        if (not np.isfinite(w_mean)) or (weight_mean_threshold > 0.0 and w_mean > weight_mean_threshold):
            supported = 0.0
            skip_reason = "weight_mean_exceeds_threshold"
            dr_value = float("nan")
            dm_value = float("nan")
            correction = float("nan")

        rows.append(
            {
                "model": gen_model.name,
                "dataset": dataset,
                "policy": getattr(policy, "name", "policy"),
                "horizon": int(horizon),
                "dr_value": dr_value,
                "dm_value": dm_value,
                "correction": correction,
                "w_mean": w_mean,
                "w_max": w_max,
                "supported": supported,
                "skip_reason": skip_reason,
            }
        )

    df = pd.DataFrame(rows)
    if "supported" in df.columns:
        df_supported = df[df["supported"] == 1.0]
    else:
        df_supported = df
    dr_vals = df_supported["dr_value"].to_numpy(dtype=np.float64) if "dr_value" in df_supported.columns else np.array([])
    dm_vals = df_supported["dm_value"].to_numpy(dtype=np.float64) if "dm_value" in df_supported.columns else np.array([])
    corr_vals = df_supported["correction"].to_numpy(dtype=np.float64) if "correction" in df_supported.columns else np.array([])

    def _mean(arr: np.ndarray) -> float:
        if arr.size == 0:
            return float("nan")
        arr = arr[np.isfinite(arr)]
        return float(np.nanmean(arr)) if arr.size else float("nan")

    summary = {
        "model": gen_model.name,
        "dataset": dataset,
        "horizon": int(horizon),
        "dr_value_mean": _mean(dr_vals),
        "dm_value_mean": _mean(dm_vals),
        "correction_mean": _mean(corr_vals),
        "policy_supported": 1.0 if not df_supported.empty else 0.0,
    }
    if not policy_supported and policy_skip_reason:
        summary["policy_skip_reason"] = policy_skip_reason
    return df, summary
