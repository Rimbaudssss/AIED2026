"""Evaluate counterfactual consistency on latent K_c and optionally plot bifurcation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import NPZSequenceDataset, TrajectoryBatch, compute_lengths
from src.model.baselines import load_rollout_model_from_checkpoint


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _batch_from_dataset(ds: NPZSequenceDataset, indices: np.ndarray) -> TrajectoryBatch:
    X = torch.as_tensor(ds.X[indices]).float()
    A = torch.as_tensor(ds.A[indices])
    T = torch.as_tensor(ds.T[indices])
    Y = torch.as_tensor(ds.Y[indices]).float()
    M = torch.as_tensor(ds.M[indices]).float()
    return TrajectoryBatch(X=X, A=A, T=T, Y=Y, mask=M, lengths=compute_lengths(M))


def _parse_int_list(text: str) -> list[int]:
    items = [s.strip() for s in text.split(",") if s.strip()]
    return [int(x) for x in items] if items else []


def _compute_cf_history_mse(
    model: torch.nn.Module,
    batch: TrajectoryBatch,
    t_list: list[int],
    action_a: int,
    action_b: int,
    seed: int,
) -> list[tuple[int, float]]:
    model.eval()
    device = next(model.parameters()).device
    batch = batch.to(device)
    bsz, seq_len = batch.T.shape[0], batch.T.shape[1]
    rng = np.random.default_rng(int(seed))
    results: list[tuple[int, float]] = []

    for t in t_list:
        t = int(t)
        if t < 0 or t >= seq_len:
            continue
        eps = torch.as_tensor(rng.standard_normal(size=(bsz, seq_len, model.cfg.d_eps)), device=device).float()
        do_a = {t: torch.full((bsz,), int(action_a), device=device, dtype=batch.T.dtype)}
        do_b = {t: torch.full((bsz,), int(action_b), device=device, dtype=batch.T.dtype)}

        ro_a = model.rollout(
            x=batch.X,
            a=batch.A,
            t_obs=batch.T,
            do_t=do_a,
            mask=batch.mask,
            eps=eps,
            steps=seq_len,
            stochastic_y=False,
        )
        ro_b = model.rollout(
            x=batch.X,
            a=batch.A,
            t_obs=batch.T,
            do_t=do_b,
            mask=batch.mask,
            eps=eps,
            steps=seq_len,
            stochastic_y=False,
        )
        if "k_c" in ro_a and "k_c" in ro_b:
            k_a = ro_a["k_c"][:, : t + 1, :]
            k_b = ro_b["k_c"][:, : t + 1, :]
        elif "k" in ro_a and "k" in ro_b:
            k_a = ro_a["k"][:, : t + 1, :]
            k_b = ro_b["k"][:, : t + 1, :]
        else:
            raise RuntimeError("rollout() did not return latent K/K_c; counterfactual consistency requires K.")
        diff_sq = (k_a - k_b).pow(2).mean(dim=-1)
        mask_k = torch.cat([torch.ones(bsz, 1, device=device), batch.mask[:, :t]], dim=1)
        mse = (diff_sq * mask_k).sum() / mask_k.sum().clamp(min=1.0)
        results.append((t, float(mse.item())))

    return results


def _project_k(k_seq: np.ndarray, method: str, pc_vec: np.ndarray | None) -> np.ndarray:
    if method == "k_norm":
        return np.linalg.norm(k_seq, axis=1)
    if method == "k_mean":
        return k_seq.mean(axis=1)
    if method == "k_pca":
        if pc_vec is None:
            raise ValueError("pc_vec required for k_pca.")
        return (k_seq - k_seq.mean(axis=0, keepdims=True)) @ pc_vec
    raise ValueError(f"Unknown plot_metric={method}")


def _score_pre_intervention(batch: TrajectoryBatch, t0: int) -> torch.Tensor:
    Y = batch.Y
    if Y.ndim == 3:
        Y = Y.squeeze(-1)
    if t0 <= 0:
        return torch.zeros(batch.Y.shape[0])
    pre_m = batch.mask[:, :t0].float()
    pre_y = Y[:, :t0].float()
    den = pre_m.sum(dim=1).clamp(min=1.0)
    return (pre_y * pre_m).sum(dim=1) / den


def _select_student_idx(
    model: torch.nn.Module,
    batch: TrajectoryBatch,
    *,
    t_intervention: int,
    action_a: int,
    action_b: int,
    plot_horizon: int,
    seed: int,
    select: str,
    causal_only: bool,
) -> int:
    select = str(select).lower()
    if select == "first":
        return 0
    if select == "median":
        score = _score_pre_intervention(batch, t_intervention)
        sorted_idx = torch.argsort(score)
        return int(sorted_idx[len(sorted_idx) // 2].item())
    if select in {"best", "worst"}:
        score = _score_pre_intervention(batch, t_intervention)
        idx = torch.argmax(score) if select == "best" else torch.argmin(score)
        return int(idx.item())
    if select == "random":
        rng = np.random.default_rng(int(seed))
        return int(rng.integers(low=0, high=int(batch.X.shape[0])))

    # default: max divergence between do(T=a) and do(T=b)
    model.eval()
    device = next(model.parameters()).device
    batch = batch.to(device)
    bsz, seq_len = batch.T.shape
    steps = min(int(plot_horizon) + 1, int(seq_len))
    t_intervention = max(0, min(int(t_intervention), steps - 1))
    rng = np.random.default_rng(int(seed))
    eps = torch.as_tensor(rng.standard_normal(size=(bsz, steps, model.cfg.d_eps)), device=device).float()

    do_a = {t_intervention: torch.full((bsz,), int(action_a), device=device, dtype=batch.T.dtype)}
    do_b = {t_intervention: torch.full((bsz,), int(action_b), device=device, dtype=batch.T.dtype)}

    ro_a = model.rollout(
        x=batch.X,
        a=batch.A,
        t_obs=batch.T,
        do_t=do_a,
        mask=batch.mask,
        eps=eps,
        steps=steps,
        stochastic_y=False,
        causal_only=causal_only,
    )
    ro_b = model.rollout(
        x=batch.X,
        a=batch.A,
        t_obs=batch.T,
        do_t=do_b,
        mask=batch.mask,
        eps=eps,
        steps=steps,
        stochastic_y=False,
        causal_only=causal_only,
    )
    score_y = None
    score_k = None
    if select in {"max_delta", "max_delta_y", "max_delta_combined"}:
        y_a = ro_a["y"]
        y_b = ro_b["y"]
        if y_a.ndim == 3:
            y_a = y_a.squeeze(-1)
        if y_b.ndim == 3:
            y_b = y_b.squeeze(-1)
        mask_y = batch.mask[:, :steps].float()
        post_mask = torch.zeros_like(mask_y)
        if t_intervention + 1 < steps:
            post_mask[:, t_intervention + 1 :] = 1.0
        delta_y = (y_a - y_b).abs() * mask_y * post_mask
        score_y = delta_y.sum(dim=1) / (mask_y * post_mask).sum(dim=1).clamp(min=1.0)

    if select in {"max_delta_k", "max_delta_combined"}:
        k_key = "k_c" if "k_c" in ro_a and "k_c" in ro_b else "k"
        k_a = ro_a[k_key]
        k_b = ro_b[k_key]
        delta_k = (k_a - k_b).norm(dim=-1)
        mask_k = torch.cat([torch.ones(bsz, 1, device=device), batch.mask[:, :steps].float()], dim=1)
        post_mask_k = torch.zeros_like(mask_k)
        if t_intervention + 1 < mask_k.shape[1]:
            post_mask_k[:, t_intervention + 1 :] = 1.0
        delta_k = delta_k * mask_k * post_mask_k
        score_k = delta_k.sum(dim=1) / (mask_k * post_mask_k).sum(dim=1).clamp(min=1.0)

    if select == "max_delta_k":
        score = score_k if score_k is not None else score_y
    elif select == "max_delta_y":
        score = score_y if score_y is not None else score_k
    else:
        # max_delta / max_delta_combined
        if score_y is None:
            score = score_k
        elif score_k is None:
            score = score_y
        else:
            score = score_y + score_k

    if score is None:
        return 0
    return int(torch.argmax(score).item())


def select_student_idx_for_model(
    model: torch.nn.Module,
    batch: TrajectoryBatch,
    *,
    t_intervention: int,
    action_a: int,
    action_b: int,
    plot_horizon: int,
    seed: int,
    select: str,
    causal_only: bool,
) -> int:
    return _select_student_idx(
        model,
        batch,
        t_intervention=t_intervention,
        action_a=action_a,
        action_b=action_b,
        plot_horizon=plot_horizon,
        seed=seed,
        select=select,
        causal_only=causal_only,
    )


def _rollout_do_pair(
    model: torch.nn.Module,
    *,
    X: torch.Tensor,
    A: torch.Tensor,
    T_obs: torch.Tensor,
    M: torch.Tensor,
    t_intervention: int,
    action_a: int,
    action_b: int,
    steps: int,
    seed: int,
    causal_only: bool,
) -> tuple[dict, dict]:
    device = X.device
    eps = None
    if hasattr(model, "cfg") and hasattr(model.cfg, "d_eps"):
        rng = np.random.default_rng(int(seed))
        eps = torch.as_tensor(rng.standard_normal(size=(X.shape[0], steps, model.cfg.d_eps)), device=device).float()
    do_a = {t_intervention: torch.tensor([int(action_a)], device=device, dtype=T_obs.dtype)}
    do_b = {t_intervention: torch.tensor([int(action_b)], device=device, dtype=T_obs.dtype)}
    ro_a = model.rollout(
        x=X,
        a=A,
        t_obs=T_obs,
        do_t=do_a,
        mask=M,
        eps=eps,
        steps=steps,
        stochastic_y=False,
        causal_only=causal_only,
    )
    ro_b = model.rollout(
        x=X,
        a=A,
        t_obs=T_obs,
        do_t=do_b,
        mask=M,
        eps=eps,
        steps=steps,
        stochastic_y=False,
        causal_only=causal_only,
    )
    return ro_a, ro_b


def _rollout_do_pair_wrapper(
    model: object,
    *,
    batch: TrajectoryBatch,
    t_intervention: int,
    action_a: int,
    action_b: int,
) -> tuple[dict, dict]:
    do_a = {int(t_intervention): int(action_a)}
    do_b = {int(t_intervention): int(action_b)}
    ro_a = model.rollout(batch, do_t=do_a, t0=0, horizon=None, teacher_forcing=False, return_logits=False)
    ro_b = model.rollout(batch, do_t=do_b, t0=0, horizon=None, teacher_forcing=False, return_logits=False)
    return ro_a, ro_b


def plot_bifurcation_multi(
    *,
    models: list[tuple[str, object, str]],
    batch: TrajectoryBatch,
    student_idx: int,
    t_intervention: int,
    action_a: int,
    action_b: int,
    plot_horizon: int,
    seed: int,
    out_path: str | None,
    out_data_path: str | None,
) -> None:
    import matplotlib.pyplot as plt

    if not models:
        raise ValueError("No models provided for multi-model bifurcation plot.")
    i = int(student_idx)
    if i < 0 or i >= batch.X.shape[0]:
        i = max(0, min(i, batch.X.shape[0] - 1))

    fig, (ax_y, ax_k) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    data: dict[str, np.ndarray] = {}
    errors: list[str] = []
    plotted = 0
    any_k = False

    for idx, (label, model, mode) in enumerate(models):
        try:
            if hasattr(model, "model"):
                model.model.eval()
                device = next(model.model.parameters()).device
            else:
                model.eval()
                device = next(model.parameters()).device
            X = batch.X[i : i + 1].to(device)
            A = batch.A[i : i + 1].to(device)
            T_obs = batch.T[i : i + 1].to(device)
            M = batch.mask[i : i + 1].to(device)
            seq_len = int(T_obs.shape[1])
            steps = min(seq_len, int(plot_horizon) + 1)
            t_i = max(0, min(int(t_intervention), steps - 1))

            if mode == "scm_raw":
                causal_only = True
                ro_a, ro_b = _rollout_do_pair(
                    model,
                    X=X,
                    A=A,
                    T_obs=T_obs,
                    M=M,
                    t_intervention=t_i,
                    action_a=action_a,
                    action_b=action_b,
                    steps=steps,
                    seed=int(seed),
                    causal_only=causal_only,
                )
                y_a = ro_a["y"].squeeze().detach().cpu().numpy()
                y_b = ro_b["y"].squeeze().detach().cpu().numpy()
            else:
                sub_lengths = compute_lengths(M)
                sub_batch = TrajectoryBatch(
                    X=X,
                    A=A,
                    T=T_obs,
                    Y=batch.Y[i : i + 1].to(device),
                    mask=M,
                    lengths=sub_lengths,
                )
                ro_a, ro_b = _rollout_do_pair_wrapper(
                    model,
                    batch=sub_batch,
                    t_intervention=t_i,
                    action_a=action_a,
                    action_b=action_b,
                )
                y_a = ro_a["Y_prob"].squeeze().detach().cpu().numpy()
                y_b = ro_b["Y_prob"].squeeze().detach().cpu().numpy()

            color = f"C{idx % 10}"
            times = np.arange(len(y_a))
            ax_y.plot(times, y_a, color=color, linewidth=2, label=f"{label}: do={action_a}")
            ax_y.plot(times, y_b, color=color, linestyle="--", linewidth=2, label=f"{label}: do={action_b}")
            plotted += 1

            data[f"y_a_{label}"] = y_a
            data[f"y_b_{label}"] = y_b

            k_label = None
            if mode == "scm_raw":
                if "k_c" in ro_a and "k_c" in ro_b:
                    k_a = ro_a["k_c"].squeeze(0).detach().cpu().numpy()
                    k_b = ro_b["k_c"].squeeze(0).detach().cpu().numpy()
                    k_label = "Kc"
                elif "k" in ro_a and "k" in ro_b:
                    k_a = ro_a["k"].squeeze(0).detach().cpu().numpy()
                    k_b = ro_b["k"].squeeze(0).detach().cpu().numpy()
                    k_label = "K"
            else:
                if "K" in ro_a and "K" in ro_b:
                    k_a = ro_a["K"].squeeze(0).detach().cpu().numpy()
                    k_b = ro_b["K"].squeeze(0).detach().cpu().numpy()
                    k_label = "K"
            if k_label is not None:
                z_a = np.linalg.norm(k_a, axis=-1)
                z_b = np.linalg.norm(k_b, axis=-1)
                times_k = np.arange(z_a.shape[0])
                ax_k.plot(times_k, z_a, color=color, linewidth=2, label=f"{label}: do={action_a}")
                ax_k.plot(times_k, z_b, color=color, linestyle="--", linewidth=2, label=f"{label}: do={action_b}")
                any_k = True
                data[f"k_a_{label}"] = k_a
                data[f"k_b_{label}"] = k_b
                data[f"k_metric_{label}"] = z_a
                data[f"k_metric_{label}_b"] = z_b
        except Exception as e:
            errors.append(f"{label}: {e}")
            continue

    if plotted == 0:
        raise RuntimeError(f"All models failed for multi-plot: {errors}")

    ax_y.axvline(x=t_intervention, color="gray", linestyle=":")
    ax_y.set_ylabel("Predicted Y")
    ax_y.legend(ncol=2, fontsize=8)
    ax_y.grid(True, alpha=0.3)

    ax_k.axvline(x=t_intervention, color="gray", linestyle=":")
    ax_k.set_ylabel("Latent K (norm)")
    ax_k.set_xlabel("Time Step")
    if any_k:
        ax_k.legend(ncol=2, fontsize=8)
        ax_k.grid(True, alpha=0.3)
    else:
        ax_k.text(0.5, 0.5, "No latent K available", ha="center", va="center", transform=ax_k.transAxes)
        ax_k.grid(True, alpha=0.3)

    fig.suptitle("Counterfactual Bifurcation (Same Student, Multi-Model)")
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
    if out_data_path:
        out_npz = Path(out_data_path)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        data["student_idx"] = np.asarray(i, dtype=np.int64)
        data["t_intervention"] = np.asarray(t_intervention, dtype=np.int64)
        data["action_a"] = np.asarray(action_a, dtype=np.int64)
        data["action_b"] = np.asarray(action_b, dtype=np.int64)
        data["model_labels"] = np.asarray([m for m, _, _ in models])
        if errors:
            data["errors"] = np.asarray(errors)
        np.savez_compressed(out_npz, **data)
    plt.show()
    plt.close(fig)


def plot_counterfactual_bifurcation(
    model: torch.nn.Module,
    batch: TrajectoryBatch,
    t_intervention: int,
    action_a: int,
    action_b: int,
    plot_horizon: int,
    plot_metric: str,
    seed: int,
    out_path: str | None,
    student_idx: int | None = None,
    student_select: str = "max_delta",
    causal_only: bool = True,
    out_data_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    model.eval()
    device = next(model.parameters()).device
    batch = batch.to(device)
    if student_idx is None:
        i = _select_student_idx(
            model,
            batch,
            t_intervention=int(t_intervention),
            action_a=int(action_a),
            action_b=int(action_b),
            plot_horizon=int(plot_horizon),
            seed=int(seed),
            select=str(student_select),
            causal_only=bool(causal_only),
        )
    else:
        i = int(student_idx)
    if i < 0 or i >= batch.X.shape[0]:
        i = max(0, min(i, batch.X.shape[0] - 1))
    X = batch.X[i : i + 1]
    A = batch.A[i : i + 1]
    T_obs = batch.T[i : i + 1]
    M = batch.mask[i : i + 1]
    seq_len = int(T_obs.shape[1])
    steps = min(seq_len, int(plot_horizon) + 1)
    t_intervention = min(int(t_intervention), steps - 1)
    rng = np.random.default_rng(int(seed))
    eps = torch.as_tensor(rng.standard_normal(size=(1, steps, model.cfg.d_eps)), device=device).float()

    do_a = {t_intervention: torch.tensor([int(action_a)], device=device, dtype=T_obs.dtype)}
    ro_a = model.rollout(
        x=X,
        a=A,
        t_obs=T_obs,
        do_t=do_a,
        mask=M,
        eps=eps,
        steps=steps,
        stochastic_y=False,
        causal_only=causal_only,
    )
    do_b = {t_intervention: torch.tensor([int(action_b)], device=device, dtype=T_obs.dtype)}
    ro_b = model.rollout(
        x=X,
        a=A,
        t_obs=T_obs,
        do_t=do_b,
        mask=M,
        eps=eps,
        steps=steps,
        stochastic_y=False,
        causal_only=causal_only,
    )

    y_a = ro_a["y"].squeeze().detach().cpu().numpy()
    y_b = ro_b["y"].squeeze().detach().cpu().numpy()

    k_label = "K"
    if "k_c" in ro_a and "k_c" in ro_b:
        k_a = ro_a["k_c"].squeeze(0).detach().cpu().numpy()
        k_b = ro_b["k_c"].squeeze(0).detach().cpu().numpy()
        k_label = "Kc"
    elif "k" in ro_a and "k" in ro_b:
        k_a = ro_a["k"].squeeze(0).detach().cpu().numpy()
        k_b = ro_b["k"].squeeze(0).detach().cpu().numpy()
    else:
        raise RuntimeError("rollout() did not return latent K/K_c; cannot plot latent bifurcation.")

    k_metric = "k_norm" if plot_metric.startswith("y") else plot_metric
    pc_vec = None
    if k_metric == "k_pca":
        concat = np.concatenate([k_a, k_b], axis=0)
        concat = concat - concat.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(concat, full_matrices=False)
        pc_vec = vt[0]
    z_a = _project_k(k_a, k_metric, pc_vec)
    z_b = _project_k(k_b, k_metric, pc_vec)

    fig, (ax_y, ax_k) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    times_y = np.arange(len(y_a))
    hist_end = t_intervention + 1
    ax_y.plot(times_y[:hist_end], y_a[:hist_end], "k-", linewidth=2, label="Historical Path")
    ax_y.plot(times_y[t_intervention:], y_a[t_intervention:], "b--", linewidth=2, label=f"Do(T={action_a})")
    ax_y.plot(times_y[t_intervention:], y_b[t_intervention:], "r--", linewidth=2, label=f"Do(T={action_b})")
    ax_y.set_ylabel("Predicted Y")
    ax_y.axvline(x=t_intervention, color="gray", linestyle=":", label="Intervention Time")
    ax_y.legend()
    ax_y.grid(True, alpha=0.3)

    times_k = np.arange(z_a.shape[0])
    ax_k.plot(times_k[:hist_end], z_a[:hist_end], "k-", linewidth=2, label="Historical Path")
    ax_k.plot(times_k[t_intervention + 1 :], z_a[t_intervention + 1 :], "b--", linewidth=2, label=f"Do(T={action_a})")
    ax_k.plot(times_k[t_intervention + 1 :], z_b[t_intervention + 1 :], "r--", linewidth=2, label=f"Do(T={action_b})")
    ax_k.set_ylabel(f"Latent {k_label} ({k_metric})")
    ax_k.axvline(x=t_intervention, color="gray", linestyle=":")
    ax_k.legend()
    ax_k.grid(True, alpha=0.3)

    ax_k.set_xlabel("Time Step")
    fig.suptitle("Counterfactual Consistency: Bifurcation Plot")
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
    if out_data_path:
        out_npz = Path(out_data_path)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            y_a=y_a,
            y_b=y_b,
            k_a=k_a,
            k_b=k_b,
            z_a=z_a,
            z_b=z_b,
            student_idx=np.asarray(i, dtype=np.int64),
            t_intervention=np.asarray(t_intervention, dtype=np.int64),
            action_a=np.asarray(action_a, dtype=np.int64),
            action_b=np.asarray(action_b, dtype=np.int64),
            k_label=np.asarray(k_label),
            plot_metric=np.asarray(plot_metric),
        )
    plt.show()
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    ap.add_argument("--data", type=str, required=True, help="Dataset .npz path")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_eval", type=int, default=512)
    ap.add_argument("--t_list", type=str, default="0,5,10")
    ap.add_argument("--action_a", type=int, default=0)
    ap.add_argument("--action_b", type=int, default=1)
    ap.add_argument("--plot", action="store_true", default=True)
    ap.add_argument("--plot_t", type=int, default=10)
    ap.add_argument("--plot_horizon", type=int, default=20)
    ap.add_argument("--plot_metric", type=str, default="k_mean", choices=["k_norm", "k_mean", "k_pca", "y_prob"])
    ap.add_argument("--plot_out", type=str, default=None)
    ap.add_argument("--plot_data_out", type=str, default=None)
    ap.add_argument(
        "--student_select",
        type=str,
        default="max_delta",
        choices=["max_delta", "max_delta_y", "max_delta_k", "max_delta_combined", "median", "best", "worst", "random", "first"],
    )
    ap.add_argument("--causal_only", action="store_true", default=True)
    ap.add_argument("--out_csv", type=str, default=None)
    args = ap.parse_args()

    device = _device_from_arg(str(args.device))
    ckpt = torch.load(args.ckpt, map_location=device)
    model, model_name = load_rollout_model_from_checkpoint(ckpt, device=device)

    if not hasattr(model, "cfg") or not hasattr(model.cfg, "d_eps"):
        raise RuntimeError(f"Model {model_name} does not expose cfg.d_eps for fixed-noise rollouts.")

    ds = NPZSequenceDataset(args.data)
    rng = np.random.default_rng(int(args.seed))
    n_eval = min(int(args.n_eval), len(ds))
    idx = rng.choice(len(ds), size=n_eval, replace=False)
    batch = _batch_from_dataset(ds, idx)

    t_list = _parse_int_list(str(args.t_list))
    results = _compute_cf_history_mse(
        model=model,
        batch=batch,
        t_list=t_list,
        action_a=int(args.action_a),
        action_b=int(args.action_b),
        seed=int(args.seed),
    )
    if not results:
        raise RuntimeError("No valid t indices for counterfactual consistency evaluation.")

    mean_mse = float(np.mean([v for _, v in results]))
    print("Counterfactual consistency (MSE_history):")
    for t, v in results:
        print(f"  t={t}: {v:.6e}")
    print(f"  mean: {mean_mse:.6e}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("t,mse_history\n")
            for t, v in results:
                f.write(f"{t},{v:.10e}\n")
            f.write(f"mean,{mean_mse:.10e}\n")

    if args.plot:
        plot_counterfactual_bifurcation(
            model=model,
            batch=batch,
            t_intervention=int(args.plot_t),
            action_a=int(args.action_a),
            action_b=int(args.action_b),
            plot_horizon=int(args.plot_horizon),
            plot_metric=str(args.plot_metric),
            seed=int(args.seed),
            out_path=args.plot_out,
            student_select=str(args.student_select),
            causal_only=bool(args.causal_only),
            out_data_path=args.plot_data_out,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
