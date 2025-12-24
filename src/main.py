from __future__ import annotations

import sys
import csv
import json
import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TextIO

import numpy as np
import pandas as pd
import torch



from torch.utils.data import DataLoader, Dataset, Subset

# Allow running as a script (e.g., `python src/main.py`) as well as a module (`python -m src.main`).
if __package__ in (None, ""):
    _repo_root_for_imports = Path(__file__).resolve().parents[1]
    if str(_repo_root_for_imports) not in sys.path:
        sys.path.insert(0, str(_repo_root_for_imports))
    # Ensure imports like `import src.main` resolve to this running module (avoid a second import copy).
    sys.modules.setdefault("src.main", sys.modules[__name__])

from src.data import (
    NPZSequenceDataset,
    SequenceBatch,
    TrajectoryBatch,
    TrajectoryDataset,
    compute_lengths,
    make_trajectory_dataloader,
    move_batch,
    move_trajectory_batch,
)
from src.baselines import load_base_model_from_checkpoint
from src.causal_estimators import GFormula, IPTWMSM
from src.eval.causal_bias import compute_causal_bias, compute_policy_values
from src.eval.tstr_trts import run_tstr_trts
from src.privacy.mia_shadow import mia_to_dataframe, run_synth_membership_inference
from src.privacy.nn_distance import compute_nn_distance
from src.utils.manifest import add_artifact, write_manifest
from src.vis import run_visualization
from src.model.baselines import (
    CRN,
    CRNConfig,
    RCGANConfig,
    RCGANGenerator,
    SeqDiffusion,
    SeqDiffusionConfig,
    
    SeqVAE,
    SeqVAEConfig,
    TimeGAN,
    TimeGANConfig,
    load_rollout_model_from_checkpoint,
)
from src.model.discriminators import SequenceDiscriminator, SequenceDiscriminatorConfig
from src.model.losses import bernoulli_nll_from_logits, grad_reverse, mmd_rbf, treatment_ce_loss, wgan_g_loss, wgan_gp_d_loss
from src.model.policy import DoIntervention, RandomPolicy, TreatmentClassifier
from src.model.scm_generator import SCMGenerator, SCMGeneratorConfig


# ========== User Config ==========
# Dataset path mapping
DATASET_PATHS = {
    "assist09": "DataSet/assist2009/assist09_processed.npz",
    "oulad": "DataSet/OULAD/oulad_processed.npz",
    "statics": "DataSet/Statics2011/statics2011_step_level.npz",
}

# Choose which datasets / models to run
ACTIVE_DATASETS = ["oulad"]  # options: ["assist09", "oulad", "statics"]
ACTIVE_MODELS = ["scm", "rcgan", "vae", "diffusion", "crn", "timegan"]  # options: ["scm", "rcgan", "vae", "diffusion", "crn", "timegan"]

# Common training knobs (defaults)
COMMON_KNOBS = dict(


    batch_size=256,
    seq_len=50,  # will truncate if npz is longer
    device="auto",
    seed=42,
    
    test_ratio=0.2,  # split held-out set for predictive fidelity
    keep_checkpoints=False,  # if False, delete *.pt after metrics/vis
)

# Model-specific knobs (override defaults)
MODEL_KNOBS = {


    "scm": dict(
        epochs_a=30,
        epochs_b=30,
        epochs_c=15,
        dynamics="transformer",
        d_k=64,
        w_do=1.0,

        w_advT=0.05,
        w_cf=0.01,
        cf_interval=10,
        grl_lambda=1.0,
        lr_advT=1e-4,
        do_time_sampling="random",
        do_num_time_samples=2,
        do_min_arm_samples=16,
        do_actions="0,1",
        cf_num_time_samples=1,
    ),
    "rcgan": dict(
        epochs_a=30,
        epochs_b=30,
        epochs_c=10,
        rnn="gru",
        rcgan_hidden=128,
    ),
    "crn": dict(
        epochs=30,
        crn_hidden=128,
        crn_grl_lambda=1.0,
    ),
    "diffusion": dict(
        epochs=30,
        diff_steps=50,
        diff_backbone="transformer",
    ),
    "vae": dict(
        epochs=30,
        vae_z_dim=32,
    ),
    "timegan": dict(
        epochs_embed=3,
        epochs_supervisor=3,
        epochs_joint=5,
        timegan_hidden=64,
        timegan_num_layers=2,
        timegan_z_dim=16,
        x_emb_dim=32,
        lambda_sup=10.0,
        lambda_mom=1.0,
        lr_embed=1e-3,
        lr_gen=1e-3,
        lr_disc=1e-3,
        max_batches_per_epoch=200,
        reservoir_size=2048,
        gen_samples=512,
    ),
}
# =================================

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p.resolve()


class _TruncatedSequenceDataset(Dataset):
    def __init__(self, base: NPZSequenceDataset, *, seq_len: int):
        self.base = base
        self.n = len(base)
        self.seq_len = min(int(seq_len), int(base.seq_len))

        self.d_x = int(base.d_x)
        self.a_is_discrete = bool(base.a_is_discrete)
        self.t_is_discrete = bool(base.t_is_discrete)
        self.y_is_discrete = bool(getattr(base, "y_is_discrete", False))
        self.d_a = int(base.d_a)
        self.d_t = int(base.d_t)
        self.d_y = int(base.d_y)
        self.a_vocab_size = None if base.a_vocab_size is None else int(base.a_vocab_size)
        self.t_vocab_size = None if base.t_vocab_size is None else int(base.t_vocab_size)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.base[int(idx)]
        ex["A"] = ex["A"][: self.seq_len]
        ex["T"] = ex["T"][: self.seq_len]
        ex["Y"] = ex["Y"][: self.seq_len]
        ex["mask"] = ex["mask"][: self.seq_len]
        return ex


def _dataset_meta(ds: Any) -> dict[str, Any]:
    return {
        "d_x": int(ds.d_x),
        "seq_len": int(ds.seq_len),
        "a_is_discrete": bool(ds.a_is_discrete),
        "t_is_discrete": bool(ds.t_is_discrete),
        "a_vocab_size": ds.a_vocab_size,
        "t_vocab_size": ds.t_vocab_size,
        "d_a": int(ds.d_a),
        "d_t": int(ds.d_t),
        "d_y": int(ds.d_y),
    }


def _make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    def collate(examples: list[dict[str, torch.Tensor]]) -> SequenceBatch:
        x = torch.stack([e["X"] for e in examples], dim=0)
        a = torch.stack([e["A"] for e in examples], dim=0)
        t = torch.stack([e["T"] for e in examples], dim=0)
        y = torch.stack([e["Y"] for e in examples], dim=0)
        m = torch.stack([e["mask"] for e in examples], dim=0)
        return SequenceBatch(X=x, A=a, T=t, Y=y, mask=m)

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        collate_fn=collate,
        drop_last=bool(drop_last),
    )


def _split_indices(n: int, *, test_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    n = int(n)
    if n <= 0:
        return [], []
    if not (0.0 <= float(test_ratio) < 1.0):
        raise ValueError(f"test_ratio must be in [0,1), got {test_ratio}")

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    n_test = int(n * float(test_ratio))
    if test_ratio > 0.0:
        n_test = max(1, min(n - 1, n_test))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return train_idx, test_idx


def _batch_from_dataset(ds: NPZSequenceDataset) -> TrajectoryBatch:
    X = torch.as_tensor(ds.X).float()
    A = torch.as_tensor(ds.A)
    T = torch.as_tensor(ds.T).long()
    Y = torch.as_tensor(ds.Y).float()
    M = torch.as_tensor(ds.M).float()
    return TrajectoryBatch(X=X, A=A, T=T, Y=Y, mask=M, lengths=compute_lengths(M))


def _perturb_batch_for_privacy(
    batch: TrajectoryBatch,
    ref_batch: TrajectoryBatch,
    meta: dict[str, Any],
    *,
    x_noise_scale: float = 1.0,
    a_flip_prob: float = 0.4,
    t_flip_prob: float = 0.4,
) -> TrajectoryBatch:
    """Perturb X/A/T to create synthetic conditions that are distinct from originals."""
    # Perturb X with Gaussian noise
    X = batch.X.float()
    x_std = ref_batch.X.float().std(dim=0, unbiased=False)
    x_std = torch.where(x_std > 0, x_std, torch.ones_like(x_std))
    X = X + torch.randn_like(X) * x_std * float(x_noise_scale)

    # Perturb A
    mask = batch.mask
    A = batch.A
    if A.ndim == 2:  # discrete
        a_vocab = int(meta.get("a_vocab_size", 0) or int(ref_batch.A.max().item()) + 1)
        a_vocab = max(1, a_vocab)
        replace_mask = (torch.rand(A.shape, device=A.device) < a_flip_prob) & (mask > 0.5)
        if replace_mask.any():
            A = A.clone()
            new_vals = torch.randint(0, a_vocab, size=A.shape, device=A.device, dtype=A.dtype)
            A[replace_mask] = new_vals[replace_mask]
    else:  # continuous
        a_std = ref_batch.A.float().std(dim=(0, 1), unbiased=False)
        a_std = torch.where(a_std > 0, a_std, torch.ones_like(a_std))
        noise = torch.randn_like(A.float()) * a_std * float(x_noise_scale)
        A = A.float() + noise * mask.unsqueeze(-1)

    # Perturb T
    T = batch.T
    if bool(meta.get("t_is_discrete", True)):
        t_vocab = int(meta.get("t_vocab_size", 0) or int(ref_batch.T.max().item()) + 1)
        t_vocab = max(1, t_vocab)
        replace_mask = (torch.rand(T.shape, device=T.device) < t_flip_prob) & (mask > 0.5)
        if replace_mask.any():
            T = T.clone()
            new_vals = torch.randint(0, t_vocab, size=T.shape, device=T.device, dtype=T.dtype)
            T[replace_mask] = new_vals[replace_mask]
    else:
        t_std = ref_batch.T.float().std(dim=0, unbiased=False)
        t_scale = float(torch.mean(t_std).item()) if t_std.numel() else 1.0
        T = T.float() + torch.randn_like(T.float()) * t_scale * float(x_noise_scale) * mask

    lengths = compute_lengths(mask)
    return TrajectoryBatch(X=X, A=A, T=T, Y=batch.Y, mask=mask, lengths=lengths)


def _subset_batch(batch: TrajectoryBatch, indices: list[int]) -> TrajectoryBatch:
    if not indices:
        return TrajectoryBatch(
            X=batch.X[:0],
            A=batch.A[:0],
            T=batch.T[:0],
            Y=batch.Y[:0],
            mask=batch.mask[:0],
            lengths=batch.lengths[:0],
        )
    idx = torch.as_tensor(indices, dtype=torch.long)
    return TrajectoryBatch(
        X=batch.X[idx],
        A=batch.A[idx],
        T=batch.T[idx],
        Y=batch.Y[idx],
        mask=batch.mask[idx],
        lengths=batch.lengths[idx],
    )


def _parse_int_list(text: str, *, default: list[int]) -> list[int]:
    if text is None:
        return list(default)
    items = [s.strip() for s in str(text).split(",") if s.strip()]
    return [int(x) for x in items] if items else list(default)


def _parse_str_list(text: str, *, default: list[str]) -> list[str]:
    if text is None:
        return list(default)
    items = [s.strip() for s in str(text).split(",") if s.strip()]
    return items if items else list(default)


def _build_subgroups(batch: TrajectoryBatch, names: list[str]) -> list[dict]:
    subgroups = []
    x0 = None
    if batch.X.shape[1] > 0:
        x0 = batch.X[:, 0].detach().cpu().numpy()
    else:
        y_mean = (batch.Y * batch.mask).sum(dim=1) / batch.mask.sum(dim=1).clamp(min=1.0)
        x0 = y_mean.detach().cpu().numpy()

    median = float(np.median(x0))
    for name in names:
        if name == "all":
            subgroups.append({"name": "all"})
        elif name == "low":
            subgroups.append({"name": "low", "mask": x0 <= median})
        elif name == "high":
            subgroups.append({"name": "high", "mask": x0 > median})
    return subgroups


def _write_vis_summary_models(report_f: TextIO, *, vis_out_dir: Path) -> None:
    """Append the visualizer's per-model summary metrics to the main report."""

    summary_models_csv = Path(vis_out_dir) / "summary_models.csv"
    if not summary_models_csv.exists():
        report_f.write("  vis_summary_models_csv: (missing)\n")
        return

    def _format_path(value: str) -> str:
        if not value:
            return ""
        try:
            p = Path(value)
            if not p.is_absolute():
                p = (Path(vis_out_dir) / p).resolve()
            else:
                p = p.resolve()
            try:
                return p.relative_to(_REPO_ROOT).as_posix()
            except Exception:
                return p.as_posix()
        except Exception:
            return str(value)

    report_f.write(f"  vis_summary_models_csv: {summary_models_csv.as_posix()}\n")
    report_f.write("  vis_summary_models:\n")

    try:
        with summary_models_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                report_f.write("    (empty)\n\n")
                return

            path_fields = {k for k in reader.fieldnames if k.endswith("_png") or k.endswith("_csv")}
            for row in reader:
                model = (row.get("model") or "").strip() or "UNKNOWN"
                report_f.write(f"    [model={model}]\n")
                for key in reader.fieldnames:
                    if key == "model":
                        continue
                    value = (row.get(key) or "").strip()
                    if key in path_fields:
                        value = _format_path(value)
                    report_f.write(f"      {key}: {value}\n")
                report_f.write("\n")
    except Exception as e:
        report_f.write(f"  vis_summary_models_error: {e}\n\n")


def write_results_summary(
    metrics_tables: dict[str, pd.DataFrame],
    out_txt: str,
) -> None:
    out_path = Path(out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    id_candidates = ["model", "dataset", "ref_estimator", "predictor", "setting", "policy", "metric"]
    with out_path.open("w", encoding="utf-8") as f:
        for name, df in metrics_tables.items():
            f.write(f"{name}\n")
            if df is None or df.empty:
                f.write("(empty)\n\n")
                continue
            id_cols = [c for c in id_candidates if c in df.columns]
            num_cols = list(df.select_dtypes(include=[np.number]).columns)
            num_cols = [c for c in num_cols if c not in id_cols]
            if not id_cols and not num_cols:
                f.write("(empty)\n\n")
                continue
            out_df = df[id_cols + num_cols].copy()
            sort_cols = [c for c in ["model", "predictor", "setting", "policy"] if c in out_df.columns]
            if sort_cols:
                out_df = out_df.sort_values(sort_cols)
            f.write(out_df.to_string(index=False))
            f.write("\n\n")


def _safe_nanmean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr))


def _format_legacy_metric(value: object) -> str:
    try:
        num = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(num):
        return "nan"
    return f"{num:.6f}"


def _append_legacy_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def _extract_tstr_metrics(df: pd.DataFrame) -> dict[str, float]:
    out = {"tstr_auc": float("nan"), "trts_auc": float("nan"), "tstr_rmse": float("nan"), "trts_rmse": float("nan")}
    if df is None or df.empty:
        return out
    for setting, prefix in (("TSTR", "tstr"), ("TRTS", "trts")):
        rows = df[df["setting"] == setting]
        if rows.empty:
            continue
        out[f"{prefix}_auc"] = float(rows["auc"].iloc[0])
        out[f"{prefix}_rmse"] = float(rows["rmse"].iloc[0])
    return out


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    scores = y_score[order]
    labels = y_true[order]

    ranks = np.empty_like(scores, dtype=np.float64)
    i = 0
    rank = 1.0
    while i < scores.shape[0]:
        j = i + 1
        while j < scores.shape[0] and scores[j] == scores[i]:
            j += 1
        avg_rank = 0.5 * (rank + (rank + (j - i) - 1.0))
        ranks[i:j] = avg_rank
        rank += float(j - i)
        i = j

    sum_ranks_pos = float(ranks[labels == 1].sum())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def _predictive_fidelity(
    model: torch.nn.Module,
    model_name: str,
    dl: DataLoader,
    *,
    device: torch.device,
    drop_first_step: bool = True,
) -> dict[str, Any]:
    if model_name == "diffusion":
        return {"auc": float("nan"), "rmse": float("nan"), "note": "predictive fidelity skipped for diffusion"}

    y_scores: list[torch.Tensor] = []
    y_true: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dl:
            batch = move_batch(batch, device)
            X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask

            if model_name == "scm":
                out = model.teacher_forcing(  # type: ignore[attr-defined]
                    x=X, a=A, t=T, y=Y, mask=M, eps=None, eps_mode="zero"
                )
                y_prob = torch.sigmoid(out["y_logits"])
            elif model_name == "rcgan":
                d_eps = int(getattr(getattr(model, "cfg", None), "d_eps", 16))
                eps = torch.zeros(A.shape[0], A.shape[1], d_eps, device=device)
                out = model.teacher_forcing(  # type: ignore[attr-defined]
                    x=X, a=A, t=T, y=Y, mask=M, eps=eps, stochastic_y=False
                )
                y_prob = torch.sigmoid(out["y_logits"])
            elif model_name == "crn":
                out = model.forward(x=X, a=A, t=T, y=Y, mask=M)  # type: ignore[call-arg]
                y_prob = torch.sigmoid(out["y_logits"])
            elif model_name == "vae":
                mu, _logvar = model.encode(x=X, a=A, t=T, y=Y, mask=M)  # type: ignore[attr-defined]
                out = model.decode(  # type: ignore[attr-defined]
                    x=X, a=A, t=T, mask=M, z=mu, y=Y, teacher_forcing=True, stochastic_y=False
                )
                y_prob = torch.sigmoid(out["y_logits"])
            elif model_name == "timegan":
                z_dim = int(getattr(getattr(model, "cfg", None), "z_dim", 16))
                z = torch.zeros(A.shape[0], A.shape[1], z_dim, device=device)
                out = model.teacher_forcing(  # type: ignore[attr-defined]
                    x=X, a=A, t=T, y=None, mask=M, z=z, stochastic_y=False
                )
                y_prob = torch.sigmoid(out["y_logits"])
            else:
                return {"auc": float("nan"), "rmse": float("nan"), "note": f"unsupported model={model_name}"}

            if y_prob.ndim == 3:
                y_prob = y_prob.squeeze(-1)
            if Y.ndim == 3:
                Y = Y.squeeze(-1)

            if drop_first_step and y_prob.shape[1] > 1:
                y_prob = y_prob[:, 1:]
                Y = Y[:, 1:]
                M = M[:, 1:]

            valid = M > 0.5
            if valid.any():
                y_scores.append(y_prob[valid].detach().cpu())
                y_true.append(Y[valid].detach().cpu())

    if not y_scores:
        return {"auc": float("nan"), "rmse": float("nan"), "note": "empty test set"}

    scores_np = torch.cat(y_scores).numpy().astype(np.float64)
    true_np = torch.cat(y_true).numpy().astype(np.float64)

    rmse = float(np.sqrt(np.mean((scores_np - true_np) ** 2)))

    is_binary = np.all((true_np >= -1e-6) & (true_np <= 1.0 + 1e-6)) and np.all(
        np.isclose(true_np, 0.0, atol=1e-6) | np.isclose(true_np, 1.0, atol=1e-6)
    )
    auc = float("nan")
    if is_binary:
        auc = _roc_auc_score(true_np.astype(np.int64), scores_np)

    return {"auc": auc, "rmse": rmse, "note": ""}


def _interventional_fidelity_mmd(
    gen: torch.nn.Module,
    model_name: str,
    dl: DataLoader,
    *,
    device: torch.device,
    seq_len: int,
    actions: list[int],
    time_indices: list[int],
) -> dict[str, Any]:
    if not hasattr(gen, "rollout"):
        return {"mmd": float("nan"), "note": f"model={model_name} has no rollout()"}

    t_is_discrete = bool(getattr(getattr(gen, "cfg", None), "t_is_discrete", True))
    if not t_is_discrete:
        return {"mmd": float("nan"), "note": "treatment is continuous; MMD eval expects discrete T"}

    time_indices = [int(t) for t in time_indices if 0 <= int(t) < int(seq_len)]
    actions = [int(a) for a in actions if int(a) >= 0]
    if not time_indices or not actions:
        return {"mmd": float("nan"), "note": "no valid time_indices/actions"}

    mmd_terms: list[float] = []
    with torch.no_grad():
        for t_idx in time_indices:
            for a_val in actions:
                mmd_sum = 0.0
                mmd_n = 0
                for batch in dl:
                    batch = move_batch(batch, device)
                    X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask
                    if not (0 <= t_idx < A.shape[1]):
                        continue

                    valid = M[:, t_idx] > 0.5
                    sel = (T[:, t_idx].long() == int(a_val)) & valid
                    if not sel.any():
                        continue

                    y_real = Y[sel, t_idx].float().view(-1, 1)
                    feat_real = torch.cat([X[sel].float(), y_real], dim=-1)

                    do = DoIntervention.single_step(t_idx, int(a_val)).as_dict(batch_size=X.shape[0], device=device)
                    ro = gen.rollout(  # type: ignore[attr-defined]
                        x=X, a=A, t_obs=T, do_t=do, mask=M, stochastic_y=False
                    )
                    y_do = ro["y"][:, t_idx].float()
                    if y_do.ndim == 1:
                        y_do = y_do.view(-1, 1)
                    feat_gen = torch.cat([X[valid].float(), y_do[valid].view(-1, 1)], dim=-1)

                    mmd_sum += float(mmd_rbf(feat_real, feat_gen).item())
                    mmd_n += 1

                if mmd_n > 0:
                    mmd_terms.append(mmd_sum / float(mmd_n))

    if not mmd_terms:
        return {"mmd": float("nan"), "note": "no valid (t,a) terms (insufficient samples)"}

    return {"mmd": float(np.mean(mmd_terms)), "note": ""}


def _policy_value_random(
    gen: torch.nn.Module,
    model_name: str,
    dl: DataLoader,
    *,
    device: torch.device,
    num_actions: int,
) -> dict[str, Any]:
    if not hasattr(gen, "rollout"):
        return {"value": float("nan"), "note": f"model={model_name} has no rollout()"}

    t_is_discrete = bool(getattr(getattr(gen, "cfg", None), "t_is_discrete", True))
    if not t_is_discrete:
        return {"value": float("nan"), "note": "treatment is continuous; policy eval expects discrete T"}

    policy = RandomPolicy(num_actions=int(num_actions))
    sum_y = 0.0
    sum_m = 0.0
    with torch.no_grad():
        for batch in dl:
            batch = move_batch(batch, device)
            X, A, M = batch.X, batch.A, batch.mask
            ro = gen.rollout(  # type: ignore[attr-defined]
                x=X, a=A, t_obs=None, policy=policy, mask=M, stochastic_y=False
            )
            y = ro["y"]
            if y.ndim == 3:
                y = y.squeeze(-1)
            sum_y += float((y * M).sum().item())
            sum_m += float(M.sum().item())

    return {"value": (sum_y / max(1.0, sum_m)), "note": ""}


def _train_scm_or_rcgan(
    model_name: str,
    *,
    train_dl: DataLoader,
    meta: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    knobs: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs_a = int(knobs.get("epochs_a", 2))
    epochs_b = int(knobs.get("epochs_b", 2))
    epochs_c = int(knobs.get("epochs_c", 2))
    lr_g = float(knobs.get("lr_g", 1e-3))
    lr_d = float(knobs.get("lr_d", 1e-4))
    gp_weight = float(knobs.get("gp_weight", 10.0))
    w_y = float(knobs.get("w_y", 1.0))
    w_adv = float(knobs.get("w_adv", 0.5))
    w_sup = float(knobs.get("w_sup", 1.0))

    # Causal regularizer weights (SCM Stage C)
    w_do = float(knobs.get("w_do", 0.0))
    w_advT = float(knobs.get("w_advT", 0.0))
    w_cf = float(knobs.get("w_cf", 0.0))
    grl_lambda = float(knobs.get("grl_lambda", 1.0))
    lr_advT = float(knobs.get("lr_advT", 1e-4))
    save_checkpoints = bool(knobs.get("save_checkpoints", True))
    ckpt_every = max(1, int(knobs.get("ckpt_every", 1)))
    disc_input = str(knobs.get("disc_input", "logits"))
    do_horizon = int(knobs.get("do_horizon", 0))
    ref_estimator = knobs.get("ref_estimator", None)

    # Causal knobs (ported from src/train.py)
    do_time_index = int(knobs.get("do_time_index", 5))
    do_time_sampling = str(knobs.get("do_time_sampling", "random"))
    do_num_time_samples = int(knobs.get("do_num_time_samples", 1))
    do_min_arm_samples = int(knobs.get("do_min_arm_samples", 8))
    do_actions_raw = knobs.get("do_actions", "0,1")
    cf_num_time_samples = int(knobs.get("cf_num_time_samples", 1))
    cf_interval = max(1, int(knobs.get("cf_interval", 10)))

    log_every = int(knobs.get("log_every", 50))
    log_every = max(1, log_every)

    if do_time_sampling not in {"fixed", "random", "all"}:
        raise ValueError(f"do_time_sampling must be one of fixed|random|all, got {do_time_sampling}")
    do_num_time_samples = max(1, int(do_num_time_samples))
    do_min_arm_samples = max(1, int(do_min_arm_samples))
    cf_num_time_samples = max(1, int(cf_num_time_samples))

    if isinstance(do_actions_raw, str):
        do_actions = [int(x.strip()) for x in do_actions_raw.split(",") if x.strip()]
    elif isinstance(do_actions_raw, (list, tuple)):
        do_actions = [int(x) for x in do_actions_raw]
    else:
        raise TypeError("do_actions must be a comma-separated string or a list/tuple of ints")
    if not do_actions:
        do_actions = [0, 1]

    a_emb_dim = int(knobs.get("a_emb_dim", 32))
    t_emb_dim = int(knobs.get("t_emb_dim", 16))
    dropout = float(knobs.get("dropout", 0.1))
    d_disc_h = int(knobs.get("d_disc_h", 128))
    disc_num_layers = int(knobs.get("disc_num_layers", 1))

    a_vocab_size = int(meta["a_vocab_size"] or 100)
    t_vocab_size = int(meta["t_vocab_size"] or 2)

    disc_cfg = SequenceDiscriminatorConfig(
        d_x=int(meta["d_x"]),
        a_is_discrete=bool(meta["a_is_discrete"]),
        a_vocab_size=a_vocab_size,
        a_emb_dim=a_emb_dim,
        d_a=int(meta["d_a"]),
        t_is_discrete=bool(meta["t_is_discrete"]),
        t_vocab_size=t_vocab_size,
        t_emb_dim=t_emb_dim,
        d_t=int(meta["d_t"]),
        d_y=1,
        d_h=d_disc_h,
        num_layers=disc_num_layers,
        dropout=dropout,
    )
    disc = SequenceDiscriminator(disc_cfg).to(device)

    d_eps = int(knobs.get("d_eps", 16))
    if model_name == "scm":
        gen_cfg = SCMGeneratorConfig(
            d_x=int(meta["d_x"]),
            d_k=int(knobs.get("d_k", 64)),
            d_eps=d_eps,
            a_is_discrete=bool(meta["a_is_discrete"]),
            a_vocab_size=a_vocab_size,
            a_emb_dim=a_emb_dim,
            d_a=int(meta["d_a"]),
            t_is_discrete=bool(meta["t_is_discrete"]),
            t_vocab_size=t_vocab_size,
            t_emb_dim=t_emb_dim,
            d_t=int(meta["d_t"]),
            y_dist="bernoulli",
            d_y=int(meta["d_y"]),
            use_y_in_dynamics=bool(knobs.get("y_in_dynamics", False)),
            dynamics=str(knobs.get("dynamics", "gru")),
            mlp_hidden=int(knobs.get("mlp_hidden", 128)),
            dropout=dropout,
            tf_n_layers=int(knobs.get("tf_n_layers", 2)),
            tf_n_heads=int(knobs.get("tf_n_heads", 4)),
            tf_ffn_hidden=int(knobs.get("tf_ffn_hidden", 256)),
            tf_max_seq_len=int(knobs.get("tf_max_seq_len", 512)),
        )
        gen: torch.nn.Module = SCMGenerator(gen_cfg).to(device)
    elif model_name == "rcgan":
        gen_cfg = RCGANConfig(
            d_x=int(meta["d_x"]),
            d_eps=d_eps,
            a_is_discrete=bool(meta["a_is_discrete"]),
            a_vocab_size=a_vocab_size,
            a_emb_dim=a_emb_dim,
            d_a=int(meta["d_a"]),
            t_is_discrete=bool(meta["t_is_discrete"]),
            t_vocab_size=t_vocab_size,
            t_emb_dim=t_emb_dim,
            d_t=int(meta["d_t"]),
            rnn=str(knobs.get("rnn", "gru")),
            d_h=int(knobs.get("rcgan_hidden", 128)),
            dropout=dropout,
            use_prev_y=bool(knobs.get("rcgan_use_prev_y", True)),
        )
        gen = RCGANGenerator(gen_cfg).to(device)
    else:
        raise ValueError(f"Unexpected model={model_name}")

    # Treatment classifier for adversarial deconfounding (Stage C; SCM only)
    t_clf: Optional[TreatmentClassifier] = None
    opt_tclf: Optional[torch.optim.Optimizer] = None
    if model_name == "scm" and w_advT > 0.0 and epochs_c > 0:
        t_clf = TreatmentClassifier(d_h=int(gen_cfg.d_k), num_actions=t_vocab_size).to(device)  # type: ignore[union-attr]
        opt_tclf = torch.optim.Adam(t_clf.parameters(), lr=lr_advT)

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d)

    last_ckpt_payload: Optional[dict[str, Any]] = None

    def save_ckpt(stage: str, epoch: int) -> Path:
        ckpt = {
            "model": model_name,
            "stage": stage,
            "epoch": int(epoch),
            "model_cfg": asdict(gen_cfg),
            "model_state": gen.state_dict(),
            "disc_cfg": asdict(disc_cfg),
            "disc_state": disc.state_dict(),
        }
        path = out_dir / f"ckpt_{model_name}_{stage}_ep{epoch}.pt"
        nonlocal last_ckpt_payload
        last_ckpt_payload = ckpt
        if save_checkpoints and (int(epoch) % ckpt_every == 0):
            torch.save(ckpt, path)
        return path

    last_ckpt = save_ckpt("A", 0)

    # Stage A: supervised pretrain
    if epochs_a > 0:
        print(f"    [Stage A] Supervised pretraining ({epochs_a} epochs)")
    gen.train()
    for ep in range(1, epochs_a + 1):
        total_loss = 0.0
        steps = 0
        for batch in train_dl:
            batch = move_batch(batch, device)
            X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask

            if model_name == "scm":
                bsz, seq_len = A.shape[0], A.shape[1]
                eps = torch.zeros(bsz, seq_len, d_eps, device=device)
                out = gen.teacher_forcing(  # type: ignore[attr-defined]
                    x=X, a=A, t=T, y=Y, mask=M, eps=eps, eps_mode="zero"
                )
                loss_y = bernoulli_nll_from_logits(out["y_logits"], Y, mask=M)

                k_tf = out["k"]
                a_enc = gen.encode_a(A.view(-1, *A.shape[2:])).view(bsz, seq_len, -1)  # type: ignore[attr-defined]
                t_enc = gen.encode_t(T.view(-1, *T.shape[2:])).view(bsz, seq_len, -1)  # type: ignore[attr-defined]
                x_rep = gen.x_proj(X.float())[:, None, :].expand(bsz, seq_len, X.shape[1])  # type: ignore[attr-defined]
                if gen_cfg.use_y_in_dynamics:
                    y_dyn = Y.float()
                    if y_dyn.ndim == 2:
                        y_dyn = y_dyn[..., None]
                    inp = torch.cat([a_enc, t_enc, x_rep, y_dyn, eps], dim=-1)
                else:
                    inp = torch.cat([a_enc, t_enc, x_rep, eps], dim=-1)
                k_pred = gen.supervisor(k_tf[:, :-1, :], inp)  # type: ignore[attr-defined]
                loss_sup = ((k_pred - k_tf[:, 1:, :]).pow(2).mean(dim=-1) * M).sum() / M.sum().clamp(min=1.0)

                loss = w_y * loss_y + w_sup * loss_sup
            else:
                out = gen.teacher_forcing(  # type: ignore[attr-defined]
                    x=X, a=A, t=T, y=Y, mask=M, stochastic_y=False
                )
                loss = bernoulli_nll_from_logits(out["y_logits"], Y, mask=M)

            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            opt_g.step()

            total_loss += float(loss.item())
            steps += 1
            if steps % log_every == 0:
                avg = total_loss / max(1, steps)
                print(f"      Stage A | Epoch {ep}/{epochs_a} | Step {steps} | AvgLoss {avg:.4f}")

        avg_loss = total_loss / max(1, steps)
        print(f"      Stage A | Epoch {ep}/{epochs_a} | AvgLoss {avg_loss:.4f}")
        last_ckpt = save_ckpt("A", ep)

    # Stage B/C: adversarial training (WGAN-GP) + supervised term (+ causal terms for SCM Stage C)
    for stage_name, epochs in [("B", epochs_b), ("C", epochs_c)]:
        if epochs <= 0:
            continue
        print(f"    [Stage {stage_name}] Adversarial training ({epochs} epochs)")
        gen.train()
        disc.train()
        if t_clf is not None:
            t_clf.train()

        enable_causal = stage_name == "C" and model_name == "scm" and (w_do > 0.0 or w_advT > 0.0 or w_cf > 0.0)
        for ep in range(1, epochs + 1):
            g_loss_accum = 0.0
            d_loss_accum = 0.0
            do_accum = 0.0
            advt_accum = 0.0
            cf_accum = 0.0
            steps = 0
            for batch in train_dl:
                batch = move_batch(batch, device)
                X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask
                bsz, seq_len = A.shape[0], A.shape[1]

                ro = gen.rollout(x=X, a=A, t_obs=T, mask=M, stochastic_y=False)  # type: ignore[attr-defined]
                y_fake = ro["y"]
                y_fake_in = y_fake
                y_real_in = Y
                if disc_input == "logits":
                    y_fake_in = ro.get("y_logits", None)
                    if y_fake_in is None:
                        y_fake_in = torch.log(torch.clamp(y_fake, 0.01, 0.99) / torch.clamp(1.0 - y_fake, 0.01, 0.99))
                    y_real_in = torch.log(torch.clamp(Y, 0.01, 0.99) / torch.clamp(1.0 - Y, 0.01, 0.99))
                elif disc_input == "sampled":
                    u = torch.rand_like(y_fake)
                    y_hard = (y_fake > u).float()
                    y_fake_in = y_hard + (y_fake - y_fake.detach())

                real_seq = disc.encode_inputs(x=X, a=A, t=T, y=y_real_in)
                fake_seq = disc.encode_inputs(x=X, a=A, t=T, y=y_fake_in.detach())

                loss_d = wgan_gp_d_loss(disc, real_seq, fake_seq, gp_weight=gp_weight, mask=M)
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
                d_loss_accum += float(loss_d.item())

                loss_do = torch.tensor(0.0, device=device)
                loss_advT = torch.tensor(0.0, device=device)
                loss_cf = torch.tensor(0.0, device=device)

                if model_name == "scm":
                    eps_seq = torch.randn(bsz, seq_len, d_eps, device=device)
                    tf = gen.teacher_forcing(  # type: ignore[attr-defined]
                        x=X, a=A, t=T, y=Y, mask=M, eps=eps_seq, eps_mode="random"
                    )
                    loss_y = bernoulli_nll_from_logits(tf["y_logits"], Y, mask=M)

                    k_tf = tf["k"]
                    a_enc = gen.encode_a(A.view(-1, *A.shape[2:])).view(bsz, seq_len, -1)  # type: ignore[attr-defined]
                    t_enc = gen.encode_t(T.view(-1, *T.shape[2:])).view(bsz, seq_len, -1)  # type: ignore[attr-defined]
                    x_feat = gen.x_proj(X.float())  # type: ignore[attr-defined]
                    x_rep = x_feat[:, None, :].expand(bsz, seq_len, X.shape[1])
                    if gen_cfg.use_y_in_dynamics:
                        y_dyn = Y.float()
                        if y_dyn.ndim == 2:
                            y_dyn = y_dyn[..., None]
                        inp = torch.cat([a_enc, t_enc, x_rep, y_dyn, eps_seq], dim=-1)
                    else:
                        inp = torch.cat([a_enc, t_enc, x_rep, eps_seq], dim=-1)
                    k_pred = gen.supervisor(k_tf[:, :-1, :], inp)  # type: ignore[attr-defined]
                    loss_sup = ((k_pred - k_tf[:, 1:, :]).pow(2).mean(dim=-1) * M).sum() / M.sum().clamp(min=1.0)

                    if enable_causal and bool(gen_cfg.t_is_discrete) and T.ndim == 2:
                        if w_do > 0.0 and do_horizon > 0 and ref_estimator is not None:
                            t0 = int(torch.randint(low=0, high=seq_len, size=(1,), device=device).item())
                            action = int(
                                do_actions[
                                    int(torch.randint(low=0, high=len(do_actions), size=(1,), device=device).item())
                                ]
                            )
                            horizon_eff = min(int(do_horizon), max(0, seq_len - t0 - 1))
                            steps = t0 + horizon_eff + 1
                            t_do = torch.full((bsz,), action, device=device, dtype=T.dtype)
                            ro_do = gen.rollout(  # type: ignore[attr-defined]
                                x=X,
                                a=A,
                                t_obs=T,
                                do_t={t0: t_do},
                                mask=M,
                                eps=eps_seq,
                                steps=steps,
                                stochastic_y=False,
                            )
                            y_do = ro_do["y"]
                            if y_do.ndim == 3 and y_do.shape[-1] == 1:
                                y_do = y_do.squeeze(-1)
                            y_slice = y_do[:, t0:steps]
                            m_slice = M[:, t0:steps]
                            gen_mu = (y_slice * m_slice).sum(dim=0) / m_slice.sum(dim=0).clamp(min=1.0)
                            batch_ref = TrajectoryBatch(
                                X=X.detach().cpu(),
                                A=A.detach().cpu(),
                                T=T.detach().cpu(),
                                Y=Y.detach().cpu(),
                                mask=M.detach().cpu(),
                                lengths=compute_lengths(M.detach().cpu()),
                            )
                            ref = ref_estimator.estimate_do(
                                batch_ref,
                                t0=t0,
                                horizon=int(horizon_eff),
                                action=int(action),
                                subgroup={"name": "all"},
                                n_boot=1,
                                seed=int(knobs.get("seed", 0)),
                            )
                            ref_mu = torch.as_tensor(ref["mu"], device=device).float()
                            loss_do = torch.mean((gen_mu[: ref_mu.shape[0]] - ref_mu) ** 2)
                            print(
                                f"      loss_do_multistep={float(loss_do.item()):.4f} "
                                f"do_horizon={int(horizon_eff)} t0_sampled={t0}"
                            )
                        elif w_do > 0.0:
                            # Fallback: single-step MMD alignment (legacy).
                            if do_time_sampling == "fixed":
                                do_time_indices = [int(do_time_index)]
                            elif do_time_sampling == "random":
                                do_time_indices = torch.randint(
                                    low=0, high=seq_len, size=(int(do_num_time_samples),), device=device
                                ).tolist()
                            else:
                                do_time_indices = list(range(seq_len))

                            do_terms = 0
                            for t_idx in do_time_indices:
                                if not (0 <= int(t_idx) < int(seq_len)):
                                    continue
                                valid = M[:, int(t_idx)] > 0.5
                                if valid.sum().item() < do_min_arm_samples:
                                    continue

                                k_t = k_tf[:, int(t_idx), :]  # pre-treatment state K_t
                                a_enc_t = gen.encode_a(A[:, int(t_idx)])  # type: ignore[attr-defined]

                                for a_val in do_actions:
                                    a_val = int(a_val)
                                    t_do = torch.full((bsz,), a_val, device=device, dtype=T.dtype)
                                    t_enc_do = gen.encode_t(t_do)  # type: ignore[attr-defined]
                                    y_inp_do = torch.cat([k_t, a_enc_t, t_enc_do, x_feat], dim=-1)
                                    y_prob_do = torch.sigmoid(gen.y_head(y_inp_do))  # type: ignore[attr-defined]
                                    feat_do = torch.cat([x_feat, k_t, y_prob_do], dim=-1)[valid]

                                    sel = (T[:, int(t_idx)].long() == a_val) & valid
                                    if sel.sum().item() < do_min_arm_samples:
                                        continue
                                    y_real = Y[sel, int(t_idx)].float().view(-1, 1)
                                    feat_real = torch.cat([x_feat[sel], k_t[sel], y_real], dim=-1)
                                    loss_do = loss_do + mmd_rbf(feat_real, feat_do)
                                    do_terms += 1

                            if do_terms > 0:
                                loss_do = loss_do / float(do_terms)

                        # (B) adversarial deconfounding (GRL): make K_t less predictive of T_t
                        if w_advT > 0.0 and t_clf is not None and opt_tclf is not None:
                            with torch.no_grad():
                                k_repr_detached = k_tf[:, :-1, :].detach()
                            t_logits_clf = t_clf(k_repr_detached)
                            loss_tclf = treatment_ce_loss(t_logits_clf, T.long(), mask=M)
                            opt_tclf.zero_grad(set_to_none=True)
                            loss_tclf.backward()
                            opt_tclf.step()

                            t_logits_adv = t_clf(grad_reverse(k_tf[:, :-1, :], lambd=grl_lambda))
                            loss_advT = treatment_ce_loss(t_logits_adv, T.long(), mask=M)

                        # (C) counterfactual consistency: K_{0:t} must not change when intervening at t
                        # This is expensive for autoregressive rollouts (especially with transformer dynamics),
                        # so we optionally compute it only every `cf_interval` steps.
                        do_cf_now = (steps % cf_interval == 0) if cf_interval > 1 else True
                        if w_cf > 0.0 and do_cf_now:
                            num_actions = max(1, int(t_vocab_size))
                            cf_time_indices = torch.randint(
                                low=0, high=seq_len, size=(int(cf_num_time_samples),), device=device
                            ).tolist()
                            ro_f = gen.rollout(  # type: ignore[attr-defined]
                                x=X, a=A, t_obs=T, mask=M, eps=eps_seq, stochastic_y=False
                            )
                            k_f = ro_f["k"]
                            cf_terms = 0
                            for t_idx in cf_time_indices:
                                if not (0 <= int(t_idx) < int(seq_len)):
                                    continue
                                alt = (T[:, int(t_idx)].long() + 1) % num_actions
                                ro_cf = gen.rollout(  # type: ignore[attr-defined]
                                    x=X, a=A, t_obs=T, do_t={int(t_idx): alt}, mask=M, eps=eps_seq, stochastic_y=False
                                )
                                k_cf = ro_cf["k"]
                                prefix_len = int(t_idx) + 1  # compare K_0..K_t
                                if prefix_len <= 0:
                                    continue
                                k1 = k_f[:, :prefix_len, :]
                                k2 = k_cf[:, :prefix_len, :]
                                mask_k = torch.cat([torch.ones(bsz, 1, device=device), M[:, : int(t_idx)]], dim=1)
                                diff = (k1 - k2).pow(2).mean(dim=-1)  # [B, prefix_len]
                                loss_cf = loss_cf + (diff * mask_k).sum() / mask_k.sum().clamp(min=1.0)
                                cf_terms += 1
                            if cf_terms > 0:
                                loss_cf = loss_cf / float(cf_terms)
                else:
                    tf = gen.teacher_forcing(  # type: ignore[attr-defined]
                        x=X, a=A, t=T, y=Y, mask=M, stochastic_y=False
                    )
                    loss_y = bernoulli_nll_from_logits(tf["y_logits"], Y, mask=M)
                    loss_sup = torch.tensor(0.0, device=device)

                fake_seq_g = disc.encode_inputs(x=X, a=A, t=T, y=y_fake_in)
                loss_adv = wgan_g_loss(disc, fake_seq_g, mask=M)

                loss_g = w_y * loss_y + w_sup * loss_sup + w_adv * loss_adv
                if enable_causal:
                    loss_g = loss_g + w_do * loss_do + w_advT * loss_advT + w_cf * loss_cf

                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_g.step()

                g_loss_accum += float(loss_g.item())
                if enable_causal:
                    do_accum += float(loss_do.item())
                    advt_accum += float(loss_advT.item())
                    cf_accum += float(loss_cf.item())
                steps += 1
                if steps % log_every == 0:
                    avg_d = d_loss_accum / max(1, steps)
                    avg_g = g_loss_accum / max(1, steps)
                    msg = f"      Stage {stage_name} | Epoch {ep}/{epochs} | Step {steps} | D {avg_d:.4f} | G {avg_g:.4f}"
                    if enable_causal:
                        msg += (
                            f" | do {do_accum / max(1, steps):.4f}"
                            f" | advT {advt_accum / max(1, steps):.4f}"
                            f" | cf {cf_accum / max(1, steps):.4f}"
                        )
                    print(msg)

            avg_d = d_loss_accum / max(1, steps)
            avg_g = g_loss_accum / max(1, steps)
            msg = f"      Stage {stage_name} | Epoch {ep}/{epochs} | D {avg_d:.4f} | G {avg_g:.4f}"
            if enable_causal:
                msg += (
                    f" | do {do_accum / max(1, steps):.4f}"
                    f" | advT {advt_accum / max(1, steps):.4f}"
                    f" | cf {cf_accum / max(1, steps):.4f}"
                )
            print(msg)
            last_ckpt = save_ckpt(stage_name, ep)

    if not save_checkpoints:
        if last_ckpt_payload is None:
            raise RuntimeError("No checkpoint payload captured; training loop may have not executed.")
        # Materialize a single checkpoint for downstream evaluation/visualization.
        torch.save(last_ckpt_payload, last_ckpt)

    return last_ckpt


def _train_vae(
    *,
    train_dl: DataLoader,
    meta: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    knobs: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(knobs.get("epochs", 10))
    lr = float(knobs.get("lr", 1e-3))
    kl_weight = float(knobs.get("vae_kl_weight", 0.1))
    dropout = float(knobs.get("dropout", 0.1))
    save_checkpoints = bool(knobs.get("save_checkpoints", True))
    a_emb_dim = int(knobs.get("a_emb_dim", 32))
    t_emb_dim = int(knobs.get("t_emb_dim", 16))

    cfg = SeqVAEConfig(
        d_x=int(meta["d_x"]),
        z_dim=int(knobs.get("vae_z_dim", 32)),
        enc_hidden=int(knobs.get("vae_enc_hidden", 128)),
        dec_hidden=int(knobs.get("vae_dec_hidden", 128)),
        dropout=dropout,
        use_prev_y=True,
        a_is_discrete=bool(meta["a_is_discrete"]),
        a_vocab_size=int(meta["a_vocab_size"] or 100),
        a_emb_dim=a_emb_dim,
        d_a=int(meta["d_a"]),
        t_is_discrete=bool(meta["t_is_discrete"]),
        t_vocab_size=int(meta["t_vocab_size"] or 2),
        t_emb_dim=t_emb_dim,
        d_t=int(meta["d_t"]),
    )
    model = SeqVAE(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log_every = max(1, int(knobs.get("log_every", 50)))

    last_ckpt_payload: Optional[dict[str, Any]] = None

    def save_ckpt(epoch: int) -> Path:
        ckpt = {"model": "vae", "epoch": int(epoch), "model_cfg": asdict(cfg), "model_state": model.state_dict()}
        path = out_dir / f"ckpt_vae_ep{epoch}.pt"
        nonlocal last_ckpt_payload
        last_ckpt_payload = ckpt
        if save_checkpoints:
            torch.save(ckpt, path)
        return path

    last_ckpt = save_ckpt(0)
    if epochs > 0:
        print(f"    [VAE] Training ({epochs} epochs)")
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_dl:
            batch = move_batch(batch, device)
            losses = model.elbo_loss(  # type: ignore[attr-defined]
                x=batch.X, a=batch.A, t=batch.T, y=batch.Y, mask=batch.mask, kl_weight=kl_weight
            )
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()
            total_loss += float(losses["loss"].item())
            steps += 1
            if steps % log_every == 0:
                print(f"      VAE | Epoch {ep}/{epochs} | Step {steps} | AvgLoss {total_loss / max(1, steps):.4f}")

        print(f"      VAE | Epoch {ep}/{epochs} | AvgLoss {total_loss / max(1, steps):.4f}")
        last_ckpt = save_ckpt(ep)

    if not save_checkpoints:
        if last_ckpt_payload is None:
            raise RuntimeError("No checkpoint payload captured; training loop may have not executed.")
        torch.save(last_ckpt_payload, last_ckpt)

    return last_ckpt


def _train_diffusion(
    *,
    train_dl: DataLoader,
    meta: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    knobs: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(knobs.get("epochs", 10))
    lr = float(knobs.get("lr", 1e-3))
    dropout = float(knobs.get("dropout", 0.1))
    save_checkpoints = bool(knobs.get("save_checkpoints", True))
    a_emb_dim = int(knobs.get("a_emb_dim", 32))
    t_emb_dim = int(knobs.get("t_emb_dim", 16))

    cfg = SeqDiffusionConfig(
        d_x=int(meta["d_x"]),
        num_steps=int(knobs.get("diff_steps", 50)),
        beta_start=float(knobs.get("diff_beta_start", 1e-4)),
        beta_end=float(knobs.get("diff_beta_end", 0.02)),
        time_emb_dim=int(knobs.get("diff_time_emb_dim", 64)),
        model_dim=int(knobs.get("diff_model_dim", 128)),
        backbone=str(knobs.get("diff_backbone", "mlp")),
        n_layers=int(knobs.get("diff_n_layers", 2)),
        n_heads=int(knobs.get("diff_n_heads", 4)),
        dropout=dropout,
        a_is_discrete=bool(meta["a_is_discrete"]),
        a_vocab_size=int(meta["a_vocab_size"] or 100),
        a_emb_dim=a_emb_dim,
        d_a=int(meta["d_a"]),
        t_is_discrete=bool(meta["t_is_discrete"]),
        t_vocab_size=int(meta["t_vocab_size"] or 2),
        t_emb_dim=t_emb_dim,
        d_t=int(meta["d_t"]),
        y_dim=1,
    )
    model = SeqDiffusion(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log_every = max(1, int(knobs.get("log_every", 50)))

    last_ckpt_payload: Optional[dict[str, Any]] = None

    def save_ckpt(epoch: int) -> Path:
        ckpt = {
            "model": "diffusion",
            "epoch": int(epoch),
            "model_cfg": asdict(cfg),
            "model_state": model.state_dict(),
        }
        path = out_dir / f"ckpt_diffusion_ep{epoch}.pt"
        nonlocal last_ckpt_payload
        last_ckpt_payload = ckpt
        if save_checkpoints:
            torch.save(ckpt, path)
        return path

    last_ckpt = save_ckpt(0)
    if epochs > 0:
        print(f"    [Diffusion] Training ({epochs} epochs)")
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_dl:
            batch = move_batch(batch, device)
            loss = model.denoising_loss(  # type: ignore[attr-defined]
                x=batch.X, a=batch.A, t=batch.T, y=batch.Y, mask=batch.mask
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            steps += 1
            if steps % log_every == 0:
                print(
                    f"      Diffusion | Epoch {ep}/{epochs} | Step {steps} | AvgLoss {total_loss / max(1, steps):.4f}"
                )

        print(f"      Diffusion | Epoch {ep}/{epochs} | AvgLoss {total_loss / max(1, steps):.4f}")
        last_ckpt = save_ckpt(ep)

    if not save_checkpoints:
        if last_ckpt_payload is None:
            raise RuntimeError("No checkpoint payload captured; training loop may have not executed.")
        torch.save(last_ckpt_payload, last_ckpt)

    return last_ckpt


def _train_crn(
    *,
    train_dl: DataLoader,
    meta: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    knobs: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(knobs.get("epochs", 10))
    lr = float(knobs.get("lr", 1e-3))
    dropout = float(knobs.get("dropout", 0.1))
    save_checkpoints = bool(knobs.get("save_checkpoints", True))
    a_emb_dim = int(knobs.get("a_emb_dim", 32))
    t_emb_dim = int(knobs.get("t_emb_dim", 16))
    w_treat = float(knobs.get("crn_w_treat", 1.0))

    cfg = CRNConfig(
        d_x=int(meta["d_x"]),
        d_h=int(knobs.get("crn_hidden", 128)),
        dropout=dropout,
        grl_lambda=float(knobs.get("crn_grl_lambda", 1.0)),
        a_is_discrete=bool(meta["a_is_discrete"]),
        a_vocab_size=int(meta["a_vocab_size"] or 100),
        a_emb_dim=a_emb_dim,
        d_a=int(meta["d_a"]),
        t_is_discrete=bool(meta["t_is_discrete"]),
        t_vocab_size=int(meta["t_vocab_size"] or 2),
        t_emb_dim=t_emb_dim,
        d_t=int(meta["d_t"]),
    )
    model = CRN(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log_every = max(1, int(knobs.get("log_every", 50)))

    last_ckpt_payload: Optional[dict[str, Any]] = None

    def save_ckpt(epoch: int) -> Path:
        ckpt = {"model": "crn", "epoch": int(epoch), "model_cfg": asdict(cfg), "model_state": model.state_dict()}
        path = out_dir / f"ckpt_crn_ep{epoch}.pt"
        nonlocal last_ckpt_payload
        last_ckpt_payload = ckpt
        if save_checkpoints:
            torch.save(ckpt, path)
        return path

    last_ckpt = save_ckpt(0)
    if epochs > 0:
        print(f"    [CRN] Training ({epochs} epochs)")
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_dl:
            batch = move_batch(batch, device)
            losses = model.loss(  # type: ignore[attr-defined]
                x=batch.X, a=batch.A, t=batch.T, y=batch.Y, mask=batch.mask, w_treat=w_treat
            )
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()
            total_loss += float(losses["loss"].item())
            steps += 1
            if steps % log_every == 0:
                print(f"      CRN | Epoch {ep}/{epochs} | Step {steps} | AvgLoss {total_loss / max(1, steps):.4f}")

        print(f"      CRN | Epoch {ep}/{epochs} | AvgLoss {total_loss / max(1, steps):.4f}")
        last_ckpt = save_ckpt(ep)

    if not save_checkpoints:
        if last_ckpt_payload is None:
            raise RuntimeError("No checkpoint payload captured; training loop may have not executed.")
        torch.save(last_ckpt_payload, last_ckpt)

    return last_ckpt


def _train_timegan(
    *,
    train_dl: DataLoader,
    meta: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    knobs: dict[str, Any],
) -> Path:
    from baselines.timegan import TimeGANTrainConfig, fit_timegan  # noqa: WPS433

    out_dir.mkdir(parents=True, exist_ok=True)

    epochs_embed = int(knobs.get("epochs_embed", 3))
    epochs_supervisor = int(knobs.get("epochs_supervisor", 3))
    epochs_joint = int(knobs.get("epochs_joint", 5))
    save_checkpoints = bool(knobs.get("save_checkpoints", True))

    a_emb_dim = int(knobs.get("a_emb_dim", 32))
    t_emb_dim = int(knobs.get("t_emb_dim", 16))
    x_emb_dim = int(knobs.get("x_emb_dim", 32))
    dropout = float(knobs.get("dropout", 0.1))

    a_vocab_size = int(meta["a_vocab_size"] or 100)
    t_vocab_size = int(meta["t_vocab_size"] or 2)

    cfg = TimeGANConfig(
        d_x=int(meta["d_x"]),
        seq_len=int(meta["seq_len"]),
        a_is_discrete=bool(meta["a_is_discrete"]),
        a_vocab_size=a_vocab_size,
        a_emb_dim=a_emb_dim,
        d_a=int(meta["d_a"]),
        t_is_discrete=bool(meta["t_is_discrete"]),
        t_vocab_size=t_vocab_size,
        t_emb_dim=t_emb_dim,
        d_t=int(meta["d_t"]),
        d_y=int(meta["d_y"]),
        x_emb_dim=x_emb_dim,
        hidden_dim=int(knobs.get("timegan_hidden", 64)),
        num_layers=int(knobs.get("timegan_num_layers", 2)),
        z_dim=int(knobs.get("timegan_z_dim", 16)),
        dropout=dropout,
    )
    model = TimeGAN(cfg).to(device)

    train_cfg = TimeGANTrainConfig(
        epochs_embed=epochs_embed,
        epochs_supervisor=epochs_supervisor,
        epochs_joint=epochs_joint,
        lr_embed=float(knobs.get("lr_embed", 1e-3)),
        lr_gen=float(knobs.get("lr_gen", 1e-3)),
        lr_disc=float(knobs.get("lr_disc", 1e-3)),
        lambda_sup=float(knobs.get("lambda_sup", 10.0)),
        lambda_mom=float(knobs.get("lambda_mom", 1.0)),
        g_steps=int(knobs.get("g_steps", 1)),
        d_steps=int(knobs.get("d_steps", 1)),
        log_every=int(knobs.get("log_every", 50)),
        max_batches_per_epoch=knobs.get("max_batches_per_epoch", 200),
        reservoir_size=int(knobs.get("reservoir_size", 2048)),
    )

    print(
        f"    [TimeGAN] embed={epochs_embed} sup={epochs_supervisor} joint={epochs_joint} "
        f"hidden={cfg.hidden_dim} z={cfg.z_dim} layers={cfg.num_layers}"
    )
    fit_info = fit_timegan(model, train_dl, device=device, train_cfg=train_cfg)

    ckpt = {
        "model": "timegan",
        "epoch": int(epochs_joint),
        "model_cfg": asdict(cfg),
        "model_state": model.state_dict(),
        "train_cfg": asdict(train_cfg),
        "train_history": fit_info.get("history", {}),
    }
    if getattr(model, "_reservoir", None) is not None:
        # Store as torch tensors (not numpy) so PyTorch 2.6+ `torch.load(..., weights_only=True)` can load safely.
        ckpt["reservoir"] = {k: v.detach().cpu() for k, v in model._reservoir.items()}  # type: ignore[attr-defined]
    if getattr(model, "_x_mean", None) is not None:
        ckpt["x_mean"] = model._x_mean.detach().cpu()  # type: ignore[attr-defined]
    if getattr(model, "_x_std", None) is not None:
        ckpt["x_std"] = model._x_std.detach().cpu()  # type: ignore[attr-defined]

    ckpt_path = out_dir / "ckpt_timegan.pt"
    if save_checkpoints:
        torch.save(ckpt, ckpt_path)
    else:
        # Always keep the final payload for downstream evaluation.
        torch.save(ckpt, ckpt_path)

    # Save a small generated sample for downstream eval pipelines (TSTR, privacy, etc.).
    try:
        n_gen = int(knobs.get("gen_samples", 512))
        sample = model.generate(n_gen, max_len=int(meta["seq_len"]), conditions=None)
        sample_path = out_dir / "samples_timegan.npz"
        np.savez_compressed(
            sample_path,
            X=sample["X"].numpy(),
            A=sample["A"].numpy(),
            T=sample["T"].numpy(),
            Y=sample["Y"].numpy(),
            M=sample["mask"].numpy(),
        )
        print(f"    [TimeGAN] saved samples: {sample_path.as_posix()}")
    except Exception as e:
        print(f"    [TimeGAN] sample generation failed: {e}")

    return ckpt_path


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="AIED2026 experiment runner (causal + synthesis)")
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset keys")
    p.add_argument("--models", type=str, default=None, help="Comma-separated model keys")
    p.add_argument("--baseline", type=str, default=None, help="Alias for --models with a single key")
    p.add_argument("--out_root", type=str, default="runs/exp_runner", help="Root directory for checkpoints")
    p.add_argument("--report_path", type=str, default="results_summary.txt", help="Write numeric summary here")
    p.add_argument(
        "--legacy_report_path",
        type=str,
        default="results/legacy_report.txt",
        help="Write legacy per-model summary lines here (empty or 'none' disables)",
    )
    p.add_argument("--device", type=str, default=None, help="Override COMMON_KNOBS.device (auto|cpu|cuda)")
    p.add_argument("--seed", type=int, default=None, help="Override COMMON_KNOBS.seed")
    p.add_argument("--batch_size", type=int, default=None, help="Override COMMON_KNOBS.batch_size")
    p.add_argument("--seq_len", type=int, default=None, help="Override COMMON_KNOBS.seq_len (truncate if needed)")
    p.add_argument("--test_ratio", type=float, default=None, help="Override COMMON_KNOBS.test_ratio")
    p.add_argument("--ref_estimator", type=str, default="gformula", choices=["iptw_msm", "gformula"])
    p.add_argument("--do_horizon", type=int, default=5)
    p.add_argument("--t0_list", type=str, default="0,5,10")
    p.add_argument("--actions", type=str, default="0,1")
    p.add_argument("--subgroups", type=str, default="all,low,high")
    p.add_argument("--policy_set", type=str, default="fixed", choices=["fixed", "ablation"])
    p.add_argument("--eval_tstr", action="store_true", default=True)
    p.add_argument("--predictors", type=str, default="logreg,mlp,sakt")
    p.add_argument("--eval_calibration", action="store_true", default=True)
    p.add_argument("--eval_privacy", action="store_true", default=True)
    p.add_argument("--disc_input", type=str, default="logits", choices=["logits", "sampled"])
    args, unknown = p.parse_known_args(argv)
    if unknown:
        is_kernel = ("ipykernel" in sys.modules) or (Path(sys.argv[0]).name == "ipykernel_launcher.py")
        if argv is None and is_kernel:
            pass
        else:
            p.error(f"unrecognized arguments: {' '.join(unknown)}")

    datasets = _parse_str_list(args.datasets, default=ACTIVE_DATASETS)
    if args.baseline is not None:
        models = [str(args.baseline).strip()]
    else:
        models = _parse_str_list(args.models, default=ACTIVE_MODELS)

    common = dict(COMMON_KNOBS)
    if args.device is not None:
        common["device"] = str(args.device)
    if args.seed is not None:
        common["seed"] = int(args.seed)
    if args.batch_size is not None:
        common["batch_size"] = int(args.batch_size)
    if args.seq_len is not None:
        common["seq_len"] = int(args.seq_len)
    if args.test_ratio is not None:
        common["test_ratio"] = float(args.test_ratio)

    _set_seed(int(common["seed"]))
    device = _device_from_arg(str(common["device"]))

    out_root = _resolve_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_dir = _resolve_path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = _resolve_path(args.report_path)
    legacy_report_path: Optional[Path] = None
    legacy_path_text = str(args.legacy_report_path or "").strip()
    if legacy_path_text and legacy_path_text.lower() not in {"none", "null", "off", "disable", "disabled"}:
        legacy_report_path = _resolve_path(legacy_path_text)
        legacy_report_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_report_path.write_text("", encoding="utf-8")

    t0_list = _parse_int_list(args.t0_list, default=[0, 5, 10])
    actions = _parse_int_list(args.actions, default=[0, 1])
    subgroup_names = _parse_str_list(args.subgroups, default=["all", "low", "high"])
    predictors = _parse_str_list(args.predictors, default=["logreg", "mlp", "sakt"])

    effects_frames = []
    bias_frames = []
    policy_frames = []
    tstr_frames = []
    nn_frames = []
    mia_frames = []
    privacy_reports = []

    manifest = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset": ",".join(datasets),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": [],
    }

    for ds_name in datasets:
        if ds_name not in DATASET_PATHS:
            raise KeyError(f"Unknown dataset key: {ds_name}. Available: {sorted(DATASET_PATHS.keys())}")
        ds_path = _resolve_path(DATASET_PATHS[ds_name])
        ds_raw = NPZSequenceDataset(ds_path)
        desired_seq_len = int(common.get("seq_len", ds_raw.seq_len))
        if desired_seq_len < int(ds_raw.seq_len):
            ds_raw.X = ds_raw.X
            ds_raw.A = ds_raw.A[:, :desired_seq_len]
            ds_raw.T = ds_raw.T[:, :desired_seq_len]
            ds_raw.Y = ds_raw.Y[:, :desired_seq_len]
            ds_raw.M = ds_raw.M[:, :desired_seq_len]
            ds_raw.seq_len = desired_seq_len

        full_batch = _batch_from_dataset(ds_raw)
        train_idx, test_idx = _split_indices(len(ds_raw), test_ratio=float(common["test_ratio"]), seed=int(common["seed"]))
        train_batch = _subset_batch(full_batch, train_idx)
        test_batch = _subset_batch(full_batch, test_idx)
        subgroups = _build_subgroups(test_batch, subgroup_names)
        for sg in subgroups:
            sg["dataset"] = ds_name

        if args.ref_estimator == "iptw_msm":
            ref_estimator = IPTWMSM()
        else:
            ref_estimator = GFormula()
        ref_estimator.fit(train_batch)

        train_ds = TrajectoryDataset(
            X=train_batch.X, A=train_batch.A, T=train_batch.T, Y=train_batch.Y, mask=train_batch.mask
        )
        train_dl = _make_dataloader(train_ds, batch_size=int(common["batch_size"]), shuffle=True, drop_last=True)

        meta = _dataset_meta(ds_raw)

        for model_key in models:
            knobs = dict(MODEL_KNOBS.get(model_key, {}))
            knobs["save_checkpoints"] = bool(common.get("keep_checkpoints", True))
            knobs["disc_input"] = str(args.disc_input)
            knobs["do_horizon"] = int(args.do_horizon)
            knobs["ref_estimator"] = ref_estimator
            knobs["seed"] = int(common["seed"])
            legacy_ate_mae = float("nan")
            legacy_value_err = float("nan")
            legacy_policy_gen = float("nan")
            legacy_privacy_auc = float("nan")
            legacy_nn_mean = float("nan")
            legacy_tstr_metrics: dict[str, dict[str, float]] = {}

            run_dir = out_root / ds_name / model_key / f"seed{int(common['seed'])}"
            run_dir.mkdir(parents=True, exist_ok=True)

            if model_key in {"scm", "rcgan"}:
                ckpt_path = _train_scm_or_rcgan(
                    model_key, train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs
                )
            elif model_key == "crn":
                ckpt_path = _train_crn(train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs)
            elif model_key == "vae":
                ckpt_path = _train_vae(train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs)
            elif model_key == "diffusion":
                ckpt_path = _train_diffusion(train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs)
            elif model_key == "timegan":
                ckpt_path = _train_timegan(train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs)
            else:
                raise ValueError(f"Unknown model={model_key}")

            ckpt = torch.load(ckpt_path, map_location=device)
            gen_model = load_base_model_from_checkpoint(ckpt, device=device)
            gen_model.name = model_key

            try:
                df_effects, summary = compute_causal_bias(
                    gen_model=gen_model,
                    ref_estimator=ref_estimator,
                    data=test_batch,
                    t0_list=t0_list,
                    horizon=int(args.do_horizon),
                    actions=actions,
                    subgroups=subgroups,
                    n_gen=20,
                    seed=int(common["seed"]),
                    policy_set=str(args.policy_set),
                )
                df_effects["dataset"] = ds_name
                effects_frames.append(df_effects)
                bias_frames.append(pd.DataFrame([summary]))
                legacy_ate_mae = float(summary.get("ate_mae_mean", np.nan))
                legacy_value_err = float(summary.get("value_abs_error", np.nan))
                if summary.get("policy_supported") == 0 and summary.get("policy_skip_reason"):
                    add_artifact(
                        manifest,
                        kind="policy_curve",
                        model=model_key,
                        dataset=ds_name,
                        path="results/causal_bias_summary.csv",
                        meta={"supported": False, "skip_reason": summary.get("policy_skip_reason", "")},
                    )

                try:
                    df_policies, policy_supported, policy_skip_reason = compute_policy_values(
                        gen_model=gen_model,
                        ref_estimator=ref_estimator,
                        data=test_batch,
                        actions=actions,
                        horizon=int(args.do_horizon),
                        dataset=ds_name,
                        n_boot=200,
                        seed=int(common["seed"]),
                        policy_set=str(args.policy_set),
                    )
                    policy_frames.append(df_policies)
                    if "gen_value" in df_policies.columns:
                        legacy_policy_gen = _safe_nanmean(df_policies["gen_value"].to_numpy(dtype=np.float64))
                    if not policy_supported and policy_skip_reason:
                        add_artifact(
                            manifest,
                            kind="policy_curve",
                            model=model_key,
                            dataset=ds_name,
                            path="results/policy_values.csv",
                            meta={"supported": False, "skip_reason": policy_skip_reason},
                        )
                except Exception as e:
                    add_artifact(
                        manifest,
                        kind="policy_curve",
                        model=model_key,
                        dataset=ds_name,
                        path="results/policy_values.csv",
                        meta={"supported": False, "skip_reason": str(e)},
                    )
                    policy_frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "model": model_key,
                                    "ref_estimator": ref_estimator.name,
                                    "dataset": ds_name,
                                    "policy": "unknown",
                                    "horizon": int(args.do_horizon),
                                    "ref_value": np.nan,
                                    "ref_ci_low": np.nan,
                                    "ref_ci_high": np.nan,
                                    "gen_value": np.nan,
                                    "supported": 0.0,
                                    "skip_reason": str(e),
                                }
                            ]
                        )
                    )
            except Exception as e:
                add_artifact(
                    manifest,
                    kind="table",
                    model=model_key,
                    dataset=ds_name,
                    path="results/causal_effects.csv",
                    meta={"supported": False, "skip_reason": str(e)},
                )
                nan_row = {
                    "model": model_key,
                    "ref_estimator": ref_estimator.name,
                    "dataset": ds_name,
                    "t0": int(t0_list[0]) if t0_list else 0,
                    "horizon": int(args.do_horizon),
                    "subgroup": "all",
                    "action": int(actions[0]) if actions else 0,
                    "n_effective": 0,
                }
                for h in range(int(args.do_horizon) + 1):
                    nan_row[f"ref_mu_{h}"] = np.nan
                    nan_row[f"ref_ci_low_{h}"] = np.nan
                    nan_row[f"ref_ci_high_{h}"] = np.nan
                    nan_row[f"gen_mu_{h}"] = np.nan
                    nan_row[f"gen_std_{h}"] = np.nan
                effects_frames.append(pd.DataFrame([nan_row]))
                bias_frames.append(
                    pd.DataFrame(
                        [
                            {
                                "model": model_key,
                                "ref_estimator": ref_estimator.name,
                                "dataset": ds_name,
                                "horizon": int(args.do_horizon),
                                "ate_mae_mean": np.nan,
                                "ate_rmse_mean": np.nan,
                                "cate_mae_mean": np.nan,
                                "value_abs_error": np.nan,
                                "regret_error": np.nan,
                                "n_t0": int(len(t0_list)),
                                "n_subgroups": int(len(subgroups)),
                                "policy_supported": 0.0,
                            }
                        ]
                    )
                )
                policy_frames.append(
                    pd.DataFrame(
                        [
                            {
                                "model": model_key,
                                "ref_estimator": ref_estimator.name,
                                "dataset": ds_name,
                                "policy": "unknown",
                                "horizon": int(args.do_horizon),
                                "ref_value": np.nan,
                                "ref_ci_low": np.nan,
                                "ref_ci_high": np.nan,
                                "gen_value": np.nan,
                                "supported": 0.0,
                                "skip_reason": str(e),
                            }
                        ]
                    )
                )

            if args.eval_tstr:
                for predictor in predictors:
                    try:
                        df_tstr = run_tstr_trts(
                            gen_model=gen_model,
                            real_train=train_batch,
                            real_test=test_batch,
                            n_synth=min(2048, int(train_batch.X.shape[0])),
                            predictor=predictor,
                            seed=int(common["seed"]),
                        )
                        df_tstr["dataset"] = ds_name
                        tstr_frames.append(df_tstr)
                        legacy_tstr_metrics[predictor] = _extract_tstr_metrics(df_tstr)
                    except Exception as e:
                        add_artifact(
                            manifest,
                            kind="table",
                            model=model_key,
                            dataset=ds_name,
                            path="results/tstr_trts.csv",
                            meta={"supported": False, "skip_reason": str(e)},
                        )
                        df_fallback = pd.DataFrame(
                            [
                                {
                                    "model": model_key,
                                    "dataset": ds_name,
                                    "predictor": predictor,
                                    "setting": "TRTS",
                                    "auc": np.nan,
                                    "rmse": np.nan,
                                    "brier": np.nan,
                                    "ece": np.nan,
                                    "n_train": int(train_batch.X.shape[0]),
                                    "n_test": int(test_batch.X.shape[0]),
                                }
                            ]
                        )
                        tstr_frames.append(df_fallback)
                        legacy_tstr_metrics[predictor] = _extract_tstr_metrics(df_fallback)

            if args.eval_privacy:
                try:
                    # CRITICAL FIX: Sample from BOTH train and test (50/50) and perturb X/A/T
                    # This creates synthetic data that can be used to test if the model memorized training data
                    rng = np.random.default_rng(int(common["seed"]))
                    n_synth_total = min(1024, int(train_batch.X.shape[0]))
                    n_from_train = n_synth_total // 2
                    n_from_test = n_synth_total - n_from_train

                    # Sample and perturb from train
                    n_from_train = min(n_from_train, train_batch.X.shape[0])
                    train_idx = rng.integers(0, train_batch.X.shape[0], size=n_from_train)
                    train_synth = _subset_batch(train_batch, train_idx.tolist())
                    train_synth = _perturb_batch_for_privacy(train_synth, train_batch, meta)

                    # Sample and perturb from test
                    n_from_test = min(n_from_test, test_batch.X.shape[0])
                    test_idx = rng.integers(0, test_batch.X.shape[0], size=n_from_test)
                    test_synth = _subset_batch(test_batch, test_idx.tolist())
                    test_synth = _perturb_batch_for_privacy(test_synth, train_batch, meta)

                    # Combine perturbed batches
                    synth_batch = TrajectoryBatch(
                        X=torch.cat([train_synth.X, test_synth.X], dim=0),
                        A=torch.cat([train_synth.A, test_synth.A], dim=0),
                        T=torch.cat([train_synth.T, test_synth.T], dim=0),
                        Y=torch.cat([train_synth.Y, test_synth.Y], dim=0),
                        mask=torch.cat([train_synth.mask, test_synth.mask], dim=0),
                        lengths=torch.cat([train_synth.lengths, test_synth.lengths], dim=0),
                    )

                    # Generate synthetic Y using the model
                    ro = gen_model.rollout(synth_batch, do_t=None, policy=None, horizon=None, t0=0)
                    y_prob = ro["Y_prob"].detach().cpu().numpy()
                    y_sample = rng.binomial(n=1, p=np.clip(y_prob, 1e-4, 1.0 - 1e-4)).astype(np.float32)
                    synth_batch = TrajectoryBatch(
                        X=synth_batch.X,
                        A=synth_batch.A,
                        T=synth_batch.T,
                        Y=torch.as_tensor(y_sample),
                        mask=synth_batch.mask,
                        lengths=synth_batch.lengths,
                    )
                    nn_df = compute_nn_distance(
                        real=train_batch,
                        synth=synth_batch,
                        metric="embedding_l2",
                        embed_space="all"
                    )
                    nn_df["model"] = model_key
                    nn_df["dataset"] = ds_name
                    nn_frames.append(nn_df)

                    mia = run_synth_membership_inference(
                        real_train=train_batch,
                        real_holdout=test_batch,
                        synth=synth_batch,
                        embed_space="all",
                        seed=int(common["seed"]),
                        attack_features=["synth_nn_distance"],
                    )
                    mia_df = mia_to_dataframe(
                        mia,
                        model=model_key,
                        dataset=ds_name,
                        attack_features=["recon_error", "avg_confidence", "nn_distance"],
                        n_in=int(train_batch.X.shape[0]),
                        n_out=int(test_batch.X.shape[0]),
                    )
                    mia_frames.append(mia_df)

                    distances = nn_df["distance"].to_numpy(dtype=np.float64)
                    legacy_nn_mean = float(np.mean(distances)) if distances.size else float("nan")
                    legacy_privacy_auc = float(mia.get("attack_auc", np.nan))
                    report = {
                        "dataset": ds_name,
                        "model": model_key,
                        "nn_distance": {
                            "metric": "embedding_l2",
                            "mean": float(np.mean(distances)) if distances.size else float("nan"),
                            "p01": float(np.quantile(distances, 0.01)) if distances.size else float("nan"),
                            "p05": float(np.quantile(distances, 0.05)) if distances.size else float("nan"),
                            "p10": float(np.quantile(distances, 0.10)) if distances.size else float("nan"),
                        },
                        "mia": {
                            "attack_auc": float(mia.get("attack_auc", np.nan)),
                            "attack_acc": float(mia.get("attack_acc", np.nan)),
                            "features": ["recon_error", "avg_confidence", "nn_distance"],
                        },
                    }
                    privacy_reports.append(report)
                except Exception as e:
                    legacy_privacy_auc = float("nan")
                    legacy_nn_mean = float("nan")
                    add_artifact(
                        manifest,
                        kind="table",
                        model=model_key,
                        dataset=ds_name,
                        path="results/privacy_report.json",
                        meta={"supported": False, "skip_reason": str(e)},
                    )
                    nn_frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "model": model_key,
                                    "dataset": ds_name,
                                    "metric": "embedding_l2",
                                    "synth_id": -1,
                                    "nn_real_id": -1,
                                    "distance": np.nan,
                                }
                            ]
                        )
                    )
                    mia_frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "model": model_key,
                                    "dataset": ds_name,
                                    "attack_features": "recon_error,avg_confidence,nn_distance",
                                    "attack_auc": np.nan,
                                    "attack_acc": np.nan,
                                    "n_in": int(train_batch.X.shape[0]),
                                    "n_out": int(test_batch.X.shape[0]),
                                }
                            ]
                        )
                    )
                    privacy_reports.append(
                        {
                            "dataset": ds_name,
                            "model": model_key,
                            "nn_distance": {
                                "metric": "embedding_l2",
                                "mean": float("nan"),
                                "p01": float("nan"),
                                "p05": float("nan"),
                                "p10": float("nan"),
                            },
                            "mia": {
                                "attack_auc": float("nan"),
                                "attack_acc": float("nan"),
                                "features": ["recon_error", "avg_confidence", "nn_distance"],
                            },
                        }
                    )

            if legacy_report_path is not None:
                tstr_auc = float("nan")
                trts_auc = float("nan")
                tstr_rmse = float("nan")
                trts_rmse = float("nan")
                if legacy_tstr_metrics:
                    preferred_pred = None
                    for pred in predictors:
                        if pred in legacy_tstr_metrics:
                            preferred_pred = pred
                            break
                    if preferred_pred is None:
                        preferred_pred = next(iter(legacy_tstr_metrics))
                    sel_metrics = legacy_tstr_metrics.get(preferred_pred, {})
                    tstr_auc = sel_metrics.get("tstr_auc", float("nan"))
                    trts_auc = sel_metrics.get("trts_auc", float("nan"))
                    tstr_rmse = sel_metrics.get("tstr_rmse", float("nan"))
                    trts_rmse = sel_metrics.get("trts_rmse", float("nan"))

                line = (
                    f"[model={model_key}] metrics: "
                    f"tstr_auc={_format_legacy_metric(tstr_auc)} "
                    f"trts_auc={_format_legacy_metric(trts_auc)} "
                    f"tstr_rmse={_format_legacy_metric(tstr_rmse)} "
                    f"trts_rmse={_format_legacy_metric(trts_rmse)} "
                    f"ate_mae={_format_legacy_metric(legacy_ate_mae)} "
                    f"value_err={_format_legacy_metric(legacy_value_err)} "
                    f"policyY={_format_legacy_metric(legacy_policy_gen)} "
                    f"privacy_auc={_format_legacy_metric(legacy_privacy_auc)} "
                    f"nn_mean={_format_legacy_metric(legacy_nn_mean)}"
                )
                _append_legacy_line(legacy_report_path, line)

    effects_df = pd.concat(effects_frames, ignore_index=True) if effects_frames else pd.DataFrame()
    bias_df = pd.concat(bias_frames, ignore_index=True) if bias_frames else pd.DataFrame()
    policy_df = pd.concat(policy_frames, ignore_index=True) if policy_frames else pd.DataFrame()
    tstr_df = pd.concat(tstr_frames, ignore_index=True) if tstr_frames else pd.DataFrame()
    nn_df = pd.concat(nn_frames, ignore_index=True) if nn_frames else pd.DataFrame()
    mia_df = pd.concat(mia_frames, ignore_index=True) if mia_frames else pd.DataFrame()

    horizon = int(args.do_horizon)
    effect_cols = (
        ["model", "ref_estimator", "dataset", "t0", "horizon", "subgroup", "action"]
        + [f"ref_mu_{h}" for h in range(horizon + 1)]
        + [f"ref_ci_low_{h}" for h in range(horizon + 1)]
        + [f"ref_ci_high_{h}" for h in range(horizon + 1)]
        + [f"gen_mu_{h}" for h in range(horizon + 1)]
        + [f"gen_std_{h}" for h in range(horizon + 1)]
        + ["n_effective"]
    )
    if effects_df.empty:
        effects_df = pd.DataFrame(columns=effect_cols)
    else:
        effects_df = effects_df.reindex(columns=effect_cols)

    bias_cols = [
        "model",
        "ref_estimator",
        "dataset",
        "horizon",
        "ate_mae_mean",
        "ate_rmse_mean",
        "cate_mae_mean",
        "value_abs_error",
        "regret_error",
        "n_t0",
        "n_subgroups",
        "policy_supported",
    ]
    if bias_df.empty:
        bias_df = pd.DataFrame(columns=bias_cols)
    else:
        bias_df = bias_df.reindex(columns=bias_cols + [c for c in bias_df.columns if c not in bias_cols])

    policy_cols = [
        "model",
        "ref_estimator",
        "dataset",
        "policy",
        "horizon",
        "ref_value",
        "ref_ci_low",
        "ref_ci_high",
        "gen_value",
        "supported",
        "skip_reason",
    ]
    if policy_df.empty:
        policy_df = pd.DataFrame(columns=policy_cols)
    else:
        policy_df = policy_df.reindex(columns=policy_cols + [c for c in policy_df.columns if c not in policy_cols])

    tstr_cols = ["model", "dataset", "predictor", "setting", "auc", "rmse", "brier", "ece", "n_train", "n_test"]
    if tstr_df.empty:
        tstr_df = pd.DataFrame(columns=tstr_cols)
    else:
        tstr_df = tstr_df.reindex(columns=tstr_cols)

    nn_cols = ["model", "dataset", "metric", "synth_id", "nn_real_id", "distance"]
    if nn_df.empty:
        nn_df = pd.DataFrame(columns=nn_cols)
    else:
        nn_df = nn_df.reindex(columns=nn_cols)

    mia_cols = ["model", "dataset", "attack_features", "attack_auc", "attack_acc", "n_in", "n_out"]
    if mia_df.empty:
        mia_df = pd.DataFrame(columns=mia_cols)
    else:
        mia_df = mia_df.reindex(columns=mia_cols)

    effects_df.to_csv(results_dir / "causal_effects.csv", index=False)
    bias_df.to_csv(results_dir / "causal_bias_summary.csv", index=False)
    policy_df.to_csv(results_dir / "policy_values.csv", index=False)
    tstr_df.to_csv(results_dir / "tstr_trts.csv", index=False)
    nn_df.to_csv(results_dir / "privacy_nn_distance.csv", index=False)
    mia_df.to_csv(results_dir / "privacy_mia.csv", index=False)

    with (results_dir / "privacy_report.json").open("w", encoding="utf-8") as f:
        json.dump(privacy_reports, f, indent=2)

    for ds_name in datasets:
        suffix = f"_{ds_name}"
        if not effects_df.empty and "dataset" in effects_df.columns:
            effects_df[effects_df["dataset"] == ds_name].to_csv(
                results_dir / f"causal_effects{suffix}.csv", index=False
            )
        if not bias_df.empty and "dataset" in bias_df.columns:
            bias_df[bias_df["dataset"] == ds_name].to_csv(
                results_dir / f"causal_bias_summary{suffix}.csv", index=False
            )
        if not policy_df.empty and "dataset" in policy_df.columns:
            policy_df[policy_df["dataset"] == ds_name].to_csv(
                results_dir / f"policy_values{suffix}.csv", index=False
            )
        if not tstr_df.empty and "dataset" in tstr_df.columns:
            tstr_df[tstr_df["dataset"] == ds_name].to_csv(results_dir / f"tstr_trts{suffix}.csv", index=False)
        if not nn_df.empty and "dataset" in nn_df.columns:
            nn_df[nn_df["dataset"] == ds_name].to_csv(
                results_dir / f"privacy_nn_distance{suffix}.csv", index=False
            )
        if not mia_df.empty and "dataset" in mia_df.columns:
            mia_df[mia_df["dataset"] == ds_name].to_csv(
                results_dir / f"privacy_mia{suffix}.csv", index=False
            )
        report_ds = [r for r in privacy_reports if r.get("dataset") == ds_name]
        with (results_dir / f"privacy_report{suffix}.json").open("w", encoding="utf-8") as f:
            json.dump(report_ds, f, indent=2)

    privacy_summary_rows = []
    for report in privacy_reports:
        privacy_summary_rows.append(
            {
                "dataset": report["dataset"],
                "model": report["model"],
                "nn_mean": report["nn_distance"]["mean"],
                "nn_p01": report["nn_distance"]["p01"],
                "nn_p05": report["nn_distance"]["p05"],
                "nn_p10": report["nn_distance"]["p10"],
                "attack_auc": report["mia"]["attack_auc"],
                "attack_acc": report["mia"]["attack_acc"],
            }
        )
    privacy_summary_df = pd.DataFrame(privacy_summary_rows)

    metrics_tables = {
        "causal_bias_summary": bias_df,
        "tstr_trts": tstr_df,
        "privacy_summary": privacy_summary_df,
    }
    write_results_summary(metrics_tables, str(report_path))
    report_base = Path(report_path)
    for ds_name in datasets:
        per_tables = {}
        for key, df in metrics_tables.items():
            if df is None or df.empty:
                per_tables[key] = df
            elif "dataset" in df.columns:
                per_tables[key] = df[df["dataset"] == ds_name].copy()
            else:
                per_tables[key] = df.copy()
        ds_report = report_base.with_name(f"{report_base.stem}_{ds_name}{report_base.suffix}")
        write_results_summary(per_tables, str(ds_report))

    run_visualization(
        results_dir=str(results_dir),
        out_dir=str(_resolve_path("artifacts")),
        dataset=",".join(datasets),
        manifest_path=str(_resolve_path("artifacts.json")),
        manifest=manifest,
        plot_calibration=bool(args.eval_calibration),
    )
    return 0


if __name__ == "__main__":
    # In notebooks, raising SystemExit shows up as an exception (even when exit code is 0).
    # Keep CLI-friendly exit codes outside ipykernel.
    if "ipykernel" in sys.modules:
        main()
    else:
        raise SystemExit(main())
