from __future__ import annotations

import sys
import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# Allow running as a script (e.g., `python src/main.py`) as well as a module (`python -m src.main`).
if __package__ in (None, ""):
    _repo_root_for_imports = Path(__file__).resolve().parents[1]
    if str(_repo_root_for_imports) not in sys.path:
        sys.path.insert(0, str(_repo_root_for_imports))

from src.data import NPZSequenceDataset, SequenceBatch, move_batch
from src.model.baselines import (
    CRN,
    CRNConfig,
    RCGANConfig,
    RCGANGenerator,
    SeqDiffusion,
    SeqDiffusionConfig,
    SeqVAE,
    SeqVAEConfig,
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
ACTIVE_MODELS = ["scm", "rcgan", "crn", "diffusion", "vae"]  # options: ["scm", "rcgan", "vae", "diffusion", "crn"]

# Common training knobs (defaults)
COMMON_KNOBS = dict(
    batch_size=512,
    seq_len=50,  # will truncate if npz is longer
    device="auto",
    seed=42,
    test_ratio=0.2,  # split held-out set for predictive fidelity
)

# Model-specific knobs (override defaults)
MODEL_KNOBS = {
    "scm": dict(
        epochs_a=5,
        epochs_b=5,
        epochs_c=5,
        dynamics="transformer",
        d_k=64,
        w_do=1.0,
        w_advT=0.1,
        w_cf=0.0,
        grl_lambda=1.0,
        lr_advT=1e-4,
        do_time_sampling="random",
        do_num_time_samples=1,
        do_min_arm_samples=8,
        do_actions="0,1",
        cf_num_time_samples=1,
    ),
    "rcgan": dict(
        epochs_a=0,
        epochs_b=10,
        epochs_c=0,
        rnn="gru",
        rcgan_hidden=128,
    ),
    "crn": dict(
        epochs=10,
        crn_hidden=128,
        crn_grl_lambda=1.0,
    ),
    "diffusion": dict(
        epochs=10,
        diff_steps=50,
        diff_backbone="transformer",
    ),
    "vae": dict(
        epochs=10,
        vae_z_dim=32,
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

    # Causal knobs (ported from src/train.py)
    do_time_index = int(knobs.get("do_time_index", 5))
    do_time_sampling = str(knobs.get("do_time_sampling", "random"))
    do_num_time_samples = int(knobs.get("do_num_time_samples", 1))
    do_min_arm_samples = int(knobs.get("do_min_arm_samples", 8))
    do_actions_raw = knobs.get("do_actions", "0,1")
    cf_num_time_samples = int(knobs.get("cf_num_time_samples", 1))

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

                real_seq = disc.encode_inputs(x=X, a=A, t=T, y=Y)
                fake_seq = disc.encode_inputs(x=X, a=A, t=T, y=y_fake.detach())

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
                        # (A) do-alignment: MMD over features (X, K_t, Y_t) across arms
                        if w_do > 0.0:
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
                        if w_cf > 0.0:
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

                fake_seq_g = disc.encode_inputs(x=X, a=A, t=T, y=y_fake)
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

    def save_ckpt(epoch: int) -> Path:
        ckpt = {"model": "vae", "epoch": int(epoch), "model_cfg": asdict(cfg), "model_state": model.state_dict()}
        path = out_dir / f"ckpt_vae_ep{epoch}.pt"
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

    def save_ckpt(epoch: int) -> Path:
        ckpt = {
            "model": "diffusion",
            "epoch": int(epoch),
            "model_cfg": asdict(cfg),
            "model_state": model.state_dict(),
        }
        path = out_dir / f"ckpt_diffusion_ep{epoch}.pt"
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

    def save_ckpt(epoch: int) -> Path:
        ckpt = {"model": "crn", "epoch": int(epoch), "model_cfg": asdict(cfg), "model_state": model.state_dict()}
        path = out_dir / f"ckpt_crn_ep{epoch}.pt"
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

    return last_ckpt


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="AIED2026 config-driven experiment runner")
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset keys (override ACTIVE_DATASETS)")
    p.add_argument("--models", type=str, default=None, help="Comma-separated model keys (override ACTIVE_MODELS)")
    p.add_argument("--out_root", type=str, default="runs/exp_runner", help="Root directory for checkpoints")
    p.add_argument("--report_path", type=str, default="results_summary.txt", help="Write summary to this .txt file")
    p.add_argument("--device", type=str, default=None, help="Override COMMON_KNOBS.device (e.g., auto|cpu|cuda)")
    p.add_argument("--seed", type=int, default=None, help="Override COMMON_KNOBS.seed")
    p.add_argument("--batch_size", type=int, default=None, help="Override COMMON_KNOBS.batch_size")
    p.add_argument("--seq_len", type=int, default=None, help="Override COMMON_KNOBS.seq_len (truncate if needed)")
    p.add_argument("--test_ratio", type=float, default=None, help="Override COMMON_KNOBS.test_ratio")
    args, unknown = p.parse_known_args(argv)
    if unknown:
        # Jupyter / IPython kernels inject extra flags like `-f <connection.json>` or `--f=...`.
        # When running via `%run src/main.py`, treat those as out-of-scope and ignore them.
        is_kernel = ("ipykernel" in sys.modules) or (Path(sys.argv[0]).name == "ipykernel_launcher.py")
        if argv is None and is_kernel:
            pass
        else:
            p.error(f"unrecognized arguments: {' '.join(unknown)}")

    datasets = ACTIVE_DATASETS if args.datasets is None else [x.strip() for x in args.datasets.split(",") if x.strip()]
    models = ACTIVE_MODELS if args.models is None else [x.strip() for x in args.models.split(",") if x.strip()]

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
    report_path = _resolve_path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    started = datetime.now().isoformat(timespec="seconds")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("AIED2026 Experiment Summary\n")
        f.write(f"Started: {started}\n")
        f.write(f"Device: {device}\n")
        f.write(f"COMMON_KNOBS: {common}\n\n")

        for ds_name in datasets:
            if ds_name not in DATASET_PATHS:
                raise KeyError(f"Unknown dataset key: {ds_name}. Available: {sorted(DATASET_PATHS.keys())}")

            ds_path = _resolve_path(DATASET_PATHS[ds_name])
            if not ds_path.exists():
                raise FileNotFoundError(f"Dataset not found: {ds_path}")

            ds_raw = NPZSequenceDataset(ds_path)
            desired_seq_len = int(common.get("seq_len", ds_raw.seq_len))
            if desired_seq_len >= int(ds_raw.seq_len):
                ds: Dataset = ds_raw
            else:
                ds = _TruncatedSequenceDataset(ds_raw, seq_len=desired_seq_len)

            meta = _dataset_meta(ds)
            train_idx, test_idx = _split_indices(
                len(ds), test_ratio=float(common["test_ratio"]), seed=int(common["seed"])
            )
            train_ds: Dataset = Subset(ds, train_idx) if train_idx else ds
            test_ds: Dataset = Subset(ds, test_idx) if test_idx else Subset(ds, [])

            train_dl = _make_dataloader(train_ds, batch_size=int(common["batch_size"]), shuffle=True, drop_last=True)
            test_dl = _make_dataloader(test_ds, batch_size=int(common["batch_size"]), shuffle=False, drop_last=False)

            ds_header = (
                f"[dataset={ds_name}] path={ds_path.as_posix()} n={len(ds)} "
                f"seq_len={meta['seq_len']} d_x={meta['d_x']} a_vocab={meta['a_vocab_size']} t_vocab={meta['t_vocab_size']}\n"
            )
            print(ds_header.strip())
            f.write(ds_header)

            for model_key in models:
                knobs = dict(MODEL_KNOBS.get(model_key, {}))
                run_dir = out_root / ds_name / model_key / f"seed{int(common['seed'])}"
                run_dir.mkdir(parents=True, exist_ok=True)

                _set_seed(int(common["seed"]))
                print(f"  -> Training model={model_key} (out={run_dir.as_posix()})")

                if model_key in {"scm", "rcgan"}:
                    ckpt_path = _train_scm_or_rcgan(
                        model_key, train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs
                    )
                elif model_key == "crn":
                    ckpt_path = _train_crn(train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs)
                elif model_key == "vae":
                    ckpt_path = _train_vae(train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs)
                elif model_key == "diffusion":
                    ckpt_path = _train_diffusion(
                        train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs
                    )
                else:
                    raise ValueError(
                        f"Unknown model={model_key}. Expected one of {['scm','rcgan','vae','diffusion','crn']}"
                    )

                ckpt = torch.load(ckpt_path, map_location=device)
                gen, loaded_name = load_rollout_model_from_checkpoint(ckpt, device=device)

                pred = _predictive_fidelity(gen, loaded_name, test_dl, device=device)

                t_vocab = int(getattr(getattr(gen, "cfg", None), "t_vocab_size", meta["t_vocab_size"] or 2))
                actions = [0, 1] if t_vocab >= 2 else [0]
                time_indices = [0, 5, 10]
                mmd = _interventional_fidelity_mmd(
                    gen,
                    loaded_name,
                    test_dl,
                    device=device,
                    seq_len=int(meta["seq_len"]),
                    actions=actions,
                    time_indices=time_indices,
                )
                pv = _policy_value_random(gen, loaded_name, test_dl, device=device, num_actions=t_vocab)

                metrics_line = (
                    f"    metrics: auc={float(pred['auc']):.4f} rmse={float(pred['rmse']):.4f} "
                    f"mmd={float(mmd['mmd']):.4f} policyY={float(pv['value']):.4f}\n"
                )
                print(metrics_line.strip())

                f.write(f"  [model={loaded_name}] ckpt={ckpt_path.as_posix()}\n")
                f.write(metrics_line)
                if pred.get("note"):
                    f.write(f"    predictive_note: {pred['note']}\n")
                if mmd.get("note"):
                    f.write(f"    interventional_note: {mmd['note']}\n")
                if pv.get("note"):
                    f.write(f"    policy_note: {pv['note']}\n")
                f.write("\n")

            f.write("\n")

        finished = datetime.now().isoformat(timespec="seconds")
        f.write(f"Finished: {finished}\n")

    print(f"Wrote summary: {report_path.as_posix()}")
    return 0


if __name__ == "__main__":
    # In notebooks, raising SystemExit shows up as an exception (even when exit code is 0).
    # Keep CLI-friendly exit codes outside ipykernel.
    if "ipykernel" in sys.modules:
        main()
    else:
        raise SystemExit(main())
