from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F

from src.baselines import BaseSeqModel
from src.data import TrajectoryBatch
from src.privacy.nn_distance import _embed


def _recon_error(gen_model: BaseSeqModel, batch: TrajectoryBatch) -> np.ndarray:
    ro = gen_model.rollout(batch, do_t=None, policy=None, horizon=None, t0=0, teacher_forcing=False)
    y_prob = ro["Y_prob"].detach().cpu().numpy()
    y_true = batch.Y.detach().cpu().numpy()
    m = batch.mask.detach().cpu().numpy()
    eps = 1e-6
    loss = -(y_true * np.log(y_prob + eps) + (1.0 - y_true) * np.log(1.0 - y_prob + eps))
    per_seq = (loss * m).sum(axis=1) / np.maximum(1.0, m.sum(axis=1))
    return per_seq.astype(np.float32)


def _avg_confidence(gen_model: BaseSeqModel, batch: TrajectoryBatch) -> np.ndarray:
    ro = gen_model.rollout(batch, do_t=None, policy=None, horizon=None, t0=0, teacher_forcing=False)
    y_prob = ro["Y_prob"].detach().cpu().numpy()
    m = batch.mask.detach().cpu().numpy()
    conf = np.abs(y_prob - 0.5)
    per_seq = (conf * m).sum(axis=1) / np.maximum(1.0, m.sum(axis=1))
    return per_seq.astype(np.float32)


def _nn_distance_feature(
    train: TrajectoryBatch,
    batch: TrajectoryBatch,
    *,
    leave_one_out: bool,
    embed_space: str,
) -> np.ndarray:
    train_emb = _embed(train, embed_space=embed_space)
    batch_emb = _embed(batch, embed_space=embed_space)
    if train_emb.shape[0] == 0 or batch_emb.shape[0] == 0:
        return np.zeros((batch_emb.shape[0],), dtype=np.float32)

    n_neighbors = 2 if leave_one_out else 1
    if train_emb.shape[0] < n_neighbors:
        n_neighbors = train_emb.shape[0]
    if n_neighbors == 0:
        return np.zeros((batch_emb.shape[0],), dtype=np.float32)

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(train_emb)
    distances, _ = nn.kneighbors(batch_emb, return_distance=True)
    if leave_one_out and n_neighbors > 1:
        dist = distances[:, 1]
    else:
        dist = distances[:, 0]
    return dist.astype(np.float32)


def _y_stats(batch: TrajectoryBatch) -> np.ndarray:
    return _embed(batch, embed_space="y_only")


def _stack_features(features: list[np.ndarray]) -> np.ndarray:
    cols = []
    for feat in features:
        if feat.ndim == 1:
            cols.append(feat[:, None])
        else:
            cols.append(feat)
    if not cols:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(cols, axis=1).astype(np.float32)


def _ensure_3d_y(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 2:
        return y.unsqueeze(-1)
    return y


def _masked_mean_per_seq(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    if mask is None:
        return x.mean(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (x * mask).sum(dim=1) / denom


def _latent_seq_norm(latent: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if latent.ndim == 3:
        norms = torch.norm(latent, dim=-1)
        return _masked_mean_per_seq(norms, mask)
    if latent.ndim == 2:
        return torch.norm(latent, dim=-1)
    return torch.zeros(latent.shape[0], device=latent.device)


def _get_model_and_name(gen_model: BaseSeqModel) -> Tuple[Optional[torch.nn.Module], str]:
    model = getattr(gen_model, "model", None)
    model_name = str(getattr(gen_model, "name", ""))
    return model, model_name


def _whitebox_forward(
    *,
    model: torch.nn.Module,
    model_name: str,
    batch: TrajectoryBatch,
) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    X = batch.X.to(device)
    A = batch.A.to(device)
    T = batch.T.to(device)
    Y = batch.Y.to(device)
    M = batch.mask.to(device)

    out: Dict[str, torch.Tensor] = {"mask": M}

    if model_name.startswith("scm"):
        tf = model.teacher_forcing(  # type: ignore[attr-defined]
            x=X, a=A, t=T, y=Y, mask=M, eps_mode="zero", causal_only=True
        )
        out["y_logits"] = tf["y_logits"]
        if "k_c" in tf:
            out["k_c"] = tf["k_c"]
        if "k_s" in tf:
            out["k_s"] = tf["k_s"]
        if "k" in tf:
            out["k"] = tf["k"]
        return out

    if model_name == "rcgan":
        tf = model.teacher_forcing(  # type: ignore[attr-defined]
            x=X, a=A, t=T, y=Y, mask=M, stochastic_y=False
        )
        out["y_logits"] = tf["y_logits"]
        return out

    if model_name == "vae":
        mu, logvar = model.encode(x=X, a=A, t=T, y=Y, mask=M)  # type: ignore[attr-defined]
        dec = model.decode(  # type: ignore[attr-defined]
            x=X, a=A, t=T, mask=M, z=mu, y=Y, teacher_forcing=True, stochastic_y=False
        )
        out["y_logits"] = dec["y_logits"]
        out["z_mu"] = mu
        out["z_logvar"] = logvar
        return out

    if model_name == "crn":
        tf = model.forward(x=X, a=A, t=T, y=Y, mask=M)  # type: ignore[attr-defined]
        out["y_logits"] = tf["y_logits"]
        if "h" in tf:
            out["h"] = tf["h"]
        return out

    if model_name == "timegan":
        tf = model.teacher_forcing(  # type: ignore[attr-defined]
            x=X, a=A, t=T, y=Y, mask=M, stochastic_y=False
        )
        out["y_logits"] = tf["y_logits"]
        if hasattr(model, "generate_hidden"):
            z_dim = int(getattr(getattr(model, "cfg", None), "z_dim", 16))
            z = torch.zeros(X.shape[0], X.shape[1], z_dim, device=device)
            out["h_hat"] = model.generate_hidden(x=X, a=A, t=T, z=z)  # type: ignore[attr-defined]
        return out

    return out


def _diffusion_loss_per_seq(
    *,
    model: torch.nn.Module,
    batch: TrajectoryBatch,
    seed: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    X = batch.X.to(device)
    A = batch.A.to(device)
    T = batch.T.to(device)
    Y = batch.Y.to(device)
    M = batch.mask.to(device)

    bsz = X.shape[0]
    rng = torch.Generator(device=device).manual_seed(int(seed))
    y0 = _ensure_3d_y(Y).float()
    timesteps = torch.randint(
        low=0,
        high=int(model.cfg.num_steps),  # type: ignore[attr-defined]
        size=(bsz,),
        device=device,
        generator=rng,
    )
    noise = torch.randn_like(y0, generator=rng)
    alpha_bar = model.alpha_bars[timesteps].view(bsz, 1, 1)  # type: ignore[attr-defined]
    y_noisy = torch.sqrt(alpha_bar) * y0 + torch.sqrt(1.0 - alpha_bar) * noise
    pred = model._predict_eps(  # type: ignore[attr-defined]
        x=X, a=A, t=T, y_noisy=y_noisy, timesteps=timesteps, mask=M
    )
    loss_t = (pred - noise).pow(2).mean(dim=-1)  # [B,T]
    per_seq = _masked_mean_per_seq(loss_t, M)
    return per_seq.detach().cpu().numpy().astype(np.float32)


def _whitebox_per_sample_features(
    *,
    gen_model: BaseSeqModel,
    batch: TrajectoryBatch,
    feature_names: List[str],
    seed: int,
) -> Dict[str, np.ndarray]:
    model, model_name = _get_model_and_name(gen_model)
    bsz = int(batch.X.shape[0])
    out: Dict[str, np.ndarray] = {}

    if model is None or bsz == 0:
        for name in feature_names:
            out[name] = np.zeros((bsz,), dtype=np.float32)
        return out

    if model_name == "diffusion":
        if "wb_loss" in feature_names:
            out["wb_loss"] = _diffusion_loss_per_seq(model=model, batch=batch, seed=seed)
        if "wb_logit_margin" in feature_names:
            out["wb_logit_margin"] = np.zeros((bsz,), dtype=np.float32)
        if "wb_latent_norm" in feature_names:
            out["wb_latent_norm"] = np.zeros((bsz,), dtype=np.float32)
        return out

    with torch.no_grad():
        wb = _whitebox_forward(model=model, model_name=model_name, batch=batch)
        mask = wb.get("mask")
        y_logits = wb.get("y_logits")

        if "wb_loss" in feature_names:
            if y_logits is None:
                out["wb_loss"] = np.zeros((bsz,), dtype=np.float32)
            else:
                y_true = _ensure_3d_y(batch.Y.to(y_logits.device)).float()
                loss = F.binary_cross_entropy_with_logits(y_logits, y_true, reduction="none")
                per_seq = _masked_mean_per_seq(loss, mask)
                out["wb_loss"] = per_seq.detach().cpu().numpy().astype(np.float32)

        if "wb_logit_margin" in feature_names:
            if y_logits is None:
                out["wb_logit_margin"] = np.zeros((bsz,), dtype=np.float32)
            else:
                margin = _masked_mean_per_seq(y_logits.abs(), mask)
                out["wb_logit_margin"] = margin.detach().cpu().numpy().astype(np.float32)

        if "wb_latent_norm" in feature_names:
            latent: Optional[torch.Tensor] = None
            if model_name.startswith("scm"):
                if "k_c" in wb and wb["k_c"].shape[-1] > 0:
                    latent = wb["k_c"]
                elif "k" in wb:
                    latent = wb["k"]
                if latent is not None:
                    mask_k = torch.cat(
                        [torch.ones(latent.shape[0], 1, device=latent.device), mask], dim=1
                    )
                    latent_norm = _latent_seq_norm(latent, mask_k)
                    out["wb_latent_norm"] = latent_norm.detach().cpu().numpy().astype(np.float32)
                else:
                    out["wb_latent_norm"] = np.zeros((bsz,), dtype=np.float32)
            elif model_name == "crn" and "h" in wb:
                latent = wb["h"]
                latent_norm = _latent_seq_norm(latent, mask)
                out["wb_latent_norm"] = latent_norm.detach().cpu().numpy().astype(np.float32)
            elif model_name == "vae" and "z_mu" in wb:
                latent = wb["z_mu"]
                latent_norm = _latent_seq_norm(latent, None)
                out["wb_latent_norm"] = latent_norm.detach().cpu().numpy().astype(np.float32)
            elif model_name == "timegan" and "h_hat" in wb:
                latent = wb["h_hat"]
                latent_norm = _latent_seq_norm(latent, mask)
                out["wb_latent_norm"] = latent_norm.detach().cpu().numpy().astype(np.float32)
            else:
                out["wb_latent_norm"] = np.zeros((bsz,), dtype=np.float32)

    for name in feature_names:
        out.setdefault(name, np.zeros((bsz,), dtype=np.float32))
    return out


def _select_head_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    head = None
    if hasattr(model, "y_head"):
        head = getattr(model, "y_head")
        if callable(head) and not isinstance(head, torch.nn.Module):
            head = head()
    if head is None and hasattr(model, "recovery"):
        head = getattr(model, "recovery")
    if head is None and hasattr(model, "out_proj"):
        head = getattr(model, "out_proj")
    if head is None and hasattr(model, "net"):
        head = getattr(model, "net")
    if isinstance(head, torch.nn.Module):
        return [p for p in head.parameters() if p.requires_grad]
    return [p for p in model.parameters() if p.requires_grad]


def _whitebox_grad_norm(
    *,
    gen_model: BaseSeqModel,
    batch: TrajectoryBatch,
    seed: int,
    max_samples: int = 256,
) -> float:
    model, model_name = _get_model_and_name(gen_model)
    if model is None:
        return float("nan")

    device = next(model.parameters()).device
    bsz = int(batch.X.shape[0])
    if bsz == 0:
        return float("nan")

    n = min(int(max_samples), bsz)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(bsz, size=n, replace=False)
    idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
    X = batch.X.to(device)[idx_t]
    A = batch.A.to(device)[idx_t]
    T = batch.T.to(device)[idx_t]
    Y = batch.Y.to(device)[idx_t]
    M = batch.mask.to(device)[idx_t]

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        if model_name == "diffusion":
            y0 = _ensure_3d_y(Y).float()
            timesteps = torch.randint(
                low=0,
                high=int(model.cfg.num_steps),  # type: ignore[attr-defined]
                size=(n,),
                device=device,
            )
            noise = torch.randn_like(y0)
            alpha_bar = model.alpha_bars[timesteps].view(n, 1, 1)  # type: ignore[attr-defined]
            y_noisy = torch.sqrt(alpha_bar) * y0 + torch.sqrt(1.0 - alpha_bar) * noise
            pred = model._predict_eps(  # type: ignore[attr-defined]
                x=X, a=A, t=T, y_noisy=y_noisy, timesteps=timesteps, mask=M
            )
            loss_t = (pred - noise).pow(2).mean(dim=-1)
            loss = _masked_mean_per_seq(loss_t, M).mean()
        else:
            sub_batch = TrajectoryBatch(
                X=X,
                A=A,
                T=T,
                Y=Y,
                mask=M,
                lengths=batch.lengths.to(device)[idx_t],
            )
            wb = _whitebox_forward(model=model, model_name=model_name, batch=sub_batch)
            y_logits = wb.get("y_logits")
            if y_logits is None:
                return float("nan")
            y_true = _ensure_3d_y(Y).float()
            loss = F.binary_cross_entropy_with_logits(y_logits, y_true, reduction="none")
            loss = _masked_mean_per_seq(loss, M).mean()

        loss.backward()
        total = 0.0
        for p in _select_head_params(model):
            if p.grad is None:
                continue
            total += float(p.grad.detach().pow(2).sum().item())
        model.zero_grad(set_to_none=True)
    return float(np.sqrt(total))


def _fit_attack_classifier(
    X_in: np.ndarray,
    X_out: np.ndarray,
    *,
    seed: int,
    class_weight: str | dict | None = "balanced",
) -> dict:
    if X_in.size == 0 or X_out.size == 0:
        return {"attack_auc": float("nan"), "attack_acc": float("nan"), "attack_balanced_acc": float("nan")}

    X = np.concatenate([X_in, X_out], axis=0)
    y = np.concatenate([np.ones(X_in.shape[0]), np.zeros(X_out.shape[0])], axis=0)
    if X.shape[0] < 2:
        return {"attack_auc": float("nan"), "attack_acc": float("nan"), "attack_balanced_acc": float("nan")}

    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(X.shape[0])
    split = int(0.7 * len(idx))
    if split <= 0:
        split = 1
    if split >= len(idx):
        split = len(idx) - 1
    train_idx = idx[:split]
    test_idx = idx[split:]

    try:
        clf = LogisticRegression(max_iter=200, class_weight=class_weight)
        clf.fit(X[train_idx], y[train_idx])
        prob = clf.predict_proba(X[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(y[test_idx], prob))
        except Exception:
            auc = float("nan")
        acc = float(accuracy_score(y[test_idx], pred))
        try:
            bal_acc = float(balanced_accuracy_score(y[test_idx], pred))
        except Exception:
            bal_acc = float("nan")
    except Exception:
        auc = float("nan")
        acc = float("nan")
        bal_acc = float("nan")

    return {"attack_auc": auc, "attack_acc": acc, "attack_balanced_acc": bal_acc}


def run_membership_inference(
    *,
    gen_model: BaseSeqModel,
    real_train: TrajectoryBatch,
    real_holdout: TrajectoryBatch,
    attack_features: List[str],
    embed_space: str = "y_only",
    seed: int = 0,
) -> dict:
    feats_in = []
    feats_out = []

    if "y_stats" in attack_features:
        feats_in.append(_y_stats(real_train))
        feats_out.append(_y_stats(real_holdout))
    if "recon_error" in attack_features:
        feats_in.append(_recon_error(gen_model, real_train))
        feats_out.append(_recon_error(gen_model, real_holdout))
    if "avg_confidence" in attack_features:
        feats_in.append(_avg_confidence(gen_model, real_train))
        feats_out.append(_avg_confidence(gen_model, real_holdout))
    if "nn_distance" in attack_features:
        feats_in.append(_nn_distance_feature(real_train, real_train, leave_one_out=True, embed_space=embed_space))
        feats_out.append(_nn_distance_feature(real_train, real_holdout, leave_one_out=False, embed_space=embed_space))
    wb_feature_names = [f for f in attack_features if f.startswith("wb_") and f != "wb_grad_norm"]
    if wb_feature_names:
        wb_in = _whitebox_per_sample_features(gen_model=gen_model, batch=real_train, feature_names=wb_feature_names, seed=int(seed))
        wb_out = _whitebox_per_sample_features(gen_model=gen_model, batch=real_holdout, feature_names=wb_feature_names, seed=int(seed))
        for name in wb_feature_names:
            feats_in.append(wb_in.get(name, np.zeros((real_train.X.shape[0],), dtype=np.float32)))
            feats_out.append(wb_out.get(name, np.zeros((real_holdout.X.shape[0],), dtype=np.float32)))
    if "wb_grad_norm" in attack_features:
        gn_in = _whitebox_grad_norm(gen_model=gen_model, batch=real_train, seed=int(seed))
        gn_out = _whitebox_grad_norm(gen_model=gen_model, batch=real_holdout, seed=int(seed) + 1)
        feats_in.append(np.full((real_train.X.shape[0],), float(gn_in), dtype=np.float32))
        feats_out.append(np.full((real_holdout.X.shape[0],), float(gn_out), dtype=np.float32))

    if not feats_in:
        return {
            "attack_auc": float("nan"),
            "attack_acc": float("nan"),
            "attack_balanced_acc": float("nan"),
            "features": attack_features,
        }

    X_in = _stack_features(feats_in)
    X_out = _stack_features(feats_out)
    result = _fit_attack_classifier(X_in, X_out, seed=int(seed))
    result["features"] = attack_features
    return result


def run_synth_membership_inference(
    *,
    real_train: TrajectoryBatch,
    real_holdout: TrajectoryBatch,
    synth: TrajectoryBatch,
    embed_space: str = "y_only",
    seed: int = 0,
    attack_features: List[str] | None = None,
) -> dict:
    feats_in = _nn_distance_feature(synth, real_train, leave_one_out=False, embed_space=embed_space)
    feats_out = _nn_distance_feature(synth, real_holdout, leave_one_out=False, embed_space=embed_space)

    X_in = feats_in.reshape(-1, 1)
    X_out = feats_out.reshape(-1, 1)
    result = _fit_attack_classifier(X_in, X_out, seed=int(seed))
    result["features"] = attack_features or ["synth_nn_distance"]
    return result


def mia_to_dataframe(result: dict, *, model: str, dataset: str, attack_features: List[str], n_in: int, n_out: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model,
                "dataset": dataset,
                "attack_features": ",".join(attack_features),
                "attack_auc": float(result.get("attack_auc", np.nan)),
                "attack_acc": float(result.get("attack_acc", np.nan)),
                "attack_balanced_acc": float(result.get("attack_balanced_acc", np.nan)),
                "n_in": int(n_in),
                "n_out": int(n_out),
            }
        ]
    )
