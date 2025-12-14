from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean()
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp(min=1.0)
    return (x * mask).sum() / denom


def bernoulli_nll_from_logits(
    logits: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if target.ndim == 2:
        target = target[..., None]
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    return masked_mean(loss, mask)


def gaussian_mse(mean: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if target.ndim == 2:
        target = target[..., None]
    loss = F.mse_loss(mean, target.float(), reduction="none")
    return masked_mean(loss, mask)


def supervisor_mse(k_pred: torch.Tensor, k_true: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # k_pred/k_true: [B, T, d_k]
    loss = F.mse_loss(k_pred, k_true, reduction="none").mean(dim=-1)  # [B,T]
    return masked_mean(loss, mask)


def gradient_penalty(
    d_fn,
    real_seq: torch.Tensor,
    fake_seq: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """WGAN-GP gradient penalty on embedded sequences."""
    bsz = real_seq.shape[0]
    alpha = torch.rand(bsz, 1, 1, device=real_seq.device)
    interp = alpha * real_seq + (1.0 - alpha) * fake_seq
    interp = interp.requires_grad_(True)
    scores = d_fn(interp, mask=mask)
    grad = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=interp,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(bsz, -1)
    norm = grad.norm(2, dim=1)
    return ((norm - 1.0) ** 2).mean()


def wgan_gp_d_loss(
    d_fn,
    real_seq: torch.Tensor,
    fake_seq: torch.Tensor,
    *,
    gp_weight: float = 10.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    d_real = d_fn(real_seq, mask=mask)
    d_fake = d_fn(fake_seq, mask=mask)
    gp = gradient_penalty(d_fn, real_seq, fake_seq, mask=mask)
    return (d_fake.mean() - d_real.mean()) + gp_weight * gp


def wgan_g_loss(d_fn, fake_seq: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return -d_fn(fake_seq, mask=mask).mean()


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # x: [N,d], y: [M,d]
    dist2 = torch.cdist(x, y) ** 2
    return torch.exp(-dist2 / (2.0 * sigma**2 + 1e-12))


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, *, sigma: Optional[float] = None) -> torch.Tensor:
    """RBF MMD^2. Works with different sample sizes."""
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    x = x.float()
    y = y.float()

    if sigma is None:
        with torch.no_grad():
            z = torch.cat([x, y], dim=0)
            if z.shape[0] > 512:
                idx = torch.randperm(z.shape[0], device=z.device)[:512]
                z = z[idx]
            dists = torch.cdist(z, z).view(-1)
            sigma = torch.median(dists[dists > 0]).clamp(min=1e-3).item()
    sigma_t = torch.tensor(float(sigma), device=x.device)

    k_xx = _rbf_kernel(x, x, sigma_t)
    k_yy = _rbf_kernel(y, y, sigma_t)
    k_xy = _rbf_kernel(x, y, sigma_t)
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


def treatment_ce_loss(
    logits: torch.Tensor, t_true: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # logits: [B,T,A], t_true: [B,T]
    bsz, seq_len, num_actions = logits.shape
    loss = F.cross_entropy(logits.view(bsz * seq_len, num_actions), t_true.view(-1).long(), reduction="none")
    loss = loss.view(bsz, seq_len)
    return masked_mean(loss, mask)


def policy_moment_loss(y_real: torch.Tensor, y_gen: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Simple per-time mean matching (placeholder for policy simulation consistency)."""
    if y_real.ndim == 2:
        y_real = y_real[..., None]
    if y_gen.ndim == 2:
        y_gen = y_gen[..., None]
    diff = (y_real.mean(dim=0) - y_gen.mean(dim=0)).abs().mean(dim=-1)  # [T]
    return masked_mean(diff, mask.mean(dim=0) if mask is not None else None)

