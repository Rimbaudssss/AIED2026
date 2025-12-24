from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.causal_estimators.base import CausalEstimator, _bootstrap_ci
from src.data import TrajectoryBatch, compute_lengths
from src.policy import Policy


class _OutcomeRNN(nn.Module):
    def __init__(self, d_in: int, d_h: int, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(d_in, d_h)
        self.cell = nn.GRUCell(d_h, d_h)
        self.out = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, 1),
        )

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.relu(self.inp(x))
        h_next = self.cell(z, h)
        y_logit = self.out(h_next)
        return h_next, y_logit


@dataclass
class _FitConfig:
    outcome_model: str = "mlp"
    state_update: str = "autoregressive"
    lr: float = 1e-3
    epochs: int = 5
    hidden: int = 64
    dropout: float = 0.1


class GFormula(CausalEstimator):
    name = "gformula"

    def __init__(self, *, outcome_model: str = "mlp", state_update: str = "autoregressive", **kwargs):
        self.cfg = _FitConfig(outcome_model=outcome_model, state_update=state_update, **kwargs)
        self.model: Optional[_OutcomeRNN] = None
        self.num_actions: Optional[int] = None
        self.seq_len: Optional[int] = None
        self.a_vocab: Optional[int] = None
        self.a_is_discrete: Optional[bool] = None
        self._device = torch.device("cpu")

    def _infer_dims(self, batch: TrajectoryBatch) -> tuple[int, int, int]:
        seq_len = int(batch.T.shape[1])
        num_actions = int(batch.T.max().item() + 1)
        d_x = int(batch.X.shape[1])
        return d_x, seq_len, num_actions

    def _encode_a(self, A: torch.Tensor) -> torch.Tensor:
        a_is_discrete = self.a_is_discrete
        if a_is_discrete is None:
            a_is_discrete = A.ndim == 2
        if a_is_discrete:
            a_vocab = self.a_vocab if self.a_vocab is not None else int(A.max().item() + 1)
            a_onehot = torch.zeros(A.shape[0], A.shape[1], a_vocab, device=A.device)
            a_onehot.scatter_(2, A.long().unsqueeze(-1), 1.0)
            return a_onehot
        return A.float()

    def _encode_t(self, T: torch.Tensor, num_actions: int) -> torch.Tensor:
        t_onehot = torch.zeros(T.shape[0], T.shape[1], num_actions, device=T.device)
        t_onehot.scatter_(2, T.long().unsqueeze(-1), 1.0)
        return t_onehot

    def fit(self, train: TrajectoryBatch, valid: Optional[TrajectoryBatch] = None, **kwargs) -> None:
        _ = (valid, kwargs)
        cfg = self.cfg
        d_x, seq_len, num_actions = self._infer_dims(train)
        self.num_actions = num_actions
        self.seq_len = seq_len
        self.a_is_discrete = train.A.ndim == 2
        if self.a_is_discrete:
            self.a_vocab = int(train.A.max().item() + 1)
        else:
            self.a_vocab = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        A = train.A.to(device)
        T = train.T.to(device)
        X = train.X.to(device)
        Y = train.Y.to(device)
        M = train.mask.to(device)

        a_feat = self._encode_a(A)
        t_feat = self._encode_t(T, num_actions=num_actions)
        x_rep = X.float()[:, None, :].expand(X.shape[0], seq_len, X.shape[1])

        y_prev = torch.zeros_like(Y)
        y_prev[:, 1:] = Y[:, :-1]
        y_prev = y_prev.float().unsqueeze(-1)

        inp = torch.cat([x_rep, a_feat, t_feat, y_prev], dim=-1)
        d_in = int(inp.shape[-1])

        self.model = _OutcomeRNN(d_in=d_in, d_h=cfg.hidden, dropout=cfg.dropout).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        for _ep in range(int(cfg.epochs)):
            h = torch.zeros(X.shape[0], cfg.hidden, device=device)
            total_loss = torch.tensor(0.0, device=device)
            for t in range(seq_len):
                h, y_logit = self.model.step(inp[:, t, :], h)
                y_logit = y_logit.view(-1)
                y_true = Y[:, t].float()
                m = M[:, t].float()
                loss = nn.functional.binary_cross_entropy_with_logits(y_logit, y_true, reduction="none")
                loss = (loss * m).sum() / m.sum().clamp(min=1.0)
                total_loss = total_loss + loss
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            opt.step()

    def _predict_do_curve(
        self,
        data: TrajectoryBatch,
        *,
        t0: int,
        horizon: int,
        action: int,
    ) -> np.ndarray:
        if self.model is None or self.num_actions is None:
            raise RuntimeError("GFormula must be fit before calling estimate.")
        device = self._device
        batch = data.to(device)
        X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask
        seq_len = T.shape[1]
        t0 = int(t0)
        horizon = int(horizon)

        a_feat = self._encode_a(A)
        x_rep = X.float()[:, None, :].expand(X.shape[0], seq_len, X.shape[1])

        h = torch.zeros(X.shape[0], self.model.cell.hidden_size, device=device)
        y_prev = torch.zeros(X.shape[0], 1, device=device)
        y_pred = []

        for t in range(seq_len):
            if t == t0:
                t_use = torch.full_like(T[:, t], int(action))
            else:
                t_use = T[:, t]
            t_feat = self._encode_t(t_use.unsqueeze(1), num_actions=self.num_actions).squeeze(1)
            inp = torch.cat([x_rep[:, t, :], a_feat[:, t, :], t_feat, y_prev], dim=-1)
            h, y_logit = self.model.step(inp, h)
            y_prob = torch.sigmoid(y_logit.view(-1))
            if t < t0:
                y_prev = Y[:, t].float().view(-1, 1)
            else:
                # Roll history forward with predicted outcomes after intervention.
                y_prev = y_prob.view(-1, 1)
            y_pred.append(y_prob)

        y_pred = torch.stack(y_pred, dim=1)  # [B,T]
        t_end = min(seq_len - 1, t0 + horizon)
        return y_pred[:, t0 : t_end + 1].detach().cpu().numpy()

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
        preds = self._predict_do_curve(data, t0=t0, horizon=horizon, action=action)
        subgroup_name = "all"
        if subgroup is not None and "name" in subgroup:
            subgroup_name = str(subgroup["name"])
        if subgroup is not None and "mask" in subgroup:
            sel = np.asarray(subgroup["mask"]).astype(bool)
            preds = preds[sel]

        if preds.size == 0:
            mu = np.full(horizon + 1, np.nan, dtype=np.float64)
            ci_low = mu.copy()
            ci_high = mu.copy()
            n_eff = 0
        else:
            rng = np.random.default_rng(int(seed))
            samples = []
            for _ in range(int(n_boot)):
                idx = rng.integers(0, preds.shape[0], size=preds.shape[0])
                samples.append(np.mean(preds[idx], axis=0))
            samples = np.stack(samples, axis=0)
            mu = np.mean(samples, axis=0)
            ci_low, ci_high = _bootstrap_ci(samples, alpha=0.05)
            n_eff = int(preds.shape[0])

        return {
            "mu": mu.astype(np.float64),
            "ci_low": ci_low.astype(np.float64),
            "ci_high": ci_high.astype(np.float64),
            "n": n_eff,
            "subgroup_key": subgroup_name,
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
        if self.model is None or self.num_actions is None:
            raise RuntimeError("GFormula must be fit before calling estimate.")
        device = self._device
        batch = data.to(device)
        X, A, Y, M = batch.X, batch.A, batch.Y, batch.mask
        seq_len = batch.T.shape[1]
        if horizon is None:
            horizon = seq_len - 1
        horizon = int(horizon)

        a_feat = self._encode_a(A)
        x_rep = X.float()[:, None, :].expand(X.shape[0], seq_len, X.shape[1])

        h = torch.zeros(X.shape[0], self.model.cell.hidden_size, device=device)
        y_prev = torch.zeros(X.shape[0], 1, device=device)
        T_hist = torch.zeros_like(batch.T)
        y_pred = []

        for t in range(seq_len):
            t_action = policy.act(
                X=X,
                A_hist=A[:, :t],
                T_hist=T_hist[:, :t],
                Y_hist=torch.stack(y_pred, dim=1) if y_pred else torch.zeros_like(batch.Y[:, :0]),
                t=t,
                mask=M[:, :t],
            )
            T_hist[:, t] = t_action
            t_feat = self._encode_t(t_action.unsqueeze(1), num_actions=self.num_actions).squeeze(1)
            inp = torch.cat([x_rep[:, t, :], a_feat[:, t, :], t_feat, y_prev], dim=-1)
            h, y_logit = self.model.step(inp, h)
            y_prob = torch.sigmoid(y_logit.view(-1))
            # Policy evaluation uses predicted outcomes to update history.
            y_prev = y_prob.view(-1, 1)
            y_pred.append(y_prob)

        y_pred = torch.stack(y_pred, dim=1)[:, : horizon + 1]
        m_slice = M[:, : horizon + 1].float()
        per_seq = (y_pred * m_slice).sum(dim=1) / m_slice.sum(dim=1).clamp(min=1.0)
        preds = per_seq.detach().cpu().numpy()

        rng = np.random.default_rng(int(seed))
        samples = []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, preds.shape[0], size=preds.shape[0])
            samples.append(np.mean(preds[idx]))
        samples = np.asarray(samples, dtype=np.float64)
        ci_low, ci_high = _bootstrap_ci(samples.reshape(-1, 1), alpha=0.05)
        return {
            "value": float(np.mean(samples)),
            "ci_low": float(ci_low.reshape(-1)[0]),
            "ci_high": float(ci_high.reshape(-1)[0]),
            "n": int(preds.shape[0]),
            "policy_name": getattr(policy, "name", "policy"),
        }
