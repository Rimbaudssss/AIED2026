from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.causal_estimators.base import CausalEstimator, _bootstrap_ci
from src.data import TrajectoryBatch, TrajectoryDataset, compute_lengths, make_trajectory_dataloader
from src.policy import Policy


class _PropensityGRU(nn.Module):
    def __init__(self, d_in: int, d_h: int, num_actions: int, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_h, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, num_actions),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = compute_lengths(mask).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=x.shape[1])
        return self.out(out)


class _BaselinePropensity(nn.Module):
    def __init__(self, d_x: int, num_actions: int, seq_len: int, d_h: int = 64, dropout: float = 0.1):
        super().__init__()
        self.time_emb = nn.Embedding(int(seq_len), d_h)
        self.net = nn.Sequential(
            nn.Linear(int(d_x) + d_h, d_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, num_actions),
        )

    def forward(self, x: torch.Tensor, t_index: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t_index.long())
        return self.net(torch.cat([x.float(), t_emb], dim=-1))


@dataclass
class _FitConfig:
    propensity_model: str = "mlp"
    stabilized: bool = True
    clip: float = 20.0
    lr: float = 1e-3
    epochs: int = 5
    hidden: int = 64
    dropout: float = 0.1
    batch_size: int = 128


class IPTWMSM(CausalEstimator):
    name = "iptw_msm"

    def __init__(self, *, propensity_model: str = "mlp", stabilized: bool = True, clip: float = 20.0, **kwargs):
        self.cfg = _FitConfig(propensity_model=propensity_model, stabilized=stabilized, clip=clip, **kwargs)
        self.propensity: Optional[_PropensityGRU] = None
        self.baseline_propensity: Optional[_BaselinePropensity] = None
        self.num_actions: Optional[int] = None
        self.seq_len: Optional[int] = None
        self._device = torch.device("cpu")

    def _infer_dims(self, batch: TrajectoryBatch) -> tuple[int, int, int]:
        seq_len = int(batch.T.shape[1])
        num_actions = int(batch.T.max().item() + 1)
        d_x = int(batch.X.shape[1])
        return d_x, seq_len, num_actions

    def _encode_inputs(self, batch: TrajectoryBatch) -> torch.Tensor:
        bsz, seq_len = batch.T.shape
        x_rep = batch.X.float()[:, None, :].expand(bsz, seq_len, batch.X.shape[1])

        if batch.A.ndim == 2:
            a_vocab = int(batch.A.max().item() + 1)
            a_onehot = torch.zeros(bsz, seq_len, a_vocab, device=batch.A.device)
            a_onehot.scatter_(2, batch.A.long().unsqueeze(-1), 1.0)
            a_feat = a_onehot
        else:
            a_feat = batch.A.float()

        t_prev = torch.zeros_like(batch.T)
        t_prev[:, 1:] = batch.T[:, :-1]
        t_vocab = int(batch.T.max().item() + 1)
        t_onehot = torch.zeros(bsz, seq_len, t_vocab, device=batch.T.device)
        t_onehot.scatter_(2, t_prev.long().unsqueeze(-1), 1.0)

        y_prev = torch.zeros_like(batch.Y)
        y_prev[:, 1:] = batch.Y[:, :-1]
        y_prev = y_prev.float().unsqueeze(-1)

        return torch.cat([x_rep, a_feat, t_onehot, y_prev], dim=-1)

    def fit(self, train: TrajectoryBatch, valid: Optional[TrajectoryBatch] = None, **kwargs) -> None:
        _ = valid
        cfg = self.cfg
        d_x, seq_len, num_actions = self._infer_dims(train)
        self.num_actions = num_actions
        self.seq_len = seq_len

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        inp = self._encode_inputs(train.to(device))
        d_in = int(inp.shape[-1])

        self.propensity = _PropensityGRU(d_in=d_in, d_h=cfg.hidden, num_actions=num_actions, dropout=cfg.dropout).to(device)
        self.baseline_propensity = _BaselinePropensity(
            d_x=d_x, num_actions=num_actions, seq_len=seq_len, d_h=cfg.hidden, dropout=cfg.dropout
        ).to(device)

        opt = torch.optim.Adam(self.propensity.parameters(), lr=cfg.lr)
        opt_base = torch.optim.Adam(self.baseline_propensity.parameters(), lr=cfg.lr)

        dataset = train
        dl = make_trajectory_dataloader(
            TrajectoryDataset(
                X=dataset.X.detach().cpu(),
                A=dataset.A.detach().cpu(),
                T=dataset.T.detach().cpu(),
                Y=dataset.Y.detach().cpu(),
                mask=dataset.mask.detach().cpu(),
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )

        for _ep in range(int(cfg.epochs)):
            for batch in dl:
                batch = batch.to(device)
                logits = self.propensity(self._encode_inputs(batch), batch.mask)
                bsz, seq_len = batch.T.shape
                loss = nn.CrossEntropyLoss(reduction="none")(
                    logits.view(bsz * seq_len, num_actions), batch.T.view(-1).long()
                )
                loss = loss.view(bsz, seq_len)
                loss = (loss * batch.mask).sum() / batch.mask.sum().clamp(min=1.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                t_idx = torch.arange(seq_len, device=device).view(1, seq_len).expand(bsz, seq_len)
                base_logits = self.baseline_propensity(batch.X, t_idx.reshape(-1)).view(bsz, seq_len, num_actions)
                loss_base = nn.CrossEntropyLoss(reduction="none")(
                    base_logits.view(bsz * seq_len, num_actions), batch.T.view(-1).long()
                )
                loss_base = loss_base.view(bsz, seq_len)
                loss_base = (loss_base * batch.mask).sum() / batch.mask.sum().clamp(min=1.0)
                opt_base.zero_grad(set_to_none=True)
                loss_base.backward()
                opt_base.step()

    def _compute_log_probs(self, data: TrajectoryBatch) -> tuple[np.ndarray, np.ndarray]:
        if self.propensity is None or self.baseline_propensity is None:
            raise RuntimeError("IPTWMSM must be fit before calling estimate.")
        device = self._device
        batch = data.to(device)
        with torch.no_grad():
            logits = self.propensity(self._encode_inputs(batch), batch.mask)
            logp_den = torch.log_softmax(logits, dim=-1)
            if self.cfg.stabilized:
                t_idx = torch.arange(batch.T.shape[1], device=device).view(1, -1).expand(batch.T.shape[0], -1)
                base_logits = self.baseline_propensity(batch.X, t_idx.reshape(-1)).view(
                    batch.T.shape[0], batch.T.shape[1], -1
                )
                logp_num = torch.log_softmax(base_logits, dim=-1)
            else:
                logp_num = torch.zeros_like(logp_den)

            t_obs = batch.T.long().unsqueeze(-1)
            logp_den_obs = torch.gather(logp_den, dim=-1, index=t_obs).squeeze(-1)
            logp_num_obs = torch.gather(logp_num, dim=-1, index=t_obs).squeeze(-1)
        return logp_num_obs.cpu().numpy(), logp_den_obs.cpu().numpy()

    def _weights_up_to(self, logp_num: np.ndarray, logp_den: np.ndarray, mask: np.ndarray, t_end: int) -> np.ndarray:
        t_end = int(t_end)
        seq_len = logp_num.shape[1]
        t_end = min(t_end, seq_len - 1)
        valid = mask[:, : t_end + 1] > 0.5
        log_ratio = (logp_num[:, : t_end + 1] - logp_den[:, : t_end + 1]) * valid
        log_w = log_ratio.sum(axis=1)
        w = np.exp(log_w)
        if self.cfg.clip is not None:
            w = np.clip(w, 0.0, float(self.cfg.clip))
        return w

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
        t0 = int(t0)
        horizon = int(horizon)
        action = int(action)
        logp_num, logp_den = self._compute_log_probs(data)
        mask = data.mask.detach().cpu().numpy()
        T = data.T.detach().cpu().numpy()
        Y = data.Y.detach().cpu().numpy()

        subgroup_name = "all"
        if subgroup is not None and "name" in subgroup:
            subgroup_name = str(subgroup["name"])
        if subgroup is not None and "mask" in subgroup:
            sel = np.asarray(subgroup["mask"]).astype(bool)
        else:
            sel = np.ones(T.shape[0], dtype=bool)

        rng = np.random.default_rng(int(seed))
        mu_samples = []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, T.shape[0], size=T.shape[0])
            idx = idx[sel[idx]]
            if idx.size == 0:
                mu_samples.append(np.full(horizon + 1, np.nan, dtype=np.float64))
                continue
            mu = np.zeros(horizon + 1, dtype=np.float64)
            for h in range(horizon + 1):
                t_end = t0 + h
                w = self._weights_up_to(logp_num[idx], logp_den[idx], mask[idx], t_end)
                arm = T[idx, t0] == action
                if not np.any(arm):
                    mu[h] = np.nan
                else:
                    y = Y[idx, t0 + h]
                    w_sel = w[arm]
                    y_sel = y[arm]
                    mu[h] = float(np.sum(w_sel * y_sel) / max(1e-6, np.sum(w_sel)))
            mu_samples.append(mu)
        mu_samples = np.stack(mu_samples, axis=0)
        mu = np.nanmean(mu_samples, axis=0)
        ci_low, ci_high = _bootstrap_ci(mu_samples, alpha=0.05)
        n_eff = int(np.sum(sel))

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
        logp_num, logp_den = self._compute_log_probs(data)
        mask = data.mask.detach().cpu().numpy()
        T = data.T.detach().cpu().numpy()
        Y = data.Y.detach().cpu().numpy()
        X = data.X.detach().cpu()
        A = data.A.detach().cpu()
        seq_len = T.shape[1]
        if horizon is None:
            horizon = seq_len - 1
        horizon = int(horizon)
        rng = np.random.default_rng(int(seed))

        def policy_actions_np() -> np.ndarray:
            actions = np.zeros_like(T)
            for t in range(seq_len):
                a = policy.act(
                    X=X,
                    A_hist=A[:, :t],
                    T_hist=torch.as_tensor(actions[:, :t]),
                    Y_hist=torch.as_tensor(Y[:, :t]),
                    t=t,
                    mask=torch.as_tensor(mask[:, :t]),
                )
                actions[:, t] = a.detach().cpu().numpy().astype(np.int64)
            return actions

        pi_actions = policy_actions_np()

        values = []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, T.shape[0], size=T.shape[0])
            w = self._weights_up_to(logp_num[idx], logp_den[idx], mask[idx], horizon)
            match = np.all(T[idx, : horizon + 1] == pi_actions[idx, : horizon + 1], axis=1)
            if not np.any(match):
                values.append(np.nan)
                continue
            y_slice = Y[idx, : horizon + 1]
            m_slice = mask[idx, : horizon + 1]
            y_mean = (y_slice * m_slice).sum(axis=1) / np.maximum(1.0, m_slice.sum(axis=1))
            w_sel = w[match]
            y_sel = y_mean[match]
            values.append(float(np.sum(w_sel * y_sel) / max(1e-6, np.sum(w_sel))))

        values = np.asarray(values, dtype=np.float64)
        ci_low, ci_high = _bootstrap_ci(values.reshape(-1, 1), alpha=0.05)
        return {
            "value": float(np.nanmean(values)),
            "ci_low": float(ci_low.reshape(-1)[0]),
            "ci_high": float(ci_high.reshape(-1)[0]),
            "n": int(values.shape[0]),
            "policy_name": getattr(policy, "name", "policy"),
        }
