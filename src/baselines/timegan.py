from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from baselines.timegan import TimeGAN, TimeGANConfig, TimeGANTrainConfig, fit_timegan
from src.baselines import BaseSeqModel
from src.data import TrajectoryBatch
from src.policy import Policy


@dataclass
class _TrainConfig:
    epochs_embed: int = 3
    epochs_supervisor: int = 3
    epochs_joint: int = 5
    hidden_dim: int = 64
    num_layers: int = 2
    noise_dim: int = 16
    x_emb_dim: int = 32
    lr: float = 1e-3
    lambda_sup: float = 10.0
    lambda_mom: float = 1.0
    max_batches_per_epoch: int = 200
    reservoir_size: int = 2048


class TimeGANModel(BaseSeqModel):
    name = "timegan"

    def __init__(self, *, hidden_dim: int = 64, num_layers: int = 2, noise_dim: int = 16, lr: float = 1e-3, **kwargs):
        super().__init__(name="timegan")
        self.cfg = _TrainConfig(hidden_dim=hidden_dim, num_layers=num_layers, noise_dim=noise_dim, lr=lr, **kwargs)
        self.model: Optional[TimeGAN] = None
        self.device = torch.device("cpu")

    def fit(self, train: TrajectoryBatch, valid: TrajectoryBatch, **kwargs) -> None:
        _ = (valid, kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        cfg = TimeGANConfig(
            d_x=int(train.X.shape[1]),
            seq_len=int(train.T.shape[1]),
            a_is_discrete=train.A.ndim == 2,
            a_vocab_size=int(train.A.max().item() + 1) if train.A.ndim == 2 else int(train.A.shape[2]),
            a_emb_dim=32,
            d_a=1 if train.A.ndim == 2 else int(train.A.shape[2]),
            t_is_discrete=True,
            t_vocab_size=int(train.T.max().item() + 1),
            t_emb_dim=16,
            d_t=1,
            d_y=1,
            x_emb_dim=int(self.cfg.x_emb_dim),
            hidden_dim=int(self.cfg.hidden_dim),
            num_layers=int(self.cfg.num_layers),
            z_dim=int(self.cfg.noise_dim),
            dropout=0.1,
        )
        self.model = TimeGAN(cfg).to(device)
        train_cfg = TimeGANTrainConfig(
            epochs_embed=int(self.cfg.epochs_embed),
            epochs_supervisor=int(self.cfg.epochs_supervisor),
            epochs_joint=int(self.cfg.epochs_joint),
            lr_embed=float(self.cfg.lr),
            lr_gen=float(self.cfg.lr),
            lr_disc=float(self.cfg.lr),
            lambda_sup=float(self.cfg.lambda_sup),
            lambda_mom=float(self.cfg.lambda_mom),
            max_batches_per_epoch=int(self.cfg.max_batches_per_epoch),
            reservoir_size=int(self.cfg.reservoir_size),
        )

        dataset = torch.utils.data.TensorDataset(train.X, train.A, train.T, train.Y, train.mask)

        def collate(batch):
            X, A, T, Y, M = zip(*batch)
            return {"X": torch.stack(X), "A": torch.stack(A), "T": torch.stack(T), "Y": torch.stack(Y), "mask": torch.stack(M)}

        dl = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True, collate_fn=collate)
        fit_timegan(self.model, dl, device=device, train_cfg=train_cfg)

    @torch.no_grad()
    def rollout(
        self,
        batch: TrajectoryBatch,
        *,
        do_t: Optional[dict] = None,
        policy: Optional[Policy] = None,
        horizon: Optional[int] = None,
        t0: int = 0,
        teacher_forcing: bool = False,
        return_logits: bool = False,
    ) -> dict:
        if self.model is None:
            raise RuntimeError("TimeGANModel must be fit before rollout.")
        _ = teacher_forcing
        X = batch.X.to(self.device)
        A = batch.A.to(self.device)
        T = batch.T.to(self.device)
        mask = batch.mask.to(self.device)
        seq_len = int(T.shape[1])
        end = seq_len if horizon is None else min(seq_len, int(t0) + int(horizon) + 1)
        end = max(0, min(seq_len, int(end)))
        t0 = max(0, min(seq_len, int(t0)))

        def _squeeze_y(y: torch.Tensor) -> torch.Tensor:
            if y.ndim == 3 and y.shape[-1] == 1:
                return y.squeeze(-1)
            return y

        def _prob_to_logit(y_prob: torch.Tensor) -> torch.Tensor:
            y_clip = torch.clamp(y_prob, 0.01, 0.99)
            return torch.log(y_clip / (1.0 - y_clip))

        def _resolve_do_value(t_index: int) -> Optional[torch.Tensor]:
            if do_t is None:
                return None
            if "all" in do_t:
                return do_t["all"]
            if "range" in do_t:
                t_start, t_end, action = do_t["range"]
                if int(t_start) <= int(t_index) <= int(t_end):
                    return action
                return None
            if t_index in do_t:
                return do_t[t_index]
            key = str(t_index)
            return do_t.get(key)

        def _to_action_tensor(value: object) -> torch.Tensor:
            if isinstance(value, torch.Tensor):
                act = value.to(self.device)
            else:
                act = torch.tensor(value, device=self.device, dtype=torch.long)
            if act.ndim == 0:
                act = act.expand(T.shape[0])
            return act.long()

        t_used = T.clone()
        if policy is None:
            if do_t is not None:
                if "all" in do_t:
                    t_used[:, :end] = int(do_t["all"])
                elif "range" in do_t:
                    t_start, t_end, action = do_t["range"]
                    t_start = max(0, int(t_start))
                    t_end = min(seq_len - 1, int(t_end))
                    t_used[:, t_start : t_end + 1] = int(action)
                else:
                    for k, v in do_t.items():
                        idx = int(k)
                        if 0 <= idx < end:
                            t_used[:, idx] = int(v)

            ro = self.model.rollout(x=X, a=A, t_obs=t_used, mask=mask, stochastic_y=False)
            y_prob = _squeeze_y(ro["y"]).float()
            out = {"Y_prob": y_prob, "T_used": t_used.long(), "mask": mask}
            if return_logits:
                y_logits = ro.get("y_logits", None)
                if y_logits is None:
                    y_logits = _prob_to_logit(y_prob)
                else:
                    y_logits = _squeeze_y(y_logits)
                out["Y_logit"] = y_logits.float()
            return out

        y_prob = torch.zeros((T.shape[0], seq_len), device=self.device, dtype=torch.float32)
        y_logit = torch.zeros_like(y_prob) if return_logits else None

        if t0 > 0:
            for t in range(t0):
                do_val = _resolve_do_value(t)
                if do_val is not None:
                    t_used[:, t] = _to_action_tensor(do_val)
            ro = self.model.rollout(x=X, a=A, t_obs=t_used, mask=mask, steps=t0, stochastic_y=False)
            y_prefix = _squeeze_y(ro["y"]).float()
            y_prob[:, :t0] = y_prefix[:, :t0]
            if return_logits and y_logit is not None:
                if "y_logits" in ro:
                    y_logit[:, :t0] = _squeeze_y(ro["y_logits"]).float()[:, :t0]
                else:
                    y_logit[:, :t0] = _prob_to_logit(y_prefix[:, :t0])

        for t in range(t0, end):
            do_val = _resolve_do_value(t)
            if do_val is not None:
                act = _to_action_tensor(do_val)
            else:
                act = policy.act(
                    X=X,
                    A_hist=A[:, :t],
                    T_hist=t_used[:, :t],
                    Y_hist=y_prob[:, :t],
                    t=t,
                    mask=mask[:, :t],
                )
                act = act.to(self.device).long()
            t_used[:, t] = act

            ro = self.model.rollout(x=X, a=A, t_obs=t_used, mask=mask, steps=t + 1, stochastic_y=False)
            y_prefix = _squeeze_y(ro["y"]).float()
            y_prob[:, t] = y_prefix[:, t]
            if return_logits and y_logit is not None:
                if "y_logits" in ro:
                    y_logit[:, t] = _squeeze_y(ro["y_logits"]).float()[:, t]
                else:
                    y_logit[:, t] = _prob_to_logit(y_prefix[:, t])

        if end < seq_len:
            ro = self.model.rollout(x=X, a=A, t_obs=t_used, mask=mask, steps=seq_len, stochastic_y=False)
            y_full = _squeeze_y(ro["y"]).float()
            y_prob[:, end:] = y_full[:, end:]
            if return_logits and y_logit is not None:
                if "y_logits" in ro:
                    y_logit[:, end:] = _squeeze_y(ro["y_logits"]).float()[:, end:]
                else:
                    y_logit[:, end:] = _prob_to_logit(y_full[:, end:])

        out = {"Y_prob": y_prob.float(), "T_used": t_used.long(), "mask": mask}
        if return_logits and y_logit is not None:
            out["Y_logit"] = y_logit.float()
        return out
