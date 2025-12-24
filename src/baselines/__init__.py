from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.data import TrajectoryBatch
from src.model.baselines import load_rollout_model_from_checkpoint
from src.policy import Policy


@dataclass
class BaseSeqModel:
    name: str

    def fit(self, train: TrajectoryBatch, valid: TrajectoryBatch, **kwargs) -> None:
        raise NotImplementedError

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
        raise NotImplementedError


class TorchSeqModel(BaseSeqModel):
    def __init__(self, *, name: str, model: torch.nn.Module):
        super().__init__(name=name)
        self.model = model

    def fit(self, train: TrajectoryBatch, valid: TrajectoryBatch, **kwargs) -> None:
        _ = (train, valid, kwargs)
        raise NotImplementedError("TorchSeqModel is a wrapper for pre-trained models.")

    def _apply_do(self, t_used: torch.Tensor, do_t: dict, *, t0: int, horizon: Optional[int]) -> torch.Tensor:
        seq_len = t_used.shape[1]
        end = seq_len if horizon is None else min(seq_len, int(t0) + int(horizon) + 1)

        if "all" in do_t:
            t_used[:, :end] = int(do_t["all"])
            return t_used
        if "range" in do_t:
            t_start, t_end, action = do_t["range"]
            t_start = max(0, int(t_start))
            t_end = min(seq_len - 1, int(t_end))
            t_used[:, t_start : t_end + 1] = int(action)
            return t_used

        for k, v in do_t.items():
            idx = int(k)
            if 0 <= idx < end:
                t_used[:, idx] = int(v)
        return t_used

    def _rollout_policy_closed_loop(
        self,
        batch: TrajectoryBatch,
        *,
        policy: Policy,
        do_t: Optional[dict],
        horizon: Optional[int],
        t0: int,
        return_logits: bool,
    ) -> dict:
        device = next(self.model.parameters()).device
        X = batch.X.to(device)
        A = batch.A.to(device)
        T = batch.T.to(device)
        mask = batch.mask.to(device)
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
                act = value.to(device)
            else:
                act = torch.tensor(value, device=device, dtype=torch.long)
            if act.ndim == 0:
                act = act.expand(T.shape[0])
            return act.long()

        t_used = T.clone()
        y_prob = torch.zeros((T.shape[0], seq_len), device=device, dtype=torch.float32)
        y_logit = torch.zeros_like(y_prob) if return_logits else None
        ro_last: dict | None = None

        if t0 > 0:
            for t in range(t0):
                do_val = _resolve_do_value(t)
                if do_val is not None:
                    t_used[:, t] = _to_action_tensor(do_val)
            ro = self.model.rollout(x=X, a=A, t_obs=t_used, mask=mask, steps=t0, stochastic_y=False)  # type: ignore[attr-defined]
            ro_last = ro
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
                act = act.to(device).long()
            t_used[:, t] = act

            ro = self.model.rollout(  # type: ignore[attr-defined]
                x=X,
                a=A,
                t_obs=t_used,
                mask=mask,
                steps=t + 1,
                stochastic_y=False,
            )
            ro_last = ro
            y_prefix = _squeeze_y(ro["y"]).float()
            y_prob[:, t] = y_prefix[:, t]
            if return_logits and y_logit is not None:
                if "y_logits" in ro:
                    y_logit[:, t] = _squeeze_y(ro["y_logits"]).float()[:, t]
                else:
                    y_logit[:, t] = _prob_to_logit(y_prefix[:, t])

        if end < seq_len:
            ro = self.model.rollout(  # type: ignore[attr-defined]
                x=X,
                a=A,
                t_obs=t_used,
                mask=mask,
                steps=seq_len,
                stochastic_y=False,
            )
            ro_last = ro
            y_full = _squeeze_y(ro["y"]).float()
            y_prob[:, end:] = y_full[:, end:]
            if return_logits and y_logit is not None:
                if "y_logits" in ro:
                    y_logit[:, end:] = _squeeze_y(ro["y_logits"]).float()[:, end:]
                else:
                    y_logit[:, end:] = _prob_to_logit(y_full[:, end:])

        out = {
            "Y_prob": y_prob.float(),
            "T_used": t_used.long(),
            "mask": mask,
        }
        if return_logits and y_logit is not None:
            out["Y_logit"] = y_logit.float()
        if ro_last is not None and "k" in ro_last:
            out["K"] = ro_last["k"]
        return out

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
        _ = teacher_forcing
        device = next(self.model.parameters()).device
        X = batch.X.to(device)
        A = batch.A.to(device)
        T = batch.T.to(device)
        mask = batch.mask.to(device)
        batch = TrajectoryBatch(
            X=X,
            A=A,
            T=T,
            Y=batch.Y.to(device),
            mask=mask,
            lengths=batch.lengths.to(device),
        )

        if policy is not None:
            if self.name == "diffusion":
                raise NotImplementedError("Diffusion does not support closed-loop policy rollouts.")
            return self._rollout_policy_closed_loop(
                batch,
                policy=policy,
                do_t=do_t,
                horizon=horizon,
                t0=t0,
                return_logits=return_logits,
            )

        t_used = T.clone()
        if do_t is not None:
            t_used = self._apply_do(t_used, do_t, t0=t0, horizon=horizon)

        ro = self.model.rollout(x=X, a=A, t_obs=t_used, mask=mask, stochastic_y=False)  # type: ignore[attr-defined]
        y_prob = ro["y"]
        if y_prob.ndim == 3 and y_prob.shape[-1] == 1:
            y_prob = y_prob.squeeze(-1)

        out = {
            "Y_prob": y_prob.float(),
            "T_used": t_used.long(),
            "mask": mask,
        }
        if return_logits:
            if "y_logits" in ro:
                y_logit = ro["y_logits"]
                if y_logit.ndim == 3 and y_logit.shape[-1] == 1:
                    y_logit = y_logit.squeeze(-1)
            else:
                y_clip = torch.clamp(y_prob, 0.01, 0.99)
                y_logit = torch.log(y_clip / (1.0 - y_clip))
            out["Y_logit"] = y_logit.float()
        if "k" in ro:
            out["K"] = ro["k"]
        return out


def wrap_torch_model(model: torch.nn.Module, *, name: str) -> BaseSeqModel:
    return TorchSeqModel(name=name, model=model)


def load_base_model_from_checkpoint(ckpt: dict, *, device: torch.device) -> BaseSeqModel:
    model, model_name = load_rollout_model_from_checkpoint(ckpt, device=device)
    return wrap_torch_model(model, name=model_name)
