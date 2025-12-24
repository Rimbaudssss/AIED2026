from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


class Policy:
    """Policy interface with explicit history inputs."""

    name: str = "policy"

    def act(
        self,
        *,
        X: torch.Tensor,
        A_hist: torch.Tensor,
        T_hist: torch.Tensor,
        Y_hist: torch.Tensor,
        t: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return action ids [B] for time index t."""
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantPolicy(Policy):
    action: int
    name: str = "constant"

    def act(
        self,
        *,
        X: torch.Tensor,
        A_hist: torch.Tensor,
        T_hist: torch.Tensor,
        Y_hist: torch.Tensor,
        t: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ = (X, A_hist, T_hist, Y_hist, t, mask)
        return torch.full((X.shape[0],), int(self.action), device=X.device, dtype=torch.long)


@dataclass(frozen=True)
class StagePolicy(Policy):
    t_switch: int
    a_before: int
    a_after: int
    name: str = "stage"

    def act(
        self,
        *,
        X: torch.Tensor,
        A_hist: torch.Tensor,
        T_hist: torch.Tensor,
        Y_hist: torch.Tensor,
        t: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ = (X, A_hist, T_hist, Y_hist, mask)
        action = self.a_before if int(t) < int(self.t_switch) else self.a_after
        return torch.full((X.shape[0],), int(action), device=X.device, dtype=torch.long)


@dataclass(frozen=True)
class HistoryThresholdPolicy(Policy):
    """Experimental policy depending on history (for ablation only)."""

    window: int
    threshold: float
    a_high: int
    a_low: int
    name: str = "history_threshold"

    def act(
        self,
        *,
        X: torch.Tensor,
        A_hist: torch.Tensor,
        T_hist: torch.Tensor,
        Y_hist: torch.Tensor,
        t: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ = (X, A_hist, T_hist)
        bsz = X.shape[0]
        if t <= 0:
            return torch.full((bsz,), int(self.a_low), device=X.device, dtype=torch.long)

        win = max(1, int(self.window))
        t_end = int(t)
        t_start = max(0, t_end - win)
        y_slice = Y_hist[:, t_start:t_end]
        if mask is None:
            denom = y_slice.shape[1]
            avg = y_slice.float().sum(dim=1) / max(1.0, float(denom))
        else:
            m_slice = mask[:, t_start:t_end].float()
            denom = m_slice.sum(dim=1).clamp(min=1.0)
            avg = (y_slice.float() * m_slice).sum(dim=1) / denom

        take_high = avg >= float(self.threshold)
        action = torch.where(take_high, torch.full_like(avg, float(self.a_high)), torch.full_like(avg, float(self.a_low)))
        return action.long()


@dataclass(frozen=True)
class PeriodicPolicy(Policy):
    period: int
    on_steps: int
    action_on: int
    action_off: int
    name: str = "periodic"

    def act(
        self,
        *,
        X: torch.Tensor,
        A_hist: torch.Tensor,
        T_hist: torch.Tensor,
        Y_hist: torch.Tensor,
        t: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ = (X, A_hist, T_hist, Y_hist, mask)
        period = max(1, int(self.period))
        on_steps = max(1, int(self.on_steps))
        phase = int(t) % period
        action = self.action_on if phase < on_steps else self.action_off
        return torch.full((X.shape[0],), int(action), device=X.device, dtype=torch.long)


def get_default_policies(T: int, action_space: list[int], *, policy_set: str = "fixed") -> list[Policy]:
    """Return fixed, interpretable policies only (no history dependence by default)."""
    actions = [int(a) for a in action_space] if action_space else [0, 1]
    actions = sorted(set(actions)) if actions else [0, 1]
    if len(actions) == 1:
        action_off = actions[0]
        action_on = actions[0]
    else:
        action_off = actions[0]
        action_on = actions[1]

    seq_len = max(1, int(T))
    if seq_len <= 1:
        early_switch = 0
        late_switch = 0
    else:
        early_switch = max(1, min(seq_len - 1, int(round(0.2 * seq_len))))
        late_switch = max(1, min(seq_len - 1, int(round(0.8 * seq_len))))

    policies: list[Policy] = [
        ConstantPolicy(action=action_off, name="never"),
        ConstantPolicy(action=action_on, name="always"),
        StagePolicy(t_switch=early_switch, a_before=action_on, a_after=action_off, name="early_on"),
        StagePolicy(t_switch=late_switch, a_before=action_off, a_after=action_on, name="late_on"),
    ]

    if policy_set == "ablation":
        period = max(2, int(round(0.2 * seq_len)))
        policies.append(
            PeriodicPolicy(period=period, on_steps=1, action_on=action_on, action_off=action_off, name="periodic")
        )
        policies.append(
            HistoryThresholdPolicy(window=5, threshold=0.5, a_high=action_on, a_low=action_off, name="history_threshold")
        )
    elif policy_set != "fixed":
        raise ValueError(f"Unknown policy_set={policy_set}")

    return policies


def default_policy_set() -> list[Policy]:
    """Deprecated alias for backward compatibility."""
    return get_default_policies(50, [0, 1], policy_set="fixed")
