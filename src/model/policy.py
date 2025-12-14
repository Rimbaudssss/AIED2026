from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class Policy:
    """Policy interface: T_t = pi(h_t)."""

    def act(
        self,
        h_t: torch.Tensor,
        *,
        t_index: int,
        x: torch.Tensor,
        a_t: torch.Tensor,
        prev_y: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, *, num_actions: int):
        self.num_actions = int(num_actions)

    def act(
        self,
        h_t: torch.Tensor,
        *,
        t_index: int,
        x: torch.Tensor,
        a_t: torch.Tensor,
        prev_y: torch.Tensor,
    ) -> torch.Tensor:
        bsz = h_t.shape[0]
        return torch.randint(low=0, high=self.num_actions, size=(bsz,), device=h_t.device)


class ConstantPolicy(Policy):
    def __init__(self, action: int):
        self.action = int(action)

    def act(
        self,
        h_t: torch.Tensor,
        *,
        t_index: int,
        x: torch.Tensor,
        a_t: torch.Tensor,
        prev_y: torch.Tensor,
    ) -> torch.Tensor:
        return torch.full((h_t.shape[0],), self.action, device=h_t.device, dtype=torch.long)


class MLPPolicy(Policy, nn.Module):
    def __init__(self, d_h: int, num_actions: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.num_actions = int(num_actions)
        self.net = nn.Sequential(
            nn.Linear(d_h, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_actions),
        )

    def act(
        self,
        h_t: torch.Tensor,
        *,
        t_index: int,
        x: torch.Tensor,
        a_t: torch.Tensor,
        prev_y: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.net(h_t)
        return torch.distributions.Categorical(logits=logits).sample()


@dataclass(frozen=True)
class DoIntervention:
    """A minimal do(T_t=a) container.

    Use `mapping[t] = action_tensor_or_scalar`.
    """

    mapping: dict[int, torch.Tensor]

    @classmethod
    def single_step(cls, t_index: int, action: int | torch.Tensor) -> "DoIntervention":
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long)
        return cls(mapping={int(t_index): action})

    def as_dict(self, *, batch_size: int, device: torch.device) -> dict[int, torch.Tensor]:
        out: dict[int, torch.Tensor] = {}
        for t_index, action in self.mapping.items():
            if action.ndim == 0:
                out[int(t_index)] = action.to(device).expand(batch_size)
            else:
                out[int(t_index)] = action.to(device)
        return out


class HistoryEncoder(nn.Module):
    """Encodes h_t from factual histories (X, A_{1:t}, T_{1:t-1}, Y_{1:t-1}).

    This is a minimal GRU encoder used by the deconfounding regularizer.
    """

    def __init__(self, d_x: int, d_a: int, d_t: int, d_y: int, d_h: int = 64):
        super().__init__()
        self.d_h = int(d_h)
        self.inp = nn.Linear(d_x + d_a + d_t + d_y, d_h)
        self.rnn = nn.GRU(input_size=d_h, hidden_size=d_h, batch_first=True)

    def forward(
        self,
        *,
        x: torch.Tensor,  # [B, d_x]
        a: torch.Tensor,  # [B, T, d_a]
        t: torch.Tensor,  # [B, T, d_t]
        y: torch.Tensor,  # [B, T, d_y]
        mask: torch.Tensor | None = None,  # [B, T]
    ) -> torch.Tensor:
        bsz, seq_len = a.shape[0], a.shape[1]
        if mask is None:
            mask = torch.ones(bsz, seq_len, device=a.device)

        x_rep = x[:, None, :].expand(bsz, seq_len, x.shape[1])
        inp = torch.cat([x_rep, a, t, y], dim=-1)
        inp = self.inp(inp)

        lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(inp, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=seq_len)
        return out  # [B, T, d_h]


class TreatmentClassifier(nn.Module):
    def __init__(self, d_h: int, num_actions: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_h, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)
