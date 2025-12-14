from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SequenceDiscriminatorConfig:
    d_x: int
    # A
    a_is_discrete: bool = True
    a_vocab_size: int = 100
    a_emb_dim: int = 32
    d_a: int = 16
    # T
    t_is_discrete: bool = True
    t_vocab_size: int = 4
    t_emb_dim: int = 16
    d_t: int = 8
    # Y representation
    d_y: int = 1
    # Model
    d_h: int = 128
    num_layers: int = 1
    dropout: float = 0.1


class SequenceDiscriminator(nn.Module):
    """WGAN-style sequence discriminator D_seq."""

    def __init__(self, cfg: SequenceDiscriminatorConfig):
        super().__init__()
        self.cfg = cfg

        self.x_proj = nn.Linear(cfg.d_x, cfg.d_x)
        if cfg.a_is_discrete:
            self.a_emb = nn.Embedding(cfg.a_vocab_size, cfg.a_emb_dim)
            d_a = cfg.a_emb_dim
        else:
            self.a_emb = nn.Linear(cfg.d_a, cfg.d_a)
            d_a = cfg.d_a

        if cfg.t_is_discrete:
            self.t_emb = nn.Embedding(cfg.t_vocab_size, cfg.t_emb_dim)
            d_t = cfg.t_emb_dim
        else:
            self.t_emb = nn.Linear(cfg.d_t, cfg.d_t)
            d_t = cfg.d_t

        self.d_in = cfg.d_x + d_a + d_t + cfg.d_y
        self.rnn = nn.GRU(
            input_size=self.d_in,
            hidden_size=cfg.d_h,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(cfg.d_h, 1)

    def encode_inputs(
        self,
        *,
        x: torch.Tensor,  # [B, d_x]
        a: torch.Tensor,  # [B, T] or [B, T, d_a]
        t: torch.Tensor,  # [B, T] or [B, T, d_t]
        y: torch.Tensor,  # [B, T] or [B, T, d_y]
    ) -> torch.Tensor:
        bsz, seq_len = a.shape[0], a.shape[1]
        x_rep = self.x_proj(x.float())[:, None, :].expand(bsz, seq_len, self.cfg.d_x)

        if self.cfg.a_is_discrete:
            a_enc = self.a_emb(a.long())
        else:
            a_enc = self.a_emb(a.float())

        if self.cfg.t_is_discrete:
            t_enc = self.t_emb(t.long())
        else:
            t_enc = self.t_emb(t.float())

        if y.ndim == 2:
            y = y[..., None]
        y = y.float()

        return torch.cat([x_rep, a_enc, t_enc, y], dim=-1)

    def forward(self, seq: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scores an embedded sequence `seq` with shape [B, T, d_in]."""
        bsz, seq_len = seq.shape[0], seq.shape[1]
        if mask is None:
            mask = torch.ones(bsz, seq_len, device=seq.device)

        lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        h_last = h_n[-1]  # [B, d_h]
        return self.head(h_last).view(-1)  # [B]


class ConditionalDiscriminator(nn.Module):
    """Optional local discriminator D_cond on (h_t, T_t, Y_t)."""

    def __init__(self, d_h: int, d_t: int, d_y: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_h + d_t + d_y, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_t: torch.Tensor, t_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        if y_t.ndim == 1:
            y_t = y_t[:, None]
        if t_t.ndim == 1:
            t_t = t_t[:, None]
        return self.net(torch.cat([h_t, t_t.float(), y_t.float()], dim=-1)).view(-1)

