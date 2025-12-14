from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .discriminators import SequenceDiscriminator, SequenceDiscriminatorConfig
from .losses import bernoulli_nll_from_logits, grad_reverse, masked_mean
from .policy import Policy
from .scm_generator import SCMGenerator, SCMGeneratorConfig


def _ensure_3d_y(y: torch.Tensor) -> torch.Tensor:
    return y if y.ndim == 3 else y.unsqueeze(-1)


@dataclass(frozen=True)
class RCGANConfig:
    d_x: int
    d_eps: int = 16
    a_is_discrete: bool = True
    a_vocab_size: int = 100
    a_emb_dim: int = 32
    d_a: int = 16
    t_is_discrete: bool = True
    t_vocab_size: int = 4
    t_emb_dim: int = 16
    d_t: int = 8
    rnn: str = "gru"  # gru | lstm
    d_h: int = 128
    dropout: float = 0.1
    use_prev_y: bool = True


class RCGANGenerator(nn.Module):
    """Recurrent Conditional GAN generator (black-box RNN)."""

    def __init__(self, cfg: RCGANConfig):
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

        self.d_in = d_a + d_t + cfg.d_x + cfg.d_eps + (1 if cfg.use_prev_y else 0)
        if cfg.rnn == "gru":
            self.rnn_cell = nn.GRUCell(self.d_in, cfg.d_h)
        elif cfg.rnn == "lstm":
            self.rnn_cell = nn.LSTMCell(self.d_in, cfg.d_h)
        else:
            raise ValueError(f"Unknown rnn={cfg.rnn}")

        self.h0 = nn.Sequential(nn.Linear(cfg.d_x + cfg.d_eps, cfg.d_h), nn.Tanh())
        self.y_head = nn.Linear(cfg.d_h, 1)

    def encode_a(self, a_t: torch.Tensor) -> torch.Tensor:
        if self.cfg.a_is_discrete:
            return self.a_emb(a_t.long())
        return self.a_emb(a_t.float())

    def encode_t(self, t_t: torch.Tensor) -> torch.Tensor:
        if self.cfg.t_is_discrete:
            return self.t_emb(t_t.long())
        return self.t_emb(t_t.float())

    def _init_state(self, x: torch.Tensor, eps0: Optional[torch.Tensor] = None):
        if eps0 is None:
            eps0 = torch.randn(x.shape[0], self.cfg.d_eps, device=x.device)
        h0 = self.h0(torch.cat([self.x_proj(x.float()), eps0], dim=-1))
        if self.cfg.rnn == "gru":
            return h0
        return (h0, torch.zeros_like(h0))

    def teacher_forcing(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,  # [B,T,d_eps]
        stochastic_y: bool = False,
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = a.shape[0], a.shape[1]
        if mask is None:
            mask = torch.ones(bsz, seq_len, device=x.device)
        if eps is None:
            eps = torch.randn(bsz, seq_len, self.cfg.d_eps, device=x.device)
        if eps.ndim != 3 or eps.shape[0] != bsz or eps.shape[1] != seq_len or eps.shape[2] != self.cfg.d_eps:
            raise ValueError(f"eps must be [B,T,d_eps]=[{bsz},{seq_len},{self.cfg.d_eps}], got {tuple(eps.shape)}")

        state = self._init_state(x, eps0=eps[:, 0, :])
        prev_y = torch.zeros(bsz, 1, device=x.device)

        y_logits_seq = []
        y_out_seq = []
        for ti in range(seq_len):
            a_enc = self.encode_a(a[:, ti])
            t_enc = self.encode_t(t[:, ti])
            x_enc = self.x_proj(x.float())
            parts = [a_enc, t_enc, x_enc, eps[:, ti, :]]
            if self.cfg.use_prev_y:
                parts.append(prev_y)
            inp = torch.cat(parts, dim=-1)

            if self.cfg.rnn == "gru":
                state = self.rnn_cell(inp, state)
                h_t = state
            else:
                h, c = state
                h, c = self.rnn_cell(inp, (h, c))
                state = (h, c)
                h_t = h

            y_logits = self.y_head(h_t)
            y_prob = torch.sigmoid(y_logits)
            y_out = torch.bernoulli(y_prob) if stochastic_y else y_prob

            m = mask[:, ti].view(bsz, 1)
            y_logits = m * y_logits + (1.0 - m) * y_logits.detach()
            y_out = m * y_out + (1.0 - m) * y_out.detach()

            y_logits_seq.append(y_logits)
            y_out_seq.append(y_out)

            if self.cfg.use_prev_y:
                if y is not None:
                    prev_y = _ensure_3d_y(y)[:, ti, :].detach()
                else:
                    prev_y = y_out.detach()

        return {"y_logits": torch.stack(y_logits_seq, dim=1), "y": torch.stack(y_out_seq, dim=1)}

    def rollout(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t_obs: Optional[torch.Tensor] = None,
        do_t: Optional[Dict[int, torch.Tensor]] = None,
        policy: Optional[Policy] = None,
        mask: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,  # [B,T,d_eps]
        steps: Optional[int] = None,
        stochastic_y: bool = False,
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = a.shape[0], a.shape[1]
        if steps is None:
            steps = seq_len
        if mask is None:
            mask = torch.ones(bsz, steps, device=x.device)
        if eps is None:
            eps = torch.randn(bsz, steps, self.cfg.d_eps, device=x.device)
        if eps.ndim != 3 or eps.shape[0] != bsz or eps.shape[1] < steps or eps.shape[2] != self.cfg.d_eps:
            raise ValueError(f"eps must be [B,>=T,d_eps]=[{bsz},>={steps},{self.cfg.d_eps}], got {tuple(eps.shape)}")

        state = self._init_state(x, eps0=eps[:, 0, :])
        prev_y = torch.zeros(bsz, 1, device=x.device)

        y_logits_seq = []
        y_out_seq = []
        t_used_seq = []
        for ti in range(steps):
            if do_t is not None and ti in do_t:
                t_t = do_t[ti]
            elif policy is not None:
                h_for_policy = state if isinstance(state, torch.Tensor) else state[0]
                t_t = policy.act(h_for_policy, t_index=ti, x=x, a_t=a[:, ti], prev_y=prev_y)
            elif t_obs is not None:
                t_t = t_obs[:, ti]
            else:
                raise ValueError("One of do_t, policy, t_obs must be provided.")

            a_enc = self.encode_a(a[:, ti])
            t_enc = self.encode_t(t_t)
            x_enc = self.x_proj(x.float())
            parts = [a_enc, t_enc, x_enc, eps[:, ti, :]]
            if self.cfg.use_prev_y:
                parts.append(prev_y)
            inp = torch.cat(parts, dim=-1)

            if self.cfg.rnn == "gru":
                state = self.rnn_cell(inp, state)
                h_t = state
            else:
                h, c = state
                h, c = self.rnn_cell(inp, (h, c))
                state = (h, c)
                h_t = h

            y_logits = self.y_head(h_t)
            y_prob = torch.sigmoid(y_logits)
            y_out = torch.bernoulli(y_prob) if stochastic_y else y_prob

            if self.cfg.use_prev_y:
                prev_y = y_out.detach()

            m = mask[:, ti].view(bsz, 1)
            y_logits = m * y_logits + (1.0 - m) * y_logits.detach()
            y_out = m * y_out + (1.0 - m) * y_out.detach()

            y_logits_seq.append(y_logits)
            y_out_seq.append(y_out)
            t_used_seq.append(t_t)

        return {
            "y_logits": torch.stack(y_logits_seq, dim=1),
            "y": torch.stack(y_out_seq, dim=1),
            "t": torch.stack(t_used_seq, dim=1),
        }


@dataclass(frozen=True)
class SeqVAEConfig:
    d_x: int
    z_dim: int = 32
    enc_hidden: int = 128
    dec_hidden: int = 128
    dropout: float = 0.1
    use_prev_y: bool = True
    a_is_discrete: bool = True
    a_vocab_size: int = 100
    a_emb_dim: int = 32
    d_a: int = 16
    t_is_discrete: bool = True
    t_vocab_size: int = 4
    t_emb_dim: int = 16
    d_t: int = 8


class SeqVAE(nn.Module):
    def __init__(self, cfg: SeqVAEConfig):
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

        self.enc_in = nn.Linear(cfg.d_x + d_a + d_t + 1, cfg.enc_hidden)
        self.encoder = nn.GRU(
            input_size=cfg.enc_hidden,
            hidden_size=cfg.enc_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.z_mu = nn.Linear(2 * cfg.enc_hidden, cfg.z_dim)
        self.z_logvar = nn.Linear(2 * cfg.enc_hidden, cfg.z_dim)

        self.dec_h0 = nn.Sequential(nn.Linear(cfg.d_x + cfg.z_dim, cfg.dec_hidden), nn.Tanh())
        self.dec_cell = nn.GRUCell(
            input_size=cfg.d_x + d_a + d_t + cfg.z_dim + (1 if cfg.use_prev_y else 0),
            hidden_size=cfg.dec_hidden,
        )
        self.y_head = nn.Linear(cfg.dec_hidden, 1)

    def encode_a(self, a_t: torch.Tensor) -> torch.Tensor:
        if self.cfg.a_is_discrete:
            return self.a_emb(a_t.long())
        return self.a_emb(a_t.float())

    def encode_t(self, t_t: torch.Tensor) -> torch.Tensor:
        if self.cfg.t_is_discrete:
            return self.t_emb(t_t.long())
        return self.t_emb(t_t.float())

    def encode(self, *, x: torch.Tensor, a: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        bsz, seq_len = a.shape[0], a.shape[1]
        x_rep = self.x_proj(x.float())[:, None, :].expand(bsz, seq_len, x.shape[1])
        a_enc = self.encode_a(a.view(-1, *a.shape[2:])).view(bsz, seq_len, -1)
        t_enc = self.encode_t(t.view(-1, *t.shape[2:])).view(bsz, seq_len, -1)
        y = _ensure_3d_y(y).float()
        inp = torch.cat([x_rep, a_enc, t_enc, y], dim=-1)
        inp = self.enc_in(inp)

        lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(inp, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.encoder(packed)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        mu = self.z_mu(h)
        logvar = self.z_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        z: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
        stochastic_y: bool = False,
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = a.shape[0], a.shape[1]
        x_enc = self.x_proj(x.float())
        h = self.dec_h0(torch.cat([x_enc, z], dim=-1))
        prev_y = torch.zeros(bsz, 1, device=x.device)

        y_logits_seq = []
        y_out_seq = []
        for ti in range(seq_len):
            a_enc = self.encode_a(a[:, ti])
            t_enc = self.encode_t(t[:, ti])

            parts = [x_enc, a_enc, t_enc, z]
            if self.cfg.use_prev_y:
                parts.append(prev_y)
            inp = torch.cat(parts, dim=-1)
            h = self.dec_cell(inp, h)

            y_logits = self.y_head(h)
            y_prob = torch.sigmoid(y_logits)
            y_out = torch.bernoulli(y_prob) if stochastic_y else y_prob

            m = mask[:, ti].view(bsz, 1)
            y_logits = m * y_logits + (1.0 - m) * y_logits.detach()
            y_out = m * y_out + (1.0 - m) * y_out.detach()

            y_logits_seq.append(y_logits)
            y_out_seq.append(y_out)

            if self.cfg.use_prev_y:
                if teacher_forcing and y is not None:
                    prev_y = _ensure_3d_y(y)[:, ti, :].detach()
                else:
                    prev_y = y_out.detach()

        return {"y_logits": torch.stack(y_logits_seq, dim=1), "y": torch.stack(y_out_seq, dim=1)}

    def elbo_loss(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x=x, a=a, t=t, y=y, mask=mask)
        z = self.reparameterize(mu, logvar)
        dec = self.decode(x=x, a=a, t=t, mask=mask, z=z, y=y, teacher_forcing=True, stochastic_y=False)
        recon = bernoulli_nll_from_logits(dec["y_logits"], y, mask=mask)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1).mean()
        loss = recon + float(kl_weight) * kl
        return {"loss": loss, "recon": recon.detach(), "kl": kl.detach()}

    def rollout(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t_obs: Optional[torch.Tensor] = None,
        do_t: Optional[Dict[int, torch.Tensor]] = None,
        policy: Optional[Policy] = None,
        mask: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        stochastic_y: bool = False,
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = a.shape[0], a.shape[1]
        if steps is None:
            steps = seq_len
        if mask is None:
            mask = torch.ones(bsz, steps, device=x.device)

        z = torch.randn(bsz, self.cfg.z_dim, device=x.device)

        if t_obs is None:
            if policy is None:
                raise ValueError("One of t_obs or policy must be provided.")
            t_list = []
            dummy_h = torch.zeros(bsz, 1, device=x.device)
            prev_y = torch.zeros(bsz, 1, device=x.device)
            for ti in range(steps):
                if do_t is not None and ti in do_t:
                    t_t = do_t[ti]
                else:
                    t_t = policy.act(dummy_h, t_index=ti, x=x, a_t=a[:, ti], prev_y=prev_y)
                t_list.append(t_t)
            t_used = torch.stack(t_list, dim=1)
        else:
            t_used = t_obs[:, :steps].clone()
            if do_t is not None:
                for ti, val in do_t.items():
                    if 0 <= int(ti) < steps:
                        t_used[:, int(ti)] = val

        dec = self.decode(x=x, a=a[:, :steps], t=t_used, mask=mask, z=z, y=None, teacher_forcing=False, stochastic_y=stochastic_y)
        return {"y_logits": dec["y_logits"], "y": dec["y"], "t": t_used}


@dataclass(frozen=True)
class SeqDiffusionConfig:
    d_x: int
    num_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    time_emb_dim: int = 64
    model_dim: int = 128
    backbone: str = "mlp"  # mlp | transformer
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    a_is_discrete: bool = True
    a_vocab_size: int = 100
    a_emb_dim: int = 32
    d_a: int = 16
    t_is_discrete: bool = True
    t_vocab_size: int = 4
    t_emb_dim: int = 16
    d_t: int = 8
    y_dim: int = 1


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, *, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class SeqDiffusion(nn.Module):
    def __init__(self, cfg: SeqDiffusionConfig):
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

        self.time_emb = nn.Embedding(cfg.num_steps, cfg.time_emb_dim)

        d_in_step = cfg.y_dim + cfg.d_x + d_a + d_t + cfg.time_emb_dim
        if cfg.backbone == "mlp":
            self.net = nn.Sequential(
                nn.Linear(d_in_step, cfg.model_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.model_dim, cfg.model_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.model_dim, cfg.y_dim),
            )
        elif cfg.backbone == "transformer":
            self.in_proj = nn.Linear(d_in_step, cfg.model_dim)
            self.blocks = nn.ModuleList(
                [
                    _TransformerBlock(
                        d_model=cfg.model_dim,
                        n_heads=cfg.n_heads,
                        ffn_hidden=cfg.model_dim * 4,
                        dropout=cfg.dropout,
                    )
                    for _ in range(cfg.n_layers)
                ]
            )
            self.out_proj = nn.Linear(cfg.model_dim, cfg.y_dim)
        else:
            raise ValueError(f"Unknown backbone={cfg.backbone}")

        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def encode_a(self, a: torch.Tensor) -> torch.Tensor:
        if self.cfg.a_is_discrete:
            return self.a_emb(a.long())
        return self.a_emb(a.float())

    def encode_t(self, t: torch.Tensor) -> torch.Tensor:
        if self.cfg.t_is_discrete:
            return self.t_emb(t.long())
        return self.t_emb(t.float())

    def _predict_eps(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        y_noisy: torch.Tensor,
        timesteps: torch.Tensor,  # [B]
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len = a.shape[0], a.shape[1]
        x_rep = self.x_proj(x.float())[:, None, :].expand(bsz, seq_len, x.shape[1])
        a_enc = self.encode_a(a.view(-1, *a.shape[2:])).view(bsz, seq_len, -1)
        t_enc = self.encode_t(t.view(-1, *t.shape[2:])).view(bsz, seq_len, -1)
        y_noisy = _ensure_3d_y(y_noisy).float()
        t_emb = self.time_emb(timesteps.long())[:, None, :].expand(bsz, seq_len, self.cfg.time_emb_dim)

        inp = torch.cat([y_noisy, x_rep, a_enc, t_enc, t_emb], dim=-1)
        if self.cfg.backbone == "mlp":
            return self.net(inp)

        h = self.in_proj(inp)
        key_padding_mask = None if mask is None else (~mask.bool())
        for blk in self.blocks:
            h = blk(h, key_padding_mask=key_padding_mask)
        return self.out_proj(h)

    def denoising_loss(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz = x.shape[0]
        y0 = _ensure_3d_y(y).float()
        timesteps = torch.randint(low=0, high=self.cfg.num_steps, size=(bsz,), device=x.device)
        noise = torch.randn_like(y0)
        alpha_bar = self.alpha_bars[timesteps].view(bsz, 1, 1)
        y_noisy = torch.sqrt(alpha_bar) * y0 + torch.sqrt(1.0 - alpha_bar) * noise
        pred = self._predict_eps(x=x, a=a, t=t, y_noisy=y_noisy, timesteps=timesteps, mask=mask)
        loss = (pred - noise).pow(2).mean(dim=-1)  # [B,T]
        return masked_mean(loss, mask)

    @torch.no_grad()
    def sample(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,  # [B,T,y_dim]
    ) -> torch.Tensor:
        bsz, seq_len = a.shape[0], a.shape[1]
        y = torch.randn(bsz, seq_len, self.cfg.y_dim, device=x.device) if eps is None else eps.clone()

        for step in reversed(range(self.cfg.num_steps)):
            ts = torch.full((bsz,), step, device=x.device, dtype=torch.long)
            pred_eps = self._predict_eps(x=x, a=a, t=t, y_noisy=y, timesteps=ts, mask=mask)

            beta = self.betas[ts].view(bsz, 1, 1)
            alpha = self.alphas[ts].view(bsz, 1, 1)
            alpha_bar = self.alpha_bars[ts].view(bsz, 1, 1)

            mean = (1.0 / torch.sqrt(alpha)) * (y - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_eps)
            if step > 0:
                y = mean + torch.sqrt(beta) * torch.randn_like(y)
            else:
                y = mean

        return y.clamp(0.0, 1.0)

    def rollout(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t_obs: Optional[torch.Tensor] = None,
        do_t: Optional[Dict[int, torch.Tensor]] = None,
        policy: Optional[Policy] = None,
        mask: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        stochastic_y: bool = False,
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = a.shape[0], a.shape[1]
        if steps is None:
            steps = seq_len
        if mask is None:
            mask = torch.ones(bsz, steps, device=x.device)

        if t_obs is None:
            if policy is None:
                raise ValueError("One of t_obs or policy must be provided.")
            t_list = []
            dummy_h = torch.zeros(bsz, 1, device=x.device)
            prev_y = torch.zeros(bsz, 1, device=x.device)
            for ti in range(steps):
                if do_t is not None and ti in do_t:
                    t_t = do_t[ti]
                else:
                    t_t = policy.act(dummy_h, t_index=ti, x=x, a_t=a[:, ti], prev_y=prev_y)
                t_list.append(t_t)
            t_used = torch.stack(t_list, dim=1)
        else:
            t_used = t_obs[:, :steps].clone()
            if do_t is not None:
                for ti, val in do_t.items():
                    if 0 <= int(ti) < steps:
                        t_used[:, int(ti)] = val

        y = self.sample(x=x, a=a[:, :steps], t=t_used, mask=mask, eps=eps)
        if stochastic_y:
            y = torch.bernoulli(y)
        return {"y": y, "t": t_used}


@dataclass(frozen=True)
class CRNConfig:
    d_x: int
    d_h: int = 128
    dropout: float = 0.1
    grl_lambda: float = 1.0
    a_is_discrete: bool = True
    a_vocab_size: int = 100
    a_emb_dim: int = 32
    d_a: int = 16
    t_is_discrete: bool = True
    t_vocab_size: int = 4
    t_emb_dim: int = 16
    d_t: int = 8


class CRN(nn.Module):
    def __init__(self, cfg: CRNConfig):
        super().__init__()
        self.cfg = cfg
        if not cfg.t_is_discrete:
            raise ValueError("CRN baseline currently supports discrete treatments only.")

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

        self.enc_cell = nn.GRUCell(input_size=cfg.d_x + d_a + d_t + 1, hidden_size=cfg.d_h)
        self.h0 = nn.Sequential(nn.Linear(cfg.d_x, cfg.d_h), nn.Tanh())

        self.t_head = nn.Sequential(
            nn.Linear(cfg.d_h, cfg.d_h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_h, cfg.t_vocab_size),
        )
        self.y_head = nn.Sequential(
            nn.Linear(cfg.d_h + cfg.d_x + d_a + d_t, cfg.d_h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_h, 1),
        )

    def encode_a(self, a_t: torch.Tensor) -> torch.Tensor:
        if self.cfg.a_is_discrete:
            return self.a_emb(a_t.long())
        return self.a_emb(a_t.float())

    def encode_t(self, t_t: torch.Tensor) -> torch.Tensor:
        if self.cfg.t_is_discrete:
            return self.t_emb(t_t.long())
        return self.t_emb(t_t.float())

    def forward(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = a.shape[0], a.shape[1]
        x_enc = self.x_proj(x.float())
        h = self.h0(x_enc)

        y = _ensure_3d_y(y).float()
        y_logits_seq = []
        t_logits_seq = []
        h_seq = []

        for ti in range(seq_len):
            h_seq.append(h)
            t_logits_seq.append(self.t_head(grad_reverse(h, lambd=self.cfg.grl_lambda)))

            a_enc = self.encode_a(a[:, ti])
            t_enc = self.encode_t(t[:, ti])
            y_inp = torch.cat([h, x_enc, a_enc, t_enc], dim=-1)
            y_logits = self.y_head(y_inp)
            y_logits_seq.append(y_logits)

            upd_inp = torch.cat([x_enc, a_enc, t_enc, y[:, ti, :]], dim=-1)
            h_next = self.enc_cell(upd_inp, h)
            m = mask[:, ti].view(bsz, 1)
            h = m * h_next + (1.0 - m) * h

        return {
            "y_logits": torch.stack(y_logits_seq, dim=1),
            "t_logits": torch.stack(t_logits_seq, dim=1),
            "h": torch.stack(h_seq, dim=1),
        }

    def loss(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        w_treat: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(x=x, a=a, t=t, y=y, mask=mask)
        loss_y = bernoulli_nll_from_logits(out["y_logits"], y, mask=mask)

        bsz, seq_len, num_actions = out["t_logits"].shape
        ce = F.cross_entropy(out["t_logits"].view(bsz * seq_len, num_actions), t.view(-1).long(), reduction="none")
        ce = ce.view(bsz, seq_len)
        loss_t = masked_mean(ce, mask)

        loss = loss_y + float(w_treat) * loss_t
        return {"loss": loss, "loss_y": loss_y.detach(), "loss_treat": loss_t.detach()}

    @torch.no_grad()
    def rollout(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t_obs: Optional[torch.Tensor] = None,
        do_t: Optional[Dict[int, torch.Tensor]] = None,
        policy: Optional[Policy] = None,
        mask: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        stochastic_y: bool = False,
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = a.shape[0], a.shape[1]
        if steps is None:
            steps = seq_len
        if mask is None:
            mask = torch.ones(bsz, steps, device=x.device)

        x_enc = self.x_proj(x.float())
        h = self.h0(x_enc)
        prev_y = torch.zeros(bsz, 1, device=x.device)

        y_logits_seq = []
        y_out_seq = []
        t_used_seq = []

        for ti in range(steps):
            if do_t is not None and ti in do_t:
                t_t = do_t[ti]
            elif policy is not None:
                t_t = policy.act(h, t_index=ti, x=x, a_t=a[:, ti], prev_y=prev_y)
            elif t_obs is not None:
                t_t = t_obs[:, ti]
            else:
                raise ValueError("One of do_t, policy, t_obs must be provided.")

            a_enc = self.encode_a(a[:, ti])
            t_enc = self.encode_t(t_t)
            y_inp = torch.cat([h, x_enc, a_enc, t_enc], dim=-1)
            y_logits = self.y_head(y_inp)
            y_prob = torch.sigmoid(y_logits)
            y_out = torch.bernoulli(y_prob) if stochastic_y else y_prob

            upd_inp = torch.cat([x_enc, a_enc, t_enc, y_out.detach()], dim=-1)
            h_next = self.enc_cell(upd_inp, h)
            m = mask[:, ti].view(bsz, 1)
            h = m * h_next + (1.0 - m) * h

            y_logits = m * y_logits + (1.0 - m) * y_logits.detach()
            y_out = m * y_out + (1.0 - m) * y_out.detach()

            prev_y = y_out.detach()
            y_logits_seq.append(y_logits)
            y_out_seq.append(y_out)
            t_used_seq.append(t_t)

        return {
            "y_logits": torch.stack(y_logits_seq, dim=1),
            "y": torch.stack(y_out_seq, dim=1),
            "t": torch.stack(t_used_seq, dim=1),
        }


def build_discriminator_from_meta(
    *,
    d_x: int,
    a_is_discrete: bool,
    a_vocab_size: int,
    d_a: int,
    t_is_discrete: bool,
    t_vocab_size: int,
    d_t: int,
    a_emb_dim: int,
    t_emb_dim: int,
    d_h: int,
    dropout: float,
) -> Tuple[SequenceDiscriminatorConfig, SequenceDiscriminator]:
    disc_cfg = SequenceDiscriminatorConfig(
        d_x=d_x,
        a_is_discrete=a_is_discrete,
        a_vocab_size=a_vocab_size,
        a_emb_dim=a_emb_dim,
        d_a=d_a,
        t_is_discrete=t_is_discrete,
        t_vocab_size=t_vocab_size,
        t_emb_dim=t_emb_dim,
        d_t=d_t,
        d_y=1,
        d_h=d_h,
        dropout=dropout,
    )
    return disc_cfg, SequenceDiscriminator(disc_cfg)


def load_rollout_model_from_checkpoint(ckpt: dict, *, device: torch.device) -> Tuple[nn.Module, str]:
    # Legacy SCM checkpoints from src/train.py
    if "model" not in ckpt and "gen_cfg" in ckpt:
        cfg = SCMGeneratorConfig(**ckpt["gen_cfg"])
        model = SCMGenerator(cfg).to(device)
        model.load_state_dict(ckpt["gen"])
        model.eval()
        return model, "scm"

    model_name = str(ckpt.get("model", "scm"))
    cfg_dict = ckpt.get("model_cfg") or ckpt.get("gen_cfg") or ckpt.get("cfg")
    state = ckpt.get("model_state") or ckpt.get("gen") or ckpt.get("state_dict")
    if cfg_dict is None or state is None:
        raise ValueError("Checkpoint missing model config/state.")

    if model_name == "scm":
        cfg = SCMGeneratorConfig(**cfg_dict)
        model = SCMGenerator(cfg).to(device)
    elif model_name == "rcgan":
        cfg = RCGANConfig(**cfg_dict)
        model = RCGANGenerator(cfg).to(device)
    elif model_name == "vae":
        cfg = SeqVAEConfig(**cfg_dict)
        model = SeqVAE(cfg).to(device)
    elif model_name == "diffusion":
        cfg = SeqDiffusionConfig(**cfg_dict)
        model = SeqDiffusion(cfg).to(device)
    elif model_name == "crn":
        cfg = CRNConfig(**cfg_dict)
        model = CRN(cfg).to(device)
    else:
        raise ValueError(f"Unknown model={model_name}")

    model.load_state_dict(state)
    model.eval()
    return model, model_name
