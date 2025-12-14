from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SCMGeneratorConfig:
    d_x: int
    d_k: int = 64
    d_eps: int = 16

    # A_t (item / skill / difficulty)
    a_is_discrete: bool = True
    a_vocab_size: int = 100
    a_emb_dim: int = 32
    d_a: int = 16  # used when a_is_discrete=False

    # T_t (teaching action)
    t_is_discrete: bool = True
    t_vocab_size: int = 4
    t_emb_dim: int = 16
    d_t: int = 8  # used when t_is_discrete=False

    # Y_t (outcome)
    y_dist: str = "bernoulli"  # bernoulli | gaussian | categorical
    d_y: int = 1  # gaussian: output dims; bernoulli: usually 1
    y_vocab_size: int = 2  # categorical
    use_y_in_dynamics: bool = False  # DKT-style feedback: K_{t+1} depends on Y_t

    dynamics: str = "gru"  # gru | mlp | transformer
    mlp_hidden: int = 128
    dropout: float = 0.1

    # Transformer dynamics (used when dynamics="transformer")
    tf_n_layers: int = 2
    tf_n_heads: int = 4
    tf_ffn_hidden: int = 256
    tf_max_seq_len: int = 512


class _CausalSelfAttentionBlock(nn.Module):
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class _TransformerDynamics(nn.Module):
    """Masked self-attention dynamics to produce K_{t+1} from history embeddings."""

    def __init__(
        self,
        *,
        d_inp: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_hidden: int,
        dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)

        self.inp_proj = nn.Linear(d_inp, d_model)
        self.pos_emb = nn.Embedding(self.max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                _CausalSelfAttentionBlock(d_model=d_model, n_heads=n_heads, ffn_hidden=ffn_hidden, dropout=dropout)
                for _ in range(int(n_layers))
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        # True means "do not attend" (future positions).
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, inp_seq: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # inp_seq: [B,T,d_inp] -> out: [B,T,d_model]
        bsz, seq_len = inp_seq.shape[0], inp_seq.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds tf_max_seq_len={self.max_seq_len}")

        x = self.inp_proj(inp_seq)
        pos = torch.arange(seq_len, device=inp_seq.device)
        x = x + self.pos_emb(pos)[None, :, :]

        attn_mask = self._causal_mask(seq_len, inp_seq.device)
        key_padding_mask = None if mask is None else (~mask.bool())

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        return self.ln_f(x)


class _MLPDynamics(nn.Module):
    def __init__(self, d_in: int, d_k: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_k),
        )

    def forward(self, k_t: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([k_t, inp], dim=-1))


class Supervisor(nn.Module):
    """TimeGAN-style stepwise supervision on latent dynamics."""

    def __init__(self, d_k: int, d_inp: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_k + d_inp, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_k),
        )

    def forward(self, k_t: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([k_t, inp], dim=-1))


class SCMGenerator(nn.Module):
    """Dynamic SCM generator with a rollout API (supports do() and policies).

    Core SCM:
      K_{t+1} = f_theta(K_t, A_t, T_t, X, eps_t)
      Y_t ~ p_theta(. | K_t, A_t, T_t, X)
    """

    def __init__(self, cfg: SCMGeneratorConfig):
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

        if cfg.y_dist == "bernoulli":
            self.d_y_dyn = 1
        elif cfg.y_dist == "gaussian":
            self.d_y_dyn = int(cfg.d_y)
        elif cfg.y_dist == "categorical":
            self.d_y_dyn = int(cfg.y_vocab_size)
        else:
            raise ValueError(f"Unknown y_dist={cfg.y_dist}")

        self.d_inp = d_a + d_t + cfg.d_x + cfg.d_eps + (self.d_y_dyn if cfg.use_y_in_dynamics else 0)

        self.k0 = nn.Sequential(nn.Linear(cfg.d_x, cfg.d_k), nn.Tanh())

        if cfg.dynamics == "gru":
            self.dynamics = nn.GRUCell(input_size=self.d_inp, hidden_size=cfg.d_k)
        elif cfg.dynamics == "mlp":
            self.dynamics = _MLPDynamics(
                d_in=cfg.d_k + self.d_inp,
                d_k=cfg.d_k,
                hidden=cfg.mlp_hidden,
                dropout=cfg.dropout,
            )
        elif cfg.dynamics == "transformer":
            self.dynamics = _TransformerDynamics(
                d_inp=self.d_inp,
                d_model=cfg.d_k,
                n_layers=cfg.tf_n_layers,
                n_heads=cfg.tf_n_heads,
                ffn_hidden=cfg.tf_ffn_hidden,
                dropout=cfg.dropout,
                max_seq_len=cfg.tf_max_seq_len,
            )
        else:
            raise ValueError(f"Unknown dynamics={cfg.dynamics}")

        d_obs_in = cfg.d_k + d_a + d_t + cfg.d_x
        if cfg.y_dist == "bernoulli":
            self.y_head = nn.Linear(d_obs_in, 1)
        elif cfg.y_dist == "gaussian":
            self.y_head = nn.Linear(d_obs_in, cfg.d_y)
        elif cfg.y_dist == "categorical":
            self.y_head = nn.Linear(d_obs_in, cfg.y_vocab_size)
        else:
            raise ValueError(f"Unknown y_dist={cfg.y_dist}")

        self.supervisor = Supervisor(cfg.d_k, self.d_inp, hidden=cfg.mlp_hidden, dropout=cfg.dropout)

    def encode_a(self, a_t: torch.Tensor) -> torch.Tensor:
        if self.cfg.a_is_discrete:
            return self.a_emb(a_t.long())
        return self.a_emb(a_t.float())

    def encode_t(self, t_t: torch.Tensor) -> torch.Tensor:
        if self.cfg.t_is_discrete:
            return self.t_emb(t_t.long())
        return self.t_emb(t_t.float())

    def _format_y_dyn_step(self, y: torch.Tensor) -> torch.Tensor:
        if self.cfg.y_dist in {"bernoulli", "gaussian"}:
            if y.ndim == 1:
                y = y[:, None]
            return y.float()
        if self.cfg.y_dist == "categorical":
            if y.ndim == 1:
                return F.one_hot(y.long(), num_classes=self.cfg.y_vocab_size).float()
            if y.ndim == 2 and y.shape[1] == 1:
                return F.one_hot(y.squeeze(-1).long(), num_classes=self.cfg.y_vocab_size).float()
            if y.ndim == 2 and y.shape[1] == self.cfg.y_vocab_size:
                return y.float()
            raise ValueError(f"y_dyn for categorical must be indices [B] or probs [B,{self.cfg.y_vocab_size}], got {tuple(y.shape)}")
        raise ValueError(self.cfg.y_dist)

    def _format_y_dyn_seq(self, y: torch.Tensor, *, seq_len: int) -> torch.Tensor:
        if self.cfg.y_dist == "categorical":
            if y.ndim == 2 and y.shape[1] == seq_len:
                return F.one_hot(y.long(), num_classes=self.cfg.y_vocab_size).float()
            if y.ndim == 3 and y.shape[1] == seq_len and y.shape[2] == self.cfg.y_vocab_size:
                return y.float()
            raise ValueError(
                f"y for categorical must be [B,T] indices or [B,T,{self.cfg.y_vocab_size}] probs, got {tuple(y.shape)}"
            )

        # bernoulli / gaussian
        if y.ndim == 2 and y.shape[1] == seq_len:
            return y.float().unsqueeze(-1) if self.d_y_dyn == 1 else y.float()
        if y.ndim == 3 and y.shape[1] == seq_len and y.shape[2] == self.d_y_dyn:
            return y.float()
        raise ValueError(f"y must be [B,T] or [B,T,{self.d_y_dyn}], got {tuple(y.shape)}")

    def init_k(self, x: torch.Tensor) -> torch.Tensor:
        return self.k0(x.float())

    def _make_inp(
        self,
        x: torch.Tensor,
        a_enc: torch.Tensor,
        t_enc: torch.Tensor,
        eps: torch.Tensor,
        *,
        y_dyn: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_enc = self.x_proj(x.float())
        parts = [a_enc, t_enc, x_enc]
        if self.cfg.use_y_in_dynamics:
            if y_dyn is None:
                raise ValueError("y_dyn is required when use_y_in_dynamics=True")
            if y_dyn.ndim == 1:
                y_dyn = y_dyn[:, None]
            parts.append(y_dyn.float())
        parts.append(eps)
        return torch.cat(parts, dim=-1)

    def step(
        self,
        k_t: torch.Tensor,
        *,
        x: torch.Tensor,
        a_t: torch.Tensor,
        t_t: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
        y_dyn: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One SCM transition step: returns (k_{t+1}, y_logits_t)."""
        if self.cfg.dynamics == "transformer":
            raise RuntimeError("step() is not supported for dynamics='transformer' (needs history); use teacher_forcing/rollout.")
        a_enc = self.encode_a(a_t)
        t_enc = self.encode_t(t_t)

        x_enc = self.x_proj(x.float())
        y_inp = torch.cat([k_t, a_enc, t_enc, x_enc], dim=-1)
        y_logits = self.y_head(y_inp)

        if self.cfg.use_y_in_dynamics:
            if y_dyn is None:
                if self.cfg.y_dist == "bernoulli":
                    y_dyn = torch.sigmoid(y_logits)
                elif self.cfg.y_dist == "gaussian":
                    y_dyn = y_logits
                else:
                    y_dyn = F.softmax(y_logits, dim=-1)
            y_dyn = self._format_y_dyn_step(y_dyn)

        if eps is None:
            eps = torch.randn(k_t.shape[0], self.cfg.d_eps, device=k_t.device)
        inp = self._make_inp(x, a_enc, t_enc, eps, y_dyn=y_dyn)

        if self.cfg.dynamics == "gru":
            k_next = self.dynamics(inp, k_t)
        else:
            k_next = self.dynamics(k_t, inp)
        return k_next, y_logits

    @torch.no_grad()
    def sample_y(self, y_logits: torch.Tensor) -> torch.Tensor:
        if self.cfg.y_dist == "bernoulli":
            return torch.bernoulli(torch.sigmoid(y_logits)).float()
        if self.cfg.y_dist == "gaussian":
            return y_logits
        if self.cfg.y_dist == "categorical":
            return torch.distributions.Categorical(logits=y_logits).sample()
        raise ValueError(self.cfg.y_dist)

    def teacher_forcing(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,  # [B,T,d_eps]
        eps_mode: str = "zero",  # zero | random
    ) -> Dict[str, torch.Tensor]:
        """Unrolls the SCM under observed (A,T) and returns latent + logits."""
        if self.cfg.dynamics == "transformer":
            return self._teacher_forcing_transformer(x=x, a=a, t=t, y=y, mask=mask, eps=eps, eps_mode=eps_mode)
        bsz, seq_len = a.shape[0], a.shape[1]
        device = x.device
        if mask is None:
            mask = torch.ones(bsz, seq_len, device=device)
        if self.cfg.use_y_in_dynamics and y is None:
            raise ValueError("teacher_forcing requires y when use_y_in_dynamics=True")

        k_t = self.init_k(x)
        k_seq = [k_t]
        y_logits_seq = []

        for ti in range(seq_len):
            if eps is None:
                if eps_mode == "zero":
                    eps_t = torch.zeros(bsz, self.cfg.d_eps, device=device)
                elif eps_mode == "random":
                    eps_t = torch.randn(bsz, self.cfg.d_eps, device=device)
                else:
                    raise ValueError(f"Unknown eps_mode={eps_mode}")
            else:
                if eps.ndim != 3 or eps.shape[0] != bsz or eps.shape[1] != seq_len or eps.shape[2] != self.cfg.d_eps:
                    raise ValueError(f"eps must be [B,T,d_eps]=[{bsz},{seq_len},{self.cfg.d_eps}], got {tuple(eps.shape)}")
                eps_t = eps[:, ti, :]

            y_t = None if y is None else y[:, ti]
            k_next, y_logits = self.step(k_t, x=x, a_t=a[:, ti], t_t=t[:, ti], eps=eps_t, y_dyn=y_t)

            # Keep padded steps stable (avoid leaking padding into training).
            m = mask[:, ti].view(bsz, 1)
            k_next = m * k_next + (1.0 - m) * k_t

            k_seq.append(k_next)
            y_logits_seq.append(y_logits)
            k_t = k_next

        return {
            "k": torch.stack(k_seq, dim=1),  # [B, T+1, d_k]
            "y_logits": torch.stack(y_logits_seq, dim=1),  # [B, T, ...]
        }

    def _teacher_forcing_transformer(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        eps: Optional[torch.Tensor],
        eps_mode: str,
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = a.shape[0], a.shape[1]
        device = x.device
        if mask is None:
            mask = torch.ones(bsz, seq_len, device=device)

        if eps is None:
            if eps_mode == "zero":
                eps_seq = torch.zeros(bsz, seq_len, self.cfg.d_eps, device=device)
            elif eps_mode == "random":
                eps_seq = torch.randn(bsz, seq_len, self.cfg.d_eps, device=device)
            else:
                raise ValueError(f"Unknown eps_mode={eps_mode}")
        else:
            if eps.ndim != 3 or eps.shape[0] != bsz or eps.shape[1] != seq_len or eps.shape[2] != self.cfg.d_eps:
                raise ValueError(f"eps must be [B,T,d_eps]=[{bsz},{seq_len},{self.cfg.d_eps}], got {tuple(eps.shape)}")
            eps_seq = eps

        if self.cfg.use_y_in_dynamics and y is None:
            raise ValueError("teacher_forcing requires y when use_y_in_dynamics=True")

        a_enc = self.encode_a(a.view(-1, *a.shape[2:])).view(bsz, seq_len, -1)
        t_enc = self.encode_t(t.view(-1, *t.shape[2:])).view(bsz, seq_len, -1)
        x_rep = self.x_proj(x.float())[:, None, :].expand(bsz, seq_len, x.shape[1])
        if self.cfg.use_y_in_dynamics:
            if y is None:
                raise ValueError("teacher_forcing requires y when use_y_in_dynamics=True")
            y_dyn = self._format_y_dyn_seq(y, seq_len=seq_len)
            inp_seq = torch.cat([a_enc, t_enc, x_rep, y_dyn, eps_seq], dim=-1)  # [B,T,d_inp]
        else:
            inp_seq = torch.cat([a_enc, t_enc, x_rep, eps_seq], dim=-1)  # [B,T,d_inp]

        k_out = self.dynamics(inp_seq, mask=mask)  # [B,T,d_k] representing K_{t+1}

        k0 = self.init_k(x)
        k_seq = [k0]
        for ti in range(seq_len):
            k_next = k_out[:, ti, :]
            m = mask[:, ti].view(bsz, 1)
            k_next = m * k_next + (1.0 - m) * k_seq[-1]
            k_seq.append(k_next)
        k_seq = torch.stack(k_seq, dim=1)  # [B,T+1,d_k]

        y_inp = torch.cat([k_seq[:, :-1, :], a_enc, t_enc, x_rep], dim=-1)
        y_logits = self.y_head(y_inp)
        y_logits = mask[:, :, None] * y_logits + (1.0 - mask[:, :, None]) * y_logits.detach()
        return {"k": k_seq, "y_logits": y_logits}

    def rollout(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t_obs: Optional[torch.Tensor] = None,
        do_t: Optional[Dict[int, torch.Tensor]] = None,
        policy: Optional["Policy"] = None,
        mask: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,  # [B,T,d_eps]
        steps: Optional[int] = None,
        stochastic_y: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Rolls out a trajectory, optionally with do(T_t=...) or a policy."""
        if self.cfg.dynamics == "transformer":
            return self._rollout_transformer(
                x=x, a=a, t_obs=t_obs, do_t=do_t, policy=policy, mask=mask, eps=eps, steps=steps, stochastic_y=stochastic_y
            )
        bsz, seq_len = a.shape[0], a.shape[1]
        device = x.device
        if steps is None:
            steps = seq_len
        if mask is None:
            mask = torch.ones(bsz, steps, device=device)

        k_t = self.init_k(x)
        k_seq = [k_t]
        y_logits_seq = []
        y_out_seq = []
        t_used_seq = []

        prev_y = torch.zeros(bsz, 1, device=device)
        for ti in range(steps):
            if do_t is not None and ti in do_t:
                t_t = do_t[ti]
            elif policy is not None:
                t_t = policy.act(k_t, t_index=ti, x=x, a_t=a[:, ti], prev_y=prev_y)
            elif t_obs is not None:
                t_t = t_obs[:, ti]
            else:
                raise ValueError("One of do_t, policy, t_obs must be provided.")

            if eps is None:
                eps_t = None
            else:
                if eps.ndim != 3 or eps.shape[0] != bsz or eps.shape[1] < steps or eps.shape[2] != self.cfg.d_eps:
                    raise ValueError(f"eps must be [B,>=T,d_eps]=[{bsz},>={steps},{self.cfg.d_eps}], got {tuple(eps.shape)}")
                eps_t = eps[:, ti, :]

            k_next, y_logits = self.step(k_t, x=x, a_t=a[:, ti], t_t=t_t, eps=eps_t)

            if self.cfg.y_dist == "bernoulli":
                y_prob = torch.sigmoid(y_logits)
                y_out = torch.bernoulli(y_prob) if stochastic_y else y_prob
                prev_y = y_out.detach() if stochastic_y else y_prob.detach()
            elif self.cfg.y_dist == "gaussian":
                y_out = y_logits
                prev_y = y_out.detach()
            else:
                y_out = F.softmax(y_logits, dim=-1)
                prev_y = y_out.detach()

            m = mask[:, ti].view(bsz, 1)
            k_next = m * k_next + (1.0 - m) * k_t
            y_logits = m * y_logits + (1.0 - m) * y_logits.detach()
            y_out = m * y_out + (1.0 - m) * y_out.detach()

            k_seq.append(k_next)
            y_logits_seq.append(y_logits)
            y_out_seq.append(y_out)
            t_used_seq.append(t_t)
            k_t = k_next

        return {
            "k": torch.stack(k_seq, dim=1),
            "y_logits": torch.stack(y_logits_seq, dim=1),
            "y": torch.stack(y_out_seq, dim=1),
            "t": torch.stack(t_used_seq, dim=1),
        }

    def _rollout_transformer(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t_obs: Optional[torch.Tensor],
        do_t: Optional[Dict[int, torch.Tensor]],
        policy: Optional["Policy"],
        mask: Optional[torch.Tensor],
        eps: Optional[torch.Tensor],
        steps: Optional[int],
        stochastic_y: bool,
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = a.shape[0], a.shape[1]
        device = x.device
        if steps is None:
            steps = seq_len
        if mask is None:
            mask = torch.ones(bsz, steps, device=device)

        if eps is not None:
            if eps.ndim != 3 or eps.shape[0] != bsz or eps.shape[1] < steps or eps.shape[2] != self.cfg.d_eps:
                raise ValueError(f"eps must be [B,>=T,d_eps]=[{bsz},>={steps},{self.cfg.d_eps}], got {tuple(eps.shape)}")

        k_t = self.init_k(x)
        k_seq = [k_t]
        y_logits_seq = []
        y_out_seq = []
        t_used_seq = []

        x_rep_t = self.x_proj(x.float())
        inp_list = []

        prev_y = torch.zeros(bsz, 1, device=device)
        for ti in range(steps):
            if do_t is not None and ti in do_t:
                t_t = do_t[ti]
            elif policy is not None:
                t_t = policy.act(k_t, t_index=ti, x=x, a_t=a[:, ti], prev_y=prev_y)
            elif t_obs is not None:
                t_t = t_obs[:, ti]
            else:
                raise ValueError("One of do_t, policy, t_obs must be provided.")

            if eps is None:
                eps_t = torch.randn(bsz, self.cfg.d_eps, device=device)
            else:
                eps_t = eps[:, ti, :]

            a_enc = self.encode_a(a[:, ti])
            t_enc = self.encode_t(t_t)
            # Observation at time t depends on current K_t and current (A_t, T_t, X).
            y_inp_t = torch.cat([k_t, a_enc, t_enc, x_rep_t], dim=-1)
            y_logits = self.y_head(y_inp_t)

            if self.cfg.y_dist == "bernoulli":
                y_prob = torch.sigmoid(y_logits)
                y_out = torch.bernoulli(y_prob) if stochastic_y else y_prob
                prev_y = y_out.detach() if stochastic_y else y_prob.detach()
            elif self.cfg.y_dist == "gaussian":
                y_out = y_logits
                prev_y = y_out.detach()
            else:
                y_out = F.softmax(y_logits, dim=-1)
                prev_y = y_out.detach()

            if self.cfg.use_y_in_dynamics:
                y_dyn_t = y_out if y_out.ndim == 2 else y_out.view(bsz, -1)
                inp_t = torch.cat([a_enc, t_enc, x_rep_t, y_dyn_t.float(), eps_t], dim=-1)  # [B,d_inp]
            else:
                inp_t = torch.cat([a_enc, t_enc, x_rep_t, eps_t], dim=-1)  # [B,d_inp]

            inp_list.append(inp_t)
            inp_seq = torch.stack(inp_list, dim=1)  # [B, ti+1, d_inp]
            k_out_seq = self.dynamics(inp_seq, mask=mask[:, : ti + 1])  # [B, ti+1, d_k]
            k_next = k_out_seq[:, -1, :]

            m = mask[:, ti].view(bsz, 1)
            k_next = m * k_next + (1.0 - m) * k_t
            y_logits = m * y_logits + (1.0 - m) * y_logits.detach()
            y_out = m * y_out + (1.0 - m) * y_out.detach()

            k_seq.append(k_next)
            y_logits_seq.append(y_logits)
            y_out_seq.append(y_out)
            t_used_seq.append(t_t)
            k_t = k_next

        return {
            "k": torch.stack(k_seq, dim=1),
            "y_logits": torch.stack(y_logits_seq, dim=1),
            "y": torch.stack(y_out_seq, dim=1),
            "t": torch.stack(t_used_seq, dim=1),
        }


from .policy import Policy  # noqa: E402  (placed at end intentionally)
