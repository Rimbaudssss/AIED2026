from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean()
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp(min=1.0)
    return (x * mask).sum() / denom


def _masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if target.ndim == 2:
        target = target[..., None]
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    return _masked_mean(loss, mask)


def _masked_mse(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if x.ndim == 2:
        x = x[..., None]
    if y.ndim == 2:
        y = y[..., None]
    loss = F.mse_loss(x.float(), y.float(), reduction="none")
    return _masked_mean(loss, mask)


def _sample_lengths_from_mask(mask: torch.Tensor) -> torch.Tensor:
    # mask: [N,T] float/bool
    m = mask.detach()
    if m.dtype != torch.float32 and m.dtype != torch.float64:
        m = m.float()
    return m.sum(dim=1).clamp(min=1.0).long()


def _moment_loss_binary(y_real: torch.Tensor, y_prob_fake: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    # Use per-position mean/std matching on probabilities (scalar series).
    if y_real.ndim == 3:
        y_real = y_real.squeeze(-1)
    if y_prob_fake.ndim == 3:
        y_prob_fake = y_prob_fake.squeeze(-1)
    if mask is None:
        real = y_real.reshape(-1).float()
        fake = y_prob_fake.reshape(-1).float()
    else:
        valid = mask > 0.5
        if not valid.any():
            return torch.zeros((), device=y_real.device)
        real = y_real[valid].float()
        fake = y_prob_fake[valid].float()
    mean_r = real.mean()
    mean_f = fake.mean()
    std_r = real.std(unbiased=False)
    std_f = fake.std(unbiased=False)
    return (mean_r - mean_f).abs() + (std_r - std_f).abs()


@dataclass(frozen=True)
class TimeGANConfig:
    d_x: int
    seq_len: int

    a_is_discrete: bool = True
    a_vocab_size: int = 100
    a_emb_dim: int = 32
    d_a: int = 1

    t_is_discrete: bool = True
    t_vocab_size: int = 2
    t_emb_dim: int = 16
    d_t: int = 1

    d_y: int = 1

    # Architecture
    x_emb_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 2
    z_dim: int = 16
    dropout: float = 0.1


@dataclass(frozen=True)
class TimeGANTrainConfig:
    epochs_embed: int = 3
    epochs_supervisor: int = 3
    epochs_joint: int = 5

    lr_embed: float = 1e-3
    lr_gen: float = 1e-3
    lr_disc: float = 1e-3

    lambda_sup: float = 10.0
    lambda_mom: float = 1.0

    g_steps: int = 1
    d_steps: int = 1

    log_every: int = 50
    max_batches_per_epoch: Optional[int] = 200

    # Used only for `generate(..., conditions=None)` convenience.
    reservoir_size: int = 2048


class TimeGAN(nn.Module):
    """Conditional TimeGAN-style generator for longitudinal outcomes.

    This implementation is tailored to the repository's sequence format:
      - static covariates: X [B, d_x]
      - per-step covariates: A_t, T_t
      - binary outcome: Y_t in {0,1} (stored as float)

    The model is trained to generate Y trajectories conditioned on (X, A, T),
    while keeping the original TimeGAN components:
      - embedder/recovery for latent representation learning
      - supervisor for temporal dynamics regularization
      - discriminator for adversarial matching in latent space
      - moment matching loss on generated outcomes
    """

    def __init__(self, cfg: TimeGANConfig):
        super().__init__()
        self.cfg = cfg

        if int(cfg.d_y) != 1:
            raise ValueError(f"TimeGAN expects d_y=1 (binary Y), got d_y={cfg.d_y}")

        if cfg.a_is_discrete:
            self.a_emb = nn.Embedding(int(cfg.a_vocab_size), int(cfg.a_emb_dim))
            d_a_enc = int(cfg.a_emb_dim)
        else:
            self.a_emb = nn.Linear(int(cfg.d_a), int(cfg.d_a))
            d_a_enc = int(cfg.d_a)

        if cfg.t_is_discrete:
            self.t_emb = nn.Embedding(int(cfg.t_vocab_size), int(cfg.t_emb_dim))
            d_t_enc = int(cfg.t_emb_dim)
        else:
            self.t_emb = nn.Linear(int(cfg.d_t), int(cfg.d_t))
            d_t_enc = int(cfg.d_t)

        self.x_proj = nn.Linear(int(cfg.d_x), int(cfg.x_emb_dim))
        self.cond_dim = int(cfg.x_emb_dim) + d_a_enc + d_t_enc

        rnn_dropout = float(cfg.dropout) if int(cfg.num_layers) > 1 else 0.0

        # Embedder E: (Y, cond) -> H
        self.embedder = nn.GRU(
            input_size=1 + self.cond_dim,
            hidden_size=int(cfg.hidden_dim),
            num_layers=int(cfg.num_layers),
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.embedder_out = nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim))

        # Recovery R: (H, cond) -> Y logits
        self.recovery = nn.Sequential(
            nn.Linear(int(cfg.hidden_dim) + self.cond_dim, int(cfg.hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.hidden_dim), 1),
        )

        # Generator G: (Z, cond) -> E_hat
        self.generator = nn.GRU(
            input_size=int(cfg.z_dim) + self.cond_dim,
            hidden_size=int(cfg.hidden_dim),
            num_layers=int(cfg.num_layers),
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.generator_out = nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim))

        # Supervisor S: H -> H (next-step dynamics constraint)
        self.supervisor = nn.GRU(
            input_size=int(cfg.hidden_dim),
            hidden_size=int(cfg.hidden_dim),
            num_layers=int(cfg.num_layers),
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.supervisor_out = nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim))

        # Discriminator D: H -> logits (real vs synthetic)
        self.discriminator = nn.GRU(
            input_size=int(cfg.hidden_dim),
            hidden_size=int(cfg.hidden_dim),
            num_layers=int(cfg.num_layers),
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.disc_out = nn.Linear(int(cfg.hidden_dim), 1)

        # Optional generation helpers filled by training.
        self._reservoir: Optional[dict[str, torch.Tensor]] = None
        self._x_mean: Optional[torch.Tensor] = None
        self._x_std: Optional[torch.Tensor] = None

    def _encode_a_seq(self, a: torch.Tensor) -> torch.Tensor:
        if self.cfg.a_is_discrete:
            return self.a_emb(a.long())
        return self.a_emb(a.float())

    def _encode_t_seq(self, t: torch.Tensor) -> torch.Tensor:
        if self.cfg.t_is_discrete:
            return self.t_emb(t.long())
        return self.t_emb(t.float())

    def _encode_a_step(self, a_t: torch.Tensor) -> torch.Tensor:
        if self.cfg.a_is_discrete:
            return self.a_emb(a_t.long())
        return self.a_emb(a_t.float())

    def _encode_t_step(self, t_t: torch.Tensor) -> torch.Tensor:
        if self.cfg.t_is_discrete:
            return self.t_emb(t_t.long())
        return self.t_emb(t_t.float())

    # Public aliases for compatibility with existing policy/eval code (SCM/RCGAN-style API).
    def encode_a(self, a: torch.Tensor) -> torch.Tensor:
        if self.cfg.a_is_discrete:
            # seq: [B,T] -> [B,T,E]; step: [B] -> [B,E]
            return self._encode_a_seq(a) if a.ndim == 2 else self._encode_a_step(a)
        # seq: [B,T,d_a] -> [B,T,d_a']; step: [B,d_a] -> [B,d_a']
        return self._encode_a_seq(a) if a.ndim == 3 else self._encode_a_step(a)

    def encode_t(self, t: torch.Tensor) -> torch.Tensor:
        if self.cfg.t_is_discrete:
            return self._encode_t_seq(t) if t.ndim == 2 else self._encode_t_step(t)
        return self._encode_t_seq(t) if t.ndim == 3 else self._encode_t_step(t)

    @property
    def y_head(self) -> nn.Module:
        # `ThresholdFailPolicy` expects `model.y_head(y_inp)` to return logits.
        # For TimeGAN the equivalent head is the recovery network.
        return self.recovery

    def _cond_seq(self, *, x: torch.Tensor, a: torch.Tensor, t: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        bsz, seq_len = a.shape[0], a.shape[1]
        if steps is None:
            steps = seq_len
        if steps > seq_len:
            raise ValueError(f"steps={steps} exceeds a.shape[1]={seq_len}")
        x_rep = self.x_proj(x.float())[:, None, :].expand(bsz, steps, self.cfg.x_emb_dim)
        a_enc = self._encode_a_seq(a[:, :steps])
        t_enc = self._encode_t_seq(t[:, :steps])
        return torch.cat([x_rep, a_enc, t_enc], dim=-1)

    def _cond_step(self, *, x: torch.Tensor, a_t: torch.Tensor, t_t: torch.Tensor) -> torch.Tensor:
        x_enc = self.x_proj(x.float())
        a_enc = self._encode_a_step(a_t)
        t_enc = self._encode_t_step(t_t)
        return torch.cat([x_enc, a_enc, t_enc], dim=-1)

    def embed(self, *, x: torch.Tensor, a: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y: [B,T] or [B,T,1]
        if y.ndim == 2:
            y = y[..., None]
        cond = self._cond_seq(x=x, a=a, t=t)
        inp = torch.cat([y.float(), cond], dim=-1)
        h, _ = self.embedder(inp)
        return torch.sigmoid(self.embedder_out(h))

    def recover_y_logits(self, *, h: torch.Tensor, x: torch.Tensor, a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond = self._cond_seq(x=x, a=a, t=t)
        return self.recovery(torch.cat([h, cond], dim=-1))

    def generate_hidden(
        self, *, x: torch.Tensor, a: torch.Tensor, t: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bsz, seq_len = a.shape[0], a.shape[1]
        if z is None:
            z = torch.randn(bsz, seq_len, int(self.cfg.z_dim), device=x.device)
        cond = self._cond_seq(x=x, a=a, t=t)
        g_inp = torch.cat([z, cond], dim=-1)
        e_hat, _ = self.generator(g_inp)
        e_hat = torch.sigmoid(self.generator_out(e_hat))
        h_hat, _ = self.supervisor(e_hat)
        return torch.sigmoid(self.supervisor_out(h_hat))

    def discriminate(self, h: torch.Tensor, *, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Return one logit per sequence (masked mean over time).
        out, _ = self.discriminator(h)
        logits_t = self.disc_out(out).squeeze(-1)  # [B,T]
        if mask is None:
            return logits_t.mean(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (logits_t * mask).sum(dim=1) / denom

    def teacher_forcing(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        stochastic_y: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Intentionally does NOT condition on y (avoid trivial reconstruction at eval time).
        bsz, seq_len = a.shape[0], a.shape[1]
        if mask is None:
            mask = torch.ones(bsz, seq_len, device=x.device)
        if z is None:
            z = torch.zeros(bsz, seq_len, int(self.cfg.z_dim), device=x.device)
        h_hat = self.generate_hidden(x=x, a=a, t=t, z=z)
        y_logits = self.recover_y_logits(h=h_hat, x=x, a=a, t=t)
        y_prob = torch.sigmoid(y_logits)
        y_out = torch.bernoulli(y_prob) if stochastic_y else y_prob
        if y_out.ndim == 2:
            y_out = y_out[..., None]
        if mask is not None:
            m = mask[..., None].float()
            y_logits = m * y_logits + (1.0 - m) * y_logits.detach()
            y_out = m * y_out + (1.0 - m) * y_out.detach()
        return {"y_logits": y_logits, "y": y_out}

    def rollout(
        self,
        *,
        x: torch.Tensor,
        a: torch.Tensor,
        t_obs: Optional[torch.Tensor] = None,
        do_t: Optional[Dict[int, torch.Tensor]] = None,
        policy: Optional[Any] = None,
        mask: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        stochastic_y: bool = False,
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = a.shape[0], a.shape[1]
        if steps is None:
            steps = seq_len
        if mask is None:
            mask = torch.ones(bsz, steps, device=x.device)
        if z is None:
            z = torch.randn(bsz, steps, int(self.cfg.z_dim), device=x.device)

        # Hidden states for generator/supervisor GRUs (num_layers, B, hidden_dim).
        h_g = torch.zeros(int(self.cfg.num_layers), bsz, int(self.cfg.hidden_dim), device=x.device)
        h_s = torch.zeros(int(self.cfg.num_layers), bsz, int(self.cfg.hidden_dim), device=x.device)
        prev_y = torch.zeros(bsz, 1, device=x.device)

        y_logits_seq: list[torch.Tensor] = []
        y_out_seq: list[torch.Tensor] = []
        t_used_seq: list[torch.Tensor] = []

        for ti in range(int(steps)):
            if do_t is not None and ti in do_t:
                t_t = do_t[ti]
            elif policy is not None:
                t_t = policy.act(h_s[-1], t_index=ti, x=x, a_t=a[:, ti], prev_y=prev_y)  # type: ignore[attr-defined]
            elif t_obs is not None:
                t_t = t_obs[:, ti]
            else:
                raise ValueError("One of do_t, policy, t_obs must be provided.")

            cond_t = self._cond_step(x=x, a_t=a[:, ti], t_t=t_t)
            inp_t = torch.cat([z[:, ti, :], cond_t], dim=-1).unsqueeze(1)  # [B,1,z+cond]

            # Generator step.
            out_g, h_g = self.generator(inp_t, h_g)
            e_t = torch.sigmoid(self.generator_out(out_g.squeeze(1)))  # [B,H]

            # Supervisor step.
            out_s, h_s = self.supervisor(e_t.unsqueeze(1), h_s)
            h_t = torch.sigmoid(self.supervisor_out(out_s.squeeze(1)))  # [B,H]

            y_logits = self.recovery(torch.cat([h_t, cond_t], dim=-1)).view(bsz, 1)
            y_prob = torch.sigmoid(y_logits)
            y_out = torch.bernoulli(y_prob) if stochastic_y else y_prob

            prev_y = y_out.detach()

            m = mask[:, ti].view(bsz, 1).float()
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

    @torch.no_grad()
    def generate(self, n: int, max_len: int, conditions: Optional[dict[str, Any]] = None) -> dict[str, torch.Tensor]:
        """Generate synthetic trajectories.

        If `conditions` is provided, it can contain:
          - X: [N,d_x]
          - A: [N,T] or [N,T,d_a]
          - T: [N,T] or [N,T,d_t]
          - mask: [N,T]
          - stochastic_y: bool

        If `conditions is None`, this method falls back to a small training-data reservoir
        (bootstrapped X/A/T/mask) created by `fit_timegan`.
        """

        n = int(n)
        max_len = int(max_len)
        if n <= 0 or max_len <= 0:
            raise ValueError("n and max_len must be positive")

        if conditions is None:
            if self._reservoir is None:
                raise ValueError("conditions=None requires a fitted reservoir, but none was found.")
            idx = torch.randint(0, self._reservoir["X"].shape[0], size=(n,))
            X = self._reservoir["X"][idx]
            A = self._reservoir["A"][idx]
            T = self._reservoir["T"][idx]
            M = self._reservoir["mask"][idx]
            stochastic_y = True
        else:
            X = torch.as_tensor(conditions["X"])
            A = torch.as_tensor(conditions["A"])
            T = torch.as_tensor(conditions["T"])
            M = torch.as_tensor(conditions.get("mask", torch.ones(A.shape[0], A.shape[1])))
            stochastic_y = bool(conditions.get("stochastic_y", True))

        if A.shape[1] < max_len:
            raise ValueError(f"conditions provide seq_len={A.shape[1]} < max_len={max_len}; pad before calling generate()")

        X = X[:n]
        A = A[:n, :max_len]
        T = T[:n, :max_len]
        M = M[:n, :max_len].float()

        ro = self.rollout(x=X.to(next(self.parameters()).device), a=A.to(next(self.parameters()).device), t_obs=T.to(next(self.parameters()).device), mask=M.to(next(self.parameters()).device), stochastic_y=stochastic_y)
        Y = ro["y"].detach().cpu()
        if Y.ndim == 3 and Y.shape[-1] == 1:
            Y = Y.squeeze(-1)

        return {"X": X.detach().cpu(), "A": A.detach().cpu(), "T": T.detach().cpu(), "Y": Y, "mask": M.detach().cpu()}


def fit_timegan(
    model: TimeGAN,
    train_dl: Any,
    *,
    device: torch.device,
    train_cfg: TimeGANTrainConfig,
) -> dict[str, Any]:
    """Fit TimeGAN baseline on a DataLoader yielding batches with X/A/T/Y/mask."""

    model = model.to(device)
    model.train()

    # Disjoint optimizers (do not share parameters across optimizers).
    opt_embed = torch.optim.Adam(
        list(model.x_proj.parameters())
        + list(model.a_emb.parameters())
        + list(model.t_emb.parameters())
        + list(model.embedder.parameters())
        + list(model.embedder_out.parameters())
        + list(model.recovery.parameters()),
        lr=float(train_cfg.lr_embed),
    )
    opt_sup = torch.optim.Adam(list(model.supervisor.parameters()) + list(model.supervisor_out.parameters()), lr=float(train_cfg.lr_gen))
    opt_gen = torch.optim.Adam(list(model.generator.parameters()) + list(model.generator_out.parameters()) + list(model.supervisor.parameters()) + list(model.supervisor_out.parameters()), lr=float(train_cfg.lr_gen))
    opt_disc = torch.optim.Adam(list(model.discriminator.parameters()) + list(model.disc_out.parameters()), lr=float(train_cfg.lr_disc))

    log_every = max(1, int(train_cfg.log_every))
    max_batches = None if train_cfg.max_batches_per_epoch is None else max(1, int(train_cfg.max_batches_per_epoch))

    # ---------- Phase 1: Embedder/Recovery pretrain (reconstruction) ----------
    history: dict[str, list[float]] = {"embed_rec": [], "sup": [], "g": [], "d": []}
    step = 0
    for ep in range(1, int(train_cfg.epochs_embed) + 1):
        total = 0.0
        n_steps = 0
        for batch_i, batch in enumerate(train_dl):
            if max_batches is not None and batch_i >= max_batches:
                break
            X = batch.X.to(device)
            A = batch.A.to(device)
            T = batch.T.to(device)
            Y = batch.Y.to(device)
            M = batch.mask.to(device)

            model.zero_grad(set_to_none=True)
            h = model.embed(x=X, a=A, t=T, y=Y)
            y_rec_logits = model.recover_y_logits(h=h, x=X, a=A, t=T)
            loss = _masked_bce_with_logits(y_rec_logits, Y, M)

            opt_embed.zero_grad(set_to_none=True)
            loss.backward()
            opt_embed.step()

            total += float(loss.item())
            n_steps += 1
            step += 1
            if step % log_every == 0:
                print(f"      TimeGAN | Embed | Ep {ep}/{train_cfg.epochs_embed} | Step {step} | Rec {total / max(1, n_steps):.4f}")

        history["embed_rec"].append(total / max(1, n_steps))
        print(f"      TimeGAN | Embed | Ep {ep}/{train_cfg.epochs_embed} | Rec {total / max(1, n_steps):.4f}")

    # ---------- Phase 2: Supervisor pretrain ----------
    for ep in range(1, int(train_cfg.epochs_supervisor) + 1):
        total = 0.0
        n_steps = 0
        for batch_i, batch in enumerate(train_dl):
            if max_batches is not None and batch_i >= max_batches:
                break
            X = batch.X.to(device)
            A = batch.A.to(device)
            T = batch.T.to(device)
            Y = batch.Y.to(device)
            M = batch.mask.to(device)

            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                h = model.embed(x=X, a=A, t=T, y=Y)
            h_sup, _ = model.supervisor(h)
            h_sup = torch.sigmoid(model.supervisor_out(h_sup))
            loss = _masked_mse(h[:, 1:, :], h_sup[:, :-1, :], M[:, 1:])

            opt_sup.zero_grad(set_to_none=True)
            loss.backward()
            opt_sup.step()

            total += float(loss.item())
            n_steps += 1
            step += 1
            if step % log_every == 0:
                print(f"      TimeGAN | Sup   | Ep {ep}/{train_cfg.epochs_supervisor} | Step {step} | SupMSE {total / max(1, n_steps):.4f}")

        history["sup"].append(total / max(1, n_steps))
        print(f"      TimeGAN | Sup   | Ep {ep}/{train_cfg.epochs_supervisor} | SupMSE {total / max(1, n_steps):.4f}")

    # ---------- Phase 3: Joint training ----------
    for ep in range(1, int(train_cfg.epochs_joint) + 1):
        g_total = 0.0
        d_total = 0.0
        g_steps = 0
        d_steps = 0

        for batch_i, batch in enumerate(train_dl):
            if max_batches is not None and batch_i >= max_batches:
                break
            X = batch.X.to(device)
            A = batch.A.to(device)
            T = batch.T.to(device)
            Y = batch.Y.to(device)
            M = batch.mask.to(device)

            # -- Generator (and supervisor) updates --
            for _ in range(int(train_cfg.g_steps)):
                model.zero_grad(set_to_none=True)
                h_hat = model.generate_hidden(x=X, a=A, t=T, z=None)
                d_fake = model.discriminate(h_hat, mask=M)
                g_adv = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

                h_hat_sup, _ = model.supervisor(h_hat)
                h_hat_sup = torch.sigmoid(model.supervisor_out(h_hat_sup))
                g_sup = _masked_mse(h_hat[:, 1:, :], h_hat_sup[:, :-1, :], M[:, 1:])

                y_hat_logits = model.recover_y_logits(h=h_hat, x=X, a=A, t=T)
                y_hat_prob = torch.sigmoid(y_hat_logits)
                g_mom = _moment_loss_binary(Y, y_hat_prob, M)

                loss_g = g_adv + float(train_cfg.lambda_sup) * g_sup + float(train_cfg.lambda_mom) * g_mom
                opt_gen.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_gen.step()

                g_total += float(loss_g.item())
                g_steps += 1

            # -- Embedder/recovery update (keep latent space meaningful) --
            model.zero_grad(set_to_none=True)
            h = model.embed(x=X, a=A, t=T, y=Y)
            y_rec_logits = model.recover_y_logits(h=h, x=X, a=A, t=T)
            e_rec = _masked_bce_with_logits(y_rec_logits, Y, M)

            h_sup, _ = model.supervisor(h)
            h_sup = torch.sigmoid(model.supervisor_out(h_sup))
            e_sup = _masked_mse(h[:, 1:, :], h_sup[:, :-1, :], M[:, 1:])
            loss_e = e_rec + float(train_cfg.lambda_sup) * e_sup

            opt_embed.zero_grad(set_to_none=True)
            loss_e.backward()
            opt_embed.step()

            # -- Discriminator updates --
            for _ in range(int(train_cfg.d_steps)):
                model.zero_grad(set_to_none=True)
                with torch.no_grad():
                    h = model.embed(x=X, a=A, t=T, y=Y)
                    h_hat = model.generate_hidden(x=X, a=A, t=T, z=None)
                d_real = model.discriminate(h, mask=M)
                d_fake = model.discriminate(h_hat, mask=M)
                loss_d = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + F.binary_cross_entropy_with_logits(
                    d_fake, torch.zeros_like(d_fake)
                )

                opt_disc.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_disc.step()

                d_total += float(loss_d.item())
                d_steps += 1

            step += 1
            if step % log_every == 0:
                print(f"      TimeGAN | Joint | Ep {ep}/{train_cfg.epochs_joint} | Step {step} | G {g_total / max(1, g_steps):.4f} | D {d_total / max(1, d_steps):.4f}")

        history["g"].append(g_total / max(1, g_steps))
        history["d"].append(d_total / max(1, d_steps))
        print(f"      TimeGAN | Joint | Ep {ep}/{train_cfg.epochs_joint} | G {g_total / max(1, g_steps):.4f} | D {d_total / max(1, d_steps):.4f}")

    # Build a small reservoir for convenient unconditional `generate()`.
    model.eval()
    reservoir: dict[str, list[torch.Tensor]] = {"X": [], "A": [], "T": [], "mask": []}
    target = int(train_cfg.reservoir_size)
    for batch in train_dl:
        reservoir["X"].append(batch.X.detach().cpu())
        reservoir["A"].append(batch.A.detach().cpu())
        reservoir["T"].append(batch.T.detach().cpu())
        reservoir["mask"].append(batch.mask.detach().cpu())
        if sum(x.shape[0] for x in reservoir["X"]) >= target:
            break
    if reservoir["X"]:
        Xr = torch.cat(reservoir["X"], dim=0)[:target]
        Ar = torch.cat(reservoir["A"], dim=0)[:target]
        Tr = torch.cat(reservoir["T"], dim=0)[:target]
        Mr = torch.cat(reservoir["mask"], dim=0)[:target]
        model._reservoir = {"X": Xr, "A": Ar, "T": Tr, "mask": Mr}

        x_mean = Xr.float().mean(dim=0)
        x_std = Xr.float().std(dim=0, unbiased=False).clamp(min=1e-6)
        model._x_mean = x_mean
        model._x_std = x_std

    return {"history": history}
