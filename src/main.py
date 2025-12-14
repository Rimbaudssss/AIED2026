from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from .data import IRTSyntheticDataset, NPZSequenceDataset, SyntheticEduDataset, make_dataloader, move_batch
from .model.baselines import CRN, CRNConfig, RCGANConfig, RCGANGenerator, SeqDiffusion, SeqDiffusionConfig, SeqVAE, SeqVAEConfig
from .model.discriminators import SequenceDiscriminator, SequenceDiscriminatorConfig
from .model.losses import bernoulli_nll_from_logits, wgan_g_loss, wgan_gp_d_loss
from .model.scm_generator import SCMGenerator, SCMGeneratorConfig


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_dataset(args) -> tuple[torch.utils.data.Dataset, dict]:
    if args.data_npz:
        ds = NPZSequenceDataset(args.data_npz)
        meta = {
            "d_x": ds.d_x,
            "seq_len": ds.seq_len,
            "a_is_discrete": ds.a_is_discrete,
            "t_is_discrete": ds.t_is_discrete,
            "a_vocab_size": ds.a_vocab_size,
            "t_vocab_size": ds.t_vocab_size,
            "d_a": ds.d_a,
            "d_t": ds.d_t,
            "d_y": ds.d_y,
        }
        return ds, meta

    if args.synthetic_kind == "logistic":
        ds = SyntheticEduDataset(
            n=args.synthetic_n,
            seq_len=args.seq_len,
            d_x=args.d_x,
            a_vocab_size=args.a_vocab_size,
            t_vocab_size=args.t_vocab_size,
            seed=args.seed,
        )
    else:
        ds = IRTSyntheticDataset(
            n=args.synthetic_n,
            seq_len=args.seq_len,
            d_x=args.d_x,
            a_vocab_size=args.a_vocab_size,
            t_vocab_size=args.t_vocab_size,
            gamma=args.synthetic_gamma,
            lr=args.synthetic_lr,
            delta=args.synthetic_delta,
            confounding=args.synthetic_confounding,
            noise_std=args.synthetic_noise,
            seed=args.seed,
        )
    meta = {
        "d_x": ds.d_x,
        "seq_len": ds.seq_len,
        "a_is_discrete": ds.a_is_discrete,
        "t_is_discrete": ds.t_is_discrete,
        "a_vocab_size": ds.a_vocab_size,
        "t_vocab_size": ds.t_vocab_size,
        "d_a": ds.d_a,
        "d_t": ds.d_t,
        "d_y": ds.d_y,
    }
    return ds, meta


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="AIED2026 unified trainer (SCM + baselines)")
    p.add_argument("--model", type=str, required=True, choices=["scm", "rcgan", "vae", "diffusion", "crn"])
    p.add_argument("--data_npz", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="runs/exp1")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)

    # Data / synthetic
    p.add_argument("--synthetic_kind", type=str, default="irt", choices=["logistic", "irt"])
    p.add_argument("--synthetic_n", type=int, default=4096)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--d_x", type=int, default=8)
    p.add_argument("--a_vocab_size", type=int, default=50)
    p.add_argument("--t_vocab_size", type=int, default=3)
    p.add_argument("--synthetic_gamma", type=float, default=0.8)
    p.add_argument("--synthetic_lr", type=float, default=0.05)
    p.add_argument("--synthetic_delta", type=float, default=0.02)
    p.add_argument("--synthetic_confounding", type=float, default=1.0)
    p.add_argument("--synthetic_noise", type=float, default=0.02)

    # Shared training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10, help="Used for non-GAN models (vae/diffusion/crn)")
    p.add_argument("--lr", type=float, default=1e-3, help="Used for non-GAN models (vae/diffusion/crn)")

    # GAN training
    p.add_argument("--epochs_a", type=int, default=2)
    p.add_argument("--epochs_b", type=int, default=2)
    p.add_argument("--epochs_c", type=int, default=2)
    p.add_argument("--lr_g", type=float, default=1e-3)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--gp_weight", type=float, default=10.0)
    p.add_argument("--w_y", type=float, default=1.0)
    p.add_argument("--w_adv", type=float, default=0.5)

    # Embeddings
    p.add_argument("--a_emb_dim", type=int, default=32)
    p.add_argument("--t_emb_dim", type=int, default=16)

    # SCM config
    p.add_argument("--d_k", type=int, default=64)
    p.add_argument("--d_eps", type=int, default=16)
    p.add_argument("--dynamics", type=str, default="gru", choices=["gru", "mlp", "transformer"])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mlp_hidden", type=int, default=128)
    p.add_argument("--tf_n_layers", type=int, default=2)
    p.add_argument("--tf_n_heads", type=int, default=4)
    p.add_argument("--tf_ffn_hidden", type=int, default=256)
    p.add_argument("--tf_max_seq_len", type=int, default=512)
    p.add_argument("--y_in_dynamics", action="store_true")
    p.add_argument("--w_sup", type=float, default=1.0)

    # RCGAN config
    p.add_argument("--rcgan_hidden", type=int, default=128)
    p.add_argument("--rcgan_rnn", type=str, default="gru", choices=["gru", "lstm"])
    p.add_argument("--rcgan_use_prev_y", action="store_true")

    # VAE config
    p.add_argument("--vae_z_dim", type=int, default=32)
    p.add_argument("--vae_enc_hidden", type=int, default=128)
    p.add_argument("--vae_dec_hidden", type=int, default=128)
    p.add_argument("--vae_kl_weight", type=float, default=1.0)

    # Diffusion config
    p.add_argument("--diff_steps", type=int, default=100)
    p.add_argument("--diff_beta_start", type=float, default=1e-4)
    p.add_argument("--diff_beta_end", type=float, default=0.02)
    p.add_argument("--diff_time_emb_dim", type=int, default=64)
    p.add_argument("--diff_model_dim", type=int, default=128)
    p.add_argument("--diff_backbone", type=str, default="mlp", choices=["mlp", "transformer"])
    p.add_argument("--diff_n_layers", type=int, default=2)
    p.add_argument("--diff_n_heads", type=int, default=4)

    # CRN config
    p.add_argument("--crn_hidden", type=int, default=128)
    p.add_argument("--crn_grl_lambda", type=float, default=1.0)
    p.add_argument("--crn_w_treat", type=float, default=1.0)

    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = _device_from_arg(args.device)

    ds, meta = _build_dataset(args)
    dl = make_dataloader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model in {"scm", "rcgan"}:
        disc_cfg = dict(
            d_x=meta["d_x"],
            a_is_discrete=meta["a_is_discrete"],
            a_vocab_size=int(meta["a_vocab_size"] or args.a_vocab_size),
            d_a=int(meta["d_a"]),
            t_is_discrete=meta["t_is_discrete"],
            t_vocab_size=int(meta["t_vocab_size"] or args.t_vocab_size),
            d_t=int(meta["d_t"]),
            a_emb_dim=args.a_emb_dim,
            t_emb_dim=args.t_emb_dim,
            d_h=128,
            dropout=args.dropout,
        )
        disc = SequenceDiscriminator(
            SequenceDiscriminatorConfig(**{**disc_cfg, "d_y": 1, "num_layers": 1})
        ).to(device)

        if args.model == "scm":
            gen_cfg = SCMGeneratorConfig(
                d_x=meta["d_x"],
                d_k=args.d_k,
                d_eps=args.d_eps,
                a_is_discrete=meta["a_is_discrete"],
                a_vocab_size=int(meta["a_vocab_size"] or args.a_vocab_size),
                a_emb_dim=args.a_emb_dim,
                d_a=int(meta["d_a"]),
                t_is_discrete=meta["t_is_discrete"],
                t_vocab_size=int(meta["t_vocab_size"] or args.t_vocab_size),
                t_emb_dim=args.t_emb_dim,
                d_t=int(meta["d_t"]),
                y_dist="bernoulli",
                d_y=int(meta["d_y"]),
                use_y_in_dynamics=bool(args.y_in_dynamics),
                dynamics=args.dynamics,
                mlp_hidden=args.mlp_hidden,
                dropout=args.dropout,
                tf_n_layers=args.tf_n_layers,
                tf_n_heads=args.tf_n_heads,
                tf_ffn_hidden=args.tf_ffn_hidden,
                tf_max_seq_len=args.tf_max_seq_len,
            )
            gen: torch.nn.Module = SCMGenerator(gen_cfg).to(device)
        else:
            gen_cfg = RCGANConfig(
                d_x=meta["d_x"],
                d_eps=args.d_eps,
                a_is_discrete=meta["a_is_discrete"],
                a_vocab_size=int(meta["a_vocab_size"] or args.a_vocab_size),
                a_emb_dim=args.a_emb_dim,
                d_a=int(meta["d_a"]),
                t_is_discrete=meta["t_is_discrete"],
                t_vocab_size=int(meta["t_vocab_size"] or args.t_vocab_size),
                t_emb_dim=args.t_emb_dim,
                d_t=int(meta["d_t"]),
                rnn=args.rcgan_rnn,
                d_h=args.rcgan_hidden,
                dropout=args.dropout,
                use_prev_y=bool(args.rcgan_use_prev_y),
            )
            gen = RCGANGenerator(gen_cfg).to(device)

        opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr_g)
        opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d)

        def save_ckpt(stage: str, epoch: int) -> None:
            ckpt = {
                "model": args.model,
                "stage": stage,
                "epoch": epoch,
                "model_cfg": asdict(gen_cfg),
                "model_state": gen.state_dict(),
                "disc_cfg": {k: v for k, v in disc_cfg.items()},
                "disc_state": disc.state_dict(),
            }
            if args.model == "scm":
                ckpt["gen_cfg"] = asdict(gen_cfg)
                ckpt["gen"] = gen.state_dict()
            torch.save(ckpt, out_dir / f"ckpt_{args.model}_{stage}_ep{epoch}.pt")

        # Stage A: supervised pretrain
        gen.train()
        for ep in range(1, args.epochs_a + 1):
            for batch in dl:
                batch = move_batch(batch, device)
                X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask

                if args.model == "scm":
                    bsz, seq_len = A.shape[0], A.shape[1]
                    eps = torch.zeros(bsz, seq_len, args.d_eps, device=device)
                    out = gen.teacher_forcing(x=X, a=A, t=T, y=Y, mask=M, eps=eps, eps_mode="zero")  # type: ignore[attr-defined]
                    loss_y = bernoulli_nll_from_logits(out["y_logits"], Y, mask=M)

                    # Supervisor (TimeGAN-style)
                    k_tf = out["k"]
                    a_enc = gen.encode_a(A.view(-1, *A.shape[2:])).view(bsz, seq_len, -1)  # type: ignore[attr-defined]
                    t_enc = gen.encode_t(T.view(-1, *T.shape[2:])).view(bsz, seq_len, -1)  # type: ignore[attr-defined]
                    x_rep = gen.x_proj(X.float())[:, None, :].expand(bsz, seq_len, X.shape[1])  # type: ignore[attr-defined]
                    if gen_cfg.use_y_in_dynamics:
                        y_dyn = Y.float()
                        if y_dyn.ndim == 2:
                            y_dyn = y_dyn[..., None]
                        inp = torch.cat([a_enc, t_enc, x_rep, y_dyn, eps], dim=-1)
                    else:
                        inp = torch.cat([a_enc, t_enc, x_rep, eps], dim=-1)
                    k_pred = gen.supervisor(k_tf[:, :-1, :], inp)  # type: ignore[attr-defined]
                    loss_sup = ((k_pred - k_tf[:, 1:, :]).pow(2).mean(dim=-1) * M).sum() / M.sum().clamp(min=1.0)

                    loss = args.w_y * loss_y + args.w_sup * loss_sup
                else:
                    out = gen.teacher_forcing(x=X, a=A, t=T, y=Y, mask=M, stochastic_y=False)  # type: ignore[attr-defined]
                    loss = bernoulli_nll_from_logits(out["y_logits"], Y, mask=M)

                opt_g.zero_grad(set_to_none=True)
                loss.backward()
                opt_g.step()

            save_ckpt("A", ep)

        # Stage B/C: adversarial training (WGAN-GP) + keep supervised term
        for stage_name, epochs in [("B", args.epochs_b), ("C", args.epochs_c)]:
            gen.train()
            disc.train()
            for ep in range(1, epochs + 1):
                for batch in dl:
                    batch = move_batch(batch, device)
                    X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask

                    # Generator rollout under observed actions
                    ro = gen.rollout(x=X, a=A, t_obs=T, mask=M, stochastic_y=False)  # type: ignore[attr-defined]
                    y_fake = ro["y"]

                    real_seq = disc.encode_inputs(x=X, a=A, t=T, y=Y)
                    fake_seq = disc.encode_inputs(x=X, a=A, t=T, y=y_fake.detach())

                    # Update D
                    loss_d = wgan_gp_d_loss(disc, real_seq, fake_seq, gp_weight=args.gp_weight, mask=M)
                    opt_d.zero_grad(set_to_none=True)
                    loss_d.backward()
                    opt_d.step()

                    # Supervised loss (keeps conditional mapping grounded)
                    if args.model == "scm":
                        bsz, seq_len = A.shape[0], A.shape[1]
                        eps = torch.randn(bsz, seq_len, args.d_eps, device=device)
                        tf = gen.teacher_forcing(x=X, a=A, t=T, y=Y, mask=M, eps=eps, eps_mode="random")  # type: ignore[attr-defined]
                        loss_y = bernoulli_nll_from_logits(tf["y_logits"], Y, mask=M)

                        k_tf = tf["k"]
                        a_enc = gen.encode_a(A.view(-1, *A.shape[2:])).view(bsz, seq_len, -1)  # type: ignore[attr-defined]
                        t_enc = gen.encode_t(T.view(-1, *T.shape[2:])).view(bsz, seq_len, -1)  # type: ignore[attr-defined]
                        x_rep = gen.x_proj(X.float())[:, None, :].expand(bsz, seq_len, X.shape[1])  # type: ignore[attr-defined]
                        if gen_cfg.use_y_in_dynamics:
                            y_dyn = Y.float()
                            if y_dyn.ndim == 2:
                                y_dyn = y_dyn[..., None]
                            inp = torch.cat([a_enc, t_enc, x_rep, y_dyn, eps], dim=-1)
                        else:
                            inp = torch.cat([a_enc, t_enc, x_rep, eps], dim=-1)
                        k_pred = gen.supervisor(k_tf[:, :-1, :], inp)  # type: ignore[attr-defined]
                        loss_sup = ((k_pred - k_tf[:, 1:, :]).pow(2).mean(dim=-1) * M).sum() / M.sum().clamp(min=1.0)
                        loss_sup = loss_sup * args.w_sup
                    else:
                        tf = gen.teacher_forcing(x=X, a=A, t=T, y=Y, mask=M, stochastic_y=False)  # type: ignore[attr-defined]
                        loss_y = bernoulli_nll_from_logits(tf["y_logits"], Y, mask=M)
                        loss_sup = torch.tensor(0.0, device=device)

                    # Update G
                    fake_seq_g = disc.encode_inputs(x=X, a=A, t=T, y=y_fake)
                    loss_adv = wgan_g_loss(disc, fake_seq_g, mask=M)
                    loss_g = args.w_y * loss_y + args.w_adv * loss_adv + loss_sup

                    opt_g.zero_grad(set_to_none=True)
                    loss_g.backward()
                    opt_g.step()

                save_ckpt(stage_name, ep)

        print(f"Saved checkpoints to: {out_dir.resolve()}")
        return 0

    if args.model == "vae":
        cfg = SeqVAEConfig(
            d_x=meta["d_x"],
            z_dim=args.vae_z_dim,
            enc_hidden=args.vae_enc_hidden,
            dec_hidden=args.vae_dec_hidden,
            dropout=args.dropout,
            use_prev_y=True,
            a_is_discrete=meta["a_is_discrete"],
            a_vocab_size=int(meta["a_vocab_size"] or args.a_vocab_size),
            a_emb_dim=args.a_emb_dim,
            d_a=int(meta["d_a"]),
            t_is_discrete=meta["t_is_discrete"],
            t_vocab_size=int(meta["t_vocab_size"] or args.t_vocab_size),
            t_emb_dim=args.t_emb_dim,
            d_t=int(meta["d_t"]),
        )
        model = SeqVAE(cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for ep in range(1, args.epochs + 1):
            model.train()
            for batch in dl:
                batch = move_batch(batch, device)
                losses = model.elbo_loss(x=batch.X, a=batch.A, t=batch.T, y=batch.Y, mask=batch.mask, kl_weight=args.vae_kl_weight)
                opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                opt.step()
            ckpt = {"model": "vae", "epoch": ep, "model_cfg": asdict(cfg), "model_state": model.state_dict()}
            torch.save(ckpt, out_dir / f"ckpt_vae_ep{ep}.pt")
        print(f"Saved checkpoints to: {out_dir.resolve()}")
        return 0

    if args.model == "diffusion":
        cfg = SeqDiffusionConfig(
            d_x=meta["d_x"],
            num_steps=args.diff_steps,
            beta_start=args.diff_beta_start,
            beta_end=args.diff_beta_end,
            time_emb_dim=args.diff_time_emb_dim,
            model_dim=args.diff_model_dim,
            backbone=args.diff_backbone,
            n_layers=args.diff_n_layers,
            n_heads=args.diff_n_heads,
            dropout=args.dropout,
            a_is_discrete=meta["a_is_discrete"],
            a_vocab_size=int(meta["a_vocab_size"] or args.a_vocab_size),
            a_emb_dim=args.a_emb_dim,
            d_a=int(meta["d_a"]),
            t_is_discrete=meta["t_is_discrete"],
            t_vocab_size=int(meta["t_vocab_size"] or args.t_vocab_size),
            t_emb_dim=args.t_emb_dim,
            d_t=int(meta["d_t"]),
            y_dim=1,
        )
        model = SeqDiffusion(cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for ep in range(1, args.epochs + 1):
            model.train()
            for batch in dl:
                batch = move_batch(batch, device)
                loss = model.denoising_loss(x=batch.X, a=batch.A, t=batch.T, y=batch.Y, mask=batch.mask)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            ckpt = {"model": "diffusion", "epoch": ep, "model_cfg": asdict(cfg), "model_state": model.state_dict()}
            torch.save(ckpt, out_dir / f"ckpt_diffusion_ep{ep}.pt")
        print(f"Saved checkpoints to: {out_dir.resolve()}")
        return 0

    if args.model == "crn":
        cfg = CRNConfig(
            d_x=meta["d_x"],
            d_h=args.crn_hidden,
            dropout=args.dropout,
            grl_lambda=args.crn_grl_lambda,
            a_is_discrete=meta["a_is_discrete"],
            a_vocab_size=int(meta["a_vocab_size"] or args.a_vocab_size),
            a_emb_dim=args.a_emb_dim,
            d_a=int(meta["d_a"]),
            t_is_discrete=meta["t_is_discrete"],
            t_vocab_size=int(meta["t_vocab_size"] or args.t_vocab_size),
            t_emb_dim=args.t_emb_dim,
            d_t=int(meta["d_t"]),
        )
        model = CRN(cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for ep in range(1, args.epochs + 1):
            model.train()
            for batch in dl:
                batch = move_batch(batch, device)
                losses = model.loss(x=batch.X, a=batch.A, t=batch.T, y=batch.Y, mask=batch.mask, w_treat=args.crn_w_treat)
                opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                opt.step()
            ckpt = {"model": "crn", "epoch": ep, "model_cfg": asdict(cfg), "model_state": model.state_dict()}
            torch.save(ckpt, out_dir / f"ckpt_crn_ep{ep}.pt")
        print(f"Saved checkpoints to: {out_dir.resolve()}")
        return 0

    raise RuntimeError(f"Unhandled model={args.model}")


if __name__ == "__main__":
    raise SystemExit(main())
