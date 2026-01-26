from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from .data import (
    IRTSyntheticDataset,
    NPZSequenceDataset,
    SyntheticEduDataset,
    TrajectoryBatch,
    compute_lengths,
    make_dataloader,
    move_batch,
)
from .causal_estimators import GFormula, IPTWMSM
from .model.discriminators import SequenceDiscriminator, SequenceDiscriminatorConfig
from .model.losses import (
    bernoulli_nll_from_logits,
    grad_reverse,
    mmd_rbf,
    supervisor_mse,
    treatment_ce_loss,
    wgan_g_loss,
    wgan_gp_d_loss,
)
from .model.policy import DoIntervention, TreatmentClassifier
from .model.scm_generator import SCMGenerator, SCMGeneratorConfig


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor, eps: float) -> torch.Tensor:
    dist2 = torch.cdist(x, y) ** 2
    return torch.exp(-dist2 / (2.0 * sigma**2 + eps))


def weighted_mmd_rbf(
    x: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
    *,
    sigma: float,
    eps: float,
) -> torch.Tensor:
    if x.ndim == 1:
        x = x[:, None]
    if z.ndim == 1:
        z = z[:, None]

    x = x.float()
    z = z.float()
    w = w.float()

    if x.numel() == 0 or z.numel() == 0:
        return torch.tensor(0.0, device=x.device)

    w = w / (w.sum() + eps)
    sigma_t = torch.tensor(float(sigma), device=x.device)

    k_xx = _rbf_kernel(x, x, sigma_t, eps)
    k_zz = _rbf_kernel(z, z, sigma_t, eps)
    k_xz = _rbf_kernel(x, z, sigma_t, eps)

    term_xx = (w[:, None] * w[None, :] * k_xx).sum()
    term_zz = k_zz.mean()
    term_xz = (w[:, None] * k_xz).sum() / float(z.shape[0])
    return (term_xx + term_zz - 2.0 * term_xz).clamp(min=0.0)


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
    p = argparse.ArgumentParser(description="Dynamic SCM × Sequence GAN × Causal constraints (PyTorch scaffold)")
    p.add_argument("--data_npz", type=str, default=None, help="Path to padded .npz with X,A,T,Y,(M)")
    p.add_argument("--out_dir", type=str, default="runs/exp1")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--seed", type=int, default=0)

    # Synthetic fallback
    p.add_argument("--synthetic_n", type=int, default=2048)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--d_x", type=int, default=8)
    p.add_argument("--a_vocab_size", type=int, default=50)
    p.add_argument("--t_vocab_size", type=int, default=3)
    p.add_argument("--synthetic_kind", type=str, default="irt", choices=["logistic", "irt"])
    p.add_argument("--synthetic_gamma", type=float, default=0.8, help="IRT sim: action(1) effect size")
    p.add_argument("--synthetic_lr", type=float, default=0.05, help="IRT sim: latent update lr")
    p.add_argument("--synthetic_delta", type=float, default=0.02, help="IRT sim: latent update action gain")
    p.add_argument("--synthetic_confounding", type=float, default=1.0, help="IRT sim: higher -> more selection bias")
    p.add_argument("--synthetic_noise", type=float, default=0.02, help="IRT sim: latent noise std")

    # Model sizes
    p.add_argument("--d_k", type=int, default=64)
    p.add_argument("--k_c_ratio", type=float, default=0.5)
    p.add_argument("--d_eps", type=int, default=16)
    p.add_argument("--a_emb_dim", type=int, default=32)
    p.add_argument("--t_emb_dim", type=int, default=16)
    p.add_argument("--d_disc_h", type=int, default=128)

    p.add_argument("--dynamics", type=str, default="gru", choices=["gru", "mlp", "transformer"])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mlp_hidden", type=int, default=128)
    p.add_argument("--y_in_dynamics", action="store_true", help="Let K_{t+1} depend on Y_t (DKT-style feedback)")

    # Transformer dynamics knobs (when --dynamics transformer)
    p.add_argument("--tf_n_layers", type=int, default=2)
    p.add_argument("--tf_n_heads", type=int, default=4)
    p.add_argument("--tf_ffn_hidden", type=int, default=256)
    p.add_argument("--tf_max_seq_len", type=int, default=512)

    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_g", type=float, default=1e-3)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--lr_advT", type=float, default=1e-4)
    p.add_argument("--epochs_a", type=int, default=2, help="Stage A supervised pretrain")
    p.add_argument("--epochs_b", type=int, default=2, help="Stage B add adversarial")
    p.add_argument("--epochs_c", type=int, default=2, help="Stage C add causal regularizers")
    p.add_argument("--gp_weight", type=float, default=10.0)

    # Loss weights (typical warm-up schedule: increase do/advT in stage C)
    p.add_argument("--w_y", type=float, default=1.0)
    p.add_argument("--w_sup", type=float, default=1.0)
    p.add_argument("--w_adv", type=float, default=0.5)
    p.add_argument("--w_do", type=float, default=0.5)
    p.add_argument("--w_advT", type=float, default=0.1)
    p.add_argument("--w_advT_causal", type=float, default=None)
    p.add_argument(
        "--advT_on_kc",
        action="store_true",
        default=False,
        help="Enable GRL on K_c (off by default; can hurt adaptive-policy settings).",
    )
    p.add_argument("--w_t_pred_spurious", type=float, default=0.1)
    p.add_argument("--w_cf", type=float, default=0.0)
    p.add_argument("--grl_lambda", type=float, default=1.0)

    # Causal regularizers knobs
    p.add_argument("--do_time_index", type=int, default=5, help="time index for do-alignment (demo)")
    p.add_argument("--do_time_sampling", type=str, default="random", choices=["fixed", "random", "all"])
    p.add_argument("--do_num_time_samples", type=int, default=1)
    p.add_argument("--do_min_arm_samples", type=int, default=8)
    p.add_argument("--do_actions", type=str, default="0,1", help="comma-separated action ids")
    p.add_argument("--cf_num_time_samples", type=int, default=1, help="counterfactual time samples per batch")
    p.add_argument("--causal_every", type=int, default=20, help="compute do-loss every N steps")
    p.add_argument("--cf_every", type=int, default=20, help="compute cf-loss every N steps")
    p.add_argument("--do_horizon", type=int, default=5, help="multi-step do-alignment horizon")
    p.add_argument("--do_huber_delta", type=float, default=1.0, help="Huber delta for do-alignment loss")
    p.add_argument(
        "--do_weighted_mmd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use propensity-weighted MMD in single-step do-alignment fallback.",
    )
    p.add_argument("--do_eps", type=float, default=0.001)
    p.add_argument("--do_w_clip", type=float, default=10.0)
    p.add_argument("--do_sigma", type=float, default=1.0)
    p.add_argument("--do_ess_min", type=float, default=5.0)
    p.add_argument("--do_warmup_epochs", type=int, default=0)
    p.add_argument("--do_debug", action="store_true", default=False)
    p.add_argument("--ref_estimator", type=str, default="iptw_msm", choices=["gformula", "iptw_msm"])
    p.add_argument("--disc_input", type=str, default="logits", choices=["logits", "sampled"])

    args = p.parse_args(argv)
    w_advT_causal = float(args.w_advT if args.w_advT_causal is None else args.w_advT_causal)
    w_t_pred_spurious = float(args.w_t_pred_spurious)

    torch.manual_seed(args.seed)
    device = _device_from_arg(args.device)

    ds, meta = _build_dataset(args)
    dl = make_dataloader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = SCMGeneratorConfig(
        d_x=meta["d_x"],
        d_k=args.d_k,
        k_c_ratio=float(args.k_c_ratio),
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
        use_y_in_dynamics=bool(args.y_in_dynamics or (args.synthetic_kind == "irt" and args.data_npz is None)),
        dynamics=args.dynamics,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
        tf_n_layers=args.tf_n_layers,
        tf_n_heads=args.tf_n_heads,
        tf_ffn_hidden=args.tf_ffn_hidden,
        tf_max_seq_len=args.tf_max_seq_len,
    )
    gen = SCMGenerator(gen_cfg).to(device)

    disc_cfg = SequenceDiscriminatorConfig(
        d_x=meta["d_x"],
        a_is_discrete=meta["a_is_discrete"],
        a_vocab_size=int(meta["a_vocab_size"] or args.a_vocab_size),
        a_emb_dim=args.a_emb_dim,
        d_a=int(meta["d_a"]),
        t_is_discrete=meta["t_is_discrete"],
        t_vocab_size=int(meta["t_vocab_size"] or args.t_vocab_size),
        t_emb_dim=args.t_emb_dim,
        d_t=int(meta["d_t"]),
        d_y=1,
        d_h=args.d_disc_h,
        dropout=args.dropout,
    )
    d_seq = SequenceDiscriminator(disc_cfg).to(device)

    # Deconfounding regularizer module (Stage C): predict T_t from latent K_c / K_s.
    t_clf = getattr(gen, "t_head", None)
    t_policy_clf = getattr(gen, "t_policy_head", None)
    opt_tclf = None
    opt_tpolicy = None
    if w_advT_causal > 0.0 and t_clf is not None:
        opt_tclf = torch.optim.Adam(t_clf.parameters(), lr=args.lr_advT)
    if w_t_pred_spurious > 0.0 and t_policy_clf is not None:
        opt_tpolicy = torch.optim.Adam(t_policy_clf.parameters(), lr=args.lr_advT)

    head_params = []
    if t_clf is not None:
        head_params.extend(list(t_clf.parameters()))
    if t_policy_clf is not None:
        head_params.extend(list(t_policy_clf.parameters()))
    head_param_ids = {id(p) for p in head_params}
    gen_params = [p for p in gen.parameters() if id(p) not in head_param_ids]

    opt_g = torch.optim.Adam(gen_params, lr=args.lr_g)
    opt_d = torch.optim.Adam(d_seq.parameters(), lr=args.lr_d)

    do_actions = [int(x.strip()) for x in args.do_actions.split(",") if x.strip()]
    loss_do_fn = torch.nn.L1Loss()

    ref_estimator = None
    if args.w_do > 0.0 and int(args.do_horizon) > 0:
        X_full = torch.as_tensor(ds.X).float()
        A_full = torch.as_tensor(ds.A)
        T_full = torch.as_tensor(ds.T)
        Y_full = torch.as_tensor(ds.Y).float()
        M_full = torch.as_tensor(ds.M).float()
        full_batch = TrajectoryBatch(
            X=X_full, A=A_full, T=T_full, Y=Y_full, mask=M_full, lengths=compute_lengths(M_full)
        )
        if args.ref_estimator == "iptw_msm":
            ref_estimator = IPTWMSM(clip=10.0)
        else:
            ref_estimator = GFormula()
        ref_estimator.fit(full_batch)

    def run_stage(name: str, epochs: int, *, use_gan: bool, use_causal: bool) -> None:
        gen.train()
        d_seq.train()
        if t_clf is not None:
            t_clf.train()
        if t_policy_clf is not None:
            t_policy_clf.train()

        causal_every = max(1, int(args.causal_every))
        cf_every = max(1, int(args.cf_every))
        global_step = 0
        for ep in range(1, epochs + 1):
            for batch in dl:
                batch = move_batch(batch, device)
                X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask

                bsz, seq_len = A.shape[0], A.shape[1]
                should_causal = (global_step % causal_every == 0)
                should_cf = (global_step % cf_every == 0)
                tf_eps_mode = "zero" if name == "A" else "random"
                eps_seq = (
                    torch.zeros(bsz, seq_len, gen_cfg.d_eps, device=device)
                    if tf_eps_mode == "zero"
                    else torch.randn(bsz, seq_len, gen_cfg.d_eps, device=device)
                )

                # 1) Teacher forcing path (latent + MLE-ish Y fitting)
                tf = gen.teacher_forcing(x=X, a=A, t=T, y=Y, mask=M, eps=eps_seq, eps_mode=tf_eps_mode)
                y_logits_tf = tf["y_logits"]  # [B,T,1]
                k_tf = tf["k"]  # [B,T+1,d_k]
                k_c = tf.get("k_c")
                k_s = tf.get("k_s")
                if k_c is None or k_s is None:
                    k_c, k_s = gen.split_k(k_tf)

                loss_y = bernoulli_nll_from_logits(y_logits_tf, Y, mask=M)

                # Supervisor predicts K_{t+1} from K_t and inputs (A,T,X,eps).
                a_enc = gen.encode_a(A.view(-1, *A.shape[2:])).view(bsz, seq_len, -1)
                t_enc = gen.encode_t(T.view(-1, *T.shape[2:])).view(bsz, seq_len, -1)
                x_rep = gen.x_proj(X.float())[:, None, :].expand(bsz, seq_len, X.shape[1])
                if gen_cfg.use_y_in_dynamics:
                    y_dyn = Y.float()
                    if y_dyn.ndim == 2:
                        y_dyn = y_dyn[..., None]
                    inp = torch.cat([a_enc, t_enc, x_rep, y_dyn, eps_seq], dim=-1)
                else:
                    inp = torch.cat([a_enc, t_enc, x_rep, eps_seq], dim=-1)
                k_pred = gen.supervisor(k_tf[:, :-1, :], inp)
                loss_sup = supervisor_mse(k_pred, k_tf[:, 1:, :], mask=M)

                loss_adv = torch.tensor(0.0, device=device)
                if use_gan:
                    # 2) Generator rollout (replay observed actions) for adversarial learning
                    ro = gen.rollout(x=X, a=A, t_obs=T, mask=M, stochastic_y=False)
                    y_fake = ro["y"]  # probabilities

                    y_fake_in = y_fake
                    y_real_in = Y
                    if args.disc_input == "logits":
                        y_fake_in = ro.get("y_logits", None)
                        if y_fake_in is None:
                            y_fake_in = torch.log(torch.clamp(y_fake, 0.01, 0.99) / torch.clamp(1.0 - y_fake, 0.01, 0.99))
                        y_real_in = torch.log(torch.clamp(Y, 0.01, 0.99) / torch.clamp(1.0 - Y, 0.01, 0.99))
                    elif args.disc_input == "sampled":
                        u = torch.rand_like(y_fake)
                        y_hard = (y_fake > u).float()
                        y_fake_in = y_hard + (y_fake - y_fake.detach())

                    real_seq = d_seq.encode_inputs(x=X, a=A, t=T, y=y_real_in)
                    fake_seq = d_seq.encode_inputs(x=X, a=A, t=T, y=y_fake_in.detach())

                    # 3) Update discriminator
                    loss_d = wgan_gp_d_loss(d_seq, real_seq, fake_seq, gp_weight=args.gp_weight, mask=M)
                    opt_d.zero_grad(set_to_none=True)
                    loss_d.backward()
                    opt_d.step()

                    # 4) Generator adversarial loss (use y_fake WITHOUT detach)
                    fake_seq_g = d_seq.encode_inputs(x=X, a=A, t=T, y=y_fake_in)
                    loss_adv = wgan_g_loss(d_seq, fake_seq_g, mask=M)

                loss_do = torch.tensor(0.0, device=device)
                loss_advT = torch.tensor(0.0, device=device)
                loss_t_pred = torch.tensor(0.0, device=device)
                loss_cf = torch.tensor(0.0, device=device)
                do_weight = float(args.w_do)
                if use_causal and int(args.do_warmup_epochs) > 0:
                    do_weight *= min(1.0, ep / float(args.do_warmup_epochs))
                if use_causal:
                    if not (gen_cfg.t_is_discrete and T.ndim == 2):
                        # do-alignment and advT are implemented for discrete actions in this scaffold.
                        pass
                    else:
                        if do_weight > 0.0 and should_causal and int(args.do_horizon) > 0 and ref_estimator is not None:
                            t0 = int(torch.randint(low=0, high=seq_len, size=(1,), device=device).item())
                            action = int(
                                do_actions[
                                    int(torch.randint(low=0, high=len(do_actions), size=(1,), device=device).item())
                                ]
                            )
                            horizon_eff = min(int(args.do_horizon), max(0, seq_len - t0 - 1))
                            steps = t0 + horizon_eff + 1
                            t_do = torch.full((bsz,), action, device=device, dtype=T.dtype)
                            do_map = {t: t_do for t in range(t0, steps)}
                            ro_do = gen.rollout(  # type: ignore[attr-defined]
                                x=X,
                                a=A,
                                t_obs=T,
                                do_t=do_map,
                                mask=M,
                                eps=eps_seq,
                                steps=steps,
                                stochastic_y=False,
                                causal_only=True,
                            )
                            y_do = ro_do["y"]
                            if y_do.ndim == 3 and y_do.shape[-1] == 1:
                                y_do = y_do.squeeze(-1)
                            y_slice = y_do[:, t0:steps]
                            m_slice = M[:, t0:steps]
                            gen_mu = (y_slice * m_slice).sum(dim=0) / m_slice.sum(dim=0).clamp(min=1.0)

                            batch_ref = TrajectoryBatch(
                                X=X.detach().cpu(),
                                A=A.detach().cpu(),
                                T=T.detach().cpu(),
                                Y=Y.detach().cpu(),
                                mask=M.detach().cpu(),
                                lengths=compute_lengths(M.detach().cpu()),
                            )
                            ref = ref_estimator.estimate_do(
                                batch_ref,
                                t0=t0,
                                horizon=int(horizon_eff),
                                action=int(action),
                                subgroup={"name": "all"},
                                n_boot=1,
                                seed=int(args.seed),
                            )
                            ref_mu = torch.as_tensor(ref["mu"], device=device).float()
                            loss_do = loss_do_fn(gen_mu[: ref_mu.shape[0]], ref_mu)
                            print(
                                f"      loss_do_multistep={float(loss_do.item()):.4f} "
                                f"do_horizon={int(horizon_eff)} t0_sampled={t0}"
                            )
                        elif do_weight > 0.0 and should_causal:
                            # Fallback: single-step MMD alignment (legacy).
                            if args.do_time_sampling == "fixed":
                                do_time_indices = [int(args.do_time_index)]
                            elif args.do_time_sampling == "random":
                                do_time_indices = (
                                    torch.randint(low=0, high=seq_len, size=(int(args.do_num_time_samples),), device=device)
                                    .tolist()
                                )
                            else:
                                do_time_indices = list(range(seq_len))

                            x_feat = gen.x_proj(X.float())
                            t_probs_all = None
                            if args.do_weighted_mmd and t_clf is not None and k_c.shape[-1] > 0:
                                with torch.no_grad():
                                    t_logits_all = t_clf(k_c[:, :-1, :].detach())
                                    t_probs_all = torch.softmax(t_logits_all, dim=-1)
                            do_terms = 0
                            for t_idx in do_time_indices:
                                if not (0 <= t_idx < seq_len):
                                    continue
                                if t_probs_all is not None and t_idx >= t_probs_all.shape[1]:
                                    continue
                                valid = M[:, t_idx] > 0.5
                                valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
                                if valid_idx.numel() == 0:
                                    continue
                                if valid.sum().item() < args.do_min_arm_samples:
                                    continue

                                k_t = k_tf[:, t_idx, :]  # [B,d_k] (pre-treatment state)
                                a_enc_t = gen.encode_a(A[:, t_idx])

                                for a_val in do_actions:
                                    a_val = int(a_val)
                                    t_do = torch.full((bsz,), a_val, device=device, dtype=T.dtype)
                                    t_enc_do = gen.encode_t(t_do)
                                    y_inp_do = gen.build_y_input(k_t, a_enc_t, t_enc_do, x_feat, causal_only=True)
                                    y_prob_do = torch.sigmoid(gen.y_head(y_inp_do))  # [B,1]
                                    feat_do = torch.cat([x_feat, k_t, y_prob_do], dim=-1)[valid]

                                    sel = (T[:, t_idx].long() == a_val) & valid
                                    sel_idx = sel.nonzero(as_tuple=False).squeeze(-1)
                                    if sel_idx.numel() < args.do_min_arm_samples:
                                        continue
                                    y_real = Y[sel_idx, t_idx].float().view(-1, 1)
                                    feat_real = torch.cat([x_feat[sel_idx], k_t[sel_idx], y_real], dim=-1)
                                    if (not torch.isfinite(feat_real).all()) or (not torch.isfinite(feat_do).all()):
                                        continue
                                    if args.do_weighted_mmd and t_probs_all is not None:
                                        e = t_probs_all[sel_idx, t_idx, a_val].detach()
                                        e = e.clamp(min=args.do_eps, max=1.0 - args.do_eps)
                                        p_a = t_probs_all[valid_idx, t_idx, a_val].detach().mean().clamp(min=args.do_eps)
                                        w_raw = (p_a / e).clamp(max=args.do_w_clip)
                                        w = w_raw / (w_raw.sum() + args.do_eps)
                                        ess = 1.0 / (w.pow(2).sum() + args.do_eps)
                                        if ess < args.do_ess_min:
                                            if args.do_debug:
                                                print(
                                                    f"      skip_weighted_mmd t={t_idx} a={a_val} "
                                                    f"ess={float(ess.item()):.2f} n={int(sel.sum().item())} "
                                                    f"w_max={float(w_raw.max().item()):.2f}"
                                                )
                                            continue
                                        loss_do = loss_do + weighted_mmd_rbf(
                                            feat_real,
                                            feat_do,
                                            w,
                                            sigma=float(args.do_sigma),
                                            eps=float(args.do_eps),
                                        )
                                    else:
                                        loss_do = loss_do + mmd_rbf(feat_real, feat_do)
                                    do_terms += 1

                            if do_terms > 0:
                                loss_do = loss_do / float(do_terms)
                            else:
                                loss_do = torch.tensor(0.0, device=device)

                        # Optional GRL on K_c (disabled by default to avoid removing policy-dependent signal).
                        if (
                            w_advT_causal > 0.0
                            and args.advT_on_kc
                            and t_clf is not None
                            and opt_tclf is not None
                            and k_c.shape[-1] > 0
                        ):
                            with torch.no_grad():
                                k_repr_detached = k_c[:, :-1, :].detach()
                            t_logits_clf = t_clf(k_repr_detached)
                            loss_tclf = treatment_ce_loss(t_logits_clf, T.long(), mask=M)
                            opt_tclf.zero_grad(set_to_none=True)
                            loss_tclf.backward()
                            opt_tclf.step()

                            # For generator update: reverse gradients through K_c.
                            t_logits_adv = t_clf(grad_reverse(k_c[:, :-1, :], lambd=args.grl_lambda))
                            loss_advT = treatment_ce_loss(t_logits_adv, T.long(), mask=M)

                        if (
                            w_t_pred_spurious > 0.0
                            and t_policy_clf is not None
                            and opt_tpolicy is not None
                            and k_s.shape[-1] > 0
                        ):
                            with torch.no_grad():
                                k_s_detached = k_s[:, :-1, :].detach()
                            t_logits_policy = t_policy_clf(k_s_detached)
                            loss_tpolicy = treatment_ce_loss(t_logits_policy, T.long(), mask=M)
                            opt_tpolicy.zero_grad(set_to_none=True)
                            loss_tpolicy.backward()
                            opt_tpolicy.step()

                            t_logits_policy = t_policy_clf(k_s[:, :-1, :])
                            loss_t_pred = treatment_ce_loss(t_logits_policy, T.long(), mask=M)

                        # Counterfactual consistency: intervening at time t must not change K_{0:t}.
                        if args.w_cf > 0.0 and should_cf:
                            num_actions = int(meta["t_vocab_size"] or args.t_vocab_size)
                            cf_time_indices = (
                                torch.randint(low=0, high=seq_len, size=(int(args.cf_num_time_samples),), device=device)
                                .tolist()
                            )
                            ro_f = gen.rollout(x=X, a=A, t_obs=T, mask=M, eps=eps_seq, stochastic_y=False)
                            k_f = ro_f["k"]  # [B,T+1,d_k]
                            cf_terms = 0
                            for t_idx in cf_time_indices:
                                if not (0 <= t_idx < seq_len):
                                    continue
                                alt = (T[:, t_idx].long() + 1) % num_actions
                                ro_cf = gen.rollout(
                                    x=X, a=A, t_obs=T, do_t={t_idx: alt}, mask=M, eps=eps_seq, stochastic_y=False
                                )
                                k_cf = ro_cf["k"]
                                prefix_len = t_idx + 1  # compare K_0..K_t (pre-intervention prefix)
                                if prefix_len <= 0:
                                    continue
                                k1 = k_f[:, :prefix_len, :]
                                k2 = k_cf[:, :prefix_len, :]
                                mask_k = torch.cat([torch.ones(bsz, 1, device=device), M[:, :t_idx]], dim=1)
                                diff = (k1 - k2).pow(2).mean(dim=-1)  # [B, prefix_len]
                                loss_cf = loss_cf + (diff * mask_k).sum() / mask_k.sum().clamp(min=1.0)
                                cf_terms += 1
                            if cf_terms > 0:
                                loss_cf = loss_cf / float(cf_terms)

                # Generator update (Stage A: supervised + sup; Stage B/C: add adv + do)
                loss_g = args.w_y * loss_y + args.w_sup * loss_sup
                if use_gan:
                    loss_g = loss_g + args.w_adv * loss_adv
                if use_causal:
                    loss_g = loss_g + do_weight * loss_do
                    loss_g = loss_g + w_advT_causal * loss_advT
                    loss_g = loss_g + w_t_pred_spurious * loss_t_pred
                    loss_g = loss_g + args.w_cf * loss_cf

                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_g.step()
                global_step += 1

            ckpt = {
                "stage": name,
                "epoch": ep,
                "gen_cfg": asdict(gen_cfg),
                "disc_cfg": asdict(disc_cfg),
                "gen": gen.state_dict(),
                "d_seq": d_seq.state_dict(),
            }
            torch.save(ckpt, out_dir / f"ckpt_{name}_ep{ep}.pt")

    run_stage("A", args.epochs_a, use_gan=False, use_causal=False)
    run_stage("B", args.epochs_b, use_gan=True, use_causal=False)
    run_stage("C", args.epochs_c, use_gan=True, use_causal=True)

    print(f"Saved checkpoints to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
