"""Generate a DKT-based semi-synthetic dataset using a pretrained LSTM oracle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


class LSTMDKTOracle(nn.Module):
    def __init__(self, n_skills: int, embed_dim: int, hidden_dim: int, d_x: int = 0):
        super().__init__()
        self.n_skills = int(n_skills)
        self.embedding = nn.Embedding(self.n_skills * 2 + 1, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, self.n_skills)
        self.h0 = nn.Linear(int(d_x), hidden_dim) if d_x > 0 else None
        self.c0 = nn.Linear(int(d_x), hidden_dim) if d_x > 0 else None

    def init_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = int(x.shape[0])
        device = x.device
        if self.h0 is None or self.c0 is None:
            h0 = torch.zeros(1, bsz, self.lstm.hidden_size, device=device)
            c0 = torch.zeros(1, bsz, self.lstm.hidden_size, device=device)
            return h0, c0
        h = torch.tanh(self.h0(x.float())).unsqueeze(0)
        c = torch.tanh(self.c0(x.float())).unsqueeze(0)
        return h, c

    def forward(self, input_seq: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None):
        emb = self.embedding(input_seq)
        out, hidden = self.lstm(emb, hidden)
        logits = self.out(out)
        return logits, hidden


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _parse_int_list(text: str) -> list[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_open_loop_policies(seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    t_switch = int(0.2 * seq_len)
    names = [b"never", b"always", b"early_on", b"late_on"]
    never = np.zeros(seq_len, dtype=np.int64)
    always = np.ones(seq_len, dtype=np.int64)
    early_on = np.concatenate([np.ones(t_switch, dtype=np.int64), np.zeros(seq_len - t_switch, dtype=np.int64)])
    late_on = np.concatenate([np.zeros(t_switch, dtype=np.int64), np.ones(seq_len - t_switch, dtype=np.int64)])
    T = np.stack([never, always, early_on, late_on], axis=0)
    return np.array(names, dtype="S"), T


@torch.no_grad()
def _simulate_with_T(
    *,
    oracle: LSTMDKTOracle,
    X: np.ndarray,
    A: np.ndarray,
    T: np.ndarray,
    u_y: np.ndarray,
    u_t: np.ndarray | None,
    f: np.ndarray | None,
    t_effect: float,
    confounding: float,
    hidden_f_confounding: float,
    hidden_f_y_effect: float,
    device: torch.device,
    batch_size: int,
    return_t: bool,
) -> dict[str, np.ndarray]:
    n, seq_len = A.shape
    n_skills = oracle.n_skills
    start_token = n_skills * 2

    Y_prob = np.zeros((n, seq_len), dtype=np.float32)
    Y = np.zeros((n, seq_len), dtype=np.float32)
    T_out = np.zeros((n, seq_len), dtype=np.int64) if return_t else None

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        sl = slice(start, end)
        x_b = torch.as_tensor(X[sl], device=device)
        a_b = torch.as_tensor(A[sl], device=device)
        t_b = torch.as_tensor(T[sl], device=device)
        u_b = torch.as_tensor(u_y[sl], device=device)
        u_t_b = torch.as_tensor(u_t[sl], device=device) if u_t is not None else None
        f_b = torch.as_tensor(f[sl], device=device) if f is not None else None

        hidden = oracle.init_state(x_b)
        prev_token = torch.full((end - start, 1), start_token, device=device, dtype=torch.long)

        for t in range(seq_len):
            logits, hidden = oracle(prev_token, hidden)
            cur_a = a_b[:, t]
            logit_a = logits[:, 0, :].gather(1, cur_a[:, None]).squeeze(1)
            prob = torch.sigmoid(logit_a)
            f_t = f_b[:, t] if f_b is not None else 0.0

            if return_t:
                p_help = torch.sigmoid(
                    confounding * (0.5 - prob) + float(hidden_f_confounding) * f_t
                )
                u_src = u_t_b[:, t] if u_t_b is not None else u_b[:, t]
                t_t = (u_src < p_help).long()
            else:
                t_t = t_b[:, t].long()

            logit_eff = logit_a + float(t_effect) * t_t.float() - float(hidden_f_y_effect) * f_t
            p = torch.sigmoid(logit_eff)
            y_t = (u_b[:, t] < p).float()

            Y_prob[sl, t] = p.detach().cpu().numpy().astype(np.float32)
            Y[sl, t] = y_t.detach().cpu().numpy().astype(np.float32)
            if T_out is not None:
                T_out[sl, t] = t_t.detach().cpu().numpy().astype(np.int64)

            prev_token = cur_a + y_t.long() * n_skills
            prev_token = prev_token.unsqueeze(1)

    out = {"Y_prob": Y_prob, "Y": Y}
    if T_out is not None:
        out["T"] = T_out
    return out


def simulate_factual(
    *,
    oracle: LSTMDKTOracle,
    rng: np.random.Generator,
    n: int,
    seq_len: int,
    d_x: int,
    n_skills: int,
    t_effect: float,
    confounding: float,
    hidden_f_confounding: float,
    hidden_f_y_effect: float,
    f_rho: float,
    f_std: float,
    f_init_std: float,
    f_clip: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    X = rng.normal(0.0, 1.0, size=(n, d_x)).astype(np.float32)
    A = rng.integers(low=0, high=n_skills, size=(n, seq_len), dtype=np.int64)
    u_y = rng.random(size=(n, seq_len)).astype(np.float32)
    u_t = rng.random(size=(n, seq_len)).astype(np.float32)
    F = np.zeros((n, seq_len), dtype=np.float32)
    F[:, 0] = rng.normal(0.0, float(f_init_std), size=n).astype(np.float32)
    for t in range(1, seq_len):
        eps = rng.normal(0.0, float(f_std), size=n).astype(np.float32)
        F[:, t] = float(f_rho) * F[:, t - 1] + eps
    if float(f_clip) > 0.0:
        F = np.clip(F, -float(f_clip), float(f_clip)).astype(np.float32)

    sim = _simulate_with_T(
        oracle=oracle,
        X=X,
        A=A,
        T=np.zeros((n, seq_len), dtype=np.int64),
        u_y=u_y,
        u_t=u_t,
        f=F,
        t_effect=float(t_effect),
        confounding=float(confounding),
        hidden_f_confounding=float(hidden_f_confounding),
        hidden_f_y_effect=float(hidden_f_y_effect),
        device=device,
        batch_size=int(batch_size),
        return_t=True,
    )

    M = np.ones((n, seq_len), dtype=np.float32)
    return {"X": X, "A": A, "T": sim["T"], "Y": sim["Y"], "M": M, "u_y": u_y, "F": F}


def simulate_given_T(
    *,
    oracle: LSTMDKTOracle,
    X: np.ndarray,
    A: np.ndarray,
    T: np.ndarray,
    u_y: np.ndarray,
    f: np.ndarray | None,
    t_effect: float,
    hidden_f_confounding: float,
    hidden_f_y_effect: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    sim = _simulate_with_T(
        oracle=oracle,
        X=X,
        A=A,
        T=T,
        u_y=u_y,
        u_t=None,
        f=f,
        t_effect=float(t_effect),
        confounding=0.0,
        hidden_f_confounding=float(hidden_f_confounding),
        hidden_f_y_effect=float(hidden_f_y_effect),
        device=device,
        batch_size=int(batch_size),
        return_t=False,
    )
    return sim["Y_prob"]


def _iter_t0(seq_len: int, t0_list: Iterable[int] | None) -> list[int]:
    if not t0_list:
        return list(range(seq_len))
    return [t for t in t0_list if 0 <= int(t) < seq_len]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None, help="Output .npz path")
    ap.add_argument("--oracle_path", type=str, default=None, help="Path to pretrained DKT LSTM weights")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seq_len", type=int, default=50)
    ap.add_argument("--d_x", type=int, default=8)
    ap.add_argument("--n_skills", type=int, default=50)
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--t_effect", type=float, default=2.0)
    ap.add_argument("--confounding", type=float, default=3.0)
    ap.add_argument("--hidden_f_confounding", type=float, default=2.0)
    ap.add_argument("--hidden_f_y_effect", type=float, default=2.0)
    ap.add_argument("--f_rho", type=float, default=0.9)
    ap.add_argument("--f_std", type=float, default=0.3)
    ap.add_argument("--f_init_std", type=float, default=0.5)
    ap.add_argument("--f_clip", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--oracle_strict", action="store_true", help="Require exact oracle checkpoint match")
    ap.add_argument("--oracle_use_cfg", action="store_true", help="Use model_cfg from oracle checkpoint if present")
    ap.add_argument("--with_oracle", action="store_true", help="Also compute Y_cf and Y_policy")
    ap.add_argument("--hmax", type=int, default=29, help="Max horizon for oracle Y_cf")
    ap.add_argument("--cf_t0_list", type=str, default="0,5,10", help="t0 indices for oracle Y_cf")
    args, _unknown = ap.parse_known_args(argv)
    auto_defaults = False
    if not args.out or not args.oracle_path:
        repo_root = Path(__file__).resolve().parents[1]
        if not args.out:
            args.out = str(repo_root / "DataSet/irt_synth/dkt_synth.npz")
            auto_defaults = True
        if not args.oracle_path:
            args.oracle_path = str(repo_root / "runs/dkt_oracle.pt")
            auto_defaults = True
        print(f"NOTE: defaulting --out to {args.out}")
        print(f"NOTE: defaulting --oracle_path to {args.oracle_path}")

    if auto_defaults:
        if not args.with_oracle:
            args.with_oracle = True
            print("NOTE: enabling --with_oracle for default run.")
        if not args.oracle_use_cfg:
            args.oracle_use_cfg = True
            print("NOTE: enabling --oracle_use_cfg for default run.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not Path(args.oracle_path).exists():
        raise FileNotFoundError(
            f"Oracle checkpoint not found: {args.oracle_path}. "
            "Run scripts/train_dkt_oracle.py first."
        )

    device = _device_from_arg(args.device)
    rng = np.random.default_rng(int(args.seed))

    ckpt = torch.load(args.oracle_path, map_location=device)
    if bool(args.oracle_use_cfg) and isinstance(ckpt, dict) and "model_cfg" in ckpt:
        cfg = ckpt["model_cfg"]
        if "n_skills" in cfg:
            args.n_skills = int(cfg["n_skills"])
        if "embed_dim" in cfg:
            args.embed_dim = int(cfg["embed_dim"])
        if "hidden_dim" in cfg:
            args.hidden_dim = int(cfg["hidden_dim"])
        if "d_x" in cfg:
            args.d_x = int(cfg["d_x"])

    oracle = LSTMDKTOracle(
        n_skills=int(args.n_skills),
        embed_dim=int(args.embed_dim),
        hidden_dim=int(args.hidden_dim),
        d_x=int(args.d_x),
    ).to(device)
    if isinstance(ckpt, nn.Module):
        state = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            state = ckpt.get("state_dict", ckpt)
    else:
        state = ckpt
    incompatible = oracle.load_state_dict(state, strict=bool(args.oracle_strict))
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        print("WARNING: Oracle checkpoint did not fully match LSTMDKTOracle.")
        if missing:
            print(f"  missing_keys={missing}")
        if unexpected:
            print(f"  unexpected_keys={unexpected}")
    oracle.eval()

    sim = simulate_factual(
        oracle=oracle,
        rng=rng,
        n=int(args.n),
        seq_len=int(args.seq_len),
        d_x=int(args.d_x),
        n_skills=int(args.n_skills),
        t_effect=float(args.t_effect),
        confounding=float(args.confounding),
        hidden_f_confounding=float(args.hidden_f_confounding),
        hidden_f_y_effect=float(args.hidden_f_y_effect),
        f_rho=float(args.f_rho),
        f_std=float(args.f_std),
        f_init_std=float(args.f_init_std),
        f_clip=float(args.f_clip),
        device=device,
        batch_size=int(args.batch_size),
    )

    save_kwargs = {
        "X": sim["X"].astype(np.float32),
        "A": sim["A"].astype(np.int64),
        "T": sim["T"].astype(np.int64),
        "Y": sim["Y"].astype(np.float32),
        "M": sim["M"].astype(np.float32),
        "F": sim["F"].astype(np.float32),
    }

    if args.with_oracle:
        t0_list = _parse_int_list(args.cf_t0_list)
        t0_vals = _iter_t0(int(args.seq_len), t0_list)
        hmax = int(args.hmax)
        Y_cf = np.full((args.n, args.seq_len, 2, hmax + 1), np.nan, dtype=np.float32)
        for t0 in t0_vals:
            for a in (0, 1):
                T_cf = sim["T"].copy()
                T_cf[:, t0] = int(a)
                Y_prob = simulate_given_T(
                    oracle=oracle,
                    X=sim["X"],
                    A=sim["A"],
                    T=T_cf,
                    u_y=sim["u_y"],
                    f=sim["F"],
                    t_effect=float(args.t_effect),
                    hidden_f_confounding=float(args.hidden_f_confounding),
                    hidden_f_y_effect=float(args.hidden_f_y_effect),
                    device=device,
                    batch_size=int(args.batch_size),
                )
                for h in range(hmax + 1):
                    t = t0 + h
                    if t >= int(args.seq_len):
                        break
                    Y_cf[:, t0, a, h] = Y_prob[:, t]

        policy_names, T_policies = build_open_loop_policies(int(args.seq_len))
        P = int(T_policies.shape[0])
        Y_policy = np.zeros((int(args.n), P, int(args.seq_len)), dtype=np.float32)
        for p in range(P):
            T_pol = np.broadcast_to(T_policies[p][None, :], (int(args.n), int(args.seq_len))).copy()
            Y_policy[:, p] = simulate_given_T(
                oracle=oracle,
                X=sim["X"],
                A=sim["A"],
                T=T_pol,
                u_y=sim["u_y"],
                f=sim["F"],
                t_effect=float(args.t_effect),
                hidden_f_confounding=float(args.hidden_f_confounding),
                hidden_f_y_effect=float(args.hidden_f_y_effect),
                device=device,
                batch_size=int(args.batch_size),
            )

        save_kwargs.update(
            {
                "Y_cf": Y_cf.astype(np.float32),
                "Y_policy": Y_policy.astype(np.float32),
                "policy_names": policy_names,
            }
        )

    np.savez(out_path, **save_kwargs)
    print(f"Wrote: {out_path}")
    print(f"  X: {sim['X'].shape}  A: {sim['A'].shape}  T: {sim['T'].shape}  Y: {sim['Y'].shape}")
    if args.with_oracle:
        print(f"  Y_cf: {save_kwargs['Y_cf'].shape} (hmax={int(args.hmax)})")
        print(f"  Y_policy: {save_kwargs['Y_policy'].shape}")
    return 0


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        main()
    else:
        raise SystemExit(main())
