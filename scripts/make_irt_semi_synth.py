"""Generate a semi-synthetic sequential dataset with oracle counterfactuals.

This script outputs an .npz compatible with src.data.NPZSequenceDataset, with optional
extra keys for oracle evaluation:

    - Y_cf: [N, T, 2, Hmax+1] float32 oracle outcome probabilities for a single-step do(T_t0=0/1)
    - Y_policy: [N, P, T] float32 oracle outcome probabilities under a small set of open-loop policies
    - policy_names: [P] bytes/strings (stored as dtype='S')

It is intentionally deterministic given --seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_open_loop_policies(seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (policy_names, T_policies) where T_policies is [P, T] with {0,1}."""
    # Names chosen to match src.policy.get_default_policies(..., policy_set='fixed')
    # (ConstantPolicy + StagePolicy variants).
    t_switch = int(0.2 * seq_len)
    names = [b"never", b"always", b"early_on", b"late_on"]
    never = np.zeros(seq_len, dtype=np.int64)
    always = np.ones(seq_len, dtype=np.int64)
    early_on = np.concatenate([np.ones(t_switch, dtype=np.int64), np.zeros(seq_len - t_switch, dtype=np.int64)])
    late_on = np.concatenate([np.zeros(t_switch, dtype=np.int64), np.ones(seq_len - t_switch, dtype=np.int64)])
    T = np.stack([never, always, early_on, late_on], axis=0)
    return np.array(names, dtype="S"), T


def simulate_factual(
    rng: np.random.Generator,
    n: int,
    seq_len: int,
    d_x: int,
    a_vocab_size: int,
    lr: float,
    delta: float,
    gamma: float,
    noise_std: float,
    confounding: float,
) -> dict[str, np.ndarray]:
    """Simulate a multidimensional IRT (MIRT) sequential process."""
    # Treat d_x as latent ability dimension (d_k).
    theta0 = rng.normal(0.0, 1.0, size=(n, d_x)).astype(np.float32)

    # Item embeddings represent which skills are tested by each item.
    item_embeddings = rng.normal(0.0, 1.0, size=(a_vocab_size, d_x)).astype(np.float32)
    item_embeddings = item_embeddings / (np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-6)

    A = rng.integers(low=0, high=a_vocab_size, size=(n, seq_len), dtype=np.int64)

    # Exogenous noises, fixed for all counterfactuals.
    eps_theta = rng.normal(0.0, noise_std, size=(n, seq_len, d_x)).astype(np.float32)
    u_y = rng.random(size=(n, seq_len)).astype(np.float32)

    theta = theta0.copy()
    theta_hist = np.zeros((n, seq_len + 1, d_x), dtype=np.float32)
    theta_hist[:, 0] = theta

    T = np.zeros((n, seq_len), dtype=np.int64)
    Y = np.zeros((n, seq_len), dtype=np.float32)

    for t in range(seq_len):
        # Confounded treatment assignment: low-ability students more likely to receive help.
        mean_ability = theta.mean(axis=1)
        p_help = sigmoid(-confounding * mean_ability)
        T[:, t] = (rng.random(size=n) < p_help).astype(np.int64)

        cur_items = item_embeddings[A[:, t]]
        knowledge_term = (theta * cur_items).sum(axis=1)
        help_flag = T[:, t].astype(np.float32)
        logits = knowledge_term + gamma * help_flag
        p = sigmoid(logits).astype(np.float32)
        y = (u_y[:, t] < p).astype(np.float32)
        Y[:, t] = y

        pred_error = (y - p)[:, None]
        learning_gain = lr * pred_error * cur_items
        intervention_gain = delta * help_flag[:, None] * cur_items
        theta = theta + learning_gain + intervention_gain + eps_theta[:, t]
        theta_hist[:, t + 1] = theta

    M = np.ones((n, seq_len), dtype=np.float32)
    return {
        "X": theta0,
        "A": A,
        "T": T,
        "Y": Y,
        "M": M,
        "theta_hist": theta_hist,
        "item_embeddings": item_embeddings,
        "theta0": theta0,
        "eps_theta": eps_theta,
        "u_y": u_y,
    }


def simulate_given_T(
    A: np.ndarray,
    T: np.ndarray,
    item_embeddings: np.ndarray,
    theta0: np.ndarray,
    eps_theta: np.ndarray,
    u_y: np.ndarray,
    lr: float,
    delta: float,
    gamma: float,
) -> np.ndarray:
    """Return Y_prob trajectory [N,T] for a fixed treatment sequence T."""
    n, seq_len = T.shape
    theta = theta0.astype(np.float32).copy()
    Y_prob = np.zeros((n, seq_len), dtype=np.float32)
    for t in range(seq_len):
        cur_items = item_embeddings[A[:, t]]
        knowledge_term = (theta * cur_items).sum(axis=1)
        help_flag = T[:, t].astype(np.float32)
        logits = knowledge_term + gamma * help_flag
        p = sigmoid(logits).astype(np.float32)
        Y_prob[:, t] = p
        y = (u_y[:, t] < p).astype(np.float32)
        pred_error = (y - p)[:, None]
        learning_gain = lr * pred_error * cur_items
        intervention_gain = delta * help_flag[:, None] * cur_items
        theta = theta + learning_gain + intervention_gain + eps_theta[:, t]
    return Y_prob


def build_oracle_counterfactuals(
    sim: dict[str, np.ndarray],
    hmax: int,
    lr: float,
    delta: float,
    gamma: float,
) -> np.ndarray:
    """Compute Y_cf = P(Y_{t0:t0+h} | do(T_{t0}=a), T_{!=t0}=factual)."""
    A = sim["A"]
    T = sim["T"]
    item_embeddings = sim["item_embeddings"]
    theta_hist = sim["theta_hist"]
    eps_theta = sim["eps_theta"]
    u_y = sim["u_y"]

    n, seq_len = T.shape
    Y_cf = np.full((n, seq_len, 2, hmax + 1), np.nan, dtype=np.float32)

    for t0 in range(seq_len):
        # Start from the *factual* state at t0.
        theta0 = theta_hist[:, t0].astype(np.float32)
        for a in (0, 1):
            theta = theta0.copy()
            for h in range(hmax + 1):
                t = t0 + h
                if t >= seq_len:
                    break
                help_flag = (np.full(n, a, dtype=np.int64) if h == 0 else T[:, t]).astype(np.float32)
                cur_items = item_embeddings[A[:, t]]
                knowledge_term = (theta * cur_items).sum(axis=1)
                logits = knowledge_term + gamma * help_flag
                p = sigmoid(logits).astype(np.float32)
                Y_cf[:, t0, a, h] = p
                y = (u_y[:, t] < p).astype(np.float32)
                pred_error = (y - p)[:, None]
                learning_gain = lr * pred_error * cur_items
                intervention_gain = delta * help_flag[:, None] * cur_items
                theta = theta + learning_gain + intervention_gain + eps_theta[:, t]

    return Y_cf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output .npz path")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seq_len", type=int, default=50)
    ap.add_argument("--d_x", type=int, default=5)
    ap.add_argument("--d_k", type=int, default=5, help="Latent ability dimension (overrides d_x if set).")
    ap.add_argument("--a_vocab_size", type=int, default=50)
    ap.add_argument("--hmax", type=int, default=10, help="Max horizon for oracle Y_cf")
    ap.add_argument("--seed", type=int, default=0)
    # DGP parameters
    ap.add_argument("--lr", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=0.75)
    ap.add_argument("--noise_std", type=float, default=0.05)
    ap.add_argument("--confounding", type=float, default=1.0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    d_k = int(args.d_k)
    if int(args.d_k) == 5 and int(args.d_x) != 5:
        d_k = int(args.d_x)
    sim = simulate_factual(
        rng=rng,
        n=args.n,
        seq_len=args.seq_len,
        d_x=d_k,
        a_vocab_size=args.a_vocab_size,
        lr=float(args.lr),
        delta=float(args.delta),
        gamma=float(args.gamma),
        noise_std=float(args.noise_std),
        confounding=float(args.confounding),
    )

    # Oracle counterfactuals for single-step do at each t0.
    Y_cf = build_oracle_counterfactuals(sim, hmax=int(args.hmax), lr=float(args.lr), delta=float(args.delta), gamma=float(args.gamma))

    # Oracle policy trajectories (open-loop fixed policies).
    policy_names, T_policies = build_open_loop_policies(args.seq_len)
    P = int(T_policies.shape[0])
    Y_policy = np.zeros((args.n, P, args.seq_len), dtype=np.float32)
    for p in range(P):
        T_pol = np.broadcast_to(T_policies[p][None, :], (args.n, args.seq_len)).copy()
        Y_policy[:, p] = simulate_given_T(
            A=sim["A"],
            T=T_pol,
            item_embeddings=sim["item_embeddings"],
            theta0=sim["theta0"],
            eps_theta=sim["eps_theta"],
            u_y=sim["u_y"],
            lr=float(args.lr),
            delta=float(args.delta),
            gamma=float(args.gamma),
        )

    np.savez(
        out_path,
        X=sim["X"].astype(np.float32),
        A=sim["A"].astype(np.int64),
        T=sim["T"].astype(np.int64),
        Y=sim["Y"].astype(np.float32),
        M=sim["M"].astype(np.float32),
        Y_cf=Y_cf.astype(np.float32),
        Y_policy=Y_policy.astype(np.float32),
        policy_names=policy_names,
    )

    print(f"Wrote: {out_path}")
    print(f"  X: {sim['X'].shape}  A: {sim['A'].shape}  T: {sim['T'].shape}  Y: {sim['Y'].shape}")
    print(f"  Y_cf: {Y_cf.shape} (hmax={args.hmax})")
    print(f"  Y_policy: {Y_policy.shape} policies={list(policy_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
