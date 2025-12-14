from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..data import NPZSequenceDataset, make_dataloader, move_batch
from ..model.baselines import load_rollout_model_from_checkpoint
from ..model.policy import ConstantPolicy, RandomPolicy


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Policy simulation on learned SCM generator")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_npz", type=str, required=True, help="Use (X,A) marginals from this dataset")
    p.add_argument("--out_csv", type=str, default="policy_sim.csv")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_actions", type=int, default=2)
    p.add_argument("--cate", action="store_true", help="Compute subgroup CATE between two constant-action policies")
    p.add_argument("--action_control", type=int, default=0)
    p.add_argument("--action_treated", type=int, default=1)
    p.add_argument("--num_groups", type=int, default=2, help="Number of subgroups for CATE (quantiles on X)")
    p.add_argument("--x_index", type=int, default=0, help="Which X dimension to group by")
    args = p.parse_args(argv)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    gen, model_name = load_rollout_model_from_checkpoint(ckpt, device=device)

    t_is_discrete = bool(getattr(getattr(gen, "cfg", None), "t_is_discrete", True))
    if not t_is_discrete:
        raise ValueError("policy_sim currently assumes discrete actions T (t_is_discrete=True).")

    ds = NPZSequenceDataset(args.data_npz)
    dl = make_dataloader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if not args.cate:
        policy = RandomPolicy(num_actions=args.num_actions)
        sums = None
        counts = None
        with torch.no_grad():
            for batch in dl:
                batch = move_batch(batch, device)
                X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask
                ro = gen.rollout(x=X, a=A, t_obs=None, policy=policy, mask=M, stochastic_y=False)
                y = ro["y"]  # [B,T,1] probs
                if y.ndim == 2:
                    y = y[..., None]
                m = M[..., None]
                if sums is None:
                    sums = (y * m).sum(dim=0)
                    counts = m.sum(dim=0)
                else:
                    sums = sums + (y * m).sum(dim=0)
                    counts = counts + m.sum(dim=0)

        means = (sums / counts.clamp(min=1.0)).view(-1).cpu().tolist()
        rows = [{"t": t_idx, "E_y_policy": val} for t_idx, val in enumerate(means)]
    else:
        if args.num_groups < 2:
            raise ValueError("--num_groups must be >= 2 for CATE")
        if not (0 <= args.x_index < ds.d_x):
            raise ValueError(f"--x_index out of range; dataset d_x={ds.d_x}")

        # Quantile groups by X[:, x_index] (proxy for initial ability).
        x0 = ds.X[:, int(args.x_index)].astype("float64")
        qs = [i / args.num_groups for i in range(args.num_groups + 1)]
        edges = torch.as_tensor(np.quantile(x0, qs), dtype=torch.float32)

        policy_c = ConstantPolicy(action=args.action_control)
        policy_t = ConstantPolicy(action=args.action_treated)

        T_len = ds.seq_len
        g = int(args.num_groups)
        sums_c = torch.zeros(g, T_len, device=device)
        sums_t = torch.zeros(g, T_len, device=device)
        counts = torch.zeros(g, T_len, device=device)

        def group_id(x_scalar: torch.Tensor) -> torch.Tensor:
            # edges: [g+1] increasing
            # returns ids in [0,g-1]
            # bucketize returns in [0,g] for boundaries; clamp to [0,g-1]
            idx = torch.bucketize(x_scalar, edges[1:-1].to(x_scalar.device), right=False)
            return idx.clamp(min=0, max=g - 1)

        with torch.no_grad():
            for batch in dl:
                batch = move_batch(batch, device)
                X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask

                gid = group_id(X[:, int(args.x_index)])  # [B]
                ro_c = gen.rollout(x=X, a=A, t_obs=None, policy=policy_c, mask=M, stochastic_y=False)
                ro_t = gen.rollout(x=X, a=A, t_obs=None, policy=policy_t, mask=M, stochastic_y=False)

                y_c = ro_c["y"]
                y_treat = ro_t["y"]
                if y_c.ndim == 3:
                    y_c = y_c.squeeze(-1)
                if y_treat.ndim == 3:
                    y_treat = y_treat.squeeze(-1)

                for gi in range(g):
                    sel = gid == gi  # [B]
                    if not sel.any():
                        continue
                    sel_m = M[sel]  # [b_g, T]
                    sums_c[gi] += (y_c[sel] * sel_m).sum(dim=0)
                    sums_t[gi] += (y_treat[sel] * sel_m).sum(dim=0)
                    counts[gi] += sel_m.sum(dim=0)

        mean_c = sums_c / counts.clamp(min=1.0)
        mean_t = sums_t / counts.clamp(min=1.0)
        cate = mean_t - mean_c

        rows = []
        edges_cpu = edges.cpu().tolist()
        for gi in range(g):
            low = edges_cpu[gi]
            high = edges_cpu[gi + 1]
            for t_idx in range(T_len):
                rows.append(
                    {
                        "group": gi,
                        "group_low": low,
                        "group_high": high,
                        "t": t_idx,
                        "E_y_control": float(mean_c[gi, t_idx].item()),
                        "E_y_treated": float(mean_t[gi, t_idx].item()),
                        "CATE": float(cate[gi, t_idx].item()),
                    }
                )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["t"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
