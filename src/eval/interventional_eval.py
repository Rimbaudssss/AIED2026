from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import torch

from ..data import NPZSequenceDataset, make_dataloader, move_batch
from ..model.baselines import load_rollout_model_from_checkpoint
from ..model.losses import mmd_rbf
from ..model.policy import DoIntervention


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Interventional fidelity eval: E[Y_t | do(T_t=a)]")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_npz", type=str, required=True)
    p.add_argument("--out_csv", type=str, default="interventional_eval.csv")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--time_indices", type=str, default="0,5,10")
    p.add_argument("--actions", type=str, default="0,1")
    p.add_argument("--mmd_xy", action="store_true", help="Also report MMD over features (X, Y) per arm")
    args = p.parse_args(argv)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    gen, model_name = load_rollout_model_from_checkpoint(ckpt, device=device)

    t_is_discrete = bool(getattr(getattr(gen, "cfg", None), "t_is_discrete", True))
    if not t_is_discrete:
        raise ValueError("interventional_eval currently assumes discrete actions T (t_is_discrete=True).")

    ds = NPZSequenceDataset(args.data_npz)
    dl = make_dataloader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    time_indices = [int(x.strip()) for x in args.time_indices.split(",") if x.strip()]
    actions = [int(x.strip()) for x in args.actions.split(",") if x.strip()]

    rows = []
    with torch.no_grad():
        for t_idx in time_indices:
            for a_val in actions:
                num_real = 0
                sum_real = 0.0
                num_gen = 0
                sum_gen = 0.0
                mmd_sum = 0.0
                mmd_n = 0

                for batch in dl:
                    batch = move_batch(batch, device)
                    X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask
                    if not (0 <= t_idx < A.shape[1]):
                        continue

                    sel = (T[:, t_idx].long() == a_val) & (M[:, t_idx] > 0.5)
                    if sel.any():
                        y_real = Y[sel, t_idx].float()
                        sum_real += float(y_real.sum().item())
                        num_real += int(sel.sum().item())

                    do = DoIntervention.single_step(t_idx, a_val).as_dict(batch_size=X.shape[0], device=device)
                    ro = gen.rollout(x=X, a=A, t_obs=T, do_t=do, mask=M, stochastic_y=False)
                    y_do = ro["y"][:, t_idx].view(-1).float()
                    sum_gen += float(y_do.sum().item())
                    num_gen += int(y_do.numel())

                    if args.mmd_xy and sel.any():
                        # Feature-level distance to reflect heterogeneity beyond mean Y.
                        feat_real = torch.cat([X[sel].float(), Y[sel, t_idx].float().view(-1, 1)], dim=-1)
                        valid = M[:, t_idx] > 0.5
                        feat_gen = torch.cat([X[valid].float(), y_do[valid].view(-1, 1)], dim=-1)
                        mmd_sum += float(mmd_rbf(feat_real, feat_gen).item())
                        mmd_n += 1

                rows.append(
                    {
                        "t": t_idx,
                        "a": a_val,
                        "E_y_real_TeqA": (sum_real / max(1, num_real)),
                        "E_y_gen_do": (sum_gen / max(1, num_gen)),
                        "MMD_XY": (mmd_sum / max(1, mmd_n)) if args.mmd_xy else "",
                        "n_real": num_real,
                        "n_gen": num_gen,
                    }
                )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["t", "a"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
