"""Train a DKT-style LSTM oracle on assist09 (or any NPZSequenceDataset)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.chdir(r"C:\Users\Administrator\Desktop\AIED2026")

# Allow running as a script (e.g., `python scripts/train_dkt_oracle.py`) from repo root.
if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from src.data import NPZSequenceDataset, make_dataloader, move_batch


class LSTMDKTOracle(nn.Module):
    def __init__(
        self,
        *,
        n_skills: int,
        embed_dim: int,
        hidden_dim: int,
        d_x: int = 0,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_skills = int(n_skills)
        self.embedding = nn.Embedding(self.n_skills * 2 + 1, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim, self.n_skills)
        self.h0 = nn.Linear(int(d_x), hidden_dim) if int(d_x) > 0 else None
        self.c0 = nn.Linear(int(d_x), hidden_dim) if int(d_x) > 0 else None
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)

    def init_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = int(x.shape[0])
        device = x.device
        if self.h0 is None or self.c0 is None:
            h0 = torch.zeros(self.num_layers, bsz, self.hidden_dim, device=device)
            c0 = torch.zeros(self.num_layers, bsz, self.hidden_dim, device=device)
            return h0, c0
        h = torch.tanh(self.h0(x.float())).unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c = torch.tanh(self.c0(x.float())).unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        return h, c

    def forward(
        self, input_seq: torch.Tensor, hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embedding(input_seq)
        out, hidden = self.lstm(emb, hidden)
        logits = self.out(out)
        return logits, hidden


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_inputs(a: torch.Tensor, y: torch.Tensor, n_skills: int) -> torch.Tensor:
    bsz, seq_len = a.shape
    start_token = n_skills * 2
    prev_a = a[:, :-1].long()
    prev_y = y[:, :-1].float()
    prev_token = prev_a + (prev_y > 0.5).long() * n_skills
    start = torch.full((bsz, 1), start_token, device=a.device, dtype=torch.long)
    return torch.cat([start, prev_token], dim=1)


def _split_indices(n: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(n)
    n_val = int(round(n * float(val_ratio)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train a DKT LSTM oracle from NPZSequenceDataset.")
    ap.add_argument("--data_npz", type=str, default="DataSet/assist2009/assist09_processed.npz")
    ap.add_argument("--out", type=str, default="runs/dkt_oracle.pt")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=10.0)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--n_skills", type=int, default=0, help="Override skill vocab size (default uses dataset)")
    ap.add_argument("--d_x", type=int, default=0, help="Override static feature dim (default uses dataset)")
    args, _unknown = ap.parse_known_args(argv)

    torch.manual_seed(int(args.seed))
    device = _device_from_arg(args.device)

    ds = NPZSequenceDataset(args.data_npz)
    n_skills = int(args.n_skills) if int(args.n_skills) > 0 else int(ds.A.max() + 1)
    d_x = int(args.d_x) if int(args.d_x) > 0 else int(ds.d_x)

    train_idx, val_idx = _split_indices(len(ds), float(args.val_ratio), int(args.seed))
    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    val_ds = torch.utils.data.Subset(ds, val_idx.tolist()) if len(val_idx) > 0 else None

    train_dl = make_dataloader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_dl = make_dataloader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0) if val_ds else None

    model = LSTMDKTOracle(
        n_skills=n_skills,
        embed_dim=int(args.embed_dim),
        hidden_dim=int(args.hidden_dim),
        d_x=d_x,
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    def run_epoch(dl, training: bool) -> float:
        model.train(training)
        total = 0.0
        steps = 0
        for batch in dl:
            batch = move_batch(batch, device)
            X, A, Y, M = batch.X, batch.A, batch.Y, batch.mask
            inp = _build_inputs(A, Y, n_skills)
            h0 = model.init_state(X)
            logits, _ = model(inp, h0)
            logits_sel = logits.gather(2, A.long().unsqueeze(-1)).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits_sel, Y.float(), reduction="none")
            loss = (loss * M).sum() / M.sum().clamp(min=1.0)

            if training:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if float(args.max_grad_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.max_grad_norm))
                opt.step()

            total += float(loss.item())
            steps += 1
        return total / max(1, steps)

    best_val = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, int(args.epochs) + 1):
        train_loss = run_epoch(train_dl, training=True)
        val_loss = run_epoch(val_dl, training=False) if val_dl else float("nan")
        print(f"Epoch {ep}/{int(args.epochs)} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

        if val_dl is None or val_loss < best_val:
            best_val = val_loss if val_dl else train_loss
            ckpt = {
                "state_dict": model.state_dict(),
                "model_cfg": {
                    "n_skills": n_skills,
                    "embed_dim": int(args.embed_dim),
                    "hidden_dim": int(args.hidden_dim),
                    "d_x": d_x,
                    "num_layers": int(args.num_layers),
                    "dropout": float(args.dropout),
                },
            }
            torch.save(ckpt, out_path)

    print(f"Saved oracle checkpoint to {out_path}")
    return 0


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        main()
    else:
        raise SystemExit(main())
