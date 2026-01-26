from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch


@dataclass
class NaNDebugConfig:
    every_n: int = 1
    check_grads: bool = True
    check_params: bool = True
    max_param_names: int = 10


class NaNDebugger:
    def __init__(self, log_path: Path, cfg: Optional[NaNDebugConfig] = None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or NaNDebugConfig()
        # Ensure the log file exists even if no NaNs are found.
        if not self.log_path.exists():
            self._log("[nan_debug] initialized")

    def _should_check(self, step: int) -> bool:
        return (step % max(1, int(self.cfg.every_n))) == 0

    def _log(self, message: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(message)
            f.write("\n")

    def _tensor_stats(self, t: torch.Tensor) -> str:
        t = t.detach()
        fin = torch.isfinite(t)
        nan_count = int(torch.isnan(t).sum().item())
        inf_count = int(torch.isinf(t).sum().item())
        if fin.any():
            t_min = float(t[fin].min().item())
            t_max = float(t[fin].max().item())
        else:
            t_min = float("nan")
            t_max = float("nan")
        return (
            f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"nan={nan_count} inf={inf_count} min={t_min:.4g} max={t_max:.4g}"
        )

    def check_tensors(
        self,
        *,
        stage: str,
        epoch: int,
        step: int,
        tag: str,
        tensors: Dict[str, torch.Tensor | None],
    ) -> None:
        if not self._should_check(step):
            return
        for name, t in tensors.items():
            if t is None or not torch.is_tensor(t):
                continue
            if not torch.isfinite(t).all():
                stats = self._tensor_stats(t)
                self._log(f"[{stage} ep{epoch} step{step}] {tag}.{name} non-finite: {stats}")

    def check_grads(self, model: torch.nn.Module, *, stage: str, epoch: int, step: int) -> None:
        if not self.cfg.check_grads or not self._should_check(step):
            return
        bad = []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad.append(name)
                if len(bad) >= int(self.cfg.max_param_names):
                    break
        if bad:
            joined = ", ".join(bad)
            self._log(f"[{stage} ep{epoch} step{step}] non-finite grads: {joined}")

    def check_params(self, model: torch.nn.Module, *, stage: str, epoch: int, step: int) -> None:
        if not self.cfg.check_params or not self._should_check(step):
            return
        bad = []
        for name, p in model.named_parameters():
            if not torch.isfinite(p).all():
                bad.append(name)
                if len(bad) >= int(self.cfg.max_param_names):
                    break
        if bad:
            joined = ", ".join(bad)
            self._log(f"[{stage} ep{epoch} step{step}] non-finite params: {joined}")
