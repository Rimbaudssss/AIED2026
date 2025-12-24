"""Standalone baseline models that are not part of `src.model`.

This package is intentionally lightweight so that baselines can be imported from both
`python -m src.main` and top-level CLIs (e.g., `python main.py`).
"""

from .timegan import TimeGAN, TimeGANConfig, TimeGANTrainConfig, fit_timegan

__all__ = ["TimeGAN", "TimeGANConfig", "TimeGANTrainConfig", "fit_timegan"]

