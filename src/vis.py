from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.metrics.calibration import plot_reliability
from src.utils.manifest import add_artifact, write_manifest


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_ate_cate_curves(
    causal_effects_csv: str,
    *,
    out_dir: str,
    manifest: dict,
    dataset: str,
) -> None:
    df = pd.read_csv(causal_effects_csv)
    if df.empty:
        return

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = [dataset]
    if "dataset" in df.columns:
        datasets = sorted(df["dataset"].dropna().unique().tolist())

    for ds_name in datasets:
        df_d = df if "dataset" not in df.columns else df[df["dataset"] == ds_name]
        if df_d.empty:
            continue
        horizon = int(df_d["horizon"].iloc[0])
        ref_cols = [f"ref_mu_{h}" for h in range(horizon + 1)]
        gen_cols = [f"gen_mu_{h}" for h in range(horizon + 1)]

        for model in df_d["model"].unique():
            df_m = df_d[df_d["model"] == model]
            for subgroup in df_m["subgroup"].unique():
                df_s = df_m[df_m["subgroup"] == subgroup]
                if df_s.empty:
                    continue
                ate_ref_list = []
                ate_gen_list = []
                for t0 in df_s["t0"].unique():
                    df_t = df_s[df_s["t0"] == t0]
                    ref_0 = df_t[df_t["action"] == 0][ref_cols].to_numpy()
                    ref_1 = df_t[df_t["action"] == 1][ref_cols].to_numpy()
                    gen_0 = df_t[df_t["action"] == 0][gen_cols].to_numpy()
                    gen_1 = df_t[df_t["action"] == 1][gen_cols].to_numpy()
                    if ref_0.size == 0 or ref_1.size == 0 or gen_0.size == 0 or gen_1.size == 0:
                        continue
                    ref_0_mean = np.nanmean(ref_0, axis=0)
                    ref_1_mean = np.nanmean(ref_1, axis=0)
                    gen_0_mean = np.nanmean(gen_0, axis=0)
                    gen_1_mean = np.nanmean(gen_1, axis=0)
                    if (
                        np.all(np.isnan(ref_0_mean))
                        or np.all(np.isnan(ref_1_mean))
                        or np.all(np.isnan(gen_0_mean))
                        or np.all(np.isnan(gen_1_mean))
                    ):
                        continue
                    ate_ref_list.append(ref_1_mean - ref_0_mean)
                    ate_gen_list.append(gen_1_mean - gen_0_mean)
                if not ate_ref_list or not ate_gen_list:
                    continue
                ate_ref = np.nanmean(np.stack(ate_ref_list, axis=0), axis=0)
                ate_gen = np.nanmean(np.stack(ate_gen_list, axis=0), axis=0)

                import matplotlib.pyplot as plt

                plt.figure(figsize=(7, 4))
                plt.plot(ate_ref, label="Ref ATE", linewidth=2)
                plt.plot(ate_gen, label="Gen ATE", linewidth=2, linestyle="--")
                plt.xlabel("Horizon")
                plt.ylabel("ATE")
                plt.title(f"ATE Curve ({model}, {subgroup}, {ds_name})")
                plt.legend()
                out_png = out_root / f"ate_{ds_name}_{model}_{subgroup}.png"
                _ensure_dir(out_png)
                plt.tight_layout()
                plt.savefig(out_png, dpi=180)
                plt.close()
                add_artifact(
                    manifest,
                    kind="ate_curve",
                    model=model,
                    dataset=ds_name,
                    path=str(out_png),
                    meta={"subgroup": subgroup, "horizon": horizon},
                )

                if subgroup != "all":
                    add_artifact(
                        manifest,
                        kind="cate_curve",
                        model=model,
                        dataset=ds_name,
                        path=str(out_png),
                        meta={"subgroup": subgroup, "horizon": horizon},
                    )


def plot_calibration_curves(
    calib_dir: str,
    *,
    out_dir: str,
    manifest: dict,
    dataset: str,
) -> None:
    calib_root = Path(calib_dir)
    if not calib_root.exists():
        return
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for npz in calib_root.glob("calibration_*.npz"):
        data = np.load(npz)
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        parts = npz.stem.split("_")
        if len(parts) < 4:
            continue
        _, model, predictor, setting = parts[0], parts[1], parts[2], parts[3]
        out_png = out_root / f"reliability_{model}_{predictor}_{setting}.png"
        plot_reliability(y_true, y_prob, n_bins=10, out_png=str(out_png))
        add_artifact(
            manifest,
            kind="calibration_plot",
            model=model,
            dataset=dataset,
            path=str(out_png),
            meta={"predictor": predictor, "setting": setting},
        )


def plot_policy_values(
    policy_values_csv: str,
    *,
    out_dir: str,
    manifest: dict,
    dataset: str,
) -> None:
    df = pd.read_csv(policy_values_csv)
    if df.empty or not {"model", "policy"}.issubset(df.columns):
        return
    if {"ref_value", "gen_value"}.issubset(df.columns):
        value_cols = ("ref_value", "gen_value")
        labels = ("Ref", "Gen")
    elif {"dr_value", "dm_value"}.issubset(df.columns):
        value_cols = ("dr_value", "dm_value")
        labels = ("DR", "DM")
    else:
        return

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for model in df["model"].unique():
        df_m = df[df["model"] == model]
        if df_m.empty:
            continue
        df_plot = (
            df_m.groupby("policy", sort=False)[list(value_cols)].mean().reset_index()
        )
        policies = df_plot["policy"].astype(str).tolist()
        ref_vals = df_plot[value_cols[0]].to_numpy(dtype=np.float64)
        gen_vals = df_plot[value_cols[1]].to_numpy(dtype=np.float64)
        if len(policies) == 0:
            continue

        import matplotlib.pyplot as plt

        x = np.arange(len(policies))
        width = 0.35
        plt.figure(figsize=(max(6, len(policies) * 1.2), 4))
        plt.bar(x - width / 2, ref_vals, width, label=labels[0], color="#4C78A8")
        plt.bar(x + width / 2, gen_vals, width, label=labels[1], color="#F58518")
        plt.xticks(x, policies, rotation=20, ha="right")
        plt.ylabel("Policy Value")
        plt.title(f"Policy Values ({model})")
        plt.legend()
        out_png = out_root / f"policy_values_{model}.png"
        _ensure_dir(out_png)
        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close()
        add_artifact(
            manifest,
            kind="policy_curve",
            model=model,
            dataset=dataset,
            path=str(out_png),
            meta={},
        )


def run_visualization(
    *,
    results_dir: str,
    out_dir: str,
    dataset: str,
    manifest_path: str,
    manifest: dict | None = None,
    plot_calibration: bool = True,
) -> dict:
    if manifest is None:
        manifest = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "dataset": dataset,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "artifacts": [],
        }

    results_root = Path(results_dir)
    causal_effects_csv = results_root / "causal_effects.csv"
    if causal_effects_csv.exists():
        plot_ate_cate_curves(str(causal_effects_csv), out_dir=out_dir, manifest=manifest, dataset=dataset)

    policy_values_csv = results_root / "policy_values.csv"
    if policy_values_csv.exists():
        plot_policy_values(str(policy_values_csv), out_dir=out_dir, manifest=manifest, dataset=dataset)
    else:
        dr_policy_values_csv = results_root / "dr_policy_values.csv"
        if dr_policy_values_csv.exists():
            plot_policy_values(str(dr_policy_values_csv), out_dir=out_dir, manifest=manifest, dataset=dataset)

    calib_dir = results_root / "calibration"
    if plot_calibration and calib_dir.exists():
        plot_calibration_curves(str(calib_dir), out_dir=out_dir, manifest=manifest, dataset=dataset)

    write_manifest(manifest, manifest_path)
    return manifest
