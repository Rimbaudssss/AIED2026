from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize


MODEL_ORDER = ["scm_causal", "rcgan", "vae", "diffusion", "crn", "timegan"]
MODEL_LABEL = {
    "scm_causal": "SCM-Causal",
    "rcgan": "RCGAN",
    "vae": "VAE",
    "diffusion": "Diffusion",
    "crn": "CRN",
    "timegan": "TimeGAN",
}
MODEL_COLOR = {
    "scm_causal": "#234f7d",
    "rcgan": "#d35f32",
    "vae": "#4c956c",
    "diffusion": "#7a5aa6",
    "crn": "#c99a2e",
    "timegan": "#5c677d",
}
DATASET_COLOR = {
    "assistments_rct88": "#4e79a7",
    "assistments_las2016": "#f28e2b",
    "assistments_abtest_study2": "#59a14f",
}
METRIC_LABEL = {
    "ate_abs_err": "ATE abs. err.",
    "policy_value_abs_err": "Policy value abs. err.",
    "policy_regret": "Policy regret",
    "sign_accuracy": "Sign acc.",
    "pearson": "Pearson r",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    short_label: str
    result_dir: Path
    source: str


DATASETS = [
    DatasetSpec(
        "assistments_rct88",
        "ASSISTments expanded RCT release",
        "RCT88/89",
        Path("artifacts/assistments_rct88_scm_models_tuned"),
        "https://osf.io/m2jqe/",
    ),
    DatasetSpec(
        "assistments_las2016",
        "LAS2016 22 randomized experiments",
        "LAS2016",
        Path("artifacts/assistments_las2016_scm_models_tuned"),
        "https://sites.google.com/site/las2016data/data/thison",
    ),
    DatasetSpec(
        "assistments_abtest_study2",
        "ASSISTments OSF Study2 A/B tests",
        "Study2",
        Path("artifacts/assistments_abtest_study2_scm_models_tuned"),
        "https://osf.io/j6esa/",
    ),
]


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.0,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.bbox": "tight",
            "savefig.dpi": 340,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _label_panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        -0.08,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax.set_title(title, loc="left", fontweight="bold", pad=8)


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def _model_labels(models: list[str]) -> list[str]:
    return [MODEL_LABEL.get(m, m) for m in models]


def _load_all(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method_frames: list[pd.DataFrame] = []
    ate_frames: list[pd.DataFrame] = []
    task_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    audit_rows: list[dict] = []

    for spec in DATASETS:
        table_dir = root / spec.result_dir / "tables"
        method = _read_csv(table_dir / "method_summary.csv")
        method["dataset"] = spec.key
        method["dataset_label"] = spec.label
        method["dataset_short"] = spec.short_label
        method_frames.append(method)

        ate = _read_csv(table_dir / "ate_recovery_by_seed.csv")
        ate["dataset"] = spec.key
        ate["dataset_label"] = spec.label
        ate["dataset_short"] = spec.short_label
        ate_frames.append(ate)

        task = _read_csv(table_dir / "task_summary.csv")
        task["dataset"] = spec.key
        task["dataset_label"] = spec.label
        task["dataset_short"] = spec.short_label
        task_frames.append(task)

        selection = _read_csv(table_dir / "selection_by_seed.csv")
        selection["dataset"] = spec.key
        selection["dataset_label"] = spec.label
        selection["dataset_short"] = spec.short_label
        selection_frames.append(selection)

        audit = json.loads((table_dir / "leakage_audit.json").read_text(encoding="utf-8"))
        audit_rows.append(
            {
                "dataset": spec.key,
                "dataset_label": spec.label,
                "dataset_short": spec.short_label,
                "n_features": len(audit.get("feature_columns", [])),
                "n_excluded_columns": len(audit.get("excluded_columns", [])),
                "n_post_treatment_dropped": len(audit.get("post_treatment_columns_explicitly_dropped", [])),
                "uses_true_ate_in_training": bool(audit.get("uses_true_ate_in_training", False)),
                "uses_arm_means_in_training": bool(audit.get("uses_arm_means_in_training", False)),
                "uses_condition_as_feature": bool(audit.get("uses_condition_as_feature", False)),
            }
        )

    return (
        pd.concat(method_frames, ignore_index=True),
        pd.concat(ate_frames, ignore_index=True),
        pd.concat(task_frames, ignore_index=True),
        pd.concat(selection_frames, ignore_index=True),
        pd.DataFrame(audit_rows),
    )


def _dataset_summary(task: pd.DataFrame, selection: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    task_sum = (
        task.groupby(["dataset", "dataset_label", "dataset_short"], as_index=False)
        .agg(
            n_tasks=("task_key", "nunique"),
            n_students=("n", "sum"),
            median_task_n=("n", "median"),
            min_task_n=("n", "min"),
            median_arm0=("n0", "median"),
            median_arm1=("n1", "median"),
            mean_outcome=("y_rate", "mean"),
            mean_true_ate=("true_ate", "mean"),
            sd_true_ate=("true_ate", "std"),
            mean_abs_true_ate=("true_ate", lambda x: float(np.mean(np.abs(x)))),
            share_positive_ate=("true_ate", lambda x: float(np.mean(np.asarray(x) > 0))),
        )
        .merge(
            selection.groupby("dataset", as_index=False).agg(mean_retention=("retention", "mean")),
            on="dataset",
            how="left",
        )
        .merge(audit, on=["dataset", "dataset_label", "dataset_short"], how="left")
    )
    task_sum["source"] = task_sum["dataset"].map({d.key: d.source for d in DATASETS})
    return task_sum


def _task_averaged_summary(ate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, dataset_short, model, label), group in ate.groupby(["dataset", "dataset_short", "model", "label"]):
        by_task = (
            group.groupby("task_key", as_index=False)
            .agg(
                true_ate=("true_ate", "mean"),
                ate_hat=("ate_hat", "mean"),
                policy_value_abs_err=("policy_value_abs_err", "mean"),
                policy_regret=("policy_regret", "mean"),
                selected_treatment_rate=("policy_selected_arm", "mean"),
                oracle_treatment_rate=("oracle_best_arm", "mean"),
            )
        )
        err = by_task["ate_hat"] - by_task["true_ate"]
        sign_mask = np.sign(by_task["ate_hat"]) == np.sign(by_task["true_ate"])
        if by_task["true_ate"].std(ddof=0) > 0 and by_task["ate_hat"].std(ddof=0) > 0:
            corr = float(np.corrcoef(by_task["true_ate"], by_task["ate_hat"])[0, 1])
        else:
            corr = np.nan
        rows.append(
            {
                "dataset": dataset,
                "dataset_short": dataset_short,
                "model": model,
                "label": label,
                "n_tasks": len(by_task),
                "taskavg_ate_abs_err": float(np.mean(np.abs(err))),
                "taskavg_ate_rmse": float(np.sqrt(np.mean(np.square(err)))),
                "taskavg_ate_bias": float(np.mean(err)),
                "taskavg_policy_value_abs_err": float(by_task["policy_value_abs_err"].mean()),
                "taskavg_policy_regret": float(by_task["policy_regret"].mean()),
                "taskavg_sign_accuracy": float(sign_mask.mean()),
                "taskavg_pearson": corr,
                "selected_treatment_rate": float(by_task["selected_treatment_rate"].mean()),
                "oracle_treatment_rate": float(by_task["oracle_treatment_rate"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _macro_summary(method: pd.DataFrame) -> pd.DataFrame:
    primary = ["ate_abs_err", "policy_value_abs_err", "policy_regret"]
    rank_frames = []
    for dataset, group in method.groupby("dataset"):
        ranks = group[["model", "label"]].copy()
        ranks["dataset"] = dataset
        for metric in primary:
            ranks[f"{metric}_rank"] = group[metric].rank(method="average", ascending=True).to_numpy()
        rank_frames.append(ranks)
    ranks_all = pd.concat(rank_frames, ignore_index=True)
    rank_cols = [f"{m}_rank" for m in primary]
    ranks_all["primary_macro_rank"] = ranks_all[rank_cols].mean(axis=1)

    macro = (
        method.groupby(["model", "label"], as_index=False)
        .agg(
            ate_abs_err=("ate_abs_err", "mean"),
            ate_rmse=("ate_rmse", "mean"),
            policy_value_abs_err=("policy_value_abs_err", "mean"),
            policy_regret=("policy_regret", "mean"),
            sign_accuracy=("sign_accuracy", "mean"),
            pearson=("pearson", "mean"),
            spearman=("spearman", "mean"),
        )
        .merge(
            ranks_all.groupby(["model", "label"], as_index=False).agg(primary_macro_rank=("primary_macro_rank", "mean")),
            on=["model", "label"],
            how="left",
        )
        .sort_values(["primary_macro_rank", "ate_abs_err"])
    )
    return macro


def _best_counts_by_task(ate: pd.DataFrame) -> pd.DataFrame:
    by_task = (
        ate.groupby(["dataset", "dataset_short", "task_key", "model", "label"], as_index=False)
        .agg(true_ate=("true_ate", "mean"), ate_hat=("ate_hat", "mean"))
    )
    by_task["task_ate_abs_err"] = (by_task["ate_hat"] - by_task["true_ate"]).abs()
    idx = by_task.groupby(["dataset", "task_key"])["task_ate_abs_err"].idxmin()
    winners = by_task.loc[idx].copy()
    counts = winners.groupby(["dataset", "dataset_short", "model", "label"], as_index=False).size()
    counts = counts.rename(columns={"size": "n_task_wins"})
    all_pairs = pd.MultiIndex.from_product(
        [[d.key for d in DATASETS], MODEL_ORDER], names=["dataset", "model"]
    ).to_frame(index=False)
    all_pairs["dataset_short"] = all_pairs["dataset"].map({d.key: d.short_label for d in DATASETS})
    all_pairs["label"] = all_pairs["model"].map(MODEL_LABEL)
    counts = all_pairs.merge(counts, on=["dataset", "dataset_short", "model", "label"], how="left")
    counts["n_task_wins"] = counts["n_task_wins"].fillna(0).astype(int)
    return counts


def _markdown_table(df: pd.DataFrame) -> str:
    clean = df.copy()
    for col in clean.columns:
        clean[col] = clean[col].map(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
    headers = list(clean.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in clean.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def _heatmap(ax: plt.Axes, matrix: pd.DataFrame, title: str, cmap: str | LinearSegmentedColormap, fmt: str, lower_better: bool) -> None:
    values = matrix.to_numpy(dtype=float)
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(valid.min()), float(valid.max())
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    ax.set_xticks(np.arange(matrix.shape[1]), _model_labels(list(matrix.columns)), rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), list(matrix.index))
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    ax.tick_params(length=0)
    threshold = vmin + 0.58 * (vmax - vmin)
    for i in range(matrix.shape[0]):
        row_vals = values[i, :]
        best = np.nanargmin(row_vals) if lower_better else np.nanargmax(row_vals)
        for j in range(matrix.shape[1]):
            color = "white" if values[i, j] > threshold else "#1f1f1f"
            weight = "bold" if j == best else "normal"
            ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", color=color, fontsize=7.9, fontweight=weight)
    for side in ["left", "bottom"]:
        ax.spines[side].set_visible(False)
    return im


def fig1_benchmark(task: pd.DataFrame, dataset_summary: pd.DataFrame, audit: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(14.2, 8.4), constrained_layout=False)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.02, 0.98], hspace=0.36, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    order = [d.key for d in DATASETS]
    labels = [d.short_label for d in DATASETS]
    ds = dataset_summary.set_index("dataset").loc[order].reset_index()
    x = np.arange(len(order))
    width = 0.34

    _label_panel(ax_a, "A", "Three real randomized intervention benchmarks")
    ax_a.bar(x - width / 2, ds["n_tasks"], width=width, color=[DATASET_COLOR[d] for d in order], alpha=0.95)
    ax_a.set_ylabel("RCT/A-B tasks")
    ax_a.set_xticks(x, labels)
    for i, val in enumerate(ds["n_tasks"]):
        ax_a.text(i - width / 2, val + max(ds["n_tasks"]) * 0.025, f"{int(val)}", ha="center", va="bottom", fontsize=8.3)
    ax2 = ax_a.twinx()
    ax2.bar(x + width / 2, ds["n_students"], width=width, color="#8d99ae", alpha=0.85)
    ax2.set_ylabel("Student-task rows")
    for i, val in enumerate(ds["n_students"]):
        ax2.text(i + width / 2, val + max(ds["n_students"]) * 0.025, f"{int(val):,}", ha="center", va="bottom", fontsize=8.3)
    ax_a.set_ylim(0, max(ds["n_tasks"]) * 1.18)
    ax2.set_ylim(0, max(ds["n_students"]) * 1.18)

    _label_panel(ax_b, "B", "Empirical treatment effects from held-out RCT arms")
    positions = np.arange(len(order))
    for i, key in enumerate(order):
        vals = task.loc[task["dataset"].eq(key), "true_ate"].to_numpy()
        jitter = np.linspace(-0.10, 0.10, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax_b.scatter(np.full(len(vals), positions[i]) + jitter, vals, s=24, color=DATASET_COLOR[key], alpha=0.75, edgecolor="white", linewidth=0.4)
        ax_b.hlines(np.mean(vals), positions[i] - 0.25, positions[i] + 0.25, color="#1f1f1f", lw=2)
    ax_b.axhline(0, color="#444444", lw=0.8, ls="--")
    ax_b.set_xticks(positions, labels)
    ax_b.set_ylabel("True ATE from randomized arms")
    ax_b.text(0.02, 0.04, "Points are experiment-level estimands; black bars are dataset means.", transform=ax_b.transAxes, fontsize=8.1, color="#555555")

    _label_panel(ax_c, "C", "Pre-treatment covariate audit")
    feature_mat = audit.set_index("dataset").loc[order][["n_features", "n_excluded_columns", "n_post_treatment_dropped"]]
    bars = ["features used", "blocked fields", "post-treatment drops"]
    offsets = [-0.24, 0.0, 0.24]
    colors = ["#457b9d", "#6c757d", "#e76f51"]
    for k, col in enumerate(feature_mat.columns):
        vals = feature_mat[col].to_numpy()
        ax_c.bar(x + offsets[k], vals, width=0.22, color=colors[k], label=bars[k], alpha=0.93)
        for i, val in enumerate(vals):
            ax_c.text(i + offsets[k], val + max(feature_mat.max()) * 0.025, str(int(val)), ha="center", va="bottom", fontsize=7.8)
    ax_c.set_xticks(x, labels)
    ax_c.set_ylabel("Column count")
    ax_c.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02))

    _label_panel(ax_d, "D", "Leakage checks used in every model run")
    checks = ["true ATE in train", "arm means in train", "T as feature"]
    check_cols = ["uses_true_ate_in_training", "uses_arm_means_in_training", "uses_condition_as_feature"]
    vals = audit.set_index("dataset").loc[order][check_cols].astype(int).to_numpy()
    cmap = LinearSegmentedColormap.from_list("audit", ["#d8f3dc", "#b00020"])
    ax_d.imshow(vals, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax_d.set_xticks(np.arange(len(checks)), checks, rotation=18, ha="right")
    ax_d.set_yticks(np.arange(len(labels)), labels)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax_d.text(j, i, "no" if vals[i, j] == 0 else "yes", ha="center", va="center", fontsize=9, fontweight="bold")
    ax_d.tick_params(length=0)
    for side in ["left", "bottom"]:
        ax_d.spines[side].set_visible(False)
    ax_d.text(0.0, -0.22, "Counterfactual values are produced only by each model's do(T=0/1) rollout interface.", transform=ax_d.transAxes, fontsize=8.3, color="#555555")

    fig.suptitle("Fig. 1. Real ASSISTments intervention benchmarks and leakage controls", x=0.02, ha="left", fontweight="bold")
    _save(fig, out_dir, "Fig1_real_rct_benchmark")


def fig2_main_results(method: pd.DataFrame, macro: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(15.0, 9.2), constrained_layout=False)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.28)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    row_order = [d.short_label for d in DATASETS]
    model_order = [m for m in MODEL_ORDER if m in method["model"].unique()]

    mats = {}
    for metric in ["ate_abs_err", "policy_value_abs_err", "policy_regret"]:
        mats[metric] = (
            method.pivot(index="dataset_short", columns="model", values=metric)
            .reindex(index=row_order, columns=model_order)
        )

    _label_panel(axes[0], "A", "")
    _heatmap(axes[0], mats["ate_abs_err"], "ATE recovery error (lower is better)", "YlOrRd", ".3f", True)
    _label_panel(axes[1], "B", "")
    _heatmap(axes[1], mats["policy_value_abs_err"], "Policy value calibration error (lower is better)", "YlGnBu", ".3f", True)
    _label_panel(axes[2], "C", "")
    _heatmap(axes[2], mats["policy_regret"], "Policy regret under selected arm (lower is better)", "PuRd", ".3f", True)

    ax = axes[3]
    _label_panel(ax, "D", "Macro rank across ATE, value, and regret")
    plot = macro.set_index("model").reindex(model_order).reset_index()
    y = np.arange(len(plot))
    bars = ax.barh(y, plot["primary_macro_rank"], color=[MODEL_COLOR[m] for m in plot["model"]], alpha=0.92)
    ax.set_yticks(y, _model_labels(list(plot["model"])))
    ax.invert_yaxis()
    ax.set_xlabel("Mean within-dataset rank (lower is better)")
    ax.set_xlim(0.8, max(6.1, plot["primary_macro_rank"].max() + 0.5))
    for bar, val in zip(bars, plot["primary_macro_rank"]):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", ha="left", fontsize=8.3)
    ax.grid(axis="x", alpha=0.22)
    ax.text(
        0.01,
        -0.18,
        "Ranks are computed inside each dataset and primary metric before macro-averaging.",
        transform=ax.transAxes,
        fontsize=8.1,
        color="#555555",
    )

    fig.suptitle("Fig. 2. Six-model results across three real intervention datasets", x=0.02, ha="left", fontweight="bold")
    _save(fig, out_dir, "Fig2_six_model_real_rct_results")


def fig3_calibration(ate: pd.DataFrame, taskavg: pd.DataFrame, out_dir: Path) -> None:
    model_order = [m for m in MODEL_ORDER if m in ate["model"].unique()]
    task_model = (
        ate.groupby(["dataset", "dataset_short", "task_key", "model", "label"], as_index=False)
        .agg(true_ate=("true_ate", "mean"), ate_hat=("ate_hat", "mean"))
    )
    fig = plt.figure(figsize=(15.4, 8.8), constrained_layout=False)
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.05, 0.95], hspace=0.42, wspace=0.30)
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    for idx, spec in enumerate(DATASETS):
        ax = axes_top[idx]
        _label_panel(ax, chr(ord("A") + idx), f"{spec.short_label}: true vs generated ATE")
        subset = task_model[task_model["dataset"].eq(spec.key)]
        lim = float(np.nanmax(np.abs(np.r_[subset["true_ate"].to_numpy(), subset["ate_hat"].to_numpy()])))
        lim = max(lim * 1.12, 0.04)
        ax.plot([-lim, lim], [-lim, lim], color="#333333", lw=0.9, ls="--")
        ax.axhline(0, color="#bbbbbb", lw=0.7)
        ax.axvline(0, color="#bbbbbb", lw=0.7)
        for model in model_order:
            g = subset[subset["model"].eq(model)]
            ax.scatter(g["true_ate"], g["ate_hat"], s=28, color=MODEL_COLOR[model], alpha=0.72, edgecolor="white", linewidth=0.35, label=MODEL_LABEL[model])
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("True ATE")
        ax.set_ylabel("Generated do-ATE")
        ax.grid(True, alpha=0.18)
    axes_top[0].legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.28))

    _label_panel(ax_d, "D", "Task-averaged ATE error")
    pivot_err = (
        taskavg.pivot(index="dataset_short", columns="model", values="taskavg_ate_abs_err")
        .reindex(index=[d.short_label for d in DATASETS], columns=model_order)
    )
    _heatmap(ax_d, pivot_err, "Task-average abs. error", "Oranges", ".3f", True)

    _label_panel(ax_e, "E", "ATE sign recovery")
    pivot_sign = (
        taskavg.pivot(index="dataset_short", columns="model", values="taskavg_sign_accuracy")
        .reindex(index=[d.short_label for d in DATASETS], columns=model_order)
    )
    _heatmap(ax_e, pivot_sign, "Task-average sign accuracy", "Greens", ".2f", False)

    _label_panel(ax_f, "F", "Correlation with randomized effects")
    pivot_corr = (
        taskavg.pivot(index="dataset_short", columns="model", values="taskavg_pearson")
        .reindex(index=[d.short_label for d in DATASETS], columns=model_order)
    )
    _heatmap(ax_f, pivot_corr, "Task-average Pearson r", "Blues", ".2f", False)

    fig.suptitle("Fig. 3. Counterfactual effect calibration on real randomized interventions", x=0.02, ha="left", fontweight="bold")
    _save(fig, out_dir, "Fig3_counterfactual_effect_calibration")


def fig4_policy_diagnostics(ate: pd.DataFrame, method: pd.DataFrame, best_counts: pd.DataFrame, out_dir: Path) -> None:
    model_order = [m for m in MODEL_ORDER if m in ate["model"].unique()]
    fig = plt.figure(figsize=(15.1, 9.0), constrained_layout=False)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _label_panel(ax_a, "A", "Treatment-arm selection vs oracle arm")
    sel = (
        ate.groupby(["dataset", "dataset_short", "model"], as_index=False)
        .agg(selected_treatment_rate=("policy_selected_arm", "mean"), oracle_treatment_rate=("oracle_best_arm", "mean"))
    )
    for spec in DATASETS:
        oracle = sel.loc[sel["dataset"].eq(spec.key), "oracle_treatment_rate"].mean()
        ax_a.axhline(oracle, color=DATASET_COLOR[spec.key], lw=1.0, ls="--", alpha=0.65)
    width = 0.12
    centers = np.arange(len(DATASETS))
    for j, model in enumerate(model_order):
        vals = [sel.loc[sel["dataset"].eq(d.key) & sel["model"].eq(model), "selected_treatment_rate"].mean() for d in DATASETS]
        ax_a.bar(centers + (j - (len(model_order) - 1) / 2) * width, vals, width=width, color=MODEL_COLOR[model], label=MODEL_LABEL[model], alpha=0.9)
    ax_a.set_xticks(centers, [d.short_label for d in DATASETS])
    ax_a.set_ylabel("Selected treatment rate")
    ax_a.set_ylim(0, 1.02)
    ax_a.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.18))
    ax_a.text(0.01, 0.03, "Dashed lines show each dataset's oracle treatment-arm rate.", transform=ax_a.transAxes, fontsize=8.1, color="#555555")

    _label_panel(ax_b, "B", "Regret-error tradeoff by model and dataset")
    for model in model_order:
        g = method[method["model"].eq(model)]
        ax_b.scatter(g["ate_abs_err"], g["policy_regret"], s=72, color=MODEL_COLOR[model], alpha=0.78, edgecolor="white", linewidth=0.7, label=MODEL_LABEL[model])
        for _, row in g.iterrows():
            ax_b.text(row["ate_abs_err"] + 0.0009, row["policy_regret"] + 0.0004, row["dataset_short"], fontsize=7.0, color=MODEL_COLOR[model])
    ax_b.set_xlabel("ATE abs. error")
    ax_b.set_ylabel("Policy regret")
    ax_b.grid(True, alpha=0.2)

    _label_panel(ax_c, "C", "Task-level winners by ATE recovery")
    counts = best_counts.copy()
    bottoms = np.zeros(len(DATASETS))
    x = np.arange(len(DATASETS))
    for model in model_order:
        vals = [
            int(counts.loc[counts["dataset"].eq(d.key) & counts["model"].eq(model), "n_task_wins"].sum())
            for d in DATASETS
        ]
        ax_c.bar(x, vals, bottom=bottoms, color=MODEL_COLOR[model], label=MODEL_LABEL[model], alpha=0.92)
        for i, val in enumerate(vals):
            if val > 0:
                ax_c.text(i, bottoms[i] + val / 2, str(val), ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottoms += np.asarray(vals)
    ax_c.set_xticks(x, [d.short_label for d in DATASETS])
    ax_c.set_ylabel("Number of experiment tasks")
    ax_c.set_ylim(0, max(bottoms) * 1.12)

    _label_panel(ax_d, "D", "Primary metric rank profile")
    primary = ["ate_abs_err", "policy_value_abs_err", "policy_regret"]
    rank_rows = []
    for dataset, group in method.groupby("dataset"):
        for metric in primary:
            ranks = group[["model", metric]].copy()
            ranks["rank"] = ranks[metric].rank(method="average", ascending=True)
            ranks["metric"] = metric
            ranks["dataset"] = dataset
            rank_rows.append(ranks[["dataset", "model", "metric", "rank"]])
    ranks = pd.concat(rank_rows, ignore_index=True)
    rank_pivot = (
        ranks.groupby(["model", "metric"], as_index=False)["rank"].mean()
        .pivot(index="model", columns="metric", values="rank")
        .reindex(index=model_order, columns=primary)
    )
    _heatmap(ax_d, rank_pivot, "Mean rank across datasets", "Purples", ".1f", True)
    ax_d.set_yticks(np.arange(len(model_order)), _model_labels(model_order))
    ax_d.set_xticks(np.arange(len(primary)), [METRIC_LABEL[m] for m in primary], rotation=25, ha="right")

    fig.suptitle("Fig. 4. Policy behavior and robustness diagnostics", x=0.02, ha="left", fontweight="bold")
    _save(fig, out_dir, "Fig4_policy_and_robustness_diagnostics")


def _write_markdown(
    out_dir: Path,
    dataset_summary: pd.DataFrame,
    method: pd.DataFrame,
    macro: pd.DataFrame,
    taskavg: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    best_macro = macro.iloc[0]
    scm_by_dataset = method[method["model"].eq("scm_causal")].set_index("dataset_short")
    macro_lines = []
    for _, row in macro.iterrows():
        macro_lines.append(
            f"- {row['label']}: ATE {row['ate_abs_err']:.4f}, value {row['policy_value_abs_err']:.4f}, regret {row['policy_regret']:.4f}, macro-rank {row['primary_macro_rank']:.2f}"
        )

    digest = [
        "# Real RCT Results Digest",
        "",
        "## Scope",
        "",
        "Only real randomized ASSISTments intervention datasets are included. Study1 from the OSF TACO release is the same 22-experiment LAS2016 data, so it is not double-counted as a third benchmark.",
        "",
        "## Dataset Sources",
        "",
    ]
    for spec in DATASETS:
        row = dataset_summary[dataset_summary["dataset"].eq(spec.key)].iloc[0]
        digest.append(
            f"- {spec.short_label}: {int(row['n_tasks'])} tasks, {int(row['n_students']):,} student-task rows, source {spec.source}"
        )
    digest.extend(
        [
            "",
            "## Main Macro Results",
            "",
            *macro_lines,
            "",
            "## SCM-Causal By Dataset",
            "",
        ]
    )
    for short, row in scm_by_dataset.iterrows():
        digest.append(
            f"- {short}: ATE {row['ate_abs_err']:.4f}, policy value {row['policy_value_abs_err']:.4f}, regret {row['policy_regret']:.4f}, sign {row['sign_accuracy']:.3f}, Pearson {row['pearson']:.3f}"
        )
    digest.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The strongest overall method by macro rank is {best_macro['label']}. Baseline models can produce do(T=0/1) estimates because they implement the same generator rollout API, but the audit verifies that true ATEs, randomized arm means, and treatment condition are not used as training features.",
            "",
            "PEHE is intentionally left undefined for these real RCTs because individual paired potential outcomes are not observed.",
        ]
    )
    (out_dir / "results_digest.md").write_text("\n".join(digest) + "\n", encoding="utf-8")

    captions = [
        "# Figure Captions",
        "",
        "Fig. 1. Real ASSISTments intervention benchmarks and leakage controls. Panel A reports the number of randomized intervention tasks and student-task rows. Panel B shows empirical randomized ATEs. Panels C-D summarize feature blocking and leakage checks.",
        "",
        "Fig. 2. Six-model results across three real intervention datasets. Heatmaps report ATE recovery error, policy value calibration error, and policy regret. Bold entries indicate the best method per dataset. Panel D shows the macro rank across the three primary metrics.",
        "",
        "Fig. 3. Counterfactual effect calibration on real randomized interventions. Panels A-C compare randomized ATEs against generated do-ATEs averaged across seeds. Panels D-F report task-averaged error, sign recovery, and Pearson correlation.",
        "",
        "Fig. 4. Policy behavior and robustness diagnostics. Panel A compares selected treatment-arm rates against oracle arm rates. Panel B shows regret-error tradeoffs. Panel C counts task-level ATE-recovery winners. Panel D reports the primary metric rank profile.",
    ]
    (out_dir / "figure_captions.md").write_text("\n".join(captions) + "\n", encoding="utf-8")

    plan = [
        "# Figure Plan",
        "",
        "- Fig1_real_rct_benchmark: one multi-panel benchmark and leakage-control figure.",
        "- Fig2_six_model_real_rct_results: one multi-panel main result figure for all six required baselines.",
        "- Fig3_counterfactual_effect_calibration: one multi-panel calibration figure using randomized ATEs.",
        "- Fig4_policy_and_robustness_diagnostics: one multi-panel policy and robustness figure.",
        "",
        "Each figure number maps to one image file, with both PNG and PDF export.",
    ]
    (out_dir / "figure_plan.md").write_text("\n".join(plan) + "\n", encoding="utf-8")

    leakage = [
        "# Leakage And Counterfactual Audit",
        "",
        "## Audit Summary",
        "",
        _markdown_table(audit),
        "",
        "## Counterfactual Generation Rule",
        "",
        "All methods are evaluated through the same do(T=0/1) rollout interface. The models do not receive true ATE labels, randomized arm means, or treatment condition as covariates.",
        "",
        "## Real-RCT Metric Constraint",
        "",
        "PEHE is not reported because real randomized logs reveal one realized outcome per student-task instance, not both individual potential outcomes.",
    ]
    (out_dir / "leakage_and_counterfactual_audit.md").write_text("\n".join(leakage) + "\n", encoding="utf-8")

    taskavg.to_csv(out_dir / "task_averaged_summary.csv", index=False)


def build_outputs(root: Path, out_dir: Path) -> None:
    _setup_style()
    method, ate, task, selection, audit = _load_all(root)
    method = method[method["model"].isin(MODEL_ORDER)].copy()
    ate = ate[ate["model"].isin(MODEL_ORDER)].copy()

    dataset_summary = _dataset_summary(task, selection, audit)
    taskavg = _task_averaged_summary(ate)
    macro = _macro_summary(method)
    best_counts = _best_counts_by_task(ate)

    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    method.to_csv(table_dir / "method_summary_by_dataset.csv", index=False)
    ate.to_csv(table_dir / "ate_recovery_all_seeds.csv", index=False)
    task.to_csv(table_dir / "task_summary_all_datasets.csv", index=False)
    selection.to_csv(table_dir / "selection_by_seed_all_datasets.csv", index=False)
    audit.to_csv(table_dir / "leakage_audit_summary.csv", index=False)
    dataset_summary.to_csv(table_dir / "dataset_summary.csv", index=False)
    taskavg.to_csv(table_dir / "task_averaged_summary.csv", index=False)
    macro.to_csv(table_dir / "method_summary_macro.csv", index=False)
    best_counts.to_csv(table_dir / "best_model_counts_by_task.csv", index=False)

    fig1_benchmark(task, dataset_summary, audit, fig_dir)
    fig2_main_results(method, macro, fig_dir)
    fig3_calibration(ate, taskavg, fig_dir)
    fig4_policy_diagnostics(ate, method, best_counts, fig_dir)
    _write_markdown(out_dir, dataset_summary, method, macro, taskavg, audit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AIED real-RCT paper figures and summary tables.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts/aied_real_rct_paper"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_outputs(args.root.resolve(), args.out_dir)


if __name__ == "__main__":
    main()
