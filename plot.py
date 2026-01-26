import os, glob, re, argparse
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODEL_ORDER = ["scm_causal", "crn", "timegan", "rcgan", "vae", "diffusion"]
MODEL_NAME = {
    "scm_causal": "CausalSeqGAN",
    "crn": "CRN",
    "timegan": "TimeGAN",
    "rcgan": "RCGAN",
    "vae": "VAE",
    "diffusion": "Diffusion",
}
DATASET_ORDER = ["assist09", "oulad", "statics", "irt_synth"]
DATASET_LABEL = {
    "assist09": "ASSIST09",
    "oulad": "OULAD",
    "statics": "Statics2011",
    "irt_synth": "IRT-Synth",
}
POLICY_ORDER = ["never", "always", "early_on", "late_on"]
POLICY_LABEL = {"never": "Never", "always": "Always", "early_on": "Early-on", "late_on": "Late-on"}

PLOT_STYLE = {
    "figure.dpi": 120,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}


def parse_sections(text: str) -> dict:
    """
    Parse a results_summary_*.txt file that is organized as:
    <section_name>
    <fixed-width table>
    <blank line>
    <section_name>
    <table> ...
    """
    blocks = re.split(r"\n\s*\n", text.strip())
    sections = {}
    for b in blocks:
        lines = b.splitlines()
        if not lines:
            continue
        name = lines[0].strip()
        table_str = "\n".join(lines[1:]).strip()
        if not table_str or table_str.strip().lower() == "(empty)":
            sections[name] = pd.DataFrame()
            continue
        try:
            df = pd.read_fwf(StringIO(table_str))
            df.columns = [str(c).strip() for c in df.columns]
            for col in df.columns:
                df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
            sections[name] = df
        except Exception:
            # If a section isn't parseable as a table, keep raw text.
            sections[name] = table_str
    return sections


def load_results(results_dir: Path) -> dict:
    for search_dir in [results_dir, results_dir / "results"]:
        paths = sorted(glob.glob(str(search_dir / "results_summary_*.txt")))
        if not paths:
            continue
        out = {}
        for p in paths:
            key = re.search(r"results_summary_(.+)\.txt", os.path.basename(p)).group(1)
            out[key] = parse_sections(Path(p).read_text(encoding="utf-8", errors="ignore"))
        return out
    print(f"[plot] No results_summary_*.txt found under: {results_dir.resolve()}")
    return {}


def _read_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if isinstance(df, pd.DataFrame) and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out


def load_dr_policy_values_from_csv(base_dir: Path) -> pd.DataFrame:
    patterns = [
        base_dir / "dr_policy_values*.csv",
        base_dir / "results" / "dr_policy_values*.csv",
    ]
    paths = []
    for pat in patterns:
        paths.extend([Path(p) for p in glob.glob(str(pat))])
    df = _read_csvs(paths)
    if df.empty:
        return df
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str).str.strip()
    for col in ["horizon", "dr_value", "w_max", "supported"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_policy_long_from_df(df: pd.DataFrame, horizon: int = 29) -> pd.DataFrame:
    cols = ["Dataset", "Model", "Policy", "PolicyLabel", "DR", "w_max", "supported"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    if not {"model", "policy"}.issubset(df.columns):
        return pd.DataFrame(columns=cols)
    work = df.copy()
    if "horizon" in work.columns:
        work = work[work["horizon"] == horizon]
    datasets = []
    if "dataset" in work.columns:
        seen = set(str(x).strip() for x in work["dataset"].dropna().unique().tolist())
        datasets = [d for d in DATASET_ORDER if d in seen]
        if not datasets:
            datasets = sorted(seen)
    else:
        datasets = DATASET_ORDER

    rows = []
    for ds in datasets:
        for m in MODEL_ORDER:
            for pol in POLICY_ORDER:
                sel = work[(work.get("dataset", ds) == ds) & (work["model"] == m) & (work["policy"] == pol)]
                if len(sel) == 0:
                    dr, wmax, sup = np.nan, np.nan, 0.0
                else:
                    dr = float(sel["dr_value"].iloc[0]) if "dr_value" in sel.columns else np.nan
                    wmax = float(sel["w_max"].iloc[0]) if "w_max" in sel.columns else np.nan
                    sup = float(sel["supported"].iloc[0]) if "supported" in sel.columns else 1.0
                rows.append(
                    {
                        "Dataset": DATASET_LABEL.get(ds, ds),
                        "Model": MODEL_NAME[m],
                        "Policy": pol,
                        "PolicyLabel": POLICY_LABEL[pol],
                        "DR": dr,
                        "w_max": wmax,
                        "supported": sup,
                    }
                )
    return pd.DataFrame(rows, columns=cols)


def load_oracle_policy_from_csv(base_dir: Path) -> pd.DataFrame:
    patterns = [
        base_dir / "oracle_policy_metrics*.csv",
        base_dir / "results" / "oracle_policy_metrics*.csv",
    ]
    paths = []
    for pat in patterns:
        paths.extend([Path(p) for p in glob.glob(str(pat))])
    df = _read_csvs(paths)
    if df.empty:
        return df
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str).str.strip()
        df = df[df["dataset"] == "irt_synth"]
    for col in ["oracle_value", "gen_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_policy_long(results: dict, horizon: int = 29) -> pd.DataFrame:
    rows = []
    for ds in DATASET_ORDER:
        sec = results.get(ds, {})
        if "dr_policy_values" not in sec:
            continue
        df = sec["dr_policy_values"].copy()
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if not {"model", "policy"}.issubset(df.columns):
            continue
        for col in ["horizon", "dr_value", "w_max", "supported"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "horizon" in df.columns:
            df = df[df["horizon"] == horizon]
        for m in MODEL_ORDER:
            df_m = df[df["model"] == m]
            for pol in POLICY_ORDER:
                r = df_m[df_m["policy"] == pol]
                if len(r) == 0:
                    dr, wmax, sup = np.nan, np.nan, 0.0
                else:
                    dr = float(r["dr_value"].iloc[0])
                    wmax = float(r["w_max"].iloc[0]) if "w_max" in r.columns else np.nan
                    sup = float(r["supported"].iloc[0]) if "supported" in r.columns else 1.0
                rows.append(
                    {
                        "Dataset": DATASET_LABEL.get(ds, ds),
                        "Model": MODEL_NAME[m],
                        "Policy": pol,
                        "PolicyLabel": POLICY_LABEL[pol],
                        "DR": dr,
                        "w_max": wmax,
                        "supported": sup,
                    }
                )
    cols = ["Dataset", "Model", "Policy", "PolicyLabel", "DR", "w_max", "supported"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def plot_dr_policy_profile(policy_long: pd.DataFrame, out_path: Path,
                           selected_models=None):
    if selected_models is None:
        selected_models = [MODEL_NAME[m] for m in MODEL_ORDER]

    required_cols = {"Dataset", "Model", "PolicyLabel", "DR", "supported"}
    if policy_long is None or policy_long.empty or not required_cols.issubset(policy_long.columns):
        print("[plot] dr_policy_values missing or empty; skip fig_dr_policy_values.")
        return

    df = policy_long[policy_long["Model"].isin(selected_models)].copy()
    if df.empty:
        print("[plot] dr_policy_values has no rows for selected models; skip fig_dr_policy_values.")
        return

    # Per-dataset line plots with 4 policies on X axis.
    policy_labels = [POLICY_LABEL[p] for p in POLICY_ORDER]
    x = np.arange(len(policy_labels))
    datasets = [DATASET_LABEL[d] for d in DATASET_ORDER if DATASET_LABEL[d] in df["Dataset"].unique()]
    if not datasets:
        datasets = sorted(df["Dataset"].unique().tolist())

    n = len(datasets)
    ncols = 2
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    markers = ["o", "s", "^", "D", "P", "X"]
    for i, ds in enumerate(datasets):
        ax = axes[i]
        df_ds = df[df["Dataset"] == ds]
        for j, m in enumerate(selected_models):
            sub = df_ds[df_ds["Model"] == m].set_index("PolicyLabel").reindex(policy_labels)
            y = sub["DR"].astype(float).to_numpy()
            sup = sub["supported"].astype(float).to_numpy()
            y = np.where((sup == 0) | np.isnan(y), np.nan, y)
            ax.plot(x, y, marker=markers[j % len(markers)], linewidth=1.6, label=m)
        ax.set_title(ds, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(policy_labels)
        ax.grid(axis="y", alpha=0.3)

    # Hide any unused subplots
    for k in range(n, len(axes)):
        axes[k].axis("off")

    axes[0].set_ylabel("DR policy value")
    # No global title (requested).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_oracle_vs_generated(results: dict, out_path: Path):
    df = pd.DataFrame()
    sec = results.get("irt_synth", {}) if isinstance(results, dict) else {}
    if isinstance(sec, dict) and "oracle_policy_metrics" in sec:
        df = sec["oracle_policy_metrics"].copy()
    if not isinstance(df, pd.DataFrame) or df.empty or "model" not in df.columns:
        df = load_oracle_policy_from_csv(out_path.parent)
    if not isinstance(df, pd.DataFrame) or df.empty or "model" not in df.columns:
        print("[plot] oracle_policy_metrics missing or malformed; skip fig_oracle_vs_generated.")
        return
    for col in ["oracle_value", "gen_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["model"].isin(MODEL_ORDER)]
    df["Model"] = df["model"].map(MODEL_NAME)
    df["Policy"] = df["policy"].map(POLICY_LABEL).fillna(df["policy"])

    plt.figure(figsize=(5.2, 5.2))
    markers = {"CausalSeqGAN": "o", "CRN": "s", "TimeGAN": "^", "RCGAN": "D", "VAE": "P", "Diffusion": "X"}

    for m in ["CausalSeqGAN", "CRN", "TimeGAN", "RCGAN", "VAE", "Diffusion"]:
        sub = df[df["Model"] == m]
        if len(sub) == 0:
            continue
        plt.scatter(
            sub["oracle_value"],
            sub["gen_value"],
            label=m,
            marker=markers.get(m, "o"),
            s=40,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )

    # y = x reference
    valid = df[["oracle_value", "gen_value"]].dropna()
    if valid.empty:
        print("[plot] oracle_policy_metrics has no finite oracle/gen values; skip fig_oracle_vs_generated.")
        return
    lo = float(valid.min().min())
    hi = float(valid.max().max())
    pad = max(0.02, (hi - lo) * 0.05)
    lims = [lo - pad, hi + pad]
    plt.plot(lims, lims, linestyle="--", color="#444", linewidth=1.0)
    plt.xlim(lims)
    plt.ylim(lims)

    # Policy labels above each oracle-value column.
    policy_labels = [POLICY_LABEL[p] for p in POLICY_ORDER]
    y_text = lims[1] - (lims[1] - lims[0]) * 0.02
    for pol in policy_labels:
        sub_pol = df[df["Policy"] == pol]
        if sub_pol.empty:
            continue
        x = float(np.nanmedian(sub_pol["oracle_value"]))
        plt.text(x, y_text, pol, ha="center", va="top", fontsize=8)
    plt.xlabel("Oracle policy value")
    plt.ylabel("Generated policy value")
    # No title (requested).
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=".", help="Directory containing results_summary_*.txt")
    ap.add_argument("--out", type=str, default=".", help="Output directory for figures")
    ap.add_argument("--horizon", type=int, default=29, help="Horizon used in dr_policy_values")
    args, _unknown = ap.parse_known_args()

    plt.rcParams.update(PLOT_STYLE)
    results_dir = Path(args.dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    dr_csv = load_dr_policy_values_from_csv(results_dir)
    if not dr_csv.empty:
        pol_long = build_policy_long_from_df(dr_csv, horizon=args.horizon)
    else:
        pol_long = build_policy_long(results, horizon=args.horizon)

    if pol_long.empty:
        print("[plot] dr_policy_values not found or empty in results_summary/csv; no policy figure will be saved.")

    plot_dr_policy_profile(pol_long, out_dir / "fig_dr_policy_values.png")
    plot_oracle_vs_generated(results, out_dir / "fig_oracle_vs_generated.png")

    print("Saved:")
    print(" -", (out_dir / "fig_dr_policy_values.png").resolve())
    print(" -", (out_dir / "fig_oracle_vs_generated.png").resolve())

if __name__ == "__main__":
    main()



