import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


MODEL_NAME = {
    "scm_causal": "CausalSeqGAN",
}


def _decode_labels(arr) -> list[str]:
    labels = []
    for item in arr:
        if isinstance(item, (bytes, bytearray)):
            labels.append(item.decode("utf-8"))
        else:
            labels.append(str(item))
    return labels


def _parse_list(text: str) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _pretty_label(label: str) -> str:
    return MODEL_NAME.get(label, label)


def _available_models_from_keys(keys: list[str]) -> list[str]:
    y_a = {k.replace("y_a_", "", 1) for k in keys if k.startswith("y_a_")}
    y_b = {k.replace("y_b_", "", 1) for k in keys if k.startswith("y_b_")}
    return sorted(y_a.intersection(y_b))


def _normalize_models(requested: list[str], available: list[str]) -> tuple[list[str], list[str]]:
    if not requested:
        return available, []
    lower_map = {m.lower(): m for m in available}
    alias_map = {v.lower(): k for k, v in MODEL_NAME.items()}
    selected = []
    missing = []
    for raw in requested:
        key = raw
        if raw in available:
            selected.append(raw)
            continue
        key_l = raw.lower()
        if key_l in lower_map:
            selected.append(lower_map[key_l])
            continue
        if key_l in alias_map and alias_map[key_l] in available:
            selected.append(alias_map[key_l])
            continue
        missing.append(raw)
    return selected, missing


def _dataset_from_path(path: Path) -> str:
    m = re.search(r"cf_bifurcation_(.+?)_all", path.name)
    return m.group(1) if m else path.stem


def _extract_series(data: dict, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    y_a = data.get(f"y_a_{label}")
    y_b = data.get(f"y_b_{label}")
    if y_a is None or y_b is None:
        raise KeyError(f"Missing y series for model={label}")

    k_a = data.get(f"k_a_{label}")
    k_b = data.get(f"k_b_{label}")
    k_metric_a = data.get(f"k_metric_{label}")
    k_metric_b = data.get(f"k_metric_{label}_b")

    if k_metric_a is None and k_a is not None:
        k_metric_a = np.linalg.norm(k_a, axis=-1)
    if k_metric_b is None and k_b is not None:
        k_metric_b = np.linalg.norm(k_b, axis=-1)

    return y_a, y_b, k_metric_a, k_metric_b


def plot_npz(
    npz_path: Path,
    *,
    models: list[str] | None = None,
    out_path: Path | None = None,
    show: bool = False,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    data_dict = {k: data[k] for k in data.files}
    labels = _decode_labels(data_dict.get("model_labels", []))
    available = _available_models_from_keys(list(data_dict.keys()))
    if labels:
        available = [m for m in labels if m in available] or available
    if not available:
        raise ValueError(f"No model series found in {npz_path}")

    selected, missing = _normalize_models(models or [], available)
    if missing:
        print(f"[plot] Warning: models not found in {npz_path.name}: {missing}")
    if not selected:
        raise ValueError("No valid models selected for plotting.")
    print(f"[plot] using models: {selected}")

    t_intervention = int(np.asarray(data_dict.get("t_intervention", 0)).reshape(-1)[0])
    action_a = int(np.asarray(data_dict.get("action_a", 0)).reshape(-1)[0])
    action_b = int(np.asarray(data_dict.get("action_b", 1)).reshape(-1)[0])
    dataset = _dataset_from_path(npz_path)

    fig, (ax_y, ax_k) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    any_k = False

    for idx, label in enumerate(selected):
        y_a, y_b, k_a, k_b = _extract_series(data_dict, label)
        color = f"C{idx % 10}"
        times = np.arange(len(y_a))
        ax_y.plot(times, y_a, color=color, linewidth=2, label=f"{_pretty_label(label)}: do={action_a}")
        ax_y.plot(times, y_b, color=color, linestyle="--", linewidth=2, label=f"{_pretty_label(label)}: do={action_b}")

        if k_a is not None and k_b is not None:
            times_k = np.arange(len(k_a))
            ax_k.plot(times_k, k_a, color=color, linewidth=2, label=f"{_pretty_label(label)}: do={action_a}")
            ax_k.plot(times_k, k_b, color=color, linestyle="--", linewidth=2, label=f"{_pretty_label(label)}: do={action_b}")
            any_k = True

    ax_y.axvline(x=t_intervention, color="gray", linestyle=":")
    ax_y.set_ylabel("Predicted Y")
    ax_y.legend(ncol=2, fontsize=8)
    ax_y.grid(True, alpha=0.3)

    ax_k.axvline(x=t_intervention, color="gray", linestyle=":")
    ax_k.set_ylabel("Latent K (norm)")
    ax_k.set_xlabel("Time Step")
    if any_k:
        ax_k.legend(ncol=2, fontsize=8)
        ax_k.grid(True, alpha=0.3)
    else:
        ax_k.text(0.5, 0.5, "No latent K available", ha="center", va="center", transform=ax_k.transAxes)
        ax_k.grid(True, alpha=0.3)

    fig.suptitle(f"Counterfactual Bifurcation (Same Student, Multi-Model) - {dataset}")
    fig.tight_layout()
    if out_path is None:
        out_path = npz_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot cf_bifurcation_*_all.npz with optional model selection.")
    ap.add_argument("--npz", type=str, default="", help="Path to a single .npz file")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts", help="Artifacts directory to scan")
    ap.add_argument("--pattern", type=str, default="cf_bifurcation_*_all.npz", help="Glob pattern under artifacts_dir")
    ap.add_argument("--models", type=str, default="", help="Comma-separated model labels to plot")
    ap.add_argument("--show", action="store_true", help="Display the plot window")
    ap.add_argument("--list-models", action="store_true", help="List models in the provided --npz and exit")
    args = ap.parse_args()

    models = _parse_list(args.models)
    if args.npz:
        paths = [Path(args.npz)]
    else:
        base = Path(args.artifacts_dir)
        paths = sorted(base.glob(args.pattern))

    if not paths:
        print("[plot] No .npz files found to plot.")
        return 1

    if args.list_models:
        path = paths[0]
        data = np.load(path, allow_pickle=True)
        labels = _decode_labels(data.get("model_labels", []))
        available = _available_models_from_keys(list(data.files))
        if labels:
            available = [m for m in labels if m in available] or available
        print(f"[plot] models in {path.name}: {available}")
        return 0

    for p in paths:
        try:
            plot_npz(p, models=models if models else None, out_path=None, show=args.show)
            print(f"[plot] wrote: {p.with_suffix('.png')}")
        except Exception as e:
            print(f"[plot] failed for {p.name}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
