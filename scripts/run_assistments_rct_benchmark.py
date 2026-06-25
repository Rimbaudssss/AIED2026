from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import load_base_model_from_checkpoint
from src.data import TrajectoryBatch, TrajectoryDataset, compute_lengths
from src.main import (
    MODEL_KNOBS,
    _make_dataloader,
    _set_seed,
    _train_crn,
    _train_diffusion,
    _train_scm_or_rcgan,
    _train_timegan,
    _train_vae,
)

DATA_DIR = ROOT / "DataSet" / "assistments_rct88" / "env" / "data"
LAS_DATA_DIR = ROOT / "DataSet" / "assistments_las2016" / "raw"
ABTEST_DATA_DIR = ROOT / "DataSet" / "assistments_abtest_remnant_osf" / "raw" / "j6esa"
PROCESSED = DATA_DIR / "processed_experiment_data.csv"
RAW_DIR = DATA_DIR / "experiment_data"
OUT_DIR = ROOT / "artifacts" / "assistments_rct88_scm_models"

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
    "scm_causal": "#1f4e79",
    "rcgan": "#e15759",
    "vae": "#59a14f",
    "diffusion": "#9c755f",
    "crn": "#4e79a7",
    "timegan": "#f28e2b",
}
SEEDS = [42, 43, 44]
EVAL_VERSION = 2
REQUIRED_RESULT_COLUMNS = {
    "eval_version",
    "true_value_t0",
    "true_value_t1",
    "model_value_t0",
    "model_value_t1",
    "ate_abs_err",
    "policy_value_abs_err",
    "policy_regret",
    "fixed_policy_value_abs_err",
    "pehe",
    "pehe_supported",
}
ID_COLS = {
    "dataset_name",
    "experiment_id",
    "dependent_measure",
    "independent_measure_pair",
    "condition",
    "student_id",
    "assignment_completed",
    "normalized_student_learning",
}
DATASET_LABELS = {
    "assistments_rct88": "ASSISTments RCT88/89",
    "assistments_las2016": "ASSISTments LAS2016 22 RCTs",
    "assistments_abtest_study2": "ASSISTments OSF Study2 11 A/B Tests",
}
LAS_PRETREATMENT_COLUMNS = [
    "Prior Problem Count",
    "Prior Correct Count",
    "Prior Percent Correct",
    "Class ID",
    "Class Section ID",
    "Class Grade",
    "Teacher ID",
    "Guessed Gender",
    "Birthyear",
    "school_id",
    "District ID",
    "State ID",
    "Prior Assignments Assigned",
    "Prior Assignment Count",
    "Prior Completion Count",
    "Prior Percent Completion",
    "Prior Class Percent Completion",
    "Z-Scored Mastery Speed",
    "Prior Homework Assigned",
    "Prior Homework Count",
    "Prior Homework Completion Count",
    "Prior Homework Percent Completion",
    "Prior Class Homework Percent Completion",
    "Z-Scored HW Mastery Speed",
]
LAS_EXCLUDED_POST_TREATMENT_COLUMNS = [
    "ExperiencedCondition",
    "Could See Video",
    "complete",
    "ProblemCount",
    "log(count)",
]
STUDY2_PRETREATMENT_COLUMNS = [
    "Prior.Problem.Count",
    "Prior.Correct.Count",
    "Prior.Percent.Correct",
    "Class.ID",
    "Class.Section.ID",
    "Class.Grade",
    "Teacher.ID",
    "Guessed.Gender",
    "Birthyear",
    "Location.ID",
    "Role.Type",
    "Location.Type",
    "School.ID",
    "District.ID",
    "State.ID",
    "Prior.Assignments.Assigned",
    "Prior.Assignment.Count",
    "Prior.Completion.Count",
    "Prior.Percent.Completion",
    "Prior.Class.Percent.Completion",
    "Z.Scored.Mastery.Speed",
    "Prior.Homework.Assigned",
    "Prior.Homework.Count",
    "Prior.Homework.Completion.Count",
    "Prior.Homework.Percent.Completion",
    "Prior.Class.Homework.Percent.Completion",
    "Z.Scored.HW.Mastery.Speed",
]
STUDY2_EXCLUDED_POST_TREATMENT_COLUMNS = [
    "Number.All.Problems.in.Posttest",
    "Number.Complete.Problems.in.Posttest",
    "Number.Correct.Problems.in.Posttest",
    "Assigment.Completed",
    "Condition",
    "actual_experiment",
    "target_assignment_id",
    "target_assignment_complete",
    "pcomplete3",
    "pcomplete2",
    "pcomplete1",
    "actual",
    "complete",
    "condition",
    "p_complete",
]


@dataclass(frozen=True)
class Task:
    task_key: str
    task_id: int
    experiment_id: str
    pair: str
    outcome: str
    n: int
    n0: int
    n1: int
    true_value_t0: float
    true_value_t1: float
    true_ate: float
    true_ate_normalized: float
    y_rate: float


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def label_panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.08, 1.08, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    ax.set_title(title, loc="left", fontweight="bold", pad=8)


def savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_processed(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "processed_experiment_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed ASSISTments RCT data: {path}")
    df = pd.read_csv(path)
    df["experiment_id"] = df["experiment_id"].astype(str)
    df["dependent_measure"] = df["dependent_measure"].astype(str)
    df["independent_measure_pair"] = df["independent_measure_pair"].astype(str)
    df["condition"] = df["condition"].astype(int)
    return df


def load_las2016(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "ThisOne.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing ASSISTments LAS2016 data: {path}")
    raw = pd.read_csv(path)
    required = {"problem_set", "User ID", "InferredCondition", "complete"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"LAS2016 file is missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "dataset_name": "assistments_las2016",
            "experiment_id": raw["problem_set"].astype(str),
            "independent_measure_pair": "Control vs. Experiment",
            "dependent_measure": "completion",
            "student_id": raw["User ID"].astype(str),
            "condition": raw["InferredCondition"].astype(str).str.strip().str.upper().map({"C": 0, "E": 1}),
            "assignment_completed": pd.to_numeric(raw["complete"], errors="coerce"),
        }
    )
    out["normalized_student_learning"] = out["assignment_completed"]

    if "Prior Percent Correct" in raw.columns:
        prior = pd.to_numeric(raw["Prior Percent Correct"], errors="coerce")
    elif "Prior Percent Completion" in raw.columns:
        prior = pd.to_numeric(raw["Prior Percent Completion"], errors="coerce")
    else:
        prior = pd.Series(np.nan, index=raw.index)
    out["student_prior_average_correctness"] = prior

    if "problem_set_name" in raw.columns:
        out["problem_set_name"] = raw["problem_set_name"].astype(str)

    for col in LAS_PRETREATMENT_COLUMNS:
        if col in raw.columns:
            out[col] = raw[col]

    out = out.dropna(subset=["condition", "assignment_completed"]).copy()
    out["condition"] = out["condition"].astype(int)
    out["assignment_completed"] = out["assignment_completed"].astype(float)
    out["normalized_student_learning"] = out["normalized_student_learning"].astype(float)
    return out


def las_inventory(data_dir: Path) -> pd.DataFrame:
    df = load_las2016(data_dir)
    rows = []
    for eid, g in df.groupby("experiment_id", sort=True):
        rows.append(
            {
                "experiment_id": str(eid),
                "has_condition_metadata": True,
                "has_raw_folder": True,
                "n_assignment_logs": 1,
                "n_problem_logs": 0,
                "n_student_logs": int(g["student_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def load_abtest_study2(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "Study2 11 New Experiments" / "TreatmentAssignment_Covariate_Outcome_Study2.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing ASSISTments OSF Study2 TACO data: {path}")
    raw = pd.read_csv(path)
    required = {"target_sequence_id", "user_id", "complete"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Study2 TACO file is missing required columns: {missing}")

    if "condition" in raw.columns:
        condition = pd.to_numeric(raw["condition"], errors="coerce")
    elif "Condition" in raw.columns:
        condition = raw["Condition"].astype(str).str.lower().map({"control": 0, "treatment": 1, "treatment1": 1})
    else:
        raise ValueError("Study2 TACO file is missing `condition`/`Condition`.")

    out = pd.DataFrame(
        {
            "dataset_name": "assistments_abtest_study2",
            "experiment_id": raw["target_sequence_id"].astype(str),
            "independent_measure_pair": "Control vs. Treatment",
            "dependent_measure": "completion",
            "student_id": raw["user_id"].astype(str),
            "condition": condition,
            "assignment_completed": pd.to_numeric(raw["complete"], errors="coerce"),
        }
    )
    out["normalized_student_learning"] = out["assignment_completed"]
    out["student_prior_average_correctness"] = pd.to_numeric(raw.get("Prior.Percent.Correct"), errors="coerce")

    for col in STUDY2_PRETREATMENT_COLUMNS:
        if col in raw.columns:
            out[col] = raw[col]

    out = out.dropna(subset=["condition", "assignment_completed"]).copy()
    out["condition"] = out["condition"].astype(int)
    out["assignment_completed"] = out["assignment_completed"].astype(float)
    out["normalized_student_learning"] = out["normalized_student_learning"].astype(float)
    return out


def abtest_study2_inventory(data_dir: Path) -> pd.DataFrame:
    df = load_abtest_study2(data_dir)
    rows = []
    for eid, g in df.groupby("experiment_id", sort=True):
        rows.append(
            {
                "experiment_id": str(eid),
                "has_condition_metadata": True,
                "has_raw_folder": True,
                "n_assignment_logs": 1,
                "n_problem_logs": 0,
                "n_student_logs": int(g["student_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def load_benchmark_data(dataset_name: str, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset_name == "assistments_rct88":
        df = load_processed(data_dir)
        df["dataset_name"] = dataset_name
        return df, raw_inventory(data_dir)
    if dataset_name == "assistments_las2016":
        return load_las2016(data_dir), las_inventory(data_dir)
    if dataset_name == "assistments_abtest_study2":
        return load_abtest_study2(data_dir), abtest_study2_inventory(data_dir)
    raise ValueError(f"Unknown dataset_name={dataset_name}")


def raw_inventory(data_dir: Path) -> pd.DataFrame:
    cond_path = data_dir / "experiment_conditions.csv"
    rows: list[dict[str, Any]] = []
    cond_ids: set[str] = set()
    if cond_path.exists():
        cond = pd.read_csv(cond_path)
        if "experiment_id" in cond.columns:
            cond_ids = set(cond["experiment_id"].astype(str))
    raw_root = data_dir / "experiment_data"
    if not raw_root.exists():
        raw_root = data_dir
    raw_ids: set[str] = set()
    if raw_root.exists():
        raw_ids = {p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("PSA")}
    for eid in sorted(cond_ids | raw_ids):
        p = raw_root / eid
        rows.append(
            {
                "experiment_id": eid,
                "has_condition_metadata": eid in cond_ids,
                "has_raw_folder": p.exists(),
                "n_assignment_logs": int(len(list(p.glob("*alog*.csv")))) if p.exists() else 0,
                "n_problem_logs": int(len(list(p.glob("*plog*.csv")))) if p.exists() else 0,
                "n_student_logs": int(len(list(p.glob("*slog*.csv")))) if p.exists() else 0,
            }
        )
    return pd.DataFrame(rows)


def category_from_pair(pair: str) -> str:
    pair = str(pair).replace("_No_Choice", "").replace("_Choice", "")
    left = pair.split(" vs. ")[0]
    left = left.split("_")[0]
    return left.strip()


def _binary_outcome(g: pd.DataFrame) -> np.ndarray:
    outcome = str(g["dependent_measure"].iloc[0])
    if outcome == "completion":
        return g["assignment_completed"].astype(float).to_numpy()
    return (g["normalized_student_learning"].astype(float).to_numpy() > 0.0).astype(float)


def _prior_y0(g: pd.DataFrame) -> np.ndarray:
    if "student_prior_average_correctness" not in g.columns:
        return np.full(len(g), 0.5, dtype=np.float32)
    y0 = pd.to_numeric(g["student_prior_average_correctness"], errors="coerce")
    med = float(y0.median()) if y0.notna().any() else 0.5
    y0 = y0.fillna(med).clip(0.0, 1.0)
    return y0.to_numpy(dtype=np.float32)


def build_tasks(df: pd.DataFrame, *, min_n: int, min_arm: int, max_tasks: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    task_rows: list[pd.DataFrame] = []
    group_cols = ["experiment_id", "independent_measure_pair", "dependent_measure"]
    for (eid, pair, outcome), g0 in df.groupby(group_cols, sort=True):
        g = g0.dropna(subset=["condition", "normalized_student_learning"]).copy()
        arms = sorted(g["condition"].astype(int).unique().tolist())
        if arms != [0, 1]:
            continue
        n0 = int((g["condition"] == 0).sum())
        n1 = int((g["condition"] == 1).sum())
        if len(g) < int(min_n) or min(n0, n1) < int(min_arm):
            continue
        y = _binary_outcome(g)
        cond = g["condition"].to_numpy()
        true_value_t0 = float(y[cond == 0].mean())
        true_value_t1 = float(y[cond == 1].mean())
        ate = float(true_value_t1 - true_value_t0)
        y_norm = g["normalized_student_learning"].astype(float).to_numpy()
        ate_norm = float(y_norm[cond == 1].mean() - y_norm[cond == 0].mean())
        task_id = len(rows)
        key = f"{eid}||{pair}||{outcome}"
        g = g.copy()
        g["task_key"] = key
        g["task_id"] = task_id
        g["y_binary"] = y.astype(np.float32)
        g["y0_prior"] = _prior_y0(g)
        task_rows.append(g)
        rows.append(
            {
                "task_key": key,
                "task_id": task_id,
                "experiment_id": str(eid),
                "independent_measure_pair": str(pair),
                "dependent_measure": str(outcome),
                "category": category_from_pair(str(pair)),
                "n": int(len(g)),
                "n0": n0,
                "n1": n1,
                "true_value_t0": true_value_t0,
                "true_value_t1": true_value_t1,
                "true_ate": ate,
                "true_ate_normalized": ate_norm,
                "y_rate": float(np.mean(y)),
            }
        )
        if max_tasks and len(rows) >= int(max_tasks):
            break
    if not rows:
        raise RuntimeError("No analyzable binary RCT tasks after filters.")
    return pd.DataFrame(rows), pd.concat(task_rows, ignore_index=True)


def prepare_features(all_rows: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    candidate_cols = [c for c in all_rows.columns if c not in ID_COLS and c not in {"task_key", "task_id", "y_binary", "y0_prior"}]
    feature_parts: list[pd.Series] = []
    names: list[str] = []
    for col in candidate_cols:
        if col == "opportunity_zone":
            s = all_rows[col].astype(str).str.lower().map({"yes": 1.0, "true": 1.0, "1": 1.0, "no": 0.0, "false": 0.0, "0": 0.0})
        else:
            s = pd.to_numeric(all_rows[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        feature_parts.append(s.astype(float))
        names.append(col)
    X = pd.concat(feature_parts, axis=1)
    X.columns = names
    med = X.median(axis=0).fillna(0.0)
    X = X.fillna(med)
    mean = X.mean(axis=0)
    std = X.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    Xs = ((X - mean) / std).to_numpy(dtype=np.float32)
    transform = {"median": med.to_dict(), "mean": mean.to_dict(), "std": std.to_dict()}
    return Xs, names, transform


def make_sequence_batch(
    rows: pd.DataFrame,
    X_all: np.ndarray,
    *,
    task_id_mode: str = "specific",
    device: torch.device | None = None,
) -> TrajectoryBatch:
    idx = rows.index.to_numpy()
    n = len(rows)
    X = torch.as_tensor(X_all[idx], dtype=torch.float32)
    A = torch.zeros((n, 2), dtype=torch.long)
    if task_id_mode == "constant":
        A[:, 1] = 1
    elif task_id_mode == "specific":
        A[:, 1] = torch.as_tensor((rows["task_id"].to_numpy(dtype=np.int64) + 1).copy(), dtype=torch.long)
    else:
        raise ValueError(f"Unknown task_id_mode={task_id_mode}")
    T = torch.zeros((n, 2), dtype=torch.long)
    T[:, 1] = torch.as_tensor(rows["condition"].to_numpy(dtype=np.int64).copy(), dtype=torch.long)
    Y = torch.zeros((n, 2), dtype=torch.float32)
    Y[:, 0] = torch.as_tensor(rows["y0_prior"].to_numpy(dtype=np.float32).copy())
    Y[:, 1] = torch.as_tensor(rows["y_binary"].to_numpy(dtype=np.float32).copy())
    M = torch.ones((n, 2), dtype=torch.float32)
    batch = TrajectoryBatch(X=X, A=A, T=T, Y=Y, mask=M, lengths=compute_lengths(M))
    return batch.to(device) if device is not None else batch


def select_biased_training_rows(
    all_rows: pd.DataFrame,
    X_all: np.ndarray,
    *,
    seed: int,
    bias_strength: float,
    min_arm: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    coefs = rng.normal(size=X_all.shape[1]).astype(np.float32)
    coefs = coefs / max(float(np.linalg.norm(coefs)), 1e-6)
    keep_indices: list[int] = []
    stats: list[dict[str, Any]] = []
    for task_key, g in all_rows.groupby("task_key", sort=False):
        idx = g.index.to_numpy()
        Xg = X_all[idx]
        score = Xg @ coefs
        score = (score - float(score.mean())) / max(float(score.std()), 1e-6)
        treat = g["condition"].to_numpy(dtype=np.float32)
        logits = float(bias_strength) * (0.75 * score + 0.55 * (2.0 * treat - 1.0) * score - 0.15 * treat)
        prob = 0.18 + 0.72 / (1.0 + np.exp(-logits))
        keep = rng.random(len(g)) < prob
        for arm in (0, 1):
            arm_pos = np.where(treat == arm)[0]
            need = min(int(min_arm), len(arm_pos))
            if keep[arm_pos].sum() < need:
                order = arm_pos[np.argsort(-prob[arm_pos])]
                keep[order[:need]] = True
        kept = idx[keep]
        keep_indices.extend(kept.tolist())
        stats.append(
            {
                "task_key": task_key,
                "seed": int(seed),
                "n_full": int(len(g)),
                "n_biased": int(len(kept)),
                "n0_biased": int(((g.loc[kept, "condition"]).to_numpy() == 0).sum()),
                "n1_biased": int(((g.loc[kept, "condition"]).to_numpy() == 1).sum()),
                "retention": float(len(kept) / max(1, len(g))),
            }
        )
    train = all_rows.loc[sorted(keep_indices)].copy()
    return train, pd.DataFrame(stats)


def latest_checkpoint(run_dir: Path, model: str) -> Path | None:
    if model == "timegan":
        p = run_dir / "ckpt_timegan.pt"
        return p if p.exists() else None
    patterns = {
        "scm_causal": "ckpt_scm_causal_*_ep*.pt",
        "rcgan": "ckpt_rcgan_*_ep*.pt",
        "vae": "ckpt_vae_ep*.pt",
        "diffusion": "ckpt_diffusion_ep*.pt",
        "crn": "ckpt_crn_ep*.pt",
    }
    files = list(run_dir.glob(patterns[model]))
    if not files:
        return None
    return sorted(files, key=lambda p: p.stat().st_mtime)[-1]


def model_knobs(model: str, args: argparse.Namespace) -> dict[str, Any]:
    knobs = dict(MODEL_KNOBS[model])
    knobs["save_checkpoints"] = False
    knobs["log_every"] = int(args.log_every)
    knobs["dropout"] = float(args.dropout)
    knobs["a_emb_dim"] = int(args.a_emb_dim)
    knobs["t_emb_dim"] = int(args.t_emb_dim)
    if model == "scm_causal":
        ea, eb, ec = [int(x) for x in args.scm_epochs]
        knobs.update(
            {
                "epochs_a": ea,
                "epochs_b": eb,
                "epochs_c": ec,
                "dynamics": str(args.scm_dynamics),
                "d_k": int(args.scm_d_k),
                "mlp_hidden": int(args.hidden),
                "y_head_hidden": int(args.scm_y_head_hidden),
                "y_head_layers": int(args.scm_y_head_layers),
                "k_c_ratio": float(args.scm_k_c_ratio),
                "w_do": float(args.scm_w_do),
                "w_cf": float(args.scm_w_cf),
                "w_advT_causal": float(args.scm_w_advT_causal),
                "w_t_pred_spurious": float(args.scm_w_t_pred_spurious),
                "tf_ffn_hidden": max(64, int(args.hidden) * 2),
                "do_time_sampling": "fixed",
                "do_time_index": 1,
                "do_num_time_samples": 1,
                "do_min_arm_samples": int(args.do_min_arm_samples),
                "do_actions": "0,1",
                "do_horizon_train": 0,
                "causal_every": int(args.causal_every),
                "cf_every": int(args.cf_every),
                "ref_estimator": None,
            }
        )
    elif model == "rcgan":
        ea, eb, ec = [int(x) for x in args.gan_epochs]
        knobs.update({"epochs_a": ea, "epochs_b": eb, "epochs_c": ec, "rcgan_hidden": int(args.hidden)})
    elif model == "crn":
        knobs.update({"epochs": int(args.epochs), "crn_hidden": int(args.hidden)})
    elif model == "vae":
        knobs.update({"epochs": int(args.epochs), "vae_enc_hidden": int(args.hidden), "vae_dec_hidden": int(args.hidden)})
    elif model == "diffusion":
        knobs.update(
            {
                "epochs": int(args.epochs),
                "diff_steps": int(args.diff_steps),
                "diff_model_dim": int(args.hidden),
                "diff_backbone": str(args.diff_backbone),
            }
        )
    elif model == "timegan":
        ee, es, ej = [int(x) for x in args.timegan_epochs]
        knobs.update(
            {
                "epochs_embed": ee,
                "epochs_supervisor": es,
                "epochs_joint": ej,
                "timegan_hidden": int(args.timegan_hidden),
                "timegan_num_layers": 2,
                "timegan_z_dim": int(args.timegan_z_dim),
                "x_emb_dim": int(args.hidden),
                "max_batches_per_epoch": int(args.timegan_max_batches),
            }
        )
    return knobs


def train_or_load_model(
    model: str,
    *,
    train_batch: TrajectoryBatch,
    meta: dict[str, Any],
    seed: int,
    device: torch.device,
    run_dir: Path,
    args: argparse.Namespace,
) -> Any:
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = latest_checkpoint(run_dir, model) if args.resume else None
    if ckpt_path is None:
        _set_seed(int(seed))
        train_ds = TrajectoryDataset(
            X=train_batch.X,
            A=train_batch.A,
            T=train_batch.T,
            Y=train_batch.Y,
            mask=train_batch.mask,
        )
        batch_size = min(int(args.batch_size), max(8, len(train_ds)))
        drop_last = len(train_ds) >= batch_size * 2
        train_dl = _make_dataloader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
        knobs = model_knobs(model, args)
        print(f"[train] seed={seed} model={model} n_train={len(train_ds)} batch={batch_size} device={device}")
        if model in {"scm_causal", "rcgan"}:
            ckpt_path = _train_scm_or_rcgan(model, train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs)
        elif model == "crn":
            ckpt_path = _train_crn(train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs)
        elif model == "vae":
            ckpt_path = _train_vae(train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs)
        elif model == "diffusion":
            ckpt_path = _train_diffusion(train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs)
        elif model == "timegan":
            ckpt_path = _train_timegan(train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs)
        else:
            raise ValueError(model)
    else:
        print(f"[resume] seed={seed} model={model} checkpoint={ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen_model = load_base_model_from_checkpoint(ckpt, device=device)
    gen_model.name = model
    return gen_model


@torch.inference_mode()
def estimate_task_values(
    gen_model: Any,
    rows: pd.DataFrame,
    X_all: np.ndarray,
    *,
    seed: int,
    model_index: int,
    mc_samples: int,
    eval_batch_size: int,
    device: torch.device,
    task_id_mode: str,
) -> dict[str, float]:
    n = len(rows)
    y0_total = 0.0
    y1_total = 0.0
    n_total = 0
    for start in range(0, n, int(eval_batch_size)):
        chunk = rows.iloc[start : start + int(eval_batch_size)]
        batch = make_sequence_batch(chunk, X_all, task_id_mode=task_id_mode, device=None)
        y0_sum = torch.zeros(len(chunk), dtype=torch.float32, device=device)
        y1_sum = torch.zeros(len(chunk), dtype=torch.float32, device=device)
        for mc in range(int(mc_samples)):
            torch.manual_seed(int(seed) * 100_000 + int(model_index) * 1_000 + int(mc))
            out0 = gen_model.rollout(batch, do_t={1: 0})
            torch.manual_seed(int(seed) * 100_000 + int(model_index) * 1_000 + int(mc))
            out1 = gen_model.rollout(batch, do_t={1: 1})
            y0 = out0["Y_prob"][:, 1].detach().to(device).float()
            y1 = out1["Y_prob"][:, 1].detach().to(device).float()
            y0_sum += y0
            y1_sum += y1
        y0_prob = y0_sum / float(mc_samples)
        y1_prob = y1_sum / float(mc_samples)
        y0_total += float(y0_prob.sum().item())
        y1_total += float(y1_prob.sum().item())
        n_total += int(len(chunk))
    mu0 = float(y0_total / max(1, n_total))
    mu1 = float(y1_total / max(1, n_total))
    return {"model_value_t0": mu0, "model_value_t1": mu1, "ate_hat": float(mu1 - mu0)}


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, g in results.groupby("model", sort=False):
        err = g["ate_hat"] - g["true_ate"]
        rows.append(
            {
                "model": model,
                "label": MODEL_LABEL.get(model, model),
                "n_estimates": int(len(g)),
                "ate_abs_err": float(np.mean(np.abs(err))),
                "ate_rmse": float(math.sqrt(np.mean(np.square(err)))),
                "ate_bias": float(np.mean(err)),
                "policy_value_abs_err": float(g["policy_value_abs_err"].mean()) if "policy_value_abs_err" in g.columns else float("nan"),
                "policy_regret": float(g["policy_regret"].mean()) if "policy_regret" in g.columns else float("nan"),
                "fixed_policy_value_abs_err": float(g["fixed_policy_value_abs_err"].mean()) if "fixed_policy_value_abs_err" in g.columns else float("nan"),
                "pehe": float("nan"),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(math.sqrt(np.mean(np.square(err)))),
                "bias": float(np.mean(err)),
                "median_abs_error": float(np.median(np.abs(err))),
                "sign_accuracy": float(np.mean(np.sign(g["ate_hat"]) == np.sign(g["true_ate"]))),
                "pearson": float(g[["ate_hat", "true_ate"]].corr(method="pearson").iloc[0, 1]),
                "spearman": float(g[["ate_hat", "true_ate"]].corr(method="spearman").iloc[0, 1]),
            }
        )
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    return out.sort_values("model").reset_index(drop=True)


def plot_figures(
    out_dir: Path,
    raw: pd.DataFrame,
    tasks: pd.DataFrame,
    selection: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    device_label: str,
) -> None:
    fig_dir = out_dir / "paper_figures"
    setup_style()

    fig = plt.figure(figsize=(13.6, 8.4))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    label_panel(ax_a, "A", "Real randomized-intervention data coverage")
    bars = [
        ("condition\nmetadata", int(raw["has_condition_metadata"].sum()) if len(raw) else 0),
        ("raw experiment\nfolders", int((raw["n_assignment_logs"] > 0).sum()) if len(raw) else 0),
        ("binary RCT\ntasks", int(len(tasks))),
        ("processed\nexperiments", int(tasks["experiment_id"].nunique())),
    ]
    ax_a.bar(np.arange(len(bars)), [x[1] for x in bars], color=["#4e79a7", "#59a14f", "#f28e2b", "#8f8f8f"], alpha=0.92)
    ax_a.set_xticks(np.arange(len(bars)), [x[0] for x in bars])
    ax_a.set_ylabel("Count")
    ax_a.set_ylim(0, max(v for _, v in bars) + 12)
    for i, (_, v) in enumerate(bars):
        ax_a.text(i, v + 1.2, str(v), ha="center", va="bottom")

    label_panel(ax_b, "B", "Outcome families")
    counts = tasks["dependent_measure"].value_counts().sort_values()
    ax_b.barh(np.arange(len(counts)), counts.values, color="#76b7b2", alpha=0.92)
    ax_b.set_yticks(np.arange(len(counts)), counts.index)
    ax_b.set_xlabel("Binary RCT tasks")
    for i, v in enumerate(counts.values):
        ax_b.text(v + 0.4, i, str(int(v)), va="center")

    label_panel(ax_c, "C", "RCT effect landscape")
    outcome_colors = {"completion": "#7f7f7f", "posttest": "#4e79a7", "problem set correctness": "#59a14f", "problems to mastery": "#f28e2b"}
    max_abs = max(float(tasks["true_ate"].abs().max()), 1e-6)
    for outcome, g in tasks.groupby("dependent_measure"):
        ax_c.scatter(
            g["n"],
            g["true_ate"],
            s=35 + 160 * g["true_ate"].abs() / max_abs,
            color=outcome_colors.get(outcome, "#999999"),
            edgecolor="white",
            linewidth=0.8,
            alpha=0.80,
            label=outcome,
        )
    ax_c.axhline(0, color="#333333", lw=0.9, ls="--")
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Task sample size (log)")
    ax_c.set_ylabel("RCT ATE on binary outcome")
    ax_c.grid(True, alpha=0.22)
    ax_c.legend(frameon=False, ncol=2, loc="best")

    label_panel(ax_d, "D", "Biased observational training sample")
    sel = selection.groupby("seed", as_index=False).agg(retention=("retention", "mean"), n_biased=("n_biased", "sum"))
    ax_d.bar(sel["seed"].astype(str), sel["retention"], color="#8f8f8f", alpha=0.90)
    ax_d.set_ylim(0, min(1.0, max(0.25, float(sel["retention"].max()) + 0.18)))
    ax_d.set_xlabel("Seed")
    ax_d.set_ylabel("Mean retention after selection")
    ax_d.grid(axis="y", alpha=0.22)
    for i, row in sel.iterrows():
        ax_d.text(i, row["retention"] + 0.018, f"{row['retention']:.2f}\nn={row['n_biased']:.0f}", ha="center", va="bottom", fontsize=8.6)
    ax_d.text(0.02, 0.04, f"GPU: {device_label}", transform=ax_d.transAxes, fontsize=8.4, color="#555555")
    fig.suptitle("Fig. 1. ASSISTments randomized-intervention benchmark for the six target models", x=0.02, ha="left", fontweight="bold")
    savefig(fig, fig_dir, "Fig1_assistments_rct_scm_benchmark")

    fig = plt.figure(figsize=(13.6, 8.4))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    summ = summary.set_index("model").reindex(MODEL_ORDER)
    label_panel(ax_a, "A", "ATE absolute error")
    x = np.arange(len(summ))
    ax_a.bar(x, summ["mae"], color=[MODEL_COLOR[m] for m in summ.index], alpha=0.92)
    ax_a.set_xticks(x, [MODEL_LABEL[m] for m in summ.index], rotation=25, ha="right")
    ax_a.set_ylabel("ATE abs err")
    ax_a.grid(axis="y", alpha=0.22)
    for xi, val in zip(x, summ["mae"]):
        ax_a.text(xi, val + 0.002, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

    label_panel(ax_b, "B", "ATE RMSE and sign recovery")
    y = np.arange(len(summ))
    norm = mpl.colors.Normalize(vmin=max(0.0, float(summ["sign_accuracy"].min()) - 0.03), vmax=min(1.0, float(summ["sign_accuracy"].max()) + 0.03))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("sign", ["#f7fbff", "#6baed6", "#08306b"])
    ax_b.hlines(y, xmin=0, xmax=summ["rmse"], color="#bdbdbd", lw=3, zorder=1)
    sc = ax_b.scatter(summ["rmse"], y, c=summ["sign_accuracy"], s=220, cmap=cmap, norm=norm, edgecolor="white", linewidth=1.0, zorder=2)
    ax_b.set_yticks(y, [MODEL_LABEL[m] for m in summ.index])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("ATE RMSE")
    ax_b.set_xlim(0, float(summ["rmse"].max()) + 0.02)
    ax_b.grid(axis="x", alpha=0.22)
    for yi, (_, row) in zip(y, summ.iterrows()):
        ax_b.text(row["rmse"] + 0.003, yi, f"sign={row['sign_accuracy']:.2f}", va="center", fontsize=8.3)
    cbar = fig.colorbar(sc, ax=ax_b, fraction=0.046, pad=0.02)
    cbar.set_label("Sign accuracy")

    label_panel(ax_c, "C", "Error distribution over tasks and seeds")
    data = [results.loc[results["model"].eq(m), "abs_error"].to_numpy(float) for m in MODEL_ORDER]
    parts = ax_c.violinplot(data, showmedians=True, widths=0.82)
    for body, model in zip(parts["bodies"], MODEL_ORDER):
        body.set_facecolor(MODEL_COLOR[model])
        body.set_alpha(0.42)
        body.set_edgecolor("none")
    parts["cmedians"].set_color("#111111")
    ax_c.set_xticks(np.arange(1, len(MODEL_ORDER) + 1), [MODEL_LABEL[m] for m in MODEL_ORDER], rotation=25, ha="right")
    ax_c.set_ylabel("Absolute ATE error")
    ax_c.grid(axis="y", alpha=0.22)

    label_panel(ax_d, "D", "Target paper metrics")
    metrics = ["ate_abs_err", "policy_value_abs_err", "policy_regret", "sign_accuracy", "pearson"]
    labels = ["ATE abs\nerr low", "Policy value\nabs err low", "Policy\nregret low", "Sign\nhigh", "Pearson\nhigh"]
    score = summ[metrics].astype(float).copy()
    for col in ["ate_abs_err", "policy_value_abs_err", "policy_regret"]:
        score[col] = 1 - (score[col] - score[col].min()) / max(score[col].max() - score[col].min(), 1e-8)
    for col in ["sign_accuracy", "pearson"]:
        score[col] = (score[col] - score[col].min()) / max(score[col].max() - score[col].min(), 1e-8)
    im = ax_d.imshow(score.values, aspect="auto", cmap=mpl.colors.LinearSegmentedColormap.from_list("score", ["#f7fbff", "#6baed6", "#08306b"]), vmin=0, vmax=1)
    ax_d.set_yticks(np.arange(len(score)), [MODEL_LABEL[m] for m in score.index])
    ax_d.set_xticks(np.arange(len(metrics)), labels)
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            ax_d.text(j, i, f"{score.values[i, j]:.2f}", ha="center", va="center", fontsize=8.4)
    cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.02)
    cbar.set_label("Normalized score")
    fig.suptitle("Fig. 2. Six-model recovery of randomized intervention effects", x=0.02, ha="left", fontweight="bold")
    savefig(fig, fig_dir, "Fig2_assistments_rct_six_model_recovery")

    fig = plt.figure(figsize=(13.6, 10.0))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)
    for idx, model in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        label_panel(ax, "ABCDEF"[idx], MODEL_LABEL[model])
        g = results[results["model"].eq(model)].groupby("task_key", as_index=False).agg(
            true_ate=("true_ate", "mean"),
            ate_hat=("ate_hat", "mean"),
            n=("n", "mean"),
            dependent_measure=("dependent_measure", "first"),
        )
        for outcome, gg in g.groupby("dependent_measure"):
            ax.scatter(gg["true_ate"], gg["ate_hat"], s=np.sqrt(gg["n"]) * 2.4, alpha=0.72, edgecolor="white", linewidth=0.8, label=outcome)
        lim = max(abs(g["true_ate"]).max(), abs(g["ate_hat"]).max()) + 0.04
        ax.plot([-lim, lim], [-lim, lim], color="#333333", lw=1.0, ls="--")
        ax.axhline(0, color="#aaaaaa", lw=0.8)
        ax.axvline(0, color="#aaaaaa", lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("RCT ATE")
        ax.set_ylabel("Estimated ATE")
        ax.grid(True, alpha=0.22)
        if idx == 0:
            ax.legend(frameon=False, fontsize=7.7, loc="best")
    fig.suptitle("Fig. 3. Calibration against real randomized-intervention effects", x=0.02, ha="left", fontweight="bold")
    savefig(fig, fig_dir, "Fig3_assistments_rct_six_model_calibration")

    fig = plt.figure(figsize=(13.6, 8.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    label_panel(ax_a, "A", "Most frequent intervention families")
    top = tasks["category"].value_counts().head(10).index.tolist()
    cat = tasks[tasks["category"].isin(top)]
    counts = cat["category"].value_counts().sort_values()
    ax_a.barh(np.arange(len(counts)), counts.values, color="#4e79a7", alpha=0.88)
    ax_a.set_yticks(np.arange(len(counts)), counts.index)
    ax_a.set_xlabel("Binary RCT tasks")

    label_panel(ax_b, "B", "RCT ATE by intervention family")
    ordered = counts.index.tolist()
    box_data = [cat.loc[cat["category"].eq(c), "true_ate"].to_numpy(float) for c in ordered]
    ax_b.boxplot(box_data, vert=False, tick_labels=ordered, patch_artist=True, medianprops={"color": "#111111"})
    ax_b.axvline(0, color="#333333", lw=0.9, ls="--")
    ax_b.set_xlabel("RCT ATE")
    ax_b.grid(axis="x", alpha=0.22)

    label_panel(ax_c, "C", "Outcome-specific ATE abs err")
    err = results.groupby(["model", "dependent_measure"], as_index=False).agg(ate_abs_err=("ate_abs_err", "mean"))
    pivot = err.pivot(index="model", columns="dependent_measure", values="ate_abs_err").reindex(MODEL_ORDER)
    im = ax_c.imshow(pivot.values, cmap=mpl.colors.LinearSegmentedColormap.from_list("err", ["#f7fbff", "#9ecae1", "#08519c"]), aspect="auto")
    ax_c.set_yticks(np.arange(len(pivot)), [MODEL_LABEL[m] for m in pivot.index])
    ax_c.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax_c.text(j, i, "" if not np.isfinite(val) else f"{val:.3f}", ha="center", va="center", fontsize=8.3)
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.02)
    cbar.set_label("ATE abs err")

    label_panel(ax_d, "D", "Best model by task")
    best = results.groupby(["task_key", "model"], as_index=False).agg(abs_error=("abs_error", "mean"))
    best = best.loc[best.groupby("task_key")["abs_error"].idxmin()]
    bc = best["model"].value_counts().reindex(MODEL_ORDER).fillna(0).astype(int)
    ax_d.bar(np.arange(len(bc)), bc.values, color=[MODEL_COLOR[m] for m in bc.index], alpha=0.90)
    ax_d.set_xticks(np.arange(len(bc)), [MODEL_LABEL[m] for m in bc.index], rotation=25, ha="right")
    ax_d.set_ylabel("Tasks with lowest mean error")
    ax_d.grid(axis="y", alpha=0.22)
    for i, v in enumerate(bc.values):
        ax_d.text(i, v + 0.4, str(int(v)), ha="center", va="bottom", fontsize=8.6)
    fig.suptitle("Fig. 4. Heterogeneity and robustness of the six target models", x=0.02, ha="left", fontweight="bold")
    savefig(fig, fig_dir, "Fig4_assistments_rct_six_model_robustness")


def write_docs(out_dir: Path, tasks: pd.DataFrame, selection: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    fig_dir = out_dir / "paper_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    best_ate = summary.sort_values("ate_abs_err").iloc[0]
    best_policy_value = summary.sort_values("policy_value_abs_err").iloc[0]
    best_policy_regret = summary.sort_values("policy_regret").iloc[0]
    dataset_label = DATASET_LABELS.get(str(args.dataset_name), str(args.dataset_name))
    notes = [
        "# ASSISTments Real-RCT Six-Model Benchmark",
        "",
        f"This run uses {dataset_label} as the real-intervention benchmark.",
        "",
        "## Models",
        "",
        "- scm_causal",
        "- rcgan",
        "- vae",
        "- diffusion",
        "- crn",
        "- timegan",
        "",
        "## Target",
        "",
        "The repository models are binary-outcome sequence models. For the processed ASSISTments RCT outcomes, the benchmark uses a binary target: completion uses assignment completion, and other standardized learning outcomes use `normalized_student_learning > 0`. The RCT ATE is computed on this same binary target.",
        "",
        "## Data",
        "",
        f"- Dataset: {dataset_label}.",
        f"- Analyzable binary RCT tasks: {len(tasks)}.",
        f"- Processed experiments represented: {tasks['experiment_id'].nunique()}.",
        f"- Seeds: {','.join(str(x) for x in args.seeds)}.",
        f"- Mean biased-sample retention: {selection['retention'].mean():.3f}.",
        "",
        "## Current Ranking",
        "",
        f"- Best ATE abs err: {best_ate['model']} ({best_ate['ate_abs_err']:.4f}).",
        f"- Best policy value abs err: {best_policy_value['model']} ({best_policy_value['policy_value_abs_err']:.4f}).",
        f"- Best policy regret: {best_policy_regret['model']} ({best_policy_regret['policy_regret']:.4f}).",
        "",
        "## PEHE",
        "",
        "PEHE is not reported for this pure real-RCT benchmark because individual-level potential outcomes are not both observed. The result table keeps `pehe` as missing and `pehe_supported=0`; PEHE should be reported on synthetic or semi-synthetic datasets with known individual treatment effects.",
        "",
    ]
    (out_dir / "benchmark_notes.md").write_text("\n".join(notes), encoding="utf-8")
    plan = [
        "# Figure Plan",
        "",
        "Each numbered figure is a single multi-panel plate.",
        "",
        "## Fig. 1. Benchmark Construction",
        "",
        "- Data coverage, outcome families, RCT effect landscape, and biased observational training design.",
        "",
        "## Fig. 2. Six-Model ATE Recovery",
        "",
        "- Main quantitative comparison across SCM-Causal, RCGAN, VAE, Diffusion, CRN, and TimeGAN using ATE abs err, policy value abs err, policy regret, sign recovery, and correlation.",
        "",
        "## Fig. 3. Calibration",
        "",
        "- Six small multiples comparing model-estimated ATE to randomized ground-truth ATE.",
        "",
        "## Fig. 4. Robustness",
        "",
        "- Intervention family, outcome-family, and best-model-by-task analysis.",
        "",
    ]
    captions = [
        "# Figure Captions",
        "",
        "Fig. 1. ASSISTments randomized-intervention benchmark for the six target models. Panels summarize the real RCT data coverage, outcome-family composition, randomized binary ATE landscape, and covariate-dependent biased sampling used to train observational sequence models.",
        "",
        "Fig. 2. Six-model recovery of randomized intervention effects. SCM-Causal, RCGAN, VAE, Diffusion, CRN, and TimeGAN are evaluated by ATE absolute error, policy value absolute error, policy regret, sign recovery, and correlation with full-sample randomized effects.",
        "",
        "Fig. 3. Calibration against real randomized-intervention effects. Each point is an analyzable binary RCT task, averaged across seeds. The diagonal indicates exact agreement with randomized ground truth.",
        "",
        "Fig. 4. Heterogeneity and robustness of the six target models. Panels show intervention-family coverage, family-level RCT effect variation, outcome-specific recovery error, and which model wins each task.",
        "",
    ]
    (fig_dir / "figure_plan.md").write_text("\n".join(plan), encoding="utf-8")
    (fig_dir / "figure_captions.md").write_text("\n".join(captions), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the six target AIED models on real ASSISTments randomized interventions.")
    parser.add_argument("--dataset_name", type=str, default="assistments_rct88", choices=sorted(DATASET_LABELS))
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--min_n", type=int, default=100)
    parser.add_argument("--min_arm", type=int, default=30)
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--bias_strength", type=float, default=1.35)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--mc_samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task_id_mode", type=str, default="specific", choices=["specific", "constant"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--scm_epochs", type=int, nargs=3, default=[8, 8, 6])
    parser.add_argument("--gan_epochs", type=int, nargs=3, default=[8, 8, 4])
    parser.add_argument("--timegan_epochs", type=int, nargs=3, default=[2, 2, 4])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--timegan_hidden", type=int, default=32)
    parser.add_argument("--timegan_z_dim", type=int, default=16)
    parser.add_argument("--timegan_max_batches", type=int, default=200)
    parser.add_argument("--scm_dynamics", type=str, default="transformer", choices=["gru", "mlp", "transformer"])
    parser.add_argument("--scm_d_k", type=int, default=32)
    parser.add_argument("--scm_k_c_ratio", type=float, default=0.5)
    parser.add_argument("--scm_y_head_hidden", type=int, default=0)
    parser.add_argument("--scm_y_head_layers", type=int, default=1)
    parser.add_argument("--scm_w_do", type=float, default=0.05)
    parser.add_argument("--scm_w_cf", type=float, default=0.02)
    parser.add_argument("--scm_w_advT_causal", type=float, default=0.02)
    parser.add_argument("--scm_w_t_pred_spurious", type=float, default=0.1)
    parser.add_argument("--diff_steps", type=int, default=30)
    parser.add_argument("--diff_backbone", type=str, default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--a_emb_dim", type=int, default=32)
    parser.add_argument("--t_emb_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--causal_every", type=int, default=10)
    parser.add_argument("--cf_every", type=int, default=10)
    parser.add_argument("--do_min_arm_samples", type=int, default=16)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--skip_plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dataset_name == "assistments_las2016" and args.data_dir == DATA_DIR:
        args.data_dir = LAS_DATA_DIR
    if args.dataset_name == "assistments_abtest_study2" and args.data_dir == DATA_DIR:
        args.data_dir = ABTEST_DATA_DIR
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    df, raw = load_benchmark_data(str(args.dataset_name), args.data_dir)
    tasks, all_rows = build_tasks(df, min_n=args.min_n, min_arm=args.min_arm, max_tasks=args.max_tasks)
    all_rows = all_rows.reset_index(drop=True)
    X_all, feature_cols, feature_transform = prepare_features(all_rows)
    meta = {
        "d_x": int(X_all.shape[1]),
        "seq_len": 2,
        "a_is_discrete": True,
        "t_is_discrete": True,
        "a_vocab_size": 2 if args.task_id_mode == "constant" else int(tasks["task_id"].max()) + 2,
        "t_vocab_size": 2,
        "d_a": 1,
        "d_t": 1,
        "d_y": 1,
    }
    tasks.to_csv(args.out_dir / "tables" / "task_summary.csv", index=False)
    raw.to_csv(args.out_dir / "tables" / "raw_inventory.csv", index=False)
    (args.out_dir / "tables" / "feature_columns.json").write_text(
        json.dumps({"feature_columns": feature_cols, "transform": feature_transform}, indent=2),
        encoding="utf-8",
    )
    if str(args.dataset_name) == "assistments_las2016":
        post_treatment_dropped = LAS_EXCLUDED_POST_TREATMENT_COLUMNS
    elif str(args.dataset_name) == "assistments_abtest_study2":
        post_treatment_dropped = STUDY2_EXCLUDED_POST_TREATMENT_COLUMNS
    else:
        post_treatment_dropped = ["assignment_completed", "normalized_student_learning"]
    leakage_audit = {
        "dataset_name": str(args.dataset_name),
        "feature_columns": feature_cols,
        "excluded_columns": sorted(ID_COLS | {"task_key", "task_id", "y_binary", "y0_prior"}),
        "post_treatment_columns_explicitly_dropped": post_treatment_dropped,
        "uses_true_ate_in_training": False,
        "uses_arm_means_in_training": False,
        "uses_condition_as_feature": False,
    }
    (args.out_dir / "tables" / "leakage_audit.json").write_text(json.dumps(leakage_audit, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "dataset_name": str(args.dataset_name),
                "models": args.models,
                "seeds": args.seeds,
                "n_tasks": int(len(tasks)),
                "n_rows_task_expanded": int(len(all_rows)),
                "d_x": int(X_all.shape[1]),
            },
            indent=2,
        )
    )

    all_result_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    result_path = args.out_dir / "tables" / "ate_recovery_by_seed.csv"
    if args.resume and result_path.exists():
        old = pd.read_csv(result_path)
        missing_cols = sorted(REQUIRED_RESULT_COLUMNS - set(old.columns))
        if missing_cols:
            print(f"[resume] existing result table is missing new metric columns; re-evaluating from checkpoints: {missing_cols}")
            done = set()
        else:
            all_result_frames.append(old)
            done = set(zip(old["seed"].astype(int), old["model"].astype(str)))
    else:
        done = set()

    for seed in args.seeds:
        train_rows, sel = select_biased_training_rows(
            all_rows,
            X_all,
            seed=int(seed),
            bias_strength=float(args.bias_strength),
            min_arm=int(args.min_arm),
        )
        selection_frames.append(sel)
        train_batch = make_sequence_batch(train_rows, X_all, task_id_mode=str(args.task_id_mode), device=None)
        for model_index, model in enumerate(args.models):
            if (int(seed), str(model)) in done:
                print(f"[skip] seed={seed} model={model} already in result table")
                continue
            run_dir = args.out_dir / "checkpoints" / f"seed{int(seed)}" / model
            gen_model = train_or_load_model(
                model,
                train_batch=train_batch,
                meta=meta,
                seed=int(seed),
                device=device,
                run_dir=run_dir,
                args=args,
            )
            rows: list[dict[str, Any]] = []
            for _, task in tasks.iterrows():
                task_rows = all_rows[all_rows["task_key"].eq(task["task_key"])]
                values = estimate_task_values(
                    gen_model,
                    task_rows,
                    X_all,
                    seed=int(seed),
                    model_index=int(model_index),
                    mc_samples=int(args.mc_samples),
                    eval_batch_size=int(args.eval_batch_size),
                    device=device,
                    task_id_mode=str(args.task_id_mode),
                )
                ate_hat = float(values["ate_hat"])
                err = float(ate_hat - task["true_ate"])
                arm0_err = abs(float(values["model_value_t0"]) - float(task["true_value_t0"]))
                arm1_err = abs(float(values["model_value_t1"]) - float(task["true_value_t1"]))
                policy_selected_arm = int(float(values["model_value_t1"]) >= float(values["model_value_t0"]))
                oracle_best_arm = int(float(task["true_value_t1"]) >= float(task["true_value_t0"]))
                policy_value_hat = float(values["model_value_t1"] if policy_selected_arm == 1 else values["model_value_t0"])
                policy_value_true = float(task["true_value_t1"] if policy_selected_arm == 1 else task["true_value_t0"])
                oracle_policy_value = float(max(float(task["true_value_t0"]), float(task["true_value_t1"])))
                rows.append(
                    {
                        "seed": int(seed),
                        "eval_version": int(EVAL_VERSION),
                        "model": model,
                        "label": MODEL_LABEL[model],
                        "task_key": task["task_key"],
                        "experiment_id": task["experiment_id"],
                        "independent_measure_pair": task["independent_measure_pair"],
                        "dependent_measure": task["dependent_measure"],
                        "category": task["category"],
                        "n": int(task["n"]),
                        "n0": int(task["n0"]),
                        "n1": int(task["n1"]),
                        "true_value_t0": float(task["true_value_t0"]),
                        "true_value_t1": float(task["true_value_t1"]),
                        "model_value_t0": float(values["model_value_t0"]),
                        "model_value_t1": float(values["model_value_t1"]),
                        "true_ate": float(task["true_ate"]),
                        "true_ate_normalized": float(task["true_ate_normalized"]),
                        "ate_hat": float(ate_hat),
                        "ate_error": err,
                        "ate_abs_err": abs(err),
                        "ate_squared_err": err * err,
                        "error": err,
                        "abs_error": abs(err),
                        "squared_error": err * err,
                        "fixed_policy_value_abs_err": float(0.5 * (arm0_err + arm1_err)),
                        "policy_selected_arm": policy_selected_arm,
                        "oracle_best_arm": oracle_best_arm,
                        "policy_value_hat": policy_value_hat,
                        "policy_value_true": policy_value_true,
                        "oracle_policy_value": oracle_policy_value,
                        "policy_value_abs_err": abs(policy_value_hat - policy_value_true),
                        "policy_regret": max(0.0, oracle_policy_value - policy_value_true),
                        "pehe": np.nan,
                        "pehe_supported": 0,
                    }
                )
            frame = pd.DataFrame(rows)
            all_result_frames.append(frame)
            pd.concat(all_result_frames, ignore_index=True).to_csv(result_path, index=False)
            print(f"[eval] seed={seed} model={model} rows={len(frame)}")

    selection = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    if args.resume and (args.out_dir / "tables" / "selection_by_seed.csv").exists():
        old_sel = pd.read_csv(args.out_dir / "tables" / "selection_by_seed.csv")
        selection = pd.concat([old_sel, selection], ignore_index=True).drop_duplicates(["seed", "task_key"], keep="last")
    selection.to_csv(args.out_dir / "tables" / "selection_by_seed.csv", index=False)
    results = pd.concat(all_result_frames, ignore_index=True)
    results = results.drop_duplicates(["seed", "model", "task_key"], keep="last")
    results["model"] = pd.Categorical(results["model"], MODEL_ORDER, ordered=True)
    results = results.sort_values(["seed", "model", "task_key"]).reset_index(drop=True)
    results.to_csv(result_path, index=False)
    summary = summarize(results)
    summary.to_csv(args.out_dir / "tables" / "method_summary.csv", index=False)
    by_outcome = (
        results.groupby(["model", "dependent_measure"], observed=False)
        .agg(mae=("abs_error", "mean"), rmse=("squared_error", lambda s: math.sqrt(float(np.mean(s)))))
        .reset_index()
    )
    by_outcome.to_csv(args.out_dir / "tables" / "method_by_outcome.csv", index=False)
    device_label = torch.cuda.get_device_name(0) if device.type == "cuda" and torch.cuda.is_available() else "CPU"
    if not args.skip_plots:
        plot_figures(args.out_dir, raw, tasks, selection, results, summary, device_label=device_label)
    write_docs(args.out_dir, tasks, selection, summary, args)
    print(f"[done] wrote six-model RCT benchmark to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
