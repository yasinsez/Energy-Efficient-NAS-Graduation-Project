#!/usr/bin/env python3
"""
train_latency_predictor.py
==========================
Evaluation of LLM latency prediction and analytical roofline modeling:
1. Feature Collision & Workload Indicator:
   - Launch configurations alone (batch_size, output_tokens, parallelism) suffer from feature collision
     between Alpaca (~54 ms) and Longbench (~452 ms) under identical hardware configurations.
   - Providing a categorical task indicator (Alpaca vs. Longbench) resolves this collision.
2. Physics-Informed Roofline Modeling:
   - Analytical Roofline latencies (via LLM-Viewer) provide a physical inductive bias,
     significantly improving linear modeling (R² = 0.2791 -> 0.5024).
3. Out-of-Distribution Generalization:
   - Evaluates generalization across unseen parallelism configurations using GroupKFold.

Note on Data Aggregation:
   Repeated measurement trials (10 replicates per setting) are aggregated to their mean
   prior to model training to prevent duplicate trial leakage between cross-validation folds.
"""

import os
import tempfile

# Prevent matplotlib directory permission warning
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GroupKFold, cross_validate, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_dataset(csv_path="reproduced_dataset.csv", model_name="Qwen3-14B"):
    """
    Loads benchmark dataset and aggregates repeated trials to unique configuration means,
    ensuring no duplicate trial leakage across cross-validation splits.
    """
    if not os.path.isabs(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Please run create_dataset.py first.")

    df_raw = pd.read_csv(csv_path)
    if model_name:
        df_raw = df_raw[df_raw["model_name"] == model_name].copy()

    # Aggregate 10 replicate runs per unique configuration into mean values
    group_cols = [
        "task",
        "model_name",
        "config",
        "tp_degree",
        "pp_degree",
        "dp_degree",
        "num_gpus",
        "workload",
        "batch_size",
        "output_tokens",
        "roofline_analytical_latency",
        "roofline_parallel_aware",
    ]
    df = df_raw.groupby(group_cols, as_index=False).agg(
        duration=("duration", "mean"),
        duration_std=("duration", "std"),
        throughput=("throughput", "mean"),
        gpu_avg_power=("gpu_avg_power", "mean"),
    )
    df["config_workload"] = df["task"] + "_" + df["config"] + "_" + df["workload"]
    return df


def build_pipeline(model, cat_cols, num_cols):
    transformers = []
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols)
        )
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("prep", preprocessor), ("reg", model)])


def evaluate_feature_set(df, feature_cols, target_col="duration", cv=None, groups=None):
    if cv is None:
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]
    X, y = df[feature_cols], df[target_col]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = {}
    predictions = {}
    for name, raw_model in models.items():
        pipe = build_pipeline(raw_model, cat_cols, num_cols)
        cv_scores = cross_validate(pipe, X, y, cv=cv, groups=groups, scoring=["r2", "neg_mean_squared_error"])
        preds = cross_val_predict(pipe, X, y, cv=cv, groups=groups)
        results[name] = {
            "R2": float(np.mean(cv_scores["test_r2"])),
            "MSE": float(-np.mean(cv_scores["test_neg_mean_squared_error"])),
        }
        predictions[name] = preds

    return results, predictions


def plot_takeaway(r2_ablation, r2_generalization, output_path=None):
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "figures", "roofline_task_omission_impact.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=300)

    # ── Panel A: In-Distribution Feature Ablation (3-Fold CV) ────────────────
    ax1 = axes[0]
    labels = ["Linear Regression", "Random Forest", "Gradient Boosting"]
    x = np.arange(len(labels))
    width = 0.26

    r2_base = [max(r2_ablation["Base"][m]["R2"], 0) for m in labels]
    r2_task = [max(r2_ablation["Task"][m]["R2"], 0) for m in labels]
    r2_rlm  = [max(r2_ablation["Roofline"][m]["R2"], 0) for m in labels]

    ax1.bar(x - width, r2_base, width, label="Parallelism Only (Feature Collision)", color="#d9534f", edgecolor="black")
    ax1.bar(x,         r2_task, width, label="+ Categorical Task (Alpaca/Longbench)", color="#337ab7", edgecolor="black", hatch="//")
    ax1.bar(x + width, r2_rlm,  width, label="+ Task + Analytical Roofline (RLM)", color="#5cb85c", edgecolor="black", hatch="\\\\")

    ax1.set_ylabel("Cross-Validated R² Score", fontsize=11, fontweight="bold")
    ax1.set_title("A. In-Distribution Feature Ablation (3-Fold CV)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(fontsize=8.0, loc="upper left")

    for i in range(len(labels)):
        ax1.annotate(f"{r2_task[i]:.2f}", xy=(x[i], r2_task[i] + 0.02), ha="center", fontsize=8.5, fontweight="bold", color="#1d4b75")
        ax1.annotate(f"{r2_rlm[i]:.2f}", xy=(x[i] + width, r2_rlm[i] + 0.02), ha="center", fontsize=8.5, fontweight="bold", color="#2e662e")

    # ── Panel B: Out-of-Distribution Generalization (GroupKFold) ────────────
    ax2 = axes[1]
    schemes = ["In-Distribution (Random 3-Fold)", "GroupKFold: Parallel Config (Unseen Degrees)"]
    x2 = np.arange(len(schemes))
    width2 = 0.26

    lr_scores = [max(r2_generalization[s]["Linear Regression"]["R2"], 0) for s in ["in_dist", "group_config"]]
    rf_scores = [max(r2_generalization[s]["Random Forest"]["R2"], 0) for s in ["in_dist", "group_config"]]
    gb_scores = [max(r2_generalization[s]["Gradient Boosting"]["R2"], 0) for s in ["in_dist", "group_config"]]

    ax2.bar(x2 - width2, lr_scores, width2, label="Linear Regression", color="#f0ad4e", edgecolor="black")
    ax2.bar(x2,          rf_scores, width2, label="Random Forest", color="#5cb85c", edgecolor="black", hatch="//")
    ax2.bar(x2 + width2, gb_scores, width2, label="Gradient Boosting", color="#5bc0de", edgecolor="black", hatch="\\\\")

    ax2.set_ylabel("Cross-Validated R² Score", fontsize=11, fontweight="bold")
    ax2.set_title("B. Out-of-Distribution Generalization (+Roofline)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(schemes, fontsize=9.5)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend(fontsize=9, loc="upper right")

    for i in range(len(schemes)):
        ax2.annotate(f"R²={rf_scores[i]:.2f}", xy=(x2[i], rf_scores[i] + 0.02), ha="center", fontsize=8.5, fontweight="bold", color="#2e662e")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved updated takeaway figure: {output_path}")


def main():
    print("=" * 88)
    print("  PHYSICS-INFORMED ROOFLINE VS. OPERATIONAL MODELING (AGGREGATED BENCHMARKS)")
    print("=" * 88)

    df = load_dataset()
    print(f"Loaded {len(df)} unique experimental configurations for Qwen3-14B (replicate runs aggregated to mean).")

    cols_base = ["batch_size", "output_tokens", "tp_degree", "pp_degree", "dp_degree", "num_gpus"]
    cols_task = cols_base + ["task"]
    cols_rlm = cols_base + ["task", "roofline_analytical_latency"]

    # 1. Feature Ablation (3-Fold CV on clean configuration dataset)
    cv_kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    res_base, _ = evaluate_feature_set(df, cols_base, cv=cv_kfold)
    res_task, _ = evaluate_feature_set(df, cols_task, cv=cv_kfold)
    res_rlm, _ = evaluate_feature_set(df, cols_rlm, cv=cv_kfold)

    r2_ablation = {
        "Base": res_base,
        "Task": res_task,
        "Roofline": res_rlm,
    }

    print("\n" + "─" * 88)
    print("1. FEATURE ABLATION (RANDOM 3-FOLD CV)")
    print("─" * 88)
    print(f"{'Method':<20} │ {'Base (Collision)':^19} │ {'+ Categorical Task':^21} │ {'+ Task + Roofline (RLM)':^25}")
    print(f"{'':<20} │ {'R²':>7}   {'MSE':>8} │ {'R²':>7}    {'MSE':>8} │ {'R²':>7}   {'MSE':>8}")
    print("─" * 88)
    for m in res_base:
        print(f"{m:<20} │ {res_base[m]['R2']:>7.4f}  {res_base[m]['MSE']:>8.0f} │ "
              f"{res_task[m]['R2']:>7.4f}   {res_task[m]['MSE']:>8.0f} │ "
              f"{res_rlm[m]['R2']:>7.4f}  {res_rlm[m]['MSE']:>8.0f}")
    print("─" * 88)

    # 2. Out-of-Distribution Generalization across Parallelism Configurations
    cv_config = GroupKFold(n_splits=4)
    res_gen_config_rlm, _ = evaluate_feature_set(df, cols_rlm, cv=cv_config, groups=df["config"])

    r2_generalization = {
        "in_dist": res_rlm,
        "group_config": res_gen_config_rlm,
    }

    print("\n" + "─" * 88)
    print("2. GENERALIZATION: IN-DISTRIBUTION VS. UNSEEN PARALLELISM CONFIGURATION (+ROOFLINE)")
    print("─" * 88)
    print(f"{'Evaluation Split':<40} │ {'Linear Reg':^14} │ {'Random Forest':^15} │ {'Gradient Boost':^15}")
    print(f"{'':<40} │ {'R²':>6}   {'MSE':>6} │ {'R²':>6}    {'MSE':>6} │ {'R²':>6}    {'MSE':>6}")
    print("─" * 88)
    print(f"{'In-Distribution (Random 3-Fold)':<40} │ {res_rlm['Linear Regression']['R2']:>6.4f} {res_rlm['Linear Regression']['MSE']:>6.0f} │ {res_rlm['Random Forest']['R2']:>6.4f}  {res_rlm['Random Forest']['MSE']:>6.0f} │ {res_rlm['Gradient Boosting']['R2']:>6.4f}  {res_rlm['Gradient Boosting']['MSE']:>6.0f}")
    print(f"{'GroupKFold: Unseen Parallelism Config':<40} │ {res_gen_config_rlm['Linear Regression']['R2']:>6.4f} {res_gen_config_rlm['Linear Regression']['MSE']:>6.0f} │ {res_gen_config_rlm['Random Forest']['R2']:>6.4f}  {res_gen_config_rlm['Random Forest']['MSE']:>6.0f} │ {res_gen_config_rlm['Gradient Boosting']['R2']:>6.4f}  {res_gen_config_rlm['Gradient Boosting']['MSE']:>6.0f}")
    print("=" * 88)

    plot_takeaway(r2_ablation, r2_generalization)


if __name__ == "__main__":
    main()
