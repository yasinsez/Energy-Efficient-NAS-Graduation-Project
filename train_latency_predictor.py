#!/usr/bin/env python3
"""
train_latency_predictor.py
==========================
Evaluates the core thesis finding:
In real-world LLM deployments, cluster parallelism configuration (TP, PP, DP, Num GPUs)
is known prior to deployment, but request prompt length / task context is dynamic and private.

- Naive ML models with cluster parallelism still fail (R² ≈ 0.25 - 0.30) because they cannot
  distinguish short prompts (~512 tokens) from long contexts (~2048 tokens).
- Pairing known cluster parallelism with the analytical Roofline model acts as a physics proxy,
  skyrocketing Random Forest R² from 0.255 to 0.9997 (99.96% MSE reduction) and Linear Regression
  R² from 0.300 to 0.884 (83.1% MSE reduction).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_dataset(csv_path="reproduced_dataset.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Please run create_dataset.py first.")
    df = pd.read_csv(csv_path)
    return df[df["model_name"] == "Qwen3-14B"].copy()


def build_pipeline(model, cat_cols, num_cols):
    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("prep", preprocessor), ("reg", model)])


def evaluate_feature_set(df, feature_cols, target_col="duration"):
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]
    X, y = df[feature_cols], df[target_col]
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = {}
    predictions = {}
    for name, raw_model in models.items():
        pipe = build_pipeline(raw_model, cat_cols, num_cols)
        cv_scores = cross_validate(pipe, X, y, cv=cv, scoring=["r2", "neg_mean_squared_error"])
        preds = cross_val_predict(pipe, X, y, cv=cv)
        results[name] = {
            "R2": np.mean(cv_scores["test_r2"]),
            "MSE": -np.mean(cv_scores["test_neg_mean_squared_error"]),
        }
        predictions[name] = preds

    return results, predictions


def plot_takeaway(df, pred_naive, pred_rlm, r2_naive, r2_rlm, output_path="figures/roofline_task_omission_impact.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # ── Panel A: R² Score across Pre-Execution Regimes ───────────────────────
    ax = axes[0]
    labels = ["Linear Regression", "Random Forest", "Gradient Boosting"]
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, r2_naive, width, label="Known Cluster Parallelism (No Task)",
                   color="#d9534f", edgecolor="black")
    bars2 = ax.bar(x + width/2, r2_rlm, width, label="Known Parallelism + Roofline (No Task)",
                   color="#5cb85c", edgecolor="black", hatch="//")

    ax.set_ylabel("Cross-Validated R² Score", fontsize=11, fontweight="bold")
    ax.set_title("A. Predictive Power: Parallelism Known, Task Unknown", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    for i in range(len(labels)):
        delta = (r2_rlm[i] - r2_naive[i]) * 100
        ax.annotate(f"+{delta:.1f}%\n(R²={r2_rlm[i]:.3f})", xy=(x[i] + width/2, r2_rlm[i] + 0.02),
                    ha="center", fontsize=8.5, fontweight="bold", color="#2e662e")

    # ── Panel B: Predicted vs Actual Latency (Random Forest) ─────────────────
    ax2 = axes[1]
    y_actual = df["duration"].values
    y_naive = pred_naive["Random Forest"]
    y_rlm = pred_rlm["Random Forest"]

    ax2.scatter(y_actual, y_naive, color="#d9534f", alpha=0.55, edgecolors="none", s=35,
                label=f"Naive RF with Parallelism (R² = {r2_naive[1]:.2f})")
    ax2.scatter(y_actual, y_rlm, color="#5cb85c", alpha=0.8, edgecolors="black", linewidth=0.5, s=40,
                label=f"Roofline RF with Parallelism (R² = {r2_rlm[1]:.4f})")

    lims = [0, max(y_actual) * 1.05]
    ax2.plot(lims, lims, "k--", alpha=0.6, label="Ideal (y = x)")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel("Actual Latency (s)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Predicted Latency (s)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Ground Truth vs. Predicted (Pre-Execution Inputs Only)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved updated key takeaway figure: {output_path}")


def main():
    print("=" * 84)
    print("  PRE-EXECUTION DEPLOYMENT EVALUATION: KNOWN PARALLELISM VS. UNKNOWN PROMPT CONTEXT")
    print("=" * 84)

    df = load_dataset()
    print(f"Loaded {len(df)} runs for Qwen3-14B across Alpaca (~512 in-tokens) and Longbench (~2048 in-tokens).")

    # 1. Real-World Deployment: Infrastructure Parallelism Known, Task/Prompt Unknown
    cols_parallel = ["batch_size", "output_tokens", "tp_degree", "pp_degree", "dp_degree", "num_gpus"]
    cols_parallel_rlm = cols_parallel + ["roofline_analytical_latency"]

    res_par_id, preds_par_id = evaluate_feature_set(df, cols_parallel)
    res_par_rlm, preds_par_rlm = evaluate_feature_set(df, cols_parallel_rlm)

    # 2. Minimal Ops (Single-GPU serving without parallelism): Task Unknown
    cols_min = ["batch_size", "output_tokens"]
    cols_min_rlm = cols_min + ["roofline_analytical_latency"]
    res_min_id, _ = evaluate_feature_set(df, cols_min)
    res_min_rlm, _ = evaluate_feature_set(df, cols_min_rlm)

    # 3. Benchmark Oracle (Synthetic: Task Labels Provided)
    cols_oracle = ["task", "batch_size", "output_tokens", "tp_degree", "pp_degree", "dp_degree", "num_gpus"]
    res_oracle_id, _ = evaluate_feature_set(df, cols_oracle)

    # Print clean summary tables
    print("\n" + "─" * 84)
    print("REALISTIC DEPLOYMENT: CLUSTER PARALLELISM KNOWN (TP, PP, DP, GPUS), TASK UNKNOWN")
    print("─" * 84)
    print(f"{'Method':<20} │ {'Parallelism Only':^22} │ {'Parallelism + Roofline':^24} │ {'R² Δ':^8}")
    print(f"{'':<20} │ {'R²':>8}   {'MSE':>10} │ {'R²':>9}   {'MSE':>10} │")
    print("─" * 84)
    for m in res_par_id:
        r2_b = res_par_id[m]["R2"]
        mse_b = res_par_id[m]["MSE"]
        r2_r = res_par_rlm[m]["R2"]
        mse_r = res_par_rlm[m]["MSE"]
        delta = (r2_r - r2_b) * 100
        print(f"{m:<20} │ {r2_b:>8.4f}   {mse_b:>10.1f} │ {r2_r:>9.4f}   {mse_r:>10.1f} │ {delta:>+7.1f}%")

    print("\n" + "─" * 84)
    print("SYNTHETIC BENCHMARK: TASK LABELS PROVIDED (MEMORIZATION EFFECT)")
    print("─" * 84)
    print(f"{'Method':<20} │ {'Oracle ID R²':>12}   {'MSE':>10} │ Notes")
    print("─" * 84)
    for m in res_oracle_id:
        print(f"{m:<20} │ {res_oracle_id[m]['R2']:>12.4f}   {res_oracle_id[m]['MSE']:>10.1f} │ "
              f"{'Memorizes discrete benchmark grid' if 'Forest' in m or 'Boost' in m else 'Linear model cannot fully memorize'}")
    print("=" * 84)

    r2_naive = [res_par_id[m]["R2"] for m in ["Linear Regression", "Random Forest", "Gradient Boosting"]]
    r2_rlm   = [res_par_rlm[m]["R2"] for m in ["Linear Regression", "Random Forest", "Gradient Boosting"]]
    plot_takeaway(df, preds_par_id, preds_par_rlm, r2_naive, r2_rlm)


if __name__ == "__main__":
    main()
