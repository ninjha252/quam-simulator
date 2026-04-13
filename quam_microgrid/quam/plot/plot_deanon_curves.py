#!/usr/bin/env python3
"""
plot_deanon_curves.py - Plot deanonymization metrics (QAN)

Generates:
1) Time-series of top1 accuracy, entropy, and top1 probability
2) Aggregate bar charts per scenario

Usage:
  python3 quam/plot_deanon_curves.py --input outputs/<tag>/trial_1/deanon --output outputs/<tag>/trial_1/plots_deanon
"""

import argparse
import glob
import os
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


TOPOLOGIES = ["ring", "star", "mesh", "two_cluster_bridge"]


def _split_scenario_topology(scenario: str) -> Tuple[str, str]:
    for topo in TOPOLOGIES:
        suffix = f"_{topo}"
        if scenario.endswith(suffix):
            return scenario[: -len(suffix)], topo
    return scenario, ""


def load_deanon(input_dir: str, scenarios: Optional[List[str]] = None) -> pd.DataFrame:
    files = glob.glob(os.path.join(input_dir, "deanon_*.csv"))
    if not files:
        raise FileNotFoundError(f"No deanon CSVs found in {input_dir}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        base = os.path.basename(f)
        parts = base.replace("deanon_", "").replace(".csv", "").rsplit("_seed", 1)
        scenario_full = parts[0] if parts else "unknown"
        scenario, topo = _split_scenario_topology(scenario_full)
        df["scenario_full"] = scenario_full
        df["scenario"] = scenario
        df["topology"] = topo
        df["seed"] = int(parts[1].split("_")[0]) if len(parts) > 1 else 0
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    if "abstained" not in out.columns:
        out["abstained"] = 0
    if "top1_margin" not in out.columns:
        out["top1_margin"] = np.nan
    if "top2_prob" not in out.columns:
        out["top2_prob"] = np.nan
    if "n_obs_window" not in out.columns:
        out["n_obs_window"] = 0
    if "prior_blend_weight" not in out.columns:
        out["prior_blend_weight"] = np.nan
    if scenarios:
        out = out[out["scenario"].isin(scenarios)]
    return out


def plot_deanon_timeseries(df: pd.DataFrame, out_path: str, bucket_s: int = 3600):
    df = df.copy()
    df["top1_correct"] = (df["true_sender"] == df["inferred_sender"]).astype(int)
    top1_cand = df["top1_candidate"] if "top1_candidate" in df.columns else df["inferred_sender"]
    df["top1_candidate_correct"] = (df["true_sender"] == top1_cand).astype(float)
    df["abstained"] = df["abstained"].fillna(0).astype(int)
    df["top1_acc_non_abstain"] = np.where(df["abstained"] == 0, df["top1_candidate_correct"], np.nan)
    df["bucket_s"] = (df["t_event_s"] // bucket_s) * bucket_s

    agg = df.groupby(["scenario", "bucket_s"]).agg(
        top1_acc_all=("top1_correct", "mean"),
        top1_acc_non_abstain=("top1_acc_non_abstain", "mean"),
        entropy=("entropy_bits", "mean"),
        top1_prob=("top1_prob", "mean"),
        abstain_rate=("abstained", "mean"),
        n=("top1_correct", "count"),
    ).reset_index()

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    scenarios = sorted(agg["scenario"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))

    for i, scen in enumerate(scenarios):
        sub = agg[agg["scenario"] == scen]
        t_hours = sub["bucket_s"] / 3600.0
        axes[0].plot(t_hours, sub["top1_acc_non_abstain"], label=scen, color=colors[i])
        axes[1].plot(t_hours, sub["entropy"], label=scen, color=colors[i])
        axes[2].plot(t_hours, sub["top1_prob"], label=scen, color=colors[i])
        axes[3].plot(t_hours, sub["abstain_rate"], label=scen, color=colors[i])

    axes[0].set_ylabel("Top‑1 Accuracy")
    axes[0].set_title("Deanonymization Accuracy Over Time (Non‑Abstain)")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Entropy (bits)")
    axes[1].set_title("Posterior Entropy Over Time")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Top‑1 Probability")
    axes[2].set_title("Top‑1 Confidence Over Time")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_ylabel("Abstain Rate")
    axes[3].set_title("Abstain/Unknown Rate Over Time")
    axes[3].set_xlabel("Time (hours)")
    axes[3].grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _scenario_calibration(sub: pd.DataFrame, n_bins: int = 10) -> Tuple[float, float]:
    if "abstained" in sub.columns:
        sub = sub[sub["abstained"].fillna(0).astype(int) == 0]
    if sub.empty:
        return float("nan"), float("nan")
    if "top1_candidate" in sub.columns:
        y = (sub["true_sender"] == sub["top1_candidate"]).astype(float).to_numpy()
    else:
        y = (sub["true_sender"] == sub["inferred_sender"]).astype(float).to_numpy()
    probs = np.clip(sub["top1_prob"].to_numpy(dtype=float), 0.0, 1.0)
    if probs.size == 0:
        return float("nan"), float("nan")
    brier = float(np.mean((probs - y) ** 2))
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i < n_bins - 1:
            m = (probs >= lo) & (probs < hi)
        else:
            m = (probs >= lo) & (probs <= hi)
        if not np.any(m):
            continue
        acc = float(np.mean(y[m]))
        conf = float(np.mean(probs[m]))
        ece += abs(acc - conf) * (np.sum(m) / probs.size)
    return float(ece), brier


def plot_deanon_summary(df: pd.DataFrame, out_path: str):
    df = df.copy()
    df["top1_correct"] = (df["true_sender"] == df["inferred_sender"]).astype(int)
    df["abstained"] = df["abstained"].fillna(0).astype(int)
    top1_cand = df["top1_candidate"] if "top1_candidate" in df.columns else df["inferred_sender"]
    df["top1_candidate_correct"] = (df["true_sender"] == top1_cand).astype(float)
    df["top1_acc_non_abstain"] = np.where(df["abstained"] == 0, df["top1_candidate_correct"], np.nan)
    agg = df.groupby("scenario").agg(
        top1_acc=("top1_correct", "mean"),
        top1_acc_non_abstain=("top1_acc_non_abstain", "mean"),
        entropy=("entropy_bits", "mean"),
        top1_prob=("top1_prob", "mean"),
        abstain_rate=("abstained", "mean"),
    ).reset_index()
    cal_rows = []
    for scen, sub in df.groupby("scenario"):
        ece, brier = _scenario_calibration(sub)
        cal_rows.append({"scenario": scen, "ece": ece, "brier": brier})
    cal = pd.DataFrame(cal_rows)
    agg = agg.merge(cal, on="scenario", how="left")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes.ravel()
    x = np.arange(len(agg))
    labels = agg["scenario"].tolist()

    ax[0].bar(x, agg["top1_acc_non_abstain"], color="steelblue")
    ax[0].set_title("Top‑1 Accuracy (Non‑Abstain)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=45, ha="right")
    ax[0].grid(True, axis="y", alpha=0.3)

    ax[1].bar(x, agg["entropy"], color="orange")
    ax[1].set_title("Entropy (bits)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, rotation=45, ha="right")
    ax[1].grid(True, axis="y", alpha=0.3)

    ax[2].bar(x, agg["top1_prob"], color="green")
    ax[2].set_title("Top‑1 Probability")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(labels, rotation=45, ha="right")
    ax[2].grid(True, axis="y", alpha=0.3)

    ax[3].bar(x, agg["abstain_rate"], color="mediumpurple")
    ax[3].set_title("Abstain Rate")
    ax[3].set_xticks(x)
    ax[3].set_xticklabels(labels, rotation=45, ha="right")
    ax[3].grid(True, axis="y", alpha=0.3)

    ax[4].bar(x, agg["ece"], color="indianred")
    ax[4].set_title("ECE (Top‑1)")
    ax[4].set_xticks(x)
    ax[4].set_xticklabels(labels, rotation=45, ha="right")
    ax[4].grid(True, axis="y", alpha=0.3)

    ax[5].bar(x, agg["brier"], color="darkcyan")
    ax[5].set_title("Brier (Top‑1)")
    ax[5].set_xticks(x)
    ax[5].set_xticklabels(labels, rotation=45, ha="right")
    ax[5].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_deanon_reliability(df: pd.DataFrame, out_path: str, n_bins: int = 10):
    df = df.copy()
    if df.empty:
        return
    if "abstained" in df.columns:
        df = df[df["abstained"].fillna(0).astype(int) == 0].copy()
    if df.empty:
        return
    if "top1_candidate" in df.columns:
        df["top1_correct"] = (df["true_sender"] == df["top1_candidate"]).astype(float)
    else:
        df["top1_correct"] = (df["true_sender"] == df["inferred_sender"]).astype(float)
    probs = np.clip(df["top1_prob"].to_numpy(dtype=float), 0.0, 1.0)
    y = df["top1_correct"].to_numpy(dtype=float)

    rows = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i < n_bins - 1:
            m = (probs >= lo) & (probs < hi)
        else:
            m = (probs >= lo) & (probs <= hi)
        if not np.any(m):
            continue
        rows.append({
            "conf": float(np.mean(probs[m])),
            "acc": float(np.mean(y[m])),
            "n": int(np.sum(m)),
        })
    if not rows:
        return
    cal = pd.DataFrame(rows).sort_values("conf")

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    axes[0].plot(cal["conf"], cal["acc"], marker="o", color="tab:blue", label="Observed")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Reliability Diagram (Non‑Abstain, All Scenarios)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].bar(cal["conf"], cal["n"], width=1.0 / n_bins * 0.9, color="tab:gray", alpha=0.8)
    axes[1].set_xlabel("Confidence (top‑1 prob)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Confidence Histogram")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot deanonymization curves")
    parser.add_argument("--input", required=True, help="deanon/ directory")
    parser.add_argument("--output", default="plots_deanon", help="Output directory")
    parser.add_argument("--bucket_s", type=int, default=3600, help="Time bucket in seconds")
    parser.add_argument("--scenarios", nargs="*", help="Filter scenarios")
    parser.add_argument("--topology", default=None, help="Filter by topology")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = load_deanon(args.input, args.scenarios)
    if args.topology:
        df = df[df.get("topology", "") == args.topology]

    plot_deanon_timeseries(df, os.path.join(args.output, "deanon_timeseries.png"), bucket_s=args.bucket_s)
    plot_deanon_summary(df, os.path.join(args.output, "deanon_summary.png"))
    plot_deanon_reliability(df, os.path.join(args.output, "deanon_reliability.png"))

    print(f"\nAll deanon plots saved to: {args.output}")


if __name__ == "__main__":
    main()
