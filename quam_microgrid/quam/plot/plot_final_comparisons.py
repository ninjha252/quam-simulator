#!/usr/bin/env python3
"""
plot_final_comparisons.py

Create publication-style comparison plots from two or more QuAM summary CSV files.

Example:
  python3 -m quam.plot_final_comparisons \
    --summaries outputs/run_24h_10nodes_qan10_randS/trial_1/summary/summary.csv \
               outputs/run_24h_10nodes_qan500_randS/trial_1/summary/summary.csv \
    --labels QAN10 QAN500 \
    --output outputs/final_comparison_24h_10nodes_randS
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_ORDER: List[str] = [
    "baseline",
    "spoof_def_none",
    "spoof_def_all",
    "exhaust_def_none",
    "exhaust_def_all",
    "quantum_def_none",
    "quantum_def_all",
    "all_attacks_def_none",
    "all_attacks_def_all",
]

TOPOLOGY_ORDER: List[str] = ["ring", "star", "mesh", "two_cluster_bridge"]


def _short_scenario(name: str) -> str:
    mapping = {
        "baseline": "base",
        "spoof_def_none": "spoof:n",
        "spoof_def_all": "spoof:a",
        "exhaust_def_none": "exh:n",
        "exhaust_def_all": "exh:a",
        "quantum_def_none": "qatt:n",
        "quantum_def_all": "qatt:a",
        "all_attacks_def_none": "all:n",
        "all_attacks_def_all": "all:a",
    }
    return mapping.get(name, name)


def _load_summaries(paths: List[str], labels: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p, label in zip(paths, labels):
        df = pd.read_csv(p)
        df["run_label"] = label
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["scenario"] = out["scenario"].astype(str)
    out["topology"] = out["topology"].astype(str)
    return out


def _ensure_output(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pivot_metric(
    df: pd.DataFrame,
    metric: str,
    scenarios: List[str],
    topologies: List[str],
) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for topo in topologies:
        sub = df[(df["topology"] == topo) & (df["scenario"].isin(scenarios))].copy()
        p = (
            sub.groupby(["scenario", "run_label"], as_index=False)[metric]
            .mean()
            .pivot(index="scenario", columns="run_label", values=metric)
            .reindex(scenarios)
        )
        result[topo] = p
    return result


def _plot_eens_by_topology(df: pd.DataFrame, out_dir: str, run_labels: List[str]) -> None:
    scenarios = SCENARIO_ORDER
    pivots = _pivot_metric(df, "eens_total_kwh", scenarios, TOPOLOGY_ORDER)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    axes = axes.ravel()
    width = 0.38 if len(run_labels) == 2 else 0.8 / max(1, len(run_labels))
    x = np.arange(len(scenarios))

    for ax, topo in zip(axes, TOPOLOGY_ORDER):
        p = pivots[topo]
        for i, label in enumerate(run_labels):
            vals = p[label].values if label in p.columns else np.zeros(len(scenarios))
            off = (i - (len(run_labels) - 1) / 2) * width
            ax.bar(x + off, vals, width=width, label=label, alpha=0.9)
        ax.set_title(f"{topo} - EENS")
        ax.set_xticks(x)
        ax.set_xticklabels([_short_scenario(s) for s in scenarios], rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylabel("kWh")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(run_labels)))
    fig.suptitle("Energy Reliability Comparison by Topology")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(out_dir, "compare_eens_by_topology.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_security_key_delivery(df: pd.DataFrame, out_dir: str, run_labels: List[str]) -> None:
    focus = ["baseline", "spoof_def_none", "all_attacks_def_none", "all_attacks_def_all"]
    sec = (
        df[df["scenario"].isin(focus)]
        .groupby(["run_label", "scenario"], as_index=False)[
            ["delivered_ratio", "dropped_no_keys_ratio", "delivered_key_wait_mean_ms"]
        ]
        .mean()
    )
    scenarios = focus
    x = np.arange(len(scenarios))
    width = 0.38 if len(run_labels) == 2 else 0.8 / max(1, len(run_labels))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("delivered_ratio", "Delivered Ratio"),
        ("dropped_no_keys_ratio", "Dropped (No Keys) Ratio"),
        ("delivered_key_wait_mean_ms", "Avg Key Wait (ms)"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for i, label in enumerate(run_labels):
            sub = sec[sec["run_label"] == label].set_index("scenario")
            vals = [float(sub.at[s, metric]) if s in sub.index else 0.0 for s in scenarios]
            off = (i - (len(run_labels) - 1) / 2) * width
            ax.bar(x + off, vals, width=width, label=label, alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([_short_scenario(s) for s in scenarios], rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(run_labels)))
    fig.suptitle("Security Delivery and Key-Use Comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(out_dir, "compare_security_delivery.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_quantum_health(df: pd.DataFrame, out_dir: str, run_labels: List[str]) -> None:
    focus = ["baseline", "quantum_def_none", "quantum_def_all", "all_attacks_def_none", "all_attacks_def_all"]
    q = (
        df[df["scenario"].isin(focus)]
        .groupby(["run_label", "scenario"], as_index=False)[
            ["qber_mean", "secret_fraction_mean", "fidelity_min_mean"]
        ]
        .mean()
    )
    scenarios = focus
    x = np.arange(len(scenarios))
    width = 0.38 if len(run_labels) == 2 else 0.8 / max(1, len(run_labels))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("qber_mean", "QBER Mean"),
        ("secret_fraction_mean", "Secret Fraction Mean"),
        ("fidelity_min_mean", "Fidelity Minimum Mean"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for i, label in enumerate(run_labels):
            sub = q[q["run_label"] == label].set_index("scenario")
            vals = [float(sub.at[s, metric]) if s in sub.index else 0.0 for s in scenarios]
            off = (i - (len(run_labels) - 1) / 2) * width
            ax.bar(x + off, vals, width=width, label=label, alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([_short_scenario(s) for s in scenarios], rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(run_labels)))
    fig.suptitle("Quantum Health Comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(out_dir, "compare_quantum_health.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_deanon(df: pd.DataFrame, out_dir: str, run_labels: List[str]) -> None:
    focus = ["spoof_def_none", "spoof_def_all", "all_attacks_def_none", "all_attacks_def_all"]
    d = (
        df[df["scenario"].isin(focus)]
        .groupby(["run_label", "scenario"], as_index=False)[
            ["deanon_top1_acc", "deanon_entropy_mean_bits", "deanon_top1prob_mean", "deanon_ece_top1"]
        ]
        .mean()
    )
    scenarios = focus
    x = np.arange(len(scenarios))
    width = 0.38 if len(run_labels) == 2 else 0.8 / max(1, len(run_labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    left_metrics = [("deanon_top1_acc", "Top-1 Accuracy"), ("deanon_top1prob_mean", "Top-1 Probability")]
    right_metrics = [("deanon_entropy_mean_bits", "Entropy (bits)"), ("deanon_ece_top1", "ECE")]

    # Left axis
    for metric, _title in left_metrics:
        for i, label in enumerate(run_labels):
            sub = d[d["run_label"] == label].set_index("scenario")
            vals = [float(sub.at[s, metric]) if s in sub.index else 0.0 for s in scenarios]
            off = (i - (len(run_labels) - 1) / 2) * (width / 2.0)
            axes[0].plot(x + off, vals, marker="o", label=f"{label}:{metric}")
    axes[0].set_title("Deanonymization Accuracy/Confidence")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([_short_scenario(s) for s in scenarios], rotation=25, ha="right")
    axes[0].grid(True, alpha=0.3)

    # Right axis
    for metric, _title in right_metrics:
        for i, label in enumerate(run_labels):
            sub = d[d["run_label"] == label].set_index("scenario")
            vals = [float(sub.at[s, metric]) if s in sub.index else 0.0 for s in scenarios]
            off = (i - (len(run_labels) - 1) / 2) * (width / 2.0)
            axes[1].plot(x + off, vals, marker="o", label=f"{label}:{metric}")
    axes[1].set_title("Deanonymization Uncertainty/Calibration")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_short_scenario(s) for s in scenarios], rotation=25, ha="right")
    axes[1].grid(True, alpha=0.3)

    axes[0].legend(fontsize=8, loc="best")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Deanonymization Comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(out_dir, "compare_deanon.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_comparison_table(df: pd.DataFrame, out_dir: str) -> None:
    table = (
        df.groupby(["run_label", "topology", "scenario"], as_index=False)[
            [
                "delivered_ratio",
                "dropped_no_keys_ratio",
                "eens_total_kwh",
                "eens_critical_kwh",
                "qber_mean",
                "secret_fraction_mean",
                "fidelity_min_mean",
                "deanon_top1_acc",
                "deanon_entropy_mean_bits",
            ]
        ]
        .mean()
    )
    table.to_csv(os.path.join(out_dir, "comparison_table.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final comparison plots from summary CSVs")
    parser.add_argument("--summaries", nargs="+", required=True, help="List of summary.csv files")
    parser.add_argument("--labels", nargs="+", required=True, help="Run labels matching --summaries")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    if len(args.summaries) != len(args.labels):
        raise ValueError("Number of --summaries must match number of --labels")

    _ensure_output(args.output)
    df = _load_summaries(args.summaries, args.labels)
    _write_comparison_table(df, args.output)
    _plot_eens_by_topology(df, args.output, args.labels)
    _plot_security_key_delivery(df, args.output, args.labels)
    _plot_quantum_health(df, args.output, args.labels)
    _plot_deanon(df, args.output, args.labels)
    print(f"Saved comparison plots to: {args.output}")


if __name__ == "__main__":
    main()
