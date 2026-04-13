#!/usr/bin/env python3
"""
report_scope_defense_metrics.py

Aggregate deanon + cover-cost metrics by attacker scope and defense mode.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def load_summaries(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if os.path.isdir(p):
            p = os.path.join(p, "summary.csv")
        if not os.path.exists(p):
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError("No summary CSVs found from --summaries paths")
    return pd.concat(frames, ignore_index=True)


def add_attack_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["attack"] = out["scenario"].astype(str).str.replace(r"_def_.*$", "", regex=True)
    out.loc[out["scenario"] == "baseline", "attack"] = "baseline"
    return out


def aggregate(df: pd.DataFrame, attack: str) -> pd.DataFrame:
    out = add_attack_label(df)
    if attack != "all":
        out = out[out["attack"] == attack].copy()

    metrics = [
        "deanon_top1_acc_non_abstain",
        "deanon_top1prob_mean",
        "deanon_entropy_mean_bits",
        "deanon_ece_top1",
        "false_allow_rate",
        "false_block_rate",
        "deadline_miss_ratio",
        "control_deadline_miss_ratio",
        "delivered_latency_mean_ms",
        "control_latency_mean_ms",
        "delivered_key_wait_mean_ms",
        "cover_overhead_ratio",
        "cover_energy_kwh",
        "cover_messages_per_real_event",
        "eens_total_kwh",
        "eens_critical_kwh",
        "total_critical_unserved_kw_sum",
    ]
    for c in metrics:
        if c not in out.columns:
            out[c] = float("nan")

    group_cols = ["attacker_scope", "defense_mode"]
    agg = (
        out.groupby(group_cols, dropna=False)[metrics]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(group_cols)
    )
    return agg


def plot_breakout(df: pd.DataFrame, out_png: str) -> None:
    if df.empty:
        return

    defenses = ["none", "ratelimit", "block", "delay", "intrusion", "adaptive", "signature", "all"]
    scopes = sorted(df["attacker_scope"].dropna().unique().tolist())

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plots = [
        ("deanon_top1_acc_non_abstain", "Deanon Top-1 Acc (Non-Abstain)"),
        ("false_allow_rate", "Control False-Allow Rate"),
        ("false_block_rate", "Control False-Block Rate"),
        ("deadline_miss_ratio", "Deadline Miss Ratio"),
        ("delivered_latency_mean_ms", "Delivered Latency Mean (ms)"),
        ("eens_critical_kwh", "Critical EENS (kWh)"),
    ]

    for ax, (col, title) in zip(axes.flat, plots):
        for scope in scopes:
            d = df[df["attacker_scope"] == scope]
            x = [i for i, de in enumerate(defenses) if de in set(d["defense_mode"])]
            y = [float(d.loc[d["defense_mode"] == defenses[i], col].iloc[0]) for i in x]
            if x:
                ax.plot(x, y, marker="o", linewidth=1.8, label=scope)
        ax.set_title(title)
        ax.set_xticks(range(len(defenses)))
        ax.set_xticklabels(defenses, rotation=35, ha="right")
        ax.grid(True, alpha=0.25)
        if col == "cover_overhead_ratio":
            ax.set_ylim(0.0, 1.0)

    axes[0, 0].legend(loc="best", frameon=True)
    fig.suptitle("Scope vs Defense: Privacy + Control + Reliability KPIs", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_cost_tradeoffs(df: pd.DataFrame, out_png: str) -> None:
    if df.empty:
        return

    defenses = ["none", "ratelimit", "block", "delay", "intrusion", "adaptive", "signature", "all"]
    scopes = sorted(df["attacker_scope"].dropna().unique().tolist())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plots = [
        ("cover_overhead_ratio", "Cover Overhead Ratio"),
        ("cover_energy_kwh", "Cover Energy (kWh)"),
        ("delivered_key_wait_mean_ms", "Key Wait Mean (ms)"),
        ("eens_total_kwh", "Total EENS (kWh)"),
    ]

    for ax, (col, title) in zip(axes.flat, plots):
        for scope in scopes:
            d = df[df["attacker_scope"] == scope]
            x = [i for i, de in enumerate(defenses) if de in set(d["defense_mode"])]
            y = [float(d.loc[d["defense_mode"] == defenses[i], col].iloc[0]) for i in x]
            if x:
                ax.plot(x, y, marker="o", linewidth=1.8, label=scope)
        ax.set_title(title)
        ax.set_xticks(range(len(defenses)))
        ax.set_xticklabels(defenses, rotation=35, ha="right")
        ax.grid(True, alpha=0.25)
        if col == "cover_overhead_ratio":
            ax.set_ylim(0.0, 1.0)

    axes[0, 0].legend(loc="best", frameon=True)
    fig.suptitle("Scope vs Defense: Cost/Availability Trade-offs", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate scope/defense metrics from summary.csv files")
    ap.add_argument("--summaries", nargs="+", required=True, help="summary.csv files or summary/ directories")
    ap.add_argument("--attack", default="spoof", help="Attack to analyze (e.g., spoof, all_attacks, all)")
    ap.add_argument("--output", required=True, help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = load_summaries(args.summaries)
    agg = aggregate(df, args.attack)

    out_csv = os.path.join(args.output, f"scope_defense_breakout_{args.attack}.csv")
    agg.to_csv(out_csv, index=False)

    out_png = os.path.join(args.output, f"scope_defense_breakout_{args.attack}.png")
    plot_breakout(agg, out_png)
    out_png_cost = os.path.join(args.output, f"scope_defense_cost_tradeoffs_{args.attack}.png")
    plot_cost_tradeoffs(agg, out_png_cost)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_png_cost}")


if __name__ == "__main__":
    main()
