#!/usr/bin/env python3
"""
plot_security_curves.py - Security-focused timeseries plots

Generates:
1) Security metrics over time (delivery, drops, key wait, secret fraction)
2) Attack impact vs baseline (delta metrics)
3) Smoothed total key pool (if quantum timeseries provided)

Usage:
  python3 quam/plot_security_curves.py \\
    --messages outputs/<tag>/trial_1/messages \\
    --timeseries outputs/<tag>/trial_1/timeseries \\
    --output outputs/<tag>/trial_1/plots_security
"""

import argparse
import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TOPOLOGIES = ["ring", "star", "mesh", "two_cluster_bridge"]


def _split_scenario_topology(scenario: str) -> Tuple[str, str]:
    for topo in TOPOLOGIES:
        suffix = f"_{topo}"
        if scenario.endswith(suffix):
            return scenario[: -len(suffix)], topo
    return scenario, ""


def load_messages(messages_dir: str, scenarios: Optional[List[str]] = None) -> pd.DataFrame:
    files = glob.glob(os.path.join(messages_dir, "messages_*.csv"))
    if not files:
        raise FileNotFoundError(f"No messages CSVs found in {messages_dir}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        base = os.path.basename(f)
        parts = base.replace("messages_", "").replace(".csv", "").rsplit("_seed", 1)
        scenario_full = parts[0] if parts else "unknown"
        scenario, topo = _split_scenario_topology(scenario_full)
        df["scenario_full"] = scenario_full
        df["scenario"] = scenario
        df["topology"] = topo
        df["seed"] = int(parts[1].split("_")[0]) if len(parts) > 1 else 0
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    if scenarios:
        out = out[out["scenario"].isin(scenarios)]
    return out


def load_timeseries(timeseries_dir: str, scenarios: Optional[List[str]] = None) -> pd.DataFrame:
    files = glob.glob(os.path.join(timeseries_dir, "quantum_*.csv"))
    if not files:
        raise FileNotFoundError(f"No quantum CSVs found in {timeseries_dir}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        base = os.path.basename(f)
        if base.startswith("quantum_"):
            base = base[len("quantum_"):]
        parts = base.replace(".csv", "").rsplit("_seed", 1)
        scenario_full = parts[0] if parts else "unknown"
        scenario, topo = _split_scenario_topology(scenario_full)
        df["scenario_full"] = scenario_full
        df["scenario"] = scenario
        df["topology"] = topo
        df["seed"] = int(parts[1].split("_")[0]) if len(parts) > 1 else 0
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    if scenarios:
        out = out[out["scenario"].isin(scenarios)]
    return out


def _status_flags(df: pd.DataFrame) -> pd.DataFrame:
    status = df["status"].astype(str)
    df = df.copy()
    df["is_delivered"] = status.str.contains("delivered")
    df["is_dropped"] = status.str.contains("dropped")
    df["is_no_keys"] = status.str.contains("no_keys")
    df["is_control"] = df["msg_type"].isin(["control_setpoint", "priority_action"])
    return df


def _bucket(df: pd.DataFrame, bucket_s: int) -> pd.DataFrame:
    df = df.copy()
    df["t_s"] = (df["created_ms"] / 1000.0).astype(float)
    df["bucket_s"] = (df["t_s"] // bucket_s) * bucket_s
    return df


def _aggregate_messages(df: pd.DataFrame, bucket_s: int) -> pd.DataFrame:
    df = _status_flags(df)
    df = _bucket(df, bucket_s)

    # latency for delivered
    latency_ms = df["delivered_ms"] - df["created_ms"]
    df["latency_ms"] = latency_ms.where(df["is_delivered"], np.nan)

    agg = df.groupby(["scenario", "seed", "bucket_s"]).agg(
        n_msgs=("msg_id", "count"),
        n_auth=("requires_auth", "sum"),
        n_attack=("is_attack", "sum"),
        n_control=("is_control", "sum"),
        delivered=("is_delivered", "sum"),
        dropped=("is_dropped", "sum"),
        dropped_no_keys=("is_no_keys", "sum"),
        key_wait_ms=("key_wait_ms", "mean"),
        latency_ms=("latency_ms", "mean"),
        qber=("qber_path_mean", "mean"),
        secret_fraction=("secret_fraction_path_mean", "mean"),
    ).reset_index()

    # ratios
    agg["delivered_ratio"] = agg["delivered"] / agg["n_msgs"].replace(0, np.nan)
    agg["dropped_ratio"] = agg["dropped"] / agg["n_msgs"].replace(0, np.nan)
    agg["no_keys_ratio_all"] = agg["dropped_no_keys"] / agg["n_msgs"].replace(0, np.nan)
    agg["no_keys_ratio_auth"] = agg["dropped_no_keys"] / agg["n_auth"].replace(0, np.nan)
    agg["attack_msg_rate"] = agg["n_attack"] / agg["n_msgs"].replace(0, np.nan)
    return agg


def _mean_ci(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    grouped = df.groupby(["scenario", "bucket_s"]).agg(
        mean=(value_col, "mean"),
        std=(value_col, "std"),
        n=("seed", "nunique"),
    ).reset_index()
    grouped["sem"] = grouped["std"] / grouped["n"].replace(0, np.nan).pow(0.5)
    grouped["ci_lo"] = grouped["mean"] - 1.96 * grouped["sem"]
    grouped["ci_hi"] = grouped["mean"] + 1.96 * grouped["sem"]
    return grouped


def _rolling(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.sort_values("bucket_s").copy()
    df["mean"] = df["mean"].rolling(window=window, min_periods=1).mean()
    df["ci_lo"] = df["ci_lo"].rolling(window=window, min_periods=1).mean()
    df["ci_hi"] = df["ci_hi"].rolling(window=window, min_periods=1).mean()
    return df


def plot_security_metrics(agg: pd.DataFrame, out_path: str, rolling: int = 5) -> None:
    metrics = [
        ("delivered_ratio", "Delivered Ratio"),
        ("no_keys_ratio_auth", "Dropped (No Keys) / Auth"),
        ("key_wait_ms", "Avg Key Wait (ms)"),
        ("secret_fraction", "Secret Fraction"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    scenarios = sorted(agg["scenario"].unique())

    for ax, (metric, title) in zip(axes, metrics):
        stats = _mean_ci(agg, metric)
        for scen in scenarios:
            sub = stats[stats["scenario"] == scen]
            sub = _rolling(sub, rolling)
            t_hours = sub["bucket_s"] / 3600.0
            ax.plot(t_hours, sub["mean"], label=scen)
            if sub["n"].max() > 1:
                ax.fill_between(t_hours, sub["ci_lo"], sub["ci_hi"], alpha=0.15)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (hours)")

    axes[0].legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_attack_impact(agg: pd.DataFrame, baseline: str, out_path: str, rolling: int = 5) -> None:
    metrics = [
        ("no_keys_ratio_auth", "Δ Dropped (No Keys) / Auth"),
        ("delivered_ratio", "Δ Delivered Ratio"),
        ("key_wait_ms", "Δ Avg Key Wait (ms)"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 9), sharex=True)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    base = _mean_ci(agg[agg["scenario"] == baseline], metrics[0][0])
    base = base[["bucket_s", "mean"]].rename(columns={"mean": "baseline"})

    scenarios = [s for s in agg["scenario"].unique() if s != baseline]
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        stats = _mean_ci(agg, metric)
        base_metric = _mean_ci(agg[agg["scenario"] == baseline], metric)
        base_metric = base_metric[["bucket_s", "mean"]].rename(columns={"mean": "baseline"})

        for scen in scenarios:
            sub = stats[stats["scenario"] == scen].merge(base_metric, on="bucket_s", how="left")
            sub["mean"] = sub["mean"] - sub["baseline"]
            sub["ci_lo"] = sub["ci_lo"] - sub["baseline"]
            sub["ci_hi"] = sub["ci_hi"] - sub["baseline"]
            sub = _rolling(sub, rolling)
            t_hours = sub["bucket_s"] / 3600.0
            ax.plot(t_hours, sub["mean"], label=scen)
            if sub["n"].max() > 1:
                ax.fill_between(t_hours, sub["ci_lo"], sub["ci_hi"], alpha=0.15)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (hours)")

    axes[0].legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_pool_smoothed(ts: pd.DataFrame, out_path: str, rolling: int = 15) -> None:
    # total pool across edges per time
    ts = ts.copy()
    ts["t_s"] = ts["t_s"].astype(float)
    pool = ts.groupby(["scenario", "seed", "t_s"]).agg(
        total_pool=("pool_level", "sum")
    ).reset_index()

    stats = pool.groupby(["scenario", "t_s"]).agg(
        mean=("total_pool", "mean"),
        std=("total_pool", "std"),
        n=("seed", "nunique"),
    ).reset_index()
    stats["sem"] = stats["std"] / stats["n"].replace(0, np.nan).pow(0.5)
    stats["ci_lo"] = stats["mean"] - 1.96 * stats["sem"]
    stats["ci_hi"] = stats["mean"] + 1.96 * stats["sem"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    for scen in sorted(stats["scenario"].unique()):
        sub = stats[stats["scenario"] == scen].sort_values("t_s")
        sub["mean"] = sub["mean"].rolling(window=rolling, min_periods=1).mean()
        sub["ci_lo"] = sub["ci_lo"].rolling(window=rolling, min_periods=1).mean()
        sub["ci_hi"] = sub["ci_hi"].rolling(window=rolling, min_periods=1).mean()
        ax.plot(sub["t_s"], sub["mean"], label=scen)
        if sub["n"].max() > 1:
            ax.fill_between(sub["t_s"], sub["ci_lo"], sub["ci_hi"], alpha=0.15)

    ax.set_title("Smoothed Total Key Pool Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total Pool (bits)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_effective_key_spend(msgs: pd.DataFrame, out_path: str, bucket_s: int, rolling: int = 5) -> None:
    df = msgs.copy()
    df = _bucket(df, bucket_s)
    df["key_bits_spent_total"] = pd.to_numeric(df.get("key_bits_spent_total"), errors="coerce").fillna(0.0)
    df["gate_decision"] = df.get("gate_decision", "").astype(str)
    df["is_allowed"] = df["gate_decision"] == "allow"
    df["key_bits_allowed"] = df["key_bits_spent_total"] * df["is_allowed"].astype(float)

    agg = df.groupby(["scenario", "seed", "bucket_s"]).agg(
        total_bits=("key_bits_spent_total", "sum"),
        allowed_bits=("key_bits_allowed", "sum"),
    ).reset_index()

    metrics = [
        ("total_bits", "Total Key Bits Spent"),
        ("allowed_bits", "Effective Key Bits Spent (Allowed)"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    scenarios = sorted(agg["scenario"].unique())

    for ax, (metric, title) in zip(axes, metrics):
        stats = _mean_ci(agg, metric)
        for scen in scenarios:
            sub = stats[stats["scenario"] == scen]
            sub = _rolling(sub, rolling)
            t_hours = sub["bucket_s"] / 3600.0
            ax.plot(t_hours, sub["mean"], label=scen)
            if sub["n"].max() > 1:
                ax.fill_between(t_hours, sub["ci_lo"], sub["ci_hi"], alpha=0.15)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Key Bits")

    axes[0].legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Security curves plotting")
    parser.add_argument("--messages", required=True, help="messages/ directory")
    parser.add_argument("--timeseries", help="timeseries/ directory (quantum)")
    parser.add_argument("--output", default="plots_security", help="Output directory")
    parser.add_argument("--bucket_s", type=int, default=60, help="Time bucket size in seconds")
    parser.add_argument("--rolling", type=int, default=5, help="Rolling window size (in buckets)")
    parser.add_argument("--scenarios", nargs="*", help="Filter scenarios")
    parser.add_argument("--topology", default=None, help="Filter by topology (ring, mesh, star, two_cluster_bridge)")
    parser.add_argument("--baseline", default="baseline", help="Baseline scenario name")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    msgs = load_messages(args.messages, args.scenarios)
    if args.topology:
        msgs = msgs[msgs.get("topology", "") == args.topology]
    agg = _aggregate_messages(msgs, args.bucket_s)

    plot_security_metrics(agg, os.path.join(args.output, "security_metrics.png"), rolling=args.rolling)
    plot_attack_impact(agg, args.baseline, os.path.join(args.output, "attack_impact_security.png"), rolling=args.rolling)

    if args.timeseries:
        ts = load_timeseries(args.timeseries, args.scenarios)
        if args.topology:
            ts = ts[ts.get("topology", "") == args.topology]
        plot_pool_smoothed(ts, os.path.join(args.output, "key_pool_smoothed.png"), rolling=max(5, args.rolling * 3))

    plot_effective_key_spend(msgs, os.path.join(args.output, "key_spend_effective.png"), bucket_s=args.bucket_s, rolling=args.rolling)

    print(f"\nAll security plots saved to: {args.output}")


if __name__ == "__main__":
    main()
