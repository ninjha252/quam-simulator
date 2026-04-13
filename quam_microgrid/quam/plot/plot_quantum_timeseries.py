#!/usr/bin/env python3
"""
plot_quantum_timeseries.py - Plot quantum timeseries (fidelity, QBER, key pool)

Usage:
  python3 quam/plot_quantum_timeseries.py --input outputs/<tag>/trial_1/timeseries --output outputs/<tag>/trial_1/plots_quantum
  python3 quam/plot_quantum_timeseries.py --input outputs/<tag>/trial_1/timeseries --output outputs/<tag>/trial_1/plots_quantum --scenarios baseline all_attacks_def_none
"""

import argparse
import os
import glob
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _shade_flag(ax, t_hours, flag_vals, color="gray", alpha=0.15):
    in_seg = False
    start = None
    for t, f in zip(t_hours, flag_vals):
        if f and not in_seg:
            in_seg = True
            start = t
        elif not f and in_seg:
            ax.axvspan(start, t, color=color, alpha=alpha)
            in_seg = False
    if in_seg and start is not None:
        ax.axvspan(start, t_hours[-1], color=color, alpha=alpha)


def _split_scenario_topology(scenario: str) -> Tuple[str, str]:
    topologies = ["ring", "star", "mesh", "two_cluster_bridge"]
    for topo in topologies:
        suffix = f"_{topo}"
        if scenario.endswith(suffix):
            return scenario[: -len(suffix)], topo
    return scenario, ""


def load_quantum_data(input_dir: str, scenarios: Optional[List[str]] = None) -> pd.DataFrame:
    pattern = os.path.join(input_dir, "quantum_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No quantum CSV files found in {input_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        basename = os.path.basename(f)
        if basename.startswith("quantum_"):
            basename = basename[len("quantum_"):]
        parts = basename.replace(".csv", "").rsplit("_seed", 1)
        scenario_full = parts[0] if parts else "unknown"
        scenario, topo = _split_scenario_topology(scenario_full)
        df["scenario_full"] = scenario_full
        df["scenario"] = scenario
        df["topology"] = topo
        df["seed"] = int(parts[1].split("_")[0]) if len(parts) > 1 else 0
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # If fidelity not present, compute from qber
    if "fidelity" not in data.columns and "qber" in data.columns:
        data["fidelity"] = 1.0 - data["qber"].clip(0, 1)

    if scenarios:
        data = data[data["scenario"].isin(scenarios)]

    return data


def plot_fidelity_curve(df: pd.DataFrame, scenario: str, output_path: str) -> None:
    data = df[df["scenario"] == scenario]
    if data.empty:
        print(f"No data for scenario: {scenario}")
        return

    # Aggregate across edges/seeds
    agg = {
        "fidelity": "mean",
        "qber": "mean" if "qber" in data.columns else "mean",
        "secret_fraction": "mean" if "secret_fraction" in data.columns else "mean",
        "pool_level": "mean" if "pool_level" in data.columns else "mean",
        "is_attack": "max" if "is_attack" in data.columns else "max",
    }
    if "abort_active" in data.columns:
        agg["abort_active"] = "max"
    grouped = data.groupby("t_s").agg(agg).reset_index()

    t_hours = grouped["t_s"] / 3600.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Fidelity curve
    axes[0].plot(t_hours, grouped["fidelity"], color="purple", linewidth=1.5, label="Fidelity")
    axes[0].set_ylabel("Fidelity")
    axes[0].set_title(f"Fidelity Over Time: {scenario}")
    axes[0].grid(True, alpha=0.3)

    # Pool level curve
    axes[1].plot(t_hours, grouped["pool_level"], color="teal", linewidth=1.5, label="Key Pool Level")
    axes[1].set_ylabel("Key Pool Level (bits)")
    axes[1].set_xlabel("Time (hours)")
    axes[1].grid(True, alpha=0.3)

    # Shade attack windows
    if "is_attack" in grouped.columns:
        attack_times = grouped[grouped["is_attack"] == 1]["t_s"].values
        if len(attack_times) > 0:
            for t_s in attack_times:
                t_hr = t_s / 3600.0
                axes[0].axvspan(t_hr - 0.01, t_hr + 0.01, alpha=0.1, color="red")
                axes[1].axvspan(t_hr - 0.01, t_hr + 0.01, alpha=0.1, color="red")


    # Shade abort (keygen disabled) windows
    if "abort_active" in grouped.columns:
        _shade_flag(axes[0], t_hours, grouped["abort_active"], color="gray", alpha=0.15)
        _shade_flag(axes[1], t_hours, grouped["abort_active"], color="gray", alpha=0.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_qber_secret(df: pd.DataFrame, scenario: str, output_path: str) -> None:
    data = df[df["scenario"] == scenario]
    if data.empty:
        return

    agg = {
        "qber": "mean" if "qber" in data.columns else "mean",
        "secret_fraction": "mean" if "secret_fraction" in data.columns else "mean",
        "is_attack": "max" if "is_attack" in data.columns else "max",
    }
    if "abort_active" in data.columns:
        agg["abort_active"] = "max"
    grouped = data.groupby("t_s").agg(agg).reset_index()

    t_hours = grouped["t_s"] / 3600.0

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    if "qber" in grouped.columns:
        ax.plot(t_hours, grouped["qber"], label="QBER", color="darkred", linewidth=1.5)
    if "secret_fraction" in grouped.columns:
        ax.plot(t_hours, grouped["secret_fraction"], label="Secret Fraction", color="navy", linewidth=1.5)

    ax.set_xlabel("Time (hours)")
    ax.set_title(f"QBER and Secret Fraction: {scenario}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Shade attack windows
    if "is_attack" in grouped.columns:
        attack_times = grouped[grouped["is_attack"] == 1]["t_s"].values
        if len(attack_times) > 0:
            for t_s in attack_times:
                t_hr = t_s / 3600.0
                ax.axvspan(t_hr - 0.01, t_hr + 0.01, alpha=0.1, color="red")


    # Shade abort (keygen disabled) windows
    if "abort_active" in grouped.columns:
        _shade_flag(ax, t_hours, grouped["abort_active"], color="gray", alpha=0.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot quantum timeseries")
    parser.add_argument("--input", required=True, help="Directory containing quantum CSVs")
    parser.add_argument("--output", default="plots_quantum", help="Output directory for plots")
    parser.add_argument("--scenarios", nargs="*", help="Filter scenarios to plot")
    parser.add_argument("--topology", default=None, help="Filter by topology (ring, mesh, star, two_cluster_bridge)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading data from: {args.input}")
    df = load_quantum_data(args.input, args.scenarios)
    if args.topology:
        df = df[df.get("topology", "") == args.topology]
    print(f"Loaded {len(df)} records for {df['scenario'].nunique()} scenarios")

    scenarios = df["scenario"].unique().tolist()
    for scenario in scenarios:
        plot_fidelity_curve(df, scenario, os.path.join(args.output, f"fidelity_{scenario}.png"))
        plot_qber_secret(df, scenario, os.path.join(args.output, f"qber_secret_{scenario}.png"))

    print(f"\nAll plots saved to: {args.output}")


if __name__ == "__main__":
    main()
