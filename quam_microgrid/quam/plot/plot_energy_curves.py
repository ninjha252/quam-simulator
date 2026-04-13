#!/usr/bin/env python3
"""
plot_energy_curves.py - Plot energy time series from QuAM simulations

Creates publication-ready figures showing:
1. Load vs generation over time
2. Served vs unserved energy
3. Cumulative EENS
4. Attack impact visualization

Usage:
  python3 plot_energy_curves.py --input outputs/experiment/trial_1/energy/
  python3 plot_energy_curves.py --input outputs/experiment/trial_1/energy/ --scenarios spoof_def_none spoof_def_all
"""

import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Optional

# Style settings for publication
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'figure.figsize': (10, 8),
})


def _split_scenario_topology(scenario: str) -> tuple:
    topologies = ["ring", "star", "mesh", "two_cluster_bridge"]
    for topo in topologies:
        suffix = f"_{topo}"
        if scenario.endswith(suffix):
            return scenario[: -len(suffix)], topo
    return scenario, ""


def load_energy_data(input_dir: str, scenarios: Optional[List[str]] = None) -> pd.DataFrame:
    """Load all energy CSV files from directory."""
    pattern = os.path.join(input_dir, "energy_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No energy CSV files found in {input_dir}")
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Extract scenario from filename
        basename = os.path.basename(f)
        parts = basename.replace("energy_", "").replace(".csv", "").rsplit("_seed", 1)
        scenario_full = parts[0] if parts else "unknown"
        scenario, topo = _split_scenario_topology(scenario_full)
        df["scenario_full"] = scenario_full
        df["scenario"] = scenario
        df["topology"] = topo
        df["seed"] = int(parts[1].split("_")[0]) if len(parts) > 1 else 0
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    
    if scenarios:
        data = data[data["scenario"].isin(scenarios)]
    
    return data


def add_effective_eens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute EENS from time series using the curtailment definition:
      curtailed_kw = total_load_kw - served_kw = shed + unserved

    Also computes an "unserved-only" EENS (excludes shedding) for comparison.
    """
    required = {"t_s", "total_load_kw", "served_kw", "unserved_kw", "shed_frac", "scenario", "seed", "microgrid"}
    if not required.issubset(set(df.columns)):
        return df

    out = df.copy()
    out = out.sort_values(["scenario", "seed", "microgrid", "t_s"])
    dt = out.groupby(["scenario", "seed", "microgrid"])["t_s"].diff().fillna(0.0)

    out["curtailed_kw"] = (out["total_load_kw"] - out["served_kw"]).clip(lower=0.0)
    out["unserved_energy_kwh"] = out["unserved_kw"] * (dt / 3600.0)
    out["curtailed_energy_kwh"] = out["curtailed_kw"] * (dt / 3600.0)

    out["eens_unserved_kwh"] = out.groupby(["scenario", "seed", "microgrid"])["unserved_energy_kwh"].cumsum()
    out["eens_effective_kwh"] = out.groupby(["scenario", "seed", "microgrid"])["curtailed_energy_kwh"].cumsum()
    return out


def plot_energy_balance(df: pd.DataFrame, scenario: str, output_path: str):
    """Plot energy balance for a single scenario."""
    data = df[df["scenario"] == scenario]
    
    if data.empty:
        print(f"No data for scenario: {scenario}")
        return
    
    # Average across seeds and microgrids
    grouped = data.groupby("t_s").agg({
        "total_load_kw": "mean",
        "gen_kw": "mean",
        "import_kw": "mean",
        "served_kw": "mean",
        "unserved_kw": "mean",
        "shed_frac": "mean",
        "is_attack_window": "max",
    }).reset_index()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Convert to hours
    t_hours = grouped["t_s"] / 3600
    
    # Panel 1: Load vs Supply
    ax1 = axes[0]
    ax1.fill_between(t_hours, 0, grouped["total_load_kw"], alpha=0.3, label="Total Load", color="red")
    ax1.plot(t_hours, grouped["gen_kw"], label="Generation", color="green", linewidth=1.5)
    ax1.plot(t_hours, grouped["gen_kw"] + grouped["import_kw"], label="Gen + Import", 
             color="blue", linewidth=1.5, linestyle="--")
    ax1.set_ylabel("Power (kW)")
    ax1.set_title(f"Energy Balance: {scenario}")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # Shade attack windows
    attack_periods = grouped[grouped["is_attack_window"] == 1]
    if not attack_periods.empty:
        for _, row in attack_periods.iterrows():
            ax1.axvspan(row["t_s"]/3600 - 0.01, row["t_s"]/3600 + 0.01, 
                       alpha=0.1, color="red", zorder=0)
    
    # Panel 2: Served vs Curtailed (shed + unserved)
    ax2 = axes[1]
    curtailed_kw = (grouped["total_load_kw"] - grouped["served_kw"]).clip(lower=0.0)
    ax2.stackplot(t_hours, grouped["served_kw"], curtailed_kw,
                  labels=["Served", "Curtailed"], colors=["#2ecc71", "#e74c3c"], alpha=0.7)
    ax2.set_ylabel("Power (kW)")
    ax2.set_title("Served vs Curtailed Load")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Shed Fraction
    ax3 = axes[2]
    ax3.fill_between(t_hours, 0, grouped["shed_frac"] * 100, alpha=0.5, color="orange")
    ax3.plot(t_hours, grouped["shed_frac"] * 100, color="darkorange", linewidth=1.5)
    ax3.set_ylabel("Shed Fraction (%)")
    ax3.set_xlabel("Time (hours)")
    ax3.set_title("Load Shedding")
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    
    # Add attack window legend
    attack_patch = mpatches.Patch(color='red', alpha=0.1, label='Attack Window')
    ax3.legend(handles=[attack_patch], loc="upper right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_eens_comparison(df: pd.DataFrame, scenarios: List[str], output_path: str):
    """Plot cumulative EENS (curtailment) comparison across scenarios."""
    has_critical = "eens_critical_cumulative_kwh" in df.columns
    nrows = 2 if has_critical else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 8 if has_critical else 4), sharex=True)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    
    for i, scenario in enumerate(scenarios):
        data = df[df["scenario"] == scenario]
        if data.empty:
            continue
        
        agg = {"eens_effective_kwh": "mean"}
        if has_critical:
            agg["eens_critical_cumulative_kwh"] = "mean"
        grouped = data.groupby("t_s").agg(agg).reset_index()
        
        t_hours = grouped["t_s"] / 3600
        
        axes[0].plot(t_hours, grouped["eens_effective_kwh"],
                     label=scenario, color=colors[i], linewidth=1.5)
        if has_critical:
            axes[1].plot(t_hours, grouped["eens_critical_cumulative_kwh"],
                         label=scenario, color=colors[i], linewidth=1.5)
    
    axes[0].set_ylabel("Cumulative EENS (kWh)")
    axes[0].set_title("Total Energy Not Served (Curtailment)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    if has_critical:
        axes[1].set_ylabel("Critical EENS (kWh)")
        axes[1].set_xlabel("Time (hours)")
        axes[1].set_title("Critical Energy Not Served")
        axes[1].legend(loc="upper left", fontsize=8)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[0].set_xlabel("Time (hours)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_effective_eens_comparison(df: pd.DataFrame, scenarios: List[str], output_path: str):
    """Plot unserved-only EENS as a diagnostic (excludes shedding)."""
    if "eens_unserved_kwh" not in df.columns:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))

    for i, scenario in enumerate(scenarios):
        data = df[df["scenario"] == scenario]
        if data.empty:
            continue
        grouped = data.groupby("t_s").agg({"eens_unserved_kwh": "mean"}).reset_index()
        t_hours = grouped["t_s"] / 3600
        ax.plot(t_hours, grouped["eens_unserved_kwh"], label=scenario, color=colors[i], linewidth=1.5)

    ax.set_ylabel("Unserved-Only EENS (kWh)")
    ax.set_xlabel("Time (hours)")
    ax.set_title("Unserved-Only EENS (Diagnostic)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_attack_impact_effective_eens(df: pd.DataFrame, baseline_scenario: str, 
                                      attack_scenarios: List[str], output_path: str):
    """Plot attack impact vs baseline using curtailment EENS."""
    if "eens_unserved_kwh" not in df.columns:
        return
    baseline = df[df["scenario"] == baseline_scenario]
    if baseline.empty:
        return

    base_grouped = baseline.groupby("t_s").agg({"eens_effective_kwh": "mean"}).reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    colors = plt.cm.Set1(np.linspace(0, 1, len(attack_scenarios)))

    for i, scenario in enumerate(attack_scenarios):
        data = df[df["scenario"] == scenario]
        if data.empty:
            continue
        grouped = data.groupby("t_s").agg({"eens_effective_kwh": "mean"}).reset_index()
        merged = pd.merge(grouped, base_grouped, on="t_s", suffixes=("", "_baseline"))
        t_hours = merged["t_s"] / 3600
        delta = merged["eens_effective_kwh"] - merged["eens_effective_kwh_baseline"]
        ax.plot(t_hours, delta, label=scenario, color=colors[i], linewidth=1.5)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax.set_ylabel("Δ Curtailment EENS (kWh)")
    ax.set_xlabel("Time (hours)")
    ax.set_title("Attack Impact vs Baseline (Curtailment EENS)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_attack_impact(df: pd.DataFrame, baseline_scenario: str, 
                       attack_scenarios: List[str], output_path: str):
    """Plot attack impact as difference from baseline."""
    baseline = df[df["scenario"] == baseline_scenario]
    if baseline.empty:
        print(f"No baseline data for: {baseline_scenario}")
        return
    
    if "curtailed_kw" not in df.columns and {"total_load_kw", "served_kw"}.issubset(df.columns):
        df = df.copy()
        df["curtailed_kw"] = (df["total_load_kw"] - df["served_kw"]).clip(lower=0.0)

    baseline_grouped = baseline.groupby("t_s").agg({
        "curtailed_kw": "mean",
        "shed_frac": "mean",
    }).reset_index()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    colors = plt.cm.Set1(np.linspace(0, 1, len(attack_scenarios)))
    
    for i, scenario in enumerate(attack_scenarios):
        data = df[df["scenario"] == scenario]
        if data.empty:
            continue
        
        grouped = data.groupby("t_s").agg({
            "curtailed_kw": "mean",
            "shed_frac": "mean",
        }).reset_index()
        
        # Merge with baseline
        merged = pd.merge(grouped, baseline_grouped, on="t_s", suffixes=("", "_baseline"))
        t_hours = merged["t_s"] / 3600
        
        # Difference in curtailment (shed + unserved)
        diff_curtailed = merged["curtailed_kw"] - merged["curtailed_kw_baseline"]
        diff_shed = (merged["shed_frac"] - merged["shed_frac_baseline"]) * 100
        
        axes[0].plot(t_hours, diff_curtailed, label=scenario, color=colors[i], linewidth=1.5)
        axes[1].plot(t_hours, diff_shed, label=scenario, color=colors[i], linewidth=1.5)
    
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[0].set_ylabel("ΔCurtailment (kW)")
    axes[0].set_title("Additional Curtailment vs Baseline")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[1].set_ylabel("ΔShed Fraction (%)")
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_title("Additional Load Shedding vs Baseline")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_defense_effectiveness(df: pd.DataFrame, attack: str,
                               defenses: List[str], output_path: str):
    """Compare defense effectiveness for a given attack."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    has_critical = "eens_critical_cumulative_kwh" in df.columns

    # Collect only defenses that actually exist in the data
    defenses_present = []
    final_eens = []
    final_critical_eens = []
    max_shed = []

    for defense in defenses:
        scenario = f"{attack}_def_{defense}"
        data = df[df["scenario"] == scenario]
        if data.empty:
            continue

        defenses_present.append(defense)
        final_eens.append(data.groupby("seed")["eens_effective_kwh"].max().mean())
        if has_critical:
            final_critical_eens.append(data.groupby("seed")["eens_critical_cumulative_kwh"].max().mean())
        max_shed.append(data.groupby("seed")["shed_frac"].max().mean() * 100)

    if not defenses_present:
        print(f"No defense scenarios found for attack '{attack}'. Skipping defense plot.")
        plt.close()
        return

    scenarios = [f"{attack}_def_{d}" for d in defenses_present]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(scenarios)))

    # Bar plots
    x = np.arange(len(defenses_present))
    width = 0.35

    ax1 = axes[0, 0]
    ax1.bar(x, final_eens, width, label='Total EENS', color='steelblue')
    if has_critical:
        ax1.bar(x + width, final_critical_eens, width, label='Critical EENS', color='coral')
    ax1.set_xlabel("Defense Mode")
    ax1.set_ylabel("EENS (kWh)")
    ax1.set_title(f"Final EENS by Defense ({attack})")
    ax1.set_xticks(x + (width / 2 if has_critical else 0))
    ax1.set_xticklabels(defenses_present, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.bar(defenses_present, max_shed, color='orange')
    ax2.set_xlabel("Defense Mode")
    ax2.set_ylabel("Max Shed Fraction (%)")
    ax2.set_title(f"Maximum Load Shedding ({attack})")
    ax2.set_xticklabels(defenses_present, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    # Time series comparison
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    for i, (scenario, color) in enumerate(zip(scenarios, colors)):
        data = df[df["scenario"] == scenario]
        if data.empty:
            continue

        agg = {"shed_frac": "mean"}
        if has_critical:
            agg["eens_critical_cumulative_kwh"] = "mean"
        else:
            agg["eens_effective_kwh"] = "mean"
        grouped = data.groupby("t_s").agg(agg).reset_index()

        t_hours = grouped["t_s"] / 3600

        if has_critical:
            ax3.plot(t_hours, grouped["eens_critical_cumulative_kwh"],
                     label=defenses_present[i], color=color, linewidth=1.5)
        else:
            ax3.plot(t_hours, grouped["eens_effective_kwh"],
                     label=defenses_present[i], color=color, linewidth=1.5)
        ax4.plot(t_hours, grouped["shed_frac"] * 100,
                 label=defenses_present[i], color=color, linewidth=1.5)

    ax3.set_xlabel("Time (hours)")
    ax3.set_ylabel("Critical EENS (kWh)" if has_critical else "Total EENS (kWh)")
    ax3.set_title("Critical EENS Over Time" if has_critical else "Total EENS Over Time (Curtailment)")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4.set_xlabel("Time (hours)")
    ax4.set_ylabel("Shed Fraction (%)")
    ax4.set_title("Load Shedding Over Time")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"Defense Effectiveness Against {attack.upper()} Attack", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot energy curves from QuAM simulations")
    parser.add_argument("--input", required=True, help="Directory containing energy CSVs")
    parser.add_argument("--output", default="plots", help="Output directory for plots")
    parser.add_argument("--scenarios", nargs="*", help="Specific scenarios to plot")
    parser.add_argument("--topology", default=None, help="Filter by topology (ring, mesh, star, two_cluster_bridge)")
    parser.add_argument("--attack", default="spoof", help="Attack type for defense comparison")
    parser.add_argument("--defenses", nargs="*", 
                       default=["none", "block", "intrusion", "all"],
                       help="Defense modes to compare")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading data from: {args.input}")
    df = load_energy_data(args.input, args.scenarios)
    if args.topology:
        df = df[df.get("topology", "") == args.topology]
    df = add_effective_eens(df)
    print(f"Loaded {len(df)} records for {df['scenario'].nunique()} scenarios")
    
    scenarios = df["scenario"].unique().tolist()
    
    # Plot individual scenarios
    for scenario in scenarios:
        output_path = os.path.join(args.output, f"energy_balance_{scenario}.png")
        plot_energy_balance(df, scenario, output_path)
    
    # EENS comparison
    if len(scenarios) > 1:
        output_path = os.path.join(args.output, "eens_comparison.png")
        plot_eens_comparison(df, scenarios, output_path)
        # Unserved-only diagnostic (excludes shedding)
        output_path = os.path.join(args.output, "eens_unserved_diagnostic.png")
        plot_effective_eens_comparison(df, scenarios, output_path)
    
    # Attack impact (if baseline exists)
    if "baseline" in scenarios:
        attack_scenarios = [s for s in scenarios if s != "baseline"]
        if attack_scenarios:
            output_path = os.path.join(args.output, "attack_impact.png")
            plot_attack_impact(df, "baseline", attack_scenarios, output_path)
            output_path = os.path.join(args.output, "attack_impact_effective_eens.png")
            plot_attack_impact_effective_eens(df, "baseline", attack_scenarios, output_path)
    
    # Defense effectiveness (only plot defenses that exist in data)
    attack_prefix = f"{args.attack}_def_"
    available_defenses = [s[len(attack_prefix):] for s in scenarios if s.startswith(attack_prefix)]
    defenses = [d for d in args.defenses if d in available_defenses]
    for d in available_defenses:
        if d not in defenses:
            defenses.append(d)
    if defenses:
        output_path = os.path.join(args.output, f"defense_effectiveness_{args.attack}.png")
        plot_defense_effectiveness(df, args.attack, defenses, output_path)
    
    print(f"\nAll plots saved to: {args.output}")


if __name__ == "__main__":
    main()
