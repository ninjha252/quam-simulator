#!/usr/bin/env python3
"""
report_quantum_layer_impact.py

Build integration-story artifacts for the quantum layer:
1) Key depletion under deanon-guided exhaustion (and other exhaustion modes)
2) Key-rate reduction under distance + finite-key + degraded mode settings
3) QAN overhead vs key usage trade-off
"""

from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def _load_csvs(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        path = p
        if os.path.isdir(path):
            path = os.path.join(path, "summary.csv")
        if not os.path.exists(path):
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No summary.csv files found")
    return pd.concat(frames, ignore_index=True)


def _attack_from_scenario(s: str) -> str:
    if s == "baseline":
        return "baseline"
    if "_def_" in s:
        return s.split("_def_", 1)[0]
    return s


def _safe_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = float("nan")
    return out


def make_depletion_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["attack"] = work["scenario"].astype(str).map(_attack_from_scenario)
    work = work[work["attack"].isin(["exhaust", "all_attacks"])].copy()
    work = _safe_num(work, [
        "deanon_top1_acc_non_abstain",
        "dropped_no_keys_ratio",
        "delivered_key_wait_mean_ms",
        "key_bits_spent_sum",
        "key_bits_spent_qan_total_sum",
        "key_bits_spent_qan_share",
        "key_bits_spent_non_qan_sum",
        "cover_overhead_ratio",
        "cover_energy_kwh",
        "cover_messages_per_real_event",
        "eens_total_kwh",
    ])
    if "exhaust_strategy" not in work.columns:
        work["exhaust_strategy"] = "unknown"
    if "attacker_scope" not in work.columns:
        work["attacker_scope"] = "global"

    agg = (
        work.groupby(["attack", "exhaust_strategy", "attacker_scope", "defense_mode"], dropna=False)[[
            "deanon_top1_acc_non_abstain",
            "dropped_no_keys_ratio",
            "delivered_key_wait_mean_ms",
            "key_bits_spent_sum",
            "key_bits_spent_qan_total_sum",
            "key_bits_spent_qan_share",
            "key_bits_spent_non_qan_sum",
            "cover_overhead_ratio",
            "cover_energy_kwh",
            "cover_messages_per_real_event",
            "eens_total_kwh",
        ]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["attack", "exhaust_strategy", "attacker_scope", "defense_mode"])
    )
    agg["qan_key_draw_share_pct"] = 100.0 * agg["key_bits_spent_qan_total_sum"] / agg["key_bits_spent_sum"].replace(0, pd.NA)
    return agg


def make_scope_defense_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["attack"] = work["scenario"].astype(str).map(_attack_from_scenario)
    if "attacker_scope" not in work.columns:
        work["attacker_scope"] = "global"
    if "exhaust_strategy" not in work.columns:
        work["exhaust_strategy"] = "uniform"

    work = _safe_num(work, [
        "deanon_top1_acc_non_abstain",
        "deanon_top1prob_mean",
        "dropped_no_keys_ratio",
        "delivered_ratio",
        "delivered_key_wait_mean_ms",
        "cover_overhead_ratio",
        "cover_energy_kwh",
        "cover_messages_per_real_event",
        "key_bits_spent_qan_share",
        "eens_total_kwh",
    ])
    agg = (
        work.groupby(
            ["attack", "exhaust_strategy", "attacker_scope", "defense_mode"],
            dropna=False,
        )[[
            "deanon_top1_acc_non_abstain",
            "deanon_top1prob_mean",
            "dropped_no_keys_ratio",
            "delivered_ratio",
            "delivered_key_wait_mean_ms",
            "cover_overhead_ratio",
            "cover_energy_kwh",
            "cover_messages_per_real_event",
            "key_bits_spent_qan_share",
            "eens_total_kwh",
        ]]
        .mean(numeric_only=True)
        .reset_index()
    )
    agg["qan_key_draw_share_pct"] = 100.0 * agg["key_bits_spent_qan_share"]
    return agg.sort_values(["attack", "exhaust_strategy", "attacker_scope", "defense_mode"])


def make_keyrate_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["attack"] = work["scenario"].astype(str).map(_attack_from_scenario)
    # baseline gives the cleanest key-rate picture; fallback to all if baseline absent
    if (work["attack"] == "baseline").any():
        work = work[work["attack"] == "baseline"].copy()

    work = _safe_num(work, [
        "avg_link_distance_km",
        "effective_key_rate_reduction_pct",
        "secret_fraction_mean",
        "dropped_no_keys_ratio",
        "delivered_key_wait_mean_ms",
        "degraded_threshold_sf",
        "finite_key_factor",
    ])
    if "finite_key_preset" not in work.columns:
        work["finite_key_preset"] = "unknown"

    agg = (
        work.groupby(["avg_link_distance_km", "finite_key_preset", "degraded_threshold_sf", "defense_mode"], dropna=False)[[
            "effective_key_rate_reduction_pct",
            "secret_fraction_mean",
            "dropped_no_keys_ratio",
            "delivered_key_wait_mean_ms",
            "finite_key_factor",
        ]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["avg_link_distance_km", "finite_key_preset", "degraded_threshold_sf", "defense_mode"])
    )
    return agg


def make_qan_cost_table(df: pd.DataFrame) -> pd.DataFrame:
    work = _safe_num(df.copy(), [
        "qan_events_requested",
        "qan_cover_rate_per_s",
        "cover_overhead_ratio",
        "cover_energy_kwh",
        "cover_messages_per_real_event",
        "key_bits_spent_qan_total_sum",
        "key_bits_spent_cover_share",
        "key_bits_spent_qan_share",
        "key_bits_spent_sum",
        "eens_total_kwh",
    ])
    if "attacker_scope" not in work.columns:
        work["attacker_scope"] = "global"

    for col, default in (
        ("qan_auth_cover", 0),
        ("qan_auth_real_notify", 0),
        ("qan_auth_sync_burst", 0),
    ):
        if col not in work.columns:
            work[col] = default

    agg = (
        work.groupby([
            "attacker_scope",
            "defense_mode",
            "qan_events_requested",
            "qan_cover_rate_per_s",
            "qan_auth_cover",
            "qan_auth_real_notify",
            "qan_auth_sync_burst",
        ], dropna=False)[[
            "cover_overhead_ratio",
            "cover_energy_kwh",
            "cover_messages_per_real_event",
            "key_bits_spent_qan_total_sum",
            "key_bits_spent_cover_share",
            "key_bits_spent_qan_share",
            "key_bits_spent_sum",
            "eens_total_kwh",
        ]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values([
            "attacker_scope",
            "defense_mode",
            "qan_events_requested",
            "qan_cover_rate_per_s",
            "qan_auth_cover",
            "qan_auth_real_notify",
        ])
    )
    agg["qan_key_draw_share_pct"] = 100.0 * agg["key_bits_spent_qan_share"]
    agg["cover_key_draw_share_pct"] = 100.0 * agg["key_bits_spent_cover_share"]
    return agg


def plot_depletion(df: pd.DataFrame, out_png: str) -> None:
    if df.empty:
        return
    plot_df = df[df["defense_mode"].isin(["none", "all", "block", "intrusion"])].copy()
    if plot_df.empty:
        plot_df = df.copy()

    x_labels = sorted(plot_df["attacker_scope"].astype(str).unique())
    strategies = sorted(plot_df["exhaust_strategy"].astype(str).unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.8 / max(1, len(strategies))

    for i, strat in enumerate(strategies):
        d = plot_df[plot_df["exhaust_strategy"] == strat]
        xs = list(range(len(x_labels)))
        ys_drop = []
        ys_wait = []
        for s in x_labels:
            ds = d[d["attacker_scope"] == s]
            ys_drop.append(float(ds["dropped_no_keys_ratio"].mean()) if not ds.empty else float("nan"))
            ys_wait.append(float(ds["delivered_key_wait_mean_ms"].mean()) if not ds.empty else float("nan"))

        xoff = [x + (i - (len(strategies) - 1) / 2) * width for x in xs]
        axes[0].bar(xoff, ys_drop, width=width, label=strat)
        axes[1].bar(xoff, ys_wait, width=width, label=strat)

    axes[0].set_title("QKD Depletion (Dropped: No Keys)")
    axes[0].set_ylabel("Dropped / Total")
    axes[1].set_title("Key Wait Latency")
    axes[1].set_ylabel("Mean key_wait (ms)")

    for ax in axes:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="best", frameon=True)
    fig.suptitle("Deanon-Guided Exhaustion Impact by Attacker Scope", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_scope_defense(df: pd.DataFrame, out_png: str) -> None:
    if df.empty:
        return
    focus = df[df["attack"].isin(["spoof", "exhaust", "all_attacks"])].copy()
    if focus.empty:
        focus = df.copy()
    # Prefer deanon-guided rows where available for clearer attacker-capability comparison.
    if (focus["exhaust_strategy"] == "deanon_guided").any():
        focus = focus[(focus["exhaust_strategy"] == "deanon_guided") | (~focus["attack"].isin(["exhaust", "all_attacks"]))]

    defenses = ["none", "ratelimit", "block", "intrusion", "all"]
    scopes = sorted(str(s) for s in focus["attacker_scope"].dropna().unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plots = [
        ("deanon_top1_acc_non_abstain", "Deanon Top-1 Accuracy (non-abstain)"),
        ("dropped_no_keys_ratio", "Dropped No-Keys Ratio"),
        ("cover_overhead_ratio", "Cover Overhead Ratio"),
        ("cover_energy_kwh", "Cover Energy (kWh)"),
    ]
    for ax, (metric, title) in zip(axes.flat, plots):
        for scope in scopes:
            d = focus[focus["attacker_scope"] == scope]
            xs, ys = [], []
            for i, defense in enumerate(defenses):
                dd = d[d["defense_mode"] == defense]
                if dd.empty:
                    continue
                xs.append(i)
                ys.append(float(dd[metric].mean()))
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=scope)
        ax.set_title(title)
        ax.set_xticks(range(len(defenses)))
        ax.set_xticklabels(defenses)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best", frameon=True)
    fig.suptitle("Defense Effectiveness Isolation by Attacker Scope", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_keyrate(df: pd.DataFrame, out_png: str) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    presets = sorted(df["finite_key_preset"].astype(str).unique())
    for p in presets:
        d = df[df["finite_key_preset"] == p].sort_values("avg_link_distance_km")
        axes[0].plot(d["avg_link_distance_km"], d["effective_key_rate_reduction_pct"], marker="o", label=p)
        axes[1].plot(d["avg_link_distance_km"], d["secret_fraction_mean"], marker="o", label=p)

    axes[0].set_title("Key-Rate Reduction vs Distance")
    axes[0].set_xlabel("Average link distance (km)")
    axes[0].set_ylabel("Effective key-rate reduction (%)")
    axes[1].set_title("Secret Fraction vs Distance")
    axes[1].set_xlabel("Average link distance (km)")
    axes[1].set_ylabel("Secret fraction (mean)")
    for ax in axes:
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="best", frameon=True)
    fig.suptitle("Distance + Finite-Key Impact", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_qan_cost(df: pd.DataFrame, out_png: str) -> None:
    if df.empty:
        return
    plot_df = df.sort_values(["qan_events_requested", "qan_cover_rate_per_s"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for auth_flag, grp in plot_df.groupby("qan_auth_cover"):
        label = "cover-auth=on" if int(auth_flag) == 1 else "cover-auth=off"
        axes[0].scatter(grp["cover_overhead_ratio"], grp["cover_energy_kwh"], s=70, alpha=0.75, label=label)
        axes[1].scatter(grp["cover_overhead_ratio"], grp["qan_key_draw_share_pct"], s=70, alpha=0.75, label=label)

    axes[0].set_title("Cover Overhead vs Cover Energy")
    axes[0].set_xlabel("Cover overhead ratio")
    axes[0].set_ylabel("Cover energy (kWh)")

    axes[1].set_title("Cover Overhead vs QAN Key Draw")
    axes[1].set_xlabel("Cover overhead ratio")
    axes[1].set_ylabel("QAN key-draw share (%)")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="best", frameon=True)
    fig.suptitle("QAN Cost-Benefit (Traffic + Keys)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantum-layer integration impact report")
    ap.add_argument("--summaries", nargs="+", required=True, help="summary.csv files or summary/ dirs")
    ap.add_argument("--output", required=True, help="output folder")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = _load_csvs(args.summaries)

    depletion = make_depletion_table(df)
    scope_defense = make_scope_defense_table(df)
    keyrate = make_keyrate_table(df)
    qan_cost = make_qan_cost_table(df)

    depletion_csv = os.path.join(args.output, "quantum_impact_depletion.csv")
    scope_defense_csv = os.path.join(args.output, "quantum_impact_scope_defense.csv")
    keyrate_csv = os.path.join(args.output, "quantum_impact_keyrate.csv")
    qan_cost_csv = os.path.join(args.output, "quantum_impact_qan_cost.csv")
    depletion.to_csv(depletion_csv, index=False)
    scope_defense.to_csv(scope_defense_csv, index=False)
    keyrate.to_csv(keyrate_csv, index=False)
    qan_cost.to_csv(qan_cost_csv, index=False)

    plot_depletion(depletion, os.path.join(args.output, "quantum_impact_depletion.png"))
    plot_scope_defense(scope_defense, os.path.join(args.output, "quantum_impact_scope_defense.png"))
    plot_keyrate(keyrate, os.path.join(args.output, "quantum_impact_keyrate.png"))
    plot_qan_cost(qan_cost, os.path.join(args.output, "quantum_impact_qan_cost.png"))

    print(f"Wrote: {depletion_csv}")
    print(f"Wrote: {scope_defense_csv}")
    print(f"Wrote: {keyrate_csv}")
    print(f"Wrote: {qan_cost_csv}")
    print(f"Wrote plots under: {args.output}")


if __name__ == "__main__":
    main()
