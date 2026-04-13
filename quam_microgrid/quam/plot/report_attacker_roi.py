#!/usr/bin/env python3
"""
report_attacker_roi.py
-----------------------

Phase 2 artifact: attacker capability ROI plot.

We want a defensible link between deanonymization capability and downstream
attack success. This is most meaningful for attacks that *use* deanon output,
e.g. deanon-guided key exhaustion.

This script consumes summary.csv outputs and plots:
  x = deanonymization accuracy (non-abstain)
  y = attack success metrics (e.g., dropped_no_keys_ratio, EENS)
with stratification by attacker scope and defense mode.
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
        frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        raise FileNotFoundError("No summary.csv files found")
    return pd.concat(frames, ignore_index=True)


def _attack_from_scenario(s: str) -> str:
    if s == "baseline":
        return "baseline"
    if "_def_" in s:
        return s.split("_def_", 1)[0]
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Report: attacker ROI (deanon acc vs attack success)")
    ap.add_argument("--summaries", nargs="+", required=True, help="summary.csv files or summary/ dirs")
    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--attack", default="exhaust", help="attack family to plot (exhaust or all_attacks)")
    ap.add_argument("--exhaust_strategy", default="deanon_guided", help="filter exhaust_strategy")
    ap.add_argument("--x_metric", default="deanon_top1_acc_non_abstain", help="x-axis metric")
    ap.add_argument("--y_metric", default="dropped_no_keys_ratio", help="y-axis metric")
    ap.add_argument("--y2_metric", default="eens_total_kwh", help="secondary y metric (2nd plot)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = _load_csvs(args.summaries)
    df["attack"] = df["scenario"].astype(str).map(_attack_from_scenario)

    # Focus on runs where deanon output plausibly matters.
    work = df[df["attack"].isin([args.attack])].copy()
    if "exhaust_strategy" in work.columns:
        work = work[work["exhaust_strategy"].astype(str) == str(args.exhaust_strategy)]

    # Basic hygiene: require x/y present and finite.
    for c in (args.x_metric, args.y_metric, args.y2_metric, "attacker_scope", "defense_mode"):
        if c not in work.columns:
            work[c] = float("nan")
    work = work.dropna(subset=[args.x_metric, args.y_metric])
    if work.empty:
        raise RuntimeError("No rows left after filtering; check attack/exhaust_strategy/x_metric/y_metric.")

    out_points = os.path.join(args.output, "attacker_roi_points.csv")
    keep_cols = [
        "scenario",
        "attack",
        "topology",
        "seed",
        "attacker_scope",
        "defense_mode",
        "exhaust_strategy",
        args.x_metric,
        args.y_metric,
        args.y2_metric,
    ]
    keep_cols = [c for c in keep_cols if c in work.columns]
    work[keep_cols].to_csv(out_points, index=False)

    scopes = sorted(str(s) for s in work["attacker_scope"].dropna().unique())
    defenses = sorted(str(d) for d in work["defense_mode"].dropna().unique())

    # Visual encoding: color by attacker scope; marker by defense.
    colors = {s: c for s, c in zip(scopes, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"])}
    markers = {d: m for d, m in zip(defenses, ["o", "s", "D", "^", "v", "P", "X"])}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def _scatter(ax, y_metric: str, title: str):
        for scope in scopes:
            for defense in defenses:
                dd = work[(work["attacker_scope"].astype(str) == scope) & (work["defense_mode"].astype(str) == defense)]
                if dd.empty:
                    continue
                ax.scatter(
                    dd[args.x_metric],
                    dd[y_metric],
                    s=70,
                    alpha=0.80,
                    color=colors.get(scope, "tab:blue"),
                    marker=markers.get(defense, "o"),
                    label=f"{scope}/{defense}",
                    edgecolors="none",
                )
        ax.set_title(title)
        ax.set_xlabel(args.x_metric)
        ax.set_ylabel(y_metric)
        ax.grid(True, alpha=0.25)

    _scatter(axes[0], args.y_metric, f"ROI: {args.attack} ({args.exhaust_strategy})")
    _scatter(axes[1], args.y2_metric, "Operational Impact")

    # De-duplicate legend entries (scope/defense pairs can be many)
    handles, labels = axes[0].get_legend_handles_labels()
    dedup = {}
    for h, l in zip(handles, labels):
        dedup[l] = h
    axes[0].legend(dedup.values(), dedup.keys(), loc="best", frameon=True, fontsize=8)

    fig.tight_layout()
    out_png = os.path.join(args.output, "attacker_roi.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"Wrote: {out_points}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()

