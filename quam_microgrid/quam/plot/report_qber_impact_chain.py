#!/usr/bin/env python3
"""
report_qber_impact_chain.py
---------------------------------

Phase 2 artifact: quantify the "QBER impact chain" promised in the abstract.

Goal: produce evidence for the causal story:
  QBER ↑ -> secret fraction ↓ -> effective key refill ↓ / abort ↑ ->
  key pool ↓ -> key wait ↑ / dropped_no_keys ↑ -> deadline misses ↑ ->
  control quality ↓ -> load shedding/curtailment ↑ -> EENS ↑

This script is intentionally *post-processing only* (no model changes).
It consumes the logs already emitted by run_full_matrix/finalmain:
  - outputs/<tag>/trial_*/timeseries/quantum_*.csv
  - outputs/<tag>/trial_*/messages/messages_*.csv
  - outputs/<tag>/trial_*/energy/energy_*.csv
  - outputs/<tag>/trial_*/summary/summary.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class RunKey:
    scenario: str
    topology: str
    seed: int


def _trial_paths(trial_dir: str, key: RunKey) -> Dict[str, str]:
    return {
        "messages": os.path.join(
            trial_dir, "messages", f"messages_{key.scenario}_{key.topology}_seed{key.seed}.csv"
        ),
        "energy": os.path.join(
            trial_dir, "energy", f"energy_{key.scenario}_{key.topology}_seed{key.seed}.csv"
        ),
        "timeseries": os.path.join(
            trial_dir, "timeseries", f"quantum_{key.scenario}_{key.topology}_seed{key.seed}.csv"
        ),
    }


def _attack_from_scenario(s: str) -> str:
    if s == "baseline":
        return "baseline"
    if "_def_" in s:
        return s.split("_def_", 1)[0]
    return s


def _status_bucket(status: str) -> str:
    s = str(status or "")
    if s.startswith("delivered"):
        return "delivered"
    if "no_keys" in s:
        return "dropped_no_keys"
    if "deadline" in s:
        return "dropped_deadline"
    if s.startswith("dropped"):
        return "dropped_other"
    return "unknown"


def _contiguous_ranges(xs: List[float], flags: List[bool]) -> List[Tuple[float, float]]:
    """Return x-range segments where flags are True (for attack-window shading)."""
    if not xs or not flags or len(xs) != len(flags):
        return []
    out: List[Tuple[float, float]] = []
    start: Optional[float] = None
    prev_x: Optional[float] = None
    for x, f in zip(xs, flags):
        if f and start is None:
            start = x
        if not f and start is not None:
            # Close segment at previous sample.
            out.append((start, prev_x if prev_x is not None else x))
            start = None
        prev_x = x
    if start is not None:
        out.append((start, xs[-1]))
    # Expand slightly to make bands visible.
    return [(a, b) for (a, b) in out if b >= a]


def _bin_time(df: pd.DataFrame, t_col: str, bin_s: int) -> pd.DataFrame:
    out = df.copy()
    out["t_bin_s"] = (out[t_col].astype(int) // int(bin_s)) * int(bin_s)
    return out


def _aggregate_timeseries(df_ts: pd.DataFrame, bin_s: int) -> pd.DataFrame:
    if df_ts.empty:
        return df_ts
    df = _bin_time(df_ts, "t_s", bin_s)
    # per-bin aggregation across edges
    agg = (
        df.groupby("t_bin_s", dropna=False)
        .agg(
            qber_mean=("qber", "mean"),
            qber_p95=("qber", lambda s: float(s.quantile(0.95)) if len(s) else float("nan")),
            secret_fraction_mean=("secret_fraction", "mean"),
            secret_fraction_p05=("secret_fraction", lambda s: float(s.quantile(0.05)) if len(s) else float("nan")),
            fidelity_min=("fidelity", "min"),
            pool_total_bits=("pool_level", "sum"),
            pool_min_bits=("pool_level", "min"),
            abort_active_frac=("abort_active", "mean"),
            intrusion_alert_frac=("intrusion_alert", "mean"),
            is_attack_window=("is_attack", "max"),
        )
        .reset_index()
        .sort_values("t_bin_s")
    )
    return agg


def _aggregate_messages(df_msg: pd.DataFrame, bin_s: int) -> pd.DataFrame:
    if df_msg.empty:
        return df_msg

    # Only authenticated traffic consumes QKD resources.
    df = df_msg.copy()
    df["created_s"] = (df["created_ms"].astype(int) // 1000).astype(int)
    df = _bin_time(df, "created_s", bin_s)
    df["bucket"] = df["status"].map(_status_bucket)

    # Baseline message logs can be large; keep aggregation simple and robust.
    def _ratio(n: float, d: float) -> float:
        return float(n) / float(d) if d else 0.0

    grouped = []
    for t_bin, g in df.groupby("t_bin_s", dropna=False):
        auth = g[g["requires_auth"] == 1]
        tot = len(auth)
        delivered = int((auth["bucket"] == "delivered").sum())
        drop_no_keys = int((auth["bucket"] == "dropped_no_keys").sum())
        drop_deadline = int((auth["bucket"] == "dropped_deadline").sum())
        drop_other = int(((auth["bucket"] == "dropped_other") | (auth["bucket"] == "unknown")).sum())

        delivered_wait = auth[(auth["bucket"] == "delivered")]["key_wait_ms"]
        delivered_wait_mean = float(delivered_wait.mean()) if len(delivered_wait) else float("nan")

        spent = auth[auth["bucket"] == "delivered"]["key_bits_spent_total"]
        spent_sum = float(spent.fillna(0.0).sum()) if "key_bits_spent_total" in auth.columns else float("nan")

        grouped.append(
            {
                "t_bin_s": int(t_bin),
                "auth_msgs_total": int(tot),
                "auth_delivered_ratio": _ratio(delivered, tot),
                "auth_dropped_no_keys_ratio": _ratio(drop_no_keys, tot),
                "auth_dropped_deadline_ratio": _ratio(drop_deadline, tot),
                "auth_dropped_other_ratio": _ratio(drop_other, tot),
                "auth_key_wait_mean_ms": delivered_wait_mean,
                "auth_key_bits_spent_sum": spent_sum,
            }
        )
    return pd.DataFrame(grouped).sort_values("t_bin_s")


def _aggregate_energy(df_en: pd.DataFrame, bin_s: int) -> pd.DataFrame:
    if df_en.empty:
        return df_en
    df = _bin_time(df_en, "t_s", bin_s)
    df["curtailed_kw"] = (df["total_load_kw"] - df["served_kw"]).clip(lower=0.0)
    agg = (
        df.groupby("t_bin_s", dropna=False)
        .agg(
            shed_frac_mean=("shed_frac", "mean"),
            shed_frac_max=("shed_frac", "max"),
            curtailed_kw_mean=("curtailed_kw", "mean"),
            unserved_critical_kw_sum=("unserved_critical_kw", "sum"),
            eens_cum_kwh_sum=("eens_cumulative_kwh", "sum"),
            eens_critical_cum_kwh_sum=("eens_critical_cumulative_kwh", "sum"),
            is_attack_window=("is_attack_window", "max"),
        )
        .reset_index()
        .sort_values("t_bin_s")
    )
    return agg


def _merge_chain(ts: pd.DataFrame, msg: pd.DataFrame, en: pd.DataFrame) -> pd.DataFrame:
    out = ts.copy()
    if not msg.empty:
        out = out.merge(msg, on="t_bin_s", how="outer")
    if not en.empty:
        out = out.merge(en, on="t_bin_s", how="outer")
    out = out.sort_values("t_bin_s")
    return out


def plot_chain(df: pd.DataFrame, *, title: str, out_png: str) -> None:
    if df.empty:
        return

    df = df.copy()
    df["t_h"] = df["t_bin_s"].astype(float) / 3600.0

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)

    # Attack window shading preference: energy -> timeseries -> fallback none
    attack_flag_col = None
    for c in ("is_attack_window", "is_attack_window_x", "is_attack_window_y", "is_attack_window_ts", "is_attack_window_en"):
        if c in df.columns:
            attack_flag_col = c
            break
    if attack_flag_col is None and "is_attack_window" in df.columns:
        attack_flag_col = "is_attack_window"
    if attack_flag_col is None and "is_attack_window" not in df.columns and "is_attack_window" in df:
        attack_flag_col = "is_attack_window"

    flags = (df.get("is_attack_window", df.get("is_attack_window_x", df.get("is_attack_window_y", pd.Series([0] * len(df))))) > 0).tolist()
    spans = _contiguous_ranges(df["t_h"].tolist(), [bool(x) for x in flags])

    def _shade(ax):
        for a, b in spans:
            ax.axvspan(a, b, color="red", alpha=0.12, linewidth=0)

    # 1) QBER
    ax = axes[0, 0]
    if "qber_mean" in df:
        ax.plot(df["t_h"], df["qber_mean"], label="QBER mean", linewidth=1.8)
    if "qber_p95" in df:
        ax.plot(df["t_h"], df["qber_p95"], label="QBER p95", linewidth=1.2, alpha=0.8)
    ax.axhline(0.11, color="k", linestyle="--", linewidth=1.0, alpha=0.5, label="Abort (11%)")
    ax.axhline(0.09, color="k", linestyle=":", linewidth=1.0, alpha=0.5, label="Recover (9%)")
    ax.set_title("QBER")
    ax.set_ylabel("QBER")
    ax.grid(True, alpha=0.25)
    _shade(ax)
    ax.legend(loc="best", frameon=True)

    # 2) Secret fraction
    ax = axes[0, 1]
    if "secret_fraction_mean" in df:
        ax.plot(df["t_h"], df["secret_fraction_mean"], label="secret_fraction mean", linewidth=1.8)
    if "secret_fraction_p05" in df:
        ax.plot(df["t_h"], df["secret_fraction_p05"], label="secret_fraction p05", linewidth=1.2, alpha=0.8)
    ax.set_title("Secret Fraction")
    ax.set_ylabel("r(QBER)")
    ax.grid(True, alpha=0.25)
    _shade(ax)
    ax.legend(loc="best", frameon=True)

    # 3) Key pool
    ax = axes[1, 0]
    if "pool_total_bits" in df:
        ax.plot(df["t_h"], df["pool_total_bits"], label="pool total (bits)", linewidth=1.8)
    if "pool_min_bits" in df:
        ax.plot(df["t_h"], df["pool_min_bits"], label="pool min (bits)", linewidth=1.2, alpha=0.8)
    ax.set_title("Key Pool")
    ax.set_ylabel("bits")
    ax.grid(True, alpha=0.25)
    _shade(ax)
    ax.legend(loc="best", frameon=True)

    # 4) Key wait + dropped no-keys
    ax = axes[1, 1]
    if "auth_key_wait_mean_ms" in df:
        ax.plot(df["t_h"], df["auth_key_wait_mean_ms"], label="key_wait mean (ms)", linewidth=1.8)
    ax2 = ax.twinx()
    if "auth_dropped_no_keys_ratio" in df:
        ax2.plot(df["t_h"], df["auth_dropped_no_keys_ratio"], label="dropped_no_keys ratio", linewidth=1.5, color="tab:orange")
        ax2.set_ylabel("ratio")
    ax.set_title("Key Delay / No-Keys Drops (Auth)")
    ax.set_ylabel("ms")
    ax.grid(True, alpha=0.25)
    _shade(ax)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", frameon=True)

    # 5) Delivery / deadline misses
    ax = axes[2, 0]
    if "auth_delivered_ratio" in df:
        ax.plot(df["t_h"], df["auth_delivered_ratio"], label="delivered ratio (auth)", linewidth=1.8)
    if "auth_dropped_deadline_ratio" in df:
        ax.plot(df["t_h"], df["auth_dropped_deadline_ratio"], label="deadline miss ratio (auth)", linewidth=1.5)
    ax.set_title("Delivery / Deadline Miss (Auth)")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("ratio")
    ax.grid(True, alpha=0.25)
    _shade(ax)
    ax.legend(loc="best", frameon=True)

    # 6) Energy impact
    ax = axes[2, 1]
    if "shed_frac_mean" in df:
        ax.plot(df["t_h"], df["shed_frac_mean"] * 100.0, label="shed mean (%)", linewidth=1.8)
    ax2 = ax.twinx()
    if "eens_cum_kwh_sum" in df:
        ax2.plot(df["t_h"], df["eens_cum_kwh_sum"], label="EENS cumulative (kWh)", linewidth=1.5, color="tab:red")
        ax2.set_ylabel("kWh")
    ax.set_title("Operational Impact")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("%")
    ax.grid(True, alpha=0.25)
    _shade(ax)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", frameon=True)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Report: QBER impact chain (Phase 2)")
    ap.add_argument("--trial", required=True, help="Trial directory (e.g., outputs/tag/trial_1)")
    ap.add_argument("--scenarios", nargs="*", default=None, help="Scenario names to include (default: baseline + quantum*)")
    ap.add_argument("--topology", default=None, help="Optional topology filter")
    ap.add_argument("--seed", type=int, default=None, help="Optional seed filter")
    ap.add_argument("--bin_s", type=int, default=300, help="Bin size in seconds (default 300 = 5 min)")
    ap.add_argument("--output", required=True, help="Output folder for report artifacts")
    args = ap.parse_args()

    summary_path = os.path.join(args.trial, "summary", "summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"missing summary.csv at: {summary_path}")

    os.makedirs(args.output, exist_ok=True)
    df_sum = pd.read_csv(summary_path, low_memory=False)

    if args.topology:
        df_sum = df_sum[df_sum["topology"].astype(str) == str(args.topology)]
    if args.seed is not None:
        df_sum = df_sum[df_sum["seed"].astype(int) == int(args.seed)]

    if args.scenarios:
        allow = set(str(s) for s in args.scenarios)
        df_sum = df_sum[df_sum["scenario"].astype(str).isin(allow)]
    else:
        # Default focus: baseline + any scenario containing quantum disturbance.
        df_sum = df_sum[
            (df_sum["scenario"].astype(str) == "baseline")
            | (df_sum["scenario"].astype(str).str.contains("quantum"))
        ]

    rows = []
    for _, r in df_sum.iterrows():
        key = RunKey(scenario=str(r["scenario"]), topology=str(r["topology"]), seed=int(r["seed"]))
        paths = _trial_paths(args.trial, key)
        if not os.path.exists(paths["timeseries"]):
            continue

        df_ts = pd.read_csv(paths["timeseries"], low_memory=False)
        df_msg = pd.read_csv(paths["messages"], low_memory=False) if os.path.exists(paths["messages"]) else pd.DataFrame()
        df_en = pd.read_csv(paths["energy"], low_memory=False) if os.path.exists(paths["energy"]) else pd.DataFrame()

        ts = _aggregate_timeseries(df_ts, args.bin_s)
        msg = _aggregate_messages(df_msg, args.bin_s) if not df_msg.empty else pd.DataFrame()
        en = _aggregate_energy(df_en, args.bin_s) if not df_en.empty else pd.DataFrame()
        chain = _merge_chain(ts, msg, en)

        out_csv = os.path.join(
            args.output, f"qber_chain_{key.scenario}_{key.topology}_seed{key.seed}.csv"
        )
        chain.to_csv(out_csv, index=False)

        out_png = os.path.join(
            args.output, f"qber_chain_{key.scenario}_{key.topology}_seed{key.seed}.png"
        )
        plot_chain(
            chain,
            title=f"QBER Impact Chain: {key.scenario} ({key.topology}, seed={key.seed})",
            out_png=out_png,
        )
        rows.append({"scenario": key.scenario, "topology": key.topology, "seed": key.seed, "csv": out_csv, "png": out_png})

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(args.output, "qber_chain_index.csv"), index=False)
        print(f"Wrote report under: {args.output}")
    else:
        print("No matching runs found (missing timeseries files?)")


if __name__ == "__main__":
    main()

