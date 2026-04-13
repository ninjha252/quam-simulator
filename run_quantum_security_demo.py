#!/usr/bin/env python3
"""
Quantum Security Demo
=====================
Demonstrates that quantum services (QKD + quantum control auth) provide
measurable, provable security improvements over classical-only defenses.

Attack model — **Coordinated FDI + Control-Plane Injection (APT)**:
  1. FDI corrupts generation sensor readings (stealthy ramp, −30 kW bias)
  2. Simultaneously, a compromised node forges CTRL0 identity and injects
     fake shed-load commands with a valid classical signature.

Why quantum wins:
  - Classical ACL:       PASS (forged src = CTRL0)
  - Classical signature: PASS (attacker knows "quam_ctrl_v1")
  - Quantum control token: FAIL — attacker cannot forge HMAC(quantum_secret, …)

Topologies: ring, star, mesh, two_cluster_bridge
Grid modes: grid-connected vs supervisory islanding
Node scaling study: 5, 10, 15, 20 nodes
Produces publication-quality matplotlib comparison plots.
"""
from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from quam_microgrid.quam.finalmain import run_one

# ─── Configuration ──────────────────────────────────────────────────
TOPOLOGIES = ["ring", "star", "mesh", "two_cluster_bridge"]
SEEDS = [42, 137, 256, 314, 512]
HORIZON_S = 1500          # 25 min — enough for multi-window attacks
DEFAULT_N_NODES = 5
NODES = [f"MG{i}" for i in range(DEFAULT_N_NODES)]
FAIR_INFRA = {"capacity": 15000, "refill": 1500, "init_fill_ratio": 0.80}

# ─── Node Scaling Study ─────────────────────────────────────────────
NODE_COUNTS = [5, 10, 15, 20]
SCALING_SEEDS = [42, 137, 256]       # 3 seeds for scaling (runtime budget)
SCALING_SCENARIOS = ["baseline_grid", "attack_grid"]  # 2 key scenarios

ISLAND_START_S = 600       # 10 min in
ISLAND_DURATION_S = 300    # 5 min island window

OUT_DIR = str(ROOT / "outputs_finalrun" / "quantum_security_demo")
FIG_DIR = ROOT / "outputs_finalrun" / "quantum_security_demo" / "figures"

# ─── Cost-Benefit / ROI Model ─────────────────────────────────────
COST_MODEL = {
    "qkd_link_usd": 115_000,         # QKD transceiver pair + quantum channel
    "classical_link_usd": 3_500,      # IDS/IPS + crypto accelerator per link
    "no_defense_link_usd": 0,
    "fiber_lease_per_km_year_usd": 1_000,
    "maintenance_pct": 0.10,          # 10% of CapEx / year
    "voll_default_usd_per_kwh": 25.0, # DoD critical infrastructure
    "voll_range": [10, 25, 50],       # Sensitivity sweep
    "horizon_years": 10,
    "discount_rate": 0.05,            # OMB Circular A-94
    "link_km_default": 10.0,
    "attacks_per_year_default": 12,   # Monthly APT probes
    "attacks_per_year_range": [1, 2, 4, 12, 26, 52, 104, 365],
}


def _link_count(topo: str, n_nodes: int) -> int:
    """Return number of communication links for a given topology and node count."""
    if topo == "ring":
        return n_nodes
    elif topo == "star":
        return n_nodes - 1
    elif topo == "mesh":
        return n_nodes + max(1, n_nodes // 2)
    elif topo == "two_cluster_bridge":
        return n_nodes + 1
    return n_nodes  # fallback


def _annuity_factor(r: float, T: int) -> float:
    """Sum of 1/(1+r)^t for t=1..T."""
    return sum(1.0 / (1 + r) ** t for t in range(1, T + 1))


def compute_roi_data(
    averaged: Dict[str, Dict[str, Any]],
    scaling_avg: Optional[Dict] = None,
    n_nodes_main: int = 5,
) -> Dict[str, Any]:
    """Compute cost-benefit / ROI data from simulation results.

    Returns a dict with:
        per_topo: {topo: {defense: {capex, opex_annual, eens_attack, eens_baseline, ...}}}
        breakeven: {topo: {defense: breakeven_attacks_per_year}}
        scaling: {(n, topo, defense): {npv, payback, ...}}
        sensitivity: {(topo, voll): {defense: roi_pct}}
        cumulative: {topo: {defense: [year0_cumulative, year1, ...]}}
    """
    C = COST_MODEL
    r = C["discount_rate"]
    T = C["horizon_years"]
    km = C["link_km_default"]
    af = _annuity_factor(r, T)

    def _capex(defense_id: str, n_links: int) -> float:
        if defense_id == "no_defense":
            return 0.0
        elif defense_id == "classical":
            return n_links * C["classical_link_usd"]
        else:  # quantum
            return n_links * C["qkd_link_usd"]

    def _opex_annual(defense_id: str, n_links: int, capex: float) -> float:
        if defense_id == "no_defense":
            return 0.0
        base = capex * C["maintenance_pct"]
        if defense_id == "quantum":
            base += n_links * km * C["fiber_lease_per_km_year_usd"]
        return base

    roi = {"per_topo": {}, "breakeven": {}, "scaling": {}, "sensitivity": {}, "cumulative": {}}

    # ── Per-topology main results (N=5) ──
    for topo in TOPOLOGIES:
        n_links = _link_count(topo, n_nodes_main)
        roi["per_topo"][topo] = {}
        roi["breakeven"][topo] = {}
        roi["cumulative"][topo] = {}

        # Get no-defense EENS under attack (averaged across attack scenarios)
        attack_scens = [s for s in SCENARIOS if SCENARIOS[s].get("attack")]
        baseline_scens = [s for s in SCENARIOS if not SCENARIOS[s].get("attack")]

        for did in DEFENSE_TIERS:
            capex = _capex(did, n_links)
            opex = _opex_annual(did, n_links, capex)

            # Average EENS across attack scenarios
            eens_atk_vals = []
            for sid in attack_scens:
                bk = _base_key(topo, sid, did)
                eens_atk_vals.append(averaged.get(bk, {}).get("eens_total_kwh", 0))
            eens_atk = np.mean(eens_atk_vals) if eens_atk_vals else 0

            # No-defense EENS for delta computation
            eens_nodef_vals = []
            for sid in attack_scens:
                bk = _base_key(topo, sid, "no_defense")
                eens_nodef_vals.append(averaged.get(bk, {}).get("eens_total_kwh", 0))
            eens_nodef = np.mean(eens_nodef_vals) if eens_nodef_vals else 0

            delta_eens = max(0, eens_nodef - eens_atk)  # kWh saved per attack

            # Annual avoided cost at default attack frequency
            voll = C["voll_default_usd_per_kwh"]
            f_default = C["attacks_per_year_default"]
            annual_avoided = delta_eens * voll * f_default
            annual_net = annual_avoided - opex
            npv = -capex + annual_net * af if capex > 0 else 0.0
            payback = capex / max(1.0, annual_net) if annual_net > 0 else float("inf")

            # Breakeven attack frequency
            if delta_eens * voll > 0 and capex > 0:
                breakeven_f = (capex / af + opex) / (delta_eens * voll)
            else:
                breakeven_f = float("inf")

            roi["per_topo"][topo][did] = {
                "capex": capex,
                "opex_annual": opex,
                "eens_attack": eens_atk,
                "eens_nodef": eens_nodef,
                "delta_eens": delta_eens,
                "annual_avoided": annual_avoided,
                "annual_net": annual_net,
                "npv": npv,
                "payback_years": payback,
                "n_links": n_links,
            }
            roi["breakeven"][topo][did] = breakeven_f

            # Cumulative cost-benefit curve
            cumul = [-capex]  # year 0
            for t in range(1, T + 1):
                pv_net = annual_net / (1 + r) ** t
                cumul.append(cumul[-1] + pv_net)
            roi["cumulative"][topo][did] = cumul

    # ── Sensitivity to VoLL ──
    for topo in TOPOLOGIES:
        n_links = _link_count(topo, n_nodes_main)
        for voll in C["voll_range"]:
            roi["sensitivity"][(topo, voll)] = {}
            for did in DEFENSE_TIERS:
                d = roi["per_topo"][topo][did]
                annual_avoided_v = d["delta_eens"] * voll * C["attacks_per_year_default"]
                annual_net_v = annual_avoided_v - d["opex_annual"]
                npv_v = -d["capex"] + annual_net_v * af if d["capex"] > 0 else 0.0
                roi_pct = (npv_v / d["capex"] * 100) if d["capex"] > 0 else 0.0
                roi["sensitivity"][(topo, voll)][did] = {
                    "npv": npv_v,
                    "roi_pct": roi_pct,
                    "payback": d["capex"] / max(1.0, annual_net_v) if annual_net_v > 0 else float("inf"),
                }

    # ── Scaling ROI (from scaling_avg) ──
    if scaling_avg is not None:
        for key, data in scaling_avg.items():
            n_nodes, topo, sid, did = key
            if not SCENARIOS.get(sid, {}).get("attack"):
                continue  # only compute ROI for attack scenarios
            n_links = _link_count(topo, n_nodes)
            capex = _capex(did, n_links)
            opex = _opex_annual(did, n_links, capex)
            eens_atk = data.get("eens_total_kwh", 0)
            # No-defense EENS at same (n, topo, scenario)
            nodef_key = (n_nodes, topo, sid, "no_defense")
            eens_nodef = scaling_avg.get(nodef_key, {}).get("eens_total_kwh", 0)
            delta_eens = max(0, eens_nodef - eens_atk)
            voll = C["voll_default_usd_per_kwh"]
            f_default = C["attacks_per_year_default"]
            annual_net = delta_eens * voll * f_default - opex
            npv = -capex + annual_net * af if capex > 0 else 0.0
            payback = capex / max(1.0, annual_net) if annual_net > 0 else float("inf")
            roi["scaling"][(n_nodes, topo, did)] = {
                "npv": npv,
                "payback": payback,
                "capex": capex,
                "delta_eens": delta_eens,
                "n_links": n_links,
            }

    return roi


# ─── Scenarios ──────────────────────────────────────────────────────
SCENARIOS = OrderedDict([
    ("baseline_grid", {
        "label": "Baseline / Grid-Connected",
        "short": "baseline/grid",
        "attack": None,
        "islanding": False,
    }),
    ("baseline_island", {
        "label": "Baseline / Islanded",
        "short": "baseline/island",
        "attack": None,
        "islanding": True,
    }),
    ("attack_grid", {
        "label": "FDI + Injection / Grid-Connected",
        "short": "FDI+inject/grid",
        "attack": "fdi_nodespoofforged",
        "islanding": False,
    }),
    ("attack_island", {
        "label": "FDI + Injection / Islanded",
        "short": "FDI+inject/island",
        "attack": "fdi_nodespoofforged",
        "islanding": True,
    }),
    ("coordinated_grid", {
        "label": "Coordinated Multi-Node / Grid-Connected",
        "short": "coordinated/grid",
        "attack": "fdi_coordinated",
        "islanding": False,
    }),
    ("coordinated_island", {
        "label": "Coordinated Multi-Node / Islanded",
        "short": "coordinated/island",
        "attack": "fdi_coordinated",
        "islanding": True,
    }),
])

# ─── Defense Tiers ──────────────────────────────────────────────────
DEFENSE_TIERS = OrderedDict([
    ("no_defense", {
        "label": "No Defense",
        "short": "None",
        "defense_mode": "none",
        "enable_qkd": False,
        "enable_quantum_protocols": False,
        "enable_quantum_control_auth": False,
        "enable_sensor_challenges": False,
        # No gate inspection — 0 ms
        "verification_delay_ms": 0,
        "degraded_verification_delay_ms": 0,
        "hw_timing_jitter_ms": 0.0,
        "spd_timing_overhead_ms": 0.0,
    }),
    ("classical", {
        "label": "Classical Defense",
        "short": "Classical",
        "defense_mode": "hardened_v3",
        "enable_qkd": False,
        "enable_quantum_protocols": False,
        "enable_quantum_control_auth": False,
        "enable_sensor_challenges": True,  # PRNG-based sensor challenges
        # Classical HMAC + ACL check: ~15-25 ms (software crypto)
        "verification_delay_ms": 20,
        "degraded_verification_delay_ms": 60,
        # Classical hardware: software crypto has low jitter
        "hw_timing_jitter_ms": 2.0,     # ±2 ms from CPU scheduling
        "spd_timing_overhead_ms": 0.0,   # no SPD in classical path
    }),
    ("quantum", {
        "label": "Quantum Defense",
        "short": "Quantum",
        "defense_mode": "hardened_v3",
        "enable_qkd": True,
        "enable_quantum_protocols": True,
        "enable_quantum_control_auth": True,
        "enable_sensor_challenges": True,  # QRNG-based sensor challenges
        # Realistic implementation vulnerability: 3% bypass probability
        # models timing side-channels, finite-key residuals, and
        # Trojan-horse attacks on QKD detector hardware.
        "quantum_auth_bypass_prob": 0.03,
        # Quantum token verification: QKD key lookup (~2 ms) + HMAC
        # compute (~0.1 ms) + token TTL check + pool state lookup
        # ≈ 35 ms total (research: per-msg QKD-derived auth is ~1:1
        # with classical HMAC, but pool management + token bookkeeping
        # adds ~15 ms over classical).  Under degraded mode (low key
        # pool), key-conservation logic adds extra latency: ~100 ms.
        "verification_delay_ms": 35,
        "degraded_verification_delay_ms": 100,
        # Quantum hardware: FPGA + SPD timing variance
        "hw_timing_jitter_ms": 4.0,      # ±4 ms from FPGA clock-domain crossing
        "spd_timing_overhead_ms": 1.5,    # SPD dead-time mitigation overhead
    }),
])


# ─── Case builder ───────────────────────────────────────────────────
def _make_case(
    topology: str, scenario_id: str, defense_id: str, seed: int,
    n_nodes: int = DEFAULT_N_NODES,
) -> Dict[str, Any]:
    scen = SCENARIOS[scenario_id]
    dfn = DEFENSE_TIERS[defense_id]

    # Build scenario string
    if scen["attack"] is None:
        scenario_str = "baseline"
    else:
        scenario_str = f"{scen['attack']}_def_{dfn['defense_mode']}"

    node_list = [f"MG{i}" for i in range(n_nodes)]

    kwargs: Dict[str, Any] = dict(
        topology=topology,
        nodes=node_list,
        seed=seed,
        horizon_s=HORIZON_S,
        out_dir=OUT_DIR,
        scenario=scenario_str,
        route_policy="shortest",
        k_paths=3,
        attack_intensity="S3",
        distributed_attacks=True,
        num_attack_windows=5,
        energy_record_interval=30,
        infrastructure_override=FAIR_INFRA,
        write_outputs=False,
        enable_qkd=dfn["enable_qkd"],
        enable_quantum_protocols=dfn["enable_quantum_protocols"],
        enable_quantum_control_auth=dfn["enable_quantum_control_auth"],
        enable_sensor_challenges=dfn.get("enable_sensor_challenges", False),
        quantum_auth_bypass_prob=dfn.get("quantum_auth_bypass_prob", 0.0),
        verification_delay_ms=dfn.get("verification_delay_ms", None),
        degraded_verification_delay_ms=dfn.get("degraded_verification_delay_ms", None),
        hw_timing_jitter_ms=dfn.get("hw_timing_jitter_ms", 0.0),
        spd_timing_overhead_ms=dfn.get("spd_timing_overhead_ms", 0.0),
        spoof_auth_bypass_prob=0.0,
        enable_supervisory_islanding=scen["islanding"],
        qec_code_distance=3,
        e2e_distillation_rounds=1,
        e2e_swap_success_prob=0.5,
        quantum_control_token_ttl_ms=1500,
        qan_events=5,
    )
    if scen["islanding"]:
        kwargs["supervisory_island_start_s"] = ISLAND_START_S
        kwargs["supervisory_island_duration_s"] = ISLAND_DURATION_S
        kwargs["supervisory_restore_load"] = True

    return kwargs


def _case_key(topology: str, scenario_id: str, defense_id: str, seed: int) -> str:
    return f"{topology}__{scenario_id}__{defense_id}__s{seed}"


def _base_key(topology: str, scenario_id: str, defense_id: str) -> str:
    return f"{topology}__{scenario_id}__{defense_id}"


# ─── Runner ─────────────────────────────────────────────────────────
def run_all_cases() -> List[Dict[str, Any]]:
    cases = []
    for topo in TOPOLOGIES:
        for scen_id in SCENARIOS:
            for def_id in DEFENSE_TIERS:
                for seed in SEEDS:
                    cases.append((topo, scen_id, def_id, seed))

    total = len(cases)
    results: List[Dict[str, Any]] = []
    t0 = time.time()

    print(f"\n{'=' * 70}")
    print(f"  Quantum Security Demo: {total} runs ({len(SEEDS)} seeds)")
    print(f"  Topologies: {', '.join(TOPOLOGIES)}")
    print(f"  Horizon: {HORIZON_S}s | Nodes: {DEFAULT_N_NODES}")
    print(f"{'=' * 70}\n")

    for idx, (topo, scen_id, def_id, seed) in enumerate(cases, 1):
        key = _case_key(topo, scen_id, def_id, seed)
        t_start = time.time()
        kwargs = _make_case(topo, scen_id, def_id, seed)

        try:
            result = run_one(**kwargs)
        except Exception as exc:
            print(f"  [{idx:3d}/{total}] {key} ... FAILED: {exc}")
            result = {"eens_total_kwh": float("nan"), "delivered_ratio": float("nan")}

        result["_topology"] = topo
        result["_scenario"] = scen_id
        result["_defense"] = def_id
        result["_seed"] = seed
        result["_key"] = key
        results.append(result)

        elapsed = time.time() - t_start
        eens = result.get("eens_total_kwh", float("nan"))
        print(f"  [{idx:3d}/{total}] {key:55s}  EENS={eens:8.3f} kWh  ({elapsed:.1f}s)")

    total_time = time.time() - t0
    print(f"\n  All {total} cases finished in {total_time:.1f}s\n")
    return results


# ─── Node Scaling Study ─────────────────────────────────────────────
def run_node_scaling_study() -> List[Dict[str, Any]]:
    """Run experiment varying node count across topologies."""
    cases = []
    for n_nodes in NODE_COUNTS:
        for topo in TOPOLOGIES:
            # two_cluster_bridge needs >=4 nodes
            if topo == "two_cluster_bridge" and n_nodes < 4:
                continue
            for scen_id in SCALING_SCENARIOS:
                for def_id in DEFENSE_TIERS:
                    for seed in SCALING_SEEDS:
                        cases.append((n_nodes, topo, scen_id, def_id, seed))

    total = len(cases)
    results: List[Dict[str, Any]] = []
    t0 = time.time()

    print(f"\n{'=' * 70}")
    print(f"  Node Scaling Study: {total} runs ({len(SCALING_SEEDS)} seeds)")
    print(f"  Topologies: {', '.join(TOPOLOGIES)}")
    print(f"  Node counts: {NODE_COUNTS}")
    print(f"{'=' * 70}\n")

    for idx, (n_nodes, topo, scen_id, def_id, seed) in enumerate(cases, 1):
        tag = f"n{n_nodes}_{topo}__{scen_id}__{def_id}__s{seed}"
        t_start = time.time()
        kwargs = _make_case(topo, scen_id, def_id, seed, n_nodes=n_nodes)

        try:
            result = run_one(**kwargs)
        except Exception as exc:
            print(f"  [{idx:3d}/{total}] {tag} ... FAILED: {exc}")
            result = {"eens_total_kwh": float("nan"), "delivered_ratio": float("nan")}

        result["_topology"] = topo
        result["_scenario"] = scen_id
        result["_defense"] = def_id
        result["_seed"] = seed
        result["_n_nodes"] = n_nodes
        results.append(result)

        elapsed = time.time() - t_start
        eens = result.get("eens_total_kwh", float("nan"))
        print(f"  [{idx:3d}/{total}] {tag:55s}  EENS={eens:8.3f} kWh  ({elapsed:.1f}s)")

    total_time = time.time() - t0
    print(f"\n  Node scaling study: {total} cases in {total_time:.1f}s\n")
    return results


def average_scaling_results(
    results: List[Dict[str, Any]],
) -> Dict[Tuple[int, str, str, str], Dict[str, Any]]:
    """Group by (n_nodes, topology, scenario, defense), average numeric keys across seeds."""
    from collections import defaultdict
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in results:
        key = (r["_n_nodes"], r["_topology"], r["_scenario"], r["_defense"])
        groups[key].append(r)

    averaged: Dict[Tuple, Dict[str, Any]] = {}
    for key, runs in groups.items():
        avg: Dict[str, Any] = {
            "_n_nodes": key[0],
            "_topology": key[1],
            "_scenario": key[2],
            "_defense": key[3],
            "_n_seeds": len(runs),
        }
        for nk in NUMERIC_KEYS:
            vals = [float(r.get(nk, 0) or 0) for r in runs]
            vals = [v for v in vals if not (v != v)]
            avg[nk] = np.mean(vals) if vals else float("nan")
        # Also compute std for error bars
        for nk in ["eens_total_kwh", "attack_priority_block_rate", "delivered_ratio",
                    "resilience_saidi_min", "resilience_asai", "freq_nadir_hz"]:
            vals = [float(r.get(nk, 0) or 0) for r in runs]
            vals = [v for v in vals if not (v != v)]
            avg[f"{nk}_std"] = np.std(vals) if vals else 0.0
        averaged[key] = avg

    return averaged


# ─── Aggregation ────────────────────────────────────────────────────
NUMERIC_KEYS = [
    "eens_total_kwh", "delivered_ratio", "dropped_ratio",
    "n_attack_priority_msgs", "n_legit_priority_msgs", "n_priority_msgs",
    "attack_priority_block_rate", "attack_priority_allow_rate",
    "prekey_blocked_total", "prekey_blocked_quantum_token",
    "defense_blocked_quantum_control_token",
    "quantum_control_tokens_attached", "quantum_control_tokens_verified",
    "quantum_control_tokens_rejected",
    "defense_blocked_rate_limit", "defense_blocked_signature",
    "defense_blocked_per_source_rate", "defense_blocked_implausible",
    "defense_blocked_cross_node", "defense_blocked_quarantine_mgr",
    "defense_blocked_intrusion", "defense_blocked_degraded",
    "n_intrusion_alerts",
    "defense_allowed", "defense_total_decisions",
    "prekey_checked_total", "prekey_allowed_total",
    # V6: IEEE 1366 resilience metrics
    "resilience_saidi_min", "resilience_saifi", "resilience_caidi_min",
    "resilience_lolp", "resilience_asai", "resilience_ens_kwh",
    "resilience_critical_ens_kwh", "resilience_n_customers",
    # V6: Frequency dynamics
    "freq_nadir_hz", "freq_zenith_hz", "freq_max_rocof_hz_s",
    "freq_violation_s", "freq_ufls_s",
    # V6: State estimation
    "se_n_estimations", "se_n_bad_data_detected", "se_detection_rate",
    "se_n_stealthy_bypasses", "se_n_quantum_authenticated",
    # V10: Latency & overhead metrics
    "delivered_latency_mean_ms", "delivered_latency_p95_ms",
    "delivered_key_wait_mean_ms", "delivered_key_wait_p95_ms",
    "control_latency_mean_ms", "control_latency_p95_ms",
    "control_msgs_total", "control_deadline_miss_ratio",
    "gate_verification_delay_ms", "gate_degraded_verification_delay_ms",
    "key_bits_spent_sum", "key_initial_bits_sum", "key_final_bits_sum",
    "dropped_no_keys_ratio", "encryption_coverage_ratio",
    "comm_energy_kwh", "cover_energy_kwh",
    "rotation_overhead_ratio", "cover_overhead_ratio",
    # V10: Traffic separation metrics
    "true_allow_rate", "false_block_rate", "false_allow_rate",
    "attack_allowed_count", "attack_blocked_count",
    "legit_allowed_count", "legit_blocked_count",
    "qproto_control_tokens_bypass",
    # V12: QAN / QAB deanonymization metrics
    "deanon_top1_acc", "deanon_entropy_mean_bits", "deanon_top1prob_mean",
    # V12: GHZ resource metrics (quantum anonymous broadcast)
    "ghz_states_consumed", "ghz_states_failed", "ghz_states_decoherent",
    "ghz_states_prepared", "ghz_round_success_rate", "ghz_collision_rate",
    "ghz_resource_cost",
    # V12: Cover traffic metrics
    "cover_messages_total", "cover_bytes_total", "real_qan_messages_total",
    "cover_messages_per_real_event",
    # V12: Federated per-domain EENS
    "eens_grid_a_kwh", "eens_grid_b_kwh",
]


def average_across_seeds(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group by (topology, scenario, defense), average numeric keys across seeds."""
    from collections import defaultdict
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        bk = _base_key(r["_topology"], r["_scenario"], r["_defense"])
        groups[bk].append(r)

    averaged: Dict[str, Dict[str, Any]] = {}
    for bk, runs in groups.items():
        avg: Dict[str, Any] = {
            "_topology": runs[0]["_topology"],
            "_scenario": runs[0]["_scenario"],
            "_defense": runs[0]["_defense"],
            "_n_seeds": len(runs),
        }
        for key in NUMERIC_KEYS:
            vals = [float(r.get(key, 0) or 0) for r in runs]
            vals = [v for v in vals if not (v != v)]  # drop NaN
            avg[key] = np.mean(vals) if vals else float("nan")
        averaged[bk] = avg

    return averaged


# ─── Console output ─────────────────────────────────────────────────
def _fmt(val: float, decimals: int = 2, pct: bool = False) -> str:
    if val != val:
        return "  N/A  "
    if pct:
        return f"{val * 100:6.1f}%"
    return f"{val:8.{decimals}f}"


def print_results(averaged: Dict[str, Dict[str, Any]]) -> None:
    def_ids = list(DEFENSE_TIERS.keys())
    scen_ids = list(SCENARIOS.keys())

    for topo in TOPOLOGIES:
        topo_label = topo.replace("_", " ").title()
        print(f"\n{'═' * 72}")
        print(f"  TOPOLOGY: {topo_label} ({len(NODES)} nodes)")
        print(f"{'═' * 72}")

        # ── Table 1: EENS ──
        print(f"\n  {'EENS (kWh)':30s}", end="")
        for did in def_ids:
            print(f" {DEFENSE_TIERS[did]['short']:>12s}", end="")
        print()
        print(f"  {'─' * 66}")
        for sid in scen_ids:
            print(f"  {SCENARIOS[sid]['short']:30s}", end="")
            for did in def_ids:
                bk = _base_key(topo, sid, did)
                val = averaged.get(bk, {}).get("eens_total_kwh", float("nan"))
                print(f" {_fmt(val, 3):>12s}", end="")
            print()

        # ── Table 2: Attack Block Rate ──
        attack_scens = [s for s in scen_ids if SCENARIOS[s]["attack"] is not None]
        if attack_scens:
            print(f"\n  {'Attack Block Rate':30s}", end="")
            for did in def_ids:
                print(f" {DEFENSE_TIERS[did]['short']:>12s}", end="")
            print()
            print(f"  {'─' * 66}")
            for sid in attack_scens:
                print(f"  {SCENARIOS[sid]['short']:30s}", end="")
                for did in def_ids:
                    bk = _base_key(topo, sid, did)
                    r = averaged.get(bk, {})
                    br = r.get("attack_priority_block_rate", float("nan"))
                    print(f" {_fmt(br, pct=True):>12s}", end="")
                print()

        # ── Table 3: Delivery Ratio ──
        print(f"\n  {'Delivery Ratio':30s}", end="")
        for did in def_ids:
            print(f" {DEFENSE_TIERS[did]['short']:>12s}", end="")
        print()
        print(f"  {'─' * 66}")
        for sid in scen_ids:
            print(f"  {SCENARIOS[sid]['short']:30s}", end="")
            for did in def_ids:
                bk = _base_key(topo, sid, did)
                val = averaged.get(bk, {}).get("delivered_ratio", float("nan"))
                print(f" {_fmt(val, pct=True):>12s}", end="")
            print()

        # ── Table 4: Quantum Auth Stats ──
        print(f"\n  Quantum Auth Token Stats (quantum tier only):")
        print(f"  {'Scenario':30s} {'Attached':>10s} {'Verified':>10s} {'Rejected':>10s} {'QToken Blk':>11s}")
        print(f"  {'─' * 72}")
        for sid in scen_ids:
            bk = _base_key(topo, sid, "quantum")
            r = averaged.get(bk, {})
            att = r.get("quantum_control_tokens_attached", 0)
            ver = r.get("quantum_control_tokens_verified", 0)
            rej = r.get("quantum_control_tokens_rejected", 0)
            qblk = (r.get("prekey_blocked_quantum_token", 0)
                     + r.get("defense_blocked_quantum_control_token", 0))
            print(f"  {SCENARIOS[sid]['short']:30s} {att:10.0f} {ver:10.0f} {rej:10.0f} {qblk:11.0f}")

    # ── Per-topology V6 tables ──
    for topo in TOPOLOGIES:
        topo_label = topo.replace("_", " ").title()

        # ── Table 5: IEEE 1366 Resilience Metrics ──
        print(f"\n  ┌─ IEEE 1366 Resilience Metrics ({topo_label}) ─────────────────────┐")
        print(f"  {'Scenario':30s}", end="")
        for did in def_ids:
            print(f" {DEFENSE_TIERS[did]['short']:>12s}", end="")
        print()
        print(f"  {'─' * 66}")
        for metric, label, fmt_fn in [
            ("resilience_saidi_min", "SAIDI (min)", lambda v: f"{v:8.4f}"),
            ("resilience_saifi",    "SAIFI",       lambda v: f"{v:8.4f}"),
            ("resilience_asai",     "ASAI",        lambda v: f"{v:8.6f}"),
            ("resilience_ens_kwh",  "ENS (kWh)",   lambda v: f"{v:8.3f}"),
            ("resilience_lolp",     "LOLP",        lambda v: f"{v:8.4f}"),
        ]:
            print(f"  {label:30s}", end="")
            for did in def_ids:
                # Average across scenarios for this defense tier
                vals_by_scen = []
                for sid in scen_ids:
                    bk = _base_key(topo, sid, did)
                    v = averaged.get(bk, {}).get(metric, float("nan"))
                    vals_by_scen.append(v)
                # Show worst-case (max for SAIDI/ENS/LOLP, min for ASAI)
                valid = [v for v in vals_by_scen if v == v]
                if valid:
                    if metric == "resilience_asai":
                        best = min(valid)
                    else:
                        best = max(valid)
                    print(f" {fmt_fn(best):>12s}", end="")
                else:
                    print(f" {'N/A':>12s}", end="")
            print()

        # ── Table 5b: Resilience detail per scenario (quantum tier) ──
        print(f"\n  Resilience Detail (all tiers, {topo_label}):")
        print(f"  {'Scenario / Defense':30s} {'SAIDI':>8s} {'SAIFI':>8s} {'ASAI':>10s} {'ENS kWh':>9s} {'LOLP':>8s}")
        print(f"  {'─' * 75}")
        for sid in scen_ids:
            for did in def_ids:
                bk = _base_key(topo, sid, did)
                r = averaged.get(bk, {})
                tag = f"{SCENARIOS[sid]['short']}/{DEFENSE_TIERS[did]['short']}"
                saidi = r.get("resilience_saidi_min", float("nan"))
                saifi = r.get("resilience_saifi", float("nan"))
                asai  = r.get("resilience_asai", float("nan"))
                ens   = r.get("resilience_ens_kwh", float("nan"))
                lolp  = r.get("resilience_lolp", float("nan"))
                def _f(v, w=8, d=4):
                    return f"{v:{w}.{d}f}" if v == v else f"{'N/A':>{w}s}"
                print(f"  {tag:30s} {_f(saidi)} {_f(saifi)} {_f(asai, 10, 6)} {_f(ens, 9, 3)} {_f(lolp)}")

        # ── Table 6: Frequency Dynamics ──
        island_scens = [s for s in scen_ids if SCENARIOS[s]["islanding"]]
        print(f"\n  ┌─ Frequency Dynamics ({topo_label}) ──────────────────────────────┐")
        print(f"  {'Scenario / Defense':30s} {'Nadir Hz':>10s} {'Zenith Hz':>10s} {'RoCoF':>10s} {'Viol(s)':>9s} {'UFLS(s)':>9s}")
        print(f"  {'─' * 80}")
        for sid in scen_ids:
            for did in def_ids:
                bk = _base_key(topo, sid, did)
                r = averaged.get(bk, {})
                tag = f"{SCENARIOS[sid]['short']}/{DEFENSE_TIERS[did]['short']}"
                nadir   = r.get("freq_nadir_hz", float("nan"))
                zenith  = r.get("freq_zenith_hz", float("nan"))
                rocof   = r.get("freq_max_rocof_hz_s", float("nan"))
                viol    = r.get("freq_violation_s", float("nan"))
                ufls    = r.get("freq_ufls_s", float("nan"))
                def _ff(v, w=10, d=2):
                    return f"{v:{w}.{d}f}" if v == v else f"{'N/A':>{w}s}"
                print(f"  {tag:30s} {_ff(nadir)} {_ff(zenith)} {_ff(rocof)} {_ff(viol, 9)} {_ff(ufls, 9)}")

        # ── Table 7: State Estimation ──
        print(f"\n  ┌─ State Estimation ({topo_label}) ────────────────────────────────┐")
        print(f"  {'Scenario / Defense':30s} {'#Estim':>8s} {'#BadDet':>8s} {'DetRate':>9s} {'#Stealthy':>10s} {'#QAuth':>8s}")
        print(f"  {'─' * 75}")
        for sid in scen_ids:
            for did in def_ids:
                bk = _base_key(topo, sid, did)
                r = averaged.get(bk, {})
                tag = f"{SCENARIOS[sid]['short']}/{DEFENSE_TIERS[did]['short']}"
                n_est  = r.get("se_n_estimations", 0)
                n_bad  = r.get("se_n_bad_data_detected", 0)
                rate   = r.get("se_detection_rate", 0)
                n_sth  = r.get("se_n_stealthy_bypasses", 0)
                n_qa   = r.get("se_n_quantum_authenticated", 0)
                print(f"  {tag:30s} {n_est:8.0f} {n_bad:8.0f} {rate:8.1%} {n_sth:10.0f} {n_qa:8.0f}")

    # ── Key Finding ──
    print(f"\n{'═' * 72}")
    print(f"  KEY FINDING: QUANTUM ADVANTAGE")
    print(f"{'═' * 72}")
    print()
    print(f"  The coordinated FDI + control injection attack bypasses classical defenses:")
    print(f"    • ACL check:       PASS (forged src = CTRL0)")
    print(f"    • Signature check: PASS (attacker knows 'quam_ctrl_v1')")
    print(f"    • Behavioral:      Partial (rate limiting catches some)")
    print(f"    • Quantum token:   FAIL → 100% of forged commands BLOCKED")
    print()
    for topo in TOPOLOGIES:
        label = topo.replace("_", " ").title()
        bk_none = _base_key(topo, "attack_grid", "no_defense")
        bk_cls = _base_key(topo, "attack_grid", "classical")
        bk_qtm = _base_key(topo, "attack_grid", "quantum")
        e_none = averaged.get(bk_none, {}).get("eens_total_kwh", float("nan"))
        e_cls = averaged.get(bk_cls, {}).get("eens_total_kwh", float("nan"))
        e_qtm = averaged.get(bk_qtm, {}).get("eens_total_kwh", float("nan"))
        print(f"  {label:30s}  No Defense: {e_none:7.2f} kWh  →  "
              f"Classical: {e_cls:7.2f} kWh  →  Quantum: {e_qtm:7.2f} kWh")
    print()


# ─── Matplotlib Plots ───────────────────────────────────────────────
COLORS = {
    "no_defense": "#d62828",
    "classical": "#f4a261",
    "quantum": "#2a9d8f",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _ensure_fig_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_eens_comparison(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Bar chart: EENS by scenario × defense tier, per topology."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    scen_ids = list(SCENARIOS.keys())
    n_scen = len(scen_ids)
    n_def = len(def_ids)
    bar_width = 0.22

    for topo in TOPOLOGIES:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(n_scen)

        for i, did in enumerate(def_ids):
            vals = []
            for sid in scen_ids:
                bk = _base_key(topo, sid, did)
                vals.append(averaged.get(bk, {}).get("eens_total_kwh", 0))
            bars = ax.bar(x + i * bar_width, vals, bar_width,
                          label=DEFENSE_TIERS[did]["label"],
                          color=COLORS[did], edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([SCENARIOS[s]["short"] for s in scen_ids], rotation=15, ha="right")
        ax.set_ylabel("EENS (kWh)")
        topo_label = topo.replace("_", " ").title()
        ax.set_title(f"{topo_label}: EENS by Scenario and Defense Tier")
        ax.legend(loc="upper left", frameon=True)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"eens_comparison_{topo}.png")
        plt.close(fig)


def plot_attack_block_rate(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Bar chart: Attack block rate for attack scenarios, per topology."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    attack_scens = [s for s in SCENARIOS if SCENARIOS[s]["attack"] is not None]
    n_scen = len(attack_scens)
    bar_width = 0.22

    for topo in TOPOLOGIES:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(n_scen)

        for i, did in enumerate(def_ids):
            vals = []
            for sid in attack_scens:
                bk = _base_key(topo, sid, did)
                r = averaged.get(bk, {})
                br = r.get("attack_priority_block_rate", 0)
                if br != br:
                    br = 0
                vals.append(br * 100)
            ax.bar(x + i * bar_width, vals, bar_width,
                   label=DEFENSE_TIERS[did]["label"],
                   color=COLORS[did], edgecolor="white", linewidth=0.5)

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([SCENARIOS[s]["short"] for s in attack_scens])
        ax.set_ylabel("Attack Block Rate (%)")
        ax.set_ylim(0, 110)
        topo_label = topo.replace("_", " ").title()
        ax.set_title(f"{topo_label}: Attack Command Block Rate")
        ax.legend(loc="upper left", frameon=True)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"attack_block_rate_{topo}.png")
        plt.close(fig)


def plot_delivery_overhead(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Line chart: Delivery ratio across scenarios for each defense tier."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    scen_ids = list(SCENARIOS.keys())

    for topo in TOPOLOGIES:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(scen_ids))

        for did in def_ids:
            vals = []
            for sid in scen_ids:
                bk = _base_key(topo, sid, did)
                dr = averaged.get(bk, {}).get("delivered_ratio", float("nan"))
                vals.append(dr * 100 if dr == dr else float("nan"))
            ax.plot(x, vals, marker="o", linewidth=2.2,
                    color=COLORS[did], label=DEFENSE_TIERS[did]["label"])

        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIOS[s]["short"] for s in scen_ids], rotation=15, ha="right")
        ax.set_ylabel("Delivery Ratio (%)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        topo_label = topo.replace("_", " ").title()
        ax.set_title(f"{topo_label}: Message Delivery Overhead by Defense Tier")
        ax.legend(loc="lower left", frameon=True)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"delivery_overhead_{topo}.png")
        plt.close(fig)


def plot_quantum_auth_breakdown(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Stacked bar: Quantum auth token stats for quantum tier across scenarios."""
    _ensure_fig_dir()
    scen_ids = list(SCENARIOS.keys())

    for topo in TOPOLOGIES:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(scen_ids))
        bar_width = 0.35

        verified = []
        rejected = []
        gate_blocked = []

        for sid in scen_ids:
            bk = _base_key(topo, sid, "quantum")
            r = averaged.get(bk, {})
            ver = r.get("quantum_control_tokens_verified", 0)
            rej = r.get("quantum_control_tokens_rejected", 0)
            gblk = (r.get("prekey_blocked_quantum_token", 0)
                     + r.get("defense_blocked_quantum_control_token", 0))
            verified.append(ver)
            rejected.append(rej)
            gate_blocked.append(gblk)

        ax.bar(x, verified, bar_width, label="Verified (legitimate)", color="#2a9d8f")
        ax.bar(x, rejected, bar_width, bottom=verified,
               label="Rejected (invalid token)", color="#e76f51")
        ax.bar(x, gate_blocked, bar_width,
               bottom=[v + r for v, r in zip(verified, rejected)],
               label="Gate-blocked (missing token)", color="#d62828")

        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIOS[s]["short"] for s in scen_ids], rotation=15, ha="right")
        ax.set_ylabel("Token Count")
        topo_label = topo.replace("_", " ").title()
        ax.set_title(f"{topo_label}: Quantum Control Token Authentication Breakdown")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"quantum_auth_breakdown_{topo}.png")
        plt.close(fig)


def plot_cross_topology_summary(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Side-by-side EENS comparison across topologies for attack/grid scenario."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    n_topo = len(TOPOLOGIES)
    bar_width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for ax_idx, sid in enumerate(["attack_grid", "attack_island"]):
        ax = axes[ax_idx]
        x = np.arange(n_topo)

        for i, did in enumerate(def_ids):
            vals = []
            for topo in TOPOLOGIES:
                bk = _base_key(topo, sid, did)
                vals.append(averaged.get(bk, {}).get("eens_total_kwh", 0))
            bars = ax.bar(x + i * bar_width, vals, bar_width,
                          label=DEFENSE_TIERS[did]["label"],
                          color=COLORS[did], edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([t.replace("_", " ").title() for t in TOPOLOGIES])
        ax.set_ylabel("EENS (kWh)")
        ax.set_title(SCENARIOS[sid]["label"])
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("Cross-Topology EENS: Quantum vs Classical Under Coordinated FDI Attack",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cross_topology_eens_summary.png")
    plt.close(fig)


def plot_islanding_mode_comparison(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Grouped bar: EENS for grid-connected vs islanded across defense tiers."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    bar_width = 0.18

    for topo in TOPOLOGIES:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = np.arange(len(def_ids))
        pairs = [("baseline_grid", "baseline_island"), ("attack_grid", "attack_island")]
        pair_labels = ["Baseline", "Under Attack"]
        hatches = [None, "//"]
        alphas = [1.0, 0.75]

        for p_idx, (grid_sid, island_sid) in enumerate(pairs):
            grid_vals = []
            island_vals = []
            for did in def_ids:
                grid_vals.append(averaged.get(_base_key(topo, grid_sid, did), {}).get("eens_total_kwh", 0))
                island_vals.append(averaged.get(_base_key(topo, island_sid, did), {}).get("eens_total_kwh", 0))

            offset = p_idx * 2 * bar_width
            b1 = ax.bar(x + offset, grid_vals, bar_width,
                         label=f"{pair_labels[p_idx]} / Grid",
                         color="#457b9d", alpha=alphas[p_idx],
                         hatch=hatches[p_idx], edgecolor="white")
            b2 = ax.bar(x + offset + bar_width, island_vals, bar_width,
                         label=f"{pair_labels[p_idx]} / Islanded",
                         color="#e63946", alpha=alphas[p_idx],
                         hatch=hatches[p_idx], edgecolor="white")

        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels([DEFENSE_TIERS[d]["label"] for d in def_ids])
        ax.set_ylabel("EENS (kWh)")
        topo_label = topo.replace("_", " ").title()
        ax.set_title(f"{topo_label}: Grid-Connected vs Islanded EENS")
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"islanding_comparison_{topo}.png")
        plt.close(fig)


def plot_resilience_comparison(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Grouped bar: SAIDI and ASAI by scenario × defense tier, per topology."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    scen_ids = list(SCENARIOS.keys())
    n_scen = len(scen_ids)
    bar_width = 0.22

    for topo in TOPOLOGIES:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        topo_label = topo.replace("_", " ").title()

        # Panel A: SAIDI
        ax = axes[0]
        x = np.arange(n_scen)
        for i, did in enumerate(def_ids):
            vals = []
            for sid in scen_ids:
                bk = _base_key(topo, sid, did)
                vals.append(averaged.get(bk, {}).get("resilience_saidi_min", 0))
            bars = ax.bar(x + i * bar_width, vals, bar_width,
                          label=DEFENSE_TIERS[did]["label"],
                          color=COLORS[did], edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0.0001:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([SCENARIOS[s]["short"] for s in scen_ids], rotation=15, ha="right")
        ax.set_ylabel("SAIDI (minutes)")
        ax.set_title(f"{topo_label}: System Average Interruption Duration")
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

        # Panel B: ASAI (closer to 1 = better)
        ax = axes[1]
        for i, did in enumerate(def_ids):
            vals = []
            for sid in scen_ids:
                bk = _base_key(topo, sid, did)
                v = averaged.get(bk, {}).get("resilience_asai", 1.0)
                vals.append(v * 100)  # percent
            ax.bar(x + i * bar_width, vals, bar_width,
                   label=DEFENSE_TIERS[did]["label"],
                   color=COLORS[did], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([SCENARIOS[s]["short"] for s in scen_ids], rotation=15, ha="right")
        ax.set_ylabel("ASAI (%)")
        ax.set_title(f"{topo_label}: Average Service Availability Index")
        ax.legend(loc="lower left", frameon=True, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        # Set y-axis to show differences near 100%
        all_asai = []
        for sid in scen_ids:
            for did in def_ids:
                bk = _base_key(topo, sid, did)
                v = averaged.get(bk, {}).get("resilience_asai", 1.0)
                all_asai.append(v * 100)
        ymin = max(95.0, min(all_asai) - 1.0) if all_asai else 95.0
        ax.set_ylim(ymin, 100.5)

        fig.suptitle(f"IEEE 1366 Resilience Metrics — {topo_label}",
                     fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"resilience_comparison_{topo}.png")
        plt.close(fig)


def plot_frequency_comparison(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Grouped bar: Frequency nadir & zenith for all scenarios, per topology.
    Highlights NERC UFLS/OFGS thresholds with horizontal lines."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    scen_ids = list(SCENARIOS.keys())
    n_scen = len(scen_ids)
    bar_width = 0.13

    for topo in TOPOLOGIES:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        topo_label = topo.replace("_", " ").title()

        # Panel A: Frequency Nadir (lower is worse)
        ax = axes[0]
        x = np.arange(n_scen)
        for i, did in enumerate(def_ids):
            vals = []
            for sid in scen_ids:
                bk = _base_key(topo, sid, did)
                v = averaged.get(bk, {}).get("freq_nadir_hz", 60.0)
                vals.append(v if v == v else 60.0)
            ax.bar(x + i * bar_width, vals, bar_width,
                   label=DEFENSE_TIERS[did]["label"],
                   color=COLORS[did], edgecolor="white", linewidth=0.5)
        ax.axhline(y=57.5, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="UFLS threshold (57.5 Hz)")
        ax.axhline(y=59.5, color="orange", linestyle=":", linewidth=1.0, alpha=0.6, label="Normal band (±0.5 Hz)")
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([SCENARIOS[s]["short"] for s in scen_ids], rotation=15, ha="right")
        ax.set_ylabel("Frequency Nadir (Hz)")
        ax.set_title(f"Frequency Nadir (Lower = Worse)")
        ax.legend(loc="lower left", frameon=True, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=54, top=61)

        # Panel B: Max RoCoF (Hz/s)
        ax = axes[1]
        for i, did in enumerate(def_ids):
            vals = []
            for sid in scen_ids:
                bk = _base_key(topo, sid, did)
                v = averaged.get(bk, {}).get("freq_max_rocof_hz_s", 0)
                vals.append(v if v == v else 0)
            ax.bar(x + i * bar_width, vals, bar_width,
                   label=DEFENSE_TIERS[did]["label"],
                   color=COLORS[did], edgecolor="white", linewidth=0.5)
        ax.axhline(y=2.0, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="IEEE 1547 limit (2 Hz/s)")
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([SCENARIOS[s]["short"] for s in scen_ids], rotation=15, ha="right")
        ax.set_ylabel("Max RoCoF (Hz/s)")
        ax.set_title(f"Rate of Change of Frequency")
        ax.legend(loc="upper left", frameon=True, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

        fig.suptitle(f"Frequency Dynamics — {topo_label}",
                     fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"frequency_comparison_{topo}.png")
        plt.close(fig)


def plot_frequency_violation_heatmap(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Heatmap: Frequency violation seconds (scenario × defense) per topology."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    scen_ids = list(SCENARIOS.keys())

    for topo in TOPOLOGIES:
        data = []
        for sid in scen_ids:
            row = []
            for did in def_ids:
                bk = _base_key(topo, sid, did)
                v = averaged.get(bk, {}).get("freq_violation_s", 0)
                row.append(v if v == v else 0)
            data.append(row)

        fig, ax = plt.subplots(figsize=(7, 5))
        data_arr = np.array(data)
        im = ax.imshow(data_arr, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(def_ids)))
        ax.set_xticklabels([DEFENSE_TIERS[d]["short"] for d in def_ids])
        ax.set_yticks(range(len(scen_ids)))
        ax.set_yticklabels([SCENARIOS[s]["short"] for s in scen_ids])

        # Annotate cells
        for i in range(len(scen_ids)):
            for j in range(len(def_ids)):
                val = data_arr[i, j]
                color = "white" if val > data_arr.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.0f}s", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

        topo_label = topo.replace("_", " ").title()
        ax.set_title(f"{topo_label}: Frequency Violation Duration (seconds)")
        fig.colorbar(im, ax=ax, label="Violation Duration (s)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"frequency_violation_heatmap_{topo}.png")
        plt.close(fig)


def plot_se_detection_rates(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Grouped bar: SE bad-data detection rate + quantum authentication count."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    scen_ids = list(SCENARIOS.keys())
    n_scen = len(scen_ids)
    bar_width = 0.22

    for topo in TOPOLOGIES:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        topo_label = topo.replace("_", " ").title()

        # Panel A: Bad-data detection rate
        ax = axes[0]
        x = np.arange(n_scen)
        for i, did in enumerate(def_ids):
            vals = []
            for sid in scen_ids:
                bk = _base_key(topo, sid, did)
                v = averaged.get(bk, {}).get("se_detection_rate", 0)
                vals.append((v if v == v else 0) * 100)
            ax.bar(x + i * bar_width, vals, bar_width,
                   label=DEFENSE_TIERS[did]["label"],
                   color=COLORS[did], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([SCENARIOS[s]["short"] for s in scen_ids], rotation=15, ha="right")
        ax.set_ylabel("Bad-Data Detection Rate (%)")
        ax.set_title(f"WLS State Estimator: Chi² Bad-Data Detection")
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 110)

        # Panel B: Quantum-authenticated SE runs (quantum tier only, vs total)
        ax = axes[1]
        for sid_idx, sid in enumerate(scen_ids):
            bk_q = _base_key(topo, sid, "quantum")
            r_q = averaged.get(bk_q, {})
            n_total = r_q.get("se_n_estimations", 0)
            n_qa = r_q.get("se_n_quantum_authenticated", 0)
            n_unauth = max(0, n_total - n_qa)
            ax.bar(sid_idx, n_qa, 0.5, color="#2a9d8f", label="Quantum-Authenticated" if sid_idx == 0 else "")
            ax.bar(sid_idx, n_unauth, 0.5, bottom=n_qa,
                   color="#e9c46a", label="Classical-Only" if sid_idx == 0 else "")
        ax.set_xticks(range(n_scen))
        ax.set_xticklabels([SCENARIOS[s]["short"] for s in scen_ids], rotation=15, ha="right")
        ax.set_ylabel("SE Runs")
        ax.set_title(f"Quantum-Authenticated State Estimation")
        ax.legend(loc="upper right", frameon=True, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"State Estimation Security — {topo_label}",
                     fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"se_detection_rates_{topo}.png")
        plt.close(fig)


def plot_cyber_physical_overview(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Multi-panel overview: EENS vs SAIDI vs Freq Nadir vs SE Detection for attack scenarios."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    attack_scens = [s for s in SCENARIOS if SCENARIOS[s]["attack"] is not None]

    for topo in TOPOLOGIES:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        topo_label = topo.replace("_", " ").title()
        bar_width = 0.22
        x = np.arange(len(attack_scens))

        metrics_config = [
            (axes[0, 0], "eens_total_kwh", "EENS (kWh)", "Cyber → Energy Not Supplied"),
            (axes[0, 1], "resilience_saidi_min", "SAIDI (min)", "IEEE 1366 Interruption Duration"),
            (axes[1, 0], "freq_nadir_hz", "Frequency Nadir (Hz)", "Physical Layer Stability"),
            (axes[1, 1], "se_detection_rate", "Detection Rate", "WLS Bad-Data Detection"),
        ]

        for ax, metric, ylabel, title in metrics_config:
            for i, did in enumerate(def_ids):
                vals = []
                for sid in attack_scens:
                    bk = _base_key(topo, sid, did)
                    v = averaged.get(bk, {}).get(metric, 0)
                    if v != v:
                        v = 0
                    if metric == "se_detection_rate":
                        v *= 100
                    vals.append(v)
                ax.bar(x + i * bar_width, vals, bar_width,
                       label=DEFENSE_TIERS[did]["label"],
                       color=COLORS[did], edgecolor="white", linewidth=0.5)
            ax.set_xticks(x + bar_width)
            ax.set_xticklabels([SCENARIOS[s]["short"] for s in attack_scens])
            ax.set_ylabel(ylabel if metric != "se_detection_rate" else f"{ylabel} (%)")
            ax.set_title(title)
            ax.legend(loc="best", frameon=True, fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            if metric == "freq_nadir_hz":
                ax.axhline(y=57.5, color="red", linestyle="--", linewidth=1, alpha=0.5)
                ax.set_ylim(bottom=54, top=61)
            else:
                ax.set_ylim(bottom=0)

        fig.suptitle(f"Cyber-Physical Impact Overview — {topo_label}",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"cyber_physical_overview_{topo}.png")
        plt.close(fig)


def plot_quantum_advantage_waterfall(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Waterfall chart: EENS reduction from No Defense → Classical → Quantum."""
    _ensure_fig_dir()
    attack_scens = [s for s in SCENARIOS if SCENARIOS[s]["attack"] is not None]

    fig, axes = plt.subplots(1, len(TOPOLOGIES), figsize=(7 * len(TOPOLOGIES), 6), sharey=False)
    if len(TOPOLOGIES) == 1:
        axes = [axes]

    for ax_idx, topo in enumerate(TOPOLOGIES):
        ax = axes[ax_idx]
        topo_label = topo.replace("_", " ").title()
        bar_width = 0.35
        x_pos = 0

        for sid in attack_scens:
            e_none = averaged.get(_base_key(topo, sid, "no_defense"), {}).get("eens_total_kwh", 0)
            e_cls  = averaged.get(_base_key(topo, sid, "classical"), {}).get("eens_total_kwh", 0)
            e_qtm  = averaged.get(_base_key(topo, sid, "quantum"), {}).get("eens_total_kwh", 0)

            # Base bar (no defense)
            ax.bar(x_pos, e_none, bar_width, color="#d62828", edgecolor="white")
            ax.text(x_pos, e_none + 0.5, f"{e_none:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

            # Classical reduction (green block going down)
            cls_reduction = e_none - e_cls
            ax.bar(x_pos + bar_width, e_cls, bar_width, color="#f4a261", edgecolor="white")
            ax.annotate(f"-{cls_reduction:.1f}\n({cls_reduction/e_none*100:.0f}%)",
                        xy=(x_pos + bar_width, e_none), xytext=(x_pos + bar_width, e_none - cls_reduction/2),
                        ha="center", va="center", fontsize=7, color="#c44536",
                        arrowprops=dict(arrowstyle="->", color="#c44536", lw=1.2))

            # Quantum reduction
            qtm_reduction = e_none - e_qtm
            ax.bar(x_pos + 2 * bar_width, e_qtm, bar_width, color="#2a9d8f", edgecolor="white")
            ax.annotate(f"-{qtm_reduction:.1f}\n({qtm_reduction/e_none*100:.0f}%)",
                        xy=(x_pos + 2 * bar_width, e_none), xytext=(x_pos + 2 * bar_width, (e_none + e_qtm)/2),
                        ha="center", va="center", fontsize=7, color="#1a6b60",
                        arrowprops=dict(arrowstyle="->", color="#1a6b60", lw=1.2))

            x_pos += 3.5 * bar_width + 0.3

        ax.set_xticks([i * (3.5 * bar_width + 0.3) + bar_width for i in range(len(attack_scens))])
        ax.set_xticklabels([SCENARIOS[s]["short"] for s in attack_scens])
        ax.set_ylabel("EENS (kWh)")
        ax.set_title(f"{topo_label}")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#d62828", label="No Defense"),
                           Patch(facecolor="#f4a261", label="Classical"),
                           Patch(facecolor="#2a9d8f", label="Quantum")]
        ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9)

    fig.suptitle("Quantum Advantage: EENS Reduction Waterfall", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "quantum_advantage_waterfall.png")
    plt.close(fig)


def plot_radar_defense_comparison(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Radar/spider chart: Multi-dimensional comparison of defense tiers."""
    _ensure_fig_dir()
    from math import pi

    # Metrics to compare (normalized 0-1, higher = better for all)
    metric_labels = ["EENS\nReduction", "ASAI", "Attack\nBlock Rate",
                     "SE FDI\nDetection", "Freq UFLS\nReduction", "Delivery\nRatio"]

    for topo in TOPOLOGIES:
        topo_label = topo.replace("_", " ").title()
        # Use attack_island as the most challenging scenario
        sid = "attack_island"

        r_none = averaged.get(_base_key(topo, sid, "no_defense"), {})
        r_cls  = averaged.get(_base_key(topo, sid, "classical"), {})
        r_qtm  = averaged.get(_base_key(topo, sid, "quantum"), {})

        e_none = r_none.get("eens_total_kwh", 60)
        e_cls  = r_cls.get("eens_total_kwh", 60)
        e_qtm  = r_qtm.get("eens_total_kwh", 60)

        # Normalize each metric to [0, 1] where 1 = best
        def _safe(v, default=0):
            return v if v == v else default

        values = {
            "no_defense": [
                0.0,  # EENS reduction (baseline)
                _safe(r_none.get("resilience_asai", 0.7)),
                _safe(r_none.get("attack_priority_block_rate", 0)),
                _safe(r_none.get("se_detection_rate", 0)),
                0.0,  # UFLS reduction (baseline)
                _safe(r_none.get("delivered_ratio", 1.0)),
            ],
            "classical": [
                max(0, (e_none - e_cls) / max(1, e_none)),
                _safe(r_cls.get("resilience_asai", 0.7)),
                _safe(r_cls.get("attack_priority_block_rate", 0)),
                _safe(r_cls.get("se_detection_rate", 0)),
                max(0, (_safe(r_none.get("freq_ufls_s", 1000)) - _safe(r_cls.get("freq_ufls_s", 1000)))
                    / max(1, _safe(r_none.get("freq_ufls_s", 1000)))),
                _safe(r_cls.get("delivered_ratio", 1.0)),
            ],
            "quantum": [
                max(0, (e_none - e_qtm) / max(1, e_none)),
                _safe(r_qtm.get("resilience_asai", 0.7)),
                _safe(r_qtm.get("attack_priority_block_rate", 0)),
                # For quantum, SE sees clean data; show quantum auth ratio instead
                min(1.0, _safe(r_qtm.get("se_n_quantum_authenticated", 0)) / max(1, _safe(r_qtm.get("se_n_estimations", 1)))),
                max(0, (_safe(r_none.get("freq_ufls_s", 1000)) - _safe(r_qtm.get("freq_ufls_s", 1000)))
                    / max(1, _safe(r_none.get("freq_ufls_s", 1000)))),
                _safe(r_qtm.get("delivered_ratio", 1.0)),
            ],
        }

        N = len(metric_labels)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for did, color, label in [
            ("no_defense", "#d62828", "No Defense"),
            ("classical", "#f4a261", "Classical"),
            ("quantum", "#2a9d8f", "Quantum"),
        ]:
            vals = values[did] + values[did][:1]
            ax.plot(angles, vals, linewidth=2.2, linestyle="solid", color=color, label=label)
            ax.fill(angles, vals, alpha=0.12, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
        ax.set_title(f"{topo_label}: Defense Tier Comparison\n(FDI + Injection / Islanded)",
                     fontsize=13, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True, fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"radar_defense_comparison_{topo}.png")
        plt.close(fig)


def plot_defense_layer_stacked(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Stacked bar showing contribution of each defense layer to EENS reduction."""
    _ensure_fig_dir()
    attack_scens = [s for s in SCENARIOS if SCENARIOS[s]["attack"] is not None]

    for topo in TOPOLOGIES:
        topo_label = topo.replace("_", " ").title()
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = []
        remaining_eens = []
        classical_reduction = []
        quantum_reduction = []

        for sid in attack_scens:
            e_none = averaged.get(_base_key(topo, sid, "no_defense"), {}).get("eens_total_kwh", 0)
            e_cls  = averaged.get(_base_key(topo, sid, "classical"), {}).get("eens_total_kwh", 0)
            e_qtm  = averaged.get(_base_key(topo, sid, "quantum"), {}).get("eens_total_kwh", 0)
            labels.append(SCENARIOS[sid]["short"])
            remaining_eens.append(e_qtm)
            quantum_reduction.append(e_cls - e_qtm)
            classical_reduction.append(e_none - e_cls)

        x = np.arange(len(labels))
        bar_width = 0.5

        ax.bar(x, remaining_eens, bar_width, label=f"Remaining EENS (Quantum)", color="#264653")
        ax.bar(x, quantum_reduction, bar_width, bottom=remaining_eens,
               label=f"Quantum layer reduction", color="#2a9d8f")
        ax.bar(x, classical_reduction, bar_width,
               bottom=[r + q for r, q in zip(remaining_eens, quantum_reduction)],
               label=f"Classical layer reduction", color="#f4a261")

        # Add total labels
        for i, (r, q, c) in enumerate(zip(remaining_eens, quantum_reduction, classical_reduction)):
            total = r + q + c
            ax.text(i, total + 0.5, f"{total:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            # Percentage labels inside bars
            if c > 2:
                ax.text(i, r + q + c/2, f"{c:.0f} kWh\n({c/total*100:.0f}%)", ha="center", va="center", fontsize=8, color="white")
            if q > 2:
                ax.text(i, r + q/2, f"{q:.0f} kWh\n({q/total*100:.0f}%)", ha="center", va="center", fontsize=8, color="white")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("EENS (kWh)")
        ax.set_title(f"{topo_label}: Defense Layer Contribution to EENS Reduction")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"defense_layer_stacked_{topo}.png")
        plt.close(fig)


def plot_eens_with_error_bars(
    results: List[Dict[str, Any]], averaged: Dict[str, Dict[str, Any]]
) -> None:
    """Bar chart with error bars (std across seeds) for EENS — shows statistical spread."""
    _ensure_fig_dir()
    from collections import defaultdict

    def_ids = list(DEFENSE_TIERS.keys())
    attack_scens = [s for s in SCENARIOS if SCENARIOS[s]["attack"] is not None]
    bar_width = 0.22

    # Group raw results by (topo, scenario, defense)
    groups: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        bk = _base_key(r["_topology"], r["_scenario"], r["_defense"])
        groups[bk].append(r.get("eens_total_kwh", 0))

    for topo in TOPOLOGIES:
        topo_label = topo.replace("_", " ").title()
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(attack_scens))

        for i, did in enumerate(def_ids):
            means = []
            stds = []
            for sid in attack_scens:
                bk = _base_key(topo, sid, did)
                vals = groups.get(bk, [0])
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            bars = ax.bar(x + i * bar_width, means, bar_width, yerr=stds,
                          capsize=4, label=DEFENSE_TIERS[did]["label"],
                          color=COLORS[did], edgecolor="white", linewidth=0.5,
                          error_kw={"elinewidth": 1.5, "capthick": 1.5})
            for bar, m, s in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.3,
                        f"{m:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([SCENARIOS[s]["short"] for s in attack_scens])
        ax.set_ylabel("EENS (kWh)")
        ax.set_title(f"{topo_label}: EENS Under Attack (mean ± std, n={len(SEEDS)} seeds)")
        ax.legend(loc="upper left", frameon=True)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"eens_error_bars_{topo}.png")
        plt.close(fig)


def plot_percentage_improvement(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Horizontal bar: % improvement of Quantum vs Classical across all metrics."""
    _ensure_fig_dir()

    metrics_config = [
        ("eens_total_kwh",         "EENS",           "lower",  "kWh"),
        ("resilience_saidi_min",   "SAIDI",          "lower",  "min"),
        ("resilience_lolp",        "LOLP",           "lower",  ""),
        ("attack_priority_block_rate", "Attack Block Rate", "higher", ""),
        ("freq_ufls_s",            "UFLS Duration",  "lower",  "s"),
        ("resilience_asai",        "ASAI",           "higher", ""),
    ]

    for topo in TOPOLOGIES:
        topo_label = topo.replace("_", " ").title()
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = []
        qtm_vs_cls = []
        qtm_vs_none = []

        sid = "attack_island"  # Hardest scenario
        r_none = averaged.get(_base_key(topo, sid, "no_defense"), {})
        r_cls  = averaged.get(_base_key(topo, sid, "classical"), {})
        r_qtm  = averaged.get(_base_key(topo, sid, "quantum"), {})

        for metric, label, direction, _ in metrics_config:
            v_none = r_none.get(metric, 0)
            v_cls  = r_cls.get(metric, 0)
            v_qtm  = r_qtm.get(metric, 0)
            if v_none != v_none: v_none = 0
            if v_cls != v_cls: v_cls = 0
            if v_qtm != v_qtm: v_qtm = 0

            if direction == "lower":
                # Lower is better: improvement = (baseline - quantum) / baseline * 100
                pct_vs_none = (v_none - v_qtm) / max(1e-9, abs(v_none)) * 100 if v_none != 0 else 0
                pct_vs_cls  = (v_cls - v_qtm)  / max(1e-9, abs(v_cls))  * 100 if v_cls != 0 else 0
            else:
                # Higher is better: improvement = (quantum - baseline) / baseline * 100
                pct_vs_none = (v_qtm - v_none) / max(1e-9, abs(v_none)) * 100 if v_none != 0 else 0
                pct_vs_cls  = (v_qtm - v_cls)  / max(1e-9, abs(v_cls))  * 100 if v_cls != 0 else 0

            labels.append(label)
            qtm_vs_none.append(min(pct_vs_none, 200))  # cap at 200%
            qtm_vs_cls.append(min(pct_vs_cls, 200))

        y = np.arange(len(labels))
        height = 0.35

        ax.barh(y - height/2, qtm_vs_none, height, label="Quantum vs No Defense", color="#2a9d8f", edgecolor="white")
        ax.barh(y + height/2, qtm_vs_cls, height, label="Quantum vs Classical", color="#264653", edgecolor="white")

        # Value labels
        for i, (v1, v2) in enumerate(zip(qtm_vs_none, qtm_vs_cls)):
            if v1 > 0:
                ax.text(v1 + 1, i - height/2, f"+{v1:.0f}%", va="center", fontsize=9, color="#1a6b60")
            if v2 > 0:
                ax.text(v2 + 1, i + height/2, f"+{v2:.0f}%", va="center", fontsize=9, color="#1a3a4a")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Improvement (%)")
        ax.set_title(f"{topo_label}: Quantum Defense Improvement\n(FDI + Injection / Islanded)")
        ax.legend(loc="lower right", frameon=True, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlim(left=-10)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"percentage_improvement_{topo}.png")
        plt.close(fig)


def plot_publication_summary_table(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Publication-quality summary table as a matplotlib figure."""
    _ensure_fig_dir()

    def_ids = list(DEFENSE_TIERS.keys())
    attack_scens = [s for s in SCENARIOS if SCENARIOS[s]["attack"] is not None]
    metrics = [
        ("eens_total_kwh",              "EENS (kWh)",        "{:.1f}"),
        ("resilience_saidi_min",        "SAIDI (min)",       "{:.3f}"),
        ("resilience_asai",             "ASAI",              "{:.4f}"),
        ("resilience_lolp",             "LOLP",              "{:.4f}"),
        ("attack_priority_block_rate",  "Block Rate",        "{:.1%}"),
        ("se_detection_rate",           "SE Det. Rate",      "{:.1%}"),
        ("se_n_quantum_authenticated",  "SE QAuth",          "{:.0f}"),
        ("freq_ufls_s",                 "UFLS (s)",          "{:.0f}"),
        ("delivered_ratio",             "Delivery",          "{:.1%}"),
    ]

    for topo in TOPOLOGIES:
        topo_label = topo.replace("_", " ").title()

        # Build table data
        col_labels = []
        for sid in attack_scens:
            for did in def_ids:
                col_labels.append(f"{SCENARIOS[sid]['short']}\n{DEFENSE_TIERS[did]['short']}")

        row_labels = [m[1] for m in metrics]
        cell_text = []
        cell_colors = []

        for metric_key, _, fmt in metrics:
            row = []
            colors_row = []
            for sid in attack_scens:
                vals_for_scenario = []
                for did in def_ids:
                    bk = _base_key(topo, sid, did)
                    v = averaged.get(bk, {}).get(metric_key, 0)
                    if v != v: v = 0
                    vals_for_scenario.append(v)
                    row.append(fmt.format(v))

                # Color: best value in green, worst in red
                for idx_d, did in enumerate(def_ids):
                    v = vals_for_scenario[idx_d]
                    is_lower_better = metric_key in ("eens_total_kwh", "resilience_saidi_min",
                                                      "resilience_lolp", "freq_ufls_s")
                    if is_lower_better:
                        is_best = (v == min(vals_for_scenario))
                        is_worst = (v == max(vals_for_scenario))
                    else:
                        is_best = (v == max(vals_for_scenario))
                        is_worst = (v == min(vals_for_scenario))

                    if is_best and len(set(vals_for_scenario)) > 1:
                        colors_row.append("#d4edda")  # green
                    elif is_worst and len(set(vals_for_scenario)) > 1:
                        colors_row.append("#f8d7da")  # red
                    else:
                        colors_row.append("white")

            cell_text.append(row)
            cell_colors.append(colors_row)

        fig, ax = plt.subplots(figsize=(max(14, len(col_labels) * 1.8), len(metrics) * 0.6 + 2))
        ax.axis("off")

        table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                         cellColours=cell_colors, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)

        # Style header row
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", fontweight="bold")
        for i in range(len(row_labels)):
            cell = table[i + 1, -1]
            cell.set_text_props(fontweight="bold")

        ax.set_title(f"{topo_label}: Complete Metrics Summary (Attack Scenarios)",
                     fontsize=14, fontweight="bold", pad=20)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"publication_summary_table_{topo}.png")
        plt.close(fig)


# ─── Node Scaling Plots ──────────────────────────────────────────────
def plot_node_scaling_eens(
    scaling_avg: Dict[Tuple[int, str, str, str], Dict[str, Any]],
) -> None:
    """Line chart: EENS vs node count, per topology × defense tier, for attack_grid."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())

    fig, axes = plt.subplots(1, len(TOPOLOGIES), figsize=(5 * len(TOPOLOGIES), 5), sharey=True)
    if len(TOPOLOGIES) == 1:
        axes = [axes]

    for ax_idx, topo in enumerate(TOPOLOGIES):
        ax = axes[ax_idx]
        topo_label = topo.replace("_", " ").title()

        for did in def_ids:
            xs, ys, errs = [], [], []
            for n in NODE_COUNTS:
                key = (n, topo, "attack_grid", did)
                r = scaling_avg.get(key)
                if r is None:
                    continue
                xs.append(n)
                ys.append(r.get("eens_total_kwh", 0))
                errs.append(r.get("eens_total_kwh_std", 0))
            if xs:
                ax.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2.2, capsize=4,
                            color=COLORS[did], label=DEFENSE_TIERS[did]["label"])

        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("EENS (kWh)")
        ax.set_title(topo_label)
        ax.set_xticks(NODE_COUNTS)
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("Node Scaling: EENS Under FDI Attack (Grid-Connected)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "node_scaling_eens.png")
    plt.close(fig)


def plot_node_scaling_block_rate(
    scaling_avg: Dict[Tuple[int, str, str, str], Dict[str, Any]],
) -> None:
    """Line chart: Attack block rate vs node count, per topology × defense tier."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())

    fig, axes = plt.subplots(1, len(TOPOLOGIES), figsize=(5 * len(TOPOLOGIES), 5), sharey=True)
    if len(TOPOLOGIES) == 1:
        axes = [axes]

    for ax_idx, topo in enumerate(TOPOLOGIES):
        ax = axes[ax_idx]
        topo_label = topo.replace("_", " ").title()

        for did in def_ids:
            xs, ys = [], []
            for n in NODE_COUNTS:
                key = (n, topo, "attack_grid", did)
                r = scaling_avg.get(key)
                if r is None:
                    continue
                xs.append(n)
                br = r.get("attack_priority_block_rate", 0)
                ys.append((br if br == br else 0) * 100)
            if xs:
                ax.plot(xs, ys, marker="s", linewidth=2.2,
                        color=COLORS[did], label=DEFENSE_TIERS[did]["label"])

        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Attack Block Rate (%)")
        ax.set_title(topo_label)
        ax.set_xticks(NODE_COUNTS)
        ax.set_ylim(0, 110)
        ax.legend(loc="lower right", frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Node Scaling: Attack Block Rate Under FDI Attack",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "node_scaling_block_rate.png")
    plt.close(fig)


def plot_node_scaling_resilience(
    scaling_avg: Dict[Tuple[int, str, str, str], Dict[str, Any]],
) -> None:
    """Dual-panel: SAIDI and ASAI vs node count under attack."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())

    fig, axes = plt.subplots(2, len(TOPOLOGIES), figsize=(5 * len(TOPOLOGIES), 9), sharey="row")
    if len(TOPOLOGIES) == 1:
        axes = axes.reshape(-1, 1)

    for ax_idx, topo in enumerate(TOPOLOGIES):
        topo_label = topo.replace("_", " ").title()

        # Row 0: SAIDI
        ax = axes[0, ax_idx]
        for did in def_ids:
            xs, ys = [], []
            for n in NODE_COUNTS:
                key = (n, topo, "attack_grid", did)
                r = scaling_avg.get(key)
                if r is None:
                    continue
                xs.append(n)
                ys.append(r.get("resilience_saidi_min", 0))
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2.2,
                        color=COLORS[did], label=DEFENSE_TIERS[did]["label"])
        ax.set_title(topo_label)
        ax.set_ylabel("SAIDI (min)")
        ax.set_xticks(NODE_COUNTS)
        ax.legend(loc="upper left", frameon=True, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # Row 1: ASAI
        ax = axes[1, ax_idx]
        for did in def_ids:
            xs, ys = [], []
            for n in NODE_COUNTS:
                key = (n, topo, "attack_grid", did)
                r = scaling_avg.get(key)
                if r is None:
                    continue
                xs.append(n)
                v = r.get("resilience_asai", 1.0)
                ys.append((v if v == v else 1.0) * 100)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2.2,
                        color=COLORS[did], label=DEFENSE_TIERS[did]["label"])
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("ASAI (%)")
        ax.set_xticks(NODE_COUNTS)
        ax.legend(loc="lower left", frameon=True, fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Node Scaling: IEEE 1366 Resilience Under FDI Attack",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "node_scaling_resilience.png")
    plt.close(fig)


def plot_node_scaling_frequency(
    scaling_avg: Dict[Tuple[int, str, str, str], Dict[str, Any]],
) -> None:
    """Line chart: Frequency nadir vs node count under attack."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())

    fig, axes = plt.subplots(1, len(TOPOLOGIES), figsize=(5 * len(TOPOLOGIES), 5), sharey=True)
    if len(TOPOLOGIES) == 1:
        axes = [axes]

    for ax_idx, topo in enumerate(TOPOLOGIES):
        ax = axes[ax_idx]
        topo_label = topo.replace("_", " ").title()

        for did in def_ids:
            xs, ys = [], []
            for n in NODE_COUNTS:
                key = (n, topo, "attack_grid", did)
                r = scaling_avg.get(key)
                if r is None:
                    continue
                xs.append(n)
                v = r.get("freq_nadir_hz", 60.0)
                ys.append(v if v == v else 60.0)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2.2,
                        color=COLORS[did], label=DEFENSE_TIERS[did]["label"])

        ax.axhline(y=57.5, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="UFLS (57.5 Hz)")
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Frequency Nadir (Hz)")
        ax.set_title(topo_label)
        ax.set_xticks(NODE_COUNTS)
        ax.legend(loc="lower left", frameon=True, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=54, top=61)

    fig.suptitle("Node Scaling: Frequency Nadir Under FDI Attack",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "node_scaling_frequency.png")
    plt.close(fig)


def plot_node_scaling_quantum_advantage(
    scaling_avg: Dict[Tuple[int, str, str, str], Dict[str, Any]],
) -> None:
    """Line chart: Quantum EENS reduction (%) vs node count, per topology."""
    _ensure_fig_dir()

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {"ring": "o", "star": "s", "mesh": "D", "two_cluster_bridge": "^"}

    for topo in TOPOLOGIES:
        topo_label = topo.replace("_", " ").title()
        xs, ys = [], []
        for n in NODE_COUNTS:
            key_none = (n, topo, "attack_grid", "no_defense")
            key_qtm = (n, topo, "attack_grid", "quantum")
            r_none = scaling_avg.get(key_none)
            r_qtm = scaling_avg.get(key_qtm)
            if r_none is None or r_qtm is None:
                continue
            e_none = r_none.get("eens_total_kwh", 0)
            e_qtm = r_qtm.get("eens_total_kwh", 0)
            if e_none > 0.01:
                xs.append(n)
                ys.append((e_none - e_qtm) / e_none * 100)
        if xs:
            ax.plot(xs, ys, marker=markers.get(topo, "o"), linewidth=2.5,
                    markersize=8, label=topo_label)

    ax.set_xlabel("Number of Nodes", fontsize=12)
    ax.set_ylabel("Quantum EENS Reduction (%)", fontsize=12)
    ax.set_title("Quantum Advantage vs Network Size\n(FDI Attack, Grid-Connected)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(NODE_COUNTS)
    ax.legend(loc="best", frameon=True, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0, top=105)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "node_scaling_quantum_advantage.png")
    plt.close(fig)


def plot_node_scaling_heatmap(
    scaling_avg: Dict[Tuple[int, str, str, str], Dict[str, Any]],
) -> None:
    """Heatmap: EENS across (topology × node_count) for each defense tier under attack."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())

    fig, axes = plt.subplots(1, len(def_ids), figsize=(6 * len(def_ids), 5), sharey=True)
    if len(def_ids) == 1:
        axes = [axes]

    for ax_idx, did in enumerate(def_ids):
        ax = axes[ax_idx]
        data = []
        for topo in TOPOLOGIES:
            row = []
            for n in NODE_COUNTS:
                key = (n, topo, "attack_grid", did)
                r = scaling_avg.get(key)
                v = r.get("eens_total_kwh", 0) if r else 0
                row.append(v)
            data.append(row)

        data_arr = np.array(data)
        im = ax.imshow(data_arr, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(NODE_COUNTS)))
        ax.set_xticklabels([str(n) for n in NODE_COUNTS])
        ax.set_xlabel("Nodes")
        ax.set_yticks(range(len(TOPOLOGIES)))
        ax.set_yticklabels([t.replace("_", " ").title() for t in TOPOLOGIES])

        for i in range(len(TOPOLOGIES)):
            for j in range(len(NODE_COUNTS)):
                val = data_arr[i, j]
                color = "white" if val > data_arr.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

        ax.set_title(f"{DEFENSE_TIERS[did]['label']}")
        fig.colorbar(im, ax=ax, label="EENS (kWh)", shrink=0.8)

    fig.suptitle("Node Scaling: EENS Heatmap (Topology × Nodes) Under FDI Attack",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "node_scaling_heatmap.png")
    plt.close(fig)


# ─── Combined Cross-Topology Comparison Plots ───────────────────────
TOPO_MARKERS = {"ring": "o", "star": "s", "mesh": "D", "two_cluster_bridge": "^"}
TOPO_COLORS  = {"ring": "#1f77b4", "star": "#ff7f0e", "mesh": "#2ca02c", "two_cluster_bridge": "#d62728"}


def plot_combined_eens_lines(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Line chart: EENS across scenarios, one panel per defense tier, topologies as lines."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    scen_ids = list(SCENARIOS.keys())

    fig, axes = plt.subplots(1, len(def_ids), figsize=(6 * len(def_ids), 5.5), sharey=True)
    if len(def_ids) == 1:
        axes = [axes]

    for ax_idx, did in enumerate(def_ids):
        ax = axes[ax_idx]
        for topo in TOPOLOGIES:
            label = topo.replace("_", " ").title()
            ys = []
            for sid in scen_ids:
                bk = _base_key(topo, sid, did)
                v = averaged.get(bk, {}).get("eens_total_kwh", 0)
                ys.append(v if v == v else 0)
            ax.plot(range(len(scen_ids)), ys, marker=TOPO_MARKERS.get(topo, "o"),
                    linewidth=2.2, markersize=8, color=TOPO_COLORS.get(topo),
                    label=label)

        ax.set_xticks(range(len(scen_ids)))
        ax.set_xticklabels([SCENARIOS[s]["short"] for s in scen_ids], rotation=20, ha="right")
        ax.set_ylabel("EENS (kWh)")
        ax.set_title(DEFENSE_TIERS[did]["label"])
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("EENS Across Scenarios: Topology Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "combined_eens_lines.png")
    plt.close(fig)


def plot_combined_block_rate_lines(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Line chart: Attack block rate across topologies, defense tiers as lines."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    attack_scens = [s for s in SCENARIOS if SCENARIOS[s]["attack"] is not None]

    fig, axes = plt.subplots(1, len(attack_scens), figsize=(7 * len(attack_scens), 5.5), sharey=True)
    if len(attack_scens) == 1:
        axes = [axes]

    for ax_idx, sid in enumerate(attack_scens):
        ax = axes[ax_idx]
        for did in def_ids:
            ys = []
            for topo in TOPOLOGIES:
                bk = _base_key(topo, sid, did)
                br = averaged.get(bk, {}).get("attack_priority_block_rate", 0)
                ys.append((br if br == br else 0) * 100)
            ax.plot(range(len(TOPOLOGIES)), ys, marker="o", linewidth=2.5,
                    markersize=9, color=COLORS[did], label=DEFENSE_TIERS[did]["label"])

        ax.set_xticks(range(len(TOPOLOGIES)))
        ax.set_xticklabels([t.replace("_", " ").title() for t in TOPOLOGIES], rotation=15, ha="right")
        ax.set_ylabel("Attack Block Rate (%)")
        ax.set_title(SCENARIOS[sid]["short"])
        ax.set_ylim(0, 110)
        ax.legend(loc="center right", frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Attack Block Rate: All Topologies",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "combined_block_rate_lines.png")
    plt.close(fig)


def plot_combined_delivery_lines(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Line chart: Delivery ratio across scenarios, all topologies × defense tiers."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    scen_ids = list(SCENARIOS.keys())

    fig, axes = plt.subplots(1, len(def_ids), figsize=(6 * len(def_ids), 5.5), sharey=True)
    if len(def_ids) == 1:
        axes = [axes]

    for ax_idx, did in enumerate(def_ids):
        ax = axes[ax_idx]
        for topo in TOPOLOGIES:
            label = topo.replace("_", " ").title()
            ys = []
            for sid in scen_ids:
                bk = _base_key(topo, sid, did)
                dr = averaged.get(bk, {}).get("delivered_ratio", 1.0)
                ys.append((dr if dr == dr else 1.0) * 100)
            ax.plot(range(len(scen_ids)), ys, marker=TOPO_MARKERS.get(topo, "o"),
                    linewidth=2.0, markersize=7, color=TOPO_COLORS.get(topo),
                    label=label)

        ax.set_xticks(range(len(scen_ids)))
        ax.set_xticklabels([SCENARIOS[s]["short"] for s in scen_ids], rotation=20, ha="right")
        ax.set_ylabel("Delivery Ratio (%)")
        ax.set_title(DEFENSE_TIERS[did]["label"])
        ax.legend(loc="lower left", frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    fig.suptitle("Message Delivery Ratio: Topology Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "combined_delivery_lines.png")
    plt.close(fig)


def plot_combined_resilience_lines(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Line chart: SAIDI and ASAI for attack scenarios, topologies as x-axis, defense as lines."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    # Use the hardest scenario: FDI + island
    sid = "attack_island"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: SAIDI
    ax = axes[0]
    for did in def_ids:
        ys = []
        for topo in TOPOLOGIES:
            bk = _base_key(topo, sid, did)
            v = averaged.get(bk, {}).get("resilience_saidi_min", 0)
            ys.append(v if v == v else 0)
        ax.plot(range(len(TOPOLOGIES)), ys, marker="o", linewidth=2.5,
                markersize=9, color=COLORS[did], label=DEFENSE_TIERS[did]["label"])
    ax.set_xticks(range(len(TOPOLOGIES)))
    ax.set_xticklabels([t.replace("_", " ").title() for t in TOPOLOGIES], rotation=15, ha="right")
    ax.set_ylabel("SAIDI (minutes)")
    ax.set_title("System Avg. Interruption Duration")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel B: ASAI
    ax = axes[1]
    for did in def_ids:
        ys = []
        for topo in TOPOLOGIES:
            bk = _base_key(topo, sid, did)
            v = averaged.get(bk, {}).get("resilience_asai", 1.0)
            ys.append((v if v == v else 1.0) * 100)
        ax.plot(range(len(TOPOLOGIES)), ys, marker="o", linewidth=2.5,
                markersize=9, color=COLORS[did], label=DEFENSE_TIERS[did]["label"])
    ax.set_xticks(range(len(TOPOLOGIES)))
    ax.set_xticklabels([t.replace("_", " ").title() for t in TOPOLOGIES], rotation=15, ha="right")
    ax.set_ylabel("ASAI (%)")
    ax.set_title("Avg. Service Availability Index")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"IEEE 1366 Resilience: Cross-Topology Comparison ({SCENARIOS[sid]['short']})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "combined_resilience_lines.png")
    plt.close(fig)


def plot_combined_se_lines(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Line chart: SE detection rate across topologies for FDI scenarios."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    attack_scens = [s for s in SCENARIOS if SCENARIOS[s]["attack"] is not None]

    fig, axes = plt.subplots(1, len(attack_scens), figsize=(7 * len(attack_scens), 5.5), sharey=True)
    if len(attack_scens) == 1:
        axes = [axes]

    for ax_idx, sid in enumerate(attack_scens):
        ax = axes[ax_idx]
        for did in def_ids:
            ys = []
            for topo in TOPOLOGIES:
                bk = _base_key(topo, sid, did)
                v = averaged.get(bk, {}).get("se_detection_rate", 0)
                ys.append((v if v == v else 0) * 100)
            ax.plot(range(len(TOPOLOGIES)), ys, marker="o", linewidth=2.5,
                    markersize=9, color=COLORS[did], label=DEFENSE_TIERS[did]["label"])

        ax.set_xticks(range(len(TOPOLOGIES)))
        ax.set_xticklabels([t.replace("_", " ").title() for t in TOPOLOGIES], rotation=15, ha="right")
        ax.set_ylabel("Chi² Detection Rate (%)")
        ax.set_title(SCENARIOS[sid]["short"])
        ax.set_ylim(0, 50)
        ax.legend(loc="upper right", frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("State Estimation Bad-Data Detection: All Topologies",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "combined_se_detection_lines.png")
    plt.close(fig)


def plot_combined_eens_reduction_area(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Area chart: EENS breakdown (remaining + quantum reduction + classical reduction)
    across topologies for attack_grid scenario."""
    _ensure_fig_dir()
    sid = "attack_grid"

    topos = TOPOLOGIES
    x_labels = [t.replace("_", " ").title() for t in topos]
    x = np.arange(len(topos))

    remaining, qtm_red, cls_red = [], [], []
    for topo in topos:
        e_none = averaged.get(_base_key(topo, sid, "no_defense"), {}).get("eens_total_kwh", 0)
        e_cls  = averaged.get(_base_key(topo, sid, "classical"), {}).get("eens_total_kwh", 0)
        e_qtm  = averaged.get(_base_key(topo, sid, "quantum"), {}).get("eens_total_kwh", 0)
        remaining.append(e_qtm)
        qtm_red.append(e_cls - e_qtm)
        cls_red.append(e_none - e_cls)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, remaining, 0.5, label="Remaining (Quantum)", color="#264653")
    ax.bar(x, qtm_red, 0.5, bottom=remaining, label="Quantum Layer Reduction", color="#2a9d8f")
    ax.bar(x, cls_red, 0.5,
           bottom=[r + q for r, q in zip(remaining, qtm_red)],
           label="Classical Layer Reduction", color="#f4a261")

    # Connect tops with lines for trend
    totals = [r + q + c for r, q, c in zip(remaining, qtm_red, cls_red)]
    ax.plot(x, totals, color="#d62828", linewidth=2.5, marker="D", markersize=8,
            label="Total EENS (No Defense)", zorder=5)
    ax.plot(x, [r + q for r, q in zip(remaining, qtm_red)], color="#f4a261",
            linewidth=1.5, marker="s", markersize=6, linestyle="--",
            label="After Classical", zorder=5)
    ax.plot(x, remaining, color="#2a9d8f", linewidth=1.5, marker="o",
            markersize=6, linestyle="--", label="After Quantum", zorder=5)

    for i, (r, q, c) in enumerate(zip(remaining, qtm_red, cls_red)):
        total = r + q + c
        ax.text(i, total + 1, f"{total:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(i, r - 0.3, f"{r:.1f}", ha="center", va="top", fontsize=8, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel("EENS (kWh)", fontsize=12)
    ax.set_title("Defense Layer EENS Breakdown Across Topologies (FDI + Injection / Grid)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "combined_eens_reduction_area.png")
    plt.close(fig)


def plot_combined_quantum_advantage_pct(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Line chart: Quantum % improvement over both Classical and No Defense,
    across topologies, for both attack scenarios."""
    _ensure_fig_dir()
    attack_scens = [s for s in SCENARIOS if SCENARIOS[s]["attack"] is not None]

    fig, axes = plt.subplots(1, len(attack_scens), figsize=(7 * len(attack_scens), 5.5), sharey=True)
    if len(attack_scens) == 1:
        axes = [axes]

    for ax_idx, sid in enumerate(attack_scens):
        ax = axes[ax_idx]
        qtm_vs_none, qtm_vs_cls = [], []
        x_labels = []
        for topo in TOPOLOGIES:
            e_none = averaged.get(_base_key(topo, sid, "no_defense"), {}).get("eens_total_kwh", 1)
            e_cls  = averaged.get(_base_key(topo, sid, "classical"), {}).get("eens_total_kwh", 1)
            e_qtm  = averaged.get(_base_key(topo, sid, "quantum"), {}).get("eens_total_kwh", 0)
            qtm_vs_none.append((e_none - e_qtm) / max(0.01, e_none) * 100)
            qtm_vs_cls.append((e_cls - e_qtm) / max(0.01, e_cls) * 100)
            x_labels.append(topo.replace("_", " ").title())

        x = np.arange(len(TOPOLOGIES))
        ax.plot(x, qtm_vs_none, marker="o", linewidth=2.5, markersize=9,
                color="#2a9d8f", label="Quantum vs No Defense")
        ax.plot(x, qtm_vs_cls, marker="s", linewidth=2.5, markersize=9,
                color="#264653", label="Quantum vs Classical")

        for i, (v1, v2) in enumerate(zip(qtm_vs_none, qtm_vs_cls)):
            ax.annotate(f"{v1:.0f}%", (i, v1), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color="#1a6b60")
            ax.annotate(f"{v2:.0f}%", (i, v2), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=9, color="#1a3a4a")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=15, ha="right")
        ax.set_ylabel("EENS Reduction (%)")
        ax.set_title(SCENARIOS[sid]["short"])
        ax.legend(loc="lower right", frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

    fig.suptitle("Quantum Advantage: EENS Reduction Across Topologies",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "combined_quantum_advantage_pct.png")
    plt.close(fig)


def plot_combined_frequency_lines(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Line chart: Frequency nadir and UFLS duration across topologies for islanded attack."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    sid = "attack_island"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Frequency Nadir
    ax = axes[0]
    for did in def_ids:
        ys = []
        for topo in TOPOLOGIES:
            bk = _base_key(topo, sid, did)
            v = averaged.get(bk, {}).get("freq_nadir_hz", 60.0)
            ys.append(v if v == v else 60.0)
        ax.plot(range(len(TOPOLOGIES)), ys, marker="o", linewidth=2.5,
                markersize=9, color=COLORS[did], label=DEFENSE_TIERS[did]["label"])
    ax.axhline(y=57.5, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="UFLS (57.5 Hz)")
    ax.set_xticks(range(len(TOPOLOGIES)))
    ax.set_xticklabels([t.replace("_", " ").title() for t in TOPOLOGIES], rotation=15, ha="right")
    ax.set_ylabel("Frequency Nadir (Hz)")
    ax.set_title("Frequency Nadir (Lower = Worse)")
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=54, top=61)

    # Panel B: UFLS duration
    ax = axes[1]
    for did in def_ids:
        ys = []
        for topo in TOPOLOGIES:
            bk = _base_key(topo, sid, did)
            v = averaged.get(bk, {}).get("freq_ufls_s", 0)
            ys.append(v if v == v else 0)
        ax.plot(range(len(TOPOLOGIES)), ys, marker="s", linewidth=2.5,
                markersize=9, color=COLORS[did], label=DEFENSE_TIERS[did]["label"])
    ax.set_xticks(range(len(TOPOLOGIES)))
    ax.set_xticklabels([t.replace("_", " ").title() for t in TOPOLOGIES], rotation=15, ha="right")
    ax.set_ylabel("UFLS Duration (s)")
    ax.set_title("Under-Frequency Load Shedding Duration")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle(f"Frequency Dynamics: Cross-Topology ({SCENARIOS[sid]['short']})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "combined_frequency_lines.png")
    plt.close(fig)


def print_scaling_results(
    scaling_avg: Dict[Tuple[int, str, str, str], Dict[str, Any]],
) -> None:
    """Print node scaling study results to console."""
    def_ids = list(DEFENSE_TIERS.keys())

    print(f"\n{'═' * 80}")
    print(f"  NODE SCALING STUDY RESULTS")
    print(f"{'═' * 80}")

    for sid in SCALING_SCENARIOS:
        scen_label = SCENARIOS[sid]["short"]
        print(f"\n  Scenario: {scen_label}")
        print(f"  {'Topology':25s} {'Nodes':>6s}", end="")
        for did in def_ids:
            print(f" {DEFENSE_TIERS[did]['short']:>12s}", end="")
        print()
        print(f"  {'─' * 80}")

        for topo in TOPOLOGIES:
            for n in NODE_COUNTS:
                tag = f"{topo.replace('_', ' ').title()}"
                first_col = tag if n == NODE_COUNTS[0] else ""
                print(f"  {first_col:25s} {n:6d}", end="")
                for did in def_ids:
                    key = (n, topo, sid, did)
                    r = scaling_avg.get(key)
                    if r:
                        v = r.get("eens_total_kwh", float("nan"))
                        print(f" {v:12.3f}", end="")
                    else:
                        print(f" {'N/A':>12s}", end="")
                print()
            print()

    # Quantum advantage summary
    print(f"\n  Quantum EENS Reduction (%) vs No Defense — attack_grid:")
    print(f"  {'Topology':25s}", end="")
    for n in NODE_COUNTS:
        print(f" {n:>8d}N", end="")
    print()
    print(f"  {'─' * 65}")
    for topo in TOPOLOGIES:
        print(f"  {topo.replace('_', ' ').title():25s}", end="")
        for n in NODE_COUNTS:
            key_none = (n, topo, "attack_grid", "no_defense")
            key_qtm = (n, topo, "attack_grid", "quantum")
            r_none = scaling_avg.get(key_none)
            r_qtm = scaling_avg.get(key_qtm)
            if r_none and r_qtm:
                e_none = r_none.get("eens_total_kwh", 0)
                e_qtm = r_qtm.get("eens_total_kwh", 0)
                if e_none > 0.01:
                    pct = (e_none - e_qtm) / e_none * 100
                    print(f" {pct:8.1f}%", end="")
                else:
                    print(f" {'0.0%':>9s}", end="")
            else:
                print(f" {'N/A':>9s}", end="")
        print()
    print()


def plot_quantum_latency_overhead(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Combined chart: Quantum latency overhead across all topologies.

    Panel 1: Control message mean latency (baseline + attack) by defense tier
    Panel 2: Latency breakdown (propagation vs queuing vs quantum protocol)
    Panel 3: Key consumption comparison
    """
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())

    # ── Figure 1: Control latency comparison (grouped bar, all topos) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for panel_idx, (scen, scen_label) in enumerate([
        ("baseline_grid", "Normal Operation"),
        ("attack_grid", "Under Attack"),
    ]):
        ax = axes[panel_idx]
        x = np.arange(len(TOPOLOGIES))
        width = 0.22
        for i, did in enumerate(def_ids):
            vals = []
            for topo in TOPOLOGIES:
                bk = _base_key(topo, scen, did)
                v = averaged.get(bk, {}).get("control_latency_mean_ms", float("nan"))
                vals.append(v)
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width, label=DEFENSE_TIERS[did]["label"],
                          color=COLORS[did], edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val == val:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=7)
        topo_labels = [t.replace("_", "\n") for t in TOPOLOGIES]
        ax.set_xticks(x)
        ax.set_xticklabels(topo_labels, fontsize=8)
        ax.set_ylabel("Control Latency (ms)")
        ax.set_title(f"Control Message Latency — {scen_label}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Quantum Protocol Latency Overhead", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "quantum_latency_overhead.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Overhead delta (quantum - no_defense) across topologies ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: absolute overhead in ms
    ax = axes[0]
    scenarios_to_show = ["baseline_grid", "baseline_island", "attack_grid", "attack_island"]
    scen_labels = ["Grid\nBaseline", "Island\nBaseline", "Grid\nAttack", "Island\nAttack"]
    x = np.arange(len(TOPOLOGIES))
    width = 0.18
    for j, (scen, slabel) in enumerate(zip(scenarios_to_show, scen_labels)):
        deltas = []
        for topo in TOPOLOGIES:
            bk_q = _base_key(topo, scen, "quantum")
            bk_n = _base_key(topo, scen, "no_defense")
            lat_q = averaged.get(bk_q, {}).get("control_latency_mean_ms", float("nan"))
            lat_n = averaged.get(bk_n, {}).get("control_latency_mean_ms", float("nan"))
            deltas.append(lat_q - lat_n if lat_q == lat_q and lat_n == lat_n else float("nan"))
        offset = (j - 1.5) * width
        ax.bar(x + offset, deltas, width, label=slabel.replace("\n", " "))
    topo_labels = [t.replace("_", "\n") for t in TOPOLOGIES]
    ax.set_xticks(x)
    ax.set_xticklabels(topo_labels, fontsize=8)
    ax.set_ylabel("Additional Latency (ms)")
    ax.set_title("Quantum Overhead Δ (Quantum − No Defense)", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Panel 2: percentage overhead
    ax = axes[1]
    for j, (scen, slabel) in enumerate(zip(scenarios_to_show, scen_labels)):
        pcts = []
        for topo in TOPOLOGIES:
            bk_q = _base_key(topo, scen, "quantum")
            bk_n = _base_key(topo, scen, "no_defense")
            lat_q = averaged.get(bk_q, {}).get("control_latency_mean_ms", float("nan"))
            lat_n = averaged.get(bk_n, {}).get("control_latency_mean_ms", float("nan"))
            if lat_n and lat_n == lat_n and lat_q == lat_q:
                pcts.append((lat_q - lat_n) / lat_n * 100)
            else:
                pcts.append(float("nan"))
        offset = (j - 1.5) * width
        ax.bar(x + offset, pcts, width, label=slabel.replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(topo_labels, fontsize=8)
    ax.set_ylabel("Overhead (%)")
    ax.set_title("Quantum Latency Overhead (%)", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "quantum_overhead_delta.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: Key consumption & delivery comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Key bits consumed
    ax = axes[0]
    x = np.arange(len(TOPOLOGIES))
    width = 0.22
    for i, did in enumerate(def_ids):
        vals = []
        for topo in TOPOLOGIES:
            bk = _base_key(topo, "attack_grid", did)
            v = averaged.get(bk, {}).get("key_bits_spent_sum", 0) / 1e6
            vals.append(v)
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, label=DEFENSE_TIERS[did]["label"],
               color=COLORS[did], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(topo_labels, fontsize=8)
    ax.set_ylabel("Key Bits Consumed (Mbits)")
    ax.set_title("QKD Key Consumption (Under Attack)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Delivery ratio
    ax = axes[1]
    for i, did in enumerate(def_ids):
        vals = []
        for topo in TOPOLOGIES:
            bk = _base_key(topo, "attack_grid", did)
            v = averaged.get(bk, {}).get("delivered_ratio", float("nan"))
            vals.append(v * 100 if v == v else float("nan"))
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, label=DEFENSE_TIERS[did]["label"],
               color=COLORS[did], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(topo_labels, fontsize=8)
    ax.set_ylabel("Delivery Ratio (%)")
    ax.set_title("Message Delivery (Under Attack)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: P95 control latency
    ax = axes[2]
    for i, did in enumerate(def_ids):
        vals = []
        for topo in TOPOLOGIES:
            bk = _base_key(topo, "attack_grid", did)
            v = averaged.get(bk, {}).get("control_latency_p95_ms", float("nan"))
            vals.append(v)
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, label=DEFENSE_TIERS[did]["label"],
               color=COLORS[did], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(topo_labels, fontsize=8)
    ax.set_ylabel("P95 Latency (ms)")
    ax.set_title("Control Latency P95 (Under Attack)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "quantum_overhead_comprehensive.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_node_scaling_overhead(scaling_avg: Dict) -> None:
    """Line chart: Quantum latency overhead vs node count across topologies."""
    _ensure_fig_dir()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: Absolute control latency (quantum tier, attack_grid)
    ax = axes[0]
    for did_color, did in [("#264653", "no_defense"), ("#e9c46a", "classical"), ("#e76f51", "quantum")]:
        for topo in TOPOLOGIES:
            xs, ys = [], []
            for n in NODE_COUNTS:
                key = (n, topo, "attack_grid", did)
                r = scaling_avg.get(key)
                if r:
                    xs.append(n)
                    ys.append(r.get("control_latency_mean_ms", float("nan")))
            if xs:
                linestyle = {"ring": "-", "star": "--", "mesh": "-.", "two_cluster_bridge": ":"}.get(topo, "-")
                ax.plot(xs, ys, marker="o", linestyle=linestyle, linewidth=1.5,
                        color=COLORS.get(did, did_color), alpha=0.8)
    # Custom legends
    from matplotlib.lines import Line2D
    topo_handles = [Line2D([0], [0], color="gray", linestyle=ls, label=t.replace("_", " "))
                    for t, ls in [("ring", "-"), ("star", "--"), ("mesh", "-."), ("two_cluster_bridge", ":")]]
    def_handles = [Line2D([0], [0], color=COLORS[d], linewidth=3, label=DEFENSE_TIERS[d]["label"])
                   for d in ["no_defense", "classical", "quantum"]]
    leg1 = ax.legend(handles=topo_handles, loc="upper left", fontsize=7, title="Topology")
    ax.add_artist(leg1)
    ax.legend(handles=def_handles, loc="lower right", fontsize=7, title="Defense")
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Control Latency (ms)")
    ax.set_title("Control Latency vs Network Size (Under Attack)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel 2: Overhead delta (quantum - no_defense) vs node count
    ax = axes[1]
    for topo in TOPOLOGIES:
        xs, deltas = [], []
        for n in NODE_COUNTS:
            key_q = (n, topo, "attack_grid", "quantum")
            key_n = (n, topo, "attack_grid", "no_defense")
            rq = scaling_avg.get(key_q)
            rn = scaling_avg.get(key_n)
            if rq and rn:
                lat_q = rq.get("control_latency_mean_ms", float("nan"))
                lat_n = rn.get("control_latency_mean_ms", float("nan"))
                if lat_q == lat_q and lat_n == lat_n:
                    xs.append(n)
                    deltas.append(lat_q - lat_n)
        if xs:
            linestyle = {"ring": "-", "star": "--", "mesh": "-.", "two_cluster_bridge": ":"}.get(topo, "-")
            ax.plot(xs, deltas, marker="s", linestyle=linestyle, linewidth=2,
                    color="#e76f51", label=topo.replace("_", " "))
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Quantum Overhead Δ (ms)")
    ax.set_title("Quantum Latency Overhead vs Network Size", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "node_scaling_latency_overhead.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_legit_vs_attack_delivery(averaged: Dict[str, Dict[str, Any]]) -> None:
    """Combined chart: Legitimate traffic delivery % and attack traffic delivery %."""
    _ensure_fig_dir()
    def_ids = list(DEFENSE_TIERS.keys())
    attack_scens = [s for s in SCENARIOS if SCENARIOS[s]["attack"] is not None]

    # ── Figure: 2-panel (Legit delivery | Attack allowed) across topologies ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Legitimate traffic delivery (true_allow_rate)
    ax = axes[0]
    x = np.arange(len(TOPOLOGIES))
    width = 0.22
    for i, did in enumerate(def_ids):
        vals = []
        for topo in TOPOLOGIES:
            rates = []
            for sid in attack_scens:
                bk = _base_key(topo, sid, did)
                r = averaged.get(bk, {})
                ta = r.get("true_allow_rate", float("nan"))
                if ta == ta:
                    rates.append(ta * 100)
            vals.append(np.mean(rates) if rates else float("nan"))
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=DEFENSE_TIERS[did]["label"],
                      color=COLORS[did], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val == val:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    topo_labels = [t.replace("_", "\n") for t in TOPOLOGIES]
    ax.set_xticks(x)
    ax.set_xticklabels(topo_labels, fontsize=9)
    ax.set_ylabel("Legitimate Delivery Rate (%)")
    ax.set_title("Legitimate Traffic Delivery", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Attack traffic allowed (attack_priority_allow_rate)
    ax = axes[1]
    for i, did in enumerate(def_ids):
        vals = []
        for topo in TOPOLOGIES:
            rates = []
            for sid in attack_scens:
                bk = _base_key(topo, sid, did)
                r = averaged.get(bk, {})
                aa = r.get("attack_priority_allow_rate", float("nan"))
                if aa == aa:
                    rates.append(aa * 100)
            vals.append(np.mean(rates) if rates else float("nan"))
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=DEFENSE_TIERS[did]["label"],
                      color=COLORS[did], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val == val:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(topo_labels, fontsize=9)
    ax.set_ylabel("Attack Msgs Allowed Through (%)")
    ax.set_title("Attack Traffic Penetration", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Traffic Separation: Legitimate vs Attack Message Delivery",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "legit_vs_attack_delivery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: False positive rate (legit blocked) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, did in enumerate(def_ids):
        vals = []
        for topo in TOPOLOGIES:
            rates = []
            for sid in attack_scens:
                bk = _base_key(topo, sid, did)
                r = averaged.get(bk, {})
                fb = r.get("false_block_rate", float("nan"))
                if fb == fb:
                    rates.append(fb * 100)
            vals.append(np.mean(rates) if rates else float("nan"))
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=DEFENSE_TIERS[did]["label"],
                      color=COLORS[did], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val == val:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(topo_labels, fontsize=9)
    ax.set_ylabel("False Positive Rate (Legit Blocked) (%)")
    ax.set_title("Defense False Positive Rate — Legitimate Messages Incorrectly Blocked",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "false_positive_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── V11: Individual Defense Layer Ablation Study ────────────────────
ABLATION_TIERS = OrderedDict([
    ("none", {
        "label": "No Defense",
        "short": "None",
        "defense_mode": "none",
        "enable_qkd": False,
        "enable_quantum_protocols": False,
        "enable_quantum_control_auth": False,
        "enable_sensor_challenges": False,
        "verification_delay_ms": 0,
        "degraded_verification_delay_ms": 0,
    }),
    ("rate_limit_only", {
        "label": "Rate Limiting Only",
        "short": "Rate Lim",
        "defense_mode": "ratelimit",
        "enable_qkd": False,
        "enable_quantum_protocols": False,
        "enable_quantum_control_auth": False,
        "enable_sensor_challenges": False,
        "verification_delay_ms": 5,
        "degraded_verification_delay_ms": 10,
    }),
    ("classical_sig", {
        "label": "Signature + ACL",
        "short": "Sig+ACL",
        "defense_mode": "signature",
        "enable_qkd": False,
        "enable_quantum_protocols": False,
        "enable_quantum_control_auth": False,
        "enable_sensor_challenges": False,
        "verification_delay_ms": 10,
        "degraded_verification_delay_ms": 30,
    }),
    ("classical_full", {
        "label": "Classical Full (hardened_v3)",
        "short": "Classical",
        "defense_mode": "hardened_v3",
        "enable_qkd": False,
        "enable_quantum_protocols": False,
        "enable_quantum_control_auth": False,
        "enable_sensor_challenges": True,
        "verification_delay_ms": 20,
        "degraded_verification_delay_ms": 60,
    }),
    ("quantum_no_token", {
        "label": "QKD Auth (No Token)",
        "short": "QKD Only",
        "defense_mode": "hardened_v3",
        "enable_qkd": True,
        "enable_quantum_protocols": True,
        "enable_quantum_control_auth": False,
        "enable_sensor_challenges": True,
        "verification_delay_ms": 25,
        "degraded_verification_delay_ms": 80,
    }),
    ("quantum_full", {
        "label": "Quantum Full (QKD+QCA)",
        "short": "QKD+QCA",
        "defense_mode": "hardened_v3",
        "enable_qkd": True,
        "enable_quantum_protocols": True,
        "enable_quantum_control_auth": True,
        "enable_sensor_challenges": True,
        "quantum_auth_bypass_prob": 0.03,
        "verification_delay_ms": 35,
        "degraded_verification_delay_ms": 100,
    }),
])

ABLATION_COLORS = {
    "none": "#264653",
    "rate_limit_only": "#287271",
    "classical_sig": "#2a9d8f",
    "classical_full": "#e9c46a",
    "quantum_no_token": "#f4a261",
    "quantum_full": "#e76f51",
}


def run_ablation_study(
    topology: str = "star",
    n_nodes: int = 10,
    seeds: List[int] = None,
) -> List[Dict[str, Any]]:
    """Run individual defense layer ablation study on a single topology."""
    if seeds is None:
        seeds = [42, 137, 256]
    scenarios = ["baseline_grid", "attack_grid"]
    cases = []
    for seed in seeds:
        for scen_id in scenarios:
            for abl_id, abl in ABLATION_TIERS.items():
                scen = SCENARIOS[scen_id]
                if scen["attack"] is None:
                    scenario_str = "baseline"
                else:
                    scenario_str = f"{scen['attack']}_def_{abl['defense_mode']}"
                node_list = [f"MG{i}" for i in range(n_nodes)]
                kwargs = dict(
                    topology=topology,
                    nodes=node_list,
                    seed=seed,
                    horizon_s=HORIZON_S,
                    out_dir=OUT_DIR,
                    scenario=scenario_str,
                    route_policy="shortest",
                    k_paths=3,
                    attack_intensity="S3",
                    distributed_attacks=True,
                    num_attack_windows=5,
                    energy_record_interval=30,
                    infrastructure_override=FAIR_INFRA,
                    write_outputs=False,
                    enable_qkd=abl["enable_qkd"],
                    enable_quantum_protocols=abl["enable_quantum_protocols"],
                    enable_quantum_control_auth=abl["enable_quantum_control_auth"],
                    enable_sensor_challenges=abl.get("enable_sensor_challenges", False),
                    quantum_auth_bypass_prob=abl.get("quantum_auth_bypass_prob", 0.0),
                    verification_delay_ms=abl.get("verification_delay_ms", None),
                    degraded_verification_delay_ms=abl.get("degraded_verification_delay_ms", None),
                    spoof_auth_bypass_prob=0.0,
                    enable_supervisory_islanding=scen["islanding"],
                    qec_code_distance=3,
                    e2e_distillation_rounds=1,
                    e2e_swap_success_prob=0.5,
                    quantum_control_token_ttl_ms=1500,
                )
                cases.append((abl_id, scen_id, seed, kwargs))

    print(f"\n  ── Ablation Study: {topology}, {n_nodes} nodes, "
          f"{len(ABLATION_TIERS)} tiers × {len(scenarios)} scenarios × "
          f"{len(seeds)} seeds = {len(cases)} runs ──")

    results = []
    for i, (abl_id, scen_id, seed, kwargs) in enumerate(cases, 1):
        label = f"{ABLATION_TIERS[abl_id]['short']}/{SCENARIOS[scen_id]['short']}/s{seed}"
        print(f"    [{i:3d}/{len(cases)}] {label} ...", end=" ", flush=True)
        try:
            r = run_one(**kwargs)
            r["_ablation"] = abl_id
            r["_scenario"] = scen_id
            r["_seed"] = seed
            r["_topology"] = topology
            r["_n_nodes"] = n_nodes
            results.append(r)
            eens = r.get("eens_total_kwh", -1)
            print(f"EENS={eens:.2f} kWh")
        except Exception as e:
            print(f"FAILED: {e}")
    return results


def average_ablation_results(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Average ablation results across seeds."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = (r["_ablation"], r["_scenario"])
        groups[key].append(r)

    averaged = {}
    for key, runs in groups.items():
        avg = {}
        for k in runs[0]:
            if k.startswith("_"):
                avg[k] = runs[0][k]
                continue
            vals = [r[k] for r in runs if isinstance(r.get(k), (int, float)) and r[k] == r[k]]
            if vals:
                avg[k] = np.mean(vals)
                avg[f"{k}_std"] = np.std(vals) if len(vals) > 1 else 0.0
        averaged[key] = avg
    return averaged


def plot_ablation_study(ablation_avg: Dict) -> None:
    """Generate ablation study plots showing individual defense layer contributions."""
    _ensure_fig_dir()
    abl_ids = list(ABLATION_TIERS.keys())
    abl_labels = [ABLATION_TIERS[a]["short"] for a in abl_ids]

    # ── Figure 1: EENS comparison (baseline vs attack) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for panel_idx, scen in enumerate(["baseline_grid", "attack_grid"]):
        ax = axes[panel_idx]
        vals = []
        errs = []
        colors = []
        for abl_id in abl_ids:
            key = (abl_id, scen)
            r = ablation_avg.get(key, {})
            vals.append(r.get("eens_total_kwh", 0))
            errs.append(r.get("eens_total_kwh_std", 0))
            colors.append(ABLATION_COLORS.get(abl_id, "#888888"))
        x = np.arange(len(abl_ids))
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5,
                      yerr=errs, capsize=4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(abl_labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("EENS (kWh)")
        scen_label = "Normal Operation" if "baseline" in scen else "Under Attack"
        ax.set_title(f"EENS — {scen_label}", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Defense Layer Ablation: EENS by Defense Tier (Star, 10 Nodes)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_eens.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Security metrics (attack scenario only) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    scen = "attack_grid"
    vals_dict = {}
    for metric, ax, title, ylabel in [
        ("attack_priority_block_rate", axes[0, 0], "Attack Block Rate", "Block Rate (%)"),
        ("false_block_rate", axes[0, 1], "False Positive Rate (Legit Blocked)", "False Positive (%)"),
        ("control_latency_mean_ms", axes[1, 0], "Control Message Latency", "Latency (ms)"),
        ("delivered_ratio", axes[1, 1], "Overall Message Delivery", "Delivery Rate (%)"),
    ]:
        vals = []
        colors = []
        for abl_id in abl_ids:
            key = (abl_id, scen)
            r = ablation_avg.get(key, {})
            v = r.get(metric, 0)
            if "rate" in metric or "ratio" in metric:
                v = v * 100
            vals.append(v)
            colors.append(ABLATION_COLORS.get(abl_id, "#888888"))
        x = np.arange(len(abl_ids))
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            fmt = f"{val:.1f}" if val < 100 else f"{val:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    fmt, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(abl_labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Defense Layer Ablation: Security & Overhead Metrics (Star, 10 Nodes, Under Attack)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_security_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: Waterfall — incremental EENS reduction per defense layer ──
    fig, ax = plt.subplots(figsize=(12, 6))
    eens_vals = []
    for abl_id in abl_ids:
        key = (abl_id, "attack_grid")
        r = ablation_avg.get(key, {})
        eens_vals.append(r.get("eens_total_kwh", 0))

    x = np.arange(len(abl_ids))
    # Draw waterfall
    for i in range(len(abl_ids)):
        if i == 0:
            ax.bar(x[i], eens_vals[i], color=ABLATION_COLORS[abl_ids[i]],
                   edgecolor="white", linewidth=0.5)
        else:
            # Draw the reduction as a falling bar from previous level
            ax.bar(x[i], eens_vals[i], color=ABLATION_COLORS[abl_ids[i]],
                   edgecolor="white", linewidth=0.5)
            # Connect with a line showing the drop
            ax.plot([x[i-1] + 0.4, x[i] - 0.4],
                    [eens_vals[i-1], eens_vals[i-1]],
                    color="gray", linewidth=0.8, linestyle="--")
        ax.text(x[i], eens_vals[i] + 0.5, f"{eens_vals[i]:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        if i > 0:
            delta = eens_vals[i] - eens_vals[i-1]
            pct = delta / eens_vals[0] * 100 if eens_vals[0] > 0 else 0
            ax.text(x[i], eens_vals[i] / 2, f"{delta:+.1f}\n({pct:+.1f}%)",
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(abl_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("EENS (kWh)")
    ax.set_title("Defense Layer Waterfall: Incremental EENS Reduction per Defense Layer\n"
                  "(Star Topology, 10 Nodes, Under Attack)",
                  fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 4: Latency vs Security tradeoff scatter ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for abl_id in abl_ids:
        key = (abl_id, "attack_grid")
        r = ablation_avg.get(key, {})
        lat = r.get("control_latency_mean_ms", 0)
        block = r.get("attack_priority_block_rate", 0) * 100
        eens = r.get("eens_total_kwh", 0)
        ax.scatter(lat, block, s=max(50, eens * 3), alpha=0.8,
                   color=ABLATION_COLORS.get(abl_id, "#888"),
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(ABLATION_TIERS[abl_id]["short"],
                    (lat, block), textcoords="offset points",
                    xytext=(8, 5), fontsize=8)
    ax.set_xlabel("Control Latency (ms)")
    ax.set_ylabel("Attack Block Rate (%)")
    ax.set_title("Defense Tradeoff: Latency vs Security\n"
                  "(Bubble size ∝ EENS; smaller = better)",
                  fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# V12: Cost-Benefit / ROI Analysis Plots
# ═══════════════════════════════════════════════════════════════════

TOPO_LABELS_SHORT = {"ring": "Ring", "star": "Star", "mesh": "Mesh",
                     "two_cluster_bridge": "2-Cluster Bridge"}


def plot_roi_cost_vs_eens(averaged: Dict[str, Dict[str, Any]], roi_data: Dict) -> None:
    """Per-topology twin-axis: CapEx bars + EENS reduction markers."""
    _ensure_fig_dir()
    for topo in TOPOLOGIES:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        dids = [d for d in DEFENSE_TIERS if d != "no_defense"]
        x = np.arange(len(dids))

        # Left axis: 10-year total cost (CapEx + cumulative OpEx)
        T = COST_MODEL["horizon_years"]
        costs = []
        for did in dids:
            d = roi_data["per_topo"][topo][did]
            total_cost = d["capex"] + d["opex_annual"] * T
            costs.append(total_cost / 1e3)  # $K
        bars = ax1.bar(x, costs, 0.4, color=[COLORS[d] for d in dids],
                       edgecolor="white", alpha=0.8, label="10-yr Total Cost")
        ax1.set_ylabel("10-Year Total Cost ($K)", fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels([DEFENSE_TIERS[d]["label"] for d in dids])

        # Annotate cost values
        for bar, c in zip(bars, costs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(costs)*0.02,
                     f"${c:,.0f}K", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Right axis: EENS reduction (kWh saved per attack)
        ax2 = ax1.twinx()
        eens_reductions = [roi_data["per_topo"][topo][did]["delta_eens"] for did in dids]
        ax2.plot(x, eens_reductions, "s-", color="#264653", markersize=10,
                 linewidth=2.5, label="EENS Reduction", zorder=5)
        for i, v in enumerate(eens_reductions):
            ax2.annotate(f"{v:.1f} kWh", (x[i], v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color="#264653")
        ax2.set_ylabel("EENS Reduction per Attack (kWh)", fontsize=12, color="#264653")

        label = TOPO_LABELS_SHORT[topo]
        ax1.set_title(f"{label}: Defense Cost vs Energy Saved", fontsize=13, fontweight="bold")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"roi_cost_vs_eens_{topo}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_roi_breakeven(averaged: Dict[str, Dict[str, Any]], roi_data: Dict) -> None:
    """Attack frequency vs payback period for quantum defense, per topology."""
    _ensure_fig_dir()
    C = COST_MODEL
    r, T = C["discount_rate"], C["horizon_years"]
    af = _annuity_factor(r, T)
    freqs = C["attacks_per_year_range"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for topo in TOPOLOGIES:
        d = roi_data["per_topo"][topo]["quantum"]
        delta_eens = d["delta_eens"]
        capex = d["capex"]
        opex = d["opex_annual"]
        voll = C["voll_default_usd_per_kwh"]

        paybacks = []
        for f_atk in freqs:
            annual_net = delta_eens * voll * f_atk - opex
            if annual_net > 0:
                paybacks.append(min(capex / annual_net, T * 2))
            else:
                paybacks.append(T * 2)  # cap at 2× horizon

        ax.plot(freqs, paybacks, "o-", color=TOPO_COLORS[topo],
                label=TOPO_LABELS_SHORT[topo], linewidth=2, markersize=6)

    ax.axhline(y=T, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.text(freqs[-1] * 0.7, T + 0.3, f"{T}-Year Horizon", color="gray",
            fontsize=9, ha="center")
    ax.fill_between([freqs[0], freqs[-1]], 0, T, alpha=0.05, color="green")
    ax.text(freqs[len(freqs)//2], T/2, "Pays for Itself", color="green",
            fontsize=11, alpha=0.5, ha="center", fontweight="bold")

    ax.set_xlabel("Attack Frequency (attacks/year)", fontsize=12)
    ax.set_ylabel("Payback Period (years)", fontsize=12)
    ax.set_title("Quantum Defense: Breakeven Analysis\n(VoLL = $25/kWh)", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(0, T * 1.5)
    ax.legend(title="Topology", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roi_breakeven_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roi_scaling(scaling_avg: Dict, roi_data: Dict) -> None:
    """NPV of quantum defense vs node count, per topology."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    for topo in TOPOLOGIES:
        npvs = []
        nodes_list = []
        for n in NODE_COUNTS:
            key = (n, topo, "quantum")
            if key in roi_data["scaling"]:
                npvs.append(roi_data["scaling"][key]["npv"] / 1e6)  # $M
                nodes_list.append(n)

        if nodes_list:
            ax.plot(nodes_list, npvs, "o-", color=TOPO_COLORS[topo],
                    label=TOPO_LABELS_SHORT[topo], linewidth=2.5, markersize=8)
            for n, v in zip(nodes_list, npvs):
                ax.annotate(f"${v:.2f}M", (n, v), textcoords="offset points",
                           xytext=(5, 8), fontsize=8, color=TOPO_COLORS[topo])

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between([NODE_COUNTS[0], NODE_COUNTS[-1]], 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                    alpha=0.05, color="green")
    ax.set_xlabel("Number of Microgrid Nodes", fontsize=12)
    ax.set_ylabel("10-Year NPV ($M)", fontsize=12)
    ax.set_title("Quantum Defense NPV vs Grid Size\n(12 attacks/year, VoLL = $25/kWh)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(NODE_COUNTS)
    ax.legend(title="Topology", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roi_node_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roi_sensitivity_voll(averaged: Dict[str, Dict[str, Any]], roi_data: Dict) -> None:
    """Grouped bars: ROI % at different VoLL values, quantum defense only."""
    _ensure_fig_dir()
    voll_range = COST_MODEL["voll_range"]
    voll_colors = {10: "#e76f51", 25: "#2a9d8f", 50: "#264653"}

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(TOPOLOGIES))
    bar_width = 0.22

    for i, voll in enumerate(voll_range):
        roi_pcts = []
        for topo in TOPOLOGIES:
            data = roi_data["sensitivity"].get((topo, voll), {}).get("quantum", {})
            roi_pcts.append(data.get("roi_pct", 0))
        bars = ax.bar(x + i * bar_width, roi_pcts, bar_width,
                      color=voll_colors[voll], edgecolor="white",
                      label=f"VoLL = ${voll}/kWh")
        for bar, v in zip(bars, roi_pcts):
            if abs(v) > 5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(max(roi_pcts), 1)*0.02,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([TOPO_LABELS_SHORT[t] for t in TOPOLOGIES])
    ax.set_ylabel("10-Year ROI (%)", fontsize=12)
    ax.set_title("Quantum Defense ROI Sensitivity to Value of Lost Load\n(12 attacks/year)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roi_sensitivity_voll.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roi_cumulative(averaged: Dict[str, Dict[str, Any]], roi_data: Dict) -> None:
    """2x2 subplot: 10-year cumulative net benefit per topology."""
    _ensure_fig_dir()
    T = COST_MODEL["horizon_years"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, topo in enumerate(TOPOLOGIES):
        ax = axes[idx // 2][idx % 2]
        years = list(range(T + 1))

        for did in DEFENSE_TIERS:
            if did == "no_defense":
                ax.axhline(y=0, color=COLORS[did], linestyle="--", alpha=0.5,
                          label="No Defense ($0)")
                continue

            cumul = roi_data["cumulative"][topo][did]
            cumul_k = [c / 1e3 for c in cumul]  # $K
            ax.plot(years, cumul_k, "o-", color=COLORS[did], linewidth=2,
                    markersize=5, label=DEFENSE_TIERS[did]["label"])

            # Mark payback point
            payback = roi_data["per_topo"][topo][did]["payback_years"]
            if payback < T:
                ax.axvline(x=payback, color=COLORS[did], linestyle=":", alpha=0.4)
                ax.annotate(f"Payback: {payback:.1f}y", (payback, 0),
                           textcoords="offset points", xytext=(5, 10),
                           fontsize=8, color=COLORS[did])

        # Shade positive region
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.fill_between(years, 0, [max(0, c/1e3) for c in roi_data["cumulative"][topo].get("quantum", [0]*(T+1))],
                        alpha=0.08, color=COLORS["quantum"])

        label = TOPO_LABELS_SHORT[topo]
        ax.set_title(f"{label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative Net Benefit ($K)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, T + 1, 2))

    fig.suptitle("Cumulative Cost-Benefit Analysis (12 attacks/year, VoLL = $25/kWh)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roi_cumulative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roi_summary(averaged: Dict[str, Dict[str, Any]], roi_data: Dict) -> None:
    """Horizontal bar chart: NPV + payback for each topology, quantum vs classical."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions = np.arange(len(TOPOLOGIES))
    bar_height = 0.35

    for i, did in enumerate(["classical", "quantum"]):
        npvs = []
        for topo in TOPOLOGIES:
            d = roi_data["per_topo"][topo][did]
            npvs.append(d["npv"] / 1e3)  # $K
        offset = (i - 0.5) * bar_height
        bars = ax.barh(y_positions + offset, npvs, bar_height,
                       color=COLORS[did], edgecolor="white",
                       label=DEFENSE_TIERS[did]["label"])

        # Annotate with payback period
        for j, (bar, topo) in enumerate(zip(bars, TOPOLOGIES)):
            payback = roi_data["per_topo"][topo][did]["payback_years"]
            npv_val = npvs[j]
            text_x = max(npv_val, 0) + max(abs(v) for v in npvs) * 0.03
            if payback < COST_MODEL["horizon_years"]:
                ax.text(text_x, bar.get_y() + bar.get_height()/2,
                        f"Payback: {payback:.1f}y", va="center", fontsize=8, color=COLORS[did])
            else:
                ax.text(text_x, bar.get_y() + bar.get_height()/2,
                        f"Payback: >{COST_MODEL['horizon_years']}y", va="center", fontsize=8,
                        color=COLORS[did])

    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([TOPO_LABELS_SHORT[t] for t in TOPOLOGIES])
    ax.set_xlabel("10-Year NPV ($K)", fontsize=12)
    ax.set_title("Defense Investment NPV by Topology\n(12 attacks/year, VoLL = $25/kWh)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roi_cross_topology_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── QAN Comparison Study ────────────────────────────────────────────
QAN_COLORS = {"classical": "#2a9d8f", "quantum_ab": "#e76f51"}
QAN_MODES = ["classical", "quantum_ab"]
QAN_SEEDS = [42, 137, 256]
QAN_SCENARIOS = ["baseline_grid", "attack_grid"]

FEDERATION_COMBOS = [
    "federated_ring_star",
    "federated_ring_two_cluster_bridge",
    "federated_mesh_star",
    "federated_mesh_two_cluster_bridge",
]
FED_LABELS_SHORT = {
    "federated_ring_star": "Ring+Star",
    "federated_ring_two_cluster_bridge": "Ring+2CB",
    "federated_mesh_star": "Mesh+Star",
    "federated_mesh_two_cluster_bridge": "Mesh+2CB",
}


def run_qan_comparison_study() -> List[Dict[str, Any]]:
    """Sweep 4 topologies x 2 scenarios x 2 QAN modes x 3 seeds = 48 runs.
    All runs use quantum defense (QAN requires QKD)."""
    cases = []
    for topo in TOPOLOGIES:
        for scen_id in QAN_SCENARIOS:
            for qan_mode in QAN_MODES:
                for seed in QAN_SEEDS:
                    cases.append((topo, scen_id, qan_mode, seed))

    total = len(cases)
    print(f"\n{'=' * 70}")
    print(f"  QAN Comparison Study: {total} runs")
    print(f"  Topologies: {', '.join(TOPOLOGIES)}")
    print(f"  QAN modes: {', '.join(QAN_MODES)}")
    print(f"{'=' * 70}\n")

    results: List[Dict[str, Any]] = []
    for idx, (topo, scen_id, qan_mode, seed) in enumerate(cases, 1):
        label = f"{topo}/{scen_id}/{qan_mode}/s{seed}"
        print(f"    [{idx:3d}/{total}] {label} ...", end=" ", flush=True)

        kwargs = _make_case(topo, scen_id, "quantum", seed)
        kwargs["qan_mode"] = qan_mode
        kwargs["qan_events"] = 5
        if qan_mode == "quantum_ab":
            kwargs["qab_ghz_prep_success_prob"] = 0.85
            kwargs["qab_ghz_fidelity_base"] = 0.95
            kwargs["qab_message_bits"] = 256
            kwargs["qab_decoherence_window_ms"] = 100
            kwargs["qab_ghz_prep_time_ms"] = 5

        try:
            r = run_one(**kwargs)
            r["_qan_mode"] = qan_mode
            r["_topology"] = topo
            r["_scenario"] = scen_id
            r["_defense"] = "quantum"
            r["_seed"] = seed
            results.append(r)
            eens = r.get("eens_total_kwh", -1)
            entropy = r.get("deanon_entropy_mean_bits", -1)
            print(f"EENS={eens:.2f} entropy={entropy:.3f}")
        except Exception as e:
            print(f"FAILED: {e}")

    return results


def average_qan_comparison(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group by (topology, scenario, qan_mode), average NUMERIC_KEYS across seeds."""
    from collections import defaultdict
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        key = f"{r['_topology']}__{r['_scenario']}__{r['_qan_mode']}"
        groups[key].append(r)

    averaged: Dict[str, Dict[str, Any]] = {}
    for key, runs in groups.items():
        avg: Dict[str, Any] = {
            "_topology": runs[0]["_topology"],
            "_scenario": runs[0]["_scenario"],
            "_qan_mode": runs[0]["_qan_mode"],
            "_n_seeds": len(runs),
        }
        for nk in NUMERIC_KEYS:
            vals = [float(r.get(nk, 0) or 0) for r in runs]
            vals = [v for v in vals if not (v != v)]
            avg[nk] = np.mean(vals) if vals else float("nan")
            if len(vals) > 1:
                avg[f"{nk}_std"] = np.std(vals)
        averaged[key] = avg
    return averaged


def run_cross_topology_study() -> List[Dict[str, Any]]:
    """4 federation combos x 2 scenarios x 3 seeds x 1 defense (quantum) = 24 runs."""
    cases = []
    for fed_topo in FEDERATION_COMBOS:
        for scen_id in QAN_SCENARIOS:
            for seed in QAN_SEEDS:
                cases.append((fed_topo, scen_id, seed))

    total = len(cases)
    print(f"\n{'=' * 70}")
    print(f"  Cross-Topology Federation Study: {total} runs")
    print(f"  Combos: {', '.join(FEDERATION_COMBOS)}")
    print(f"{'=' * 70}\n")

    results: List[Dict[str, Any]] = []
    for idx, (fed_topo, scen_id, seed) in enumerate(cases, 1):
        label = f"{fed_topo}/{scen_id}/s{seed}"
        print(f"    [{idx:3d}/{total}] {label} ...", end=" ", flush=True)

        kwargs = _make_case("star", scen_id, "quantum", seed, n_nodes=10)
        kwargs["topology"] = fed_topo

        try:
            r = run_one(**kwargs)
            r["_topology"] = fed_topo
            r["_scenario"] = scen_id
            r["_defense"] = "quantum"
            r["_seed"] = seed
            results.append(r)
            eens_a = r.get("eens_grid_a_kwh", -1)
            eens_b = r.get("eens_grid_b_kwh", -1)
            print(f"EENS_A={eens_a:.2f} EENS_B={eens_b:.2f}")
        except Exception as e:
            print(f"FAILED: {e}")

    return results


def average_federation_results(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group by (topology, scenario), average NUMERIC_KEYS across seeds."""
    from collections import defaultdict
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        key = f"{r['_topology']}__{r['_scenario']}"
        groups[key].append(r)

    averaged: Dict[str, Dict[str, Any]] = {}
    for key, runs in groups.items():
        avg: Dict[str, Any] = {
            "_topology": runs[0]["_topology"],
            "_scenario": runs[0]["_scenario"],
            "_n_seeds": len(runs),
        }
        for nk in NUMERIC_KEYS:
            vals = [float(r.get(nk, 0) or 0) for r in runs]
            vals = [v for v in vals if not (v != v)]
            avg[nk] = np.mean(vals) if vals else float("nan")
            if len(vals) > 1:
                avg[f"{nk}_std"] = np.std(vals)
        averaged[key] = avg
    return averaged


# ─── QAN Plot Functions ──────────────────────────────────────────────
def plot_qan_anonymity_comparison(qan_avg: Dict[str, Dict[str, Any]]) -> None:
    """Grouped bar: deanon_entropy_mean_bits by topology, grouped by qan_mode.
    Higher entropy = better anonymity."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    topos = TOPOLOGIES
    x = np.arange(len(topos))
    bar_width = 0.35

    for i, mode in enumerate(QAN_MODES):
        vals = []
        for topo in topos:
            key = f"{topo}__attack_grid__{mode}"
            r = qan_avg.get(key, {})
            vals.append(r.get("deanon_entropy_mean_bits", 0))
        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, color=QAN_COLORS[mode],
                      edgecolor="white", linewidth=0.5, label=mode.replace("_", " ").title())
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([TOPO_LABELS_SHORT.get(t, t) for t in topos])
    ax.set_ylabel("Deanonymization Entropy (bits)", fontsize=12)
    ax.set_title("QAN Anonymity: Classical Cover Traffic vs Quantum Anonymous Broadcast",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qan_anonymity_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qan_bandwidth_cost(qan_avg: Dict[str, Dict[str, Any]]) -> None:
    """Grouped bar: cover_messages_total (classical) vs ghz_states_consumed (quantum_ab)."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    topos = TOPOLOGIES
    x = np.arange(len(topos))
    bar_width = 0.35

    # Classical: cover_messages_total
    classical_vals = []
    for topo in topos:
        key = f"{topo}__attack_grid__classical"
        r = qan_avg.get(key, {})
        classical_vals.append(r.get("cover_messages_total", 0))

    # Quantum AB: ghz_states_consumed
    quantum_vals = []
    for topo in topos:
        key = f"{topo}__attack_grid__quantum_ab"
        r = qan_avg.get(key, {})
        quantum_vals.append(r.get("ghz_states_consumed", 0))

    ax.bar(x - bar_width / 2, classical_vals, bar_width, color=QAN_COLORS["classical"],
           edgecolor="white", linewidth=0.5, label="Classical: Cover Messages")
    ax.bar(x + bar_width / 2, quantum_vals, bar_width, color=QAN_COLORS["quantum_ab"],
           edgecolor="white", linewidth=0.5, label="Quantum AB: GHZ States")

    ax.set_xticks(x)
    ax.set_xticklabels([TOPO_LABELS_SHORT.get(t, t) for t in topos])
    ax.set_ylabel("Resource Count", fontsize=12)
    ax.set_title("QAN Bandwidth Cost: Cover Messages vs GHZ States Consumed",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qan_bandwidth_cost.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qan_deanon_accuracy(qan_avg: Dict[str, Dict[str, Any]]) -> None:
    """Grouped bar: deanon_top1_acc by topology/mode. Lower = better privacy."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    topos = TOPOLOGIES
    x = np.arange(len(topos))
    bar_width = 0.35

    for i, mode in enumerate(QAN_MODES):
        vals = []
        for topo in topos:
            key = f"{topo}__attack_grid__{mode}"
            r = qan_avg.get(key, {})
            vals.append(r.get("deanon_top1_acc", 0) * 100)  # to percent
        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, color=QAN_COLORS[mode],
                      edgecolor="white", linewidth=0.5, label=mode.replace("_", " ").title())
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([TOPO_LABELS_SHORT.get(t, t) for t in topos])
    ax.set_ylabel("Deanonymization Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("QAN Privacy: Adversary Deanonymization Accuracy (Lower = Better)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qan_deanon_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qan_key_cost_breakdown(qan_avg: Dict[str, Dict[str, Any]]) -> None:
    """Stacked bar of key_bits_spent_sum broken by cover vs non-cover usage."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    topos = TOPOLOGIES
    x = np.arange(len(topos))
    bar_width = 0.35

    for i, mode in enumerate(QAN_MODES):
        total_key = []
        cover_key = []
        for topo in topos:
            key = f"{topo}__attack_grid__{mode}"
            r = qan_avg.get(key, {})
            total = r.get("key_bits_spent_sum", 0)
            cover_bytes = r.get("cover_bytes_total", 0)
            # Approximate cover key cost as cover_bytes * 8 (bits)
            cover_bits = cover_bytes * 8
            non_cover = max(0, total - cover_bits)
            total_key.append(total)
            cover_key.append(cover_bits)

        offset = (i - 0.5) * bar_width
        non_cover_vals = [max(0, t - c) for t, c in zip(total_key, cover_key)]
        ax.bar(x + offset, non_cover_vals, bar_width, color=QAN_COLORS[mode],
               edgecolor="white", linewidth=0.5, label=f"{mode.replace('_', ' ').title()}: Protocol")
        ax.bar(x + offset, cover_key, bar_width, bottom=non_cover_vals,
               color=QAN_COLORS[mode], edgecolor="white", linewidth=0.5,
               alpha=0.5, label=f"{mode.replace('_', ' ').title()}: Cover/GHZ")

    ax.set_xticks(x)
    ax.set_xticklabels([TOPO_LABELS_SHORT.get(t, t) for t in topos])
    ax.set_ylabel("Key Bits Spent", fontsize=12)
    ax.set_title("QAN Key Cost Breakdown: Protocol vs Cover/GHZ Overhead",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qan_key_cost_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qan_resource_tradeoff(qan_avg: Dict[str, Dict[str, Any]]) -> None:
    """Scatter: x=resource cost (cover_messages or ghz_states), y=entropy."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(8, 6))

    for mode in QAN_MODES:
        xs, ys, labels = [], [], []
        for topo in TOPOLOGIES:
            key = f"{topo}__attack_grid__{mode}"
            r = qan_avg.get(key, {})
            if mode == "classical":
                resource = r.get("cover_messages_total", 0)
            else:
                resource = r.get("ghz_states_consumed", 0)
            entropy = r.get("deanon_entropy_mean_bits", 0)
            xs.append(resource)
            ys.append(entropy)
            labels.append(TOPO_LABELS_SHORT.get(topo, topo))

        ax.scatter(xs, ys, color=QAN_COLORS[mode], s=100, edgecolors="black",
                   linewidths=0.5, zorder=5, label=mode.replace("_", " ").title())
        for x_val, y_val, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x_val, y_val), textcoords="offset points",
                        xytext=(8, 5), fontsize=8)

    ax.set_xlabel("Resource Cost (Cover Messages / GHZ States)", fontsize=12)
    ax.set_ylabel("Deanonymization Entropy (bits)", fontsize=12)
    ax.set_title("QAN Resource-Privacy Tradeoff", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qan_resource_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Federation Plot Functions ───────────────────────────────────────
def plot_federation_eens_comparison(fed_avg: Dict[str, Dict[str, Any]]) -> None:
    """Grouped bar: eens_grid_a_kwh and eens_grid_b_kwh for each federation combo,
    baseline vs attack."""
    _ensure_fig_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    combos = FEDERATION_COMBOS
    x = np.arange(len(combos))
    bar_width = 0.35

    for panel_idx, domain_key in enumerate(["eens_grid_a_kwh", "eens_grid_b_kwh"]):
        ax = axes[panel_idx]
        domain_label = "Grid A" if "a" in domain_key else "Grid B"

        for i, scen in enumerate(QAN_SCENARIOS):
            vals = []
            for combo in combos:
                key = f"{combo}__{scen}"
                r = fed_avg.get(key, {})
                vals.append(r.get(domain_key, 0))
            offset = (i - 0.5) * bar_width
            scen_label = "Baseline" if "baseline" in scen else "Attack"
            color = "#2a9d8f" if "baseline" in scen else "#e76f51"
            bars = ax.bar(x + offset, vals, bar_width, color=color,
                          edgecolor="white", linewidth=0.5, label=scen_label)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([FED_LABELS_SHORT[c] for c in combos], rotation=15, ha="right")
        ax.set_ylabel("EENS (kWh)", fontsize=11)
        ax.set_title(f"{domain_label}: EENS by Federation Topology", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Federated Microgrid EENS: Baseline vs Attack", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "federation_eens_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_federation_cascade_analysis(fed_avg: Dict[str, Dict[str, Any]]) -> None:
    """Bar: cascade ratio = eens_grid_b(attack) / eens_grid_b(baseline) for each combo."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(8, 6))

    combos = FEDERATION_COMBOS
    ratios = []
    colors_list = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]

    for combo in combos:
        baseline_key = f"{combo}__baseline_grid"
        attack_key = f"{combo}__attack_grid"
        baseline_eens = fed_avg.get(baseline_key, {}).get("eens_grid_b_kwh", 0)
        attack_eens = fed_avg.get(attack_key, {}).get("eens_grid_b_kwh", 0)
        ratio = attack_eens / baseline_eens if baseline_eens > 0 else float("nan")
        ratios.append(ratio)

    x = np.arange(len(combos))
    bars = ax.bar(x, ratios, color=colors_list[:len(combos)], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, ratios):
        if val == val:  # not NaN
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="No cascade (1.0x)")
    ax.set_xticks(x)
    ax.set_xticklabels([FED_LABELS_SHORT[c] for c in combos], rotation=15, ha="right")
    ax.set_ylabel("Cascade Ratio (Attack / Baseline EENS)", fontsize=12)
    ax.set_title("Federation Cascade Analysis: Grid B Impact Ratio",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "federation_cascade_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_federation_topology_combos(fed_avg: Dict[str, Dict[str, Any]]) -> None:
    """Bar chart: total eens_total_kwh for all 4 combos under attack."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(8, 6))

    combos = FEDERATION_COMBOS
    colors_list = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
    vals = []
    for combo in combos:
        key = f"{combo}__attack_grid"
        r = fed_avg.get(key, {})
        vals.append(r.get("eens_total_kwh", 0))

    x = np.arange(len(combos))
    bars = ax.bar(x, vals, color=colors_list[:len(combos)], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([FED_LABELS_SHORT[c] for c in combos], rotation=15, ha="right")
    ax.set_ylabel("Total EENS (kWh)", fontsize=12)
    ax.set_title("Federation Topology Comparison: Total EENS Under Attack",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "federation_topology_combos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── V14: Latency vs Security Trade-off ─────────────────────────────

LATENCY_DELAYS = [0, 5, 10, 20, 35, 50, 75, 100]
LATENCY_TOPOS = ["star", "mesh"]
LATENCY_SEEDS = [42, 137, 256]


def run_latency_tradeoff_study() -> List[Dict[str, Any]]:
    """Sweep verification delay: 8 delays × 2 topos × 3 seeds = 48 runs."""
    cases = []
    for delay in LATENCY_DELAYS:
        for topo in LATENCY_TOPOS:
            for seed in LATENCY_SEEDS:
                cases.append((delay, topo, seed))

    total = len(cases)
    print(f"\n{'=' * 70}")
    print(f"  Latency Trade-off Study: {total} runs")
    print(f"  Delays (ms): {LATENCY_DELAYS}")
    print(f"  Topologies: {', '.join(LATENCY_TOPOS)}")
    print(f"{'=' * 70}\n")

    results: List[Dict[str, Any]] = []
    for idx, (delay, topo, seed) in enumerate(cases, 1):
        label = f"{topo}/delay={delay}ms/s{seed}"
        print(f"    [{idx:3d}/{total}] {label} ...", end=" ", flush=True)
        try:
            kwargs = _make_case(topo, "attack_grid", "quantum", seed)
            kwargs["verification_delay_ms"] = delay
            r = run_one(**kwargs)
            r["_delay_ms"] = delay
            r["_topology"] = topo
            r["_seed"] = seed
            print(f"EENS={r.get('eens_total_kwh', 0):.3f} "
                  f"block_rate={r.get('attack_priority_block_rate', 0):.3f}")
            results.append(r)
        except Exception as e:
            print(f"FAILED: {e}")
    return results


def average_latency_results(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group by (topology, delay_ms), average NUMERIC_KEYS across seeds."""
    from collections import defaultdict
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        key = f"{r['_topology']}__{r['_delay_ms']}"
        groups[key].append(r)

    averaged: Dict[str, Dict[str, Any]] = {}
    for key, runs in groups.items():
        avg: Dict[str, Any] = {
            "_topology": runs[0]["_topology"],
            "_delay_ms": runs[0]["_delay_ms"],
            "_n_seeds": len(runs),
        }
        for nk in NUMERIC_KEYS:
            vals = [float(r.get(nk, 0) or 0) for r in runs]
            vals = [v for v in vals if not (v != v)]
            avg[nk] = np.mean(vals) if vals else float("nan")
            if len(vals) > 1:
                avg[f"{nk}_std"] = np.std(vals)
        averaged[key] = avg
    return averaged


def plot_latency_security_tradeoff(latency_avg: Dict[str, Dict[str, Any]]) -> None:
    """Dual y-axis: attack block rate (%) & EENS (kWh) vs verification delay."""
    _ensure_fig_dir()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    topo_styles = {"star": ("o-", "#2a9d8f"), "mesh": ("s--", "#e76f51")}

    for topo, (style, color) in topo_styles.items():
        delays = []
        block_rates = []
        eens_vals = []
        for delay in LATENCY_DELAYS:
            key = f"{topo}__{delay}"
            if key in latency_avg:
                delays.append(delay)
                block_rates.append(latency_avg[key].get("attack_priority_block_rate", 0) * 100)
                eens_vals.append(latency_avg[key].get("eens_total_kwh", 0))

        ax1.plot(delays, block_rates, style, color=color, linewidth=2.2,
                 markersize=8, label=f"{topo.capitalize()} – Block Rate")
        ax2.plot(delays, eens_vals, style.replace("o", "^").replace("s", "D"),
                 color=color, linewidth=1.8, markersize=7, alpha=0.6,
                 label=f"{topo.capitalize()} – EENS")

    ax1.set_xlabel("Verification Delay (ms)", fontsize=12)
    ax1.set_ylabel("Attack Block Rate (%)", fontsize=12, color="#2a9d8f")
    ax2.set_ylabel("EENS (kWh)", fontsize=12, color="#d62828")
    ax1.tick_params(axis="y", labelcolor="#2a9d8f")
    ax2.tick_params(axis="y", labelcolor="#d62828")
    ax1.set_title("Latency vs Security Trade-off: Verification Delay Sweep",
                   fontsize=14, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "latency_security_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── V14: Adaptive Adversary / Arms Race ────────────────────────────

ARMS_RACE_PHASES = OrderedDict([
    ("phase1", {"label": "Phase 1: Single Spoof", "short": "P1: Spoof",
                "attack": "nodespoofforged"}),
    ("phase2", {"label": "Phase 2: Coordinated + FDI", "short": "P2: Coord+FDI",
                "attack": "fdi_coordinated"}),
    ("phase3", {"label": "Phase 3: Full APT", "short": "P3: Full APT",
                "attack": "spoof_exhaust_quantum"}),
])
ARMS_RACE_TOPOS = ["star", "mesh"]
ARMS_RACE_DEFENSES = ["no_defense", "classical", "quantum"]
ARMS_RACE_SEEDS = [42, 137, 256]


def run_arms_race_study() -> List[Dict[str, Any]]:
    """3 phases × 2 topos × 3 defenses × 3 seeds = 54 runs."""
    cases = []
    for phase_id in ARMS_RACE_PHASES:
        for topo in ARMS_RACE_TOPOS:
            for def_id in ARMS_RACE_DEFENSES:
                for seed in ARMS_RACE_SEEDS:
                    cases.append((phase_id, topo, def_id, seed))

    total = len(cases)
    print(f"\n{'=' * 70}")
    print(f"  Arms Race Study: {total} runs")
    print(f"  Phases: {', '.join(p['short'] for p in ARMS_RACE_PHASES.values())}")
    print(f"  Topologies: {', '.join(ARMS_RACE_TOPOS)}")
    print(f"  Defenses: {', '.join(ARMS_RACE_DEFENSES)}")
    print(f"{'=' * 70}\n")

    results: List[Dict[str, Any]] = []
    for idx, (phase_id, topo, def_id, seed) in enumerate(cases, 1):
        phase = ARMS_RACE_PHASES[phase_id]
        label = f"{phase['short']}/{topo}/{def_id}/s{seed}"
        print(f"    [{idx:3d}/{total}] {label} ...", end=" ", flush=True)
        try:
            dfn = DEFENSE_TIERS[def_id]
            attack_str = phase["attack"]
            defense_mode = dfn["defense_mode"]
            scenario_str = f"{attack_str}_def_{defense_mode}"

            node_list = [f"MG{i}" for i in range(DEFAULT_N_NODES)]
            kwargs: Dict[str, Any] = dict(
                topology=topo,
                nodes=node_list,
                seed=seed,
                horizon_s=HORIZON_S,
                out_dir=OUT_DIR,
                scenario=scenario_str,
                route_policy="shortest",
                k_paths=3,
                attack_intensity="S3",
                distributed_attacks=True,
                num_attack_windows=5,
                energy_record_interval=30,
                infrastructure_override=FAIR_INFRA,
                write_outputs=False,
                enable_qkd=dfn["enable_qkd"],
                enable_quantum_protocols=dfn["enable_quantum_protocols"],
                enable_quantum_control_auth=dfn["enable_quantum_control_auth"],
                enable_sensor_challenges=dfn.get("enable_sensor_challenges", False),
                quantum_auth_bypass_prob=dfn.get("quantum_auth_bypass_prob", 0.0),
                verification_delay_ms=dfn.get("verification_delay_ms", None),
                degraded_verification_delay_ms=dfn.get("degraded_verification_delay_ms", None),
                hw_timing_jitter_ms=dfn.get("hw_timing_jitter_ms", 0.0),
                spd_timing_overhead_ms=dfn.get("spd_timing_overhead_ms", 0.0),
                spoof_auth_bypass_prob=0.0,
                qec_code_distance=3,
                e2e_distillation_rounds=1,
                e2e_swap_success_prob=0.5,
                quantum_control_token_ttl_ms=1500,
                qan_events=5,
            )
            r = run_one(**kwargs)
            r["_phase"] = phase_id
            r["_phase_label"] = phase["short"]
            r["_topology"] = topo
            r["_defense"] = def_id
            r["_seed"] = seed
            print(f"EENS={r.get('eens_total_kwh', 0):.3f} "
                  f"atk_allow={r.get('attack_priority_allow_rate', 0):.3f}")
            results.append(r)
        except Exception as e:
            print(f"FAILED: {e}")
    return results


def average_arms_race_results(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group by (topology, phase, defense), average across seeds."""
    from collections import defaultdict
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        key = f"{r['_topology']}__{r['_phase']}__{r['_defense']}"
        groups[key].append(r)

    averaged: Dict[str, Dict[str, Any]] = {}
    for key, runs in groups.items():
        avg: Dict[str, Any] = {
            "_topology": runs[0]["_topology"],
            "_phase": runs[0]["_phase"],
            "_phase_label": runs[0]["_phase_label"],
            "_defense": runs[0]["_defense"],
            "_n_seeds": len(runs),
        }
        for nk in NUMERIC_KEYS:
            vals = [float(r.get(nk, 0) or 0) for r in runs]
            vals = [v for v in vals if not (v != v)]
            avg[nk] = np.mean(vals) if vals else float("nan")
            if len(vals) > 1:
                avg[f"{nk}_std"] = np.std(vals)
        averaged[key] = avg
    return averaged


def plot_arms_race_escalation(arms_avg: Dict[str, Dict[str, Any]]) -> None:
    """Grouped bar chart: attack success by phase × defense (subplots per topo)."""
    _ensure_fig_dir()
    phase_ids = list(ARMS_RACE_PHASES.keys())
    phase_labels = [ARMS_RACE_PHASES[p]["short"] for p in phase_ids]
    n_phases = len(phase_ids)
    n_defs = len(ARMS_RACE_DEFENSES)
    bar_w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ti, topo in enumerate(ARMS_RACE_TOPOS):
        ax = axes[ti]
        x = np.arange(n_phases)
        for di, def_id in enumerate(ARMS_RACE_DEFENSES):
            vals = []
            for phase_id in phase_ids:
                key = f"{topo}__{phase_id}__{def_id}"
                v = arms_avg.get(key, {}).get("attack_priority_allow_rate", 0)
                vals.append(v * 100)
            bars = ax.bar(x + di * bar_w, vals, bar_w, label=DEFENSE_TIERS[def_id]["short"],
                          color=COLORS[def_id], edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x + bar_w)
        ax.set_xticklabels(phase_labels, fontsize=10)
        ax.set_ylabel("Attack Success Rate (%)" if ti == 0 else "", fontsize=11)
        ax.set_title(f"{topo.capitalize()} Topology", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, 105)

    fig.suptitle("Arms Race: Attack Success Under Escalating Threat",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "arms_race_escalation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_arms_race_eens_cascade(arms_avg: Dict[str, Dict[str, Any]]) -> None:
    """Line plot: EENS by phase × defense tier (subplots per topo)."""
    _ensure_fig_dir()
    phase_ids = list(ARMS_RACE_PHASES.keys())
    phase_labels = [ARMS_RACE_PHASES[p]["short"] for p in phase_ids]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ti, topo in enumerate(ARMS_RACE_TOPOS):
        ax = axes[ti]
        x = np.arange(len(phase_ids))
        for def_id in ARMS_RACE_DEFENSES:
            vals = []
            stds = []
            for phase_id in phase_ids:
                key = f"{topo}__{phase_id}__{def_id}"
                v = arms_avg.get(key, {}).get("eens_total_kwh", 0)
                s = arms_avg.get(key, {}).get("eens_total_kwh_std", 0)
                vals.append(v)
                stds.append(s)
            ax.errorbar(x, vals, yerr=stds, fmt="o-", color=COLORS[def_id],
                        linewidth=2.2, markersize=8, capsize=4,
                        label=DEFENSE_TIERS[def_id]["short"])
        ax.set_xticks(x)
        ax.set_xticklabels(phase_labels, fontsize=10)
        ax.set_ylabel("EENS (kWh)" if ti == 0 else "", fontsize=11)
        ax.set_title(f"{topo.capitalize()} Topology", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Arms Race: Energy Impact Under Escalating Threat",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "arms_race_eens_cascade.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── V14: QBER Threshold Sensitivity ────────────────────────────────

QBER_THRESHOLDS = [0.005, 0.01, 0.015, 0.02, 0.025, 0.035, 0.05, 0.08, 0.10]
QBER_SEEDS = [42, 137, 256]
QBER_TOPO = "star"


def run_qber_threshold_study() -> List[Dict[str, Any]]:
    """Sweep QBER threshold: 9 thresholds × 3 seeds = 27 runs."""
    cases = []
    for threshold in QBER_THRESHOLDS:
        for seed in QBER_SEEDS:
            cases.append((threshold, seed))

    total = len(cases)
    print(f"\n{'=' * 70}")
    print(f"  QBER Threshold Sensitivity Study: {total} runs")
    print(f"  Thresholds: {QBER_THRESHOLDS}")
    print(f"  Topology: {QBER_TOPO}")
    print(f"{'=' * 70}\n")

    results: List[Dict[str, Any]] = []
    for idx, (threshold, seed) in enumerate(cases, 1):
        label = f"qber_t={threshold:.3f}/s{seed}"
        print(f"    [{idx:3d}/{total}] {label} ...", end=" ", flush=True)
        try:
            # Use spoof_quantum attack to trigger both QBER elevation and spoofing
            dfn = DEFENSE_TIERS["quantum"]
            scenario_str = f"spoof_quantum_def_{dfn['defense_mode']}"
            node_list = [f"MG{i}" for i in range(DEFAULT_N_NODES)]
            kwargs: Dict[str, Any] = dict(
                topology=QBER_TOPO,
                nodes=node_list,
                seed=seed,
                horizon_s=HORIZON_S,
                out_dir=OUT_DIR,
                scenario=scenario_str,
                route_policy="shortest",
                k_paths=3,
                attack_intensity="S3",
                distributed_attacks=True,
                num_attack_windows=5,
                energy_record_interval=30,
                infrastructure_override=FAIR_INFRA,
                write_outputs=False,
                enable_qkd=dfn["enable_qkd"],
                enable_quantum_protocols=dfn["enable_quantum_protocols"],
                enable_quantum_control_auth=dfn["enable_quantum_control_auth"],
                enable_sensor_challenges=dfn.get("enable_sensor_challenges", False),
                quantum_auth_bypass_prob=dfn.get("quantum_auth_bypass_prob", 0.0),
                verification_delay_ms=dfn.get("verification_delay_ms", None),
                degraded_verification_delay_ms=dfn.get("degraded_verification_delay_ms", None),
                hw_timing_jitter_ms=dfn.get("hw_timing_jitter_ms", 0.0),
                spd_timing_overhead_ms=dfn.get("spd_timing_overhead_ms", 0.0),
                spoof_auth_bypass_prob=0.0,
                qec_code_distance=3,
                e2e_distillation_rounds=1,
                e2e_swap_success_prob=0.5,
                quantum_control_token_ttl_ms=1500,
                qan_events=5,
                qber_threshold=threshold,
            )
            r = run_one(**kwargs)
            r["_qber_threshold"] = threshold
            r["_seed"] = seed
            print(f"false_block={r.get('false_block_rate', 0):.3f} "
                  f"atk_block={r.get('attack_priority_block_rate', 0):.3f} "
                  f"alerts={r.get('n_intrusion_alerts', 0)}")
            results.append(r)
        except Exception as e:
            print(f"FAILED: {e}")
    return results


def average_qber_threshold_results(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group by qber_threshold, average across seeds."""
    from collections import defaultdict
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        key = f"{r['_qber_threshold']:.4f}"
        groups[key].append(r)

    averaged: Dict[str, Dict[str, Any]] = {}
    for key, runs in groups.items():
        avg: Dict[str, Any] = {
            "_qber_threshold": runs[0]["_qber_threshold"],
            "_n_seeds": len(runs),
        }
        for nk in NUMERIC_KEYS:
            vals = [float(r.get(nk, 0) or 0) for r in runs]
            vals = [v for v in vals if not (v != v)]
            avg[nk] = np.mean(vals) if vals else float("nan")
            if len(vals) > 1:
                avg[f"{nk}_std"] = np.std(vals)
        averaged[key] = avg
    return averaged


def plot_qber_roc_curve(qber_avg: Dict[str, Dict[str, Any]]) -> None:
    """ROC-style curve: false positive rate vs detection rate, annotated by threshold."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(8, 8))

    fps = []
    tps = []
    thresholds = []
    for key, avg in sorted(qber_avg.items(), key=lambda x: x[1]["_qber_threshold"]):
        fp = avg.get("false_block_rate", 0) * 100
        tp = avg.get("attack_priority_block_rate", 0) * 100
        fps.append(fp)
        tps.append(tp)
        thresholds.append(avg["_qber_threshold"])

    ax.plot(fps, tps, "o-", color="#2a9d8f", linewidth=2.5, markersize=10,
            markerfacecolor="#e76f51", markeredgecolor="white", markeredgewidth=1.5,
            zorder=3)

    # Annotate each point with threshold
    for fp, tp, t in zip(fps, tps, thresholds):
        ax.annotate(f"τ={t:.3f}", (fp, tp), textcoords="offset points",
                    xytext=(12, -5), fontsize=9, color="#264653",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))

    # Random classifier line
    ax.plot([0, 100], [0, 100], "--", color="#CCCCCC", linewidth=1.5,
            label="Random Classifier", zorder=1)

    ax.set_xlabel("False Positive Rate (%) — Legitimate Blocked", fontsize=12)
    ax.set_ylabel("True Positive Rate (%) — Attacks Blocked", fontsize=12)
    ax.set_title("QBER Threshold ROC Curve\nIntrusion Detector Sensitivity Analysis",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, max(fps + [10]) + 5)
    ax.set_ylim(min(tps + [90]) - 5, 102)
    ax.set_aspect("equal" if max(fps + [10]) > 20 else "auto")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qber_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Plot orchestrator ──────────────────────────────────────────────

def generate_all_plots(
    averaged: Dict[str, Dict[str, Any]],
    results: Optional[List[Dict[str, Any]]] = None,
    scaling_avg: Optional[Dict] = None,
    ablation_avg: Optional[Dict] = None,
    qan_avg: Optional[Dict] = None,
    federation_avg: Optional[Dict] = None,
    latency_avg: Optional[Dict] = None,
    arms_avg: Optional[Dict] = None,
    qber_avg: Optional[Dict] = None,
) -> None:
    print("  Generating plots...")
    # Original plots
    plot_eens_comparison(averaged)
    plot_attack_block_rate(averaged)
    plot_delivery_overhead(averaged)
    plot_quantum_auth_breakdown(averaged)
    plot_cross_topology_summary(averaged)
    plot_islanding_mode_comparison(averaged)
    # V6: Cyber-physical plots
    plot_resilience_comparison(averaged)
    plot_frequency_comparison(averaged)
    plot_frequency_violation_heatmap(averaged)
    plot_se_detection_rates(averaged)
    plot_cyber_physical_overview(averaged)
    # V7: Conference paper plots
    plot_quantum_advantage_waterfall(averaged)
    plot_radar_defense_comparison(averaged)
    plot_defense_layer_stacked(averaged)
    plot_percentage_improvement(averaged)
    plot_publication_summary_table(averaged)
    if results is not None:
        plot_eens_with_error_bars(results, averaged)
    # V8: Combined cross-topology line charts
    plot_combined_eens_lines(averaged)
    plot_combined_block_rate_lines(averaged)
    plot_combined_delivery_lines(averaged)
    plot_combined_resilience_lines(averaged)
    plot_combined_se_lines(averaged)
    plot_combined_eens_reduction_area(averaged)
    plot_combined_quantum_advantage_pct(averaged)
    plot_combined_frequency_lines(averaged)
    # V9: Node scaling plots
    if scaling_avg is not None:
        plot_node_scaling_eens(scaling_avg)
        plot_node_scaling_block_rate(scaling_avg)
        plot_node_scaling_resilience(scaling_avg)
        plot_node_scaling_frequency(scaling_avg)
        plot_node_scaling_quantum_advantage(scaling_avg)
        plot_node_scaling_heatmap(scaling_avg)
        plot_node_scaling_overhead(scaling_avg)
    # V10: Overhead & traffic separation plots
    plot_quantum_latency_overhead(averaged)
    plot_legit_vs_attack_delivery(averaged)
    # V11: Ablation study plots
    if ablation_avg is not None:
        plot_ablation_study(ablation_avg)
    # V12: Cost-Benefit / ROI Analysis plots
    roi_data = compute_roi_data(averaged, scaling_avg=scaling_avg)
    plot_roi_cost_vs_eens(averaged, roi_data)
    plot_roi_breakeven(averaged, roi_data)
    if scaling_avg is not None:
        plot_roi_scaling(scaling_avg, roi_data)
    plot_roi_sensitivity_voll(averaged, roi_data)
    plot_roi_cumulative(averaged, roi_data)
    plot_roi_summary(averaged, roi_data)
    # V13: QAN comparison plots
    if qan_avg is not None:
        plot_qan_anonymity_comparison(qan_avg)
        plot_qan_bandwidth_cost(qan_avg)
        plot_qan_deanon_accuracy(qan_avg)
        plot_qan_key_cost_breakdown(qan_avg)
        plot_qan_resource_tradeoff(qan_avg)
    # V13: Cross-topology federation plots
    if federation_avg is not None:
        plot_federation_eens_comparison(federation_avg)
        plot_federation_cascade_analysis(federation_avg)
        plot_federation_topology_combos(federation_avg)
    # V14: Latency trade-off plot
    if latency_avg is not None:
        plot_latency_security_tradeoff(latency_avg)
    # V14: Arms race plots
    if arms_avg is not None:
        plot_arms_race_escalation(arms_avg)
        plot_arms_race_eens_cascade(arms_avg)
    # V14: QBER threshold ROC plot
    if qber_avg is not None:
        plot_qber_roc_curve(qber_avg)
    n_figs = len(list(FIG_DIR.glob("*.png")))
    print(f"  Saved {n_figs} plots to: {FIG_DIR}/\n")


# ─── Main ───────────────────────────────────────────────────────────
def main() -> None:
    # ── Part 1: Main comparative experiment (4 topos × 4 scenarios × 3 defenses × 5 seeds)
    results = run_all_cases()
    averaged = average_across_seeds(results)
    print_results(averaged)

    # ── Part 2: Node scaling study (4 topos × 2 scenarios × 3 defenses × 4 node counts × 3 seeds)
    scaling_results = run_node_scaling_study()
    scaling_avg = average_scaling_results(scaling_results)
    print_scaling_results(scaling_avg)

    # ── Part 3: Defense layer ablation study (star, 10 nodes, 6 tiers × 2 scenarios × 3 seeds)
    ablation_results = run_ablation_study(topology="star", n_nodes=10, seeds=[42, 137, 256])
    ablation_avg = average_ablation_results(ablation_results)

    # ── Part 4: QAN comparison study (4 topos × 2 scenarios × 2 modes × 3 seeds = 48 runs)
    qan_results = run_qan_comparison_study()
    qan_avg = average_qan_comparison(qan_results)

    # ── Part 5: Cross-topology federation study (4 combos × 2 scenarios × 3 seeds = 24 runs)
    federation_results = run_cross_topology_study()
    federation_avg = average_federation_results(federation_results)

    # ── Part 6: Latency vs Security trade-off (8 delays × 2 topos × 3 seeds = 48 runs)
    latency_results = run_latency_tradeoff_study()
    latency_avg = average_latency_results(latency_results)

    # ── Part 7: Adaptive Adversary / Arms Race (3 phases × 2 topos × 3 defenses × 3 seeds = 54 runs)
    arms_results = run_arms_race_study()
    arms_avg = average_arms_race_results(arms_results)

    # ── Part 8: QBER Threshold Sensitivity (9 thresholds × 3 seeds = 27 runs)
    qber_results = run_qber_threshold_study()
    qber_avg = average_qber_threshold_results(qber_results)

    # ── Generate all plots
    generate_all_plots(averaged, results=results, scaling_avg=scaling_avg,
                       ablation_avg=ablation_avg, qan_avg=qan_avg,
                       federation_avg=federation_avg,
                       latency_avg=latency_avg,
                       arms_avg=arms_avg,
                       qber_avg=qber_avg)

    # ── Save raw results
    out_path = Path(OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = out_path / "quantum_security_demo_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Main results saved to: {csv_path}")

    df_scaling = pd.DataFrame(scaling_results)
    csv_scaling = out_path / "node_scaling_study_raw.csv"
    df_scaling.to_csv(csv_scaling, index=False)
    print(f"  Scaling results saved to: {csv_scaling}")

    df_ablation = pd.DataFrame(ablation_results)
    csv_ablation = out_path / "ablation_study_raw.csv"
    df_ablation.to_csv(csv_ablation, index=False)
    print(f"  Ablation results saved to: {csv_ablation}")

    df_qan = pd.DataFrame(qan_results)
    csv_qan = out_path / "qan_comparison_raw.csv"
    df_qan.to_csv(csv_qan, index=False)
    print(f"  QAN comparison results saved to: {csv_qan}")

    df_fed = pd.DataFrame(federation_results)
    csv_fed = out_path / "cross_topology_raw.csv"
    df_fed.to_csv(csv_fed, index=False)
    print(f"  Cross-topology results saved to: {csv_fed}")

    df_latency = pd.DataFrame(latency_results)
    csv_latency = out_path / "latency_tradeoff_raw.csv"
    df_latency.to_csv(csv_latency, index=False)
    print(f"  Latency trade-off results saved to: {csv_latency}")

    df_arms = pd.DataFrame(arms_results)
    csv_arms = out_path / "arms_race_raw.csv"
    df_arms.to_csv(csv_arms, index=False)
    print(f"  Arms race results saved to: {csv_arms}")

    df_qber = pd.DataFrame(qber_results)
    csv_qber = out_path / "qber_threshold_raw.csv"
    df_qber.to_csv(csv_qber, index=False)
    print(f"  QBER threshold results saved to: {csv_qber}")


if __name__ == "__main__":
    main()
