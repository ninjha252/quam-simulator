#!/usr/bin/env python3
"""
Master re-run script: 1-hour horizon, STRESSED infrastructure
=============================================================
Physical microgrid params are tightened so islanding and attacks
produce visible dynamics in energy figures.

Changes from FAIR run:
  - battery_capacity_kwh:  100 → 60
  - battery_init_kwh:       50 → 20
  - import_cap_kw:           60 → 35
  - smr_capacity_kw:         60 → 40
  - QKD infra: capacity=10000, refill=1000 (moderate stress)
"""
import os, sys, time, csv, tempfile, glob, json, shutil
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quam_microgrid"))

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
HORIZON_S = 3600
ISLAND_START_S = 1200
ISLAND_DURATION_S = 600

OUT_BASE = ROOT / "outputs_finalrun" / "quantum_security_demo"
FINAL_DIR = OUT_BASE / "final_new"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# Stressed physical microgrid parameters
MG_STRESS = {
    "battery_capacity_kwh": 60.0,      # was 100
    "battery_init_kwh": 20.0,          # was 50
    "import_cap_kw": 35.0,             # was 60
    "smr_capacity_kw": 40.0,           # was 60
}
# Moderately stressed QKD infra
STRESSED_INFRA = {"capacity": 10000, "refill": 1000, "init_fill_ratio": 0.70}

# ═══════════════════════════════════════════════════════════════
# PHASE 0: Monkey-patch the main study constants
# ═══════════════════════════════════════════════════════════════
import run_quantum_security_demo as demo
demo.HORIZON_S = HORIZON_S
demo.ISLAND_START_S = ISLAND_START_S
demo.ISLAND_DURATION_S = ISLAND_DURATION_S
demo.OUT_DIR = str(FINAL_DIR)
demo.FIG_DIR = FINAL_DIR / "figures"
demo.FIG_DIR.mkdir(parents=True, exist_ok=True)

# Patch FAIR_INFRA and MG_PARAMS in the demo module
demo.FAIR_INFRA = STRESSED_INFRA
if hasattr(demo, "DEFAULT_INFRA"):
    demo.DEFAULT_INFRA = STRESSED_INFRA

# Patch _make_case: replace infra + inject microgrid_param_overrides
_orig_make_case = demo._make_case
def _patched_make_case(topology, scenario_id, defense_id, seed, n_nodes=5):
    result = _orig_make_case(topology, scenario_id, defense_id, seed, n_nodes)
    result["infrastructure_override"] = STRESSED_INFRA
    result["microgrid_param_overrides"] = MG_STRESS
    return result

if hasattr(demo, "_make_case"):
    demo._make_case = _patched_make_case
    print("  Patched _make_case with stressed MG params")

print("=" * 70)
print(f"  QuAM Full Re-run — STRESSED infrastructure — {HORIZON_S}s")
print(f"  MG stress: battery=60kWh(init 20), import=35kW, SMR=40kW")
print(f"  QKD stress: cap=10000, refill=1000, init=70%")
print(f"  Output: {FINAL_DIR}")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Run the full study
# ═══════════════════════════════════════════════════════════════
t0 = time.time()
print("\n[PHASE 1] Running full quantum_security_demo study (stressed)...")
try:
    demo.main()
except SystemExit:
    pass
t1 = time.time()
print(f"\n  Phase 1 done in {t1-t0:.0f}s")

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Extra seeds for node scaling
# ═══════════════════════════════════════════════════════════════
print("\n[PHASE 2] Running extra seeds for node scaling...")

from quam.finalmain import run_one

TOPOLOGIES = ["ring", "star", "mesh", "two_cluster_bridge"]
NODE_COUNTS = [5, 10, 15, 20]
DEFENSE_MAP = {
    "none":      {"scenario_suffix": "def_none",        "enable_qkd": False, "enable_quantum_protocols": False, "enable_quantum_control_auth": False},
    "classical": {"scenario_suffix": "def_hardened_v3", "enable_qkd": False, "enable_quantum_protocols": False, "enable_quantum_control_auth": False},
    "quantum":   {"scenario_suffix": "def_hardened_v3", "enable_qkd": True,  "enable_quantum_protocols": True,  "enable_quantum_control_auth": True},
}
EXTRA_SEEDS = [691, 823, 999, 1117, 1331, 1543, 1777]
ORIG_SEEDS = [42, 137, 256]
ALL_SEEDS = ORIG_SEEDS + EXTRA_SEEDS

existing_csv = FINAL_DIR / "node_scaling_study_raw.csv"
existing_keys = set()
if existing_csv.exists():
    with open(existing_csv) as f:
        for r in csv.DictReader(f):
            existing_keys.add((r.get("_topology",""), r.get("_n_nodes",""),
                              r.get("_scenario",""), r.get("_defense",""), r.get("seed","")))

extra_rows = []
configs = []
for topo in TOPOLOGIES:
    for n in NODE_COUNTS:
        for scen in ["baseline_grid", "attack_grid"]:
            for def_id, dcfg in DEFENSE_MAP.items():
                for seed in ALL_SEEDS:
                    key = (topo, str(n), scen, def_id, str(seed))
                    if key not in existing_keys:
                        configs.append((topo, n, scen, def_id, dcfg, seed))

total_extra = len(configs)
print(f"  Need to run {total_extra} extra configs")

for ci, (topo, n, scen, def_id, dcfg, seed) in enumerate(configs):
    nodes = [f"MG{i}" for i in range(n)]
    if "attack" in scen:
        scenario_str = f"fdi_spoof_exhaust_qdisturb_{dcfg['scenario_suffix']}"
    else:
        scenario_str = "baseline"

    kwargs = {
        "topology": topo, "nodes": nodes, "seed": seed,
        "horizon_s": HORIZON_S, "out_dir": tempfile.mkdtemp(),
        "scenario": scenario_str,
        "infrastructure_override": STRESSED_INFRA,
        "microgrid_param_overrides": MG_STRESS,
        "write_outputs": False,
    }
    if dcfg.get("enable_qkd"):
        kwargs["enable_qkd"] = True
    if dcfg.get("enable_quantum_protocols"):
        kwargs["enable_quantum_protocols"] = True
    if dcfg.get("enable_quantum_control_auth"):
        kwargs["enable_quantum_control_auth"] = True

    t_s = time.time()
    try:
        result = run_one(**kwargs)
        result["_topology"] = topo
        result["_n_nodes"] = n
        result["_scenario"] = scen
        result["_defense"] = def_id
        result["seed"] = seed
        extra_rows.append(result)
        dt = time.time() - t_s
        label = f"n{n}_{topo}__{scen}__{def_id}__s{seed}"
        print(f"  [{ci+1:3d}/{total_extra}] {label:55s} EENS={result.get('eens_total_kwh',0):8.3f} kWh  ({dt:.1f}s)")
    except Exception as e:
        print(f"  [{ci+1}/{total_extra}] FAILED: {e}")
    shutil.rmtree(kwargs["out_dir"], ignore_errors=True)

if extra_rows:
    print(f"  Merging {len(extra_rows)} extra rows...")
    if existing_csv.exists():
        with open(existing_csv) as f:
            existing_rows = list(csv.DictReader(f))
    else:
        existing_rows = []
    all_rows = existing_rows + [{k: str(v) for k, v in r.items()} for r in extra_rows]
    all_keys = sorted(set().union(*(r.keys() for r in all_rows)))
    with open(existing_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(all_rows)
    print(f"  Merged: {len(all_rows)} total rows → {existing_csv}")

t2 = time.time()
print(f"\n  Phase 2 done in {t2-t1:.0f}s")

# ═══════════════════════════════════════════════════════════════
# PHASE 3: Generate ALL figures
# ═══════════════════════════════════════════════════════════════
print("\n[PHASE 3] Generating all figures...")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAT_DIR = FINAL_DIR / "materials"
MAT_DIR.mkdir(parents=True, exist_ok=True)

# Global: bump legend font for all subsequent figures
plt.rcParams.update({"legend.fontsize": 10, "font.size": 12})

# ── 3a: v3 figures ──
print("\n  [3a] v3 main figures...")
import generate_paper_materials_v3 as v3
v3.MAT_DIR = MAT_DIR
v3.HORIZON_S = HORIZON_S
v3.RAW_CSV = str(FINAL_DIR / "quantum_security_demo_raw.csv")
for attr in ["ABLATION_CSV", "ARMS_CSV", "TOPO_CSV", "SCALING_CSV", "LATENCY_CSV", "QBER_CSV", "QAN_CSV"]:
    csv_name = getattr(v3, attr, "").split("/")[-1] if hasattr(v3, attr) else ""
    if csv_name:
        setattr(v3, attr, str(FINAL_DIR / csv_name))
try:
    v3.main()
except Exception as e:
    print(f"  v3 error: {e}")
    import traceback; traceback.print_exc()

# ── 3b: v4 security figures ──
print("\n  [3b] v4 security figures...")
import generate_paper_materials_v4_security as v4
v4.MAT_DIR = MAT_DIR
v4.BASE = str(FINAL_DIR)
try:
    v4.main()
except Exception as e:
    print(f"  v4 error: {e}")
    import traceback; traceback.print_exc()

# ── 3c: v5b quantum overhead figures ──
print("\n  [3c] v5b quantum overhead figures...")
import generate_paper_materials_v5b_quantum_overhead as v5b
v5b.MAT_DIR = MAT_DIR
v5b.SCALING_CSV = str(FINAL_DIR / "node_scaling_study_raw.csv")
try:
    v5b.main()
except Exception as e:
    print(f"  v5b error: {e}")
    import traceback; traceback.print_exc()

# ── 3d: Energy physical figure (STRESSED — visible islanding) ──
print("\n  [3d] Energy physical figure (stressed infra)...")
from quam.finalmain import run_one as run_one_sim

def gen_energy_physical():
    tmp1 = tempfile.mkdtemp()
    print("    Running grid-connected baseline...")
    run_one_sim(topology="star", nodes=[f"MG{i}" for i in range(5)], seed=42,
                horizon_s=HORIZON_S, out_dir=tmp1, scenario="baseline",
                infrastructure_override=STRESSED_INFRA,
                microgrid_param_overrides=MG_STRESS,
                write_outputs=True)

    tmp2 = tempfile.mkdtemp()
    print("    Running islanded baseline...")
    run_one_sim(topology="star", nodes=[f"MG{i}" for i in range(5)], seed=42,
                horizon_s=HORIZON_S, out_dir=tmp2, scenario="baseline",
                infrastructure_override=STRESSED_INFRA,
                microgrid_param_overrides=MG_STRESS,
                write_outputs=True,
                enable_supervisory_islanding=True,
                supervisory_island_start_s=ISLAND_START_S,
                supervisory_island_duration_s=ISLAND_DURATION_S,
                supervisory_restore_load=True)

    def parse_energy(out_dir):
        files = glob.glob(f"{out_dir}/energy/energy_*.csv")
        by_t = defaultdict(lambda: {"gen":0,"load":0,"served":0,"unserved":0,
                                     "solar":0,"wind":0,"smr":0,"import_kw":0,
                                     "batt_discharge":0,"batt_charge":0,
                                     "batt_kwh":[],"shed":[],"count":0})
        for fp in files:
            with open(fp) as f:
                for r in csv.DictReader(f):
                    t = int(float(r.get("t_s",0)))
                    by_t[t]["gen"] += float(r.get("gen_kw",0) or 0)
                    by_t[t]["load"] += float(r.get("total_load_kw",0) or 0)
                    by_t[t]["served"] += float(r.get("served_kw",0) or 0)
                    by_t[t]["unserved"] += max(0, float(r.get("total_load_kw",0) or 0) - float(r.get("served_kw",0) or 0))
                    by_t[t]["solar"] += float(r.get("solar_kw",0) or 0)
                    by_t[t]["wind"] += float(r.get("wind_kw",0) or 0)
                    by_t[t]["smr"] += float(r.get("smr_kw",0) or 0)
                    by_t[t]["import_kw"] += float(r.get("import_kw",0) or 0)
                    by_t[t]["batt_discharge"] += float(r.get("battery_discharge_kw",0) or 0)
                    by_t[t]["batt_charge"] += float(r.get("battery_charge_kw",0) or 0)
                    by_t[t]["batt_kwh"].append(float(r.get("battery_kwh",0) or 0))
                    by_t[t]["shed"].append(float(r.get("shed_frac",0) or 0))
                    by_t[t]["count"] += 1
        return by_t

    data_gc = parse_energy(tmp1)
    data_is = parse_energy(tmp2)

    plt.rcParams.update({"font.family":"sans-serif","axes.grid":False,
                         "axes.spines.top":False,"axes.spines.right":False,"font.size":12,
                         "legend.fontsize":10})

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="col")

    def smooth(arr, w=30):
        if len(arr) < w: return arr
        return np.convolve(arr, np.ones(w)/w, mode="same")

    def plot_mode(axcol, data, title, island_start=None, island_end=None, is_right=False):
        times = sorted(data.keys())
        t_min = np.array([t/60 for t in times])
        gen = smooth(np.array([data[t]["gen"] for t in times]))
        load = smooth(np.array([data[t]["load"] for t in times]))
        served = smooth(np.array([data[t]["served"] for t in times]))
        unserved = smooth(np.array([data[t]["unserved"] for t in times]))
        solar = smooth(np.array([data[t]["solar"] for t in times]))
        wind = smooth(np.array([data[t]["wind"] for t in times]))
        smr = smooth(np.array([data[t]["smr"] for t in times]))
        imp = smooth(np.array([data[t]["import_kw"] for t in times]))
        batt_kwh = smooth(np.array([np.mean(data[t]["batt_kwh"]) for t in times]))
        batt_net = smooth(np.array([data[t]["batt_discharge"] - data[t]["batt_charge"] for t in times]))
        shed = smooth(np.array([np.mean(data[t]["shed"]) for t in times]))

        ax0, ax1, ax2 = axcol
        # Stacked generation area
        ax0.fill_between(t_min, 0, smr, alpha=0.5, color="#66bb6a", label="SMR")
        ax0.fill_between(t_min, smr, smr+solar, alpha=0.5, color="#ffb300", label="Solar")
        ax0.fill_between(t_min, smr+solar, smr+solar+wind, alpha=0.5, color="#42a5f5", label="Wind")
        ax0.fill_between(t_min, smr+solar+wind, smr+solar+wind+imp, alpha=0.4, color="#ab47bc", label="Grid Import")
        # Battery discharge contribution
        batt_d_smooth = smooth(np.array([data[t]["batt_discharge"] for t in times]))
        ax0.fill_between(t_min, smr+solar+wind+imp, smr+solar+wind+imp+batt_d_smooth,
                         alpha=0.3, color="#ef5350", label="Battery Disch.")
        ax0.plot(t_min, load, "-", color="#d62828", linewidth=2, label="Total Load")
        ax0.plot(t_min, served, "--", color="#264653", linewidth=1.5, alpha=0.8, label="Served")
        ax0.legend(fontsize=9, loc="upper right", ncol=2, framealpha=0.9)
        ax0.set_title(title, fontsize=14, fontweight="bold", color="#264653", pad=8)
        ax0.set_ylabel("Power (kW)", fontsize=13)

        # Battery SOC
        ax1.plot(t_min, batt_kwh, "-", color="#ef5350", linewidth=2, label="Avg Battery SOC")
        ax1.set_ylabel("Battery (kWh)", fontsize=13)
        ax1b = ax1.twinx()
        ax1b.plot(t_min, batt_net, "-", color="#1565c0", linewidth=1, alpha=0.5, label="Net Discharge")
        ax1b.axhline(0, color="#ccc", linewidth=0.5)
        ax1b.set_ylabel("Net Disch. (kW)", fontsize=10, color="#1565c0")
        ax1b.tick_params(axis="y", labelcolor="#1565c0")
        h1,l1 = ax1.get_legend_handles_labels()
        h2,l2 = ax1b.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, fontsize=9, loc="upper right", framealpha=0.9)

        # Shed + unserved
        ax2.plot(t_min, shed*100, "-", color="#e76f51", linewidth=2, label="Shed Fraction")
        ax2.set_ylabel("Shed (%)", fontsize=13, color="#e76f51")
        ax2.tick_params(axis="y", labelcolor="#e76f51")
        ax2b = ax2.twinx()
        ax2b.fill_between(t_min, 0, unserved, alpha=0.3, color="#d62828", label="Unserved Load")
        ax2b.set_ylabel("Unserved (kW)", fontsize=10, color="#d62828")
        ax2b.tick_params(axis="y", labelcolor="#d62828")
        ax2.set_xlabel("Time (min)", fontsize=13)
        h1,l1 = ax2.get_legend_handles_labels()
        h2,l2 = ax2b.get_legend_handles_labels()
        ax2.legend(h1+h2, l1+l2, fontsize=9, loc="upper left", framealpha=0.9)

        # Island shading
        if island_start is not None:
            for ax in [ax0, ax1, ax2]:
                ax.axvspan(island_start/60, island_end/60, alpha=0.15, color="#1a1a2e", zorder=0)
                ax.axvline(island_start/60, color="#264653", linewidth=1.5, linestyle="--", alpha=0.7)
                ax.axvline(island_end/60, color="#264653", linewidth=1.5, linestyle="--", alpha=0.7)
            ylim = ax0.get_ylim()
            ax0.text((island_start+island_end)/2/60, ylim[1]*0.98,
                     "ISLANDED", ha="center", va="top", fontsize=12, color="#1a1a2e",
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                               edgecolor="#1a1a2e", alpha=0.95, linewidth=1.5))

    plot_mode([axes[0,0], axes[1,0], axes[2,0]], data_gc, "Grid-Connected (Stressed)")
    plot_mode([axes[0,1], axes[1,1], axes[2,1]], data_is, "Islanded Mode (Stressed)",
              island_start=ISLAND_START_S, island_end=ISLAND_START_S+ISLAND_DURATION_S, is_right=True)
    fig.tight_layout(h_pad=1.5, w_pad=2.5)
    for ext in ("pdf","png"):
        fig.savefig(str(MAT_DIR / f"fig_energy_physical.{ext}"), dpi=600,
                    bbox_inches="tight", pad_inches=0.15)
        print(f"    ✓ fig_energy_physical.{ext}")
    plt.close(fig)
    shutil.rmtree(tmp1, ignore_errors=True)
    shutil.rmtree(tmp2, ignore_errors=True)

try:
    gen_energy_physical()
except Exception as e:
    print(f"  energy_physical error: {e}")
    import traceback; traceback.print_exc()

# ── 3e: Ablation figures ──
print("\n  [3e] Ablation figures...")
if (ROOT / "gen_ablation_figures.py").exists():
    exec(open(str(ROOT / "gen_ablation_figures.py")).read())

# ── 3f: Topology diagrams ──
print("\n  [3f] Topology diagrams...")
exec(open(str(ROOT / "gen_topology_diagrams.py")).read()) if (ROOT / "gen_topology_diagrams.py").exists() else None

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
t3 = time.time()
n_pdf = len(list(MAT_DIR.glob("*.pdf")))
n_png = len(list(MAT_DIR.glob("*.png")))
print(f"\n{'='*70}")
print(f"  ALL DONE! Total time: {(t3-t0)/60:.1f} minutes")
print(f"  Output: {FINAL_DIR}")
print(f"  Figures: {n_pdf} PDFs, {n_png} PNGs")
print(f"{'='*70}")
