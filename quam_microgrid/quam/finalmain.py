#!/usr/bin/env python3
"""
main_final_v2.py - FIXED QuAM Runner

FIXES from v1:
1. Microgrid NOW REQUIRES control messages to maintain balance
2. Without messages, shed_frac drifts up → causes EENS
3. Control messages actively reduce shed_frac
4. Intrusion detection blocks ALL auth messages during alert (not just PRIORITY)
5. Rate limiting now properly counted
6. Grid is load-constrained (needs active control)

Key Changes:
- MicrogridState.drift_rate: Without control, shed increases 0.5%/min
- Control messages actively reduce shed_frac (operational benefit)
- Message drops directly impact grid operations
"""

from __future__ import annotations

import argparse
import os
import random
import csv
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import hashlib

import simpy

# Import existing modules
from .model import (
    Message, MsgType, MicrogridState, MicrogridParams, DeliveryStatus,
    ActionDecision, GridMode, SimpleDCPowerFlow, PowerLine,
    FrequencyDynamics, FrequencyDynamicsConfig, WLSStateEstimator,
)
from .network import build_topology, build_links, NetworkSim, edge_key, summarize_link_congestion
from .network_metrics import NetworkActivityTracker
from .generation import GenerationMix, SolarProfile, WindProfile, SMRProfile
from .quantum import (
    QuantumAugmentation, KeyPolicy, QKDKeyPool, QuantumLinkHealth, QKDLinkParameters,
    QBERWindow, EaveWindow, edge_key, secret_fraction_bb84, fidelity_from_qber,
    apply_rotation_policy, get_finite_key_params
)
from .threat import (
    TrafficAnalyzer,
    CoverTrafficTracker,
    QANConfig,
    QANOrchestrator,
    SpoofConfig,
    SpoofingAttack,
    ExhaustConfig,
    KeyExhaustionAttack,
    TargetedExhaustConfig,
    TargetedKeyExhaustionAttack,
    ExhaustTargetStrategy,
    make_emit_with_observer,
    parse_action_from_message,
    should_ignore_as_stale,
    IntrusionDetector,
    AdmissionGate,
    PolicyGate,
    get_defense_config,
    secret_fraction_to_qber_approx,
    InsiderThreatConfig,
    InsiderThreatAttack,
    NodeLevelSpoofConfig,
    NodeLevelSpoofingAttack,
    CoordinatedAttackConfig,
    CoordinatedMultiNodeAttack,
    FDIAttackConfig,
    FalseDataInjectionAttack,
    MITMAttackConfig,
    ClassicalMITMAttack,
    QuarantineManager,
    QABConfig,
    GHZResourceTracker,
    QuantumAnonymousBroadcast,
)
from .quantum_protocols import (
    QuantumProtocolConfig, QuantumTLSConfig, PingPongVariant,
    QRNGSensorChallengeConfig, QRNGSensorChallenger,
)
from .metrics import QuAMLogger, compute_resilience_metrics

# ============================================================================
# FIXED: Microgrid that NEEDS control messages
# ============================================================================

@dataclass
class OperationalMicrogridState:
    """
    Microgrid that REQUIRES control messages to maintain stability.
    
    Key behavior:
    - Without control messages, shed_frac drifts UP (simulates need for coordination)
    - Successful control messages reduce shed_frac
    - This creates REAL operational consequences for message drops
    """
    params: MicrogridParams
    
    # Mode
    mode: GridMode = GridMode.GRID_TIED
    
    # Load shedding
    shed_frac: float = 0.0
    shed_target: float = 0.0
    
    # FIXED: Drift without control
    last_control_t_s: int = 0
    drift_rate_per_min: float = 0.005  # 0.5% per minute without control
    max_drift_shed: float = 0.30  # Max drift-induced shed
    
    # Accumulators
    eens_total_kwh: float = 0.0
    eens_critical_kwh: float = 0.0
    critical_outage_minutes: float = 0.0
    
    # Attack state
    forced_shed_until_s: int = 0
    forced_shed_frac: float = 0.0
    restoration_until_s: int = 0
    quarantine_until_s: int = 0

    # NEW: Network/Control coupling state
    comm_load_kw: float = 0.0
    comm_energy_kwh: float = 0.0
    control_quality: float = 1.0
    control_on_time_ratio: float = 1.0
    control_drop_ratio: float = 0.0
    avg_control_latency_ms: float = float("nan")
    last_control_arrival_s: float = 0.0
    
    # Debug
    last_total_load_kw: float = 0.0
    last_served_kw: float = 0.0
    last_gen_kw: float = 0.0
    last_import_kw: float = 0.0
    last_import_cap_kw: float = 0.0
    last_unserved_kw: float = 0.0
    
    # Tracking
    control_msgs_received: int = 0
    action_log: List = field(default_factory=list)

    def __post_init__(self) -> None:
        self.comm_load_kw = float(self.params.comm_base_kw)
        cap = max(0.0, self.params.battery_capacity_kwh)
        init = max(0.0, min(self.params.battery_init_kwh, cap))
        self.battery_kwh = init
    
    def receive_control(self, t_s: int):
        """Called when a valid control message is received."""
        self.control_msgs_received += 1
        self.last_control_t_s = t_s
        # Control message helps reduce shed (up to a point)
        self.shed_frac = max(0, self.shed_frac - 0.02)  # Each control reduces shed by 2%
        self.shed_target = max(0.0, self.shed_target - 0.01)
    
    def step(self, *, t_s: int, dt_s: int, gen_kw_sample: float,
             solar_kw: float = 0.0, wind_kw: float = 0.0, smr_kw: float = 0.0):
        """Step with drift modeling."""
        # Store per-source generation for logging
        self.last_solar_kw = solar_kw
        self.last_wind_kw = wind_kw
        self.last_smr_kw = smr_kw

        # Mode auto-transitions (time-based recovery)
        if self.mode == GridMode.RESTORATION and t_s >= self.restoration_until_s:
            self.mode = GridMode.GRID_TIED
        if self.mode == GridMode.QUARANTINE and t_s >= self.quarantine_until_s:
            self.mode = GridMode.GRID_TIED

        # FIXED: Apply drift if no recent control
        time_since_control_s = t_s - self.last_control_t_s
        if time_since_control_s > 30:  # Start drifting after 30s without control
            drift_minutes = (time_since_control_s - 30) / 60.0
            drift_shed = min(self.max_drift_shed, drift_minutes * self.drift_rate_per_min)
            self.shed_target = max(self.shed_target, drift_shed)

        forced_active = (t_s <= self.forced_shed_until_s)
        if forced_active:
            self.shed_target = max(self.shed_target, self.forced_shed_frac)
        elif self.forced_shed_until_s > 0 or self.forced_shed_frac > 0.0:
            # Expired attack overlay must be cleared or previous values keep relatching.
            self.forced_shed_until_s = 0
            self.forced_shed_frac = 0.0

        penalty = 0.0
        if self.control_quality < 1.0:
            penalty = max(0.0, self.params.control_quality_shed_gain * (1.0 - self.control_quality))

        # Recovery runs whenever forced overlay is inactive.
        if not forced_active:
            decay = self.params.recovery_rate_per_min * (dt_s / 60.0)
            if decay > 0:
                self.shed_target = max(0.0, self.shed_target - decay)
            if penalty > 0.0:
                self.shed_target = max(self.shed_target, penalty)
        
        # Move shed_frac toward target
        self.shed_frac = max(0, min(0.9, self.shed_frac))
        self.shed_target = max(0, min(0.9, self.shed_target))
        
        max_step = 0.25 * dt_s  # 25% per second max change
        delta = self.shed_target - self.shed_frac
        if abs(delta) <= max_step:
            self.shed_frac = self.shed_target
        else:
            self.shed_frac += max_step * (1 if delta > 0 else -1)
        
        # Compute energy balance
        p = self.params
        
        # FIXED: Load is now higher relative to supply (grid is constrained)
        total_load = p.base_load_kw + p.ai_load_kw + max(0.0, float(self.comm_load_kw))
        critical_load = min(p.critical_load_kw, total_load)
        
        # Apply shedding
        served_demand = total_load * (1.0 - self.shed_frac)
        
        # Supply
        gen = max(0, gen_kw_sample)
        if self.mode == GridMode.GRID_TIED:
            import_cap = p.import_cap_kw
        elif self.mode == GridMode.RESTORATION:
            import_cap = 0.5 * p.import_cap_kw
        else:
            import_cap = 0.0
        
        deficit = max(0, served_demand - gen)
        import_kw = min(import_cap, deficit)
        
        supply = gen + import_kw
        remaining_deficit = max(0.0, served_demand - supply)
        
        discharge_kw = 0.0
        if remaining_deficit > 0.0 and p.battery_capacity_kwh > 0.0:
            max_discharge_kw = min(
                p.battery_max_discharge_kw,
                (self.battery_kwh * 3600.0 / dt_s) if dt_s > 0 else 0.0,
            )
            discharge_kw = min(remaining_deficit, max_discharge_kw)
            if discharge_kw > 0.0:
                self.battery_kwh -= discharge_kw * (dt_s / 3600.0)
                remaining_deficit = max(0.0, remaining_deficit - discharge_kw)
        
        charge_kw = 0.0
        if remaining_deficit <= 0.0 and p.battery_capacity_kwh > 0.0:
            surplus_kw = max(0.0, supply - served_demand)
            headroom_kwh = max(0.0, p.battery_capacity_kwh - self.battery_kwh)
            max_charge_kw = min(
                p.battery_max_charge_kw,
                (headroom_kwh * 3600.0 / dt_s) if dt_s > 0 else 0.0,
            )
            charge_kw = min(surplus_kw, max_charge_kw)
            if charge_kw > 0.0:
                self.battery_kwh += charge_kw * (dt_s / 3600.0)
        
        served = max(0.0, served_demand - remaining_deficit)
        unserved = max(0.0, remaining_deficit)
        
        # Allocate to critical first
        served_critical = min(critical_load, served)
        unserved_critical = max(0, critical_load - served_critical)
        
        # Accumulate EENS using load curtailment (shed + unserved)
        curtailed_kw = max(0.0, total_load - served)
        self.eens_total_kwh += curtailed_kw * (dt_s / 3600.0)
        self.eens_critical_kwh += unserved_critical * (dt_s / 3600.0)
        
        if unserved_critical > 0.1:
            self.critical_outage_minutes += dt_s / 60.0
        
        # Debug
        self.last_total_load_kw = total_load
        self.last_served_kw = served
        self.last_gen_kw = gen
        self.last_import_kw = import_kw
        self.last_import_cap_kw = import_cap
        self.last_unserved_kw = unserved
        self.last_battery_discharge_kw = discharge_kw
        self.last_battery_charge_kw = charge_kw

    # -------------------------
    # Network/Control coupling
    # -------------------------

    def update_comm_state(self, *, stats: Dict[str, Any], dt_s: int) -> None:
        try:
            self.comm_load_kw = float(stats.get("comm_load_kw", self.comm_load_kw))
        except Exception:
            pass
        try:
            self.comm_energy_kwh += float(stats.get("comm_energy_kwh_inc", 0.0))
        except Exception:
            pass
        try:
            self.control_quality = float(stats.get("control_quality", self.control_quality))
            self.control_on_time_ratio = float(stats.get("control_on_time_ratio", self.control_on_time_ratio))
            self.control_drop_ratio = float(stats.get("control_drop_ratio", self.control_drop_ratio))
        except Exception:
            pass
        try:
            self.avg_control_latency_ms = float(stats.get("avg_control_latency_ms", self.avg_control_latency_ms))
        except Exception:
            pass
        last_ctrl = stats.get("last_control_arrival_s")
        if last_ctrl is not None:
            try:
                self.last_control_arrival_s = float(last_ctrl)
            except Exception:
                pass
    
    def apply_action(self, *, now_s: int, msg: Message, action, decision):
        """Apply action from message."""
        from .model import ActionType, ActionLogEntry
        
        self.action_log.append(ActionLogEntry(
            t_s=now_s, msg_id=msg.msg_id, src=msg.src, dst=msg.dst,
            action_type=action.action_type.value if action else "",
            decision=decision.decision, decision_reason=decision.reason,
            applied=(decision.decision == "allow"),
        ))
        
        if decision.decision != "allow":
            return False
        
        if action.action_type == ActionType.SHED_LOAD_EMERGENCY:
            if msg.is_attack and action.target_shed_frac is not None:
                target = max(0.0, min(float(action.target_shed_frac), 0.9))
            elif action.reason == "priority_action_shed":
                total_load = self.last_total_load_kw
                if total_load <= 0.0:
                    total_load = self.params.base_load_kw + self.params.ai_load_kw + max(0.0, float(self.comm_load_kw))
                supply = max(0.0, self.last_gen_kw) + max(0.0, self.last_import_kw)
                deficit_ratio = 0.0 if total_load <= 0.0 else max(0.0, (total_load - supply) / total_load)
                cq = max(0.0, min(1.0, float(getattr(self, "control_quality", 1.0))))
                base = 0.15 + 0.6 * (1.0 - cq) + 0.25 * deficit_ratio
                jitter = random.uniform(-0.1, 0.1)
                target = max(0.05, min(base + jitter, 0.9))
            else:
                if action.target_shed_frac is None:
                    target = 0.7
                else:
                    target = float(action.target_shed_frac)
                target = max(0.0, min(float(target), 0.9))

            # Apply target directly so spoof/priority actions are not monotonic-only increases.
            self.shed_target = max(0.0, min(0.9, float(target)))
            # Control setpoints carry duration_s=0 and should not create forced overlays.
            dur = int(action.duration_s) if action.duration_s is not None else 30
            if dur > 0:
                self.forced_shed_frac = float(target)
                self.forced_shed_until_s = max(self.forced_shed_until_s, now_s + dur)
        
        elif action.action_type == ActionType.RESTORE_LOAD:
            self.shed_target = max(0, self.shed_target - 0.5)
            self.forced_shed_until_s = 0
            self.forced_shed_frac = 0

        elif action.action_type == ActionType.ISLAND_NOW:
            self.mode = GridMode.ISLANDED

        elif action.action_type == ActionType.RECONNECT_GRID:
            self.mode = GridMode.RESTORATION
            self.restoration_until_s = now_s + self.params.restoration_duration_s

        elif action.action_type == ActionType.QUARANTINE:
            dur = int(action.duration_s) if action.duration_s is not None else 60
            self.mode = GridMode.QUARANTINE
            self.quarantine_until_s = now_s + dur

        elif action.action_type in (ActionType.OPEN_TIELINE, ActionType.CLOSE_TIELINE):
            pass

        return True


# ============================================================================
# ENERGY LOGGER
# ============================================================================

@dataclass
class EnergyRecord:
    t_s: int
    microgrid: str
    total_load_kw: float
    gen_kw: float
    import_kw: float
    served_kw: float
    unserved_kw: float
    shed_frac: float
    shed_target: float
    battery_kwh: float
    battery_discharge_kw: float
    battery_charge_kw: float
    time_since_control_s: int
    eens_cumulative_kwh: float
    is_attack_window: bool
    active_attack: str
    control_msgs_received: int
    comm_load_kw: float
    comm_energy_kwh: float
    control_quality: float
    control_on_time_ratio: float
    control_drop_ratio: float
    avg_control_latency_ms: float
    mode: str
    import_cap_kw: float
    # Per-source generation (defaults for backward compatibility)
    solar_kw: float = 0.0
    wind_kw: float = 0.0
    smr_kw: float = 0.0
    # Power flow (SimpleDCPowerFlow)
    line_loss_kw: float = 0.0
    voltage_violation: bool = False


class EnergyLogger:
    def __init__(self):
        self.records: List[EnergyRecord] = []
    
    def record(self, t_s: int, mg_name: str, mg, is_attack: bool = False, attack_label: str = "none"):
        self.records.append(EnergyRecord(
            t_s=t_s,
            microgrid=mg_name,
            total_load_kw=mg.last_total_load_kw,
            gen_kw=mg.last_gen_kw,
            import_kw=mg.last_import_kw,
            served_kw=mg.last_served_kw,
            unserved_kw=mg.last_unserved_kw,
            shed_frac=mg.shed_frac,
            shed_target=mg.shed_target,
            battery_kwh=mg.battery_kwh,
            battery_discharge_kw=mg.last_battery_discharge_kw,
            battery_charge_kw=mg.last_battery_charge_kw,
            time_since_control_s=t_s - mg.last_control_t_s,
            eens_cumulative_kwh=mg.eens_total_kwh,
            is_attack_window=is_attack,
            active_attack=attack_label,
            control_msgs_received=mg.control_msgs_received,
            comm_load_kw=getattr(mg, "comm_load_kw", 0.0),
            comm_energy_kwh=getattr(mg, "comm_energy_kwh", 0.0),
            control_quality=getattr(mg, "control_quality", float("nan")),
            control_on_time_ratio=getattr(mg, "control_on_time_ratio", float("nan")),
            control_drop_ratio=getattr(mg, "control_drop_ratio", float("nan")),
            avg_control_latency_ms=getattr(mg, "avg_control_latency_ms", float("nan")),
            mode=str(getattr(getattr(mg, "mode", None), "value", getattr(mg, "mode", "unknown"))),
            import_cap_kw=getattr(mg, "last_import_cap_kw", float("nan")),
            solar_kw=getattr(mg, "last_solar_kw", 0.0),
            wind_kw=getattr(mg, "last_wind_kw", 0.0),
            smr_kw=getattr(mg, "last_smr_kw", 0.0),
        ))
    
    def write_csv(self, path: str):
        if not self.records:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.records[0]).keys()))
            writer.writeheader()
            for r in self.records:
                writer.writerow(asdict(r))


# ============================================================================
# QUANTUM TIME SERIES
# ============================================================================

class QuantumTimeSeriesLogger:
    def __init__(self):
        self.records = []
    
    def record(self, **kwargs):
        self.records.append(kwargs)
    
    def write_csv(self, path: str):
        if not self.records:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.records[0].keys()))
            writer.writeheader()
            writer.writerows(self.records)


# ============================================================================
# UTILITIES
# ============================================================================

def generate_attack_windows(
    rng, horizon_s, total_dur, num_win,
    min_gap_s: int = 120, edge_buffer_s: int = 300
) -> List[Tuple[int, int]]:
    if num_win <= 0 or total_dur <= 0:
        return []
    
    usable = max(0, horizon_s - 2 * edge_buffer_s)
    if usable <= 0:
        return []
    
    max_gap = max(0, (usable - total_dur) // max(1, num_win - 1))
    min_gap_s = min(min_gap_s, max_gap)
    
    avg = max(60, total_dur // num_win)
    min_dur = max(60, avg // 2)
    max_dur = max(min_dur, avg * 2)
    
    durations = []
    remaining = total_dur
    for i in range(num_win):
        if i == num_win - 1:
            dur = max(min_dur, remaining)
        else:
            max_for_i = remaining - min_dur * (num_win - i - 1)
            if max_for_i < min_dur:
                dur = max(60, remaining // (num_win - i))
            else:
                dur = rng.randint(min_dur, min(max_dur, max_for_i))
        durations.append(dur)
        remaining -= dur
    
    available = [(edge_buffer_s, horizon_s - edge_buffer_s)]
    windows = []
    for dur in durations:
        candidates = [(a, b) for (a, b) in available if b - a >= dur]
        if not candidates:
            break
        a, b = rng.choice(candidates)
        start = rng.randint(a, b - dur)
        end = start + dur
        windows.append((start, end))
        
        block_start = start - min_gap_s
        block_end = end + min_gap_s
        new_available = []
        for (x, y) in available:
            if y <= block_start or x >= block_end:
                new_available.append((x, y))
            else:
                if x < block_start:
                    new_available.append((x, block_start))
                if y > block_end:
                    new_available.append((block_end, y))
        available = new_available
    
    windows.sort(key=lambda w: w[0])
    return windows




def _split_window(rng, start_s: int, end_s: int, n_segments: int):
    if n_segments <= 1 or end_s - start_s <= 1:
        return [(start_s, end_s)]
    n_segments = max(1, n_segments)
    # Ensure we have enough room for splits
    if end_s - start_s <= n_segments:
        return [(start_s, end_s)]
    cut_points = sorted(rng.sample(range(start_s + 1, end_s), n_segments - 1))
    pts = [start_s] + cut_points + [end_s]
    return list(zip(pts[:-1], pts[1:]))


def _sample_qber(cfg: dict, rng) -> float:
    mean = cfg.get("qber_mean", cfg.get("qber", 0.05))
    std = cfg.get("qber_std", 0.0)
    q = rng.gauss(mean, std) if std > 0 else mean
    q_min = cfg.get("qber_min", 0.0)
    q_max = cfg.get("qber_max", 0.5)
    return max(q_min, min(q_max, q))

def _mk_outdir(tag: str) -> str:
    base = os.path.join("outputs", tag)
    num = 1
    if os.path.exists(base):
        existing = [d for d in os.listdir(base) if d.startswith("trial_")]
        if existing:
            nums = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
            num = max(nums) + 1 if nums else 1
    
    trial = os.path.join(base, f"trial_{num}")
    for f in ["messages", "deanon", "summary", "timeseries", "energy"]:
        os.makedirs(os.path.join(trial, f), exist_ok=True)
    return trial


def parse_scenario(scenario: str) -> Tuple[List[str], str]:
    if scenario == "baseline":
        return [], "none"
    if "_def_" in scenario:
        parts = scenario.rsplit("_def_", 1)
        atk, defense = parts[0], parts[1]
        if atk == "all_attacks":
            return ["spoof", "exhaust", "quantum"], defense
        elif atk == "all_attacks_v2":
            return ["spoof", "exhaust", "quantum", "insider"], defense
        # Handle compound attack names that should not be split:
        # "nodespoof" and "nodespoofforged" are single attack types.
        elif atk in ("nodespoof", "nodespoofforged"):
            return [atk], defense
        elif "_" in atk:
            return atk.split("_"), defense
        return [atk], defense
    return [], "none"


QUANTUM_DEFENSE_STRATEGIES = {
    "ratelimit_v2",
    "intrusion_v2",
    "plausibility",
    "correlation",
    "quarantine_v2",
    "hardened",
    "hardened_v2",
    "hardened_v3",
    "quantum_only",
}


def get_quantum_defense_config(defense_mode: str, enable_qkd: bool) -> Dict[str, Any]:
    """
    Map defense mode to quantum-layer controls.
    """
    mode = str(defense_mode or "none").lower()
    enabled = bool(enable_qkd and mode in QUANTUM_DEFENSE_STRATEGIES)
    if not enabled:
        return {
            "enabled": False,
            "enable_priority_reservation": False,
            "reservation_ratio": 0.0,
            "enable_source_key_rate_limit": False,
            "source_key_rate_bits_per_s": 0.0,
            "enable_emergency_mode": False,
            "emergency_threshold_ratio": 0.0,
            "emergency_tag_bits": 0,
        }

    reservation_ratio = 0.20
    source_rate = 1000.0
    if mode == "hardened":
        reservation_ratio = 0.30
        source_rate = 800.0
    elif mode in {"ratelimit_v2", "quarantine_v2"}:
        reservation_ratio = 0.25
        source_rate = 900.0

    return {
        "enabled": True,
        "enable_priority_reservation": True,
        "reservation_ratio": reservation_ratio,
        "enable_source_key_rate_limit": True,
        "source_key_rate_bits_per_s": source_rate,
        "enable_emergency_mode": True,
        "emergency_threshold_ratio": 0.10,
        "emergency_tag_bits": 64,
    }


# ============================================================================
# MAIN SIMULATION
# ============================================================================

DEFAULT_NODES = ["MG0", "MG1", "MG2"]


def run_one(
    *,
    scenario: str,
    topology: str,
    seed: int,
    horizon_s: int,
    out_dir: str,
    nodes: List[str],
    route_policy: str = "shortest",
    k_paths: int = 3,
    use_grid_link_types: bool = False,
    attack_intensity: str = "S3",
    attack_duration: Optional[int] = None,
    distributed_attacks: bool = False,
    num_attack_windows: int = 5,
    exhaust_strategy: str = "uniform",
    exhaust_focus: float = 0.8,
    qan_events: int = 3,
    qan_cover_rate_per_s: float = 4.0,
    qan_window_s: int = 5,
    qan_mixing_delay_ms: int = 40,
    qan_auth_cover: bool = False,
    qan_auth_real: bool = False,
    qan_auth_sync: bool = False,
    qan_mode: str = "classical",
    qab_ghz_prep_success_prob: float = 0.85,
    qab_ghz_fidelity_base: float = 0.95,
    qab_message_bits: int = 256,
    qab_decoherence_window_ms: int = 100,
    qab_ghz_prep_time_ms: int = 5,
    energy_record_interval: int = 30,
    enable_qkd: bool = True,
    rotation_policy: str = "none",
    verification_delay_ms: Optional[int] = None,
    degraded_verification_delay_ms: Optional[int] = None,
    enable_power_sharing: bool = False,
    link_distance_km: float = 10.0,
    fiber_loss_db_per_km: float = 0.2,
    finite_key_preset: str = "disabled",
    finite_key_block_bits: Optional[int] = None,
    finite_key_security_log: Optional[int] = None,
    degraded_threshold_preset: str = "moderate",
    random_window_intensity: bool = False,
    window_intensity_pool: Optional[List[str]] = None,
    qrng_pool_bits: Optional[float] = None,
    qrng_rate_bits_per_s: Optional[float] = None,
    enable_quantum_protocols: bool = False,
    pingpong_variant: str = "ghz",
    pingpong_interval_s: float = 5.0,
    infrastructure_override: Optional[Dict[str, Any]] = None,
    compromised_node: Optional[str] = None,
    auth_model: str = "per_hop",  # "per_hop" or "e2e_relay"
    spoof_auth_bypass_prob: float = 0.0,  # Probability attacker forges auth credentials
    enable_sensor_challenges: bool = False,  # QRNG measurement challenges
    sensor_challenge_interval_s: float = 15.0,  # Mean challenge interval
    # V5: Quantum transport tuning
    qec_code_distance: int = 3,
    e2e_distillation_rounds: int = 1,
    e2e_swap_success_prob: float = 0.5,
    quantum_qber_override: Optional[float] = None,
    quantum_target_edge_count: int = 2,
    # V5: Quantum control authentication
    enable_quantum_control_auth: bool = False,
    quantum_control_token_ttl_ms: int = 1500,
    # V6: Realistic quantum auth bypass probability (implementation flaws)
    quantum_auth_bypass_prob: float = 0.0,
    # V11: Hardware timing jitter model
    hw_timing_jitter_ms: float = 0.0,
    spd_timing_overhead_ms: float = 0.0,
    # V5: Supervisory islanding
    enable_supervisory_islanding: bool = False,
    supervisory_island_start_s: Optional[int] = None,
    supervisory_island_duration_s: int = 0,
    supervisory_restore_load: bool = True,
    # V5: Microgrid parameter overrides
    microgrid_param_overrides: Optional[Dict[str, Any]] = None,
    # V5: Output control
    write_outputs: bool = True,
    # V14: QBER threshold sweep
    qber_threshold: float = 0.025,
) -> Dict[str, Any]:
    
    attacks, defense_mode = parse_scenario(scenario)
    
    rng = random.Random(seed)
    env = simpy.Environment()
    
    _msg_id = [0]
    def msg_id_fn():
        _msg_id[0] += 1
        return _msg_id[0]
    
    # Network
    if str(topology).startswith("ieee"):
        graph = build_topology(topology, [], rng)
        managed_nodes = sorted(list(graph.nodes()))
    else:
        managed_nodes = list(nodes)
        graph = build_topology(topology, managed_nodes, rng)

    # V5: Dedicated supervisory controller overlay
    _is_federated = graph.graph.get("federated", False)
    if _is_federated:
        # Federated: two controllers, one per domain
        ctrl_a = "CTRL_A"
        ctrl_b = "CTRL_B"
        graph.add_node(ctrl_a)
        graph.add_node(ctrl_b)
        graph.nodes[ctrl_a]["role"] = "central_controller"
        graph.nodes[ctrl_a]["domain"] = "grid_a"
        graph.nodes[ctrl_b]["role"] = "central_controller"
        graph.nodes[ctrl_b]["domain"] = "grid_b"
        for mg_node in managed_nodes:
            domain = graph.nodes.get(mg_node, {}).get("domain", "grid_a")
            ctrl = ctrl_a if domain == "grid_a" else ctrl_b
            graph.add_edge(ctrl, mg_node)
            graph.nodes[mg_node]["role"] = "microgrid"
        controller_node = ctrl_a  # Primary for backward compat
        controller_nodes = (ctrl_a, ctrl_b)
        nodes = [ctrl_a, ctrl_b] + list(managed_nodes)
    else:
        controller_node = "CTRL0"
        while controller_node in managed_nodes:
            controller_node = f"CTRL0_{rng.randint(1, 999)}"
        controller_nodes = (controller_node,)

        graph.add_node(controller_node)
        for mg_node in managed_nodes:
            graph.add_edge(controller_node, mg_node)
            graph.nodes[mg_node]["role"] = "microgrid"
        graph.nodes[controller_node]["role"] = "central_controller"

        nodes = [controller_node] + list(managed_nodes)
    edges = list(graph.edges())
    links = build_links(env, graph, rng, use_grid_link_types=use_grid_link_types)

    # V3: SimpleDC power flow model over physical microgrid interconnect only
    _ctrl_set = set(controller_nodes)
    physical_edges = [(u, v) for (u, v) in edges if u not in _ctrl_set and v not in _ctrl_set]
    power_lines = [
        PowerLine(
            from_bus=u, to_bus=v,
            resistance_ohm=0.08,
            max_current_a=200.0,
            length_km=link_distance_km,
            voltage_nominal_v=480.0,
        )
        for u, v in physical_edges
    ]
    power_flow = SimpleDCPowerFlow(lines=power_lines, buses=managed_nodes)

    # Attack config
    atk_cfgs = {
        "S1": {"capacity": 15000, "refill": 1500, "exhaust_rate": 2.0,
               "qber_mean": 0.03, "qber_std": 0.006, "qber_min": 0.015, "qber_max": 0.07},
        "S2": {"capacity": 10000, "refill": 1000, "exhaust_rate": 5.0,
               "qber_mean": 0.05, "qber_std": 0.010, "qber_min": 0.02, "qber_max": 0.10},
        "S3": {"capacity": 6000, "refill": 600, "exhaust_rate": 10.0,
               "qber_mean": 0.08, "qber_std": 0.015, "qber_min": 0.03, "qber_max": 0.14},
        "S4": {"capacity": 4000, "refill": 400, "exhaust_rate": 15.0,
               "qber_mean": 0.10, "qber_std": 0.018, "qber_min": 0.04, "qber_max": 0.18},
        "S5": {"capacity": 2500, "refill": 250, "exhaust_rate": 20.0,
               "qber_mean": 0.14, "qber_std": 0.020, "qber_min": 0.08, "qber_max": 0.22},
    }
    atk_cfg = atk_cfgs.get(attack_intensity, atk_cfgs["S3"])
    # Spoofing intervals: realistic SCADA attack rates.
    # Old values (S1=420s .. S5=180s) were far too sparse — real attackers
    # can send multiple forged commands per second once they have network access.
    spoof_window_cfgs = {
        "S1": {"forced_shed_frac": 0.35, "harm_duration_s": 25, "interval_s": 30},
        "S2": {"forced_shed_frac": 0.45, "harm_duration_s": 35, "interval_s": 15},
        "S3": {"forced_shed_frac": 0.55, "harm_duration_s": 45, "interval_s": 8},
        "S4": {"forced_shed_frac": 0.62, "harm_duration_s": 60, "interval_s": 4},
        "S5": {"forced_shed_frac": 0.70, "harm_duration_s": 75, "interval_s": 2},
    }
    
    # Quantum layer
    key_policy = KeyPolicy(
        tag_bits=384 if enable_qkd else 0,
        nonce_bits=64,
        max_key_wait_ms=1500 if enable_qkd else 0,
        verify_cost_factor=1.5,
    )
    apply_rotation_policy(key_policy, rotation_policy)

    link_params = QKDLinkParameters(
        distance_km=link_distance_km,
        fiber_loss_db_per_km=fiber_loss_db_per_km,
    )
    finite_key_params = get_finite_key_params(
        finite_key_preset,
        block_size_bits=finite_key_block_bits,
        security_parameter_log=finite_key_security_log,
    )
    quantum_def_cfg = get_quantum_defense_config(defense_mode, enable_qkd)
    if infrastructure_override and enable_qkd:
        _pool_cap = infrastructure_override["capacity"]
        _pool_refill = infrastructure_override["refill"]
        _pool_init = infrastructure_override.get("init_fill_ratio", 0.50)
    else:
        _pool_cap = atk_cfg["capacity"] if enable_qkd else 1000000
        _pool_refill = atk_cfg["refill"] if enable_qkd else 1000000
        _pool_init = 0.25 if enable_qkd else 1.0
    default_pool = QKDKeyPool(
        capacity_bits=_pool_cap,
        base_refill_bits_per_s=_pool_refill,
        init_fill_ratio=_pool_init,
        link_params=link_params,
    )
    default_health = QuantumLinkHealth(baseline_qber=0.01)
    
    per_edge_distance_km = {edge_key(u, v): link_distance_km for (u, v) in edges}

    # V2: Multi-protocol quantum layer config
    q_proto_cfg = None
    if enable_quantum_protocols:
        _pp_variant_map = {"bell": PingPongVariant.BELL, "ghz": PingPongVariant.GHZ, "cluster": PingPongVariant.CLUSTER}
        q_proto_cfg = QuantumProtocolConfig(
            enable_pingpong_ids=True,
            ids_probe_interval_s=pingpong_interval_s,
            ids_variant=_pp_variant_map.get(str(pingpong_variant).lower(), PingPongVariant.GHZ),
        )

    qlayer = QuantumAugmentation(
        env=env, rng=rng, key_policy=key_policy,
        default_pool=default_pool, default_health=default_health,
        per_edge_distance_km=per_edge_distance_km,
        finite_key_params=finite_key_params,
        qrng_rate_bits_per_s=float(qrng_rate_bits_per_s) if qrng_rate_bits_per_s is not None else 1000.0,
        qrng_capacity_bits=float(qrng_pool_bits) if qrng_pool_bits is not None else 8000.0,
        enable_priority_reservation=bool(quantum_def_cfg["enable_priority_reservation"]),
        reservation_ratio=float(quantum_def_cfg["reservation_ratio"]),
        enable_source_key_rate_limit=bool(quantum_def_cfg["enable_source_key_rate_limit"]),
        source_key_rate_bits_per_s=float(quantum_def_cfg["source_key_rate_bits_per_s"]),
        enable_emergency_mode=bool(quantum_def_cfg["enable_emergency_mode"]),
        emergency_threshold_ratio=float(quantum_def_cfg["emergency_threshold_ratio"]),
        emergency_tag_bits=int(quantum_def_cfg["emergency_tag_bits"]),
        quantum_protocol_config=q_proto_cfg,
        auth_model=auth_model,
        graph=graph,
        qec_code_distance=qec_code_distance,
        e2e_distillation_rounds=e2e_distillation_rounds,
        e2e_swap_success_prob=e2e_swap_success_prob,
        enable_quantum_control_auth=enable_quantum_control_auth,
        quantum_control_token_ttl_ms=quantum_control_token_ttl_ms,
        quantum_auth_bypass_prob=quantum_auth_bypass_prob,
    )

    
    # Intrusion detector
    intrusion_detector = IntrusionDetector(qber_threshold=qber_threshold)
    
    # FIXED: Use OperationalMicrogridState that needs control
    mg_params = MicrogridParams(
        name="default",
        base_load_kw=120.0,  # Baseline demand
        ai_load_kw=30.0,     # Smaller AI component
        critical_load_kw=110.0,
        gen_kw_mean=100.0,   # Legacy Gaussian fallback
        gen_kw_sigma=20.0,
        import_cap_kw=60.0,   # Supply > demand in baseline
        # Realistic generation mix (Solar + Wind + SMR)
        # Sized so avg gen ≈ 105 kW → with 60 kW import, supply ≈ 165 kW > 150 kW demand
        generation_model="realistic",
        solar_capacity_kw=70.0,      # 70 kW PV array (~55% CF at noon → 38 kW avg)
        wind_capacity_kw=50.0,       # 50 kW turbine (~25% CF → 12.5 kW avg)
        smr_capacity_kw=60.0,        # 60 kW SMR module (~90% CF → 54 kW avg)
        wind_mean_speed_ms=9.0,      # IEC Class II site (better wind resource)
    )
    if microgrid_param_overrides:
        for key, value in microgrid_param_overrides.items():
            if hasattr(mg_params, key):
                setattr(mg_params, key, value)
    microgrids = {n: OperationalMicrogridState(params=mg_params) for n in managed_nodes}

    # Initialize generation profiles (one per microgrid, different weather)
    gen_profiles: Dict[str, GenerationMix] = {}
    for idx, n in enumerate(managed_nodes):
        gm = GenerationMix(
            solar=SolarProfile(
                capacity_kw=mg_params.solar_capacity_kw,
                time_of_day_h=mg_params.solar_time_of_day_h,
                cloud_transition_s=mg_params.solar_cloud_transition_s,
            ),
            wind=WindProfile(
                capacity_kw=mg_params.wind_capacity_kw,
                mean_speed_ms=mg_params.wind_mean_speed_ms,
                turbulence_intensity=mg_params.wind_turbulence_intensity,
                correlation_s=mg_params.wind_correlation_s,
            ),
            smr=SMRProfile(
                capacity_kw=mg_params.smr_capacity_kw,
                availability=mg_params.smr_availability,
            ),
        )
        gm.seed(seed, node_idx=idx)
        gen_profiles[n] = gm

    # V6: Frequency dynamics — one per microgrid
    p_rated_kw = mg_params.solar_capacity_kw + mg_params.wind_capacity_kw + mg_params.smr_capacity_kw
    freq_cfg = FrequencyDynamicsConfig()
    freq_dynamics: Dict[str, FrequencyDynamics] = {
        n: FrequencyDynamics(cfg=freq_cfg, p_rated_kw=p_rated_kw,
                             seed=seed + idx)
        for idx, n in enumerate(managed_nodes)
    }

    # V6: WLS State Estimator for central controller
    state_estimator = None
    try:
        se_adjacency = [(u, v) for u, v in physical_edges]
        state_estimator = WLSStateEstimator(
            bus_names=list(managed_nodes),
            adjacency=se_adjacency,
            line_susceptance=10.0,
            sigma=0.15,  # 15% per-unit — sensor noise; model mismatch handled by tighter alpha
        )
    except Exception:
        pass  # numpy not available or topology too small

    # Network activity tracker (comms-energy coupling)
    net_tracker = NetworkActivityTracker(window_s=mg_params.control_window_s)
    
    # Defense
    gate_cfg = get_defense_config(defense_mode, degraded_threshold_preset)
    if controller_nodes:
        gate_cfg.allowed_control_sources = controller_nodes
    if enable_quantum_control_auth:
        gate_cfg.require_quantum_control_token = True
    if verification_delay_ms is not None:
        gate_cfg.verification_delay_ms = max(0, int(verification_delay_ms))
    if degraded_verification_delay_ms is not None:
        gate_cfg.degraded_verification_delay_ms = max(0, int(degraded_verification_delay_ms))
    # V11: Hardware timing jitter
    if hw_timing_jitter_ms > 0:
        gate_cfg.hw_timing_jitter_ms = float(hw_timing_jitter_ms)
    if spd_timing_overhead_ms > 0:
        gate_cfg.spd_timing_overhead_ms = float(spd_timing_overhead_ms)
    use_intrusion = bool(getattr(gate_cfg, "block_during_intrusion", False))
    admission_gate = AdmissionGate(
        gate_cfg,
        intrusion_detector if use_intrusion else None,
    )
    gate = PolicyGate(
        gate_cfg,
        intrusion_detector if use_intrusion else None,
        microgrids=microgrids,
        rng=rng,
    )
    qlayer.preauth_decider = admission_gate.decide

    # V4: QRNG Sensor Measurement Challenges (active FDI defense)
    sensor_challenger = None
    if enable_sensor_challenges:
        _qrng_for_challenges = None
        _nonce_for_challenges = None
        if enable_qkd and hasattr(qlayer, "nonce_mgr") and qlayer.nonce_mgr is not None:
            _qrng_for_challenges = qlayer.nonce_mgr.qrng
            _nonce_for_challenges = qlayer.nonce_mgr
        challenge_cfg = QRNGSensorChallengeConfig(
            mean_challenge_interval_s=sensor_challenge_interval_s,
            enabled=True,
        )
        challenge_quarantine = QuarantineManager(duration_s=120, cooldown_s=60)
        sensor_challenger = QRNGSensorChallenger(
            cfg=challenge_cfg,
            qrng=_qrng_for_challenges,
            nonce_mgr=_nonce_for_challenges,
            quarantine_mgr=challenge_quarantine,
            rng=rng,
        )
        # Pre-compute adaptive thresholds from source capacities
        for name in managed_nodes:
            sensor_challenger.compute_adaptive_threshold(
                node=name,
                solar_cap=mg_params.solar_capacity_kw,
                wind_cap=mg_params.wind_capacity_kw,
                smr_cap=mg_params.smr_capacity_kw,
            )

    # V5: Telemetry cache for central controller
    controller_telemetry_cache: Dict[str, Dict[str, Any]] = {}

    # On deliver hook
    def on_deliver_hook(env_inner, msg):
        # Telemetry to controller is cached, not acted upon
        if msg.dst == controller_node and msg.msg_type == MsgType.TELEMETRY:
            controller_telemetry_cache[msg.src] = {
                "t_s": int(env_inner.now),
                **(msg.payload or {}),
            }
            return

        mg = microgrids.get(msg.dst)
        if mg is None:
            return

        # Ensure payload exists so later logging/metrics have consistent fields.
        msg.payload = msg.payload or {}
        
        # FIXED: Control messages help the grid
        if msg.msg_type == MsgType.CONTROL_SETPOINT and not msg.is_attack:
            mg.receive_control(int(env_inner.now))
        
        action = parse_action_from_message(msg)
        if action is None:
            return
        
        decision, delay = gate.decide(env=env_inner, msg=msg, action=action,
                                       staleness_ms=mg.params.staleness_ms)

        # Record gate decision at delivery-time for later integrity metrics.
        # Note: decision may be overridden at apply-time if verification delay causes a
        # deadline miss (see _apply()).
        msg.payload["action"] = str(getattr(action.action_type, "value", action.action_type))
        msg.payload["gate_decision"] = str(getattr(decision, "decision", decision))
        msg.payload["gate_reason"] = str(getattr(decision, "reason", ""))
        msg.payload["gate_delay_ms"] = int(delay) if delay is not None else 0
        
        def _apply():
            if delay > 0:
                yield env_inner.timeout(delay / 1000.0)

            # Verification delay can make an otherwise-allowed action miss its deadline.
            # Model: if (network latency + verification delay) exceeds deadline_ms, the
            # control action is ignored as "too late to matter".
            apply_decision = decision
            if str(getattr(decision, "decision", "")).lower() == "allow":
                try:
                    now_ms = int(env_inner.now * 1000)
                    age_ms = now_ms - int(getattr(msg, "created_ms", now_ms))
                    dl_ms = int(getattr(msg, "deadline_ms", -1))
                    if delay and dl_ms > 0 and age_ms > dl_ms:
                        apply_decision = ActionDecision("ignore", "verify_delay_missed_deadline")
                except Exception:
                    pass

            # Update gate decision fields if apply-time override happened.
            if apply_decision is not decision:
                msg.payload["gate_decision"] = str(getattr(apply_decision, "decision", apply_decision))
                msg.payload["gate_reason"] = str(getattr(apply_decision, "reason", ""))

            mg.apply_action(now_s=int(env_inner.now), msg=msg, action=action, decision=apply_decision)
        
        env_inner.process(_apply())
    
    # Network
    net = NetworkSim(
        env=env, g=graph, links=links, rng=rng,
        route_policy=route_policy,
        k_paths=k_paths,
        pre_send_hook=qlayer.pre_send_hook,
        on_deliver_hook=on_deliver_hook,
        on_message_final=net_tracker.observe_message,
        drop_if_miss_deadline=True,
    )
    
    # Loggers
    logger = QuAMLogger()
    quantum_ts = QuantumTimeSeriesLogger()
    energy_ts = EnergyLogger()
    analyzer = TrafficAnalyzer(sender_candidates=nodes)
    
    created_msgs = []
    
    def base_emit(msg):
        created_msgs.append(msg)
        net.send(msg)
    
    emit = make_emit_with_observer(base_emit=base_emit, analyzer=analyzer)
    
    # Attack windows
    total_attack_s = attack_duration or int(horizon_s / 3)
    if distributed_attacks and attacks:
        attack_windows = generate_attack_windows(rng, horizon_s, total_attack_s, num_attack_windows)
    elif attacks:
        start = max(300, horizon_s // 3)
        attack_windows = [(start, min(start + total_attack_s, horizon_s - 300))]
    else:
        attack_windows = []

    allowed_levels = [k for k in atk_cfgs.keys() if k.startswith("S")]
    if window_intensity_pool:
        custom_levels = [str(x).strip() for x in window_intensity_pool if str(x).strip() in atk_cfgs]
        if custom_levels:
            allowed_levels = custom_levels
    window_intensity_map: Dict[Tuple[int, int], str] = {}
    for (start, end) in attack_windows:
        lvl = rng.choice(allowed_levels) if random_window_intensity else attack_intensity
        window_intensity_map[(start, end)] = lvl
    
    def get_attack_status(t_s):
        for s, e in attack_windows:
            if s <= t_s <= e:
                return True, ",".join(attacks)
        return False, "none"
    
    # V5: Structured three-way traffic with proper control architecture
    def _background():
        control_sources = list(controller_nodes) or list(nodes)
        control_targets = [n for n in managed_nodes if n not in control_sources] or list(managed_nodes) or list(nodes)
        telemetry_sources = list(managed_nodes) or list(nodes)
        telemetry_targets = list(controller_nodes) or list(nodes)
        peer_coord_topics = (
            "reserve_share",
            "battery_soc",
            "renewable_forecast",
            "islanding_readiness",
            "restoration_status",
        )
        while env.now < horizon_s:
            rnd = rng.random()
            if rnd < 0.60:
                # 60% — Routine control setpoints from designated controller(s)
                src = rng.choice(control_sources)
                dst = rng.choice([n for n in control_targets if n != src] or control_targets or nodes)
                msg = Message(
                    msg_id=msg_id_fn(), created_ms=int(env.now * 1000),
                    src=src, dst=dst, msg_type=MsgType.CONTROL_SETPOINT,
                    priority=1, deadline_ms=500, size_bytes=260,
                    requires_auth=True, payload={"shed_frac_target": 0.0},
                )
            elif rnd < 0.75:
                # 15% — Critical priority actions from designated controller(s)
                src = rng.choice(control_sources)
                dst = rng.choice([n for n in control_targets if n != src] or control_targets or nodes)
                msg = Message(
                    msg_id=msg_id_fn(), created_ms=int(env.now * 1000),
                    src=src, dst=dst, msg_type=MsgType.PRIORITY_ACTION,
                    priority=2, deadline_ms=300, size_bytes=280,
                    requires_auth=True,
                    payload={
                        "action": "shed_load_emergency",
                        "shed_frac_target": 0.0,
                        "control_signature": "quam_ctrl_v1",
                        "control_sender_role": "controller",
                    },
                )
            else:
                # 25% — Telemetry and peer coordination
                src = rng.choice(telemetry_sources)
                mg_src = microgrids.get(src)
                if len(managed_nodes) > 1 and rng.random() < 0.30:
                    # 30% of telemetry = peer coordination (MG↔MG)
                    peer_targets = [n for n in managed_nodes if n != src]
                    dst = rng.choice(peer_targets or telemetry_targets or nodes)
                    requires_auth = True
                    payload = {
                        "telemetry_role": "peer_coordination",
                        "coordination_topic": rng.choice(peer_coord_topics),
                        "reported_gen_kw": float(getattr(mg_src, "last_gen_kw", 0.0)) if mg_src else 0.0,
                        "reported_load_kw": float(getattr(mg_src, "last_total_load_kw", 0.0)) if mg_src else 0.0,
                        "battery_kwh": float(getattr(mg_src, "battery_kwh", 0.0)) if mg_src else 0.0,
                        "grid_mode": str(getattr(getattr(mg_src, "mode", None), "value", "unknown")),
                        "available_export_kw": max(
                            0.0,
                            float(getattr(mg_src, "last_gen_kw", 0.0)) - float(getattr(mg_src, "last_total_load_kw", 0.0)),
                        ) if mg_src else 0.0,
                    }
                else:
                    # 70% of telemetry = state reports (MG→CTRL0)
                    dst = rng.choice([n for n in telemetry_targets if n != src] or telemetry_targets or nodes)
                    requires_auth = False
                    payload = {
                        "telemetry_role": "state_report",
                        "reported_gen_kw": float(getattr(mg_src, "last_gen_kw", 0.0)) if mg_src else 0.0,
                        "reported_load_kw": float(getattr(mg_src, "last_total_load_kw", 0.0)) if mg_src else 0.0,
                        "battery_kwh": float(getattr(mg_src, "battery_kwh", 0.0)) if mg_src else 0.0,
                        "grid_mode": str(getattr(getattr(mg_src, "mode", None), "value", "unknown")),
                        "control_quality": float(getattr(mg_src, "control_quality", 1.0)) if mg_src else 1.0,
                    }
                msg = Message(
                    msg_id=msg_id_fn(), created_ms=int(env.now * 1000),
                    src=src, dst=dst, msg_type=MsgType.TELEMETRY,
                    priority=0, deadline_ms=800, size_bytes=220,
                    requires_auth=requires_auth,
                    payload=payload,
                )
            emit(msg)
            yield env.timeout(rng.uniform(0.3, 0.8))

    env.process(_background())

    # V5: Supervisory islanding controller
    def _supervisor_control():
        if not enable_supervisory_islanding:
            return

        island_start = int(
            supervisory_island_start_s
            if supervisory_island_start_s is not None
            else max(60, int(0.55 * horizon_s))
        )
        island_duration = int(
            supervisory_island_duration_s
            if supervisory_island_duration_s > 0
            else max(60, int(0.20 * horizon_s))
        )
        island_end = min(horizon_s - 1, island_start + island_duration)
        if island_start >= horizon_s or island_end <= island_start:
            return

        if island_start > env.now:
            yield env.timeout(island_start - env.now)

        # Phase 1: ISLAND_NOW
        for dst in managed_nodes:
            emit(Message(
                msg_id=msg_id_fn(),
                created_ms=int(env.now * 1000),
                src=controller_node,
                dst=dst,
                msg_type=MsgType.PRIORITY_ACTION,
                priority=2,
                deadline_ms=300,
                size_bytes=280,
                requires_auth=True,
                payload={
                    "action": "island_now",
                    "duration_s": island_duration,
                    "control_signature": "quam_ctrl_v1",
                    "control_sender_role": "controller",
                    "supervisor_reason": "planned_islanding",
                },
            ))
            yield env.timeout(0.05)

        if island_end > env.now:
            yield env.timeout(island_end - env.now)

        # Phase 2: RECONNECT_GRID
        for dst in managed_nodes:
            emit(Message(
                msg_id=msg_id_fn(),
                created_ms=int(env.now * 1000),
                src=controller_node,
                dst=dst,
                msg_type=MsgType.PRIORITY_ACTION,
                priority=2,
                deadline_ms=300,
                size_bytes=280,
                requires_auth=True,
                payload={
                    "action": "reconnect_grid",
                    "duration_s": mg_params.restoration_duration_s,
                    "control_signature": "quam_ctrl_v1",
                    "control_sender_role": "controller",
                    "supervisor_reason": "planned_reconnect",
                },
            ))
            yield env.timeout(0.05)

        # Phase 3: RESTORE_LOAD (optional)
        if supervisory_restore_load:
            yield env.timeout(2.0)
            for dst in managed_nodes:
                emit(Message(
                    msg_id=msg_id_fn(),
                    created_ms=int(env.now * 1000),
                    src=controller_node,
                    dst=dst,
                    msg_type=MsgType.PRIORITY_ACTION,
                    priority=2,
                    deadline_ms=300,
                    size_bytes=280,
                    requires_auth=True,
                    payload={
                        "action": "restore_load",
                        "duration_s": 0,
                        "control_signature": "quam_ctrl_v1",
                        "control_sender_role": "controller",
                        "supervisor_reason": "post_reconnect_restore",
                    },
                ))
                yield env.timeout(0.05)

    env.process(_supervisor_control())

    # Microgrid stepper
    def _stepper():
        while env.now < horizon_s:
            t_s = int(env.now)
            is_attack, attack_label = get_attack_status(t_s)
            
            for name, mg in microgrids.items():
                stats = net_tracker.get_stats(
                    node=name,
                    now_s=float(env.now),
                    window_s=mg.params.control_window_s,
                    dt_s=1.0,
                    comm_base_kw=mg.params.comm_base_kw,
                    energy_per_byte_j=mg.params.energy_per_byte_j,
                    energy_per_key_bit_j=mg.params.energy_per_key_bit_j,
                    control_drop_penalty=mg.params.control_drop_penalty,
                    control_on_time_deadline_ms=mg.params.control_on_time_deadline_ms,
                )
                mg.update_comm_state(stats=stats, dt_s=1)

                # Generation: realistic DER mix or legacy Gaussian
                if mg.params.generation_model == "realistic" and name in gen_profiles:
                    solar_kw, wind_kw, smr_kw = gen_profiles[name].get_power(t_s)
                    gen = solar_kw + wind_kw + smr_kw
                else:
                    gen = max(0, rng.gauss(mg.params.gen_kw_mean, mg.params.gen_kw_sigma))
                    solar_kw = wind_kw = smr_kw = 0.0
                mg.step(t_s=t_s, dt_s=1, gen_kw_sample=gen,
                        solar_kw=solar_kw, wind_kw=wind_kw, smr_kw=smr_kw)

                # V6: Frequency dynamics step
                fd = freq_dynamics.get(name)
                if fd is not None:
                    is_island = (mg.mode in (GridMode.ISLANDED, GridMode.QUARANTINE))
                    # Effective generation includes battery response (fast freq support)
                    eff_gen = (mg.last_gen_kw + mg.last_import_kw
                               + mg.last_battery_discharge_kw - mg.last_battery_charge_kw)
                    fd.step(
                        t_s=t_s, dt_s=1.0,
                        gen_kw=eff_gen,
                        load_kw=mg.last_total_load_kw,
                        smr_kw=smr_kw, solar_kw=solar_kw, wind_kw=wind_kw,
                        is_islanded=is_island,
                        record=(t_s % 5 == 0),
                    )

                # V4: QRNG Measurement Challenge
                if sensor_challenger is not None and sensor_challenger.should_challenge(t_s, name):
                    # Expected values: from physics model (ground truth)
                    exp_s, exp_w, exp_m = solar_kw, wind_kw, smr_kw
                    # Reported values: what sensor "tells" controller
                    rep_s, rep_w, rep_m = solar_kw, wind_kw, smr_kw
                    fdi_active_now = False
                    if "fdi" in attacks:
                        for _aw_start, _aw_end in attack_windows:
                            if _aw_start <= t_s < _aw_end:
                                fdi_active_now = True
                                elapsed = t_s - _aw_start
                                ramp_frac = min(1.0, elapsed / 60.0)
                                gen_bias = -30.0 * ramp_frac
                                total_gen = max(1.0, solar_kw + wind_kw + smr_kw)
                                rep_s = max(0.0, solar_kw + gen_bias * (solar_kw / total_gen))
                                rep_w = max(0.0, wind_kw + gen_bias * (wind_kw / total_gen))
                                rep_m = max(0.0, smr_kw + gen_bias * (smr_kw / total_gen))
                                break
                    sensor_challenger.execute_challenge(
                        t_s=t_s, node=name,
                        expected_solar_kw=exp_s, expected_wind_kw=exp_w,
                        expected_smr_kw=exp_m,
                        reported_solar_kw=rep_s, reported_wind_kw=rep_w,
                        reported_smr_kw=rep_m,
                        fdi_active=fdi_active_now,
                    )

                if t_s % energy_record_interval == 0:
                    energy_ts.record(t_s, name, mg, is_attack, attack_label)

            # V3: DC power flow — compute line losses and voltage drops
            if t_s % energy_record_interval == 0 and energy_ts.records:
                injections = {}
                for name, mg in microgrids.items():
                    # Net injection = generation - total load (kW)
                    injections[name] = mg.last_gen_kw - mg.last_total_load_kw
                pf_result = power_flow.solve(injections)
                # Annotate the most recent energy records with power flow data
                # The last len(nodes) records correspond to the nodes just recorded
                n_nodes = len(microgrids)
                for rec in energy_ts.records[-n_nodes:]:
                    rec_name = rec.microgrid
                    rec.line_loss_kw = pf_result.total_loss_kw / max(1, n_nodes)
                    rec.voltage_violation = rec_name in pf_result.voltage_violations

            # V6: State estimation (every 10s) — detect FDI attacks
            #
            # Architecture:
            #   - SE reads bus injection measurements (gen − load) per node
            #   - During FDI attack windows the attacker corrupts sensor
            #     telemetry with a −30 kW ramp bias (non-stealthy: NOT a = Hc)
            #   - Classical chi-squared BDD can detect this non-stealthy bias
            #     because the attack vector is NOT in the column space of H
            #   - Quantum-authenticated channels (QKD + quantum control auth)
            #     prevent the attacker from injecting false data, so SE sees
            #     clean measurements → no residual inflation
            #
            # Correct behaviour:
            #   Baseline:                  0% detection (clean data)
            #   FDI / No Defense:          Partial detection via chi-squared
            #   FDI / Classical:           Same chi-squared detection + sensor
            #                              challenges add quarantine-based defense
            #   FDI / Quantum:             0% detection (quantum auth blocks FDI
            #                              at the channel level); 100% auth count
            #
            if state_estimator is not None and t_s % 10 == 0:
                # Determine if FDI is active right now
                _se_fdi_active = False
                _se_fdi_bias_kw = 0.0
                if "fdi" in attacks:
                    for _aw_s, _aw_e in attack_windows:
                        if _aw_s <= t_s < _aw_e:
                            _se_fdi_active = True
                            _elapsed = t_s - _aw_s
                            _ramp = min(1.0, _elapsed / 60.0)
                            _se_fdi_bias_kw = -30.0 * _ramp
                            break

                # Quantum auth: attacker CANNOT corrupt authenticated channels
                _se_quantum_auth = (enable_qkd and enable_quantum_control_auth)

                bus_inj = {}
                for name, mg in microgrids.items():
                    true_gen = mg.last_gen_kw
                    true_load = mg.last_total_load_kw
                    net_inj = true_gen - true_load

                    # Apply FDI bias unless quantum-authenticated
                    if _se_fdi_active and not _se_quantum_auth:
                        # Non-stealthy bias: proportional to generation share
                        total_gen_node = max(1.0, true_gen)
                        net_inj += _se_fdi_bias_kw  # attacker under-reports gen

                    # Normalize to per-unit
                    bus_inj[name] = net_inj / max(1.0, p_rated_kw)

                z = state_estimator.make_measurement_vector(bus_inj)
                _x_hat, _r, J = state_estimator.estimate(z)
                is_bad = state_estimator.is_bad_data(J)

                # Quantum-authenticated telemetry counter
                if _se_quantum_auth:
                    state_estimator.n_quantum_authenticated += 1

            # QBER recording
            if t_s % 10 == 0:  # FIXED: More frequent QBER recording
                for ek in qlayer.pools:
                    qber = qlayer.health[ek].qber_at(t_s)
                    intrusion_detector.record_qber(t_s, ek, qber)

                    if t_s % 30 == 0:
                        pingpong_alert = qlayer.pingpong_has_recent_alert(t_s)
                        # Key budget metrics
                        pool = qlayer.pools[ek]
                        eff_refill = qlayer.effective_refill_bits_per_s(ek, t_s)
                        consumed_rate = pool.total_consumed_bits / max(1, t_s)  # avg over sim
                        utilization = consumed_rate / max(1e-9, eff_refill) if eff_refill > 0 else 0.0
                        headroom_s = pool.level_bits / max(1e-9, consumed_rate) if consumed_rate > 0 else float("inf")
                        quantum_ts.record(
                            t_s=t_s, edge=f"{ek[0]}-{ek[1]}",
                            qber=qber, secret_fraction=secret_fraction_bb84(qber, qlayer.finite_key_params),
                            fidelity=fidelity_from_qber(qber),
                            pool_level=qlayer.pools[ek].level_bits,
                            is_attack=is_attack,
                            intrusion_alert=intrusion_detector.has_recent_alert(t_s),
                            abort_active=int(qlayer.health[ek].abort_active()),
                            pingpong_alert=int(pingpong_alert),
                            # V3: Key budget metrics
                            key_gen_rate_bits_per_s=round(eff_refill, 2),
                            key_consume_rate_bits_per_s=round(consumed_rate, 2),
                            key_utilization_ratio=round(min(utilization, 10.0), 4),
                            key_headroom_s=round(min(headroom_s, 99999), 1),
                            pool_capacity_bits=float(pool.capacity_bits),
                            auth_model=auth_model,
                        )

                # V2: Run Ping-Pong IDS probes
                qlayer.run_pingpong_probes(t_s)
            
            yield env.timeout(1)
    
    env.process(_stepper())

    # QAN events — dispatch classical vs quantum anonymous broadcast
    qan_specs = []
    cover_tracker = None
    ghz_tracker = None
    if qan_events > 0:
        if qan_mode == "quantum_ab":
            # ── Quantum Anonymous Broadcast (GHZ DC-net) ──
            ghz_tracker = GHZResourceTracker()
            qab_cfg = QABConfig(
                ghz_prep_success_prob=float(qab_ghz_prep_success_prob),
                ghz_fidelity_base=float(qab_ghz_fidelity_base),
                message_bits=int(qab_message_bits),
                decoherence_window_ms=int(qab_decoherence_window_ms),
                ghz_prep_time_ms=int(qab_ghz_prep_time_ms),
                auth_real_notify=bool(qan_auth_real),
            )
            qan = QuantumAnonymousBroadcast(
                env=env,
                rng=rng,
                cfg=qab_cfg,
                msg_id_fn=msg_id_fn,
                emit_fn=emit,
                n_participants=len(nodes),
                ghz_tracker=ghz_tracker,
            )
        else:
            # ── Classical QAN (cover-traffic mixing) ──
            cover_tracker = CoverTrafficTracker(energy_per_byte_j=mg_params.energy_per_byte_j)
            qan_cfg = QANConfig(
                cover_rate_per_s=float(qan_cover_rate_per_s),
                window_s=int(qan_window_s),
                mixing_delay_ms=int(qan_mixing_delay_ms),
                auth_cover=bool(qan_auth_cover),
                auth_real_notify=bool(qan_auth_real),
                auth_sync_burst=bool(qan_auth_sync),
            )
            qan = QANOrchestrator(
                env=env,
                rng=rng,
                cfg=qan_cfg,
                msg_id_fn=msg_id_fn,
                emit_fn=emit,
                cover_tracker=cover_tracker,
            )

        for i in range(qan_events):
            t_ev = rng.randint(int(horizon_s * 0.2), int(horizon_s * 0.8))
            true_sender = rng.choice(nodes)
            receiver = rng.choice([n for n in nodes if n != true_sender] or nodes)
            spec = qan.schedule_event(true_sender=true_sender, candidates=nodes,
                                      receiver=receiver, t_event_s=t_ev)
            qan_specs.append({"spec": spec, "true_sender": true_sender,
                             "receiver": receiver, "t_event_s": t_ev})
        if write_outputs and cover_tracker is not None:
            cover_tracker.write_csv(os.path.join(out_dir, "cover", f"cover_{scenario}_{topology}_seed{seed}.csv"))

    # Schedule attacks
    if "spoof" in attacks:
        # ── Smart attacker targeting ──
        # A sophisticated attacker (auth_bypass_prob > 0) employs stealth tactics:
        #   1. Picks a FIXED (controller, victim) pair per window to avoid
        #      cross-node correlation detection.
        #   2. Spaces attacks ≥35s apart to stay under per-source rate limits.
        #   3. Varies timing with large jitter to avoid pattern detection.
        # A naive attacker (auth_bypass_prob=0) uses random targeting (caught anyway).
        smart_attacker = spoof_auth_bypass_prob > 0
        for start, end in attack_windows:
            lvl = window_intensity_map.get((start, end), attack_intensity)
            s_cfg = spoof_window_cfgs.get(lvl, spoof_window_cfgs["S3"])
            interval_s = int(s_cfg.get("interval_s", 300))
            spoof_cfg = SpoofConfig(
                use_islanding=False,
                forced_shed_frac=float(s_cfg.get("forced_shed_frac", 0.55)),
                harm_duration_s=int(s_cfg.get("harm_duration_s", 45)),
                auth_bypass_prob=spoof_auth_bypass_prob,
            )
            spoof = SpoofingAttack(env=env, rng=rng, cfg=spoof_cfg, msg_id_fn=msg_id_fn, emit_fn=emit)

            if smart_attacker:
                # Smart: fixed pair per window, longer interval, bigger jitter
                fixed_controller = rng.choice(list(controller_nodes) or nodes)
                fixed_victim = rng.choice([n for n in managed_nodes if n != fixed_controller] or managed_nodes or nodes)
                stealth_interval = max(interval_s, 120)  # ≥2 min between attacks
                n_spoofs = max(1, (end - start) // max(60, stealth_interval))
                jitter_s = max(20, min(90, stealth_interval // 3))
            else:
                n_spoofs = max(1, (end - start) // max(60, interval_s))
                jitter_s = max(10, min(60, interval_s // 4))

            for j in range(n_spoofs):
                if smart_attacker:
                    t = start + j * stealth_interval + rng.randint(0, jitter_s)
                    controller = fixed_controller
                    victim = fixed_victim
                else:
                    t = start + j * interval_s + rng.randint(0, jitter_s)
                    controller = rng.choice(list(controller_nodes) or nodes)
                    victim = rng.choice([n for n in managed_nodes if n != controller] or managed_nodes or nodes)
                if t < end:
                    spoof.schedule_spoof(t_spoof_s=t, controller=controller, victim=victim,
                                        inferred_sender=controller, label=f"spoof_{lvl.lower()}")
    
    if "exhaust" in attacks:
        try:
            strategy = ExhaustTargetStrategy(exhaust_strategy)
        except Exception:
            strategy = ExhaustTargetStrategy.UNIFORM

        for start, end in attack_windows:
            lvl = window_intensity_map.get((start, end), attack_intensity)
            win_cfg = atk_cfgs.get(lvl, atk_cfgs["S3"])
            if strategy == ExhaustTargetStrategy.UNIFORM:
                ex_cfg = ExhaustConfig(start_s=start, end_s=end, rate_per_s=win_cfg["exhaust_rate"])
                ex = KeyExhaustionAttack(env=env, rng=rng, cfg=ex_cfg, msg_id_fn=msg_id_fn, emit_fn=emit)
                ex.schedule(src_nodes=nodes, dst_nodes=nodes)
            else:
                ex_cfg = TargetedExhaustConfig(
                    start_s=start,
                    end_s=end,
                    rate_per_s=win_cfg["exhaust_rate"],
                    target_strategy=strategy,
                    focus_ratio=exhaust_focus,
                )
                ex = TargetedKeyExhaustionAttack(
                    env=env,
                    rng=rng,
                    cfg=ex_cfg,
                    msg_id_fn=msg_id_fn,
                    emit_fn=emit,
                    topology=graph,
                    traffic_observer=analyzer,
                )
                ex.schedule()
    
    if "quantum" in attacks:
        target_edge_count = max(1, min(int(quantum_target_edge_count), len(edges))) if edges else 0
        quantum_target_edges = edges[:target_edge_count]
        for start, end in attack_windows:
            lvl = window_intensity_map.get((start, end), attack_intensity)
            win_cfg = atk_cfgs.get(lvl, atk_cfgs["S3"])
            n_segments = rng.randint(3, 5)
            segments = _split_window(rng, start, end, n_segments)
            for (u, v) in quantum_target_edges:
                ek = edge_key(u, v)
                for seg_start, seg_end in segments:
                    qber_val = (
                        float(quantum_qber_override)
                        if quantum_qber_override is not None
                        else _sample_qber(win_cfg, rng)
                    )
                    window = QBERWindow(start_s=seg_start, end_s=seg_end,
                                       absolute_qber=qber_val, label="quantum_disturb")
                    qlayer.add_qber_window(ek, window)

        # ── Random eavesdropping events (intercept-resend) ──
        # A real attacker performing quantum disturbance also performs
        # eavesdropping: intercept a fraction of qubits, measure them,
        # and forward copies.  These appear as random bursts within the
        # attack windows.  Ping-Pong IDS should detect them.
        # Target the SAME edges as quantum disturbance (edges[:2]) since
        # Eve physically taps the same fibre she is disturbing.
        eve_target_edges = quantum_target_edges
        for start, end in attack_windows:
            dur = end - start
            if dur <= 0:
                continue
            # Schedule 2-4 random eavesdropping bursts per attack window
            n_eve_bursts = rng.randint(2, 4)
            for _ in range(n_eve_bursts):
                burst_start = rng.randint(start, max(start, end - 30))
                burst_dur = rng.randint(15, min(60, max(15, dur // 3)))
                burst_end = min(end, burst_start + burst_dur)
                # Intercept fraction: 0.15-0.45 (realistic for information gain)
                intercept = round(rng.uniform(0.15, 0.45), 2)
                for (u, v) in eve_target_edges:
                    ek = edge_key(u, v)
                    qlayer.add_eve_window(ek, EaveWindow(
                        start_s=burst_start, end_s=burst_end,
                        intercept_fraction=intercept,
                        label="eavesdrop_burst",
                    ))

    # V2: Insider threat attack
    if "insider" in attacks:
        for start, end in attack_windows:
            insider_node = rng.choice(list(controller_nodes) or nodes)
            victim_nodes = [n for n in managed_nodes if n != insider_node] or managed_nodes or nodes
            insider_cfg = InsiderThreatConfig(
                start_s=start, end_s=end,
                rate_per_s=0.5,
                use_legitimate_credentials=True,
                target_shed_frac=0.60,
                harm_duration_s=45,
            )
            insider = InsiderThreatAttack(
                env=env, rng=rng, cfg=insider_cfg,
                msg_id_fn=msg_id_fn, emit_fn=emit,
            )
            insider.schedule(insider_node=insider_node, victim_nodes=victim_nodes)

    # Node-level spoofing: honest source (src=compromised_node)
    if "nodespoof" in attacks:
        comp_node = compromised_node or sorted(managed_nodes)[len(managed_nodes) // 2]
        ns_controller_node = rng.choice(list(controller_nodes) or nodes)
        victim_nodes = [n for n in managed_nodes if n != comp_node] or managed_nodes or nodes
        for start, end in attack_windows:
            lvl = window_intensity_map.get((start, end), attack_intensity)
            s_cfg = spoof_window_cfgs.get(lvl, spoof_window_cfgs["S3"])
            ns_cfg = NodeLevelSpoofConfig(
                compromised_node=comp_node,
                controller_node=ns_controller_node,
                forge_source=False,
                rate_per_s=0.5,
                forced_shed_frac=float(s_cfg.get("forced_shed_frac", 0.55)),
                harm_duration_s=int(s_cfg.get("harm_duration_s", 45)),
                label=f"nodespoof_{lvl.lower()}",
            )
            ns_attack = NodeLevelSpoofingAttack(
                env=env, rng=rng, cfg=ns_cfg,
                msg_id_fn=msg_id_fn, emit_fn=emit,
            )
            ns_attack.schedule(start_s=start, end_s=end, victim_nodes=victim_nodes)

    # Node-level spoofing: forged source (src=controller, injection_node=compromised)
    if "nodespoofforged" in attacks:
        comp_node = compromised_node or sorted(managed_nodes)[len(managed_nodes) // 2]
        nsf_controller_node = rng.choice(list(controller_nodes) or nodes)
        victim_nodes = [n for n in managed_nodes if n != comp_node] or managed_nodes or nodes
        for start, end in attack_windows:
            lvl = window_intensity_map.get((start, end), attack_intensity)
            s_cfg = spoof_window_cfgs.get(lvl, spoof_window_cfgs["S3"])
            ns_cfg = NodeLevelSpoofConfig(
                compromised_node=comp_node,
                controller_node=nsf_controller_node,
                forge_source=True,
                rate_per_s=0.5,
                forced_shed_frac=float(s_cfg.get("forced_shed_frac", 0.55)),
                harm_duration_s=int(s_cfg.get("harm_duration_s", 45)),
                label=f"nodespoofforged_{lvl.lower()}",
            )
            ns_attack = NodeLevelSpoofingAttack(
                env=env, rng=rng, cfg=ns_cfg,
                msg_id_fn=msg_id_fn, emit_fn=emit,
            )
            ns_attack.schedule(start_s=start, end_s=end, victim_nodes=victim_nodes)

    # V11: Coordinated multi-node attack (APT)
    if "coordinated" in attacks:
        # Compromise ~30% of non-controller nodes (min 2)
        non_ctrl = [n for n in managed_nodes if n not in controller_nodes]
        n_compromised = max(2, len(non_ctrl) // 3)
        compromised_set = sorted(non_ctrl)[:n_compromised]
        coord_ctrl = rng.choice(list(controller_nodes) or nodes)
        victim_nodes = [n for n in managed_nodes if n not in compromised_set] or managed_nodes or nodes
        for start, end in attack_windows:
            lvl = window_intensity_map.get((start, end), attack_intensity)
            s_cfg = spoof_window_cfgs.get(lvl, spoof_window_cfgs["S3"])
            coord_cfg = CoordinatedAttackConfig(
                compromised_nodes=compromised_set,
                controller_node=coord_ctrl,
                forge_source=True,
                rate_per_s_per_node=0.08,
                forced_shed_frac=float(s_cfg.get("forced_shed_frac", 0.55)),
                harm_duration_s=int(s_cfg.get("harm_duration_s", 45)),
                coordination_phase_offset_s=2.0,
                shed_frac_jitter=0.05,
                label=f"coordinated_{lvl.lower()}",
            )
            coord_attack = CoordinatedMultiNodeAttack(
                env=env, rng=rng, cfg=coord_cfg,
                msg_id_fn=msg_id_fn, emit_fn=emit,
            )
            coord_attack.schedule(start_s=start, end_s=end, victim_nodes=victim_nodes)

    # V3: False Data Injection attack
    if "fdi" in attacks:
        fdi_ctrl = rng.choice(list(controller_nodes) or nodes)
        sensor_nodes = [n for n in managed_nodes if n != fdi_ctrl] or managed_nodes or nodes
        for start, end in attack_windows:
            fdi_cfg = FDIAttackConfig(
                start_s=start, end_s=end,
                injection_rate_per_s=0.5,
                gen_bias_kw=-30.0,
                load_bias_kw=20.0,
                target_sensors="generation",
                stealthy=True,
                ramp_duration_s=60.0,
                fdi_forge_prob=0.15,
                label="fdi",
            )
            fdi_attack = FalseDataInjectionAttack(
                env=env, rng=rng, cfg=fdi_cfg,
                msg_id_fn=msg_id_fn, emit_fn=emit,
            )
            fdi_attack.schedule(
                sensor_nodes=sensor_nodes,
                controller_node=fdi_ctrl,
            )
        # Register FDI start times for challenge detection latency metrics
        if sensor_challenger is not None:
            for start, end in attack_windows:
                for sn in sensor_nodes:
                    sensor_challenger.register_fdi_start(sn, start)

    # V3: Classical Man-in-the-Middle attack
    mitm_attack = None
    if "mitm" in attacks:
        for start, end in attack_windows:
            mitm_cfg = MITMAttackConfig(
                start_s=start, end_s=end,
                intercept_prob=0.3,
                modify_prob=0.8,
                shed_override_frac=0.70,
                target_edges=None,  # All edges
                classical_forge_prob=0.10,
                label="mitm",
            )
            mitm_attack = ClassicalMITMAttack(
                env=env, rng=rng, cfg=mitm_cfg,
            )
            # MITM operates on messages in transit — activation handled
            # by the stepper loop checking mitm_attack.is_active(t_s)

    # Run
    env.run(until=horizon_s)

    # Deanon results
    _deanon_correct = []
    _deanon_entropy = []
    _deanon_top1prob = []
    for i, info in enumerate(qan_specs):
        analyzer.arm_event(info["spec"])
        result = analyzer.infer_sender()
        logger.record_deanon_result(
            event_id=f"{scenario}_{topology}_seed{seed}_ev{i}",
            t_event_s=info["t_event_s"],
            receiver=info["receiver"],
            true_sender=info["true_sender"],
            inferred_sender=result["top1"],
            top1_prob=result["top1_prob"],
            entropy_bits=result["entropy_bits"],
            top1_candidate=result.get("top1_candidate", result.get("top1", "")),
            top2_prob=result.get("top2_prob", float("nan")),
            top1_margin=result.get("top1_margin", float("nan")),
            abstained=1 if bool(result.get("abstained", False)) else 0,
            n_obs_window=int(result.get("n_obs_window", 0)),
            prior_blend_weight=result.get("prior_blend_weight", float("nan")),
        )
        _deanon_correct.append(1.0 if result["top1"] == info["true_sender"] else 0.0)
        _deanon_entropy.append(float(result.get("entropy_bits", 0.0)))
        _deanon_top1prob.append(float(result.get("top1_prob", 0.0)))

    # Record messages
    for m in created_msgs:
        logger.record_message(m)
    
    # Write outputs
    if write_outputs:
        logger.write_message_csv(os.path.join(out_dir, "messages", f"messages_{scenario}_{topology}_seed{seed}.csv"))
        logger.write_deanon_csv(os.path.join(out_dir, "deanon", f"deanon_{scenario}_{topology}_seed{seed}.csv"))
        energy_ts.write_csv(os.path.join(out_dir, "energy", f"energy_{scenario}_{topology}_seed{seed}.csv"))
        quantum_ts.write_csv(os.path.join(out_dir, "timeseries", f"quantum_{scenario}_{topology}_seed{seed}.csv"))
    
    # Metrics
    total_eens = sum(mg.eens_total_kwh for mg in microgrids.values())
    critical_eens = sum(mg.eens_critical_kwh for mg in microgrids.values())

    # Per-domain EENS for federated topologies
    _eens_grid_a = 0.0
    _eens_grid_b = 0.0
    if _is_federated:
        for mg_name, mg in microgrids.items():
            domain = graph.nodes.get(mg_name, {}).get("domain", "grid_a")
            if domain == "grid_b":
                _eens_grid_b += mg.eens_total_kwh
            else:
                _eens_grid_a += mg.eens_total_kwh
    total_control = sum(mg.control_msgs_received for mg in microgrids.values())
    total_comm_energy = sum(getattr(mg, "comm_energy_kwh", 0.0) for mg in microgrids.values())
    avg_control_quality = (sum(getattr(mg, "control_quality", 1.0) for mg in microgrids.values()) /
                           max(1, len(microgrids)))
    if qlayer.pools:
        avg_link_distance_km = sum(p.link_params.distance_km for p in qlayer.pools.values()) / len(qlayer.pools)
        avg_distance_factor = sum(p.link_params.distance_factor() for p in qlayer.pools.values()) / len(qlayer.pools)
        effective_key_rate_reduction_pct = (1.0 - avg_distance_factor) * 100.0
    else:
        avg_link_distance_km = float("nan")
        avg_distance_factor = float("nan")
        effective_key_rate_reduction_pct = float("nan")
    
    delivered = [m for m in created_msgs if "delivered" in str(m.status)]

    congestion_stats = summarize_link_congestion(links)

    # ------------------------------------------------------------------
    # Sanity checks / invariants (debugging safety net)
    # ------------------------------------------------------------------
    # Key pool conservation (per edge):
    #   initial + added_effective - consumed == final
    # Note: add_bits() caps at capacity, so we also track spilled bits for transparency.
    pool_abs_errs: List[float] = []
    key_initial_sum = 0.0
    key_added_sum = 0.0
    key_spilled_sum = 0.0
    key_consumed_sum = 0.0
    key_final_sum = 0.0
    key_failed_consume_bits_sum = 0.0
    key_failed_consume_count_sum = 0
    key_reserved_saves_sum = 0
    key_source_rate_blocks_sum = 0
    key_emergency_grants_sum = 0
    for pool in qlayer.pools.values():
        key_initial_sum += float(getattr(pool, "initial_level_bits", 0.0))
        key_added_sum += float(getattr(pool, "total_added_bits", 0.0))
        key_spilled_sum += float(getattr(pool, "total_spilled_bits", 0.0))
        key_consumed_sum += float(getattr(pool, "total_consumed_bits", 0.0))
        key_final_sum += float(getattr(pool, "level_bits", 0.0))
        key_failed_consume_bits_sum += float(getattr(pool, "total_failed_consume_bits", 0.0))
        key_failed_consume_count_sum += int(getattr(pool, "failed_consume_count", 0))
        key_reserved_saves_sum += int(getattr(pool, "reserved_saves", 0))
        key_source_rate_blocks_sum += int(getattr(pool, "source_rate_blocks", 0))
        key_emergency_grants_sum += int(getattr(pool, "emergency_grants", 0))

        err = (float(getattr(pool, "initial_level_bits", 0.0)) + float(getattr(pool, "total_added_bits", 0.0))) - (
            float(getattr(pool, "total_consumed_bits", 0.0)) + float(getattr(pool, "level_bits", 0.0))
        )
        pool_abs_errs.append(abs(float(err)))

    key_conservation_abs_err_bits_max = max(pool_abs_errs) if pool_abs_errs else 0.0
    key_conservation_abs_err_bits_mean = (sum(pool_abs_errs) / len(pool_abs_errs)) if pool_abs_errs else 0.0

    # Hard invariant: keys in a pool are conserved (within generous tolerance).
    # If this trips, there is a double-counting or mutation bug in QKD accounting.
    if key_conservation_abs_err_bits_max > 1000.0:
        raise AssertionError(
            f"QKD key conservation violated: max_abs_err_bits={key_conservation_abs_err_bits_max:.2f}"
        )

    # Routing diagnostics (ECMP alternative availability)
    ecmp_stats = getattr(net, "_ecmp_stats", None)
    ecmp_total = float(ecmp_stats.get("total", 0.0)) if isinstance(ecmp_stats, dict) else 0.0
    ecmp_had_alt = float(ecmp_stats.get("had_alternatives", 0.0)) if isinstance(ecmp_stats, dict) else 0.0
    ecmp_alt_paths_mean = (
        float(ecmp_stats.get("alt_paths_sum", 0.0)) / ecmp_total
        if (isinstance(ecmp_stats, dict) and ecmp_total > 0)
        else 0.0
    )
    ecmp_alt_paths_max = float(ecmp_stats.get("alt_paths_max", 0.0)) if isinstance(ecmp_stats, dict) else 0.0
    ecmp_had_alt_ratio = (ecmp_had_alt / ecmp_total) if ecmp_total > 0 else 0.0

    # Attack traffic distribution (by attempted hop / edge)
    attack_hits = getattr(net, "_edge_hits_attack", None)
    total_hits = getattr(net, "_edge_hits_total", None)

    def _fmt_hits(counter_obj, n: int = 10) -> str:
        try:
            items = list(counter_obj.most_common(n))
        except Exception:
            return ""
        return "|".join([f"{u}-{v}:{int(c)}" for (u, v), c in items])

    net_edge_hits_attack_top10 = _fmt_hits(attack_hits, 10) if attack_hits else ""
    net_edge_hits_total_top10 = _fmt_hits(total_hits, 10) if total_hits else ""
    prekey_blocked_msgs = [
        m for m in created_msgs
        if int((m.payload or {}).get("prekey_blocked", 0)) == 1
    ]
    key_bits_saved_prekey_est_sum = float(sum(
        float((m.payload or {}).get("key_bits_saved_prekey_est", 0.0))
        for m in prekey_blocked_msgs
    ))
    emergency_mode_msg_count = int(sum(
        1 for m in created_msgs if int((m.payload or {}).get("emergency_mode", 0)) == 1
    ))
    reduced_tag_msg_count = int(sum(
        1 for m in created_msgs if int((m.payload or {}).get("reduced_auth_tag", 0)) == 1
    ))
    
    extra = {
        "attacks": ",".join(attacks) if attacks else "none",
        "defense_mode": defense_mode,
        "auth_model": auth_model,
        "qec_code_distance": int(qec_code_distance),
        "e2e_distillation_rounds": int(e2e_distillation_rounds),
        "e2e_swap_success_prob": float(e2e_swap_success_prob),
        "enable_quantum_control_auth": int(bool(enable_quantum_control_auth)),
        "quantum_control_token_ttl_ms": int(quantum_control_token_ttl_ms),
        "spoof_auth_bypass_prob": spoof_auth_bypass_prob,
        "compromised_node": compromised_node or "",
        "central_controller_node": controller_node,
        "controller_nodes": ",".join(controller_nodes) if controller_nodes else "",
        "managed_microgrid_nodes": ",".join(managed_nodes) if managed_nodes else "",
        "network_node_count_total": len(nodes),
        "physical_microgrid_count": len(managed_nodes),
        "enable_supervisory_islanding": int(bool(enable_supervisory_islanding)),
        "supervisory_island_start_s": int(supervisory_island_start_s) if supervisory_island_start_s is not None else -1,
        "supervisory_island_duration_s": int(supervisory_island_duration_s),
        "attack_intensity": attack_intensity,
        "attack_intensity_mode": "random_windows" if random_window_intensity else "fixed",
        "attack_window_intensity_sequence": ";".join(
            f"{s}-{e}:{window_intensity_map.get((s, e), attack_intensity)}" for (s, e) in attack_windows
        ) if attack_windows else "none",
        "enable_qkd": enable_qkd,
        "eens_total_kwh": total_eens,
        "eens_critical_kwh": critical_eens,
        "generation_model": mg_params.generation_model,
        "solar_capacity_kw": mg_params.solar_capacity_kw,
        "wind_capacity_kw": mg_params.wind_capacity_kw,
        "smr_capacity_kw": mg_params.smr_capacity_kw,
        "control_msgs_received": total_control,
        "comm_energy_kwh": total_comm_energy,
        "avg_control_quality": avg_control_quality,
        "avg_link_distance_km": avg_link_distance_km,
        "avg_distance_factor": avg_distance_factor,
        "effective_key_rate_reduction_pct": effective_key_rate_reduction_pct,
        "degraded_threshold_sf": gate.cfg.degraded_secret_fraction,
        "degraded_threshold_qber_approx": secret_fraction_to_qber_approx(gate.cfg.degraded_secret_fraction),
        "degraded_mode_triggers": gate.stats.get("degraded_mode_triggers", 0),
        "n_attack_windows": len(attack_windows),
        "total_attack_duration_s": sum(e - s for s, e in attack_windows),
        "n_intrusion_alerts": len(intrusion_detector.alerts),
        "defense_blocked_degraded": gate.stats["blocked_degraded"],
        "defense_blocked_intrusion": gate.stats["blocked_intrusion"],
        "defense_blocked_rate_limit": gate.stats["blocked_rate_limit"],
        "defense_blocked_signature": gate.stats["blocked_signature"],
        "defense_blocked_per_source_rate": gate.stats.get("blocked_per_source_rate", 0),
        "defense_blocked_implausible": gate.stats.get("blocked_implausible", 0),
        "defense_blocked_cross_node": gate.stats.get("blocked_cross_node", 0),
        "defense_blocked_quarantine_mgr": gate.stats.get("blocked_quarantine_mgr", 0),
        "defense_blocked_quantum_control_token": gate.stats.get("blocked_quantum_control_token", 0),
        "defense_allowed": gate.stats["allowed"],
        "prekey_checked_total": admission_gate.stats.get("checked_total", 0),
        "prekey_allowed_total": admission_gate.stats.get("allowed_total", 0),
        "prekey_blocked_total": admission_gate.stats.get("blocked_total", 0),
        "prekey_blocked_rate_limit": admission_gate.stats.get("blocked_prekey_rate_limit", 0),
        "prekey_blocked_per_source_rate": admission_gate.stats.get("blocked_prekey_per_source_rate", 0),
        "prekey_blocked_intrusion": admission_gate.stats.get("blocked_prekey_intrusion", 0),
        "prekey_blocked_cross_node": admission_gate.stats.get("blocked_prekey_cross_node", 0),
        "prekey_blocked_quarantine_mgr": admission_gate.stats.get("blocked_prekey_quarantine_mgr", 0),
        "prekey_blocked_degraded": admission_gate.stats.get("blocked_prekey_degraded", 0),
        "prekey_blocked_quantum_token": admission_gate.stats.get("blocked_prekey_quantum_token", 0),
        "prekey_key_bits_saved_est_sum": key_bits_saved_prekey_est_sum,
        "quantum_defense_enabled": int(bool(getattr(qlayer, "_enable_priority_reservation", False) or getattr(qlayer, "_enable_source_key_rate_limit", False))),
        "quantum_priority_reservation_enabled": int(bool(getattr(qlayer, "_enable_priority_reservation", False))),
        "quantum_reservation_ratio": float(getattr(qlayer, "_reservation_ratio", 0.0)),
        "quantum_source_key_rate_limit_enabled": int(bool(getattr(qlayer, "_enable_source_key_rate_limit", False))),
        "quantum_source_key_rate_bits_per_s": float(getattr(qlayer, "_source_key_rate_bits_per_s", 0.0)),
        "quantum_emergency_mode_enabled": int(bool(getattr(qlayer, "_enable_emergency_mode", False))),
        "quantum_emergency_threshold_ratio": float(getattr(qlayer, "_emergency_threshold_ratio", 0.0)),
        "quantum_emergency_tag_bits": int(getattr(qlayer, "_emergency_tag_bits", 0)),
        "quantum_reserved_saves_sum": int(key_reserved_saves_sum),
        "quantum_source_rate_blocks_sum": int(key_source_rate_blocks_sum),
        "quantum_emergency_grants_sum": int(key_emergency_grants_sum),
        "quantum_emergency_mode_msg_count": int(emergency_mode_msg_count),
        "quantum_reduced_tag_msg_count": int(reduced_tag_msg_count),
        "quantum_control_tokens_attached": int(qlayer.protocol_stats.get("control_tokens_attached", 0)),
        "quantum_control_tokens_verified": int(qlayer.protocol_stats.get("control_tokens_verified", 0)),
        "quantum_control_tokens_rejected": int(qlayer.protocol_stats.get("control_tokens_rejected", 0)),
        # QAN settings (for experiment sweep bookkeeping)
        "qan_events_requested": int(qan_events),
        "qan_cover_rate_per_s": float(qan_cover_rate_per_s),
        "qan_window_s": int(qan_window_s),
        "qan_mixing_delay_ms": int(qan_mixing_delay_ms),
        "qan_auth_cover": int(bool(qan_auth_cover)),
        "qan_auth_real_notify": int(bool(qan_auth_real)),
        "qan_auth_sync_burst": int(bool(qan_auth_sync)),
        # Deanonymization summary metrics
        "deanon_top1_acc": float(sum(_deanon_correct) / max(1, len(_deanon_correct))) if _deanon_correct else 0.0,
        "deanon_entropy_mean_bits": float(sum(_deanon_entropy) / max(1, len(_deanon_entropy))) if _deanon_entropy else 0.0,
        "deanon_top1prob_mean": float(sum(_deanon_top1prob) / max(1, len(_deanon_top1prob))) if _deanon_top1prob else 0.0,
        # Federated per-domain EENS
        "eens_grid_a_kwh": float(_eens_grid_a),
        "eens_grid_b_kwh": float(_eens_grid_b),
        "is_federated": int(_is_federated),
        # Gate delay overrides (if used)
        "gate_verification_delay_ms": int(gate_cfg.verification_delay_ms),
        "gate_degraded_verification_delay_ms": int(gate_cfg.degraded_verification_delay_ms),
        # Sanity / invariants
        "key_conservation_abs_err_bits_max": float(key_conservation_abs_err_bits_max),
        "key_conservation_abs_err_bits_mean": float(key_conservation_abs_err_bits_mean),
        "key_initial_bits_sum": float(key_initial_sum),
        "key_added_bits_sum": float(key_added_sum),
        "key_spilled_bits_sum": float(key_spilled_sum),
        "key_consumed_bits_sum": float(key_consumed_sum),
        "key_final_bits_sum": float(key_final_sum),
        "key_failed_consume_bits_sum": float(key_failed_consume_bits_sum),
        "key_failed_consume_count_sum": int(key_failed_consume_count_sum),
        # Routing diagnostics
        "ecmp_total_path_calls": float(ecmp_total),
        "ecmp_had_alternatives": float(ecmp_had_alt),
        "ecmp_had_alternatives_ratio": float(ecmp_had_alt_ratio),
        "ecmp_alt_paths_mean": float(ecmp_alt_paths_mean),
        "ecmp_alt_paths_max": float(ecmp_alt_paths_max),
        # Attack traffic diagnostics
        "net_edge_hits_attack_top10": str(net_edge_hits_attack_top10),
        "net_edge_hits_total_top10": str(net_edge_hits_total_top10),
        **congestion_stats,
    }
    if cover_tracker is not None:
        extra.update(cover_tracker.get_summary())
    if ghz_tracker is not None:
        extra.update(ghz_tracker.get_summary())
    extra["qan_mode"] = str(qan_mode)
    if qlayer.nonce_mgr is not None:
        extra.update(qlayer.nonce_mgr.qrng_stats())

    # V2: Multi-protocol quantum layer stats
    proto_stats = qlayer.get_protocol_stats()
    for k, v in proto_stats.items():
        extra[f"qproto_{k}"] = v
    extra["enable_quantum_protocols"] = int(enable_quantum_protocols)

    # V3: E2E relay pool stats
    e2e_stats = qlayer.get_e2e_pool_stats()
    for k, v in e2e_stats.items():
        extra[f"e2e_{k}" if not k.startswith("e2e_") else k] = v

    # V4: QRNG Sensor Challenge metrics
    if sensor_challenger is not None:
        challenge_stats = sensor_challenger.get_stats()
        for k, v in challenge_stats.items():
            if not isinstance(v, dict):
                extra[k] = v

    # V6: Resilience metrics (SAIDI / SAIFI / LOLP / ASAI)
    resilience = compute_resilience_metrics(microgrids, horizon_s, n_customers_per_mg=200)
    extra.update(resilience)

    # V6: Frequency dynamics aggregate stats
    freq_nadir_all = min(
        (fd.nadir_hz for fd in freq_dynamics.values()), default=60.0
    )
    freq_zenith_all = max(
        (fd.zenith_hz for fd in freq_dynamics.values()), default=60.0
    )
    freq_max_rocof_all = max(
        (fd.max_rocof for fd in freq_dynamics.values()), default=0.0
    )
    freq_violation_s_sum = sum(fd.violation_s for fd in freq_dynamics.values())
    freq_ufls_s_sum = sum(fd.ufls_s for fd in freq_dynamics.values())
    extra.update({
        "freq_nadir_hz": round(freq_nadir_all, 4),
        "freq_zenith_hz": round(freq_zenith_all, 4),
        "freq_max_rocof_hz_s": round(freq_max_rocof_all, 4),
        "freq_violation_s": round(freq_violation_s_sum, 2),
        "freq_ufls_s": round(freq_ufls_s_sum, 2),
    })

    # V6: State estimator stats
    if state_estimator is not None:
        extra.update(state_estimator.get_stats())

    return logger.summarize_run(
        scenario=scenario, topology=topology, seed=seed,
        horizon_ms=horizon_s * 1000, extra=extra,
    )


# ============================================================================
# CLI
# ============================================================================

def get_all_scenarios():
    scenarios = ["baseline"]
    for atk in ["spoof", "exhaust", "quantum", "insider", "spoof_exhaust",
                 "all_attacks", "all_attacks_v2"]:
        for defense in [
            "none",
            "ratelimit",
            "block",
            "delay",
            "intrusion",
            "adaptive",
            "signature",
            "all",
            "ratelimit_v2",
            "intrusion_v2",
            "plausibility",
            "correlation",
            "quarantine_v2",
            "hardened",
            "hardened_v2",
            "gate_only",
            "quantum_only",
        ]:
            scenarios.append(f"{atk}_def_{defense}")
    return scenarios


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="experiment")
    ap.add_argument("--horizon_s", type=int, default=3600)
    ap.add_argument("--nodes", nargs="*", default=DEFAULT_NODES)
    ap.add_argument("--topologies", nargs="*", default=["ring"],
                    help="Network topologies (e.g., ring, star, mesh, two_cluster_bridge, ieee13, ieee34, ieee37, ieee123)")
    ap.add_argument("--route_policy", default="shortest",
                    choices=["shortest", "ecmp", "k_shortest", "k_shortest_weighted", "disjoint", "load_aware"],
                    help="Routing policy for message delivery")
    ap.add_argument("--k_paths", type=int, default=3,
                    help="Number of candidate paths for k_shortest policies")
    ap.add_argument("--use_grid_link_types", action="store_true",
                    help="Infer backbone/feeder/lateral link params automatically (affects latency/loss/bandwidth)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--scenarios", nargs="*", default=["baseline", "spoof_def_none", "spoof_def_all"])
    ap.add_argument("--attack_intensity", default="S3")
    ap.add_argument("--attack_duration", type=int, default=None)
    ap.add_argument("--distributed_attacks", action="store_true")
    ap.add_argument("--num_attack_windows", type=int, default=5)
    ap.add_argument("--exhaust_strategy", default="uniform",
                   choices=["uniform", "bottleneck", "bridge", "high_traffic", "single_link", "bypassable", "star_center"],
                   help="Key exhaustion targeting strategy")
    ap.add_argument("--exhaust_focus", type=float, default=0.8,
                   help="Fraction of exhaustion traffic focused on targets (0-1)")
    ap.add_argument("--qan_events", type=int, default=3)
    ap.add_argument("--qan_cover_rate_per_s", type=float, default=4.0,
                    help="Cover messages per second during QAN event window (0 disables cover)")
    ap.add_argument("--qan_window_s", type=int, default=5,
                    help="QAN event half-window (covers run from t_event-window to t_event+window)")
    ap.add_argument("--qan_mixing_delay_ms", type=int, default=40,
                    help="Random mixing jitter (ms) applied to the real notify time within the QAN window")
    ap.add_argument("--qan_auth_cover", action="store_true",
                    help="Require auth (key usage) for QAN cover messages")
    ap.add_argument("--qan_auth_real", action="store_true",
                    help="Require auth (key usage) for the real QAN notify message")
    ap.add_argument("--qan_auth_sync", action="store_true",
                    help="Require auth (key usage) for sync-burst cover messages")
    ap.add_argument("--energy_interval", type=int, default=10)
    ap.add_argument("--rotation_policy", default="none",
                    choices=["none", "conservative", "moderate", "aggressive", "paranoid"])
    ap.add_argument("--verification_delay_ms", type=int, default=None,
                    help="Override gate verification delay (ms) for all defenses (useful for sweeps)")
    ap.add_argument("--degraded_verification_delay_ms", type=int, default=None,
                    help="Override extra verification delay (ms) in degraded mode (useful for sweeps)")
    ap.add_argument("--link_distance_km", type=float, default=10.0,
                    help="Default QKD link distance in km (affects key rate)")
    ap.add_argument("--fiber_loss_db_km", type=float, default=0.2,
                    help="Fiber attenuation in dB/km")
    ap.add_argument("--finite_key", default="disabled",
                    choices=["disabled", "large_block", "medium_block", "small_block", "high_security"],
                    help="Finite-key correction preset")
    ap.add_argument("--finite_key_block_bits", type=int, default=None,
                    help="Override finite-key block size (bits)")
    ap.add_argument("--finite_key_security_log", type=int, default=None,
                    help="Override -log10(eps) for finite-key correction")
    ap.add_argument("--degraded_threshold", default="moderate",
                    help="Degraded mode threshold preset or float (0-1)")
    ap.add_argument("--random_window_intensity", action="store_true",
                    help="Randomize attack intensity per attack window (drawn from S1..S5)")
    ap.add_argument("--window_intensity_pool", nargs="*", default=["S1", "S2", "S3", "S4", "S5"],
                    help="Allowed intensity levels for per-window randomization")
    ap.add_argument("--qrng_pool_bits", type=float, default=None,
                    help="Override QRNG pool capacity (bits). If unset, uses default.")
    ap.add_argument("--qrng_rate_bits_per_s", type=float, default=None,
                    help="Override QRNG generation rate (bits/s). If unset, uses default.")
    ap.add_argument("--enable_power_sharing", action="store_true")
    ap.add_argument("--no_qkd", action="store_true")
    # V2: Multi-protocol quantum layer
    ap.add_argument("--enable_quantum_protocols", action="store_true",
                    help="Enable multi-protocol quantum layer (KAK, E91, Ping-Pong IDS)")
    ap.add_argument("--pingpong_variant", default="ghz",
                    choices=["bell", "ghz", "cluster"],
                    help="Ping-Pong IDS variant (bell=50%%, ghz=75%%, cluster=94%% detection)")
    ap.add_argument("--pingpong_interval_s", type=float, default=5.0,
                    help="Ping-Pong IDS probe interval in seconds")
    # V5: Quantum transport tuning
    ap.add_argument("--qec_code_distance", type=int, default=3)
    ap.add_argument("--e2e_distillation_rounds", type=int, default=1)
    ap.add_argument("--e2e_swap_success_prob", type=float, default=0.5)
    ap.add_argument("--enable_quantum_control_auth", action="store_true")
    ap.add_argument("--quantum_control_token_ttl_ms", type=int, default=1500)
    # V5: Supervisory islanding
    ap.add_argument("--enable_supervisory_islanding", action="store_true")
    ap.add_argument("--supervisory_island_start_s", type=int, default=None)
    ap.add_argument("--supervisory_island_duration_s", type=int, default=0)
    ap.add_argument("--no_supervisory_restore_load", action="store_true")
    ap.add_argument("--list_scenarios", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    
    if args.list_scenarios:
        print("\nScenarios:")
        for s in get_all_scenarios():
            print(f"  {s}")
        return
    
    scenarios = get_all_scenarios() if "all" in args.scenarios else args.scenarios
    out_dir = _mk_outdir(args.tag)
    
    print(f"\n{'='*60}")
    print(f"QuAM v2 (FIXED)")
    print(f"{'='*60}")
    print(f"Horizon: {args.horizon_s}s ({args.horizon_s/3600:.1f}h)")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")
    
    rows = []
    total = len(scenarios) * len(args.seeds) * len(args.topologies)
    current = 0
    
    for topology in args.topologies:
        for seed in args.seeds:
            for scenario in scenarios:
                current += 1
                print(f"[{current}/{total}] {scenario} | seed={seed}")
                
                row = run_one(
                    scenario=scenario, topology=topology, seed=seed,
                    horizon_s=args.horizon_s, out_dir=out_dir, nodes=args.nodes,
                    route_policy=args.route_policy,
                    k_paths=args.k_paths,
                    use_grid_link_types=args.use_grid_link_types,
                    attack_intensity=args.attack_intensity,
                    attack_duration=args.attack_duration,
                    distributed_attacks=args.distributed_attacks,
                    num_attack_windows=args.num_attack_windows,
                    exhaust_strategy=args.exhaust_strategy,
                    exhaust_focus=args.exhaust_focus,
                    qan_events=args.qan_events,
                    qan_cover_rate_per_s=args.qan_cover_rate_per_s,
                    qan_window_s=args.qan_window_s,
                    qan_mixing_delay_ms=args.qan_mixing_delay_ms,
                    qan_auth_cover=args.qan_auth_cover,
                    qan_auth_real=args.qan_auth_real,
                    qan_auth_sync=args.qan_auth_sync,
                    energy_record_interval=args.energy_interval,
                    enable_qkd=not args.no_qkd,
                    rotation_policy=args.rotation_policy,
                    verification_delay_ms=args.verification_delay_ms,
                    degraded_verification_delay_ms=args.degraded_verification_delay_ms,
                    enable_power_sharing=args.enable_power_sharing,
                    link_distance_km=args.link_distance_km,
                    fiber_loss_db_per_km=args.fiber_loss_db_km,
                    finite_key_preset=args.finite_key,
                    finite_key_block_bits=args.finite_key_block_bits,
                    finite_key_security_log=args.finite_key_security_log,
                    degraded_threshold_preset=args.degraded_threshold,
                    random_window_intensity=args.random_window_intensity,
                    window_intensity_pool=args.window_intensity_pool,
                    qrng_pool_bits=args.qrng_pool_bits,
                    qrng_rate_bits_per_s=args.qrng_rate_bits_per_s,
                    enable_quantum_protocols=args.enable_quantum_protocols,
                    pingpong_variant=args.pingpong_variant,
                    pingpong_interval_s=args.pingpong_interval_s,
                )
                rows.append(row)
                
                delivered = row.get("delivered_ratio", 0) * 100
                eens = row.get("eens_critical_kwh", 0)
                ctrl = row.get("control_msgs_received", 0)
                blocked_total = (
                    row.get("defense_blocked_degraded", 0)
                    + row.get("defense_blocked_intrusion", 0)
                    + row.get("defense_blocked_rate_limit", 0)
                    + row.get("defense_blocked_signature", 0)
                    + row.get("defense_blocked_per_source_rate", 0)
                    + row.get("defense_blocked_implausible", 0)
                    + row.get("defense_blocked_cross_node", 0)
                    + row.get("defense_blocked_quarantine_mgr", 0)
                    + row.get("prekey_blocked_total", 0)
                )
                blocked_rate = row.get("defense_blocked_rate_limit", 0)
                print(f"  → Delivered: {delivered:.1f}%, EENS: {eens:.2f}kWh, Control: {ctrl}, Blocked: {blocked_total} (rate_limit {blocked_rate})\n")
    
    summary_path = os.path.join(out_dir, "summary", "summary.csv")
    QuAMLogger.write_summary_csv(summary_path, rows)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
