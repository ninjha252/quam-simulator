"""
common.py - Shared infrastructure for QuAM runners

This module provides common setup functions used by all runner scripts.
Import this to avoid code duplication across run_spoof.py, run_exhaust.py, etc.
"""

from __future__ import annotations

import os
import random
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional, Callable
from collections import defaultdict
import hashlib

import simpy

# Import existing modules
from .model import Message, MsgType, MicrogridState, MicrogridParams, DeliveryStatus, ActionDecision
from .network import build_topology, build_links, NetworkSim, edge_key, summarize_link_congestion
from .network_metrics import NetworkActivityTracker
try:
    from .power_sharing import PowerSharingCoordinator
except Exception:
    PowerSharingCoordinator = None  # optional module

from .quantum import (
    QuantumAugmentation, KeyPolicy, QKDKeyPool, QuantumLinkHealth, QKDLinkParameters,
    QBERWindow, secret_fraction_bb84, fidelity_from_qber, apply_rotation_policy,
    get_finite_key_params
)
from .threat import (
    Observation,
    TrafficAnalyzer, QANConfig, QANOrchestrator,
    SpoofConfig, SpoofingAttack,
    ExhaustConfig, KeyExhaustionAttack,
    IntrusionDetector,
    AdmissionGate,
    CoverTrafficTracker,
    AttackerObservationConfig, ObservationScope,
    GateConfig, PolicyGate, get_defense_config,
    secret_fraction_to_qber_approx,
    make_emit_with_observer,
    parse_action_from_message,
    InsiderThreatConfig, InsiderThreatAttack,
)
from .quantum_protocols import (
    QuantumProtocolConfig, QuantumTLSConfig, PingPongVariant, QuantumProtocol,
)
from .metrics import QuAMLogger, EnergyLogger


# ============================================================================
# ATTACK CONFIGURATIONS
# ============================================================================

ATTACK_INTENSITIES = {
    "S1": {"capacity": 15000, "refill": 1500, "exhaust_rate": 2.0, "qber": 0.03},
    "S2": {"capacity": 10000, "refill": 1000, "exhaust_rate": 5.0, "qber": 0.05},
    "S3": {"capacity": 6000, "refill": 600, "exhaust_rate": 10.0, "qber": 0.08},
    "S4": {"capacity": 4000, "refill": 400, "exhaust_rate": 15.0, "qber": 0.10},
    "S5": {"capacity": 2500, "refill": 250, "exhaust_rate": 20.0, "qber": 0.14},
}

# Fixed QKD infrastructure — decoupled from attack strength.
# Baseline traffic costs ~672 bits/s (tag=384 + nonce=64, x1.5 verify, ~1 msg/s).
# refill=2000 gives comfortable headroom so baseline is never key-starved.
FIXED_INFRASTRUCTURE = {
    "capacity": 20000,
    "refill": 2000,
    "init_fill_ratio": 0.50,
}

# Attack-only parameters (infrastructure stays fixed via FIXED_INFRASTRUCTURE).
ATTACK_LEVELS = {
    "A1": {"exhaust_rate": 2.0, "qber": 0.02},
    "A2": {"exhaust_rate": 5.0, "qber": 0.04},
    "A3": {"exhaust_rate": 10.0, "qber": 0.08},
    "A4": {"exhaust_rate": 20.0, "qber": 0.12},
    "A5": {"exhaust_rate": 40.0, "qber": 0.18},
}

# Defense modes that should also turn on quantum-layer resource protection.
QUANTUM_DEFENSE_STRATEGIES = {
    "ratelimit_v2",
    "intrusion_v2",
    "plausibility",
    "correlation",
    "quarantine_v2",
    "hardened",
    "hardened_balanced",
    "hardened_strong",
    "hardened_v2",
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

    # Balanced defaults: still protective, but avoids near-binary behavior.
    reservation_ratio = 0.12
    source_rate = 4000.0
    source_rate_enabled = True
    emergency_threshold_ratio = 0.04
    emergency_tag_bits = 160

    if mode == "hardened_strong":
        reservation_ratio = 0.30
        source_rate = 800.0
        source_rate_enabled = True
        emergency_threshold_ratio = 0.10
        emergency_tag_bits = 64
    elif mode in {"hardened", "hardened_balanced"}:
        reservation_ratio = 0.12
        source_rate = 4000.0
        source_rate_enabled = True
        emergency_threshold_ratio = 0.04
        emergency_tag_bits = 160
    elif mode == "quantum_only":
        # In ablation, disable source key-rate limiting so quantum-only is not
        # unrealistically perfect against exhaustion.
        reservation_ratio = 0.12
        source_rate = 0.0
        source_rate_enabled = False
        emergency_threshold_ratio = 0.04
        emergency_tag_bits = 160
    elif mode in {"ratelimit_v2", "quarantine_v2"}:
        reservation_ratio = 0.25
        source_rate = 1200.0
        source_rate_enabled = True
        emergency_threshold_ratio = 0.08
        emergency_tag_bits = 96

    return {
        "enabled": True,
        "enable_priority_reservation": True,
        "reservation_ratio": reservation_ratio,
        "enable_source_key_rate_limit": source_rate_enabled,
        "source_key_rate_bits_per_s": source_rate,
        "enable_emergency_mode": True,
        "emergency_threshold_ratio": emergency_threshold_ratio,
        "emergency_tag_bits": emergency_tag_bits,
    }


# ============================================================================
# SIMULATION CONTEXT
# ============================================================================

@dataclass
class SimContext:
    """Container for all simulation components."""
    env: simpy.Environment
    rng: random.Random
    graph: Any
    edges: List[Tuple[str, str]]
    links: Dict
    qlayer: QuantumAugmentation
    microgrids: Dict[str, MicrogridState]
    net: NetworkSim
    admission_gate: AdmissionGate
    gate: PolicyGate
    intrusion_detector: IntrusionDetector
    net_tracker: NetworkActivityTracker
    phys_rng: random.Random
    
    # Loggers
    logger: QuAMLogger
    energy_logger: EnergyLogger
    analyzer: TrafficAnalyzer
    quantum_ts: "QuantumTimeSeriesLogger"
    cover_tracker: CoverTrafficTracker
    power_coordinator: Optional[PowerSharingCoordinator]
    finite_key_preset: str

    # V2: Quantum protocol config
    quantum_protocol_config: Optional[QuantumProtocolConfig] = None

    # State
    created_msgs: List[Message] = field(default_factory=list)
    attack_windows: List[Tuple[int, int]] = field(default_factory=list)

    # Callbacks
    msg_id_fn: Callable[[], int] = None
    emit_fn: Callable[[Message], None] = None


# ============================================================================
# QUANTUM TIME SERIES LOGGER
# ============================================================================

class QuantumTimeSeriesLogger:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def record(self, **kwargs: Any) -> None:
        self.records.append(dict(kwargs))

    def write_csv(self, path: str) -> None:
        if not self.records:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.records[0].keys()))
            writer.writeheader()
            writer.writerows(self.records)


def create_sim_context(
    *,
    seed: int,
    nodes: List[str],
    topology: str,
    defense_mode: str,
    attack_intensity: str = "S3",
    infrastructure_override: Optional[Dict] = None,
    route_policy: str = "shortest",
    k_paths: int = 3,
    use_grid_link_types: bool = False,
    enable_qkd: bool = True,
    rotation_policy: str = "none",
    enable_power_sharing: bool = False,
    link_distance_km: float = 10.0,
    fiber_loss_db_per_km: float = 0.2,
    finite_key_preset: str = "disabled",
    finite_key_block_bits: Optional[int] = None,
    finite_key_security_log: Optional[int] = None,
    degraded_threshold_preset: str = "moderate",
    attacker_scope: str = "global",
    attacker_tap_node: Optional[str] = None,
    attacker_tap_edge: Optional[Tuple[str, str]] = None,
    quantum_protocol_config: Optional[QuantumProtocolConfig] = None,
) -> SimContext:
    """Create a complete simulation context with all components initialized."""
    
    rng = random.Random(seed)
    # Separate RNG for physics so attacks/traffic do not perturb generation samples.
    phys_rng = random.Random(seed + 1000003)
    env = simpy.Environment()
    
    # Message ID generator
    _msg_id = [0]
    def msg_id_fn():
        _msg_id[0] += 1
        return _msg_id[0]
    
    # Network
    if str(topology).startswith("ieee"):
        # IEEE feeders define their own node IDs; override the provided node list.
        graph = build_topology(topology, [], rng)
        nodes = sorted(list(graph.nodes()))
    else:
        graph = build_topology(topology, nodes, rng)
    edges = list(graph.edges())
    links = build_links(env, graph, rng, use_grid_link_types=use_grid_link_types)

    # Deanonymization attacker observation model (tap locations depend on topology)
    try:
        obs_scope = ObservationScope(str(attacker_scope))
    except Exception:
        obs_scope = ObservationScope.GLOBAL
    obs_cfg = AttackerObservationConfig(
        scope=obs_scope,
        tap_node=attacker_tap_node,
        tap_edge=attacker_tap_edge,
    )
    # Auto-pick a deterministic tap location when using localized scopes.
    # Without a tap, the attacker observes nothing (all-abstain) which is not
    # a meaningful capability baseline for scope comparisons.
    if obs_cfg.scope == ObservationScope.NODE_TAP:
        if obs_cfg.tap_node is not None and obs_cfg.tap_node not in graph:
            obs_cfg.tap_node = None
        if not obs_cfg.tap_node:
            degrees = list(graph.degree())
            if degrees:
                max_deg = max(d for _, d in degrees)
                candidates = sorted([n for n, d in degrees if d == max_deg])
                obs_cfg.tap_node = candidates[0] if candidates else None
    elif obs_cfg.scope == ObservationScope.EDGE_TAP:
        if obs_cfg.tap_edge is not None:
            a, b = obs_cfg.tap_edge
            norm = tuple(sorted((str(a), str(b))))
            if not graph.has_edge(*norm):
                obs_cfg.tap_edge = None
            else:
                obs_cfg.tap_edge = norm
        if not obs_cfg.tap_edge:
            try:
                import networkx as nx
                bc = nx.edge_betweenness_centrality(graph)
                if bc:
                    # Pick the highest betweenness edge; stable tie-break.
                    best = sorted(
                        bc.items(),
                        key=lambda kv: (-float(kv[1]), tuple(sorted((str(kv[0][0]), str(kv[0][1]))))),
                    )[0][0]
                    obs_cfg.tap_edge = tuple(sorted((str(best[0]), str(best[1]))))
            except Exception:
                obs_cfg.tap_edge = None
    
    # Attack config
    atk_cfg = ATTACK_INTENSITIES.get(attack_intensity, ATTACK_INTENSITIES["S3"])
    
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
    if infrastructure_override:
        _pool_cap = infrastructure_override["capacity"]
        _pool_refill = infrastructure_override["refill"]
        _pool_init = infrastructure_override.get("init_fill_ratio", 0.50)
    else:
        _pool_cap = atk_cfg["capacity"] if enable_qkd else 1000000
        _pool_refill = atk_cfg["refill"] if enable_qkd else 1000000
        _pool_init = 0.50 if enable_qkd else 1.0

    default_pool = QKDKeyPool(
        capacity_bits=_pool_cap,
        base_refill_bits_per_s=_pool_refill,
        init_fill_ratio=_pool_init,
        link_params=link_params,
    )
    default_health = QuantumLinkHealth(baseline_qber=0.01)
    
    finite_key_params = get_finite_key_params(
        finite_key_preset,
        block_size_bits=finite_key_block_bits,
        security_parameter_log=finite_key_security_log,
    )
    quantum_def_cfg = get_quantum_defense_config(defense_mode, enable_qkd)

    per_edge_distance_km = {edge_key(u, v): link_distance_km for (u, v) in edges}

    qlayer = QuantumAugmentation(
        env=env, rng=rng, key_policy=key_policy,
        default_pool=default_pool, default_health=default_health,
        per_edge_distance_km=per_edge_distance_km,
        finite_key_params=finite_key_params,
        enable_priority_reservation=bool(quantum_def_cfg["enable_priority_reservation"]),
        reservation_ratio=float(quantum_def_cfg["reservation_ratio"]),
        enable_source_key_rate_limit=bool(quantum_def_cfg["enable_source_key_rate_limit"]),
        source_key_rate_bits_per_s=float(quantum_def_cfg["source_key_rate_bits_per_s"]),
        enable_emergency_mode=bool(quantum_def_cfg["enable_emergency_mode"]),
        emergency_threshold_ratio=float(quantum_def_cfg["emergency_threshold_ratio"]),
        emergency_tag_bits=int(quantum_def_cfg["emergency_tag_bits"]),
        quantum_protocol_config=quantum_protocol_config,
    )

    # Intrusion detector
    intrusion_detector = IntrusionDetector(qber_threshold=0.025)
    
    # Microgrids
    mg_params = MicrogridParams(
        name="default",
        base_load_kw=120.0,
        ai_load_kw=60.0,
        critical_load_kw=110.0,
        gen_kw_mean=130.0,
        gen_kw_sigma=15.0,
        import_cap_kw=60.0,
    )
    microgrids = {n: MicrogridState(params=mg_params) for n in nodes}

    cover_tracker = CoverTrafficTracker(energy_per_byte_j=mg_params.energy_per_byte_j)
    power_coordinator = None

    # Network activity tracker (for comms-energy coupling)
    net_tracker = NetworkActivityTracker(window_s=mg_params.control_window_s)
    
    # Defense gate
    gate_cfg = get_defense_config(defense_mode, degraded_threshold_preset)
    controller_nodes = tuple([sorted(nodes)[0]]) if nodes else tuple()
    if getattr(gate_cfg, "enable_control_acl", False) and not tuple(getattr(gate_cfg, "allowed_control_sources", ())):
        gate_cfg.allowed_control_sources = controller_nodes
    use_intrusion = bool(getattr(gate_cfg, "block_during_intrusion", False))
    admission_gate = AdmissionGate(
        gate_cfg,
        intrusion_detector if use_intrusion else None,
    )
    gate = PolicyGate(
        gate_cfg,
        intrusion_detector if use_intrusion else None,
        microgrids=microgrids,
    )
    # Pre-key admission runs inside quantum pre_send_hook so blocks happen before
    # key consumption on authenticated traffic.
    qlayer.preauth_decider = admission_gate.decide
    
    # Custom on_deliver_hook
    def on_deliver_hook(env_inner: simpy.Environment, msg: Message) -> None:
        action = parse_action_from_message(msg)
        if action is None:
            return
        
        mg = microgrids.get(msg.dst)
        if mg is None:
            return
        
        decision, delay_ms = gate.decide(env=env_inner, msg=msg, action=action, 
                                         staleness_ms=mg.params.staleness_ms)
        
        def _apply():
            if delay_ms > 0:
                yield env_inner.timeout(delay_ms / 1000.0)
            mg.apply_action(now_s=int(env_inner.now), msg=msg, action=action, decision=decision)
        
        env_inner.process(_apply())
    
    # Network
    analyzer = TrafficAnalyzer(sender_candidates=nodes, observation_cfg=obs_cfg)

    def on_hop_observe(env_inner: simpy.Environment, msg: Message, u: str, v: str) -> None:
        if bool(getattr(msg, "requires_anon", False)):
            obs_type = "anon_meta"
            obs_size = 200
            obs_channel = "anon"
        else:
            obs_type = msg.msg_type.value if hasattr(msg.msg_type, "value") else str(msg.msg_type)
            obs_size = int(msg.size_bytes)
            obs_channel = obs_type
        lu, lv = edge_key(u, v)
        analyzer.observe(Observation(
            t_ms=int(env_inner.now * 1000),
            src=msg.src,
            dst=msg.dst,
            msg_type=obs_type,
            size_bytes=obs_size,
            requires_anon=bool(getattr(msg, "requires_anon", False)),
            obs_channel=f"{lu}-{lv}" if obs_channel == "anon" else obs_channel,
            link_u=lu,
            link_v=lv,
        ))

    net = NetworkSim(
        env=env, g=graph, links=links, rng=rng,
        route_policy=route_policy,
        k_paths=k_paths,
        pre_send_hook=qlayer.pre_send_hook,
        on_deliver_hook=on_deliver_hook,
        on_message_final=net_tracker.observe_message,
        on_hop_observe=on_hop_observe,
        drop_if_miss_deadline=True,
    )
    
    # Loggers
    logger = QuAMLogger()
    energy_logger = EnergyLogger()
    quantum_ts = QuantumTimeSeriesLogger()
    
    created_msgs: List[Message] = []
    
    def base_emit(msg: Message):
        created_msgs.append(msg)
        net.send(msg)
    
    # Use hop-level observations for deanonymization; creation-time tap disabled
    # to avoid duplicating observations and to keep local-tap semantics strict.
    emit_fn = make_emit_with_observer(
        base_emit=base_emit,
        analyzer=analyzer,
        observe_created_events=False,
    )
    
    return SimContext(
        env=env,
        rng=rng,
        graph=graph,
        edges=edges,
        links=links,
        qlayer=qlayer,
        microgrids=microgrids,
        net=net,
        admission_gate=admission_gate,
        gate=gate,
        intrusion_detector=intrusion_detector,
        net_tracker=net_tracker,
        phys_rng=phys_rng,
        logger=logger,
        energy_logger=energy_logger,
        analyzer=analyzer,
        quantum_ts=quantum_ts,
        cover_tracker=cover_tracker,
        power_coordinator=power_coordinator,
        finite_key_preset=str(finite_key_preset),
        quantum_protocol_config=quantum_protocol_config,
        created_msgs=created_msgs,
        msg_id_fn=msg_id_fn,
        emit_fn=emit_fn,
    )


# ============================================================================
# BACKGROUND PROCESSES
# ============================================================================

def schedule_background_workload(ctx: SimContext, horizon_s: int, nodes: List[str]):
    """Schedule background control/telemetry traffic."""
    # Guard against mismatched node lists (e.g., IEEE topologies define their own nodes).
    nodes = [n for n in nodes if n in ctx.graph] or list(ctx.microgrids.keys())
    controller_nodes = [sorted(nodes)[0]] if nodes else []

    def _process():
        while ctx.env.now < horizon_s:
            rnd = ctx.rng.random()
            if rnd < 0.60:
                # 60% — Routine control: designated controller emits setpoints.
                src = ctx.rng.choice(controller_nodes or nodes)
                dst = ctx.rng.choice([n for n in nodes if n != src] or nodes)
                msg = Message(
                    msg_id=ctx.msg_id_fn(), created_ms=int(ctx.env.now * 1000),
                    src=src, dst=dst, msg_type=MsgType.CONTROL_SETPOINT,
                    priority=1, deadline_ms=500, size_bytes=260,
                    # Baseline control should not shed load unless needed.
                    # Use a neutral setpoint to avoid artificial curtailment in baseline runs.
                    requires_auth=True,
                    payload={
                        "shed_frac_target": 0.0,
                        "control_signature": "quam_ctrl_v1",
                        "control_sender_role": "controller",
                    },
                )
            elif rnd < 0.75:
                # 15% — Critical priority actions (emergency shed, islanding, etc.)
                # These use Quantum-TLS (KAK) when quantum protocols are enabled,
                # transmitting WITHOUT consuming key bits from the QKD pool.
                src = ctx.rng.choice(controller_nodes or nodes)
                dst = ctx.rng.choice([n for n in nodes if n != src] or nodes)
                msg = Message(
                    msg_id=ctx.msg_id_fn(), created_ms=int(ctx.env.now * 1000),
                    src=src, dst=dst, msg_type=MsgType.PRIORITY_ACTION,
                    priority=2, deadline_ms=300, size_bytes=280,
                    requires_auth=True,
                    payload={
                        "action": "emergency_shed",
                        "shed_frac_target": 0.0,
                        "control_signature": "quam_ctrl_v1",
                        "control_sender_role": "controller",
                    },
                )
            else:
                # 25% — Telemetry readings
                src = ctx.rng.choice(nodes)
                dst = ctx.rng.choice([n for n in nodes if n != src] or nodes)
                msg = Message(
                    msg_id=ctx.msg_id_fn(), created_ms=int(ctx.env.now * 1000),
                    src=src, dst=dst, msg_type=MsgType.TELEMETRY,
                    priority=0, deadline_ms=800, size_bytes=220,
                    requires_auth=False,
                )
            ctx.emit_fn(msg)
            yield ctx.env.timeout(ctx.rng.uniform(0.5, 1.5))
    
    ctx.env.process(_process())


def schedule_microgrid_stepper(
    ctx: SimContext,
    horizon_s: int,
    energy_interval: int = 10,
    quantum_interval_s: int = 30,
):
    """Schedule microgrid physics updates and energy logging."""
    q_interval = max(1, int(quantum_interval_s))

    def _process():
        while ctx.env.now < horizon_s:
            t_s = int(ctx.env.now)
            is_attack = any(s <= t_s <= e for s, e in ctx.attack_windows)
            attack_label = "active" if is_attack else "none"
            
            for name, mg in ctx.microgrids.items():
                # Update comms/control coupling from recent network activity
                stats = ctx.net_tracker.get_stats(
                    node=name,
                    now_s=float(ctx.env.now),
                    window_s=mg.params.control_window_s,
                    dt_s=1.0,
                    comm_base_kw=mg.params.comm_base_kw,
                    energy_per_byte_j=mg.params.energy_per_byte_j,
                    energy_per_key_bit_j=mg.params.energy_per_key_bit_j,
                    control_drop_penalty=mg.params.control_drop_penalty,
                    control_on_time_deadline_ms=mg.params.control_on_time_deadline_ms,
                )
                mg.update_comm_state(stats=stats, dt_s=1)

                gen = max(0, ctx.phys_rng.gauss(mg.params.gen_kw_mean, mg.params.gen_kw_sigma))
                mg.step(t_s=t_s, dt_s=1, gen_kw_sample=gen)
                
                # Energy logging
                if t_s % energy_interval == 0:
                    ctx.energy_logger.record(t_s, name, mg, is_attack, attack_label)
            
            # Quantum time series + intrusion detection
            if t_s % q_interval == 0:
                for ek in ctx.qlayer.pools:
                    qber = ctx.qlayer.health[ek].qber_at(t_s)
                    ctx.intrusion_detector.record_qber(t_s, ek, qber)

                    # V2: Ping-Pong IDS probes
                    pingpong_alert = False
                    if ctx.qlayer.pingpong_has_recent_alert(t_s):
                        pingpong_alert = True

                    ctx.quantum_ts.record(
                        t_s=t_s,
                        edge=f"{ek[0]}-{ek[1]}",
                        qber=qber,
                        secret_fraction=secret_fraction_bb84(qber, ctx.qlayer.finite_key_params),
                        fidelity=fidelity_from_qber(qber),
                        pool_level=ctx.qlayer.pools[ek].level_bits,
                        is_attack=is_attack,
                        intrusion_alert=ctx.intrusion_detector.has_recent_alert(t_s),
                        abort_active=int(ctx.qlayer.health[ek].abort_active()),
                        pingpong_alert=int(pingpong_alert),
                    )

                # Run Ping-Pong IDS probes periodically
                ctx.qlayer.run_pingpong_probes(t_s)
            
            yield ctx.env.timeout(1)
    
    ctx.env.process(_process())


# ============================================================================
# OUTPUT HELPERS
# ============================================================================



def _split_window(rng, start_s: int, end_s: int, n_segments: int):
    if n_segments <= 1 or end_s - start_s <= 1:
        return [(start_s, end_s)]
    n_segments = max(1, n_segments)
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


def make_output_dir(tag: str) -> str:
    """Create output directory with trial numbering."""
    base_dir = os.path.join("outputs", tag)
    trial_num = 1
    if os.path.exists(base_dir):
        existing = [d for d in os.listdir(base_dir) if d.startswith("trial_")]
        if existing:
            nums = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
            trial_num = max(nums) + 1 if nums else 1
    
    trial_dir = os.path.join(base_dir, f"trial_{trial_num}")
    for folder in ["messages", "deanon", "summary", "timeseries", "energy", "cover"]:
        os.makedirs(os.path.join(trial_dir, folder), exist_ok=True)
    
    return trial_dir


def finalize_and_save(
    ctx: SimContext,
    scenario: str,
    topology: str,
    seed: int,
    horizon_s: int,
    out_dir: str,
    defense_mode: str,
    attack_intensity: str,
    attacks: List[str],
    enable_qkd: bool = True,
    run_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record all messages and save outputs."""
    
    # Record messages
    for m in ctx.created_msgs:
        ctx.logger.record_message(m)
    
    # Record action events
    msg_attack_lookup = {m.msg_id: m.is_attack for m in ctx.created_msgs}
    for name, mg in ctx.microgrids.items():
        for entry in mg.action_log:
            is_attack = 1 if msg_attack_lookup.get(entry.msg_id, False) else 0
            ctx.logger.record_action_event(
                dst=name, action=entry.action_type,
                decision=entry.decision, is_attack=is_attack,
            )
    
    # Write outputs
    ctx.logger.write_message_csv(os.path.join(out_dir, "messages", 
                                 f"messages_{scenario}_{topology}_seed{seed}.csv"))
    ctx.logger.write_deanon_csv(os.path.join(out_dir, "deanon", 
                                f"deanon_{scenario}_{topology}_seed{seed}.csv"))
    ctx.energy_logger.write_csv(os.path.join(out_dir, "energy",
                                f"energy_{scenario}_{topology}_seed{seed}.csv"))
    ctx.quantum_ts.write_csv(os.path.join(out_dir, "timeseries",
                                f"quantum_{scenario}_{topology}_seed{seed}.csv"))
    ctx.cover_tracker.write_csv(os.path.join(out_dir, "cover",
                                f"cover_{scenario}_{topology}_seed{seed}.csv"))
    
    # Compute summary
    energy_stats = ctx.energy_logger.get_summary_stats()
    # Distance statistics (from QKD pools)
    distances = [p.link_params.distance_km for p in ctx.qlayer.pools.values()]
    factors = [p.link_params.distance_factor() for p in ctx.qlayer.pools.values()]
    avg_link_distance_km = sum(distances) / len(distances) if distances else 0.0
    avg_distance_factor = sum(factors) / len(factors) if factors else 1.0
    effective_key_rate_reduction_pct = max(0.0, (1.0 - avg_distance_factor) * 100.0)
    finite_key_enabled = int(bool(ctx.qlayer.finite_key_params is not None and ctx.qlayer.finite_key_params.enabled))
    finite_key_factor = float(ctx.qlayer.finite_key_params.finite_key_factor()) if ctx.qlayer.finite_key_params else 1.0
    finite_key_block_size = int(ctx.qlayer.finite_key_params.block_size_bits) if ctx.qlayer.finite_key_params else 0
    finite_key_security_log = int(ctx.qlayer.finite_key_params.security_parameter_log) if ctx.qlayer.finite_key_params else 0

    
    cover_stats = ctx.cover_tracker.get_summary()
    congestion_stats = summarize_link_congestion(ctx.links)

    # ------------------------------------------------------------------
    # Sanity checks / invariants
    # ------------------------------------------------------------------
    # Key pool conservation (per edge):
    #   initial + added_effective - consumed == final
    # `add_bits()` caps at capacity, so we separately track spilled bits for transparency.
    pool_abs_errs = []
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
    for pool in ctx.qlayer.pools.values():
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
    ecmp_stats = getattr(ctx.net, "_ecmp_stats", None)
    ecmp_total = float(ecmp_stats.get("total", 0.0)) if isinstance(ecmp_stats, dict) else 0.0
    ecmp_had_alt = float(ecmp_stats.get("had_alternatives", 0.0)) if isinstance(ecmp_stats, dict) else 0.0
    ecmp_alt_paths_mean = (float(ecmp_stats.get("alt_paths_sum", 0.0)) / ecmp_total) if (isinstance(ecmp_stats, dict) and ecmp_total > 0) else 0.0
    ecmp_alt_paths_max = float(ecmp_stats.get("alt_paths_max", 0.0)) if isinstance(ecmp_stats, dict) else 0.0
    ecmp_had_alt_ratio = (ecmp_had_alt / ecmp_total) if ecmp_total > 0 else 0.0

    # Attack traffic distribution (by attempted hop / edge)
    attack_hits = getattr(ctx.net, "_edge_hits_attack", None)
    total_hits = getattr(ctx.net, "_edge_hits_total", None)
    def _fmt_hits(counter_obj, n: int = 10) -> str:
        try:
            items = list(counter_obj.most_common(n))
        except Exception:
            return ""
        return "|".join([f"{u}-{v}:{int(c)}" for (u, v), c in items])

    net_edge_hits_attack_top10 = _fmt_hits(attack_hits, 10) if attack_hits else ""
    net_edge_hits_total_top10 = _fmt_hits(total_hits, 10) if total_hits else ""

    prekey_blocked_msgs = [
        m for m in ctx.created_msgs
        if int((m.payload or {}).get("prekey_blocked", 0)) == 1
    ]
    key_bits_saved_prekey_est_sum = float(sum(
        float((m.payload or {}).get("key_bits_saved_prekey_est", 0.0))
        for m in prekey_blocked_msgs
    ))
    emergency_mode_msg_count = int(sum(
        1 for m in ctx.created_msgs if int((m.payload or {}).get("emergency_mode", 0)) == 1
    ))
    reduced_tag_msg_count = int(sum(
        1 for m in ctx.created_msgs if int((m.payload or {}).get("reduced_auth_tag", 0)) == 1
    ))

    extra = {
        "attacks": ",".join(attacks) if attacks else "none",
        "defense_mode": defense_mode,
        "attack_intensity": attack_intensity,
        "enable_qkd": enable_qkd,
        "eens_total_kwh": sum(mg.eens_total_kwh for mg in ctx.microgrids.values()),
        "eens_critical_kwh": sum(mg.eens_critical_kwh for mg in ctx.microgrids.values()),
        "critical_outage_minutes": sum(mg.critical_outage_minutes for mg in ctx.microgrids.values()),
        "n_attack_windows": len(ctx.attack_windows),
        "total_attack_duration_s": sum(e - s for s, e in ctx.attack_windows),
        "n_intrusion_alerts": len(ctx.intrusion_detector.alerts),
        "defense_blocked_degraded": ctx.gate.stats["blocked_degraded"],
        "defense_blocked_rate_limit": ctx.gate.stats["blocked_rate_limit"],
        "defense_blocked_intrusion": ctx.gate.stats["blocked_intrusion"],
        "defense_blocked_signature": ctx.gate.stats["blocked_signature"],
        "defense_blocked_per_source_rate": ctx.gate.stats.get("blocked_per_source_rate", 0),
        "defense_blocked_implausible": ctx.gate.stats.get("blocked_implausible", 0),
        "defense_blocked_cross_node": ctx.gate.stats.get("blocked_cross_node", 0),
        "defense_blocked_quarantine_mgr": ctx.gate.stats.get("blocked_quarantine_mgr", 0),
        "defense_blocked_control_acl": ctx.gate.stats.get("blocked_control_acl", 0),
        "defense_blocked_control_signature": ctx.gate.stats.get("blocked_control_signature", 0),
        "defense_blocked_source_global_rate": ctx.gate.stats.get("blocked_source_global_rate", 0),
        "defense_delayed_degraded": ctx.gate.stats["delayed_degraded"],
        "degraded_threshold_sf": ctx.gate.cfg.degraded_secret_fraction,
        "degraded_threshold_qber_approx": secret_fraction_to_qber_approx(ctx.gate.cfg.degraded_secret_fraction),
        "degraded_mode_triggers": ctx.gate.stats.get("degraded_mode_triggers", 0),
        "prekey_checked_total": ctx.admission_gate.stats.get("checked_total", 0),
        "prekey_allowed_total": ctx.admission_gate.stats.get("allowed_total", 0),
        "prekey_blocked_total": ctx.admission_gate.stats.get("blocked_total", 0),
        "prekey_blocked_rate_limit": ctx.admission_gate.stats.get("blocked_prekey_rate_limit", 0),
        "prekey_blocked_per_source_rate": ctx.admission_gate.stats.get("blocked_prekey_per_source_rate", 0),
        "prekey_blocked_intrusion": ctx.admission_gate.stats.get("blocked_prekey_intrusion", 0),
        "prekey_blocked_cross_node": ctx.admission_gate.stats.get("blocked_prekey_cross_node", 0),
        "prekey_blocked_quarantine_mgr": ctx.admission_gate.stats.get("blocked_prekey_quarantine_mgr", 0),
        "prekey_blocked_degraded": ctx.admission_gate.stats.get("blocked_prekey_degraded", 0),
        "prekey_blocked_acl": ctx.admission_gate.stats.get("blocked_prekey_acl", 0),
        "prekey_blocked_signature": ctx.admission_gate.stats.get("blocked_prekey_signature", 0),
        "prekey_blocked_source_global_rate": ctx.admission_gate.stats.get("blocked_prekey_source_global_rate", 0),
        "prekey_key_bits_saved_est_sum": key_bits_saved_prekey_est_sum,
        "quantum_defense_enabled": int(bool(getattr(ctx.qlayer, "_enable_priority_reservation", False) or getattr(ctx.qlayer, "_enable_source_key_rate_limit", False))),
        "quantum_priority_reservation_enabled": int(bool(getattr(ctx.qlayer, "_enable_priority_reservation", False))),
        "quantum_reservation_ratio": float(getattr(ctx.qlayer, "_reservation_ratio", 0.0)),
        "quantum_source_key_rate_limit_enabled": int(bool(getattr(ctx.qlayer, "_enable_source_key_rate_limit", False))),
        "quantum_source_key_rate_bits_per_s": float(getattr(ctx.qlayer, "_source_key_rate_bits_per_s", 0.0)),
        "quantum_emergency_mode_enabled": int(bool(getattr(ctx.qlayer, "_enable_emergency_mode", False))),
        "quantum_emergency_threshold_ratio": float(getattr(ctx.qlayer, "_emergency_threshold_ratio", 0.0)),
        "quantum_emergency_tag_bits": int(getattr(ctx.qlayer, "_emergency_tag_bits", 0)),
        "quantum_reserved_saves_sum": int(key_reserved_saves_sum),
        "quantum_source_rate_blocks_sum": int(key_source_rate_blocks_sum),
        "quantum_emergency_grants_sum": int(key_emergency_grants_sum),
        "quantum_emergency_mode_msg_count": int(emergency_mode_msg_count),
        "quantum_reduced_tag_msg_count": int(reduced_tag_msg_count),
        "avg_link_distance_km": avg_link_distance_km,
        "avg_distance_factor": avg_distance_factor,
        "effective_key_rate_reduction_pct": effective_key_rate_reduction_pct,
        "finite_key_preset": str(ctx.finite_key_preset),
        "finite_key_enabled": finite_key_enabled,
        "finite_key_factor": finite_key_factor,
        "finite_key_block_size": finite_key_block_size,
        "finite_key_security_log": finite_key_security_log,
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
        **energy_stats,
        **cover_stats,
        **congestion_stats,
    }

    if run_meta:
        extra.update(run_meta)

    if ctx.qlayer.nonce_mgr is not None:
        extra.update(ctx.qlayer.nonce_mgr.qrng_stats())

    # V2: Multi-protocol quantum layer stats
    proto_stats = ctx.qlayer.get_protocol_stats()
    for k, v in proto_stats.items():
        extra[f"qproto_{k}"] = v

    return ctx.logger.summarize_run(
        scenario=scenario, topology=topology, seed=seed,
        horizon_ms=horizon_s * 1000, extra=extra,
    )
