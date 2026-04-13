#!/usr/bin/env python3
"""
run_full_matrix.py - Run all attack + defense combinations

This is the comprehensive runner that tests all combinations for the conference paper.
Outputs complete energy time series for plotting.

Usage:
  # Quick test (1 hour)
  python3 -m quam.runners.run_full_matrix --tag quick_test --horizon_s 3600 --seeds 0

  # Full 24-hour experiment
  python3 -m quam.runners.run_full_matrix --tag full_24h --horizon_s 86400 --seeds 0 1 2 3 4
  
  # Specific attacks only
  python3 -m quam.runners.run_full_matrix --attacks spoof quantum --defenses none block all
"""

import argparse
import os
import sys
from typing import List, Dict, Any, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Support running as script (quam.*) or as package module (relative imports)
try:
    from quam.common import (
        create_sim_context, schedule_background_workload, schedule_microgrid_stepper,
        make_output_dir, finalize_and_save, ATTACK_INTENSITIES,
        FIXED_INFRASTRUCTURE, ATTACK_LEVELS,
    )
    from quam.model import Message, MsgType
    from quam.quantum import QBERWindow, edge_key
    from quam.threat import (
        SpoofConfig, SpoofingAttack,
        ExhaustConfig, KeyExhaustionAttack, TargetedExhaustConfig, TargetedKeyExhaustionAttack, ExhaustTargetStrategy,
        QANConfig, QANOrchestrator,
    )
    from quam.metrics import QuAMLogger
except Exception:  # fallback if imported as part of a package
    from .common import (
        create_sim_context, schedule_background_workload, schedule_microgrid_stepper,
        make_output_dir, finalize_and_save, ATTACK_INTENSITIES,
        FIXED_INFRASTRUCTURE, ATTACK_LEVELS,
    )
    from .model import Message, MsgType
    from .quantum import QBERWindow, edge_key
    from .threat import (
        SpoofConfig, SpoofingAttack,
        ExhaustConfig, KeyExhaustionAttack, TargetedExhaustConfig, TargetedKeyExhaustionAttack, ExhaustTargetStrategy,
        QANConfig, QANOrchestrator,
    )
    from .metrics import QuAMLogger


DEFAULT_NODES = ["MG0", "MG1", "MG2"]

ALL_ATTACKS = ["spoof", "exhaust", "quantum"]
ALL_DEFENSES = [
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
    "hardened_balanced",
    "hardened_strong",
    "gate_only",
    "quantum_only",
]


QAN_DEFENSE_PROFILES: Dict[str, Dict[str, Any]] = {
    # Low cover for "none" intentionally keeps anonymity weaker.
    "none": {
        "profile_name": "minimal",
        "cover_rate_per_s": 1.0,
        "window_s": 3,
        "mixing_delay_ms": 20,
        "sync_burst_per_candidate": False,
    },
    "ratelimit": {
        "profile_name": "standard",
        "cover_rate_per_s": 2.5,
        "window_s": 4,
        "mixing_delay_ms": 30,
        "sync_burst_per_candidate": False,
    },
    "block": {
        "profile_name": "hardened",
        "cover_rate_per_s": 4.0,
        "window_s": 5,
        "mixing_delay_ms": 40,
        "sync_burst_per_candidate": True,
    },
    "delay": {
        "profile_name": "hardened",
        "cover_rate_per_s": 4.5,
        "window_s": 5,
        "mixing_delay_ms": 55,
        "sync_burst_per_candidate": True,
    },
    "intrusion": {
        "profile_name": "hardened_plus",
        "cover_rate_per_s": 5.0,
        "window_s": 6,
        "mixing_delay_ms": 55,
        "sync_burst_per_candidate": True,
    },
    "adaptive": {
        "profile_name": "hardened_plus",
        "cover_rate_per_s": 5.0,
        "window_s": 6,
        "mixing_delay_ms": 60,
        "sync_burst_per_candidate": True,
    },
    "signature": {
        "profile_name": "hardened_plus",
        "cover_rate_per_s": 5.0,
        "window_s": 6,
        "mixing_delay_ms": 60,
        "sync_burst_per_candidate": True,
    },
    "all": {
        "profile_name": "max_cover",
        "cover_rate_per_s": 8.0,
        "window_s": 8,
        "mixing_delay_ms": 80,
        "sync_burst_per_candidate": True,
    },
    "ratelimit_v2": {
        "profile_name": "hardened_plus",
        "cover_rate_per_s": 5.0,
        "window_s": 6,
        "mixing_delay_ms": 60,
        "sync_burst_per_candidate": True,
    },
    "intrusion_v2": {
        "profile_name": "hardened_plus",
        "cover_rate_per_s": 5.0,
        "window_s": 6,
        "mixing_delay_ms": 60,
        "sync_burst_per_candidate": True,
    },
    "plausibility": {
        "profile_name": "hardened_plus",
        "cover_rate_per_s": 5.0,
        "window_s": 6,
        "mixing_delay_ms": 60,
        "sync_burst_per_candidate": True,
    },
    "correlation": {
        "profile_name": "hardened_plus",
        "cover_rate_per_s": 5.0,
        "window_s": 6,
        "mixing_delay_ms": 60,
        "sync_burst_per_candidate": True,
    },
    "quarantine_v2": {
        "profile_name": "hardened_plus",
        "cover_rate_per_s": 5.0,
        "window_s": 6,
        "mixing_delay_ms": 60,
        "sync_burst_per_candidate": True,
    },
    "hardened": {
        "profile_name": "max_cover",
        "cover_rate_per_s": 8.0,
        "window_s": 8,
        "mixing_delay_ms": 80,
        "sync_burst_per_candidate": True,
    },
    "hardened_balanced": {
        "profile_name": "max_cover",
        "cover_rate_per_s": 8.0,
        "window_s": 8,
        "mixing_delay_ms": 80,
        "sync_burst_per_candidate": True,
    },
    "hardened_strong": {
        "profile_name": "max_cover",
        "cover_rate_per_s": 8.0,
        "window_s": 8,
        "mixing_delay_ms": 80,
        "sync_burst_per_candidate": True,
    },
    # Ablation-specific modes
    "gate_only": {
        "profile_name": "max_cover",
        "cover_rate_per_s": 8.0,
        "window_s": 8,
        "mixing_delay_ms": 80,
        "sync_burst_per_candidate": True,
    },
    "quantum_only": {
        "profile_name": "max_cover",
        "cover_rate_per_s": 8.0,
        "window_s": 8,
        "mixing_delay_ms": 80,
        "sync_burst_per_candidate": True,
    },
}


def build_qan_config_for_defense(defense_mode: str) -> Tuple[QANConfig, str]:
    cfg = QAN_DEFENSE_PROFILES.get(defense_mode, QAN_DEFENSE_PROFILES["none"])
    return (
        QANConfig(
            cover_rate_per_s=float(cfg["cover_rate_per_s"]),
            window_s=int(cfg["window_s"]),
            mixing_delay_ms=int(cfg["mixing_delay_ms"]),
            sync_burst_per_candidate=bool(cfg["sync_burst_per_candidate"]),
        ),
        str(cfg["profile_name"]),
    )


def generate_attack_windows(
    rng, horizon_s: int, total_duration: int, num_windows: int,
    min_gap_s: int = 120, edge_buffer_s: int = 300
) -> List[Tuple[int, int]]:
    """Generate random, non-overlapping attack windows across the horizon."""
    if num_windows <= 0 or total_duration <= 0:
        return []
    
    usable = max(0, horizon_s - 2 * edge_buffer_s)
    if usable <= 0:
        return []
    
    max_gap = max(0, (usable - total_duration) // max(1, num_windows - 1))
    min_gap_s = min(min_gap_s, max_gap)
    
    avg_dur = max(60, total_duration // num_windows)
    min_dur = max(60, avg_dur // 2)
    max_dur = max(min_dur, avg_dur * 2)
    
    durations = []
    remaining = total_duration
    for i in range(num_windows):
        if i == num_windows - 1:
            dur = max(min_dur, remaining)
        else:
            max_for_i = remaining - min_dur * (num_windows - i - 1)
            if max_for_i < min_dur:
                dur = max(60, remaining // (num_windows - i))
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


def run_scenario(
    *,
    attacks: List[str],
    defense_mode: str,
    seed: int,
    nodes: List[str],
    topology: str,
    horizon_s: int,
    out_dir: str,
    attack_intensity: str,
    distributed_attacks: bool,
    num_attack_windows: int,
    exhaust_strategy: str,
    exhaust_focus: float,
    energy_interval: int,
    quantum_interval_s: int,
    attack_ramp_s: int,
    qan_events: int,
    infrastructure_override: Optional[Dict] = None,
    route_policy: str = "shortest",
    k_paths: int = 3,
    use_grid_link_types: bool = False,
    qan_auth_cover: bool = False,
    qan_auth_real: bool = False,
    qan_auth_sync: bool = False,
    deanon_guided_pulse_s: int = 90,
    deanon_guided_min_top1_prob: float = 0.0,
    deanon_guided_require_non_abstain: bool = False,
    link_distance_km: float = 10.0,
    fiber_loss_db_per_km: float = 0.2,
    finite_key_preset: str = "disabled",
    finite_key_block_bits: Optional[int] = None,
    finite_key_security_log: Optional[int] = None,
    degraded_threshold_preset: str = "moderate",
    attacker_scope: str = "global",
    attacker_tap_node: Optional[str] = None,
    attacker_tap_edge: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """Run a single attack+defense scenario."""
    
    # Build scenario name
    if not attacks:
        scenario = "baseline"
    else:
        attack_str = "_".join(sorted(attacks)) if len(attacks) < 3 else "all_attacks"
        scenario = f"{attack_str}_def_{defense_mode}"
    
    # Create context
    ctx = create_sim_context(
        seed=seed,
        nodes=nodes,
        topology=topology,
        defense_mode=defense_mode,
        attack_intensity=attack_intensity,
        infrastructure_override=infrastructure_override,
        route_policy=route_policy,
        k_paths=k_paths,
        use_grid_link_types=use_grid_link_types,
        link_distance_km=link_distance_km,
        fiber_loss_db_per_km=fiber_loss_db_per_km,
        finite_key_preset=finite_key_preset,
        finite_key_block_bits=finite_key_block_bits,
        finite_key_security_log=finite_key_security_log,
        degraded_threshold_preset=degraded_threshold_preset,
        attacker_scope=attacker_scope,
        attacker_tap_node=attacker_tap_node,
        attacker_tap_edge=attacker_tap_edge,
    )

    # If topology overrides node IDs (e.g., IEEE feeders), ensure all downstream logic
    # uses the effective node list.
    nodes = sorted(list(ctx.microgrids.keys()))
    
    # Attack windows
    if infrastructure_override:
        _atk_level = ATTACK_LEVELS.get(attack_intensity, ATTACK_LEVELS["A3"])
        atk_cfg = {**infrastructure_override, **_atk_level}
    else:
        atk_cfg = ATTACK_INTENSITIES.get(attack_intensity, ATTACK_INTENSITIES["S3"])
    
    if attacks:
        total_attack_s = int(horizon_s / 3)
        if distributed_attacks:
            ctx.attack_windows = generate_attack_windows(
                ctx.rng, horizon_s, total_attack_s, num_attack_windows)
        else:
            start = max(300, horizon_s // 3)
            ctx.attack_windows = [(start, min(start + total_attack_s, horizon_s - 300))]
    
    # Background workload
    schedule_background_workload(ctx, horizon_s, nodes)
    
    # Microgrid stepper
    schedule_microgrid_stepper(
        ctx,
        horizon_s,
        energy_interval=energy_interval,
        quantum_interval_s=quantum_interval_s,
    )
    
    # QAN events (defense-aware anonymity profile)
    qan_cfg, qan_profile = build_qan_config_for_defense(defense_mode)
    qan_cfg.auth_cover = bool(qan_auth_cover)
    qan_cfg.auth_real_notify = bool(qan_auth_real)
    qan_cfg.auth_sync_burst = bool(qan_auth_sync)
    qan = QANOrchestrator(env=ctx.env, rng=ctx.rng, cfg=qan_cfg,
                          msg_id_fn=ctx.msg_id_fn, emit_fn=ctx.emit_fn,
                          cover_tracker=ctx.cover_tracker)
    
    qan_specs = []
    for i in range(qan_events):
        t_ev = ctx.rng.randint(int(horizon_s * 0.2), int(horizon_s * 0.8))
        true_sender = ctx.rng.choice(nodes)
        receiver = ctx.rng.choice([n for n in nodes if n != true_sender] or nodes)
        spec = qan.schedule_event(true_sender=true_sender, candidates=nodes,
                                  receiver=receiver, t_event_s=t_ev)
        qan_specs.append({"spec": spec, "true_sender": true_sender,
                         "receiver": receiver, "t_event_s": t_ev})

    deanon_guided_stats: Dict[str, int] = {
        "deanon_guided_events_considered": 0,
        "deanon_guided_pulses_started": 0,
        "deanon_guided_msgs_sent": 0,
        "deanon_guided_events_skipped_scope_gate": 0,
    }

    def _window_at(t_s: int) -> Optional[Tuple[int, int]]:
        for s, e in ctx.attack_windows:
            if s <= t_s <= e:
                return (s, e)
        return None

    def _schedule_deanon_guided_exhaustion():
        """
        Deanon-guided exhaustion:
        For each QAN event, infer sender from observed metadata and launch a short
        auth flood focused on inferred sender -> receiver path while inside attack windows.
        """
        if not ctx.attack_windows:
            return
        ordered = sorted(qan_specs, key=lambda x: int(x["t_event_s"]))
        for qi in ordered:
            deanon_guided_stats["deanon_guided_events_considered"] += 1
            spec = qi["spec"]
            eval_t = int(qi["t_event_s"]) + int(spec.window_s)
            now_s = int(ctx.env.now)
            if eval_t > now_s:
                yield ctx.env.timeout(eval_t - now_s)

            win = _window_at(int(ctx.env.now))
            if win is None:
                continue

            # Estimate sender using attacker observation scope.
            ctx.analyzer.arm_event(spec)
            infer = ctx.analyzer.infer_sender()
            top1_prob = float(infer.get("top1_prob", 0.0) or 0.0)
            abstained = bool(infer.get("abstained", False))
            if deanon_guided_require_non_abstain and abstained:
                deanon_guided_stats["deanon_guided_events_skipped_scope_gate"] += 1
                continue
            if top1_prob < float(deanon_guided_min_top1_prob):
                deanon_guided_stats["deanon_guided_events_skipped_scope_gate"] += 1
                continue
            guessed = str(infer.get("top1_candidate") or infer.get("top1") or "")
            receiver = str(qi["receiver"])
            if guessed in ("", "unknown") or guessed not in nodes or guessed == receiver:
                pool = [n for n in nodes if n != receiver]
                guessed = ctx.rng.choice(pool or nodes)

            _, win_end = win
            pulse = int(max(20, min(int(deanon_guided_pulse_s), win_end - int(ctx.env.now))))
            if pulse <= 0:
                continue
            deanon_guided_stats["deanon_guided_pulses_started"] += 1

            inter = 1.0 / max(0.1, float(atk_cfg["exhaust_rate"]))
            t_end = float(ctx.env.now) + float(pulse)
            while float(ctx.env.now) <= t_end:
                msg = Message(
                    msg_id=ctx.msg_id_fn(),
                    created_ms=int(ctx.env.now * 1000),
                    src=guessed,
                    dst=receiver,
                    msg_type=MsgType.CONTROL_SETPOINT,
                    priority=1,
                    deadline_ms=350,
                    size_bytes=240,
                    requires_auth=True,
                    requires_anon=False,
                    is_attack=True,
                    attack_label="key_exhaust_deanon_guided",
                    payload={
                        "attack": True,
                        "attack_label": "key_exhaust_deanon_guided",
                        "target_strategy": "deanon_guided",
                        "deanon_top1_prob": float(infer.get("top1_prob", 0.0)),
                        "deanon_abstained": int(bool(infer.get("abstained", False))),
                        "deanon_obs_window": int(infer.get("n_obs_window", 0)),
                        "deanon_inferred_sender": guessed,
                        "shed_frac_target": 0.0,
                    },
                )
                ctx.emit_fn(msg)
                deanon_guided_stats["deanon_guided_msgs_sent"] += 1
                jitter = ctx.rng.uniform(-0.2 * inter, 0.2 * inter)
                yield ctx.env.timeout(max(0.001, inter + jitter))
    
    def _window_rate_segments(start_s: int, end_s: int, full_rate: float) -> List[Tuple[int, int, float]]:
        """Optional linear ramp-in/out segments for smoother attack transients."""
        dur = int(end_s) - int(start_s)
        ramp = max(0, int(attack_ramp_s))
        if dur <= 1:
            return []
        if ramp <= 0 or dur <= 2 * ramp:
            return [(int(start_s), int(end_s), float(full_rate))]

        segs: List[Tuple[int, int, float]] = []
        steps = 5
        up_dt = max(1, ramp // steps)

        # Ramp up
        cur = int(start_s)
        for i in range(steps):
            nxt = min(int(start_s + ramp), cur + up_dt)
            if nxt > cur:
                scale = float(i + 1) / float(steps)
                segs.append((cur, nxt, float(full_rate) * scale))
            cur = nxt

        # Flat
        flat_s = int(start_s + ramp)
        flat_e = int(end_s - ramp)
        if flat_e > flat_s:
            segs.append((flat_s, flat_e, float(full_rate)))

        # Ramp down
        cur = int(end_s - ramp)
        for i in range(steps):
            nxt = min(int(end_s), cur + up_dt)
            if nxt > cur:
                scale = float(steps - i) / float(steps)
                segs.append((cur, nxt, float(full_rate) * scale))
            cur = nxt

        # Ensure coverage to end
        if segs and segs[-1][1] < int(end_s):
            s0, _, r0 = segs[-1]
            segs[-1] = (s0, int(end_s), r0)
        return segs

    # Schedule attacks
    if "spoof" in attacks:
        spoof_cfg = SpoofConfig(use_islanding=False, forced_shed_frac=0.70, harm_duration_s=45)
        spoof = SpoofingAttack(env=ctx.env, rng=ctx.rng, cfg=spoof_cfg,
                               msg_id_fn=ctx.msg_id_fn, emit_fn=ctx.emit_fn)
        for start, end in ctx.attack_windows:
            n_spoofs = max(1, (end - start) // 300)
            for j in range(n_spoofs):
                t = start + j * 300 + ctx.rng.randint(0, 60)
                if t < end:
                    controller = ctx.rng.choice(nodes)
                    victim = ctx.rng.choice([n for n in nodes if n != controller] or nodes)
                    spoof.schedule_spoof(t_spoof_s=t, controller=controller, victim=victim,
                                        inferred_sender=controller, label="spoof")
    
    if "exhaust" in attacks:
        try:
            strategy = ExhaustTargetStrategy(exhaust_strategy)
        except Exception:
            strategy = ExhaustTargetStrategy.UNIFORM

        if strategy == ExhaustTargetStrategy.DEANON_GUIDED:
            ctx.env.process(_schedule_deanon_guided_exhaustion())
        else:
            for start, end in ctx.attack_windows:
                for seg_start, seg_end, seg_rate in _window_rate_segments(start, end, float(atk_cfg["exhaust_rate"])):
                    if strategy == ExhaustTargetStrategy.UNIFORM:
                        ex_cfg = ExhaustConfig(start_s=seg_start, end_s=seg_end, rate_per_s=seg_rate)
                        ex = KeyExhaustionAttack(env=ctx.env, rng=ctx.rng, cfg=ex_cfg,
                                                msg_id_fn=ctx.msg_id_fn, emit_fn=ctx.emit_fn)
                        ex.schedule(src_nodes=nodes, dst_nodes=nodes)
                    else:
                        ex_cfg = TargetedExhaustConfig(
                            start_s=seg_start,
                            end_s=seg_end,
                            rate_per_s=seg_rate,
                            target_strategy=strategy,
                            focus_ratio=exhaust_focus,
                        )
                        ex = TargetedKeyExhaustionAttack(
                            env=ctx.env,
                            rng=ctx.rng,
                            cfg=ex_cfg,
                            msg_id_fn=ctx.msg_id_fn,
                            emit_fn=ctx.emit_fn,
                            topology=ctx.graph,
                            traffic_observer=ctx.analyzer,
                        )
                        ex.schedule()
    
    if "quantum" in attacks:
        # Apply quantum disturbance to bottleneck edges (high edge-betweenness),
        # rather than an arbitrary prefix of ctx.edges. This better matches the
        # "attack shortest-path/bottleneck" story and scales across topologies.
        quantum_target_edges: List[Tuple[str, str]] = []
        try:
            import networkx as nx
            bc = nx.edge_betweenness_centrality(ctx.graph)
            ranked = sorted(
                bc.items(),
                key=lambda kv: (-float(kv[1]), tuple(sorted((str(kv[0][0]), str(kv[0][1]))))),
            )
            # Target the top ~20% (at least 2) but cap to keep attack localized.
            n_pick = max(2, min(len(ranked), max(2, len(ranked) // 5)))
            quantum_target_edges = [tuple(sorted((str(u), str(v)))) for (u, v), _ in ranked[:n_pick]]
        except Exception:
            quantum_target_edges = [tuple(sorted((str(u), str(v)))) for (u, v) in ctx.edges[:2]]

        for start, end in ctx.attack_windows:
            for (u, v) in quantum_target_edges:
                ek = edge_key(u, v)
                base_q = float(getattr(ctx.qlayer.health.get(ek), "baseline_qber", 0.01))
                target_q = float(atk_cfg["qber"])
                for seg_start, seg_end, seg_rate in _window_rate_segments(start, end, 1.0):
                    seg_q = base_q + float(seg_rate) * max(0.0, target_q - base_q)
                    window = QBERWindow(
                        start_s=seg_start,
                        end_s=seg_end,
                        absolute_qber=seg_q,
                        segment_count=1,
                        segment_qber_std=0.0,
                        label="quantum_disturb",
                    )
                    ctx.qlayer.add_qber_window(ek, window)
    
    # Run
    ctx.env.run(until=horizon_s)
    
    # Process deanon
    for i, qan_info in enumerate(qan_specs):
        ctx.analyzer.arm_event(qan_info["spec"])
        result = ctx.analyzer.infer_sender()
        ctx.logger.record_deanon_result(
            event_id=f"{scenario}_{topology}_seed{seed}_ev{i}",
            t_event_s=qan_info["t_event_s"],
            receiver=qan_info["receiver"],
            true_sender=qan_info["true_sender"],
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
    
    return finalize_and_save(
        ctx=ctx,
        scenario=scenario,
        topology=topology,
        seed=seed,
        horizon_s=horizon_s,
        out_dir=out_dir,
        defense_mode=defense_mode,
        attack_intensity=attack_intensity,
        attacks=attacks,
        run_meta={
            "attacker_scope": attacker_scope,
            # Tap may be auto-selected in create_sim_context; record effective tap.
            "attacker_tap_node": (getattr(ctx.analyzer.observation_cfg, "tap_node", None) or ""),
            "attacker_tap_edge": (
                "-".join(getattr(ctx.analyzer.observation_cfg, "tap_edge", None))
                if getattr(ctx.analyzer.observation_cfg, "tap_edge", None)
                else ""
            ),
            "n_nodes": int(len(nodes)),
            "nodes_csv": ",".join(str(n) for n in nodes),
            "exhaust_strategy": str(exhaust_strategy),
            "exhaust_focus": float(exhaust_focus),
            "deanon_guided_min_top1_prob": float(deanon_guided_min_top1_prob),
            "deanon_guided_require_non_abstain": int(bool(deanon_guided_require_non_abstain)),
            "qan_profile": qan_profile,
            "qan_events_requested": int(qan_events),
            "qan_cover_rate_per_s": float(qan_cfg.cover_rate_per_s),
            "qan_window_s": int(qan_cfg.window_s),
            "qan_mixing_delay_ms": int(qan_cfg.mixing_delay_ms),
            "qan_sync_burst": int(bool(qan_cfg.sync_burst_per_candidate)),
            "qan_auth_cover": int(bool(qan_cfg.auth_cover)),
            "qan_auth_real_notify": int(bool(qan_cfg.auth_real_notify)),
            "qan_auth_sync_burst": int(bool(qan_cfg.auth_sync_burst)),
            "quantum_target_edges": (
                "|".join(f"{u}-{v}" for (u, v) in (quantum_target_edges if "quantum" in attacks else []))
            ),
            **deanon_guided_stats,
        },
    )


def get_all_scenarios(attacks: List[str], defenses: List[str]) -> List[Tuple[List[str], str]]:
    """Generate all attack+defense combinations."""
    scenarios = []
    
    # Baseline
    scenarios.append(([], "none"))
    
    # Single attacks
    for atk in attacks:
        for defense in defenses:
            scenarios.append(([atk], defense))
    
    # Combined attacks (if multiple)
    if len(attacks) > 1:
        for defense in defenses:
            scenarios.append((attacks, defense))
    
    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Run full attack+defense matrix")
    parser.add_argument("--tag", default="full_matrix")
    parser.add_argument("--horizon_s", type=int, default=3600)
    parser.add_argument("--nodes", nargs="*", default=DEFAULT_NODES)
    parser.add_argument("--topology", default="ring")
    parser.add_argument("--route_policy", default="shortest",
                        choices=["shortest", "ecmp", "k_shortest", "k_shortest_weighted", "disjoint", "load_aware"],
                        help="Routing policy for message delivery")
    parser.add_argument("--k_paths", type=int, default=3,
                        help="Number of candidate paths for k_shortest policies")
    parser.add_argument("--use_grid_link_types", action="store_true",
                        help="Infer backbone/feeder/lateral link params automatically (affects latency/loss/bandwidth)")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--attacks", nargs="*", default=ALL_ATTACKS)
    parser.add_argument("--defenses", nargs="*", default=["none", "block", "intrusion", "all", "hardened"])
    parser.add_argument("--attack_intensity", default="S3")
    parser.add_argument("--fixed_infrastructure", action="store_true",
                        help="Use FIXED_INFRASTRUCTURE for pool sizing; "
                             "attack_intensity then selects from ATTACK_LEVELS (A1-A5)")
    parser.add_argument("--link_distance_km", type=float, default=10.0,
                        help="Default QKD link distance in km (affects key rate)")
    parser.add_argument("--fiber_loss_db_km", type=float, default=0.2,
                        help="Fiber attenuation in dB/km")
    parser.add_argument("--finite_key", default="disabled",
                        choices=["disabled", "large_block", "medium_block", "small_block", "high_security"],
                        help="Finite-key correction preset")
    parser.add_argument("--finite_key_block_bits", type=int, default=None,
                        help="Override finite-key block size (bits)")
    parser.add_argument("--finite_key_security_log", type=int, default=None,
                        help="Override -log10(eps) for finite-key correction")
    parser.add_argument("--degraded_threshold", default="moderate",
                        help="Degraded mode threshold preset or float (0-1)")
    parser.add_argument("--attacker_scope", default="global",
                        choices=["global", "node_tap", "edge_tap"],
                        help="Deanonymization attacker visibility model")
    parser.add_argument("--tap_node", default=None,
                        help="Tapped node id when --attacker_scope node_tap")
    parser.add_argument("--tap_edge", default=None,
                        help="Tapped edge when --attacker_scope edge_tap (format: A-B)")
    parser.add_argument("--distributed_attacks", action="store_true")
    parser.add_argument("--num_attack_windows", type=int, default=5)
    parser.add_argument("--exhaust_strategy", default="uniform",
                        choices=["uniform", "bottleneck", "bridge", "high_traffic", "single_link", "bypassable", "star_center", "deanon_guided"],
                        help="Key exhaustion targeting strategy")
    parser.add_argument("--exhaust_focus", type=float, default=0.8,
                        help="Fraction of exhaustion traffic focused on targets (0-1)")
    parser.add_argument("--deanon_guided_pulse_s", type=int, default=90,
                        help="Burst duration (s) for each deanon-guided exhaustion pulse")
    parser.add_argument("--deanon_guided_min_top1_prob", type=float, default=0.0,
                        help="Only launch deanon-guided pulse when inferred top1_prob exceeds this threshold")
    parser.add_argument("--deanon_guided_require_non_abstain", action="store_true",
                        help="Skip deanon-guided pulse for abstained/unknown inferences")
    parser.add_argument("--energy_interval", type=int, default=10)
    parser.add_argument("--quantum_interval_s", type=int, default=30,
                        help="Quantum telemetry sampling interval in seconds (set 1 for smooth plots)")
    parser.add_argument("--attack_ramp_s", type=int, default=0,
                        help="Linear ramp-in/out duration (seconds) for exhaust and quantum attacks")
    parser.add_argument("--qan_events", type=int, default=24)
    parser.add_argument("--qan_auth_cover", action="store_true",
                        help="Require auth (key usage) for QAN cover messages")
    parser.add_argument("--qan_auth_real", action="store_true",
                        help="Require auth (key usage) for real QAN_NOTIFY messages")
    parser.add_argument("--qan_auth_sync", action="store_true",
                        help="Require auth (key usage) for sync-burst cover messages")
    parser.add_argument("--list", action="store_true", help="List scenarios without running")
    args = parser.parse_args()
    
    scenarios = get_all_scenarios(args.attacks, args.defenses)
    
    if args.list:
        print("\nScenarios to run:")
        for i, (attacks, defense) in enumerate(scenarios, 1):
            atk_str = "_".join(attacks) if attacks else "baseline"
            print(f"  {i:2d}. {atk_str}_def_{defense}")
        print(f"\nTotal: {len(scenarios)} scenarios x {len(args.seeds)} seeds = {len(scenarios) * len(args.seeds)} runs")
        return
    
    out_dir = make_output_dir(args.tag)
    
    print(f"\n{'='*70}")
    print(f"QuAM Full Matrix Experiment")
    print(f"{'='*70}")
    print(f"Horizon: {args.horizon_s}s ({args.horizon_s/3600:.1f}h)")
    print(f"Attacks: {args.attacks}")
    print(f"Defenses: {args.defenses}")
    print(f"Seeds: {args.seeds}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Total runs: {len(scenarios) * len(args.seeds)}")
    print(f"Output: {out_dir}")
    print(f"{'='*70}\n")
    
    rows = []
    total = len(scenarios) * len(args.seeds)
    current = 0
    
    for seed in args.seeds:
        for attacks, defense in scenarios:
            current += 1
            atk_str = "_".join(attacks) if attacks else "baseline"
            print(f"[{current}/{total}] {atk_str}_def_{defense} | seed={seed}")
            
            row = run_scenario(
                attacks=attacks,
                defense_mode=defense,
                seed=seed,
                nodes=args.nodes,
                topology=args.topology,
                horizon_s=args.horizon_s,
                out_dir=out_dir,
                attack_intensity=args.attack_intensity,
                infrastructure_override=FIXED_INFRASTRUCTURE if args.fixed_infrastructure else None,
                distributed_attacks=args.distributed_attacks,
                num_attack_windows=args.num_attack_windows,
                exhaust_strategy=args.exhaust_strategy,
                exhaust_focus=args.exhaust_focus,
                energy_interval=args.energy_interval,
                quantum_interval_s=args.quantum_interval_s,
                attack_ramp_s=args.attack_ramp_s,
                qan_events=args.qan_events,
                route_policy=args.route_policy,
                k_paths=args.k_paths,
                use_grid_link_types=args.use_grid_link_types,
                qan_auth_cover=args.qan_auth_cover,
                qan_auth_real=args.qan_auth_real,
                qan_auth_sync=args.qan_auth_sync,
                deanon_guided_pulse_s=args.deanon_guided_pulse_s,
                deanon_guided_min_top1_prob=args.deanon_guided_min_top1_prob,
                deanon_guided_require_non_abstain=args.deanon_guided_require_non_abstain,
                link_distance_km=args.link_distance_km,
                fiber_loss_db_per_km=args.fiber_loss_db_km,
                finite_key_preset=args.finite_key,
                finite_key_block_bits=args.finite_key_block_bits,
                finite_key_security_log=args.finite_key_security_log,
                degraded_threshold_preset=args.degraded_threshold,
                attacker_scope=args.attacker_scope,
                attacker_tap_node=args.tap_node,
                attacker_tap_edge=(
                    tuple(args.tap_edge.split("-", 1)) if isinstance(args.tap_edge, str) and "-" in args.tap_edge
                    else None
                ),
            )
            rows.append(row)
            
            delivered = row.get("delivered_ratio", 0) * 100
            eens = row.get("eens_critical_kwh", 0)
            blocked = sum([
                row.get("defense_blocked_degraded", 0),
                row.get("defense_blocked_intrusion", 0),
                row.get("defense_blocked_signature", 0),
            ])
            print(f"  → Delivered: {delivered:.1f}%, EENS: {eens:.2f}kWh, Blocked: {blocked}\n")
    
    summary_path = os.path.join(out_dir, "summary", "summary.csv")
    QuAMLogger.write_summary_csv(summary_path, rows)
    
    print(f"\n{'='*70}")
    print(f"COMPLETE!")
    print(f"{'='*70}")
    print(f"Summary: {summary_path}")
    print(f"Energy data: {out_dir}/energy/")
    print(f"\nTo plot energy curves:")
    print(f"  python3 plot_energy_curves.py --input {out_dir}/energy/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
