#!/usr/bin/env python3
"""
run_spoof.py - Run spoofing attack scenarios

This script focuses on spoofing attacks with various defense configurations.
Use this for detailed analysis of spoof attack impact.

Usage:
  python3 -m quam.runners.run_spoof --tag spoof_study --horizon_s 3600 --seeds 0 1 2
  python3 -m quam.runners.run_spoof --defenses none block all --seeds 0 1 2 3 4
"""

import argparse
import os
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .common import (
    create_sim_context, schedule_background_workload, schedule_microgrid_stepper,
    make_output_dir, finalize_and_save, ATTACK_INTENSITIES
)
from .model import Message, MsgType
from .quantum import QBERWindow, edge_key
from .threat import SpoofConfig, SpoofingAttack, QANConfig, QANOrchestrator
from .metrics import QuAMLogger


DEFAULT_NODES = ["MG0", "MG1", "MG2"]


def run_spoof_scenario(
    *,
    seed: int,
    nodes: list,
    topology: str,
    defense_mode: str,
    horizon_s: int,
    out_dir: str,
    attack_intensity: str = "S3",
    num_spoofs: int = 10,
    spoof_interval_s: int = 120,
    forced_shed_frac: float = 0.70,
    harm_duration_s: int = 45,
    energy_interval: int = 10,
    link_distance_km: float = 10.0,
    fiber_loss_db_per_km: float = 0.2,
    finite_key_preset: str = "disabled",
    finite_key_block_bits: Optional[int] = None,
    finite_key_security_log: Optional[int] = None,
    degraded_threshold_preset: str = "conservative",
):
    """Run a single spoof attack scenario."""
    
    scenario = f"spoof_def_{defense_mode}"
    
    # Create simulation context
    ctx = create_sim_context(
        seed=seed,
        nodes=nodes,
        topology=topology,
        defense_mode=defense_mode,
        attack_intensity=attack_intensity,
        link_distance_km=link_distance_km,
        fiber_loss_db_per_km=fiber_loss_db_per_km,
        finite_key_preset=finite_key_preset,
        finite_key_block_bits=finite_key_block_bits,
        finite_key_security_log=finite_key_security_log,
        degraded_threshold_preset=degraded_threshold_preset,
    )
    
    # Define attack window
    attack_start = max(300, horizon_s // 4)
    attack_end = min(horizon_s - 300, attack_start + num_spoofs * spoof_interval_s)
    ctx.attack_windows = [(attack_start, attack_end)]
    
    # Schedule background workload
    schedule_background_workload(ctx, horizon_s, nodes)
    
    # Schedule microgrid stepper with energy logging
    schedule_microgrid_stepper(ctx, horizon_s, energy_interval)
    
    # Schedule spoof attacks
    spoof_cfg = SpoofConfig(
        use_islanding=False,
        forced_shed_frac=forced_shed_frac,
        harm_duration_s=harm_duration_s,
    )
    spoof = SpoofingAttack(
        env=ctx.env, rng=ctx.rng, cfg=spoof_cfg,
        msg_id_fn=ctx.msg_id_fn, emit_fn=ctx.emit_fn,
    )
    
    for i in range(num_spoofs):
        t = attack_start + i * spoof_interval_s + ctx.rng.randint(0, 30)
        if t < attack_end:
            controller = ctx.rng.choice(nodes)
            victim = ctx.rng.choice([n for n in nodes if n != controller] or nodes)
            spoof.schedule_spoof(
                t_spoof_s=t,
                controller=controller,
                victim=victim,
                inferred_sender=controller,
                label="spoof",
            )
    
    # Optionally add quantum disturbance to amplify spoof impact
    atk_cfg = ATTACK_INTENSITIES.get(attack_intensity, ATTACK_INTENSITIES["S3"])
    for (u, v) in ctx.edges[:2]:
        ek = edge_key(u, v)
        seg_count = ctx.rng.randint(3, 5)
        seg_std = max(0.002, 0.2 * float(atk_cfg["qber"]))
        window = QBERWindow(
            start_s=attack_start, end_s=attack_end,
            absolute_qber=atk_cfg["qber"],
            segment_count=seg_count,
            segment_qber_std=seg_std,
            label="quantum_disturb",
        )
        ctx.qlayer.add_qber_window(ek, window)
    
    # Run simulation
    ctx.env.run(until=horizon_s)
    
    # Save outputs and return summary
    return finalize_and_save(
        ctx=ctx,
        scenario=scenario,
        topology=topology,
        seed=seed,
        horizon_s=horizon_s,
        out_dir=out_dir,
        defense_mode=defense_mode,
        attack_intensity=attack_intensity,
        attacks=["spoof", "quantum"],
    )


def main():
    parser = argparse.ArgumentParser(description="Run spoofing attack scenarios")
    parser.add_argument("--tag", default="spoof_study")
    parser.add_argument("--horizon_s", type=int, default=3600)
    parser.add_argument("--nodes", nargs="*", default=DEFAULT_NODES)
    parser.add_argument("--topology", default="ring")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--defenses", nargs="*", default=["none", "block", "intrusion", "all"])
    parser.add_argument("--attack_intensity", default="S3")
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
    parser.add_argument("--degraded_threshold", default="conservative",
                        help="Degraded mode threshold preset or float (0-1)")
    parser.add_argument("--num_spoofs", type=int, default=10)
    parser.add_argument("--spoof_interval", type=int, default=120)
    parser.add_argument("--energy_interval", type=int, default=10)
    args = parser.parse_args()
    
    out_dir = make_output_dir(args.tag)
    
    print(f"\n{'='*60}")
    print(f"Spoof Attack Study")
    print(f"{'='*60}")
    print(f"Horizon: {args.horizon_s}s ({args.horizon_s/3600:.1f}h)")
    print(f"Defenses: {args.defenses}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")
    
    rows = []
    total = len(args.defenses) * len(args.seeds)
    current = 0
    
    for seed in args.seeds:
        for defense in args.defenses:
            current += 1
            print(f"[{current}/{total}] spoof_def_{defense} | seed={seed}")
            
            row = run_spoof_scenario(
                seed=seed,
                nodes=args.nodes,
                topology=args.topology,
                defense_mode=defense,
                horizon_s=args.horizon_s,
                out_dir=out_dir,
                attack_intensity=args.attack_intensity,
                num_spoofs=args.num_spoofs,
                spoof_interval_s=args.spoof_interval,
                energy_interval=args.energy_interval,
                link_distance_km=args.link_distance_km,
                fiber_loss_db_per_km=args.fiber_loss_db_km,
                finite_key_preset=args.finite_key,
                finite_key_block_bits=args.finite_key_block_bits,
                finite_key_security_log=args.finite_key_security_log,
                degraded_threshold_preset=args.degraded_threshold,
            )
            rows.append(row)
            
            delivered = row.get("delivered_ratio", 0) * 100
            eens = row.get("eens_critical_kwh", 0)
            blocked = row.get("defense_blocked_degraded", 0) + row.get("defense_blocked_intrusion", 0)
            blocked_rate = row.get("defense_blocked_rate_limit", 0)
            blocked_total = blocked + blocked_rate
            print(f"  → Delivered: {delivered:.1f}%, EENS: {eens:.2f}kWh, Blocked: {blocked_total} (rate_limit {blocked_rate})\n")
    
    # Also run baseline
    print(f"Running baseline...")
    baseline_row = run_baseline(args, out_dir)
    rows.insert(0, baseline_row)
    
    summary_path = os.path.join(out_dir, "summary", "summary.csv")
    QuAMLogger.write_summary_csv(summary_path, rows)
    print(f"\nSummary: {summary_path}")


def run_baseline(args, out_dir):
    """Run baseline (no attack) for comparison."""
    ctx = create_sim_context(
        seed=args.seeds[0],
        nodes=args.nodes,
        topology=args.topology,
        defense_mode="none",
        attack_intensity=args.attack_intensity,
        link_distance_km=args.link_distance_km,
        fiber_loss_db_per_km=args.fiber_loss_db_km,
        finite_key_preset=args.finite_key,
        finite_key_block_bits=args.finite_key_block_bits,
        finite_key_security_log=args.finite_key_security_log,
        degraded_threshold_preset=args.degraded_threshold,
    )
    
    schedule_background_workload(ctx, args.horizon_s, args.nodes)
    schedule_microgrid_stepper(ctx, args.horizon_s, args.energy_interval)
    
    ctx.env.run(until=args.horizon_s)
    
    return finalize_and_save(
        ctx=ctx,
        scenario="baseline",
        topology=args.topology,
        seed=args.seeds[0],
        horizon_s=args.horizon_s,
        out_dir=out_dir,
        defense_mode="none",
        attack_intensity=args.attack_intensity,
        attacks=[],
    )


if __name__ == "__main__":
    main()
