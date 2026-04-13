"""
metrics.py 

Result logging + metric computation for QuAM.

Design goals:
- Minimal dependencies (pandas optional; csv always supported)
- Produces:
    (1) per-message logs (CSV)
    (2) per-run summary (dict and/or CSV row)
- Computes the metrics promised in the abstract:
    Privacy:
        - deanonymization top-1 accuracy (per event)
        - posterior entropy (bits)
        - top1 probability / confidence
        - confidence calibration quality (ECE/Brier)
        - abstain/unknown rate
    Integrity:
        - spoof acceptance rate (attack actions allowed/applied)
        - false allow/block (if oracle labels exist)
    Key sufficiency:
        - dropped_no_keys ratio
        - key wait statistics (mean/p95 on delivered)
        - key bits spent (if logged by quantum.py)
    Availability/operations:
        - delivery latency, deadline miss ratio, drop ratios
        - (Optional) microgrid outcomes: unserved energy proxy if present on microgrid state

Expected integration:
- Runner calls:
    logger.record_message(msg)
- If you use TrafficAnalyzer:
    logger.record_deanon_result(event_id, true_sender, inferred_sender, top1_prob, entropy_bits)
- At end of run:
    summary = logger.summarize_run(...)
    logger.write_csvs(out_dir, run_tag)


Changes from original:
1. Added EnergyRecord dataclass
2. Added EnergyLogger class
3. Added write_energy_csv() method

This module does not import network.py or quantum.py directly.
"""


from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import csv
import os
import math
import statistics


# -------------------------
# Utilities (unchanged)
# -------------------------

def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _nanmean(xs: List[float]) -> float:
    ys = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(float(x))]
    return float(sum(ys) / len(ys)) if ys else float("nan")


def _quantile(xs: List[float], q: float) -> float:
    ys = sorted([float(x) for x in xs if isinstance(x, (int, float)) and not math.isnan(float(x))])
    if not ys:
        return float("nan")
    q = max(0.0, min(1.0, float(q)))
    k = (len(ys) - 1) * q
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - k) + ys[hi] * (k - lo)


# -------------------------
# Core log record types
# -------------------------

@dataclass
class DeanonRecord:
    event_id: str
    t_event_s: int
    receiver: str
    true_sender: str
    inferred_sender: str
    top1_prob: float
    entropy_bits: float
    top1_candidate: str = ""
    top2_prob: float = float("nan")
    top1_margin: float = float("nan")
    abstained: int = 0
    n_obs_window: int = 0
    prior_blend_weight: float = float("nan")


@dataclass
class MessageLogRow:
    msg_id: int
    created_ms: int
    src: str
    dst: str
    msg_type: str
    priority: int
    deadline_ms: int
    size_bytes: int

    requires_auth: int
    requires_anon: int

    is_attack: int
    attack_label: str

    status: str
    delivered_ms: int
    total_ms: float
    key_wait_ms: float

    qber_path_mean: float
    fidelity_path_min: float
    secret_fraction_path_mean: float
    path_total_distance_km: float
    path_avg_distance_factor: float
    finite_key_block_size: float
    finite_key_correction: float
    finite_key_factor: float
    key_bits_spent_total: float
    key_rotation_bits_total: float
    nonce_quality: str
    qrng_pool_bits: float
    qrng_fallback: int

    action: str
    gate_decision: str
    gate_reason: str
    gate_delay_ms: float

    # Network/routing diagnostics
    net_route_policy: str
    net_candidate_paths: int
    net_path_hops: int

    # V2: Multi-protocol quantum layer
    quantum_protocol: str = ""
    quantum_protocol_reason: str = ""

    # Confidentiality: was this message encrypted with QKD-derived keys?
    encrypted: int = 0


# -------------------------
# NEW: Energy Time Series Record
# -------------------------

@dataclass
class EnergyRecord:
    """Single timestep energy state for plotting."""
    t_s: int
    microgrid: str
    
    # Load (kW)
    total_load_kw: float
    critical_load_kw: float
    noncritical_load_kw: float
    
    # Supply (kW)
    gen_kw: float
    import_kw: float
    import_cap_kw: float
    
    # Balance (kW)
    served_kw: float
    unserved_kw: float
    unserved_critical_kw: float

    # Battery (kWh / kW)
    battery_kwh: float
    battery_discharge_kw: float
    battery_charge_kw: float

    # Control state
    shed_frac: float
    shed_target: float
    forced_shed_active: bool
    mode: str

    # Comms/control coupling
    comm_load_kw: float
    comm_energy_kwh: float
    control_quality: float
    control_on_time_ratio: float
    control_drop_ratio: float
    avg_control_latency_ms: float
    
    # Cumulative EENS (kWh)
    eens_cumulative_kwh: float
    eens_critical_cumulative_kwh: float
    
    # Attack context
    is_attack_window: bool
    active_attack: str


# -------------------------
# NEW: Energy Logger
# -------------------------

class EnergyLogger:
    """Logs energy state time series for all microgrids."""
    
    def __init__(self):
        self.records: List[EnergyRecord] = []
    
    def record(self, t_s: int, mg_name: str, mg: Any,
               is_attack: bool = False, attack_label: str = "none") -> None:
        """
        Record current energy state from a MicrogridState object.
        
        Args:
            t_s: Current simulation time in seconds
            mg_name: Name/ID of the microgrid
            mg: MicrogridState object
            is_attack: Whether an attack is currently active
            attack_label: Type of attack (e.g., "spoof", "exhaust", "quantum")
        """
        # Get import capacity based on mode
        mode_val = mg.mode.value if hasattr(mg.mode, 'value') else str(mg.mode)
        if mode_val == "grid_tied":
            import_cap = mg.params.import_cap_kw
        elif mode_val == "restoration":
            import_cap = 0.5 * mg.params.import_cap_kw
        else:
            import_cap = 0.0
        
        self.records.append(EnergyRecord(
            t_s=t_s,
            microgrid=mg_name,
            total_load_kw=mg.last_total_load_kw,
            critical_load_kw=getattr(mg, 'last_critical_load_kw', mg.params.critical_load_kw),
            noncritical_load_kw=max(0, mg.last_total_load_kw - mg.params.critical_load_kw),
            gen_kw=mg.last_gen_kw,
            import_kw=mg.last_import_kw,
            import_cap_kw=import_cap,
            served_kw=mg.last_served_kw,
            unserved_kw=mg.last_unserved_kw,
            unserved_critical_kw=getattr(mg, 'last_unserved_critical_kw', 0),
            battery_kwh=getattr(mg, 'battery_kwh', 0.0),
            battery_discharge_kw=getattr(mg, 'last_battery_discharge_kw', 0.0),
            battery_charge_kw=getattr(mg, 'last_battery_charge_kw', 0.0),
            shed_frac=mg.shed_frac,
            shed_target=mg.shed_target,
            forced_shed_active=(t_s <= mg.forced_shed_until_s),
            mode=mode_val,
            comm_load_kw=getattr(mg, "comm_load_kw", 0.0),
            comm_energy_kwh=getattr(mg, "comm_energy_kwh", 0.0),
            control_quality=getattr(mg, "control_quality", float("nan")),
            control_on_time_ratio=getattr(mg, "control_on_time_ratio", float("nan")),
            control_drop_ratio=getattr(mg, "control_drop_ratio", float("nan")),
            avg_control_latency_ms=getattr(mg, "avg_control_latency_ms", float("nan")),
            eens_cumulative_kwh=mg.eens_total_kwh,
            eens_critical_cumulative_kwh=mg.eens_critical_kwh,
            is_attack_window=is_attack,
            active_attack=attack_label,
        ))
    
    def write_csv(self, path: str) -> None:
        """Write energy time series to CSV."""
        if not self.records:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.records[0]).keys()))
            writer.writeheader()
            for r in self.records:
                writer.writerow(asdict(r))
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Compute summary statistics from energy records."""
        if not self.records:
            return {}
        
        # Group by microgrid
        by_mg: Dict[str, List[EnergyRecord]] = {}
        for r in self.records:
            by_mg.setdefault(r.microgrid, []).append(r)
        
        total_unserved = sum(r.unserved_kw for r in self.records)
        total_critical_unserved = sum(r.unserved_critical_kw for r in self.records)
        
        attack_records = [r for r in self.records if r.is_attack_window]
        attack_unserved = sum(r.unserved_kw for r in attack_records)
        
        max_shed = max(r.shed_frac for r in self.records)
        avg_shed_during_attack = _nanmean([r.shed_frac for r in attack_records]) if attack_records else 0

        # Comms energy is cumulative per microgrid; take last value per microgrid
        comm_energy_by_mg: Dict[str, float] = {}
        control_quality_samples: List[float] = []
        control_latency_samples: List[float] = []
        for r in self.records:
            comm_energy_by_mg[r.microgrid] = r.comm_energy_kwh
            if isinstance(r.control_quality, (int, float)) and not math.isnan(float(r.control_quality)):
                control_quality_samples.append(float(r.control_quality))
            if isinstance(r.avg_control_latency_ms, (int, float)) and not math.isnan(float(r.avg_control_latency_ms)):
                control_latency_samples.append(float(r.avg_control_latency_ms))

        return {
            "energy_records_count": len(self.records),
            "total_unserved_kw_sum": total_unserved,
            "total_critical_unserved_kw_sum": total_critical_unserved,
            "attack_unserved_kw_sum": attack_unserved,
            "max_shed_frac": max_shed,
            "avg_shed_during_attack": avg_shed_during_attack,
            "comm_energy_kwh_sum": float(sum(comm_energy_by_mg.values())) if comm_energy_by_mg else 0.0,
            "control_quality_mean": _nanmean(control_quality_samples) if control_quality_samples else float("nan"),
            "control_latency_mean_ms": _nanmean(control_latency_samples) if control_latency_samples else float("nan"),
            "control_latency_p95_ms": _quantile(control_latency_samples, 0.95) if control_latency_samples else float("nan"),
        }


# -------------------------
# Logger (mostly unchanged)
# -------------------------

@dataclass
class QuAMLogger:
    messages: List[MessageLogRow] = field(default_factory=list)
    deanons: List[DeanonRecord] = field(default_factory=list)
    action_events: List[Dict[str, Any]] = field(default_factory=list)

    def record_message(self, msg: Any) -> None:
        payload = getattr(msg, "payload", {}) or {}

        row = MessageLogRow(
            msg_id=_safe_int(getattr(msg, "msg_id", -1)),
            created_ms=_safe_int(getattr(msg, "created_ms", -1)),
            src=str(getattr(msg, "src", "")),
            dst=str(getattr(msg, "dst", "")),
            msg_type=str(getattr(getattr(msg, "msg_type", ""), "value", getattr(msg, "msg_type", ""))),
            priority=_safe_int(getattr(msg, "priority", 0)),
            deadline_ms=_safe_int(getattr(msg, "deadline_ms", -1)),
            size_bytes=_safe_int(getattr(msg, "size_bytes", 0)),

            requires_auth=1 if bool(getattr(msg, "requires_auth", False)) else 0,
            requires_anon=1 if bool(getattr(msg, "requires_anon", False)) else 0,

            is_attack=1 if bool(getattr(msg, "is_attack", False)) else 0,
            attack_label=str(getattr(msg, "attack_label", "") or payload.get("attack_label", "") or ""),

            status=str(getattr(getattr(msg, "status", ""), "value", getattr(msg, "status", ""))),
            delivered_ms=_safe_int(getattr(msg, "delivered_ms", -1)),
            # Message stores network latency as total_latency_ms (set on delivery).
            # Older code used total_ms; keep a fallback for backward compatibility.
            total_ms=_safe_float(getattr(msg, "total_latency_ms", getattr(msg, "total_ms", float("nan")))),
            key_wait_ms=_safe_float(getattr(msg, "key_wait_ms", float("nan"))),

            qber_path_mean=_safe_float(payload.get("qber_path_mean", float("nan"))),
            fidelity_path_min=_safe_float(payload.get("fidelity_path_min", float("nan"))),
            secret_fraction_path_mean=_safe_float(payload.get("secret_fraction_path_mean", float("nan"))),
            path_total_distance_km=_safe_float(payload.get("path_total_distance_km", float("nan"))),
            path_avg_distance_factor=_safe_float(payload.get("path_avg_distance_factor", float("nan"))),
            finite_key_block_size=_safe_float(payload.get("finite_key_block_size", float("nan"))),
            finite_key_correction=_safe_float(payload.get("finite_key_correction", float("nan"))),
            finite_key_factor=_safe_float(payload.get("finite_key_factor", float("nan"))),
            key_bits_spent_total=_safe_float(payload.get("key_bits_spent_total", float("nan"))),
            key_rotation_bits_total=_safe_float(payload.get("key_rotation_bits_total", float("nan"))),
            nonce_quality=str(payload.get("nonce_quality", "")),
            qrng_pool_bits=_safe_float(payload.get("qrng_pool_bits", float("nan"))),
            qrng_fallback=_safe_int(payload.get("qrng_fallback", 0), 0),

            action=str(payload.get("action", "")),
            gate_decision=str(payload.get("gate_decision", "")),
            gate_reason=str(payload.get("gate_reason", "")),
            gate_delay_ms=_safe_float(payload.get("gate_delay_ms", float("nan"))),

            net_route_policy=str(payload.get("net_route_policy", "")),
            net_candidate_paths=_safe_int(payload.get("net_candidate_paths", 0), 0),
            net_path_hops=_safe_int(payload.get("net_path_hops", 0), 0),

            quantum_protocol=str(payload.get("quantum_protocol", "")),
            quantum_protocol_reason=str(payload.get("quantum_protocol_reason", "")),
            encrypted=_safe_int(payload.get("encrypted", 0), 0),
        )
        self.messages.append(row)

    def record_deanon_result(
        self,
        *,
        event_id: str,
        t_event_s: int,
        receiver: str,
        true_sender: str,
        inferred_sender: str,
        top1_prob: float,
        entropy_bits: float,
        top1_candidate: str = "",
        top2_prob: float = float("nan"),
        top1_margin: float = float("nan"),
        abstained: int = 0,
        n_obs_window: int = 0,
        prior_blend_weight: float = float("nan"),
    ) -> None:
        self.deanons.append(DeanonRecord(
            event_id=str(event_id),
            t_event_s=int(t_event_s),
            receiver=str(receiver),
            true_sender=str(true_sender),
            inferred_sender=str(inferred_sender),
            top1_prob=float(top1_prob),
            entropy_bits=float(entropy_bits),
            top1_candidate=str(top1_candidate),
            top2_prob=float(top2_prob),
            top1_margin=float(top1_margin),
            abstained=int(abstained),
            n_obs_window=int(n_obs_window),
            prior_blend_weight=float(prior_blend_weight),
        ))

    def record_action_event(self, **kwargs: Any) -> None:
        self.action_events.append(dict(kwargs))

    # -------------------------
    # Summaries
    # -------------------------

    def summarize_privacy(self) -> Dict[str, Any]:
        if not self.deanons:
            return {
                "n_events": 0,
                "deanon_top1_acc": float("nan"),
                "deanon_entropy_mean_bits": float("nan"),
                "deanon_top1prob_mean": float("nan"),
                "deanon_top1_acc_non_abstain": float("nan"),
                "deanon_abstain_rate": float("nan"),
                "deanon_brier_top1": float("nan"),
                "deanon_ece_top1": float("nan"),
            }

        n = len(self.deanons)
        correct_flags = [1 if (r.true_sender == r.inferred_sender) else 0 for r in self.deanons]
        correct = sum(correct_flags)
        non_abstain = [r for r in self.deanons if int(getattr(r, "abstained", 0)) == 0 and r.inferred_sender != "unknown"]
        non_abstain_correct = sum(1 for r in non_abstain if r.true_sender == r.inferred_sender)

        probs = [clamp(float(r.top1_prob), 0.0, 1.0) for r in self.deanons]
        brier = _nanmean([(p - y) ** 2 for p, y in zip(probs, correct_flags)])

        # ECE on top-1 confidence (binary outcome: top-1 correct or not).
        n_bins = 10
        ece_sum = 0.0
        for i in range(n_bins):
            lo = i / n_bins
            hi = (i + 1) / n_bins
            if i < n_bins - 1:
                idx = [j for j, p in enumerate(probs) if lo <= p < hi]
            else:
                idx = [j for j, p in enumerate(probs) if lo <= p <= hi]
            if not idx:
                continue
            acc = sum(correct_flags[j] for j in idx) / len(idx)
            conf = sum(probs[j] for j in idx) / len(idx)
            ece_sum += abs(acc - conf) * (len(idx) / n)

        return {
            "n_events": n,
            "deanon_top1_acc": correct / n,
            "deanon_entropy_mean_bits": _nanmean([r.entropy_bits for r in self.deanons]),
            "deanon_top1prob_mean": _nanmean([r.top1_prob for r in self.deanons]),
            "deanon_top1_acc_non_abstain": (non_abstain_correct / len(non_abstain)) if non_abstain else float("nan"),
            "deanon_abstain_rate": _nanmean([float(int(getattr(r, "abstained", 0))) for r in self.deanons]),
            "deanon_brier_top1": brier,
            "deanon_ece_top1": ece_sum,
        }

    def summarize_delivery(self) -> Dict[str, Any]:
        n = len(self.messages)
        if n == 0:
            return {}

        def _effective_total_ms(m: MessageLogRow) -> float:
            """
            Effective end-to-end latency for control actions can include gate verification delay.

            - total_ms is network/queueing latency (from Message).
            - gate_delay_ms is the post-delivery verification delay (from PolicyGate).
            """
            base = float(m.total_ms) if isinstance(m.total_ms, (int, float)) else float("nan")
            if math.isnan(base):
                return base
            gd = float(getattr(m, "gate_delay_ms", float("nan")))
            if not math.isnan(gd):
                base += gd
            return base

        def _is_deadline_miss(m: MessageLogRow) -> bool:
            status = str(m.status).lower()
            if ("late" in status) or ("expired" in status):
                return True
            eff = _effective_total_ms(m)
            return (not math.isnan(eff)) and (m.deadline_ms > 0) and (eff > m.deadline_ms)

        def _is_control_msg(m: MessageLogRow) -> bool:
            t = str(m.msg_type).lower()
            return t in ("control_setpoint", "priority_action")

        delivered = [m for m in self.messages if "delivered" in m.status]
        dropped = [m for m in self.messages if "dropped" in m.status]
        dropped_no_keys = [m for m in self.messages if "no_keys" in m.status]
        deadline_miss = [m for m in self.messages if _is_deadline_miss(m)]

        late = [
            m for m in delivered
            if ("late" in m.status)
            or (not math.isnan(_effective_total_ms(m)) and m.deadline_ms > 0 and _effective_total_ms(m) > m.deadline_ms)
        ]

        delivered_lat = [m.total_ms for m in delivered if not math.isnan(m.total_ms)]
        delivered_key_wait = [m.key_wait_ms for m in delivered if not math.isnan(m.key_wait_ms)]
        control_msgs = [m for m in self.messages if _is_control_msg(m)]
        control_delivered = [m for m in control_msgs if "delivered" in str(m.status)]
        control_lat = [_effective_total_ms(m) for m in control_delivered if not math.isnan(_effective_total_ms(m))]
        control_deadline_miss = [m for m in control_msgs if _is_deadline_miss(m)]

        return {
            "n_msgs": n,
            "delivered_ratio": len(delivered) / n,
            "dropped_ratio": len(dropped) / n,
            "dropped_no_keys_ratio": len(dropped_no_keys) / n,
            "deadline_miss_ratio": len(deadline_miss) / n,
            "late_ratio": (len(late) / len(delivered)) if delivered else float("nan"),
            "delivered_latency_mean_ms": _nanmean(delivered_lat),
            "delivered_latency_p95_ms": _quantile(delivered_lat, 0.95),
            "delivered_key_wait_mean_ms": _nanmean(delivered_key_wait),
            "delivered_key_wait_p95_ms": _quantile(delivered_key_wait, 0.95),
            "control_msgs_total": len(control_msgs),
            "control_deadline_miss_ratio": (
                len(control_deadline_miss) / len(control_msgs)
            ) if control_msgs else float("nan"),
            "control_latency_mean_ms": _nanmean(control_lat),
            "control_latency_p95_ms": _quantile(control_lat, 0.95),
            # Encryption coverage: fraction of delivered messages that were
            # encrypted with QKD-derived keys (confidentiality metric)
            "encryption_coverage_ratio": (
                sum(1 for m in delivered if int(getattr(m, "encrypted", 0) or 0) == 1)
                / len(delivered)
            ) if delivered else 0.0,
        }

    def summarize_integrity(self) -> Dict[str, Any]:
        def _is_allowed(decision: str) -> bool:
            d = str(decision).strip().lower()
            return d in ("allow", "applied")

        if self.action_events:
            pri = [e for e in self.action_events if str(e.get("action", "")).strip() != ""]
            attack_pri = [e for e in pri if int(e.get("is_attack", 0)) == 1]
            legit_pri = [e for e in pri if int(e.get("is_attack", 0)) == 0]
            allowed_attack = [e for e in attack_pri if _is_allowed(str(e.get("decision", "")))]
            blocked_attack = [e for e in attack_pri if not _is_allowed(str(e.get("decision", "")))]
            allowed_legit = [e for e in legit_pri if _is_allowed(str(e.get("decision", "")))]
            blocked_legit = [e for e in legit_pri if not _is_allowed(str(e.get("decision", "")))]
            return {
                "n_action_events": len(self.action_events),
                "n_attack_action_events": len(attack_pri),
                "n_legit_action_events": len(legit_pri),
                "attack_action_allow_rate": (len(allowed_attack) / len(attack_pri)) if attack_pri else float("nan"),
                "attack_action_block_rate": (len(blocked_attack) / len(attack_pri)) if attack_pri else float("nan"),
                "false_allow_rate": (len(allowed_attack) / len(attack_pri)) if attack_pri else float("nan"),
                "false_block_rate": (len(blocked_legit) / len(legit_pri)) if legit_pri else float("nan"),
                "true_allow_rate": (len(allowed_legit) / len(legit_pri)) if legit_pri else float("nan"),
                "attack_allowed_count": len(allowed_attack),
                "attack_blocked_count": len(blocked_attack),
                "legit_allowed_count": len(allowed_legit),
                "legit_blocked_count": len(blocked_legit),
            }

        # Fall back to message-level gate fields recorded in Message.payload (see finalmain.py).
        # Use *all* actionable control messages (control_setpoint + priority_action) so we can
        # quantify availability cost (false blocks) even when there are no legitimate priority
        # actions in the workload.
        action_msgs = [m for m in self.messages if str(getattr(m, "action", "")).strip() != ""]
        attack_actions = [m for m in action_msgs if m.is_attack == 1]
        legit_actions = [m for m in action_msgs if m.is_attack == 0]

        allowed_attack = [m for m in attack_actions if _is_allowed(str(m.gate_decision))]
        blocked_attack = [m for m in attack_actions if not _is_allowed(str(m.gate_decision))]
        allowed_legit = [m for m in legit_actions if _is_allowed(str(m.gate_decision))]
        blocked_legit = [m for m in legit_actions if not _is_allowed(str(m.gate_decision))]

        pri_msgs = [m for m in self.messages if str(m.msg_type).lower() == "priority_action"]
        attack_pri = [m for m in pri_msgs if m.is_attack == 1]
        legit_pri = [m for m in pri_msgs if m.is_attack == 0]
        allowed_attack_pri = [m for m in attack_pri if _is_allowed(str(m.gate_decision))]
        blocked_attack_pri = [m for m in attack_pri if not _is_allowed(str(m.gate_decision))]
        return {
            "n_priority_msgs": len(pri_msgs),
            "n_attack_priority_msgs": len(attack_pri),
            "n_legit_priority_msgs": len(legit_pri),
            # Spoof acceptance (priority-action attacks).
            "attack_priority_allow_rate": (
                len(allowed_attack_pri) / len(attack_pri)
            ) if attack_pri else float("nan"),
            "attack_priority_block_rate": (
                len(blocked_attack_pri) / len(attack_pri)
            ) if attack_pri else float("nan"),
            # Overall integrity / availability trade-off across all actionable messages.
            "false_allow_rate": (len(allowed_attack) / len(attack_actions)) if attack_actions else float("nan"),
            "false_block_rate": (len(blocked_legit) / len(legit_actions)) if legit_actions else float("nan"),
            "true_allow_rate": (len(allowed_legit) / len(legit_actions)) if legit_actions else float("nan"),
            "attack_allowed_count": len(allowed_attack),
            "attack_blocked_count": len(blocked_attack),
            "legit_allowed_count": len(allowed_legit),
            "legit_blocked_count": len(blocked_legit),
        }

    def summarize_quantum_health(self) -> Dict[str, Any]:
        qbers = [m.qber_path_mean for m in self.messages if not math.isnan(m.qber_path_mean)]
        fmins = [m.fidelity_path_min for m in self.messages if not math.isnan(m.fidelity_path_min)]
        sfs = [m.secret_fraction_path_mean for m in self.messages if not math.isnan(m.secret_fraction_path_mean)]
        spent = [m.key_bits_spent_total for m in self.messages if not math.isnan(m.key_bits_spent_total)]
        rot_bits = [m.key_rotation_bits_total for m in self.messages if not math.isnan(m.key_rotation_bits_total)]
        dist_totals = [m.path_total_distance_km for m in self.messages if not math.isnan(m.path_total_distance_km)]
        dist_factors = [m.path_avg_distance_factor for m in self.messages if not math.isnan(m.path_avg_distance_factor)]

        key_bits_spent_sum = float(sum(spent)) if spent else 0.0
        rotation_bits_total = float(sum(rot_bits)) if rot_bits else 0.0
        rotation_events_total = int(sum(1 for x in rot_bits if x > 0))
        rotation_overhead_ratio = rotation_bits_total / key_bits_spent_sum if key_bits_spent_sum > 0 else float("nan")

        # MessageLogRow.msg_type stores MsgType.value strings (lowercase).
        spent_cover = float(sum(
            m.key_bits_spent_total
            for m in self.messages
            if (not math.isnan(m.key_bits_spent_total)) and str(m.msg_type) == "cover"
        ))
        spent_qan_real = float(sum(
            m.key_bits_spent_total
            for m in self.messages
            if (not math.isnan(m.key_bits_spent_total)) and str(m.msg_type) == "qan_notify"
        ))
        spent_qan_total = spent_cover + spent_qan_real
        spent_non_qan = max(0.0, key_bits_spent_sum - spent_qan_total)
        spent_qan_share = spent_qan_total / key_bits_spent_sum if key_bits_spent_sum > 0 else float("nan")
        spent_cover_share = spent_cover / key_bits_spent_sum if key_bits_spent_sum > 0 else float("nan")

        avg_dist = _nanmean(dist_totals)
        avg_factor = _nanmean(dist_factors)
        reduction_pct = (1.0 - avg_factor) * 100.0 if not math.isnan(avg_factor) else float("nan")

        return {
            "qber_mean": _nanmean(qbers),
            "qber_p95": _quantile(qbers, 0.95),
            "fidelity_min_mean": _nanmean(fmins),
            "secret_fraction_mean": _nanmean(sfs),
            "key_bits_spent_mean": _nanmean(spent),
            "key_bits_spent_sum": key_bits_spent_sum,
            "rotation_events_total": rotation_events_total,
            "rotation_bits_total": rotation_bits_total,
            "rotation_overhead_ratio": rotation_overhead_ratio,
            "key_bits_spent_qan_cover_sum": spent_cover,
            "key_bits_spent_qan_real_sum": spent_qan_real,
            "key_bits_spent_qan_total_sum": spent_qan_total,
            "key_bits_spent_non_qan_sum": spent_non_qan,
            "key_bits_spent_qan_share": spent_qan_share,
            "key_bits_spent_cover_share": spent_cover_share,
            "path_total_distance_km_mean": avg_dist,
            "path_avg_distance_factor_mean": avg_factor,
            "effective_key_rate_reduction_pct": reduction_pct,
        }

    def summarize_quantum_protocols(self) -> Dict[str, Any]:
        """Summarize multi-protocol quantum layer usage."""
        proto_msgs = [m for m in self.messages if str(m.quantum_protocol).strip() != ""]
        if not proto_msgs:
            return {}

        n = len(proto_msgs)
        by_proto: Dict[str, int] = {}
        for m in proto_msgs:
            p = str(m.quantum_protocol)
            by_proto[p] = by_proto.get(p, 0) + 1

        # KAK messages (zero key consumption)
        kak_msgs = [m for m in proto_msgs if "kak" in str(m.quantum_protocol).lower()
                     or "quantum_tls" in str(m.quantum_protocol).lower()]
        kak_key_saved = sum(
            _safe_float(getattr(m, "key_bits_spent_total", 0), 0.0)
            for m in kak_msgs
            if _safe_float(getattr(m, "key_bits_spent_total", float("nan")), float("nan")) == 0.0
        )

        result = {
            "qproto_msgs_total": n,
            "qproto_kak_count": len(kak_msgs),
            "qproto_kak_ratio": len(kak_msgs) / n if n > 0 else 0.0,
        }
        for proto, count in by_proto.items():
            result[f"qproto_{proto}_count"] = count
            result[f"qproto_{proto}_ratio"] = count / n if n > 0 else 0.0

        return result

    def summarize_run(
        self,
        *,
        scenario: str,
        topology: str,
        seed: int,
        horizon_ms: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out.update({
            "scenario": str(scenario),
            "topology": str(topology),
            "seed": int(seed),
            "horizon_ms": int(horizon_ms),
        })
        out.update(self.summarize_privacy())
        out.update(self.summarize_integrity())
        out.update(self.summarize_delivery())
        out.update(self.summarize_quantum_health())
        out.update(self.summarize_quantum_protocols())

        if extra:
            for k, v in extra.items():
                out[str(k)] = v
        return out

    # -------------------------
    # CSV writers
    # -------------------------

    def write_message_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = [asdict(m) for m in self.messages]
        if not rows:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(list(asdict(MessageLogRow(
                    msg_id=0, created_ms=0, src="", dst="", msg_type="", priority=0, deadline_ms=0, size_bytes=0,
                    requires_auth=0, requires_anon=0, is_attack=0, attack_label="",
                    status="", delivered_ms=0, total_ms=float("nan"), key_wait_ms=float("nan"),
                    qber_path_mean=float("nan"), fidelity_path_min=float("nan"), secret_fraction_path_mean=float("nan"),
                    path_total_distance_km=float("nan"), path_avg_distance_factor=float("nan"),
                    finite_key_block_size=float("nan"), finite_key_correction=float("nan"), finite_key_factor=float("nan"),
                    key_bits_spent_total=float("nan"), key_rotation_bits_total=float("nan"),
                    nonce_quality="", qrng_pool_bits=float("nan"), qrng_fallback=0,
                    action="", gate_decision="", gate_reason="", gate_delay_ms=float("nan"),
                    net_route_policy="", net_candidate_paths=0, net_path_hops=0,
                    quantum_protocol="", quantum_protocol_reason="",
                )).keys()))
            return

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_deanon_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = [asdict(r) for r in self.deanons]
        if not rows:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "event_id",
                    "t_event_s",
                    "receiver",
                    "true_sender",
                    "inferred_sender",
                    "top1_prob",
                    "entropy_bits",
                    "top1_candidate",
                    "top2_prob",
                    "top1_margin",
                    "abstained",
                    "n_obs_window",
                    "prior_blend_weight",
                ])
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def write_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not rows:
            return
        keys: List[str] = sorted({k for r in rows for k in r.keys()})
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


# -------------------------
# IEEE/NERC Standard Resilience Metrics
# -------------------------

def compute_resilience_metrics(
    microgrids: Dict[str, Any],
    horizon_s: int,
    n_customers_per_mg: int = 200,
) -> Dict[str, float]:
    """
    Compute IEEE 1366-2012 / NERC standard power system reliability indices.

    Metrics
    -------
    SAIDI : System Average Interruption Duration Index (min/customer)
    SAIFI : System Average Interruption Frequency Index (events/customer)
    CAIDI : Customer Average Interruption Duration Index (min/event)
    LOLP  : Loss of Load Probability (fraction of time with unserved load)
    ASAI  : Average Service Availability Index (fraction)
    ENS   : Energy Not Served (kWh)

    Parameters
    ----------
    microgrids : dict mapping name → OperationalMicrogridState
    horizon_s  : simulation horizon (seconds)
    n_customers_per_mg : assumed customer count per microgrid node
    """
    n_mg = max(1, len(microgrids))
    total_customers = n_mg * n_customers_per_mg

    cust_int_minutes = 0.0      # sum of (customer × interruption-duration)
    cust_int_events = 0         # sum of (customer × interruption-events)
    total_lol_s = 0.0
    total_ens = 0.0
    total_crit_ens = 0.0

    for _name, mg in microgrids.items():
        outage_min = float(getattr(mg, "critical_outage_minutes", 0.0))
        ens = float(getattr(mg, "eens_total_kwh", 0.0))
        crit_ens = float(getattr(mg, "eens_critical_kwh", 0.0))

        # Customer-interruption-minutes (each MG serves n_customers_per_mg)
        cust_int_minutes += outage_min * n_customers_per_mg

        # Interruption events: approximate from outage duration
        # Assume average interruption lasts ~5 min; at least 1 if any outage
        if outage_min > 0.01:
            n_events = max(1, int(round(outage_min / 5.0)))
            cust_int_events += n_events * n_customers_per_mg

        total_lol_s += outage_min * 60.0
        total_ens += ens
        total_crit_ens += crit_ens

    saidi = cust_int_minutes / max(1, total_customers)
    saifi = cust_int_events / max(1, total_customers)
    caidi = saidi / max(0.001, saifi)

    # LOLP: average across MGs of (outage-seconds / horizon)
    lolp = total_lol_s / max(1, n_mg * horizon_s)
    lolp = max(0.0, min(1.0, lolp))

    # ASAI: 1 − (total customer-interruption-hours / total customer-hours)
    horizon_min = horizon_s / 60.0
    asai = 1.0 - saidi / max(1.0, horizon_min)
    asai = max(0.0, min(1.0, asai))

    return {
        "resilience_saidi_min": round(saidi, 4),
        "resilience_saifi": round(saifi, 4),
        "resilience_caidi_min": round(caidi, 4),
        "resilience_lolp": round(lolp, 6),
        "resilience_asai": round(asai, 6),
        "resilience_ens_kwh": round(total_ens, 4),
        "resilience_critical_ens_kwh": round(total_crit_ens, 4),
        "resilience_n_customers": total_customers,
    }


# -------------------------
# Plotting helpers
# -------------------------

def plot_quick_conference_figures(summary_rows: List[Dict[str, Any]], out_dir: str) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []

    def _group_mean(field: str) -> Tuple[List[str], List[float]]:
        by: Dict[str, List[float]] = {}
        for r in summary_rows:
            scen = str(r.get("scenario", ""))
            val = _safe_float(r.get(field, float("nan")))
            if not math.isnan(val):
                by.setdefault(scen, []).append(val)
        xs = sorted(by.keys())
        ys = [_nanmean(by[x]) for x in xs]
        return xs, ys

    # Deanonymization accuracy
    xs, ys = _group_mean("deanon_top1_acc")
    if xs:
        plt.figure()
        plt.bar(xs, ys)
        plt.ylabel("Top-1 deanonymization accuracy")
        plt.xticks(rotation=30, ha="right")
        p = os.path.join(out_dir, "fig_deanon_top1_acc.png")
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()
        saved.append(p)

    # Dropped no keys
    xs, ys = _group_mean("dropped_no_keys_ratio")
    if xs:
        plt.figure()
        plt.bar(xs, ys)
        plt.ylabel("Dropped (no keys) ratio")
        plt.xticks(rotation=30, ha="right")
        p = os.path.join(out_dir, "fig_drop_no_keys.png")
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()
        saved.append(p)

    # Spoof acceptance
    xs, ys = _group_mean("attack_priority_allow_rate")
    if xs:
        plt.figure()
        plt.bar(xs, ys)
        plt.ylabel("Attack priority allow rate")
        plt.xticks(rotation=30, ha="right")
        p = os.path.join(out_dir, "fig_attack_allow_rate.png")
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()
        saved.append(p)

    return saved
