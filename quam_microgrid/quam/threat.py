"""
threat.py

Threat + defense layer for QuAM:
- QAN event orchestration (real notify + cover traffic)
- Traffic analysis adversary (posterior over sender, top-1, entropy)
- Priority-action spoofing campaign (target selected from deanonymization output)
- Key exhaustion campaign (authenticated message flood)
- Quantum disturbance campaign (raise QBER -> reduce secret key rate -> key scarcity)
- Policy gate for allow/block/ignore + verification delay trade-off + quantum-aware degraded mode

This module:
- Creates Message objects (from model.py)
- Does not depend on network.py; it emits messages via callbacks provided by the runner.

Key integration points:
- Runner provides:
    - msg_id generator
    - emit(msg): schedules msg into the network (net.send(msg))
    - on_deliver_hook: calls gate decision + applies actions

Assumptions (conference-scope):
- Attacker can observe message creation events (timing/volume metadata).
- Quantum disturbance is modeled as an increase in QBER on selected QKD edges (abstracted).

Changes from original:
1. Extended GateConfig with new defense options
2. Added DefenseStrategy enum
3. Added IntrusionDetector class
4. Extended PolicyGate with new blocking logic
5. Added statistics tracking

NEW DEFENSE STRATEGIES:
- intrusion: Block during QBER anomaly
- adaptive: Tighten rate limit when degraded
- signature: Block repeated identical commands
- quarantine: Isolate microgrid during attack
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
from enum import Enum
import math
import random
import hashlib
import csv
import os
import networkx as nx

import simpy

from .model import (
    Message,
    MsgType,
    ControlAction,
    ActionDecision,
    ActionType,
    parse_action_from_message,
    should_ignore_as_stale,
)

from .quantum import QuantumAugmentation, QBERWindow, edge_key, secret_fraction_bb84


# -------------------------
# Utilities
# -------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def softmax(scores: Dict[str, float], temperature: float = 2.0) -> Dict[str, float]:
    if not scores:
        return {}
    t = max(1e-6, float(temperature))
    m = max(scores.values())
    exps = {k: math.exp((v - m) / t) for k, v in scores.items()}
    s = sum(exps.values())
    n = len(scores)
    return {k: (exps[k] / s) if s > 0 else (1.0 / n) for k in scores}


def entropy_bits(posterior: Dict[str, float]) -> float:
    h = 0.0
    for p in posterior.values():
        if p > 1e-15:
            h -= p * math.log2(p)
    return h


# -------------------------
# NEW: Defense Strategy Enum
# -------------------------

class DefenseStrategy(str, Enum):
    NONE = "none"
    RATE_LIMIT = "ratelimit"
    BLOCK_DEGRADED = "block"
    DELAY = "delay"
    INTRUSION = "intrusion"
    ADAPTIVE = "adaptive"
    SIGNATURE = "signature"
    QUARANTINE = "quarantine"
    ALL = "all"
    # IMPROVED strategies
    RATE_LIMIT_V2 = "ratelimit_v2"
    INTRUSION_V2 = "intrusion_v2"
    PLAUSIBILITY = "plausibility"
    CORRELATION = "correlation"
    QUARANTINE_V2 = "quarantine_v2"
    HARDENED = "hardened"
    HARDENED_BALANCED = "hardened_balanced"
    HARDENED_STRONG = "hardened_strong"
    GATE_ONLY = "gate_only"
    QUANTUM_ONLY = "quantum_only"
    HARDENED_V2 = "hardened_v2"


# -------------------------
# NEW: Intrusion Detector
# -------------------------

@dataclass
class IntrusionDetector:
    """
    Detects quantum channel attacks via QBER monitoring.
    """
    qber_threshold: float = 0.025  # Alert if avg QBER exceeds this
    window_size_s: int = 30
    min_samples: int = 3
    
    _history: Dict[Tuple[str, str], List[Tuple[int, float]]] = field(default_factory=dict)
    alerts: List[Dict] = field(default_factory=list)
    
    def record_qber(self, t_s: int, edge: Tuple[str, str], qber: float) -> Optional[Dict]:
        if edge not in self._history:
            self._history[edge] = []
        
        self._history[edge].append((t_s, qber))
        
        cutoff = t_s - self.window_size_s
        self._history[edge] = [(t, q) for t, q in self._history[edge] if t >= cutoff]
        
        recent = [q for _, q in self._history[edge]]
        if len(recent) >= self.min_samples:
            avg_qber = sum(recent) / len(recent)
            if avg_qber > self.qber_threshold:
                alert = {
                    "t_s": t_s,
                    "edge": f"{edge[0]}-{edge[1]}",
                    "avg_qber": avg_qber,
                    "threshold": self.qber_threshold,
                    "samples": len(recent),
                }
                self.alerts.append(alert)
                return alert
        return None
    
    def has_recent_alert(self, t_s: int, lookback_s: int = 60) -> bool:
        return any(t_s - a["t_s"] < lookback_s for a in self.alerts)
    
    def get_recent_alerts(self, t_s: int, lookback_s: int = 60) -> List[Dict]:
        return [a for a in self.alerts if t_s - a["t_s"] < lookback_s]


# -------------------------
# NEW: Command Signature Tracker
# -------------------------

@dataclass
class SignatureTracker:
    """
    Tracks command signatures to detect repeated/replay attacks.
    """
    cooldown_s: int = 30
    max_repetitions: int = 2
    
    _recent: Dict[str, List[int]] = field(default_factory=dict)  # signature -> list of times
    
    def _compute_signature(self, msg: Message, action: ControlAction) -> str:
        """Compute a signature for a command based on src, dst, action, params."""
        data = f"{msg.src}:{msg.dst}:{action.action_type.value}:{action.target_shed_frac}:{action.duration_s}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def is_repeated(self, t_s: int, msg: Message, action: ControlAction) -> bool:
        sig = self._compute_signature(msg, action)
        
        # Clean old entries
        cutoff = t_s - self.cooldown_s
        if sig in self._recent:
            self._recent[sig] = [t for t in self._recent[sig] if t >= cutoff]
        
        # Check if repeated -- also check action-type level (not just exact params)
        if sig in self._recent and len(self._recent[sig]) >= self.max_repetitions:
            return True
        
        # Record this command
        if sig not in self._recent:
            self._recent[sig] = []
        self._recent[sig].append(t_s)
        
        return False


# -------------------------
# NEW: Per-Source Rate Limiter
# -------------------------

@dataclass
class PerSourceRateLimiter:
    """
    Track message rates per (src, dst) pair instead of a single global counter.

    This prevents an attacker from consuming the global rate budget and causing
    legitimate messages to be rate-limited alongside attack traffic.
    """
    max_rate_per_s: float = 2.0
    window_s: int = 10
    burst_multiplier: float = 3.0
    burst_window_s: int = 2

    _counters: Dict[str, List[int]] = field(default_factory=dict)

    def _key(self, src: str, dst: str) -> str:
        return f"{src}->{dst}"

    def check_and_record(self, t_s: int, src: str, dst: str) -> bool:
        """Returns True if rate limit is exceeded (should block)."""
        k = self._key(src, dst)
        if k not in self._counters:
            self._counters[k] = []

        cutoff = t_s - self.window_s
        self._counters[k] = [t for t in self._counters[k] if t >= cutoff]

        count = len(self._counters[k])
        allowed_in_window = int(self.max_rate_per_s * self.window_s)

        burst_cutoff = t_s - self.burst_window_s
        burst_count = sum(1 for t in self._counters[k] if t >= burst_cutoff)
        burst_allowed = int(self.max_rate_per_s * self.burst_multiplier * self.burst_window_s)

        self._counters[k].append(t_s)
        return count >= allowed_in_window or burst_count >= burst_allowed


# -------------------------
# NEW: Global Source Rate Limiter
# -------------------------

@dataclass
class SourceRateLimiter:
    """
    Track message rates per source across all destinations.

    This catches fan-out floods that evade per-(src,dst) limits by rotating
    destinations.
    """
    max_rate_per_s: float = 2.0
    window_s: int = 10
    burst_multiplier: float = 3.0
    burst_window_s: int = 2

    _counters: Dict[str, List[int]] = field(default_factory=dict)

    def check_and_record(self, t_s: int, src: str) -> bool:
        s = str(src)
        if s not in self._counters:
            self._counters[s] = []

        cutoff = t_s - self.window_s
        self._counters[s] = [t for t in self._counters[s] if t >= cutoff]

        count = len(self._counters[s])
        allowed_in_window = int(self.max_rate_per_s * self.window_s)

        burst_cutoff = t_s - self.burst_window_s
        burst_count = sum(1 for t in self._counters[s] if t >= burst_cutoff)
        burst_allowed = int(self.max_rate_per_s * self.burst_multiplier * self.burst_window_s)

        self._counters[s].append(t_s)
        return count >= allowed_in_window or burst_count >= burst_allowed


# -------------------------
# NEW: Behavioral Plausibility Checker
# -------------------------

@dataclass
class PlausibilityChecker:
    """
    Validate whether PRIORITY_ACTION commands are plausible for current grid state.

    V11 TUNED: Relaxed thresholds to reduce false positive rate from ~12% to <3%.
    Key changes:
      - Raised healthy_shed_threshold 0.30 → 0.40 (legitimate protective shedding
        can reach 35% during fast frequency transients)
      - Raised healthy_deficit_max 0.05 → 0.10 (renewable variability causes
        deficit_ratio to fluctuate 3-8% even in normal operation)
      - Tightened excessive_step: added minimum step size of 0.35 (attack uses 0.55)
      - Added frequency-aware gating: if frequency is within ±1 Hz of nominal,
        high-shed commands are more suspicious
    """
    max_shed_step: float = 0.35          # was 0.25 — too tight for island recovery
    healthy_shed_threshold: float = 0.40  # was 0.30 — legitimate shed can reach 0.35
    healthy_deficit_max: float = 0.10     # was 0.05 — renewable variability ±8%

    _cmd_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=lambda: {
        "checked": 0,
        "blocked_excessive_step": 0,
        "blocked_implausible": 0,
        "blocked_frequency_mismatch": 0,
        "passed": 0,
    })

    def check(
        self,
        *,
        msg: Message,
        action: ControlAction,
        current_shed: float,
        deficit_ratio: float,
        control_quality: float,
        frequency_hz: float = 60.0,   # V11: frequency-aware gating
    ) -> Tuple[bool, str]:
        """Returns (is_plausible, reason)."""
        self.stats["checked"] += 1

        if action.action_type != ActionType.SHED_LOAD_EMERGENCY:
            self.stats["passed"] += 1
            return True, "non_shed_action"

        requested = float(action.target_shed_frac) if action.target_shed_frac is not None else 0.0

        # Check 1: Excessive step from near-zero shedding
        step = requested - current_shed
        if step > self.max_shed_step and current_shed < 0.10:
            # Only block if conditions are clearly healthy (high quality + low deficit)
            if control_quality > 0.6 and deficit_ratio < 0.15:
                self.stats["blocked_excessive_step"] += 1
                return False, f"excessive_step({step:.2f}_from_{current_shed:.2f})"

        # Check 2: High shed when grid is healthy (core implausibility check)
        if (
            requested >= self.healthy_shed_threshold
            and deficit_ratio < self.healthy_deficit_max
            and control_quality > 0.7
        ):
            self.stats["blocked_implausible"] += 1
            return False, f"implausible(shed={requested:.2f}_deficit={deficit_ratio:.2f})"

        # Check 3: Frequency-mismatch detection (V11)
        # If frequency is very close to nominal (±0.5 Hz) but someone requests
        # high emergency shed (>=0.45), this is suspicious — real emergencies
        # show frequency deviation first.
        freq_deviation = abs(frequency_hz - 60.0)
        if (
            requested >= 0.45
            and freq_deviation < 0.5
            and deficit_ratio < 0.08
        ):
            self.stats["blocked_frequency_mismatch"] += 1
            return False, f"freq_mismatch(shed={requested:.2f}_freq_dev={freq_deviation:.2f}Hz)"

        dst = msg.dst
        if dst not in self._cmd_history:
            self._cmd_history[dst] = []
        self._cmd_history[dst].append({
            "t_ms": int(msg.created_ms),
            "shed": requested,
            "src": msg.src,
        })
        self._cmd_history[dst] = self._cmd_history[dst][-20:]

        self.stats["passed"] += 1
        return True, "plausible"


# -------------------------
# NEW: Cross-Node Attack Correlator
# -------------------------

@dataclass
class CrossNodeCorrelator:
    """
    Detect coordinated attacks by observing one source targeting many nodes quickly.
    """
    correlation_window_s: int = 30
    max_simultaneous_targets: int = 1
    include_control_setpoint: bool = False

    _recent_priority_actions: List[Dict[str, Any]] = field(default_factory=list)
    _blocked_sources: Dict[str, int] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=lambda: {
        "correlated_attacks_detected": 0,
        "messages_blocked_by_correlation": 0,
    })

    def record_and_check(self, t_s: int, msg: Message) -> Tuple[bool, str]:
        """Returns (is_suspicious, reason). True means block."""
        consider = (msg.msg_type == MsgType.PRIORITY_ACTION) or (
            self.include_control_setpoint and msg.msg_type == MsgType.CONTROL_SETPOINT
        )
        if not consider:
            return False, ""

        if msg.src in self._blocked_sources:
            if t_s < self._blocked_sources[msg.src]:
                self.stats["messages_blocked_by_correlation"] += 1
                return True, "correlated_attack_source"
            del self._blocked_sources[msg.src]

        self._recent_priority_actions.append({"t_s": t_s, "src": msg.src, "dst": msg.dst})
        cutoff = t_s - self.correlation_window_s
        self._recent_priority_actions = [r for r in self._recent_priority_actions if r["t_s"] >= cutoff]

        src_targets = set(r["dst"] for r in self._recent_priority_actions if r["src"] == msg.src)
        if len(src_targets) > self.max_simultaneous_targets:
            self._blocked_sources[msg.src] = t_s + self.correlation_window_s
            self.stats["correlated_attacks_detected"] += 1
            return True, f"coordinated_attack({len(src_targets)}_targets)"

        return False, ""


# -------------------------
# NEW: Quarantine Manager
# -------------------------

@dataclass
class QuarantineManager:
    """
    Safe-mode isolation: block external PRIORITY_ACTION while allowing CONTROL_SETPOINT.
    """
    duration_s: int = 120
    cooldown_s: int = 60

    _quarantine_until: Dict[str, int] = field(default_factory=dict)
    _last_trigger: Dict[str, int] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=lambda: {
        "quarantines_activated": 0,
        "messages_blocked_quarantine": 0,
        "legitimate_allowed_during_quarantine": 0,
    })

    def trigger(self, t_s: int, node: str) -> bool:
        if node in self._last_trigger and (t_s - self._last_trigger[node]) < self.cooldown_s:
            return False
        self._quarantine_until[node] = t_s + self.duration_s
        self._last_trigger[node] = t_s
        self.stats["quarantines_activated"] += 1
        return True

    def is_quarantined(self, t_s: int, node: str) -> bool:
        if node not in self._quarantine_until:
            return False
        if t_s >= self._quarantine_until[node]:
            del self._quarantine_until[node]
            return False
        return True

    def check_message(self, t_s: int, msg: Message) -> Tuple[bool, str]:
        """Returns (should_block, reason)."""
        if not self.is_quarantined(t_s, msg.dst):
            return False, ""

        if msg.msg_type == MsgType.CONTROL_SETPOINT:
            self.stats["legitimate_allowed_during_quarantine"] += 1
            return False, ""

        if msg.msg_type == MsgType.PRIORITY_ACTION:
            self.stats["messages_blocked_quarantine"] += 1
            return True, "quarantine_active"

        return False, ""


# -------------------------
# Observation tap
# -------------------------

@dataclass
class Observation:
    t_ms: int
    src: str
    dst: str
    msg_type: str
    size_bytes: int
    requires_anon: bool = False
    obs_channel: str = ""
    link_u: str = ""
    link_v: str = ""


class ObservationScope(str, Enum):
    """Attacker observability model for deanonymization."""
    GLOBAL = "global"
    NODE_TAP = "node_tap"
    EDGE_TAP = "edge_tap"


@dataclass
class AttackerObservationConfig:
    """Configuration for what traffic the deanonymization attacker can see."""
    scope: ObservationScope = ObservationScope.GLOBAL
    tap_node: Optional[str] = None
    tap_edge: Optional[Tuple[str, str]] = None


EmitFn = Callable[[Message], None]


def make_emit_with_observer(
    *,
    base_emit: EmitFn,
    analyzer: Optional["TrafficAnalyzer"] = None,
    observe_created_events: bool = True,
) -> EmitFn:
    def _emit(msg: Message) -> None:
        if analyzer is not None and bool(observe_created_events):
            # For anonymized channels we intentionally mask observables to avoid
            # leaking real-vs-cover signatures to the passive observer.
            if bool(getattr(msg, "requires_anon", False)):
                obs_type = "anon_meta"
                obs_size = 200
                obs_channel = "anon"
            else:
                obs_type = msg.msg_type.value if hasattr(msg.msg_type, "value") else str(msg.msg_type)
                obs_size = int(msg.size_bytes)
                obs_channel = obs_type
            analyzer.observe(Observation(
                t_ms=int(msg.created_ms),
                src=msg.src,
                dst=msg.dst,
                msg_type=obs_type,
                size_bytes=obs_size,
                requires_anon=bool(getattr(msg, "requires_anon", False)),
                obs_channel=obs_channel,
                link_u="",
                link_v="",
            ))
        base_emit(msg)
    return _emit


@dataclass
class CoverTrafficTracker:
    """Track QAN cover traffic costs."""
    cover_messages_sent: int = 0
    cover_bytes_sent: int = 0
    real_qan_messages_sent: int = 0
    real_qan_bytes_sent: int = 0
    energy_per_byte_j: float = 5e-6
    cover_timeseries: List[Dict[str, Any]] = field(default_factory=list)

    def record_cover(self, t_s: int, msg: Message) -> None:
        self.cover_messages_sent += 1
        self.cover_bytes_sent += int(getattr(msg, 'size_bytes', 0))
        self.cover_timeseries.append({
            "t_s": int(t_s),
            "type": "cover",
            "bytes": int(getattr(msg, 'size_bytes', 0)),
        })

    def record_real_qan(self, t_s: int, msg: Message) -> None:
        self.real_qan_messages_sent += 1
        self.real_qan_bytes_sent += int(getattr(msg, 'size_bytes', 0))
        self.cover_timeseries.append({
            "t_s": int(t_s),
            "type": "real_qan",
            "bytes": int(getattr(msg, 'size_bytes', 0)),
        })

    def get_summary(self) -> Dict[str, Any]:
        total_qan_bytes = self.cover_bytes_sent + self.real_qan_bytes_sent
        cover_energy_kwh = (self.cover_bytes_sent * self.energy_per_byte_j) / 3.6e6
        return {
            "cover_messages_total": self.cover_messages_sent,
            "cover_bytes_total": self.cover_bytes_sent,
            "real_qan_messages_total": self.real_qan_messages_sent,
            "real_qan_bytes_total": self.real_qan_bytes_sent,
            "cover_overhead_ratio": self.cover_bytes_sent / max(1, total_qan_bytes),
            "cover_energy_kwh": cover_energy_kwh,
            "cover_messages_per_real_event": self.cover_messages_sent / max(1, self.real_qan_messages_sent),
        }

    def write_csv(self, path: str) -> None:
        if not self.cover_timeseries:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["t_s", "type", "bytes"])
            writer.writeheader()
            writer.writerows(self.cover_timeseries)


# -------------------------
# GHZ Resource Tracker (for Quantum Anonymous Broadcast)
# -------------------------

@dataclass
class GHZResourceTracker:
    """Track GHZ state consumption for quantum anonymous broadcast protocol."""
    ghz_states_prepared: int = 0
    ghz_states_consumed: int = 0
    ghz_states_failed: int = 0
    ghz_states_decoherent: int = 0
    total_rounds_attempted: int = 0
    total_rounds_succeeded: int = 0
    collisions_detected: int = 0
    real_qab_messages_sent: int = 0
    timeseries: List[Dict[str, Any]] = field(default_factory=list)

    def record_ghz_round(self, t_s: int, success: bool, decoherent: bool = False) -> None:
        self.total_rounds_attempted += 1
        self.ghz_states_prepared += 1
        if decoherent:
            self.ghz_states_decoherent += 1
        elif success:
            self.total_rounds_succeeded += 1
            self.ghz_states_consumed += 1
        else:
            self.ghz_states_failed += 1
        self.timeseries.append({
            "t_s": int(t_s), "type": "ghz_round",
            "success": success, "decoherent": decoherent,
        })

    def record_collision(self, t_s: int) -> None:
        self.collisions_detected += 1
        self.timeseries.append({"t_s": int(t_s), "type": "collision"})

    def record_real_qab(self, t_s: int, msg: Message) -> None:
        self.real_qab_messages_sent += 1
        self.timeseries.append({
            "t_s": int(t_s), "type": "real_qab",
            "bytes": int(getattr(msg, 'size_bytes', 0)),
        })

    def get_summary(self) -> Dict[str, Any]:
        """Return summary compatible with CoverTrafficTracker keys + GHZ-specific."""
        return {
            # CoverTrafficTracker-compatible keys (zero for QAB — no cover traffic)
            "cover_messages_total": 0,
            "cover_bytes_total": 0,
            "real_qan_messages_total": self.real_qab_messages_sent,
            "real_qan_bytes_total": 0,
            "cover_overhead_ratio": 0.0,
            "cover_energy_kwh": 0.0,
            "cover_messages_per_real_event": 0.0,
            # GHZ-specific metrics
            "ghz_states_consumed": self.ghz_states_consumed,
            "ghz_states_failed": self.ghz_states_failed,
            "ghz_states_decoherent": self.ghz_states_decoherent,
            "ghz_states_prepared": self.ghz_states_prepared,
            "ghz_round_success_rate": (
                self.total_rounds_succeeded / max(1, self.total_rounds_attempted)
            ),
            "ghz_collision_rate": (
                self.collisions_detected / max(1, self.total_rounds_attempted)
            ),
            "ghz_resource_cost": self.ghz_states_prepared,
        }

    def write_csv(self, path: str) -> None:
        if not self.timeseries:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        keys = ["t_s", "type", "success", "decoherent", "bytes"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.timeseries)


# -------------------------
# Traffic analysis adversary
# -------------------------

@dataclass
class QANEventSpec:
    sender_candidates: List[str]
    receiver: str
    t_event_s: int
    window_s: int = 5
    mixing_delay_ms: int = 40
    cover_rate_per_s: float = 3.0


class TrafficAnalyzer:
    def __init__(
        self,
        sender_candidates: List[str],
        *,
        softmax_temperature: float = 0.80,
        # With strong anonymity defenses, posterior mass is near-uniform.
        # Keep abstain thresholds low so calibration can be measured instead of
        # collapsing to all-unknown predictions.
        abstain_top1_prob: float = 0.10,
        abstain_margin: float = 0.0,
        min_window_obs: int = 1,
        weak_evidence_obs: int = 3,
        weak_prior_weight: float = 0.05,
        observation_cfg: Optional[AttackerObservationConfig] = None,
    ):
        self.sender_candidates = list(sender_candidates)
        self._obs: List[Observation] = []
        self._event: Optional[QANEventSpec] = None
        self.softmax_temperature = max(1e-6, float(softmax_temperature))
        self.abstain_top1_prob = clamp(float(abstain_top1_prob), 0.0, 1.0)
        self.abstain_margin = clamp(float(abstain_margin), 0.0, 1.0)
        self.min_window_obs = max(0, int(min_window_obs))
        self.weak_evidence_obs = max(1, int(weak_evidence_obs))
        self.weak_prior_weight = clamp(float(weak_prior_weight), 0.0, 1.0)
        self.observation_cfg = observation_cfg or AttackerObservationConfig()
        if self.observation_cfg.tap_edge is not None:
            a, b = self.observation_cfg.tap_edge
            self.observation_cfg.tap_edge = tuple(sorted((str(a), str(b))))

        # Sender-feature weights are intentionally modest to avoid overconfident
        # posteriors while still exposing a capability gap across attacker scopes.
        self._feature_weights = {
            "count": 0.60,
            "offset": 0.55,
            "near": 0.75,
            "burst": 0.25,
        }
        # Lower observability should reduce effective signal from the same
        # metadata and increase regularization toward a uniform prior.
        self._scope_signal_gain = {
            ObservationScope.GLOBAL: 1.00,
            ObservationScope.NODE_TAP: 0.65,
            ObservationScope.EDGE_TAP: 0.45,
        }
        self._scope_prior_blend = {
            ObservationScope.GLOBAL: 0.55,
            ObservationScope.NODE_TAP: 0.72,
            ObservationScope.EDGE_TAP: 0.82,
        }

    def _is_observable(self, obs: Observation) -> bool:
        cfg = self.observation_cfg
        scope = cfg.scope
        if scope == ObservationScope.GLOBAL:
            return True

        # Localized attacker relies on link-level taps.
        if not obs.link_u or not obs.link_v:
            return False

        if scope == ObservationScope.NODE_TAP:
            if not cfg.tap_node:
                return False
            return (obs.link_u == cfg.tap_node) or (obs.link_v == cfg.tap_node)

        if scope == ObservationScope.EDGE_TAP:
            if not cfg.tap_edge:
                return False
            return tuple(sorted((obs.link_u, obs.link_v))) == cfg.tap_edge

        return True

    def _blend_uniform_prior(self, posterior: Dict[str, float], weight: float) -> Dict[str, float]:
        if not posterior:
            return {}
        n = len(posterior)
        if n <= 0:
            return {}
        w = clamp(float(weight), 0.0, 1.0)
        if w <= 0.0:
            return posterior
        u = 1.0 / n
        out = {k: (1.0 - w) * float(v) + w * u for k, v in posterior.items()}
        z = sum(out.values())
        return {k: (v / z if z > 0 else u) for k, v in out.items()}

    @staticmethod
    def _zscore_map(values: Dict[str, float]) -> Dict[str, float]:
        if not values:
            return {}
        keys = list(values.keys())
        arr = [float(values[k]) for k in keys]
        mu = sum(arr) / len(arr)
        var = sum((x - mu) ** 2 for x in arr) / max(1, len(arr))
        sd = max(1e-6, math.sqrt(var))
        return {k: (float(values[k]) - mu) / sd for k in keys}

    def arm_event(self, event: QANEventSpec, *, clear_obs: bool = False) -> None:
        """
        Arm the analyzer for a specific QAN event.

        By default we keep the observation history so post-run inference can
        still access messages emitted during the event window.
        """
        self._event = event
        if clear_obs:
            self._obs.clear()

    def observe(self, obs: Observation) -> None:
        self._obs.append(obs)

    def infer_sender(self) -> Dict[str, Any]:
        if self._event is None:
            n = len(self.sender_candidates) or 1
            top1 = self.sender_candidates[0] if self.sender_candidates else ""
            return {
                "posterior": {s: 1.0 / n for s in self.sender_candidates},
                "top1": top1,
                "top1_candidate": top1,
                "top1_prob": 1.0 / n,
                "top2_prob": 1.0 / n if n > 1 else 0.0,
                "top1_margin": 0.0,
                "abstained": False,
                "n_obs_window": 0,
                "prior_blend_weight": 0.0,
                "entropy_bits": math.log2(n) if n > 1 else 0,
                "features": {},
            }

        e = self._event
        t0_ms = e.t_event_s * 1000
        lo = (e.t_event_s - e.window_s) * 1000
        hi = (e.t_event_s + e.window_s) * 1000

        window_obs = [
            o for o in self._obs
            if lo <= o.t_ms <= hi
            and o.dst == e.receiver
            and bool(getattr(o, "requires_anon", False))
            and self._is_observable(o)
        ]

        # Receiver is not a valid sender candidate for the event.
        candidate_senders = [s for s in e.sender_candidates if s != e.receiver]
        if not candidate_senders:
            candidate_senders = list(e.sender_candidates)

        feats: Dict[str, Dict[str, float]] = {}
        for s in candidate_senders:
            obs_s = [o for o in window_obs if o.src == s]
            count = float(len(obs_s))

            if obs_s:
                closest_offset_ms = float(min(abs(o.t_ms - t0_ms) for o in obs_s))
            else:
                closest_offset_ms = float(e.window_s * 1000 + 1)

            mid = t0_ms
            early_ct = float(sum(1 for o in obs_s if o.t_ms <= mid))
            late_ct = float(sum(1 for o in obs_s if o.t_ms > mid))
            burst = (early_ct - late_ct) / (count + 1e-9) if count > 0 else 0.0

            near_event_ms = 500
            near_event_ct = float(sum(1 for o in obs_s if abs(o.t_ms - t0_ms) <= near_event_ms))
            feats[s] = {
                "count": count,
                "closest_offset_ms": closest_offset_ms,
                "near_event_ct": near_event_ct,
                "burst": burst,
            }

        count_map = {k: float(v["count"]) for k, v in feats.items()}
        offset_map = {k: float(v["closest_offset_ms"]) for k, v in feats.items()}
        near_map = {k: float(v["near_event_ct"]) for k, v in feats.items()}
        burst_map = {k: float(v["burst"]) for k, v in feats.items()}

        z_count = self._zscore_map(count_map)
        z_offset = self._zscore_map(offset_map)
        z_near = self._zscore_map(near_map)
        z_burst = self._zscore_map(burst_map)

        scope = self.observation_cfg.scope
        signal_gain = self._scope_signal_gain.get(scope, 0.65)
        anon_damp = 0.65 if e.window_s > 0 else 1.0
        w_count = self._feature_weights["count"] * signal_gain * anon_damp
        w_offset = self._feature_weights["offset"] * signal_gain * anon_damp
        w_near = self._feature_weights["near"] * signal_gain * anon_damp
        w_burst = self._feature_weights["burst"] * signal_gain * anon_damp

        scores: Dict[str, float] = {}
        for s in feats.keys():
            scores[s] = (
                w_count * z_count.get(s, 0.0)
                - w_offset * z_offset.get(s, 0.0)
                + w_near * z_near.get(s, 0.0)
                + w_burst * z_burst.get(s, 0.0)
            )

        effective_temp = self.softmax_temperature / max(0.25, signal_gain)
        post = softmax(scores, temperature=effective_temp)

        n_obs_window = len(window_obs)
        anon_share = (
            sum(1 for o in window_obs if bool(getattr(o, "requires_anon", False))) / n_obs_window
            if n_obs_window > 0 else 0.0
        )
        n_candidates = max(1, len(candidate_senders))
        seen_candidates = sum(1 for s in candidate_senders if feats.get(s, {}).get("count", 0.0) > 0.0)
        coverage = seen_candidates / n_candidates

        base_blend = self._scope_prior_blend.get(scope, 0.72)
        anon_blend = 0.15 if anon_share > 0.80 else 0.05
        coverage_blend = (1.0 - coverage) * 0.35
        weak_blend = 0.0
        if n_obs_window < self.weak_evidence_obs:
            evidence_frac = n_obs_window / max(1, self.weak_evidence_obs)
            weak_blend = self.weak_prior_weight * (1.0 - evidence_frac)
        blend_w = clamp(base_blend + anon_blend + coverage_blend + weak_blend, 0.0, 0.95)
        post = self._blend_uniform_prior(post, blend_w)

        if post:
            ranked = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
            top1_candidate = ranked[0][0]
            top1_prob = float(ranked[0][1])
            top2_prob = float(ranked[1][1]) if len(ranked) > 1 else 0.0

            # Avoid fixed sender bias when posteriors are effectively tied.
            # Deterministic lexicographic MAP creates artificial over-accuracy.
            band_eps = 0.01
            top_band = [k for (k, p) in ranked if (top1_prob - float(p)) <= band_eps]
            if len(top_band) > 1:
                key = f"{e.t_event_s}:{e.receiver}:{n_obs_window}:{len(top_band)}"
                h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
                top1_candidate = sorted(top_band)[h % len(top_band)]
                top1_prob = float(post.get(top1_candidate, top1_prob))
                vals = sorted((float(v) for v in post.values()), reverse=True)
                top2_prob = vals[1] if len(vals) > 1 else 0.0
        else:
            top1_candidate = ""
            top1_prob = 0.0
            top2_prob = 0.0

        top1_margin = max(0.0, top1_prob - top2_prob)
        abstained = (
            (n_obs_window < self.min_window_obs)
            or (top1_prob < self.abstain_top1_prob)
            or (top1_margin < self.abstain_margin)
        )
        top1 = "unknown" if abstained else top1_candidate

        return {
            "posterior": post,
            "top1": top1,
            "top1_candidate": top1_candidate,
            "top1_prob": top1_prob,
            "top2_prob": top2_prob,
            "top1_margin": top1_margin,
            "abstained": abstained,
            "n_obs_window": n_obs_window,
            "anon_share": anon_share,
            "observability_coverage": coverage,
            "scope_signal_gain": signal_gain,
            "prior_blend_weight": blend_w,
            "entropy_bits": float(entropy_bits(post)),
            "features": feats,
        }


# -------------------------
# QAN Orchestrator
# -------------------------

MsgIdFn = Callable[[], int]


@dataclass
class QANConfig:
    cover_rate_per_s: float = 3.0
    window_s: int = 5
    mixing_delay_ms: int = 40
    sync_burst_per_candidate: bool = True
    sync_burst_jitter_ms: int = 10
    indistinguishable_meta: bool = True
    qan_size_bytes: int = 220
    cover_size_bytes: int = 180
    qan_deadline_ms: int = 250
    cover_deadline_ms: int = 400
    # Optional key-cost model: allow QAN/cover traffic to consume auth keys.
    auth_real_notify: bool = False
    auth_cover: bool = False
    auth_sync_burst: bool = False


@dataclass
class QABConfig:
    """
    Configuration for GHZ-based Quantum Anonymous Broadcast protocol.

    Unlike classical QAN (cover-traffic mixing), QAB uses N-party GHZ
    entangled states to implement a quantum DC-net.  Each anonymous bit
    is transmitted by having all participants measure their share of a
    GHZ state; the XOR of outcomes reveals only the sender's bit.

    Anonymity is information-theoretic: even N-2 colluding participants
    cannot identify the sender (vs. classical cover traffic which is
    defeated by a global passive adversary).

    References:
        Christandl & Wehner, "Quantum Anonymous Transmissions" (2005)
        Broadbent & Tapp, "Information-Theoretic Security Without an
            Honest Majority" (2007)
    """
    # GHZ state preparation parameters
    ghz_prep_success_prob: float = 0.85      # Success probability per GHZ attempt
    ghz_fidelity_base: float = 0.95          # Base fidelity for 2-party GHZ
    ghz_fidelity_decay_per_node: float = 0.02  # Fidelity reduction per additional party
    ghz_fidelity_threshold: float = 0.50     # Below this, entanglement is useless

    # Protocol parameters
    bits_per_round: int = 1                  # Anonymous bits per GHZ round
    message_bits: int = 256                  # Total payload to transmit anonymously
    decoherence_window_ms: int = 100         # Quantum memory lifetime for GHZ state
    ghz_prep_time_ms: int = 5               # Time to prepare one N-party GHZ state
    collision_detection: bool = True         # Detect >1 simultaneous sender

    # Wire format (matches classical QAN for observer compatibility)
    qab_size_bytes: int = 220
    qab_deadline_ms: int = 250

    # Optional authentication of the final notification
    auth_real_notify: bool = False


class QANOrchestrator:
    def __init__(self, env: simpy.Environment, rng: random.Random, cfg: QANConfig, 
                 msg_id_fn: MsgIdFn, emit_fn: EmitFn,
                 cover_tracker: Optional[CoverTrafficTracker] = None):
        self.env = env
        self.rng = rng
        self.cfg = cfg
        self.msg_id_fn = msg_id_fn
        self.emit_fn = emit_fn
        self.cover_tracker = cover_tracker

    def _wire_profile(self, *, is_real: bool) -> Tuple[int, int, int]:
        # When enabled, both real/cover notifications look identical on wire.
        if self.cfg.indistinguishable_meta:
            return (int(self.cfg.qan_size_bytes), int(self.cfg.qan_deadline_ms), 1)
        if is_real:
            return (int(self.cfg.qan_size_bytes), int(self.cfg.qan_deadline_ms), 1)
        return (int(self.cfg.cover_size_bytes), int(self.cfg.cover_deadline_ms), 0)

    def schedule_event(self, *, true_sender: str, candidates: List[str], 
                       receiver: str, t_event_s: int) -> QANEventSpec:
        sender_candidates = [c for c in candidates if c != receiver]
        if true_sender not in sender_candidates:
            sender_candidates.append(true_sender)
        if not sender_candidates:
            sender_candidates = [true_sender]

        spec = QANEventSpec(
            sender_candidates=list(sender_candidates),
            receiver=receiver,
            t_event_s=int(t_event_s),
            window_s=int(self.cfg.window_s),
            mixing_delay_ms=int(self.cfg.mixing_delay_ms),
            cover_rate_per_s=float(self.cfg.cover_rate_per_s),
        )
        self.env.process(self._run_event(true_sender=true_sender, spec=spec))
        return spec

    def _run_event(self, *, true_sender: str, spec: QANEventSpec):
        start_s = spec.t_event_s - spec.window_s
        if start_s > int(self.env.now):
            yield self.env.timeout(start_s - int(self.env.now))

        if self.cfg.sync_burst_per_candidate:
            self.env.process(self._emit_sync_burst(spec))

        # Emit covers and real QAN notify concurrently so the true notify stays
        # centered around the event time instead of drifting to window end.
        self.env.process(self._emit_covers(spec))
        self.env.process(self._emit_real_notify(true_sender=true_sender, spec=spec))

    def _emit_sync_burst(self, spec: QANEventSpec):
        now_s = int(self.env.now)
        if spec.t_event_s > now_s:
            yield self.env.timeout(spec.t_event_s - now_s)

        for sender in spec.sender_candidates:
            jitter_ms = self.rng.uniform(
                -float(self.cfg.sync_burst_jitter_ms),
                float(self.cfg.sync_burst_jitter_ms),
            )
            if jitter_ms > 0:
                yield self.env.timeout(jitter_ms / 1000.0)

            size, deadline, priority = self._wire_profile(is_real=False)
            msg = Message(
                msg_id=self.msg_id_fn(),
                created_ms=int(self.env.now * 1000),
                src=sender,
                dst=spec.receiver,
                msg_type=MsgType.COVER,
                priority=priority,
                deadline_ms=deadline,
                size_bytes=size,
                requires_auth=bool(self.cfg.auth_sync_burst),
                requires_anon=True,
                is_attack=False,
                payload={"qan_event_t_s": spec.t_event_s, "sync_burst": True},
            )
            self.emit_fn(msg)
            if self.cover_tracker is not None:
                self.cover_tracker.record_cover(int(self.env.now), msg)

            if jitter_ms < 0:
                yield self.env.timeout(abs(jitter_ms) / 1000.0)

    def _emit_covers(self, spec: QANEventSpec):
        total_duration_s = 2 * spec.window_s + 1
        n_cover = int(max(0.0, round(spec.cover_rate_per_s * total_duration_s)))

        offsets = sorted(self.rng.uniform(0.0, float(total_duration_s)) for _ in range(n_cover))
        prev = 0.0
        for off in offsets:
            yield self.env.timeout(max(0.0, off - prev))
            prev = off

            src = self.rng.choice(spec.sender_candidates)
            mix_s = self.rng.uniform(0.0, spec.mixing_delay_ms / 1000.0)
            if mix_s > 0:
                yield self.env.timeout(mix_s)

            size, deadline, priority = self._wire_profile(is_real=False)
            msg = Message(
                msg_id=self.msg_id_fn(),
                created_ms=int(self.env.now * 1000),
                src=src,
                dst=spec.receiver,
                msg_type=MsgType.COVER,
                priority=priority,
                deadline_ms=deadline,
                size_bytes=size,
                requires_auth=bool(self.cfg.auth_cover),
                requires_anon=True,
                is_attack=False,
                payload={"qan_event_t_s": spec.t_event_s},
            )
            self.emit_fn(msg)
            if self.cover_tracker is not None:
                self.cover_tracker.record_cover(int(self.env.now), msg)

    def _emit_real_notify(self, *, true_sender: str, spec: QANEventSpec):
        now_s = int(self.env.now)
        if spec.t_event_s > now_s:
            yield self.env.timeout(spec.t_event_s - now_s)

        mix_s = self.rng.uniform(0.0, spec.mixing_delay_ms / 1000.0)
        if mix_s > 0:
            yield self.env.timeout(mix_s)

        size, deadline, priority = self._wire_profile(is_real=True)
        qan = Message(
            msg_id=self.msg_id_fn(),
            created_ms=int(self.env.now * 1000),
            src=true_sender,
            dst=spec.receiver,
            msg_type=MsgType.QAN_NOTIFY,
            priority=priority,
            deadline_ms=deadline,
            size_bytes=size,
            requires_auth=bool(self.cfg.auth_real_notify),
            requires_anon=True,
            is_attack=False,
            payload={"qan_event_t_s": spec.t_event_s, "qan_true_sender": true_sender},
        )
        self.emit_fn(qan)
        if self.cover_tracker is not None:
            self.cover_tracker.record_real_qan(int(self.env.now), qan)


# -------------------------
# Quantum Anonymous Broadcast (GHZ-based DC-net)
# -------------------------

class QuantumAnonymousBroadcast:
    """
    GHZ-based quantum anonymous broadcast protocol.

    Implements an abstracted quantum DC-net using N-party GHZ entangled
    states.  Each anonymous notification requires `message_bits` GHZ
    rounds; each round:
      1. Prepare N-party GHZ state (probabilistic success).
      2. All participants measure in agreed basis.
      3. XOR of measurement outcomes reveals the sender's bit.
      4. Collision detection flags multi-sender interference.

    Unlike QANOrchestrator, this class emits **zero cover traffic**.
    Anonymity is information-theoretic from the quantum protocol, not
    statistical from traffic mixing.

    API matches QANOrchestrator for drop-in compatibility with
    TrafficAnalyzer and the finalmain.py wiring.
    """

    def __init__(
        self,
        env: "simpy.Environment",
        rng: random.Random,
        cfg: QABConfig,
        msg_id_fn,
        emit_fn,
        n_participants: int,
        ghz_tracker: Optional[GHZResourceTracker] = None,
    ):
        self.env = env
        self.rng = rng
        self.cfg = cfg
        self.msg_id_fn = msg_id_fn
        self.emit_fn = emit_fn
        self.n_participants = max(2, n_participants)
        self.ghz_tracker = ghz_tracker or GHZResourceTracker()
        self.stats = {
            "events_scheduled": 0,
            "events_completed": 0,
            "events_failed": 0,
        }

    # ── Public API (matches QANOrchestrator) ─────────────────────────

    def schedule_event(
        self,
        *,
        true_sender: str,
        candidates: List[str],
        receiver: str,
        t_event_s: int,
    ) -> QANEventSpec:
        """Schedule a quantum anonymous broadcast event."""
        spec = QANEventSpec(
            sender_candidates=list(candidates),
            receiver=receiver,
            t_event_s=t_event_s,
            window_s=0,            # No cover window needed
            mixing_delay_ms=0,     # No mixing delay
            cover_rate_per_s=0.0,  # No cover traffic
        )
        self.stats["events_scheduled"] += 1
        self.env.process(self._run_event(true_sender=true_sender, spec=spec))
        return spec

    # ── GHZ Physics Model ────────────────────────────────────────────

    def _ghz_fidelity(self, n: int) -> float:
        """
        N-party GHZ state fidelity.

        F(N) = F_base * (1 - decay)^(N-2)

        For N=2 (Bell pair): F = F_base.
        Each additional party reduces fidelity due to gate errors in
        the fan-out circuit (CNOT chain from source qubit).
        """
        return self.cfg.ghz_fidelity_base * (
            (1.0 - self.cfg.ghz_fidelity_decay_per_node) ** max(0, n - 2)
        )

    def _attempt_ghz_round(self) -> Tuple[bool, bool]:
        """
        Attempt one GHZ DC-net round.

        Returns (success, decoherent):
          - success=True: GHZ state prepared, measurements valid
          - decoherent=True: state decohered before all measurements complete
        """
        # Step 1: GHZ state preparation (probabilistic)
        if self.rng.random() > self.cfg.ghz_prep_success_prob:
            return False, False

        # Step 2: Fidelity check — below threshold means entanglement is
        # too degraded for reliable anonymous transmission
        fidelity = self._ghz_fidelity(self.n_participants)
        # Add small random noise to fidelity
        fidelity += self.rng.gauss(0, 0.01)
        if fidelity < self.cfg.ghz_fidelity_threshold:
            return False, False

        # Step 3: Decoherence check — quantum memory must hold long enough
        # for all N parties to measure.  Model as exponential decay.
        measurement_time_ms = self.n_participants * 0.5  # 0.5 ms per party
        decay_prob = 1.0 - math.exp(-measurement_time_ms / self.cfg.decoherence_window_ms)
        if self.rng.random() < decay_prob:
            return False, True  # Decoherent

        return True, False

    # ── Event Execution ──────────────────────────────────────────────

    def _run_event(self, *, true_sender: str, spec: QANEventSpec):
        """SimPy process: execute quantum anonymous broadcast rounds."""
        # Wait until scheduled event time
        if spec.t_event_s > self.env.now:
            yield self.env.timeout(spec.t_event_s - self.env.now)

        rounds_needed = max(1, self.cfg.message_bits // max(1, self.cfg.bits_per_round))
        rounds_completed = 0
        max_attempts = rounds_needed * 5  # Give up after 5× retries
        attempts = 0

        while rounds_completed < rounds_needed and attempts < max_attempts:
            attempts += 1

            # GHZ state preparation time
            prep_time_s = self.cfg.ghz_prep_time_ms / 1000.0
            yield self.env.timeout(prep_time_s)

            success, decoherent = self._attempt_ghz_round()
            t_now = int(self.env.now)
            self.ghz_tracker.record_ghz_round(t_now, success, decoherent)

            if success:
                rounds_completed += self.cfg.bits_per_round
            # Failed rounds are simply retried (standard for GHZ protocols)

        if rounds_completed >= rounds_needed:
            self.stats["events_completed"] += 1
            self._emit_qab_notify(true_sender=true_sender, spec=spec)
        else:
            self.stats["events_failed"] += 1

    def _emit_qab_notify(self, *, true_sender: str, spec: QANEventSpec):
        """Emit the anonymous notification after all GHZ rounds succeed."""
        msg = Message(
            msg_id=self.msg_id_fn(),
            created_ms=int(self.env.now * 1000),
            src=true_sender,
            dst=spec.receiver,
            msg_type=MsgType.QAN_NOTIFY,
            priority=2,
            deadline_ms=self.cfg.qab_deadline_ms,
            size_bytes=self.cfg.qab_size_bytes,
            requires_auth=bool(self.cfg.auth_real_notify),
            requires_anon=True,
            is_attack=False,
            payload={
                "qan_event_t_s": spec.t_event_s,
                "qan_true_sender": true_sender,
                "qab_protocol": "ghz_dcnet",
                "ghz_rounds_used": self.cfg.message_bits,
            },
        )
        self.emit_fn(msg)
        self.ghz_tracker.record_real_qab(int(self.env.now), msg)


# -------------------------
# Attacks (unchanged from original)
# -------------------------

@dataclass
class SpoofConfig:
    spoof_size_bytes: int = 260
    spoof_deadline_ms: int = 220
    use_islanding: bool = False
    forced_shed_frac: float = 0.75
    harm_duration_s: int = 45
    # Probability that the attacker can forge/steal authentication credentials.
    # Models attacker sophistication:
    #   0.0  = naive attacker (no signature knowledge)
    #   0.15 = moderate attacker (partial insider knowledge / implementation exploit)
    #   0.40 = advanced attacker (classical HMAC compromise, side-channel attack)
    #   1.0  = perfect forgery (insider-equivalent)
    # When forge succeeds, message bypasses signature check and must be caught
    # by behavioral defenses (plausibility, rate limiting, quarantine, IDS).
    auth_bypass_prob: float = 0.0
    # Whether the attacker attempts to forge quantum control tokens.
    # When True, the attacker generates a fake HMAC token (random hex) that
    # will fail verification at the receiver.  This models a realistic
    # adversary who knows the protocol format but lacks the QKD-derived
    # HMAC secret.
    attempt_quantum_token_forge: bool = True


class SpoofingAttack:
    def __init__(self, env: simpy.Environment, rng: random.Random, cfg: SpoofConfig, 
                 msg_id_fn: MsgIdFn, emit_fn: EmitFn):
        self.env = env
        self.rng = rng
        self.cfg = cfg
        self.msg_id_fn = msg_id_fn
        self.emit_fn = emit_fn
        self.cover_tracker = None

    def schedule_spoof(self, *, t_spoof_s: int, controller: str, victim: str, 
                       inferred_sender: str, label: str = "priority_spoof") -> None:
        self.env.process(self._run(t_spoof_s=t_spoof_s, controller=controller, 
                                   victim=victim, inferred_sender=inferred_sender, label=label))

    def _run(self, *, t_spoof_s: int, controller: str, victim: str, inferred_sender: str, label: str):
        now_s = int(self.env.now)
        if t_spoof_s > now_s:
            yield self.env.timeout(t_spoof_s - now_s)

        # ── Attacker sophistication: probabilistic auth bypass ──
        # With auth_bypass_prob, the attacker may forge/steal valid credentials.
        # This models real-world scenarios where attackers have partial insider
        # knowledge, exploit implementation side-channels, or compromise
        # classical HMAC keys.
        auth_forged = (
            self.cfg.auth_bypass_prob > 0
            and self.rng.random() < self.cfg.auth_bypass_prob
        )

        if self.cfg.use_islanding:
            payload = {"action": "island_now"}
        elif auth_forged:
            # Sophisticated attacker: when auth is forged, craft plausible
            # commands that evade behavioral checks.  Use small shed fraction
            # (below plausibility threshold of 0.25) to avoid detection.
            # Still harmful: even 10-15% unnecessary shed causes grid instability.
            subtle_shed = self.rng.uniform(0.08, 0.20)
            payload = {
                "action": "shed_load_emergency",
                "forced_shed_frac": round(subtle_shed, 3),
                "harm_duration_s": int(self.cfg.harm_duration_s),
            }
        else:
            # Naive attacker: no auth bypass, sends aggressive commands.
            # Will be caught by signature check anyway.
            payload = {
                "action": "shed_load_emergency",
                "forced_shed_frac": float(self.cfg.forced_shed_frac),
                "harm_duration_s": int(self.cfg.harm_duration_s),
            }

        payload.update({"attack": True, "attack_label": label, "inferred_sender": inferred_sender})

        if auth_forged:
            payload["control_signature"] = "quam_ctrl_v1"
            payload["control_sender_role"] = "controller"
            payload["auth_forged"] = True  # Track for analysis

        # ── Quantum token forgery attempt ──
        # A realistic attacker who knows the protocol format will generate
        # a fake quantum control token (random hex HMAC).  Without the
        # QKD-derived secret, this token will fail HMAC verification at the
        # receiver's quantum layer — but the attacker still tries.
        if self.cfg.attempt_quantum_token_forge:
            fake_nonce = self.rng.getrandbits(64)
            fake_expiry_ms = int(self.env.now * 1000) + 2000
            fake_token = f"{self.rng.getrandbits(128):032x}"  # random 32-char hex
            payload["nonce"] = fake_nonce
            payload["quantum_control_expiry_ms"] = fake_expiry_ms
            payload["quantum_control_token"] = fake_token
            payload["quantum_token_forged"] = True  # tracking only

        msg = Message(
            msg_id=self.msg_id_fn(),
            created_ms=int(self.env.now * 1000),
            src=controller,
            dst=victim,
            msg_type=MsgType.PRIORITY_ACTION,
            priority=2,
            deadline_ms=self.cfg.spoof_deadline_ms,
            size_bytes=self.cfg.spoof_size_bytes,
            requires_auth=True,
            requires_anon=False,
            is_attack=True,
            attack_label=label,
            payload=payload,
        )
        self.emit_fn(msg)


@dataclass
class ExhaustConfig:
    start_s: int = 210
    end_s: int = 320
    rate_per_s: float = 3.0
    size_bytes: int = 240
    deadline_ms: int = 350
    label: str = "key_exhaust"



class ExhaustTargetStrategy(str, Enum):
    """Targeting strategies for key exhaustion attacks."""
    UNIFORM = "uniform"
    BOTTLENECK = "bottleneck"
    BRIDGE = "bridge"
    HIGH_TRAFFIC = "high_traffic"
    SINGLE_LINK = "single_link"
    BYPASSABLE = "bypassable"
    STAR_CENTER = "star_center"
    DEANON_GUIDED = "deanon_guided"


@dataclass
class TargetedExhaustConfig(ExhaustConfig):
    """Configuration for targeted key exhaustion."""
    target_strategy: ExhaustTargetStrategy = ExhaustTargetStrategy.UNIFORM
    target_edges: Optional[List[Tuple[str, str]]] = None
    target_nodes: Optional[List[str]] = None
    focus_ratio: float = 0.8
    adapt_to_defense: bool = False


class TargetedKeyExhaustionAttack:
    """Key exhaustion attack with topology-aware targeting."""

    def __init__(
        self,
        env: simpy.Environment,
        rng: random.Random,
        cfg: TargetedExhaustConfig,
        msg_id_fn: MsgIdFn,
        emit_fn: EmitFn,
        topology: nx.Graph,
        traffic_observer: Optional["TrafficAnalyzer"] = None,
    ):
        self.env = env
        self.rng = rng
        self.cfg = cfg
        self.msg_id_fn = msg_id_fn
        self.emit_fn = emit_fn
        self.topology = topology
        self.traffic_observer = traffic_observer
        self._all_edges = list(topology.edges()) if topology is not None else []
        self._primary_targets = self._identify_targets()

        self.stats = {
            "messages_sent": 0,
            "targeted_messages": 0,
            "decoy_messages": 0,
        }

    def _identify_targets(self) -> List[Tuple[str, str]]:
        if self.cfg.target_edges:
            return list(self.cfg.target_edges)

        if self.cfg.target_nodes:
            targets = [
                (u, v) for (u, v) in self._all_edges
                if u in self.cfg.target_nodes or v in self.cfg.target_nodes
            ]
            return targets or list(self._all_edges)

        strategy = self.cfg.target_strategy
        if strategy == ExhaustTargetStrategy.UNIFORM:
            return list(self._all_edges)

        if not self._all_edges:
            return []

        if strategy == ExhaustTargetStrategy.BOTTLENECK:
            bc = nx.edge_betweenness_centrality(self.topology)
            ranked = sorted(bc.items(), key=lambda x: -x[1])
            n_targets = max(2, len(ranked) // 5)
            return [e for e, _ in ranked[:n_targets]]

        if strategy == ExhaustTargetStrategy.BRIDGE:
            bridges = list(nx.bridges(self.topology))
            if bridges:
                return bridges
            bc = nx.edge_betweenness_centrality(self.topology)
            ranked = sorted(bc.items(), key=lambda x: -x[1])
            n_targets = max(2, len(ranked) // 5)
            return [e for e, _ in ranked[:n_targets]]

        if strategy == ExhaustTargetStrategy.HIGH_TRAFFIC and self.traffic_observer:
            return self._identify_high_traffic_edges()

        if strategy == ExhaustTargetStrategy.SINGLE_LINK:
            bc = nx.edge_betweenness_centrality(self.topology)
            if bc:
                return [max(bc.items(), key=lambda x: x[1])[0]]
            return [self._all_edges[0]]

        if strategy == ExhaustTargetStrategy.BYPASSABLE:
            # Choose a single edge that is (a) used by deterministic shortest-path routing for
            # many src/dst pairs but (b) has at least one equal-cost shortest-path bypass.
            #
            # This is specifically useful for validating ECMP boundary conditions:
            # - If no equal-cost bypass exists, ECMP cannot avoid the attacked resource.
            # - If a bypass exists for some flows, ECMP can probabilistically steer traffic away,
            #   reducing contention/key usage on the attacked edge.
            picked = self._pick_bypassable_edge()
            return [picked] if picked is not None else [self._all_edges[0]]

        if strategy == ExhaustTargetStrategy.STAR_CENTER:
            degrees = dict(self.topology.degree())
            if degrees:
                hub = max(degrees.items(), key=lambda x: x[1])[0]
                return [(hub, n) for n in self.topology.neighbors(hub)]

        return list(self._all_edges)

    def _identify_high_traffic_edges(self) -> List[Tuple[str, str]]:
        if not self.traffic_observer:
            return list(self._all_edges)
        edge_counts: Dict[Tuple[str, str], int] = {}
        for obs in getattr(self.traffic_observer, "_obs", []):
            ek = tuple(sorted((obs.src, obs.dst)))
            edge_counts[ek] = edge_counts.get(ek, 0) + 1
        if not edge_counts:
            return list(self._all_edges)
        ranked = sorted(edge_counts.items(), key=lambda x: -x[1])
        n_targets = max(2, len(ranked) // 3)
        return [e for e, _ in ranked[:n_targets]]

    def _pick_bypassable_edge(self) -> Optional[Tuple[str, str]]:
        """
        Pick an edge whose use is avoidable under ECMP without increasing hop count.

        Score an edge by counting ordered src/dst pairs where:
        - deterministic routing (nx.shortest_path) uses the edge, AND
        - there exists an alternative shortest path that avoids the edge.

        This intentionally approximates the boundary condition where ECMP can help:
        ECMP only has leverage when multiple equal-cost shortest paths exist.
        """
        if self.topology is None or not self._all_edges:
            return None

        nodes = list(self.topology.nodes())
        if len(nodes) < 2:
            return None

        # For large graphs, bound work by sampling pairs.
        if len(nodes) <= 40:
            pairs = [(s, t) for s in nodes for t in nodes if s != t]
        else:
            pairs = []
            # 500 pairs is enough to identify a "good" bypassable edge while keeping runtime small.
            while len(pairs) < 500:
                s = self.rng.choice(nodes)
                t = self.rng.choice(nodes)
                if s != t:
                    pairs.append((s, t))

        scores: Dict[Tuple[str, str], int] = {}
        for s, t in pairs:
            try:
                det_path = nx.shortest_path(self.topology, s, t)
            except nx.NetworkXNoPath:
                continue

            det_edges = [
                tuple(sorted((det_path[i], det_path[i + 1])))
                for i in range(len(det_path) - 1)
            ]
            if not det_edges:
                continue

            # Enumerate all shortest paths for small graphs; sample a few for large graphs.
            try:
                if len(nodes) <= 30:
                    shortest_paths = list(nx.all_shortest_paths(self.topology, s, t))
                else:
                    # Sample up to 8 shortest paths (enough to detect bypass existence).
                    shortest_paths = []
                    gen = nx.all_shortest_paths(self.topology, s, t)
                    for _ in range(8):
                        try:
                            shortest_paths.append(next(gen))
                        except StopIteration:
                            break
            except nx.NetworkXNoPath:
                continue

            if len(shortest_paths) <= 1:
                continue

            # Precompute edge-sets for each shortest path.
            path_edge_sets = []
            for p in shortest_paths:
                es = {
                    tuple(sorted((p[i], p[i + 1])))
                    for i in range(len(p) - 1)
                }
                path_edge_sets.append(es)

            # For each edge used by deterministic routing, see if any equal-cost path avoids it.
            for e in det_edges:
                avoidable = any(e not in es for es in path_edge_sets)
                if avoidable:
                    scores[e] = scores.get(e, 0) + 1

        if not scores:
            return None

        # Tie-breaker: prefer higher edge betweenness (more "impact") if scores match.
        bc: Dict[Tuple[str, str], float] = {}
        try:
            bc_raw = nx.edge_betweenness_centrality(self.topology)
            # Normalize undirected edge keys for consistent lookup.
            bc = {tuple(sorted(e)): float(v) for e, v in bc_raw.items()}
        except Exception:
            bc = {}

        best_edge = max(
            scores.items(),
            key=lambda kv: (kv[1], float(bc.get(kv[0], 0.0))),
        )[0]

        return best_edge

    def _pick_target_edge(self) -> Optional[Tuple[str, str]]:
        if not self._all_edges:
            return None
        focus = max(0.0, min(1.0, float(self.cfg.focus_ratio)))
        if self._primary_targets and self.rng.random() < focus:
            self.stats["targeted_messages"] += 1
            return self.rng.choice(self._primary_targets)
        self.stats["decoy_messages"] += 1
        return self.rng.choice(self._all_edges)

    def schedule(self) -> None:
        self.env.process(self._run())

    def _run(self):
        now_s = int(self.env.now)
        if self.cfg.start_s > now_s:
            yield self.env.timeout(self.cfg.start_s - now_s)

        if self.cfg.end_s <= self.cfg.start_s or self.cfg.rate_per_s <= 0:
            return

        inter = 1.0 / float(self.cfg.rate_per_s)
        while int(self.env.now) <= int(self.cfg.end_s):
            edge = self._pick_target_edge()
            if edge is None:
                return
            u, v = edge
            if self.rng.random() < 0.5:
                src, dst = u, v
            else:
                src, dst = v, u

            msg = Message(
                msg_id=self.msg_id_fn(),
                created_ms=int(self.env.now * 1000),
                src=src,
                dst=dst,
                msg_type=MsgType.CONTROL_SETPOINT,
                priority=1,
                deadline_ms=self.cfg.deadline_ms,
                size_bytes=self.cfg.size_bytes,
                requires_auth=True,
                requires_anon=False,
                is_attack=True,
                attack_label=self.cfg.label,
                payload={
                    "attack": True,
                    "attack_label": self.cfg.label,
                    "target_strategy": self.cfg.target_strategy.value,
                    "shed_frac_target": 0.0,
                },
            )
            self.emit_fn(msg)
            self.stats["messages_sent"] += 1

            jitter = self.rng.uniform(-0.2 * inter, 0.2 * inter)
            yield self.env.timeout(max(0.001, inter + jitter))

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "strategy": self.cfg.target_strategy.value,
            "primary_targets": [f"{u}-{v}" for u, v in self._primary_targets],
            "focus_ratio": float(self.cfg.focus_ratio),
        }


class KeyExhaustionAttack:
    def __init__(self, env: simpy.Environment, rng: random.Random, cfg: ExhaustConfig, 
                 msg_id_fn: MsgIdFn, emit_fn: EmitFn):
        self.env = env
        self.rng = rng
        self.cfg = cfg
        self.msg_id_fn = msg_id_fn
        self.emit_fn = emit_fn
        self.cover_tracker = None

    def schedule(self, *, src_nodes: List[str], dst_nodes: List[str]) -> None:
        self.env.process(self._run(src_nodes=src_nodes, dst_nodes=dst_nodes))

    def _run(self, *, src_nodes: List[str], dst_nodes: List[str]):
        now_s = int(self.env.now)
        if self.cfg.start_s > now_s:
            yield self.env.timeout(self.cfg.start_s - now_s)

        if self.cfg.end_s <= self.cfg.start_s or self.cfg.rate_per_s <= 0:
            return

        inter = 1.0 / float(self.cfg.rate_per_s)
        while int(self.env.now) <= int(self.cfg.end_s):
            src = self.rng.choice(src_nodes)
            dst = self.rng.choice(dst_nodes)
            if src == dst:
                dst = self.rng.choice([d for d in dst_nodes if d != src] or dst_nodes)

            msg = Message(
                msg_id=self.msg_id_fn(),
                created_ms=int(self.env.now * 1000),
                src=src,
                dst=dst,
                msg_type=MsgType.CONTROL_SETPOINT,
                priority=1,
                deadline_ms=self.cfg.deadline_ms,
                size_bytes=self.cfg.size_bytes,
                requires_auth=True,
                requires_anon=False,
                is_attack=True,
                attack_label=self.cfg.label,
                payload={"attack": True, "attack_label": self.cfg.label, "shed_frac_target": 0.0},
            )
            self.emit_fn(msg)
            if self.cover_tracker is not None:
                self.cover_tracker.record_cover(int(self.env.now), msg)

            jitter = self.rng.uniform(-0.2 * inter, 0.2 * inter)
            yield self.env.timeout(max(0.001, inter + jitter))


@dataclass
class QuantumDisturbConfig:
    start_s: int = 220
    end_s: int = 300
    mode: str = "delta"
    delta_qber: float = 0.06
    absolute_qber: float = 0.25
    target_edges: Optional[List[Tuple[str, str]]] = None
    label: str = "quantum_disturb"


class QuantumDisturbanceAttack:
    def __init__(self, env: simpy.Environment, rng: random.Random, cfg: QuantumDisturbConfig, 
                 qlayer: QuantumAugmentation):
        self.env = env
        self.rng = rng
        self.cfg = cfg
        self.qlayer = qlayer

    def schedule(self, *, candidate_edges: List[Tuple[str, str]]) -> None:
        self.env.process(self._run(candidate_edges=candidate_edges))

    def _run(self, *, candidate_edges: List[Tuple[str, str]]):
        now_s = int(self.env.now)
        if self.cfg.start_s > now_s:
            yield self.env.timeout(self.cfg.start_s - now_s)

        edges = list(self.cfg.target_edges) if self.cfg.target_edges else list(candidate_edges)
        if not edges:
            return

        if self.cfg.target_edges is None:
            self.rng.shuffle(edges)
            edges = edges[: min(2, len(edges))]

        for (u, v) in edges:
            ek = edge_key(u, v)
            if self.cfg.mode == "absolute":
                w = QBERWindow(start_s=self.cfg.start_s, end_s=self.cfg.end_s, 
                              absolute_qber=float(self.cfg.absolute_qber), label=self.cfg.label)
            else:
                w = QBERWindow(start_s=self.cfg.start_s, end_s=self.cfg.end_s, 
                              delta_qber=float(self.cfg.delta_qber), label=self.cfg.label)
            self.qlayer.add_qber_window(ek, w)


# -------------------------
# EXTENDED: Policy gate config
# -------------------------

@dataclass
class GateConfig:
    """
    Extended defense configuration with ALL strategies.
    
    Default degraded_secret_fraction is moderate (0.50).
    """
    # Basic verification
    verification_delay_ms: int = 0
    
    # Quantum-aware degraded mode with hysteresis
    degraded_secret_fraction: float = 0.50  # enter threshold (moderate default)
    degraded_recover_secret_fraction: float = 0.60  # exit threshold
    degraded_recover_hold_s: int = 60  # require healthy SF for this long before exit
    degraded_verification_delay_ms: int = 120
    block_priority_in_degraded: bool = False
    
    # Oracle (for upper-bound analysis)
    oracle_block_attacks: bool = False
    
    # Staleness
    enforce_staleness: bool = True
    
    # Rate limiting
    auth_rate_limit_per_s: float = 0.0  # 0 = disabled
    rate_limit_window_s: int = 5
    
    # NEW: Intrusion detection
    block_during_intrusion: bool = False
    intrusion_lookback_s: int = 60
    # IMPROVED: Selective intrusion blocking (only block PRIORITY_ACTION, not all auth)
    intrusion_selective: bool = False
    
    # NEW: Adaptive rate limiting
    adaptive_rate_limit: bool = False
    normal_rate_limit_per_s: float = 10.0
    degraded_rate_limit_per_s: float = 2.0
    
    # NEW: Signature-based blocking
    block_repeated_commands: bool = False
    command_cooldown_s: int = 30
    max_command_repetitions: int = 2
    
    # NEW: Quarantine mode
    enable_quarantine: bool = False
    quarantine_duration_s: int = 60

    # V11: Hardware timing jitter model
    # Models realistic crypto-hardware processing time variance from:
    #   - FPGA clock domain crossing jitter (±2-5 ms)
    #   - HSM busy-wait under concurrent requests (0-15 ms)
    #   - SPD (Single Photon Detector) dead-time effects on key lookup
    # Jitter is additive: actual_delay = base_delay + Uniform(-jitter, +jitter)
    hw_timing_jitter_ms: float = 0.0  # 0 = deterministic (original behavior)
    # SPD dark-count rate effect on verification confidence
    # Higher dark counts → more conservative verification → slightly longer delay
    spd_dark_count_rate_hz: float = 100.0  # typical InGaAs SPD
    spd_timing_overhead_ms: float = 0.0    # extra delay from dark-count mitigation

    # ---- IMPROVED DEFENSE OPTIONS ----

    # Per-source rate limiting (replaces/supplements global rate limit)
    enable_per_source_rate_limit: bool = False
    per_source_max_rate_per_s: float = 2.0
    per_source_window_s: int = 10
    per_source_burst_multiplier: float = 3.0

    # Behavioral plausibility checking
    enable_plausibility_check: bool = False
    plausibility_max_shed_step: float = 0.35          # V11: relaxed from 0.25
    plausibility_healthy_shed_threshold: float = 0.40  # V11: relaxed from 0.30
    plausibility_healthy_deficit_max: float = 0.10     # V11: relaxed from 0.05

    # Cross-node attack correlation
    enable_cross_node_correlation: bool = False
    correlation_window_s: int = 30
    max_simultaneous_targets: int = 1

    # Actual quarantine manager
    enable_quarantine_manager: bool = False
    quarantine_manager_duration_s: int = 120
    quarantine_manager_cooldown_s: int = 60

    # Control-plane authz (realistic ACL + lightweight app signature)
    enable_control_acl: bool = False
    allowed_control_sources: Tuple[str, ...] = ()
    require_control_signature: bool = False
    control_signature_value: str = "quam_ctrl_v1"

    # Quantum control-token enforcement (QTLS-bound OTP from QuantumAugmentation)
    require_quantum_control_token: bool = False

    # Source-level control flood guard (across all destinations)
    enable_source_global_rate_limit: bool = False
    source_global_max_rate_per_s: float = 1.0
    source_global_window_s: int = 10
    source_global_burst_multiplier: float = 2.0

    # Correlation scope extension
    correlation_include_control_setpoint: bool = False


# -------------------------
# NEW: Pre-key admission gate
# -------------------------

class AdmissionGate:
    """
    Lightweight pre-key admission gate.

    Runs before QKD key consumption to stop abusive traffic from burning key
    pools. Keep checks metadata-focused and cheap:
    - rate limiting (global/per-source)
    - intrusion-aware selective blocking
    - coordinated multi-target correlation
    - quarantine guard
    - degraded-mode PRIORITY_ACTION block
    """

    def __init__(
        self,
        cfg: GateConfig,
        intrusion_detector: Optional[IntrusionDetector] = None,
    ):
        self.cfg = cfg
        self.intrusion_detector = intrusion_detector
        self._control_acl_sources = {str(s) for s in (cfg.allowed_control_sources or ())}

        self._win_start_s: Optional[int] = None
        self._auth_count_in_window: int = 0

        self._degraded_active: bool = False
        self._degraded_recover_start_s: Optional[int] = None

        self.per_source_limiter = PerSourceRateLimiter(
            max_rate_per_s=cfg.per_source_max_rate_per_s,
            window_s=cfg.per_source_window_s,
            burst_multiplier=cfg.per_source_burst_multiplier,
        ) if cfg.enable_per_source_rate_limit else None

        self.source_global_limiter = SourceRateLimiter(
            max_rate_per_s=cfg.source_global_max_rate_per_s,
            window_s=cfg.source_global_window_s,
            burst_multiplier=cfg.source_global_burst_multiplier,
        ) if cfg.enable_source_global_rate_limit else None

        self.cross_node_correlator = CrossNodeCorrelator(
            correlation_window_s=cfg.correlation_window_s,
            max_simultaneous_targets=cfg.max_simultaneous_targets,
            include_control_setpoint=cfg.correlation_include_control_setpoint,
        ) if cfg.enable_cross_node_correlation else None

        self.quarantine_mgr = QuarantineManager(
            duration_s=cfg.quarantine_manager_duration_s,
            cooldown_s=cfg.quarantine_manager_cooldown_s,
        ) if cfg.enable_quarantine_manager else None

        self.stats = {
            "checked_total": 0,
            "allowed_total": 0,
            "blocked_total": 0,
            "blocked_prekey_rate_limit": 0,
            "blocked_prekey_per_source_rate": 0,
            "blocked_prekey_intrusion": 0,
            "blocked_prekey_cross_node": 0,
            "blocked_prekey_quarantine_mgr": 0,
            "blocked_prekey_degraded": 0,
            "blocked_prekey_acl": 0,
            "blocked_prekey_signature": 0,
            "blocked_prekey_quantum_token": 0,
            "blocked_prekey_source_global_rate": 0,
            "degraded_mode_triggers": 0,
            "degraded_mode_recovers": 0,
        }

    @staticmethod
    def _is_authenticated_control(msg: Message) -> bool:
        return bool(
            msg.requires_auth
            and msg.msg_type in (MsgType.CONTROL_SETPOINT, MsgType.PRIORITY_ACTION)
        )

    @staticmethod
    def _is_authenticated_priority(msg: Message) -> bool:
        return bool(msg.requires_auth and msg.msg_type == MsgType.PRIORITY_ACTION)

    def _get_effective_rate_limit(self, degraded_active: bool) -> float:
        if not self.cfg.adaptive_rate_limit:
            return self.cfg.auth_rate_limit_per_s
        if degraded_active:
            return self.cfg.degraded_rate_limit_per_s
        return self.cfg.normal_rate_limit_per_s

    def _update_degraded_mode(self, *, now_s: int, sf: Optional[float], requires_auth: bool) -> bool:
        if not requires_auth:
            return self._degraded_active
        if sf is None:
            return self._degraded_active

        enter_sf = float(self.cfg.degraded_secret_fraction)
        exit_sf = max(enter_sf, float(self.cfg.degraded_recover_secret_fraction))
        hold_s = max(1, int(self.cfg.degraded_recover_hold_s))

        if not self._degraded_active:
            if sf < enter_sf:
                self._degraded_active = True
                self._degraded_recover_start_s = None
                self.stats["degraded_mode_triggers"] += 1
            return self._degraded_active

        if sf > exit_sf:
            if self._degraded_recover_start_s is None:
                self._degraded_recover_start_s = now_s
            elif (now_s - self._degraded_recover_start_s) >= hold_s:
                self._degraded_active = False
                self._degraded_recover_start_s = None
                self.stats["degraded_mode_recovers"] += 1
        else:
            self._degraded_recover_start_s = None

        return self._degraded_active

    def _rate_limit_hit(self, now_s: int, msg: Message, degraded_active: bool) -> bool:
        effective_limit = self._get_effective_rate_limit(degraded_active)
        if effective_limit <= 0 or not msg.requires_auth:
            return False

        win = max(1, int(self.cfg.rate_limit_window_s))
        if self._win_start_s is None or now_s - self._win_start_s >= win:
            self._win_start_s = now_s
            self._auth_count_in_window = 0

        self._auth_count_in_window += 1
        allowed = int(effective_limit * win)
        return self._auth_count_in_window > allowed

    def decide(
        self,
        env: simpy.Environment,
        msg: Message,
        path_nodes: Optional[List[str]] = None,
    ) -> Tuple[str, int]:
        """
        Return (decision, delay_ms). decision in {"allow", "block"}.
        """
        now_s = int(env.now)
        msg.payload = msg.payload or {}
        self.stats["checked_total"] += 1
        is_control_auth = self._is_authenticated_control(msg)
        is_priority_auth = self._is_authenticated_priority(msg)

        sf = None
        try:
            sf = float(msg.payload.get("secret_fraction_path_mean"))
        except Exception:
            sf = None
        degraded_active = self._update_degraded_mode(now_s=now_s, sf=sf, requires_auth=msg.requires_auth)

        # 0) Control-plane authorization checks (ACL + app signature)
        if is_priority_auth and self.cfg.enable_control_acl and self._control_acl_sources:
            if str(msg.src) not in self._control_acl_sources:
                self.stats["blocked_total"] += 1
                self.stats["blocked_prekey_acl"] += 1
                msg.payload["prekey_gate_reason"] = "prekey_control_acl_source"
                msg.payload["prekey_gate_decision"] = "block"
                return "block", 0

        if is_priority_auth and self.cfg.require_control_signature:
            sig = str((msg.payload or {}).get("control_signature", ""))
            if sig != str(self.cfg.control_signature_value):
                self.stats["blocked_total"] += 1
                self.stats["blocked_prekey_signature"] += 1
                msg.payload["prekey_gate_reason"] = "prekey_control_signature_invalid"
                msg.payload["prekey_gate_decision"] = "block"
                return "block", 0

        # Quantum control-token enforcement (QTLS-bound OTP)
        if is_priority_auth and self.cfg.require_quantum_control_token:
            valid = int((msg.payload or {}).get("quantum_control_token_valid", 0) or 0)
            if valid != 1:
                self.stats["blocked_total"] += 1
                self.stats["blocked_prekey_quantum_token"] += 1
                msg.payload["prekey_gate_reason"] = str(
                    (msg.payload or {}).get("quantum_control_token_reason",
                                           "prekey_quantum_control_token_invalid")
                )
                msg.payload["prekey_gate_decision"] = "block"
                return "block", 0

        if (
            is_control_auth
            and self.source_global_limiter is not None
            and self.source_global_limiter.check_and_record(now_s, str(msg.src))
        ):
            self.stats["blocked_total"] += 1
            self.stats["blocked_prekey_source_global_rate"] += 1
            msg.payload["prekey_gate_reason"] = "prekey_source_global_rate_limited"
            msg.payload["prekey_gate_decision"] = "block"
            return "block", 0

        # 1) Quarantine manager
        if self.quarantine_mgr is not None:
            should_block, reason = self.quarantine_mgr.check_message(now_s, msg)
            if should_block:
                self.stats["blocked_total"] += 1
                self.stats["blocked_prekey_quarantine_mgr"] += 1
                msg.payload["prekey_gate_reason"] = reason
                msg.payload["prekey_gate_decision"] = "block"
                return "block", 0

        # 2) Cross-node coordinated attack correlation
        if self.cross_node_correlator is not None:
            # Avoid penalizing designated controllers that legitimately fan out
            # control-setpoints to many nodes.
            acl_known_src = bool(self._control_acl_sources and str(msg.src) in self._control_acl_sources)
            if not acl_known_src:
                is_suspicious, reason = self.cross_node_correlator.record_and_check(now_s, msg)
                if is_suspicious:
                    self.stats["blocked_total"] += 1
                    self.stats["blocked_prekey_cross_node"] += 1
                    if self.quarantine_mgr is not None:
                        self.quarantine_mgr.trigger(now_s, msg.dst)
                    msg.payload["prekey_gate_reason"] = reason
                    msg.payload["prekey_gate_decision"] = "block"
                    return "block", 0

        # 3) Per-source rate limiting
        if (
            self.per_source_limiter is not None
            and is_control_auth
            and self.per_source_limiter.check_and_record(now_s, msg.src, msg.dst)
        ):
            self.stats["blocked_total"] += 1
            self.stats["blocked_prekey_per_source_rate"] += 1
            msg.payload["prekey_gate_reason"] = "prekey_per_source_rate_limited"
            msg.payload["prekey_gate_decision"] = "block"
            return "block", 0

        # 4) Global/adaptive rate limiting
        if is_control_auth and self._rate_limit_hit(now_s, msg, degraded_active):
            self.stats["blocked_total"] += 1
            self.stats["blocked_prekey_rate_limit"] += 1
            msg.payload["prekey_gate_reason"] = "prekey_auth_rate_limited"
            msg.payload["prekey_gate_decision"] = "block"
            return "block", 0

        # 5) Intrusion selective/full blocking
        if (
            self.cfg.block_during_intrusion
            and self.intrusion_detector is not None
            and self.intrusion_detector.has_recent_alert(now_s, self.cfg.intrusion_lookback_s)
        ):
            if self.cfg.intrusion_selective:
                if msg.msg_type == MsgType.PRIORITY_ACTION and msg.requires_auth:
                    self.stats["blocked_total"] += 1
                    self.stats["blocked_prekey_intrusion"] += 1
                    if self.quarantine_mgr is not None:
                        self.quarantine_mgr.trigger(now_s, msg.dst)
                    msg.payload["prekey_gate_reason"] = "prekey_intrusion_alert_selective"
                    msg.payload["prekey_gate_decision"] = "block"
                    return "block", 0
            else:
                if msg.requires_auth:
                    self.stats["blocked_total"] += 1
                    self.stats["blocked_prekey_intrusion"] += 1
                    msg.payload["prekey_gate_reason"] = "prekey_intrusion_alert_active"
                    msg.payload["prekey_gate_decision"] = "block"
                    return "block", 0

        # 6) Degraded-mode PRIORITY_ACTION block
        if (
            degraded_active
            and msg.requires_auth
            and msg.msg_type == MsgType.PRIORITY_ACTION
            and self.cfg.block_priority_in_degraded
        ):
            self.stats["blocked_total"] += 1
            self.stats["blocked_prekey_degraded"] += 1
            if self.quarantine_mgr is not None:
                self.quarantine_mgr.trigger(now_s, msg.dst)
            msg.payload["prekey_gate_reason"] = "prekey_degraded_secret_fraction"
            msg.payload["prekey_gate_decision"] = "block"
            return "block", 0

        self.stats["allowed_total"] += 1
        msg.payload["prekey_gate_reason"] = "prekey_allow"
        msg.payload["prekey_gate_decision"] = "allow"
        return "allow", 0


# -------------------------
# EXTENDED: Policy gate
# -------------------------

class PolicyGate:
    """
    Extended PolicyGate with all defense strategies and statistics tracking.

    IMPROVED: Added per-source rate limiting, plausibility checking,
    cross-node correlation, selective intrusion response, and actual quarantine.
    """
    
    def __init__(self, cfg: GateConfig,
                 intrusion_detector: Optional[IntrusionDetector] = None,
                 microgrids: Optional[Dict[str, Any]] = None,
                 rng: Optional[random.Random] = None):
        self.cfg = cfg
        self.intrusion_detector = intrusion_detector
        self._microgrids = microgrids or {}
        self._hw_rng = rng or random.Random(42)  # for hardware timing jitter
        self._control_acl_sources = {str(s) for s in (cfg.allowed_control_sources or ())}
        self.signature_tracker = SignatureTracker(
            cooldown_s=cfg.command_cooldown_s,
            max_repetitions=cfg.max_command_repetitions,
        ) if cfg.block_repeated_commands else None
        
        # Rate limiting state
        self._win_start_s: Optional[int] = None
        self._auth_count_in_window: int = 0

        # Degraded-mode hysteresis state
        self._degraded_active: bool = False
        self._degraded_recover_start_s: Optional[int] = None

        # IMPROVED: Per-source rate limiter
        self.per_source_limiter = PerSourceRateLimiter(
            max_rate_per_s=cfg.per_source_max_rate_per_s,
            window_s=cfg.per_source_window_s,
            burst_multiplier=cfg.per_source_burst_multiplier,
        ) if cfg.enable_per_source_rate_limit else None

        self.source_global_limiter = SourceRateLimiter(
            max_rate_per_s=cfg.source_global_max_rate_per_s,
            window_s=cfg.source_global_window_s,
            burst_multiplier=cfg.source_global_burst_multiplier,
        ) if cfg.enable_source_global_rate_limit else None

        # IMPROVED: Plausibility checker
        self.plausibility_checker = PlausibilityChecker(
            max_shed_step=cfg.plausibility_max_shed_step,
            healthy_shed_threshold=cfg.plausibility_healthy_shed_threshold,
            healthy_deficit_max=cfg.plausibility_healthy_deficit_max,
        ) if cfg.enable_plausibility_check else None

        # IMPROVED: Cross-node correlator
        self.cross_node_correlator = CrossNodeCorrelator(
            correlation_window_s=cfg.correlation_window_s,
            max_simultaneous_targets=cfg.max_simultaneous_targets,
            include_control_setpoint=cfg.correlation_include_control_setpoint,
        ) if cfg.enable_cross_node_correlation else None

        # IMPROVED: Quarantine manager
        self.quarantine_mgr = QuarantineManager(
            duration_s=cfg.quarantine_manager_duration_s,
            cooldown_s=cfg.quarantine_manager_cooldown_s,
        ) if cfg.enable_quarantine_manager else None
        
        # Statistics
        self.stats = {
            "total_decisions": 0,
            "allowed": 0,
            "blocked_stale": 0,
            "blocked_oracle": 0,
            "blocked_rate_limit": 0,
            "blocked_degraded": 0,
            "blocked_intrusion": 0,
            "blocked_signature": 0,
            "delayed_degraded": 0,
            "degraded_mode_triggers": 0,
            "degraded_mode_recovers": 0,
            "quarantine_triggered": 0,
            # IMPROVED stats
            "blocked_per_source_rate": 0,
            "blocked_implausible": 0,
            "blocked_cross_node": 0,
            "blocked_quarantine_mgr": 0,
            "blocked_control_acl": 0,
            "blocked_control_signature": 0,
            "blocked_quantum_control_token": 0,
            "blocked_source_global_rate": 0,
        }

    def _jittered_delay(self, base_ms: int) -> int:
        """Apply hardware timing jitter to verification delay.

        Models realistic crypto-hardware processing variance:
          - FPGA clock-domain crossing: ±2-5 ms
          - HSM contention under load: 0-15 ms
          - SPD dead-time overhead: configurable
        """
        jitter = self.cfg.hw_timing_jitter_ms
        spd_overhead = self.cfg.spd_timing_overhead_ms
        if jitter <= 0.0 and spd_overhead <= 0.0:
            return base_ms
        noise = self._hw_rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
        return max(0, int(base_ms + noise + spd_overhead))

    @staticmethod
    def _is_authenticated_control(msg: Message) -> bool:
        return bool(
            msg.requires_auth
            and msg.msg_type in (MsgType.CONTROL_SETPOINT, MsgType.PRIORITY_ACTION)
        )

    @staticmethod
    def _is_authenticated_priority(msg: Message) -> bool:
        return bool(msg.requires_auth and msg.msg_type == MsgType.PRIORITY_ACTION)

    def _get_effective_rate_limit(self, degraded_active: bool) -> float:
        """Get effective rate limit based on secret fraction (adaptive)."""
        if not self.cfg.adaptive_rate_limit:
            return self.cfg.auth_rate_limit_per_s
        if degraded_active:
            return self.cfg.degraded_rate_limit_per_s
        return self.cfg.normal_rate_limit_per_s

    def _update_degraded_mode(self, *, now_s: int, sf: Optional[float], requires_auth: bool) -> bool:
        """
        Update degraded-mode state with hysteresis.
        Enter when SF drops below degraded_secret_fraction.
        Exit only after SF stays above degraded_recover_secret_fraction for hold_s.
        """
        if not requires_auth:
            return self._degraded_active
        if sf is None:
            return self._degraded_active

        enter_sf = float(self.cfg.degraded_secret_fraction)
        exit_sf = max(enter_sf, float(self.cfg.degraded_recover_secret_fraction))
        hold_s = max(1, int(self.cfg.degraded_recover_hold_s))

        if not self._degraded_active:
            if sf < enter_sf:
                self._degraded_active = True
                self._degraded_recover_start_s = None
                self.stats["degraded_mode_triggers"] += 1
            return self._degraded_active

        # Already degraded: recover only after sustained healthy SF.
        if sf > exit_sf:
            if self._degraded_recover_start_s is None:
                self._degraded_recover_start_s = now_s
            elif (now_s - self._degraded_recover_start_s) >= hold_s:
                self._degraded_active = False
                self._degraded_recover_start_s = None
                self.stats["degraded_mode_recovers"] += 1
        else:
            self._degraded_recover_start_s = None

        return self._degraded_active

    def _rate_limit_hit(self, now_s: int, msg: Message, degraded_active: bool = False) -> bool:
        effective_limit = self._get_effective_rate_limit(degraded_active)
        if effective_limit <= 0 or not msg.requires_auth:
            return False

        win = max(1, int(self.cfg.rate_limit_window_s))
        if self._win_start_s is None or now_s - self._win_start_s >= win:
            self._win_start_s = now_s
            self._auth_count_in_window = 0

        self._auth_count_in_window += 1
        allowed = int(effective_limit * win)
        return self._auth_count_in_window > allowed

    def _get_grid_state(self, dst: str) -> Dict[str, float]:
        """Get current grid state for plausibility checking."""
        mg = self._microgrids.get(dst)
        if mg is None:
            return {"shed": 0.0, "deficit_ratio": 0.0, "control_quality": 1.0,
                    "frequency_hz": 60.0}

        shed = float(getattr(mg, "shed_frac", getattr(mg, "shed_target", 0.0)))
        total_load = float(getattr(mg, "last_total_load_kw", 100.0))
        gen = float(getattr(mg, "last_gen_kw", 0.0))
        import_kw = float(getattr(mg, "last_import_kw", 0.0))
        supply = gen + import_kw
        deficit_ratio = max(0.0, (total_load - supply) / total_load) if total_load > 0 else 0.0
        cq = float(getattr(mg, "control_quality", 1.0))
        # V11: Frequency for plausibility cross-check
        freq_hz = float(getattr(mg, "frequency_hz", 60.0))
        return {"shed": shed, "deficit_ratio": deficit_ratio, "control_quality": cq,
                "frequency_hz": freq_hz}

    def decide(self, *, env: simpy.Environment, msg: Message, 
               action: ControlAction, staleness_ms: int) -> Tuple[ActionDecision, int]:
        """
        Make a defense decision with all strategies.

        Defense pipeline order:
        1. Staleness
        2. Oracle
        3. Quarantine manager
        4. Cross-node correlation
        5. Per-source rate limiting
        6. Global rate limiting
        7. Intrusion detection (selective or full)
        8. Behavioral plausibility
        9. Signature replay check
        10. Quantum degraded-mode policy
        11. Allow with optional delay

        Returns: (ActionDecision, delay_ms)
        """
        now_ms = int(env.now * 1000)
        now_s = int(env.now)
        self.stats["total_decisions"] += 1
        is_control_auth = self._is_authenticated_control(msg)
        is_priority_auth = self._is_authenticated_priority(msg)
        
        # Get secret fraction for quantum-aware decisions
        sf = None
        try:
            sf = float(msg.payload.get("secret_fraction_path_mean"))
        except Exception:
            sf = None
        degraded_active = self._update_degraded_mode(now_s=now_s, sf=sf, requires_auth=msg.requires_auth)
        
        # 1. Staleness check
        if self.cfg.enforce_staleness and should_ignore_as_stale(msg, now_ms=now_ms, staleness_ms=staleness_ms):
            self.stats["blocked_stale"] += 1
            return ActionDecision("ignore", "stale_command"), 0

        # 1b. Control-plane authorization checks (ACL + app signature)
        if is_priority_auth and self.cfg.enable_control_acl and self._control_acl_sources:
            if str(msg.src) not in self._control_acl_sources:
                self.stats["blocked_control_acl"] += 1
                return ActionDecision("block", "control_acl_source"), 0

        if is_priority_auth and self.cfg.require_control_signature:
            sig = str((msg.payload or {}).get("control_signature", ""))
            if sig != str(self.cfg.control_signature_value):
                self.stats["blocked_control_signature"] += 1
                return ActionDecision("block", "control_signature_invalid"), 0

        # Quantum control-token enforcement (QTLS-bound OTP)
        if is_priority_auth and self.cfg.require_quantum_control_token:
            valid = int((msg.payload or {}).get("quantum_control_token_valid", 0) or 0)
            if valid != 1:
                self.stats["blocked_quantum_control_token"] += 1
                reason = str(
                    (msg.payload or {}).get("quantum_control_token_reason",
                                           "quantum_control_token_invalid")
                )
                return ActionDecision("block", reason), 0

        if (
            is_control_auth
            and self.source_global_limiter is not None
            and self.source_global_limiter.check_and_record(now_s, str(msg.src))
        ):
            self.stats["blocked_source_global_rate"] += 1
            return ActionDecision("block", "source_global_rate_limited"), 0
        
        # 2. Oracle attack blocking (for upper-bound analysis)
        if self.cfg.oracle_block_attacks and msg.is_attack:
            self.stats["blocked_oracle"] += 1
            return ActionDecision("block", "oracle_attack_label"), 0
        
        # 3. Quarantine manager
        if self.quarantine_mgr is not None:
            should_block, reason = self.quarantine_mgr.check_message(now_s, msg)
            if should_block:
                self.stats["blocked_quarantine_mgr"] += 1
                return ActionDecision("block", reason), 0

        # 4. Cross-node correlation
        if self.cross_node_correlator is not None:
            acl_known_src = bool(self._control_acl_sources and str(msg.src) in self._control_acl_sources)
            if not acl_known_src:
                is_suspicious, reason = self.cross_node_correlator.record_and_check(now_s, msg)
                if is_suspicious:
                    self.stats["blocked_cross_node"] += 1
                    if self.quarantine_mgr is not None:
                        self.quarantine_mgr.trigger(now_s, msg.dst)
                    return ActionDecision("block", reason), 0

        # 5. Per-source rate limiting
        if (
            self.per_source_limiter is not None
            and is_control_auth
            and self.per_source_limiter.check_and_record(now_s, msg.src, msg.dst)
        ):
            self.stats["blocked_per_source_rate"] += 1
            return ActionDecision("block", "per_source_rate_limited"), 0

        # 6. Global rate limiting
        if is_control_auth and self._rate_limit_hit(now_s, msg, degraded_active):
            self.stats["blocked_rate_limit"] += 1
            return ActionDecision("block", "auth_rate_limited"), 0
        
        # 7. Intrusion detection blocking
        if (
            self.cfg.block_during_intrusion
            and self.intrusion_detector
            and self.intrusion_detector.has_recent_alert(now_s, self.cfg.intrusion_lookback_s)
        ):
            if self.cfg.intrusion_selective:
                # Block only high-risk actions; allow control setpoints through.
                if msg.msg_type == MsgType.PRIORITY_ACTION and msg.requires_auth:
                    self.stats["blocked_intrusion"] += 1
                    if self.quarantine_mgr is not None:
                        self.quarantine_mgr.trigger(now_s, msg.dst)
                    return ActionDecision("block", "intrusion_alert_selective"), 0
            else:
                if msg.requires_auth:
                    self.stats["blocked_intrusion"] += 1
                    return ActionDecision("block", "intrusion_alert_active"), 0
        
        # 8. Behavioral plausibility check
        if self.plausibility_checker is not None and msg.msg_type == MsgType.PRIORITY_ACTION:
            grid_state = self._get_grid_state(msg.dst)
            is_plausible, reason = self.plausibility_checker.check(
                msg=msg,
                action=action,
                current_shed=grid_state["shed"],
                deficit_ratio=grid_state["deficit_ratio"],
                control_quality=grid_state["control_quality"],
                frequency_hz=grid_state.get("frequency_hz", 60.0),
            )
            if not is_plausible:
                self.stats["blocked_implausible"] += 1
                return ActionDecision("block", reason), 0

        # 9. Signature-based blocking
        if (self.cfg.block_repeated_commands and 
            self.signature_tracker and
            msg.msg_type == MsgType.PRIORITY_ACTION and
            self.signature_tracker.is_repeated(now_s, msg, action)):
            self.stats["blocked_signature"] += 1
            return ActionDecision("block", "repeated_command"), 0
        
        # 10. Quantum-aware degraded mode
        if degraded_active and msg.requires_auth:
            # Block priority actions in degraded mode if configured
            if msg.msg_type == MsgType.PRIORITY_ACTION and self.cfg.block_priority_in_degraded:
                self.stats["blocked_degraded"] += 1
                
                # Trigger quarantine when available
                if self.quarantine_mgr is not None:
                    self.quarantine_mgr.trigger(now_s, msg.dst)
                    self.stats["quarantine_triggered"] += 1
                elif self.cfg.enable_quarantine:
                    self.stats["quarantine_triggered"] += 1
                
                return ActionDecision("block", "degraded_secret_fraction"), 0
            
            # Otherwise allow with extra delay
            self.stats["delayed_degraded"] += 1
            delay = self.cfg.verification_delay_ms + self.cfg.degraded_verification_delay_ms
            return ActionDecision("allow", "degraded_with_delay"), self._jittered_delay(delay)

        # 11. Normal allow
        self.stats["allowed"] += 1
        return ActionDecision("allow", "policy_allow"), self._jittered_delay(int(self.cfg.verification_delay_ms))


# -------------------------
# Delivery hook helper
# -------------------------

def make_on_deliver_hook(*, microgrids: Dict[str, Any], gate: PolicyGate) -> Callable[[simpy.Environment, Message], None]:
    """
    Create on_deliver_hook for NetworkSim.
    """
    def _hook(env: simpy.Environment, msg: Message) -> None:
        action = parse_action_from_message(msg)
        if action is None:
            return

        mg = microgrids.get(msg.dst, None)
        if mg is None:
            return

        decision, verify_delay_ms = gate.decide(env=env, msg=msg, action=action, 
                                                 staleness_ms=mg.params.staleness_ms)

        def _apply_later():
            if verify_delay_ms > 0:
                yield env.timeout(verify_delay_ms / 1000.0)
            mg.apply_action(now_s=int(env.now), msg=msg, action=action, decision=decision)

        env.process(_apply_later())

    return _hook


# -------------------------
# Degraded threshold presets
# -------------------------

DEGRADED_THRESHOLD_PRESETS = {
    "paranoid": {
        "secret_fraction": 0.90,
        "qber_approx": 0.010,
        "description": "Ultra-sensitive, may false-alarm on normal noise",
    },
    "conservative": {
        "secret_fraction": 0.85,
        "qber_approx": 0.015,
        "description": "High security, low tolerance for noise",
    },
    "balanced": {
        "secret_fraction": 0.70,
        "qber_approx": 0.030,
        "description": "Balanced security and availability",
    },
    "moderate": {
        "secret_fraction": 0.50,
        "qber_approx": 0.050,
        "description": "Tolerates moderate noise, good availability",
    },
    "permissive": {
        "secret_fraction": 0.20,
        "qber_approx": 0.080,
        "description": "High availability, detects only severe attacks",
    },
    "minimal": {
        "secret_fraction": 0.05,
        "qber_approx": 0.105,
        "description": "Near theoretical limit, maximum availability",
    },
}


def get_degraded_threshold(preset: str) -> float:
    name = str(preset or "moderate").lower()
    if name in DEGRADED_THRESHOLD_PRESETS:
        return float(DEGRADED_THRESHOLD_PRESETS[name]["secret_fraction"])
    try:
        return float(name)
    except Exception:
        return float(DEGRADED_THRESHOLD_PRESETS["moderate"]["secret_fraction"])


def qber_to_secret_fraction(qber: float) -> float:
    return secret_fraction_bb84(qber)


def secret_fraction_to_qber_approx(sf: float) -> float:
    if sf >= 1.0:
        return 0.0
    if sf <= 0.0:
        return 0.11
    lo, hi = 0.0, 0.11
    for _ in range(50):
        mid = (lo + hi) / 2
        if qber_to_secret_fraction(mid) > sf:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# -------------------------
# Defense config factory
# -------------------------

def get_defense_config(strategy: str, degraded_threshold: str = "moderate") -> GateConfig:
    """
    Factory function to get defense configuration by strategy name.

    ORIGINAL strategies:
    - none, ratelimit, block, delay, intrusion, adaptive, signature, quarantine, all

    IMPROVED strategies:
    - ratelimit_v2: per-source rate limiting
    - intrusion_v2: selective intrusion blocking (PRIORITY_ACTION only)
    - plausibility: behavioral plausibility checks
    - correlation: cross-node coordinated attack detection
    - quarantine_v2: actual quarantine safe-mode with selective blocking
    - hardened: all improved defenses combined
    - hardened_balanced: main-paper profile (same gate hardening, milder quantum layer)
    - hardened_strong: upper-bound profile (legacy stronger quantum behavior)
    - gate_only: all improved gate defenses, no quantum-layer changes
    - quantum_only: no gate hardening (quantum-layer defenses only)
    """
    
    threshold = get_degraded_threshold(degraded_threshold)
    recover_threshold = min(0.99, threshold + 0.10)
    recover_hold_s = 60

    configs = {
        "none": GateConfig(
            verification_delay_ms=0,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            block_priority_in_degraded=False,
            auth_rate_limit_per_s=0.0,
        ),
        
        "ratelimit": GateConfig(
            verification_delay_ms=0,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            block_priority_in_degraded=False,
            auth_rate_limit_per_s=0.2,
            rate_limit_window_s=5,
        ),
        
        "block": GateConfig(
            verification_delay_ms=0,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            block_priority_in_degraded=True,
            auth_rate_limit_per_s=0.0,
        ),
        
        "delay": GateConfig(
            verification_delay_ms=100,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            degraded_verification_delay_ms=200,
            block_priority_in_degraded=False,
            auth_rate_limit_per_s=0.0,
        ),
        
        "intrusion": GateConfig(
            verification_delay_ms=0,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            block_priority_in_degraded=False,
            auth_rate_limit_per_s=0.0,
            block_during_intrusion=True,
            intrusion_lookback_s=60,
        ),
        
        "adaptive": GateConfig(
            verification_delay_ms=0,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            block_priority_in_degraded=False,
            adaptive_rate_limit=True,
            normal_rate_limit_per_s=10.0,
            degraded_rate_limit_per_s=2.0,
        ),
        
        "signature": GateConfig(
            verification_delay_ms=0,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            block_priority_in_degraded=False,
            auth_rate_limit_per_s=0.0,
            block_repeated_commands=True,
            command_cooldown_s=30,
            max_command_repetitions=2,
        ),
        
        "quarantine": GateConfig(
            verification_delay_ms=0,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            block_priority_in_degraded=True,
            auth_rate_limit_per_s=0.0,
            enable_quarantine=True,
            quarantine_duration_s=60,
        ),
        
        "all": GateConfig(
            verification_delay_ms=50,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            degraded_verification_delay_ms=150,
            block_priority_in_degraded=True,
            auth_rate_limit_per_s=0.2,
            rate_limit_window_s=5,
            block_during_intrusion=True,
            intrusion_lookback_s=60,
            adaptive_rate_limit=True,
            normal_rate_limit_per_s=10.0,
            degraded_rate_limit_per_s=2.0,
            block_repeated_commands=True,
            command_cooldown_s=30,
            max_command_repetitions=2,
            enable_quarantine=True,
            quarantine_duration_s=60,
        ),

        # ---- IMPROVED STRATEGIES ----

        "ratelimit_v2": GateConfig(
            verification_delay_ms=0,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            block_priority_in_degraded=False,
            enable_per_source_rate_limit=True,
            per_source_max_rate_per_s=0.5,
            per_source_window_s=10,
            per_source_burst_multiplier=2.0,
        ),

        "intrusion_v2": GateConfig(
            verification_delay_ms=0,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            block_priority_in_degraded=False,
            block_during_intrusion=True,
            intrusion_lookback_s=60,
            intrusion_selective=True,
        ),

        "plausibility": GateConfig(
            verification_delay_ms=0,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            block_priority_in_degraded=False,
            enable_plausibility_check=True,
            plausibility_max_shed_step=0.35,
            plausibility_healthy_shed_threshold=0.40,
            plausibility_healthy_deficit_max=0.10,
        ),

        "correlation": GateConfig(
            verification_delay_ms=0,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            block_priority_in_degraded=False,
            enable_cross_node_correlation=True,
            correlation_window_s=30,
            max_simultaneous_targets=1,
        ),

        "quarantine_v2": GateConfig(
            verification_delay_ms=0,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            block_priority_in_degraded=True,
            enable_quarantine_manager=True,
            quarantine_manager_duration_s=120,
            quarantine_manager_cooldown_s=60,
        ),

        "hardened_strong": GateConfig(
            verification_delay_ms=50,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            degraded_verification_delay_ms=150,
            block_priority_in_degraded=True,
            # Per-source rate limiting (primary)
            enable_per_source_rate_limit=True,
            per_source_max_rate_per_s=0.5,
            per_source_window_s=10,
            per_source_burst_multiplier=2.0,
            # Selective intrusion response
            block_during_intrusion=True,
            intrusion_lookback_s=60,
            intrusion_selective=True,
            # Adaptive global fallback
            adaptive_rate_limit=True,
            normal_rate_limit_per_s=5.0,
            degraded_rate_limit_per_s=1.0,
            # Replay tightening
            block_repeated_commands=True,
            command_cooldown_s=60,
            max_command_repetitions=1,
            # Behavioral plausibility
            enable_plausibility_check=True,
            plausibility_max_shed_step=0.35,
            plausibility_healthy_shed_threshold=0.40,
            plausibility_healthy_deficit_max=0.10,
            # Cross-node correlation
            enable_cross_node_correlation=True,
            correlation_window_s=30,
            max_simultaneous_targets=1,
            # Quarantine manager
            enable_quarantine_manager=True,
            quarantine_manager_duration_s=120,
            quarantine_manager_cooldown_s=60,
            # Control-plane hardening
            enable_control_acl=True,
            require_control_signature=True,
            enable_source_global_rate_limit=True,
            source_global_max_rate_per_s=1.0,
            source_global_window_s=10,
            source_global_burst_multiplier=2.0,
            correlation_include_control_setpoint=True,
        ),
        "hardened_balanced": GateConfig(
            verification_delay_ms=50,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            degraded_verification_delay_ms=150,
            block_priority_in_degraded=True,
            enable_per_source_rate_limit=True,
            per_source_max_rate_per_s=1.5,
            per_source_window_s=10,
            per_source_burst_multiplier=4.0,
            block_during_intrusion=True,
            intrusion_lookback_s=60,
            intrusion_selective=True,
            adaptive_rate_limit=True,
            normal_rate_limit_per_s=8.0,
            degraded_rate_limit_per_s=2.0,
            block_repeated_commands=True,
            command_cooldown_s=60,
            max_command_repetitions=1,
            enable_plausibility_check=True,
            plausibility_max_shed_step=0.35,
            plausibility_healthy_shed_threshold=0.40,
            plausibility_healthy_deficit_max=0.10,
            enable_cross_node_correlation=True,
            correlation_window_s=30,
            max_simultaneous_targets=1,
            enable_quarantine_manager=True,
            quarantine_manager_duration_s=120,
            quarantine_manager_cooldown_s=60,
            enable_control_acl=True,
            require_control_signature=True,
            enable_source_global_rate_limit=True,
            source_global_max_rate_per_s=4.0,
            source_global_window_s=10,
            source_global_burst_multiplier=4.0,
            correlation_include_control_setpoint=False,
        ),
        # Keep "hardened" as the default alias used by existing scripts; this
        # now maps to the balanced main-paper profile.
        "hardened": GateConfig(
            verification_delay_ms=50,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            degraded_verification_delay_ms=150,
            block_priority_in_degraded=True,
            enable_per_source_rate_limit=True,
            per_source_max_rate_per_s=1.5,
            per_source_window_s=10,
            per_source_burst_multiplier=4.0,
            block_during_intrusion=True,
            intrusion_lookback_s=60,
            intrusion_selective=True,
            adaptive_rate_limit=True,
            normal_rate_limit_per_s=8.0,
            degraded_rate_limit_per_s=2.0,
            block_repeated_commands=True,
            command_cooldown_s=60,
            max_command_repetitions=1,
            enable_plausibility_check=True,
            plausibility_max_shed_step=0.35,
            plausibility_healthy_shed_threshold=0.40,
            plausibility_healthy_deficit_max=0.10,
            enable_cross_node_correlation=True,
            correlation_window_s=30,
            max_simultaneous_targets=1,
            enable_quarantine_manager=True,
            quarantine_manager_duration_s=120,
            quarantine_manager_cooldown_s=60,
            enable_control_acl=True,
            require_control_signature=True,
            enable_source_global_rate_limit=True,
            source_global_max_rate_per_s=4.0,
            source_global_window_s=10,
            source_global_burst_multiplier=4.0,
            correlation_include_control_setpoint=False,
        ),

        # Gate-only ablation: same gate hardening as hardened mode.
        "gate_only": GateConfig(
            verification_delay_ms=50,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            degraded_verification_delay_ms=150,
            block_priority_in_degraded=True,
            enable_per_source_rate_limit=True,
            per_source_max_rate_per_s=0.5,
            per_source_window_s=10,
            per_source_burst_multiplier=2.0,
            block_during_intrusion=True,
            intrusion_lookback_s=60,
            intrusion_selective=True,
            adaptive_rate_limit=True,
            normal_rate_limit_per_s=5.0,
            degraded_rate_limit_per_s=1.0,
            block_repeated_commands=True,
            command_cooldown_s=60,
            max_command_repetitions=1,
            enable_plausibility_check=True,
            plausibility_max_shed_step=0.35,
            plausibility_healthy_shed_threshold=0.40,
            plausibility_healthy_deficit_max=0.10,
            enable_cross_node_correlation=True,
            correlation_window_s=30,
            max_simultaneous_targets=1,
            enable_quarantine_manager=True,
            quarantine_manager_duration_s=120,
            quarantine_manager_cooldown_s=60,
            enable_control_acl=True,
            require_control_signature=True,
            enable_source_global_rate_limit=True,
            source_global_max_rate_per_s=1.0,
            source_global_window_s=10,
            source_global_burst_multiplier=2.0,
            correlation_include_control_setpoint=True,
        ),

        # Quantum-only ablation: leave gate at baseline/none posture.
        "quantum_only": GateConfig(
            verification_delay_ms=0,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            block_priority_in_degraded=False,
            auth_rate_limit_per_s=0.0,
        ),

        # V2: Hardened with multi-protocol quantum layer
        # Quarantine: 45s (was 120s) — realistic: shorter lockout, requires
        # 3+ simultaneous targets (was 1) before cross-node correlation triggers.
        # A sophisticated attacker targeting one node at a time evades correlation.
        "hardened_v2": GateConfig(
            verification_delay_ms=50,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            degraded_verification_delay_ms=150,
            block_priority_in_degraded=True,
            enable_per_source_rate_limit=True,
            per_source_max_rate_per_s=1.5,
            per_source_window_s=10,
            per_source_burst_multiplier=4.0,
            block_during_intrusion=True,
            intrusion_lookback_s=60,
            intrusion_selective=True,
            adaptive_rate_limit=True,
            normal_rate_limit_per_s=8.0,
            degraded_rate_limit_per_s=2.0,
            block_repeated_commands=True,
            command_cooldown_s=60,
            max_command_repetitions=1,
            enable_plausibility_check=True,
            plausibility_max_shed_step=0.35,
            plausibility_healthy_shed_threshold=0.40,
            plausibility_healthy_deficit_max=0.10,
            enable_cross_node_correlation=True,
            correlation_window_s=30,
            max_simultaneous_targets=3,
            enable_quarantine_manager=True,
            quarantine_manager_duration_s=45,
            quarantine_manager_cooldown_s=30,
            enable_control_acl=True,
            require_control_signature=True,
            enable_source_global_rate_limit=True,
            source_global_max_rate_per_s=4.0,
            source_global_window_s=10,
            source_global_burst_multiplier=4.0,
            correlation_include_control_setpoint=False,
        ),

        # ── hardened_v3: Tuned gate that does NOT block legitimate traffic ──
        # Key changes from v2:
        #   - block_priority_in_degraded=False  (was True — stops collateral blocking)
        #   - intrusion_lookback_s=20           (was 60 — shorter memory)
        #   - max_simultaneous_targets=6        (was 3 — harder to trigger correlation)
        #   - command_cooldown_s=120, max_reps=3 (was 60/1 — allows legitimate repeats)
        #   - plausibility_healthy_shed_threshold=0.50 (was 0.25 — less restrictive)
        #   - quarantine_duration=20s           (was 45s — shorter lockout)
        #   - higher rate limits throughout      (lets legitimate traffic through)
        "hardened_v3": GateConfig(
            verification_delay_ms=50,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            degraded_verification_delay_ms=150,
            block_priority_in_degraded=False,
            enable_per_source_rate_limit=True,
            per_source_max_rate_per_s=2.0,
            per_source_window_s=10,
            per_source_burst_multiplier=4.0,
            block_during_intrusion=True,
            intrusion_lookback_s=20,
            intrusion_selective=True,
            adaptive_rate_limit=True,
            normal_rate_limit_per_s=10.0,
            degraded_rate_limit_per_s=4.0,
            block_repeated_commands=True,
            command_cooldown_s=60,              # was 120 — shorter cooldown reduces FP on legitimate repeats
            max_command_repetitions=5,           # was 3 — controllers legitimately send 3-4 repeats
            enable_plausibility_check=True,
            # V11: Tuned thresholds to reduce false positive rate (<3%)
            # Attack uses shed_frac=0.55 with no deficit — well above these
            plausibility_max_shed_step=0.40,    # was 0.25 — islanding recovery needs big steps
            plausibility_healthy_shed_threshold=0.50,
            plausibility_healthy_deficit_max=0.12, # was 0.10 — renewable variability margin
            enable_cross_node_correlation=True,
            correlation_window_s=20,     # was 30 — shorter window reduces FP from legitimate bursts
            max_simultaneous_targets=8,  # was 6 — controller often sends to multiple nodes quickly
            enable_quarantine_manager=True,
            quarantine_manager_duration_s=20,
            quarantine_manager_cooldown_s=20,
            enable_control_acl=True,
            require_control_signature=True,
            enable_source_global_rate_limit=True,
            source_global_max_rate_per_s=6.0,
            source_global_window_s=10,
            source_global_burst_multiplier=4.0,
            correlation_include_control_setpoint=False,
        ),

        # ── Ablation presets (isolate individual behavioral defenses) ──

        "plausibility_only": GateConfig(
            verification_delay_ms=50,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            degraded_verification_delay_ms=150,
            enable_plausibility_check=True,
            plausibility_max_shed_step=0.35,
            plausibility_healthy_shed_threshold=0.40,
            plausibility_healthy_deficit_max=0.10,
        ),

        "correlator_only": GateConfig(
            verification_delay_ms=50,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            degraded_verification_delay_ms=150,
            enable_cross_node_correlation=True,
            correlation_window_s=30,
            max_simultaneous_targets=1,
        ),

        "quarantine_only": GateConfig(
            verification_delay_ms=50,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            degraded_verification_delay_ms=150,
            enable_quarantine_manager=True,
            quarantine_manager_duration_s=120,
            quarantine_manager_cooldown_s=60,
        ),

        "all_behavioral": GateConfig(
            verification_delay_ms=50,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            degraded_verification_delay_ms=150,
            enable_plausibility_check=True,
            plausibility_max_shed_step=0.35,
            plausibility_healthy_shed_threshold=0.40,
            plausibility_healthy_deficit_max=0.10,
            enable_cross_node_correlation=True,
            correlation_window_s=30,
            max_simultaneous_targets=1,
            enable_quarantine_manager=True,
            quarantine_manager_duration_s=120,
            quarantine_manager_cooldown_s=60,
        ),
    }

    return configs.get(strategy, configs["none"])


# -------------------------
# V2: Insider Threat Attack
# -------------------------

@dataclass
class InsiderThreatConfig:
    """
    Configuration for insider threat attack.

    An insider has legitimate credentials but issues malicious commands.
    Unlike spoofing, insider messages pass ACL/signature checks.

    Parameters:
        start_s: Attack start time (seconds)
        end_s: Attack end time (seconds)
        rate_per_s: Malicious command rate
        use_legitimate_credentials: If True, messages carry valid control signature
        target_shed_frac: Load shedding fraction to command
        harm_duration_s: Duration of each malicious action
    """
    start_s: int = 600
    end_s: int = 1800
    rate_per_s: float = 0.1
    use_legitimate_credentials: bool = True
    target_shed_frac: float = 0.60
    harm_duration_s: int = 45
    label: str = "insider_threat"


class InsiderThreatAttack:
    """
    Insider threat: an authorized node sends malicious control commands.

    Key difference from SpoofingAttack:
    - Messages come from a legitimate controller node
    - Messages carry valid control_signature
    - Messages are CONTROL_SETPOINT (not PRIORITY_ACTION) to blend in
    - Bypasses ACL and signature checks by design

    This tests behavioral/plausibility defenses rather than auth defenses.
    """

    def __init__(
        self,
        env: simpy.Environment,
        rng: random.Random,
        cfg: InsiderThreatConfig,
        msg_id_fn,
        emit_fn,
    ):
        self.env = env
        self.rng = rng
        self.cfg = cfg
        self.msg_id_fn = msg_id_fn
        self.emit_fn = emit_fn
        self.stats = {
            "commands_sent": 0,
            "windows_scheduled": 0,
        }

    def schedule(self, insider_node: str, victim_nodes: list):
        """
        Schedule insider attack campaign.

        Args:
            insider_node: The compromised legitimate controller
            victim_nodes: Target microgrid nodes
        """
        self.stats["windows_scheduled"] += 1
        self.env.process(self._attack_process(insider_node, victim_nodes))

    def _attack_process(self, insider_node: str, victim_nodes: list):
        """SimPy process: insider sends malicious commands at configured rate."""
        yield self.env.timeout(max(0, self.cfg.start_s - self.env.now))

        interval = 1.0 / max(0.01, self.cfg.rate_per_s)

        while self.env.now < self.cfg.end_s:
            victim = self.rng.choice(victim_nodes)

            payload = {
                "action_type": "shed_load_emergency",
                "target_shed_frac": self.cfg.target_shed_frac,
                "duration_s": self.cfg.harm_duration_s,
                "reason": "priority_action_shed",
                "attack_label": self.cfg.label,
            }

            # Use legitimate credentials if configured
            if self.cfg.use_legitimate_credentials:
                payload["control_signature"] = "quam_ctrl_v1"
                payload["control_sender_role"] = "controller"

            msg = Message(
                msg_id=self.msg_id_fn(),
                created_ms=int(self.env.now * 1000),
                src=insider_node,
                dst=victim,
                msg_type=MsgType.PRIORITY_ACTION,
                priority=2,
                deadline_ms=300,
                size_bytes=280,
                requires_auth=True,
                is_attack=True,
                attack_label=self.cfg.label,
                payload=payload,
            )
            self.emit_fn(msg)
            self.stats["commands_sent"] += 1

            jitter = self.rng.uniform(0.8, 1.2)
            yield self.env.timeout(interval * jitter)


# ------------------------------------------
# Node-Level Spoofing Attack (Medium Case)
# ------------------------------------------

@dataclass
class NodeLevelSpoofConfig:
    """
    Configuration for a node-level spoofing attack.

    A non-controller node is compromised and sends malicious
    PRIORITY_ACTION commands.  Two sub-variants:

    forge_source=False  →  src = compromised_node (honest).
        Caught immediately by ACL.  Tests behavioral defenses
        when ACL is disabled.

    forge_source=True   →  src = controller (forged header),
        but injection_node = compromised_node.  Message routes
        from the compromised node's physical position.  Passes
        ACL but routing path differs from genuine controller
        traffic.
    """
    compromised_node: str = "MG5"
    controller_node: str = "MG0"
    forge_source: bool = False
    rate_per_s: float = 0.1
    forced_shed_frac: float = 0.55
    harm_duration_s: int = 45
    label: str = "node_level_spoof"


class NodeLevelSpoofingAttack:
    """
    Node-level attacker: operates from a single compromised non-controller node.

    Key differences from infrastructure-level SpoofingAttack:
    - Attacker is physically at a specific node position in the network
    - If forge_source=False: src=compromised_node (easily caught by ACL)
    - If forge_source=True: src=controller but injection_node=compromised_node
      (message routes from compromised node, not controller)

    This models a more realistic medium-threat-level adversary who has
    breached a single microgrid but not the network infrastructure.
    """

    def __init__(
        self,
        env: simpy.Environment,
        rng: random.Random,
        cfg: NodeLevelSpoofConfig,
        msg_id_fn,
        emit_fn,
    ):
        self.env = env
        self.rng = rng
        self.cfg = cfg
        self.msg_id_fn = msg_id_fn
        self.emit_fn = emit_fn
        self.stats = {
            "commands_sent": 0,
            "windows_scheduled": 0,
        }

    def schedule(self, start_s: int, end_s: int, victim_nodes: list):
        """Schedule a node-level spoofing campaign for one attack window."""
        self.stats["windows_scheduled"] += 1
        self.env.process(self._attack_process(start_s, end_s, victim_nodes))

    def _attack_process(self, start_s: int, end_s: int, victim_nodes: list):
        """SimPy process: compromised node sends malicious commands."""
        if start_s > self.env.now:
            yield self.env.timeout(start_s - self.env.now)

        interval = 1.0 / max(0.01, self.cfg.rate_per_s)

        while self.env.now < end_s:
            victim = self.rng.choice(victim_nodes)

            payload = {
                "action": "shed_load_emergency",
                "forced_shed_frac": float(self.cfg.forced_shed_frac),
                "harm_duration_s": int(self.cfg.harm_duration_s),
                "attack": True,
                "attack_label": self.cfg.label,
                "compromised_node": self.cfg.compromised_node,
                "forge_source": self.cfg.forge_source,
            }

            if self.cfg.forge_source:
                # Forged source: header says controller, but physical origin
                # is the compromised node (injection_node).
                msg_src = self.cfg.controller_node
                inj_node = self.cfg.compromised_node
                payload["control_signature"] = "quam_ctrl_v1"
                payload["control_sender_role"] = "controller"
                # Attacker attempts to forge a quantum control token.
                # Without the QKD-derived HMAC secret, the best the
                # attacker can do is generate a random 128-bit hex string
                # that will fail HMAC verification at the receiver.
                fake_token = f"{self.rng.getrandbits(128):032x}"
                payload["quantum_control_token"] = fake_token
                payload["quantum_control_expiry_ms"] = int(self.env.now * 1000) + 2000
                payload["nonce"] = self.rng.getrandbits(64)
                payload["quantum_token_forged"] = True
            else:
                # Honest source: header says compromised node.
                msg_src = self.cfg.compromised_node
                inj_node = None

            msg = Message(
                msg_id=self.msg_id_fn(),
                created_ms=int(self.env.now * 1000),
                src=msg_src,
                dst=victim,
                msg_type=MsgType.PRIORITY_ACTION,
                priority=2,
                deadline_ms=300,
                size_bytes=280,
                requires_auth=True,
                is_attack=True,
                attack_label=self.cfg.label,
                injection_node=inj_node,
                payload=payload,
            )
            self.emit_fn(msg)
            self.stats["commands_sent"] += 1

            jitter = self.rng.uniform(0.8, 1.2)
            yield self.env.timeout(interval * jitter)


# =============================================================================
# COORDINATED MULTI-NODE ATTACK (APT / State-Level Adversary)
# =============================================================================

@dataclass
class CoordinatedAttackConfig:
    """
    Configuration for coordinated multi-node attacks.

    Models an Advanced Persistent Threat (APT) / state-level adversary that
    has compromised multiple microgrid nodes simultaneously.  The attacker
    coordinates injection timing across nodes to:
      1. Overwhelm per-source rate limiters (different sources)
      2. Create cross-node confusion (legitimate-looking distributed commands)
      3. Exhaust QKD key pools faster (more authenticated channels under load)

    This addresses the "single attacker" limitation of NodeLevelSpoofingAttack.
    """
    compromised_nodes: List[str] = field(default_factory=lambda: ["MG1", "MG3"])
    controller_node: str = "MG0"
    forge_source: bool = True
    rate_per_s_per_node: float = 0.08  # Low per-node to avoid per-source triggers
    forced_shed_frac: float = 0.55
    harm_duration_s: int = 45
    # Coordination: stagger timing so messages arrive in bursts from different nodes
    coordination_phase_offset_s: float = 2.0  # seconds between each node's injection cycle
    # Diversified attack payloads: vary shed fraction slightly per node
    shed_frac_jitter: float = 0.05  # ±5% variation
    label: str = "coordinated_multi_node"


class CoordinatedMultiNodeAttack:
    """
    Multi-node coordinated attacker: operates from N compromised nodes
    simultaneously, each forging the controller identity.

    Key threat properties vs single-node attack:
    - Per-source rate limits are less effective (N distinct sources)
    - Cross-node correlation detector sees commands from multiple injection points
    - Higher aggregate injection rate (N × rate_per_node)
    - QKD pool stress distributed across more edges
    """

    def __init__(
        self,
        env: simpy.Environment,
        rng: random.Random,
        cfg: CoordinatedAttackConfig,
        msg_id_fn,
        emit_fn,
    ):
        self.env = env
        self.rng = rng
        self.cfg = cfg
        self.msg_id_fn = msg_id_fn
        self.emit_fn = emit_fn
        self.stats = {
            "commands_sent": 0,
            "commands_per_node": {n: 0 for n in cfg.compromised_nodes},
            "windows_scheduled": 0,
        }

    def schedule(self, start_s: int, end_s: int, victim_nodes: list):
        """Schedule coordinated attack from all compromised nodes."""
        self.stats["windows_scheduled"] += 1
        for idx, comp_node in enumerate(self.cfg.compromised_nodes):
            phase = idx * self.cfg.coordination_phase_offset_s
            self.env.process(
                self._node_attack_process(comp_node, start_s + phase, end_s, victim_nodes)
            )

    def _node_attack_process(self, comp_node: str, start_s: float,
                             end_s: float, victim_nodes: list):
        """SimPy process: single compromised node's injection loop."""
        if start_s > self.env.now:
            yield self.env.timeout(start_s - self.env.now)

        interval = 1.0 / max(0.01, self.cfg.rate_per_s_per_node)

        while self.env.now < end_s:
            victim = self.rng.choice(victim_nodes)

            # Slightly varied shed fraction per command (evade pattern detection)
            shed = self.cfg.forced_shed_frac + self.rng.uniform(
                -self.cfg.shed_frac_jitter, self.cfg.shed_frac_jitter
            )
            shed = max(0.1, min(0.95, shed))

            payload = {
                "action": "shed_load_emergency",
                "forced_shed_frac": float(shed),
                "harm_duration_s": int(self.cfg.harm_duration_s),
                "attack": True,
                "attack_label": self.cfg.label,
                "compromised_node": comp_node,
                "forge_source": self.cfg.forge_source,
                "coordinated": True,
                "n_attackers": len(self.cfg.compromised_nodes),
            }

            if self.cfg.forge_source:
                msg_src = self.cfg.controller_node
                inj_node = comp_node
                payload["control_signature"] = "quam_ctrl_v1"
                payload["control_sender_role"] = "controller"
                fake_token = f"{self.rng.getrandbits(128):032x}"
                payload["quantum_control_token"] = fake_token
                payload["quantum_control_expiry_ms"] = int(self.env.now * 1000) + 2000
                payload["nonce"] = self.rng.getrandbits(64)
                payload["quantum_token_forged"] = True
            else:
                msg_src = comp_node
                inj_node = None

            msg = Message(
                msg_id=self.msg_id_fn(),
                created_ms=int(self.env.now * 1000),
                src=msg_src,
                dst=victim,
                msg_type=MsgType.PRIORITY_ACTION,
                priority=2,
                deadline_ms=300,
                size_bytes=280,
                requires_auth=True,
                is_attack=True,
                attack_label=self.cfg.label,
                injection_node=inj_node,
                payload=payload,
            )
            self.emit_fn(msg)
            self.stats["commands_sent"] += 1
            self.stats["commands_per_node"][comp_node] = \
                self.stats["commands_per_node"].get(comp_node, 0) + 1

            jitter = self.rng.uniform(0.8, 1.2)
            yield self.env.timeout(interval * jitter)


# =============================================================================
# FALSE DATA INJECTION (FDI) ATTACK
# =============================================================================

@dataclass
class FDIAttackConfig:
    """
    Configuration for False Data Injection attacks on CPS telemetry.

    FDI corrupts sensor/telemetry data to cause incorrect control decisions.
    This is the canonical CPS attack — if state estimation is wrong, controllers
    make physically harmful decisions (NIST CPS framework).

    Two attack vectors:
    1. Measurement corruption: bias generation/load readings
    2. Timestamp manipulation: replay old valid readings (future extension)
    """
    start_s: int = 0
    end_s: int = 600
    injection_rate_per_s: float = 0.5     # corrupted readings per second
    gen_bias_kw: float = -30.0            # bias on generation readings (negative = underreport)
    load_bias_kw: float = 20.0            # bias on load readings (positive = overreport)
    target_sensors: str = "generation"    # "generation", "load", or "both"
    stealthy: bool = True                 # gradual ramp vs sudden jump
    ramp_duration_s: float = 60.0         # time to ramp bias from 0 to full (if stealthy)
    fdi_forge_prob: float = 0.15          # prob of forging classical HMAC (bypassing auth)
    label: str = "fdi"


class FalseDataInjectionAttack:
    """
    Corrupts SCADA sensor/telemetry data to cause incorrect control decisions.

    Models the canonical CPS attack: corrupting measurements that feed
    state estimation.  Unlike spoofing (forges control COMMANDS), FDI
    corrupts the INPUTS to the controller's decision-making process.

    Attack chain:
    1. Attacker intercepts sensor readings on telemetry channel
    2. Modifies generation/load values (bias injection)
    3. Controller receives falsified state estimation
    4. Controller makes wrong shed/dispatch decisions
    5. Grid experiences unnecessary EENS despite being physically healthy

    Quantum defense: QKD-authenticated telemetry with QRNG nonce
    freshness checking detects measurement corruption (tag mismatch)
    and timestamp replay (nonce sequence violation).

    With QKD active: auth tags are information-theoretically secure →
    FDI modifications detected with probability ~1.0.
    With classical HMAC: forging probability = fdi_forge_prob.
    Without defense: all corrupted readings accepted.
    """

    def __init__(
        self,
        env: simpy.Environment,
        rng: random.Random,
        cfg: FDIAttackConfig,
        msg_id_fn: Callable[[], int],
        emit_fn: Callable[[Message], None],
    ):
        self.env = env
        self.rng = rng
        self.cfg = cfg
        self.msg_id_fn = msg_id_fn
        self.emit_fn = emit_fn
        self.stats: Dict[str, int] = {
            "injected_count": 0,
            "detected_count": 0,
            "accepted_count": 0,
        }

    def schedule(
        self,
        sensor_nodes: List[str],
        controller_node: str,
    ) -> None:
        """Schedule FDI injection on sensor → controller telemetry channels."""
        self.env.process(
            self._inject_loop(sensor_nodes, controller_node)
        )

    def _compute_bias(self, t_s: float) -> Tuple[float, float]:
        """Compute current bias values, applying stealth ramp if enabled."""
        elapsed = t_s - self.cfg.start_s
        if self.cfg.stealthy and elapsed < self.cfg.ramp_duration_s:
            ramp_frac = elapsed / self.cfg.ramp_duration_s
        else:
            ramp_frac = 1.0

        gen_bias = self.cfg.gen_bias_kw * ramp_frac
        load_bias = self.cfg.load_bias_kw * ramp_frac
        return gen_bias, load_bias

    def _inject_loop(
        self,
        sensor_nodes: List[str],
        controller_node: str,
    ):
        """Coroutine: inject corrupted telemetry readings."""
        yield self.env.timeout(max(0, self.cfg.start_s - self.env.now))

        interval = 1.0 / max(0.01, self.cfg.injection_rate_per_s)

        while self.env.now < self.cfg.end_s:
            t_s = float(self.env.now)
            gen_bias, load_bias = self._compute_bias(t_s)

            # Pick a random sensor node to corrupt
            sensor = self.rng.choice(sensor_nodes)

            payload: Dict[str, Any] = {
                "attack": True,
                "attack_label": self.cfg.label,
                "attack_type": "fdi",
                "fdi_gen_bias_kw": gen_bias,
                "fdi_load_bias_kw": load_bias,
                "fdi_target_sensors": self.cfg.target_sensors,
                "fdi_stealthy": self.cfg.stealthy,
                "fdi_forge_prob": self.cfg.fdi_forge_prob,
                "action": "telemetry_inject",
            }

            msg = Message(
                msg_id=self.msg_id_fn(),
                created_ms=int(t_s * 1000),
                src=sensor,
                dst=controller_node,
                msg_type=MsgType.CONTROL_SETPOINT,  # Telemetry uses same channel
                priority=1,  # Lower priority than PRIORITY_ACTION
                deadline_ms=500,
                size_bytes=200,
                requires_auth=True,
                is_attack=True,
                attack_label=self.cfg.label,
                payload=payload,
            )
            self.emit_fn(msg)
            self.stats["injected_count"] += 1

            jitter = self.rng.uniform(0.8, 1.2)
            yield self.env.timeout(interval * jitter)


# =============================================================================
# CLASSICAL MAN-IN-THE-MIDDLE (MITM) ATTACK
# =============================================================================

@dataclass
class MITMAttackConfig:
    """
    Configuration for classical Man-in-the-Middle attacks.

    MITM intercepts and modifies messages in transit on specific network
    links. QKD authentication tags detect modification; classical HMAC
    can potentially be forged if the shared secret is compromised.
    """
    start_s: int = 0
    end_s: int = 600
    intercept_prob: float = 0.3          # probability of intercepting any message
    modify_prob: float = 0.8             # probability of modifying intercepted message
    shed_override_frac: float = 0.70     # shed fraction to inject in modified messages
    target_edges: Optional[List[Tuple[str, str]]] = None  # specific links (None = all)
    classical_forge_prob: float = 0.10   # prob of forging classical HMAC after modification
    label: str = "mitm"


class ClassicalMITMAttack:
    """
    Intercepts and modifies messages in transit on specific network links.

    Models an attacker with physical access to network infrastructure who
    can read and modify classical messages. Key security property:

    - QKD authentication: tag computed with quantum key → ANY modification
      causes tag mismatch → message dropped (information-theoretic security).
    - Classical HMAC: modification detected IF shared secret is intact.
      Attacker with partial key compromise can forge valid HMAC with
      probability classical_forge_prob.
    - No defense: all modifications succeed.

    This attack operates at the NETWORK LAYER — it modifies messages
    already in transit, unlike spoofing which creates new messages.
    """

    def __init__(
        self,
        env: simpy.Environment,
        rng: random.Random,
        cfg: MITMAttackConfig,
    ):
        self.env = env
        self.rng = rng
        self.cfg = cfg
        self.stats: Dict[str, int] = {
            "intercepted_count": 0,
            "modified_count": 0,
            "detected_count": 0,
            "successful_count": 0,
        }
        self._active = False

    def is_active(self, t_s: float) -> bool:
        """Check if MITM is active at time t_s."""
        return self.cfg.start_s <= t_s < self.cfg.end_s

    def should_intercept(self, src: str, dst: str) -> bool:
        """Decide whether to intercept a message on edge (src, dst)."""
        if not self._active:
            return False
        # Check if this edge is targeted
        if self.cfg.target_edges is not None:
            edge = (src, dst)
            edge_rev = (dst, src)
            if edge not in self.cfg.target_edges and edge_rev not in self.cfg.target_edges:
                return False
        return self.rng.random() < self.cfg.intercept_prob

    def try_modify_message(self, msg: Message, has_qkd_auth: bool) -> bool:
        """
        Attempt to modify an intercepted message.

        Returns True if modification succeeded (got past authentication).
        Returns False if modification was detected (tag mismatch).
        """
        self.stats["intercepted_count"] += 1

        if self.rng.random() > self.cfg.modify_prob:
            return False  # Attacker chose not to modify

        self.stats["modified_count"] += 1

        # Inject malicious control payload
        if msg.payload is None:
            msg.payload = {}
        msg.payload["mitm_modified"] = 1
        msg.payload["attack"] = True
        msg.payload["attack_label"] = self.cfg.label
        msg.payload["attack_type"] = "mitm"
        msg.payload["action"] = "shed"
        msg.payload["shed_frac"] = self.cfg.shed_override_frac

        if has_qkd_auth:
            # QKD tag: information-theoretically secure → modification ALWAYS detected
            self.stats["detected_count"] += 1
            return False  # Detected
        else:
            # Classical HMAC: can be forged with some probability
            if self.rng.random() < self.cfg.classical_forge_prob:
                self.stats["successful_count"] += 1
                msg.is_attack = True
                msg.attack_label = self.cfg.label
                return True  # Modification succeeded
            else:
                self.stats["detected_count"] += 1
                return False  # Detected

    def activate(self) -> None:
        """Start the MITM attack."""
        self._active = True

    def deactivate(self) -> None:
        """Stop the MITM attack."""
        self._active = False
