"""
network_metrics.py

Tracks per-microgrid network activity over a sliding window so the microgrid
model can couple operational energy to network behavior (bytes, auth cost,
control timeliness/drops).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Any

from .model import MsgType


@dataclass
class NetEvent:
    t_s: float
    node: str
    direction: str  # "tx" or "rx"
    size_bytes: int
    key_bits: float
    is_control: bool
    control_for_quality: bool
    delivered_on_time: bool
    delivered_late: bool
    dropped: bool
    latency_ms: Optional[float]


class NetworkActivityTracker:
    """
    Sliding-window tracker for per-node network activity.

    Events are duplicated per message for src (tx) and dst (rx) so that
    comm energy can be attributed to both endpoints. Control quality is
    derived from control messages directed to the destination node.
    """

    def __init__(self, window_s: int = 60):
        self.window_s = int(window_s)
        self._events: Dict[str, Deque[NetEvent]] = defaultdict(deque)
        self._last_poll_t: Dict[str, float] = {}
        self._last_ctrl_stats: Dict[str, Dict[str, float]] = {}

    def observe_message(self, env, msg) -> None:
        """Record a finalized message (delivered or dropped)."""
        t_s = float(env.now)

        status_val = getattr(msg, "status", "")
        if hasattr(status_val, "value"):
            status_val = status_val.value
        status_val = str(status_val)

        delivered_on_time = "delivered_on_time" in status_val
        delivered_late = "delivered_late" in status_val
        dropped = "dropped" in status_val

        latency_ms = getattr(msg, "total_latency_ms", None)
        if latency_ms is None and getattr(msg, "delivered_ms", None) is not None:
            try:
                latency_ms = int(msg.delivered_ms) - int(msg.created_ms)
            except Exception:
                latency_ms = None

        msg_type = getattr(msg, "msg_type", None)
        if hasattr(msg_type, "value"):
            msg_type = msg_type.value
        is_control = msg_type in (MsgType.CONTROL_SETPOINT.value, MsgType.PRIORITY_ACTION.value)
        payload = getattr(msg, "payload", {}) or {}
        is_attack_like = bool(getattr(msg, "is_attack", False)) or bool(payload.get("attack", False))
        control_for_quality = bool(is_control and not is_attack_like)

        key_bits_total = 0.0
        try:
            key_bits_total = float(payload.get("key_bits_spent_total", 0.0))
        except Exception:
            key_bits_total = 0.0

        key_bits_share = key_bits_total * 0.5

        src = str(getattr(msg, "src", ""))
        dst = str(getattr(msg, "dst", ""))
        size_bytes = int(getattr(msg, "size_bytes", 0))

        if src:
            self._events[src].append(NetEvent(
                t_s=t_s,
                node=src,
                direction="tx",
                size_bytes=size_bytes,
                key_bits=key_bits_share,
                is_control=is_control,
                control_for_quality=False,
                delivered_on_time=delivered_on_time,
                delivered_late=delivered_late,
                dropped=dropped,
                latency_ms=latency_ms,
            ))

        if dst:
            self._events[dst].append(NetEvent(
                t_s=t_s,
                node=dst,
                direction="rx",
                size_bytes=size_bytes,
                key_bits=key_bits_share,
                is_control=is_control,
                control_for_quality=control_for_quality,  # ignore attack-like controls in CQ
                delivered_on_time=delivered_on_time,
                delivered_late=delivered_late,
                dropped=dropped,
                latency_ms=latency_ms,
            ))

    def _prune(self, node: str, now_s: float, window_s: int) -> None:
        cutoff = now_s - float(window_s)
        dq = self._events.get(node)
        if not dq:
            return
        while dq and dq[0].t_s < cutoff:
            dq.popleft()

    def get_stats(
        self,
        *,
        node: str,
        now_s: float,
        window_s: int,
        dt_s: float,
        comm_base_kw: float,
        energy_per_byte_j: float,
        energy_per_key_bit_j: float,
        control_drop_penalty: float,
        control_on_time_deadline_ms: int,
    ) -> Dict[str, Any]:
        """
        Compute per-node network stats for the current window.

        Returns:
            comm_load_kw, comm_energy_kwh_inc, control_quality, ratios, latency, last_control_arrival_s
        """
        window_s = max(1, int(window_s))
        now_s = float(now_s)

        self._prune(node, now_s, window_s)
        events = list(self._events.get(node, []))

        # Control quality metrics
        control_events = [e for e in events if e.control_for_quality]
        total_control = len(control_events)
        on_time = sum(1 for e in control_events if e.delivered_on_time)
        late = sum(1 for e in control_events if e.delivered_late)
        dropped = sum(1 for e in control_events if e.dropped)

        if total_control:
            on_time_ratio = on_time / total_control
            drop_ratio = dropped / total_control
        else:
            # No control samples yet in window -> keep last known (or default to healthy)
            last = self._last_ctrl_stats.get(node, {})
            on_time_ratio = float(last.get("control_on_time_ratio", 1.0))
            drop_ratio = float(last.get("control_drop_ratio", 0.0))

        delivered_control_lat = [e.latency_ms for e in control_events if e.latency_ms is not None and not e.dropped]
        if delivered_control_lat:
            avg_latency_ms = float(sum(delivered_control_lat) / len(delivered_control_lat))
        else:
            last = self._last_ctrl_stats.get(node, {})
            avg_latency_ms = float(last.get("avg_control_latency_ms", float("nan")))

        # Penalize control quality by drop ratio
        control_quality = max(0.0, min(1.0, on_time_ratio - (control_drop_penalty * drop_ratio)))

        # Comms energy (window-average load + incremental energy)
        total_bytes = sum(e.size_bytes for e in events)
        total_key_bits = sum(e.key_bits for e in events)
        comm_energy_j_window = (total_bytes * float(energy_per_byte_j)) + (total_key_bits * float(energy_per_key_bit_j))
        comm_load_kw = float(comm_base_kw) + (comm_energy_j_window / float(window_s)) / 1000.0

        last_poll = self._last_poll_t.get(node, None)
        if last_poll is None:
            last_poll = now_s - float(dt_s)
        recent_energy_j = 0.0
        for e in events:
            if e.t_s > last_poll:
                recent_energy_j += (e.size_bytes * float(energy_per_byte_j)) + (e.key_bits * float(energy_per_key_bit_j))
        comm_energy_kwh_inc = recent_energy_j / 3.6e6

        # Last delivered control arrival
        delivered_controls = [e.t_s for e in control_events if not e.dropped]
        if delivered_controls:
            last_control_arrival_s = max(delivered_controls)
        else:
            last_control_arrival_s = self._last_ctrl_stats.get(node, {}).get("last_control_arrival_s")

        # update last poll
        self._last_poll_t[node] = now_s
        self._last_ctrl_stats[node] = {
            "control_quality": float(control_quality),
            "control_on_time_ratio": float(on_time_ratio),
            "control_drop_ratio": float(drop_ratio),
            "avg_control_latency_ms": float(avg_latency_ms) if avg_latency_ms == avg_latency_ms else float("nan"),
            "last_control_arrival_s": float(last_control_arrival_s) if last_control_arrival_s is not None else float("nan"),
        }

        return {
            "comm_load_kw": comm_load_kw,
            "comm_energy_kwh_inc": comm_energy_kwh_inc,
            "control_quality": control_quality,
            "control_on_time_ratio": on_time_ratio,
            "control_drop_ratio": drop_ratio,
            "avg_control_latency_ms": avg_latency_ms,
            "last_control_arrival_s": last_control_arrival_s,
            "control_total": total_control,
            "control_on_time": on_time,
            "control_late": late,
            "control_dropped": dropped,
            "control_on_time_deadline_ms": control_on_time_deadline_ms,
        }
