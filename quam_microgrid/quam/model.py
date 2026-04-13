"""
model.py 
1. Added energy_timeseries to MicrogridState
2. Added record_energy_state() method
3. Call record_energy_state() in step() method
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# -------------------------
# Message types and statuses
# -------------------------

class MsgType(str, Enum):
    TELEMETRY = "telemetry"
    CONTROL_SETPOINT = "control_setpoint"
    PRIORITY_ACTION = "priority_action"
    QAN_NOTIFY = "qan_notify"
    COVER = "cover"


class DeliveryStatus(str, Enum):
    CREATED = "created"
    DELIVERED_ON_TIME = "delivered_on_time"
    DELIVERED_LATE = "delivered_late"
    DROPPED_LOSS = "dropped_loss"
    DROPPED_NO_KEYS = "dropped_no_keys"
    DROPPED_EXPIRED = "dropped_expired"
    DROPPED_BLOCKED = "dropped_blocked"  # NEW: for defense blocking


# -------------------------
# Operational state & actions
# -------------------------

class GridMode(str, Enum):
    GRID_TIED = "grid_tied"
    ISLANDED = "islanded"
    RESTORATION = "restoration"
    QUARANTINE = "quarantine"  # NEW: for attack isolation


class ActionType(str, Enum):
    SHED_LOAD_EMERGENCY = "shed_load_emergency"
    RESTORE_LOAD = "restore_load"
    ISLAND_NOW = "island_now"
    RECONNECT_GRID = "reconnect_grid"
    OPEN_TIELINE = "open_tieline"
    CLOSE_TIELINE = "close_tieline"
    QUARANTINE = "quarantine"  # NEW


@dataclass(frozen=True)
class ControlAction:
    action_type: ActionType
    target_shed_frac: Optional[float] = None
    duration_s: Optional[int] = None
    reason: str = ""


@dataclass(frozen=True)
class ActionDecision:
    decision: str  # "allow" | "block" | "ignore"
    reason: str = ""


@dataclass
class ActionLogEntry:
    t_s: int
    msg_id: int
    src: str
    dst: str
    action_type: str
    decision: str
    decision_reason: str
    applied: bool
    note: str = ""


# -------------------------
# Message schema
# -------------------------

@dataclass
class Message:
    msg_id: int
    created_ms: int
    src: str
    dst: str
    msg_type: MsgType

    priority: int
    deadline_ms: int
    size_bytes: int

    requires_auth: bool = False
    requires_anon: bool = False

    is_attack: bool = False
    attack_label: str = ""

    # For node-level attacker: physical injection point differs from forged src
    injection_node: Optional[str] = None

    delivered_ms: Optional[int] = None
    status: DeliveryStatus = DeliveryStatus.CREATED
    drop_reason: str = ""
    total_latency_ms: Optional[int] = None
    key_wait_ms: int = 0

    payload: Dict[str, Any] = field(default_factory=dict)

    def mark_delivered(self, delivered_ms: int) -> None:
        self.delivered_ms = delivered_ms
        self.total_latency_ms = delivered_ms - self.created_ms
        if self.total_latency_ms <= self.deadline_ms:
            self.status = DeliveryStatus.DELIVERED_ON_TIME
        else:
            self.status = DeliveryStatus.DELIVERED_LATE

    def mark_dropped(self, status: DeliveryStatus, reason: str) -> None:
        self.status = status
        self.drop_reason = reason
        self.delivered_ms = None
        self.total_latency_ms = None


# -------------------------
# Microgrid parameters
# -------------------------

@dataclass
class MicrogridParams:
    name: str = "default"

    # Loads (kW)
    base_load_kw: float = 120.0
    ai_load_kw: float = 60.0
    critical_load_kw: float = 110.0

    # Generation (kW) — legacy Gaussian model
    gen_kw_mean: float = 130.0
    gen_kw_sigma: float = 15.0

    # Realistic generation model selection
    generation_model: str = "gaussian"   # "gaussian" (legacy) or "realistic"

    # Realistic DER capacities (kW rated)
    solar_capacity_kw: float = 60.0
    wind_capacity_kw: float = 40.0
    smr_capacity_kw: float = 50.0

    # Solar tuning
    solar_time_of_day_h: float = 12.0        # sim t=0 clock hour
    solar_cloud_transition_s: float = 300.0   # mean cloud state sojourn

    # Wind tuning
    wind_mean_speed_ms: float = 7.0
    wind_turbulence_intensity: float = 0.15
    wind_correlation_s: float = 30.0

    # SMR tuning
    smr_availability: float = 0.90

    # Grid-tied import capability (kW)
    import_cap_kw: float = 60.0

    # AI pulse window (seconds)
    ai_pulse_start_s: int = 200
    ai_pulse_end_s: int = 400

    # Load shedding dynamics
    max_shed_frac: float = 0.90
    shed_ramp_per_s: float = 0.25
    # Spoof plausibility guardrails
    spoof_attack_max_step_frac: float = 0.10
    spoof_unconfirmed_cap_frac: float = 0.20
    spoof_confirm_deficit_ratio: float = 0.10
    spoof_confirm_control_quality: float = 0.75
    spoof_confirm_unserved_kw: float = 2.0

    # Command handling
    staleness_ms: int = 250

    # Restoration
    restoration_duration_s: int = 20
    
    # NEW: Energy recording interval
    energy_record_interval_s: int = 1  # Record every N seconds

    # Battery Energy Storage System (BESS)
    battery_capacity_kwh: float = 100.0
    battery_init_kwh: float = 50.0
    battery_max_discharge_kw: float = 50.0
    battery_max_charge_kw: float = 50.0

    # NEW: Network/Control coupling
    control_window_s: int = 60
    control_on_time_deadline_ms: int = 500
    control_drop_penalty: float = 0.5
    # Lower gain avoids hard saturation of shed response when control quality drops.
    control_quality_shed_gain: float = 0.18
    recovery_rate_per_min: float = 0.02
    recovery_control_quality_min: float = 0.8

    # NEW: Comms energy model
    comm_base_kw: float = 0.05
    energy_per_byte_j: float = 5e-6
    energy_per_key_bit_j: float = 1e-9


# -------------------------
# QKD Infrastructure Cost Model
# -------------------------

@dataclass
class QKDInfrastructureCost:
    """
    Deployment-level cost model for QKD-secured microgrid networks.
    Based on current market prices for commercial QKD equipment.

    References:
    - IDQuantique Cerberis XGR (commercial QKD system)
    - Toshiba Quantum Key Distribution systems
    - Beijing-Shanghai QKD backbone deployment costs
    """
    # Capital costs (USD)
    qkd_link_transmitter_usd: float = 150_000.0   # per link endpoint (Alice)
    qkd_link_receiver_usd: float = 100_000.0      # per link endpoint (Bob)
    qkd_link_fiber_per_km_usd: float = 5_000.0    # dedicated quantum channel fiber
    qrng_unit_usd: float = 50_000.0               # per-node QRNG module
    classical_switch_usd: float = 5_000.0          # classical network switch per node

    # Operational power (kW)
    detector_cooling_kw: float = 0.5               # cooling per single-photon detector
    qkd_equipment_kw_per_link: float = 1.0         # total power per QKD link
    classical_equipment_kw_per_link: float = 0.1    # baseline classical network power
    qrng_kw: float = 0.05                          # QRNG module power

    # Economic parameters
    electricity_cost_per_kwh_usd: float = 0.12     # average commercial rate
    maintenance_pct_of_capex: float = 0.10          # annual maintenance as % of capex

    def compute_costs(
        self,
        num_nodes: int,
        num_qkd_links: int,
        total_fiber_km: float,
        has_qrng: bool = True,
    ) -> Dict[str, float]:
        """
        Compute full deployment cost breakdown.

        Returns dict with capex, opex, and comparison metrics.
        """
        # Capital expenditure breakdown
        tx_cost = num_qkd_links * self.qkd_link_transmitter_usd
        rx_cost = num_qkd_links * self.qkd_link_receiver_usd
        fiber_cost = total_fiber_km * self.qkd_link_fiber_per_km_usd
        qrng_cost = (num_nodes * self.qrng_unit_usd) if has_qrng else 0.0
        classical_cost = num_nodes * self.classical_switch_usd
        capex_quantum = tx_cost + rx_cost + fiber_cost + qrng_cost
        capex_classical = classical_cost + (total_fiber_km * 1_000.0)  # shared fiber cheaper
        capex_total = capex_quantum + capex_classical

        # Operational power
        qkd_power_kw = num_qkd_links * self.qkd_equipment_kw_per_link
        cooling_kw = num_qkd_links * 2 * self.detector_cooling_kw  # 2 detectors per link
        qrng_power_kw = (num_nodes * self.qrng_kw) if has_qrng else 0.0
        classical_power_kw = num_qkd_links * self.classical_equipment_kw_per_link
        total_power_kw = qkd_power_kw + cooling_kw + qrng_power_kw + classical_power_kw

        # Annual operational cost
        annual_energy_kwh = total_power_kw * 8760  # hours per year
        annual_energy_cost = annual_energy_kwh * self.electricity_cost_per_kwh_usd
        annual_maintenance = capex_total * self.maintenance_pct_of_capex
        opex_annual = annual_energy_cost + annual_maintenance

        # Comparison metrics
        classical_only_capex = capex_classical
        quantum_premium_pct = ((capex_total - classical_only_capex) / max(1, classical_only_capex)) * 100.0
        cost_per_node = capex_total / max(1, num_nodes)

        return {
            "capex_transmitters_usd": tx_cost,
            "capex_receivers_usd": rx_cost,
            "capex_fiber_usd": fiber_cost,
            "capex_qrng_usd": qrng_cost,
            "capex_classical_usd": capex_classical,
            "capex_quantum_usd": capex_quantum,
            "capex_total_usd": capex_total,
            "opex_power_kw": total_power_kw,
            "opex_qkd_power_kw": qkd_power_kw,
            "opex_cooling_kw": cooling_kw,
            "opex_qrng_power_kw": qrng_power_kw,
            "opex_classical_power_kw": classical_power_kw,
            "opex_annual_energy_kwh": annual_energy_kwh,
            "opex_annual_energy_cost_usd": annual_energy_cost,
            "opex_annual_maintenance_usd": annual_maintenance,
            "opex_annual_total_usd": opex_annual,
            "cost_per_node_usd": cost_per_node,
            "classical_only_capex_usd": classical_only_capex,
            "quantum_premium_pct": quantum_premium_pct,
            "num_nodes": num_nodes,
            "num_qkd_links": num_qkd_links,
            "total_fiber_km": total_fiber_km,
        }


# -------------------------
# NEW: Energy Time Series Record
# -------------------------

@dataclass
class EnergyRecord:
    """Single timestep energy state for plotting."""
    t_s: int
    
    # Load
    total_load_kw: float
    critical_load_kw: float
    noncritical_load_kw: float
    
    # Supply
    gen_kw: float
    import_kw: float
    import_cap_kw: float

    # Balance
    served_kw: float
    unserved_kw: float
    unserved_critical_kw: float
    unserved_noncritical_kw: float

    # Battery
    battery_kwh: float
    battery_discharge_kw: float
    battery_charge_kw: float

    # Control state
    shed_frac: float
    shed_target: float
    forced_shed_active: bool

    # Mode
    mode: str

    # Comms/control coupling
    comm_load_kw: float
    comm_energy_kwh: float
    control_quality: float
    control_on_time_ratio: float
    control_drop_ratio: float
    avg_control_latency_ms: float

    # Cumulative EENS
    eens_total_kwh: float
    eens_critical_kwh: float

    # Per-source generation (defaults so existing callers aren't broken)
    solar_kw: float = 0.0
    wind_kw: float = 0.0
    smr_kw: float = 0.0

    # Power flow (SimpleDCPowerFlow)
    line_loss_kw: float = 0.0
    voltage_violation: bool = False


# -------------------------
# SimpleDC Power Flow Model
# -------------------------

@dataclass
class PowerLine:
    """A distribution line in the microgrid."""
    from_bus: str
    to_bus: str
    resistance_ohm: float = 0.08     # line resistance (Ω) — ~0.008 Ω/km × 10km for MV distribution
    max_current_a: float = 200.0     # thermal limit (A)
    length_km: float = 5.0           # physical length
    voltage_nominal_v: float = 480.0 # nominal distribution voltage (V)


@dataclass
class PowerFlowResult:
    """Output of a DC power flow solve."""
    line_flows_kw: Dict[Tuple[str, str], float]
    line_losses_kw: Dict[Tuple[str, str], float]
    bus_voltages_pu: Dict[str, float]             # per-unit voltage
    total_loss_kw: float
    voltage_violations: List[str]                  # buses outside ±5%
    thermal_violations: List[Tuple[str, str]]      # overloaded lines


class SimpleDCPowerFlow:
    """
    Simplified DC power flow for radial microgrid distribution.

    Models:
    - Line losses via I²R (current² × resistance)
    - Voltage drop along lines (ΔV = I × R)
    - Per-unit voltage at each bus
    - Thermal limits on lines

    NOT a full AC power flow (no reactive power, no Newton-Raphson).
    Appropriate for a cybersecurity simulator where exact power flow
    is not the research focus but physical realism matters.

    The model assumes a radial (tree) or weakly meshed topology and
    uses a single-iteration DC approximation: power flow on each line
    is proportional to the net injection difference.
    """

    def __init__(self, lines: List[PowerLine], buses: List[str]):
        self.lines = lines
        self.buses = buses
        self._line_map: Dict[Tuple[str, str], PowerLine] = {}
        for line in lines:
            key = (line.from_bus, line.to_bus)
            self._line_map[key] = line
            # Bidirectional lookup
            self._line_map[(line.to_bus, line.from_bus)] = line

    def solve(self, injections: Dict[str, float]) -> PowerFlowResult:
        """
        Given net power injection at each bus (generation - load in kW),
        compute flows and losses.

        Positive injection = net generation surplus at bus.
        Negative injection = net load at bus.

        Uses simplified DC power flow: assumes voltage angles are small,
        resistance is the dominant impedance, and computes I = P / V_nom.
        """
        line_flows: Dict[Tuple[str, str], float] = {}
        line_losses: Dict[Tuple[str, str], float] = {}
        bus_voltages: Dict[str, float] = {b: 1.0 for b in self.buses}
        voltage_violations: List[str] = []
        thermal_violations: List[Tuple[str, str]] = []
        total_loss = 0.0

        # For each line, estimate power flow as the deficit at the downstream bus
        # In a radial network, power flows from surplus to deficit buses
        for line in self.lines:
            key = (line.from_bus, line.to_bus)
            # Net flow: positive means power flows from_bus → to_bus
            inj_from = injections.get(line.from_bus, 0.0)
            inj_to = injections.get(line.to_bus, 0.0)

            # Power flows toward the bus with larger deficit (more negative injection)
            flow_kw = (inj_from - inj_to) / 2.0  # Split difference

            # Current from power: I = P / V (simplified, single-phase equivalent)
            v_nom = line.voltage_nominal_v
            if v_nom > 0:
                current_a = abs(flow_kw * 1000.0) / v_nom  # kW → W → A
            else:
                current_a = 0.0

            # I²R losses
            loss_kw = (current_a ** 2 * line.resistance_ohm) / 1000.0  # W → kW
            line_flows[key] = flow_kw
            line_losses[key] = loss_kw
            total_loss += loss_kw

            # Voltage drop: ΔV = I × R (per-unit: ΔV/V_nom)
            if v_nom > 0:
                delta_v_pu = (current_a * line.resistance_ohm) / v_nom
                # Apply voltage drop to receiving bus
                bus_voltages[line.to_bus] = min(
                    bus_voltages[line.to_bus],
                    1.0 - delta_v_pu,
                )

            # Thermal limit check
            if current_a > line.max_current_a:
                thermal_violations.append(key)

        # Voltage violation check (±5%)
        for bus, v_pu in bus_voltages.items():
            if v_pu < 0.95 or v_pu > 1.05:
                voltage_violations.append(bus)

        return PowerFlowResult(
            line_flows_kw=line_flows,
            line_losses_kw=line_losses,
            bus_voltages_pu=bus_voltages,
            total_loss_kw=total_loss,
            voltage_violations=voltage_violations,
            thermal_violations=thermal_violations,
        )


# -------------------------
# Microgrid state
# -------------------------

@dataclass
class MicrogridState:
    params: MicrogridParams

    # Modes
    mode: GridMode = GridMode.GRID_TIED

    # Load shedding control state
    shed_frac: float = 0.0
    shed_target: float = 0.0

    # Accumulators
    eens_total_kwh: float = 0.0
    eens_critical_kwh: float = 0.0
    eens_noncritical_kwh: float = 0.0
    critical_outage_minutes: float = 0.0

    # Attack harm overlays
    forced_shed_until_s: int = 0
    forced_shed_frac: float = 0.0

    # Battery state
    battery_kwh: float = 0.0
    last_battery_discharge_kw: float = 0.0
    last_battery_charge_kw: float = 0.0

    # Mode timers
    restoration_until_s: int = 0
    quarantine_until_s: int = 0  # NEW

    # Logs
    action_log: List[ActionLogEntry] = field(default_factory=list)
    
    # NEW: Energy time series for plotting
    energy_timeseries: List[EnergyRecord] = field(default_factory=list)

    # NEW: Network/Control coupling state
    comm_load_kw: float = 0.0
    comm_energy_kwh: float = 0.0
    control_quality: float = 1.0
    control_on_time_ratio: float = 1.0
    control_drop_ratio: float = 0.0
    avg_control_latency_ms: float = float("nan")
    last_control_arrival_s: float = 0.0

    # Debugging / last-step state
    last_total_load_kw: float = 0.0
    last_critical_load_kw: float = 0.0
    last_noncritical_load_kw: float = 0.0
    last_served_kw: float = 0.0
    last_import_kw: float = 0.0
    last_gen_kw: float = 0.0
    last_unserved_kw: float = 0.0
    last_unserved_critical_kw: float = 0.0
    last_unserved_noncritical_kw: float = 0.0

    def __post_init__(self) -> None:
        # Initialize comm load with baseline comm power
        self.comm_load_kw = float(self.params.comm_base_kw)
        cap = max(0.0, self.params.battery_capacity_kwh)
        init = max(0.0, min(self.params.battery_init_kwh, cap))
        self.battery_kwh = init

    def _pulse_scale(self, t_s: int) -> float:
        p = self.params
        return 1.0 if (p.ai_pulse_start_s <= t_s <= p.ai_pulse_end_s) else 0.2

    def current_total_load_kw(self, t_s: int) -> float:
        p = self.params
        return p.base_load_kw + self._pulse_scale(t_s) * p.ai_load_kw

    def current_critical_load_kw(self, t_s: int) -> float:
        return max(0.0, min(self.params.critical_load_kw, self.current_total_load_kw(t_s)))

    def clamp_targets(self) -> None:
        p = self.params
        self.shed_target = max(0.0, min(self.shed_target, p.max_shed_frac))
        self.shed_frac = max(0.0, min(self.shed_frac, p.max_shed_frac))

    def apply_forced_overlay_if_active(self, t_s: int) -> None:
        if t_s <= self.forced_shed_until_s:
            self.shed_target = max(self.shed_target, self.forced_shed_frac)

    def _update_shed_actuator(self, dt_s: int) -> None:
        p = self.params
        self.clamp_targets()
        delta = self.shed_target - self.shed_frac
        max_step = p.shed_ramp_per_s * dt_s
        if abs(delta) <= max_step:
            self.shed_frac = self.shed_target
        else:
            self.shed_frac += max_step * (1.0 if delta > 0 else -1.0)
        self.clamp_targets()

    def _current_import_cap_kw(self) -> float:
        if self.mode == GridMode.GRID_TIED:
            return self.params.import_cap_kw
        if self.mode == GridMode.RESTORATION:
            return 0.5 * self.params.import_cap_kw
        # ISLANDED, QUARANTINE = no import
        return 0.0

    def _update_mode_timers(self, t_s: int) -> None:
        if self.mode == GridMode.RESTORATION and t_s >= self.restoration_until_s:
            self.mode = GridMode.GRID_TIED
        if self.mode == GridMode.QUARANTINE and t_s >= self.quarantine_until_s:
            self.mode = GridMode.GRID_TIED

    def step(self, *, t_s: int, dt_s: int, gen_kw_sample: float,
             solar_kw: float = 0.0, wind_kw: float = 0.0, smr_kw: float = 0.0,
             record_energy: bool = True, attack_label: str = "") -> None:
        """
        Advance operational state by dt_s seconds.
        
        NEW: record_energy parameter controls whether to store energy state.
        """
        self._update_mode_timers(t_s)
        forced_active = (t_s <= self.forced_shed_until_s)
        if forced_active:
            self.apply_forced_overlay_if_active(t_s)
        elif self.forced_shed_until_s > 0 or self.forced_shed_frac > 0.0:
            # Clear expired attack overlay so old forced values do not latch forever.
            self.forced_shed_until_s = 0
            self.forced_shed_frac = 0.0

        penalty = 0.0
        if self.control_quality < 1.0:
            penalty = max(0.0, self.params.control_quality_shed_gain * (1.0 - self.control_quality))

        # Recovery should run whenever forced attack overlay is inactive.
        if not forced_active:
            decay = self.params.recovery_rate_per_min * (dt_s / 60.0)
            if decay > 0:
                self.shed_target = max(0.0, self.shed_target - decay)
            if penalty > 0.0:
                # Control-quality penalty acts as a floor, not a sticky ratchet.
                self.shed_target = max(self.shed_target, penalty)

        self._update_shed_actuator(dt_s)

        total_load_kw = self.current_total_load_kw(t_s) + max(0.0, float(self.comm_load_kw))
        critical_load_kw = self.current_critical_load_kw(t_s)
        noncritical_load_kw = max(0.0, total_load_kw - critical_load_kw)

        served_demand_kw = total_load_kw * (1.0 - self.shed_frac)

        gen_kw = max(0.0, gen_kw_sample)
        import_cap_kw = self._current_import_cap_kw()
        deficit_kw = max(0.0, served_demand_kw - gen_kw)
        import_kw = min(import_cap_kw, deficit_kw)

        supply_kw = gen_kw + import_kw
        remaining_deficit_kw = max(0.0, served_demand_kw - supply_kw)

        # Battery discharge to cover remaining deficit (if available)
        discharge_kw = 0.0
        if remaining_deficit_kw > 0.0 and self.params.battery_capacity_kwh > 0.0:
            max_discharge_kw = min(
                self.params.battery_max_discharge_kw,
                (self.battery_kwh * 3600.0 / dt_s) if dt_s > 0 else 0.0,
            )
            discharge_kw = min(remaining_deficit_kw, max_discharge_kw)
            if discharge_kw > 0.0:
                self.battery_kwh -= discharge_kw * (dt_s / 3600.0)
                remaining_deficit_kw = max(0.0, remaining_deficit_kw - discharge_kw)

        # Battery charge from surplus (if any)
        charge_kw = 0.0
        if remaining_deficit_kw <= 0.0 and self.params.battery_capacity_kwh > 0.0:
            surplus_kw = max(0.0, supply_kw - served_demand_kw)
            headroom_kwh = max(0.0, self.params.battery_capacity_kwh - self.battery_kwh)
            max_charge_kw = min(
                self.params.battery_max_charge_kw,
                (headroom_kwh * 3600.0 / dt_s) if dt_s > 0 else 0.0,
            )
            charge_kw = min(surplus_kw, max_charge_kw)
            if charge_kw > 0.0:
                self.battery_kwh += charge_kw * (dt_s / 3600.0)

        served_kw = max(0.0, served_demand_kw - remaining_deficit_kw)
        unserved_kw = max(0.0, remaining_deficit_kw)

        # Allocate served power to critical first
        served_to_critical_kw = min(critical_load_kw, served_kw)
        remaining_kw = max(0.0, served_kw - served_to_critical_kw)
        served_to_noncritical_kw = min(noncritical_load_kw, remaining_kw)

        unserved_critical_kw = max(0.0, critical_load_kw - served_to_critical_kw)
        unserved_noncritical_kw = max(0.0, noncritical_load_kw - served_to_noncritical_kw)

        # Accumulate EENS (kWh) using load curtailment (shed + unserved)
        curtailed_kw = max(0.0, total_load_kw - served_kw)
        self.eens_total_kwh += curtailed_kw * (dt_s / 3600.0)
        self.eens_critical_kwh += unserved_critical_kw * (dt_s / 3600.0)
        # Noncritical curtailment = total curtailed minus critical unserved
        noncritical_curtailed_kw = max(0.0, curtailed_kw - unserved_critical_kw)
        self.eens_noncritical_kwh += noncritical_curtailed_kw * (dt_s / 3600.0)

        if unserved_critical_kw > 1e-9:
            self.critical_outage_minutes += (dt_s / 60.0)

        # Store last values for debugging/access
        self.last_total_load_kw = total_load_kw
        self.last_critical_load_kw = critical_load_kw
        self.last_noncritical_load_kw = noncritical_load_kw
        self.last_served_kw = served_kw
        self.last_import_kw = import_kw
        self.last_gen_kw = gen_kw
        self.last_unserved_kw = unserved_kw
        self.last_unserved_critical_kw = unserved_critical_kw
        self.last_unserved_noncritical_kw = unserved_noncritical_kw
        self.last_battery_discharge_kw = discharge_kw
        self.last_battery_charge_kw = charge_kw

        # NEW: Record energy state for plotting
        if record_energy and t_s % self.params.energy_record_interval_s == 0:
            self.energy_timeseries.append(EnergyRecord(
                t_s=t_s,
                total_load_kw=total_load_kw,
                critical_load_kw=critical_load_kw,
                noncritical_load_kw=noncritical_load_kw,
                gen_kw=gen_kw,
                import_kw=import_kw,
                import_cap_kw=import_cap_kw,
                served_kw=served_kw,
                unserved_kw=unserved_kw,
                unserved_critical_kw=unserved_critical_kw,
                unserved_noncritical_kw=unserved_noncritical_kw,
                battery_kwh=self.battery_kwh,
                battery_discharge_kw=discharge_kw,
                battery_charge_kw=charge_kw,
                shed_frac=self.shed_frac,
                shed_target=self.shed_target,
                forced_shed_active=(t_s <= self.forced_shed_until_s),
                mode=self.mode.value,
                comm_load_kw=float(self.comm_load_kw),
                comm_energy_kwh=float(self.comm_energy_kwh),
                control_quality=float(self.control_quality),
                control_on_time_ratio=float(self.control_on_time_ratio),
                control_drop_ratio=float(self.control_drop_ratio),
                avg_control_latency_ms=float(self.avg_control_latency_ms),
                eens_total_kwh=self.eens_total_kwh,
                eens_critical_kwh=self.eens_critical_kwh,
                solar_kw=solar_kw,
                wind_kw=wind_kw,
                smr_kw=smr_kw,
            ))

    # -------------------------
    # Network/Control coupling
    # -------------------------

    def update_comm_state(self, *, stats: Dict[str, Any], dt_s: int) -> None:
        """
        Update comm load and control quality from network stats.
        stats should be produced by NetworkActivityTracker.
        """
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

    # -------------------------
    # Action application & logging
    # -------------------------

    def log_action(
        self,
        *,
        t_s: int,
        msg: Message,
        action: ControlAction,
        decision: ActionDecision,
        applied: bool,
        note: str = "",
    ) -> None:
        self.action_log.append(ActionLogEntry(
            t_s=t_s,
            msg_id=msg.msg_id,
            src=msg.src,
            dst=msg.dst,
            action_type=action.action_type.value,
            decision=decision.decision,
            decision_reason=decision.reason,
            applied=applied,
            note=note,
        ))

    def apply_action(self, *, now_s: int, msg: Message, action: ControlAction, decision: ActionDecision) -> bool:
        self.clamp_targets()
        action_note = ""

        if decision.decision != "allow":
            self.log_action(t_s=now_s, msg=msg, action=action, decision=decision, applied=False)
            return False

        p = self.params

        if action.action_type == ActionType.SHED_LOAD_EMERGENCY:
            if msg.is_attack and action.target_shed_frac is not None:
                requested = max(0.0, min(float(action.target_shed_frac), p.max_shed_frac))
                total_load = self.last_total_load_kw
                if total_load <= 0.0:
                    total_load = self.current_total_load_kw(now_s) + max(0.0, float(self.comm_load_kw))
                supply = (
                    max(0.0, self.last_gen_kw)
                    + max(0.0, self.last_import_kw)
                    + max(0.0, self.last_battery_discharge_kw)
                )
                deficit_ratio = 0.0 if total_load <= 0.0 else max(0.0, (total_load - supply) / total_load)
                cq = max(0.0, min(1.0, float(getattr(self, "control_quality", 1.0))))
                anomaly_confirmed = (
                    deficit_ratio >= p.spoof_confirm_deficit_ratio
                    or self.last_unserved_kw >= p.spoof_confirm_unserved_kw
                    or cq <= p.spoof_confirm_control_quality
                )
                if anomaly_confirmed:
                    target = requested
                else:
                    max_step_cap = min(p.max_shed_frac, self.shed_target + p.spoof_attack_max_step_frac)
                    conservative_cap = min(p.max_shed_frac, p.spoof_unconfirmed_cap_frac)
                    target = min(requested, max_step_cap, conservative_cap)
                    if target + 1e-9 < requested:
                        action_note = (
                            f"spoof_plausibility_clip req={requested:.3f} "
                            f"applied={target:.3f} deficit={deficit_ratio:.3f} cq={cq:.3f}"
                        )
            elif action.reason == "priority_action_shed":
                total_load = self.last_total_load_kw
                if total_load <= 0.0:
                    total_load = self.current_total_load_kw(now_s) + max(0.0, float(self.comm_load_kw))
                supply = max(0.0, self.last_gen_kw) + max(0.0, self.last_import_kw)
                deficit_ratio = 0.0 if total_load <= 0.0 else max(0.0, (total_load - supply) / total_load)
                cq = max(0.0, min(1.0, float(getattr(self, "control_quality", 1.0))))
                base = 0.15 + 0.6 * (1.0 - cq) + 0.25 * deficit_ratio
                jitter = random.uniform(-0.1, 0.1)
                target = max(0.05, min(base + jitter, p.max_shed_frac))
            else:
                target = action.target_shed_frac if action.target_shed_frac is not None else min(0.7, p.max_shed_frac)
                target = max(0.0, min(float(target), p.max_shed_frac))

            # Apply target directly so spoof/priority actions do not get stuck as monotonic increases.
            self.shed_target = target

            dur = int(action.duration_s) if action.duration_s is not None else 30
            if dur > 0:
                self.forced_shed_frac = target
                self.forced_shed_until_s = max(self.forced_shed_until_s, now_s + dur)

        elif action.action_type == ActionType.RESTORE_LOAD:
            self.shed_target = max(0.0, self.shed_target - 0.5)
            if action.duration_s is not None and action.duration_s <= 0:
                self.forced_shed_until_s = 0
                self.forced_shed_frac = 0.0

        elif action.action_type == ActionType.ISLAND_NOW:
            self.mode = GridMode.ISLANDED

        elif action.action_type == ActionType.RECONNECT_GRID:
            self.mode = GridMode.RESTORATION
            self.restoration_until_s = now_s + p.restoration_duration_s

        elif action.action_type == ActionType.QUARANTINE:
            # NEW: Quarantine mode
            self.mode = GridMode.QUARANTINE
            dur = int(action.duration_s) if action.duration_s is not None else 60
            self.quarantine_until_s = now_s + dur

        elif action.action_type in (ActionType.OPEN_TIELINE, ActionType.CLOSE_TIELINE):
            pass

        self.clamp_targets()
        self.log_action(t_s=now_s, msg=msg, action=action, decision=decision, applied=True, note=action_note)
        return True


# -------------------------
# Parsing helpers
# -------------------------

def parse_action_from_message(msg: Message) -> Optional[ControlAction]:
    if msg.msg_type == MsgType.CONTROL_SETPOINT:
        if "shed_frac_target" not in msg.payload:
            return None
        try:
            target = float(msg.payload.get("shed_frac_target"))
        except (TypeError, ValueError):
            return None
        return ControlAction(
            action_type=ActionType.SHED_LOAD_EMERGENCY,
            target_shed_frac=target,
            duration_s=0,
            reason="control_setpoint"
        )

    if msg.msg_type == MsgType.PRIORITY_ACTION:
        act = msg.payload.get("action", None)
        if act is None:
            return None
        try:
            act_t = ActionType(str(act))
        except ValueError:
            return None

        if act_t == ActionType.SHED_LOAD_EMERGENCY:
            # Legitimate traffic uses "shed_frac_target"; attack traffic uses
            # "forced_shed_frac".  Default to 0.0 (no shedding) so that
            # legitimate commands don't accidentally impose 70 % forced-shed.
            frac = msg.payload.get(
                "shed_frac_target",
                msg.payload.get("forced_shed_frac", 0.0),
            )
            dur = msg.payload.get(
                "duration_s",
                msg.payload.get("harm_duration_s", 0),
            )
            try:
                frac_f = float(frac)
            except (TypeError, ValueError):
                frac_f = 0.0
            try:
                dur_i = int(dur)
            except (TypeError, ValueError):
                dur_i = 0
            return ControlAction(
                action_type=act_t,
                target_shed_frac=frac_f,
                duration_s=dur_i,
                reason="priority_action_shed"
            )

        if act_t == ActionType.RESTORE_LOAD:
            return ControlAction(action_type=act_t, reason="priority_action_restore")

        if act_t in (ActionType.ISLAND_NOW, ActionType.RECONNECT_GRID, 
                     ActionType.OPEN_TIELINE, ActionType.CLOSE_TIELINE,
                     ActionType.QUARANTINE):
            dur = msg.payload.get("duration_s", 60)
            return ControlAction(action_type=act_t, duration_s=dur, reason="priority_action_mode")

    return None


def should_ignore_as_stale(msg: Message, *, now_ms: int, staleness_ms: int) -> bool:
    if msg.delivered_ms is None:
        return True
    age_ms = now_ms - msg.created_ms
    return age_ms > staleness_ms


# =========================================================================
# Frequency Dynamics — Aggregate Swing Equation + Droop Control
# =========================================================================

@dataclass
class FrequencyDynamicsConfig:
    """
    Configuration for microgrid frequency dynamics model.

    References:
        - Kundur, "Power System Stability and Control" (1994)
        - Rocabert et al., "Control of Power Converters in AC Microgrids"
          IEEE Trans. Power Electron. 27(11), 2012
    """
    f_nominal_hz: float = 60.0
    # Inertia constants (seconds)
    h_smr_s: float = 3.0           # Synchronous machine (SMR) inertia
    h_inverter_s: float = 0.5      # Virtual synchronous generator inertia
    # Damping & droop
    d_pu_per_hz: float = 1.5       # Load-frequency damping coefficient
    droop_pct: float = 5.0         # Primary frequency droop (%)
    # Thresholds
    normal_band_hz: float = 0.5    # ±0.5 Hz normal operating band
    ufls_threshold_hz: float = 57.5 # Under-Frequency Load Shedding trip point
    ofgs_threshold_hz: float = 62.5 # Over-Frequency Generation Shedding


class FrequencyDynamics:
    """
    Aggregate swing equation model for microgrid frequency stability.

    Grid-tied: frequency held by infinite bus with stochastic ambient
               noise (σ = 0.03 Hz, Ornstein-Uhlenbeck process) modelling
               real-world interconnection frequency deviations per
               NERC BAL-001 (CPS1/CPS2).
    Islanded:  swing equation governs frequency from power imbalance.

    Physics (per-unit):
        2H · dΔf/dt = ΔP − D · Δf/f₀
    where
        H   = aggregate inertia (generation-weighted average)
        ΔP  = (P_gen − P_load) / P_rated
        D   = load–frequency damping coefficient
        Δf  = f − f₀

    Droop control adds primary response:
        ΔP_droop = −(1/R) · Δf/f₀     (R = droop fraction)
    """

    def __init__(self, cfg: Optional[FrequencyDynamicsConfig] = None,
                 p_rated_kw: float = 150.0,
                 seed: int = 0):
        self.cfg = cfg or FrequencyDynamicsConfig()
        self.p_rated = max(1.0, p_rated_kw)
        # RNG for grid-tied ambient frequency noise
        self._rng = random.Random(seed ^ 0xF8EE)  # deterministic per-node

        # State
        self.frequency_hz: float = self.cfg.f_nominal_hz
        self.delta_f: float = 0.0
        self.rocof: float = 0.0        # df/dt (Hz/s)

        # Statistics
        self.nadir_hz: float = self.cfg.f_nominal_hz
        self.zenith_hz: float = self.cfg.f_nominal_hz
        self.max_rocof: float = 0.0
        self.violation_s: float = 0.0   # Time outside normal band
        self.ufls_s: float = 0.0        # Time below UFLS threshold
        self.total_steps: int = 0

        # Time series (sampled every 5 s for plotting)
        self.history: List[Tuple[int, float, float]] = []  # (t_s, freq_hz, rocof)

    # -----------------------------------------------------------------
    def _h_aggregate(self, smr_kw: float, inv_kw: float) -> float:
        """Generation-weighted aggregate inertia constant."""
        total = max(1.0, smr_kw + inv_kw)
        h = (self.cfg.h_smr_s * smr_kw +
             self.cfg.h_inverter_s * inv_kw) / total
        return max(0.1, h)

    # -----------------------------------------------------------------
    def step(self, *, t_s: int, dt_s: float,
             gen_kw: float, load_kw: float,
             smr_kw: float = 0.0, solar_kw: float = 0.0, wind_kw: float = 0.0,
             is_islanded: bool = False, record: bool = True) -> None:
        """Advance frequency by one timestep."""
        self.total_steps += 1
        f0 = self.cfg.f_nominal_hz

        if not is_islanded:
            # Grid-tied: infinite bus restores frequency quickly
            self.delta_f *= max(0.0, 1.0 - 2.0 * dt_s)   # ~0.5 s time constant
            # Add stochastic ambient noise (Ornstein-Uhlenbeck process)
            # Models real-world interconnection frequency variations
            # observed in NERC Eastern/Western Interconnect data:
            #   σ_ambient ≈ 0.02-0.04 Hz, mean-reversion τ ≈ 4 s
            sigma_ambient = 0.03      # Hz, ambient noise std dev
            theta_ou = 0.25           # mean-reversion rate (1/τ)
            noise = self._rng.gauss(0.0, sigma_ambient * (dt_s ** 0.5))
            self.delta_f += -theta_ou * self.delta_f * dt_s + noise
            self.delta_f = max(-0.15, min(0.15, self.delta_f))  # clamp ±0.15 Hz
            self.rocof = -self.delta_f * 2.0
        else:
            h_agg = self._h_aggregate(smr_kw, solar_kw + wind_kw)

            # Per-unit power imbalance
            delta_p = (gen_kw - load_kw) / self.p_rated

            # Primary droop response
            R = max(0.01, self.cfg.droop_pct / 100.0)
            droop = -(1.0 / R) * (self.delta_f / f0)
            droop = max(-0.3, min(0.3, droop))

            # Damping
            damping = self.cfg.d_pu_per_hz * (self.delta_f / f0)

            # Swing equation
            accel = (delta_p + droop - damping) * f0 / (2.0 * h_agg)

            self.rocof = accel
            self.delta_f += accel * dt_s
            self.delta_f = max(-5.0, min(5.0, self.delta_f))

        self.frequency_hz = f0 + self.delta_f

        # Track statistics
        self.nadir_hz = min(self.nadir_hz, self.frequency_hz)
        self.zenith_hz = max(self.zenith_hz, self.frequency_hz)
        self.max_rocof = max(self.max_rocof, abs(self.rocof))
        if abs(self.delta_f) > self.cfg.normal_band_hz:
            self.violation_s += dt_s
        if self.frequency_hz < self.cfg.ufls_threshold_hz:
            self.ufls_s += dt_s

        if record and t_s % 5 == 0:
            self.history.append((t_s, self.frequency_hz, self.rocof))

    # -----------------------------------------------------------------
    def get_stats(self) -> Dict[str, float]:
        return {
            "freq_final_hz": round(self.frequency_hz, 4),
            "freq_nadir_hz": round(self.nadir_hz, 4),
            "freq_zenith_hz": round(self.zenith_hz, 4),
            "freq_max_rocof_hz_s": round(self.max_rocof, 4),
            "freq_violation_s": round(self.violation_s, 2),
            "freq_ufls_s": round(self.ufls_s, 2),
        }


# =========================================================================
# WLS State Estimator + Chi-Squared Bad-Data Detection
# =========================================================================

class WLSStateEstimator:
    """
    Weighted Least Squares DC state estimator with bad-data detection
    and stealthy FDI attack construction.

    Given measurement vector z (bus injections + line flows) and Jacobian H:
        x̂  = (H'WH)⁻¹ H'W z          (WLS estimate)
        r   = z − H x̂                  (measurement residual)
        J   = r'W r   ~ χ²(m − n)      (test statistic)

    Bad data detected when J > χ²(m−n, α).

    Stealthy FDI construction:
        a = H c  ⟹  residual unchanged  (Liu et al., IEEE TSG 2011)
        Quantum-authenticated measurements prevent injection of a.

    References:
        - Abur & Expósito, "Power System State Estimation" (2004)
        - Liu, Ning, Reiter, "False Data Injection Attacks against State
          Estimation in Electric Power Grids" (IEEE TSG, 2011)
    """

    def __init__(self, bus_names: List[str],
                 adjacency: List[Tuple[str, str]],
                 line_susceptance: float = 10.0,
                 sigma: float = 0.02):
        if not _HAS_NUMPY:
            raise RuntimeError("WLSStateEstimator requires numpy")

        self.bus_names = list(bus_names)
        self.bus_idx = {n: i for i, n in enumerate(self.bus_names)}
        self.n_bus = len(bus_names)
        self.n_state = self.n_bus - 1           # reference bus excluded
        self.adjacency_idx = [
            (self.bus_idx[a], self.bus_idx[b])
            for a, b in adjacency
            if a in self.bus_idx and b in self.bus_idx
        ]
        self.n_line = len(self.adjacency_idx)
        self.n_meas = self.n_bus + self.n_line
        self.b = line_susceptance
        self.sigma = sigma

        # Build Jacobian + precompute gain matrix
        self._build_H()
        self.W = _np.eye(self.n_meas) / (sigma ** 2)
        self.G = self.H.T @ self.W @ self.H

        # Chi-squared threshold via Wilson-Hilferty
        # Use alpha = 0.005 (z = 2.576) — tighter confidence to reduce
        # false positives from DC power-flow model mismatch in small
        # systems while retaining sensitivity to non-stealthy FDI bias.
        dof = max(1, self.n_meas - self.n_state)
        z_a = 2.576   # alpha ≈ 0.005 (one-tail)
        wh = dof * (1.0 - 2.0 / (9 * dof) + z_a * (2.0 / (9 * dof)) ** 0.5) ** 3
        self.chi2_threshold = max(wh, dof + 2.0 * dof ** 0.5)
        self.dof = dof

        # Statistics
        self.n_estimations = 0
        self.n_bad_data_detected = 0
        self.n_stealthy_bypasses = 0
        self.n_quantum_authenticated = 0
        self.total_J = 0.0
        self.max_J = 0.0

    # -----------------------------------------------------------------
    def _build_H(self) -> None:
        H = _np.zeros((self.n_meas, self.n_state))

        # Admittance matrix B (n_bus × n_bus)
        B = _np.zeros((self.n_bus, self.n_bus))
        for (i, j) in self.adjacency_idx:
            B[i, j] -= self.b
            B[j, i] -= self.b
            B[i, i] += self.b
            B[j, j] += self.b

        # Injection rows: ∂P_i/∂θ_k  (k = 1..n_bus-1, bus 0 = ref)
        for i in range(self.n_bus):
            for k in range(self.n_state):
                H[i, k] = B[i, k + 1]

        # Line-flow rows: ∂P_ij/∂θ_k = ±b
        for idx, (i, j) in enumerate(self.adjacency_idx):
            row = self.n_bus + idx
            if i > 0:
                H[row, i - 1] = self.b
            if j > 0:
                H[row, j - 1] = -self.b

        self.H = H

    # -----------------------------------------------------------------
    def estimate(self, z):
        """
        WLS estimation.
        Returns (x_hat, residual, J_statistic).
        """
        self.n_estimations += 1
        z = _np.asarray(z, dtype=float)
        try:
            x_hat = _np.linalg.solve(self.G, self.H.T @ self.W @ z)
        except _np.linalg.LinAlgError:
            x_hat = _np.zeros(self.n_state)

        r = z - self.H @ x_hat
        J = float(r.T @ self.W @ r)

        self.total_J += J
        self.max_J = max(self.max_J, J)
        return x_hat, r, J

    def is_bad_data(self, J: float) -> bool:
        """Chi-squared bad-data detection."""
        detected = J > self.chi2_threshold
        if detected:
            self.n_bad_data_detected += 1
        return detected

    # -----------------------------------------------------------------
    def construct_stealthy_fdi(self, target_state_change):
        """
        Construct stealthy FDI vector a = Hc.

        z_attacked  = z + Hc
        x̂_attacked = x̂ + c
        r_attacked  = r    (residual UNCHANGED — undetectable)
        """
        c = _np.asarray(target_state_change, dtype=float)
        self.n_stealthy_bypasses += 1
        return self.H @ c

    # -----------------------------------------------------------------
    def make_measurement_vector(self, bus_injections: Dict[str, float],
                                line_flows: Optional[Dict] = None):
        """Build measurement vector from bus/line data."""
        z = _np.zeros(self.n_meas)
        for name, val in bus_injections.items():
            if name in self.bus_idx:
                z[self.bus_idx[name]] = val
        if line_flows:
            for idx, (i, j) in enumerate(self.adjacency_idx):
                src, dst = self.bus_names[i], self.bus_names[j]
                if (src, dst) in line_flows:
                    z[self.n_bus + idx] = line_flows[(src, dst)]
        return z

    # -----------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        avg_J = self.total_J / max(1, self.n_estimations)
        return {
            "se_n_estimations": self.n_estimations,
            "se_n_bad_data_detected": self.n_bad_data_detected,
            "se_n_stealthy_bypasses": self.n_stealthy_bypasses,
            "se_n_quantum_authenticated": self.n_quantum_authenticated,
            "se_avg_residual_J": round(avg_J, 4),
            "se_max_residual_J": round(self.max_J, 4),
            "se_chi2_threshold": round(self.chi2_threshold, 4),
            "se_detection_rate": round(
                self.n_bad_data_detected / max(1, self.n_estimations), 4
            ),
        }
