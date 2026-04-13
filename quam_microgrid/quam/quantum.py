"""
Quantum augmentation layer for QuAM:
- Per-edge QKD key pools with dynamic refill driven by QBER/fidelity
- Link health model: QBER(t) schedules for disturbance windows
- Secret key fraction model: r(q) = max(0, 1 - 2 h2(q))
- Key allocation policy integrated with network.py via pre_send_hook
- Optional replay protection and nonce generation (QRNG vs non-QRNG)

Scope and intent:
- This is an abstraction, not a quantum state simulator.
- We model the operational consequence reviewers care about:
    attack/noise -> QBER up -> secret key rate down -> key scarcity -> delays/drops

Dependencies:
- simpy
- standard library
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Any, Callable
import hashlib
import hmac
import math
import os
import random
from collections import defaultdict
from enum import Enum

import simpy
import networkx as nx

from .model import Message, DeliveryStatus, MsgType

# Lazy import to avoid circular dependency; resolved at runtime.
_quantum_protocols = None

def _get_quantum_protocols():
    global _quantum_protocols
    if _quantum_protocols is None:
        from . import quantum_protocols as _qp
        _quantum_protocols = _qp
    return _quantum_protocols


# -------------------------
# Utilities
# -------------------------

Edge = Tuple[str, str]
PreAuthDecision = Tuple[str, int]
PreAuthDecider = Callable[[simpy.Environment, Message, List[str]], PreAuthDecision]


class NoiseModel(str, Enum):
    """Quantum channel noise models for QKD."""
    DEPOLARIZING = "depolarizing"
    DEPHASING = "dephasing"
    BIT_FLIP = "bit_flip"
    AMPLITUDE_DAMPING = "amplitude_damping"
    GENERIC = "generic"


def edge_key(u: str, v: str) -> Edge:
    return tuple(sorted((u, v)))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def h2(q: float) -> float:
    """
    Binary entropy h2(q) in bits.
    Defined for q in [0,1]. We clamp internally to avoid log(0).
    """
    q = clamp(q, 1e-12, 1.0 - 1e-12)
    return -(q * math.log2(q) + (1.0 - q) * math.log2(1.0 - q))

@dataclass
class FiniteKeyParameters:
    """
    Parameters for finite-key security analysis.

    In practical QKD, finite block sizes require additional privacy
    amplification, reducing the secret key rate compared to asymptotic bounds.
    """
    block_size_bits: int = 10_000_000
    security_parameter_log: int = 10
    enabled: bool = True

    def epsilon_security(self) -> float:
        return 10.0 ** (-self.security_parameter_log)

    def correction_term(self) -> float:
        """Δ ≈ 7 * sqrt(log2(2/ε) / n) (simplified)."""
        import math
        if not self.enabled or self.block_size_bits <= 0:
            return 0.0
        n = float(self.block_size_bits)
        log_term = 1.0 + self.security_parameter_log * math.log2(10)
        c = 7.0
        return c * math.sqrt(log_term / n)

    def finite_key_factor(self) -> float:
        correction = self.correction_term()
        return max(0.0, 1.0 - correction)

    def min_block_size_for_positive_rate(self, qber: float) -> int:
        import math
        r_asymptotic = max(0.001, 1.0 - 2.0 * h2(clamp(qber, 0.0, 0.5)))
        log_term = 1.0 + self.security_parameter_log * math.log2(10)
        c = 7.0
        min_n = (c / r_asymptotic) ** 2 * log_term
        return int(math.ceil(min_n))


FINITE_KEY_PRESETS = {
    "disabled": FiniteKeyParameters(enabled=False),
    "large_block": FiniteKeyParameters(block_size_bits=100_000_000, security_parameter_log=10),
    "medium_block": FiniteKeyParameters(block_size_bits=10_000_000, security_parameter_log=10),
    "small_block": FiniteKeyParameters(block_size_bits=1_000_000, security_parameter_log=10),
    "high_security": FiniteKeyParameters(block_size_bits=10_000_000, security_parameter_log=15),
}


def get_finite_key_params(preset: str = "disabled", *, block_size_bits: Optional[int] = None,
                          security_parameter_log: Optional[int] = None) -> FiniteKeyParameters:
    name = str(preset or "disabled").lower()
    base = FINITE_KEY_PRESETS.get(name, FINITE_KEY_PRESETS["disabled"])
    params = FiniteKeyParameters(
        block_size_bits=base.block_size_bits,
        security_parameter_log=base.security_parameter_log,
        enabled=base.enabled,
    )
    if block_size_bits is not None:
        params.block_size_bits = int(block_size_bits)
    if security_parameter_log is not None:
        params.security_parameter_log = int(security_parameter_log)
    return params



def entanglement_swap_qber(per_hop_qber: float, num_hops: int) -> float:
    """
    Compute effective end-to-end QBER after entanglement swapping across
    multiple relay hops (Werner state model).

    Physics: Each hop produces a Werner state with fidelity F = 1 - (4/3)q.
    Entanglement swapping (Bell state measurement at each relay) multiplies
    fidelities: F_eff = F^n for n identical hops.
    Effective QBER is recovered from: q_eff = (3/4)(1 - F_eff).

    For 1 hop: q_eff = q (no swapping needed).
    For 2 hops at q=0.01: q_eff ≈ 0.0198 (nearly doubles).
    For 4 hops at q=0.03: q_eff ≈ 0.114  (approaches abort threshold).

    References:
        Bennett et al., "Purification of Noisy Entanglement" (1996)
        Briegel et al., "Quantum Repeaters" (1998)
    """
    q = clamp(per_hop_qber, 0.0, 0.5)
    n = max(1, int(num_hops))
    if n == 1:
        return q
    # Werner fidelity per hop
    f_per_hop = max(0.0, 1.0 - (4.0 / 3.0) * q)
    # Entanglement swapping multiplies fidelities
    f_eff = f_per_hop ** n
    # Convert back to QBER
    q_eff = (3.0 / 4.0) * (1.0 - f_eff)
    return clamp(q_eff, 0.0, 0.5)


def entanglement_swap_rate_factor(num_hops: int, p_swap: float = 0.5) -> float:
    """
    Rate reduction factor from entanglement swapping.

    Each relay performs a Bell state measurement (BSM) with success probability
    p_swap. For n hops, there are (n-1) relays, so success probability
    scales as p_swap^(n-1).

    With linear optics BSM: p_swap = 0.5 (theoretical maximum).
    With deterministic BSM (future tech): p_swap ≈ 1.0.

    For 2 hops: factor = 0.5 (50% of attempts succeed).
    For 4 hops: factor = 0.125 (12.5%).
    """
    n = max(1, int(num_hops))
    if n <= 1:
        return 1.0
    return max(0.0, float(p_swap)) ** (n - 1)


def surface_code_logical_qber(
    physical_qber: float,
    code_distance: int = 3,
    p_threshold: float = 0.01,
) -> float:
    """
    Logical QBER after quantum error correction with surface codes.

    Surface codes are the leading QEC architecture for near-term quantum
    networks. They correct errors by encoding logical qubits into a 2D
    lattice of (d × d) physical qubits, where d is the code distance.

    Logical error rate (per logical operation):
        p_L ≈ 0.03 × (p_phys / p_th)^((d+1)/2)

    where:
        p_phys = physical error rate (QBER)
        p_th   = threshold (~1% for surface codes)
        d      = code distance (3, 5, 7, ...)

    Below threshold (p_phys < p_th): exponential suppression with d.
    Above threshold: QEC provides no benefit (returns physical QBER).

    Physical overhead: d² physical qubits per logical qubit.
    Rate overhead: 1/d² (fraction of qubits carrying useful information).

    References:
        Fowler et al., "Surface codes: Towards practical large-scale
        quantum computation" (2012)
        Raussendorf & Harrington, "Fault-tolerant quantum computation
        with high threshold in two dimensions" (2007)

    Examples:
        d=3: 9 physical qubits, overhead factor 1/9
            QBER=0.005 → p_L = 0.03 × 0.125 = 0.00375
            QBER=0.01  → p_L = 0.03 × 1.0   = 0.03 (at threshold)
        d=5: 25 physical qubits, overhead factor 1/25
            QBER=0.005 → p_L = 0.03 × 0.031 = 0.00094
    """
    q = clamp(physical_qber, 0.0, 0.5)
    d = max(3, int(code_distance))

    # Above threshold: QEC cannot help
    if q >= p_threshold:
        return q

    # Below threshold: exponential suppression
    ratio = q / p_threshold
    exponent = (d + 1) / 2.0
    p_logical = 0.03 * (ratio ** exponent)

    return clamp(p_logical, 0.0, q)  # Never worse than physical


def distill_fidelity_bbpssw(fidelity: float, num_rounds: int = 1) -> float:
    """
    Entanglement distillation using BBPSSW recurrence protocol.

    Takes 2^num_rounds noisy Bell pairs and produces 1 higher-fidelity pair.
    Each round: 2 pairs → 1 purified pair (50% rate cost per round).

    BBPSSW single-round formula (Bennett et al. 1996):
        F' = (F² + (1-F)²/9) / (F² + 2F(1-F)/3 + 5(1-F)²/9)

    Distillation is beneficial when F > 0.5 (above the entanglement threshold).
    Below F=0.5, the state is separable and cannot be purified.

    Rate overhead: 2^num_rounds input pairs per output pair.
        1 round:  2:1 (50% rate)
        2 rounds: 4:1 (25% rate)
        3 rounds: 8:1 (12.5% rate)

    References:
        Bennett, Brassard, Popescu, Schumacher, Smolin, Wootters,
        "Purification of Noisy Entanglement and Faithful Teleportation
        via Noisy Channels" (1996)
    """
    f = clamp(fidelity, 0.0, 1.0)
    rounds = max(0, int(num_rounds))

    for _ in range(rounds):
        if f <= 0.5:
            return f  # Below entanglement threshold, cannot purify

        # BBPSSW recurrence
        f_sq = f * f
        one_minus_f = 1.0 - f
        omf_sq = one_minus_f * one_minus_f

        numerator = f_sq + omf_sq / 9.0
        denominator = f_sq + 2.0 * f * one_minus_f / 3.0 + 5.0 * omf_sq / 9.0

        if denominator < 1e-15:
            return f
        f = numerator / denominator

    return clamp(f, 0.0, 1.0)


def distillation_rate_factor(num_rounds: int = 1) -> float:
    """Rate cost of entanglement distillation: 1/2^rounds."""
    return 1.0 / (2.0 ** max(0, int(num_rounds)))


def fidelity_to_qber(fidelity: float) -> float:
    """Convert Werner state fidelity to QBER: q = (3/4)(1 - F)."""
    return (3.0 / 4.0) * (1.0 - clamp(fidelity, 0.0, 1.0))


def qber_to_fidelity(qber: float) -> float:
    """Convert QBER to Werner state fidelity: F = 1 - (4/3)q."""
    return max(0.0, 1.0 - (4.0 / 3.0) * clamp(qber, 0.0, 0.5))


def secret_fraction_bb84(
    qber: float,
    finite_key_params: Optional[FiniteKeyParameters] = None,
    reconciliation_efficiency: float = 1.16,
) -> float:
    """
    BB84 secret key fraction using Shor-Preskill bound with CASCADE
    error correction.

    Asymptotic (one-way):
        r = max(0, 1 - h2(q) - f_EC * h2(q))

    where f_EC = 1.16 is the CASCADE reconciliation efficiency
    (Brassard & Salvail 1993).  The old ``1 - 2 h2(q)`` was the
    two-way bound — correct only for interactive reconciliation.

    Finite-key: r_finite ≈ r_inf * (1 - Δ)
    """
    q = clamp(qber, 0.0, 0.5)
    r_asymptotic = max(0.0, 1.0 - h2(q) - reconciliation_efficiency * h2(q))

    if finite_key_params is not None and finite_key_params.enabled:
        r_asymptotic *= finite_key_params.finite_key_factor()

    return max(0.0, r_asymptotic)



def fidelity_from_qber(qber: float, noise_model: NoiseModel = NoiseModel.DEPOLARIZING) -> float:
    """
    Compute fidelity from QBER for different quantum channel noise models.

    For BB84 QKD, the relationship depends on dominant noise:
    - Depolarizing (common in fiber): F = 1 - (4/3) * QBER
    - Dephasing/Bit-flip: F = 1 - 2 * QBER
    - Amplitude damping: approximate (between depolarizing and generic)
    - Generic: F = 1 - QBER (simple proxy)
    """
    q = clamp(qber, 0.0, 0.5)

    if noise_model == NoiseModel.DEPOLARIZING:
        fidelity = 1.0 - (4.0 / 3.0) * q
    elif noise_model in (NoiseModel.DEPHASING, NoiseModel.BIT_FLIP):
        fidelity = 1.0 - 2.0 * q
    elif noise_model == NoiseModel.AMPLITUDE_DAMPING:
        fidelity = 1.0 - 1.5 * q
    else:
        fidelity = 1.0 - q

    return clamp(fidelity, 0.0, 1.0)



# -------------------------
# Link health and disturbance schedules
# -------------------------
@dataclass
class QuantumConfig:
    # sensible defaults for conference scope
    base_qber: float = 0.01
    refill_bits_per_s: float = 2000.0
    init_fill_ratio: float = 0.30
    auth_cost_bits: int = 256
    anon_cost_bits: int = 128


@dataclass(frozen=True)
class QBERWindow:
    """
    A time window where the QBER is modified.

    Interpretation:
    - If absolute_qber is set, qber(t) becomes that value in the window.
    - Else, delta_qber is added to baseline in the window.

    Optional segment parameters allow piecewise variation within a window.
    If segment_count > 1 or segment_qber_std > 0, the window will be expanded
    into sub-windows with per-segment QBER values (see QuantumLinkHealth.add_window).

    Times are in seconds (simulation time).
    """
    start_s: int
    end_s: int
    delta_qber: float = 0.0
    absolute_qber: Optional[float] = None
    label: str = ""
    segment_count: int = 1
    segment_qber_std: float = 0.0
    segment_qber_min: Optional[float] = None
    segment_qber_max: Optional[float] = None


@dataclass(frozen=True)
class EaveWindow:
    """
    A time window where an eavesdropper actively intercepts qubits.

    The intercept_fraction (0-1) controls what fraction of qubits Eve
    measures.  Higher fractions are easier to detect via Ping-Pong IDS
    but yield more information to the attacker.

    Realistic intercept-resend attack: Eve intercepts a fraction of qubits,
    measures them, and forwards copies.  The measurement disturbs quantum
    states, which Ping-Pong probes detect via Bell inequality degradation.
    """
    start_s: int
    end_s: int
    intercept_fraction: float = 0.3   # 0-1: fraction of qubits Eve intercepts
    label: str = ""


@dataclass
class QuantumLinkHealth:
    """
    Maintains a baseline QBER and optional disturbance windows.

    For conference scope, QBER is the main observable. Fidelity is derived for reporting.
    Includes low-level noise and smooth rise/fall dynamics for realism.
    Also supports an abort state with hysteresis (QBER > threshold => no keygen).
    """
    baseline_qber: float = 0.01
    windows: List[QBERWindow] = field(default_factory=list)
    eve_windows: List[EaveWindow] = field(default_factory=list)
    rng: Optional[random.Random] = None
    noise_model: NoiseModel = NoiseModel.DEPOLARIZING


    # Noise + dynamics
    noise_std: float = 0.002          # small background fluctuation
    noise_tau_s: float = 30.0         # correlation time (s)
    rise_tau_s: float = 20.0          # attack rise time constant (s)
    fall_tau_s: float = 120.0         # recovery time constant (s)
    min_qber: float = 0.001           # floor to avoid zero
    max_qber: float = 0.5

    # Abort / hysteresis for key generation
    abort_qber: float = 0.11          # enter abort if qber >= this
    recover_qber: float = 0.09        # exit abort if qber <= this
    abort_hold_s: int = 60            # minimum abort duration (s)
    recover_hold_s: int = 120         # time below recover_qber before keygen resumes (s)

    _last_t_s: int = field(default=-1, init=False)
    _last_qber: float = field(default=0.01, init=False)
    _noise_state: float = field(default=0.0, init=False)
    _abort_active: bool = field(default=False, init=False)
    _abort_until_s: int = field(default=-1, init=False)
    _recover_start_s: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        self._last_qber = clamp(self.baseline_qber, self.min_qber, self.max_qber)

    def _window_target(self, t_s: int) -> float:
        q = self.baseline_qber
        for w in self.windows:
            if w.start_s <= t_s <= w.end_s:
                if w.absolute_qber is not None:
                    q = w.absolute_qber
                else:
                    q = q + w.delta_qber
        return q

    def _update_noise(self, dt_s: int) -> None:
        if self.rng is None or self.noise_std <= 0:
            return
        tau = max(self.noise_tau_s, 1e-6)
        alpha = math.exp(-float(dt_s) / tau)
        # Ornstein–Uhlenbeck style update (mean 0)
        sigma = float(self.noise_std)
        self._noise_state = (
            alpha * self._noise_state
            + math.sqrt(max(0.0, 1.0 - alpha * alpha)) * sigma * self.rng.gauss(0.0, 1.0)
        )

    def _update_abort_state(self, t_s: int, q: float) -> None:
        t_s = int(t_s)
        if self._abort_active:
            if self._abort_until_s >= 0 and t_s < self._abort_until_s:
                return
            # Hold expired; require QBER to stay low for a recovery window
            if q <= self.recover_qber:
                if self._recover_start_s < 0:
                    self._recover_start_s = t_s
                if (t_s - self._recover_start_s) >= int(self.recover_hold_s):
                    self._abort_active = False
                    self._abort_until_s = -1
                    self._recover_start_s = -1
            else:
                # QBER rose again; reset recovery timer
                self._recover_start_s = -1
        else:
            if q >= self.abort_qber:
                self._abort_active = True
                self._abort_until_s = t_s + int(self.abort_hold_s)
                self._recover_start_s = -1

    def qber_at(self, t_s: int) -> float:
        t_s = int(t_s)
        if self._last_t_s < 0:
            # Initialize on first call
            self._last_t_s = t_s
            if self.rng is not None and self.noise_std > 0:
                self._noise_state = self.rng.gauss(0.0, self.noise_std)
            target = self._window_target(t_s) + self._noise_state
            self._last_qber = clamp(target, self.min_qber, self.max_qber)
            self._update_abort_state(t_s, self._last_qber)
            return self._last_qber

        dt = t_s - self._last_t_s
        if dt <= 0:
            return self._last_qber

        self._update_noise(dt)
        target = self._window_target(t_s) + self._noise_state
        target = clamp(target, self.min_qber, self.max_qber)

        tau = self.rise_tau_s if target > self._last_qber else self.fall_tau_s
        alpha = 1.0 - math.exp(-float(dt) / max(tau, 1e-6))
        self._last_qber = self._last_qber + alpha * (target - self._last_qber)
        self._last_t_s = t_s
        self._last_qber = clamp(self._last_qber, self.min_qber, self.max_qber)
        self._update_abort_state(t_s, self._last_qber)
        return self._last_qber

    def fidelity_at(self, t_s: int) -> float:
        return fidelity_from_qber(self.qber_at(t_s), self.noise_model)

    def abort_active(self) -> bool:
        return bool(self._abort_active)

    def add_window(self, window: QBERWindow) -> None:
        # Expand into segments if requested to avoid flat QBER within a long window.
        if window.segment_count <= 1 and window.segment_qber_std <= 0:
            self.windows.append(window)
            return

        total_dur = int(window.end_s) - int(window.start_s)
        if total_dur <= 0:
            self.windows.append(window)
            return

        n_seg = max(1, int(window.segment_count))
        seg_len = max(1, total_dur // n_seg)
        rng = self.rng or random.Random(0)

        # Convert delta to absolute for segment sampling (baseline used for reference).
        base_abs = window.absolute_qber if window.absolute_qber is not None else (self.baseline_qber + window.delta_qber)

        start = int(window.start_s)
        for i in range(n_seg):
            end = int(window.end_s) if i == n_seg - 1 else min(int(window.end_s), start + seg_len)
            q = base_abs
            if window.segment_qber_std > 0:
                q += rng.gauss(0.0, float(window.segment_qber_std))
            if window.segment_qber_min is not None:
                q = max(float(window.segment_qber_min), q)
            if window.segment_qber_max is not None:
                q = min(float(window.segment_qber_max), q)
            q = clamp(q, self.min_qber, self.max_qber)

            label = window.label
            if label and n_seg > 1:
                label = f"{label}:seg{i+1}/{n_seg}"

            self.windows.append(QBERWindow(
                start_s=start,
                end_s=end,
                absolute_qber=q,
                label=label,
            ))
            start = end

    def add_eve_window(self, window: EaveWindow) -> None:
        """Register an eavesdropping window on this link."""
        self.eve_windows.append(window)

    def eve_intercept_at(self, t_s: int) -> float:
        """Return the maximum Eve intercept fraction active at time *t_s*."""
        frac = 0.0
        for w in self.eve_windows:
            if w.start_s <= t_s <= w.end_s:
                frac = max(frac, w.intercept_fraction)
        return frac


# -------------------------
# QKD key pool
# -------------------------

@dataclass
class QKDLinkParameters:
    """
    Physical parameters for a QKD link.

    NOTE:
    - We model distance-dependent channel loss via standard fiber attenuation.
    - Detector efficiency is included in total_efficiency() for completeness.
    - distance_factor() intentionally returns channel_transmittance only, so the
      base_refill_bits_per_s can already include detector effects if desired.
    """
    distance_km: float = 10.0
    fiber_loss_db_per_km: float = 0.2  # SMF-28 @ 1550nm

    detector_efficiency: float = 0.15
    dark_count_rate_per_ns: float = 1e-7

    source_rate_ghz: float = 1.0
    mean_photon_number: float = 0.1

    def channel_transmittance(self) -> float:
        """η = 10^(-αL/10)."""
        return 10.0 ** (-self.fiber_loss_db_per_km * self.distance_km / 10.0)

    def total_efficiency(self) -> float:
        """Include detector efficiency for reference."""
        return self.channel_transmittance() * self.detector_efficiency

    def distance_factor(self) -> float:
        """Normalized distance factor used to scale the base key rate."""
        return self.channel_transmittance()

    def max_practical_distance_km(self, min_rate_bps: float = 100.0) -> float:
        """Approximate max distance for a given key-rate threshold."""
        import math
        base_rate = 1e6
        ratio = min_rate_bps / (base_rate * max(self.detector_efficiency, 1e-12))
        if ratio >= 1.0:
            return 0.0
        return -10.0 / max(self.fiber_loss_db_per_km, 1e-9) * math.log10(ratio)


@dataclass
class QKDKeyPool:
    """
    Per-edge key pool (bits).

    The pool refills continuously, but the refill rate is multiplied by the secret fraction
    r(qber(t)). When r=0, the link yields no secret key (effective abort regime).

    IMPROVED: Supports key reservation for high-priority messages and per-source
    consumption rate limiting to defend against key exhaustion attacks.
    """
    capacity_bits: int
    base_refill_bits_per_s: float  # rate at ~0 km
    init_fill_ratio: float = 0.30

    # Physical link params (distance-dependent loss)
    link_params: QKDLinkParameters = field(default_factory=QKDLinkParameters)

    # IMPROVED: key reservation for critical messages.
    # Non-critical traffic can only consume above this floor.
    reservation_ratio: float = 0.0  # 0.0 disabled, 0.3 reserves 30%

    # IMPROVED: per-source consumption tracking and caps.
    enable_source_rate_limit: bool = False
    source_rate_limit_bits_per_s: float = 1000.0
    source_rate_window_s: int = 5

    level_bits: float = field(init=False)
    # Accounting counters for sanity checks / invariants.
    initial_level_bits: float = field(init=False, default=0.0)
    total_added_bits: float = field(init=False, default=0.0)     # stored in pool
    total_spilled_bits: float = field(init=False, default=0.0)   # generated but could not be stored (capacity cap)
    total_consumed_bits: float = field(init=False, default=0.0)  # successfully consumed
    total_failed_consume_bits: float = field(init=False, default=0.0)
    failed_consume_count: int = field(init=False, default=0)

    # IMPROVED: additional defense telemetry.
    reserved_saves: int = field(init=False, default=0)
    source_rate_blocks: int = field(init=False, default=0)
    emergency_grants: int = field(init=False, default=0)

    # Internal source-consumption window: src -> [(t_s, bits)]
    _source_consumption: Dict[str, List[Tuple[int, float]]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.level_bits = float(self.capacity_bits) * clamp(self.init_fill_ratio, 0.0, 1.0)
        # Snapshot initial level for conservation checks.
        self.initial_level_bits = float(self.level_bits)
        self.total_added_bits = 0.0
        self.total_spilled_bits = 0.0
        self.total_consumed_bits = 0.0
        self.total_failed_consume_bits = 0.0
        self.failed_consume_count = 0
        self.reserved_saves = 0
        self.source_rate_blocks = 0
        self.emergency_grants = 0
        self._source_consumption = {}

    def reserved_floor_bits(self) -> float:
        """Minimum pool level reserved for critical traffic."""
        return float(self.capacity_bits) * clamp(self.reservation_ratio, 0.0, 0.9)

    def available_for_normal(self) -> float:
        """Bits available for non-critical traffic."""
        return max(0.0, self.level_bits - self.reserved_floor_bits())

    def distance_adjusted_refill_rate(self) -> float:
        """Base refill rate scaled by link distance."""
        return float(self.base_refill_bits_per_s) * float(self.link_params.distance_factor())

    def add_bits(self, bits: float) -> None:
        inc = max(0.0, float(bits))
        if inc <= 0.0:
            return
        pre = float(self.level_bits)
        cap = float(self.capacity_bits)
        post = min(cap, pre + inc)
        effective = max(0.0, post - pre)
        spilled = max(0.0, inc - effective)
        self.level_bits = post
        self.total_added_bits += effective
        self.total_spilled_bits += spilled

    def can_consume(self, bits: float) -> bool:
        return self.level_bits >= bits

    def can_consume_priority(self, bits: float, is_critical: bool) -> bool:
        """Priority-aware sufficiency check."""
        if self.reservation_ratio <= 0.0 or is_critical:
            return self.level_bits >= bits
        return self.available_for_normal() >= bits

    def consume(self, bits: float) -> bool:
        b = float(bits)
        if b <= 0:
            return True
        if self.level_bits + 1e-12 >= b:
            self.level_bits -= b
            self.total_consumed_bits += b
            return True
        self.total_failed_consume_bits += b
        self.failed_consume_count += 1
        return False

    def consume_priority(self, bits: float, is_critical: bool) -> bool:
        """Priority-aware consumption that enforces reservation floor."""
        b = float(bits)
        if b <= 0:
            return True

        if self.reservation_ratio > 0.0 and not is_critical:
            if self.level_bits - b < self.reserved_floor_bits() - 1e-12:
                self.reserved_saves += 1
                self.total_failed_consume_bits += b
                self.failed_consume_count += 1
                return False

        if self.level_bits + 1e-12 >= b:
            self.level_bits -= b
            self.total_consumed_bits += b
            return True

        self.total_failed_consume_bits += b
        self.failed_consume_count += 1
        return False

    def check_source_rate(self, t_s: int, src: str, bits: float) -> bool:
        """
        Return True if source is within key-consumption rate budget.
        """
        if not self.enable_source_rate_limit:
            return True

        cutoff = int(t_s) - int(self.source_rate_window_s)
        if src in self._source_consumption:
            self._source_consumption[src] = [
                (t, b) for (t, b) in self._source_consumption[src] if t >= cutoff
            ]

        recent_bits = sum(b for _, b in self._source_consumption.get(src, []))
        allowed_bits = float(self.source_rate_limit_bits_per_s) * float(self.source_rate_window_s)
        if recent_bits + float(bits) > allowed_bits:
            self.source_rate_blocks += 1
            return False
        return True

    def record_source_consumption(self, t_s: int, src: str, bits: float) -> None:
        """Record post-consumption usage for source-rate accounting."""
        if not self.enable_source_rate_limit:
            return
        if src not in self._source_consumption:
            self._source_consumption[src] = []
        self._source_consumption[src].append((int(t_s), float(bits)))


# -------------------------
# Quantum RNG (QRNG)
# -------------------------

@dataclass
class QuantumRNG:
    """
    Simple QRNG model with fixed bit rate and finite buffer.

    This acts as a service layer for nonce generation. If the QRNG buffer is
    exhausted, callers can fall back to a weaker PRNG.
    """
    rate_bits_per_s: float = 1000.0
    capacity_bits: float = 8000.0

    level_bits: float = 0.0
    last_t_s: int = 0

    # Stats
    bits_generated_total: float = 0.0
    bits_used_total: float = 0.0

    def _refill(self, t_s: int) -> None:
        t_s = int(t_s)
        if t_s <= self.last_t_s:
            return
        dt = float(t_s - self.last_t_s)
        add = max(0.0, float(self.rate_bits_per_s)) * dt
        if add > 0:
            self.level_bits = min(float(self.capacity_bits), self.level_bits + add)
            self.bits_generated_total += add
        self.last_t_s = t_s

    def can_consume(self, bits: float, t_s: int) -> bool:
        self._refill(t_s)
        return self.level_bits >= float(bits)

    def consume(self, bits: float, t_s: int) -> bool:
        b = float(bits)
        if b <= 0:
            return True
        self._refill(t_s)
        if self.level_bits + 1e-12 >= b:
            self.level_bits -= b
            self.bits_used_total += b
            return True
        return False


# -------------------------
# Nonce and replay protection
# -------------------------

@dataclass
class NonceManager:
    """
    Nonce generation and replay cache.

    - If use_qrng=True, nonces are sampled uniformly in [0, 2^nonce_bits).
    - If use_qrng=False, nonces may repeat more easily (we simulate this by using a smaller internal space).
    """
    rng: random.Random
    nonce_bits: int = 64
    use_qrng: bool = True
    replay_cache_size: int = 50_000
    qrng: Optional[QuantumRNG] = None

    _seen: Dict[Tuple[str, str, int], int] = field(default_factory=dict)  # (src,dst,nonce)->time_s
    last_quality: str = "qrng"
    last_qrng_pool_bits: float = float("nan")
    qrng_fallbacks: int = 0
    qrng_shortage_events: int = 0

    def gen_nonce(self, t_s: int) -> int:
        if self.nonce_bits <= 0:
            return 0
        if self.use_qrng and self.qrng is not None:
            if self.qrng.consume(self.nonce_bits, t_s):
                self.last_quality = "qrng"
                self.last_qrng_pool_bits = float(self.qrng.level_bits)
                return self.rng.getrandbits(self.nonce_bits)
            # QRNG starved: fall back to weak PRNG
            self.qrng_shortage_events += 1
        # Non-QRNG or QRNG depleted: smaller effective space increases collision probability
        # Non-QRNG: smaller effective space increases collision probability
        effective_bits = max(8, self.nonce_bits // 4)
        self.last_quality = "weak"
        self.qrng_fallbacks += 1
        if self.qrng is not None:
            self.last_qrng_pool_bits = float(self.qrng.level_bits)
        return self.rng.getrandbits(effective_bits)

    def is_replay(self, src: str, dst: str, nonce: int) -> bool:
        return (src, dst, int(nonce)) in self._seen

    def remember(self, src: str, dst: str, nonce: int, t_s: int) -> None:
        key = (src, dst, int(nonce))
        self._seen[key] = int(t_s)
        if len(self._seen) > self.replay_cache_size:
            # light eviction: remove some oldest entries
            # (conference scope, no need for perfect LRU)
            items = sorted(self._seen.items(), key=lambda kv: kv[1])
            for (k, _) in items[: max(1, self.replay_cache_size // 10)]:
                self._seen.pop(k, None)

    def qrng_stats(self) -> Dict[str, float]:
        if self.qrng is None:
            return {
                "qrng_enabled": 0,
                "qrng_rate_bits_per_s": float("nan"),
                "qrng_capacity_bits": float("nan"),
                "qrng_pool_bits": float("nan"),
                "qrng_bits_generated_total": float("nan"),
                "qrng_bits_used_total": float("nan"),
                "qrng_fallbacks": float(self.qrng_fallbacks),
                "qrng_shortage_events": float(self.qrng_shortage_events),
            }
        return {
            "qrng_enabled": 1,
            "qrng_rate_bits_per_s": float(self.qrng.rate_bits_per_s),
            "qrng_capacity_bits": float(self.qrng.capacity_bits),
            "qrng_pool_bits": float(self.qrng.level_bits),
            "qrng_bits_generated_total": float(self.qrng.bits_generated_total),
            "qrng_bits_used_total": float(self.qrng.bits_used_total),
            "qrng_fallbacks": float(self.qrng_fallbacks),
            "qrng_shortage_events": float(self.qrng_shortage_events),
        }


# -------------------------
# Quantum augmentation manager
# -------------------------

ROTATION_POLICIES = {
    "none": {"rotate_every_msgs": 0, "rotate_bits": 0},
    "conservative": {"rotate_every_msgs": 100, "rotate_bits": 256},
    "moderate": {"rotate_every_msgs": 50, "rotate_bits": 384},
    "aggressive": {"rotate_every_msgs": 20, "rotate_bits": 512},
    "paranoid": {"rotate_every_msgs": 10, "rotate_bits": 768},
}

def apply_rotation_policy(key_policy: "KeyPolicy", policy: str) -> None:
    """Apply a named rotation policy to a KeyPolicy instance."""
    name = str(policy or "none").lower()
    cfg = ROTATION_POLICIES.get(name, ROTATION_POLICIES["none"])
    key_policy.rotate_every_msgs = int(cfg.get("rotate_every_msgs", 0))
    key_policy.rotate_bits = int(cfg.get("rotate_bits", 0))


@dataclass
class KeyPolicy:
    """
    Key usage policy.

    tag_bits + nonce_bits are charged per hop for authenticated messages.
    encryption_key_bits models symmetric encryption (AES-256) key consumption
    per message — enables the simulator to track confidentiality coverage.
    rotate_bits is charged when a rotation event occurs (session rekey).
    verify_cost_factor models additional consumption for verification intensity.

    Total per-hop cost = tag_bits + nonce_bits + encryption_key_bits
                       = 256 + 64 + 256 = 576 bits (with encryption enabled)
    """
    tag_bits: int = 256
    nonce_bits: int = 64

    # Encryption key consumption (models AES-256 symmetric encryption)
    # Set to 0 to disable encryption tracking (auth-only mode)
    encryption_key_bits: int = 256

    # Rotation
    rotate_every_msgs: int = 0          # 0 disables rotation
    rotate_bits: int = 512

    # Verification cost scaling:
    # 1.0 means baseline, >1 means more expensive verification posture
    verify_cost_factor: float = 1.0

    # Wait policy
    max_key_wait_ms: int = 2000
    key_wait_tick_ms: int = 50          # granularity of waiting in pre_send_hook


class QuantumAugmentation:
    """
    Owns:
    - QKD key pools per edge
    - QBER/fidelity health per edge
    - Nonce manager (optional)
    - Refill processes per edge

    Exposes:
    - pre_send_hook(env, msg, path_nodes) -> simpy.Event
      compatible with NetworkSim(pre_send_hook=...)
    """

    def __init__(
        self,
        env: simpy.Environment,
        rng: random.Random,
        *,
        key_policy: KeyPolicy,
        default_pool: QKDKeyPool,
        default_health: QuantumLinkHealth,
        per_edge_pool: Optional[Dict[Edge, QKDKeyPool]] = None,
        per_edge_health: Optional[Dict[Edge, QuantumLinkHealth]] = None,
        per_edge_distance_km: Optional[Dict[Edge, float]] = None,
        finite_key_params: Optional[FiniteKeyParameters] = None,
        preauth_decider: Optional[PreAuthDecider] = None,
        enable_replay_protection: bool = True,
        use_qrng_nonces: bool = True,
        qrng_rate_bits_per_s: float = 1000.0,
        qrng_capacity_bits: float = 8000.0,
        # IMPROVED: quantum-layer traffic shaping under key stress.
        enable_priority_reservation: bool = False,
        reservation_ratio: float = 0.30,
        enable_source_key_rate_limit: bool = False,
        source_key_rate_bits_per_s: float = 1000.0,
        # IMPROVED: emergency reduced-tag mode for critical control.
        enable_emergency_mode: bool = False,
        emergency_threshold_ratio: float = 0.10,
        emergency_tag_bits: int = 64,
        # V2: Multi-protocol quantum layer
        quantum_protocol_config: Optional[Any] = None,
        # V3: Dual authentication model
        auth_model: str = "per_hop",  # "per_hop" or "e2e_relay"
        graph: Optional[Any] = None,  # networkx.Graph for E2E path computation
        # V4: error-correction / repeater tuning
        qec_code_distance: int = 3,
        e2e_distillation_rounds: int = 1,
        e2e_swap_success_prob: float = 0.5,
        # V5: QTLS-bound one-time control token for PRIORITY_ACTION auth
        enable_quantum_control_auth: bool = False,
        quantum_control_token_ttl_ms: int = 1500,
        # V6: Realistic quantum auth bypass probability.
        # Models implementation imperfections:  timing side-channels,
        # partial key leakage via photon-number-splitting, Trojan-horse
        # attacks on QKD hardware, or finite-key statistical margin.
        # Set to 0.0 for information-theoretic ideal;  realistic range
        # is 0.01 – 0.05 (1–5%).
        quantum_auth_bypass_prob: float = 0.0,
    ):
        self.env = env
        self.rng = rng
        self.key_policy = key_policy
        self._auth_model = str(auth_model) if auth_model in ("per_hop", "e2e_relay") else "per_hop"
        self._graph = graph  # networkx graph (needed for e2e_relay)

        self.pools: Dict[Edge, QKDKeyPool] = {}
        self.health: Dict[Edge, QuantumLinkHealth] = {}

        # E2E relay pools: virtual node-pair key pools
        self._e2e_pools: Dict[Edge, QKDKeyPool] = {}
        self._e2e_path_cache: Dict[Edge, List[Edge]] = {}  # pair -> list of edge keys
        self._e2e_refill_procs_started: Dict[Edge, bool] = {}

        self._default_pool_template = default_pool
        self._default_health_template = default_health

        self._per_edge_pool = per_edge_pool or {}
        self._per_edge_health = per_edge_health or {}
        self._edge_distances: Dict[Edge, float] = per_edge_distance_km or {}
        self.finite_key_params: Optional[FiniteKeyParameters] = finite_key_params
        self.preauth_decider = preauth_decider
        # Quantum-layer defense switches.
        self._enable_priority_reservation = bool(enable_priority_reservation)
        self._reservation_ratio = float(reservation_ratio) if enable_priority_reservation else 0.0
        self._enable_source_key_rate_limit = bool(enable_source_key_rate_limit)
        self._source_key_rate_bits_per_s = float(source_key_rate_bits_per_s)
        self._enable_emergency_mode = bool(enable_emergency_mode)
        self._emergency_threshold_ratio = float(emergency_threshold_ratio)
        self._emergency_tag_bits = int(emergency_tag_bits)
        self._qec_code_distance = max(1, int(qec_code_distance))
        self._distillation_rounds = max(0, int(e2e_distillation_rounds))
        self._e2e_swap_success_prob = clamp(float(e2e_swap_success_prob), 0.0, 1.0)
        self._enable_quantum_control_auth = bool(enable_quantum_control_auth)
        self._quantum_control_token_ttl_ms = max(100, int(quantum_control_token_ttl_ms))
        self._quantum_control_secret = f"{self.rng.getrandbits(128):032x}"
        self._quantum_auth_bypass_prob = clamp(float(quantum_auth_bypass_prob), 0.0, 1.0)

        if self._enable_priority_reservation:
            self._default_pool_template.reservation_ratio = self._reservation_ratio
        if self._enable_source_key_rate_limit:
            self._default_pool_template.enable_source_rate_limit = True
            self._default_pool_template.source_rate_limit_bits_per_s = self._source_key_rate_bits_per_s

        qrng = QuantumRNG(
            rate_bits_per_s=qrng_rate_bits_per_s,
            capacity_bits=qrng_capacity_bits,
        ) if use_qrng_nonces else None

        self.nonce_mgr = NonceManager(
            rng=rng,
            nonce_bits=key_policy.nonce_bits,
            use_qrng=use_qrng_nonces,
            qrng=qrng,
        ) if enable_replay_protection else None

        # Rotation counters (edge-level)
        self._edge_msg_counter: Dict[Edge, int] = {}

        # Per-edge RNG for link noise
        self._edge_rngs: Dict[Edge, random.Random] = {}

        # Rotation stats
        self.rotation_stats = {
            "total_rotations": 0,
            "rotation_bits_consumed": 0,
            "rotations_by_edge": defaultdict(int),
        }

        # Background refill processes are launched when edges are registered
        self._refill_procs_started: Dict[Edge, bool] = {}

        # V2: Multi-protocol quantum layer (Quantum-TLS, E91, Ping-Pong IDS)
        self._protocol_config = quantum_protocol_config  # QuantumProtocolConfig or None
        self._qtls = None
        self._e91 = None
        self._pingpong_ids = None
        if quantum_protocol_config is not None:
            qp = _get_quantum_protocols()
            cfg = quantum_protocol_config
            if cfg.enable_pingpong_ids:
                self._pingpong_ids = qp.PingPongIDS(
                    probe_interval_s=cfg.ids_probe_interval_s,
                    variant=cfg.ids_variant,
                )
            self._e91 = qp.E91KeyDistribution(
                bell_test_fraction=cfg.e91_bell_test_fraction,
            )
            self._qtls = qp.QuantumTLS(
                config=cfg.qtls_config,
            )

        # Protocol usage stats
        self.protocol_stats: Dict[str, int] = {
            "bb84_used": 0,
            "e91_used": 0,
            "kak_used": 0,
            "qtls_used": 0,
            "qtls_fallback_e91": 0,
            "qtls_fallback_bb84": 0,
            "classical_used": 0,
            "pingpong_probes": 0,
            "pingpong_detections": 0,
            "control_tokens_attached": 0,
            "control_tokens_verified": 0,
            "control_tokens_rejected": 0,
        }

    def now_s(self) -> int:
        return int(self.env.now)

    def _get_edge_distance(self, ek: Edge) -> float:
        """Return configured distance (km) for an edge, or default."""
        if ek in self._edge_distances:
            return float(self._edge_distances[ek])
        default_lp = getattr(self._default_pool_template, "link_params", None)
        if default_lp is not None:
            return float(default_lp.distance_km)
        return 10.0

    def register_path_edges(self, path_nodes: List[str]) -> List[Edge]:
        edges: List[Edge] = []
        for i in range(len(path_nodes) - 1):
            ek = edge_key(path_nodes[i], path_nodes[i + 1])
            edges.append(ek)
            self._ensure_edge_initialized(ek)
        return edges

    def _ensure_edge_initialized(self, ek: Edge) -> None:
        if ek not in self.pools:
            # Clone pool template for edge
            src = self._per_edge_pool.get(ek, None)
            if src is None:
                tmpl = self._default_pool_template
                dist_km = self._get_edge_distance(ek)
                link_params = replace(tmpl.link_params, distance_km=dist_km)
                pool = QKDKeyPool(
                    capacity_bits=tmpl.capacity_bits,
                    base_refill_bits_per_s=tmpl.base_refill_bits_per_s,
                    init_fill_ratio=tmpl.init_fill_ratio,
                    link_params=link_params,
                    reservation_ratio=self._reservation_ratio,
                    enable_source_rate_limit=self._enable_source_key_rate_limit,
                    source_rate_limit_bits_per_s=self._source_key_rate_bits_per_s,
                )
            else:
                dist_km = self._get_edge_distance(ek)
                link_params = replace(src.link_params, distance_km=dist_km)
                pool = QKDKeyPool(
                    capacity_bits=src.capacity_bits,
                    base_refill_bits_per_s=src.base_refill_bits_per_s,
                    init_fill_ratio=src.init_fill_ratio,
                    link_params=link_params,
                    reservation_ratio=self._reservation_ratio,
                    enable_source_rate_limit=self._enable_source_key_rate_limit,
                    source_rate_limit_bits_per_s=self._source_key_rate_bits_per_s,
                )
            self.pools[ek] = pool

        if ek not in self.health:
            src_h = self._per_edge_health.get(ek, None)
            if src_h is None:
                tmpl_h = self._default_health_template
                # Per-edge RNG for realistic noise
                if ek not in self._edge_rngs:
                    self._edge_rngs[ek] = random.Random(self.rng.randint(0, 2**31 - 1))
                h = QuantumLinkHealth(
                    baseline_qber=tmpl_h.baseline_qber,
                    windows=list(tmpl_h.windows),
                    rng=self._edge_rngs[ek],
                    noise_model=tmpl_h.noise_model,
                    noise_std=tmpl_h.noise_std,
                    noise_tau_s=tmpl_h.noise_tau_s,
                    rise_tau_s=tmpl_h.rise_tau_s,
                    fall_tau_s=tmpl_h.fall_tau_s,
                    min_qber=tmpl_h.min_qber,
                    max_qber=tmpl_h.max_qber,
                )
            else:
                if ek not in self._edge_rngs:
                    self._edge_rngs[ek] = random.Random(self.rng.randint(0, 2**31 - 1))
                h = QuantumLinkHealth(
                    baseline_qber=src_h.baseline_qber,
                    windows=list(src_h.windows),
                    rng=self._edge_rngs[ek],
                    noise_model=src_h.noise_model,
                    noise_std=src_h.noise_std,
                    noise_tau_s=src_h.noise_tau_s,
                    rise_tau_s=src_h.rise_tau_s,
                    fall_tau_s=src_h.fall_tau_s,
                    min_qber=src_h.min_qber,
                    max_qber=src_h.max_qber,
                )
            self.health[ek] = h

        if ek not in self._edge_msg_counter:
            self._edge_msg_counter[ek] = 0

        if not self._refill_procs_started.get(ek, False):
            self._refill_procs_started[ek] = True
            self.env.process(self._refill_process(ek))

    def add_qber_window(self, ek: Edge, window: QBERWindow) -> None:
        self._ensure_edge_initialized(ek)
        self.health[ek].add_window(window)

    def add_eve_window(self, ek: Edge, window: EaveWindow) -> None:
        """Schedule an eavesdropping window on a quantum link."""
        self._ensure_edge_initialized(ek)
        self.health[ek].add_eve_window(window)

    def _channel_condition_factor(self, ek: Edge, t_s: int) -> float:
        """
        Dynamic channel condition model — time-varying degradation from
        environmental effects not captured by QBER monitoring alone.

        Models three real-world phenomena:
          1. **Diurnal temperature drift**: Fiber refractive index varies
             with temperature (dn/dT ≈ 1e-5 /°C), causing alignment drift
             in free-space segments and splice loss variations.
             Modelled as slow sinusoidal (period ~ 24 h, amplitude ±5%).
          2. **Turbulence / vibration bursts**: Short-duration coupling
             loss spikes from construction, seismic micro-tremors, or
             HVAC vibration near fiber runs.  Modelled as random bursts
             (Poisson, λ=0.001/s, duration 5-20 s, -10 to -30% throughput).
          3. **Equipment calibration drift**: Detector dark-count rates
             and timing jitter increase between calibration cycles.
             Linear ramp of ~2% degradation per 1000 s, resetting
             periodically (models auto-recalibration every ~900 s).

        Returns a multiplicative factor in (0.5, 1.05] applied to base refill.
        """
        # Deterministic per-edge seed from edge key
        edge_hash = hash(ek) & 0xFFFFFFFF

        # 1. Diurnal thermal drift: slow sinusoid (±5%)
        #    Use period = 900 s in simulation (compressed from 24 h)
        thermal_phase = (edge_hash % 360) * math.pi / 180.0
        thermal_factor = 1.0 + 0.05 * math.sin(2.0 * math.pi * t_s / 900.0 + thermal_phase)

        # 2. Turbulence bursts: pseudo-random, edge-specific
        burst_rng = random.Random(edge_hash ^ (t_s // 15))
        burst_active = burst_rng.random() < 0.015  # ~1.5% of 15-s windows
        burst_factor = 1.0
        if burst_active:
            burst_factor = 1.0 - burst_rng.uniform(0.10, 0.30)  # 10-30% loss

        # 3. Equipment calibration drift: sawtooth (0-2% loss, resets every ~900 s)
        cal_cycle = t_s % 900
        cal_factor = 1.0 - 0.02 * (cal_cycle / 900.0)

        combined = thermal_factor * burst_factor * cal_factor
        return max(0.50, min(1.05, combined))

    def effective_refill_bits_per_s(self, ek: Edge, t_s: int) -> float:
        """
        Dynamic refill with surface code QEC and real-time channel conditions:
            1. Physical QBER from channel monitoring
            2. Apply surface code QEC → logical QBER (exponentially suppressed)
            3. Secret fraction from logical QBER
            4. Channel condition factor (temperature, vibration, calibration)
            5. R(t) = R0 * distance_factor * secret_fraction(q_logical)
                       * channel_condition * (1 / d²)  [QEC encoding overhead]

        The QEC overhead (1/d²) models that d² physical qubits are needed
        per logical qubit, reducing raw throughput. However, the dramatic
        QBER reduction more than compensates at moderate code distances.

        Net effect at QBER=0.005, d=3:
            Without QEC: r = secret_fraction(0.005) = 0.880
            With QEC:    r = secret_fraction(0.00375) × (1/9) = 0.093
            → QEC overhead dominates at low QBER (expected — QEC not needed)

        Net effect at QBER=0.08, d=3:
            Without QEC: r = secret_fraction(0.08) = 0.456 (severely degraded)
            With QEC:    QBER > threshold → no QEC benefit, same as without
            → QEC cannot help above threshold

        Key insight: QEC's value is in maintaining key generation during
        moderate attacks (QBER 0.005-0.01) where it prevents the cascade
        failure. We apply QEC only when it provides net benefit.
        """
        pool = self.pools[ek]
        q_physical = self.health[ek].qber_at(t_s)
        if self.health[ek].abort_active():
            return 0.0

        # Dynamic channel condition multiplier (temperature, vibration, calibration)
        channel_factor = self._channel_condition_factor(ek, t_s)

        qec_distance = self._qec_code_distance
        r_without_qec = secret_fraction_bb84(q_physical, self.finite_key_params)
        if qec_distance <= 1:
            distance_adjusted = pool.distance_adjusted_refill_rate()
            return distance_adjusted * r_without_qec * channel_factor

        # Apply surface code QEC if beneficial
        q_logical = surface_code_logical_qber(q_physical, qec_distance)
        qec_overhead = 1.0 / (qec_distance ** 2)
        r_with_qec = secret_fraction_bb84(q_logical, self.finite_key_params) * qec_overhead

        # Use QEC only when it provides net benefit
        r = max(r_with_qec, r_without_qec)

        distance_adjusted = pool.distance_adjusted_refill_rate()
        return distance_adjusted * r * channel_factor

    def _refill_process(self, ek: Edge):
        """
        Refill loop: every 1 simulated second, add effective_refill(t) bits.
        """
        while True:
            t_s = self.now_s()
            refill = self.effective_refill_bits_per_s(ek, t_s)
            self.pools[ek].add_bits(refill)
            yield self.env.timeout(1)

    # -------------------------
    # E2E relay pool management
    # -------------------------

    def _init_e2e_pool(self, src: str, dst: str) -> Edge:
        """
        Lazily initialize an E2E key pool for a (src, dst) pair.
        The pool's dynamic refill is computed from the bottleneck link rate
        along the shortest path, divided by the number of hops (relay cost).
        """
        pair = edge_key(src, dst)
        if pair in self._e2e_pools:
            return pair

        # Compute shortest path on the network graph
        path_nodes: List[str] = [src, dst]  # fallback
        if self._graph is not None:
            try:
                path_nodes = nx.shortest_path(self._graph, src, dst)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        num_hops = max(1, len(path_nodes) - 1)
        path_edges = [edge_key(path_nodes[i], path_nodes[i + 1])
                      for i in range(len(path_nodes) - 1)]

        # Ensure all underlying link pools are initialized
        for ek in path_edges:
            self._ensure_edge_initialized(ek)

        # Cache the path edges for dynamic refill
        self._e2e_path_cache[pair] = path_edges

        # Create E2E pool with same capacity as link pools
        tmpl = self._default_pool_template
        pool = QKDKeyPool(
            capacity_bits=tmpl.capacity_bits,
            base_refill_bits_per_s=tmpl.base_refill_bits_per_s,
            init_fill_ratio=tmpl.init_fill_ratio,
            link_params=QKDLinkParameters(),  # no distance penalty for virtual pool
        )
        self._e2e_pools[pair] = pool

        # Start refill process
        if not self._e2e_refill_procs_started.get(pair, False):
            self._e2e_refill_procs_started[pair] = True
            self.env.process(self._e2e_refill_process(pair))

        return pair

    def _e2e_effective_refill(self, pair: Edge) -> float:
        """
        Dynamic E2E refill rate using entanglement swapping model.

        Physics (Werner state model, Briegel et al. 1998):
        1. Each link generates entangled pairs at its raw rate.
        2. Entanglement swapping at each relay has success probability
           p_swap = 0.5 (linear optics Bell state measurement).
        3. QBER accumulates across swaps: q_eff = (3/4)(1 - F^n)
           where F = 1 - (4/3)q per hop.
        4. E2E secret fraction uses E91 (entanglement-based) with q_eff.

        Rate formula:
            R_e2e = bottleneck(raw_rates) × p_swap^(n-1) × r_e91(q_eff)
                    ÷ r_bb84(q_per_hop)   [undo per-hop fraction already baked in]

        This is more accurate than the old `bottleneck / num_hops` model
        because it captures the exponential rate decay from BSM success
        probability and the QBER accumulation from swapping.
        """
        path_edges = self._e2e_path_cache.get(pair, [])
        if not path_edges:
            return 0.0

        t_s = self.now_s()
        num_hops = max(1, len(path_edges))

        # Collect per-hop effective rates and QBERs
        edge_rates = []
        per_hop_qbers = []
        for ek in path_edges:
            if ek in self.pools:
                edge_rates.append(self.effective_refill_bits_per_s(ek, t_s))
                per_hop_qbers.append(self.health[ek].qber_at(t_s))
            else:
                edge_rates.append(0.0)
                per_hop_qbers.append(0.05)

        if not edge_rates or max(edge_rates) <= 0:
            return 0.0

        # Bottleneck raw rate (already includes per-hop BB84 fraction)
        bottleneck = min(edge_rates)

        # Average per-hop QBER
        avg_qber = sum(per_hop_qbers) / len(per_hop_qbers)

        # Entanglement swapping: QBER accumulates, rate decreases
        q_eff = entanglement_swap_qber(avg_qber, num_hops)
        swap_factor = entanglement_swap_rate_factor(
            num_hops,
            p_swap=getattr(self, "_e2e_swap_success_prob", 0.5),
        )

        # Entanglement distillation (BBPSSW): purify noisy pairs
        # Convert q_eff to fidelity, distill, convert back
        f_raw = qber_to_fidelity(q_eff)
        distill_rounds = getattr(self, "_distillation_rounds", 1)
        f_distilled = distill_fidelity_bbpssw(f_raw, distill_rounds)
        q_distilled = fidelity_to_qber(f_distilled)
        distill_rate = distillation_rate_factor(distill_rounds)

        # Use distillation only when it provides net benefit
        # (distillation reduces QBER but also halves rate per round)
        from .quantum_protocols import secret_fraction_e91
        r_e91_raw = secret_fraction_e91(q_eff)
        r_e91_distilled = secret_fraction_e91(q_distilled) * distill_rate

        # Pick whichever path yields more key bits
        if r_e91_distilled > r_e91_raw:
            r_e91 = r_e91_distilled
            q_final = q_distilled
        else:
            r_e91 = r_e91_raw
            q_final = q_eff

        # Undo the per-hop BB84 fraction already baked into bottleneck rate
        r_bb84_per_hop = secret_fraction_bb84(avg_qber)
        if r_bb84_per_hop > 1e-9:
            raw_rate = bottleneck / r_bb84_per_hop
        else:
            raw_rate = 0.0

        # E2E rate: raw_rate × swap_success × E91_fraction
        e2e_rate = raw_rate * swap_factor * r_e91

        # Store metrics for reporting
        self._e2e_last_q_eff = getattr(self, "_e2e_last_q_eff", {})
        self._e2e_last_q_eff[pair] = q_eff

        return max(0.0, e2e_rate)

    def _e2e_refill_process(self, pair: Edge):
        """
        Refill loop for E2E pool using entanglement swapping rate.
        Updates every simulated second.
        """
        while True:
            refill = self._e2e_effective_refill(pair)
            self._e2e_pools[pair].add_bits(refill)
            yield self.env.timeout(1)

    def bits_required_e2e(self, msg: Message) -> float:
        """
        Key usage cost for E2E relay model — fixed per message, regardless of hops.
        Cost = tag_bits + nonce_bits + encryption_key_bits (no verify_cost_factor
        multiplication since verification happens once at the endpoint).
        """
        if not msg.requires_auth:
            return 0.0
        return float(
            self.key_policy.tag_bits
            + self.key_policy.nonce_bits
            + self.key_policy.encryption_key_bits
        )

    def bits_required_per_hop(self, msg: Message) -> float:
        """
        Key usage cost per hop.

        - Authenticated messages pay tag + nonce + encryption key.
        - Verification intensity scales cost.
        - encryption_key_bits (default 256) models AES-256 symmetric
          encryption; set to 0 for auth-only mode.
        """
        if not msg.requires_auth:
            return 0.0
        base = float(
            self.key_policy.tag_bits
            + self.key_policy.nonce_bits
            + self.key_policy.encryption_key_bits
        )
        return base * float(self.key_policy.verify_cost_factor)

    def _is_critical_message(self, msg: Message) -> bool:
        """
        Critical messages receive key-allocation priority.
        """
        if msg.msg_type != MsgType.CONTROL_SETPOINT:
            return False
        # Do not grant reservation priority to attack traffic that reuses
        # control-setpoint message type.
        if bool(getattr(msg, "is_attack", False)):
            return False
        try:
            if bool(msg.payload.get("attack", False)):
                return False
        except Exception:
            pass
        return True

    def _emergency_bits_per_hop(self, msg: Message) -> float:
        """
        Reduced key cost for critical messages in emergency mode.
        """
        if not msg.requires_auth:
            return 0.0
        return float(self._emergency_tag_bits + self.key_policy.nonce_bits)

    # ── V5: Quantum Control Authentication (QTLS-bound OTP) ──

    @staticmethod
    def _is_attack_message(msg: Message) -> bool:
        if bool(getattr(msg, "is_attack", False)):
            return True
        try:
            return bool((msg.payload or {}).get("attack", False))
        except Exception:
            return False

    def _priority_action_needs_quantum_token(self, msg: Message) -> bool:
        return bool(
            self._enable_quantum_control_auth
            and msg.requires_auth
            and msg.msg_type == MsgType.PRIORITY_ACTION
        )

    def _quantum_control_material(self, msg: Message, nonce: int, expiry_ms: int) -> str:
        payload = msg.payload or {}
        action = str(payload.get("action", ""))
        sender_role = str(payload.get("control_sender_role", ""))
        return "|".join([
            str(msg.src),
            str(msg.dst),
            str(getattr(msg.msg_type, "value", msg.msg_type)),
            action,
            sender_role,
            str(int(getattr(msg, "created_ms", 0) or 0)),
            str(int(nonce)),
            str(int(expiry_ms)),
            str(self._auth_model),
        ])

    def _compute_quantum_control_token(self, msg: Message, nonce: int, expiry_ms: int) -> str:
        material = self._quantum_control_material(msg, nonce, expiry_ms)
        digest = hashlib.sha256(f"{self._quantum_control_secret}|{material}".encode("utf-8")).hexdigest()
        return digest[:32]

    def _maybe_attach_quantum_control_token(self, msg: Message) -> None:
        if not self._priority_action_needs_quantum_token(msg):
            return

        payload = msg.payload or {}

        # ── Physical-origin check ──
        # The quantum control secret is held by the legitimate controller.
        # A message can only receive a valid HMAC token if it physically
        # originates from a node that possesses the secret.  If the message
        # has an injection_node that differs from its claimed src, it was
        # injected by a compromised node impersonating the controller —
        # this node does NOT have the QKD-derived HMAC secret and therefore
        # cannot produce a valid token.
        #
        # Without this check, the shared quantum layer would naively sign
        # any message that passes through it, including forged ones.
        inj = getattr(msg, "injection_node", None)
        if inj is not None and inj != msg.src:
            # Compromised node injecting with forged source header.
            # Attacker doesn't have the quantum secret — generate a
            # fake token that will fail HMAC verification.
            fake_token = f"{self.rng.getrandbits(128):032x}"
            nonce = int(payload.get("nonce", self.rng.getrandbits(64)))
            expiry_ms = int(getattr(msg, "created_ms", 0) or 0) + 2000
            payload["nonce"] = nonce
            payload["quantum_control_expiry_ms"] = expiry_ms
            payload["quantum_control_token"] = fake_token
            payload["quantum_token_forged"] = True
            msg.payload = payload
            return

        # If the message already carries a token (e.g. attacker's forged
        # attempt from SpoofingAttack), don't overwrite — let verification handle it.
        if str(payload.get("quantum_control_token", "")).strip():
            return

        if "nonce" not in payload:
            if self.nonce_mgr is not None:
                payload["nonce"] = int(self.nonce_mgr.gen_nonce(self.now_s()))
                payload["nonce_quality"] = str(self.nonce_mgr.last_quality)
                payload["qrng_pool_bits"] = float(self.nonce_mgr.last_qrng_pool_bits)
                payload["qrng_fallback"] = 1 if self.nonce_mgr.last_quality != "qrng" else 0
            else:
                payload["nonce"] = int(self.rng.getrandbits(max(16, self.key_policy.nonce_bits)))
                payload["nonce_quality"] = "classical_rng"
                payload["qrng_pool_bits"] = float("nan")
                payload["qrng_fallback"] = 1

        nonce = int(payload.get("nonce", 0))
        expiry_ms = int(getattr(msg, "created_ms", 0) or 0) + self._quantum_control_token_ttl_ms
        payload["quantum_control_mode"] = "qtls_bound_otp"
        payload["quantum_control_expiry_ms"] = expiry_ms
        payload["quantum_control_token"] = self._compute_quantum_control_token(msg, nonce, expiry_ms)
        msg.payload = payload
        self.protocol_stats["control_tokens_attached"] += 1

    def verify_quantum_control_token(self, msg: Message) -> Tuple[bool, str]:
        if not self._priority_action_needs_quantum_token(msg):
            return True, "not_required"

        payload = msg.payload or {}
        token = str(payload.get("quantum_control_token", ""))
        if not token:
            self.protocol_stats["control_tokens_rejected"] += 1
            return False, "missing_quantum_control_token"

        try:
            nonce = int(payload.get("nonce", 0))
            expiry_ms = int(payload.get("quantum_control_expiry_ms", -1))
        except Exception:
            self.protocol_stats["control_tokens_rejected"] += 1
            return False, "malformed_quantum_control_token"

        if expiry_ms < 0:
            self.protocol_stats["control_tokens_rejected"] += 1
            return False, "missing_quantum_control_expiry"

        now_ms = int(self.env.now * 1000)
        if now_ms > expiry_ms:
            self.protocol_stats["control_tokens_rejected"] += 1
            return False, "expired_quantum_control_token"

        expected = self._compute_quantum_control_token(msg, nonce, expiry_ms)
        if not hmac.compare_digest(token, expected):
            # ── Realistic bypass: implementation-level vulnerability ──
            # Even with information-theoretically secure QKD, real
            # hardware has side-channels (photon-number-splitting,
            # Trojan-horse detector blinding, timing correlations in
            # HMAC comparison, finite-key statistical residuals).
            # With probability quantum_auth_bypass_prob, the attacker
            # succeeds despite an invalid HMAC.
            if (self._quantum_auth_bypass_prob > 0
                    and self.rng.random() < self._quantum_auth_bypass_prob):
                self.protocol_stats["control_tokens_verified"] += 1
                self.protocol_stats.setdefault("control_tokens_bypass", 0)
                self.protocol_stats["control_tokens_bypass"] += 1
                return True, "bypass_implementation_vuln"

            self.protocol_stats["control_tokens_rejected"] += 1
            return False, "invalid_quantum_control_token"

        self.protocol_stats["control_tokens_verified"] += 1
        return True, "ok"

    def _is_pool_emergency(self, edges: List[Edge]) -> bool:
        """
        Emergency when any edge pool falls below configured threshold.
        """
        if not self._enable_emergency_mode:
            return False
        for ek in edges:
            pool = self.pools[ek]
            threshold = float(pool.capacity_bits) * self._emergency_threshold_ratio
            if pool.level_bits <= threshold:
                return True
        return False

    def maybe_rotation_cost(self, ek: Edge) -> float:
        """
        Optional rotation cost per edge, charged when rotate_every_msgs triggers.
        """
        if self.key_policy.rotate_every_msgs <= 0:
            return 0.0
        self._edge_msg_counter[ek] += 1
        if self._edge_msg_counter[ek] % self.key_policy.rotate_every_msgs == 0:
            self.rotation_stats["total_rotations"] += 1
            self.rotation_stats["rotation_bits_consumed"] += int(self.key_policy.rotate_bits)
            self.rotation_stats["rotations_by_edge"][ek] += 1
            return float(self.key_policy.rotate_bits)
        return 0.0

    # -------------------------
    # Hook for network.py
    # -------------------------

    def pre_send_hook(self, env: simpy.Environment, msg: Message, path_nodes: List[str]) -> simpy.Event:
        """
        SimPy hook called before network traversal.

        Behavior:
        - If replay protection enabled, ensure a nonce exists and detect replay (optional).
        - Compute path edges and total required keys.
        - If insufficient keys, wait up to max_key_wait_ms for refills.
        - If still insufficient, mark DROPPED_NO_KEYS.
        - If sufficient, consume keys and annotate msg.payload with QBER/Fidelity statistics.

        This hook does not model verification itself; it models resource usage and delay.
        """
        return env.process(self._pre_send_process(msg, path_nodes))

    def _pre_send_process(self, msg: Message, path_nodes: List[str]):
        start_ms = int(self.env.now * 1000)
        msg.payload = msg.payload or {}
        try:
            trace_id = int(os.getenv("QUAM_TRACE_MSG_ID", "-1"))
        except Exception:
            trace_id = -1

        # Ensure edges exist and refill processes are running
        edges = self.register_path_edges(path_nodes)
        per_hop_est = self.bits_required_per_hop(msg)
        key_bits_saved_est = max(0.0, per_hop_est * float(len(edges)))

        # V5: Quantum control auth — attach and verify token
        if self._enable_quantum_control_auth:
            self._maybe_attach_quantum_control_token(msg)
            tok_ok, tok_reason = self.verify_quantum_control_token(msg)
            if self._priority_action_needs_quantum_token(msg):
                msg.payload["quantum_control_token_valid"] = 1 if tok_ok else 0
                msg.payload["quantum_control_token_reason"] = tok_reason

        if self.preauth_decider is not None:
            # annotate health so gate can use secret fraction/QBER
            try:
                self._annotate_health(msg, edges)
            except Exception:
                pass
            decision, delay_ms = self.preauth_decider(self.env, msg, path_nodes)
            msg.payload["gate_decision"] = decision
            msg.payload["gate_reason"] = (
                msg.payload.get("prekey_gate_reason")
                or msg.payload.get("gate_reason")
                or "preauth"
            )
            if decision != "allow":
                msg.payload["prekey_blocked"] = 1
                msg.payload["key_bits_saved_prekey_est"] = float(key_bits_saved_est)
                msg.mark_dropped(DeliveryStatus.DROPPED_BLOCKED, "preauth_blocked")
                msg.key_wait_ms = 0
                return
            if delay_ms and delay_ms > 0:
                yield self.env.timeout(delay_ms / 1000.0)

        # Attach or generate nonce for replay protection if desired
        if self.nonce_mgr is not None and msg.requires_auth:
            if "nonce" not in msg.payload:
                msg.payload["nonce"] = int(self.nonce_mgr.gen_nonce(self.now_s()))
                msg.payload["nonce_quality"] = str(self.nonce_mgr.last_quality)
                msg.payload["qrng_pool_bits"] = float(self.nonce_mgr.last_qrng_pool_bits)
                msg.payload["qrng_fallback"] = 1 if self.nonce_mgr.last_quality != "qrng" else 0
            nonce = int(msg.payload.get("nonce", 0))
            if self.nonce_mgr.is_replay(msg.src, msg.dst, nonce):
                # Treat replay as a failure upstream (policy gate can also block later)
                msg.mark_dropped(DeliveryStatus.DROPPED_LOSS, "replay_detected")
                return
            self.nonce_mgr.remember(msg.src, msg.dst, nonce, self.now_s())

        # Compute required keys with priority/emergency policy.
        is_critical = self._is_critical_message(msg)
        is_emergency = self._is_pool_emergency(edges)
        if is_emergency and is_critical and self._enable_emergency_mode:
            per_hop = self._emergency_bits_per_hop(msg)
            msg.payload["emergency_mode"] = 1
            msg.payload["reduced_auth_tag"] = 1
            for ek in edges:
                self.pools[ek].emergency_grants += 1
        else:
            per_hop = per_hop_est
            msg.payload["emergency_mode"] = 0
            msg.payload["reduced_auth_tag"] = 0

        # V2: Protocol selection for all message types when protocols enabled
        # PRIORITY_ACTION → Quantum-TLS (KAK), CONTROL_SETPOINT → E91, TELEMETRY → Classical
        if self._protocol_config is not None:
            qp = _get_quantum_protocols()
            t_s = self.now_s()
            # Compute channel metrics for protocol selection
            avg_qber = 0.01
            noise_model_str = "generic"
            security_score = 1.0
            if edges:
                qbers = [self.health[ek].qber_at(t_s) for ek in edges]
                avg_qber = sum(qbers) / len(qbers)
                noise_model_str = self.health[edges[0]].noise_model.value
                if self._pingpong_ids is not None:
                    security_score = self._pingpong_ids.get_channel_security_score(t_s)

            protocol, reason = qp.select_protocol_for_message(
                msg, self._protocol_config,
                qber=avg_qber,
                noise_model=noise_model_str,
                security_score=security_score,
            )

            # ── Realistic quantum protocol latency overhead ──
            # These model real hardware/computation costs:
            #   gate_delay_ms   = auth computation per hop (HMAC, tag verify)
            #   qp_handshake_ms = protocol-specific handshake overhead
            # They feed into network.py's quantum_latency breakdown.
            n_hops = max(1, len(edges))

            if protocol == qp.QuantumProtocol.QUANTUM_TLS and self._qtls is not None:
                success, proto_used, metrics = self._qtls.transmit_priority_action(
                    msg=msg, qber=avg_qber, noise_model=noise_model_str,
                    t_s=t_s, rng=self.rng, security_score=security_score,
                )
                msg.payload["quantum_protocol"] = proto_used
                msg.payload["quantum_protocol_reason"] = reason
                msg.payload.update({f"qp_{k}": v for k, v in metrics.items()})

                if success and metrics.get("key_bits_consumed", -1) == 0:
                    # KAK 3-stage: 3 channel passes + ping-pong probes
                    # ~5ms computation + 2x extra propagation (3 passes vs 1)
                    kak_compute_ms = 5.0 + self.rng.gauss(0, 0.5)
                    # 2 extra passes × avg propagation per hop × n_hops
                    avg_prop_per_hop_ms = 3.0  # ~3ms for 10km fiber
                    kak_extra_prop_ms = 2.0 * avg_prop_per_hop_ms * n_hops
                    pp_probe_ms = 1.0 * metrics.get("pingpong_probes", 1)
                    msg.payload["gate_delay_ms"] = round(kak_compute_ms, 2)
                    msg.payload["qp_handshake_ms"] = round(kak_extra_prop_ms + pp_probe_ms, 2)
                    self.protocol_stats["kak_used"] += 1
                    self.protocol_stats["qtls_used"] += 1
                    msg.payload["key_bits_spent_total"] = 0.0
                    msg.payload["key_bits_per_hop"] = 0.0
                    msg.key_wait_ms = 0
                    self._annotate_health(msg, edges)
                    return  # Skip key consumption entirely
                elif "e91" in proto_used:
                    # Fallback to E91: Bell test overhead
                    msg.payload["gate_delay_ms"] = round(3.0 * n_hops + self.rng.gauss(0, 0.3), 2)
                    msg.payload["qp_handshake_ms"] = round(2.0 + self.rng.gauss(0, 0.2), 2)
                    self.protocol_stats["qtls_fallback_e91"] += 1
                    self.protocol_stats["e91_used"] += 1
                else:
                    # Fallback to BB84
                    msg.payload["gate_delay_ms"] = round(2.0 * n_hops + self.rng.gauss(0, 0.2), 2)
                    msg.payload["qp_handshake_ms"] = round(1.0, 2)
                    self.protocol_stats["qtls_fallback_bb84"] += 1
            elif protocol == qp.QuantumProtocol.E91:
                # E91: Bell test subset + HMAC auth per hop (~3ms/hop + 2ms handshake)
                msg.payload["gate_delay_ms"] = round(3.0 * n_hops + self.rng.gauss(0, 0.3), 2)
                msg.payload["qp_handshake_ms"] = round(2.0 + self.rng.gauss(0, 0.2), 2)
                self.protocol_stats["e91_used"] += 1
                msg.payload["quantum_protocol"] = "e91"
                msg.payload["quantum_protocol_reason"] = reason
            elif protocol == qp.QuantumProtocol.CLASSICAL:
                # Classical: no quantum overhead
                msg.payload["gate_delay_ms"] = 0.0
                msg.payload["qp_handshake_ms"] = 0.0
                self.protocol_stats["classical_used"] += 1
                msg.payload["quantum_protocol"] = "classical"
                msg.payload["quantum_protocol_reason"] = reason
            else:
                # BB84: HMAC auth per hop (~2ms/hop + 1ms handshake)
                msg.payload["gate_delay_ms"] = round(2.0 * n_hops + self.rng.gauss(0, 0.2), 2)
                msg.payload["qp_handshake_ms"] = round(1.0 + self.rng.gauss(0, 0.1), 2)
                self.protocol_stats["bb84_used"] += 1
                msg.payload["quantum_protocol"] = "bb84"
                msg.payload["quantum_protocol_reason"] = reason
        else:
            # No protocol config → still add BB84 auth overhead when QKD enabled
            if edges:
                n_hops = len(edges)
                msg.payload["gate_delay_ms"] = round(2.0 * n_hops + self.rng.gauss(0, 0.2), 2)
                msg.payload["qp_handshake_ms"] = round(1.0 + self.rng.gauss(0, 0.1), 2)

        if trace_id > 0 and int(getattr(msg, "msg_id", -1)) == trace_id:
            before_levels = [(ek, float(self.pools[ek].level_bits)) for ek in edges]
            print(f"[MSG {trace_id}] Key requirement per hop: {per_hop:.1f} bits")
            print(f"[MSG {trace_id}] Pool levels before: {before_levels}")
        if per_hop <= 0.0:
            # Still annotate link health if present (useful for analysis)
            self._annotate_health(msg, edges)
            msg.key_wait_ms = 0
            msg.payload["auth_model"] = self._auth_model
            return

        # ── E2E relay auth model: consume from virtual node-pair pool once ──
        if self._auth_model == "e2e_relay":
            pair = self._init_e2e_pool(msg.src, msg.dst)
            e2e_cost = self.bits_required_e2e(msg)
            if is_emergency and is_critical and self._enable_emergency_mode:
                e2e_cost = float(self._emergency_tag_bits + self.key_policy.nonce_bits)
                msg.payload["emergency_mode"] = 1

            # Wait for E2E pool
            waited_ms = 0
            tick = max(1, int(self.key_policy.key_wait_tick_ms))
            max_wait = max(0, int(self.key_policy.max_key_wait_ms))

            while True:
                if self._e2e_pools[pair].can_consume(e2e_cost):
                    break
                if waited_ms >= max_wait:
                    # GRACEFUL DEGRADATION: fall back to classical auth
                    # instead of dropping the message entirely.
                    # This prevents the cascade failure where key exhaustion
                    # → dropped control messages → blind controllers → EENS spike.
                    msg.key_wait_ms = waited_ms
                    msg.payload["key_bits_spent_total"] = 0.0
                    msg.payload["key_bits_per_hop"] = 0.0
                    msg.payload["key_bits_e2e"] = 0.0
                    msg.payload["auth_model"] = "classical_fallback"
                    msg.payload["classical_fallback"] = 1
                    msg.payload["e2e_pool_pair"] = f"{pair[0]}-{pair[1]}"
                    msg.payload["e2e_pool_level"] = float(self._e2e_pools[pair].level_bits)
                    self._classical_fallback_count = getattr(self, "_classical_fallback_count", 0) + 1
                    self._annotate_health(msg, edges)
                    break  # Proceed with classical auth instead of dropping
                yield self.env.timeout(tick / 1000.0)
                waited_ms = int(self.env.now * 1000) - start_ms

            if msg.payload.get("classical_fallback") != 1:
                ok = self._e2e_pools[pair].consume(e2e_cost)
                if not ok:
                    # Race condition: fall back to classical
                    msg.key_wait_ms = waited_ms
                    msg.payload["key_bits_spent_total"] = 0.0
                    msg.payload["key_bits_per_hop"] = 0.0
                    msg.payload["key_bits_e2e"] = 0.0
                    msg.payload["auth_model"] = "classical_fallback"
                    msg.payload["classical_fallback"] = 1
                    msg.payload["e2e_pool_pair"] = f"{pair[0]}-{pair[1]}"
                    msg.payload["e2e_pool_level"] = float(self._e2e_pools[pair].level_bits)
                    self._classical_fallback_count = getattr(self, "_classical_fallback_count", 0) + 1
                    self._annotate_health(msg, edges)
                else:
                    msg.key_wait_ms = waited_ms
                    msg.payload["key_bits_spent_total"] = float(e2e_cost)
                    msg.payload["key_bits_per_hop"] = 0.0
                    msg.payload["key_bits_e2e"] = float(e2e_cost)
                    msg.payload["auth_model"] = "e2e_relay"
                    msg.payload["e2e_pool_pair"] = f"{pair[0]}-{pair[1]}"
                    msg.payload["e2e_pool_level"] = float(self._e2e_pools[pair].level_bits)
            msg.payload["e2e_effective_refill"] = float(self._e2e_effective_refill(pair))
            msg.payload["is_critical_msg"] = 1 if is_critical else 0
            if trace_id > 0 and int(getattr(msg, "msg_id", -1)) == trace_id:
                print(f"[MSG {trace_id}] E2E pool {pair}: level={self._e2e_pools[pair].level_bits:.1f}")
            self._annotate_health(msg, edges)
            return

        # ── Per-hop auth model (default, trusted-node relay) ──
        msg.payload["auth_model"] = "per_hop"

        # Optional source key-consumption guard to slow exhaustion floods.
        if self._enable_source_key_rate_limit:
            t_s = self.now_s()
            for ek in edges:
                if not self.pools[ek].check_source_rate(t_s, msg.src, per_hop):
                    msg.mark_dropped(DeliveryStatus.DROPPED_BLOCKED, "source_key_rate_exceeded")
                    msg.key_wait_ms = 0
                    msg.payload["drop_reason_detail"] = "source_key_consumption_rate_limited"
                    self._annotate_health(msg, edges)
                    return

        # Rotation costs (charged once per edge for this message if triggered)
        rot_cost_by_edge: Dict[Edge, float] = {ek: self.maybe_rotation_cost(ek) for ek in edges}

        required_by_edge: Dict[Edge, float] = {ek: per_hop + rot_cost_by_edge[ek] for ek in edges}

        # Wait loop if needed
        waited_ms = 0
        tick = max(1, int(self.key_policy.key_wait_tick_ms))
        max_wait = max(0, int(self.key_policy.max_key_wait_ms))

        classical_fallback_hop = False
        while True:
            if self._all_edges_sufficient_priority(required_by_edge, is_critical):
                break

            if waited_ms >= max_wait:
                # GRACEFUL DEGRADATION: fall back to classical auth
                # instead of dropping the message.
                classical_fallback_hop = True
                msg.key_wait_ms = waited_ms
                msg.payload["key_bits_spent_total"] = 0.0
                msg.payload["key_bits_per_hop"] = 0.0
                msg.payload["auth_model"] = "classical_fallback"
                msg.payload["classical_fallback"] = 1
                self._classical_fallback_count = getattr(self, "_classical_fallback_count", 0) + 1
                self._annotate_health(msg, edges)
                break  # Proceed with classical auth instead of dropping

            yield self.env.timeout(tick / 1000.0)
            waited_ms = int(self.env.now * 1000) - start_ms

        # Consume keys (skip if fallen back to classical)
        spent_total = 0.0
        if not classical_fallback_hop:
            for ek, req in required_by_edge.items():
                ok = self.pools[ek].consume_priority(req, is_critical)
                if not ok:
                    # Race condition: fall back to classical
                    classical_fallback_hop = True
                    msg.key_wait_ms = waited_ms
                    msg.payload["key_bits_spent_total"] = 0.0
                    msg.payload["key_bits_per_hop"] = 0.0
                    msg.payload["auth_model"] = "classical_fallback"
                    msg.payload["classical_fallback"] = 1
                    self._classical_fallback_count = getattr(self, "_classical_fallback_count", 0) + 1
                    self._annotate_health(msg, edges)
                    break
                spent_total += req
            if self._enable_source_key_rate_limit:
                self.pools[ek].record_source_consumption(self.now_s(), msg.src, req)

        msg.key_wait_ms = waited_ms
        msg.payload["key_bits_spent_total"] = float(spent_total)
        msg.payload["key_bits_per_hop"] = float(per_hop)
        msg.payload["key_rotation_bits_total"] = float(sum(rot_cost_by_edge.values()))
        msg.payload["is_critical_msg"] = 1 if is_critical else 0
        # Track encryption coverage — True when QKD keys available for encryption
        msg.payload["encrypted"] = 1 if self.key_policy.encryption_key_bits > 0 else 0
        if trace_id > 0 and int(getattr(msg, "msg_id", -1)) == trace_id:
            after_levels = [(ek, float(self.pools[ek].level_bits)) for ek in edges]
            print(f"[MSG {trace_id}] Pool levels after: {after_levels}")
        self._annotate_health(msg, edges)

    def _all_edges_sufficient(self, required_by_edge: Dict[Edge, float]) -> bool:
        for ek, req in required_by_edge.items():
            if not self.pools[ek].can_consume(req):
                return False
        return True

    def _all_edges_sufficient_priority(self, required_by_edge: Dict[Edge, float], is_critical: bool) -> bool:
        """
        Priority-aware edge sufficiency: non-critical traffic respects reservation floor.
        """
        for ek, req in required_by_edge.items():
            if not self.pools[ek].can_consume_priority(req, is_critical):
                return False
        return True

    # V2: Ping-Pong IDS probing (called from stepper process)
    def run_pingpong_probes(self, t_s: int) -> List[Dict[str, Any]]:
        """Run Ping-Pong IDS probes on all registered edges."""
        results = []
        if self._pingpong_ids is None:
            return results
        for ek in self.pools:
            qber = self.health[ek].qber_at(t_s)
            eve_frac = self.health[ek].eve_intercept_at(t_s)
            dist_km = float(self.pools[ek].link_params.distance_km)
            result = self._pingpong_ids.send_probe(
                t_s=t_s, edge=ek, qber=qber, rng=self.rng,
                eve_intercept_fraction=eve_frac,
                distance_km=dist_km,
            )
            self.protocol_stats["pingpong_probes"] += 1
            if result.eve_detected:
                self.protocol_stats["pingpong_detections"] += 1
            results.append({
                "t_s": t_s,
                "edge": f"{ek[0]}-{ek[1]}",
                "bell_value": result.bell_value,
                "eve_detected": result.eve_detected,
                "confidence": result.confidence,
            })
        return results

    def get_e2e_pool_stats(self) -> Dict[str, Any]:
        """Get E2E relay pool statistics for summary output."""
        if not self._e2e_pools:
            return {}
        stats: Dict[str, Any] = {
            "e2e_pool_count": len(self._e2e_pools),
            "auth_model": self._auth_model,
            "classical_fallback_count": getattr(self, "_classical_fallback_count", 0),
            "e2e_distillation_rounds": int(getattr(self, "_distillation_rounds", 1)),
            "e2e_swap_success_prob": float(getattr(self, "_e2e_swap_success_prob", 0.5)),
        }
        e2e_initial = 0.0
        e2e_added = 0.0
        e2e_consumed = 0.0
        e2e_spilled = 0.0
        e2e_final = 0.0
        e2e_failed = 0.0
        for pair, pool in self._e2e_pools.items():
            e2e_initial += float(getattr(pool, "initial_level_bits", 0.0))
            e2e_added += float(getattr(pool, "total_added_bits", 0.0))
            e2e_consumed += float(getattr(pool, "total_consumed_bits", 0.0))
            e2e_spilled += float(getattr(pool, "total_spilled_bits", 0.0))
            e2e_final += float(pool.level_bits)
            e2e_failed += float(getattr(pool, "total_failed_consume_bits", 0.0))
        stats["e2e_key_initial_sum"] = e2e_initial
        stats["e2e_key_added_sum"] = e2e_added
        stats["e2e_key_consumed_sum"] = e2e_consumed
        stats["e2e_key_spilled_sum"] = e2e_spilled
        stats["e2e_key_final_sum"] = e2e_final
        stats["e2e_key_failed_consume_sum"] = e2e_failed
        # Conservation check: initial + added_effective = consumed + final
        # (spilled is tracked separately — it's generated but never stored)
        lhs = e2e_initial + e2e_added
        rhs = e2e_consumed + e2e_final
        stats["e2e_key_conservation_delta"] = abs(lhs - rhs)

        # Entanglement swapping metrics
        e2e_q_eff = getattr(self, "_e2e_last_q_eff", {})
        if e2e_q_eff:
            avg_q_eff = sum(e2e_q_eff.values()) / len(e2e_q_eff)
            stats["e2e_entanglement_swap_qber_avg"] = round(avg_q_eff, 6)
            stats["e2e_entanglement_swap_pairs"] = len(e2e_q_eff)
            # Average path length
            avg_hops = sum(len(self._e2e_path_cache.get(p, [])) for p in e2e_q_eff) / max(1, len(e2e_q_eff))
            stats["e2e_avg_path_hops"] = round(avg_hops, 2)
            # Swap success factor for average path
            stats["e2e_swap_success_factor"] = round(
                entanglement_swap_rate_factor(
                    int(avg_hops),
                    p_swap=getattr(self, "_e2e_swap_success_prob", 0.5),
                ),
                4,
            )

        return stats

    def get_protocol_stats(self) -> Dict[str, Any]:
        """Get multi-protocol usage statistics."""
        stats = dict(self.protocol_stats)
        stats["qec_code_distance"] = int(getattr(self, "_qec_code_distance", 3))
        stats["e2e_distillation_rounds"] = int(getattr(self, "_distillation_rounds", 1))
        stats["e2e_swap_success_prob"] = float(getattr(self, "_e2e_swap_success_prob", 0.5))
        stats["control_auth_enabled"] = int(bool(getattr(self, "_enable_quantum_control_auth", False)))
        stats["control_token_ttl_ms"] = int(getattr(self, "_quantum_control_token_ttl_ms", 0))
        if self._pingpong_ids is not None:
            stats.update({f"ids_{k}": v for k, v in self._pingpong_ids.get_stats().items()})
        if self._qtls is not None:
            stats.update({f"qtls_{k}": v for k, v in self._qtls.get_stats().items()})
        return stats

    def pingpong_has_recent_alert(self, t_s: int, lookback_s: int = 60) -> bool:
        """Check for recent Ping-Pong IDS alerts."""
        if self._pingpong_ids is None:
            return False
        return self._pingpong_ids.has_recent_alert(t_s, lookback_s)

    def _annotate_health(self, msg: Message, edges: List[Edge]) -> None:
        """
        Attach path-level health summary to msg.payload:
        - mean QBER, min fidelity, mean secret fraction
        - per-edge list for debugging (optional)
        """
        t_s = self.now_s()
        if not edges:
            return

        q_list: List[float] = []
        f_list: List[float] = []
        r_list: List[float] = []

        per_edge: List[Dict[str, Any]] = []
        for ek in edges:
            q = self.health[ek].qber_at(t_s)
            f = fidelity_from_qber(q, self.health[ek].noise_model)
            r = secret_fraction_bb84(q, self.finite_key_params)
            q_list.append(q)
            f_list.append(f)
            r_list.append(r)
            distance_km = float(self.pools[ek].link_params.distance_km)
            distance_factor = float(self.pools[ek].link_params.distance_factor())
            per_edge.append({
                "edge": f"{ek[0]}-{ek[1]}",
                "qber": float(q),
                "fidelity": float(f),
                "secret_fraction": float(r),
                "pool_level_bits": float(self.pools[ek].level_bits),
                "distance_km": distance_km,
                "distance_factor": distance_factor,
                "noise_model": self.health[ek].noise_model.value,
            })

        msg.payload["qber_path_mean"] = float(sum(q_list) / len(q_list))
        msg.payload["fidelity_path_min"] = float(min(f_list))
        msg.payload["secret_fraction_path_mean"] = float(sum(r_list) / len(r_list))

        total_distance = sum(self.pools[ek].link_params.distance_km for ek in edges)
        avg_distance_factor = (sum(self.pools[ek].link_params.distance_factor() for ek in edges) / len(edges))
        msg.payload["path_total_distance_km"] = float(total_distance)
        msg.payload["path_avg_distance_factor"] = float(avg_distance_factor)

        if self.finite_key_params is not None and self.finite_key_params.enabled:
            msg.payload["finite_key_block_size"] = int(self.finite_key_params.block_size_bits)
            msg.payload["finite_key_security_log"] = int(self.finite_key_params.security_parameter_log)
            msg.payload["finite_key_correction"] = float(self.finite_key_params.correction_term())
            msg.payload["finite_key_factor"] = float(self.finite_key_params.finite_key_factor())

        # Keep this optional list small. You can disable later if logs get large.
        msg.payload["quantum_path_edges"] = per_edge


# -------------------------
# Convenience constructors (for runner)
# -------------------------

def make_default_quantum_layer(
    env: simpy.Environment,
    rng: random.Random,
    *,
    capacity_bits: int = 50_000,
    base_refill_bits_per_s: float = 1_000.0,
    init_fill_ratio: float = 0.30,
    baseline_qber: float = 0.01,
    key_policy: Optional[KeyPolicy] = None,
    finite_key_params: Optional[FiniteKeyParameters] = None,
    enable_replay_protection: bool = True,
    use_qrng_nonces: bool = True,
    quantum_protocol_config: Optional[Any] = None,
) -> QuantumAugmentation:
    """
    Creates a QuantumAugmentation instance with default pool and health templates.
    """
    kp = key_policy or KeyPolicy()
    default_pool = QKDKeyPool(
        capacity_bits=capacity_bits,
        base_refill_bits_per_s=base_refill_bits_per_s,
        init_fill_ratio=init_fill_ratio,
    )
    default_health = QuantumLinkHealth(
        baseline_qber=baseline_qber,
        windows=[],
    )
    return QuantumAugmentation(
        env=env,
        rng=rng,
        key_policy=kp,
        default_pool=default_pool,
        default_health=default_health,
        per_edge_pool=None,
        per_edge_health=None,
        finite_key_params=finite_key_params,
        enable_replay_protection=enable_replay_protection,
        use_qrng_nonces=use_qrng_nonces,
        quantum_protocol_config=quantum_protocol_config,
    )
