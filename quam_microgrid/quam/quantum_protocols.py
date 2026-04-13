"""
quantum_protocols.py

Multi-layer quantum protocol implementations for QuAM:
- Layer 1: Ping-Pong Intrusion Detection System (IDS)
- Layer 2: E91 Entanglement-Based QKD
- Layer 3: Quantum-TLS (KAK Three-Stage + Ping-Pong QSDC)

These protocols enhance microgrid security where:
- PRIORITY_ACTION messages use Quantum-TLS (NO key consumption)
- CONTROL_SETPOINT messages use E91-derived keys
- Continuous Ping-Pong monitoring detects channel attacks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from enum import Enum
import math
import random

if TYPE_CHECKING:
    from .quantum import QuantumLinkHealth, FiniteKeyParameters

from .model import Message, MsgType


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class QuantumProtocol(str, Enum):
    """Available quantum protocols."""
    BB84 = "bb84"
    E91 = "e91"
    KAK_THREE_STAGE = "kak"
    QUANTUM_TLS = "qtls"      # KAK + Ping-Pong combined
    CLASSICAL = "classical"   # Fallback, no quantum


class PingPongVariant(str, Enum):
    """Ping-Pong protocol variants with different detection rates."""
    BELL = "bell"       # Original, 50% detection
    GHZ = "ghz"         # GHZ-enhanced, 75% detection
    CLUSTER = "cluster" # 6-qubit cluster, 94% detection


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value to range [lo, hi]."""
    return max(lo, min(hi, x))


def h2(q: float) -> float:
    """Binary entropy function h2(q) in bits."""
    q = clamp(q, 1e-12, 1.0 - 1e-12)
    return -(q * math.log2(q) + (1.0 - q) * math.log2(1.0 - q))


# =============================================================================
# LAYER 1: PING-PONG INTRUSION DETECTION SYSTEM
# =============================================================================

@dataclass
class PingPongProbeResult:
    """Result of a single Ping-Pong probe."""
    t_s: int
    edge: Tuple[str, str]
    bell_value: float
    quantum_bound: float        # 2sqrt2 ~ 2.83
    classical_bound: float      # 2.0
    eve_detected: bool
    confidence: float           # 0-1
    variant: PingPongVariant
    eve_active: bool = False    # True if Eve was intercepting during this probe


@dataclass
class PingPongIDS:
    """
    Quantum Intrusion Detection using Ping-Pong protocol probes.

    Mechanism:
    - Periodically sends entangled Bell/GHZ states through channel
    - Measures correlation via Bell inequality test
    - Eavesdropper measurement disturbs entanglement -> detectable

    Detection rates by variant:
    - BELL: 50% (original Bostrom-Felbinger)
    - GHZ: 75% (3-particle GHZ state)
    - CLUSTER: 94% (6-qubit cluster state)
    """
    probe_interval_s: float = 5.0
    variant: PingPongVariant = PingPongVariant.GHZ

    # Bell inequality bounds
    bell_quantum_max: float = 2.828      # 2*sqrt(2)
    bell_classical_bound: float = 2.0
    detection_threshold_ratio: float = 0.75

    # False alarm model — realistic detector noise causes occasional false triggers
    # Rate scales with QBER: higher channel noise → harder to distinguish from Eve
    false_alarm_base_prob: float = 0.005   # 0.5% base false alarm per probe
    false_alarm_qber_scale: float = 0.04   # additional 4% at QBER=0.11

    # State
    probe_history: List[PingPongProbeResult] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)

    # Statistics
    total_probes: int = field(default=0, init=False)
    total_detections: int = field(default=0, init=False)
    probes_during_eve: int = field(default=0, init=False)
    detections_during_eve: int = field(default=0, init=False)

    def get_base_detection_rate(self) -> float:
        """Theoretical base detection rate for current variant."""
        rates = {
            PingPongVariant.BELL: 0.50,
            PingPongVariant.GHZ: 0.75,
            PingPongVariant.CLUSTER: 0.94,
        }
        return rates.get(self.variant, 0.50)

    def effective_detection_probability(
        self, qber: float, eve_intercept_fraction: float,
        distance_km: float = 10.0,
    ) -> float:
        """
        Dynamic detection probability that depends on channel conditions.

        In the real world, detection effectiveness varies based on:
          - Channel noise (QBER):  high noise masks Eve's disturbance
          - Eve's aggressiveness:  a cautious Eve (low intercept) is harder to catch
          - Link distance:         longer fibres lose photons, degrading entanglement
          - Variant:               multi-qubit states (GHZ/Cluster) are more robust

        Model:
          P_eff = P_base × noise_robustness × intercept_sensitivity × distance_factor

        noise_robustness: (1 - qber/0.11)^α   where α varies by variant
          Bell:     α=1.0  (fragile — 2 qubits, easily confused by noise)
          GHZ:      α=0.6  (moderate — 3-party correlations filter noise)
          Cluster:  α=0.3  (robust — 6-qubit redundancy)

        intercept_sensitivity: eve_intercept^β   where β varies by variant
          Bell:     β=0.8  (needs strong signal — requires large disturbance)
          GHZ:      β=0.5  (moderate — detects mid-range intercepts)
          Cluster:  β=0.3  (sensitive — detects even cautious eavesdroppers)

        distance_factor: exp(-distance / λ)   where λ varies by variant
          Bell:     λ=80 km  (short range — entanglement fragile)
          GHZ:      λ=120 km (medium range — multi-photon resilience)
          Cluster:  λ=200 km (long range — qubit redundancy compensates loss)
        """
        import math

        p_base = self.get_base_detection_rate()

        # Variant-specific parameters
        params = {
            PingPongVariant.BELL:    {"noise_alpha": 1.0, "intercept_beta": 0.8, "dist_lambda": 80.0},
            PingPongVariant.GHZ:     {"noise_alpha": 0.6, "intercept_beta": 0.5, "dist_lambda": 120.0},
            PingPongVariant.CLUSTER: {"noise_alpha": 0.3, "intercept_beta": 0.3, "dist_lambda": 200.0},
        }
        p = params.get(self.variant, params[PingPongVariant.BELL])

        # 1. Noise robustness: higher QBER → harder to detect Eve over noise floor
        #    At QBER=0 → factor=1.0, at QBER=0.11 → factor=0.0
        noise_ratio = clamp(qber / 0.11, 0.0, 1.0)
        noise_robustness = max(0.0, (1.0 - noise_ratio)) ** p["noise_alpha"]

        # 2. Intercept sensitivity: how aggressively Eve intercepts
        #    Higher intercept → bigger disturbance → easier to detect
        #    At intercept=0 → 0, at intercept=1 → 1
        if eve_intercept_fraction > 0:
            intercept_sensitivity = clamp(eve_intercept_fraction, 0.0, 1.0) ** p["intercept_beta"]
        else:
            intercept_sensitivity = 0.0

        # 3. Distance penalty: longer links → photon loss → lower entanglement quality
        dist_factor = math.exp(-distance_km / p["dist_lambda"])

        p_eff = p_base * noise_robustness * intercept_sensitivity * dist_factor
        return clamp(p_eff, 0.0, 1.0)

    def send_probe(
        self,
        t_s: int,
        edge: Tuple[str, str],
        qber: float,
        rng: random.Random,
        eve_intercept_fraction: float = 0.0,
        distance_km: float = 10.0,
    ) -> PingPongProbeResult:
        """
        Send a Ping-Pong probe and evaluate for eavesdropping.

        Args:
            t_s: Current simulation time (seconds)
            edge: Quantum link being probed (node_a, node_b)
            qber: Current QBER on the channel
            rng: Random number generator
            eve_intercept_fraction: Fraction of qubits Eve intercepts (0-1)
            distance_km: Physical distance of the quantum link

        Returns:
            PingPongProbeResult with detection outcome
        """
        self.total_probes += 1

        # Perfect entanglement gives S = 2*sqrt(2) ~ 2.83
        s_max = self.bell_quantum_max

        # Channel noise reduces correlation
        # At QBER ~ 0.146, Bell value drops to classical bound
        noise_factor = 1.0 - (qber / 0.146) * 0.293
        noise_factor = clamp(noise_factor, 0.0, 1.0)

        # Eve's interception reduces correlation further
        eve_factor = 1.0 - eve_intercept_fraction

        # Add small random jitter
        jitter = rng.gauss(0, 0.02)

        # Compute observed Bell value
        bell_value = s_max * noise_factor * eve_factor + jitter
        bell_value = clamp(bell_value, 0.0, s_max)

        # Detection: Bell value below threshold indicates Eve
        threshold = self.detection_threshold_ratio * s_max
        eve_below_threshold = bell_value < threshold

        # Dynamic detection probability based on channel conditions.
        # The probability depends on variant, QBER, intercept fraction,
        # and link distance — NOT a static constant.
        if eve_below_threshold and eve_intercept_fraction > 0:
            detection_prob = self.effective_detection_probability(
                qber=qber,
                eve_intercept_fraction=eve_intercept_fraction,
                distance_km=distance_km,
            )
            eve_detected = rng.random() < detection_prob
        elif eve_intercept_fraction == 0:
            # False alarm model: realistic detector noise, dark counts, and
            # environmental fluctuations can cause sporadic false triggers.
            # Rate scales with channel QBER (noisier channel → more ambiguity).
            qber_ratio = clamp(qber / 0.11, 0.0, 1.0)
            fp_prob = self.false_alarm_base_prob + self.false_alarm_qber_scale * qber_ratio
            eve_detected = rng.random() < fp_prob
        else:
            eve_detected = eve_below_threshold

        # Confidence: how far below threshold
        if eve_detected:
            confidence = clamp(
                (threshold - bell_value) / (threshold - self.bell_classical_bound),
                0.0, 1.0
            )
        else:
            confidence = 0.0

        eve_active = eve_intercept_fraction > 0

        result = PingPongProbeResult(
            t_s=t_s,
            edge=edge,
            bell_value=bell_value,
            quantum_bound=s_max,
            classical_bound=self.bell_classical_bound,
            eve_detected=eve_detected,
            confidence=confidence,
            variant=self.variant,
            eve_active=eve_active,
        )

        self.probe_history.append(result)

        # Track per-eve-window statistics for accurate detection rate
        if eve_active:
            self.probes_during_eve += 1
            if eve_detected:
                self.detections_during_eve += 1

        if eve_detected:
            self.total_detections += 1
            self.alerts.append({
                "t_s": t_s,
                "edge": f"{edge[0]}-{edge[1]}",
                "bell_value": bell_value,
                "confidence": confidence,
                "alert_type": "pingpong_eve_detected",
            })

        return result

    def get_channel_security_score(self, t_s: int, window_s: int = 30) -> float:
        """
        Compute channel security score based on recent probes.

        Returns:
            1.0 = fully secure (no detections)
            0.0 = compromised (all probes detected Eve)
            0.5 = unknown (no data)
        """
        recent = [r for r in self.probe_history if t_s - r.t_s < window_s]
        if not recent:
            return 0.5  # Unknown

        secure_probes = sum(1 for r in recent if not r.eve_detected)
        return secure_probes / len(recent)

    def has_recent_alert(self, t_s: int, lookback_s: int = 60) -> bool:
        """Check for recent Eve detection alerts."""
        return any(t_s - a["t_s"] < lookback_s for a in self.alerts)

    def get_stats(self) -> Dict[str, Any]:
        """Get IDS statistics."""
        return {
            "total_probes": self.total_probes,
            "total_detections": self.total_detections,
            "detection_rate": (
                self.total_detections / self.total_probes
                if self.total_probes > 0 else 0.0
            ),
            "probes_during_eve": self.probes_during_eve,
            "detections_during_eve": self.detections_during_eve,
            "eve_detection_rate": (
                self.detections_during_eve / self.probes_during_eve
                if self.probes_during_eve > 0 else 0.0
            ),
            "variant": self.variant.value,
        }


# =============================================================================
# LAYER 2: E91 ENTANGLEMENT-BASED QKD
# =============================================================================

@dataclass
class E91KeyDistribution:
    """
    E91 (Ekert 1991) entanglement-based QKD.

    Advantages over BB84:
    - Built-in Bell test verifies security during key generation
    - Source can be untrusted (device-independent security possible)
    - Better suited for network topologies with central entanglement source

    Mechanism:
    - Central source generates entangled Bell pairs
    - One photon to Alice, one to Bob
    - Each measures in one of 3 bases
    - Matching bases -> key bits
    - Non-matching bases -> Bell test for security verification
    """
    # E91 uses only 2/9 of basis combinations for key generation
    key_generation_efficiency: float = 0.222  # 2/9
    bell_test_fraction: float = 0.333         # ~1/3 pairs for Bell test

    def secret_fraction_e91(
        self,
        qber: float,
        finite_key_params: Optional[Any] = None,
        reconciliation_efficiency: float = 1.16,
    ) -> float:
        """
        E91 secret key rate using Shor-Preskill bound with CASCADE
        error correction.

        Lower than BB84 due to Bell test overhead, but provides
        stronger security guarantees via Bell inequality violation.

        r_e91 = eta_key * max(0, 1 - h2(q) - f_EC * h2(q))

        where eta_key ~ 2/9 and f_EC = 1.16 (CASCADE efficiency).
        The old ``1 - 2*h2(q)`` was the two-way bound.
        """
        q = clamp(qber, 0.0, 0.5)

        # Shor-Preskill bound with CASCADE reconciliation
        r_asymptotic = self.key_generation_efficiency * max(
            0.0, 1.0 - h2(q) - reconciliation_efficiency * h2(q)
        )

        # Apply finite-key corrections if provided
        if finite_key_params is not None:
            if hasattr(finite_key_params, 'enabled') and finite_key_params.enabled:
                if hasattr(finite_key_params, 'finite_key_factor'):
                    r_asymptotic *= finite_key_params.finite_key_factor()

        return max(0.0, r_asymptotic)

    def verify_bell_inequality(
        self,
        qber: float,
        rng: random.Random,
        memory: Optional['QuantumMemoryModel'] = None,
        storage_time_ms: float = 0.0,
    ) -> Tuple[bool, float]:
        """
        Simulate Bell inequality verification, optionally including
        decoherence effects from quantum memory storage.

        Decoherence reduces entanglement fidelity, which degrades the
        Bell parameter S toward the classical bound of 2.  If S falls
        below 2, quantum correlations can no longer be confirmed and
        E91 security guarantees are lost.

        Returns:
            (is_secure, bell_value)
            is_secure: True if Bell inequality violated (quantum correlation confirmed)
        """
        s_max = 2.828  # 2*sqrt(2)

        # QBER degrades Bell value
        noise_factor = 1.0 - (qber / 0.146) * 0.293
        noise_factor = clamp(noise_factor, 0.0, 1.0)

        # Decoherence penalty: fidelity loss reduces Bell parameter
        if memory is not None and storage_time_ms > 0:
            f_loss = memory.fidelity_loss(storage_time_ms)
            # Bell parameter scales with fidelity: S ~ S_max * (1 - f_loss)
            noise_factor *= (1.0 - f_loss)

        # Add measurement noise
        jitter = rng.gauss(0, 0.03)

        bell_value = s_max * noise_factor + jitter
        bell_value = clamp(bell_value, 0.0, s_max)

        # Secure if Bell inequality violated (S > 2)
        is_secure = bell_value > 2.0

        return is_secure, bell_value


# Module-level convenience function for E91 secret fraction
# (used by quantum.py entanglement swapping refill without class instantiation)
_default_e91 = E91KeyDistribution()

def secret_fraction_e91(
    qber: float,
    finite_key_params: Optional[Any] = None,
    reconciliation_efficiency: float = 1.16,
) -> float:
    """
    Module-level E91 secret fraction computation.

    Wraps E91KeyDistribution.secret_fraction_e91() for standalone use,
    particularly in the entanglement swapping refill model (quantum.py).

    Returns the fraction of entangled pairs that produce secret key bits,
    accounting for the 2/9 key generation efficiency of E91 protocol.
    """
    return _default_e91.secret_fraction_e91(qber, finite_key_params, reconciliation_efficiency)


# =============================================================================
# QUANTUM MEMORY DECOHERENCE MODEL
# =============================================================================

@dataclass
class QuantumMemoryModel:
    """
    NV-center diamond quantum memory decoherence parameters.

    Models fidelity degradation during quantum state storage,
    critical for multi-pass protocols like KAK Three-Stage where
    qubits must be stored between channel passes.

    Default values correspond to NV-center at room temperature:
        T1 ~ 1 ms   (relaxation / amplitude damping)
        T2 ~ 0.5 ms (dephasing / phase damping)

    References:
        - Bar-Gill et al., Nature Comm. 4, 1743 (2013)
        - Abobeih et al., Nature 576, 411-415 (2019)
    """
    t1_ms: float = 1.0         # Relaxation time (amplitude damping)
    t2_ms: float = 0.5         # Dephasing time (phase damping), T2 <= 2*T1
    gate_time_ms: float = 0.01 # Single-qubit gate time

    def fidelity_loss(self, storage_time_ms: float) -> float:
        """Fidelity degradation from storage: F_loss = 1 - exp(-t / T2)."""
        if storage_time_ms <= 0:
            return 0.0
        return 1.0 - math.exp(-storage_time_ms / self.t2_ms)

    def qber_penalty(self, storage_time_ms: float) -> float:
        """
        QBER increase from decoherence.

        Uses depolarizing approximation: delta_QBER = (2/3) * F_loss.
        This models the dominant T2 dephasing process as an effective
        depolarizing channel acting on stored qubits.
        """
        return (2.0 / 3.0) * self.fidelity_loss(storage_time_ms)


# =============================================================================
# LAYER 3: KAK THREE-STAGE PROTOCOL
# =============================================================================

@dataclass
class KAKThreeStage:
    """
    KAK Three-Stage Quantum Cryptography Protocol.

    Unlike BB84/E91 which distribute keys for classical encryption,
    KAK transmits the message DIRECTLY via quantum states.

    Mechanism:
    1. Alice prepares message qubit |M>, applies random rotation R_A
       -> Sends R_A|M> to Bob
    2. Bob applies his random rotation R_B
       -> Sends R_B*R_A|M> back to Alice
    3. Alice removes her rotation (applies R_A^-1)
       -> Sends R_B|M> to Bob
    4. Bob removes his rotation (applies R_B^-1)
       -> Recovers |M>

    KEY ADVANTAGE: No pre-shared key required!
    - Immune to key exhaustion attacks
    - Perfect for emergency PRIORITY_ACTION messages

    LIMITATIONS:
    - Requires 3 channel traversals (3x latency)
    - Only secure under collective noise (dephasing, rotation)
    - NOT secure under amplitude damping noise
    - Requires low QBER (< 3.8% per pass)
    """
    rotation_precision_bits: int = 8  # 256 possible rotation angles

    # Compatible noise models
    compatible_noise_models: Tuple[str, ...] = ("dephasing", "generic", "depolarizing")

    # Maximum QBER for secure operation
    # After 3 passes: QBER_eff = 1 - (1-q)^3
    # For QBER_eff < 11%: q < 3.8%
    max_single_pass_qber: float = 0.038

    def compute_effective_qber(
        self,
        single_pass_qber: float,
        memory: Optional[QuantumMemoryModel] = None,
        round_trip_ms: float = 0.5,
    ) -> float:
        """
        Compute effective QBER after 3 channel passes, optionally
        including quantum memory decoherence.

        Channel contribution:
            QBER_channel = 1 - (1 - q)^3

        Memory contribution (if provided):
            Qubits are stored for ~2 round-trip times (between
            passes 1→2 and 2→3).  Decoherence adds:
            QBER_mem = (2/3) * (1 - exp(-2*RTT / T2))

        Total: QBER_eff = QBER_channel + QBER_mem, clamped to [0, 0.5]
        """
        q = clamp(single_pass_qber, 0.0, 0.5)
        q_channel = 1.0 - (1.0 - q) ** 3

        if memory is not None:
            # State stored for ~2 round trips (passes 1→2 and 2→3)
            storage_ms = 2.0 * round_trip_ms
            q_mem = memory.qber_penalty(storage_ms)
            q_channel = min(0.5, q_channel + q_mem)

        return q_channel

    def is_channel_compatible(self, qber: float, noise_model: str) -> Tuple[bool, str]:
        """
        Check if KAK can operate securely on this channel.

        Returns:
            (is_compatible, reason)
        """
        # Check noise model
        if noise_model.lower() not in self.compatible_noise_models:
            return False, f"noise_model_incompatible:{noise_model}"

        # Check QBER
        if qber > self.max_single_pass_qber:
            return False, f"qber_too_high:{qber:.4f}_max:{self.max_single_pass_qber}"

        return True, "compatible"

    def compute_transmission_cost(self, message_bits: int) -> Dict[str, Any]:
        """
        Compute resource costs for KAK transmission.

        KEY POINT: No key bits consumed!
        """
        qubits_per_pass = message_bits

        return {
            "channel_passes": 3,
            "qubits_per_pass": qubits_per_pass,
            "total_qubits": qubits_per_pass * 3,
            "key_bits_consumed": 0,  # THE KEY ADVANTAGE!
            "extra_latency_factor": 3.0,  # 3x normal latency
        }


# =============================================================================
# LAYER 3: QUANTUM-TLS (KAK + PING-PONG COMBINED)
# =============================================================================

@dataclass
class QuantumTLSConfig:
    """Configuration for Quantum-TLS protocol."""
    # KAK settings
    kak_enabled: bool = True
    kak_max_qber: float = 0.038

    # Ping-Pong interleaving during KAK transmission
    pingpong_check_frequency: float = 0.1  # Check every 10% of transmission
    pingpong_variant: PingPongVariant = PingPongVariant.GHZ

    # Fallback behavior
    fallback_to_e91: bool = True
    fallback_to_bb84: bool = True

    # Channel requirements
    min_security_score: float = 0.8


@dataclass
class QuantumTLS:
    """
    Quantum Transport Layer Security for PRIORITY_ACTION messages.

    Combines:
    1. KAK Three-Stage: Secure transmission WITHOUT pre-shared key
    2. Ping-Pong: Real-time eavesdrop detection during transmission

    This is "Quantum Secure Direct Communication" (QSDC).

    Why this matters for microgrids:
    - PRIORITY_ACTION = emergency commands (island, shed load)
    - These MUST get through even during key exhaustion attacks
    - KAK doesn't use key pools -> immune to exhaustion
    - Ping-Pong detects MITM during transmission
    """
    config: QuantumTLSConfig = field(default_factory=QuantumTLSConfig)
    kak: KAKThreeStage = field(default_factory=KAKThreeStage)
    ids: PingPongIDS = field(default_factory=lambda: PingPongIDS(variant=PingPongVariant.GHZ))

    # Statistics
    stats: Dict[str, int] = field(default_factory=lambda: {
        "transmissions_attempted": 0,
        "transmissions_kak_success": 0,
        "transmissions_fallback_e91": 0,
        "transmissions_fallback_bb84": 0,
        "transmissions_failed": 0,
        "eve_detections_during_tx": 0,
    })

    def transmit_priority_action(
        self,
        msg: Message,
        qber: float,
        noise_model: str,
        t_s: int,
        rng: random.Random,
        security_score: float = 1.0,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Transmit PRIORITY_ACTION using Quantum-TLS.

        Args:
            msg: The PRIORITY_ACTION message
            qber: Current single-pass QBER
            noise_model: Channel noise model
            t_s: Current time
            rng: Random number generator
            security_score: Channel security from IDS (0-1)

        Returns:
            (success, protocol_used, metrics)
        """
        self.stats["transmissions_attempted"] += 1

        # Validate message type
        if msg.msg_type != MsgType.PRIORITY_ACTION:
            return False, "error", {"error": "not_priority_action"}

        # Check channel security
        if security_score < self.config.min_security_score:
            # Channel may be compromised, but emergency msg must go through
            # Log warning but continue with fallback
            pass

        # Try KAK
        kak_compatible, kak_reason = self.kak.is_channel_compatible(qber, noise_model)

        if kak_compatible and self.config.kak_enabled:
            # Perform KAK transmission with interleaved Ping-Pong checks
            return self._transmit_with_kak(msg, qber, t_s, rng)

        # KAK not viable - try fallbacks
        if self.config.fallback_to_e91:
            self.stats["transmissions_fallback_e91"] += 1
            return True, "e91_fallback", {
                "kak_failed_reason": kak_reason,
                "protocol": "e91",
                "key_bits_required": self._estimate_key_bits(msg),
            }

        if self.config.fallback_to_bb84:
            self.stats["transmissions_fallback_bb84"] += 1
            return True, "bb84_fallback", {
                "kak_failed_reason": kak_reason,
                "protocol": "bb84",
                "key_bits_required": self._estimate_key_bits(msg),
            }

        # No viable protocol
        self.stats["transmissions_failed"] += 1
        return False, "no_viable_protocol", {"kak_reason": kak_reason}

    def _transmit_with_kak(
        self,
        msg: Message,
        qber: float,
        t_s: int,
        rng: random.Random,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Execute KAK transmission with Ping-Pong verification."""
        message_bits = msg.size_bytes * 8
        cost = self.kak.compute_transmission_cost(message_bits)

        # Number of Ping-Pong probes during transmission
        n_probes = max(1, int(message_bits * self.config.pingpong_check_frequency / 100))

        # Effective QBER after 3 passes
        qber_effective = self.kak.compute_effective_qber(qber)

        # Simulate Ping-Pong checks during KAK
        detections = 0
        for _ in range(n_probes):
            probe = self.ids.send_probe(
                t_s=t_s,
                edge=("src", "dst"),  # Placeholder
                qber=qber_effective,
                rng=rng,
            )
            if probe.eve_detected:
                detections += 1

        # Too many detections -> abort
        detection_ratio = detections / n_probes if n_probes > 0 else 0.0
        if detection_ratio > 0.2:  # More than 20% detected Eve
            self.stats["eve_detections_during_tx"] += 1
            # Fall back to E91 (which at least uses authenticated keys)
            if self.config.fallback_to_e91:
                self.stats["transmissions_fallback_e91"] += 1
                return True, "e91_fallback", {
                    "kak_aborted_reason": "eve_detected",
                    "detection_ratio": detection_ratio,
                    "protocol": "e91",
                    "key_bits_required": self._estimate_key_bits(msg),
                }
            return False, "eve_detected", {"detection_ratio": detection_ratio}

        # Success!
        self.stats["transmissions_kak_success"] += 1

        return True, "quantum_tls", {
            "protocol": "kak_three_stage",
            "channel_passes": cost["channel_passes"],
            "total_qubits": cost["total_qubits"],
            "key_bits_consumed": 0,  # THE KEY ADVANTAGE
            "pingpong_probes": n_probes,
            "pingpong_detections": detections,
            "qber_single_pass": qber,
            "qber_effective": qber_effective,
        }

    def _estimate_key_bits(self, msg: Message) -> int:
        """Estimate key bits needed for fallback encryption."""
        # Rough estimate: 256 bits auth tag + 64 bits nonce + message encryption
        return 256 + 64 + (msg.size_bytes * 8)

    def get_stats(self) -> Dict[str, Any]:
        """Get Quantum-TLS statistics."""
        total = self.stats["transmissions_attempted"]
        return {
            **self.stats,
            "kak_success_rate": (
                self.stats["transmissions_kak_success"] / total
                if total > 0 else 0.0
            ),
            "fallback_rate": (
                (self.stats["transmissions_fallback_e91"] +
                 self.stats["transmissions_fallback_bb84"]) / total
                if total > 0 else 0.0
            ),
        }


# =============================================================================
# PROTOCOL CONFIGURATION AND SELECTION
# =============================================================================

@dataclass
class QuantumProtocolConfig:
    """Configuration for multi-protocol quantum layer."""

    # Protocol selection by message type
    priority_action_protocol: QuantumProtocol = QuantumProtocol.QUANTUM_TLS
    control_setpoint_protocol: QuantumProtocol = QuantumProtocol.E91
    telemetry_protocol: QuantumProtocol = QuantumProtocol.CLASSICAL

    # Ping-Pong IDS settings
    enable_pingpong_ids: bool = True
    ids_probe_interval_s: float = 5.0
    ids_variant: PingPongVariant = PingPongVariant.GHZ

    # E91 settings
    e91_bell_test_fraction: float = 0.333

    # Quantum-TLS settings
    qtls_config: QuantumTLSConfig = field(default_factory=QuantumTLSConfig)

    # Fallback behavior
    fallback_chain: List[QuantumProtocol] = field(
        default_factory=lambda: [
            QuantumProtocol.QUANTUM_TLS,
            QuantumProtocol.E91,
            QuantumProtocol.BB84,
        ]
    )


def select_protocol_for_message(
    msg: Message,
    config: QuantumProtocolConfig,
    qber: float = 0.01,
    noise_model: str = "generic",
    security_score: float = 1.0,
) -> Tuple[QuantumProtocol, str]:
    """
    Select appropriate quantum protocol based on message type and channel state.

    Args:
        msg: Message to transmit
        config: Protocol configuration
        qber: Current channel QBER
        noise_model: Channel noise model
        security_score: Channel security score from IDS (0-1)

    Returns:
        (protocol, selection_reason)
    """
    # Get preferred protocol for message type
    if msg.msg_type == MsgType.PRIORITY_ACTION:
        preferred = config.priority_action_protocol
    elif msg.msg_type == MsgType.CONTROL_SETPOINT:
        preferred = config.control_setpoint_protocol
    else:
        preferred = config.telemetry_protocol

    # Check if Quantum-TLS is viable
    if preferred == QuantumProtocol.QUANTUM_TLS:
        # KAK constraints
        if qber > 0.038:
            return QuantumProtocol.E91, "qber_too_high_for_kak"
        if noise_model.lower() not in ("dephasing", "generic", "depolarizing"):
            return QuantumProtocol.E91, "noise_incompatible_for_kak"
        if security_score < 0.5:
            # Channel very insecure, but emergency msg -> try anyway
            return QuantumProtocol.QUANTUM_TLS, "emergency_despite_low_security"

    return preferred, "preferred_protocol"


# =============================================================================
# QRNG-BASED MEASUREMENT CHALLENGE SYSTEM
# Active defense for compromised sensor detection (FDI defense layer)
# =============================================================================

@dataclass
class QRNGSensorChallengeConfig:
    """
    Configuration for QRNG-based measurement challenge system.

    This active defense detects compromised sensors that pass FDI attacks
    with valid QKD authentication tags (because the sensor itself generates
    them). The controller uses QRNG to issue challenges at unpredictable
    times, comparing sensor-reported values against its physics model.

    Quantum advantage: QRNG timing is information-theoretically random.
    A quantum-capable attacker cannot predict when challenges occur, so
    cannot selectively report truthfully only during challenge windows.
    With classical PRNG, an attacker knowing the seed can predict ~50%
    of challenge timings.

    Analogous to BB84 basis reconciliation applied to CPS sensor integrity.
    """
    # Challenge frequency
    mean_challenge_interval_s: float = 15.0   # mean seconds between challenges
    min_interval_s: float = 5.0               # minimum gap to same node

    # QRNG resource consumption
    qrng_bits_per_challenge: int = 16         # bits consumed per timing decision

    # Tolerance thresholds (fraction of source capacity)
    solar_tolerance_frac: float = 0.20        # 20% — high cloud variability
    wind_tolerance_frac: float = 0.25         # 25% — O-U fluctuations
    smr_tolerance_frac: float = 0.05          # 5% — near-constant baseload

    # EWMA deviation tracking
    ewma_alpha: float = 0.3                   # higher = more responsive

    # Quarantine triggering
    consecutive_failures_to_quarantine: int = 3

    # Classical PRNG fallback penalty
    prng_detection_rate_factor: float = 0.5   # attacker predicts half

    # Check mode
    per_source_check: bool = True             # per solar/wind/smr vs total-only

    # Master switch
    enabled: bool = True


@dataclass
class SensorChallengeResult:
    """Result of a single QRNG measurement challenge."""
    t_s: int
    node: str

    # Expected values (from controller's physics model)
    expected_solar_kw: float
    expected_wind_kw: float
    expected_smr_kw: float
    expected_total_kw: float

    # Reported values (from sensor — possibly biased by FDI)
    reported_solar_kw: float
    reported_wind_kw: float
    reported_smr_kw: float
    reported_total_kw: float

    # Deviations
    deviation_total_kw: float

    # Outcome
    passed: bool
    failure_source: str        # "" if passed, else "solar"/"wind"/"smr"/"total"

    # QRNG quality
    qrng_quality: str          # "qrng" or "weak" (PRNG fallback)

    # Ground truth (for metrics only — not available to controller)
    fdi_active: bool


class QRNGSensorChallenger:
    """
    QRNG-based active measurement verification for compromised sensor detection.

    Mechanism (analogous to BB84 basis reconciliation applied to CPS):
    1. QRNG generates unpredictable challenge timing
    2. Controller computes expected generation from its physics model
    3. Compares against sensor-reported values
    4. EWMA tracks per-node deviation history
    5. Consecutive failures trigger quarantine

    Quantum advantage:
    - QRNG timing is information-theoretically random
    - Quantum-capable attacker cannot predict when challenges occur
    - With classical PRNG, attacker can predict ~50% of challenges
      (knows seed → knows when to report truthfully)

    Defense stack position:
    - Layer 0: No defense (all readings accepted)
    - Layer 1: Classical HMAC (detects network-level FDI only)
    - Layer 2: QKD authentication (same gap for compromised sensors)
    - Layer 3: QRNG Measurement Challenges (detects compromised sensors)
    """

    def __init__(
        self,
        cfg: QRNGSensorChallengeConfig,
        qrng,                    # Optional[QuantumRNG] — from quantum.py
        nonce_mgr,               # Optional[NonceManager] — from quantum.py
        quarantine_mgr,          # QuarantineManager — from threat.py
        rng: random.Random,
    ):
        self.cfg = cfg
        self.qrng = qrng
        self.nonce_mgr = nonce_mgr
        self.quarantine_mgr = quarantine_mgr
        self.rng = rng

        # Per-node tracking
        self._last_challenge_t: Dict[str, int] = {}
        self._ewma_deviation: Dict[str, float] = {}
        self._consecutive_failures: Dict[str, int] = {}
        self._adaptive_threshold: Dict[str, float] = {}

        # Results history
        self.results: List[SensorChallengeResult] = []

        # Statistics
        self.stats: Dict[str, int] = {
            "challenges_sent": 0,
            "challenges_passed": 0,
            "challenges_failed": 0,
            "quarantines_triggered": 0,
            "challenges_with_qrng": 0,
            "challenges_with_prng_fallback": 0,
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
        }

        # Detection latency tracking
        self._first_detection_t: Dict[str, Optional[int]] = {}
        self._fdi_start_t: Dict[str, Optional[int]] = {}

    def compute_adaptive_threshold(
        self,
        node: str,
        solar_cap: float,
        wind_cap: float,
        smr_cap: float,
    ) -> float:
        """Compute per-node threshold from source capacities via RSS."""
        solar_tol = solar_cap * self.cfg.solar_tolerance_frac
        wind_tol = wind_cap * self.cfg.wind_tolerance_frac
        smr_tol = smr_cap * self.cfg.smr_tolerance_frac
        # Root-sum-of-squares of individual tolerances
        total_tol = math.sqrt(solar_tol ** 2 + wind_tol ** 2 + smr_tol ** 2)
        self._adaptive_threshold[node] = total_tol
        return total_tol

    def should_challenge(self, t_s: int, node: str) -> bool:
        """
        Decide whether to issue a challenge at time t_s for this node.

        Uses QRNG to generate unpredictable timing. With classical PRNG
        fallback, effective detection rate is halved (attacker can predict).
        """
        if not self.cfg.enabled:
            return False

        # Respect minimum interval
        last = self._last_challenge_t.get(node, -999)
        if (t_s - last) < self.cfg.min_interval_s:
            return False

        # Try to consume QRNG bits for timing decision
        use_qrng = False
        if self.qrng is not None and self.qrng.can_consume(
            self.cfg.qrng_bits_per_challenge, t_s
        ):
            self.qrng.consume(self.cfg.qrng_bits_per_challenge, t_s)
            use_qrng = True

        # Probability of challenge this second
        p_challenge = 1.0 / max(1.0, self.cfg.mean_challenge_interval_s)

        # PRNG fallback: attacker can predict timing → reduced detection
        if not use_qrng:
            p_challenge *= self.cfg.prng_detection_rate_factor

        return self.rng.random() < p_challenge

    def execute_challenge(
        self,
        t_s: int,
        node: str,
        expected_solar_kw: float,
        expected_wind_kw: float,
        expected_smr_kw: float,
        reported_solar_kw: float,
        reported_wind_kw: float,
        reported_smr_kw: float,
        fdi_active: bool = False,
    ) -> SensorChallengeResult:
        """
        Execute a measurement challenge: compare expected vs reported values.

        Args:
            expected_*: Values from controller's physics model (gen_profiles).
            reported_*: Values from sensor (may be biased by FDI).
            fdi_active: Ground truth flag (for TP/FP/TN/FN metrics only).
        """
        self._last_challenge_t[node] = t_s

        # Generate nonce for challenge freshness
        qrng_quality = "qrng"
        if self.nonce_mgr is not None:
            self.nonce_mgr.gen_nonce(t_s)
            qrng_quality = self.nonce_mgr.last_quality
        elif self.qrng is None:
            qrng_quality = "weak"

        # Compute deviations
        expected_total = expected_solar_kw + expected_wind_kw + expected_smr_kw
        reported_total = reported_solar_kw + reported_wind_kw + reported_smr_kw
        dev_solar = abs(expected_solar_kw - reported_solar_kw)
        dev_wind = abs(expected_wind_kw - reported_wind_kw)
        dev_smr = abs(expected_smr_kw - reported_smr_kw)
        dev_total = abs(expected_total - reported_total)

        # Determine pass/fail against adaptive threshold
        passed = True
        failure_source = ""

        threshold = self._adaptive_threshold.get(node, 0.0)
        if threshold <= 0:
            # Shouldn't happen if compute_adaptive_threshold was called
            threshold = 15.0  # fallback

        if dev_total > threshold:
            passed = False
            # Identify dominant deviation source
            if dev_solar >= dev_wind and dev_solar >= dev_smr:
                failure_source = "solar"
            elif dev_wind >= dev_smr:
                failure_source = "wind"
            else:
                failure_source = "smr"

        # Update EWMA deviation tracker
        old_ewma = self._ewma_deviation.get(node, 0.0)
        new_ewma = (
            self.cfg.ewma_alpha * dev_total
            + (1.0 - self.cfg.ewma_alpha) * old_ewma
        )
        self._ewma_deviation[node] = new_ewma

        # Update consecutive failure count
        if not passed:
            self._consecutive_failures[node] = (
                self._consecutive_failures.get(node, 0) + 1
            )
        else:
            self._consecutive_failures[node] = 0

        # Trigger quarantine if threshold exceeded
        if (
            self._consecutive_failures.get(node, 0)
            >= self.cfg.consecutive_failures_to_quarantine
        ):
            self.quarantine_mgr.trigger(t_s, node)
            self._consecutive_failures[node] = 0
            self.stats["quarantines_triggered"] += 1

        # Update statistics
        self.stats["challenges_sent"] += 1
        if passed:
            self.stats["challenges_passed"] += 1
        else:
            self.stats["challenges_failed"] += 1

        if qrng_quality == "qrng":
            self.stats["challenges_with_qrng"] += 1
        else:
            self.stats["challenges_with_prng_fallback"] += 1

        # Confusion matrix (uses ground truth)
        if fdi_active and not passed:
            self.stats["true_positives"] += 1
            if (
                node not in self._first_detection_t
                or self._first_detection_t[node] is None
            ):
                self._first_detection_t[node] = t_s
        elif fdi_active and passed:
            self.stats["false_negatives"] += 1
        elif not fdi_active and not passed:
            self.stats["false_positives"] += 1
        else:
            self.stats["true_negatives"] += 1

        result = SensorChallengeResult(
            t_s=t_s,
            node=node,
            expected_solar_kw=expected_solar_kw,
            expected_wind_kw=expected_wind_kw,
            expected_smr_kw=expected_smr_kw,
            expected_total_kw=expected_total,
            reported_solar_kw=reported_solar_kw,
            reported_wind_kw=reported_wind_kw,
            reported_smr_kw=reported_smr_kw,
            reported_total_kw=reported_total,
            deviation_total_kw=dev_total,
            passed=passed,
            failure_source=failure_source,
            qrng_quality=qrng_quality,
            fdi_active=fdi_active,
        )
        self.results.append(result)
        return result

    def register_fdi_start(self, node: str, t_s: int) -> None:
        """Register when FDI attack starts on a node (for latency metrics).

        Only records the earliest start time per node (multiple attack
        windows should not overwrite with later start times).
        """
        if node not in self._fdi_start_t or self._fdi_start_t[node] is None:
            self._fdi_start_t[node] = t_s
            self._first_detection_t[node] = None

    def get_detection_latency_s(self) -> Dict[str, float]:
        """Compute detection latency per node where FDI was detected."""
        latencies: Dict[str, float] = {}
        for node, det_t in self._first_detection_t.items():
            start_t = self._fdi_start_t.get(node)
            if det_t is not None and start_t is not None:
                latencies[node] = float(det_t - start_t)
        return latencies

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive challenge statistics for metrics export."""
        total = max(1, self.stats["challenges_sent"])
        tp = self.stats["true_positives"]
        fp = self.stats["false_positives"]
        tn = self.stats["true_negatives"]
        fn = self.stats["false_negatives"]

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (
            2.0 * precision * recall / max(1e-9, precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        fpr = fp / max(1, fp + tn)

        latencies = self.get_detection_latency_s()
        mean_latency = (
            sum(latencies.values()) / len(latencies)
            if latencies
            else float("nan")
        )

        return {
            "sc_challenges_total": self.stats["challenges_sent"],
            "sc_challenges_passed": self.stats["challenges_passed"],
            "sc_challenges_failed": self.stats["challenges_failed"],
            "sc_quarantines_triggered": self.stats["quarantines_triggered"],
            "sc_qrng_count": self.stats["challenges_with_qrng"],
            "sc_prng_fallback_count": self.stats["challenges_with_prng_fallback"],
            "sc_qrng_ratio": self.stats["challenges_with_qrng"] / total,
            "sc_true_positives": tp,
            "sc_false_positives": fp,
            "sc_true_negatives": tn,
            "sc_false_negatives": fn,
            "sc_precision": precision,
            "sc_recall": recall,
            "sc_f1_score": f1,
            "sc_false_positive_rate": fpr,
            "sc_detection_latency_mean_s": mean_latency,
        }
