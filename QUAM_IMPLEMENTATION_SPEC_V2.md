# QuAM Quantum Protocol Implementation Specification v2

## Document Purpose

This specification provides **detailed implementation instructions** for enhancing the QuAM (Quantum-Augmented Microgrid Security Simulation) framework with:

1. **Multi-layer quantum protocols** (KAK Three-Stage, Ping-Pong IDS, E91)
2. **Insider threat model** (compromised coordinator)
3. **Reframed key exhaustion** attack (insider-based, not external)

**Target**: Feed this document to Claude Code for implementation.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Current System Summary](#2-current-system-summary)
3. [Phase 1: Create quantum_protocols.py](#3-phase-1-create-quantum_protocolspy)
4. [Phase 2: Modify quantum.py](#4-phase-2-modify-quantumpy)
5. [Phase 3: Modify threat.py](#5-phase-3-modify-threatpy)
6. [Phase 4: Modify common.py](#6-phase-4-modify-commonpy)
7. [Phase 5: Modify finalmain.py](#7-phase-5-modify-finalmainpy)
8. [Phase 6: Modify metrics.py](#8-phase-6-modify-metricspy)
9. [Testing Checklist](#9-testing-checklist)

---

## 1. Architecture Overview

### Network Model (UNCHANGED)

```
                              MG0 (Coordinator)
                             ┌───────────────┐
                             │ Sends:        │
                             │ - CONTROL     │
                             │ - PRIORITY    │
                             │               │
                             │ Receives:     │
                             │ - TELEMETRY   │
                             └───────┬───────┘
                                     │
                      QKD-secured inter-microgrid links
                                     │
                    ┌────────────────┼
                    │                │                
                    ▼                ▼                
              ┌──────────┐    ┌──────────┐    
              │   MG1    │◄──►│   MG2    │    
              │          │    │          │    
              │ Sends:   │    │ Sends:   │    
              │ -TELEMETRY    │ -TELEMETRY    
              │          │    │          │    
              │ Receives:│    │ Receives:│    
              │ -CONTROL │    │ -CONTROL │    
              │ -PRIORITY│    │ -PRIORITY│    
              └──────────┘    └──────────┘    
```

### Message Flow Rules (UNCHANGED)

| Message Type | Valid Senders | Valid Receivers | Auth Required |
|--------------|---------------|-----------------|---------------|
| `CONTROL_SETPOINT` | MG0 only | MG1, MG2, ... | Yes |
| `PRIORITY_ACTION` | MG0 only | MG1, MG2, ... | Yes |
| `TELEMETRY` | Any | Any | No (optional) |
| `QAN_NOTIFY` | Any | Any | Yes |
| `COVER` | Any | Any | No |

### NEW: Quantum Protocol Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 3: QUANTUM-TLS                         │
│              (For PRIORITY_ACTION messages ONLY)                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • KAK Three-Stage: Direct quantum communication          │  │
│  │ • NO pre-shared key required (immune to key exhaustion!) │  │
│  │ • Interleaved Ping-Pong probes for eavesdrop detection   │  │
│  │ • Fallback to E91/BB84 if channel conditions poor        │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 2: E91 QKD                             │
│              (For CONTROL_SETPOINT messages)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Entanglement-based key distribution                    │  │
│  │ • Built-in Bell test for eavesdrop verification          │  │
│  │ • Replaces BB84 for control messages                     │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 1: PING-PONG IDS                       │
│              (Continuous channel monitoring)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Periodic Bell pair probes through quantum channel      │  │
│  │ • Detects eavesdropping via correlation degradation      │  │
│  │ • Feeds alerts to PolicyGate for defense decisions       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### NEW: Two Attack Scenarios

| Scenario | Attacker | Capabilities | Realistic? |
|----------|----------|--------------|------------|
| **External Attacker** | Outside network | Spoof messages, disturb QBER | ✅ Yes |
| **Insider (Compromised MG0)** | Controls coordinator | Legitimate keys, send malicious commands | ✅ Yes |



---

## 2. Current System Summary

### Key Files

| File | Purpose |
|------|---------|
| `model.py` | Message, MsgType, MicrogridState dataclasses |
| `network.py` | Topology, links, message delivery |
| `quantum.py` | QKD key pools, QBER, BB84 secret fraction |
| `threat.py` | Attacks (spoof, exhaust), defenses (PolicyGate) |
| `common.py` | SimContext, scheduling functions |
| `finalmain.py` | Main runner script |
| `metrics.py` | Logging and metrics |

### Current BB84 Implementation (in `quantum.py`)

```python
def secret_fraction_bb84(qber: float, finite_key_params=None) -> float:
    """r = max(0, 1 - 2*h2(QBER))"""
    q = clamp(qber, 0.0, 0.5)
    r_asymptotic = max(0.0, 1.0 - 2.0 * h2(q))
    # ... finite key corrections ...
    return max(0.0, r_asymptotic)
```

### Current Key Exhaustion Attack (in `threat.py`)

```python
class KeyExhaustionAttack:
    """Floods authenticated messages to drain key pools"""
    # PROBLEM: Assumes external attacker can consume keys
    # SOLUTION: Reframe as insider attack from compromised MG0
```

---

## 3. Phase 1: Create `quantum_protocols.py`

**Create a new file**: `/mnt/project/quantum_protocols.py`

### 3.1 File Header and Imports

```python
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
```

### 3.2 Layer 1: Ping-Pong IDS

```python
# =============================================================================
# LAYER 1: PING-PONG INTRUSION DETECTION SYSTEM
# =============================================================================

@dataclass
class PingPongProbeResult:
    """Result of a single Ping-Pong probe."""
    t_s: int
    edge: Tuple[str, str]
    bell_value: float
    quantum_bound: float        # 2√2 ≈ 2.83
    classical_bound: float      # 2.0
    eve_detected: bool
    confidence: float           # 0-1
    variant: PingPongVariant


@dataclass
class PingPongIDS:
    """
    Quantum Intrusion Detection using Ping-Pong protocol probes.
    
    Mechanism:
    - Periodically sends entangled Bell/GHZ states through channel
    - Measures correlation via Bell inequality test
    - Eavesdropper measurement disturbs entanglement → detectable
    
    Detection rates by variant:
    - BELL: 50% (original Boström-Felbinger)
    - GHZ: 75% (3-particle GHZ state)
    - CLUSTER: 94% (6-qubit cluster state)
    """
    probe_interval_s: float = 5.0
    variant: PingPongVariant = PingPongVariant.GHZ
    
    # Bell inequality bounds
    bell_quantum_max: float = 2.828      # 2√2
    bell_classical_bound: float = 2.0
    detection_threshold_ratio: float = 0.75
    
    # State
    probe_history: List[PingPongProbeResult] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Statistics
    total_probes: int = field(default=0, init=False)
    total_detections: int = field(default=0, init=False)
    
    def get_detection_rate(self) -> float:
        """Theoretical detection rate for current variant."""
        rates = {
            PingPongVariant.BELL: 0.50,
            PingPongVariant.GHZ: 0.75,
            PingPongVariant.CLUSTER: 0.94,
        }
        return rates.get(self.variant, 0.50)
    
    def send_probe(
        self,
        t_s: int,
        edge: Tuple[str, str],
        qber: float,
        rng: random.Random,
        eve_intercept_fraction: float = 0.0,
    ) -> PingPongProbeResult:
        """
        Send a Ping-Pong probe and evaluate for eavesdropping.
        
        Args:
            t_s: Current simulation time (seconds)
            edge: Quantum link being probed (node_a, node_b)
            qber: Current QBER on the channel
            rng: Random number generator
            eve_intercept_fraction: Fraction of qubits Eve intercepts (0-1)
        
        Returns:
            PingPongProbeResult with detection outcome
        """
        self.total_probes += 1
        
        # Perfect entanglement gives S = 2√2 ≈ 2.83
        s_max = self.bell_quantum_max
        
        # Channel noise reduces correlation
        # At QBER ≈ 0.146, Bell value drops to classical bound
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
        eve_detected = bell_value < threshold
        
        # Confidence: how far below threshold
        if eve_detected:
            confidence = clamp(
                (threshold - bell_value) / (threshold - self.bell_classical_bound),
                0.0, 1.0
            )
        else:
            confidence = 0.0
        
        result = PingPongProbeResult(
            t_s=t_s,
            edge=edge,
            bell_value=bell_value,
            quantum_bound=s_max,
            classical_bound=self.bell_classical_bound,
            eve_detected=eve_detected,
            confidence=confidence,
            variant=self.variant,
        )
        
        self.probe_history.append(result)
        
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
            "variant": self.variant.value,
        }
```

### 3.3 Layer 2: E91 Key Distribution

```python
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
    - Central source generates entangled Bell pairs |Φ+⟩
    - One photon to Alice, one to Bob
    - Each measures in one of 3 bases (0°, 45°, 90° for Alice; 45°, 90°, 135° for Bob)
    - Matching bases (45°,45°) and (90°,90°) → key bits
    - Non-matching bases → Bell test for security verification
    """
    # E91 uses only 2/9 of basis combinations for key generation
    # (rest used for Bell test)
    key_generation_efficiency: float = 0.222  # 2/9
    bell_test_fraction: float = 0.333         # ~1/3 pairs for Bell test
    
    def secret_fraction_e91(
        self, 
        qber: float,
        finite_key_params: Optional[Any] = None
    ) -> float:
        """
        E91 secret key rate.
        
        Lower than BB84 due to Bell test overhead, but provides
        stronger security guarantees.
        
        r_e91 = η_key × max(0, 1 - 2×h2(QBER))
        
        where η_key ≈ 2/9
        """
        q = clamp(qber, 0.0, 0.5)
        
        # Base rate with Bell test overhead
        r_asymptotic = self.key_generation_efficiency * max(0.0, 1.0 - 2.0 * h2(q))
        
        # Apply finite-key corrections if provided
        if finite_key_params is not None:
            if hasattr(finite_key_params, 'enabled') and finite_key_params.enabled:
                if hasattr(finite_key_params, 'finite_key_factor'):
                    r_asymptotic *= finite_key_params.finite_key_factor()
        
        return max(0.0, r_asymptotic)
    
    def verify_bell_inequality(self, qber: float, rng: random.Random) -> Tuple[bool, float]:
        """
        Simulate Bell inequality verification.
        
        Returns:
            (is_secure, bell_value)
            is_secure: True if Bell inequality violated (quantum correlation confirmed)
        """
        s_max = 2.828  # 2√2
        
        # QBER degrades Bell value
        noise_factor = 1.0 - (qber / 0.146) * 0.293
        noise_factor = clamp(noise_factor, 0.0, 1.0)
        
        # Add measurement noise
        jitter = rng.gauss(0, 0.03)
        
        bell_value = s_max * noise_factor + jitter
        bell_value = clamp(bell_value, 0.0, s_max)
        
        # Secure if Bell inequality violated (S > 2)
        is_secure = bell_value > 2.0
        
        return is_secure, bell_value
```

### 3.4 Layer 3: KAK Three-Stage Protocol

```python
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
    1. Alice prepares message qubit |M⟩, applies random rotation R_A
       → Sends R_A|M⟩ to Bob
    2. Bob applies his random rotation R_B
       → Sends R_B·R_A|M⟩ back to Alice
    3. Alice removes her rotation (applies R_A⁻¹)
       → Sends R_B|M⟩ to Bob
    4. Bob removes his rotation (applies R_B⁻¹)
       → Recovers |M⟩
    
    KEY ADVANTAGE: No pre-shared key required!
    - Immune to key exhaustion attacks
    - Perfect for emergency PRIORITY_ACTION messages
    
    LIMITATIONS:
    - Requires 3 channel traversals (3× latency)
    - Only secure under collective noise (dephasing, rotation)
    - NOT secure under amplitude damping noise
    - Requires low QBER (< 3.8% per pass)
    """
    rotation_precision_bits: int = 8  # 256 possible rotation angles
    
    # Compatible noise models
    compatible_noise_models: Tuple[str, ...] = ("dephasing", "generic")
    
    # Maximum QBER for secure operation
    # After 3 passes: QBER_eff = 1 - (1-q)³
    # For QBER_eff < 11%: q < 3.8%
    max_single_pass_qber: float = 0.038
    
    def compute_effective_qber(self, single_pass_qber: float) -> float:
        """
        Compute effective QBER after 3 channel passes.
        
        QBER_effective ≈ 1 - (1 - QBER)³
        """
        q = clamp(single_pass_qber, 0.0, 0.5)
        return 1.0 - (1.0 - q) ** 3
    
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
            "extra_latency_factor": 3.0,  # 3× normal latency
        }
```

### 3.5 Layer 3: Quantum-TLS (KAK + Ping-Pong Combined)

```python
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
    - KAK doesn't use key pools → immune to exhaustion
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
        
        # Too many detections → abort
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
```

### 3.6 Protocol Configuration and Selection

```python
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
        if noise_model.lower() not in ("dephasing", "generic"):
            return QuantumProtocol.E91, "noise_incompatible_for_kak"
        if security_score < 0.5:
            # Channel very insecure, but emergency msg → try anyway
            return QuantumProtocol.QUANTUM_TLS, "emergency_despite_low_security"
    
    return preferred, "preferred_protocol"
```

---

## 4. Phase 2: Modify `quantum.py`

### 4.1 Add Imports

At the top of `quantum.py`, add:

```python
# Add after existing imports
from .quantum_protocols import (
    QuantumProtocol,
    QuantumProtocolConfig,
    PingPongIDS,
    PingPongVariant,
    E91KeyDistribution,
    KAKThreeStage,
    QuantumTLS,
    QuantumTLSConfig,
    select_protocol_for_message,
)
```

### 4.2 Modify `QuantumAugmentation.__init__`

Add protocol support to the constructor:

```python
class QuantumAugmentation:
    def __init__(
        self,
        env: simpy.Environment,
        rng: random.Random,
        *,
        key_policy: Optional[KeyPolicy] = None,
        default_pool: Optional[QKDKeyPool] = None,
        default_health: Optional[QuantumLinkHealth] = None,
        per_edge_pool: Optional[Dict[Edge, QKDKeyPool]] = None,
        per_edge_health: Optional[Dict[Edge, QuantumLinkHealth]] = None,
        finite_key_params: Optional[FiniteKeyParameters] = None,
        enable_replay_protection: bool = True,
        use_qrng_nonces: bool = True,
        # NEW: Protocol configuration
        protocol_config: Optional[QuantumProtocolConfig] = None,
    ):
        # ... existing initialization code ...
        
        # NEW: Initialize protocol handlers
        self.protocol_config = protocol_config or QuantumProtocolConfig()
        
        # Layer 1: Ping-Pong IDS
        if self.protocol_config.enable_pingpong_ids:
            self.pingpong_ids = PingPongIDS(
                probe_interval_s=self.protocol_config.ids_probe_interval_s,
                variant=self.protocol_config.ids_variant,
            )
        else:
            self.pingpong_ids = None
        
        # Layer 2: E91
        self.e91 = E91KeyDistribution(
            bell_test_fraction=self.protocol_config.e91_bell_test_fraction
        )
        
        # Layer 3: Quantum-TLS
        self.quantum_tls = QuantumTLS(
            config=self.protocol_config.qtls_config,
        )
        if self.pingpong_ids:
            self.quantum_tls.ids = self.pingpong_ids
```

### 4.3 Add Protocol Selection to `pre_send_hook`

Modify the `pre_send_hook` method to select protocols:

```python
def pre_send_hook(self, env: simpy.Environment, msg: Message, 
                  path_nodes: List[str]) -> simpy.Event:
    """
    SimPy hook called before network traversal.
    
    ENHANCED: Now selects appropriate quantum protocol based on message type.
    """
    start_ms = int(env.now * 1000)
    t_s = int(env.now)
    
    edges = [edge_key(path_nodes[i], path_nodes[i + 1]) 
             for i in range(len(path_nodes) - 1)]
    
    if not edges:
        return self.env.event().succeed()
    
    # Initialize edges
    for ek in edges:
        self._ensure_edge_initialized(ek)
    
    # Get channel state for first edge (representative)
    first_edge = edges[0]
    qber = self.health[first_edge].qber_at(t_s)
    noise_model = self.health[first_edge].noise_model.value
    
    # Get security score from Ping-Pong IDS
    security_score = 1.0
    if self.pingpong_ids:
        security_score = self.pingpong_ids.get_channel_security_score(t_s)
    
    # Select protocol
    protocol, selection_reason = select_protocol_for_message(
        msg=msg,
        config=self.protocol_config,
        qber=qber,
        noise_model=noise_model,
        security_score=security_score,
    )
    
    # Record selection in message payload
    msg.payload["quantum_protocol"] = protocol.value
    msg.payload["protocol_selection_reason"] = selection_reason
    msg.payload["channel_security_score"] = security_score
    
    # Handle based on selected protocol
    if protocol == QuantumProtocol.QUANTUM_TLS:
        return self.env.process(self._handle_quantum_tls(msg, edges, t_s))
    elif protocol == QuantumProtocol.E91:
        return self.env.process(self._handle_e91(msg, edges, t_s))
    elif protocol == QuantumProtocol.CLASSICAL:
        # No quantum processing needed
        msg.payload["key_bits_spent_total"] = 0.0
        msg.key_wait_ms = 0
        return self.env.event().succeed()
    else:
        # Default to existing BB84 handling
        return self.env.process(self._handle_bb84(msg, edges, t_s, start_ms))


def _handle_quantum_tls(self, msg: Message, edges: List[Edge], t_s: int):
    """
    Handle PRIORITY_ACTION with Quantum-TLS (KAK + Ping-Pong).
    
    KEY ADVANTAGE: Consumes ZERO key bits!
    """
    first_edge = edges[0]
    qber = self.health[first_edge].qber_at(t_s)
    noise_model = self.health[first_edge].noise_model.value
    security_score = 1.0
    if self.pingpong_ids:
        security_score = self.pingpong_ids.get_channel_security_score(t_s)
    
    # Attempt Quantum-TLS transmission
    success, protocol_used, metrics = self.quantum_tls.transmit_priority_action(
        msg=msg,
        qber=qber,
        noise_model=noise_model,
        t_s=t_s,
        rng=self.rng,
        security_score=security_score,
    )
    
    # Record metrics
    msg.payload["qtls_success"] = success
    msg.payload["qtls_protocol"] = protocol_used
    msg.payload["qtls_metrics"] = metrics
    
    if success and protocol_used == "quantum_tls":
        # KAK success - no keys consumed!
        msg.payload["key_bits_spent_total"] = 0.0
        msg.key_wait_ms = 0
        
        # Model 3-pass latency (simplified)
        extra_delay_ms = 30  # 3 passes × ~10ms
        yield self.env.timeout(extra_delay_ms / 1000.0)
    
    elif success and "fallback" in protocol_used:
        # Fallback to keyed protocol
        if "e91" in protocol_used:
            yield from self._handle_e91_inner(msg, edges, t_s)
        else:
            yield from self._handle_bb84_inner(msg, edges, t_s)
    
    else:
        # Failed - drop message
        from .model import DeliveryStatus
        msg.mark_dropped(DeliveryStatus.DROPPED_NO_KEYS, "qtls_failed")
    
    # Annotate path health
    self._annotate_health(msg, edges)


def _handle_e91(self, msg: Message, edges: List[Edge], t_s: int):
    """Handle with E91 entanglement-based keys."""
    yield from self._handle_e91_inner(msg, edges, t_s)


def _handle_e91_inner(self, msg: Message, edges: List[Edge], t_s: int):
    """E91 inner implementation with key consumption."""
    # E91 uses key pool similar to BB84, but with different rate
    per_hop_est = self.bits_required_per_hop(msg)
    
    for ek in edges:
        qber = self.health[ek].qber_at(t_s)
        e91_fraction = self.e91.secret_fraction_e91(qber, self.finite_key_params)
        msg.payload["e91_secret_fraction"] = e91_fraction
    
    # Use existing key consumption logic
    # (delegate to _handle_bb84_inner which handles key pool)
    yield from self._handle_bb84_inner(msg, edges, t_s)


def _handle_bb84(self, msg: Message, edges: List[Edge], t_s: int, start_ms: int):
    """Original BB84 handling (renamed for clarity)."""
    yield from self._handle_bb84_inner(msg, edges, t_s)


def _handle_bb84_inner(self, msg: Message, edges: List[Edge], t_s: int):
    """
    BB84 inner implementation - the existing key consumption logic.
    
    This is essentially the existing pre_send_hook logic, extracted.
    """
    # ... existing key consumption logic from pre_send_hook ...
    # (The existing code that waits for keys, consumes them, etc.)
    
    # For now, just yield to maintain generator structure
    yield self.env.timeout(0)
    # TODO: Move existing key consumption logic here
```

### 4.4 Add Ping-Pong Probe Scheduling

Add a method to schedule periodic Ping-Pong probes:

```python
def schedule_pingpong_probes(self, edges: List[Edge]):
    """Schedule periodic Ping-Pong IDS probes on all edges."""
    if not self.pingpong_ids:
        return
    
    def _probe_process():
        while True:
            yield self.env.timeout(self.pingpong_ids.probe_interval_s)
            t_s = int(self.env.now)
            
            for ek in edges:
                if ek not in self.health:
                    continue
                
                qber = self.health[ek].qber_at(t_s)
                self.pingpong_ids.send_probe(
                    t_s=t_s,
                    edge=ek,
                    qber=qber,
                    rng=self.rng,
                )
    
    self.env.process(_probe_process())
```

---

## 5. Phase 3: Modify `threat.py`

### 5.1 Add Insider Threat Classes

Add after the existing attack classes:

```python
# =============================================================================
# INSIDER THREAT MODEL
# =============================================================================

class InsiderThreatType(str, Enum):
    """Types of insider attacks from compromised coordinator."""
    MALICIOUS_CONTROL = "malicious_control"    # Send harmful commands
    KEY_EXHAUSTION = "insider_key_exhaustion"  # Abuse key access to drain pools
    FALSE_EMERGENCY = "false_emergency"        # Fake PRIORITY_ACTION


@dataclass
class CompromisedCoordinatorConfig:
    """
    Configuration for compromised coordinator (MG0) attack.
    
    The attacker has taken control of MG0 and can:
    - Send malicious CONTROL_SETPOINT commands
    - Send malicious PRIORITY_ACTION commands  
    - Flood authenticated messages to exhaust keys
    - All using LEGITIMATE credentials and key access
    """
    # Attack timing
    start_s: int = 120
    end_s: int = 480
    
    # Attack types to execute
    attack_types: List[InsiderThreatType] = field(
        default_factory=lambda: [InsiderThreatType.MALICIOUS_CONTROL]
    )
    
    # Target selection
    target_nodes: Optional[List[str]] = None  # None = all non-coordinator nodes
    
    # Malicious control parameters
    malicious_control_rate_per_s: float = 0.5
    forced_shed_frac: float = 0.75           # Harmful shed level
    use_priority_action: bool = True          # Use PRIORITY_ACTION (highest priority)
    
    # Key exhaustion parameters (insider version)
    # Insider CAN exhaust keys because they have legitimate access!
    insider_exhaust_rate_per_s: float = 5.0
    insider_exhaust_size_bytes: int = 500
    
    # Evasion
    mimic_legitimate_timing: bool = True      # Look like normal traffic
    
    # Labels for analysis
    label: str = "insider_compromised_coordinator"


class CompromisedCoordinatorAttack:
    """
    Simulates attacks from a compromised coordinator microgrid.
    
    THIS IS THE REALISTIC KEY EXHAUSTION SCENARIO:
    - Attacker controls MG0 (the coordinator)
    - Has legitimate QKD key access
    - Can send authenticated messages
    - Can drain key pools using legitimate credentials
    
    Unlike external attackers who cannot access keys, an insider
    with coordinator access poses a much more serious threat.
    """
    
    def __init__(
        self,
        env: simpy.Environment,
        rng: random.Random,
        cfg: CompromisedCoordinatorConfig,
        msg_id_fn: Callable[[], int],
        emit_fn: Callable[[Message], None],
        coordinator_node: str,
        all_nodes: List[str],
    ):
        self.env = env
        self.rng = rng
        self.cfg = cfg
        self.msg_id_fn = msg_id_fn
        self.emit_fn = emit_fn
        self.coordinator_node = coordinator_node
        self.all_nodes = all_nodes
        
        # Determine targets (all nodes except coordinator)
        if cfg.target_nodes:
            self.targets = [n for n in cfg.target_nodes if n != coordinator_node]
        else:
            self.targets = [n for n in all_nodes if n != coordinator_node]
        
        # Statistics
        self.stats = {
            "malicious_control_sent": 0,
            "malicious_priority_sent": 0,
            "insider_exhaust_sent": 0,
            "total_insider_messages": 0,
        }
    
    def schedule(self) -> None:
        """Schedule the insider attack."""
        self.env.process(self._run())
    
    def _run(self):
        """Main attack loop."""
        # Wait until attack starts
        if self.cfg.start_s > 0:
            yield self.env.timeout(self.cfg.start_s)
        
        while self.env.now < self.cfg.end_s:
            if not self.targets:
                yield self.env.timeout(1)
                continue
            
            # Execute configured attack types
            for attack_type in self.cfg.attack_types:
                if attack_type == InsiderThreatType.MALICIOUS_CONTROL:
                    self._send_malicious_control()
                elif attack_type == InsiderThreatType.KEY_EXHAUSTION:
                    self._send_insider_exhaust()
                elif attack_type == InsiderThreatType.FALSE_EMERGENCY:
                    self._send_false_emergency()
            
            # Wait based on attack rate
            interval = 1.0 / max(0.1, self.cfg.malicious_control_rate_per_s)
            if self.cfg.mimic_legitimate_timing:
                interval *= self.rng.uniform(0.7, 1.3)
            
            yield self.env.timeout(interval)
    
    def _send_malicious_control(self) -> None:
        """Send malicious control command."""
        target = self.rng.choice(self.targets)
        
        if self.cfg.use_priority_action:
            msg_type = MsgType.PRIORITY_ACTION
            priority = 2
            deadline_ms = 200
            payload = {
                "action": "shed_load_emergency",
                "forced_shed_frac": self.cfg.forced_shed_frac,
                "harm_duration_s": 60,
            }
            self.stats["malicious_priority_sent"] += 1
        else:
            msg_type = MsgType.CONTROL_SETPOINT
            priority = 1
            deadline_ms = 500
            payload = {
                "shed_frac_target": self.cfg.forced_shed_frac,
            }
            self.stats["malicious_control_sent"] += 1
        
        # Mark as attack (for analysis only - network doesn't see this)
        payload["attack"] = True
        payload["attack_label"] = self.cfg.label
        payload["attack_type"] = "insider_malicious_control"
        
        # Add legitimate-looking fields (insider has these!)
        payload["control_signature"] = "quam_ctrl_v1"
        payload["control_sender_role"] = "controller"
        
        msg = Message(
            msg_id=self.msg_id_fn(),
            created_ms=int(self.env.now * 1000),
            src=self.coordinator_node,  # Legitimate source!
            dst=target,
            msg_type=msg_type,
            priority=priority,
            deadline_ms=deadline_ms,
            size_bytes=280,
            requires_auth=True,
            requires_anon=False,
            is_attack=True,
            attack_label=self.cfg.label,
            payload=payload,
        )
        
        self.emit_fn(msg)
        self.stats["total_insider_messages"] += 1
    
    def _send_insider_exhaust(self) -> None:
        """
        Send authenticated messages to exhaust key pools.
        
        THIS IS REALISTIC because the insider HAS legitimate key access!
        """
        target = self.rng.choice(self.targets)
        
        payload = {
            "status_request": True,
            # Hidden attack markers
            "attack": True,
            "attack_label": self.cfg.label,
            "attack_type": "insider_key_exhaust",
        }
        
        msg = Message(
            msg_id=self.msg_id_fn(),
            created_ms=int(self.env.now * 1000),
            src=self.coordinator_node,
            dst=target,
            msg_type=MsgType.CONTROL_SETPOINT,  # Requires auth
            priority=1,
            deadline_ms=500,
            size_bytes=self.cfg.insider_exhaust_size_bytes,  # Larger = more key usage
            requires_auth=True,
            requires_anon=False,
            is_attack=True,
            attack_label=self.cfg.label,
            payload=payload,
        )
        
        self.emit_fn(msg)
        self.stats["insider_exhaust_sent"] += 1
        self.stats["total_insider_messages"] += 1
    
    def _send_false_emergency(self) -> None:
        """Send false PRIORITY_ACTION emergency."""
        target = self.rng.choice(self.targets)
        
        payload = {
            "action": "island_now",  # Unnecessary islanding
            "reason": "false_emergency",
            "attack": True,
            "attack_label": self.cfg.label,
            "attack_type": "insider_false_emergency",
        }
        
        msg = Message(
            msg_id=self.msg_id_fn(),
            created_ms=int(self.env.now * 1000),
            src=self.coordinator_node,
            dst=target,
            msg_type=MsgType.PRIORITY_ACTION,
            priority=2,
            deadline_ms=200,
            size_bytes=260,
            requires_auth=True,
            requires_anon=False,
            is_attack=True,
            attack_label=self.cfg.label,
            payload=payload,
        )
        
        self.emit_fn(msg)
        self.stats["malicious_priority_sent"] += 1
        self.stats["total_insider_messages"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get attack statistics."""
        return dict(self.stats)
```

### 5.2 Add New Defense Strategy

Add to the `DefenseStrategy` enum:

```python
class DefenseStrategy(str, Enum):
    # ... existing strategies ...
    QUANTUM_PROTOCOL_DEFENSE = "quantum_protocol_defense"
```

### 5.3 Update `get_defense_config`

Add the new strategy configuration:

```python
def get_defense_config(strategy: str, degraded_threshold_preset: str = "moderate") -> GateConfig:
    # ... existing code ...
    
    configs = {
        # ... existing configs ...
        
        "quantum_protocol_defense": GateConfig(
            # Basic settings
            verification_delay_ms=50,
            degraded_verification_delay_ms=150,
            degraded_secret_fraction=threshold,
            degraded_recover_secret_fraction=recover_threshold,
            degraded_recover_hold_s=recover_hold_s,
            
            # Block high-risk actions in degraded state
            block_priority_in_degraded=True,
            
            # Rate limiting
            enable_per_source_rate_limit=True,
            per_source_max_rate_per_s=2.0,
            per_source_window_s=10,
            per_source_burst_multiplier=3.0,
            
            # Intrusion detection (integrates with Ping-Pong IDS)
            block_during_intrusion=True,
            intrusion_lookback_s=60,
            intrusion_selective=True,
            
            # Behavioral plausibility (helps detect insider attacks)
            enable_plausibility_check=True,
            plausibility_max_shed_step=0.25,
            plausibility_healthy_shed_threshold=0.30,
            plausibility_healthy_deficit_max=0.05,
            
            # Cross-node correlation (detects coordinated insider attacks)
            enable_cross_node_correlation=True,
            correlation_window_s=30,
            max_simultaneous_targets=2,
            
            # Adaptive rate limiting
            adaptive_rate_limit=True,
            normal_rate_limit_per_s=5.0,
            degraded_rate_limit_per_s=1.0,
            
            # Command repetition blocking
            block_repeated_commands=True,
            command_cooldown_s=30,
            max_command_repetitions=2,
            
            # Control ACL
            enable_control_acl=True,
            require_control_signature=True,
        ),
    }
    
    return configs.get(strategy, configs["none"])
```

### 5.4 Integrate Ping-Pong IDS with PolicyGate

Modify `PolicyGate` to use Ping-Pong alerts:

```python
class PolicyGate:
    def __init__(
        self,
        cfg: GateConfig,
        intrusion_detector: Optional[IntrusionDetector] = None,
        microgrids: Optional[Dict[str, Any]] = None,
        # NEW: Ping-Pong IDS integration
        pingpong_ids: Optional["PingPongIDS"] = None,
    ):
        # ... existing init ...
        self.pingpong_ids = pingpong_ids
    
    def evaluate(self, msg: Message, action: Optional[ControlAction], 
                 now_s: int, degraded_active: bool) -> Tuple[ActionDecision, int]:
        # ... existing checks ...
        
        # NEW: Check Ping-Pong IDS alerts
        if self.pingpong_ids and self.cfg.block_during_intrusion:
            if self.pingpong_ids.has_recent_alert(now_s, self.cfg.intrusion_lookback_s):
                if self.cfg.intrusion_selective:
                    # Block only PRIORITY_ACTION during quantum channel alert
                    if msg.msg_type == MsgType.PRIORITY_ACTION and msg.requires_auth:
                        self.stats["blocked_pingpong_alert"] += 1
                        return ActionDecision("block", "pingpong_eve_detected"), 0
                else:
                    if msg.requires_auth:
                        self.stats["blocked_pingpong_alert"] += 1
                        return ActionDecision("block", "pingpong_channel_insecure"), 0
        
        # ... rest of existing checks ...
```

---

## 6. Phase 4: Modify `common.py`

### 6.1 Add Imports

```python
# Add to imports
from .quantum_protocols import (
    QuantumProtocolConfig,
    PingPongVariant,
    QuantumTLSConfig,
)
from .threat import (
    CompromisedCoordinatorConfig,
    CompromisedCoordinatorAttack,
    InsiderThreatType,
)
```

### 6.2 Add Insider Attack Scheduling

```python
def schedule_insider_attack(
    ctx: "SimContext",
    start_s: int,
    end_s: int,
    attack_types: Optional[List[str]] = None,
    forced_shed_frac: float = 0.75,
    control_rate_per_s: float = 0.5,
    exhaust_rate_per_s: float = 5.0,
    use_priority_action: bool = True,
) -> CompromisedCoordinatorAttack:
    """
    Schedule an insider attack from compromised coordinator.
    
    Args:
        ctx: Simulation context
        start_s: Attack start time
        end_s: Attack end time
        attack_types: List of attack types ("malicious_control", "key_exhaustion", "false_emergency")
        forced_shed_frac: Malicious shed level (0-1)
        control_rate_per_s: Rate of malicious commands
        exhaust_rate_per_s: Rate of key exhaustion flood
        use_priority_action: Use PRIORITY_ACTION (True) or CONTROL_SETPOINT (False)
    
    Returns:
        The configured attack object
    """
    # Parse attack types
    types = []
    if attack_types:
        for t in attack_types:
            if t == "malicious_control":
                types.append(InsiderThreatType.MALICIOUS_CONTROL)
            elif t == "key_exhaustion":
                types.append(InsiderThreatType.KEY_EXHAUSTION)
            elif t == "false_emergency":
                types.append(InsiderThreatType.FALSE_EMERGENCY)
    else:
        types = [InsiderThreatType.MALICIOUS_CONTROL]
    
    # Get coordinator node (first node by convention)
    nodes = list(ctx.microgrids.keys())
    coordinator = sorted(nodes)[0] if nodes else "mg0"
    
    cfg = CompromisedCoordinatorConfig(
        start_s=start_s,
        end_s=end_s,
        attack_types=types,
        malicious_control_rate_per_s=control_rate_per_s,
        forced_shed_frac=forced_shed_frac,
        use_priority_action=use_priority_action,
        insider_exhaust_rate_per_s=exhaust_rate_per_s,
    )
    
    attack = CompromisedCoordinatorAttack(
        env=ctx.env,
        rng=ctx.rng,
        cfg=cfg,
        msg_id_fn=ctx.msg_id_fn,
        emit_fn=ctx.emit_fn,
        coordinator_node=coordinator,
        all_nodes=nodes,
    )
    
    attack.schedule()
    return attack
```

### 6.3 Add Protocol Configuration Helper

```python
def get_quantum_protocol_config(
    enable_quantum_tls: bool = True,
    enable_pingpong_ids: bool = True,
    ids_variant: str = "ghz",
    e91_bell_test_fraction: float = 0.333,
) -> QuantumProtocolConfig:
    """
    Create quantum protocol configuration.
    
    Args:
        enable_quantum_tls: Use Quantum-TLS for PRIORITY_ACTION
        enable_pingpong_ids: Enable Ping-Pong intrusion detection
        ids_variant: IDS variant ("bell", "ghz", "cluster")
        e91_bell_test_fraction: Fraction of E91 pairs for Bell test
    
    Returns:
        QuantumProtocolConfig
    """
    variant_map = {
        "bell": PingPongVariant.BELL,
        "ghz": PingPongVariant.GHZ,
        "cluster": PingPongVariant.CLUSTER,
    }
    
    from .quantum_protocols import QuantumProtocol
    
    return QuantumProtocolConfig(
        priority_action_protocol=(
            QuantumProtocol.QUANTUM_TLS if enable_quantum_tls 
            else QuantumProtocol.BB84
        ),
        control_setpoint_protocol=QuantumProtocol.E91,
        enable_pingpong_ids=enable_pingpong_ids,
        ids_variant=variant_map.get(ids_variant.lower(), PingPongVariant.GHZ),
        e91_bell_test_fraction=e91_bell_test_fraction,
    )
```

### 6.4 Update QUANTUM_DEFENSE_STRATEGIES

```python
QUANTUM_DEFENSE_STRATEGIES = {
    "ratelimit_v2",
    "intrusion_v2",
    "plausibility",
    "correlation",
    "quarantine_v2",
    "hardened",
    "hardened_balanced",
    "hardened_strong",
    "quantum_only",
    "quantum_protocol_defense",  # NEW
}
```

---

## 7. Phase 5: Modify `finalmain.py`

### 7.1 Add CLI Arguments

```python
parser.add_argument("--enable-quantum-protocols", action="store_true",
                    help="Enable multi-layer quantum protocols (KAK, E91, Ping-Pong)")
parser.add_argument("--quantum-tls", action="store_true",
                    help="Use Quantum-TLS for PRIORITY_ACTION messages")
parser.add_argument("--pingpong-ids", action="store_true",
                    help="Enable Ping-Pong intrusion detection")
parser.add_argument("--ids-variant", type=str, default="ghz",
                    choices=["bell", "ghz", "cluster"],
                    help="Ping-Pong IDS variant")

# Insider attack arguments
parser.add_argument("--enable-insider-attack", action="store_true",
                    help="Enable insider attack (compromised coordinator)")
parser.add_argument("--insider-attack-types", type=str, default="malicious_control",
                    help="Comma-separated insider attack types: malicious_control,key_exhaustion,false_emergency")
parser.add_argument("--insider-shed-frac", type=float, default=0.75,
                    help="Malicious shed fraction for insider attack")
parser.add_argument("--insider-rate", type=float, default=0.5,
                    help="Insider attack rate per second")
```

### 7.2 Initialize Quantum Protocols

```python
# After creating QuantumAugmentation, configure protocols
protocol_config = None
if args.enable_quantum_protocols or args.quantum_tls or args.pingpong_ids:
    protocol_config = get_quantum_protocol_config(
        enable_quantum_tls=args.quantum_tls,
        enable_pingpong_ids=args.pingpong_ids,
        ids_variant=args.ids_variant,
    )

# Pass to QuantumAugmentation
qlayer = QuantumAugmentation(
    env=env,
    rng=rng,
    # ... existing args ...
    protocol_config=protocol_config,
)

# Schedule Ping-Pong probes if enabled
if protocol_config and protocol_config.enable_pingpong_ids:
    all_edges = list(qlayer.pools.keys())
    qlayer.schedule_pingpong_probes(all_edges)
```

### 7.3 Schedule Insider Attack

```python
# Schedule insider attack if enabled
insider_attack = None
if args.enable_insider_attack:
    attack_types = args.insider_attack_types.split(",")
    for start, end in attack_windows:
        insider_attack = schedule_insider_attack(
            ctx=ctx,
            start_s=start,
            end_s=end,
            attack_types=attack_types,
            forced_shed_frac=args.insider_shed_frac,
            control_rate_per_s=args.insider_rate,
        )
```

### 7.4 Pass Ping-Pong IDS to PolicyGate

```python
# When creating PolicyGate
gate = PolicyGate(
    cfg=gate_cfg,
    intrusion_detector=intrusion_detector if use_intrusion else None,
    microgrids=microgrids,
    pingpong_ids=qlayer.pingpong_ids if hasattr(qlayer, 'pingpong_ids') else None,
)
```

---

## 8. Phase 6: Modify `metrics.py`

### 8.1 Add Protocol Metrics

```python
@dataclass
class QuantumProtocolMetrics:
    """Metrics for quantum protocol usage."""
    protocol_selections: Dict[str, int] = field(default_factory=lambda: {
        "quantum_tls": 0,
        "e91": 0,
        "bb84": 0,
        "classical": 0,
    })
    
    qtls_success: int = 0
    qtls_fallback_e91: int = 0
    qtls_fallback_bb84: int = 0
    qtls_failed: int = 0
    
    pingpong_probes: int = 0
    pingpong_detections: int = 0
    
    insider_messages_total: int = 0
    insider_messages_blocked: int = 0
    insider_messages_allowed: int = 0
    
    def record_protocol_selection(self, protocol: str) -> None:
        if protocol in self.protocol_selections:
            self.protocol_selections[protocol] += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_selections": self.protocol_selections,
            "qtls_success": self.qtls_success,
            "qtls_fallback_e91": self.qtls_fallback_e91,
            "qtls_fallback_bb84": self.qtls_fallback_bb84,
            "qtls_failed": self.qtls_failed,
            "qtls_success_rate": (
                self.qtls_success / (self.qtls_success + self.qtls_fallback_e91 + 
                                     self.qtls_fallback_bb84 + self.qtls_failed)
                if (self.qtls_success + self.qtls_fallback_e91 + 
                    self.qtls_fallback_bb84 + self.qtls_failed) > 0 else 0.0
            ),
            "pingpong_probes": self.pingpong_probes,
            "pingpong_detections": self.pingpong_detections,
            "pingpong_detection_rate": (
                self.pingpong_detections / self.pingpong_probes
                if self.pingpong_probes > 0 else 0.0
            ),
            "insider_messages_total": self.insider_messages_total,
            "insider_messages_blocked": self.insider_messages_blocked,
            "insider_block_rate": (
                self.insider_messages_blocked / self.insider_messages_total
                if self.insider_messages_total > 0 else 0.0
            ),
        }
```

---

## 9. Testing Checklist

### Basic Functionality

- [ ] `quantum_protocols.py` imports without errors
- [ ] `PingPongIDS.send_probe()` returns valid results
- [ ] `E91KeyDistribution.secret_fraction_e91()` computes correct rates
- [ ] `KAKThreeStage.is_channel_compatible()` validates correctly
- [ ] `QuantumTLS.transmit_priority_action()` works for PRIORITY_ACTION
- [ ] Protocol selection chooses correct protocol per message type

### Integration

- [ ] `QuantumAugmentation` initializes with `protocol_config`
- [ ] `pre_send_hook` selects protocols correctly
- [ ] PRIORITY_ACTION uses Quantum-TLS (0 key bits)
- [ ] CONTROL_SETPOINT uses E91
- [ ] Ping-Pong probes scheduled and running
- [ ] PolicyGate receives Ping-Pong alerts

### Attacks

- [ ] `CompromisedCoordinatorAttack` sends messages from coordinator
- [ ] Insider messages marked as `is_attack=True`
- [ ] Insider key exhaustion consumes keys
- [ ] Insider malicious control has high shed fraction

### Defenses

- [ ] `quantum_protocol_defense` strategy loads correctly
- [ ] Plausibility check blocks implausible insider commands
- [ ] Ping-Pong alerts trigger gate blocking
- [ ] Quantum-TLS survives key exhaustion

### Metrics

- [ ] Protocol selection counts recorded
- [ ] QTLS success/fallback rates computed
- [ ] Insider attack stats captured

---

## Summary

This specification covers:

1. **NEW FILE**: `quantum_protocols.py` with KAK, Ping-Pong IDS, E91, Quantum-TLS
2. **MODIFY**: `quantum.py` - protocol selection in `pre_send_hook`
3. **MODIFY**: `threat.py` - insider threat, new defense strategy
4. **MODIFY**: `common.py` - scheduling helpers
5. **MODIFY**: `finalmain.py` - CLI arguments and initialization
6. **MODIFY**: `metrics.py` - protocol metrics

**Key Research Contributions**:
- Quantum-TLS (KAK + Ping-Pong) for PRIORITY_ACTION → immune to key exhaustion
- Realistic insider threat model (compromised coordinator)
- Multi-layer quantum defense with behavioral detection

**REMOVED** (unrealistic):
- External attacker key exhaustion
- Peer-to-peer control between microgrids
