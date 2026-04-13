# QuAM: Quantum-Augmented Microgrid Simulator

A four-layer discrete-event cyber-physical simulator for evaluating quantum-augmented security in networked microgrids.

## Overview

QuAM models the interaction between four coupled layers:
1. **Physical Layer** — power generation (solar, wind, SMR), battery storage, loads, and grid stability
2. **Communication Layer** — configurable topologies (star, ring, mesh, two-cluster bridge) with latency, jitter, and packet-loss modeling
3. **Quantum Security Layer** — QKD key pools, Ping-Pong IDS, QCA token authentication, QTLS, and QRNG sensor challenges
4. **Threat-Defense Layer** — configurable attack injection (FDI, spoofing, coordinated multi-node, MITM, key-exhaustion) and multi-stage defense pipelines

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.11+.

## Quick Start

Run a basic simulation with the quantum security demo:

```bash
python run_quantum_security_demo.py
```

To reproduce the paper experiments (1-hour horizon, stressed infrastructure):

```bash
python rerun_all_1hr_stressed.py
```

## Simulator Parameters

| Parameter | Value |
|-----------|-------|
| Solar / Wind / SMR capacity | 80 / 80 / 40 kW per MG |
| Battery capacity / initial SOC | 60 kWh / 20 kWh |
| Import cap | 35 kW per MG |
| Base load / AI load | 120 / 30 kW per MG |
| QKD key pool capacity | 10,000 bits |
| Key refill rate | 1,000 bits/s |
| IDS threshold (τ) | 2.5% QBER |
| QKD abort threshold | 11% QBER |

## Project Structure

```
quam-simulator/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── run_quantum_security_demo.py      # Main experiment runner
├── rerun_all_1hr_stressed.py         # Paper reproduction script
├── QUAM_IMPLEMENTATION_SPEC_V2.md    # Detailed technical specification
└── quam_microgrid/
    └── quam/
        ├── finalmain.py              # Core simulation loop
        ├── model.py                  # Data models (MicrogridParams, EnergyRecord)
        ├── quantum.py                # QKD implementation
        ├── quantum_protocols.py      # Protocol abstractions (E91, Kak, BB84)
        ├── threat.py                 # Attack models and scenarios
        ├── network.py                # Network topology and communication
        ├── metrics.py                # Performance metrics
        ├── common.py                 # Shared utilities
        ├── generation.py             # Energy generation profiles
        ├── network_metrics.py        # Network-specific metrics
        └── plot/                     # Visualization scripts
```

## Defense Tiers

- **No defense**: No authentication, no rate limiting
- **Classical**: HMAC + rate limiting + classical IDS
- **Quantum**: QKD + Ping-Pong IDS + QCA tokens + QTLS

## Citation

If you use QuAM in your research, please cite:

```bibtex
@inproceedings{jha2026quam,
  title={A Novel Quantum Augmented Framework to Improve Microgrid Cybersecurity},
  author={Jha, Nitin and Parakh, Abhishek and Subramaniam, Mahadevan},
  booktitle={Proceedings of SPIE Defense + Commercial Sensing},
  year={2026}
}
```

## Acknowledgments

This work is partly sponsored by the National Science Foundation (NSF) awards numbers 2324924 and 2324925.

## License

MIT License. See [LICENSE](LICENSE) for details.
