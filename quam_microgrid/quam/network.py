"""
network.py

Discrete-event packet/network simulation for QuAM.

Responsibilities:
- Topology creation (multiple simple topologies)
- Link models: latency, jitter, bandwidth, loss, queueing
- Message delivery with deadlines and drop reasons
- Hooks for integration:
    - pre_send_hook(env, msg, path_nodes) -> simpy.Event (may drop msg)
    - on_deliver_hook(env, msg) -> None

Non-responsibilities:
- No QKD, no QAN logic, no spoof logic, no microgrid actions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from collections import Counter
import itertools
import os
import random

import simpy
import networkx as nx

from .model import Message, DeliveryStatus


# -------------------------
# Link model
# -------------------------

@dataclass(frozen=True)
class LinkParams:
    """
    Link parameters for a single undirected edge.

    bandwidth_bytes_per_s models a simple serialization delay:
        tx_time_s = msg.size_bytes / bandwidth_bytes_per_s
    """
    base_latency_ms: int = 15
    jitter_ms: int = 8
    bandwidth_bytes_per_s: float = 80_000.0
    loss_prob: float = 0.001
    queue_capacity: int = 1  # SimPy resource capacity


@dataclass
class LinkModel:
    u: str
    v: str
    params: LinkParams
    resource: simpy.Resource
    # Queue statistics (captured opportunistically in _tx_over_link()).
    total_queued: int = 0
    total_served: int = 0
    max_queue_depth: int = 0
    cumulative_queue_time_ms: float = 0.0

    def queue_depth(self) -> int:
        """Current number of items either waiting or being served on this link."""
        return int(len(getattr(self.resource, "queue", []))) + int(getattr(self.resource, "count", 0))

    def utilization(self) -> float:
        """Fraction of link 'capacity' in use (capacity == parallel transmissions)."""
        cap = int(getattr(self.params, "queue_capacity", 0))
        if cap <= 0:
            return 0.0
        return float(getattr(self.resource, "count", 0)) / float(cap)

    def is_congested(self, threshold: float = 0.8) -> bool:
        """
        Congestion heuristic.

        SimPy Resources have unbounded queues; treat any backlog as congested,
        otherwise fall back to high utilization.
        """
        if int(len(getattr(self.resource, "queue", []))) > 0:
            return True
        return self.utilization() > float(threshold)

    def record_queue_entry(self, now_s: float) -> float:
        """Call when a message begins contending for this link."""
        self.total_queued += 1
        # Include this message in the depth estimate. The SimPy request is not
        # yet enqueued when this hook runs.
        depth = self.queue_depth() + 1
        if depth > self.max_queue_depth:
            self.max_queue_depth = depth
        return float(now_s)

    def record_queue_exit(self, entry_time_s: float, exit_time_s: float) -> None:
        """Call when a message acquires the resource (exits queue)."""
        self.total_served += 1
        self.cumulative_queue_time_ms += (float(exit_time_s) - float(entry_time_s)) * 1000.0

    def avg_queue_time_ms(self) -> float:
        if self.total_served <= 0:
            return 0.0
        return float(self.cumulative_queue_time_ms) / float(self.total_served)


def edge_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


# -------------------------
# Topology builders
# -------------------------

def build_ieee_topology(name: str) -> nx.Graph:
    """
    Build IEEE test feeder topologies commonly used in power systems research.

    Supported topologies:
    - "ieee13": IEEE 13-bus test feeder (12 edges, small distribution)
    - "ieee34": IEEE 34-bus test feeder (33 edges, rural distribution)
    - "ieee37": IEEE 37-bus test feeder (36 edges, underground network)
    - "ieee123": IEEE 123-bus test feeder (122 edges, large system)

    Returns:
        NetworkX Graph with node IDs as strings.
    """
    g = nx.Graph()

    if name == "ieee13":
        # IEEE 13 Node Test Feeder
        edges = [
            ("650", "632"), ("632", "633"), ("632", "645"), ("632", "671"),
            ("633", "634"), ("645", "646"), ("671", "684"), ("671", "680"),
            ("671", "692"), ("684", "611"), ("684", "652"), ("692", "675"),
        ]
        g.add_edges_from(edges)

        # Node type annotations are optional; used for future per-edge parameter tuning.
        g.graph["node_types"] = {
            "650": "substation",
            "632": "junction", "633": "junction", "671": "junction",
            "684": "junction", "692": "junction",
            "634": "load", "645": "load", "646": "load", "680": "load",
            "611": "load", "652": "load", "675": "load",
        }

    elif name == "ieee34":
        # IEEE 34 Node Test Feeder (simplified - main trunk + laterals)
        edges = [
            # Main trunk
            ("800", "802"), ("802", "806"), ("806", "808"), ("808", "810"),
            ("810", "812"), ("812", "814"), ("814", "850"), ("850", "816"),
            ("816", "818"), ("818", "820"), ("820", "822"), ("822", "824"),
            ("824", "826"), ("826", "828"), ("828", "830"), ("830", "854"),
            ("854", "856"), ("856", "852"), ("852", "832"), ("832", "858"),
            ("858", "834"), ("834", "860"), ("860", "836"), ("836", "840"),
            # Laterals
            ("816", "862"), ("832", "888"), ("888", "890"),
            ("834", "842"), ("842", "844"), ("844", "846"), ("846", "848"),
            ("858", "864"), ("834", "866"), ("866", "868"),
        ]
        g.add_edges_from(edges)

    elif name == "ieee37":
        # IEEE 37 Node Test Feeder (underground network)
        edges = [
            ("799", "701"), ("701", "702"), ("702", "703"), ("703", "727"),
            ("703", "730"), ("704", "714"), ("704", "720"), ("705", "742"),
            ("706", "725"), ("707", "724"), ("707", "722"), ("708", "733"),
            ("708", "732"), ("709", "731"), ("709", "708"), ("710", "735"),
            ("710", "736"), ("711", "741"), ("711", "740"), ("712", "742"),
            ("713", "704"), ("714", "718"), ("720", "707"), ("720", "706"),
            ("727", "744"), ("730", "709"), ("733", "734"), ("734", "737"),
            ("734", "738"), ("737", "738"), ("738", "711"), ("744", "728"),
            ("744", "729"), ("775", "709"), ("799", "775"),
        ]
        g.add_edges_from(edges)

    elif name == "ieee123":
        # IEEE 123 Node Test Feeder (large unbalanced system)
        # Backbone + laterals (coarse graph abstraction for cyber studies).
        edges = [
            ("150", "149"), ("149", "1"), ("1", "2"), ("1", "3"), ("1", "7"),
            ("3", "4"), ("3", "5"), ("5", "6"), ("7", "8"), ("8", "12"),
            ("8", "9"), ("8", "13"), ("9", "14"), ("13", "34"), ("13", "18"),
            ("14", "11"), ("14", "10"), ("15", "16"), ("15", "17"), ("18", "19"),
            ("18", "21"), ("19", "20"), ("21", "22"), ("21", "23"), ("23", "24"),
            ("23", "25"), ("25", "26"), ("25", "28"), ("26", "27"), ("26", "31"),
            ("27", "33"), ("28", "29"), ("29", "30"), ("30", "250"), ("31", "32"),
            ("34", "15"), ("35", "36"), ("35", "40"), ("36", "37"), ("36", "38"),
            ("38", "39"), ("40", "41"), ("40", "42"), ("42", "43"), ("42", "44"),
            ("44", "45"), ("44", "47"), ("45", "46"), ("47", "48"), ("47", "49"),
            ("49", "50"), ("50", "51"), ("51", "151"), ("52", "53"), ("53", "54"),
            ("54", "55"), ("54", "57"), ("55", "56"), ("57", "58"), ("57", "60"),
            ("58", "59"), ("60", "61"), ("60", "62"), ("62", "63"), ("63", "64"),
            ("64", "65"), ("65", "66"), ("66", "67"), ("67", "68"), ("67", "72"),
            ("67", "97"), ("68", "69"), ("69", "70"), ("70", "71"), ("72", "73"),
            ("72", "76"), ("73", "74"), ("74", "75"), ("76", "77"), ("76", "86"),
            ("77", "78"), ("78", "79"), ("78", "80"), ("80", "81"), ("81", "82"),
            ("81", "84"), ("82", "83"), ("84", "85"), ("86", "87"), ("87", "88"),
            ("87", "89"), ("89", "90"), ("89", "91"), ("91", "92"), ("91", "93"),
            ("93", "94"), ("93", "95"), ("95", "96"), ("97", "98"), ("98", "99"),
            ("99", "100"), ("100", "450"), ("101", "102"), ("101", "105"),
            ("102", "103"), ("103", "104"), ("105", "106"), ("105", "108"),
            ("106", "107"), ("108", "109"), ("108", "300"), ("109", "110"),
            ("110", "111"), ("110", "112"), ("112", "113"), ("113", "114"),
            ("135", "35"), ("152", "52"), ("160", "67"), ("197", "101"),
        ]
        g.add_edges_from(edges)

    else:
        raise ValueError(
            f"Unknown IEEE topology: {name}. Supported: ieee13, ieee34, ieee37, ieee123"
        )

    return g


def build_topology(name: str, nodes: List[str], rng: random.Random) -> nx.Graph:
    """
    Build a small graph topology for experiments.

    Supported names:
    - "ring"
    - "star" (node[0] is hub)
    - "mesh" (ring + random chords)
    - "two_cluster_bridge" (two clusters linked by a bridge)

    Notes:
    - For conference scope, keep n small (5 to 12).
    """
    # Federated topologies: two independent grids joined by a tie-line.
    if name.startswith("federated_"):
        return build_federated_topology(name, nodes, rng)

    # IEEE feeder topologies define their own node IDs; ignore `nodes`.
    if name.startswith("ieee"):
        return build_ieee_topology(name)

    if len(nodes) <= 0:
        raise ValueError("need at least 1 node")

    # Allow degenerate topologies for sanity checks (e.g., 2 nodes, 1 link).
    # Some experiments use short horizons and minimal graphs to validate invariants.
    if len(nodes) == 1:
        g = nx.Graph()
        g.add_node(nodes[0])
        if name in ("ring", "star", "mesh"):
            return g
        raise ValueError(f"topology {name} needs more nodes (got 1)")

    g = nx.Graph()
    g.add_nodes_from(nodes)

    if name == "ring":
        for i in range(len(nodes)):
            g.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])

    elif name == "star":
        hub = nodes[0]
        for n in nodes[1:]:
            g.add_edge(hub, n)

    elif name == "mesh":
        # Start from ring then add a few chords
        for i in range(len(nodes)):
            g.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])
        # Add chords: about n/2 additional edges
        extra = max(1, len(nodes) // 2)
        candidates = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 2, len(nodes))]
        rng.shuffle(candidates)
        added = 0
        for (u, v) in candidates:
            if added >= extra:
                break
            if not g.has_edge(u, v):
                g.add_edge(u, v)
                added += 1

    elif name == "two_cluster_bridge":
        # Split into two clusters, connect each internally as a ring, then add one bridge
        mid = len(nodes) // 2
        a = nodes[:mid]
        b = nodes[mid:]
        if len(a) < 2 or len(b) < 2:
            raise ValueError("two_cluster_bridge needs at least 4 nodes")

        for i in range(len(a)):
            g.add_edge(a[i], a[(i + 1) % len(a)])
        for i in range(len(b)):
            g.add_edge(b[i], b[(i + 1) % len(b)])

        # Bridge connects the controllers (MG0 of each sub-cluster)
        # a[0] = MG0 (main controller), b[0] = MG(n/2) (cluster-B controller)
        g.add_edge(a[0], b[0])

    else:
        raise ValueError(f"unknown topology name: {name}")

    return g


# ── Federated Topology: two independent grids + tie-line ──────────

# Tie-line parameters: higher latency, lower bandwidth, higher loss
TIELINE_PARAMS = LinkParams(
    base_latency_ms=50,          # 3× higher than normal (15 ms)
    jitter_ms=20,                # Higher jitter on long-haul link
    bandwidth_bytes_per_s=40_000.0,  # Half normal bandwidth
    loss_prob=0.005,             # 5× higher loss
    queue_capacity=1,
)

# Mapping from topology suffix to canonical name
_TOPO_ALIASES = {
    "ring": "ring",
    "star": "star",
    "mesh": "mesh",
    "two_cluster_bridge": "two_cluster_bridge",
    "two_cluster": "two_cluster_bridge",
    "tcb": "two_cluster_bridge",
}


def _parse_federated_name(name: str) -> Tuple[str, str]:
    """
    Parse 'federated_<topo_a>_<topo_b>' into (topo_a, topo_b).

    Handles multi-word topology names like 'two_cluster_bridge'.
    Strategy: try all known topology names greedily from the left.
    """
    suffix = name.replace("federated_", "", 1)

    # Try all possible split points
    for alias_a, canon_a in sorted(_TOPO_ALIASES.items(), key=lambda x: -len(x[0])):
        if suffix.startswith(alias_a + "_"):
            rest = suffix[len(alias_a) + 1:]
            canon_b = _TOPO_ALIASES.get(rest)
            if canon_b is not None:
                return canon_a, canon_b

    # Fallback: split on first underscore
    parts = suffix.split("_", 1)
    if len(parts) == 2:
        a = _TOPO_ALIASES.get(parts[0], parts[0])
        b = _TOPO_ALIASES.get(parts[1], parts[1])
        return a, b

    raise ValueError(f"Cannot parse federated topology name: {name}")


def build_federated_topology(
    name: str, nodes: List[str], rng: random.Random,
) -> nx.Graph:
    """
    Build a federated topology: two independent sub-grids joined by a
    single inter-domain tie-line.

    The node list is split in half:
      - Grid A: nodes[:mid]  with topology topo_a
      - Grid B: nodes[mid:]  with topology topo_b

    Each node is annotated with graph.nodes[n]["domain"] = "grid_a"|"grid_b".
    The tie-line connects nodes_a[0] <-> nodes_b[0] and is annotated with
    graph[u][v]["tie_line"] = True.

    Examples:
      "federated_ring_star"               → ring(A) + star(B)
      "federated_ring_two_cluster_bridge"  → ring(A) + two_cluster_bridge(B)
      "federated_mesh_star"               → mesh(A) + star(B)
    """
    topo_a, topo_b = _parse_federated_name(name)

    mid = len(nodes) // 2
    nodes_a = nodes[:mid]
    nodes_b = nodes[mid:]

    if len(nodes_a) < 2 or len(nodes_b) < 2:
        raise ValueError(
            f"Federated topology needs >= 4 nodes (got {len(nodes)}; "
            f"grid_a={len(nodes_a)}, grid_b={len(nodes_b)})"
        )

    # Build sub-grids independently
    g_a = build_topology(topo_a, nodes_a, rng)
    g_b = build_topology(topo_b, nodes_b, rng)

    # Compose into single graph
    g = nx.compose(g_a, g_b)

    # Domain annotations
    for n in nodes_a:
        g.nodes[n]["domain"] = "grid_a"
    for n in nodes_b:
        g.nodes[n]["domain"] = "grid_b"

    # Tie-line connecting the "head" of each sub-grid
    tie_u, tie_v = nodes_a[0], nodes_b[0]
    g.add_edge(tie_u, tie_v)
    g.edges[tie_u, tie_v]["tie_line"] = True
    g.edges[tie_u, tie_v]["link_type"] = "tie"

    # Store federation metadata on graph
    g.graph["federated"] = True
    g.graph["topo_a"] = topo_a
    g.graph["topo_b"] = topo_b
    g.graph["nodes_a"] = list(nodes_a)
    g.graph["nodes_b"] = list(nodes_b)
    g.graph["tie_line"] = (tie_u, tie_v)

    return g


def get_link_params_for_type(link_type: str) -> LinkParams:
    """
    Factory for link parameters by infrastructure type.

    This is a lightweight proxy for different comms link qualities in
    distribution grids (backbone/feeder/lateral/tie).
    """
    configs: Dict[str, LinkParams] = {
        "backbone": LinkParams(
            base_latency_ms=5,
            jitter_ms=2,
            bandwidth_bytes_per_s=1_000_000.0,
            loss_prob=0.0001,
            queue_capacity=10,
        ),
        "feeder": LinkParams(
            base_latency_ms=15,
            jitter_ms=5,
            bandwidth_bytes_per_s=100_000.0,
            loss_prob=0.001,
            queue_capacity=5,
        ),
        "lateral": LinkParams(
            base_latency_ms=25,
            jitter_ms=10,
            bandwidth_bytes_per_s=50_000.0,
            loss_prob=0.005,
            queue_capacity=2,
        ),
        "tie": LinkParams(
            base_latency_ms=40,
            jitter_ms=15,
            bandwidth_bytes_per_s=20_000.0,
            loss_prob=0.01,
            queue_capacity=3,
        ),
    }
    return configs.get(str(link_type), configs["feeder"])


def infer_link_type(g: nx.Graph, u: str, v: str) -> str:
    """
    Infer link type based on local graph structure.

    Heuristics:
    - Edges to leaf nodes (degree 1) are laterals
    - Edges between high-degree nodes are backbone
    - Otherwise feeder
    """
    u_deg = int(g.degree(u))
    v_deg = int(g.degree(v))

    if u_deg == 1 or v_deg == 1:
        return "lateral"
    if u_deg >= 4 and v_deg >= 4:
        return "backbone"
    return "feeder"


def assign_link_types(g: nx.Graph) -> Dict[Tuple[str, str], str]:
    """Assign a link type to every edge in g.  Respects explicit edge attributes."""
    out: Dict[Tuple[str, str], str] = {}
    for (u, v) in g.edges():
        # Explicit link_type from federated topology takes precedence
        explicit = g.edges[u, v].get("link_type")
        if explicit:
            out[edge_key(u, v)] = str(explicit)
        else:
            out[edge_key(u, v)] = infer_link_type(g, u, v)
    return out


def build_links(
    env: simpy.Environment,
    g: nx.Graph,
    rng: random.Random,
    *,
    default_params: LinkParams = LinkParams(),
    per_edge_params: Optional[Dict[Tuple[str, str], LinkParams]] = None,
    use_grid_link_types: bool = False,
) -> Dict[Tuple[str, str], LinkModel]:
    """
    Create LinkModel objects for all edges in g.

    If per_edge_params is provided, it overrides default_params for those edges.
    """
    # Optionally infer per-edge parameters based on link "type".
    if use_grid_link_types:
        inferred = assign_link_types(g)
        merged: Dict[Tuple[str, str], LinkParams] = dict(per_edge_params) if per_edge_params else {}
        for ek, ltype in inferred.items():
            merged.setdefault(ek, get_link_params_for_type(ltype))
        per_edge_params = merged

    links: Dict[Tuple[str, str], LinkModel] = {}
    for (u, v) in g.edges():
        ek = edge_key(u, v)
        p = per_edge_params.get(ek, default_params) if per_edge_params else default_params

        # Optional small randomization for realism while staying controlled
        # You can remove this if you want deterministic per-edge params.
        base = max(1, int(p.base_latency_ms + rng.randint(-2, 2)))
        jitter = max(0, int(p.jitter_ms + rng.randint(-2, 2)))
        bw = max(1.0, float(p.bandwidth_bytes_per_s * (0.9 + 0.2 * rng.random())))
        loss = min(0.2, max(0.0, float(p.loss_prob)))

        p2 = LinkParams(
            base_latency_ms=base,
            jitter_ms=jitter,
            bandwidth_bytes_per_s=bw,
            loss_prob=loss,
            queue_capacity=p.queue_capacity,
        )

        links[ek] = LinkModel(
            u=u,
            v=v,
            params=p2,
            resource=simpy.Resource(env, capacity=p2.queue_capacity),
        )
    return links


# -------------------------
# Network simulator
# -------------------------

PreSendHook = Callable[[simpy.Environment, Message, List[str]], simpy.Event]
OnDeliverHook = Callable[[simpy.Environment, Message], None]
HopObserveHook = Callable[[simpy.Environment, Message, str, str], None]


class NetworkSim:
    """
    NetworkSim delivers messages across a topology with per-link queueing, delay, jitter, and loss.

    Integration points:
    - pre_send_hook: allocate keys, admission control, or any pre-flight logic that may block/drop
    - on_deliver_hook: called after delivery classification (on-time/late) to apply semantics elsewhere
    """

    def __init__(
        self,
        env: simpy.Environment,
        g: nx.Graph,
        links: Dict[Tuple[str, str], LinkModel],
        rng: random.Random,
        *,
        route_policy: str = "shortest",
        k_paths: int = 3,
        pre_send_hook: Optional[PreSendHook] = None,
        on_deliver_hook: Optional[OnDeliverHook] = None,
        on_message_final: Optional[OnDeliverHook] = None,
        on_hop_observe: Optional[HopObserveHook] = None,
        drop_if_miss_deadline: bool = False,
    ):
        self.env = env
        self.g = g
        self.links = links
        self.rng = rng

        self.route_policy = route_policy
        self.k_paths = int(k_paths)
        self.pre_send_hook = pre_send_hook
        self.on_deliver_hook = on_deliver_hook
        self.on_message_final = on_message_final
        self.on_hop_observe = on_hop_observe
        self.drop_if_miss_deadline = drop_if_miss_deadline
        self._edge_hits_total: Counter[Tuple[str, str]] = Counter()
        self._edge_hits_attack: Counter[Tuple[str, str]] = Counter()

        # Routing policy diagnostics (used for sanity checks / experiments).
        # For ECMP specifically, track how often equal-cost alternatives exist.
        self._ecmp_stats: Dict[str, float] = {
            "total": 0.0,
            "had_alternatives": 0.0,
            "alt_paths_sum": 0.0,
            "alt_paths_max": 0.0,
        }

    def now_ms(self) -> int:
        return int(self.env.now * 1000)

    def compute_path(self, src: str, dst: str, msg: Optional[Message] = None) -> List[str]:
        """
        Compute a path from src to dst based on routing policy.

        Policies:
        - "shortest": deterministic shortest path
        - "ecmp": equal-cost multi-path (random among shortest paths)
        - "k_shortest": random selection among k shortest simple paths
        - "k_shortest_weighted": prefer shorter paths (inverse-length weighting)
        - "disjoint": random edge-disjoint path (fallback to shortest)
        - "load_aware": avoid links with backlog using queue-depth weights
        """
        if src == dst:
            return [src]

        if self.route_policy == "shortest":
            if msg is not None:
                msg.payload = msg.payload or {}
                msg.payload["net_route_policy"] = "shortest"
                msg.payload["net_candidate_paths"] = 1
            return nx.shortest_path(self.g, src, dst)

        if self.route_policy == "ecmp":
            # Equal-Cost Multi-Path: diversity without increasing hop count.
            # This matters in QuAM because key usage is per-hop; k-shortest can
            # backfire by selecting longer paths under key starvation.
            paths = list(nx.all_shortest_paths(self.g, src, dst))
            if msg is not None:
                msg.payload = msg.payload or {}
                msg.payload["net_route_policy"] = "ecmp"
                msg.payload["net_candidate_paths"] = int(len(paths))

            # Diagnostics: does ECMP actually have alternatives on this topology?
            self._ecmp_stats["total"] += 1.0
            self._ecmp_stats["alt_paths_sum"] += float(len(paths))
            self._ecmp_stats["alt_paths_max"] = max(self._ecmp_stats["alt_paths_max"], float(len(paths)))
            if len(paths) > 1:
                self._ecmp_stats["had_alternatives"] += 1.0
            if not paths:
                return nx.shortest_path(self.g, src, dst)
            return self.rng.choice(paths)

        if self.route_policy in ("k_shortest", "k_shortest_weighted"):
            paths = list(itertools.islice(nx.shortest_simple_paths(self.g, src, dst), max(1, self.k_paths)))
            if msg is not None:
                msg.payload = msg.payload or {}
                msg.payload["net_route_policy"] = str(self.route_policy)
                msg.payload["net_candidate_paths"] = int(len(paths))
            if len(paths) <= 1:
                return paths[0]
            if self.route_policy == "k_shortest":
                return self.rng.choice(paths)
            weights = [1.0 / max(1, len(p)) for p in paths]
            return self.rng.choices(paths, weights=weights, k=1)[0]

        if self.route_policy == "disjoint":
            try:
                disjoint_paths = list(nx.edge_disjoint_paths(self.g, src, dst))
                if msg is not None:
                    msg.payload = msg.payload or {}
                    msg.payload["net_route_policy"] = "disjoint"
                    msg.payload["net_candidate_paths"] = int(len(disjoint_paths))
                if disjoint_paths:
                    return self.rng.choice(disjoint_paths)
            except Exception:
                pass
            if msg is not None:
                msg.payload = msg.payload or {}
                msg.payload["net_route_policy"] = "disjoint"
                msg.payload["net_candidate_paths"] = int(msg.payload.get("net_candidate_paths", 0) or 0) or 1
            return nx.shortest_path(self.g, src, dst)

        if self.route_policy == "load_aware":
            if msg is not None:
                msg.payload = msg.payload or {}
                msg.payload["net_route_policy"] = "load_aware"
                msg.payload["net_candidate_paths"] = 1
            return self._compute_load_aware_path(src, dst)

        raise ValueError(f"Unknown route_policy: {self.route_policy}")

    def _compute_load_aware_path(self, src: str, dst: str) -> List[str]:
        """
        Compute a path avoiding congested links by inflating edge weights based on
        current link queue depth.
        """
        weighted = nx.Graph()
        weighted.add_nodes_from(self.g.nodes())
        for (u, v) in self.g.edges():
            ek = edge_key(u, v)
            link = self.links.get(ek)
            qd = link.queue_depth() if link is not None else 0
            # Higher weight = less preferred; backlog dominates.
            weighted.add_edge(u, v, weight=1.0 + (float(qd) * 10.0))
        return nx.shortest_path(weighted, src, dst, weight="weight")

    def send(self, msg: Message) -> simpy.Event:
        return self.env.process(self._send_process(msg))

    def _finalize(self, msg: Message) -> None:
        if self.on_message_final is not None:
            self.on_message_final(self.env, msg)

    def _send_process(self, msg: Message):
        # basic validation
        if msg.src not in self.g or msg.dst not in self.g:
            msg.mark_dropped(DeliveryStatus.DROPPED_LOSS, "unknown_node")
            self._finalize(msg)
            return

        # For node-level attacker with forged source: route from the physical
        # injection point, not the forged src field.  The src field stays as
        # the attacker's claimed identity for ACL / gate checks.
        route_origin = msg.src
        inj = getattr(msg, "injection_node", None)
        if inj and inj in self.g and inj != msg.dst:
            route_origin = inj

        try:
            path = self.compute_path(route_origin, msg.dst, msg)
        except nx.NetworkXNoPath:
            msg.mark_dropped(DeliveryStatus.DROPPED_LOSS, "no_path")
            self._finalize(msg)
            return

        # Annotate message with routing outcome for later analysis.
        msg.payload = msg.payload or {}
        msg.payload["net_path_hops"] = int(max(0, len(path) - 1))

        # Track edge usage (attempted hops) for sanity / attack targeting checks.
        for i in range(len(path) - 1):
            ek = edge_key(path[i], path[i + 1])
            self._edge_hits_total[ek] += 1
            if bool(getattr(msg, "is_attack", False)):
                self._edge_hits_attack[ek] += 1

        # Optional trace for debugging a single message end-to-end.
        try:
            trace_id = int(os.getenv("QUAM_TRACE_MSG_ID", "-1"))
        except Exception:
            trace_id = -1
        if trace_id > 0 and int(getattr(msg, "msg_id", -1)) == trace_id:
            edges = [edge_key(path[i], path[i + 1]) for i in range(len(path) - 1)]
            print(f"[MSG {trace_id}] Created: src={msg.src} dst={msg.dst}")
            print(f"[MSG {trace_id}] Path: {path}")
            print(f"[MSG {trace_id}] Edges: {edges}")

        # Pre-flight hook (keys/admission control). The hook may:
        # - yield time (waiting)
        # - mark msg as dropped and return
        if self.pre_send_hook is not None:
            yield self.pre_send_hook(self.env, msg, path)
            if msg.status in (DeliveryStatus.DROPPED_NO_KEYS, DeliveryStatus.DROPPED_LOSS, DeliveryStatus.DROPPED_EXPIRED, DeliveryStatus.DROPPED_BLOCKED):
                self._finalize(msg)
                return

        # ── Quantum protocol overhead: real SimPy delay ──
        # gate_delay_ms  = auth computation (HMAC, tag verify) per hop
        # qp_handshake_ms = protocol handshake (E91 Bell test, KAK 3-pass)
        # These were annotated by quantum.py _pre_send_process
        _gate_ms = float((msg.payload or {}).get("gate_delay_ms", 0))
        _hand_ms = float((msg.payload or {}).get("qp_handshake_ms", 0))
        _q_overhead_ms = _gate_ms + _hand_ms
        if _q_overhead_ms > 0:
            yield self.env.timeout(_q_overhead_ms / 1000.0)

        # ── Latency breakdown tracking ──
        total_propagation_ms = 0.0
        total_queuing_ms = 0.0
        hop_before_ms = self.now_ms()

        # Traverse hop-by-hop
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            ek = edge_key(u, v)
            link = self.links.get(ek, None)
            if link is None:
                msg.mark_dropped(DeliveryStatus.DROPPED_LOSS, "missing_link_model")
                self._finalize(msg)
                return

            hop_start_ms = self.now_ms()
            ok = yield self.env.process(self._tx_over_link(link, msg))
            hop_elapsed_ms = self.now_ms() - hop_start_ms
            if not ok:
                msg.mark_dropped(DeliveryStatus.DROPPED_LOSS, "link_loss")
                self._finalize(msg)
                return

            # Track propagation (base latency + jitter + serialization) and queuing
            prop_ms = float(link.params.base_latency_ms) + float(msg.size_bytes) / max(1.0, link.params.bandwidth_bytes_per_s) * 1000.0
            queue_ms = max(0.0, hop_elapsed_ms - prop_ms)
            total_propagation_ms += prop_ms
            total_queuing_ms += queue_ms

            if self.on_hop_observe is not None:
                self.on_hop_observe(self.env, msg, u, v)

            # Optional hard deadline drop
            if self.drop_if_miss_deadline:
                elapsed = self.now_ms() - msg.created_ms
                if elapsed > msg.deadline_ms:
                    msg.mark_dropped(DeliveryStatus.DROPPED_EXPIRED, "deadline_missed_in_transit")
                    self._finalize(msg)
                    return

        # Delivered: classify by deadline
        msg.mark_delivered(self.now_ms())

        # ── Annotate latency breakdown in payload ──
        msg.payload = msg.payload or {}
        key_wait = float(msg.key_wait_ms or 0)
        gate_delay = float(msg.payload.get("gate_delay_ms", 0))
        protocol_overhead = float(msg.payload.get("qp_handshake_ms", 0))
        classical_latency = total_propagation_ms + total_queuing_ms
        quantum_latency = key_wait + gate_delay + protocol_overhead
        msg.payload["propagation_ms"] = round(total_propagation_ms, 2)
        msg.payload["queuing_ms"] = round(total_queuing_ms, 2)
        msg.payload["protocol_overhead_ms"] = round(protocol_overhead, 2)
        msg.payload["classical_latency_ms"] = round(classical_latency, 2)
        msg.payload["quantum_latency_ms"] = round(quantum_latency, 2)

        # Deliver hook: apply semantics outside network layer
        if self.on_deliver_hook is not None:
            self.on_deliver_hook(self.env, msg)

        self._finalize(msg)

    def _tx_over_link(self, link: LinkModel, msg: Message) -> simpy.Event:
        """
        Simple per-link service model:
        - probabilistic loss
        - resource-based queueing
        - propagation + jitter + serialization delay
        """
        # Loss check (per hop)
        if self.rng.random() < link.params.loss_prob:
            yield self.env.timeout(0)
            return False

        entry_time = link.record_queue_entry(self.env.now)
        with link.resource.request() as req:
            yield req
            link.record_queue_exit(entry_time, self.env.now)

            jitter = self.rng.randint(0, link.params.jitter_ms)
            prop_s = (link.params.base_latency_ms + jitter) / 1000.0
            tx_s = float(msg.size_bytes) / max(1.0, link.params.bandwidth_bytes_per_s)

            yield self.env.timeout(prop_s + tx_s)

        return True


# -------------------------
# Reporting helpers
# -------------------------

def summarize_link_congestion(links: Dict[Tuple[str, str], LinkModel]) -> Dict[str, float]:
    """Aggregate queue/congestion statistics across all links."""
    if not links:
        return {}

    max_depths = [int(getattr(l, "max_queue_depth", 0)) for l in links.values()]
    avg_queue_times = [float(l.avg_queue_time_ms()) for l in links.values()]
    # Utilization at the *end* of the run is still useful as a sanity signal, but
    # the more meaningful metric for experiments is whether a link ever built
    # backlog beyond its capacity.
    utilizations = [float(l.utilization()) for l in links.values()]
    congested = sum(
        1
        for l in links.values()
        if int(getattr(l, "max_queue_depth", 0)) > int(getattr(l.params, "queue_capacity", 0))
    )

    return {
        "link_max_queue_depth_max": float(max(max_depths)) if max_depths else 0.0,
        "link_max_queue_depth_mean": float(sum(max_depths) / len(max_depths)) if max_depths else 0.0,
        "link_avg_queue_time_ms_mean": float(sum(avg_queue_times) / len(avg_queue_times)) if avg_queue_times else 0.0,
        "link_utilization_max": float(max(utilizations)) if utilizations else 0.0,
        "links_congested_count": float(congested),
    }
