"""
topology/graph.py
=================
Venue camera topology graph for SAFEPASS (Track P3).

A directed NetworkX graph G = (V, E) where every node is a CCTV camera and every
edge is a physically walkable corridor between two camera fields of view.

Node attributes:
    camera_id, camera_name, zone_name, coordinates (x, y) in metres

Edge attributes (Ci -> Cj):
    distance_meters       walking distance along the corridor
    min_transit_seconds   d / v_max   (v_max = 4.5 m/s, a running adult)
    max_transit_seconds   d / v_min   (v_min = 0.9 m/s, a slow walker / child)
    prior_probability     manual cold-start prior P(Cj | Ci); outgoing priors sum to 1
    learned_probability   (optional) P(Cj | Ci) estimated by topology.learner
    learned_travel_time   (optional) log-normal travel-time model fitted by topology.learner

`coordinates` and `distance_meters` live only on the graph; they are deliberately
not part of contract/schema.py. `to_camera_nodes()` exports the frozen CameraNode view.

Usage:
    from topology.graph import VenueTopologyGraph
    g = VenueTopologyGraph.default_venue()
    g.get_transit_window("CAM-01", "CAM-04")   # path-based, via CAM-03
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import networkx as nx

from contract.schema import CameraNode

V_MAX_MPS = 4.5
V_MIN_MPS = 0.9
TYPICAL_WALK_MPS = 1.3
DEFAULT_TORTUOSITY = 1.15

_STD_NORMAL = NormalDist()


# --------------------------------------------------------------------------- #
# Travel-time distribution
# --------------------------------------------------------------------------- #
@dataclass
class LogNormalTravelTime:
    """
    Log-normal travel-time model: ln(dt) ~ Normal(mu, sigma^2).

    `n` is the (effective) number of observations behind the fit and `source`
    records provenance: "learned-edge", "learned-pair", "composed" or "prior".
    """

    mu: float
    sigma: float
    n: float = 0.0
    source: str = "learned-edge"

    def __post_init__(self) -> None:
        self.sigma = max(float(self.sigma), 1e-3)
        self.mu = float(self.mu)

    @property
    def median(self) -> float:
        return math.exp(self.mu)

    @property
    def mode(self) -> float:
        return math.exp(self.mu - self.sigma ** 2)

    @property
    def mean(self) -> float:
        return math.exp(self.mu + 0.5 * self.sigma ** 2)

    @property
    def variance(self) -> float:
        s2 = self.sigma ** 2
        return (math.exp(s2) - 1.0) * math.exp(2.0 * self.mu + s2)

    def pdf(self, t: float) -> float:
        if t <= 0:
            return 0.0
        z = (math.log(t) - self.mu) / self.sigma
        return math.exp(-0.5 * z * z) / (t * self.sigma * math.sqrt(2.0 * math.pi))

    def log_pdf(self, t: float) -> float:
        if t <= 0:
            return -math.inf
        z = (math.log(t) - self.mu) / self.sigma
        return -0.5 * z * z - math.log(t * self.sigma * math.sqrt(2.0 * math.pi))

    def cdf(self, t: float) -> float:
        if t <= 0:
            return 0.0
        return _STD_NORMAL.cdf((math.log(t) - self.mu) / self.sigma)

    def quantile(self, q: float) -> float:
        q = min(max(q, 1e-9), 1.0 - 1e-9)
        return math.exp(self.mu + self.sigma * _STD_NORMAL.inv_cdf(q))

    def score(self, t: float) -> float:
        """Likelihood of `t` normalised by the likelihood at the mode, in [0, 1]."""
        if t <= 0:
            return 0.0
        return float(min(1.0, math.exp(self.log_pdf(t) - self.log_pdf(self.mode))))

    def interval(self, coverage: float = 0.9) -> Tuple[float, float]:
        tail = (1.0 - coverage) / 2.0
        return self.quantile(tail), self.quantile(1.0 - tail)

    @classmethod
    def from_moments(cls, mean: float, variance: float, n: float = 0.0, source: str = "composed") -> "LogNormalTravelTime":
        mean = max(mean, 1e-6)
        s2 = math.log(1.0 + max(variance, 1e-12) / (mean * mean))
        return cls(mu=math.log(mean) - 0.5 * s2, sigma=math.sqrt(s2), n=n, source=source)

    @classmethod
    def sum_of(cls, parts: Iterable["LogNormalTravelTime"]) -> "LogNormalTravelTime":
        """Fenton-Wilkinson approximation of a sum of independent log-normals."""
        parts = list(parts)
        if len(parts) == 1:
            return parts[0]
        mean = sum(p.mean for p in parts)
        var = sum(p.variance for p in parts)
        return cls.from_moments(mean, var, n=min(p.n for p in parts), source="composed")

    @classmethod
    def physical_prior(cls, distance_m: float, typical_speed: float = TYPICAL_WALK_MPS, sigma: float = 0.35) -> "LogNormalTravelTime":
        """Weak prior centred on walking the corridor at a typical adult pace."""
        return cls(mu=math.log(max(distance_m, 1e-3) / typical_speed), sigma=sigma, n=0.0, source="prior")

    def to_dict(self) -> Dict[str, Any]:
        return {"dist": "lognormal", "mu": self.mu, "sigma": self.sigma, "n": self.n, "source": self.source}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LogNormalTravelTime":
        return cls(mu=d["mu"], sigma=d["sigma"], n=d.get("n", 0.0), source=d.get("source", "learned-edge"))


def _edge_key(a: str, b: str) -> str:
    return f"{a}->{b}"


def _split_edge_key(key: str) -> Tuple[str, str]:
    a, b = key.split("->")
    return a, b


# --------------------------------------------------------------------------- #
# Venue graph
# --------------------------------------------------------------------------- #
class VenueTopologyGraph:
    """Directed camera topology graph with physical transit bounds and transition priors."""

    def __init__(
        self,
        venue_name: str = "SAFEPASS Venue",
        v_max: float = V_MAX_MPS,
        v_min: float = V_MIN_MPS,
    ):
        if not (0 < v_min < v_max):
            raise ValueError("Require 0 < v_min < v_max")
        self.venue_name = venue_name
        self.v_max = float(v_max)
        self.v_min = float(v_min)
        self.G = nx.DiGraph()
        self.use_learned = True
        self.learned_pairs: Dict[Tuple[str, str], LogNormalTravelTime] = {}
        self.learned_model_meta: Optional[Dict[str, Any]] = None
        self._window_cache: Dict[Tuple[str, str, bool, int], Optional[Tuple[float, float]]] = {}

    # ------------------------------------------------------------------ build
    def add_camera(
        self,
        camera_id: str,
        camera_name: str,
        zone_name: str,
        coordinates: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.G.add_node(
            camera_id,
            camera_id=camera_id,
            camera_name=camera_name,
            zone_name=zone_name,
            coordinates=(float(coordinates[0]), float(coordinates[1])),
        )
        self._window_cache.clear()

    def add_corridor(
        self,
        cam_a: str,
        cam_b: str,
        distance_meters: Optional[float] = None,
        bidirectional: bool = True,
        tortuosity: float = DEFAULT_TORTUOSITY,
    ) -> None:
        """Adds a walkable corridor. Distance defaults to straight-line distance x tortuosity."""
        for cam in (cam_a, cam_b):
            if cam not in self.G:
                raise KeyError(f"Unknown camera '{cam}'; add_camera() first")
        if cam_a == cam_b:
            raise ValueError("Self-loop corridors are not allowed")
        if distance_meters is None:
            (xa, ya), (xb, yb) = self.G.nodes[cam_a]["coordinates"], self.G.nodes[cam_b]["coordinates"]
            distance_meters = round(math.hypot(xb - xa, yb - ya) * tortuosity, 1)
        if distance_meters <= 0:
            raise ValueError("distance_meters must be positive")

        pairs = [(cam_a, cam_b), (cam_b, cam_a)] if bidirectional else [(cam_a, cam_b)]
        for u, v in pairs:
            self.G.add_edge(
                u,
                v,
                distance_meters=float(distance_meters),
                min_transit_seconds=distance_meters / self.v_max,
                max_transit_seconds=distance_meters / self.v_min,
                prior_probability=None,
            )
        self._window_cache.clear()

    def set_priors(self, camera_id: str, weights: Dict[str, float]) -> None:
        """Sets manual prior P(Cj | Ci) for the outgoing edges of `camera_id` (normalised to 1)."""
        succ = set(self.G.successors(camera_id))
        unknown = set(weights) - succ
        if unknown:
            raise KeyError(f"{camera_id} has no corridor to {sorted(unknown)}")
        total = float(sum(weights.get(v, 0.0) for v in succ))
        if total <= 0:
            raise ValueError("Prior weights must have positive sum")
        for v in succ:
            self.G.edges[camera_id, v]["prior_probability"] = weights.get(v, 0.0) / total

    def normalise_priors(self) -> None:
        """Ensures every node's outgoing priors sum to 1 (missing priors become uniform)."""
        for u in self.G.nodes:
            succ = list(self.G.successors(u))
            if not succ:
                continue
            vals = [self.G.edges[u, v].get("prior_probability") for v in succ]
            if any(p is None for p in vals) or sum(vals) <= 0:
                for v in succ:
                    self.G.edges[u, v]["prior_probability"] = 1.0 / len(succ)
            else:
                total = sum(vals)
                for v, p in zip(succ, vals):
                    self.G.edges[u, v]["prior_probability"] = p / total

    def validate(self, tol: float = 1e-6) -> None:
        for u in self.G.nodes:
            succ = list(self.G.successors(u))
            if not succ:
                continue
            s = sum(self.G.edges[u, v]["prior_probability"] for v in succ)
            if abs(s - 1.0) > tol:
                raise ValueError(f"Outgoing priors of {u} sum to {s:.4f}, expected 1.0")

    # ---------------------------------------------------------------- queries
    @property
    def cameras(self) -> List[str]:
        return sorted(self.G.nodes)

    def get_camera(self, camera_id: str) -> Dict[str, Any]:
        return dict(self.G.nodes[camera_id])

    def camera_name(self, camera_id: str) -> str:
        if camera_id in self.G:
            return self.G.nodes[camera_id]["camera_name"]
        return camera_id

    def get_adjacent_cameras(self, camera_id: str) -> List[str]:
        """Cameras directly reachable from `camera_id` (outgoing corridors), sorted by id."""
        if camera_id not in self.G:
            raise KeyError(f"Unknown camera '{camera_id}'")
        return sorted(self.G.successors(camera_id))

    def are_adjacent(self, cam_a: str, cam_b: str) -> bool:
        return self.G.has_edge(cam_a, cam_b)

    def has_path(self, cam_a: str, cam_b: str) -> bool:
        if cam_a not in self.G or cam_b not in self.G:
            return False
        return nx.has_path(self.G, cam_a, cam_b)

    def get_shortest_path(self, cam_a: str, cam_b: str, weight: Optional[str] = "distance_meters") -> Optional[List[str]]:
        """Shortest walkable path (by distance by default). Returns None if unreachable."""
        if cam_a not in self.G or cam_b not in self.G:
            return None
        try:
            return nx.shortest_path(self.G, cam_a, cam_b, weight=weight)
        except nx.NetworkXNoPath:
            return None

    def get_most_likely_path(self, cam_a: str, cam_b: str, max_extra_hops: int = 1,
                             source: str = "auto") -> Optional[List[str]]:
        """
        Most probable walkable route for an unseen (blind-spot) traversal: among simple
        paths with at most (min hops + max_extra_hops) hops, maximise the product of
        transition probabilities P(Cj|Ci). Ties break toward the shorter distance.
        """
        hops = self.hop_distance(cam_a, cam_b)
        if hops is None:
            return None
        if hops <= 1:
            return [cam_a, cam_b] if hops == 1 else [cam_a]
        best, best_key = None, None
        for p in nx.all_simple_paths(self.G, cam_a, cam_b, cutoff=hops + max_extra_hops):
            logp = sum(math.log(max(self.get_transition_probability(u, v, source), 1e-12))
                       for u, v in zip(p[:-1], p[1:]))
            key = (logp, -self.get_path_distance(p))
            if best_key is None or key > best_key:
                best, best_key = p, key
        return best

    def hop_distance(self, cam_a: str, cam_b: str) -> Optional[int]:
        path = self.get_shortest_path(cam_a, cam_b, weight=None)
        return None if path is None else len(path) - 1

    def get_path_distance(self, path: List[str]) -> float:
        return float(sum(self.G.edges[u, v]["distance_meters"] for u, v in zip(path[:-1], path[1:])))

    def get_edge_transit(self, cam_a: str, cam_b: str) -> Tuple[float, float]:
        e = self.G.edges[cam_a, cam_b]
        return e["min_transit_seconds"], e["max_transit_seconds"]

    def get_path_transit_window(self, path: List[str]) -> Tuple[float, float]:
        """Sum of per-edge [min, max] transit along an explicit path."""
        lo = hi = 0.0
        for u, v in zip(path[:-1], path[1:]):
            a, b = self.get_edge_transit(u, v)
            lo += a
            hi += b
        return lo, hi

    def get_transit_window(
        self,
        cam_a: str,
        cam_b: str,
        include_alternatives: bool = False,
        max_extra_hops: int = 1,
    ) -> Optional[Tuple[float, float]]:
        """
        Path-based physical transit window [dt_min, dt_max] in seconds from cam_a to cam_b.

        Multi-hop routes sum per-edge bounds along the shortest (by distance) path.
        With include_alternatives=True, every simple path with at most
        (shortest hop count + max_extra_hops) hops is considered, taking the min of
        mins and the max of maxes. Returns None when cam_b is unreachable and
        (0, 0) for cam_a == cam_b.
        """
        key = (cam_a, cam_b, include_alternatives, max_extra_hops)
        if key in self._window_cache:
            return self._window_cache[key]

        if cam_a == cam_b:
            result: Optional[Tuple[float, float]] = (0.0, 0.0) if cam_a in self.G else None
        else:
            path = self.get_shortest_path(cam_a, cam_b)
            if path is None:
                result = None
            else:
                lo, hi = self.get_path_transit_window(path)
                if include_alternatives:
                    hops = self.hop_distance(cam_a, cam_b) or (len(path) - 1)
                    for alt in nx.all_simple_paths(self.G, cam_a, cam_b, cutoff=hops + max_extra_hops):
                        a, b = self.get_path_transit_window(alt)
                        lo, hi = min(lo, a), max(hi, b)
                result = (lo, hi)
        self._window_cache[key] = result
        return result

    def get_transit_bounds_int(self, cam_a: str, cam_b: str, **kwargs: Any) -> Optional[Tuple[int, int]]:
        w = self.get_transit_window(cam_a, cam_b, **kwargs)
        if w is None:
            return None
        return int(math.floor(w[0])), int(math.ceil(w[1]))

    # ------------------------------------------------- transition probabilities
    def get_transition_probability(self, cam_a: str, cam_b: str, source: str = "auto") -> float:
        """P(cam_b | cam_a). source: 'auto' (learned if loaded, else prior), 'learned' or 'prior'."""
        if not self.G.has_edge(cam_a, cam_b):
            return 0.0
        e = self.G.edges[cam_a, cam_b]
        learned = e.get("learned_probability")
        if source == "prior" or (source == "auto" and (learned is None or not self.use_learned)):
            return float(e["prior_probability"])
        if learned is None:
            raise ValueError(f"No learned probability for {cam_a}->{cam_b}")
        return float(learned)

    def get_outgoing_distribution(self, camera_id: str, source: str = "auto") -> Dict[str, float]:
        succ = self.get_adjacent_cameras(camera_id)
        probs = {v: self.get_transition_probability(camera_id, v, source) for v in succ}
        total = sum(probs.values())
        if total <= 0:
            return {v: 1.0 / len(succ) for v in succ} if succ else {}
        return {v: p / total for v, p in probs.items()}

    @property
    def has_learned_model(self) -> bool:
        return self.learned_model_meta is not None

    # ----------------------------------------------------- travel-time models
    def get_travel_time_distribution(self, cam_a: str, cam_b: str) -> Optional[LogNormalTravelTime]:
        """
        Learned travel-time distribution from cam_a to cam_b, or None (use manual fallback).

        Preference: a directly learned pair model (also covers non-adjacent blind-spot
        pairs), else a Fenton-Wilkinson composition of learned edge models along the
        shortest path when every edge on it has a learned model.
        """
        if not self.use_learned or cam_a == cam_b:
            return None
        pair = self.learned_pairs.get((cam_a, cam_b))
        if pair is not None:
            return pair
        path = self.get_shortest_path(cam_a, cam_b)
        if path is None:
            return None
        parts = []
        for u, v in zip(path[:-1], path[1:]):
            d = self.G.edges[u, v].get("learned_travel_time")
            if d is None:
                return None
            parts.append(d)
        return LogNormalTravelTime.sum_of(parts)

    def get_expected_transit_window(self, cam_a: str, cam_b: str, coverage: float = 0.9) -> Optional[Tuple[float, float]]:
        """
        Expected arrival window: central `coverage` interval of the learned travel-time
        distribution (lower end clipped to the physical minimum), falling back to the
        physical path-based window when nothing has been learned.
        """
        phys = self.get_transit_window(cam_a, cam_b)
        if phys is None:
            return None
        dist = self.get_travel_time_distribution(cam_a, cam_b)
        if dist is None:
            return phys
        lo, hi = dist.interval(coverage)
        lo = max(lo, phys[0])
        return lo, max(hi, lo)

    # ---------------------------------------------------------- learned model
    def load_learned_model(self, source: Union[str, Dict[str, Any]]) -> None:
        """Loads a model exported by TransitionLearner.to_json() (path, JSON string or dict)."""
        model = _load_json_source(source)
        self.clear_learned_model()
        for key, info in model.get("edges", {}).items():
            a, b = _split_edge_key(key)
            if not self.G.has_edge(a, b):
                continue
            e = self.G.edges[a, b]
            if info.get("probability") is not None:
                e["learned_probability"] = float(info["probability"])
            if info.get("travel_time"):
                e["learned_travel_time"] = LogNormalTravelTime.from_dict(info["travel_time"])
        for key, info in model.get("pairs", {}).items():
            a, b = _split_edge_key(key)
            if a in self.G and b in self.G and info.get("travel_time"):
                self.learned_pairs[(a, b)] = LogNormalTravelTime.from_dict(info["travel_time"])

        for u in self.G.nodes:
            succ = list(self.G.successors(u))
            learned = [self.G.edges[u, v].get("learned_probability") for v in succ]
            if succ and all(p is not None for p in learned) and sum(learned) > 0:
                total = sum(learned)
                for v, p in zip(succ, learned):
                    self.G.edges[u, v]["learned_probability"] = p / total
            else:
                for v in succ:
                    self.G.edges[u, v].pop("learned_probability", None)

        self.learned_model_meta = {k: v for k, v in model.items() if k not in ("edges", "pairs")}
        self.use_learned = True

    def clear_learned_model(self) -> None:
        for _, _, e in self.G.edges(data=True):
            e.pop("learned_probability", None)
            e.pop("learned_travel_time", None)
        self.learned_pairs = {}
        self.learned_model_meta = None

    # ------------------------------------------------------- contract export
    def get_camera_node(self, camera_id: str) -> CameraNode:
        n = self.G.nodes[camera_id]
        succ = self.get_adjacent_cameras(camera_id)
        return CameraNode(
            camera_id=camera_id,
            camera_name=n["camera_name"],
            zone_name=n["zone_name"],
            connected_cameras=succ,
            transit_bounds={v: self.get_transit_bounds_int(camera_id, v) for v in succ},
        )

    def to_camera_nodes(self) -> List[CameraNode]:
        return [self.get_camera_node(c) for c in self.cameras]

    # --------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        nodes = []
        for c in self.cameras:
            n = self.G.nodes[c]
            nodes.append({
                "camera_id": c,
                "camera_name": n["camera_name"],
                "zone_name": n["zone_name"],
                "coordinates": list(n["coordinates"]),
            })
        edges = []
        for u, v, e in sorted(self.G.edges(data=True)):
            rec = {
                "source": u,
                "target": v,
                "distance_meters": e["distance_meters"],
                "min_transit_seconds": round(e["min_transit_seconds"], 3),
                "max_transit_seconds": round(e["max_transit_seconds"], 3),
                "prior_probability": e["prior_probability"],
            }
            if e.get("learned_probability") is not None:
                rec["learned_probability"] = e["learned_probability"]
            if e.get("learned_travel_time") is not None:
                rec["learned_travel_time"] = e["learned_travel_time"].to_dict()
            edges.append(rec)
        out: Dict[str, Any] = {
            "venue_name": self.venue_name,
            "v_max_mps": self.v_max,
            "v_min_mps": self.v_min,
            "nodes": nodes,
            "edges": edges,
        }
        if self.learned_pairs:
            out["learned_pairs"] = {_edge_key(a, b): d.to_dict() for (a, b), d in self.learned_pairs.items()}
        if self.learned_model_meta is not None:
            out["learned_model_meta"] = self.learned_model_meta
        return out

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VenueTopologyGraph":
        g = cls(
            venue_name=data.get("venue_name", "SAFEPASS Venue"),
            v_max=data.get("v_max_mps", V_MAX_MPS),
            v_min=data.get("v_min_mps", V_MIN_MPS),
        )
        for n in data["nodes"]:
            g.add_camera(n["camera_id"], n["camera_name"], n["zone_name"], tuple(n.get("coordinates", (0.0, 0.0))))
        for e in data["edges"]:
            g.add_corridor(e["source"], e["target"], e["distance_meters"], bidirectional=False)
            attrs = g.G.edges[e["source"], e["target"]]
            attrs["prior_probability"] = e.get("prior_probability")
            if e.get("learned_probability") is not None:
                attrs["learned_probability"] = float(e["learned_probability"])
            if e.get("learned_travel_time"):
                attrs["learned_travel_time"] = LogNormalTravelTime.from_dict(e["learned_travel_time"])
        for key, d in data.get("learned_pairs", {}).items():
            g.learned_pairs[_split_edge_key(key)] = LogNormalTravelTime.from_dict(d)
        g.learned_model_meta = data.get("learned_model_meta")
        g.normalise_priors()
        return g

    @classmethod
    def from_json(cls, source: str) -> "VenueTopologyGraph":
        """Builds a graph from a JSON file path or a JSON string."""
        return cls.from_dict(_load_json_source(source))

    def copy(self) -> "VenueTopologyGraph":
        return VenueTopologyGraph.from_dict(self.to_dict())

    # -------------------------------------------------------- default venue
    @classmethod
    def default_venue(cls) -> "VenueTopologyGraph":
        """
        8-camera campus venue. Coordinates are metres on a site plan; corridor
        distances are straight-line x 1.15 tortuosity.

        Blind-spot gaps (no direct corridor, must traverse an intermediate camera):
            CAM-01 Main Entrance  <-> CAM-04 East Escalator   (via CAM-03 Central Atrium)
            CAM-08 West Parking   <-> CAM-03 Central Atrium   (via CAM-01)
            CAM-06 South Exit     <-> CAM-07 Food Court       (via CAM-04)
        """
        g = cls(venue_name="SAFEPASS Default Campus Venue")
        cams = [
            ("CAM-01", "Main Entrance Gate", "Entrance Plaza", (0.0, 0.0)),
            ("CAM-02", "North Corridor", "North Wing", (0.0, 70.0)),
            ("CAM-03", "Central Atrium", "Central Atrium", (60.0, 40.0)),
            ("CAM-04", "East Escalator Terminal", "East Wing", (130.0, 40.0)),
            ("CAM-05", "Library Hallway", "Library Block", (60.0, 110.0)),
            ("CAM-06", "South Exit Turnstiles", "South Gate", (70.0, -40.0)),
            ("CAM-07", "Food Court Plaza", "Food Court", (130.0, 110.0)),
            ("CAM-08", "West Parking Stairwell", "West Parking", (-60.0, 10.0)),
        ]
        for cid, name, zone, xy in cams:
            g.add_camera(cid, name, zone, xy)

        corridors = [
            ("CAM-01", "CAM-02"), ("CAM-01", "CAM-03"), ("CAM-01", "CAM-06"), ("CAM-01", "CAM-08"),
            ("CAM-02", "CAM-03"), ("CAM-02", "CAM-05"),
            ("CAM-03", "CAM-04"), ("CAM-03", "CAM-05"), ("CAM-03", "CAM-06"),
            ("CAM-04", "CAM-06"), ("CAM-04", "CAM-07"),
            ("CAM-05", "CAM-07"),
        ]
        for a, b in corridors:
            g.add_corridor(a, b)

        priors = {
            "CAM-01": {"CAM-02": 0.25, "CAM-03": 0.45, "CAM-06": 0.10, "CAM-08": 0.20},
            "CAM-02": {"CAM-01": 0.30, "CAM-03": 0.30, "CAM-05": 0.40},
            "CAM-03": {"CAM-01": 0.20, "CAM-02": 0.15, "CAM-04": 0.30, "CAM-05": 0.20, "CAM-06": 0.15},
            "CAM-04": {"CAM-03": 0.40, "CAM-06": 0.20, "CAM-07": 0.40},
            "CAM-05": {"CAM-02": 0.35, "CAM-03": 0.35, "CAM-07": 0.30},
            "CAM-06": {"CAM-01": 0.40, "CAM-03": 0.40, "CAM-04": 0.20},
            "CAM-07": {"CAM-04": 0.55, "CAM-05": 0.45},
            "CAM-08": {"CAM-01": 1.00},
        }
        for cam, w in priors.items():
            g.set_priors(cam, w)
        g.validate()
        return g

    def __repr__(self) -> str:
        return (f"VenueTopologyGraph('{self.venue_name}', cameras={self.G.number_of_nodes()}, "
                f"edges={self.G.number_of_edges()}, learned={self.has_learned_model})")


def to_seconds(ts: Any) -> float:
    """Float seconds from a number, datetime or ISO 8601 string (only differences are used)."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, datetime):
        return ts.timestamp()
    return datetime.fromisoformat(str(ts)).timestamp()


def _load_json_source(source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(source, dict):
        return source
    if isinstance(source, str) and os.path.exists(source):
        with open(source, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(source)
