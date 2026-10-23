"""
topology/learner.py
===================
Learned camera-transition model for SAFEPASS (Track P3).

Instead of trusting hand-set priors, the venue topology is estimated from
sighting logs (simulator output now, real P1/P2 re-ID output later):

Transition probabilities (Dirichlet / Laplace-smoothed counts, cold-started from priors):
    P(Cj | Ci) = (n_ij + alpha + kappa * pi_ij) / (n_i + alpha * K_i + kappa)
        n_ij   observed transitions Ci -> Cj
        pi_ij  manual prior from VenueTopologyGraph
        kappa  prior strength in pseudo-transitions (default 10)
        alpha  Laplace pseudo-count per edge (default 1); K_i = out-degree of Ci
    With no data the estimate is the prior blended with uniform; with alpha = 0 it
    equals the manual prior exactly.

Travel time dt (last sighting at Ci -> first sighting at Cj) per edge:
    ln(dt) ~ Normal(mu, sigma^2), shrunk toward a weak physical prior
    (median d / 1.3 m/s, sigma 0.35) with `time_prior_strength` pseudo-samples.
    A histogram of dt is kept for plotting/diagnostics.

Robustness to false re-ID matches: each person's sightings are first reduced to
the topology-consistent chain (dynamic programme over physical transit bounds
only, so the cleaning does not depend on the priors being learned); any
remaining transition faster than the physical minimum is rejected.

Blind-spot transitions (consecutive sightings at non-adjacent cameras) are
modelled as pair-level travel-time distributions; optionally their counts can
be attributed along the shortest path (`multihop_weight`, off by default).

Usage:
    learner = TransitionLearner(graph).fit(result.sighting_log())
    learner.to_json("eval/p3_learned_model.json")
    graph.load_learned_model(learner.export())
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from topology.graph import LogNormalTravelTime, VenueTopologyGraph, _edge_key, to_seconds

SightingLike = Union[Any, Dict[str, Any], Tuple[str, str, float]]


@dataclass
class _Obs:
    person_id: str
    camera_id: str
    t: float
    confidence: float = 1.0


@dataclass
class _VisitObs:
    camera_id: str
    first: float
    last: float


def _normalise_sighting(s: SightingLike) -> _Obs:
    """Accepts simulator Sightings, contract Candidates, dicts or (person, camera, t) tuples."""
    if isinstance(s, tuple):
        conf = float(s[3]) if len(s) > 3 else 1.0
        return _Obs(str(s[0]), str(s[1]), to_seconds(s[2]), conf)
    if isinstance(s, dict):
        pid = s.get("person_id") or s.get("case_id")
        conf = s.get("confidence", s.get("fused_score", 1.0))
        return _Obs(str(pid), str(s["camera_id"]), to_seconds(s["timestamp"]), float(conf))
    pid = getattr(s, "person_id", None) or getattr(s, "case_id")
    conf = getattr(s, "confidence", None)
    if conf is None:
        conf = getattr(s, "fused_score", 1.0)
    return _Obs(str(pid), str(s.camera_id), to_seconds(s.timestamp), float(conf))


class TransitionLearner:
    """Estimates P(Cj | Ci) and travel-time distributions from sighting logs."""

    def __init__(
        self,
        graph: VenueTopologyGraph,
        alpha: float = 1.0,
        prior_strength: float = 10.0,
        time_prior_strength: float = 3.0,
        min_time_samples: int = 3,
        same_visit_gap_seconds: float = 20.0,
        reject_teleports: bool = True,
        teleport_tolerance_seconds: float = 0.0,
        multihop_weight: float = 0.0,
        track_cleaning: str = "chain",
        histogram_bins: int = 20,
    ):
        """
        track_cleaning: how each person's raw sightings are cleaned before counting.
            "chain"  - keep the topology-consistent DP chain (physical bounds only,
                       not the priors being learned), discarding look-alike detours
            "filter" - only drop visits that teleport from the last accepted visit
            "none"   - use the raw sequence
        multihop_weight: count credited to each edge on the shortest path for a
            blind-spot (non-adjacent) transition. 0 by default: the true route through
            the blind spot is unknown, and shortest-path attribution biases estimates.
        """
        if track_cleaning not in ("chain", "filter", "none"):
            raise ValueError("track_cleaning must be 'chain', 'filter' or 'none'")
        self.graph = graph
        self.alpha = float(alpha)
        self.prior_strength = float(prior_strength)
        self.time_prior_strength = float(time_prior_strength)
        self.min_time_samples = int(min_time_samples)
        self.same_visit_gap_seconds = float(same_visit_gap_seconds)
        self.reject_teleports = reject_teleports
        self.teleport_tolerance_seconds = float(teleport_tolerance_seconds)
        self.multihop_weight = float(multihop_weight)
        self.track_cleaning = track_cleaning
        self.histogram_bins = int(histogram_bins)
        self._physical_pruner = None
        self.reset()

    def reset(self) -> "TransitionLearner":
        self.counts: Dict[Tuple[str, str], float] = defaultdict(float)
        self.edge_dt: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.pair_dt: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.stats = {"sightings": 0, "visits": 0, "direct_transitions": 0,
                      "multihop_transitions": 0, "rejected_teleports": 0,
                      "rejected_unreachable": 0, "ambiguous_returns": 0,
                      "discarded_sightings": 0}
        return self

    def _clean_chain(self, obs: List[_Obs]) -> List[_Obs]:
        """Keeps the physically consistent chain of sightings (see trajectory.select_consistent_chain)."""
        from topology.pruner import SpatiotemporalPruner
        from topology.trajectory import select_consistent_chain

        if self._physical_pruner is None:
            self._physical_pruner = SpatiotemporalPruner(self.graph, use_learned=False)
        idx = select_consistent_chain([o.camera_id for o in obs], [o.t for o in obs],
                                      [o.confidence for o in obs], self._physical_pruner)
        self.stats["discarded_sightings"] += len(obs) - len(idx)
        return [obs[i] for i in idx]

    # ------------------------------------------------------------ ingestion
    def _visits(self, obs: List[_Obs]) -> List[_VisitObs]:
        """Collapses one person's time-ordered sightings into camera visits."""
        visits: List[_VisitObs] = []
        for o in obs:
            if visits and visits[-1].camera_id == o.camera_id and o.t - visits[-1].last <= self.same_visit_gap_seconds:
                visits[-1].last = o.t
            else:
                visits.append(_VisitObs(o.camera_id, o.t, o.t))
        return visits

    def update(self, sightings: Iterable[SightingLike]) -> "TransitionLearner":
        """Incrementally adds a batch of sightings (grouped per person internally)."""
        per_person: Dict[str, List[_Obs]] = defaultdict(list)
        for s in sightings:
            o = _normalise_sighting(s)
            if o.camera_id in self.graph.G:
                per_person[o.person_id].append(o)
                self.stats["sightings"] += 1
        for pid, obs in per_person.items():
            obs.sort(key=lambda o: o.t)
            if self.track_cleaning == "chain":
                obs = self._clean_chain(obs)
            visits = self._visits(obs)
            if self.track_cleaning != "none":
                visits = self._filter_outliers(visits)
            self.stats["visits"] += len(visits)
            for a, b in zip(visits[:-1], visits[1:]):
                self._add_transition(a, b)
        return self

    def _is_teleport(self, a: _VisitObs, b: _VisitObs) -> bool:
        if a.camera_id == b.camera_id:
            return False
        window = self.graph.get_transit_window(a.camera_id, b.camera_id)
        return window is None or (b.first - a.last) < window[0] - self.teleport_tolerance_seconds

    def _filter_outliers(self, visits: List[_VisitObs]) -> List[_VisitObs]:
        """
        Drops visits that are physically unreachable from the last accepted visit
        (typically false re-ID matches), then re-merges same-camera visits that the
        outlier had split apart.
        """
        if not self.reject_teleports or len(visits) < 2:
            return visits
        kept = [visits[0]]
        for v in visits[1:]:
            if self._is_teleport(kept[-1], v):
                self.stats["rejected_teleports"] += 1
                continue
            last = kept[-1]
            if last.camera_id == v.camera_id and v.first - last.last <= self.same_visit_gap_seconds:
                last.last = max(last.last, v.last)
            else:
                kept.append(v)
        return kept

    def fit(self, sightings: Iterable[SightingLike]) -> "TransitionLearner":
        return self.reset().update(sightings)

    def _add_transition(self, a: _VisitObs, b: _VisitObs) -> None:
        dt = b.first - a.last
        if a.camera_id == b.camera_id:
            self.stats["ambiguous_returns"] += 1  # left and came back unseen: route unknown
            return
        window = self.graph.get_transit_window(a.camera_id, b.camera_id)
        if window is None:
            self.stats["rejected_unreachable"] += 1
            return
        if self.reject_teleports and dt < window[0] - self.teleport_tolerance_seconds:
            self.stats["rejected_teleports"] += 1
            return
        dt = max(dt, 1e-3)
        if self.graph.are_adjacent(a.camera_id, b.camera_id):
            self.counts[(a.camera_id, b.camera_id)] += 1.0
            self.edge_dt[(a.camera_id, b.camera_id)].append(dt)
            self.stats["direct_transitions"] += 1
        else:
            self.pair_dt[(a.camera_id, b.camera_id)].append(dt)
            self.stats["multihop_transitions"] += 1
            if self.multihop_weight > 0:
                path = self.graph.get_shortest_path(a.camera_id, b.camera_id)
                for u, v in zip(path[:-1], path[1:]):
                    self.counts[(u, v)] += self.multihop_weight

    # ------------------------------------------------------------ estimates
    def transition_probabilities(self, camera_id: str) -> Dict[str, float]:
        succ = self.graph.get_adjacent_cameras(camera_id)
        if not succ:
            return {}
        k = len(succ)
        n_i = sum(self.counts.get((camera_id, v), 0.0) for v in succ)
        denom = n_i + self.alpha * k + self.prior_strength
        out = {}
        for v in succ:
            prior = self.graph.get_transition_probability(camera_id, v, source="prior")
            out[v] = (self.counts.get((camera_id, v), 0.0) + self.alpha + self.prior_strength * prior) / denom
        total = sum(out.values())
        return {v: p / total for v, p in out.items()}

    def _fit_lognormal(self, samples: Sequence[float], prior: Optional[LogNormalTravelTime],
                       source: str) -> Optional[LogNormalTravelTime]:
        n = len(samples)
        if n < self.min_time_samples:
            return None
        logs = np.log(np.asarray(samples, dtype=float))
        if prior is None or self.time_prior_strength <= 0:
            mu = float(logs.mean())
            var = float(logs.var(ddof=1)) if n > 1 else 0.25
            return LogNormalTravelTime(mu, math.sqrt(max(var, 1e-4)), n=n, source=source)
        k = self.time_prior_strength
        mu = float((k * prior.mu + logs.sum()) / (k + n))
        var = float((k * prior.sigma ** 2 + ((logs - mu) ** 2).sum()) / (k + n))
        return LogNormalTravelTime(mu, math.sqrt(max(var, 1e-4)), n=n, source=source)

    def edge_travel_time(self, a: str, b: str) -> Optional[LogNormalTravelTime]:
        prior = LogNormalTravelTime.physical_prior(self.graph.G.edges[a, b]["distance_meters"])
        return self._fit_lognormal(self.edge_dt.get((a, b), []), prior, "learned-edge")

    def pair_travel_time(self, a: str, b: str) -> Optional[LogNormalTravelTime]:
        path = self.graph.get_shortest_path(a, b)
        prior = None
        if path is not None:
            prior = LogNormalTravelTime.physical_prior(self.graph.get_path_distance(path))
        return self._fit_lognormal(self.pair_dt.get((a, b), []), prior, "learned-pair")

    def _histogram(self, samples: Sequence[float]) -> Dict[str, List[float]]:
        if not samples:
            return {"bin_edges": [], "counts": []}
        counts, edges = np.histogram(np.asarray(samples), bins=self.histogram_bins)
        return {"bin_edges": [round(float(x), 3) for x in edges], "counts": [int(c) for c in counts]}

    # --------------------------------------------------------------- export
    def export(self) -> Dict[str, Any]:
        edges: Dict[str, Any] = {}
        for a in self.graph.cameras:
            probs = self.transition_probabilities(a)
            for b, p in probs.items():
                samples = self.edge_dt.get((a, b), [])
                tt = self.edge_travel_time(a, b)
                edges[_edge_key(a, b)] = {
                    "probability": p,
                    "prior_probability": self.graph.get_transition_probability(a, b, source="prior"),
                    "count": self.counts.get((a, b), 0.0),
                    "n_time_samples": len(samples),
                    "travel_time": tt.to_dict() if tt else None,
                    "observed_mean_seconds": float(np.mean(samples)) if samples else None,
                    "observed_std_seconds": float(np.std(samples)) if samples else None,
                    "histogram": self._histogram(samples),
                }
        pairs: Dict[str, Any] = {}
        for (a, b), samples in sorted(self.pair_dt.items()):
            tt = self.pair_travel_time(a, b)
            pairs[_edge_key(a, b)] = {
                "n_time_samples": len(samples),
                "travel_time": tt.to_dict() if tt else None,
                "histogram": self._histogram(samples),
            }
        return {
            "model": "safepass-p3-transition-model",
            "version": 1,
            "venue_name": self.graph.venue_name,
            "params": {
                "alpha": self.alpha,
                "prior_strength": self.prior_strength,
                "time_prior_strength": self.time_prior_strength,
                "min_time_samples": self.min_time_samples,
                "same_visit_gap_seconds": self.same_visit_gap_seconds,
                "reject_teleports": self.reject_teleports,
                "multihop_weight": self.multihop_weight,
                "track_cleaning": self.track_cleaning,
            },
            "stats": dict(self.stats),
            "edges": edges,
            "pairs": pairs,
        }

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.export(), indent=indent)
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def apply_to(self, graph: Optional[VenueTopologyGraph] = None) -> VenueTopologyGraph:
        """Loads the current estimates into `graph` (default: the learner's own graph)."""
        g = graph or self.graph
        g.load_learned_model(self.export())
        return g


def transition_kl(p: Dict[str, Dict[str, float]], q: Dict[str, Dict[str, float]],
                  weights: Optional[Dict[str, float]] = None) -> float:
    """Mean (optionally weighted) KL(p || q) over source cameras; p is the reference."""
    total, wsum = 0.0, 0.0
    for c, pc in p.items():
        qc = q.get(c, {})
        kl = sum(pv * math.log(pv / max(qc.get(k, 0.0), 1e-12)) for k, pv in pc.items() if pv > 0)
        w = 1.0 if weights is None else weights.get(c, 0.0)
        total += w * kl
        wsum += w
    return total / max(wsum, 1e-12)
