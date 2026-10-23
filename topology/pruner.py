"""
topology/pruner.py
==================
Spatiotemporal candidate pruning for SAFEPASS (Track P3).

For an ordered pair of sightings A = (C_A, t_A) -> B = (C_B, t_B), dt = t_B - t_A:

    1. dt < 0 (minus clock-skew tolerance)   -> invalid, 0.0   (chronology violation)
    2. C_A == C_B                            -> valid,   1.0
    3. no walkable path C_A -> C_B           -> invalid, 0.0
    4. dt < path-based dt_min                -> invalid, 0.0   (teleportation)
    5. otherwise valid, score in [0, 1]:
         learned model:  pdf(dt) / pdf(mode) of the learned log-normal travel time
                         for this pair (edge, blind-spot pair, or composed path).
                         Beyond dt_max the likelihood decays naturally.
         manual fallback: 1.0 inside [dt_min, dt_max], and beyond dt_max
                         exp(-lambda * (dt - dt_max) / dt_max)
                         i.e. lambda is a decay rate per unit of *relative* overshoot
                         (default 1.0: a gap of 2 x dt_max scores e^-1 ~ 0.37).

dt_min / dt_max are path-based (summed along the shortest path; see
VenueTopologyGraph.get_transit_window).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from contract.schema import Candidate
from topology.graph import LogNormalTravelTime, VenueTopologyGraph, to_seconds

DEFAULT_DECAY_LAMBDA = 1.0


@dataclass
class PairScore:
    is_valid: bool
    score: float
    reason: str
    dt_seconds: float
    window: Optional[Tuple[float, float]] = None


class SpatiotemporalPruner:
    def __init__(
        self,
        graph: VenueTopologyGraph,
        decay_lambda: float = DEFAULT_DECAY_LAMBDA,
        use_learned: bool = True,
        clock_skew_tolerance_seconds: float = 0.0,
        min_valid_score: float = 0.0,
    ):
        """
        decay_lambda: manual-fallback decay rate per relative overshoot beyond dt_max.
        use_learned: use learned travel-time distributions when the graph has them.
        clock_skew_tolerance_seconds: slack subtracted from dt_min (unsynchronised NVRs).
        min_valid_score: valid pairs scoring strictly below this are marked invalid (0 = never).
        """
        self.graph = graph
        self.decay_lambda = float(decay_lambda)
        self.use_learned = use_learned
        self.tolerance = float(clock_skew_tolerance_seconds)
        self.min_valid_score = float(min_valid_score)
        self._dist_cache: Dict[Tuple[str, str], Optional[LogNormalTravelTime]] = {}

    def clear_cache(self) -> None:
        self._dist_cache.clear()

    def _distribution(self, a: str, b: str) -> Optional[LogNormalTravelTime]:
        if not self.use_learned:
            return None
        key = (a, b)
        if key not in self._dist_cache:
            self._dist_cache[key] = self.graph.get_travel_time_distribution(a, b)
        return self._dist_cache[key]

    def manual_score(self, dt: float, window: Tuple[float, float]) -> float:
        lo, hi = window
        if dt <= hi:
            return 1.0
        return float(math.exp(-self.decay_lambda * (dt - hi) / max(hi, 1e-6)))

    # ------------------------------------------------------------ core rule
    def score_pair(self, cam_a: str, t_a: Any, cam_b: str, t_b: Any) -> PairScore:
        dt = to_seconds(t_b) - to_seconds(t_a)
        if dt < -self.tolerance:
            return PairScore(False, 0.0, "chronology", dt)
        if cam_a == cam_b:
            return PairScore(True, 1.0, "same_camera", dt, (0.0, 0.0))

        window = self.graph.get_transit_window(cam_a, cam_b)
        if window is None:
            return PairScore(False, 0.0, "no_path", dt)
        if dt < window[0] - self.tolerance:
            return PairScore(False, 0.0, "teleportation", dt, window)

        dist = self._distribution(cam_a, cam_b)
        if dist is not None:
            score, reason = dist.score(dt), "learned"
        else:
            score, reason = self.manual_score(dt, window), "manual"
        if dt > window[1]:
            reason += "_beyond_max"
        valid = score >= self.min_valid_score
        return PairScore(valid, float(score) if valid else 0.0, reason if valid else "low_score", dt, window)

    def is_teleportation(self, cam_a: str, t_a: Any, cam_b: str, t_b: Any) -> bool:
        return self.score_pair(cam_a, t_a, cam_b, t_b).reason == "teleportation"

    # ----------------------------------------------------- Candidate-level
    def score_candidates(self, a: Candidate, b: Candidate) -> PairScore:
        return self.score_pair(a.camera_id, a.timestamp, b.camera_id, b.timestamp)

    def prune_against_anchor(
        self,
        candidates: Sequence[Candidate],
        anchor_camera_id: str,
        anchor_timestamp: Any,
    ) -> List[Candidate]:
        """
        Scores every candidate against a confirmed anchor sighting (e.g. the intake
        "last seen" camera/time) and returns updated copies, time-ordered.
        """
        out = []
        for c in candidates:
            ps = self.score_pair(anchor_camera_id, anchor_timestamp, c.camera_id, c.timestamp)
            out.append(c.model_copy(update={"is_spatially_valid": ps.is_valid, "topology_score": ps.score}))
        return sorted(out, key=lambda c: to_seconds(c.timestamp))

    def prune_sequence(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        """
        Chronological chain pruning: each candidate is scored against the most recent
        candidate that was itself spatially valid. The first candidate is the anchor
        (score 1.0). Use TrajectoryReconstructor for globally consistent selection.
        """
        ordered = sorted(candidates, key=lambda c: to_seconds(c.timestamp))
        out: List[Candidate] = []
        last: Optional[Candidate] = None
        for c in ordered:
            if last is None:
                upd = c.model_copy(update={"is_spatially_valid": True, "topology_score": 1.0})
            else:
                ps = self.score_candidates(last, c)
                upd = c.model_copy(update={"is_spatially_valid": ps.is_valid, "topology_score": ps.score})
            out.append(upd)
            if upd.is_spatially_valid:
                last = upd
        return out

    def prune(
        self,
        candidates: Sequence[Candidate],
        anchor_camera_id: Optional[str] = None,
        anchor_timestamp: Any = None,
    ) -> List[Candidate]:
        """Anchor pruning when an anchor is given, chain pruning otherwise."""
        if anchor_camera_id is not None and anchor_timestamp is not None:
            return self.prune_against_anchor(candidates, anchor_camera_id, anchor_timestamp)
        return self.prune_sequence(candidates)
