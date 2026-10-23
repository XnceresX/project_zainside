"""
topology/trajectory.py
======================
Topology-consistent trajectory reconstruction for SAFEPASS (Track P3).

Input: candidate sightings for one case (from P1/P2 + SpatiotemporalPruner).
    1. Drop candidates with is_spatially_valid == False.
    2. Bucket the rest into time steps of `time_step_seconds`.
    3. Resolve ambiguity with combined = biometric confidence (fused_score) x topology_score:
        - method="greedy": per time step keep the highest combined score, where the
          topology score is recomputed against the previously accepted point (or the
          stored score when rescore=False).
        - method="dp" (default): choose the chain of at most one candidate per time
          step that maximises the sum of combined scores, with topology scores taken
          pairwise between consecutive chosen points. A single early look-alike
          cannot derail the rest of the trajectory, unlike greedy.
    4. Emit ordered contract TrajectoryPoints and flag blind-spot intervals:
       consecutive visits on non-adjacent cameras, or gaps longer than the expected
       transit (upper end of the learned travel-time interval when a model is loaded,
       else the physical path-based dt_max).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from contract.schema import Candidate, TrajectoryPoint
from topology.graph import VenueTopologyGraph, to_seconds
from topology.pruner import SpatiotemporalPruner


@dataclass
class BlindSpotInterval:
    from_camera: str
    to_camera: str
    start_timestamp: str
    end_timestamp: str
    gap_seconds: float
    reason: str                       # "non_adjacent" | "gap_exceeds_transit" | "unseen_return"
    expected_window_seconds: Optional[Tuple[float, float]] = None
    inferred_path: List[str] = field(default_factory=list)  # most likely unseen intermediate cameras


@dataclass
class ReconstructionResult:
    points: List[TrajectoryPoint]
    blind_spots: List[BlindSpotInterval]
    selected: List[Candidate]
    dropped_invalid: int = 0

    @property
    def camera_sequence(self) -> List[str]:
        seq: List[str] = []
        for p in self.points:
            if not seq or seq[-1] != p.camera_id:
                seq.append(p.camera_id)
        return seq


def _bio(c: Candidate) -> float:
    return float(c.fused_score)


def select_consistent_chain(
    cameras: Sequence[str],
    times: Sequence[float],
    scores: Sequence[float],
    pruner: SpatiotemporalPruner,
    time_step_seconds: float = 1.0,
    start_scores: Optional[Sequence[float]] = None,
    max_lookback: int = 60,
) -> List[int]:
    """
    Dynamic programme over time-sorted sightings: picks at most one sighting per time
    step so that consecutive picks are spatially valid, maximising
    sum(score_j * topology(prev -> j)). Returns indices of the chosen chain.
    """
    n = len(cameras)
    if n == 0:
        return []
    step = [int(math.floor(t / time_step_seconds)) for t in times]
    best = [float(scores[j]) * (1.0 if start_scores is None else float(start_scores[j])) for j in range(n)]
    back: List[Optional[int]] = [None] * n
    for j in range(n):
        for i in range(max(0, j - max_lookback), j):
            if step[i] >= step[j]:
                continue
            ps = pruner.score_pair(cameras[i], times[i], cameras[j], times[j])
            if not ps.is_valid or ps.score <= 0:
                continue
            val = best[i] + float(scores[j]) * ps.score
            if val > best[j]:
                best[j], back[j] = val, i
    j: Optional[int] = max(range(n), key=lambda k: (best[k], times[k]))
    chain = []
    while j is not None:
        chain.append(j)
        j = back[j]
    return chain[::-1]


class TrajectoryReconstructor:
    def __init__(
        self,
        graph: VenueTopologyGraph,
        pruner: Optional[SpatiotemporalPruner] = None,
        time_step_seconds: float = 1.0,
        method: str = "dp",
        rescore: bool = True,
        max_lookback: int = 60,
        same_camera_gap_seconds: float = 20.0,
        blind_spot_coverage: float = 0.98,
    ):
        """
        blind_spot_coverage: an adjacent-camera gap is flagged when it exceeds the upper
            end of the expected transit window: the central `blind_spot_coverage`
            interval of the learned travel time if loaded, else physical dt_max.
        """
        if method not in ("dp", "greedy"):
            raise ValueError("method must be 'dp' or 'greedy'")
        self.graph = graph
        self.pruner = pruner or SpatiotemporalPruner(graph)
        self.time_step = float(time_step_seconds)
        self.method = method
        self.rescore = rescore
        self.max_lookback = int(max_lookback)
        self.same_camera_gap = float(same_camera_gap_seconds)
        self.blind_spot_coverage = float(blind_spot_coverage)

    # -------------------------------------------------------------- select
    def _greedy(self, cands: List[Candidate], times: List[float]) -> List[int]:
        steps: Dict[int, List[int]] = {}
        for i, t in enumerate(times):
            steps.setdefault(int(math.floor(t / self.time_step)), []).append(i)
        chosen: List[int] = []
        for step in sorted(steps):
            best, best_score = None, 0.0
            for i in steps[step]:
                if self.rescore and chosen:
                    j = chosen[-1]
                    ps = self.pruner.score_pair(cands[j].camera_id, times[j], cands[i].camera_id, times[i])
                    if not ps.is_valid:
                        continue
                    topo = ps.score
                else:
                    topo = 1.0 if cands[i].topology_score is None else cands[i].topology_score
                score = _bio(cands[i]) * topo
                if score > best_score:
                    best, best_score = i, score
            if best is not None:
                chosen.append(best)
        return chosen

    def _dp(self, cands: List[Candidate], times: List[float]) -> List[int]:
        return select_consistent_chain(
            [c.camera_id for c in cands], times, [_bio(c) for c in cands], self.pruner,
            self.time_step,
            start_scores=[1.0 if c.topology_score is None else c.topology_score for c in cands],
            max_lookback=self.max_lookback,
        )

    def _pairwise_topology(self, chosen: List[Candidate], times: List[float]) -> List[float]:
        scores = []
        for k, c in enumerate(chosen):
            if k == 0:
                scores.append(1.0 if c.topology_score is None else float(c.topology_score))
            else:
                scores.append(self.pruner.score_pair(chosen[k - 1].camera_id, times[k - 1],
                                                     c.camera_id, times[k]).score)
        return scores

    # ------------------------------------------------------------ public
    def reconstruct(self, candidates: Sequence[Candidate]) -> ReconstructionResult:
        valid = [c for c in candidates if c.is_spatially_valid]
        dropped = len(candidates) - len(valid)
        valid.sort(key=lambda c: to_seconds(c.timestamp))
        times = [to_seconds(c.timestamp) for c in valid]

        idx = self._dp(valid, times) if self.method == "dp" else self._greedy(valid, times)
        chosen = [valid[i] for i in idx]
        ctimes = [times[i] for i in idx]
        topo = self._pairwise_topology(chosen, ctimes) if self.rescore else [
            1.0 if c.topology_score is None else c.topology_score for c in chosen]

        selected, points = [], []
        for c, s in zip(chosen, topo):
            selected.append(c.model_copy(update={"topology_score": s, "is_spatially_valid": True}))
            points.append(TrajectoryPoint(
                camera_id=c.camera_id,
                camera_name=c.camera_name or self.graph.camera_name(c.camera_id),
                timestamp=c.timestamp,
                confidence=float(min(1.0, max(0.0, _bio(c) * s))),
                snapshot_url=c.frame_crop_path,
            ))
        return ReconstructionResult(points, self.find_blind_spots(points), selected, dropped)

    def reconstruct_points(self, candidates: Sequence[Candidate]) -> List[TrajectoryPoint]:
        return self.reconstruct(candidates).points

    def find_blind_spots(self, points: Sequence[TrajectoryPoint]) -> List[BlindSpotInterval]:
        """Flags unobserved stretches between consecutive trajectory points."""
        out: List[BlindSpotInterval] = []
        for a, b in zip(points[:-1], points[1:]):
            gap = to_seconds(b.timestamp) - to_seconds(a.timestamp)
            if a.camera_id == b.camera_id:
                if gap > self.same_camera_gap:
                    out.append(BlindSpotInterval(a.camera_id, b.camera_id, a.timestamp, b.timestamp,
                                                 gap, "unseen_return"))
                continue
            if self.pruner.use_learned:
                window = self.graph.get_expected_transit_window(a.camera_id, b.camera_id, self.blind_spot_coverage)
            else:
                window = self.graph.get_transit_window(a.camera_id, b.camera_id)
            path = self.graph.get_most_likely_path(a.camera_id, b.camera_id) or []
            if not self.graph.are_adjacent(a.camera_id, b.camera_id):
                out.append(BlindSpotInterval(a.camera_id, b.camera_id, a.timestamp, b.timestamp,
                                             gap, "non_adjacent", window, path[1:-1]))
            elif window is not None and gap > window[1]:
                out.append(BlindSpotInterval(a.camera_id, b.camera_id, a.timestamp, b.timestamp,
                                             gap, "gap_exceeds_transit", window, []))
        return out
