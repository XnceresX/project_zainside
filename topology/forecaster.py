"""
topology/forecaster.py
======================
Blind-spot next-camera forecasting for SAFEPASS (Track P3).

Given where/when a person was last seen, predict which adjacent cameras they
will appear at next and when:

    P(Cj | Ci)  from the learned transition model (fallback: manual priors)
    momentum    if >= 2 history points exist, the previous camera (the heading
                the person came from) is multiplied by `backtrack_penalty`
    renormalise so probabilities sum to 1

The arrival window (seconds after last_seen_timestamp) is the union over the
predicted cameras of the learned central-coverage interval, or the physical
path-based [dt_min, dt_max] window when nothing has been learned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from contract.schema import SpatialPrediction, TrajectoryPoint
from topology.graph import VenueTopologyGraph, to_seconds

HistoryItem = Union[TrajectoryPoint, str, Dict[str, Any]]


@dataclass
class ForecastDetail:
    prediction: SpatialPrediction
    per_camera_windows: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    previous_camera: Optional[str] = None
    momentum_applied: bool = False
    source: str = "prior"


class BlindSpotForecaster:
    def __init__(
        self,
        graph: VenueTopologyGraph,
        backtrack_penalty: float = 0.35,
        use_learned: bool = True,
        coverage: float = 0.9,
        top_k: Optional[int] = None,
        window_mass: float = 1.0,
    ):
        """
        backtrack_penalty: multiplier on P(return to previous camera) when momentum applies
                           (1.0 disables momentum, 0.0 forbids immediate backtracking).
        coverage: central probability mass of learned travel-time used for windows.
        top_k: keep only the k most likely cameras (None = all adjacent cameras).
        window_mass: the arrival window is the union over the most likely cameras whose
                     cumulative probability first reaches this mass (1.0 = all listed).
        """
        if not 0.0 <= backtrack_penalty <= 1.0:
            raise ValueError("backtrack_penalty must be in [0, 1]")
        self.graph = graph
        self.backtrack_penalty = backtrack_penalty
        self.use_learned = use_learned
        self.coverage = coverage
        self.top_k = top_k
        self.window_mass = window_mass

    @staticmethod
    def _camera_of(item: HistoryItem) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return item["camera_id"]
        return item.camera_id

    @staticmethod
    def _time_of(item: HistoryItem) -> Optional[float]:
        ts = item.get("timestamp") if isinstance(item, dict) else getattr(item, "timestamp", None)
        return None if ts is None else to_seconds(ts)

    def previous_camera(self, last_seen_camera_id: str, history: Optional[Sequence[HistoryItem]]) -> Optional[str]:
        """Most recent history camera different from the last-seen camera (the heading origin)."""
        if not history or len(history) < 2:
            return None
        items = list(history)
        if all(self._time_of(h) is not None for h in items):
            items.sort(key=lambda h: self._time_of(h))
        for h in reversed(items):
            cam = self._camera_of(h)
            if cam != last_seen_camera_id:
                return cam
        return None

    def _window(self, a: str, b: str) -> Tuple[float, float]:
        if self.use_learned:
            w = self.graph.get_expected_transit_window(a, b, self.coverage)
        else:
            w = self.graph.get_transit_window(a, b)
        return w if w is not None else (0.0, 0.0)

    def forecast_detailed(
        self,
        last_seen_camera_id: str,
        last_seen_timestamp: Any = None,
        trajectory_history: Optional[Sequence[HistoryItem]] = None,
    ) -> ForecastDetail:
        if last_seen_camera_id not in self.graph.G:
            raise KeyError(f"Unknown camera '{last_seen_camera_id}'")
        source = "learned" if (self.use_learned and self.graph.has_learned_model) else "prior"
        probs = self.graph.get_outgoing_distribution(last_seen_camera_id, source="auto" if source == "learned" else "prior")
        if not probs:
            pred = SpatialPrediction(current_camera_id=last_seen_camera_id, candidate_next_cameras=[],
                                     transition_probabilities=[], estimated_arrival_window_seconds=(0, 0))
            return ForecastDetail(pred, source=source)

        prev = self.previous_camera(last_seen_camera_id, trajectory_history)
        momentum = prev is not None and prev in probs and len(probs) > 1 and self.backtrack_penalty < 1.0
        if momentum:
            probs[prev] *= self.backtrack_penalty
        total = sum(probs.values())
        ranked = sorted(((c, p / total) for c, p in probs.items()), key=lambda x: (-x[1], x[0]))
        if self.top_k is not None:
            ranked = ranked[: self.top_k]
            s = sum(p for _, p in ranked)
            ranked = [(c, p / s) for c, p in ranked]

        windows = {c: self._window(last_seen_camera_id, c) for c, _ in ranked}
        lo, hi, mass = math.inf, 0.0, 0.0
        for c, p in ranked:
            lo, hi = min(lo, windows[c][0]), max(hi, windows[c][1])
            mass += p
            if mass >= self.window_mass - 1e-9:
                break

        pred = SpatialPrediction(
            current_camera_id=last_seen_camera_id,
            candidate_next_cameras=[c for c, _ in ranked],
            transition_probabilities=[float(p) for _, p in ranked],
            estimated_arrival_window_seconds=(int(math.floor(lo)), int(math.ceil(hi))),
        )
        return ForecastDetail(pred, windows, prev, momentum, source)

    def forecast(
        self,
        last_seen_camera_id: str,
        last_seen_timestamp: Any = None,
        trajectory_history: Optional[Sequence[HistoryItem]] = None,
    ) -> SpatialPrediction:
        """
        Returns a contract SpatialPrediction. The arrival window is relative to
        last_seen_timestamp (seconds); add it to the timestamp for absolute times.
        """
        return self.forecast_detailed(last_seen_camera_id, last_seen_timestamp, trajectory_history).prediction
