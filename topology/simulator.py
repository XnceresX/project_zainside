"""
topology/simulator.py
=====================
Synthetic venue walker simulator for SAFEPASS (Track P3).

Produces independent ground truth so topology evaluation is not circular:
    - N walkers with per-person speed ~ Normal(1.3, 0.3) m/s (clipped)
    - route choice sampled from transition probabilities, with an optional
      backtrack aversion (people rarely turn straight back)
    - random dwell at cameras and random stops / loitering inside corridors
    - blind-spot traversals: a whole camera visit can go unobserved
    - per-frame detection dropout inside an observed visit
    - injected false biometric matches: a sighting of a *different* person at a
      *distant* camera emitted as a look-alike candidate for a target

All randomness flows from one seeded numpy Generator, so runs are reproducible.
Times are float seconds from `base_time`; ISO strings are produced on export.
No images or real people are involved: everything is synthetic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from contract.schema import Candidate
from topology.graph import VenueTopologyGraph

DEFAULT_BASE_TIME = datetime(2026, 10, 23, 10, 0, 0)


@dataclass
class Sighting:
    """One camera detection attributed to `person_id`."""
    person_id: str
    camera_id: str
    timestamp: float
    confidence: float = 1.0
    is_false_match: bool = False
    source_person_id: Optional[str] = None  # who was really seen (differs for look-alikes)

    def iso(self, base_time: datetime = DEFAULT_BASE_TIME) -> str:
        return (base_time + timedelta(seconds=self.timestamp)).isoformat()


@dataclass
class Visit:
    """Ground-truth presence of a person inside one camera's field of view."""
    person_id: str
    camera_id: str
    arrival_time: float
    departure_time: float
    observed: bool
    walk_seconds_to_next: Optional[float] = None


@dataclass
class SimulatorConfig:
    speed_mean_mps: float = 1.3
    speed_std_mps: float = 0.3
    speed_clip_mps: Tuple[float, float] = (0.6, 2.4)
    edge_speed_jitter: float = 0.10          # per-corridor multiplicative speed noise (std)
    path_overhead: Tuple[float, float] = (1.0, 1.1)  # walked distance / corridor distance
    dwell_seconds: Tuple[float, float] = (2.0, 8.0)
    p_loiter_at_camera: float = 0.10
    loiter_mean_seconds: float = 30.0
    p_stop_in_corridor: float = 0.20
    stop_mean_seconds: float = 20.0
    p_blind_visit: float = 0.15              # whole visit unobserved (blind-spot traversal)
    p_frame_detect: float = 0.80
    sample_interval_seconds: float = 1.0
    backtrack_factor: float = 0.35           # multiplier on returning to the previous camera
    entry_cameras: Optional[Dict[str, float]] = None
    exit_cameras: Optional[Sequence[str]] = None
    p_exit: float = 0.5
    min_hops: int = 3
    max_hops: int = 12
    start_window_seconds: float = 1800.0
    genuine_confidence: Tuple[float, float] = (0.78, 0.08)
    lookalike_confidence: Tuple[float, float] = (0.72, 0.08)


@dataclass
class SimulationResult:
    trajectories: Dict[str, List[Visit]]
    sightings: List[Sighting]
    false_matches: List[Sighting] = field(default_factory=list)
    speeds: Dict[str, float] = field(default_factory=dict)
    base_time: datetime = DEFAULT_BASE_TIME
    seed: Optional[int] = None

    @property
    def person_ids(self) -> List[str]:
        return list(self.trajectories)

    def sighting_log(self, include_false_matches: bool = True) -> List[Sighting]:
        """Time-ordered log of (person_id, camera_id, timestamp) as a re-ID system would emit."""
        log = list(self.sightings) + (list(self.false_matches) if include_false_matches else [])
        return sorted(log, key=lambda s: (s.timestamp, s.person_id, s.camera_id))

    def sightings_for(self, person_id: str, include_false_matches: bool = True) -> List[Sighting]:
        return [s for s in self.sighting_log(include_false_matches) if s.person_id == person_id]

    def candidates_for(self, person_id: str, case_id: Optional[str] = None,
                       graph: Optional[VenueTopologyGraph] = None) -> Tuple[List[Candidate], Dict[str, bool]]:
        """
        Exports the target's sightings (genuine + look-alikes) as contract Candidates.
        Returns (candidates, labels) where labels[candidate_id] is True for genuine sightings.
        """
        case_id = case_id or f"CASE-{person_id}"
        cands, labels = [], {}
        for i, s in enumerate(self.sightings_for(person_id)):
            cid = f"{person_id}-S{i:04d}"
            cands.append(Candidate(
                case_id=case_id,
                candidate_id=cid,
                camera_id=s.camera_id,
                camera_name=graph.camera_name(s.camera_id) if graph else s.camera_id,
                timestamp=s.iso(self.base_time),
                face_score=s.confidence,
                fused_score=s.confidence,
                topology_score=None,
                is_spatially_valid=True,
            ))
            labels[cid] = not s.is_false_match
        return cands, labels

    def ground_truth_transitions(self) -> List[Dict[str, object]]:
        """Every true camera-to-camera move, including ones through blind spots."""
        out = []
        for pid, visits in self.trajectories.items():
            for a, b in zip(visits[:-1], visits[1:]):
                out.append({
                    "person_id": pid,
                    "from_camera": a.camera_id,
                    "to_camera": b.camera_id,
                    "departure_time": a.departure_time,
                    "arrival_time": b.arrival_time,
                    "from_observed": a.observed,
                    "to_observed": b.observed,
                })
        return out

    def to_dict(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "base_time": self.base_time.isoformat(),
            "speeds": self.speeds,
            "trajectories": {p: [asdict(v) for v in vs] for p, vs in self.trajectories.items()},
            "sightings": [asdict(s) for s in self.sightings],
            "false_matches": [asdict(s) for s in self.false_matches],
        }

    def to_json(self, path: Optional[str] = None) -> str:
        text = json.dumps(self.to_dict(), indent=1)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text


class TrajectorySimulator:
    """
    Simulates walkers on `graph`. The graph supplies corridor distances and,
    unless `transition_probs` overrides them, the route-choice probabilities.
    Pass a perturbed "true" venue to keep the ground truth independent of the
    manual priors used by the system under test.
    """

    def __init__(
        self,
        graph: VenueTopologyGraph,
        config: Optional[SimulatorConfig] = None,
        transition_probs: Optional[Dict[str, Dict[str, float]]] = None,
        seed: int = 42,
    ):
        self.graph = graph
        self.cfg = config or SimulatorConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.transition_probs = transition_probs or {
            c: graph.get_outgoing_distribution(c, source="prior") for c in graph.cameras
        }
        cams = graph.cameras
        default_entry = {c: w for c, w in {"CAM-01": 0.5, "CAM-08": 0.3, "CAM-06": 0.2}.items() if c in graph.G}
        self.entry = self.cfg.entry_cameras or default_entry or {c: 1.0 for c in cams}
        default_exit = [c for c in ("CAM-01", "CAM-06", "CAM-08") if c in graph.G]
        self.exits = set(self.cfg.exit_cameras or default_exit)

    # ------------------------------------------------------------- helpers
    def _sample_speed(self) -> float:
        lo, hi = self.cfg.speed_clip_mps
        return float(np.clip(self.rng.normal(self.cfg.speed_mean_mps, self.cfg.speed_std_mps), lo, hi))

    def _choose_next(self, cam: str, prev: Optional[str]) -> Optional[str]:
        probs = dict(self.transition_probs.get(cam, {}))
        if not probs:
            return None
        if prev is not None and prev in probs and len(probs) > 1:
            probs[prev] *= self.cfg.backtrack_factor
        keys = sorted(probs)
        p = np.array([probs[k] for k in keys], dtype=float)
        p /= p.sum()
        return keys[int(self.rng.choice(len(keys), p=p))]

    def _walk_seconds(self, a: str, b: str, speed: float) -> float:
        d = self.graph.G.edges[a, b]["distance_meters"] * self.rng.uniform(*self.cfg.path_overhead)
        v = speed * max(0.3, self.rng.normal(1.0, self.cfg.edge_speed_jitter))
        t = d / max(v, 0.3)
        if self.rng.random() < self.cfg.p_stop_in_corridor:
            t += self.rng.exponential(self.cfg.stop_mean_seconds)
        return float(t)

    def _confidence(self, mean_std: Tuple[float, float]) -> float:
        return float(np.clip(self.rng.normal(*mean_std), 0.30, 0.99))

    # ---------------------------------------------------------- simulation
    def simulate_person(self, person_id: str, start_time: float) -> Tuple[List[Visit], List[Sighting], float]:
        cfg = self.cfg
        speed = self._sample_speed()
        entries = sorted(self.entry)
        w = np.array([self.entry[c] for c in entries], dtype=float)
        cam = entries[int(self.rng.choice(len(entries), p=w / w.sum()))]
        prev: Optional[str] = None
        t = start_time
        visits: List[Visit] = []
        sightings: List[Sighting] = []

        for hop in range(cfg.max_hops + 1):
            dwell = self.rng.uniform(*cfg.dwell_seconds)
            if self.rng.random() < cfg.p_loiter_at_camera:
                dwell += self.rng.exponential(cfg.loiter_mean_seconds)
            observed = bool(self.rng.random() >= cfg.p_blind_visit)
            visit = Visit(person_id, cam, t, t + dwell, observed)
            if observed:
                frames = np.arange(t, t + dwell + 1e-9, cfg.sample_interval_seconds)
                hits = [ft for ft in frames if self.rng.random() < cfg.p_frame_detect]
                if not hits:
                    hits = [float(frames[int(self.rng.integers(len(frames)))])]
                for ft in hits:
                    sightings.append(Sighting(person_id, cam, float(ft),
                                              self._confidence(cfg.genuine_confidence),
                                              False, person_id))
            visits.append(visit)

            if hop >= cfg.min_hops and cam in self.exits and self.rng.random() < cfg.p_exit:
                break
            if hop == cfg.max_hops:
                break
            nxt = self._choose_next(cam, prev)
            if nxt is None:
                break
            walk = self._walk_seconds(cam, nxt, speed)
            visit.walk_seconds_to_next = walk
            prev, cam, t = cam, nxt, visit.departure_time + walk

        return visits, sightings, speed

    def simulate(
        self,
        n_people: int = 100,
        p_false_match: float = 0.0,
        target_ids: Optional[Sequence[str]] = None,
        min_false_hops: int = 2,
        false_time_window: float = 30.0,
        id_prefix: str = "P",
        base_time: datetime = DEFAULT_BASE_TIME,
    ) -> SimulationResult:
        """
        Simulates `n_people` walkers. With p_false_match > 0, each genuine sighting of a
        target (default: all walkers) spawns, with that probability, a look-alike candidate.
        """
        trajectories: Dict[str, List[Visit]] = {}
        sightings: List[Sighting] = []
        speeds: Dict[str, float] = {}
        for i in range(n_people):
            pid = f"{id_prefix}{i:04d}"
            start = float(self.rng.uniform(0.0, self.cfg.start_window_seconds))
            visits, sights, speed = self.simulate_person(pid, start)
            trajectories[pid] = visits
            sightings.extend(sights)
            speeds[pid] = speed

        result = SimulationResult(trajectories, sorted(sightings, key=lambda s: s.timestamp),
                                  speeds=speeds, base_time=base_time, seed=self.seed)
        if p_false_match > 0:
            targets = list(target_ids) if target_ids is not None else list(trajectories)
            result.false_matches = self.inject_false_matches(
                result, targets, p_false_match, min_false_hops, false_time_window)
        return result

    def inject_false_matches(
        self,
        result: SimulationResult,
        target_ids: Sequence[str],
        p: float,
        min_hops: int = 2,
        time_window: float = 30.0,
    ) -> List[Sighting]:
        """
        For every genuine target sighting, with probability p, emit a look-alike: a real
        sighting of a different person at a camera >= `min_hops` away from where the
        target actually is, within +/- `time_window` s. If no such sighting exists, a
        synthetic one at a random distant camera is fabricated.
        """
        cams = self.graph.cameras
        hops = {a: {b: self.graph.hop_distance(a, b) for b in cams} for a in cams}
        times = np.array([s.timestamp for s in result.sightings])
        by_order = result.sightings
        targets = set(target_ids)
        out: List[Sighting] = []

        for s in by_order:
            if s.person_id not in targets or self.rng.random() >= p:
                continue
            lo = int(np.searchsorted(times, s.timestamp - time_window, side="left"))
            hi = int(np.searchsorted(times, s.timestamp + time_window, side="right"))
            pool = [o for o in by_order[lo:hi]
                    if o.person_id != s.person_id
                    and (hops[s.camera_id][o.camera_id] or 0) >= min_hops]
            if pool:
                o = pool[int(self.rng.integers(len(pool)))]
                cam, ts, src = o.camera_id, o.timestamp, o.person_id
            else:
                distant = [c for c in cams if (hops[s.camera_id][c] or 0) >= min_hops]
                if not distant:
                    continue
                cam = distant[int(self.rng.integers(len(distant)))]
                ts = s.timestamp + float(self.rng.uniform(-time_window, time_window))
                src = "SYNTHETIC-LOOKALIKE"
            out.append(Sighting(s.person_id, cam, float(max(ts, 0.0)),
                                self._confidence(self.cfg.lookalike_confidence), True, src))
        return sorted(out, key=lambda x: x.timestamp)


def perturbed_truth_venue(
    graph: VenueTopologyGraph,
    seed: int = 7,
    concentration: float = 8.0,
    distance_scale: Tuple[float, float] = (1.0, 1.3),
) -> Tuple[VenueTopologyGraph, Dict[str, Dict[str, float]]]:
    """
    Builds a hidden "true" venue that differs from the manual one: transition
    probabilities ~ Dirichlet(concentration x manual prior) and corridor distances
    stretched by U(distance_scale) (obstacles, queues, detours). Distances are only
    ever stretched, so physical minimum transit bounds of the manual graph stay valid.
    Returns (truth_graph, true_transition_probs).
    """
    rng = np.random.default_rng(seed)
    truth = graph.copy()
    seen = set()
    for u, v in sorted(truth.G.edges):
        key = tuple(sorted((u, v)))
        if key in seen:
            continue
        seen.add(key)
        scale = float(rng.uniform(*distance_scale))
        for a, b in ((u, v), (v, u)):
            if truth.G.has_edge(a, b):
                e = truth.G.edges[a, b]
                e["distance_meters"] = round(e["distance_meters"] * scale, 1)
                e["min_transit_seconds"] = e["distance_meters"] / truth.v_max
                e["max_transit_seconds"] = e["distance_meters"] / truth.v_min
    truth._window_cache.clear()

    probs: Dict[str, Dict[str, float]] = {}
    for c in truth.cameras:
        succ = truth.get_adjacent_cameras(c)
        if not succ:
            continue
        alpha = np.array([max(truth.G.edges[c, s]["prior_probability"], 1e-3) * concentration for s in succ])
        p = rng.dirichlet(alpha)
        probs[c] = {s: float(x) for s, x in zip(succ, p)}
        truth.set_priors(c, probs[c])
    return truth, probs


def iso_to_seconds(ts: str, base_time: datetime = DEFAULT_BASE_TIME) -> float:
    return (datetime.fromisoformat(ts) - base_time).total_seconds()


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="SAFEPASS P3 trajectory simulator")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-false", type=float, default=0.1)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    g = VenueTopologyGraph.default_venue()
    res = TrajectorySimulator(g, seed=args.seed).simulate(args.n, p_false_match=args.p_false)
    print(f"Walkers: {len(res.trajectories)} | genuine sightings: {len(res.sightings)} | "
          f"look-alikes: {len(res.false_matches)}")
    blind = sum(1 for vs in res.trajectories.values() for v in vs if not v.observed)
    total = sum(len(vs) for vs in res.trajectories.values())
    print(f"Visits: {total} | unobserved (blind-spot) visits: {blind} ({blind / max(total, 1):.1%})")
    if args.out:
        res.to_json(args.out)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    _main()
