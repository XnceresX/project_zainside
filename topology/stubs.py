"""
topology/stubs.py
=================
Generates schema-valid stub JSON for P4 (dashboard / backend) and P1/P2 so they can
integrate against P3 outputs without waiting for real camera feeds.

All content is synthetic (seeded simulator walkers, no images, no real people).

    python -m topology.stubs          # writes topology/stubs/*.json

Files (contract types from contract/schema.py unless noted):
    camera_nodes.json        List[CameraNode]
    candidates_pruned.json   List[Candidate]      (topology_score / is_spatially_valid filled)
    trajectory.json          List[TrajectoryPoint]
    spatial_prediction.json  SpatialPrediction
    search_case_result.json  SearchCaseResult     (all of the above for one case)
    venue_graph.json         P3-internal graph format (coordinates, distances, priors, learned model)
    blind_spots.json         P3 extension, not in the schema: flagged unobserved intervals
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict

from contract.schema import SearchCaseResult
from topology.forecaster import BlindSpotForecaster
from topology.graph import VenueTopologyGraph
from topology.learner import TransitionLearner
from topology.pruner import SpatiotemporalPruner
from topology.simulator import TrajectorySimulator
from topology.trajectory import TrajectoryReconstructor

STUB_DIR = os.path.join("topology", "stubs")
CASE_ID = "CASE-DEMO-0001"


def build_stubs(seed: int = 2026) -> Dict[str, object]:
    g = VenueTopologyGraph.default_venue()
    train = TrajectorySimulator(g, seed=seed).simulate(150, p_false_match=0.1)
    g.load_learned_model(TransitionLearner(g).fit(train.sighting_log()).export())

    sim = TrajectorySimulator(g, seed=seed + 1)
    res = sim.simulate(40, p_false_match=0.15)
    target = res.people_with_blind_spots(g)[0]
    cands, _ = res.candidates_for(target, case_id=CASE_ID, graph=g)

    first = cands[0]
    pruned = SpatiotemporalPruner(g).prune(cands, anchor_camera_id=first.camera_id, anchor_timestamp=first.timestamp)
    rec = TrajectoryReconstructor(g).reconstruct(pruned)
    selected = {c.candidate_id: c for c in rec.selected}
    # Final per-candidate view: the selected chain carries pairwise scores; rejected
    # candidates keep their anchor score but are marked invalid if not in the chain.
    final = [selected.get(c.candidate_id, c.model_copy(update={"is_spatially_valid": False}))
             for c in pruned]

    last = rec.points[-1]
    pred = BlindSpotForecaster(g).forecast(last.camera_id, last.timestamp, rec.points)
    case = SearchCaseResult(
        case_id=CASE_ID,
        reported_person_name="Synthetic Walker (demo)",
        report_timestamp=first.timestamp,
        last_seen_camera_id=first.camera_id,
        status="active",
        candidates=final,
        reconstructed_trajectory=rec.points,
        predicted_next_zone=pred,
    )
    return {
        "camera_nodes": [n.model_dump() for n in g.to_camera_nodes()],
        "candidates_pruned": [c.model_dump() for c in final],
        "trajectory": [p.model_dump() for p in rec.points],
        "spatial_prediction": pred.model_dump(),
        "search_case_result": case.model_dump(),
        "venue_graph": g.to_dict(),
        "blind_spots": [asdict(b) for b in rec.blind_spots],
    }


def write_stubs(out_dir: str = STUB_DIR, seed: int = 2026) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, payload in build_stubs(seed).items():
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        paths[name] = path
    return paths


if __name__ == "__main__":
    for name, path in write_stubs().items():
        print(f"  {name:<20} -> {path}")
