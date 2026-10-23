"""
topology/test_pipeline.py
=========================
Tests for the P3 topology pipeline: graph, simulator, learner, pruner,
forecaster and trajectory reconstruction. Everything runs on synthetic data.

Run:
    python -m topology.test_pipeline
    (also collected by: python -m pytest topology/test_pipeline.py)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from contract.schema import Candidate, CameraNode, SpatialPrediction, TrajectoryPoint
from topology.forecaster import BlindSpotForecaster
from topology.graph import V_MAX_MPS, V_MIN_MPS, VenueTopologyGraph
from topology.learner import TransitionLearner, transition_kl
from topology.pruner import SpatiotemporalPruner
from topology.simulator import TrajectorySimulator, perturbed_truth_venue
from topology.trajectory import TrajectoryReconstructor

BASE = datetime(2026, 10, 23, 10, 0, 0)


def ts(sec: float) -> str:
    return (BASE + timedelta(seconds=sec)).isoformat()


def to_s(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def cand(cid: str, cam: str, sec: float, score: float = 0.8, case: str = "CASE-T") -> Candidate:
    return Candidate(case_id=case, candidate_id=cid, camera_id=cam, camera_name=cam,
                     timestamp=ts(sec), fused_score=score)


def _empirical_transitions(result) -> dict:
    counts = defaultdict(Counter)
    for t in result.ground_truth_transitions():
        counts[t["from_camera"]][t["to_camera"]] += 1
    return {a: {b: n / sum(c.values()) for b, n in c.items()} for a, c in counts.items()}


# --------------------------------------------------------------------------- graph
def test_graph_structure():
    g = VenueTopologyGraph.default_venue()
    assert g.G.number_of_nodes() == 8
    out_deg = dict(g.G.out_degree())
    in_deg = dict(g.G.in_degree())
    assert out_deg["CAM-03"] == 5 and in_deg["CAM-03"] == 5
    assert out_deg["CAM-08"] == 1 and g.get_adjacent_cameras("CAM-08") == ["CAM-01"]
    assert g.get_adjacent_cameras("CAM-01") == ["CAM-02", "CAM-03", "CAM-06", "CAM-08"]
    for c in g.cameras:
        s = sum(g.get_transition_probability(c, v, "prior") for v in g.get_adjacent_cameras(c))
        assert abs(s - 1.0) < 1e-9, f"priors of {c} sum to {s}"
    for u, v, e in g.G.edges(data=True):
        assert math.isclose(e["min_transit_seconds"], e["distance_meters"] / V_MAX_MPS)
        assert math.isclose(e["max_transit_seconds"], e["distance_meters"] / V_MIN_MPS)
    assert not g.are_adjacent("CAM-01", "CAM-04"), "default venue needs a blind-spot gap"
    print("  [PASS] nodes/degrees/priors/edge bounds")


def test_graph_paths_and_windows():
    g = VenueTopologyGraph.default_venue()
    assert g.get_shortest_path("CAM-01", "CAM-04") == ["CAM-01", "CAM-03", "CAM-04"]
    assert g.get_shortest_path("CAM-08", "CAM-07")[0] == "CAM-08"
    assert g.hop_distance("CAM-08", "CAM-07") == 4

    w13, w34 = g.get_transit_window("CAM-01", "CAM-03"), g.get_transit_window("CAM-03", "CAM-04")
    w14 = g.get_transit_window("CAM-01", "CAM-04")
    assert math.isclose(w14[0], w13[0] + w34[0]) and math.isclose(w14[1], w13[1] + w34[1])
    alt = g.get_transit_window("CAM-01", "CAM-04", include_alternatives=True)
    assert alt[0] <= w14[0] + 1e-9 and alt[1] > w14[1]
    assert g.get_transit_window("CAM-02", "CAM-02") == (0.0, 0.0)

    iso = VenueTopologyGraph()
    iso.add_camera("A", "A", "z", (0, 0))
    iso.add_camera("B", "B", "z", (10, 0))
    iso.add_camera("C", "C", "z", (50, 0))
    iso.add_corridor("A", "B", bidirectional=False)
    iso.normalise_priors()
    assert iso.get_transit_window("A", "C") is None and iso.get_shortest_path("A", "C") is None
    assert iso.get_transit_window("B", "A") is None, "one-way corridor must not be reversible"
    print(f"  [PASS] shortest paths + multi-hop window CAM-01->CAM-04 = ({w14[0]:.1f}, {w14[1]:.1f}) s")


def test_graph_serialisation_and_contract():
    g = VenueTopologyGraph.default_venue()
    g2 = VenueTopologyGraph.from_json(g.to_json())
    assert g2.cameras == g.cameras and g2.G.number_of_edges() == g.G.number_of_edges()
    for u, v in g.G.edges:
        assert math.isclose(g.G.edges[u, v]["prior_probability"], g2.G.edges[u, v]["prior_probability"])
    assert g2.G.nodes["CAM-03"]["coordinates"] == g.G.nodes["CAM-03"]["coordinates"]
    nodes = g.to_camera_nodes()
    assert all(isinstance(n, CameraNode) for n in nodes)
    n1 = next(n for n in nodes if n.camera_id == "CAM-01")
    assert set(n1.transit_bounds) == set(n1.connected_cameras)
    lo, hi = n1.transit_bounds["CAM-03"]
    assert isinstance(lo, int) and isinstance(hi, int) and lo < hi
    assert "coordinates" not in CameraNode.model_fields and "distance_meters" not in CameraNode.model_fields
    print("  [PASS] JSON round-trip + CameraNode export")


# ----------------------------------------------------------------------- simulator
def test_simulator_reproducible_and_physical():
    g = VenueTopologyGraph.default_venue()
    r1 = TrajectorySimulator(g, seed=5).simulate(40, p_false_match=0.2)
    r2 = TrajectorySimulator(g, seed=5).simulate(40, p_false_match=0.2)
    r3 = TrajectorySimulator(g, seed=6).simulate(40, p_false_match=0.2)
    key = lambda r: [(s.person_id, s.camera_id, round(s.timestamp, 6)) for s in r.sighting_log()]
    assert key(r1) == key(r2), "same seed must reproduce"
    assert key(r1) != key(r3), "different seed should differ"

    big = TrajectorySimulator(g, seed=11).simulate(300)
    speeds = list(big.speeds.values())
    assert abs(sum(speeds) / len(speeds) - 1.3) < 0.06
    blind = sum(1 for vs in big.trajectories.values() for v in vs if not v.observed)
    assert blind > 0, "expected blind-spot traversals"
    for vs in big.trajectories.values():
        for a, b in zip(vs[:-1], vs[1:]):
            assert g.are_adjacent(a.camera_id, b.camera_id)
            assert b.arrival_time - a.departure_time >= g.get_transit_window(a.camera_id, b.camera_id)[0]

    # look-alikes are at distant cameras relative to where the target really was
    assert len(r1.false_matches) > 0
    for f in r1.false_matches:
        assert f.is_false_match and f.source_person_id != f.person_id
        genuine = [s for s in r1.sightings if s.person_id == f.person_id and abs(s.timestamp - f.timestamp) <= 30.0]
        assert any(g.hop_distance(s.camera_id, f.camera_id) >= 2 for s in genuine)
    print(f"  [PASS] seeded, mean speed {sum(speeds) / len(speeds):.2f} m/s, "
          f"{blind} blind visits, {len(r1.false_matches)} look-alikes")


# ------------------------------------------------------------------------- learner
def test_learner_cold_start_and_learning():
    g = VenueTopologyGraph.default_venue()
    exact = TransitionLearner(g, alpha=0.0)
    for c in g.cameras:
        for v, p in exact.transition_probabilities(c).items():
            assert math.isclose(p, g.get_transition_probability(c, v, "prior"))

    truth, probs = perturbed_truth_venue(g, seed=3)
    train = TrajectorySimulator(truth, transition_probs=probs, seed=21).simulate(400, p_false_match=0.1)
    ref = _empirical_transitions(TrajectorySimulator(truth, transition_probs=probs, seed=99).simulate(2000))
    learner = TransitionLearner(g).fit(train.sighting_log())
    manual = {c: g.get_outgoing_distribution(c, "prior") for c in g.cameras}
    learned = {c: learner.transition_probabilities(c) for c in g.cameras}
    kl_manual, kl_learned = transition_kl(ref, manual), transition_kl(ref, learned)
    assert kl_learned < kl_manual * 0.6, (kl_manual, kl_learned)
    assert learner.stats["discarded_sightings"] >= 0.9 * len(train.false_matches), "look-alikes should be discarded"
    assert learner.stats["multihop_transitions"] > 0
    raw = TransitionLearner(g, track_cleaning="none").fit(train.sighting_log())
    assert raw.stats["rejected_teleports"] > 0, "without cleaning, look-alikes surface as teleports"
    assert transition_kl(ref, {c: raw.transition_probabilities(c) for c in g.cameras}) > kl_learned

    model_json = learner.to_json()
    g2 = VenueTopologyGraph.default_venue()
    g2.load_learned_model(model_json)
    assert g2.has_learned_model
    for c in g2.cameras:
        assert abs(sum(g2.get_outgoing_distribution(c).values()) - 1.0) < 1e-9
    d = g2.get_travel_time_distribution("CAM-01", "CAM-03")
    assert d is not None and d.source == "learned-edge"
    assert g2.get_travel_time_distribution("CAM-01", "CAM-04") is not None
    assert math.isclose(g2.get_transition_probability("CAM-01", "CAM-03"),
                        learned["CAM-01"]["CAM-03"], rel_tol=1e-9)
    g3 = VenueTopologyGraph.from_json(g2.to_json())
    assert g3.get_travel_time_distribution("CAM-01", "CAM-03") is not None
    print(f"  [PASS] cold start = priors (alpha=0); KL manual {kl_manual:.3f} -> learned {kl_learned:.3f}")


# Stated tolerance for convergence to the simulator's true edge probabilities, with
# 1500 walkers and 10% injected look-alikes. The worst-observed node here (CAM-07, ~290
# transitions at p ~ 0.5) has binomial s.e. ~0.03, so 0.06 is about 2 s.e. on the worst edge.
LEARNER_MAX_ABS_TOL = 0.06
LEARNER_MEAN_ABS_TOL = 0.025


def test_learner_converges_to_true_probabilities():
    from topology.simulator import SimulatorConfig
    g = VenueTopologyGraph.default_venue()
    truth, true_probs = perturbed_truth_venue(g, seed=3)
    cfg = SimulatorConfig(backtrack_factor=1.0)  # pure first-order routing: true_probs ARE the edge probabilities
    big = TrajectorySimulator(truth, cfg, transition_probs=true_probs, seed=21).simulate(1500, p_false_match=0.1)

    def errors(n_people):
        keep = set(list(big.trajectories)[:n_people])
        L = TransitionLearner(g).fit(s for s in big.sighting_log() if s.person_id in keep)
        return [abs(L.transition_probabilities(c)[v] - p) for c in true_probs for v, p in true_probs[c].items()]

    manual = [abs(g.get_transition_probability(c, v, "prior") - p) for c in true_probs for v, p in true_probs[c].items()]
    e100, e1500 = errors(100), errors(1500)
    mean = lambda xs: sum(xs) / len(xs)
    assert max(e1500) <= LEARNER_MAX_ABS_TOL, max(e1500)
    assert mean(e1500) <= LEARNER_MEAN_ABS_TOL, mean(e1500)
    assert mean(e1500) < mean(e100) < mean(manual), "error must shrink as data arrives"
    print(f"  [PASS] |P_learned - P_true|: mean {mean(manual):.3f} (manual) -> {mean(e100):.3f} (100 walkers) "
          f"-> {mean(e1500):.3f} (1500), max {max(e1500):.3f} <= tol {LEARNER_MAX_ABS_TOL}")


# -------------------------------------------------------------------------- pruner
def test_pruner_rules():
    g = VenueTopologyGraph.default_venue()
    p = SpatiotemporalPruner(g)
    lo, hi = g.get_transit_window("CAM-01", "CAM-03")

    r = p.score_pair("CAM-01", 0, "CAM-03", lo * 0.5)
    assert not r.is_valid and r.score == 0.0 and r.reason == "teleportation"
    r = p.score_pair("CAM-01", 0, "CAM-03", (lo + hi) / 2)
    assert r.is_valid and r.score == 1.0 and r.reason == "manual"
    r = p.score_pair("CAM-05", 10, "CAM-05", 12)
    assert r.is_valid and r.score == 1.0
    assert not p.score_pair("CAM-05", 10, "CAM-05", 5).is_valid

    r2 = p.score_pair("CAM-01", 0, "CAM-03", 2 * hi)
    assert r2.is_valid and math.isclose(r2.score, math.exp(-1.0))
    assert p.score_pair("CAM-01", 0, "CAM-03", 3 * hi).score < r2.score
    harsh = SpatiotemporalPruner(g, decay_lambda=3.0).score_pair("CAM-01", 0, "CAM-03", 2 * hi)
    assert harsh.score < r2.score

    w14 = g.get_transit_window("CAM-01", "CAM-04")
    mid_edge = (lo + w14[0]) / 2  # enough for one hop, too fast for the two-hop path
    assert p.score_pair("CAM-01", 0, "CAM-04", mid_edge).reason == "teleportation"
    assert p.score_pair("CAM-01", 0, "CAM-04", w14[0] + 5).is_valid

    iso = VenueTopologyGraph()
    iso.add_camera("A", "A", "z", (0, 0))
    iso.add_camera("B", "B", "z", (10, 0))
    assert SpatiotemporalPruner(iso).score_pair("A", 0, "B", 1000).reason == "no_path"

    cands = [cand("c1", "CAM-03", 5), cand("c2", "CAM-03", 60), cand("c3", "CAM-07", 8)]
    out = {c.candidate_id: c for c in p.prune_against_anchor(cands, "CAM-01", ts(0))}
    assert not out["c1"].is_spatially_valid and out["c1"].topology_score == 0.0
    assert out["c2"].is_spatially_valid and out["c2"].topology_score == 1.0
    assert not out["c3"].is_spatially_valid
    print(f"  [PASS] teleport (dt<{lo:.1f}s) rejected, realistic transit accepted, decay e^-1 at 2*dt_max")


def test_pruner_learned_scores():
    g = VenueTopologyGraph.default_venue()
    res = TrajectorySimulator(g, seed=4).simulate(300)
    g.load_learned_model(TransitionLearner(g).fit(res.sighting_log()).export())
    p = SpatiotemporalPruner(g)
    d = g.get_travel_time_distribution("CAM-01", "CAM-03")
    at_mode = p.score_pair("CAM-01", 0, "CAM-03", d.mode)
    assert at_mode.reason == "learned" and at_mode.score > 0.99
    far = p.score_pair("CAM-01", 0, "CAM-03", 6 * d.mode)
    assert far.is_valid and far.score < 0.05
    assert p.score_pair("CAM-01", 0, "CAM-03", 1.0).score == 0.0
    manual = SpatiotemporalPruner(g, use_learned=False).score_pair("CAM-01", 0, "CAM-03", d.mode)
    assert manual.reason == "manual"
    print(f"  [PASS] learned likelihood: mode {d.mode:.1f}s -> 1.0, 6x mode -> {far.score:.3f}")


# ---------------------------------------------------------------------- forecaster
def test_forecaster():
    g = VenueTopologyGraph.default_venue()
    f = BlindSpotForecaster(g, backtrack_penalty=0.3)
    pred = f.forecast("CAM-03", ts(0))
    assert isinstance(pred, SpatialPrediction)
    assert set(pred.candidate_next_cameras) == set(g.get_adjacent_cameras("CAM-03"))
    assert abs(sum(pred.transition_probabilities) - 1.0) < 1e-9
    assert pred.transition_probabilities == sorted(pred.transition_probabilities, reverse=True)
    assert pred.candidate_next_cameras[0] == "CAM-04"
    windows = [g.get_transit_window("CAM-03", c) for c in pred.candidate_next_cameras]
    assert pred.estimated_arrival_window_seconds == (math.floor(min(w[0] for w in windows)),
                                                     math.ceil(max(w[1] for w in windows)))

    hist = [TrajectoryPoint(camera_id="CAM-01", camera_name="x", timestamp=ts(0), confidence=0.9),
            TrajectoryPoint(camera_id="CAM-03", camera_name="x", timestamp=ts(70), confidence=0.9)]
    no_mom = dict(zip(pred.candidate_next_cameras, pred.transition_probabilities))
    det = f.forecast_detailed("CAM-03", ts(70), hist)
    mom = dict(zip(det.prediction.candidate_next_cameras, det.prediction.transition_probabilities))
    assert det.momentum_applied and det.previous_camera == "CAM-01"
    assert mom["CAM-01"] < no_mom["CAM-01"] * 0.5
    assert abs(sum(mom.values()) - 1.0) < 1e-9
    assert not f.forecast_detailed("CAM-03", ts(70), hist[1:]).momentum_applied
    assert f.forecast("CAM-08").candidate_next_cameras == ["CAM-01"]

    # windows valid for every camera, with manual and with learned models
    learned = VenueTopologyGraph.default_venue()
    learned.load_learned_model(TransitionLearner(learned).fit(
        TrajectorySimulator(learned, seed=12).simulate(150).sighting_log()).export())
    for graph in (g, learned):
        fc = BlindSpotForecaster(graph)
        for c in graph.cameras:
            d = fc.forecast_detailed(c, ts(0), hist)
            lo, hi = d.prediction.estimated_arrival_window_seconds
            assert isinstance(lo, int) and isinstance(hi, int) and 0 <= lo < hi
            assert abs(sum(d.prediction.transition_probabilities) - 1.0) < 1e-9
            assert len(d.prediction.candidate_next_cameras) == len(d.prediction.transition_probabilities)
            for cam, (wlo, whi) in d.per_camera_windows.items():
                assert graph.are_adjacent(c, cam)
                assert wlo >= graph.get_transit_window(c, cam)[0] - 1e-9 and wlo <= whi
                assert lo <= wlo + 1 and hi >= whi - 1
    print(f"  [PASS] forecast CAM-03 -> {pred.candidate_next_cameras[0]} "
          f"(p={pred.transition_probabilities[0]:.2f}), window {pred.estimated_arrival_window_seconds}, "
          f"momentum P(back to CAM-01) {no_mom['CAM-01']:.2f} -> {mom['CAM-01']:.2f}")


# ---------------------------------------------------------------------- trajectory
def test_trajectory_reconstruction():
    g = VenueTopologyGraph.default_venue()
    pr = SpatiotemporalPruner(g)
    cands = [cand(f"a{i}", "CAM-01", i, 0.80) for i in range(4)]
    cands += [cand("fake1", "CAM-07", 2.2, 0.90)]                       # look-alike, same time step
    cands += [cand(f"b{i}", "CAM-03", 70 + i, 0.78) for i in range(3)]
    cands += [cand(f"c{i}", "CAM-04", 140 + i, 0.75) for i in range(2)]  # CAM-03 seen; ok
    cands += [cand(f"d{i}", "CAM-07", 400 + i, 0.75) for i in range(2)]  # long gap
    invalid = cand("bad", "CAM-02", 75, 0.95).model_copy(update={"is_spatially_valid": False})

    for method in ("dp", "greedy"):
        res = TrajectoryReconstructor(g, pr, method=method).reconstruct(cands + [invalid])
        ids = [c.candidate_id for c in res.selected]
        assert "fake1" not in ids and "bad" not in ids, (method, ids)
        assert res.dropped_invalid == 1
        assert res.camera_sequence == ["CAM-01", "CAM-03", "CAM-04", "CAM-07"]
        assert all(isinstance(p, TrajectoryPoint) for p in res.points)
        assert [p.timestamp for p in res.points] == sorted(p.timestamp for p in res.points)
        gaps = [(b.from_camera, b.to_camera, b.reason) for b in res.blind_spots]
        assert ("CAM-04", "CAM-07", "gap_exceeds_transit") in gaps

    skip = [cand(f"x{i}", "CAM-01", i, 0.8) for i in range(3)] + [cand(f"y{i}", "CAM-04", 100 + i, 0.8) for i in range(3)]
    bs = TrajectoryReconstructor(g, pr).reconstruct(skip).blind_spots
    assert bs and bs[0].reason == "non_adjacent" and bs[0].inferred_path == ["CAM-03"]

    # early look-alike: greedy gets derailed, DP recovers the genuine chain
    trap = [cand("early_fake", "CAM-07", 0.0, 0.95)] + [cand(f"g{i}", "CAM-01", 2 + i, 0.8) for i in range(6)]
    dp_ids = [c.candidate_id for c in TrajectoryReconstructor(g, pr, method="dp").reconstruct(trap).selected]
    gr_ids = [c.candidate_id for c in TrajectoryReconstructor(g, pr, method="greedy").reconstruct(trap).selected]
    assert "early_fake" not in dp_ids and len(dp_ids) == 6
    assert gr_ids == ["early_fake"]
    print("  [PASS] invalid dropped, look-alike resolved, blind spots flagged, DP beats greedy on early fake")


def test_ground_truth_blind_spot_flagged():
    """A simulated ground-truth trajectory with an unseen camera is rebuilt and the gap flagged."""
    g = VenueTopologyGraph.default_venue()
    res = TrajectorySimulator(g, seed=42).simulate(40, p_false_match=0.1)
    people = res.people_with_blind_spots(g)
    assert people, "simulation should contain non-adjacent blind-spot traversals"
    base = res.base_time.timestamp()
    rec = TrajectoryReconstructor(g)
    checked = 0
    for pid in people[:5]:
        cands, labels = res.candidates_for(pid, graph=g)
        out = rec.reconstruct(cands)
        assert all(labels[c.candidate_id] for c in out.selected), "look-alike leaked into trajectory"
        for a, hidden, b in res.hidden_gaps(pid):
            if a.camera_id == b.camera_id or g.are_adjacent(a.camera_id, b.camera_id):
                continue
            t_hidden = base + hidden[0].arrival_time
            flags = [f for f in out.blind_spots
                     if to_s(f.start_timestamp) <= t_hidden <= to_s(f.end_timestamp)]
            assert flags, f"{pid}: hidden visit to {hidden[0].camera_id} not flagged"
            f = flags[0]
            assert (f.from_camera, f.to_camera, f.reason) == (a.camera_id, b.camera_id, "non_adjacent")
            route = [f.from_camera] + f.inferred_path + [f.to_camera]
            assert all(g.are_adjacent(u, v) for u, v in zip(route[:-1], route[1:]))
            checked += 1
    assert checked > 0
    print(f"  [PASS] {checked} ground-truth blind-spot gaps reconstructed and flagged (non_adjacent)")


def test_most_likely_path():
    g = VenueTopologyGraph.default_venue()
    # equal-length routes CAM-01->CAM-05: via CAM-02 (0.25*0.40) beats via CAM-03 (0.45*0.20)
    assert g.get_most_likely_path("CAM-01", "CAM-05") == ["CAM-01", "CAM-02", "CAM-05"]
    g.set_priors("CAM-03", {"CAM-01": 0.1, "CAM-02": 0.1, "CAM-04": 0.1, "CAM-05": 0.6, "CAM-06": 0.1})
    assert g.get_most_likely_path("CAM-01", "CAM-05") == ["CAM-01", "CAM-03", "CAM-05"]
    assert g.get_most_likely_path("CAM-01", "CAM-03") == ["CAM-01", "CAM-03"]
    print("  [PASS] blind-spot route inference follows transition probabilities")


def test_visualizer_camera_graph():
    import os
    from topology.visualizer import demo_camera_graph
    path = demo_camera_graph(os.path.join("eval", "plots", "p3_camera_graph.png"))
    assert os.path.exists(path) and os.path.getsize(path) > 10_000
    print(f"  [PASS] saved {path}")


def test_stubs_match_schema():
    from contract.schema import SearchCaseResult
    from topology.stubs import build_stubs
    s = build_stubs()
    nodes = [CameraNode(**n) for n in s["camera_nodes"]]
    assert len(nodes) == 8 and all(n.transit_bounds for n in nodes)
    cands = [Candidate(**c) for c in s["candidates_pruned"]]
    assert any(not c.is_spatially_valid for c in cands) and all(c.topology_score is not None for c in cands)
    pts = [TrajectoryPoint(**p) for p in s["trajectory"]]
    pred = SpatialPrediction(**s["spatial_prediction"])
    case = SearchCaseResult(**s["search_case_result"])
    assert case.predicted_next_zone == pred and case.reconstructed_trajectory == pts
    assert abs(sum(pred.transition_probabilities) - 1.0) < 1e-9
    VenueTopologyGraph.from_dict(s["venue_graph"])
    print(f"  [PASS] stubs valid: {len(nodes)} CameraNodes, {len(cands)} Candidates, {len(pts)} TrajectoryPoints")


# ---------------------------------------------------------------------- end to end
def test_end_to_end():
    g = VenueTopologyGraph.default_venue()
    truth, probs = perturbed_truth_venue(g, seed=8)
    train = TrajectorySimulator(truth, transition_probs=probs, seed=31).simulate(300, p_false_match=0.1)
    g.load_learned_model(TransitionLearner(g).fit(train.sighting_log()).export())

    test = TrajectorySimulator(truth, transition_probs=probs, seed=32).simulate(
        60, p_false_match=0.2, target_ids=[f"P{i:04d}" for i in range(10)])
    pr = SpatiotemporalPruner(g)
    rec = TrajectoryReconstructor(g, pr)
    tp = fp = fn = 0
    for i in range(10):
        cands, labels = test.candidates_for(f"P{i:04d}", graph=g)
        res = rec.reconstruct(cands)
        chosen = {c.candidate_id for c in res.selected}
        tp += sum(1 for c in chosen if labels[c])
        fp += sum(1 for c in chosen if not labels[c])
        fn += sum(1 for c, ok in labels.items() if ok and c not in chosen)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    assert precision > 0.95 and recall > 0.9, (precision, recall)
    print(f"  [PASS] simulate -> learn -> prune -> reconstruct: precision {precision:.3f}, recall {recall:.3f}")


TESTS = [
    ("Graph structure", test_graph_structure),
    ("Graph paths & windows", test_graph_paths_and_windows),
    ("Graph serialisation & contract", test_graph_serialisation_and_contract),
    ("Simulator", test_simulator_reproducible_and_physical),
    ("Learner: cold start, robustness, export", test_learner_cold_start_and_learning),
    ("Learner: convergence to true probabilities", test_learner_converges_to_true_probabilities),
    ("Pruner rules", test_pruner_rules),
    ("Pruner learned scores", test_pruner_learned_scores),
    ("Forecaster", test_forecaster),
    ("Trajectory reconstruction", test_trajectory_reconstruction),
    ("Ground-truth blind spot", test_ground_truth_blind_spot_flagged),
    ("Most-likely blind-spot route", test_most_likely_path),
    ("End-to-end", test_end_to_end),
    ("Visualizer: p3_camera_graph.png", test_visualizer_camera_graph),
    ("Stub JSON vs schema", test_stubs_match_schema),
]


def main() -> None:
    print("=" * 60)
    print("  SAFEPASS P3 Topology Pipeline -- Tests")
    print("=" * 60)
    for i, (name, fn) in enumerate(TESTS, 1):
        print(f"\n[{i}/{len(TESTS)}] {name}")
        fn()
    print("\n" + "=" * 60)
    print("  ALL P3 TESTS PASSED [PASS]")
    print("=" * 60)


if __name__ == "__main__":
    main()
