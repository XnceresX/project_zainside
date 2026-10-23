# Venue Camera Topology & Spatial Pruning (Track P3)

**Owner:** Person 3
**Status:** stretch track. It never blocks P1/P2 integration, and every component falls back to the manual graph.

| File | Purpose |
|---|---|
| `graph.py` | `VenueTopologyGraph`: a NetworkX DiGraph of cameras (`coordinates`) and corridors (`distance_meters`, `min/max_transit_seconds = d/4.5, d/0.9`, `prior_probability`). Transit windows are path-based. Exports JSON and contract `CameraNode`s. |
| `simulator.py` | `TrajectorySimulator`: seeded synthetic walkers, blind-spot traversals and look-alike injection. `perturbed_truth_venue()` builds a hidden ground-truth venue. |
| `learner.py` | `TransitionLearner`: learns `P(Cj\|Ci)` with Laplace/Dirichlet smoothing from the priors, plus log-normal travel times, from sighting logs. Exports JSON that `VenueTopologyGraph.load_learned_model()` can load. |
| `pruner.py` | `SpatiotemporalPruner`: same-camera, no-path and teleportation rules, plus a learned likelihood score (manual fallback: exponential decay beyond `dt_max`, λ = 1). |
| `forecaster.py` | `BlindSpotForecaster`: produces a `SpatialPrediction`, with a momentum (backtrack) penalty and a path-based or learned arrival window. |
| `trajectory.py` | `TrajectoryReconstructor`: selects a DP (or greedy) topology-consistent chain and returns `TrajectoryPoint`s plus blind-spot intervals. |
| `visualizer.py` | Matplotlib plots: venue graph, camera graph with `[dt_min, dt_max]` corridor labels + highlighted trajectory (`python -m topology.visualizer`), travel-time fits, forecast, trajectory. |
| `test_pipeline.py` | `python -m topology.test_pipeline` |
| `stubs.py` | `python -m topology.stubs` regenerates schema-valid stub JSON in `topology/stubs/` for P4/P1/P2 integration. |

Evaluation: `python -m eval.topology_baseline`. See `eval/EVAL_REPORT_P3.md`.
