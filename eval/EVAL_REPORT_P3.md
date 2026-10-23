# SAFEPASS Track P3: Venue Topology, Learned Transition Model, Spatial Pruning & Blind-Spot Forecasting
## Evaluation Report

**Date:** October 2026
**Track:** P3 - Venue Camera Topology (stretch track; does not block P1/P2)
**Modules:** `topology/`, `eval/topology_baseline.py`
**Data:** **all results in this report are on simulated data** from `topology/simulator.py`. No images or real people, no minors' data. The simulator's assumptions are listed in §2 and their consequences in §4.
**Reproduce:** `python -m eval.topology_baseline` (about 30 s, seeded) and `python -m topology.test_pipeline`
**Raw numbers:** `eval/results_topology_baseline.json`. Learned model: `eval/p3_learned_model.json`

---

## 1. Summary

P3 learns the venue's camera topology from sighting logs instead of trusting hand-set values. It then uses that model to reject impossible re-ID matches, rebuild trajectories and forecast where a person will reappear after a blind spot.

| Question | Manual priors / baseline | Learned model (P3) |
|---|---|---|
| Trajectory precision / recall (15% look-alikes) | 0.885 / 0.989 (biometric only) | **1.000 / 1.000** |
| False candidates kept out of the trajectory | 18.6% (biometric only) | **100%** (manual: 99.8%) |
| Operator search space (cameras × time vs watch everything) | -43.2% (manual), misses 40% of next sightings | **-27.9% of cells, 86.7% of next sightings inside it** |
| Transition model error, weighted KL(true ‖ model) | 0.193 (uniform: 0.138) | **0.0055** (400 walkers) |
| Travel-time window coverage on held-out transits | 67.4% (physical `[d/4.5, d/0.9]`) | **95.0%** (nominal 90%) |
| Genuine vs look-alike ROC-AUC, cross-camera | 0.686 (biometric only) | 0.882 (× manual topology), **0.926** (× learned) |
| Look-alikes rejected outright as teleports | - | **72.2%**, with **0.0%** of genuine sightings rejected |
| Trajectory F1 (sighting-level) | 0.934 (biometric only), 0.743 (greedy) | **1.000** (DP); **0.981** under the 50%, 1-hop stress test |
| Blind-spot flag precision / recall | 0.26 / 1.00 (physical windows) | **0.89 / 0.80** |
| Next-camera top-1 / top-3 after a blind spot | 0.348 / 0.801 (manual priors) | **0.565 / 0.954** (learned + momentum) |
| Latency | - | 3.5 µs per `score_pair`, 0.10 ms per `forecast` |

The main finding: **topology alone turns a weak biometric separator (AUC 0.69) into a strong one (0.93), and the transition model can be learned from noisy re-ID logs without being corrupted by look-alikes.**

---

## 2. Setup: avoiding circular evaluation

The system only sees the **manual** venue (`VenueTopologyGraph.default_venue()`, 8 cameras, 12 two-way corridors). Ground truth comes from a **hidden true venue** made by `perturbed_truth_venue(seed=7)`:

- True `P(Cj|Ci)` ~ Dirichlet(8 × manual prior). The manual priors are a plausible guess that is wrong in specifics.
- True corridor lengths are the manual lengths × U(1.0, 1.3), for obstacles, queues and detours. Corridors are only ever stretched, so the manual physical minimum stays a valid hard bound.

How walkers behave in the simulator:
- Speed ~ N(1.3, 0.3) m/s, clipped to [0.6, 2.4], plus ±10% noise per corridor.
- Dwell U(2, 8) s at each camera, with a 10% chance of loitering ~Exp(30 s).
- A 20% chance of stopping in a corridor ~Exp(20 s).
- Backtrack aversion of 0.35.
- 15% of camera visits are fully unobserved (blind-spot traversals), and 80% per-frame detection at 1 Hz otherwise.

Look-alikes: with probability *p*, each genuine target sighting spawns a false match. It is a real sighting of a *different* walker at a camera at least 2 hops away (1 hop in the stress test), within ±30 s. Look-alike confidence is ~N(0.72, 0.08) against ~N(0.78, 0.08) for genuine sightings, so biometrics alone separate them poorly.

| Split | Seed | Walkers | Look-alike rate |
|---|---|---|---|
| Train (learner input) | 101 | 400 | 0.10 (logs are noisy, as real re-ID output would be) |
| Test | 202 | 250 (60 targets) | 0.15 |
| Stress test | 404 | 125 (30 targets) | 0.15 / 0.30 / 0.50 at ≥2 hops and ≥1 hop |
| Reference transition frequencies | 303 | 3000 | - |

The reference `P(Cj|Ci)` is the *empirical* first-order transition frequency of the true walkers, not the Dirichlet parameters, because backtrack aversion changes the effective first-order statistics.

---

## 3. Results

### E0: Task 7 ablation - no topology vs manual priors vs learned priors

Simulated data with injected false matches (60 targets, 15% look-alike rate). Per target, the operator has the confirmed trajectory up to the second-to-last camera visit and must decide where to search next.

| | Precision | Recall | False-candidate rejection | Next-camera top-1 | Next-camera top-3 | Search-space reduction | Next sighting inside searched space |
|---|---|---|---|---|---|---|---|
| (a) No topology (biometric only) | 0.885 | 0.989 | 0.186 | 0.125* | 0.375* | 0% (all 8 cameras) | 1.000 by definition |
| (b) Manual priors | 1.000 | 1.000 | 0.998 | 0.217 | 0.650 | 56.8% | **0.600** |
| (c) Learned priors | 1.000 | 1.000 | **1.000** | **0.417** | **0.850** | 27.9% of cells kept | **0.867** |

*Expected values of a random pick over all 8 cameras.

Reading the table:

- **Without topology, 81% of look-alike sightings end up in the operator's timeline**, and the next camera to watch is a coin toss across the whole venue.
- **Both topology conditions reconstruct the trajectory perfectly and reject ≥99.8% of the false candidates.** The gain of the pipeline is dominated by topology structure (physical bounds + DP chain), not by whether the priors are manual or learned.
- **Where learned priors matter: knowing *where* and *when* to look.** With manual priors, pruning to the top-mass cameras cuts the searched space by 57%, but 40% of the next true sightings fall outside it, because the manual arrival window `[d/4.5, d/0.9]` covers only 67% of real transits (E2) and the priors rank the next camera barely better than chance (top-1 0.217 vs 0.125). The learned model cuts the space to 28% of cells while keeping 86.7% of next sightings inside it.
- The learned condition's reduction comes almost entirely from **camera pruning** (40.6% of cameras on average): its union arrival window (mean 188 s over the top-mass cameras) is often wider than the remaining search period (mean 129 s), so time pruning adds little until the window is refined per camera (per-true-camera windows are 100 s wide at 94.6% coverage, E5).
- Search-space accounting: cells = (fraction of cameras searched) × (window width / remaining search period), capped at 1; "next sighting inside" = the next ground-truth observed visit starts in a searched camera within the searched window.

### E1: Learning `P(Cj|Ci)` (`plots/p3_learning_curve.png`, `plots/p3_venue_graph.png`)

| Training walkers | 0 (cold) | 10 | 25 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|---|---|
| Learned, noisy logs + chain cleaning | 0.163 | 0.079 | 0.044 | 0.022 | 0.013 | 0.009 | **0.0055** |
| Learned, noisy logs, no cleaning | 0.163 | 0.088 | 0.072 | 0.050 | 0.036 | 0.028 | 0.023 |
| Learned, clean logs (upper bound) | 0.163 | 0.079 | 0.044 | 0.022 | 0.013 | 0.009 | 0.0053 |
| Manual priors (constant) | 0.193 | | | | | | |

- About 10 walkers' worth of logs already halves the error of the manual priors.
- **Chain cleaning** reduces each person's log to its topology-consistent dynamic-programming chain, using physical bounds only (not the priors being learned). It matches the clean-log upper bound: it discarded 2,080 sightings, which is about the number of injected look-alikes. Without cleaning, look-alikes leave 4× more error at 400 walkers.
- With the default Laplace α = 1, the cold start is the prior blended toward uniform (0.163 here). With α = 0 it equals the manual priors exactly, which is tested.
- Convergence to the simulator's *true* edge probabilities is a unit test with a stated tolerance: with 1,500 walkers (10% look-alikes) and first-order routing, mean |P_learned − P_true| falls 0.096 (manual) → 0.017 and max ≤ 0.06 (about two standard errors of the worst-observed node), monotonically in the number of walkers.

### E2: Travel-time model (`plots/p3_travel_time_fits.png`)

Held-out test set: 1,554 observed adjacent transits, with Δt measured from the last sighting at Ci to the first sighting at Cj.

| Window | Coverage | Mean width |
|---|---|---|
| Physical `[d/v_max, d/v_min]` | 67.4% | 76 s |
| Learned 90% log-normal interval | 95.0% | 100 s |

- The manual `dt_max = d / 0.9 m/s` is **too tight**: corridor stops, slow walkers and longer-than-surveyed corridors push about a third of real transits past it. That is why the pruner only *decays* the score beyond `dt_max` and never rejects.
- `dt_min` held as a hard bound: the fastest genuine transit was 1.89× the physical minimum, and genuine sightings had a 0% false-rejection rate.
- The log-normal fit is unimodal, while real Δt is a walk mode plus a stop tail (visible on CAM-01→CAM-03). A two-component mixture is the obvious next step.

### E3: Spatiotemporal pruner (`plots/p3_pruner_roc.png`)

There were 432 genuine and 528 look-alike cross-camera candidates. Each was scored against the target's latest earlier genuine sighting.

| | Topology AUC | Biometric × topology AUC | Look-alikes rejected (invalid) | Genuine rejected |
|---|---|---|---|---|
| Biometric only | - | 0.686 | - | - |
| Manual fallback (uniform window + exp. decay, λ = 1) | 0.833 | 0.882 | 72.2% | **0.0%** |
| Learned log-normal | **0.910** | **0.926** | 72.2% | **0.0%** |

- Hard rejection is identical for both, because the teleportation rule uses the same physical `dt_min`.
- The learned model wins on the 28% of look-alikes that are physically *possible*. It ranks them by how *likely* the elapsed time is: mean score 0.13 for look-alikes against 0.74 for genuine.

### E4: Trajectory reconstruction (`plots/p3_trajectory_example.png`, `plots/p3_reconstruction_forecast.png`)

60 targets at a 15% look-alike rate.

| Method | Precision | Recall | F1 | Camera-sequence error |
|---|---|---|---|---|
| Biometric only (best score per 1 s step) | 0.885 | 0.989 | 0.934 | 0.984 |
| Greedy (spec rule, rescored against previous point) | 0.906 | 0.629 | 0.743 | 0.30 |
| **DP chain, manual** | 1.000 | 1.000 | 1.000 | 0.002 |
| **DP chain, learned** | 1.000 | 1.000 | 1.000 | 0.004 |

Greedy gets **derailed**: once it accepts one plausible look-alike, the genuine sightings that follow look like teleports, so it drops them. There is a unit test for this. The DP (`select_consistent_chain`) is the default: at most one candidate per time step, maximising Σ biometric × pairwise topology.

**Stress test (F1):**

| Look-alike distance | p | Biometric | Greedy | DP manual | DP learned |
|---|---|---|---|---|---|
| ≥2 hops | 0.15 | 0.932 | 0.789 | 1.000 | 1.000 |
| ≥2 hops | 0.30 | 0.883 | 0.600 | 0.998 | 0.999 |
| ≥2 hops | 0.50 | 0.821 | 0.457 | 0.997 | 1.000 |
| ≥1 hop (adjacent) | 0.15 | 0.937 | 0.801 | 0.991 | 0.996 |
| ≥1 hop (adjacent) | 0.30 | 0.880 | 0.720 | 0.983 | 0.993 |
| ≥1 hop (adjacent) | 0.50 | 0.816 | 0.515 | 0.968 | **0.981** |

The learned model matters most when look-alikes are at adjacent cameras, where hard physical bounds rarely rule them out.

**Blind-spot flagging.** Ground truth here is a hidden camera visit between two observed ones.

| | Recall | Precision | Non-adjacent flag precision | Gap flag precision |
|---|---|---|---|---|
| DP + physical windows | 1.000 | 0.257 | 0.923 | 0.103 (165 flags) |
| DP + learned 98% windows | 0.797 | **0.889** | **1.000** | 0.556 (9 flags) |

Because the physical `dt_max` is too tight (E2), "gap exceeds transit" fires on almost every slow walk. Learned windows remove most of those false flags. Recall drops on detours through a hidden camera whose timing still fits a direct transit, e.g. CAM-01→(CAM-03)→CAM-02 against CAM-01→CAM-02. `blind_spot_coverage` trades precision against recall.

### E5: Blind-spot forecaster (1,833 events)

| Configuration | Top-1 | Top-2 | Top-3 | NLL | Coverage of arrival window for the true camera | Width |
|---|---|---|---|---|---|---|
| Uniform over neighbours | 0.339 | 0.642 | 0.880 | 1.156 | 75.5% | 89 s |
| Manual priors | 0.348 | 0.607 | 0.801 | 1.209 | 67.3% | 76 s |
| Manual + momentum | 0.406 | 0.684 | 0.839 | 1.161 | 67.3% | 76 s |
| Learned | 0.529 | 0.810 | 0.954 | 1.027 | 94.6% | 100 s |
| **Learned + momentum** | **0.565** | **0.818** | **0.951** | **0.978** | 94.6% | 100 s |

- With these perturbed priors, manual priors are *worse than uniform* on NLL. That is exactly the failure mode hand-set topology risks.
- Momentum (a backtrack penalty of 0.35 on the previous camera) adds about 4–6 points of top-1 accuracy on both bases.

### E6: Latency

`score_pair` takes 3.5 µs with cached path windows and distributions, and `forecast` takes 0.10 ms. Reconstructing a full case trajectory with the DP (114 candidates) takes about 10 ms, which is negligible next to P1/P2 inference.

---

## 4. Limitations

1. **Synthetic only.** Walkers follow a first-order Markov route with backtrack aversion, and the learner and forecaster share that model class. Real crowds have group movement, time-of-day effects and congestion. Validate on P1/P2 output from the demo videos once available.
2. **Momentum penalty equals the simulator's backtrack factor (0.35).** The momentum gain is therefore optimistic. The right fix is to learn second-order `P(Ck | Cj, Ci)` from logs, which `TransitionLearner` can be extended to do, or to tune the penalty on held-out data.
3. **Look-alike model.** Look-alikes are independent per sighting. Real false matches may be *persistent*: the same look-alike matched over a whole visit, so a competing chain could form. The DP's robustness to that is untested.
4. **The physical `dt_max` assumes no stops.** Treat it as a soft bound; the code already does.
5. **Unimodal log-normal.** It misses the stop tail; see E2.
6. **Clocks.** Cameras are assumed time-synchronised. `clock_skew_tolerance_seconds` exists on the pruner but was not evaluated.

---

## 5. Integration notes (P4 / fusion)

```python
from topology import VenueTopologyGraph, SpatiotemporalPruner, TrajectoryReconstructor, BlindSpotForecaster

g = VenueTopologyGraph.default_venue()            # or VenueTopologyGraph.from_json("venue.json")
g.load_learned_model("eval/p3_learned_model.json")  # optional; manual priors are used if absent

pruned = SpatiotemporalPruner(g).prune(candidates, anchor_camera_id=case.last_seen_camera_id,
                                       anchor_timestamp=case.report_timestamp)
rec = TrajectoryReconstructor(g).reconstruct(pruned)       # rec.points -> SearchCaseResult.reconstructed_trajectory
last = rec.points[-1]
pred = BlindSpotForecaster(g).forecast(last.camera_id, last.timestamp, rec.points)  # -> predicted_next_zone
```

- All outputs are frozen-contract types (`Candidate`, `TrajectoryPoint`, `SpatialPrediction`, `CameraNode`). `contract/schema.py` is unchanged.
- `coordinates` and `distance_meters` live only on the NetworkX graph.
- The `estimated_arrival_window_seconds` window is relative to `last_seen_timestamp`.
- `TransitionLearner` accepts simulator `Sighting`s, contract `Candidate`s (keyed by `case_id`), dicts or `(person_id, camera_id, timestamp)` tuples. It can be refit as real logs arrive.
- **Stub JSON for integration is committed in `topology/stubs/`** (`python -m topology.stubs` regenerates): `camera_nodes.json` (List[CameraNode]), `candidates_pruned.json` (List[Candidate]), `trajectory.json` (List[TrajectoryPoint]), `spatial_prediction.json`, `search_case_result.json` (SearchCaseResult combining all of them), `venue_graph.json` (P3-internal graph format), `blind_spots.json` (P3 extension, not in the schema). All content is seeded synthetic data.
- `python -m topology.visualizer` writes `eval/plots/p3_camera_graph.png`: the venue with every corridor labelled `[dt_min, dt_max]` and a reconstructed synthetic trajectory highlighted (dashed red = blind-spot hops along the inferred route, ringed cameras = inferred unseen cameras).
