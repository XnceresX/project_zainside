"""
eval/topology_baseline.py
=========================
Track P3 evaluation: learned topology vs manual priors on synthetic ground truth.

Ground truth comes from a hidden "true" venue (topology.simulator.perturbed_truth_venue):
transition probabilities drawn around the manual priors and corridors stretched by
obstacles. The system under test only ever sees the manual graph plus sighting logs,
so the evaluation is not circular.

Experiments:
    E1  Learning curve      KL(true || model) of P(Cj|Ci) vs number of training walkers
    E2  Travel-time model   90% interval coverage / width on held-out transitions
    E3  Pruner              genuine vs look-alike discrimination (ROC-AUC), teleport rejection
    E4  Reconstruction      precision / recall of selected sightings, blind-spot flagging
    E5  Forecaster          next-camera top-k, NLL, arrival-window coverage
    E6  Latency             per-call cost of the pruner / forecaster

Run:
    python -m eval.topology_baseline            # full run (~1-2 min)
    python -m eval.topology_baseline --quick    # smaller sizes
Outputs:
    eval/results_topology_baseline.json, eval/p3_learned_model.json, eval/plots/p3_*.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from contract.schema import TrajectoryPoint  # noqa: E402
from topology.forecaster import BlindSpotForecaster  # noqa: E402
from topology.graph import VenueTopologyGraph, to_seconds  # noqa: E402
from topology.learner import TransitionLearner, transition_kl  # noqa: E402
from topology.pruner import SpatiotemporalPruner  # noqa: E402
from topology.simulator import SimulationResult, TrajectorySimulator, perturbed_truth_venue  # noqa: E402
from topology.trajectory import TrajectoryReconstructor  # noqa: E402
from topology import visualizer as viz  # noqa: E402

PLOT_DIR = os.path.join("eval", "plots")
RESULTS_PATH = os.path.join("eval", "results_topology_baseline.json")
MODEL_PATH = os.path.join("eval", "p3_learned_model.json")


# --------------------------------------------------------------------------- helpers
def empirical_transitions(result: SimulationResult) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    for t in result.ground_truth_transitions():
        counts[t["from_camera"]][t["to_camera"]] += 1
    probs = {a: {b: n / sum(c.values()) for b, n in c.items()} for a, c in counts.items()}
    return probs, {a: sum(c.values()) for a, c in counts.items()}


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Mann-Whitney ROC-AUC with ties counted as 0.5."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def roc_curve(pos: Sequence[float], neg: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    thr = np.unique(np.concatenate([pos, neg, [np.inf, -np.inf]]))[::-1]
    pos, neg = np.asarray(pos), np.asarray(neg)
    tpr = np.array([(pos >= t).mean() for t in thr])
    fpr = np.array([(neg >= t).mean() for t in thr])
    return fpr, tpr


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[-1]


def collapse(seq: Sequence[str]) -> List[str]:
    out: List[str] = []
    for c in seq:
        if not out or out[-1] != c:
            out.append(c)
    return out


def learned_graph(manual: VenueTopologyGraph, log) -> Tuple[VenueTopologyGraph, TransitionLearner]:
    learner = TransitionLearner(manual).fit(log)
    g = manual.copy()
    g.load_learned_model(learner.export())
    return g, learner


# ------------------------------------------------------------------------ experiments
def e1_learning_curve(manual, truth, probs, ref, ref_w, sizes, seed) -> Dict:
    manual_dist = {c: manual.get_outgoing_distribution(c, "prior") for c in manual.cameras}
    out = {"sizes": list(sizes), "kl_manual": transition_kl(ref, manual_dist, ref_w),
           "kl_learned": [], "kl_learned_no_cleaning": [], "kl_learned_clean_logs": [],
           "kl_uniform": transition_kl(ref, {c: {v: 1 / len(manual.get_adjacent_cameras(c))
                                                  for v in manual.get_adjacent_cameras(c)}
                                             for c in manual.cameras}, ref_w)}
    big = TrajectorySimulator(truth, transition_probs=probs, seed=seed).simulate(max(sizes), p_false_match=0.1)
    for n in sizes:
        keep = set(list(big.trajectories)[:n])
        noisy = [s for s in big.sighting_log(True) if s.person_id in keep]
        clean = [s for s in big.sighting_log(False) if s.person_id in keep]
        for key, log, kw in (("kl_learned", noisy, {}),
                             ("kl_learned_no_cleaning", noisy, {"track_cleaning": "none"}),
                             ("kl_learned_clean_logs", clean, {})):
            L = TransitionLearner(manual, **kw).fit(log)
            out[key].append(transition_kl(ref, {c: L.transition_probabilities(c) for c in manual.cameras}, ref_w))
    return out


def e2_travel_time(manual, learned, test: SimulationResult) -> Dict:
    """Held-out observed adjacent transitions: dt = first sighting at Cj - last sighting at Ci."""
    rows = []
    for pid, visits in test.trajectories.items():
        sights = [s for s in test.sightings if s.person_id == pid]
        for a, b in zip(visits[:-1], visits[1:]):
            if not (a.observed and b.observed):
                continue
            ta = max(s.timestamp for s in sights if s.camera_id == a.camera_id and a.arrival_time - 1e-6 <= s.timestamp <= a.departure_time + 1e-6)
            tb = min(s.timestamp for s in sights if s.camera_id == b.camera_id and b.arrival_time - 1e-6 <= s.timestamp <= b.departure_time + 1e-6)
            rows.append((a.camera_id, b.camera_id, tb - ta))
    phys_cov, phys_w, l_cov, l_w, nll_l = [], [], [], [], []
    for a, b, dt in rows:
        lo, hi = manual.get_transit_window(a, b)
        phys_cov.append(lo <= dt <= hi)
        phys_w.append(hi - lo)
        lw = learned.get_expected_transit_window(a, b, 0.9)
        l_cov.append(lw[0] <= dt <= lw[1])
        l_w.append(lw[1] - lw[0])
        d = learned.get_travel_time_distribution(a, b)
        nll_l.append(-d.log_pdf(dt) if d else float("nan"))
    return {
        "n_transitions": len(rows),
        "physical_window_coverage": float(np.mean(phys_cov)),
        "physical_window_mean_width_s": float(np.mean(phys_w)),
        "learned_90_interval_coverage": float(np.mean(l_cov)),
        "learned_90_interval_mean_width_s": float(np.mean(l_w)),
        "learned_mean_nll": float(np.nanmean(nll_l)),
        "min_dt_over_physical_min": float(min(dt / manual.get_transit_window(a, b)[0] for a, b, dt in rows)),
    }


def _target_candidates(test: SimulationResult, targets: Sequence[str], graph: VenueTopologyGraph):
    return {t: test.candidates_for(t, graph=graph) for t in targets}


def e3_pruner(manual, learned, cand_sets) -> Dict:
    """Each cross-camera candidate is scored against the target's latest earlier genuine sighting."""
    pr = {"manual": SpatiotemporalPruner(manual, use_learned=False), "learned": SpatiotemporalPruner(learned)}
    scores = {k: {"genuine": [], "lookalike": []} for k in pr}
    bio = {"genuine": [], "lookalike": []}
    valid = {k: {"genuine": [], "lookalike": []} for k in pr}
    reasons = {k: Counter() for k in pr}
    for cands, labels in cand_sets.values():
        cands = sorted(cands, key=lambda c: to_seconds(c.timestamp))
        genuine = [c for c in cands if labels[c.candidate_id]]
        gt = [to_seconds(c.timestamp) for c in genuine]
        for c in cands:
            t = to_seconds(c.timestamp)
            k = int(np.searchsorted(gt, t, side="left")) - 1
            if k < 0 or genuine[k].camera_id == c.camera_id:
                continue
            anchor = genuine[k]
            cls = "genuine" if labels[c.candidate_id] else "lookalike"
            bio[cls].append(c.fused_score)
            for name, p in pr.items():
                ps = p.score_pair(anchor.camera_id, anchor.timestamp, c.camera_id, c.timestamp)
                scores[name][cls].append(ps.score)
                valid[name][cls].append(ps.is_valid)
                reasons[name][(cls, ps.reason.replace("_beyond_max", ""))] += 1
    out = {"n_genuine_pairs": len(bio["genuine"]), "n_lookalike_pairs": len(bio["lookalike"]),
           "auc_biometric_only": auc(bio["genuine"], bio["lookalike"])}
    for name in pr:
        g, l = scores[name]["genuine"], scores[name]["lookalike"]
        comb_g = [a * b for a, b in zip(bio["genuine"], g)]
        comb_l = [a * b for a, b in zip(bio["lookalike"], l)]
        out[name] = {
            "auc_topology": auc(g, l),
            "auc_combined": auc(comb_g, comb_l),
            "lookalike_rejection_rate": 1.0 - float(np.mean(valid[name]["lookalike"])),
            "genuine_false_rejection_rate": 1.0 - float(np.mean(valid[name]["genuine"])),
            "mean_score_genuine": float(np.mean(g)),
            "mean_score_lookalike": float(np.mean(l)),
            "reasons": {f"{c}:{r}": n for (c, r), n in sorted(reasons[name].items())},
        }
    out["_raw"] = {"bio": bio, "scores": scores}
    return out


def _blind_spot_metrics(test: SimulationResult, pid: str, result) -> Counter:
    """
    Ground truth "blind spot" = a hidden (unobserved) camera visit between observed ones.
    A flag is counted correct if a hidden visit starts inside it. Gap flags can also be
    legitimately caused by corridor stops, so their precision is reported separately.
    """
    visits = test.trajectories[pid]
    first_obs = next((i for i, v in enumerate(visits) if v.observed), None)
    last_obs = max((i for i, v in enumerate(visits) if v.observed), default=None)
    hidden = [v for v in visits[first_obs:last_obs + 1] if not v.observed] if first_obs is not None else []
    base = test.base_time.timestamp()
    flags = [(to_seconds(b.start_timestamp), to_seconds(b.end_timestamp), b.reason) for b in result.blind_spots]
    c: Counter = Counter()
    c["hidden_total"] += len(hidden)
    c["hidden_hit"] += sum(1 for v in hidden if any(s <= base + v.arrival_time <= e for s, e, _ in flags))
    for s, e, reason in flags:
        c[f"{reason}_total"] += 1
        c[f"{reason}_good"] += any(s <= base + v.arrival_time <= e for v in hidden)
    return c


def _blind_spot_summary(c: Counter) -> Dict[str, float]:
    total = sum(v for k, v in c.items() if k.endswith("_total") and k != "hidden_total")
    good = sum(v for k, v in c.items() if k.endswith("_good"))
    out = {"blind_spot_recall": c["hidden_hit"] / max(c["hidden_total"], 1),
           "blind_spot_precision": good / max(total, 1)}
    for reason in ("non_adjacent", "gap_exceeds_transit", "unseen_return"):
        if c[f"{reason}_total"]:
            out[f"precision_{reason}"] = c[f"{reason}_good"] / c[f"{reason}_total"]
            out[f"n_{reason}"] = c[f"{reason}_total"]
    return out


def e4_reconstruction(manual, learned, test, cand_sets) -> Dict:
    pm, pl = SpatiotemporalPruner(manual, use_learned=False), SpatiotemporalPruner(learned)
    methods = {
        "biometric_only": None,
        "greedy_manual": TrajectoryReconstructor(manual, pm, method="greedy"),
        "greedy_learned": TrajectoryReconstructor(learned, pl, method="greedy"),
        "dp_manual": TrajectoryReconstructor(manual, pm, method="dp"),
        "dp_learned": TrajectoryReconstructor(learned, pl, method="dp"),
    }
    out = {}
    example = None
    for name, rec in methods.items():
        tp = fp = fn = 0
        seq_err, seq_len = 0, 0
        bs: Counter = Counter()
        for pid, (cands, labels) in cand_sets.items():
            if rec is None:
                steps: Dict[int, object] = {}
                for c in cands:
                    k = int(math.floor(to_seconds(c.timestamp)))
                    if k not in steps or c.fused_score > steps[k].fused_score:
                        steps[k] = c
                chosen = [steps[k] for k in sorted(steps)]
                seq = collapse([c.camera_id for c in chosen])
            else:
                res = rec.reconstruct(cands)
                chosen = res.selected
                seq = res.camera_sequence
                bs.update(_blind_spot_metrics(test, pid, res))
                if name == "dp_learned" and example is None and sum(not v for v in labels.values()) >= 3:
                    example = (pid, res, [c for c in cands if c.candidate_id not in {x.candidate_id for x in chosen}])
            ids = {c.candidate_id for c in chosen}
            tp += sum(1 for c in ids if labels[c])
            fp += sum(1 for c in ids if not labels[c])
            fn += sum(1 for c, ok in labels.items() if ok and c not in ids)
            gt_seq = collapse([s.camera_id for s in test.sightings_for(pid, include_false_matches=False)])
            seq_err += edit_distance(seq, gt_seq)
            seq_len += len(gt_seq)
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[name] = {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
                     "false_points": fp, "camera_sequence_error_rate": seq_err / max(seq_len, 1)}
        if rec is not None:
            out[name].update(_blind_spot_summary(bs))
    return out, example


def e4b_stress(manual, learned, truth, probs, n_test, targets, rates, hop_levels) -> Dict:
    """Reconstruction F1 as look-alikes get more frequent and closer (1 hop = adjacent camera)."""
    rows = []
    for hops in hop_levels:
        for p in rates:
            test = TrajectorySimulator(truth, transition_probs=probs, seed=404).simulate(
                n_test, p_false_match=p, target_ids=targets, min_false_hops=hops)
            res, _ = e4_reconstruction(manual, learned, test, _target_candidates(test, targets, manual))
            rows.append({"min_false_hops": hops, "p_false": p,
                         **{f"{m}_{k}": res[m][k] for m in ("biometric_only", "greedy_learned", "dp_manual", "dp_learned")
                            for k in ("precision", "recall", "f1")}})
    return {"rows": rows}


def e0_ablation(manual, learned, test: SimulationResult, cand_sets) -> Dict:
    """
    Task 7 ablation on simulator data with injected false matches:
        (a) no topology    - biometric confidence only, search every camera
        (b) manual priors  - full P3 pipeline on the hand-set graph
        (c) learned priors - full P3 pipeline on the learned model
    Reported per condition: trajectory precision/recall, false-candidate (look-alike)
    rejection rate, next-camera top-1/top-3, and search-space reduction, i.e. the
    fraction of (camera x time) cells an operator would have to watch, relative to
    watching every camera for the whole remaining search period.
    """
    n_cams = len(manual.cameras)
    pr_m = SpatiotemporalPruner(manual, use_learned=False)
    pr_l = SpatiotemporalPruner(learned)
    conds = {
        "no_topology": None,
        "manual": (TrajectoryReconstructor(manual, pr_m, method="dp"), BlindSpotForecaster(manual)),
        "learned": (TrajectoryReconstructor(learned, pr_l, method="dp"), BlindSpotForecaster(learned)),
    }
    out = {}
    for name, setup in conds.items():
        tp = fp = fn = 0
        n_look = n_look_kept = 0
        cam_frac, win_s, span_s, hit = [], [], [], []
        top1 = top3 = 0
        n_events = 0
        base = test.base_time.timestamp()
        for pid, (cands, labels) in cand_sets.items():
            visits = test.trajectories[pid]
            if setup is None:
                steps: Dict[int, object] = {}
                for c in cands:
                    k = int(math.floor(to_seconds(c.timestamp)))
                    if k not in steps or c.fused_score > steps[k].fused_score:
                        steps[k] = c
                chosen = [steps[k] for k in sorted(steps)]
            else:
                chosen = setup[0].reconstruct(cands).selected
            ids = {c.candidate_id for c in chosen}
            tp += sum(1 for c in ids if labels[c])
            fp += sum(1 for c in ids if not labels[c])
            fn += sum(1 for c, ok in labels.items() if ok and c not in ids)
            n_look += sum(1 for ok in labels.values() if not ok)
            n_look_kept += sum(1 for c in ids if not labels[c])

            # mid-search scenario: the operator has confirmed points up to an anchor that
            # is not in the final camera visit, and must decide where to look next
            chosen_sorted = sorted(chosen, key=lambda c: to_seconds(c.timestamp))
            if not chosen_sorted:
                continue
            groups: List[List] = [[]]
            for c in chosen_sorted:
                if groups[-1] and (c.camera_id != groups[-1][-1].camera_id
                                   or to_seconds(c.timestamp) - to_seconds(groups[-1][-1].timestamp) > 20.0):
                    groups.append([])
                groups[-1].append(c)
            anchor_group = groups[-2] if len(groups) >= 2 else groups[0]
            anchor = anchor_group[-1]
            t_last = to_seconds(anchor.timestamp) - base
            obs = [v for v in visits if v.observed]
            nxt = [v for v in obs if v.arrival_time > t_last + 1e-6]
            span = (visits[-1].departure_time - t_last) if nxt else 0.0
            if not nxt:
                continue
            arrive = nxt[0].arrival_time - t_last
            true_cam = nxt[0].camera_id
            if setup is None:
                cams, w = list(manual.cameras), span          # watch everything until period end
                top1 += 1 / n_cams
                top3 += min(3 / n_cams, 1.0)
                n_events += 1
            else:
                fc = setup[1]
                hist = [c for c in chosen_sorted if to_seconds(c.timestamp) <= to_seconds(anchor.timestamp)]
                pred = fc.forecast(anchor.camera_id, anchor.timestamp, hist)
                cams, w = [], 0.0
                mass = 0.0
                for c, p in zip(pred.candidate_next_cameras, pred.transition_probabilities):
                    cams.append(c)
                    mass += p
                    if mass >= 0.9 - 1e-9:
                        break
                w = pred.estimated_arrival_window_seconds[1]
                top1 += pred.candidate_next_cameras[0] == true_cam
                top3 += true_cam in pred.candidate_next_cameras[:3]
                n_events += 1
            cam_frac.append(len(cams) / n_cams)
            win_s.append(w)
            span_s.append(span)
            hit.append(true_cam in cams and w >= arrive >= 0)
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        searched = [f * (w / max(s, 1e-9)) for f, w, s in zip(cam_frac, win_s, span_s)]
        out[name] = {
            "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
            "false_candidate_rejection_rate": 1 - n_look_kept / max(n_look, 1),
            "next_camera_top1": top1 / max(n_events, 1),
            "next_camera_top3": top3 / max(n_events, 1),
            "searched_cell_fraction": float(np.mean(searched)),
            "search_space_reduction": 1 - float(np.mean(searched)),
            "searched_camera_fraction": float(np.mean(cam_frac)),
            "searched_window_mean_s": float(np.mean(win_s)),
            "search_baseline_span_mean_s": float(np.mean(span_s)),
            "next_sighting_inside_searched_space": float(np.mean(hit)),
        }
    return out


def e5_forecaster(manual, learned, test: SimulationResult) -> Dict:
    configs = {
        "uniform": None,
        "manual": BlindSpotForecaster(manual, backtrack_penalty=1.0, use_learned=False),
        "manual+momentum": BlindSpotForecaster(manual, backtrack_penalty=0.35, use_learned=False),
        "learned": BlindSpotForecaster(learned, backtrack_penalty=1.0),
        "learned+momentum": BlindSpotForecaster(learned, backtrack_penalty=0.35),
    }
    events = []
    for pid, visits in test.trajectories.items():
        sights = [s for s in test.sightings if s.person_id == pid]
        history: List[TrajectoryPoint] = []
        for a, b in zip(visits[:-1], visits[1:]):
            if a.observed:
                last = max(s.timestamp for s in sights if s.camera_id == a.camera_id
                           and a.arrival_time - 1e-6 <= s.timestamp <= a.departure_time + 1e-6)
                history.append(TrajectoryPoint(camera_id=a.camera_id, camera_name=a.camera_id,
                                               timestamp=_iso(test, last), confidence=1.0))
                events.append((a.camera_id, last, list(history[-5:]), b.camera_id, b.arrival_time - last))
    out = {"n_events": len(events)}
    for name, fc in configs.items():
        top1 = top2 = top3 = 0.0
        nll, cov_cam, cov_union, width_cam, width_union = [], [], [], [], []
        for cam, last, hist, true_next, true_dt in events:
            if fc is None:
                succ = manual.get_adjacent_cameras(cam)
                ranked = {c: 1 / len(succ) for c in succ}
                win_c = win_u = (min(manual.get_transit_window(cam, c)[0] for c in succ),
                                 max(manual.get_transit_window(cam, c)[1] for c in succ))
                top1 += 1 / len(succ)            # expected accuracy of a random tie-break
                top2 += min(2 / len(succ), 1.0)
                top3 += min(3 / len(succ), 1.0)
            else:
                det = fc.forecast_detailed(cam, None, hist)
                pred = det.prediction
                ranked = dict(zip(pred.candidate_next_cameras, pred.transition_probabilities))
                win_c = det.per_camera_windows[true_next]
                win_u = pred.estimated_arrival_window_seconds
                top1 += pred.candidate_next_cameras[0] == true_next
                top2 += true_next in pred.candidate_next_cameras[:2]
                top3 += true_next in pred.candidate_next_cameras[:3]
            nll.append(-math.log(max(ranked.get(true_next, 0.0), 1e-12)))
            cov_cam.append(win_c[0] <= true_dt <= win_c[1])
            cov_union.append(win_u[0] <= true_dt <= win_u[1])
            width_cam.append(win_c[1] - win_c[0])
            width_union.append(win_u[1] - win_u[0])
        n = max(len(events), 1)
        out[name] = {"top1": top1 / n, "top2": top2 / n, "top3": top3 / n, "nll": float(np.mean(nll)),
                     "per_camera_window_coverage": float(np.mean(cov_cam)),
                     "per_camera_window_width_s": float(np.mean(width_cam)),
                     "union_window_coverage": float(np.mean(cov_union)),
                     "union_window_width_s": float(np.mean(width_union))}
    return out


def _iso(test: SimulationResult, sec: float) -> str:
    from datetime import timedelta
    return (test.base_time + timedelta(seconds=sec)).isoformat()


def e6_latency(learned) -> Dict:
    pr = SpatiotemporalPruner(learned)
    fc = BlindSpotForecaster(learned)
    cams = learned.cameras
    pairs = [(a, b) for a in cams for b in cams]
    for a, b in pairs:
        pr.score_pair(a, 0.0, b, 60.0)
    n = 20000
    t0 = time.perf_counter()
    for i in range(n):
        a, b = pairs[i % len(pairs)]
        pr.score_pair(a, 0.0, b, 60.0 + (i % 50))
    t1 = time.perf_counter()
    for i in range(2000):
        fc.forecast(cams[i % len(cams)])
    t2 = time.perf_counter()
    return {"pruner_score_pair_us": (t1 - t0) / n * 1e6, "forecast_ms": (t2 - t1) / 2000 * 1e3}


# ------------------------------------------------------------------------------ plots
def plot_learning_curve(e1: Dict, path: str) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    xs = [max(s, 1) for s in e1["sizes"]]
    ax.plot(xs, e1["kl_learned"], "o-", color="#c53030", label="learned (noisy logs, chain cleaning)")
    ax.plot(xs, e1["kl_learned_no_cleaning"], "s--", color="#dd6b20", label="learned (noisy logs, no cleaning)")
    ax.plot(xs, e1["kl_learned_clean_logs"], "^:", color="#2f855a", label="learned (clean logs)")
    ax.axhline(e1["kl_manual"], color="#2b6cb0", ls="-.", label="manual priors")
    ax.axhline(e1["kl_uniform"], color="#718096", ls=":", label="uniform")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["cold\nstart" if s == 0 else str(s) for s in e1["sizes"]], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("training walkers")
    ax.set_ylabel("weighted KL(true || model)")
    ax.set_title("E1: learning P(Cj|Ci) from sighting logs")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_pruner(e3: Dict, path: str) -> str:
    raw = e3["_raw"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    fpr, tpr = roc_curve(raw["bio"]["genuine"], raw["bio"]["lookalike"])
    ax.plot(fpr, tpr, color="#718096", label=f"biometric only (AUC {e3['auc_biometric_only']:.3f})")
    for name, col in (("manual", "#2b6cb0"), ("learned", "#c53030")):
        g = [a * b for a, b in zip(raw["bio"]["genuine"], raw["scores"][name]["genuine"])]
        l = [a * b for a, b in zip(raw["bio"]["lookalike"], raw["scores"][name]["lookalike"])]
        fpr, tpr = roc_curve(g, l)
        ax.plot(fpr, tpr, color=col, label=f"biometric x topology ({name}) AUC {e3[name]['auc_combined']:.3f}")
    ax.plot([0, 1], [0, 1], color="#cbd5e0", ls=":")
    ax.set_xlabel("false positive rate (look-alikes kept)")
    ax.set_ylabel("true positive rate (genuine kept)")
    ax.set_title("E3: genuine vs look-alike cross-camera candidates")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    ax.hist(raw["scores"]["learned"]["genuine"], bins=bins, alpha=0.6, color="#2f855a", label="genuine", density=True)
    ax.hist(raw["scores"]["learned"]["lookalike"], bins=bins, alpha=0.6, color="#e53e3e", label="look-alike", density=True)
    ax.set_xlabel("learned topology_score")
    ax.set_title("Topology score distribution (learned)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_summary(e4: Dict, e5: Dict, path: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    names = list(e4)
    x = np.arange(len(names))
    for k, (metric, col) in enumerate((("precision", "#2b6cb0"), ("recall", "#2f855a"), ("f1", "#c53030"))):
        ax.bar(x + (k - 1) * 0.25, [e4[n][metric] for n in names], 0.25, label=metric, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylim(min(0.5, min(e4[n]["precision"] for n in names) - 0.05), 1.01)
    ax.set_title("E4: trajectory reconstruction (sighting-level)")
    ax.legend(fontsize=8)
    ax = axes[1]
    names = [n for n in e5 if n != "n_events"]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [e5[n]["top1"] for n in names], 0.4, label="top-1", color="#2b6cb0")
    ax.bar(x + 0.2, [e5[n]["top3"] for n in names], 0.4, label="top-3", color="#90cdf4")
    for i, n in enumerate(names):
        ax.text(i, e5[n]["top3"] + 0.02, f"NLL {e5[n]['nll']:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("+", "\n+") for n in names], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("E5: blind-spot next-camera forecasting")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="SAFEPASS P3 topology evaluation")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--truth-seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(PLOT_DIR, exist_ok=True)

    n_train = 200 if args.quick else 400
    n_test = 120 if args.quick else 250
    n_targets = 30 if args.quick else 60
    n_ref = 1000 if args.quick else 3000
    sizes = [0, 10, 25, 50, 100, n_train] if args.quick else [0, 10, 25, 50, 100, 200, 400]
    p_false_train, p_false_test = 0.10, 0.15

    t_start = time.time()
    manual = VenueTopologyGraph.default_venue()
    truth, probs = perturbed_truth_venue(manual, seed=args.truth_seed)
    ref, ref_w = empirical_transitions(TrajectorySimulator(truth, transition_probs=probs, seed=303).simulate(n_ref))

    print("[E1] learning curve ...")
    e1 = e1_learning_curve(manual, truth, probs, ref, ref_w, sizes, seed=101)

    train = TrajectorySimulator(truth, transition_probs=probs, seed=101).simulate(n_train, p_false_match=p_false_train)
    learned, learner = learned_graph(manual, train.sighting_log())
    learner.to_json(MODEL_PATH)

    targets = [f"P{i:04d}" for i in range(n_targets)]
    test = TrajectorySimulator(truth, transition_probs=probs, seed=202).simulate(
        n_test, p_false_match=p_false_test, target_ids=targets)
    cand_sets = _target_candidates(test, targets, manual)

    print("[E0] ablation (no topology / manual / learned) ...")
    e0 = e0_ablation(manual, learned, test, cand_sets)
    print("[E2] travel time ...")
    e2 = e2_travel_time(manual, learned, test)
    print("[E3] pruner ...")
    e3 = e3_pruner(manual, learned, cand_sets)
    print("[E4] reconstruction ...")
    e4, example = e4_reconstruction(manual, learned, test, cand_sets)
    print("[E4b] reconstruction stress sweep ...")
    e4b = e4b_stress(manual, learned, truth, probs, n_test // 2 if not args.quick else 60,
                     targets[: n_targets // 2], rates=[0.15, 0.3, 0.5], hop_levels=[2, 1])
    print("[E5] forecaster ...")
    e5 = e5_forecaster(manual, learned, test)
    e6 = e6_latency(learned)

    print("[plots] ...")
    plots = {
        "venue": viz.plot_prior_vs_learned(learned, os.path.join(PLOT_DIR, "p3_venue_graph.png")),
        "learning_curve": plot_learning_curve(e1, os.path.join(PLOT_DIR, "p3_learning_curve.png")),
        "travel_time": viz.plot_travel_time_fits(
            learned, learner.edge_dt, os.path.join(PLOT_DIR, "p3_travel_time_fits.png"),
            edges=[("CAM-01", "CAM-03"), ("CAM-03", "CAM-04"), ("CAM-02", "CAM-05"),
                   ("CAM-04", "CAM-07"), ("CAM-06", "CAM-01"), ("CAM-05", "CAM-03")]),
        "pruner": plot_pruner(e3, os.path.join(PLOT_DIR, "p3_pruner_roc.png")),
        "summary": plot_summary(e4, e5, os.path.join(PLOT_DIR, "p3_reconstruction_forecast.png")),
    }
    fc = BlindSpotForecaster(learned)
    det = fc.forecast_detailed("CAM-03", None, ["CAM-01", "CAM-03"])
    plots["forecast"] = viz.plot_forecast(learned, det.prediction, os.path.join(PLOT_DIR, "p3_forecast_example.png"),
                                          previous_camera=det.previous_camera)
    if example:
        pid, res, rejected = example
        plots["trajectory"] = viz.plot_trajectory(learned, res, os.path.join(PLOT_DIR, "p3_trajectory_example.png"),
                                                  rejected=rejected, title=f"{pid}: DP reconstruction (learned)")

    e3.pop("_raw")
    results = {
        "config": {"truth_seed": args.truth_seed, "n_train": n_train, "n_test": n_test, "n_targets": n_targets,
                   "n_reference": n_ref, "p_false_train": p_false_train, "p_false_test": p_false_test,
                   "quick": args.quick},
        "learner_stats": learner.stats,
        "e0_ablation": e0,
        "e1_learning_curve": e1,
        "e2_travel_time": e2,
        "e3_pruner": e3,
        "e4_reconstruction": e4,
        "e4b_reconstruction_stress": e4b,
        "e5_forecaster": e5,
        "e6_latency": e6,
        "plots": plots,
        "runtime_seconds": time.time() - t_start,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print("  SAFEPASS P3 TOPOLOGY EVALUATION (SIMULATED DATA ONLY)")
    print("=" * 65)
    hdr = f"{'':<13}{'P':>7}{'R':>7}{'F1':>7}{'rejF':>7}{'top1':>7}{'top3':>7}{'srch-':>8}{'hit':>7}"
    print("ABLATION (a) no topology | (b) manual priors | (c) learned priors")
    print(hdr)
    for k, v in e0.items():
        print(f"{k:<13}{v['precision']:>7.3f}{v['recall']:>7.3f}{v['f1']:>7.3f}"
              f"{v['false_candidate_rejection_rate']:>7.3f}{v['next_camera_top1']:>7.3f}{v['next_camera_top3']:>7.3f}"
              f"{100 * v['search_space_reduction']:>7.1f}%{v['next_sighting_inside_searched_space']:>7.3f}")
    print(f"(rejF = injected false candidates kept out of the trajectory; srch- = mean reduction of camera x time "
          f"cells searched vs watching all {len(manual.cameras)} cameras until period end; "
          f"hit = next true sighting inside that reduced space)")
    print(f"E1 KL manual {e1['kl_manual']:.4f} | learned@{sizes[-1]} {e1['kl_learned'][-1]:.4f} "
          f"| uniform {e1['kl_uniform']:.4f}")
    print(f"E2 physical window cov {e2['physical_window_coverage']:.3f} (w {e2['physical_window_mean_width_s']:.0f}s) "
          f"| learned 90% cov {e2['learned_90_interval_coverage']:.3f} (w {e2['learned_90_interval_mean_width_s']:.0f}s)")
    print(f"E3 AUC bio {e3['auc_biometric_only']:.3f} | combined manual {e3['manual']['auc_combined']:.3f} "
          f"| combined learned {e3['learned']['auc_combined']:.3f} | look-alike rejection "
          f"{e3['learned']['lookalike_rejection_rate']:.3f} | genuine FRR {e3['learned']['genuine_false_rejection_rate']:.3f}")
    for k, v in e4.items():
        bs = (f" | blind-spot R {v['blind_spot_recall']:.3f} P {v['blind_spot_precision']:.3f} "
              f"(non-adj P {v.get('precision_non_adjacent', float('nan')):.3f}, "
              f"gap P {v.get('precision_gap_exceeds_transit', float('nan')):.3f} n={v.get('n_gap_exceeds_transit', 0)})"
              if "blind_spot_recall" in v else "")
        print(f"E4 {k:<15} P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} "
              f"seqErr {v['camera_sequence_error_rate']:.3f}{bs}")
    for r in e4b["rows"]:
        print(f"E4b hops>={r['min_false_hops']} p={r['p_false']:.2f}  F1 bio {r['biometric_only_f1']:.3f} "
              f"greedy {r['greedy_learned_f1']:.3f} dp_manual {r['dp_manual_f1']:.3f} dp_learned {r['dp_learned_f1']:.3f}")
    for k, v in e5.items():
        if k != "n_events":
            print(f"E5 {k:<17} top1 {v['top1']:.3f} top2 {v['top2']:.3f} top3 {v['top3']:.3f} NLL {v['nll']:.3f} "
                  f"camWin cov {v['per_camera_window_coverage']:.3f} w {v['per_camera_window_width_s']:.0f}s")
    print(f"E6 score_pair {e6['pruner_score_pair_us']:.1f} us | forecast {e6['forecast_ms']:.3f} ms")
    print(f"Saved {RESULTS_PATH}, {MODEL_PATH}; runtime {results['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
