"""
topology/visualizer.py
======================
Matplotlib plots for the P3 topology module (venue map, learned travel times,
forecasts and reconstructed trajectories). Uses the non-interactive Agg backend
so it works headless; every function saves a PNG and returns its path.

    python -m topology.visualizer     # writes eval/plots/p3_camera_graph.png
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from contract.schema import SpatialPrediction  # noqa: E402
from topology.graph import LogNormalTravelTime, VenueTopologyGraph, to_seconds  # noqa: E402
from topology.trajectory import ReconstructionResult  # noqa: E402

_NODE_COLOR = "#2b6cb0"
_EDGE_COLOR = "#a0aec0"


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_venue(ax, graph: VenueTopologyGraph, source: str = "auto", show_probs: bool = True,
               annotate_distance: bool = False) -> None:
    """Draws cameras at their coordinates; arrow width scales with P(Cj | Ci)."""
    pos = {c: graph.G.nodes[c]["coordinates"] for c in graph.cameras}
    for u, v in graph.G.edges:
        p = graph.get_transition_probability(u, v, source=source)
        (x1, y1), (x2, y2) = pos[u], pos[v]
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                connectionstyle="arc3,rad=0.12", color=_EDGE_COLOR,
                                lw=0.6 + 5.0 * p, alpha=0.85, shrinkA=14, shrinkB=14)
        ax.add_patch(arrow)
        if show_probs:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = y2 - y1, -(x2 - x1)
            n = math.hypot(dx, dy) or 1.0
            ax.text(mx + 6 * dx / n, my + 6 * dy / n, f"{p:.2f}", fontsize=6, color="#4a5568",
                    ha="center", va="center")
        if annotate_distance and u < v:
            d = graph.G.edges[u, v]["distance_meters"]
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{d:.0f} m", fontsize=6, color="#718096",
                    ha="center", va="center", bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
    for c, (x, y) in pos.items():
        ax.scatter([x], [y], s=420, color=_NODE_COLOR, zorder=3, edgecolors="white", linewidths=1.5)
        ax.text(x, y, c.replace("CAM-", ""), color="white", fontsize=8, ha="center", va="center",
                zorder=4, weight="bold")
        ax.text(x, y - 11, graph.camera_name(c), fontsize=6.5, ha="center", va="top", color="#1a202c")
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 30, max(xs) + 30)
    ax.set_ylim(min(ys) - 30, max(ys) + 25)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def plot_camera_graph(
    graph: VenueTopologyGraph,
    path: str = os.path.join("eval", "plots", "p3_camera_graph.png"),
    trajectory: Optional[Union[ReconstructionResult, Sequence[str]]] = None,
    title: Optional[str] = None,
) -> str:
    """
    Camera graph with every corridor labelled [dt_min, dt_max] (seconds, path-based
    physical bounds) and an optional highlighted trajectory. Blind-spot hops between
    non-adjacent cameras are drawn dashed along the inferred shortest path, and the
    unseen intermediate cameras are ringed.
    """
    fig, ax = plt.subplots(figsize=(10, 7.5))
    pos = {c: graph.G.nodes[c]["coordinates"] for c in graph.cameras}

    drawn = set()
    for u, v in graph.G.edges:
        key = tuple(sorted((u, v)))
        if key in drawn:
            continue
        drawn.add(key)
        (x1, y1), (x2, y2) = pos[u], pos[v]
        two_way = graph.G.has_edge(v, u)
        ax.plot([x1, x2], [y1, y2], color=_EDGE_COLOR, lw=2, zorder=1, ls="-" if two_way else ":")
        lo, hi = graph.get_transit_bounds_int(u, v)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"[{lo}, {hi}] s", fontsize=7, ha="center", va="center",
                color="#2d3748", zorder=7, bbox=dict(fc="white", ec="#cbd5e0", alpha=0.85, pad=1.2, boxstyle="round"))

    if trajectory is not None:
        seq = trajectory.camera_sequence if isinstance(trajectory, ReconstructionResult) else collapse_sequence(trajectory)
        for k, (a, b) in enumerate(zip(seq[:-1], seq[1:])):
            blind = not graph.are_adjacent(a, b)
            route = (graph.get_most_likely_path(a, b) or [a, b]) if blind else [a, b]
            col, ls = ("#e53e3e", "--") if blind else ("#dd6b20", "-")
            for s, t in zip(route[:-1], route[1:]):
                ax.annotate("", xy=pos[t], xytext=pos[s], zorder=3,
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=3, ls=ls, shrinkA=15, shrinkB=15,
                                            connectionstyle=f"arc3,rad={0.08 * ((k % 3) - 1)}"))
            for hidden in route[1:-1]:
                ax.scatter(*pos[hidden], s=1100, facecolors="none", edgecolors="#e53e3e", linewidths=2,
                           linestyle="--", zorder=4)
        steps: Dict[str, List[str]] = {}
        for k, cam in enumerate(seq):
            steps.setdefault(cam, []).append(str(k + 1))
        for cam, labels in steps.items():
            x, y = pos[cam]
            ax.text(x + 10, y + 10, ",".join(labels), fontsize=8, color="white", weight="bold", zorder=6,
                    bbox=dict(fc="#dd6b20", ec="none", boxstyle="round", pad=0.25))
        ax.plot([], [], color="#dd6b20", lw=3, label="observed transition")
        ax.plot([], [], color="#e53e3e", lw=3, ls="--", label="blind-spot (inferred route)")
        ax.legend(loc="lower right", fontsize=8)

    for c, (x, y) in pos.items():
        ax.scatter([x], [y], s=520, color=_NODE_COLOR, zorder=5, edgecolors="white", linewidths=1.5)
        ax.text(x, y, c.replace("CAM-", ""), color="white", fontsize=9, ha="center", va="center", zorder=6, weight="bold")
        ax.text(x, y - 12, f"{c}\n{graph.camera_name(c)}", fontsize=7, ha="center", va="top", color="#1a202c", zorder=6)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 35, max(xs) + 35)
    ax.set_ylim(min(ys) - 35, max(ys) + 25)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or f"{graph.venue_name}: corridors labelled [dt_min, dt_max]")
    return _save(fig, path)


def collapse_sequence(cams: Sequence[str]) -> List[str]:
    out: List[str] = []
    for c in cams:
        if not out or out[-1] != c:
            out.append(c)
    return out


def plot_venue_graph(graph: VenueTopologyGraph, path: str, source: str = "auto",
                     title: Optional[str] = None) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_venue(ax, graph, source=source, annotate_distance=True)
    ax.set_title(title or f"{graph.venue_name} - P(Cj|Ci) ({source})")
    return _save(fig, path)


def plot_prior_vs_learned(graph: VenueTopologyGraph, path: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    draw_venue(axes[0], graph, source="prior")
    axes[0].set_title("Manual cold-start priors")
    draw_venue(axes[1], graph, source="auto")
    axes[1].set_title("Learned transition model")
    return _save(fig, path)


def plot_travel_time_fits(graph: VenueTopologyGraph, samples: Dict[Tuple[str, str], Sequence[float]],
                          path: str, edges: Optional[List[Tuple[str, str]]] = None, max_panels: int = 6) -> str:
    """Histogram of observed dt per edge vs learned log-normal and the physical window."""
    keys = edges or sorted(samples, key=lambda k: -len(samples[k]))[:max_panels]
    cols = 3
    rows = max(1, math.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax, (a, b) in zip(axes.flat, keys):
        data = np.asarray(samples.get((a, b), []), dtype=float)
        lo, hi = graph.get_transit_window(a, b)
        if len(data):
            ax.hist(data, bins=20, density=True, color="#90cdf4", edgecolor="white", label=f"observed (n={len(data)})")
        xmax = max(hi * 1.6, float(data.max()) * 1.05 if len(data) else hi)
        xs = np.linspace(0.5, xmax, 300)
        dist = graph.get_travel_time_distribution(a, b)
        if dist is not None:
            ax.plot(xs, [dist.pdf(x) for x in xs], color="#c53030", lw=1.8, label="learned log-normal")
        prior = LogNormalTravelTime.physical_prior(graph.get_path_distance(graph.get_shortest_path(a, b)))
        ax.plot(xs, [prior.pdf(x) for x in xs], color="#718096", lw=1, ls="--", label="physical prior")
        ax.axvspan(lo, hi, color="#68d391", alpha=0.15, label="[dt_min, dt_max]")
        ax.axvline(lo, color="#2f855a", lw=1)
        ax.set_title(f"{a} -> {b}", fontsize=9)
        ax.set_xlabel("dt (s)")
        ax.legend(fontsize=6)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    fig.suptitle("Learned travel-time distributions vs physical bounds")
    fig.tight_layout()
    return _save(fig, path)


def plot_forecast(graph: VenueTopologyGraph, prediction: SpatialPrediction, path: str,
                  previous_camera: Optional[str] = None) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_venue(ax, graph, show_probs=False)
    pos = {c: graph.G.nodes[c]["coordinates"] for c in graph.cameras}
    cur = prediction.current_camera_id
    ax.scatter(*pos[cur], s=900, facecolors="none", edgecolors="#d69e2e", linewidths=3, zorder=5)
    if previous_camera:
        ax.annotate("", xy=pos[cur], xytext=pos[previous_camera],
                    arrowprops=dict(arrowstyle="-|>", color="#4a5568", lw=2, ls="--"))
    for cam, p in zip(prediction.candidate_next_cameras, prediction.transition_probabilities):
        ax.annotate("", xy=pos[cam], xytext=pos[cur],
                    arrowprops=dict(arrowstyle="-|>", color="#dd6b20", lw=1 + 8 * p, alpha=0.8,
                                    shrinkA=14, shrinkB=14))
        x, y = pos[cam]
        ax.text(x, y + 11, f"p={p:.2f}", fontsize=8, ha="center", color="#c05621", weight="bold")
    lo, hi = prediction.estimated_arrival_window_seconds
    ax.set_title(f"Blind-spot forecast from {cur}: arrival in [{lo}, {hi}] s")
    return _save(fig, path)


def demo_camera_graph(path: str = os.path.join("eval", "plots", "p3_camera_graph.png"), seed: int = 42) -> str:
    """Simulates walkers, reconstructs one whose ground truth contains a blind spot, and plots it."""
    from topology.simulator import TrajectorySimulator
    from topology.trajectory import TrajectoryReconstructor

    g = VenueTopologyGraph.default_venue()
    res = TrajectorySimulator(g, seed=seed).simulate(40, p_false_match=0.1)
    pid = res.people_with_blind_spots(g)[0]
    cands, _ = res.candidates_for(pid, graph=g)
    rec = TrajectoryReconstructor(g).reconstruct(cands)
    truth = [h.camera_id for _, hs, _ in res.hidden_gaps(pid) for h in hs]
    inferred = [c for b in rec.blind_spots for c in b.inferred_path]
    return plot_camera_graph(g, path, rec, title=f"Synthetic walker {pid}: reconstructed route\n"
                                                 f"blind-spot cameras inferred: {', '.join(inferred) or '-'} | "
                                                 f"ground truth unseen: {', '.join(truth)}")


def plot_trajectory(graph: VenueTopologyGraph, result: ReconstructionResult, path: str,
                    rejected: Optional[Sequence] = None, title: Optional[str] = None) -> str:
    """Reconstructed route on the venue map; dashed red segments are blind-spot intervals."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.1, 1]})
    draw_venue(ax, graph, show_probs=False)
    pos = {c: graph.G.nodes[c]["coordinates"] for c in graph.cameras}
    seq = result.camera_sequence
    blind = {(b.from_camera, b.to_camera) for b in result.blind_spots if b.reason != "unseen_return"}
    for k, (a, b) in enumerate(zip(seq[:-1], seq[1:])):
        ls, col = ("--", "#e53e3e") if (a, b) in blind else ("-", "#2f855a")
        off = (k % 3 - 1) * 2.5
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.annotate("", xy=(x2 + off, y2 + off), xytext=(x1 + off, y1 + off),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2, ls=ls, shrinkA=12, shrinkB=12))
    ax.set_title(title or "Reconstructed trajectory (dashed = blind-spot interval)")

    if result.points:
        t0 = to_seconds(result.points[0].timestamp)
        cams = graph.cameras
        ax2.scatter([to_seconds(p.timestamp) - t0 for p in result.points],
                    [cams.index(p.camera_id) for p in result.points],
                    c=[p.confidence for p in result.points], cmap="viridis", s=14, label="selected", zorder=3)
        if rejected:
            ax2.scatter([to_seconds(c.timestamp) - t0 for c in rejected],
                        [cams.index(c.camera_id) for c in rejected],
                        marker="x", color="#e53e3e", s=22, label="rejected candidates", zorder=2)
        for b in result.blind_spots:
            ax2.axvspan(to_seconds(b.start_timestamp) - t0, to_seconds(b.end_timestamp) - t0,
                        color="#feb2b2", alpha=0.35)
        ax2.set_yticks(range(len(cams)))
        ax2.set_yticklabels([f"{c} {graph.camera_name(c)}" for c in cams], fontsize=7)
        ax2.set_xlabel("time since first sighting (s)")
        ax2.set_title("Timeline (shaded = blind-spot intervals)")
        ax2.legend(fontsize=7, loc="lower right")
    return _save(fig, path)


if __name__ == "__main__":
    print(f"Saved: {demo_camera_graph()}")
