"""
eval/appearance_baseline.py
===========================
Track P2 · Task 7: Ablation Evaluation – Appearance & Fusion Baselines.

Evaluates and compares all fusion strategies on a synthetic or real
paired dataset.  Generates a JSON results report and (optionally)
matplotlib curves.

Fusion strategies evaluated:
    1. face_only       – ArcFace only
    2. app_only        – OSNet appearance only
    3. color_only      – HSV colour only
    4. fixed_equal     – Equal weights (1/3, 1/3, 1/3)
    5. fixed_P2        – Hand-tuned P2 weights (0.4, 0.35, 0.25)
    6. dynamic         – Quality-adaptive rule-based (THIS TRACK)
    7. learned         – MLP learned fusion weights (if model available)

Metrics computed per strategy:
    * AUC-ROC
    * EER (Equal Error Rate)
    * TAR @ FAR=1%
    * TAR @ FAR=0.1%
    * AP (Average Precision)

Usage::

    python -m eval.appearance_baseline \
        --dataset synthetic \
        --n-pairs 2000 \
        --output eval/results_p2_ablation.json

    python -m eval.appearance_baseline \
        --dataset data/pairs.json \
        --output eval/results_p2_ablation.json \
        --plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# NumPy 2.x renamed np.trapz → np.trapezoid
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

# Add project root to sys.path when run as __main__
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.engine import FusionEngine, fuse_scores
from fusion.calibration import ScoreCalibrator


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_roc(scores: np.ndarray, labels: np.ndarray):
    """Return (fpr, tpr, thresholds) arrays."""
    try:
        from sklearn.metrics import roc_curve
        return roc_curve(labels, scores)
    except ImportError:
        # Minimal pure-numpy ROC
        thresholds = np.linspace(0, 1, 200)[::-1]
        fpr_list, tpr_list = [], []
        pos = labels.sum()
        neg = len(labels) - pos
        for t in thresholds:
            tp = ((scores >= t) & (labels == 1)).sum()
            fp = ((scores >= t) & (labels == 0)).sum()
            fpr_list.append(fp / max(neg, 1))
            tpr_list.append(tp / max(pos, 1))
        return np.array(fpr_list), np.array(tpr_list), thresholds


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Trapezoidal AUC."""
    sorted_idx = np.argsort(fpr)
    return float(_trapz(tpr[sorted_idx], fpr[sorted_idx]))


def compute_eer(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Equal Error Rate (FPR = 1 - TPR)."""
    fnr = 1.0 - tpr
    diff = np.abs(fpr - fnr)
    idx = np.argmin(diff)
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_tar_at_far(
    fpr: np.ndarray, tpr: np.ndarray, target_far: float
) -> float:
    """TAR (True Accept Rate) at a specific FAR value."""
    # Find first threshold where FAR <= target_far
    idx = np.searchsorted(fpr, target_far, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def compute_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average Precision (area under precision-recall curve)."""
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(labels, scores))
    except ImportError:
        # Pure-numpy fallback
        sorted_idx = np.argsort(scores)[::-1]
        labels_sorted = labels[sorted_idx].astype(float)
        pos_total = labels.sum()
        if pos_total == 0:
            return 0.0
        tp_cumsum = np.cumsum(labels_sorted)
        precisions = tp_cumsum / np.arange(1, len(labels_sorted) + 1)
        recalls = tp_cumsum / pos_total
        # Only include points where recall increases
        rec_diff = np.diff(np.concatenate([[0], recalls]))
        return float(np.sum(precisions * rec_diff))


def evaluate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    name: str = "model",
) -> Dict[str, float]:
    """Compute full metric suite for a score array."""
    fpr, tpr, _ = compute_roc(scores, labels)
    auc = compute_auc(fpr, tpr)
    eer = compute_eer(fpr, tpr)
    tar_1 = compute_tar_at_far(fpr, tpr, 0.01)
    tar_01 = compute_tar_at_far(fpr, tpr, 0.001)
    ap = compute_ap(scores, labels)

    metrics = {
        "auc": round(auc, 4),
        "eer": round(eer, 4),
        "tar_at_far1pct": round(tar_1, 4),
        "tar_at_far01pct": round(tar_01, 4),
        "average_precision": round(ap, 4),
    }
    print(
        f"  [{name:20s}]  AUC={auc:.4f}  EER={eer:.4f}  "
        f"TAR@FAR1%={tar_1:.4f}  TAR@FAR0.1%={tar_01:.4f}  AP={ap:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------------------

def generate_synthetic_pairs(
    n_pairs: int = 2000,
    genuine_ratio: float = 0.5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic similarity scores for genuine (label=1) and
    impostor (label=0) pairs, with correlated quality variables.

    Returns
    -------
    dict with keys:
        face_sims, app_sims, color_sims   np.ndarray (N,)
        q_face, q_body                    np.ndarray (N,)
        labels                            np.ndarray (N,) int
    """
    rng = np.random.RandomState(seed)
    n_genuine = int(n_pairs * genuine_ratio)
    n_impostor = n_pairs - n_genuine

    def _sim(mu: float, sigma: float, n: int) -> np.ndarray:
        return np.clip(rng.normal(mu, sigma, n), 0.01, 0.99)

    # Genuine pairs: high similarity, variable quality
    g_face = _sim(0.78, 0.10, n_genuine)
    g_app = _sim(0.72, 0.12, n_genuine)
    g_color = _sim(0.68, 0.15, n_genuine)
    g_q_face = np.clip(rng.beta(5, 2, n_genuine), 0.1, 1.0)
    g_q_body = np.clip(rng.beta(4, 2, n_genuine), 0.1, 1.0)

    # Impostor pairs: low similarity
    i_face = _sim(0.45, 0.12, n_impostor)
    i_app = _sim(0.42, 0.13, n_impostor)
    i_color = _sim(0.40, 0.15, n_impostor)
    i_q_face = np.clip(rng.beta(3, 2, n_impostor), 0.1, 1.0)
    i_q_body = np.clip(rng.beta(3, 2, n_impostor), 0.1, 1.0)

    # Simulate quality degradation: randomly zero some modalities
    face_avail = rng.rand(n_pairs) > 0.10   # 10% missing face
    app_avail = rng.rand(n_pairs) > 0.05    # 5% missing appearance

    face_sims = np.concatenate([g_face, i_face])
    app_sims = np.concatenate([g_app, i_app])
    color_sims = np.concatenate([g_color, i_color])
    q_face = np.concatenate([g_q_face, i_q_face])
    q_body = np.concatenate([g_q_body, i_q_body])
    labels = np.array([1] * n_genuine + [0] * n_impostor)

    # Apply availability mask (set missing to median)
    face_sims = np.where(face_avail, face_sims, np.nan)
    app_sims = np.where(app_avail, app_sims, np.nan)

    # Shuffle
    perm = rng.permutation(n_pairs)
    return {
        "face_sims": face_sims[perm],
        "app_sims": app_sims[perm],
        "color_sims": color_sims[perm],
        "q_face": q_face[perm],
        "q_body": q_body[perm],
        "labels": labels[perm],
    }


# ---------------------------------------------------------------------------
# Fusion strategy runner
# ---------------------------------------------------------------------------

STRATEGIES = {
    "face_only": {"mode": "face_only"},
    "app_only": {"mode": "app_only"},
    "color_only": {"mode": "color_only"},
    "fixed_equal": {"mode": "fixed", "fixed_weights": (1/3, 1/3, 1/3)},
    "fixed_P2": {"mode": "fixed", "fixed_weights": (0.40, 0.35, 0.25)},
    "dynamic": {"mode": "dynamic"},
}


def run_strategy(
    name: str,
    config: dict,
    data: Dict[str, np.ndarray],
    calibrate: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Run a single fusion strategy and return (scores, metrics)."""
    engine = FusionEngine(**config)
    n = len(data["labels"])
    scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        fs = data["face_sims"][i]
        as_ = data["app_sims"][i]
        cs = data["color_sims"][i]

        fs = None if (isinstance(fs, float) and np.isnan(fs)) else float(fs)
        as_ = None if (isinstance(as_, float) and np.isnan(as_)) else float(as_)

        result = engine.fuse(
            face_sim=fs,
            app_sim=as_,
            color_sim=float(cs),
            q_face=float(data["q_face"][i]),
            q_body=float(data["q_body"][i]),
        )
        scores[i] = result.fused_score

    labels = data["labels"]

    if calibrate:
        # Split calibration: use first 20% as cal set
        n_cal = max(int(n * 0.20), 10)
        calib = ScoreCalibrator(method="auto", modality=name)
        calib.fit(scores[:n_cal], labels[:n_cal])
        scores = calib.transform(scores)

    metrics = evaluate_scores(scores, labels, name=name)
    return scores, metrics


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------

def plot_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    data_labels: np.ndarray,
    save_path: str = "eval/plots/p2_roc.png",
) -> None:
    """Plot ROC curves for all strategies."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        for (name, scores), color in zip(results.items(), colors):
            fpr, tpr, _ = compute_roc(scores, data_labels)
            auc = compute_auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=1.8)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Track P2 – Fusion Ablation ROC Curves")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Saved ROC to {save_path}")
    except ImportError:
        print("[plot] matplotlib not available. Skipping ROC plot.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ablation(
    n_pairs: int = 2000,
    output_path: str = "eval/results_p2_ablation.json",
    plot: bool = False,
    calibrate: bool = True,
    seed: int = 42,
) -> Dict[str, dict]:
    """
    Execute the full P2 ablation evaluation.

    Returns
    -------
    dict mapping strategy_name → metrics_dict
    """
    print("\n" + "=" * 70)
    print(" SAFEPASS Track P2 – Fusion Ablation Evaluation")
    print("=" * 70)

    print(f"\n[data] Generating {n_pairs} synthetic pairs (seed={seed})...")
    data = generate_synthetic_pairs(n_pairs=n_pairs, seed=seed)
    n_genuine = int(data["labels"].sum())
    print(f"       Genuine: {n_genuine} | Impostor: {n_pairs - n_genuine}")

    print("\n[eval] Running strategies:")
    all_results: Dict[str, dict] = {}
    score_cache: Dict[str, np.ndarray] = {}

    for name, config in STRATEGIES.items():
        try:
            scores, metrics = run_strategy(name, config, data, calibrate=calibrate)
            all_results[name] = metrics
            score_cache[name] = scores
        except Exception as exc:
            print(f"  [{name}] FAILED: {exc}")
            all_results[name] = {"error": str(exc)}

    # Summary table
    print("\n" + "-" * 70)
    print(f"{'Strategy':<22} {'AUC':>7} {'EER':>7} {'TAR@1%':>9} {'AP':>7}")
    print("-" * 70)
    for name, m in all_results.items():
        if "error" not in m:
            print(
                f"{name:<22} {m.get('auc', 0):>7.4f} "
                f"{m.get('eer', 0):>7.4f} "
                f"{m.get('tar_at_far1pct', 0):>9.4f} "
                f"{m.get('average_precision', 0):>7.4f}"
            )
    print("-" * 70)

    # Save results
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_pairs": n_pairs,
        "seed": seed,
        "calibrate": calibrate,
        "strategies": all_results,
    }
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[saved] Results written to {output_path}")

    if plot and score_cache:
        plot_roc_curves(score_cache, data["labels"])

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Track P2 – Fusion Ablation Evaluation")
    parser.add_argument("--n-pairs", type=int, default=2000, help="Number of pairs for synthetic eval")
    parser.add_argument("--output", type=str, default="eval/results_p2_ablation.json")
    parser.add_argument("--plot", action="store_true", help="Save ROC curve plots")
    parser.add_argument("--no-calibrate", action="store_true", help="Disable score calibration")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_ablation(
        n_pairs=args.n_pairs,
        output_path=args.output,
        plot=args.plot,
        calibrate=not args.no_calibrate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
