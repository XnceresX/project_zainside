"""
fusion/calibration.py
=====================
Track P2 · Task 5: Score Calibration.

Raw cosine similarity scores from different modalities (face ArcFace,
OSNet appearance, HSV colour) are not directly comparable or probability-
calibrated.  This module applies post-hoc calibration to map each modality's
raw scores to well-calibrated probabilities in [0, 1].

Two calibration backends:
    1. PlattCalibrator   – Logistic regression (Platt scaling).
                           Effective for small calibration sets (< 500 pairs).
    2. IsotonicCalibrator – Non-parametric isotonic regression.
                            More flexible; needs ≥ 200 samples.

Both implement the same ``fit(scores, labels)`` / ``transform(scores)`` API.

ScoreCalibrator auto-selects backend based on dataset size and wraps both
modalities into a single unified interface.

Public API
----------
    ScoreCalibrator(method="auto")
        .fit(raw_scores, binary_labels)    -> self
        .transform(raw_scores) -> np.ndarray  calibrated probabilities
        .fit_transform(raw_scores, labels) -> np.ndarray
        .save(path) / .load(path)

    calibrate_scores(scores, labels, method="auto") -> np.ndarray  # one-shot

Calibration removes systematic biases between modalities so the weighted
average in the fusion engine is meaningful.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Literal

import numpy as np


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

class PlattCalibrator:
    """
    Logistic regression (Platt scaling) calibration.
    Maps raw similarity scores → probabilities via sigmoid function:
        P(match | s) = 1 / (1 + exp(A*s + B))
    """

    def __init__(self):
        self._A: float = -1.0   # default: approximate sigmoid
        self._B: float = 0.5
        self._fitted: bool = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        """
        Fit logistic regression to (score, label) pairs.

        Parameters
        ----------
        scores : np.ndarray, shape (N,)
            Raw similarity scores in [0, 1].
        labels : np.ndarray, shape (N,)
            Binary ground-truth labels (1 = match, 0 = non-match).
        """
        scores = np.clip(scores, 1e-6, 1.0 - 1e-6).reshape(-1, 1)
        labels = labels.astype(np.float64)

        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1.0, max_iter=500)
            lr.fit(scores, labels)
            self._A = float(lr.coef_[0][0])
            self._B = float(lr.intercept_[0])
            self._fitted = True
        except ImportError:
            # Fallback: closed-form approximation using mean separation
            pos_mean = float(scores[labels == 1].mean()) if (labels == 1).any() else 0.7
            neg_mean = float(scores[labels == 0].mean()) if (labels == 0).any() else 0.3
            mid = (pos_mean + neg_mean) / 2.0
            slope = 8.0 / (pos_mean - neg_mean + 1e-6)
            self._A = slope
            self._B = -slope * mid
            self._fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply sigmoid calibration to raw scores."""
        s = np.asarray(scores, dtype=np.float64)
        return (1.0 / (1.0 + np.exp(-(self._A * s + self._B)))).astype(np.float32)

    def to_dict(self) -> dict:
        return {"type": "platt", "A": self._A, "B": self._B, "fitted": self._fitted}

    def from_dict(self, d: dict) -> "PlattCalibrator":
        self._A = float(d.get("A", -1.0))
        self._B = float(d.get("B", 0.5))
        self._fitted = bool(d.get("fitted", False))
        return self


class IsotonicCalibrator:
    """
    Non-parametric isotonic regression calibration.
    Learns a monotonic mapping from raw scores to probabilities.
    Requires sklearn; falls back to PlattCalibrator if unavailable.
    """

    def __init__(self):
        self._model = None
        self._fitted: bool = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        try:
            from sklearn.isotonic import IsotonicRegression
            self._model = IsotonicRegression(out_of_bounds="clip")
            self._model.fit(scores, labels)
            self._fitted = True
        except ImportError:
            # Fall back to Platt
            self._model = PlattCalibrator().fit(scores, labels)
            self._fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.asarray(scores, dtype=np.float32)
        s = np.asarray(scores, dtype=np.float64)
        # IsotonicRegression uses .predict(); PlattCalibrator uses .transform()
        if hasattr(self._model, "predict"):
            return self._model.predict(s).astype(np.float32)
        else:
            return self._model.transform(s).astype(np.float32)

    def to_dict(self) -> dict:
        return {"type": "isotonic", "fitted": self._fitted}

    def from_dict(self, d: dict) -> "IsotonicCalibrator":
        self._fitted = bool(d.get("fitted", False))
        return self


# ---------------------------------------------------------------------------
# Unified ScoreCalibrator
# ---------------------------------------------------------------------------

class ScoreCalibrator:
    """
    Unified score calibration wrapper supporting Platt and Isotonic backends.

    Parameters
    ----------
    method : "platt" | "isotonic" | "auto"
        Calibration backend.  "auto" picks Platt for N < 200, Isotonic otherwise.
    modality : str
        Label for logging (e.g. "face", "appearance", "color").

    Usage::

        calib = ScoreCalibrator(method="auto", modality="appearance")
        calib.fit(raw_scores_train, labels_train)
        calibrated = calib.transform(raw_scores_test)
    """

    def __init__(
        self,
        method: Literal["platt", "isotonic", "auto"] = "auto",
        modality: str = "generic",
    ):
        self.method = method
        self.modality = modality
        self._calibrator = None
        self._n_train: int = 0

    @property
    def fitted(self) -> bool:
        return self._calibrator is not None and self._calibrator.fitted

    def _select_backend(self, n_samples: int):
        if self.method == "platt":
            return PlattCalibrator()
        elif self.method == "isotonic":
            return IsotonicCalibrator()
        else:  # auto
            # Platt for small sets (fast, stable); Isotonic for >= 200 (more flexible)
            return PlattCalibrator() if n_samples < 200 else IsotonicCalibrator()

    def fit(self, raw_scores: np.ndarray, binary_labels: np.ndarray) -> "ScoreCalibrator":
        """
        Fit calibration model on (score, label) pairs.

        Parameters
        ----------
        raw_scores : np.ndarray, shape (N,)
            Raw cosine-similarity scores from a single modality.
        binary_labels : np.ndarray, shape (N,)
            1 = genuine pair (same person), 0 = impostor pair.
        """
        scores = np.asarray(raw_scores, dtype=np.float64).flatten()
        labels = np.asarray(binary_labels, dtype=np.float64).flatten()
        assert len(scores) == len(labels), "scores and labels must have equal length"

        self._n_train = len(scores)
        self._calibrator = self._select_backend(self._n_train)
        self._calibrator.fit(scores, labels)
        print(
            f"[ScoreCalibrator:{self.modality}] Fitted {type(self._calibrator).__name__} "
            f"on {self._n_train} samples."
        )
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw scores.

        Returns
        -------
        np.ndarray of calibrated probabilities in [0, 1].
        """
        scores = np.asarray(raw_scores, dtype=np.float64)
        if self._calibrator is None:
            # Uncalibrated pass-through
            return np.clip(scores, 0.0, 1.0).astype(np.float32)
        return np.clip(self._calibrator.transform(scores), 0.0, 1.0)

    def fit_transform(
        self, raw_scores: np.ndarray, binary_labels: np.ndarray
    ) -> np.ndarray:
        """Fit and immediately transform the training scores (for cross-val)."""
        self.fit(raw_scores, binary_labels)
        return self.transform(raw_scores)

    def save(self, path: str) -> None:
        """Serialize calibrator parameters to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "method": self.method,
            "modality": self.modality,
            "n_train": self._n_train,
            "calibrator": self._calibrator.to_dict() if self._calibrator else None,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[ScoreCalibrator:{self.modality}] Saved to {path}")

    def load(self, path: str) -> "ScoreCalibrator":
        """Load calibrator parameters from a JSON file."""
        with open(path) as f:
            state = json.load(f)
        self.method = state.get("method", "auto")
        self.modality = state.get("modality", "generic")
        self._n_train = state.get("n_train", 0)
        cal_data = state.get("calibrator")
        if cal_data:
            backend_type = cal_data.get("type", "platt")
            if backend_type == "isotonic":
                self._calibrator = IsotonicCalibrator().from_dict(cal_data)
            else:
                self._calibrator = PlattCalibrator().from_dict(cal_data)
        print(f"[ScoreCalibrator:{self.modality}] Loaded from {path}")
        return self


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def calibrate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    method: str = "auto",
) -> np.ndarray:
    """
    One-shot calibration: fit on (scores, labels), return calibrated scores.

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
    labels : np.ndarray, shape (N,)
    method : str

    Returns
    -------
    np.ndarray of calibrated scores in [0, 1].
    """
    calib = ScoreCalibrator(method=method)
    return calib.fit_transform(scores, labels)
