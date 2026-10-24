"""
fusion/learned.py
=================
Track P2 · Task 6: Learned Quality-Adaptive Weight Predictor (MLP).

Replaces the hand-crafted ``_quality_to_weights()`` thresholds with a
small multi-layer perceptron (MLP) that learns to predict optimal fusion
weights from quality feature vectors.

Architecture
------------
    Input  (7-d): [q_face, q_body, blur_face, blur_body,
                   occ_face, completeness, resolution_body]
    Hidden (32): ReLU
    Hidden (16): ReLU
    Output (3) : Softmax  →  [w_face, w_app, w_color]  (sum to 1.0)

Training objective: minimise mean-squared error between predicted fused
score and true match label, or equivalently, minimise cross-entropy with
soft supervision labels generated from optimal oracle weights on held-out
labelled pairs.

Backends
--------
    * torch.nn.Module   – preferred, GPU compatible, full training loop.
    * numpy fallback    – parameter snapshot loaded from JSON, inference only.

Public API
----------
    LearnedFusion(hidden_sizes=(32, 16), device="cpu")
        .predict_weights(quality_vec) -> (w_face, w_app, w_color)
        .fuse(face_sim, app_sim, color_sim, quality_vec) -> float
        .fit(dataset)           # requires torch
        .save(path) / .load(path)

    LearnedFusionDataset        – helper dataset class for training

The rule-based FusionEngine is always available as a fallback; this module
upgrades the weight predictor when labelled pair data is available.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple, Dict, Any

import numpy as np


# ---------------------------------------------------------------------------
# Quality feature vector builder
# ---------------------------------------------------------------------------

QUALITY_FEATURE_DIM = 7

def build_quality_vector(
    q_face: float = 0.5,
    q_body: float = 0.5,
    blur_face: float = 0.5,
    blur_body: float = 0.5,
    occ_face: float = 0.0,
    completeness_body: float = 0.5,
    resolution_body: float = 0.5,
) -> np.ndarray:
    """
    Build a QUALITY_FEATURE_DIM-dimensional input vector for the MLP.

    Parameters (all in [0, 1])
    ----------
    q_face            : composite face quality (from face/quality.py)
    q_body            : composite body quality (from appearance/quality.py)
    blur_face         : face blur score
    blur_body         : body crop blur score
    occ_face          : face occlusion score  (0=clear, 1=occluded)
    completeness_body : body crop completeness
    resolution_body   : body crop resolution score

    Returns
    -------
    np.ndarray, shape (7,), float32
    """
    return np.array([
        q_face, q_body,
        blur_face, blur_body,
        occ_face, completeness_body, resolution_body,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# NumPy inference-only MLP (no torch dependency)
# ---------------------------------------------------------------------------

class _NumpyMLP:
    """
    Minimal 2-hidden-layer MLP in pure NumPy for inference-only deployments.
    Weights are loaded from a JSON snapshot trained by the torch module.
    """

    def __init__(self):
        self._params: Optional[Dict[str, np.ndarray]] = None

    def load_params(self, params: Dict[str, Any]) -> None:
        """Load weight dict (keys: W1, b1, W2, b2, W3, b3)."""
        self._params = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x (N, 7) → weights (N, 3) via softmax."""
        if self._params is None:
            # Default: uniform weights
            n = x.shape[0] if x.ndim == 2 else 1
            return np.full((n, 3), 1 / 3, dtype=np.float32)

        z = x @ self._params["W1"].T + self._params["b1"]
        z = np.maximum(z, 0)   # ReLU
        z = z @ self._params["W2"].T + self._params["b2"]
        z = np.maximum(z, 0)   # ReLU
        logits = z @ self._params["W3"].T + self._params["b3"]

        # Softmax
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# LearnedFusion
# ---------------------------------------------------------------------------

class LearnedFusion:
    """
    Learned quality-adaptive fusion weight predictor.

    Uses a small MLP to map quality features → (w_face, w_app, w_color).
    Falls back to the rule-based FusionEngine when not fitted.

    Parameters
    ----------
    hidden_sizes : Tuple[int, ...]
        Hidden layer sizes. Default: (32, 16).
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        hidden_sizes: Tuple[int, ...] = (32, 16),
        device: str = "cpu",
    ):
        self.hidden_sizes = hidden_sizes
        self.device = device
        self._torch_model = None
        self._numpy_model = _NumpyMLP()
        self._fitted = False
        self._torch = None

        self._try_build_torch_model()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def uses_torch(self) -> bool:
        return self._torch_model is not None

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _try_build_torch_model(self):
        """Build torch MLP if torch is available."""
        try:
            import torch
            import torch.nn as nn
            self._torch = torch

            layers = []
            in_dim = QUALITY_FEATURE_DIM
            for h in self.hidden_sizes:
                layers += [nn.Linear(in_dim, h), nn.ReLU()]
                in_dim = h
            layers += [nn.Linear(in_dim, 3), nn.Softmax(dim=-1)]

            self._torch_model = nn.Sequential(*layers)
            if self.device == "cuda" and torch.cuda.is_available():
                self._torch_model = self._torch_model.cuda()
            else:
                self.device = "cpu"

        except ImportError:
            self._torch_model = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_weights(
        self,
        quality_vec: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Predict fusion weights from a quality feature vector.

        Parameters
        ----------
        quality_vec : np.ndarray, shape (7,) or (1, 7)

        Returns
        -------
        (w_face, w_app, w_color), each in [0, 1], summing to 1.0.
        """
        x = np.asarray(quality_vec, dtype=np.float32).reshape(1, -1)

        if self._torch_model is not None and self._fitted:
            try:
                import torch
                with torch.no_grad():
                    t = torch.tensor(x, dtype=torch.float32)
                    if self.device == "cuda":
                        t = t.cuda()
                    w = self._torch_model(t).cpu().numpy()[0]
                return float(w[0]), float(w[1]), float(w[2])
            except Exception as exc:
                print(f"[LearnedFusion] Torch inference error: {exc}. Using NumPy fallback.")

        if self._fitted:
            w = self._numpy_model.predict(x)[0]
            return float(w[0]), float(w[1]), float(w[2])

        # Not fitted → default rule-based fallback
        from fusion.engine import _quality_to_weights
        q_face = float(quality_vec[0]) if len(quality_vec) > 0 else 0.5
        q_body = float(quality_vec[1]) if len(quality_vec) > 1 else 0.5
        return _quality_to_weights(q_face, q_body)

    def fuse(
        self,
        face_sim: Optional[float],
        app_sim: Optional[float],
        color_sim: Optional[float],
        quality_vec: np.ndarray,
    ) -> float:
        """
        Predict weights and compute weighted fused score.

        Returns
        -------
        float in [0, 1]
        """
        w_face, w_app, w_color = self.predict_weights(quality_vec)

        score = (
            w_face * (face_sim or 0.0) +
            w_app * (app_sim or 0.0) +
            w_color * (color_sim or 0.0)
        )
        return float(round(min(max(score, 0.0), 1.0), 4))

    # ------------------------------------------------------------------
    # Training (requires torch)
    # ------------------------------------------------------------------

    def fit(
        self,
        quality_vectors: np.ndarray,
        face_sims: np.ndarray,
        app_sims: np.ndarray,
        color_sims: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> Dict[str, List[float]]:
        """
        Train the MLP on labelled pairs.

        The training signal is: given quality, predict weights such that
        the resulting fused score is 1.0 for genuine pairs and 0.0 for
        impostors (MSE loss against binary labels).

        Parameters
        ----------
        quality_vectors : np.ndarray, shape (N, 7)
        face_sims, app_sims, color_sims : np.ndarray, shape (N,)
        labels : np.ndarray, shape (N,) binary 0/1
        epochs : int
        lr : float
        batch_size : int

        Returns
        -------
        dict with "train_loss" history.
        """
        if self._torch_model is None:
            print("[LearnedFusion] torch not available. Training skipped.")
            return {"train_loss": []}

        import torch
        import torch.nn as nn
        import torch.optim as optim

        X = torch.tensor(quality_vectors, dtype=torch.float32)
        F = torch.tensor(face_sims, dtype=torch.float32)
        A = torch.tensor(app_sims, dtype=torch.float32)
        C = torch.tensor(color_sims, dtype=torch.float32)
        Y = torch.tensor(labels, dtype=torch.float32)

        if self.device == "cuda":
            X, F, A, C, Y = X.cuda(), F.cuda(), A.cuda(), C.cuda(), Y.cuda()

        optimizer = optim.Adam(self._torch_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        history = {"train_loss": []}
        n = len(X)

        self._torch_model.train()
        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            steps = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                x_b, f_b, a_b, c_b, y_b = X[idx], F[idx], A[idx], C[idx], Y[idx]

                optimizer.zero_grad()
                weights = self._torch_model(x_b)  # (B, 3)
                fused = (
                    weights[:, 0] * f_b +
                    weights[:, 1] * a_b +
                    weights[:, 2] * c_b
                )
                loss = criterion(fused, y_b)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                steps += 1

            avg_loss = epoch_loss / max(steps, 1)
            history["train_loss"].append(avg_loss)

            if (epoch + 1) % 20 == 0:
                print(f"[LearnedFusion] Epoch {epoch+1}/{epochs}  loss={avg_loss:.5f}")

        self._torch_model.eval()
        self._fitted = True

        # Sync weights to NumPy model for serialisation
        self._sync_to_numpy()
        print("[LearnedFusion] Training complete.")
        return history

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _sync_to_numpy(self):
        """Copy torch weights to the NumPy fallback model."""
        if self._torch_model is None:
            return
        try:
            params = {}
            state = self._torch_model.state_dict()
            linear_idx = 0
            for key, val in state.items():
                if "weight" in key:
                    params[f"W{linear_idx + 1}"] = val.cpu().numpy().tolist()
                elif "bias" in key:
                    params[f"b{linear_idx + 1}"] = val.cpu().numpy().tolist()
                    linear_idx += 1
            self._numpy_model.load_params(params)
        except Exception as exc:
            print(f"[LearnedFusion] NumPy sync failed: {exc}")

    def save(self, path: str) -> None:
        """Save model weights and hyperparameters to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state: Dict[str, Any] = {
            "hidden_sizes": list(self.hidden_sizes),
            "device": self.device,
            "fitted": self._fitted,
            "params": {},
        }
        if self._numpy_model._params is not None:
            state["params"] = {
                k: v.tolist() for k, v in self._numpy_model._params.items()
            }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[LearnedFusion] Saved to {path}")

    def load(self, path: str) -> "LearnedFusion":
        """Load model weights from a JSON snapshot."""
        with open(path) as f:
            state = json.load(f)
        self.hidden_sizes = tuple(state.get("hidden_sizes", (32, 16)))
        self.device = state.get("device", "cpu")
        self._fitted = state.get("fitted", False)
        if state.get("params"):
            self._numpy_model.load_params(state["params"])
        # Rebuild torch model with loaded weights
        self._try_build_torch_model()
        if self._torch_model is not None and state.get("params"):
            try:
                import torch
                params = state["params"]
                linear_layers = [
                    m for m in self._torch_model.modules()
                    if hasattr(m, "weight") and hasattr(m, "bias")
                ]
                for i, layer in enumerate(linear_layers):
                    wk = f"W{i + 1}"
                    bk = f"b{i + 1}"
                    if wk in params and bk in params:
                        layer.weight.data = torch.tensor(params[wk], dtype=torch.float32)
                        layer.bias.data = torch.tensor(params[bk], dtype=torch.float32)
            except Exception as exc:
                print(f"[LearnedFusion] Could not restore torch weights: {exc}")
        print(f"[LearnedFusion] Loaded from {path}")
        return self


# ---------------------------------------------------------------------------
# Training dataset helper
# ---------------------------------------------------------------------------

class LearnedFusionDataset:
    """
    Helper to build a training dataset for the LearnedFusion MLP.

    Each sample is a (quality_vector, face_sim, app_sim, color_sim, label) tuple.
    """

    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    def add(
        self,
        quality_vec: np.ndarray,
        face_sim: float,
        app_sim: float,
        color_sim: float,
        label: int,
    ) -> None:
        """Add a single training record (label=1 genuine, label=0 impostor)."""
        self._records.append({
            "quality_vec": quality_vec.tolist(),
            "face_sim": float(face_sim),
            "app_sim": float(app_sim),
            "color_sim": float(color_sim),
            "label": int(label),
        })

    def to_arrays(self):
        """Return (quality_vectors, face_sims, app_sims, color_sims, labels)."""
        q = np.array([r["quality_vec"] for r in self._records], dtype=np.float32)
        f = np.array([r["face_sim"] for r in self._records], dtype=np.float32)
        a = np.array([r["app_sim"] for r in self._records], dtype=np.float32)
        c = np.array([r["color_sim"] for r in self._records], dtype=np.float32)
        y = np.array([r["label"] for r in self._records], dtype=np.float32)
        return q, f, a, c, y

    def __len__(self) -> int:
        return len(self._records)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self._records, f, indent=2)

    def load(self, path: str) -> "LearnedFusionDataset":
        with open(path) as f:
            self._records = json.load(f)
        return self
