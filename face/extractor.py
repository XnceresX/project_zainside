"""
face/extractor.py
=================
Face embedding extraction and cosine similarity matching (Track P1).
Supports ArcFace and MobileFaceNet ONNX models with automatic
shape handling (both NCHW and NHWC formats) and RGB normalization.
Includes deterministic histogram fallback if ONNX weights are not loaded.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence
import cv2
import numpy as np


DEFAULT_MODEL_PATHS = [
    os.path.join("data", "models", "arcface.onnx"),
    os.path.join("data", "models", "w600k_mbf.onnx"),
    os.path.join("models", "arcface.onnx"),
]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two 1D feature vectors mapped to [0, 1]."""
    if vec1 is None or vec2 is None or len(vec1) == 0 or len(vec2) == 0:
        return 0.0
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm <= 1e-8:
        return 0.0
    sim = dot / norm
    # Normalize from [-1, 1] to [0, 1]
    return float(round((sim + 1.0) / 2.0, 4))


class FaceFeatureExtractor:
    """
    Face feature extractor supporting ArcFace / MobileFaceNet ONNX models.
    Extracts 512-dimensional normalized embeddings from face crops.
    """

    def __init__(self, model_path: Optional[str] = None, use_tta: bool = True):
        self.model_path = model_path
        self.use_tta = use_tta
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.is_nhwc = False
        self.backend = "histogram_fallback"

        resolved_path = self._resolve_model_path(model_path)
        if resolved_path:
            self._load_model(resolved_path)
        else:
            print("[FaceExtractor] WARNING: no ONNX model found; using grayscale-histogram fallback, not face recognition.")

    @property
    def is_loaded(self) -> bool:
        """Returns True if a real ONNX model is loaded and ready."""
        return self.session is not None

    @property
    def status(self) -> dict:
        """Runtime model information suitable for startup logs and health checks."""
        return {
            "ready": self.is_loaded,
            "backend": "onnx" if self.is_loaded else self.backend,
            "model_path": self.model_path if self.is_loaded else None,
            "recognition_capable": self.is_loaded,
            "use_tta": self.use_tta,
        }

    def _resolve_model_path(self, model_path: Optional[str]) -> Optional[str]:
        if model_path and os.path.exists(model_path):
            return model_path
        for candidate in DEFAULT_MODEL_PATHS:
            if os.path.exists(candidate):
                return candidate
        return model_path

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"[FaceExtractor] Model not found at '{model_path}'. Using fallback extractor.")
            return

        try:
            import onnxruntime as ort
            # Use CPU execution provider with low logging
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            self.input_name = inputs[0].name
            self.input_shape = inputs[0].shape
            self.output_name = outputs[0].name

            # Detect layout: [batch, H, W, 3] vs [batch, 3, H, W]
            if len(self.input_shape) == 4 and self.input_shape[-1] == 3:
                self.is_nhwc = True
            else:
                self.is_nhwc = False

            self.model_path = model_path
            self.backend = "onnx"
            print(f"[FaceExtractor] Loaded ONNX model from {model_path} (Layout: {'NHWC' if self.is_nhwc else 'NCHW'}, TTA: {self.use_tta})")
        except Exception as e:
            print(f"[FaceExtractor] Warning: Could not load ONNX model ({e}). Using fallback extractor.")
            self.session = None
        if self.session is None:
            print("[FaceExtractor] WARNING: using grayscale-histogram fallback; this is not face recognition.")

    def _prepare_tensor(self, face_crop: np.ndarray, use_tta: Optional[bool] = None) -> np.ndarray:
        if use_tta is None:
            use_tta = self.use_tta
        resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 2:
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) / 127.5) - 1.0
        if use_tta:
            flipped = cv2.flip(rgb, 1)
            flip_blob = (flipped.astype(np.float32) / 127.5) - 1.0
            if self.is_nhwc:
                return np.stack((blob, flip_blob), axis=0)
            return np.stack((np.transpose(blob, (2, 0, 1)), np.transpose(flip_blob, (2, 0, 1))), axis=0)
        if self.is_nhwc:
            return blob[np.newaxis, ...]
        return np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

    def _fallback_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_crop, (64, 64))
        if len(resized.shape) == 3 and resized.shape[2] >= 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        emb = np.concatenate([hist, hist])
        norm = np.linalg.norm(emb) + 1e-8
        return (emb / norm).astype(np.float32)

    def extract_embeddings(
        self,
        face_crops: Sequence[np.ndarray],
        use_tta: Optional[bool] = None,
    ) -> List[np.ndarray]:
        """Extract multiple embeddings with one ONNX call when possible."""
        if not face_crops:
            return []
        if use_tta is None:
            use_tta = self.use_tta
        valid = [crop is not None and crop.size > 0 for crop in face_crops]
        if self.session is None:
            return [self._fallback_embedding(crop) if ok else np.zeros(512, np.float32)
                    for crop, ok in zip(face_crops, valid)]

        # Build the whole batch together; on fixed-batch ONNX models, fall back
        # to per-crop inference while preserving the same public API.
        try:
            tensors = []
            for crop, ok in zip(face_crops, valid):
                if not ok:
                    continue
                tensors.append(self._prepare_tensor(crop, use_tta=use_tta))
            if tensors:
                batch = np.concatenate(tensors, axis=0)
                raw = self.session.run([self.output_name], {self.input_name: batch})[0]
                raw = np.asarray(raw).reshape((-1, np.asarray(raw).shape[-1]))
                embeddings = []
                cursor = 0
                for ok in valid:
                    if not ok:
                        embeddings.append(np.zeros(512, np.float32))
                        continue
                    if use_tta:
                        emb = raw[cursor] + raw[cursor + 1]
                        cursor += 2
                    else:
                        emb = raw[cursor]
                        cursor += 1
                    embeddings.append((emb / (np.linalg.norm(emb) + 1e-8)).astype(np.float32))
                return embeddings
            return [np.zeros(512, np.float32) for _ in face_crops]
        except Exception as exc:
            print(f"[FaceExtractor] Batched inference unavailable ({exc}); retrying crops individually.")
            result = []
            for crop, ok in zip(face_crops, valid):
                if not ok:
                    result.append(np.zeros(512, np.float32))
                    continue
                try:
                    tensor = self._prepare_tensor(crop, use_tta=use_tta)
                    raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
                    raw = np.asarray(raw).reshape((-1, np.asarray(raw).shape[-1]))
                    emb = raw[0] + raw[1] if use_tta else raw[0]
                    result.append((emb / (np.linalg.norm(emb) + 1e-8)).astype(np.float32))
                except Exception as single_exc:
                    print(f"[FaceExtractor] Inference failed ({single_exc}); returning an invalid embedding for this crop.")
                    result.append(np.zeros(512, np.float32))
            return result

    def extract_embedding(self, face_crop: np.ndarray, use_tta: Optional[bool] = None) -> np.ndarray:
        """
        Extracts 512-dimensional normalized embedding from a face crop.
        Accepts BGR image crop from OpenCV / FaceDetector.
        When use_tta is True, applies flip-averaged Test-Time Augmentation.
        """
        return self.extract_embeddings([face_crop], use_tta=use_tta)[0]

    def compare(self, crop1: np.ndarray, crop2: np.ndarray) -> float:
        """Direct similarity comparison between two face crops."""
        emb1 = self.extract_embedding(crop1)
        emb2 = self.extract_embedding(crop2)
        return cosine_similarity(emb1, emb2)
