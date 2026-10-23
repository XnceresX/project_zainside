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
from typing import Optional
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
    norm = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-8
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

        resolved_path = self._resolve_model_path(model_path)
        if resolved_path:
            self._load_model(resolved_path)

    @property
    def is_loaded(self) -> bool:
        """Returns True if a real ONNX model is loaded and ready."""
        return self.session is not None

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
            print(f"[FaceExtractor] Loaded ONNX model from {model_path} (Layout: {'NHWC' if self.is_nhwc else 'NCHW'}, TTA: {self.use_tta})")
        except Exception as e:
            print(f"[FaceExtractor] Warning: Could not load ONNX model ({e}). Using fallback extractor.")
            self.session = None

    def extract_embedding(self, face_crop: np.ndarray, use_tta: Optional[bool] = None) -> np.ndarray:
        """
        Extracts 512-dimensional normalized embedding from a face crop.
        Accepts BGR image crop from OpenCV / FaceDetector.
        When use_tta is True, applies flip-averaged Test-Time Augmentation.
        """
        if face_crop is None or face_crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        if use_tta is None:
            use_tta = self.use_tta

        if self.session is not None:
            try:
                # Preprocess to 112x112 RGB
                resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_AREA)
                if len(resized.shape) == 2:
                    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                elif resized.shape[2] == 4:
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
                else:
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                # Normalized float tensor in [-1, 1]
                blob_orig = (rgb.astype(np.float32) / 127.5) - 1.0

                if use_tta:
                    # Test-Time Augmentation: joint batch inference with horizontal flip
                    rgb_flip = cv2.flip(rgb, 1)
                    blob_flip = (rgb_flip.astype(np.float32) / 127.5) - 1.0

                    if self.is_nhwc:
                        # [2, 112, 112, 3]
                        batch = np.stack([blob_orig, blob_flip], axis=0)
                    else:
                        # [2, 3, 112, 112]
                        batch = np.stack([
                            np.transpose(blob_orig, (2, 0, 1)),
                            np.transpose(blob_flip, (2, 0, 1))
                        ], axis=0)

                    outputs = self.session.run([self.output_name], {self.input_name: batch})
                    e_orig = outputs[0][0].flatten().astype(np.float32)
                    e_flip = outputs[0][1].flatten().astype(np.float32)
                    emb = e_orig + e_flip
                else:
                    if self.is_nhwc:
                        blob = np.expand_dims(blob_orig, axis=0)
                    else:
                        blob = np.transpose(blob_orig, (2, 0, 1))[np.newaxis, :]
                    outputs = self.session.run([self.output_name], {self.input_name: blob})
                    emb = outputs[0].flatten().astype(np.float32)

                norm = np.linalg.norm(emb) + 1e-8
                return emb / norm
            except Exception as e:
                print(f"[FaceExtractor] Inference error ({e}), falling back to color-histogram features.")

        # Deterministic fallback pseudo-embedding generated from spatial-color histogram
        resized = cv2.resize(face_crop, (64, 64))
        if len(resized.shape) == 3 and resized.shape[2] >= 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        # Tile to 512-d
        emb = np.concatenate([hist, hist])
        norm = np.linalg.norm(emb) + 1e-8
        return (emb / norm).astype(np.float32)

    def compare(self, crop1: np.ndarray, crop2: np.ndarray) -> float:
        """Direct similarity comparison between two face crops."""
        emb1 = self.extract_embedding(crop1)
        emb2 = self.extract_embedding(crop2)
        return cosine_similarity(emb1, emb2)
