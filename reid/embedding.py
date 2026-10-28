"""
reid/embedding.py
=================
Appearance & ReID Feature Extraction Engine.
Supports:
  1. TorchReID OSNet (512-d) deep neural embeddings via appearance.extractor.
  2. Spatial LAB/HSV color-geometry histograms (deterministic fallback).
  3. Lightweight fast-crop stub embedding from ReID-framework.
"""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two feature vectors."""
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a.flatten(), b.flatten()) / denom)


def stub_embedding_for_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Deterministic normalized embedding stub from ReID-framework.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(8, dtype=np.float32)

    resized = cv2.resize(crop, (16, 16))
    mean = resized.reshape(-1, 3).mean(axis=0) / 255.0
    std = resized.reshape(-1, 3).std(axis=0) / 255.0
    aspect = np.array([(x2 - x1) / max(1.0, y2 - y1), (x2 - x1) * (y2 - y1) / 10000.0])
    vector = np.concatenate([mean, std, aspect]).astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else vector


class ReIDFeatureExtractor:
    """
    Unified ReID Feature Extractor combining Deep OSNet and Fast Spatial Embeddings.
    """

    def __init__(self, use_osnet: bool = True) -> None:
        self.use_osnet = use_osnet
        self._osnet_extractor = None

        if self.use_osnet:
            try:
                from appearance.extractor import AppearanceExtractor
                self._osnet_extractor = AppearanceExtractor()
                print("[ReIDFeatureExtractor] Loaded AppearanceExtractor (OSNet/HOG).")
            except Exception as e:
                print(f"[ReIDFeatureExtractor] Notice: using lightweight stub embedding: {e}")
                self._osnet_extractor = None

    def extract(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extracts a normalized ReID embedding for the given bounding box in frame.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return np.zeros(512 if self._osnet_extractor else 8, dtype=np.float32)

        if self._osnet_extractor is not None:
            try:
                emb = self._osnet_extractor.extract_embedding(crop)
                norm = np.linalg.norm(emb)
                return (emb / norm).astype(np.float32) if norm > 1e-8 else emb
            except Exception:
                pass

        return stub_embedding_for_crop(frame, bbox)
