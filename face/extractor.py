"""
face/extractor.py
=================
Face embedding extraction and cosine similarity matching (Track P1).
Designed to wrap MobileFaceNet / ArcFace ONNX models.
"""

from typing import Optional
import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two 1D feature vectors."""
    if vec1 is None or vec2 is None:
        return 0.0
    dot = np.dot(vec1, vec2)
    norm = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-8
    sim = dot / norm
    # Normalize from [-1, 1] to [0, 1]
    return float(round((sim + 1.0) / 2.0, 4))


class FaceFeatureExtractor:
    """
    Face feature extractor supporting MobileFaceNet / ArcFace.
    Includes a lightweight deterministic feature fallback for testing
    when ONNX model weights are not loaded.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        try:
            import onnxruntime as ort
            self.model = ort.InferenceSession(model_path)
            print(f"[FaceExtractor] Loaded ONNX model from {model_path}")
        except Exception as e:
            print(f"[FaceExtractor] Warning: Could not load ONNX model ({e}). Using mock extractor.")
            self.model = None

    def extract_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Extracts 512-dimensional normalized embedding from face crop.
        """
        if face_crop is None or face_crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        if self.model is not None:
            # Preprocess crop to 112x112, normalize, and run inference
            import cv2
            resized = cv2.resize(face_crop, (112, 112))
            blob = resized.astype(np.float32) / 127.5 - 1.0
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis, :]
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: blob})
            emb = outputs[0].flatten()
            norm = np.linalg.norm(emb) + 1e-8
            return emb / norm

        # Fallback pseudo-embedding generated from spatial-color histogram
        import cv2
        resized = cv2.resize(face_crop, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        # Tile to 512-d
        emb = np.concatenate([hist, hist])
        norm = np.linalg.norm(emb) + 1e-8
        return (emb / norm).astype(np.float32)

    def compare(self, crop1: np.ndarray, crop2: np.ndarray) -> float:
        """Direct similarity comparison between two face images."""
        emb1 = self.extract_embedding(crop1)
        emb2 = self.extract_embedding(crop2)
        return cosine_similarity(emb1, emb2)
