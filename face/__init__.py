"""
face/__init__.py
================
Face extraction and quality estimation module (Track P1).

Provides:
    - FaceDetector         : MediaPipe-based face detection + mesh landmarks
    - DetectedFace         : Data class for a single detected face
    - FacePose             : Head pose angles (yaw, pitch, roll)
    - FaceFeatureExtractor : 512-d face embedding extraction (ArcFace / fallback)
    - estimate_face_quality: Composite quality scoring with fusion weight advice
"""

from face.detector import FaceDetector, DetectedFace, FacePose
from face.quality import estimate_face_quality
from face.extractor import FaceFeatureExtractor

__all__ = [
    "FaceDetector",
    "DetectedFace",
    "FacePose",
    "FaceFeatureExtractor",
    "estimate_face_quality",
]
