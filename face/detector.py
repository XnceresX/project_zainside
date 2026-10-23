"""
face/detector.py
================
Face detection pipeline for SAFEPASS (Track P1).

Provides multi-backend face detection and landmark analysis:
  1. OpenCV YuNet (cv2.FaceDetectorYN) - Fast, lightweight ONNX detector
     providing bounding boxes and 5 facial landmarks (eyes, nose, mouth).
  2. MediaPipe Face Detection + Mesh (when available).
  3. OpenCV Haar Cascade (built-in fallback, zero extra files needed).

Usage:
    from face.detector import FaceDetector

    detector = FaceDetector()
    results  = detector.detect(bgr_frame)
    # results is a list of DetectedFace objects
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_YUNET_PATHS = [
    os.path.join("data", "models", "yunet.onnx"),
    os.path.join("data", "models", "face_detection_yunet_2023mar.onnx"),
    os.path.join("models", "yunet.onnx"),
]

# Canonical 5-point facial reference coordinates for 112x112 ArcFace
ARCFACE_CANONICAL_5PTS = np.array([
    [38.2946, 51.6963],   # Right eye (in image coords: viewer's left)
    [73.5318, 51.5014],   # Left eye (in image coords: viewer's right)
    [56.0252, 71.7366],   # Nose tip
    [41.5493, 92.3655],   # Right mouth corner
    [70.7299, 92.2041],   # Left mouth corner
], dtype=np.float32)


def align_face_5points(
    image: np.ndarray,
    landmarks: Optional[np.ndarray],
    output_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Applies 2D similarity transform (Umeyama / LMEDS) to warp face onto
    canonical ArcFace 112x112 coordinate space.
    """
    if landmarks is None or len(landmarks) < 5 or image is None or image.size == 0:
        if image is not None and image.size > 0:
            return cv2.resize(image, output_size)
        return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    src_pts = landmarks[:5].astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src_pts, ARCFACE_CANONICAL_5PTS, method=cv2.LMEDS)
    if M is None:
        return cv2.resize(image, output_size)

    warped = cv2.warpAffine(
        image,
        M,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FacePose:
    """Head pose angles estimated from face landmarks."""
    yaw: float = 0.0    # Left/right rotation in degrees (-90 to +90)
    pitch: float = 0.0  # Up/down rotation in degrees (-90 to +90)
    roll: float = 0.0   # Tilt rotation in degrees (-180 to +180)


@dataclass
class DetectedFace:
    """All information extracted for a single detected face in a frame."""
    bbox: Tuple[int, int, int, int]           # (x1, y1, x2, y2) in pixel coords
    face_crop: np.ndarray = field(repr=False) # 112x112 similarity-aligned face crop (optimal for ArcFace)
    detection_confidence: float = 0.0         # Detection confidence [0-1]
    landmarks: Optional[np.ndarray] = None    # (N, 2 or 3) landmark coordinates in pixels
    pose: FacePose = field(default_factory=FacePose)
    landmark_visibility: float = 1.0          # Visibility ratio of key facial features [0-1]
    raw_crop: Optional[np.ndarray] = field(default=None, repr=False) # Original unaligned bounding box crop


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class FaceDetector:
    """
    Robust face detector supporting OpenCV YuNet, MediaPipe, and Haar Cascade.
    Extracts face bounding boxes, crops, 5-point landmarks, and head pose.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        detection_confidence: float = 0.5,
        max_faces: int = 10,
    ):
        self.detection_confidence = detection_confidence
        self.max_faces = max_faces
        self.backend = "none"
        self._yunet = None
        self._haar = None
        self._mp_detector = None
        self._mp_mesh = None

        # 1. Try OpenCV YuNet
        yunet_path = self._resolve_yunet_path(model_path)
        if yunet_path and hasattr(cv2, "FaceDetectorYN"):
            try:
                # Initialize with a default size; updated dynamically per frame
                self._yunet = cv2.FaceDetectorYN.create(
                    model=yunet_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=detection_confidence,
                    nms_threshold=0.3,
                    top_k=max_faces,
                )
                self.backend = "yunet"
                self.model_path = yunet_path
            except Exception as e:
                self._yunet = None

        # 2. Try MediaPipe solutions if YuNet wasn't initialized
        if self.backend == "none":
            try:
                import mediapipe as mp
                if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                    self._mp_detector = mp.solutions.face_detection.FaceDetection(
                        model_selection=1,
                        min_detection_confidence=detection_confidence,
                    )
                    self.backend = "mediapipe"
            except Exception:
                self._mp_detector = None

        # 3. Always prepare Haar Cascade as fallback
        haar_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(haar_path):
            self._haar = cv2.CascadeClassifier(haar_path)
            if self.backend == "none":
                self.backend = "haar"

    def _resolve_yunet_path(self, model_path: Optional[str]) -> Optional[str]:
        if model_path and os.path.exists(model_path):
            return model_path
        for candidate in DEFAULT_YUNET_PATHS:
            if os.path.exists(candidate):
                return candidate
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        Detect all faces in a BGR frame.
        Returns a list of DetectedFace objects sorted by confidence descending.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        if self._yunet is not None:
            return self._detect_yunet(frame, h, w)
        elif self._mp_detector is not None:
            return self._detect_mediapipe(frame, h, w)
        elif self._haar is not None:
            return self._detect_haar(frame, h, w)
        else:
            return []

    def detect_single(self, frame: np.ndarray) -> Optional[DetectedFace]:
        """Convenience: return the highest-confidence face, or None."""
        faces = self.detect(frame)
        return faces[0] if faces else None

    # ------------------------------------------------------------------
    # Backend 1: OpenCV YuNet
    # ------------------------------------------------------------------

    def _detect_yunet(self, frame: np.ndarray, h: int, w: int) -> List[DetectedFace]:
        self._yunet.setInputSize((w, h))
        _, detections = self._yunet.detect(frame)
        if detections is None or len(detections) == 0:
            return []

        faces: List[DetectedFace] = []
        for det in detections[:self.max_faces]:
            x, y, bw, bh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            conf = float(det[14])

            # 5 Landmarks: right eye, left eye, nose, right mouth, left mouth
            landmarks_5 = np.array([
                [det[4], det[5]],    # Right eye
                [det[6], det[7]],    # Left eye
                [det[8], det[9]],    # Nose tip
                [det[10], det[11]],  # Right mouth corner
                [det[12], det[13]],  # Left mouth corner
            ], dtype=np.float32)

            # Pad bounding box by 10%
            pad_x = int(bw * 0.10)
            pad_y = int(bh * 0.10)
            x1 = max(x - pad_x, 0)
            y1 = max(y - pad_y, 0)
            x2 = min(x + bw + pad_x, w)
            y2 = min(y + bh + pad_y, h)

            raw_crop = frame[y1:y2, x1:x2].copy()
            if raw_crop.size == 0:
                continue

            # 5-point affine similarity alignment into canonical 112x112 space
            aligned_crop = align_face_5points(frame, landmarks_5, (112, 112))

            pose = self._estimate_pose_from_5_landmarks(landmarks_5)
            visibility = self._estimate_visibility_from_landmarks(landmarks_5, h, w)

            faces.append(DetectedFace(
                bbox=(x1, y1, x2, y2),
                face_crop=aligned_crop,
                detection_confidence=conf,
                landmarks=landmarks_5,
                pose=pose,
                landmark_visibility=visibility,
                raw_crop=raw_crop,
            ))

        faces.sort(key=lambda f: f.detection_confidence, reverse=True)
        return faces

    # ------------------------------------------------------------------
    # Backend 2: MediaPipe
    # ------------------------------------------------------------------

    def _detect_mediapipe(self, frame: np.ndarray, h: int, w: int) -> List[DetectedFace]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_detector.process(rgb)
        if not results.detections:
            return []

        faces: List[DetectedFace] = []
        for det in results.detections[:self.max_faces]:
            bbox_rel = det.location_data.relative_bounding_box
            x1 = max(int(bbox_rel.xmin * w), 0)
            y1 = max(int(bbox_rel.ymin * h), 0)
            bw = int(bbox_rel.width * w)
            bh = int(bbox_rel.height * h)
            x2 = min(x1 + bw, w)
            y2 = min(y1 + bh, h)

            pad_x = int(bw * 0.10)
            pad_y = int(bh * 0.10)
            x1p = max(x1 - pad_x, 0)
            y1p = max(y1 - pad_y, 0)
            x2p = min(x2 + pad_x, w)
            y2p = min(y2 + pad_y, h)

            raw_crop = frame[y1p:y2p, x1p:x2p].copy()
            if raw_crop.size == 0:
                continue

            # Approximate 5 landmarks if relative keypoints are present
            landmarks_5 = None
            if hasattr(det.location_data, "relative_keypoints") and len(det.location_data.relative_keypoints) >= 4:
                kps = det.location_data.relative_keypoints
                re = [kps[0].x * w, kps[0].y * h]
                le = [kps[1].x * w, kps[1].y * h]
                nose = [kps[2].x * w, kps[2].y * h]
                mc = [kps[3].x * w, kps[3].y * h]
                dx = (le[0] - re[0]) * 0.25
                rm = [mc[0] - dx, mc[1]]
                lm = [mc[0] + dx, mc[1]]
                landmarks_5 = np.array([re, le, nose, rm, lm], dtype=np.float32)

            aligned_crop = align_face_5points(frame, landmarks_5, (112, 112)) if landmarks_5 is not None else cv2.resize(raw_crop, (112, 112))
            conf = float(det.score[0]) if det.score else 0.5

            faces.append(DetectedFace(
                bbox=(x1p, y1p, x2p, y2p),
                face_crop=aligned_crop,
                detection_confidence=conf,
                landmarks=landmarks_5,
                pose=FacePose(yaw=0.0, pitch=0.0, roll=0.0),
                landmark_visibility=1.0,
                raw_crop=raw_crop,
            ))

        faces.sort(key=lambda f: f.detection_confidence, reverse=True)
        return faces

    # ------------------------------------------------------------------
    # Backend 3: Haar Cascade Fallback
    # ------------------------------------------------------------------

    def _detect_haar(self, frame: np.ndarray, h: int, w: int) -> List[DetectedFace]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(rects) == 0:
            return []

        faces: List[DetectedFace] = []
        for (x, y, bw, bh) in rects[:self.max_faces]:
            pad_x = int(bw * 0.10)
            pad_y = int(bh * 0.10)
            x1 = max(x - pad_x, 0)
            y1 = max(y - pad_y, 0)
            x2 = min(x + bw + pad_x, w)
            y2 = min(y + bh + pad_y, h)

            raw_crop = frame[y1:y2, x1:x2].copy()
            if raw_crop.size == 0:
                continue

            aligned_crop = cv2.resize(raw_crop, (112, 112))

            faces.append(DetectedFace(
                bbox=(x1, y1, x2, y2),
                face_crop=aligned_crop,
                detection_confidence=0.85,
                pose=FacePose(yaw=0.0, pitch=0.0, roll=0.0),
                landmark_visibility=1.0,
                raw_crop=raw_crop,
            ))

        return faces

    # ------------------------------------------------------------------
    # Geometric Pose & Visibility from 5 Landmarks
    # ------------------------------------------------------------------

    def _estimate_pose_from_5_landmarks(self, lm: np.ndarray) -> FacePose:
        """
        Computes head pose angles (roll, yaw, pitch) from 5 landmarks:
        lm[0] = right eye, lm[1] = left eye, lm[2] = nose,
        lm[3] = right mouth, lm[4] = left mouth.
        """
        re, le, nose, rm, lm_corner = lm[0], lm[1], lm[2], lm[3], lm[4]

        # 1. Roll: angle of eye line relative to horizontal
        dx = le[0] - re[0]
        dy = le[1] - re[1]
        roll = math.degrees(math.atan2(dy, dx)) if (dx != 0 or dy != 0) else 0.0

        # 2. Yaw: nose horizontal offset relative to eye center
        eye_mid = (re + le) / 2.0
        eye_dist = max(float(np.linalg.norm(le - re)), 1.0)
        yaw = float((nose[0] - eye_mid[0]) / (eye_dist / 2.0) * 45.0)
        yaw = max(min(yaw, 90.0), -90.0)

        # 3. Pitch: vertical nose position relative to eye-mouth distance
        mouth_mid = (rm + lm_corner) / 2.0
        vert_span = max(float(mouth_mid[1] - eye_mid[1]), 1.0)
        ratio = (nose[1] - eye_mid[1]) / vert_span
        pitch = float((ratio - 0.55) * 60.0)
        pitch = max(min(pitch, 90.0), -90.0)

        return FacePose(
            yaw=round(yaw, 1),
            pitch=round(pitch, 1),
            roll=round(roll, 1),
        )

    def _estimate_visibility_from_landmarks(self, lm: np.ndarray, h: int, w: int) -> float:
        """Fraction of 5 facial landmarks inside frame boundaries."""
        visible = sum(1 for pt in lm if 0 <= pt[0] <= w and 0 <= pt[1] <= h)
        return round(visible / len(lm), 4)

    def close(self):
        """Release any persistent resources."""
        if self._mp_detector is not None and hasattr(self._mp_detector, "close"):
            self._mp_detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
