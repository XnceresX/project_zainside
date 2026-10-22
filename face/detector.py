"""
face/detector.py
================
Face detection pipeline for SAFEPASS (Track P1).

Uses MediaPipe Face Detection for lightweight, CPU-friendly face localisation,
and MediaPipe Face Mesh (468 landmarks) for pose estimation (yaw, pitch, roll)
and landmark-based occlusion scoring.

Usage:
    from face.detector import FaceDetector

    detector = FaceDetector()
    results  = detector.detect(bgr_frame)
    # results is a list of DetectedFace objects
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FacePose:
    """Head pose angles estimated from face mesh landmarks."""
    yaw: float = 0.0    # Left/right rotation in degrees
    pitch: float = 0.0  # Up/down rotation in degrees
    roll: float = 0.0   # Tilt rotation in degrees


@dataclass
class DetectedFace:
    """All information extracted for a single detected face in a frame."""
    bbox: Tuple[int, int, int, int]           # (x1, y1, x2, y2) in pixel coords
    face_crop: np.ndarray = field(repr=False) # BGR face crop from the frame
    detection_confidence: float = 0.0         # MediaPipe detection confidence [0-1]
    landmarks: Optional[np.ndarray] = None    # (N, 3) normalized landmark coords
    pose: FacePose = field(default_factory=FacePose)
    landmark_visibility: float = 1.0          # Fraction of visible key landmarks [0-1]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class FaceDetector:
    """
    Two-stage face analysis:
      1. MediaPipe Face Detection  → bounding box + confidence
      2. MediaPipe Face Mesh       → 468 landmarks → pose + occlusion

    Parameters
    ----------
    detection_confidence : float
        Minimum confidence for the face detector (default 0.5).
    mesh_confidence : float
        Minimum confidence for the face mesh model (default 0.5).
    max_faces : int
        Maximum number of faces to detect per frame (default 10).
    enable_mesh : bool
        Whether to run the heavier face mesh stage.  Set False if you
        only need bounding boxes (e.g. for quick person filtering).
    """

    # Indices of the 6 key landmarks used for head-pose (PnP) estimation.
    # These are the canonical MediaPipe Face Mesh indices:
    #   1  = Nose tip
    #  33  = Right eye right corner
    # 263  = Left eye left corner
    #  61  = Right mouth corner
    # 291  = Left mouth corner
    # 199  = Chin / lower lip centre
    _POSE_LANDMARK_IDS = [1, 33, 263, 61, 291, 199]

    # Corresponding approximate 3-D model points (in a generic face model,
    # units are arbitrary but consistent).
    _MODEL_POINTS = np.array([
        (0.0,    0.0,    0.0),     # Nose tip
        (-30.0, -30.0,  -30.0),    # Right eye right corner
        (30.0,  -30.0,  -30.0),    # Left eye left corner
        (-25.0,  25.0,  -15.0),    # Right mouth corner
        (25.0,   25.0,  -15.0),    # Left mouth corner
        (0.0,    45.0,  -10.0),    # Chin
    ], dtype=np.float64)

    # Key facial landmark indices for occlusion/visibility checks.
    # Left eye, right eye, nose, mouth corners, forehead, chin.
    _VISIBILITY_LANDMARK_IDS = [
        1,    # nose tip
        33,   # right eye outer
        133,  # right eye inner
        362,  # left eye inner
        263,  # left eye outer
        61,   # right mouth
        291,  # left mouth
        10,   # forehead top
        152,  # chin bottom
    ]

    def __init__(
        self,
        detection_confidence: float = 0.5,
        mesh_confidence: float = 0.5,
        max_faces: int = 10,
        enable_mesh: bool = True,
    ):
        import mediapipe as mp

        self._mp_face_detection = mp.solutions.face_detection
        self._mp_face_mesh = mp.solutions.face_mesh

        self._detector = self._mp_face_detection.FaceDetection(
            model_selection=1,  # 1 = full-range model (better for varied distances)
            min_detection_confidence=detection_confidence,
        )

        self._mesh = None
        if enable_mesh:
            self._mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=mesh_confidence,
                min_tracking_confidence=mesh_confidence,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        Detect all faces in a BGR frame.

        Returns a list of DetectedFace objects sorted by descending
        detection confidence.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Stage 1: Face Detection → bounding boxes
        det_results = self._detector.process(rgb)
        if not det_results.detections:
            return []

        faces: List[DetectedFace] = []

        for detection in det_results.detections:
            bbox_rel = detection.location_data.relative_bounding_box
            x1 = max(int(bbox_rel.xmin * w), 0)
            y1 = max(int(bbox_rel.ymin * h), 0)
            bw = int(bbox_rel.width * w)
            bh = int(bbox_rel.height * h)
            x2 = min(x1 + bw, w)
            y2 = min(y1 + bh, h)

            # Pad slightly for better crop quality (10% each side)
            pad_x = int(bw * 0.10)
            pad_y = int(bh * 0.10)
            x1p = max(x1 - pad_x, 0)
            y1p = max(y1 - pad_y, 0)
            x2p = min(x2 + pad_x, w)
            y2p = min(y2 + pad_y, h)

            crop = frame[y1p:y2p, x1p:x2p].copy()
            if crop.size == 0:
                continue

            conf = detection.score[0] if detection.score else 0.0

            faces.append(DetectedFace(
                bbox=(x1p, y1p, x2p, y2p),
                face_crop=crop,
                detection_confidence=float(conf),
            ))

        # Stage 2: Face Mesh → landmarks, pose, occlusion
        if self._mesh is not None:
            mesh_results = self._mesh.process(rgb)
            if mesh_results.multi_face_landmarks:
                self._assign_mesh_to_faces(
                    faces, mesh_results.multi_face_landmarks, h, w
                )

        # Sort by confidence descending
        faces.sort(key=lambda f: f.detection_confidence, reverse=True)
        return faces

    def detect_single(self, frame: np.ndarray) -> Optional[DetectedFace]:
        """Convenience: return the highest-confidence face, or None."""
        faces = self.detect(frame)
        return faces[0] if faces else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assign_mesh_to_faces(
        self,
        faces: List[DetectedFace],
        mesh_landmarks_list,
        frame_h: int,
        frame_w: int,
    ):
        """
        Match each mesh result to the closest detected face bbox,
        then compute pose and visibility from the matched landmarks.
        """
        for mesh_lm in mesh_landmarks_list:
            # Convert landmarks to pixel coordinates
            pts = np.array([
                (lm.x * frame_w, lm.y * frame_h, lm.z * frame_w)
                for lm in mesh_lm.landmark
            ])

            # Find the face bbox whose centre is closest to the mesh nose tip
            nose = pts[1][:2]  # index 1 = nose tip
            best_face = None
            best_dist = float("inf")
            for face in faces:
                cx = (face.bbox[0] + face.bbox[2]) / 2
                cy = (face.bbox[1] + face.bbox[3]) / 2
                dist = math.hypot(nose[0] - cx, nose[1] - cy)
                if dist < best_dist:
                    best_dist = dist
                    best_face = face

            if best_face is None:
                continue

            best_face.landmarks = pts
            best_face.pose = self._estimate_pose(pts, frame_h, frame_w)
            best_face.landmark_visibility = self._estimate_visibility(pts, frame_h, frame_w)

    def _estimate_pose(
        self, landmarks: np.ndarray, frame_h: int, frame_w: int
    ) -> FacePose:
        """
        Estimate head pose (yaw, pitch, roll) via Perspective-n-Point (PnP)
        using 6 canonical face landmarks.
        """
        image_points = np.array([
            landmarks[idx][:2] for idx in self._POSE_LANDMARK_IDS
        ], dtype=np.float64)

        # Approximate camera intrinsics (assuming no lens distortion)
        focal_length = frame_w
        center = (frame_w / 2.0, frame_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            self._MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return FacePose()

        rmat, _ = cv2.Rodrigues(rvec)
        # Decompose rotation matrix into Euler angles
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        if sy > 1e-6:
            pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
            yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
            yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll = 0.0

        return FacePose(
            yaw=round(yaw, 2),
            pitch=round(pitch, 2),
            roll=round(roll, 2),
        )

    def _estimate_visibility(
        self, landmarks: np.ndarray, frame_h: int, frame_w: int
    ) -> float:
        """
        Estimate fraction of key facial landmarks that are within the frame
        and have reasonable z-depth.  A landmark is considered 'visible' if
        its (x, y) falls inside the image bounds.

        Returns a float in [0.0, 1.0].
        """
        visible = 0
        total = len(self._VISIBILITY_LANDMARK_IDS)

        for idx in self._VISIBILITY_LANDMARK_IDS:
            x, y = landmarks[idx][0], landmarks[idx][1]
            if 0 <= x <= frame_w and 0 <= y <= frame_h:
                visible += 1

        return round(visible / total, 4) if total > 0 else 1.0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Release MediaPipe resources."""
        self._detector.close()
        if self._mesh is not None:
            self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
