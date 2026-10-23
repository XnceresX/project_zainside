"""
face/detector.py
================
Face detection pipeline for SAFEPASS (Track P1).

Provides multi-backend face detection, anatomical anti-ghost verification,
sub-pixel landmark analysis, and 5-point affine canonical alignment:
  1. SCRFD (InsightFace SOTA Backbone) - Anchor-pyramid ONNX detector
     specifically designed for ArcFace, outstanding small-face recall.
  2. OpenCV YuNet (cv2.FaceDetectorYN) - Fast, lightweight ONNX detector.
  3. MediaPipe Face Detection (when available).
  4. OpenCV Haar Cascade (built-in fallback, zero extra files needed).
  5. Multi-Scale Sliced Inference (SAHI) for high-resolution CCTV crowd streams.
  6. Anatomical Anti-Ghost Filter: eliminates non-face false alarms via
     biological facial proportions (aspect ratio, inter-ocular distance,
     vertical nose hierarchy, eye-mouth parallelism).

Usage:
    from face.detector import FaceDetector

    detector = FaceDetector()
    results  = detector.detect(bgr_frame)
    # results is a list of DetectedFace objects with 112x112 aligned crops
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_SCRFD_PATHS = [
    os.path.join("data", "models", "scrfd_2.5g_bnkps.onnx"),
    os.path.join("data", "models", "scrfd_10g_bnkps.onnx"),
    os.path.join("models", "scrfd_2.5g_bnkps.onnx"),
]

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


def verify_anatomical_face_geometry(
    bbox: Tuple[int, int, int, int],
    landmarks: Optional[np.ndarray],
    strict: bool = False,
) -> bool:
    """
    Validates biological face geometric proportions to eliminate false-alarm
    ghost detections (shirt folds, foliage, wall textures, car grilles).

    Checks:
      1. Bounding box aspect ratio (height / width in [0.65, 1.85]).
      2. Inter-ocular distance relative to box width (in [0.15, 0.70]).
      3. Vertical landmark hierarchy (eyes_y <= nose_y <= mouth_y).
      4. Bilateral eye-mouth parallelism (|theta_eyes - theta_mouth| <= 30 deg).
    """
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)

    # 1. Aspect ratio check (human faces: height / width in [0.65, 1.85])
    aspect_ratio = h / float(w)
    if aspect_ratio < 0.65 or aspect_ratio > 1.85:
        return False

    if landmarks is None or len(landmarks) < 5:
        return not strict

    re, le, nose, rm, lm = landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4]

    # 2. Inter-ocular distance ratio
    eye_dist = float(np.linalg.norm(le - re))
    eye_ratio = eye_dist / float(w)
    if eye_ratio < 0.15 or eye_ratio > 0.70:
        return False

    # 3. Vertical landmark hierarchy
    eye_mid_y = (re[1] + le[1]) / 2.0
    mouth_mid_y = (rm[1] + lm[1]) / 2.0
    nose_y = float(nose[1])

    # Allow slight tolerance (5% of height) for moderate pitch angles
    tol = 0.05 * h
    if not (eye_mid_y - tol <= nose_y <= mouth_mid_y + tol):
        return False

    # 4. Bilateral eye-mouth parallelism
    dx_eyes = float(le[0] - re[0])
    dy_eyes = float(le[1] - re[1])
    dx_mouth = float(lm[0] - rm[0])
    dy_mouth = float(lm[1] - rm[1])

    angle_eyes = math.degrees(math.atan2(dy_eyes, dx_eyes)) if (dx_eyes != 0 or dy_eyes != 0) else 0.0
    angle_mouth = math.degrees(math.atan2(dy_mouth, dx_mouth)) if (dx_mouth != 0 or dy_mouth != 0) else 0.0

    angle_diff = abs(angle_eyes - angle_mouth)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    if angle_diff > 30.0:  # Eyes and mouth line must be roughly parallel
        return False

    return True


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
    Robust multi-backend face detector supporting:
      - SCRFD (InsightFace SOTA FPN backbone, ultra-high recall on small faces)
      - OpenCV YuNet (Fast, lightweight ONNX detector)
      - MediaPipe Face Detection
      - OpenCV Haar Cascade
    Features anatomical anti-ghost filtering, 5-point canonical alignment, and
    multi-scale tiled inference (SAHI) for crowded CCTV feeds.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: Optional[str] = None,  # "scrfd", "yunet", "mediapipe", "haar", or None (auto)
        detection_confidence: float = 0.5,
        max_faces: int = 10,
        enable_anatomical_filter: bool = True,
    ):
        self.detection_confidence = detection_confidence
        self.max_faces = max_faces
        self.enable_anatomical_filter = enable_anatomical_filter
        self.backend = "none"
        self.model_path = None

        self._scrfd = None
        self._scrfd_input_name = None
        self._yunet = None
        self._haar = None
        self._mp_detector = None

        # 1. Try SCRFD (Highest accuracy, InsightFace companion to ArcFace)
        if backend in (None, "scrfd"):
            scrfd_path = self._resolve_model_path(model_path, DEFAULT_SCRFD_PATHS)
            if scrfd_path and os.path.exists(scrfd_path):
                try:
                    import onnxruntime as ort
                    opts = ort.SessionOptions()
                    opts.log_severity_level = 3
                    self._scrfd = ort.InferenceSession(scrfd_path, sess_options=opts, providers=["CPUExecutionProvider"])
                    self._scrfd_input_name = self._scrfd.get_inputs()[0].name
                    self.backend = "scrfd"
                    self.model_path = scrfd_path
                except Exception as e:
                    self._scrfd = None

        # 2. Try OpenCV YuNet
        if self.backend == "none" and backend in (None, "yunet"):
            yunet_path = self._resolve_model_path(model_path, DEFAULT_YUNET_PATHS)
            if yunet_path and hasattr(cv2, "FaceDetectorYN"):
                try:
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
                except Exception:
                    self._yunet = None

        # 3. Try MediaPipe
        if self.backend == "none" and backend in (None, "mediapipe"):
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

        # 4. Fallback to Haar Cascade
        haar_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(haar_path):
            self._haar = cv2.CascadeClassifier(haar_path)
            if self.backend == "none":
                self.backend = "haar"

    def _resolve_model_path(self, model_path: Optional[str], default_paths: List[str]) -> Optional[str]:
        if model_path and os.path.exists(model_path):
            return model_path
        for candidate in default_paths:
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

        if self.backend == "scrfd" and self._scrfd is not None:
            return self._detect_scrfd(frame, h, w)
        elif self.backend == "yunet" and self._yunet is not None:
            return self._detect_yunet(frame, h, w)
        elif self.backend == "mediapipe" and self._mp_detector is not None:
            return self._detect_mediapipe(frame, h, w)
        elif self._haar is not None:
            return self._detect_haar(frame, h, w)
        else:
            return []

    def detect_single(self, frame: np.ndarray) -> Optional[DetectedFace]:
        """Convenience: return the highest-confidence face, or None."""
        faces = self.detect(frame)
        return faces[0] if faces else None

    def detect_sliced(
        self,
        frame: np.ndarray,
        tile_size: int = 640,
        overlap: float = 0.20,
        nms_threshold: float = 0.40,
    ) -> List[DetectedFace]:
        """
        Multi-scale sliced inference (SAHI) for high-resolution CCTV crowd streams.
        Slices frame into overlapping tiles to detect tiny distant faces, then merges
        with full-frame detections.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        candidates = list(self.detect(frame))

        # Only slice if frame is substantially larger than tile_size
        if w > tile_size or h > tile_size:
            stride = int(tile_size * (1.0 - overlap))
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    x_end = min(x + tile_size, w)
                    y_end = min(y + tile_size, h)
                    tile = frame[y:y_end, x:x_end]
                    if tile.shape[0] < 100 or tile.shape[1] < 100:
                        continue

                    tile_faces = self.detect(tile)
                    for f in tile_faces:
                        gx1, gy1, gx2, gy2 = f.bbox[0] + x, f.bbox[1] + y, f.bbox[2] + x, f.bbox[3] + y
                        g_lm = f.landmarks + np.array([x, y], dtype=np.float32) if f.landmarks is not None else None
                        g_aligned = align_face_5points(frame, g_lm, (112, 112)) if g_lm is not None else f.face_crop
                        candidates.append(DetectedFace(
                            bbox=(gx1, gy1, gx2, gy2),
                            face_crop=g_aligned,
                            detection_confidence=f.detection_confidence,
                            landmarks=g_lm,
                            pose=f.pose,
                            landmark_visibility=f.landmark_visibility,
                            raw_crop=frame[gy1:gy2, gx1:gx2].copy() if gy2 > gy1 and gx2 > gx1 else f.raw_crop,
                        ))

        if not candidates:
            return []

        cv_boxes = [[int(f.bbox[0]), int(f.bbox[1]), int(f.bbox[2] - f.bbox[0]), int(f.bbox[3] - f.bbox[1])] for f in candidates]
        scores = [f.detection_confidence for f in candidates]
        indices = cv2.dnn.NMSBoxes(cv_boxes, scores, self.detection_confidence, nms_threshold)
        if len(indices) == 0:
            return []

        merged: List[DetectedFace] = []
        for idx in indices[:self.max_faces]:
            i = idx[0] if isinstance(idx, (list, np.ndarray)) else idx
            merged.append(candidates[i])

        merged.sort(key=lambda f: f.detection_confidence, reverse=True)
        return merged

    # ------------------------------------------------------------------
    # Backend 1: SCRFD (InsightFace)
    # ------------------------------------------------------------------

    def _detect_scrfd(self, frame: np.ndarray, h: int, w: int) -> List[DetectedFace]:
        target_size = 640
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, :]

        outputs = self._scrfd.run(None, {self._scrfd_input_name: blob})
        scores_list = [outputs[0], outputs[1], outputs[2]]
        bboxes_list = [outputs[3], outputs[4], outputs[5]]
        kps_list = [outputs[6], outputs[7], outputs[8]]

        strides = [8, 16, 32]
        all_boxes, all_scores, all_kps = [], [], []

        for stride, scores, bboxes, kps in zip(strides, scores_list, bboxes_list, kps_list):
            scores = scores[0].flatten()
            bboxes = bboxes[0] * stride
            kps = kps[0] * stride

            h_s, w_s = target_size // stride, target_size // stride
            anchor_centers = np.stack(np.mgrid[:h_s, :w_s][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            anchor_centers = np.repeat(anchor_centers, 2, axis=0)

            pos_mask = scores >= self.detection_confidence
            if not np.any(pos_mask):
                continue

            s_scores = scores[pos_mask]
            s_anchors = anchor_centers[pos_mask]
            s_bboxes = bboxes[pos_mask]
            s_kps = kps[pos_mask]

            x1 = s_anchors[:, 0] - s_bboxes[:, 0]
            y1 = s_anchors[:, 1] - s_bboxes[:, 1]
            x2 = s_anchors[:, 0] + s_bboxes[:, 2]
            y2 = s_anchors[:, 1] + s_bboxes[:, 3]
            boxes = np.stack([x1, y1, x2, y2], axis=1)
            pts = s_kps.reshape((-1, 5, 2)) + s_anchors[:, np.newaxis, :]

            all_boxes.append(boxes)
            all_scores.append(s_scores)
            all_kps.append(pts)

        if not all_boxes:
            return []

        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_kps = np.concatenate(all_kps, axis=0)

        # Scale coordinates back from 640x640 padded to original frame
        all_boxes[:, [0, 2]] = (all_boxes[:, [0, 2]] - pad_x) / scale
        all_boxes[:, [1, 3]] = (all_boxes[:, [1, 3]] - pad_y) / scale
        all_kps[:, :, 0] = (all_kps[:, :, 0] - pad_x) / scale
        all_kps[:, :, 1] = (all_kps[:, :, 1] - pad_y) / scale

        # Clip to frame boundary
        all_boxes[:, [0, 2]] = np.clip(all_boxes[:, [0, 2]], 0, w)
        all_boxes[:, [1, 3]] = np.clip(all_boxes[:, [1, 3]], 0, h)

        # OpenCV NMS
        cv_boxes = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in all_boxes]
        indices = cv2.dnn.NMSBoxes(cv_boxes, all_scores.tolist(), self.detection_confidence, 0.4)
        if len(indices) == 0:
            return []

        faces: List[DetectedFace] = []
        for idx in indices[:self.max_faces]:
            i = idx[0] if isinstance(idx, (list, np.ndarray)) else idx
            x1, y1, x2, y2 = int(all_boxes[i][0]), int(all_boxes[i][1]), int(all_boxes[i][2]), int(all_boxes[i][3])
            conf = float(all_scores[i])
            landmarks_5 = all_kps[i].astype(np.float32)

            # Anatomical anti-ghost filter
            if self.enable_anatomical_filter and not verify_anatomical_face_geometry((x1, y1, x2, y2), landmarks_5):
                continue

            raw_crop = frame[y1:y2, x1:x2].copy()
            if raw_crop.size == 0:
                continue

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
    # Backend 2: OpenCV YuNet
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

            # Anatomical anti-ghost filter
            if self.enable_anatomical_filter and not verify_anatomical_face_geometry((x1, y1, x2, y2), landmarks_5):
                continue

            raw_crop = frame[y1:y2, x1:x2].copy()
            if raw_crop.size == 0:
                continue

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
    # Backend 3: MediaPipe
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

            if self.enable_anatomical_filter and landmarks_5 is not None:
                if not verify_anatomical_face_geometry((x1p, y1p, x2p, y2p), landmarks_5):
                    continue

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
    # Backend 4: Haar Cascade Fallback
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

            if self.enable_anatomical_filter and not verify_anatomical_face_geometry((x1, y1, x2, y2), None):
                continue

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
        dx = float(le[0] - re[0])
        dy = float(le[1] - re[1])
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
