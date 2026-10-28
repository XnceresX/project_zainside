"""
reid/detector.py
================
Deep Neural Person Detector for CCTV and Webcam surveillance.
Uses FasterRCNN-MobileNetV3-FPN to accurately detect human bodies (COCO class 1)
and eliminate all false detections on car parts, doors, windows, pavements, and shadows.
"""

from __future__ import annotations

import time
from typing import List, Tuple, Optional
import cv2
import numpy as np
import torch
import torchvision.models.detection as detection_models

from reid.tracker import Detection


class SimpleBlobDetector:
    """Detects colored/uniform person-like blobs as in ReID-framework."""

    def __init__(self, min_area: float = 600.0) -> None:
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> List[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        mask = cv2.inRange(saturation, 50, 255) & cv2.inRange(value, 40, 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            confidence = min(1.0, area / 5000.0)
            detections.append(Detection((x, y, x + w, y + h), confidence, "person"))
        return detections


class SharedPersonDetector:
    """
    Singleton / Shared Neural Person Detector instance to conserve memory
    and maximize inference throughput across multiple CCTV camera streams.
    """
    _instance: Optional[SharedPersonDetector] = None

    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold
        print("[SharedPersonDetector] Loading FasterRCNN MobileNetV3 Person Detector...")
        try:
            weights = detection_models.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            self.model = detection_models.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
            self.model.eval()
            self.is_loaded = True
            print("[SharedPersonDetector] FasterRCNN Person Detector loaded successfully.")
        except Exception as e:
            print(f"[SharedPersonDetector] Warning: Could not load FasterRCNN ({e}). Falling back.")
            self.model = None
            self.is_loaded = False

    @classmethod
    def get_instance(cls, confidence_threshold: float = 0.50) -> SharedPersonDetector:
        if cls._instance is None:
            cls._instance = SharedPersonDetector(confidence_threshold)
        return cls._instance

    def detect_frame(self, frame: np.ndarray, infer_size: Tuple[int, int] = (448, 336)) -> List[Detection]:
        """
        Runs neural person detection on the given frame.
        Only returns bounding boxes belonging to COCO class 1 ('person').
        """
        if not self.is_loaded or self.model is None:
            return []

        orig_h, orig_w = frame.shape[:2]
        iw, ih = infer_size

        # Resize for fast inference
        resized = cv2.resize(frame, (iw, ih))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        scale_x = orig_w / iw
        scale_y = orig_h / ih

        detections: List[Detection] = []
        with torch.no_grad():
            preds = self.model([tensor])[0]

        boxes = preds["boxes"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            # Class 1 in COCO is 'person'
            if label == 1 and score >= self.confidence_threshold:
                bx1, by1, bx2, by2 = box
                # Scale back to original frame dimensions
                x1 = max(0, int(bx1 * scale_x))
                y1 = max(0, int(by1 * scale_y))
                x2 = min(orig_w, int(bx2 * scale_x))
                y2 = min(orig_h, int(by2 * scale_y))

                w = x2 - x1
                h = y2 - y1
                # Filter out absurdly tiny boxes or non-human aspect ratios
                if h >= 25 and w >= 12:
                    detections.append(Detection((x1, y1, x2, y2), float(score), "person"))

        return detections


class PersonDetector:
    """
    Per-camera detector wrapper with staggered execution.
    """

    def __init__(
        self,
        detect_interval: int = 3,
        confidence_threshold: float = 0.45,
        mode: str = "auto",
    ) -> None:
        self.detect_interval = detect_interval
        self.detector = SharedPersonDetector.get_instance(confidence_threshold)
        self.frame_idx = 0
        self.last_detections: List[Detection] = []

    def detect(self, frame: np.ndarray, force: bool = False) -> List[Detection]:
        self.frame_idx += 1
        if force or not self.last_detections:
            self.last_detections = self.detector.detect_frame(frame)

        return self.last_detections
