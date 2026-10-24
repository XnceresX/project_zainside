"""
face/webcam_live.py
===================
Real-Time Laptop Front Webcam Face & Appearance Multimodal Person Identification Engine
for SAFEPASS (Tracks P1 & P2).

Features:
  1. Live Webcam Feed from laptop front camera (cv2.VideoCapture).
  2. Multi-backend Face Detection (YuNet / SCRFD / Haar Cascade) with 5-point landmarks.
  3. ArcFace 512-Dimensional Deep Embeddings with 3D Multi-View Synthesis (P1).
  4. OSNet Re-ID 512-Dimensional Deep Body Appearance Embeddings (P2).
  5. HSV Upper & Lower Garment Clothing Colour Extraction & Semantic Tagging (P2).
  6. Learned Quality-Adaptive Dynamic Multimodal Score Fusion (FusionEngine).
  7. Interactive Multimodal Enrollment:
     - Press 'E' in live webcam feed to enroll Name + Appearance details (upper/lower clothing colors & notes).
     - Live auto-detection of current shirt & trousers colors to pre-fill the enrollment prompt!
     - CLI pre-enrollment via --enroll, --upper, --lower, --desc.
  8. High-Tech Surveillance HUD:
     - Neon Green corner-reticle on Face for Identified candidates, Amber for Unknown.
     - Torso / Body outline box displaying live detected clothing colors.
     - Dynamic Fusion score breakdown: Face %, Body %, Clothing Color %, and live weights.
  9. Dual Modes:
     - Native Desktop OpenCV GUI Window (cv2.imshow)
     - Standalone Web Streaming Server (--web) at http://localhost:8000

Usage:
  # Native Desktop Webcam Window:
  python -m face.webcam_live

  # Specify Camera Index:
  python -m face.webcam_live --cam 0

  # Enroll an existing photo upfront with clothing details:
  python -m face.webcam_live --enroll "Zain" path/to/my_photo.jpg --upper "navy" --lower "black" --desc "Navy hoodie, dark trousers"

  # Web Dashboard Mode:
  python -m face.webcam_live --web --port 8000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pickle
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# P1 Modules
from face.detector import FaceDetector, DetectedFace
from face.extractor import FaceFeatureExtractor, cosine_similarity
from face.multiview import (
    MultiViewEnrollmentEngine,
    MultiViewGalleryItem,
    synthesize_3d_perspective_view,
    synthesize_multiview_gallery,
)
from face.quality import estimate_face_quality

# P2 Modules
from appearance.extractor import AppearanceExtractor, appearance_cosine_similarity
from appearance.color import ClothingColorExtractor, ColorFeatures
from appearance.quality import BodyQualityEstimator
from fusion.engine import FusionEngine, _quality_to_weights
from contract.schema import ModalitySimilarities, QualityMetrics


# ---------------------------------------------------------------------------
# Torso & Body Crop Extraction from Face Bounding Box
# ---------------------------------------------------------------------------

def extract_body_crop_from_face(
    frame: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Intelligently estimates and extracts the person's torso and clothing bounding box
    directly from a detected face location in typical laptop webcam or CCTV viewports.

    Returns:
        (body_crop, (bx1, by1, bx2, by2))
    """
    fh, fw = frame.shape[:2]
    fx1, fy1, fx2, fy2 = face_bbox
    face_w = max(fx2 - fx1, 1)
    face_h = max(fy2 - fy1, 1)
    fcx = (fx1 + fx2) // 2

    # Shoulder width: ~2.4x face width
    half_bw = int(face_w * 1.25)
    bx1 = max(0, fcx - half_bw)
    bx2 = min(fw, fcx + half_bw)

    # Torso height: starts at lower chin/neck, extends downwards ~3.2x face height
    by1 = min(fh - 10, int(fy2 + 0.05 * face_h))
    by2 = min(fh, int(fy2 + 3.2 * face_h))

    # If seated close to webcam, adjust to ensure minimum height
    if (by2 - by1) < 60:
        by1 = max(0, int(fy1))
        by2 = fh

    body_crop = frame[by1:by2, bx1:bx2]
    if body_crop.size == 0 or body_crop.shape[0] < 10 or body_crop.shape[1] < 10:
        body_crop = frame[max(0, fy1):min(fh, fy2), max(0, fx1):min(fw, fx2)]
        bx1, by1, bx2, by2 = fx1, fy1, fx2, fy2

    return body_crop, (bx1, by1, bx2, by2)


# ---------------------------------------------------------------------------
# Multimodal Identity Dataclass
# ---------------------------------------------------------------------------

@dataclass
class MultimodalGalleryItem:
    """
    Represents an enrolled person with multi-view ArcFace embeddings, OSNet body re-ID
    embedding, clothing HSV color signatures, and human-readable appearance details.
    """
    name: str
    face_item: MultiViewGalleryItem
    height: str = ""                                           # e.g. "175 cm" or "5'9"
    appearance_embedding: Optional[np.ndarray] = None          # (512,) OSNet vector
    color_features: Optional[ColorFeatures] = None             # Upper/Lower HSV histograms & labels
    upper_clothing_color: str = "unknown"                      # e.g. "navy", "black", "red"
    lower_clothing_color: str = "unknown"                      # e.g. "black", "blue", "grey"
    appearance_desc: str = ""                                  # e.g. "Navy hoodie, dark trousers, glasses"
    enrolled_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def match_face(self, probe_crop: np.ndarray, extractor: FaceFeatureExtractor) -> Tuple[float, str]:
        """Matches face against multi-angle synthesized gallery views."""
        return self.face_item.match(probe_crop, extractor)

    def match_appearance(self, probe_body_emb: np.ndarray) -> Optional[float]:
        """Matches body re-ID embedding using cosine similarity."""
        if self.appearance_embedding is None or probe_body_emb is None:
            return None
        return appearance_cosine_similarity(probe_body_emb, self.appearance_embedding)

    def match_color(
        self,
        probe_features: ColorFeatures,
        color_extractor: ClothingColorExtractor,
    ) -> Optional[float]:
        """
        Matches clothing color histograms with semantic tag agreement bonus.
        """
        if self.color_features is None or probe_features is None:
            return None

        up_sim = color_extractor.hist_similarity(probe_features.upper_hist, self.color_features.upper_hist)
        low_sim = color_extractor.hist_similarity(probe_features.lower_hist, self.color_features.lower_hist)
        base_sim = 0.55 * up_sim + 0.45 * low_sim

        # Semantic color keyword bonus: if user entered specific clothing color name
        up_tag = self.upper_clothing_color.lower().strip()
        probe_up = probe_features.upper_label.lower().strip()
        if up_tag and up_tag != "unknown" and probe_up != "unknown":
            if up_tag in probe_up or probe_up in up_tag:
                base_sim = min(1.0, base_sim + 0.08)

        low_tag = self.lower_clothing_color.lower().strip()
        probe_low = probe_features.lower_label.lower().strip()
        if low_tag and low_tag != "unknown" and probe_low != "unknown":
            if low_tag in probe_low or probe_low in low_tag:
                base_sim = min(1.0, base_sim + 0.05)

        return float(round(base_sim, 4))


class GalleryUnpickler(pickle.Unpickler):
    """Custom unpickler that safely resolves classes regardless of whether run as __main__ or imported."""
    def find_class(self, module, name):
        if name == "MultimodalGalleryItem":
            return MultimodalGalleryItem
        if name == "MultiViewGalleryItem":
            return MultiViewGalleryItem
        if name == "ColorFeatures":
            return ColorFeatures
        return super().find_class(module, name)


class IdentityGallery:
    """
    Manages persistent enrolled identities with multimodal Biometric (Face) + Appearance (Body/Color)
    features stored in data/identities/.
    """

    def __init__(self, storage_dir: str = os.path.join("data", "identities")):
        self.storage_dir = storage_dir
        self.db_path = os.path.join(storage_dir, "gallery_registry.pkl")
        self.identities: Dict[str, MultimodalGalleryItem] = {}
        os.makedirs(storage_dir, exist_ok=True)
        self.load()

    def load(self):
        """Loads registered identities from disk, maintaining legacy compatibility."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    raw_data = GalleryUnpickler(f).load()

                self.identities = {}
                for k, v in raw_data.items():
                    if isinstance(v, MultimodalGalleryItem):
                        self.identities[k] = v
                    elif isinstance(v, MultiViewGalleryItem):
                        pid_name = getattr(v, "person_id", getattr(v, "name", k))
                        self.identities[k] = MultimodalGalleryItem(
                            name=pid_name,
                            face_item=v,
                            appearance_desc="Legacy face-only profile",
                        )
                    else:
                        print(f"[IdentityGallery] Unknown item type for key {k}: {type(v)}")

                print(f"[IdentityGallery] Loaded {len(self.identities)} registered identities from {self.db_path}")
                for pid, item in self.identities.items():
                    print(f"  * Enrolled: {pid} (Upper: {item.upper_clothing_color}, Lower: {item.lower_clothing_color})")
            except Exception as e:
                print(f"[IdentityGallery] Warning: Could not load gallery cache ({e}). Starting fresh.")
                self.identities = {}

    def save(self):
        """Saves registered identities to disk."""
        try:
            with open(self.db_path, "wb") as f:
                pickle.dump(self.identities, f)
            print(f"[IdentityGallery] Successfully saved {len(self.identities)} identities to disk.")
        except Exception as e:
            print(f"[IdentityGallery] Error saving gallery to disk: {e}")

    def enroll_person(
        self,
        person_name: str,
        face_img: np.ndarray,
        enrollment_engine: MultiViewEnrollmentEngine,
        appearance_extractor: Optional[AppearanceExtractor] = None,
        color_extractor: Optional[ClothingColorExtractor] = None,
        body_crop: Optional[np.ndarray] = None,
        upper_color: str = "unknown",
        lower_color: str = "unknown",
        appearance_desc: str = "",
    ) -> MultimodalGalleryItem:
        """
        Enrolls or updates a person with both Face biometrics (11 3D viewpoints) and Appearance
        (OSNet body embedding, clothing HSV histograms, and clothing color tags).
        """
        clean_name = person_name.strip()
        if not clean_name:
            raise ValueError("Person name cannot be empty.")

        # 1. Biometric Face Multi-View Synthesis
        face_item = enrollment_engine.enroll(clean_name, face_img)

        # 2. Body Appearance Re-ID Embedding
        app_emb = None
        if body_crop is not None and appearance_extractor is not None:
            try:
                app_emb = appearance_extractor.extract_embedding(body_crop)
            except Exception as ex:
                print(f"[IdentityGallery] Appearance extraction warning: {ex}")

        # 3. Clothing Color Histograms & Labels
        color_feats = None
        if body_crop is not None and color_extractor is not None:
            try:
                color_feats = color_extractor.extract(body_crop)
                if upper_color == "unknown" or not upper_color:
                    upper_color = color_feats.upper_label
                if lower_color == "unknown" or not lower_color:
                    lower_color = color_feats.lower_label
            except Exception as ex:
                print(f"[IdentityGallery] Color extraction warning: {ex}")

        # 4. Merge or Create Item
        if clean_name in self.identities:
            existing = self.identities[clean_name]
            # Accumulate face angles
            idx_offset = len(existing.face_item.embeddings)
            for k, v in face_item.embeddings.items():
                existing.face_item.embeddings[f"{k}_{idx_offset}"] = v
            for k, v in face_item.aligned_crops.items():
                existing.face_item.aligned_crops[f"{k}_{idx_offset}"] = v

            # Update appearance if provided
            if app_emb is not None:
                existing.appearance_embedding = app_emb
            if color_feats is not None:
                existing.color_features = color_feats
            if upper_color and upper_color != "unknown":
                existing.upper_clothing_color = upper_color
            if lower_color and lower_color != "unknown":
                existing.lower_clothing_color = lower_color
            if appearance_desc:
                existing.appearance_desc = appearance_desc

            item = existing
            print(f"[IdentityGallery] Updated multimodal profile for '{clean_name}'.")
        else:
            item = MultimodalGalleryItem(
                name=clean_name,
                face_item=face_item,
                appearance_embedding=app_emb,
                color_features=color_feats,
                upper_clothing_color=upper_color,
                lower_clothing_color=lower_color,
                appearance_desc=appearance_desc,
            )
            self.identities[clean_name] = item

        # Save thumbnails for surveillance review
        face_thumb_path = os.path.join(self.storage_dir, f"{clean_name.replace(' ', '_')}.jpg")
        try:
            cv2.imwrite(face_thumb_path, face_img)
        except Exception:
            pass

        if body_crop is not None:
            body_thumb_path = os.path.join(self.storage_dir, f"{clean_name.replace(' ', '_')}_body.jpg")
            try:
                cv2.imwrite(body_thumb_path, body_crop)
            except Exception:
                pass

        self.save()
        return item

    def match(
        self,
        probe_face_crop: np.ndarray,
        probe_body_crop: np.ndarray,
        face_extractor: FaceFeatureExtractor,
        appearance_extractor: AppearanceExtractor,
        color_extractor: ClothingColorExtractor,
        fusion_engine: FusionEngine,
        q_face: float = 0.8,
        q_body: float = 0.8,
        threshold: float = 0.70,
        min_margin: float = 0.03,
    ) -> Dict[str, Any]:
        """
        Matches a probe against all enrolled identities using quality-adaptive multimodal fusion.
        Computes Face similarity + Appearance similarity + Clothing Color similarity.
        """
        if not self.identities:
            return {
                "person_name": "UNKNOWN",
                "fused_score": 0.0,
                "is_match": False,
                "face_score": 0.0,
                "appearance_score": 0.0,
                "color_score": 0.0,
                "weights": (1.0, 0.0, 0.0),
                "matched_view": "none",
                "upper_color": "unknown",
                "lower_color": "unknown",
                "height": "",
                "desc": "",
            }

        # Extract probe body appearance and color once
        probe_body_emb = None
        if probe_body_crop is not None and probe_body_crop.size > 0:
            try:
                probe_body_emb = appearance_extractor.extract_embedding(probe_body_crop)
            except Exception:
                pass

        probe_color_feats = None
        if probe_body_crop is not None and probe_body_crop.size > 0:
            try:
                probe_color_feats = color_extractor.extract(probe_body_crop)
            except Exception:
                pass

        scores: List[Dict[str, Any]] = []

        for person_name, item in self.identities.items():
            # Face similarity
            face_sim, matched_view = item.match_face(probe_face_crop, face_extractor)

            # Appearance similarity
            app_sim = item.match_appearance(probe_body_emb) if probe_body_emb is not None else None

            # Color similarity
            color_sim = (
                item.match_color(probe_color_feats, color_extractor)
                if probe_color_feats is not None
                else None
            )

            # Adaptive Fusion
            fusion_res = fusion_engine.fuse(
                face_sim=face_sim,
                app_sim=app_sim,
                color_sim=color_sim,
                q_face=q_face,
                q_body=q_body,
            )
            fused = fusion_res.fused_score

            # Active weights
            weights = fusion_engine._compute_weights(
                q_face,
                q_body,
                face_available=(face_sim is not None),
                app_available=(app_sim is not None),
                color_available=(color_sim is not None),
            )

            scores.append({
                "person_name": person_name,
                "fused_score": fused,
                "face_score": face_sim if face_sim is not None else 0.0,
                "appearance_score": app_sim if app_sim is not None else 0.0,
                "color_score": color_sim if color_sim is not None else 0.0,
                "weights": weights,
                "matched_view": matched_view,
                "upper_color": item.upper_clothing_color,
                "lower_color": item.lower_clothing_color,
                "height": getattr(item, "height", ""),
                "desc": item.appearance_desc,
            })

        # Sort descending by fused score
        scores.sort(key=lambda s: s["fused_score"], reverse=True)

        best = scores[0]
        second_score = scores[1]["fused_score"] if len(scores) > 1 else 0.0
        margin_ok = (len(scores) == 1) or ((best["fused_score"] - second_score) >= min_margin)

        if best["fused_score"] >= threshold and margin_ok:
            best["is_match"] = True
            return best
        else:
            best["is_match"] = False
            best["person_name"] = "UNKNOWN"
            return best

    def clear(self):
        """Clears all enrolled identities."""
        self.identities.clear()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        print("[IdentityGallery] Cleared all identities.")


# ---------------------------------------------------------------------------
# HUD Rendering Utilities
# ---------------------------------------------------------------------------

def draw_corner_rect(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    corner_len: int = 18,
):
    """Draws sleek sci-fi corner brackets instead of plain full rectangles."""
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    cl = min(corner_len, w // 3, h // 3)

    # Semi-transparent box background fill
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)

    # Thin full border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Top-Left Corner
    cv2.line(img, (x1, y1), (x1 + cl, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + cl), color, thickness, cv2.LINE_AA)

    # Top-Right Corner
    cv2.line(img, (x2, y1), (x2 - cl, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + cl), color, thickness, cv2.LINE_AA)

    # Bottom-Left Corner
    cv2.line(img, (x1, y2), (x1 + cl, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - cl), color, thickness, cv2.LINE_AA)

    # Bottom-Right Corner
    cv2.line(img, (x2, y2), (x2 - cl, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - cl), color, thickness, cv2.LINE_AA)


def draw_body_rect(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (255, 200, 0),
    label: str = "APPEARANCE",
):
    """Draws a dashed/bracketed torso region box indicating clothing appearance tracking."""
    x1, y1, x2, y2 = bbox
    # Thin translucent border
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Label tab at bottom-right or top
    cv2.putText(img, label, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def draw_landmarks(img: np.ndarray, landmarks: Optional[np.ndarray]):
    """Draws 5 anatomical facial landmarks with connecting alignment lines."""
    if landmarks is None or len(landmarks) < 5:
        return

    pts = landmarks[:5].astype(int)
    # Eye line (cyan)
    cv2.line(img, tuple(pts[0]), tuple(pts[1]), (255, 255, 0), 1, cv2.LINE_AA)
    # Nose to mid-eyes
    mid_eyes = ((pts[0] + pts[1]) // 2).astype(int)
    cv2.line(img, tuple(mid_eyes), tuple(pts[2]), (255, 255, 0), 1, cv2.LINE_AA)
    # Mouth line (amber)
    cv2.line(img, tuple(pts[3]), tuple(pts[4]), (0, 200, 255), 1, cv2.LINE_AA)

    # Nodes
    colors = [
        (0, 255, 255),  # Right eye
        (0, 255, 255),  # Left eye
        (0, 255, 120),  # Nose
        (0, 165, 255),  # Right mouth
        (0, 165, 255),  # Left mouth
    ]
    for pt, c in zip(pts, colors):
        cv2.circle(img, tuple(pt), 4, c, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(pt), 6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud_header(
    frame: np.ndarray,
    fps: float,
    enrolled_count: int,
    backend_name: str,
    blink_state: bool,
):
    """Draws top surveillance status telemetry bar."""
    h, w = frame.shape[:2]

    # Glass header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (15, 18, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, 48), (w, 48), (0, 200, 255), 1, cv2.LINE_AA)

    # Title & Live indicator
    live_col = (0, 0, 255) if blink_state else (100, 100, 100)
    cv2.circle(frame, (18, 24), 6, live_col, -1, cv2.LINE_AA)
    cv2.putText(frame, "SAFEPASS MULTIMODAL SURVEILLANCE", (32, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(frame, "| CAM 0: LAPTOP WEBCAM", (340, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 175, 190), 1, cv2.LINE_AA)

    # Backend info & enrolled count & FPS
    stats_text = f"FPS: {fps:4.1f}  |  Enrolled: {enrolled_count}  |  FUSION: FACE + OSNET + COLOR"
    (tw, th), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, stats_text, (w - tw - 16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)


def draw_hud_footer(frame: np.ndarray, alert_msg: Optional[str] = None):
    """Draws bottom control shortcuts guide and notification messages."""
    h, w = frame.shape[:2]

    # Footer bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 34), (w, h), (15, 18, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, h - 34), (w, h - 34), (0, 150, 200), 1, cv2.LINE_AA)

    if alert_msg:
        cv2.putText(frame, alert_msg, (16, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 120), 2, cv2.LINE_AA)
    else:
        guide = "[E] Enroll Face + Appearance  |  [S] Snapshot  |  [M] Mirror  |  [L] Landmarks  |  [C] Clear  |  [Q] Quit"
        cv2.putText(frame, guide, (16, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 195, 210), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Interactive Multimodal Enrollment Prompt
# ---------------------------------------------------------------------------

def prompt_enrollment_multimodal(
    auto_upper: str = "unknown",
    auto_lower: str = "unknown",
) -> Optional[Dict[str, str]]:
    """
    Displays a sleek dark-themed Tkinter dialog to input:
      - Person's Name
      - Upper Clothing Color (pre-filled with auto-detected value)
      - Lower Clothing Color (pre-filled with auto-detected value)
      - Appearance Details / Notes
    Falls back to console if GUI fails.
    """
    try:
        import tkinter as tk
        from tkinter import ttk

        result = {}

        root = tk.Tk()
        root.title("SAFEPASS - Multimodal Identity & Appearance Enrollment")
        root.geometry("460x360")
        root.configure(bg="#0b0f19")
        root.attributes("-topmost", True)
        root.resizable(False, False)

        # Style
        font_head = ("Segoe UI", 12, "bold")
        font_lbl = ("Segoe UI", 10)
        font_entry = ("Segoe UI", 10)

        # Header
        lbl_head = tk.Label(
            root,
            text="ENROLL NEW MULTIMODAL IDENTITY",
            font=font_head,
            fg="#00e5ff",
            bg="#0b0f19",
            pady=12,
        )
        lbl_head.pack()

        form_frame = tk.Frame(root, bg="#111827", padx=20, pady=15, relief="ridge", bd=1)
        form_frame.pack(padx=20, pady=5, fill="both", expand=True)

        # Name
        tk.Label(form_frame, text="Full Name / Person ID:", font=font_lbl, fg="#94a3b8", bg="#111827").grid(row=0, column=0, sticky="w", pady=4)
        name_var = tk.StringVar(value="")
        entry_name = tk.Entry(form_frame, textvariable=name_var, font=font_entry, bg="#1f2937", fg="#ffffff", insertbackground="white", width=28)
        entry_name.grid(row=0, column=1, sticky="w", pady=4)
        entry_name.focus_set()

        # Upper Clothing
        tk.Label(form_frame, text="Upper Garment (Shirt/Top):", font=font_lbl, fg="#94a3b8", bg="#111827").grid(row=1, column=0, sticky="w", pady=4)
        upper_var = tk.StringVar(value=auto_upper.title() if auto_upper != "unknown" else "")
        entry_upper = tk.Entry(form_frame, textvariable=upper_var, font=font_entry, bg="#1f2937", fg="#38bdf8", insertbackground="white", width=28)
        entry_upper.grid(row=1, column=1, sticky="w", pady=4)

        # Lower Clothing
        tk.Label(form_frame, text="Lower Garment (Pants/Skirt):", font=font_lbl, fg="#94a3b8", bg="#111827").grid(row=2, column=0, sticky="w", pady=4)
        lower_var = tk.StringVar(value=auto_lower.title() if auto_lower != "unknown" else "")
        entry_lower = tk.Entry(form_frame, textvariable=lower_var, font=font_entry, bg="#1f2937", fg="#38bdf8", insertbackground="white", width=28)
        entry_lower.grid(row=2, column=1, sticky="w", pady=4)

        # Appearance Notes
        tk.Label(form_frame, text="Appearance Notes / Tags:", font=font_lbl, fg="#94a3b8", bg="#111827").grid(row=3, column=0, sticky="w", pady=4)
        desc_var = tk.StringVar(value="")
        entry_desc = tk.Entry(form_frame, textvariable=desc_var, font=font_entry, bg="#1f2937", fg="#ffffff", insertbackground="white", width=28)
        entry_desc.grid(row=3, column=1, sticky="w", pady=4)

        # Note
        lbl_hint = tk.Label(
            form_frame,
            text=f"* Auto-detected colors: Upper [{auto_upper}] | Lower [{auto_lower}]",
            font=("Segoe UI", 8, "italic"),
            fg="#00e5ff",
            bg="#111827",
        )
        lbl_hint.grid(row=4, column=0, columnspan=2, sticky="w", pady=8)

        def on_submit():
            name = name_var.get().strip()
            if name:
                result["name"] = name
                result["upper"] = upper_var.get().strip().lower() or auto_upper
                result["lower"] = lower_var.get().strip().lower() or auto_lower
                result["desc"] = desc_var.get().strip()
                root.destroy()

        def on_cancel():
            root.destroy()

        btn_frame = tk.Frame(root, bg="#0b0f19", pady=10)
        btn_frame.pack()

        btn_save = tk.Button(
            btn_frame,
            text="Save & Enroll",
            command=on_submit,
            bg="#00e5ff",
            fg="#0b0f19",
            font=("Segoe UI", 10, "bold"),
            padx=16,
            pady=4,
            relief="flat",
        )
        btn_save.pack(side="left", padx=10)

        btn_cancel = tk.Button(
            btn_frame,
            text="Cancel",
            command=on_cancel,
            bg="#374151",
            fg="#e5e7eb",
            font=("Segoe UI", 10),
            padx=16,
            pady=4,
            relief="flat",
        )
        btn_cancel.pack(side="left", padx=10)

        root.bind("<Return>", lambda event: on_submit())
        root.bind("<Escape>", lambda event: on_cancel())

        root.mainloop()

        return result if result else None

    except Exception as e:
        print(f"\n[ENROLLMENT PROMPT] Tkinter error ({e}), falling back to console.")
        print("\n" + "=" * 55)
        print("  SAFEPASS ENROLLMENT (CONSOLE FALLBACK)")
        print(f"  Auto-detected: Upper [{auto_upper}] | Lower [{auto_lower}]")
        print("=" * 55)
        try:
            print("Enter Person Name: ", end="", flush=True)
            name = sys.stdin.readline().strip()
            if not name:
                return None
            print(f"Upper Clothing Color (default: {auto_upper}): ", end="", flush=True)
            upper = sys.stdin.readline().strip() or auto_upper
            print(f"Lower Clothing Color (default: {auto_lower}): ", end="", flush=True)
            lower = sys.stdin.readline().strip() or auto_lower
            print("Appearance Notes: ", end="", flush=True)
            desc = sys.stdin.readline().strip()
            return {"name": name, "upper": upper.lower(), "lower": lower.lower(), "desc": desc}
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Core Webcam Processor (Tracks P1 & P2 Multimodal)
# ---------------------------------------------------------------------------

class WebcamFaceIdentifier:
    """
    Coordinates real-time video capture, face detection, ArcFace embedding extraction,
    OSNet body appearance re-ID, clothing color profiling, and dynamic multimodal fusion.
    """

    def __init__(
        self,
        cam_index: int = 0,
        threshold: float = 0.70,
        mirror: bool = True,
        show_landmarks: bool = True,
    ):
        self.cam_index = cam_index
        self.threshold = threshold
        self.mirror = mirror
        self.show_landmarks = show_landmarks

        print("=" * 70)
        print("  SAFEPASS Tracks P1 & P2 -- Multimodal Face & Appearance Re-ID Engine")
        print("=" * 70)
        print("[Init] Initializing Face Detector & ArcFace Feature Extractor...")
        self.detector = FaceDetector(detection_confidence=0.55)
        self.face_extractor = FaceFeatureExtractor()
        self.enrollment_engine = MultiViewEnrollmentEngine(self.detector, self.face_extractor)

        print("[Init] Initializing OSNet Body Re-ID & Clothing Color Extractors...")
        self.appearance_extractor = AppearanceExtractor()
        self.color_extractor = ClothingColorExtractor()
        self.body_quality_estimator = BodyQualityEstimator()
        self.fusion_engine = FusionEngine(mode="dynamic")

        self.gallery = IdentityGallery()

        self.alert_message: Optional[str] = None
        self.alert_expiry: float = 0.0

        # Snapshot folder
        os.makedirs(os.path.join("data", "snapshots"), exist_ok=True)

    def set_alert(self, msg: str, duration_sec: float = 3.5):
        self.alert_message = msg
        self.alert_expiry = time.time() + duration_sec

    def process_frame(
        self,
        frame: np.ndarray,
        enroll_flag: bool = False,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Runs detection, quality metrics, multimodal feature extraction, and fusion on a single frame.
        Returns the annotated HUD frame and list of detected candidate metadata.
        """
        if self.mirror:
            frame = cv2.flip(frame, 1)

        display = frame.copy()
        faces = self.detector.detect(frame)
        candidates_meta = []

        now = time.time()
        enrolled_this_frame = False

        for idx, face in enumerate(faces):
            fx1, fy1, fx2, fy2 = face.bbox
            fw_box = fx2 - fx1
            fh_box = fy2 - fy1

            # 1. Face Quality Estimation
            yaw_val = face.pose.yaw if face.pose else 0.0
            pitch_val = face.pose.pitch if face.pose else 0.0
            face_quality = estimate_face_quality(
                face_image=face.face_crop,
                detection_conf=face.detection_confidence,
                yaw=yaw_val,
                pitch=pitch_val,
                landmark_visibility=face.landmark_visibility,
            )
            q_face = face_quality.recommended_face_weight

            # 2. Extract Torso / Body Crop
            body_crop, body_bbox = extract_body_crop_from_face(frame, face.bbox)

            # 3. Body Quality Estimation
            body_q_dict = self.body_quality_estimator.estimate(body_crop, bbox=body_bbox)
            q_body = body_q_dict.get("q_body", 0.5)

            # 4. Live Detected Clothing Colors
            color_feats = self.color_extractor.extract(body_crop)
            live_upper_color = color_feats.upper_label
            live_lower_color = color_feats.lower_label

            # 5. Multimodal Matching & Quality-Adaptive Fusion
            match_res = self.gallery.match(
                probe_face_crop=face.face_crop,
                probe_body_crop=body_crop,
                face_extractor=self.face_extractor,
                appearance_extractor=self.appearance_extractor,
                color_extractor=self.color_extractor,
                fusion_engine=self.fusion_engine,
                q_face=q_face,
                q_body=q_body,
                threshold=self.threshold,
            )

            identity = match_res["person_name"]
            fused_score = match_res["fused_score"]
            is_identified = match_res["is_match"]
            face_score = match_res["face_score"]
            app_score = match_res["appearance_score"]
            col_score = match_res["color_score"]
            w_f, w_a, w_c = match_res["weights"]

            # Handle live enrollment keypress
            if enroll_flag and not enrolled_this_frame:
                enrolled_this_frame = True
                prompt_data = prompt_enrollment_multimodal(
                    auto_upper=live_upper_color,
                    auto_lower=live_lower_color,
                )
                if prompt_data:
                    name = prompt_data["name"]
                    upper = prompt_data.get("upper", live_upper_color)
                    lower = prompt_data.get("lower", live_lower_color)
                    desc = prompt_data.get("desc", "")
                    try:
                        self.gallery.enroll_person(
                            person_name=name,
                            face_img=face.face_crop,
                            enrollment_engine=self.enrollment_engine,
                            appearance_extractor=self.appearance_extractor,
                            color_extractor=self.color_extractor,
                            body_crop=body_crop,
                            upper_color=upper,
                            lower_color=lower,
                            appearance_desc=desc,
                        )
                        self.set_alert(f"[SUCCESS] Enrolled '{name}' (Upper: {upper.title()}, Lower: {lower.title()})!")
                        identity = name
                        fused_score = 1.0
                        is_identified = True
                    except Exception as err:
                        self.set_alert(f"[ERROR] Enrollment failed: {err}")

            # Color scheme: Neon Green for Identified, Amber/Orange for Unknown
            if is_identified:
                box_color = (0, 255, 100)      # Neon Emerald
                badge_bg = (10, 40, 15)
                text_col = (0, 255, 120)
                ht_tag = f" [{match_res.get('height')}]" if match_res.get("height") else ""
                status_label = f"ID: {identity.upper()}{ht_tag} [{fused_score * 100:.1f}% MATCH]"
            else:
                box_color = (0, 165, 255)      # Amber / Orange
                badge_bg = (30, 20, 10)
                text_col = (0, 200, 255)
                status_label = f"UNKNOWN ({fused_score * 100:.1f}%)"

            # Draw Torso / Body outline box
            draw_body_rect(
                display,
                body_bbox,
                color=(0, 200, 255) if is_identified else (180, 140, 0),
                label=f"CLOTHING: {live_upper_color.upper()} / {live_lower_color.upper()}",
            )

            # Draw Corner Box on Face
            draw_corner_rect(display, face.bbox, box_color, thickness=2)

            # Draw Landmarks if enabled
            if self.show_landmarks and face.landmarks is not None:
                draw_landmarks(display, face.landmarks)

            # Identification Badge Pill
            badge_h = 26
            badge_y1 = max(fy1 - badge_h - 4, 52)
            badge_y2 = badge_y1 + badge_h

            (label_w, _), _ = cv2.getTextSize(status_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
            badge_w = max(label_w + 16, 140)

            # Semi-transparent badge background
            pill_overlay = display.copy()
            cv2.rectangle(pill_overlay, (fx1, badge_y1), (fx1 + badge_w, badge_y2), badge_bg, -1)
            cv2.addWeighted(pill_overlay, 0.85, display, 0.15, 0, display)
            cv2.rectangle(display, (fx1, badge_y1), (fx1 + badge_w, badge_y2), box_color, 1, cv2.LINE_AA)

            # Badge text
            cv2.putText(
                display,
                status_label,
                (fx1 + 8, badge_y2 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                text_col,
                2,
                cv2.LINE_AA,
            )

            # Multimodal breakdown banner under face box
            telemetry_text = (
                f"Face: {face_score * 100:.0f}% (w:{w_f:.2f}) | "
                f"Body: {app_score * 100:.0f}% (w:{w_a:.2f}) | "
                f"Color: {col_score * 100:.0f}% (w:{w_c:.2f})"
            )
            cv2.putText(
                display,
                telemetry_text,
                (fx1, min(fy2 + 18, display.shape[0] - 42)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (200, 230, 255),
                1,
                cv2.LINE_AA,
            )

            # Secondary detail text: clothing details or blur
            detail_line = (
                f"Upper: {live_upper_color.title()} | Lower: {live_lower_color.title()} | "
                f"Blur: {face_quality.blur_score:.2f} | Yaw: {yaw_val:+2.0f} deg"
            )
            cv2.putText(
                display,
                detail_line,
                (fx1, min(fy2 + 34, display.shape[0] - 38)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (140, 180, 210),
                1,
                cv2.LINE_AA,
            )

            candidates_meta.append({
                "person_id": identity,
                "fused_score": fused_score,
                "is_match": is_identified,
                "face_score": face_score,
                "appearance_score": app_score,
                "color_score": col_score,
                "weights": (w_f, w_a, w_c),
                "bbox": face.bbox,
                "body_bbox": body_bbox,
                "detected_upper_color": live_upper_color,
                "detected_lower_color": live_lower_color,
                "quality": {
                    "blur": face_quality.blur_score,
                    "occlusion": face_quality.occlusion_score,
                    "q_face": q_face,
                    "q_body": q_body,
                },
            })

        return display, candidates_meta

    def run_gui(self):
        """
        Runs native desktop OpenCV interactive webcam window with direct hardware capture.
        """
        print(f"\n[Camera] Opening video capture on camera index {self.cam_index} (DirectShow backend)...")
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[Camera] DirectShow fallback to default backend...")
            cap = cv2.VideoCapture(self.cam_index)

        if not cap.isOpened():
            print(f"[FATAL ERROR] Could not open camera {self.cam_index}. Check that webcam is connected & not in use.")
            return

        # Attempt to set HD resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Stream initialized: {actual_w}x{actual_h} px")

        win_name = "SAFEPASS P1 & P2 - Live Face & Appearance Multimodal Re-ID"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, actual_w, actual_h)

        fps = 0.0
        frame_count = 0
        t0 = time.time()
        enroll_requested = False

        print("\n" + "=" * 70)
        print("  CAMERA FEED ACTIVE -- WINDOW CONTROLS:")
        print("    [E]   Enroll Detected Face + Clothing Appearance with Name")
        print("    [S]   Save Snapshot (Surveillance Frame)")
        print("    [M]   Toggle Horizontal Mirroring")
        print("    [L]   Toggle 5-Point Anatomical Landmarks")
        print("    [C]   Clear All Enrolled Identities")
        print("    [Q]   Exit Application")
        print("=" * 70 + "\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARN] Failed to read webcam frame.")
                    time.sleep(0.05)
                    continue

                frame_count += 1
                dt = time.time() - t0
                if dt >= 0.5:
                    fps = frame_count / dt
                    frame_count = 0
                    t0 = time.time()

                # Process detection & multimodal identification
                vis_frame, _ = self.process_frame(frame, enroll_flag=enroll_requested)
                enroll_requested = False

                # Top & Bottom HUD
                blink = (int(time.time() * 2) % 2 == 0)
                draw_hud_header(
                    vis_frame,
                    fps=fps,
                    enrolled_count=len(self.gallery.identities),
                    backend_name=self.detector.backend,
                    blink_state=blink,
                )

                active_alert = self.alert_message if time.time() < self.alert_expiry else None
                draw_hud_footer(vis_frame, alert_msg=active_alert)

                cv2.imshow(win_name, vis_frame)

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                    print("\n[SAFEPASS] Exiting webcam session.")
                    break
                elif key in (ord('e'), ord('E')):
                    enroll_requested = True
                elif key in (ord('s'), ord('S')):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_path = os.path.join("data", "snapshots", f"snap_{ts}.jpg")
                    cv2.imwrite(snap_path, vis_frame)
                    self.set_alert(f"[SAVED] Snapshot recorded to {snap_path}")
                    print(f"[Snapshot] Saved: {snap_path}")
                elif key in (ord('m'), ord('M')):
                    self.mirror = not self.mirror
                    self.set_alert(f"Mirror mode: {'ON' if self.mirror else 'OFF'}")
                elif key in (ord('l'), ord('L')):
                    self.show_landmarks = not self.show_landmarks
                    self.set_alert(f"Landmarks display: {'ON' if self.show_landmarks else 'OFF'}")
                elif key in (ord('c'), ord('C')):
                    self.gallery.clear()
                    self.set_alert("[GALLERY] All registered identities cleared.")

        finally:
            cap.release()
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Web Streaming Dashboard Mode (FastAPI + MJPEG)
# ---------------------------------------------------------------------------

def run_web_dashboard(identifier: WebcamFaceIdentifier, host: str = "0.0.0.0", port: int = 8000):
    """
    Spawns a FastAPI server serving live MJPEG stream and interactive multimodal web interface.
    """
    import uvicorn
    from fastapi import FastAPI, Response, Form
    from fastapi.responses import HTMLResponse, StreamingResponse

    app = FastAPI(title="SAFEPASS Live Multimodal Surveillance")

    cap = cv2.VideoCapture(identifier.cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(identifier.cam_index)

    last_raw_frame = None
    lock = threading.Lock()

    def frame_generator():
        fps = 0.0
        fc = 0
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.04)
                continue

            fc += 1
            dt = time.time() - t0
            if dt >= 0.5:
                fps = fc / dt
                fc = 0
                t0 = time.time()

            vis, _ = identifier.process_frame(frame)
            blink = (int(time.time() * 2) % 2 == 0)
            draw_hud_header(
                vis,
                fps=fps,
                enrolled_count=len(identifier.gallery.identities),
                backend_name=identifier.detector.backend,
                blink_state=blink,
            )
            active_alert = identifier.alert_message if time.time() < identifier.alert_expiry else None
            draw_hud_footer(vis, alert_msg=active_alert)

            with lock:
                nonlocal last_raw_frame
                last_raw_frame = frame.copy()

            success, buffer = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    @app.get("/video_feed")
    def video_feed():
        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/api/enroll")
    def api_enroll(
        name: str = Form(...),
        upper_color: str = Form(""),
        lower_color: str = Form(""),
        appearance_desc: str = Form(""),
    ):
        with lock:
            if last_raw_frame is None:
                return {"status": "error", "message": "No webcam frame available."}
            faces = identifier.detector.detect(last_raw_frame)
            if not faces:
                return {"status": "error", "message": "No face detected in current webcam frame."}

            face = faces[0]
            body_crop, _ = extract_body_crop_from_face(last_raw_frame, face.bbox)

            try:
                identifier.gallery.enroll_person(
                    person_name=name,
                    face_img=face.face_crop,
                    enrollment_engine=identifier.enrollment_engine,
                    appearance_extractor=identifier.appearance_extractor,
                    color_extractor=identifier.color_extractor,
                    body_crop=body_crop,
                    upper_color=upper_color,
                    lower_color=lower_color,
                    appearance_desc=appearance_desc,
                )
                identifier.set_alert(f"[WEB] Enrolled '{name}' (Upper: {upper_color or 'Auto'}, Lower: {lower_color or 'Auto'})!")
                return {"status": "success", "person": name}
            except Exception as e:
                return {"status": "error", "message": str(e)}

    @app.get("/", response_class=HTMLResponse)
    def index():
        identities_rows = []
        for name, item in identifier.gallery.identities.items():
            thumb = f"/thumbs/{name.replace(' ', '_')}.jpg"
            identities_rows.append(
                f"""
                <div class="identity-card">
                  <div class="identity-name">{name}</div>
                  <div class="identity-tag">Upper: <span style="color:#38bdf8">{item.upper_clothing_color.title()}</span></div>
                  <div class="identity-tag">Lower: <span style="color:#38bdf8">{item.lower_clothing_color.title()}</span></div>
                  {f'<div class="identity-desc">{item.appearance_desc}</div>' if item.appearance_desc else ''}
                </div>
                """
            )
        id_html = "".join(identities_rows) if identities_rows else "<p style='color:#64748b'>No registered identities yet. Use the form below to enroll.</p>"

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
          <title>SAFEPASS Multimodal Live Surveillance</title>
          <style>
            body {{ background: #0b0f19; color: #f1f5f9; font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; padding: 20px; text-align: center; }}
            .container {{ max-width: 1100px; margin: 0 auto; }}
            h1 {{ color: #00e5ff; margin-bottom: 5px; font-size: 26px; }}
            .subtitle {{ color: #94a3b8; font-size: 14px; margin-bottom: 15px; }}
            .video-box {{ border: 2px solid #00e5ff; border-radius: 8px; overflow: hidden; box-shadow: 0 0 25px rgba(0, 229, 255, 0.25); margin: 15px 0; }}
            .panel {{ background: #111827; padding: 20px; border-radius: 8px; border: 1px solid #1f2937; margin-top: 15px; text-align: left; }}
            .form-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
            .form-group {{ display: flex; flex-direction: column; gap: 6px; }}
            label {{ font-size: 13px; color: #94a3b8; }}
            input[type=text] {{ padding: 10px; border-radius: 6px; border: 1px solid #374151; background: #1f2937; color: white; font-size: 14px; }}
            button {{ grid-column: span 2; padding: 12px; border: none; border-radius: 6px; background: #00e5ff; color: #0b0f19; font-weight: bold; cursor: pointer; font-size: 15px; transition: 0.2s; }}
            button:hover {{ background: #38bdf8; box-shadow: 0 0 15px rgba(0, 229, 255, 0.4); }}
            .identities-grid {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 15px; }}
            .identity-card {{ background: #1e293b; padding: 10px 16px; border-radius: 8px; border: 1px solid #334155; text-align: left; min-width: 160px; }}
            .identity-name {{ font-weight: bold; color: #f1f5f9; font-size: 15px; margin-bottom: 4px; }}
            .identity-tag {{ font-size: 12px; color: #94a3b8; }}
            .identity-desc {{ font-size: 11px; color: #64748b; margin-top: 4px; font-style: italic; }}
          </style>
        </head>
        <body>
          <div class="container">
            <h1>SAFEPASS MULTIMODAL SURVEILLANCE</h1>
            <div class="subtitle">Real-Time Laptop Webcam Face (P1) + Appearance Re-ID & Clothing Color Fusion (P2)</div>
            <div class="video-box">
              <img src="/video_feed" style="width:100%; display:block;" />
            </div>
            <div class="panel">
              <h3 style="color:#00e5ff; margin-top:0;">Enroll Identity & Appearance Details</h3>
              <form id="enrollForm" onsubmit="event.preventDefault(); enroll();" class="form-grid">
                <div class="form-group">
                  <label for="nameInput">Person Full Name *</label>
                  <input type="text" id="nameInput" placeholder="e.g. Zain" required />
                </div>
                <div class="form-group">
                  <label for="upperInput">Upper Garment Color (Shirt / Hoodie)</label>
                  <input type="text" id="upperInput" placeholder="e.g. Navy Blue, Black, Red (or leave blank to auto-detect)" />
                </div>
                <div class="form-group">
                  <label for="lowerInput">Lower Garment Color (Pants / Jeans)</label>
                  <input type="text" id="lowerInput" placeholder="e.g. Black, Blue Denim (or leave blank to auto-detect)" />
                </div>
                <div class="form-group">
                  <label for="descInput">Appearance Notes / Tags</label>
                  <input type="text" id="descInput" placeholder="e.g. Glasses, sports jacket" />
                </div>
                <button type="submit">Capture & Enroll Person with Appearance</button>
              </form>

              <h3 style="margin-top:25px; color:#38bdf8">Enrolled Multimodal Identities ({len(identifier.gallery.identities)})</h3>
              <div class="identities-grid">{id_html}</div>
            </div>
          </div>
          <script>
            async function enroll() {{
              const name = document.getElementById('nameInput').value.trim();
              if(!name) return;
              const fd = new FormData();
              fd.append('name', name);
              fd.append('upper_color', document.getElementById('upperInput').value.trim());
              fd.append('lower_color', document.getElementById('lowerInput').value.trim());
              fd.append('appearance_desc', document.getElementById('descInput').value.trim());

              const res = await fetch('/api/enroll', {{ method: 'POST', body: fd }});
              const data = await res.json();
              if(data.status === 'success') {{
                alert('Enrolled ' + data.person + ' successfully with appearance profile!');
                location.reload();
              }} else {{
                alert('Error: ' + data.message);
              }}
            }}
          </script>
        </body>
        </html>
        """

    print(f"\n[SAFEPASS Web Dashboard] Starting multimodal live stream at: http://localhost:{port}")
    uvicorn.run(app, host=host, port=port)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SAFEPASS Live Webcam Multimodal Face & Appearance Identification Engine (Tracks P1 & P2)."
    )
    parser.add_argument("--cam", type=int, default=0, help="Camera device index (default: 0 for laptop front webcam).")
    parser.add_argument("--threshold", type=float, default=0.70, help="Multimodal fused score match threshold (default: 0.70).")
    parser.add_argument("--no-mirror", action="store_true", help="Disable horizontal selfie mirror flip.")
    parser.add_argument("--enroll", nargs=2, metavar=("NAME", "IMAGE_PATH"), help="Pre-enroll person from an image file upfront.")
    parser.add_argument("--photos", nargs="+", metavar="PATH", help="Additional face photos (different angles) to capture full facial structure.")
    parser.add_argument("--height", type=str, default="175 cm", help="Person height (e.g. '175 cm' or '5 ft 9 in').")
    parser.add_argument("--upper", type=str, default="unknown", help="Optional upper clothing color for CLI enrollment (e.g. 'black').")
    parser.add_argument("--lower", type=str, default="unknown", help="Optional lower clothing color for CLI enrollment (e.g. 'blue').")
    parser.add_argument("--desc", type=str, default="", help="Optional appearance description for CLI enrollment.")
    parser.add_argument("--clear", action="store_true", help="Forget/wipe previous test identities before enrolling.")
    parser.add_argument("--web", action="store_true", help="Launch web streaming dashboard server.")
    parser.add_argument("--port", type=int, default=8000, help="Web dashboard port (default: 8000).")
    parser.add_argument("--gui", action="store_true", help="Launch live webcam GUI window immediately after CLI enrollment.")

    args = parser.parse_args()

    identifier = WebcamFaceIdentifier(
        cam_index=args.cam,
        threshold=args.threshold,
        mirror=not args.no_mirror,
    )

    if args.clear:
        identifier.gallery.clear()
        print("[Gallery] Cleared previous gallery identities.")

    # Optional pre-enrollment from command line
    if args.enroll:
        name, img_path = args.enroll
        all_photo_paths = [img_path]
        if args.photos:
            all_photo_paths.extend(args.photos)

        primary_img = cv2.imread(img_path)
        if primary_img is None:
            print(f"[ERROR] Could not decode image: {img_path}")
            sys.exit(1)

        print(f"[Enrollment] Enrolling '{name}' (Height: {args.height}) with {len(all_photo_paths)} face photo(s)...")
        body_crop, _ = extract_body_crop_from_face(primary_img, (0, 0, primary_img.shape[1], primary_img.shape[0]))
        item = identifier.gallery.enroll_person(
            person_name=name,
            face_img=primary_img,
            enrollment_engine=identifier.enrollment_engine,
            appearance_extractor=identifier.appearance_extractor,
            color_extractor=identifier.color_extractor,
            body_crop=body_crop,
            upper_color=args.upper,
            lower_color=args.lower,
            appearance_desc=args.desc,
        )
        item.height = args.height

        # Accumulate remaining photos if provided
        for extra_p in all_photo_paths[1:]:
            extra_img = cv2.imread(extra_p)
            if extra_img is not None:
                sub_item = identifier.enrollment_engine.enroll(name, extra_img)
                offset = len(item.face_item.embeddings)
                for k, v in sub_item.embeddings.items():
                    item.face_item.embeddings[f"{k}_{offset}"] = v
                for k, v in sub_item.aligned_crops.items():
                    item.face_item.aligned_crops[f"{k}_{offset}"] = v
                print(f"  + Added extra angle photo: {extra_p}")

        identifier.gallery.save()
        print(f"[Enrollment] Successfully registered '{name}' ({args.height}) with {len(item.face_item.embeddings)} multi-angle facial embeddings into gallery.")

    # Run in selected mode
    if args.web:
        run_web_dashboard(identifier, port=args.port)
    elif args.enroll and not args.gui:
        print("[Enrollment] Finished. Launch live webcam identification with: run_webcam.bat or run_studio.bat")
    else:
        identifier.run_gui()


if __name__ == "__main__":
    main()
