"""
backend/analyzer.py
===================
Video Person Re-Identification & Multimodal Analysis Engine for Web Dashboard.
Extracts person detections, local tracking, OSNet deep embeddings, and torso
color/luminance gating to produce frame-by-frame Re-ID analysis.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# Ensure root SAFE_PASS is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from reid.detector import PersonDetector
from reid.tracker import SimpleTracker
from reid.embedding import ReIDFeatureExtractor, cosine_similarity
def analyze_torso_appearance(crop: np.ndarray) -> Tuple[float, float, str, np.ndarray]:
    """Extracts torso luminance and clothing category for verification gating."""
    if crop is None or crop.size == 0:
        return 0.0, 0.0, "unknown", np.zeros((10, 10, 3), dtype=np.uint8)
    h, w = crop.shape[:2]
    y1, y2 = int(0.15 * h), int(0.55 * h)
    x1, x2 = int(0.15 * w), int(0.85 * w)
    torso = crop[y1:y2, x1:x2]
    if torso.size == 0:
        torso = crop
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    v_mean = float(np.mean(hsv[:, :, 2]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    if v_mean < 80:
        cloth_type = "dark"
    elif v_mean > 170 and s_mean < 50:
        cloth_type = "white"
    else:
        cloth_type = "light"
    return v_mean, s_mean, cloth_type, torso


def check_clothing_compatibility(
    ref_v: float, ref_cloth: str, target_v: float, target_cloth: str
) -> Tuple[bool, str]:
    """Rejects intruder if clothing color/luminance strongly diverges from reference target."""
    if ref_cloth == "dark" and target_cloth in ("white", "light") and (target_v - ref_v) > 55:
        return True, f"Torso mismatch: Target is {target_cloth} (V={target_v:.0f}) vs Ref {ref_cloth} (V={ref_v:.0f})"
    if ref_cloth in ("white", "light") and target_cloth == "dark" and (ref_v - target_v) > 55:
        return True, f"Torso mismatch: Target is {target_cloth} (V={target_v:.0f}) vs Ref {ref_cloth} (V={ref_v:.0f})"
    return False, ""


CACHE_FILE = root_dir / "data" / "reid_cache.json"


def image_to_base64(img: np.ndarray, quality: int = 85) -> str:
    """Converts a BGR image to base64 JPEG data URL."""
    if img is None or img.size == 0:
        return ""
    success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return ""
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


class ReIDWebAnalyzer:
    """
    Manages video analysis and Re-ID cache for web dashboard.
    """

    def __init__(self, sim_threshold: float = 0.58) -> None:
        self.sim_threshold = sim_threshold
        self.detector = PersonDetector(detect_interval=4, confidence_threshold=0.48)
        self.extractor = ReIDFeatureExtractor(use_osnet=True)
        self.cache: Dict[str, Any] = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Analyzer] Warning loading cache: {e}")
        return {
            "reference_target": None,
            "cameras": {},
            "unknown_alerts": {}
        }

    def save_cache(self) -> None:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)
        print(f"[Analyzer] Saved Re-ID cache to {CACHE_FILE}")

    def analyze_reference_camera(self, video_path: str, cam_id: str = "cam1", cam_name: str = "Camera 1 (Walkway)") -> Dict[str, Any]:
        """
        Processes Camera 1 to identify and enroll the primary referral target.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        tracker = SimpleTracker(max_distance=120.0, max_missed=20, min_hits=2)
        candidate_tracks: Dict[int, Dict[str, Any]] = {}
        frames_data: Dict[str, List[Dict[str, Any]]] = {}

        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            f_idx += 1
            disp_frame = cv2.resize(frame, (960, 640))

            dets = self.detector.detect(disp_frame, force=(f_idx % 4 == 1))
            tracks = tracker.update(dets)

            frame_dets = []
            for trk in tracks:
                x1, y1, x2, y2 = trk.bbox
                h_box = y2 - y1
                w_box = x2 - x1
                if h_box < 60 or (h_box / max(1, w_box)) < 1.15:
                    continue

                crop = disp_frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                tid = trk.local_id
                if tid not in candidate_tracks:
                    emb = self.extractor.extract(disp_frame, trk.bbox)
                    v_mean, s_mean, cloth_type, _ = analyze_torso_appearance(crop)
                    candidate_tracks[tid] = {
                        "hits": 0,
                        "first_box": trk.bbox,
                        "last_box": trk.bbox,
                        "best_crop": crop.copy(),
                        "best_h": h_box,
                        "embedding": emb,
                        "v_mean": v_mean,
                        "s_mean": s_mean,
                        "cloth_type": cloth_type,
                        "best_frame": f_idx,
                    }
                else:
                    ct = candidate_tracks[tid]
                    ct["hits"] += 1
                    ct["last_box"] = trk.bbox
                    if (f_idx % 4 == 1):
                        emb = self.extractor.extract(disp_frame, trk.bbox)
                        v_mean, s_mean, cloth_type, _ = analyze_torso_appearance(crop)
                        ct["embedding"] = 0.85 * ct["embedding"] + 0.15 * emb
                        ct["embedding"] /= np.linalg.norm(ct["embedding"])
                        ct["v_mean"] = 0.85 * ct["v_mean"] + 0.15 * v_mean

                    if h_box > ct["best_h"]:
                        ct["best_h"] = h_box
                        ct["best_crop"] = crop.copy()
                        ct["best_frame"] = f_idx

                # Check if this track is the dominant moving human
                b0 = candidate_tracks[tid]["first_box"]
                bN = candidate_tracks[tid]["last_box"]
                disp = ((bN[0] - b0[0]) ** 2 + (bN[1] - b0[1]) ** 2) ** 0.5
                is_target = (candidate_tracks[tid]["hits"] >= 8 and disp >= 20.0 and h_box >= 120)

                frame_dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "track_id": tid,
                    "is_target": is_target,
                    "is_match": is_target,
                    "is_unknown": False,
                    "score": 1.0 if is_target else 0.5,
                    "label": "Person_01 [ENROLLED REF]" if is_target else f"Tracking #{tid}",
                    "color": "#00e5ff" if is_target else "#94a3b8",
                    "cloth_type": candidate_tracks[tid]["cloth_type"],
                    "v_mean": round(candidate_tracks[tid]["v_mean"], 1),
                })

            frames_data[str(f_idx)] = frame_dets
        cap.release()

        # Find primary dominant target
        best_tid = None
        max_score = 0
        for tid, ct in candidate_tracks.items():
            b0 = ct["first_box"]
            bN = ct["last_box"]
            disp = ((bN[0] - b0[0]) ** 2 + (bN[1] - b0[1]) ** 2) ** 0.5
            score = ct["hits"] * ct["best_h"] * (disp + 10)
            if disp >= 20.0 and score > max_score:
                max_score = score
                best_tid = tid

        if best_tid is not None:
            primary = candidate_tracks[best_tid]
            ref_profile = {
                "id": "Person_01",
                "cloth_type": primary["cloth_type"],
                "v_mean": round(primary["v_mean"], 1),
                "s_mean": round(primary["s_mean"], 1),
                "best_frame": primary["best_frame"],
                "source_cam_id": cam_id,
                "source_cam_name": cam_name,
                "crop_b64": image_to_base64(primary["best_crop"]),
                "embedding": primary["embedding"].tolist(),
            }
            self.cache["reference_target"] = ref_profile
            print(f"[Analyzer] Enrolled Primary Target: Person_01 ({primary['cloth_type']}, V={primary['v_mean']:.1f})")

        cam_data = {
            "id": cam_id,
            "name": cam_name,
            "path": video_path,
            "total_frames": f_idx,
            "fps": fps,
            "width": orig_w,
            "height": orig_h,
            "frames": frames_data,
            "summary": {
                "status": "REFERENCE",
                "score": 1.0,
                "frame": self.cache["reference_target"]["best_frame"] if self.cache.get("reference_target") else 1,
                "crop_b64": self.cache["reference_target"]["crop_b64"] if self.cache.get("reference_target") else "",
                "label": "ENROLLED REFERENCE TARGET",
            }
        }
        self.cache["cameras"][cam_id] = cam_data
        self.save_cache()
        return cam_data

    def analyze_investigation_camera(self, video_path: str, cam_id: str, cam_name: str) -> Dict[str, Any]:
        """
        Analyzes a subsequent camera against the enrolled reference target.
        Applies strict multimodal gating: confirms true matches, rejects unknown persons.
        """
        ref = self.cache.get("reference_target")
        if not ref or "embedding" not in ref:
            raise ValueError("Reference target must be enrolled first from Camera 1.")

        ref_emb = np.array(ref["embedding"], dtype=np.float32)
        ref_v = float(ref["v_mean"])
        ref_cloth = ref["cloth_type"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        tracker = SimpleTracker(max_distance=120.0, max_missed=20, min_hits=2)
        frames_data: Dict[str, List[Dict[str, Any]]] = {}

        best_match_crop = None
        best_match_score = 0.0
        best_match_frame = 0

        best_unknown_crop = None
        best_unknown_reason = ""
        best_unknown_frame = 0
        best_unknown_score = 0.0

        track_cache: Dict[int, Tuple[int, np.ndarray, float, float, str]] = {}

        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            f_idx += 1
            disp_frame = cv2.resize(frame, (960, 640))

            dets = self.detector.detect(disp_frame, force=(f_idx % 4 == 1))
            tracks = tracker.update(dets)

            frame_dets = []
            for trk in tracks:
                x1, y1, x2, y2 = trk.bbox
                h_box = y2 - y1
                w_box = x2 - x1
                if h_box < 60 or (h_box / max(1, w_box)) < 1.15:
                    continue

                crop = disp_frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                cached = track_cache.get(trk.local_id)
                if cached is None or (f_idx - cached[0]) >= 4:
                    emb = self.extractor.extract(disp_frame, trk.bbox)
                    v_mean, s_mean, cloth_type, _ = analyze_torso_appearance(crop)
                    track_cache[trk.local_id] = (f_idx, emb, v_mean, s_mean, cloth_type)
                else:
                    _, emb, v_mean, s_mean, cloth_type = cached

                sim = cosine_similarity(emb, ref_emb)
                is_incomp, reason = check_clothing_compatibility(ref_v, ref_cloth, v_mean, cloth_type)

                if is_incomp:
                    # DEFINITIVE REJECTION (Different person / black mask)
                    is_match = False
                    is_unknown = True
                    score = 0.0
                    label = f"UNKNOWN: Diff Clothes/Mask (0.0%)"
                    color = "#ef4444"  # Crimson Red

                    if best_unknown_crop is None or h_box > best_unknown_crop.shape[0]:
                        best_unknown_crop = crop.copy()
                        best_unknown_reason = reason
                        best_unknown_frame = f_idx
                        best_unknown_score = sim

                elif sim >= self.sim_threshold:
                    # CONFIRMED MATCH
                    is_match = True
                    is_unknown = False
                    score = sim
                    label = f"MATCH: Person_01 ({sim * 100:.1f}%)"
                    color = "#10b981"  # Emerald Green

                    if sim > best_match_score:
                        best_match_score = sim
                        best_match_crop = crop.copy()
                        best_match_frame = f_idx

                else:
                    # Scanning / low confidence
                    is_match = False
                    is_unknown = False
                    score = sim
                    label = f"Scanning... ({sim * 100:.1f}%)"
                    color = "#f59e0b"  # Amber

                frame_dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "track_id": trk.local_id,
                    "is_target": False,
                    "is_match": is_match,
                    "is_unknown": is_unknown,
                    "score": round(score, 3),
                    "label": label,
                    "color": color,
                    "cloth_type": cloth_type,
                    "v_mean": round(v_mean, 1),
                    "reason": reason if is_incomp else ""
                })

            frames_data[str(f_idx)] = frame_dets
        cap.release()

        # Build camera summary
        if best_match_score >= self.sim_threshold:
            summary = {
                "status": "MATCH",
                "score": round(best_match_score, 3),
                "frame": best_match_frame,
                "crop_b64": image_to_base64(best_match_crop),
                "label": f"CONFIRMED MATCH ({best_match_score*100:.1f}%)",
            }
        elif best_unknown_crop is not None:
            summary = {
                "status": "REJECTED_UNKNOWN",
                "score": 0.0,
                "frame": best_unknown_frame,
                "crop_b64": image_to_base64(best_unknown_crop),
                "reason": best_unknown_reason,
                "label": "REJECTED: UNKNOWN PERSON",
            }
            # Record security alert
            self.cache["unknown_alerts"][cam_id] = {
                "cam_id": cam_id,
                "cam_name": cam_name,
                "frame": best_unknown_frame,
                "reason": best_unknown_reason,
                "crop_b64": image_to_base64(best_unknown_crop),
                "raw_sim": round(best_unknown_score, 3)
            }
        else:
            summary = {
                "status": "NOT_OBSERVED",
                "score": 0.0,
                "frame": 0,
                "crop_b64": "",
                "label": "NOT OBSERVED",
            }

        cam_data = {
            "id": cam_id,
            "name": cam_name,
            "path": video_path,
            "total_frames": f_idx,
            "fps": fps,
            "width": orig_w,
            "height": orig_h,
            "frames": frames_data,
            "summary": summary
        }
        self.cache["cameras"][cam_id] = cam_data
        self.save_cache()
        return cam_data


def precompute_all_cctv(data_dir: str = "data/videos") -> None:
    """Pre-analyzes cam1, cam2, cam3, cam4 and creates persistent cache."""
    analyzer = ReIDWebAnalyzer(sim_threshold=0.58)

    cams = [
        ("cam1", "Camera 1 (Walkway)", f"{data_dir}/cam1.mp4"),
        ("cam2", "Camera 2 (Garage)", f"{data_dir}/cam2.mp4"),
        ("cam3", "Camera 3 (Staircase)", f"{data_dir}/cam3.mp4"),
        ("cam4", "Camera 4 (Staircase Intruder)", f"{data_dir}/cam4.mp4"),
    ]

    print("\n[Analyzer] Pre-computing Camera 1 Reference Enrollment...")
    analyzer.analyze_reference_camera(cams[0][2], cam_id=cams[0][0], cam_name=cams[0][1])

    for cid, cname, cpath in cams[1:]:
        print(f"\n[Analyzer] Pre-computing Re-ID for {cname} ({cpath})...")
        analyzer.analyze_investigation_camera(cpath, cam_id=cid, cam_name=cname)

    print("\n[Analyzer] ALL 4 CAMERAS SUCCESSFULLY PRE-COMPUTED & CACHED!")


if __name__ == "__main__":
    precompute_all_cctv()
