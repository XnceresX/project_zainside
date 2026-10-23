"""
face/video_search.py
====================
Multi-Camera Video Ingestion & Person Identification Engine for SAFEPASS (Track P1).

Processes multiple asynchronous CCTV video files (.mp4, .avi, etc.):
  1. Enrolls target person from 1 single photo using 3D Multi-View synthesis.
  2. Ingests multiple video streams (sampled at configurable FPS).
  3. Detects faces, filters false alarms with anatomical checks, estimates quality.
  4. Matches detected faces against multi-view gallery embeddings.
  5. Outputs:
     - Chronological cross-camera trajectory timeline.
     - Exported candidate detection records conforming to contract.schema.Candidate.
     - (Optional) Annotated output videos with surveillance HUD and tracking overlays.

Usage:
  # 1. Generate sample CCTV video feeds to test:
  python -m face.video_search --create-demo-videos

  # 2. Search for enrolled person across multiple video files:
  python -m face.video_search --target data/samples/sample_face_1.jpg --videos data/videos/*.mp4
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from contract.schema import Candidate
from face.detector import FaceDetector, DetectedFace
from face.extractor import FaceFeatureExtractor, cosine_similarity
from face.multiview import MultiViewEnrollmentEngine, MultiViewGalleryItem, synthesize_3d_perspective_view
from face.quality import estimate_face_quality


class VideoSearchEngine:
    """
    Coordinates multi-camera video feed search for target identities.
    """

    def __init__(
        self,
        threshold: float = 0.65,
        sample_interval_sec: float = 0.5,
        output_dir: str = "eval/video_results",
    ):
        self.threshold = threshold
        self.sample_interval_sec = sample_interval_sec
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print("[VideoSearchEngine] Initializing SCRFD-2.5G & ArcFace TTA...")
        self.detector = FaceDetector()
        self.extractor = FaceFeatureExtractor()
        self.enrollment_engine = MultiViewEnrollmentEngine(self.detector, self.extractor)

    def enroll_target(self, person_id: str, image_path: str) -> MultiViewGalleryItem:
        """Enrolls a single frontal image into a 3D multi-view gallery item."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target reference image not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image at: {image_path}")

        print(f"[Enrollment] Enrolling target '{person_id}' from {image_path}...")
        item = self.enrollment_engine.enroll(person_id, img)
        print(f"[Enrollment] Generated viewpoints: {list(item.embeddings.keys())}")
        return item

    def process_video_stream(
        self,
        video_path: str,
        gallery_item: MultiViewGalleryItem,
        camera_id: str = "CAM-01",
        camera_name: str = "Surveillance Feed",
        start_time_iso: Optional[str] = None,
        save_annotated_video: bool = False,
    ) -> List[Candidate]:
        """
        Processes a single CCTV video stream and returns all matching candidates.
        """
        if not os.path.exists(video_path):
            print(f"[WARN] Video not found: {video_path}")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        frame_step = max(1, int(round(fps * self.sample_interval_sec)))

        print(f"\n[VideoSearch] Searching in {camera_id} ('{camera_name}'):")
        print(f"  File: {video_path} ({total_frames} frames, {fps:.1f} FPS, {duration_sec:.1f}s)")
        print(f"  Sampling 1 frame every {self.sample_interval_sec:.2f}s (step={frame_step} frames)...")

        base_dt = datetime.fromisoformat(start_time_iso) if start_time_iso else datetime.now()

        writer = None
        if save_annotated_video:
            out_vid_path = os.path.join(self.output_dir, f"annotated_{camera_id}.mp4")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_vid_path, fourcc, fps, (w, h))

        candidates: List[Candidate] = []
        frame_idx = 0
        match_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_sec = frame_idx / fps
            current_time = base_dt + timedelta(seconds=current_sec)
            iso_time = current_time.isoformat()

            # Process frame if on sample step
            if frame_idx % frame_step == 0:
                detections = self.detector.detect(frame)

                for det_i, det in enumerate(detections):
                    q_score = estimate_face_quality(
                        face_image=det.face_crop,
                        detection_conf=det.detection_confidence,
                        yaw=det.pose.yaw,
                        pitch=det.pose.pitch,
                        landmark_visibility=det.landmark_visibility,
                    )

                    # Match with Multi-View Gallery
                    sim, best_view = gallery_item.match(det.face_crop, self.extractor)

                    if sim >= self.threshold:
                        match_count += 1
                        cand_id = f"{camera_id}_f{frame_idx}_d{det_i}"

                        # Save candidate face crop
                        crop_filename = f"cand_{cand_id}.jpg"
                        crop_path = os.path.join(self.output_dir, crop_filename)
                        cv2.imwrite(crop_path, det.face_crop)

                        cand = Candidate(
                            case_id=gallery_item.person_id,
                            candidate_id=cand_id,
                            camera_id=camera_id,
                            camera_name=camera_name,
                            timestamp=iso_time,
                            frame_crop_path=crop_path,
                            face_score=float(sim),
                            appearance_score=None,
                            color_score=None,
                            q_face=float(q_score.blur_score),
                            q_body=1.0,
                            fused_score=float(sim),
                            topology_score=1.0,
                            is_spatially_valid=True,
                        )
                        candidates.append(cand)

                        # Annotate frame
                        x1, y1, x2, y2 = det.bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{gallery_item.person_id}: {sim:.2f} ({best_view})"
                        cv2.putText(
                            frame, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                        )
                    else:
                        # Draw non-matching face in gray
                        x1, y1, x2, y2 = det.bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)

            if writer is not None:
                writer.write(frame)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()
            print(f"  Annotated video saved: {out_vid_path}")

        print(f"  Matches found in {camera_id}: {match_count} detections")
        return candidates

    def search_all_videos(
        self,
        video_configs: List[Dict[str, str]],
        target_image_path: str,
        target_person_id: str = "PERSON-001",
    ) -> List[Candidate]:
        """
        Runs multi-camera cross-search across all provided video streams.
        Reconstructs the chronological motion trajectory of the person.
        """
        gallery_item = self.enroll_target(target_person_id, target_image_path)
        all_candidates: List[Candidate] = []

        print("\n" + "=" * 65)
        print(f"  SCANNING {len(video_configs)} CCTV VIDEO STREAMS FOR {target_person_id}")
        print("=" * 65)

        for cfg in video_configs:
            cands = self.process_video_stream(
                video_path=cfg["video_path"],
                gallery_item=gallery_item,
                camera_id=cfg["camera_id"],
                camera_name=cfg["camera_name"],
                start_time_iso=cfg.get("start_time"),
                save_annotated_video=cfg.get("save_annotated", False),
            )
            all_candidates.extend(cands)

        # Sort candidates chronologically to build trajectory
        all_candidates.sort(key=lambda c: c.timestamp)

        print("\n" + "=" * 65)
        print(f"  RECONSTRUCTED MOTION TRAJECTORY ACROSS CAMERAS ({len(all_candidates)} hits)")
        print("=" * 65)

        if not all_candidates:
            print("  No matching sightings found across video feeds.")
        else:
            for idx, c in enumerate(all_candidates):
                print(f"  [{idx+1:02d}] {c.timestamp[11:19]} | {c.camera_id} ({c.camera_name:<20}) | "
                      f"Sim: {c.fused_score:.3f} | Quality: {c.q_face:.2f} | Status: VERIFIED")

        return all_candidates


def create_demo_surveillance_videos(
    output_dir: str = "data/videos",
    duration_sec: int = 4,
    fps: int = 20,
):
    """
    Generates 3 synthetic CCTV camera video clips (.mp4) using actual sample faces
    to allow instant out-of-the-box demonstration of multi-camera search.
    
    Cameras generated:
      - cam_01_entrance.mp4   (Subject enters facing front)
      - cam_02_corridor.mp4   (Subject walks down corridor at left yaw angle)
      - cam_03_escalator.mp4  (Subject exits up escalator at right yaw angle)
    """
    os.makedirs(output_dir, exist_ok=True)

    face1_path = os.path.join("data", "samples", "sample_face_1.jpg")
    face2_path = os.path.join("data", "samples", "sample_face_2.jpg")

    if not os.path.exists(face1_path):
        print(f"[ERROR] Sample face not found: {face1_path}")
        return []

    target_face = cv2.imread(face1_path)
    stranger_face = cv2.imread(face2_path) if os.path.exists(face2_path) else None

    cameras = [
        {
            "id": "CAM-01",
            "name": "Main Entrance Gate",
            "filename": "cam_01_entrance.mp4",
            "yaw": 0.0,
            "has_target": True,
            "has_stranger": False,
        },
        {
            "id": "CAM-02",
            "name": "North Corridor Gate",
            "filename": "cam_02_corridor.mp4",
            "yaw": -30.0,
            "has_target": True,
            "has_stranger": True,
        },
        {
            "id": "CAM-03",
            "name": "East Escalator Terminal",
            "filename": "cam_03_escalator.mp4",
            "yaw": 30.0,
            "has_target": True,
            "has_stranger": False,
        },
    ]

    w, h = 640, 480
    total_frames = duration_sec * fps
    generated_configs = []

    print(f"\n[MockGenerator] Generating {len(cameras)} synthetic CCTV video feeds in '{output_dir}'...")

    for c_idx, cam in enumerate(cameras):
        vid_path = os.path.join(output_dir, cam["filename"])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

        # Synthetic camera background (hallway simulation)
        bg = np.full((h, w, 3), 30 + c_idx * 15, dtype=np.uint8)
        # Perspective floor lines
        cv2.line(bg, (0, h), (w // 2 - 50, h // 2), (60, 60, 60), 2)
        cv2.line(bg, (w, h), (w // 2 + 50, h // 2), (60, 60, 60), 2)

        # Prepare target face rotated to camera's perspective
        target_warped = synthesize_3d_perspective_view(target_face, yaw_degrees=cam["yaw"])

        stranger_warped = None
        if cam["has_stranger"] and stranger_face is not None:
            stranger_warped = synthesize_3d_perspective_view(stranger_face, yaw_degrees=15.0)

        # Generate movement across frames
        for f in range(total_frames):
            frame = bg.copy()
            t = f / float(total_frames)

            # Target person moving forward in frame
            if cam["has_target"]:
                sz = int(240 + t * 80)  # scales from 240px to 320px
                px = int(w * 0.20 + t * 40)
                py = int(h * 0.15 + t * 30)
                t_resized = cv2.resize(target_warped, (sz, sz))
                # Paste into frame
                y2 = min(py + sz, h)
                x2 = min(px + sz, w)
                rh, rw = y2 - py, x2 - px
                if rh > 0 and rw > 0:
                    frame[py:y2, px:x2] = t_resized[:rh, :rw]

            # Stranger person in background
            if stranger_warped is not None:
                ssz = 180
                sx = int(w * 0.65 - t * 25)
                sy = int(h * 0.18)
                s_resized = cv2.resize(stranger_warped, (ssz, ssz))
                y2 = min(sy + ssz, h)
                x2 = min(sx + ssz, w)
                rh, rw = y2 - sy, x2 - sx
                if rh > 0 and rw > 0:
                    frame[sy:y2, sx:x2] = s_resized[:rh, :rw]

            # Surveillance HUD overlay
            cv2.putText(
                frame, f"{cam['id']}: {cam['name'].upper()} [REC]",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            time_str = (datetime.now() + timedelta(seconds=c_idx*120 + f/fps)).strftime("%H:%M:%S.%f")[:-4]
            cv2.putText(
                frame, f"TIMESTAMP: {time_str} | YAW: {cam['yaw']} DEG",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
            )

            # Subtle CCTV sensor noise with proper clipping
            noise = np.random.normal(0, 3, frame.shape).astype(np.float32)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            writer.write(frame)

        writer.release()
        print(f"  Created: {vid_path} ({cam['name']})")

        generated_configs.append({
            "camera_id": cam["id"],
            "camera_name": cam["name"],
            "video_path": vid_path,
            "start_time": (datetime.now() + timedelta(minutes=c_idx * 2)).isoformat(),
            "save_annotated": True,
        })

    return generated_configs


def main():
    parser = argparse.ArgumentParser(description="SAFEPASS Multi-Camera Video Search Engine (Track P1)")
    parser.add_argument("--target", type=str, default="data/samples/sample_face_1.jpg", help="Path to single frontal reference photo")
    parser.add_argument("--target-id", type=str, default="STUDENT-4021", help="Person ID / Case ID")
    parser.add_argument("--videos", nargs="*", default=None, help="Paths or globs of CCTV video files")
    parser.add_argument("--create-demo-videos", action="store_true", help="Generate 3 sample multi-camera CCTV video files and test")
    parser.add_argument("--threshold", type=float, default=0.65, help="Cosine similarity match threshold")
    parser.add_argument("--sample-interval", type=float, default=0.5, help="Sample rate in seconds (default: 0.5s)")
    args = parser.parse_args()

    engine = VideoSearchEngine(threshold=args.threshold, sample_interval_sec=args.sample_interval)

    video_configs = []

    if args.create_demo_videos or not args.videos:
        video_configs = create_demo_surveillance_videos()
    elif args.videos:
        for vid_arg in args.videos:
            for v_path in glob.glob(vid_arg):
                base = os.path.splitext(os.path.basename(v_path))[0]
                video_configs.append({
                    "camera_id": base[:10].upper(),
                    "camera_name": f"Camera Feed {base}",
                    "video_path": v_path,
                })

    if not video_configs:
        print("[ERROR] No video feeds provided. Use --create-demo-videos or pass --videos <path>.")
        sys.exit(1)

    candidates = engine.search_all_videos(
        video_configs=video_configs,
        target_image_path=args.target,
        target_person_id=args.target_id,
    )

    print(f"\n[DONE] Successfully identified {len(candidates)} sightings across {len(video_configs)} video streams.")


if __name__ == "__main__":
    main()
