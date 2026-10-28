"""
cctv/reid_cctv.py
=================
Multi-Camera & Video CCTV Person Re-Identification Engine.
Features:
  1. Accurate Neural Person Detection (FasterRCNN MobileNetV3 FPN) - Rejects car parts,
     doors, pavements, and static artifacts.
  2. Multi-View Mode: View all cameras tiled in grid ('0'), or single full-screen cameras ('1', '2', '3', 'TAB').
  3. Interactive Pause & Cross-Camera Comparison Inspector: Pause ('SPACE') to inspect
     detected persons side-by-side across cameras and verify similarity scores.
  4. Frame-by-Frame Stepping ('N' or Right Arrow while paused).
  5. Deep OSNet 512-dim Body Re-ID Embeddings & Persistent Global Identity.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# Ensure root SAFE_PASS is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from reid import (
    GlobalIDManager,
    PersonDetector,
    ReIDFeatureExtractor,
    SimpleTracker,
    load_zones,
    zone_confidence,
)


def get_color_for_id(identity_str: str) -> Tuple[int, int, int]:
    """Generates a stable, vibrant BGR color for any ID string."""
    hash_val = hash(identity_str)
    r = (hash_val & 0xFF0000) >> 16
    g = (hash_val & 0x00FF00) >> 8
    b = hash_val & 0x0000FF
    return (max(50, b % 255), max(180, (g + 100) % 255), max(80, (r + 50) % 255))


class MultiCameraCCTVReIDEngine:
    """
    Coordinates multi-camera CCTV streams with neural person detection,
    cross-camera verification inspector, and multi-view switching.
    """

    def __init__(
        self,
        sources: List[str],
        camera_ids: Optional[List[str]] = None,
        zone_file: Optional[str] = None,
        similarity_threshold: float = 0.52,
        handoff_threshold: float = 0.48,
        use_osnet: bool = True,
        display_width: int = 640,
        display_height: int = 480,
    ) -> None:
        self.sources = sources if sources else ["0"]
        self.display_width = display_width
        self.display_height = display_height

        if camera_ids and len(camera_ids) == len(self.sources):
            self.camera_ids = camera_ids
        else:
            self.camera_ids = [f"cam{i+1}" for i in range(len(self.sources))]

        self.zones = load_zones(zone_file)

        # Initialize per-camera neural detectors and trackers
        print("[CCTV Engine] Initializing Neural Person Detectors & Trackers...")
        self.detectors: Dict[str, PersonDetector] = {
            cam_id: PersonDetector(detect_interval=3, confidence_threshold=0.50)
            for cam_id in self.camera_ids
        }
        self.trackers: Dict[str, SimpleTracker] = {
            cam_id: SimpleTracker(max_distance=120.0, max_missed=20, min_hits=2)
            for cam_id in self.camera_ids
        }
        self.extractor = ReIDFeatureExtractor(use_osnet=use_osnet)
        self.global_ids = GlobalIDManager(
            similarity_threshold=similarity_threshold,
            handoff_threshold=handoff_threshold,
            id_prefix="Person_",
        )

        # Storage for latest person crops per Global ID (for Cross-Camera Inspector)
        self.person_snapshots: Dict[str, Dict[str, Any]] = {}

        self.captures: Dict[str, cv2.VideoCapture] = {}

        # View mode: -1 = All Cameras Grid, 0 = Cam 1, 1 = Cam 2, etc.
        self.view_mode = -1

    def _open_captures(self) -> bool:
        for cam_id, src in zip(self.camera_ids, self.sources):
            idx = int(src) if src.isdigit() else src
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                print(f"[CCTV] Warning: Could not open source '{src}' for {cam_id}")
            self.captures[cam_id] = cap
        return len(self.captures) > 0

    def draw_hud(
        self,
        frame: np.ndarray,
        camera_id: str,
        frame_index: int,
        tracks: list,
        paused: bool = False,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # Draw handoff zones if defined
        for zone in self.zones.get(camera_id, []):
            zx1, zy1, zx2, zy2 = zone.xyxy
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 180, 0), 2)
            cv2.putText(
                frame,
                f"ZONE: {zone.zone_id}",
                (zx1, max(20, zy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )

        # Top Bar Overlay
        header = frame.copy()
        cv2.rectangle(header, (0, 0), (w, 36), (15, 20, 28), -1)
        cv2.addWeighted(header, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, 36), (w, 36), (0, 200, 255), 1)

        pause_tag = " [PAUSED - INSPECTOR ACTIVE]" if paused else ""
        cam_text = f"CAM: {camera_id.upper()} | FRAME: {frame_index:05d} | TRACKS: {len(tracks)}{pause_tag}"
        cv2.putText(frame, cam_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)

        return frame

    def render_inspector_modal(self, canvas: np.ndarray) -> np.ndarray:
        """
        Overlays an interactive Cross-Camera Inspection & Verification Panel
        when the video is paused.
        """
        ch, cw = canvas.shape[:2]
        panel_h = min(200, int(ch * 0.35))
        panel_y = ch - panel_h

        # Semi-transparent bottom inspector sheet
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, panel_y), (cw, ch), (10, 14, 23), -1)
        cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0, canvas)
        cv2.line(canvas, (0, panel_y), (cw, panel_y), (0, 229, 255), 2)

        # Title bar
        title_str = (
            f"PAUSED · CROSS-CAMERA PERSON VERIFICATION INSPECTOR  |  "
            f"[SPACE] Resume  |  [N/Right] Step Next Frame  |  [0] Grid  |  [1-3] Single Cam  |  [S] Save Snapshot"
        )
        cv2.putText(canvas, title_str, (15, panel_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 229, 255), 1, cv2.LINE_AA)

        # Display side-by-side thumbnails of active identities
        recent_ids = list(self.person_snapshots.keys())[-6:]
        thumb_w = 110
        thumb_h = panel_h - 45
        start_x = 15

        if not recent_ids:
            cv2.putText(canvas, "No confirmed person identities tracked yet.", (15, panel_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1)
            return canvas

        for gid in recent_ids:
            snap_data = self.person_snapshots[gid]
            crop = snap_data.get("crop")
            cam = snap_data.get("camera_id", "cam?")
            dets = snap_data.get("total_detections", 1)

            card_x1 = start_x
            card_x2 = card_x1 + thumb_w
            card_y1 = panel_y + 35
            card_y2 = panel_y + 35 + thumb_h

            if card_x2 > cw - 10:
                break

            color = get_color_for_id(gid)
            cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), (25, 33, 48), -1)
            cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), color, 1)

            if crop is not None and crop.size > 0:
                img_h = thumb_h - 40
                scaled_crop = cv2.resize(crop, (thumb_w - 6, max(10, img_h)))
                canvas[card_y1 + 3 : card_y1 + 3 + scaled_crop.shape[0], card_x1 + 3 : card_x1 + 3 + scaled_crop.shape[1]] = scaled_crop

            # Label badge
            cv2.putText(canvas, gid, (card_x1 + 5, card_y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Seen: {cam} (Hits:{dets})", (card_x1 + 5, card_y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 210, 240), 1, cv2.LINE_AA)

            start_x += thumb_w + 12

        return canvas

    def run(self, show: bool = True, max_frames: Optional[int] = None) -> None:
        """Runs the real-time CCTV person re-identification pipeline."""
        if not self._open_captures():
            print("[CCTV] Error: No camera sources could be opened.")
            return

        win_title = "SAFEPASS CCTV Person Re-ID Stream (Multi-Camera & Verification)"
        if show:
            cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

        frame_index = 0
        print("\n" + "=" * 65)
        print("  SAFEPASS CCTV PERSON RE-IDENTIFICATION RUNNING")
        print("  - Mode: Neural Person Tracking & Cross-Camera Re-ID")
        print(f"  - Cameras: {', '.join(self.camera_ids)}")
        print("  - View Controls:")
        print("      [0] or [A]: View ALL Cameras Together (Grid)")
        print("      [1], [2], [3]: View Specific Camera Full-Screen")
        print("      [TAB] or [V]: Cycle through views")
        print("      [SPACE]: PAUSE & open Cross-Camera Verification Inspector")
        print("      [N] or [Right]: Step 1 frame forward while paused")
        print("      [S]: Save Snapshot  |  [Q]: Exit")
        print("=" * 65 + "\n")

        paused = False
        step_forward = False
        last_views: List[np.ndarray] = []

        while True:
            if not paused or step_forward:
                step_forward = False
                views: List[np.ndarray] = []
                active_any = False

                for cam_idx, cam_id in enumerate(self.camera_ids):
                    cap = self.captures.get(cam_id)
                    if cap is None or not cap.isOpened():
                        blank = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
                        cv2.putText(blank, f"{cam_id}: OFFLINE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        views.append(blank)
                        continue

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue

                    active_any = True
                    # Resize to standard processing/display size
                    frame = cv2.resize(frame, (self.display_width, self.display_height))

                    # 1. Detect Persons using Neural Detector (staggered across cameras for 3x speed)
                    run_neural = (cam_idx == (frame_index % len(self.camera_ids)))
                    detections = self.detectors[cam_id].detect(frame, force=run_neural)

                    # 2. Local Track Update (with min_hits filtering)
                    tracks = self.trackers[cam_id].update(detections)

                    # 3. HUD base
                    self.draw_hud(frame, cam_id, frame_index, tracks, paused)

                    # 4. Extract ReID features & assign persistent Global IDs
                    for track in tracks:
                        embedding = self.extractor.extract(frame, track.bbox)
                        zconf = zone_confidence(cam_id, track.bbox, self.zones)
                        global_id = self.global_ids.assign(
                            camera_id=cam_id,
                            local_id=track.local_id,
                            embedding=embedding,
                            frame_index=frame_index,
                            zone_confidence=zconf,
                        )

                        color = get_color_for_id(global_id)
                        x1, y1, x2, y2 = track.bbox

                        # Save crop snapshot for Cross-Camera Inspector
                        crop = frame[max(0, y1):min(self.display_height, y2), max(0, x1):min(self.display_width, x2)]
                        if crop.size > 0:
                            self.person_snapshots[global_id] = {
                                "crop": crop.copy(),
                                "camera_id": cam_id,
                                "local_id": track.local_id,
                                "frame": frame_index,
                                "total_detections": self.global_ids.identities[global_id].total_detections if global_id in self.global_ids.identities else 1,
                            }

                        # Person bounding box with rounded corner accents
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Label badge
                        badge_label = f"{global_id} ({cam_id}:{track.local_id})"
                        if zconf > 0.1:
                            badge_label += f" Z:{zconf:.2f}"

                        # Badge pill background
                        (tw, th), _ = cv2.getTextSize(badge_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        by1 = max(0, y1 - th - 8)
                        by2 = y1
                        cv2.rectangle(frame, (x1, by1), (x1 + tw + 10, by2), color, -1)
                        cv2.putText(
                            frame,
                            badge_label,
                            (x1 + 5, by2 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                    views.append(frame)

                if not active_any and not paused:
                    print("[CCTV] All video feeds completed.")
                    break

                if views:
                    last_views = views
                frame_index += 1
                if max_frames and frame_index >= max_frames:
                    break

            # Assemble Canvas based on active view mode
            if last_views:
                if self.view_mode >= 0 and self.view_mode < len(last_views):
                    # Single camera full-screen view
                    canvas = cv2.resize(last_views[self.view_mode], (self.display_width * 2, self.display_height * 2))
                    cam_title = f"[SINGLE VIEW: {self.camera_ids[self.view_mode].upper()}] (Press [0] for All Cameras Grid)"
                    cv2.putText(canvas, cam_title, (20, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 255), 2)
                else:
                    # All cameras tiled together in grid
                    if len(last_views) == 1:
                        canvas = last_views[0]
                    elif len(last_views) == 2:
                        canvas = np.hstack(last_views)
                    elif len(last_views) <= 4:
                        row1 = np.hstack(last_views[:2])
                        row2 = np.hstack(last_views[2:4]) if len(last_views) == 4 else np.hstack([last_views[2], np.zeros_like(last_views[0])])
                        canvas = np.vstack([row1, row2])
                    else:
                        canvas = np.hstack(last_views[:3])

                # If paused, render the Cross-Camera Person Inspector
                if paused:
                    canvas = self.render_inspector_modal(canvas)

                if show:
                    cv2.imshow(win_title, canvas)

            # Keyboard Input Handling
            if show:
                key = cv2.waitKey(20 if paused else 1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key == 32:  # SPACE = Toggle Pause & Inspector
                    paused = not paused
                    print(f"[CCTV] {'PAUSED · Verification Inspector Active' if paused else 'RESUMED'}")
                elif paused and key in (ord("n"), ord("N"), 83, 2555904):  # 'N' or Right arrow to step
                    step_forward = True
                elif key in (ord("0"), ord("a"), ord("A")):  # '0' = All together
                    self.view_mode = -1
                    print("[CCTV View] Switched to ALL CAMERAS GRID")
                elif key == ord("1") and len(self.camera_ids) >= 1:
                    self.view_mode = 0
                    print(f"[CCTV View] Switched to SINGLE VIEW: {self.camera_ids[0]}")
                elif key == ord("2") and len(self.camera_ids) >= 2:
                    self.view_mode = 1
                    print(f"[CCTV View] Switched to SINGLE VIEW: {self.camera_ids[1]}")
                elif key == ord("3") and len(self.camera_ids) >= 3:
                    self.view_mode = 2
                    print(f"[CCTV View] Switched to SINGLE VIEW: {self.camera_ids[2]}")
                elif key in (9, ord("v"), ord("V")):  # TAB or 'V' to cycle
                    self.view_mode = (self.view_mode + 2) % (len(self.camera_ids) + 1) - 1
                    mode_name = "ALL CAMERAS" if self.view_mode == -1 else self.camera_ids[self.view_mode]
                    print(f"[CCTV View] Cycled view to: {mode_name}")
                elif key in (ord("s"), ord("S")):
                    snap_name = f"cctv_snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(snap_name, canvas)
                    print(f"[CCTV] Snapshot saved to {snap_name}")

        # Cleanup
        for cap in self.captures.values():
            cap.release()
        if show:
            cv2.destroyAllWindows()

        print("\n" + "=" * 65)
        print("  RE-IDENTIFICATION SESSION SUMMARY")
        print(f"  Total Unique Persons Identified: {len(self.global_ids.identities)}")
        for gid, identity in self.global_ids.identities.items():
            print(f"    * {gid}: Last seen in {identity.camera_id} (Detections: {identity.total_detections})")
        print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="SAFEPASS Multi-Camera CCTV Person Re-ID Engine.")
    parser.add_argument("--sources", nargs="+", default=None, help="Webcam index (0) or video file path(s)")
    parser.add_argument("--camera-ids", nargs="+", default=None, help="Camera IDs (e.g. cam1 cam2)")
    parser.add_argument("--zones", default=None, help="Handoff zones JSON file")
    parser.add_argument("--no-osnet", action="store_true", help="Disable OSNet and use fast stub")
    parser.add_argument("--sim-thresh", type=float, default=0.52, help="ReID cosine similarity threshold")
    parser.add_argument("--handoff-thresh", type=float, default=0.48, help="Handoff composite score threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    sources = args.sources
    if not sources:
        import glob
        found_videos = sorted(glob.glob("data/videos/*.mp4") + glob.glob("data/videos/*.avi") + glob.glob("data/videos/*.mkv"))
        if found_videos:
            sources = found_videos
            print(f"[CCTV] Auto-detected {len(sources)} CCTV video feed(s) from data/videos/:")
            for v in sources:
                print(f"  -> {v}")
        else:
            sources = ["0"]

    engine = MultiCameraCCTVReIDEngine(
        sources=sources,
        camera_ids=args.camera_ids,
        zone_file=args.zones,
        similarity_threshold=args.sim_thresh,
        handoff_threshold=args.handoff_thresh,
        use_osnet=not args.no_osnet,
    )
    engine.run(show=True, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
