"""
reid/zone_editor.py
===================
Interactive rectangle handoff-zone editor for CCTV camera views.
Ported from https://github.com/meethu51/ReID-framework.git
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2


def run_zone_editor(
    source: str,
    camera_id: str,
    output_path: str,
    display_width: int = 960,
    display_height: int = 540,
) -> None:
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        print(f"[ZoneEditor] Error: Could not read frame from source: {source}")
        return

    original_h, original_w = frame.shape[:2]
    scale_x = original_w / display_width
    scale_y = original_h / display_height
    display = cv2.resize(frame, (display_width, display_height))
    drawing = {"start": None, "end": None}
    win_name = "Draw handoff zone: drag mouse, [S] save, [R] reset, [Q] quit"

    def redraw() -> None:
        canvas = display.copy()
        if drawing["start"] and drawing["end"]:
            cv2.rectangle(canvas, drawing["start"], drawing["end"], (0, 220, 220), 2)
        cv2.imshow(win_name, canvas)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["start"] = (x, y)
            drawing["end"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing["start"]:
            drawing["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing["start"]:
            drawing["end"] = (x, y)
        redraw()

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            drawing["start"] = None
            drawing["end"] = None
            redraw()
        if key == ord("s") and drawing["start"] and drawing["end"]:
            x1, y1 = drawing["start"]
            x2, y2 = drawing["end"]
            ox1, ox2 = sorted((int(x1 * scale_x), int(x2 * scale_x)))
            oy1, oy2 = sorted((int(y1 * scale_y), int(y2 * scale_y)))
            data = {
                "zones": [
                    {
                        "camera_id": camera_id,
                        "zone_id": f"{camera_id}_handoff_zone",
                        "type": "rect",
                        "xyxy": [ox1, oy1, ox2, oy2],
                        "description": "Cross-camera topological handoff transition zone",
                    }
                ]
            }
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"[ZoneEditor] Saved handoff zone definition to {out_file}")
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Draw a CCTV handoff zone rectangle.")
    parser.add_argument("--source", required=True, help="Camera index or video file")
    parser.add_argument("--camera-id", required=True, help="Camera ID (e.g. cam1)")
    parser.add_argument("--output", default="data/handoff_zones.json", help="Output JSON path")
    args = parser.parse_args()
    run_zone_editor(args.source, args.camera_id, args.output)


if __name__ == "__main__":
    main()
