"""
backend/server.py
=================
FastAPI High-Performance Backend Server for SAFEPASS CCTV AI Command Center.
Provides REST APIs for dynamic camera management, frame serving, Re-ID analysis,
and live investigation controls.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure root SAFE_PASS is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.analyzer import ReIDWebAnalyzer, CACHE_FILE


app = FastAPI(title="SAFEPASS CCTV AI Command Center API", version="2.0.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Analyzer instance
analyzer = ReIDWebAnalyzer(sim_threshold=0.58)

import threading

# Cache video capture handles and thread locks for concurrency safety
_cap_pool: Dict[str, cv2.VideoCapture] = {}
_global_read_lock = threading.Lock()
_jpeg_cache: Dict[str, bytes] = {}
MAX_JPEG_CACHE = 600


def read_frame_safe(file_path: str, frame_no: int) -> Tuple[bool, Optional[np.ndarray]]:
    """Thread-safe frame decoder preventing libavcodec async_lock assertion crashes."""
    abs_path = str(Path(file_path).resolve())
    with _global_read_lock:
        cap = _cap_pool.get(abs_path)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(abs_path)
            _cap_pool[abs_path] = cap

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
        ret, frame = cap.read()
        if not ret or frame is None:
            # Reopen on stream failure
            cap.release()
            cap = cv2.VideoCapture(abs_path)
            _cap_pool[abs_path] = cap
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
            ret, frame = cap.read()

        return ret, frame


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/cameras")
def get_cameras():
    """Returns all active cameras and their status summaries."""
    cameras_dict = analyzer.cache.get("cameras", {})
    cam_list = []
    for cid, cdat in cameras_dict.items():
        cam_list.append({
            "id": cid,
            "name": cdat.get("name", cid),
            "path": cdat.get("path"),
            "total_frames": cdat.get("total_frames", 0),
            "fps": cdat.get("fps", 20.0),
            "width": cdat.get("width", 960),
            "height": cdat.get("height", 640),
            "summary": cdat.get("summary", {})
        })
    return {
        "cameras": cam_list,
        "total_cameras": len(cam_list),
        "reference_target": analyzer.cache.get("reference_target")
    }


@app.get("/api/cameras/{cam_id}")
def get_camera_detail(cam_id: str):
    """Returns detailed metadata and summary for a single camera."""
    cameras_dict = analyzer.cache.get("cameras", {})
    if cam_id not in cameras_dict:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cameras_dict[cam_id]


@app.get("/api/cameras/{cam_id}/analysis")
def get_camera_analysis(cam_id: str):
    """Returns the frame-by-frame bounding boxes and Re-ID status."""
    cameras_dict = analyzer.cache.get("cameras", {})
    if cam_id not in cameras_dict:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {
        "cam_id": cam_id,
        "frames": cameras_dict[cam_id].get("frames", {})
    }


@app.get("/api/cameras/{cam_id}/frame")
def get_camera_frame(cam_id: str, f: int = Query(1, ge=1)):
    """
    Returns JPEG binary data for frame `f` of camera `cam_id`.
    Thread-safe and cached for ultra-fast scrubbing and smooth playback.
    """
    cache_key = f"{cam_id}_{f}"
    if cache_key in _jpeg_cache:
        return Response(content=_jpeg_cache[cache_key], media_type="image/jpeg")

    cameras_dict = analyzer.cache.get("cameras", {})
    if cam_id not in cameras_dict:
        raise HTTPException(status_code=404, detail="Camera not found")

    cdat = cameras_dict[cam_id]
    vpath = cdat.get("path")
    if not vpath or not Path(vpath).exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    total_f = cdat.get("total_frames", 1)
    frame_no = max(1, min(f, total_f))

    ret, frame = read_frame_safe(vpath, frame_no)
    if not ret or frame is None:
        frame = cv2.imread(str(root_dir / "cctv_verification_report.jpg"))
        if frame is None:
            raise HTTPException(status_code=500, detail="Could not read frame")

    # Encode to JPEG
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise HTTPException(status_code=500, detail="Frame encode error")

    raw_bytes = buffer.tobytes()
    # Cache management
    if len(_jpeg_cache) > MAX_JPEG_CACHE:
        # Evict oldest entry
        first_k = next(iter(_jpeg_cache))
        del _jpeg_cache[first_k]
    _jpeg_cache[cache_key] = raw_bytes

    return Response(content=raw_bytes, media_type="image/jpeg")


@app.post("/api/cameras/upload")
async def upload_camera(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None)
):
    """
    Uploads a new video file (.mp4, .avi, .mov, .mkv), registers the camera,
    and analyzes it against the reference target.
    """
    videos_dir = root_dir / "data" / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "upload.mp4"
    dest_path = videos_dir / filename

    # Avoid collision
    counter = 1
    base_stem = dest_path.stem
    while dest_path.exists():
        dest_path = videos_dir / f"{base_stem}_{counter}{dest_path.suffix}"
        counter += 1

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cam_id = f"cam_{dest_path.stem.lower()}"
    cam_name = name or f"Camera {len(analyzer.cache.get('cameras', {})) + 1}: {dest_path.stem}"

    try:
        # If no reference target enrolled yet, enroll from this video
        if not analyzer.cache.get("reference_target"):
            cam_data = analyzer.analyze_reference_camera(str(dest_path), cam_id=cam_id, cam_name=cam_name)
        else:
            cam_data = analyzer.analyze_investigation_camera(str(dest_path), cam_id=cam_id, cam_name=cam_name)

        return {
            "status": "success",
            "message": f"Camera '{cam_name}' added and analyzed successfully.",
            "camera": {
                "id": cam_id,
                "name": cam_name,
                "path": str(dest_path),
                "total_frames": cam_data["total_frames"],
                "fps": cam_data["fps"],
                "summary": cam_data["summary"],
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Re-ID analysis error: {e}")


@app.delete("/api/cameras/{cam_id}")
def delete_camera(cam_id: str):
    """Deletes a camera from the active investigation."""
    cameras_dict = analyzer.cache.get("cameras", {})
    if cam_id not in cameras_dict:
        raise HTTPException(status_code=404, detail="Camera not found")

    del cameras_dict[cam_id]
    if cam_id in analyzer.cache.get("unknown_alerts", {}):
        del analyzer.cache["unknown_alerts"][cam_id]

    analyzer.save_cache()
    return {"status": "success", "message": f"Camera '{cam_id}' deleted."}


class RenameRequest(BaseModel):
    name: str

@app.patch("/api/cameras/{cam_id}")
def rename_camera(cam_id: str, req: RenameRequest):
    """Updates camera display name."""
    cameras_dict = analyzer.cache.get("cameras", {})
    if cam_id not in cameras_dict:
        raise HTTPException(status_code=404, detail="Camera not found")

    cameras_dict[cam_id]["name"] = req.name
    analyzer.save_cache()
    return {"status": "success", "cam_id": cam_id, "name": req.name}


@app.get("/api/reid/reference")
def get_reference_target():
    """Returns the current enrolled referral subject."""
    ref = analyzer.cache.get("reference_target")
    if not ref:
        return {"enrolled": False}
    return {
        "enrolled": True,
        "id": ref.get("id", "Person_01"),
        "cloth_type": ref.get("cloth_type"),
        "v_mean": ref.get("v_mean"),
        "source_cam_name": ref.get("source_cam_name"),
        "crop_b64": ref.get("crop_b64"),
    }


@app.get("/api/reid/summary")
def get_reid_summary():
    """Returns the cross-camera verification summary report data."""
    ref = analyzer.cache.get("reference_target")
    cameras_dict = analyzer.cache.get("cameras", {})
    alerts_dict = analyzer.cache.get("unknown_alerts", {})

    cam_summaries = []
    for cid, cdat in cameras_dict.items():
        cam_summaries.append({
            "cam_id": cid,
            "cam_name": cdat.get("name", cid),
            "summary": cdat.get("summary", {})
        })

    # Only report security alerts for cameras where the target was NOT matched
    matched_cams = {cid for cid, cdat in cameras_dict.items() if cdat.get("summary", {}).get("status") in ("MATCH", "REFERENCE")}
    active_alerts = [alert for cid, alert in alerts_dict.items() if cid not in matched_cams]

    return {
        "reference_target": ref,
        "camera_summaries": cam_summaries,
        "security_alerts": active_alerts,
        "total_cameras": len(cameras_dict),
        "matched_count": len(matched_cams),
        "alert_count": len(active_alerts),
    }


# ---------------------------------------------------------------------------
# Mount Static Frontend
# ---------------------------------------------------------------------------

frontend_dir = root_dir / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/")
def serve_index():
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "SAFEPASS CCTV AI Command Center API running. Frontend loading..."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)
