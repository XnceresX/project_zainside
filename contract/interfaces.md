# SAFEPASS Shared Contract & Interfaces Guide

This document defines the boundary contracts between the 4 parallel project tracks.

---

## 1. Track Ownership

| Track | Owner | Folder | Core Responsibility |
|---|---|---|---|
| **P1** | Track 1 | `face/` | MobileFaceNet / ArcFace extraction, face quality scoring (blur, size, pose, occlusion) |
| **P2** | Track 2 | `appearance/`, `fusion/` | OSNet body Re-ID, HSV clothing color, quality-adaptive fusion formula |
| **P3** | Track 3 | `topology/` | NetworkX camera graph, spatial validation, transit bounds $[\Delta t_{\min}, \Delta t_{\max}]$, blind-spot forecasting |
| **P4** | Track 4 | `backend/`, `frontend/` | FastAPI REST endpoints, SQLite DB, intake & review UI, end-to-end orchestration |

---

## 2. Shared Data Contract (`contract/schema.py`)

All tracks import standard models from `contract.schema`:
- `QualityMetrics`: Blur and occlusion quality scores.
- `Candidate`: Candidate detection score breakdown across cameras.
- `CameraNode`: Metadata and physical edge constraints for CCTV feeds.
- `SpatialPrediction`: Predicted high-likelihood next cameras with arrival windows.
- `SearchCaseResult`: Complete payload delivered to the operator UI.

---

## 3. Rules of Engagement

1. **Do not modify `contract/schema.py`** without unanimous consensus from all 4 track leads.
2. **Work strictly on your branch:**
   - `p1-face`
   - `p2-appearance`
   - `p3-topology`
   - `p4-backend`
3. **Commit often** and never commit model weights (`*.onnx`, `*.pt`, `*.pth`) or video files (`*.mp4`) to Git.
