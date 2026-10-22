# SAFE_PASS

> **Intelligent Cross-Camera Re-Identification and Search Coordination System**  
> Reactive, multimodal person re-identification with visual quality-adaptive fusion and camera-topology spatial pruning.

---

## 👥 Team & Track Ownership

To enable parallel execution across the 4 team members without merge conflicts, each member works exclusively within their assigned folder and branch:

| Person | Track | Folder | Ownership & Scope |
|:---:|:---|:---|:---|
| **P1** | **Face Pipeline & Quality** | `face/`, `eval/` | MobileFaceNet / ArcFace extraction, face quality scoring (Laplacian blur, resolution penalty, pose, occlusion), face baseline evaluation |
| **P2** | **Appearance & Fusion** | `appearance/`, `fusion/`, `eval/` | OSNet body Re-ID, HSV clothing color analysis, dynamic quality-aware fusion formula, ablation tables |
| **P3** | **Venue Topology** | `topology/` | NetworkX camera graph, transition bounds $[\Delta t_{\min}, \Delta t_{\max}]$, spatial pruning, blind-spot forecasting |
| **P4** | **Backend & Dashboard** | `backend/`, `frontend/` | FastAPI backend, SQLite DB, role-based security, audit logging, operator web dashboard, end-to-end integration |

*Note: `contract/` contains the frozen shared schema (`contract/schema.py`) and is modified only upon unanimous team agreement.*

---

## 📁 Repository Structure

```
SAFE_PASS/
├── README.md
├── requirements.txt
├── .gitignore
├── contract/              # Shared data contract & interfaces
│   ├── schema.py          # Frozen Pydantic schemas (Candidate, QualityMetrics, etc.)
│   └── interfaces.md      # Inter-module communication guide
├── face/                  # [P1 ONLY] Face extraction & quality metrics
│   ├── __init__.py
│   ├── extractor.py       # MobileFaceNet / ArcFace embeddings & cosine similarity
│   └── quality.py         # Laplacian blur, size penalty, occlusion heuristics
├── appearance/            # [P2 ONLY] OSNet body Re-ID & clothing color
├── fusion/                # [P2 ONLY] Quality-adaptive dynamic fusion engine
├── topology/              # [P3 ONLY] Camera graph & spatial search pruning
├── backend/               # [P4 ONLY] FastAPI REST server, DB, audit log
├── frontend/              # [P4 ONLY] Security operator dashboard UI
├── eval/                  # [P1 & P2] Degradation tests and benchmark scripts
├── docs/                  # Project specifications, strategies, and conversation logs
└── data/                  # [Ignored in Git] Datasets and sample video feeds
```

---

## 🌿 Git Branching Workflow

1. Clone and create your dedicated branch:
   ```bash
   git clone https://github.com/itsgracious/SAFE_PASS.git
   cd SAFE_PASS
   
   # For P1 (Face Track):
   git checkout -b p1-face
   
   # For P2: git checkout -b p2-appearance
   # For P3: git checkout -b p3-topology
   # For P4: git checkout -b p4-backend
   ```

2. Daily synchronization:
   ```bash
   # Sync latest main
   git checkout main
   git pull
   git checkout p1-face
   git merge main

   # Commit and push only your assigned folder
   git add face/ eval/
   git commit -m "face: implement blur and occlusion quality scoring"
   git push -u origin p1-face
   ```

3. **Golden Rule:** Never push directly to `main` and never commit large models (`*.onnx`, `*.pt`, `*.pth`) or videos (`*.mp4`).

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt
```
