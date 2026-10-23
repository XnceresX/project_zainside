# Evaluation & Benchmarks (Shared: P1 & P2)

**Owners:** Person 1 (Face) & Person 2 (Appearance/Fusion)  
**Scope:**
- Synthetic degradation test generator (Gaussian blur, downscaling, occlusion patches).
- CMC Rank-1, Rank-5, and mAP evaluation scripts.
- Ablation table generation: Face-only vs. OSNet-only vs. Static Fusion vs. Adaptive Quality Fusion.

## SCface face-recognition evaluation

The SCface daytime protocol compares each subject's frontal mugshot with
surveillance face crops from cameras 1–5, reported separately at 4.2 m, 2.6 m,
and 1.0 m. Place the extracted dataset under `data/scface` (or pass another
directory) and run:

```bash
python -m eval.scface --dataset_dir data/scface
```

The evaluator expects the official filenames, such as `001_frontal.jpg` and
`001_cam1_1.jpg`. It requires a loaded ONNX recognition model and writes
`eval/results_scface.json`. The dataset images are pre-cropped faces, so this
measures identification rather than full-frame detection or video tracking.
Obtain SCface directly from its distributor and follow its release agreement.
