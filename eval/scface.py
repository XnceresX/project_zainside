"""Evaluate the face recognizer using the official SCface daytime protocol.

SCface provides pre-cropped face images named like ``001_frontal.jpg`` and
``001_cam1_1.jpg``. The frontal mugshots form the gallery; each camera and
distance is evaluated as a separate probe condition.

Usage:
    python -m eval.scface --dataset_dir data/scface
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from eval.face_baseline import compute_cmc_and_map
from face.extractor import FaceFeatureExtractor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
FRONTAL_RE = re.compile(r"^(?P<id>\d+)_frontal\.(?:jpg|jpeg|png|bmp)$", re.IGNORECASE)
PROBE_RE = re.compile(
    r"^(?P<id>\d+)_cam(?P<camera>[1-5])_(?P<distance>[1-3])\.(?:jpg|jpeg|png|bmp)$",
    re.IGNORECASE,
)
DISTANCE_METERS = {"1": 4.2, "2": 2.6, "3": 1.0}


def discover_scface_images(dataset_dir: Path) -> Tuple[Dict[str, Path], Dict[Tuple[str, str, str], Path]]:
    """Index the SCface filename convention, allowing nested directories."""
    gallery: Dict[str, Path] = {}
    probes: Dict[Tuple[str, str, str], Path] = {}
    for path in dataset_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name = path.name
        match = FRONTAL_RE.match(name)
        if match:
            gallery[match.group("id")] = path
            continue
        match = PROBE_RE.match(name)
        if match:
            key = (match.group("id"), match.group("camera"), match.group("distance"))
            probes[key] = path
    return gallery, probes


def evaluate_scface(dataset_dir: Path, model_path: str | None = None, use_tta: bool = True) -> dict:
    gallery_paths, probe_paths = discover_scface_images(dataset_dir)
    if not gallery_paths:
        raise ValueError(
            f"No SCface frontal mugshots found under {dataset_dir}. Expected names such as 001_frontal.jpg."
        )
    if not probe_paths:
        raise ValueError(
            f"No SCface daytime probes found under {dataset_dir}. Expected names such as 001_cam1_1.jpg."
        )

    extractor = FaceFeatureExtractor(model_path=model_path, use_tta=use_tta)
    if not extractor.is_loaded:
        raise RuntimeError(
            "SCface evaluation requires a loaded ONNX face-recognition model; "
            "the histogram fallback is not valid for biometric evaluation."
        )

    # Compute gallery features once and reuse them for all 15 camera/distance runs.
    gallery_ids = sorted(gallery_paths)
    gallery_images = []
    valid_gallery_ids: List[str] = []
    for person_id in gallery_ids:
        image = cv2.imread(str(gallery_paths[person_id]), cv2.IMREAD_COLOR)
        if image is not None:
            gallery_images.append(image)
            valid_gallery_ids.append(person_id)
    if not gallery_images:
        raise ValueError("Could not read any SCface frontal gallery images.")
    gallery_embeddings = extractor.extract_embeddings(gallery_images)
    gallery_matrix = np.asarray(gallery_embeddings, dtype=np.float32)

    conditions = []
    for camera in range(1, 6):
        for distance in range(1, 4):
            ids = [
                person_id for person_id in valid_gallery_ids
                if (person_id, str(camera), str(distance)) in probe_paths
            ]
            if not ids:
                conditions.append({
                    "camera": camera,
                    "distance_label": distance,
                    "distance_meters": DISTANCE_METERS[str(distance)],
                    "num_probes": 0,
                    "rank_1": None,
                    "rank_5": None,
                    "rank_10": None,
                    "mAP": None,
                })
                continue

            probe_images = []
            readable_ids = []
            for person_id in ids:
                image = cv2.imread(
                    str(probe_paths[(person_id, str(camera), str(distance))]),
                    cv2.IMREAD_COLOR,
                )
                if image is not None:
                    probe_images.append(image)
                    readable_ids.append(person_id)
            if not probe_images:
                conditions.append({
                    "camera": camera,
                    "distance_label": distance,
                    "distance_meters": DISTANCE_METERS[str(distance)],
                    "num_probes": 0,
                    "rank_1": None,
                    "rank_5": None,
                    "rank_10": None,
                    "mAP": None,
                })
                continue

            probe_matrix = np.asarray(extractor.extract_embeddings(probe_images), dtype=np.float32)
            # cosine_similarity() maps valid cosine scores to [0, 1]. A zero
            # vector is assigned 0 (rather than the misleading midpoint 0.5).
            dots = probe_matrix @ gallery_matrix.T
            norms = np.linalg.norm(probe_matrix, axis=1)[:, None] * np.linalg.norm(gallery_matrix, axis=1)[None, :]
            similarities = np.zeros_like(dots, dtype=np.float32)
            valid = norms > 1e-8
            similarities[valid] = ((dots[valid] / norms[valid]) + 1.0) / 2.0

            cmc, mean_ap = compute_cmc_and_map(similarities, valid_gallery_ids, readable_ids)
            conditions.append({
                "camera": camera,
                "distance_label": distance,
                "distance_meters": DISTANCE_METERS[str(distance)],
                "num_probes": len(readable_ids),
                "rank_1": cmc[1],
                "rank_5": cmc[5],
                "rank_10": cmc[10],
                "mAP": mean_ap,
            })

    return {
        "dataset": "SCface",
        "protocol": "daytime: frontal mugshot gallery; cameras 1-5 evaluated separately at distances 4.2m, 2.6m, and 1.0m",
        "model": extractor.status,
        "num_gallery": len(valid_gallery_ids),
        "conditions": conditions,
        "notes": [
            "SCface inputs are already-cropped face images; this evaluates recognition, not full-frame face detection.",
            "Each camera-distance condition is closed-set identification against identities with a frontal gallery image.",
            "The script does not select a verification threshold; report verification using the dataset's official protocol files if used.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SCface daytime face-identification evaluation")
    parser.add_argument("--dataset_dir", required=True, help="Extracted SCface root directory")
    parser.add_argument("--model_path", default=None, help="Optional ArcFace/MobileFaceNet ONNX path")
    parser.add_argument("--no_tta", action="store_true", help="Disable horizontal-flip test-time augmentation")
    parser.add_argument("--output_json", default="eval/results_scface.json", help="Results JSON path")
    args = parser.parse_args()

    result = evaluate_scface(Path(args.dataset_dir), args.model_path, use_tta=not args.no_tta)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("SCface daytime identification results (Rank-1)")
    print("camera | distance | probes | rank-1")
    for row in result["conditions"]:
        value = "n/a" if row["rank_1"] is None else f"{row['rank_1']:.4f}"
        print(f"cam{row['camera']} | {row['distance_meters']:.1f}m | {row['num_probes']} | {value}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
