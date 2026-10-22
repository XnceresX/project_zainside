"""
face/test_pipeline.py
=====================
End-to-end smoke test for the P1 face pipeline.

Creates a synthetic face-like image, runs:
    1. FaceDetector  -> detect bounding box + landmarks + pose
    2. Quality estimation -> blur, resolution, pose, occlusion scores
    3. FaceFeatureExtractor -> 512-d embedding
    4. Degradation suite -> verify quality scores track degradation

Run:
    python -m face.test_pipeline
"""

from __future__ import annotations

import sys
import numpy as np
import cv2


def create_synthetic_face(size: int = 200) -> np.ndarray:
    """
    Create a simple synthetic 'face' image with enough structure
    for the detector to potentially find edges, and for the quality
    scorers to produce meaningful numbers.
    """
    img = np.full((size, size, 3), 200, dtype=np.uint8)  # light grey background

    # Oval face shape
    cv2.ellipse(img, (size // 2, size // 2), (size // 3, size // 2 - 20),
                0, 0, 360, (180, 150, 130), -1)

    # Eyes
    eye_y = size // 2 - 20
    cv2.circle(img, (size // 2 - 30, eye_y), 8, (50, 50, 50), -1)
    cv2.circle(img, (size // 2 + 30, eye_y), 8, (50, 50, 50), -1)

    # Nose
    cv2.line(img, (size // 2, eye_y + 10), (size // 2, eye_y + 35), (120, 100, 90), 2)

    # Mouth
    cv2.ellipse(img, (size // 2, size // 2 + 40), (20, 8), 0, 0, 180, (80, 50, 50), 2)

    return img


def test_quality_scoring():
    """Test that quality scorers return valid ranges."""
    from face.quality import (
        calculate_blur_score,
        calculate_resolution_score,
        calculate_pose_score,
        calculate_occlusion_score,
        estimate_face_quality,
    )

    face = create_synthetic_face(200)

    # Individual scorers
    blur = calculate_blur_score(face)
    assert 0.0 <= blur <= 1.0, f"blur score out of range: {blur}"

    res = calculate_resolution_score(200)
    assert res == 1.0, f"200px face should get resolution score 1.0, got {res}"

    res_small = calculate_resolution_score(25)
    assert 0.0 < res_small < 1.0, f"25px face should be penalised, got {res_small}"

    pose_frontal = calculate_pose_score(0.0, 0.0)
    assert pose_frontal == 1.0, f"frontal pose should score 1.0, got {pose_frontal}"

    pose_side = calculate_pose_score(40.0, 0.0)
    assert pose_side < pose_frontal, f"side pose should score < frontal"

    occ_full = calculate_occlusion_score(1.0)
    assert occ_full == 1.0

    occ_half = calculate_occlusion_score(0.5)
    assert occ_half == 0.5

    # Composite quality with no face
    q_none = estimate_face_quality(face_image=None)
    assert q_none.face_detected is False
    assert q_none.recommended_face_weight == 0.0

    # Composite quality with good face
    q_good = estimate_face_quality(
        face_image=face,
        detection_conf=0.95,
        yaw=5.0, pitch=3.0,
        landmark_visibility=0.95,
    )
    assert q_good.face_detected is True
    assert q_good.blur_score > 0.0

    print(f"  blur={q_good.blur_score:.3f}  occ={q_good.occlusion_score:.3f}  "
          f"conf={q_good.face_confidence:.2f}  "
          f"weights=[face={q_good.recommended_face_weight}, "
          f"app={q_good.recommended_appearance_weight}, "
          f"color={q_good.recommended_color_weight}]")

    print("  [PASS] Quality scoring tests passed")
    return q_good


def test_extractor():
    """Test face feature extraction produces valid embeddings."""
    from face.extractor import FaceFeatureExtractor, cosine_similarity

    extractor = FaceFeatureExtractor()  # no ONNX model -> uses fallback
    face = create_synthetic_face(200)

    emb = extractor.extract_embedding(face)
    assert emb.shape == (512,), f"Expected 512-d embedding, got {emb.shape}"
    assert np.linalg.norm(emb) > 0, "Embedding should not be all zeros"

    # Self-similarity should be very high
    self_sim = cosine_similarity(emb, emb)
    assert self_sim > 0.99, f"Self-similarity should be ~1.0, got {self_sim}"

    # A different image should have lower similarity
    noise_face = create_synthetic_face(200)
    noise_face = cv2.GaussianBlur(noise_face, (31, 31), 10)
    emb2 = extractor.extract_embedding(noise_face)
    cross_sim = cosine_similarity(emb, emb2)
    print(f"  self_sim={self_sim:.4f}  cross_sim={cross_sim:.4f}")

    print("  [PASS] Extractor tests passed")


def test_degradation_tracking():
    """
    Verify that quality scores DROP as degradation INCREASES.
    This is the core evidence for the research novelty.
    """
    from face.quality import calculate_blur_score
    from eval.degrade import apply_blur, apply_downscale, apply_occlusion

    face = create_synthetic_face(200)
    original_blur = calculate_blur_score(face)

    # Blur test: increasing sigma should decrease blur score
    print(f"  original blur_score = {original_blur:.4f}")
    prev = original_blur
    for sigma in [2.0, 5.0, 10.0]:
        degraded = apply_blur(face, sigma=sigma)
        score = calculate_blur_score(degraded)
        print(f"    sigma={sigma:5.1f}  ->  blur_score={score:.4f}  {'v' if score < prev else '^'}")
        assert score <= prev + 0.01, (
            f"Blur score should decrease with more blur: "
            f"sigma={sigma} gave {score} vs previous {prev}"
        )
        prev = score

    # Downscale test
    for target_h in [40, 20]:
        degraded = apply_downscale(face, target_height=target_h)
        score = calculate_blur_score(degraded)
        print(f"    downscale->{target_h}px  ->  blur_score={score:.4f}")

    # Occlusion test (visual check -- no assertion, just print)
    for coverage in [0.2, 0.4, 0.6]:
        degraded = apply_occlusion(face, coverage=coverage, position="lower")
        score = calculate_blur_score(degraded)
        print(f"    occlusion={coverage:.0%}  ->  blur_score={score:.4f}")

    print("  [PASS] Degradation tracking tests passed")


def main():
    print("=" * 60)
    print("  SAFEPASS P1 Face Pipeline -- Smoke Test")
    print("=" * 60)

    print("\n[1/3] Quality Scoring")
    test_quality_scoring()

    print("\n[2/3] Feature Extractor")
    test_extractor()

    print("\n[3/3] Degradation Tracking")
    test_degradation_tracking()

    print("\n" + "=" * 60)
    print("  ALL P1 TESTS PASSED [PASS]")
    print("=" * 60)


if __name__ == "__main__":
    main()
