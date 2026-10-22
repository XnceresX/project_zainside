"""
face/quality.py
===============
Quality estimation algorithms for facial crops (Track P1).
Evaluates blur (Laplacian variance), crop resolution, and occlusion index.
"""

import cv2
import numpy as np
from contract.schema import QualityMetrics


def calculate_blur_score(face_image: np.ndarray) -> float:
    """
    Computes normalized blur score via variance of Laplacian.
    Returns:
        float in [0.0, 1.0], where 0.0 is severely blurred and 1.0 is sharp.
    """
    if face_image is None or face_image.size == 0:
        return 0.0

    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Empirical normalization: variance >= 200 is considered sharp
    normalized = min(max(variance / 200.0, 0.0), 1.0)
    return float(round(normalized, 4))


def calculate_resolution_penalty(height: int, min_height: int = 50) -> float:
    """
    Penalizes crops below min_height pixels (standard face recognition degradation threshold).
    """
    if height <= 0:
        return 0.0
    if height >= min_height:
        return 1.0
    return float(round(height / min_height, 4))


def estimate_face_quality(
    face_image: Optional[np.ndarray],
    detection_conf: float = 0.9,
    occlusion_ratio: float = 0.1
) -> QualityMetrics:
    """
    Complete quality scoring for facial input.
    Derives recommended dynamic fusion weights based on crop health.
    """
    if face_image is None or face_image.size == 0:
        return QualityMetrics(
            blur_score=0.0,
            occlusion_score=1.0,
            face_detected=False,
            face_confidence=0.0,
            recommended_face_weight=0.0,
            recommended_appearance_weight=0.6,
            recommended_color_weight=0.4,
        )

    h, w = face_image.shape[:2]
    blur_score = calculate_blur_score(face_image)
    res_factor = calculate_resolution_penalty(h)

    # Effective quality considering blur and resolution
    effective_quality = blur_score * res_factor * (1.0 - occlusion_ratio)

    # Dynamic weight distribution based on quality
    if effective_quality > 0.65:
        w_face = 0.70
        w_app = 0.20
        w_color = 0.10
    elif effective_quality > 0.35:
        w_face = 0.40
        w_app = 0.40
        w_color = 0.20
    else:
        # Occluded or blurred face: shift weight to clothing/appearance
        w_face = 0.10
        w_app = 0.50
        w_color = 0.40

    return QualityMetrics(
        blur_score=blur_score,
        occlusion_score=float(round(occlusion_ratio, 4)),
        face_detected=True,
        face_confidence=detection_conf,
        recommended_face_weight=w_face,
        recommended_appearance_weight=w_app,
        recommended_color_weight=w_color,
    )
