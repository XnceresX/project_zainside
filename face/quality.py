"""
face/quality.py
===============
Quality estimation algorithms for facial crops (Track P1).

Evaluates multiple quality dimensions that directly affect face recognition
accuracy. These scores drive the adaptive fusion weights: when face quality
is low, the system shifts reliance to body appearance and clothing color.

Quality dimensions:
    1. Blur         — Normalised variance of Laplacian
    2. Resolution   — Pixel height of the face crop
    3. Pose         — Absolute yaw and pitch angles (from FaceDetector mesh)
    4. Occlusion    — Landmark visibility ratio (from FaceDetector mesh)
    5. Composite    — Weighted combination of all dimensions
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from contract.schema import QualityMetrics


# ---------------------------------------------------------------------------
# Individual quality scorers
# ---------------------------------------------------------------------------

def calculate_blur_score(face_image: np.ndarray) -> float:
    """
    Normalised blur score via variance of the Laplacian operator.

    Returns:
        float in [0.0, 1.0] — 0.0 is severely blurred, 1.0 is sharp.
    """
    if face_image is None or face_image.size == 0:
        return 0.0

    gray = (
        cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        if len(face_image.shape) == 3
        else face_image
    )
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Empirical normalisation: variance >= 200 is considered sharp
    normalised = min(max(variance / 200.0, 0.0), 1.0)
    return float(round(normalised, 4))


def calculate_resolution_score(height: int, min_height: int = 50) -> float:
    """
    Penalises crops below *min_height* pixels.

    Below ~50 px face recognition accuracy degrades sharply (documented in
    ArcFace and MobileFaceNet papers). Returns 1.0 for crops at or above
    the threshold.
    """
    if height <= 0:
        return 0.0
    if height >= min_height:
        return 1.0
    return float(round(height / min_height, 4))


def calculate_pose_score(yaw: float, pitch: float) -> float:
    """
    Converts head-pose angles into a quality score.

    Frontal faces (yaw ≈ 0, pitch ≈ 0) score highest.  Beyond ±45° the
    score drops to near zero because most face-recognition models are
    trained predominantly on near-frontal images.
    """
    max_angle = 45.0
    yaw_penalty = max(1.0 - abs(yaw) / max_angle, 0.0)
    pitch_penalty = max(1.0 - abs(pitch) / max_angle, 0.0)
    return float(round(yaw_penalty * pitch_penalty, 4))


def calculate_occlusion_score(landmark_visibility: float) -> float:
    """
    Converts landmark visibility ratio into an occlusion quality score.

    Parameters
    ----------
    landmark_visibility : float
        Fraction of key facial landmarks visible in the frame [0-1].
        Comes from ``DetectedFace.landmark_visibility``.

    Returns
    -------
    float
        0.0 = heavily occluded, 1.0 = fully visible.
    """
    return float(round(max(min(landmark_visibility, 1.0), 0.0), 4))


# ---------------------------------------------------------------------------
# Composite quality & dynamic weight recommendation
# ---------------------------------------------------------------------------

def estimate_face_quality(
    face_image: Optional[np.ndarray] = None,
    detection_conf: float = 0.9,
    yaw: float = 0.0,
    pitch: float = 0.0,
    landmark_visibility: float = 1.0,
) -> QualityMetrics:
    """
    Compute a comprehensive face quality assessment and recommend dynamic
    fusion weights for the multimodal matcher.

    Parameters
    ----------
    face_image : ndarray or None
        BGR face crop.  If None, returns a "no face" quality report.
    detection_conf : float
        Face detector confidence score.
    yaw, pitch : float
        Head pose angles in degrees (from ``FacePose``).
    landmark_visibility : float
        Fraction of visible key landmarks (from ``DetectedFace``).

    Returns
    -------
    QualityMetrics
        Contains individual scores and recommended fusion weights.
    """

    # --- No face available --------------------------------------------------
    if face_image is None or face_image.size == 0:
        return QualityMetrics(
            blur_score=0.0,
            occlusion_score=1.0,        # fully occluded
            face_detected=False,
            face_confidence=0.0,
            recommended_face_weight=0.0,
            recommended_appearance_weight=0.60,
            recommended_color_weight=0.40,
        )

    # --- Individual scores ---------------------------------------------------
    h = face_image.shape[0]
    blur     = calculate_blur_score(face_image)
    res      = calculate_resolution_score(h)
    pose     = calculate_pose_score(yaw, pitch)
    occlusion = calculate_occlusion_score(landmark_visibility)

    # Composite effective quality  (weighted geometric-ish blend)
    #   blur and occlusion matter most; pose and resolution are secondary
    effective_quality = (
        0.35 * blur
        + 0.25 * occlusion
        + 0.20 * pose
        + 0.20 * res
    )

    # --- Dynamic weight recommendation --------------------------------------
    #   High quality  → trust face heavily
    #   Medium        → balanced split
    #   Low           → fall back to appearance & colour
    if effective_quality > 0.65:
        w_face  = 0.70
        w_app   = 0.20
        w_color = 0.10
    elif effective_quality > 0.40:
        w_face  = 0.40
        w_app   = 0.35
        w_color = 0.25
    else:
        w_face  = 0.10
        w_app   = 0.50
        w_color = 0.40

    return QualityMetrics(
        blur_score=blur,
        occlusion_score=float(round(1.0 - occlusion, 4)),  # schema: 0=visible, 1=occluded
        face_detected=True,
        face_confidence=detection_conf,
        recommended_face_weight=w_face,
        recommended_appearance_weight=w_app,
        recommended_color_weight=w_color,
    )
