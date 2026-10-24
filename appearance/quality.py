"""
appearance/quality.py
=====================
Track P2 · Task 3: Body Crop Quality Estimation.

Estimates the quality of a person bounding-box crop for use in adaptive
fusion weight computation.  Three orthogonal quality dimensions are scored:

    1. Completeness  – What fraction of the canonical body is visible?
                       Derived from bounding-box aspect ratio and crop height.
    2. Blur          – Laplacian variance (same metric as P1 face quality).
    3. Resolution    – Crop height relative to a minimum usable threshold.
    4. Occlusion     – Edge density + saturation-based occlusion heuristic.

The composite body quality ``q_body`` ∈ [0, 1] directly feeds the fusion
engine:  high quality → trust appearance heavily; low quality → fall back
to face / colour.

Public API
----------
    BodyQualityEstimator()
        .estimate(body_crop, bbox=None) -> dict

    estimate_body_quality(body_crop, bbox=None) -> float   # convenience
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Threshold constants (tuned empirically for CCTV re-ID quality)
# ---------------------------------------------------------------------------

MIN_HEIGHT_PX = 80          # Below this crops are too small for Re-ID
GOOD_HEIGHT_PX = 180        # At this height, resolution quality tops out
TARGET_ASPECT = 0.40        # Width/Height ratio for a "full-body" crop
MAX_ASPECT_RATIO = 1.00     # Anything wider-than-tall is probably not a person


# ---------------------------------------------------------------------------
# Individual quality metrics
# ---------------------------------------------------------------------------

def _blur_score(image: np.ndarray) -> float:
    """Normalised Laplacian variance in [0, 1]. 0 = blurry, 1 = sharp."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(min(max(variance / 150.0, 0.0), 1.0))


def _resolution_score(height: int) -> float:
    """Height-based resolution quality in [0, 1]."""
    if height <= 0:
        return 0.0
    if height >= GOOD_HEIGHT_PX:
        return 1.0
    if height < MIN_HEIGHT_PX:
        return float(height / MIN_HEIGHT_PX) * 0.5  # severe penalty below min
    return float(0.5 + 0.5 * (height - MIN_HEIGHT_PX) / (GOOD_HEIGHT_PX - MIN_HEIGHT_PX))


def _completeness_score(crop: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> float:
    """
    Estimates how complete the body crop is.

    A typical standing person occupies a ~2.5:1 tall crop (height:width ≈ 2.5).
    If the bounding box is provided, check whether any dimension hits the frame
    border (indicating truncation).

    Returns a completeness score in [0, 1].
    """
    if crop is None or crop.size == 0:
        return 0.0

    h, w = crop.shape[:2]
    if w == 0 or h == 0:
        return 0.0

    aspect = w / h  # width-to-height ratio

    # Penalty for wrong aspect (too square = truncated, too wide = background crop)
    ideal_deviation = abs(aspect - TARGET_ASPECT)
    aspect_score = max(1.0 - ideal_deviation * 2.0, 0.0)

    # If bbox touches frame edge, crop is likely incomplete (truncated)
    border_penalty = 1.0
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        if x1 <= 2 or y1 <= 2:
            border_penalty *= 0.80
        if x2 < 0 or y2 < 0:
            border_penalty *= 0.80

    return float(min(aspect_score * border_penalty, 1.0))


def _occlusion_score(image: np.ndarray) -> float:
    """
    Estimates visible body fraction using a saturation + edge analysis.
    
    Very-low-saturation patches AND nearly-zero-edge-density regions near
    the borders are considered occluded / missing body parts.

    Returns a visibility score in [0, 1].  1.0 = fully visible.
    """
    if image is None or image.size == 0:
        return 0.5

    h, w = image.shape[:2]
    if h < 10 or w < 10:
        return 0.5

    # Convert to HSV for saturation analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)

    # Near-black (occluder / background) detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_ratio = float(np.count_nonzero(gray < 10)) / gray.size
    white_ratio = float(np.count_nonzero(gray > 245)) / gray.size

    # Low saturation in large contiguous blocks → occlusion
    low_sat_ratio = float(np.count_nonzero(saturation < 20)) / saturation.size

    # Edge density in the bottom half of the crop
    bottom_half = gray[h // 2:]
    edges = cv2.Canny(bottom_half, 30, 100)
    edge_density = float(np.count_nonzero(edges)) / (bottom_half.size + 1)

    # Combine: penalise extreme uniformity (occlusion / background)
    occlusion_penalty = (black_ratio * 0.5 + white_ratio * 0.3 + max(low_sat_ratio - 0.5, 0.0) * 0.4)
    visibility = max(1.0 - occlusion_penalty, 0.1)

    # Reward well-structured lower body
    if edge_density > 0.05:
        visibility = min(visibility * 1.1, 1.0)

    return float(round(visibility, 4))


# ---------------------------------------------------------------------------
# Composite estimator class
# ---------------------------------------------------------------------------

class BodyQualityEstimator:
    """
    Estimates body crop quality for adaptive multimodal fusion.

    Usage::

        estimator = BodyQualityEstimator()
        quality = estimator.estimate(body_crop, bbox=(x1, y1, x2, y2))
        q_body = quality["q_body"]  # composite score in [0, 1]
    """

    def estimate(
        self,
        body_crop: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Compute body crop quality metrics.

        Parameters
        ----------
        body_crop : np.ndarray
            BGR image of a detected person bounding box.
        bbox : (x1, y1, x2, y2), optional
            Original bounding box coordinates in the frame for border detection.

        Returns
        -------
        dict with keys:
            blur_score       float [0, 1]
            resolution_score float [0, 1]
            completeness     float [0, 1]
            occlusion        float [0, 1]
            q_body           float [0, 1]  ← composite quality
            usable           bool          ← True if crop meets minimum standards
        """
        if body_crop is None or body_crop.size == 0:
            return {
                "blur_score": 0.0,
                "resolution_score": 0.0,
                "completeness": 0.0,
                "occlusion": 0.0,
                "q_body": 0.0,
                "usable": False,
            }

        h, w = body_crop.shape[:2]

        blur = _blur_score(body_crop)
        resolution = _resolution_score(h)
        completeness = _completeness_score(body_crop, bbox)
        visibility = _occlusion_score(body_crop)

        # Composite with bottleneck component (mirrors P1 face quality logic)
        composite_base = (
            0.30 * blur
            + 0.25 * resolution
            + 0.25 * completeness
            + 0.20 * visibility
        )
        bottleneck = min(blur, resolution, completeness)
        q_body = float(round(0.65 * composite_base + 0.35 * bottleneck, 4))

        usable = (
            q_body > 0.15
            and h >= MIN_HEIGHT_PX
            and w >= int(MIN_HEIGHT_PX * TARGET_ASPECT * 0.5)
        )

        return {
            "blur_score": round(blur, 4),
            "resolution_score": round(resolution, 4),
            "completeness": round(completeness, 4),
            "occlusion": round(visibility, 4),
            "q_body": q_body,
            "usable": usable,
        }

    def get_appearance_weight_recommendation(self, q_body: float) -> float:
        """
        Map body quality to a recommended appearance weight for fusion.

        High q_body → heavy reliance on appearance;
        Low q_body  → reduce appearance weight, shift to colour / face.

        Returns
        -------
        float in [0.1, 0.7]
        """
        if q_body >= 0.65:
            return 0.65
        elif q_body >= 0.35:
            return 0.40
        else:
            return 0.15


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def estimate_body_quality(
    body_crop: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> float:
    """
    One-line body quality estimate.

    Returns
    -------
    float: composite q_body in [0, 1].
    """
    estimator = BodyQualityEstimator()
    result = estimator.estimate(body_crop, bbox)
    return result["q_body"]
