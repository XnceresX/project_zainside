"""
eval/degrade.py
===============
Synthetic image degradation toolkit for SAFEPASS evaluation (Track P1).

Applies controlled degradations to test images so we can measure how
face quality scores and matching accuracy degrade under real-world conditions:
    - Gaussian blur (simulates motion blur / out-of-focus cameras)
    - Downscaling  (simulates long-range CCTV crops)
    - Occlusion    (simulates hats, masks, hands, crowd overlap)

Usage:
    from eval.degrade import apply_blur, apply_downscale, apply_occlusion
    blurred = apply_blur(image, sigma=5.0)
"""

from __future__ import annotations

import random
from typing import Tuple

import cv2
import numpy as np


def apply_blur(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    Apply Gaussian blur to simulate motion blur or out-of-focus cameras.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian kernel.
        Larger values = more blur.  Typical test range: 1.0 – 10.0.
    """
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_downscale(
    image: np.ndarray,
    target_height: int = 20,
) -> np.ndarray:
    """
    Downscale the image to *target_height* pixels and then scale it back
    to the original size.  This simulates the quality loss from distant
    CCTV cameras where a face occupies very few pixels.

    Parameters
    ----------
    target_height : int
        Intermediate height to shrink to (default 20 px).
    """
    h, w = image.shape[:2]
    if h <= target_height:
        return image.copy()

    ratio = target_height / h
    small = cv2.resize(image, (max(int(w * ratio), 1), target_height),
                       interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored


def apply_occlusion(
    image: np.ndarray,
    coverage: float = 0.3,
    position: str = "random",
    color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Overlay a rectangular patch to simulate occlusion (hat, mask, hand,
    another person's body).

    Parameters
    ----------
    coverage : float
        Fraction of the image area to occlude (0.0 – 1.0).
    position : str
        Where to place the patch.
        'lower'  — lower half (simulates mask / scarf).
        'upper'  — upper half (simulates hat / cap).
        'random' — random position (simulates crowd overlap).
    color : tuple
        BGR colour of the patch.
    """
    out = image.copy()
    h, w = out.shape[:2]
    patch_area = int(h * w * coverage)

    # Roughly square patch
    side = int(patch_area ** 0.5)
    pw = min(side, w)
    ph = min(side, h)

    if position == "lower":
        x1, y1 = (w - pw) // 2, h - ph
    elif position == "upper":
        x1, y1 = (w - pw) // 2, 0
    else:  # random
        x1 = random.randint(0, max(w - pw, 0))
        y1 = random.randint(0, max(h - ph, 0))

    out[y1:y1 + ph, x1:x1 + pw] = color
    return out


def generate_degradation_suite(
    image: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Generate a standard suite of degraded versions of a single image.
    Returns a dict mapping degradation label → degraded image.
    """
    suite = {
        "original": image.copy(),
        "blur_sigma_2": apply_blur(image, sigma=2.0),
        "blur_sigma_5": apply_blur(image, sigma=5.0),
        "blur_sigma_10": apply_blur(image, sigma=10.0),
        "downscale_40px": apply_downscale(image, target_height=40),
        "downscale_20px": apply_downscale(image, target_height=20),
        "occlusion_20pct_lower": apply_occlusion(image, coverage=0.20, position="lower"),
        "occlusion_40pct_lower": apply_occlusion(image, coverage=0.40, position="lower"),
        "occlusion_30pct_upper": apply_occlusion(image, coverage=0.30, position="upper"),
        "occlusion_50pct_random": apply_occlusion(image, coverage=0.50, position="random"),
    }
    return suite
