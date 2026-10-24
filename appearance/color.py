"""
appearance/color.py
===================
Track P2 · Task 2 & 8: Clothing Colour Feature Extraction.

Extracts upper and lower garment colour signatures from a body crop
using HSV histograms.  Supports both global (whole-crop) and regional
(upper/lower half) extraction.

Similarity is computed using Bhattacharyya coefficient (OpenCV HISTCMP_BHATTACHARYYA
inverted) or Jensen-Shannon divergence, both mapped to [0, 1].

Public API
----------
    ClothingColorExtractor()
        .extract(body_crop) -> ColorFeatures
        .extract_upper(body_crop) -> np.ndarray   # HSV histogram
        .extract_lower(body_crop) -> np.ndarray   # HSV histogram
        .compare(crop1, crop2) -> float           # overall colour similarity
        .compare_upper(h1, h2) -> float           # upper garment similarity
        .compare_lower(h1, h2) -> float           # lower garment similarity

    ColorFeatures (dataclass):
        upper_hist   np.ndarray  shape (H*S,)
        lower_hist   np.ndarray  shape (H*S,)
        upper_label  str         dominant colour name (e.g. "red")
        lower_label  str         dominant colour name
        combined_sim float       pre-computed self-similarity (debug)

Design notes
------------
* HSV is preferred over BGR for lighting robustness.
* Hue bins: 18 (20° intervals), Saturation bins: 8 → 144-d per region.
* Achromatic pixels (S < 30, V < 30 or V > 225) are masked out to avoid
  background noise from white walls / dark shadows.
* Dominant colour label is extracted via the peak hue bin with saturation
  weighting, mapped to 11 basic colour terms.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Histogram parameters
# ---------------------------------------------------------------------------

H_BINS = 18    # 360° / 18 = 20° per bin  (hue discrimination)
S_BINS = 8     # saturation levels
V_BINS = 8     # value (brightness) levels — optional, kept separate
HIST_SIZE = H_BINS * S_BINS   # 144-d final vector

# HSV ranges for calcHist
H_RANGE = [0, 180]   # OpenCV uses 0-180 for Hue
S_RANGE = [0, 256]
V_RANGE = [0, 256]

# Achromatic mask thresholds (pixels to exclude from colour histogram)
SAT_MIN = 30    # Pixels with S < SAT_MIN are near-grey
VAL_MIN = 25    # Pixels with V < VAL_MIN are near-black
VAL_MAX = 230   # Pixels with V > VAL_MAX are near-white


# ---------------------------------------------------------------------------
# Basic colour vocabulary (OpenCV hue = 0-180; each hue value × 2 = real °)
# ---------------------------------------------------------------------------

_COLOUR_LABELS = [
    (0,   15,  "red"),
    (15,  22,  "orange"),
    (22,  38,  "yellow"),
    (38,  75,  "green"),
    (75,  100, "cyan"),
    (100, 130, "blue"),
    (130, 150, "purple"),
    (150, 165, "magenta"),
    (165, 180, "red"),     # Red wraps around
]

_ACHROMATIC_LABELS = {
    "black": (0, 30),       # V range
    "grey": (30, 180),
    "white": (180, 256),
}


def _hue_to_label(peak_hue_bin: int, is_achromatic: bool, mean_val: float) -> str:
    """Map a peak hue bin index to a colour label string."""
    if is_achromatic:
        v = mean_val
        for label, (lo, hi) in _ACHROMATIC_LABELS.items():
            if lo <= v < hi:
                return label
        return "grey"
    hue_val = peak_hue_bin * (180 / H_BINS)
    for lo, hi, name in _COLOUR_LABELS:
        if lo <= hue_val < hi:
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# Dataclass for colour features
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ColorFeatures:
    """Holds HSV histogram features for upper and lower garment regions."""
    upper_hist: np.ndarray          # shape (HIST_SIZE,)
    lower_hist: np.ndarray          # shape (HIST_SIZE,)
    upper_label: str                # dominant colour name
    lower_label: str                # dominant colour name
    upper_dominant_hsv: Tuple[int, int, int]  # (H, S, V) median of dominant pixels
    lower_dominant_hsv: Tuple[int, int, int]

    def to_dict(self) -> dict:
        return {
            "upper_label": self.upper_label,
            "lower_label": self.lower_label,
            "upper_dominant_hsv": self.upper_dominant_hsv,
            "lower_dominant_hsv": self.lower_dominant_hsv,
        }


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class ClothingColorExtractor:
    """
    Extracts and compares HSV clothing colour histograms from body crops.

    The body crop is divided into:
        Upper region: top 40% of the crop (torso / shirt)
        Lower region: bottom 50% of the crop (trousers / skirt)
        (The 10% overlap in the middle is discarded to reduce mixing.)

    Parameters
    ----------
    h_bins : int
        Number of hue bins (default 18).
    s_bins : int
        Number of saturation bins (default 8).
    remove_background : bool
        If True, use achromatic masking to suppress background pixels.
    """

    def __init__(
        self,
        h_bins: int = H_BINS,
        s_bins: int = S_BINS,
        remove_background: bool = True,
    ):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.hist_size = h_bins * s_bins
        self.remove_background = remove_background

    # ------------------------------------------------------------------
    # Histogram computation
    # ------------------------------------------------------------------

    def _compute_hist(self, region: np.ndarray) -> np.ndarray:
        """
        Compute a 2D (H×S) colour histogram from an image region (BGR).
        Returns a 1D float32 array of length self.hist_size, L1-normalised.
        """
        if region is None or region.size == 0:
            return np.zeros(self.hist_size, dtype=np.float32)

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Build mask for chromatic foreground pixels
        if self.remove_background:
            s_ch = hsv[:, :, 1]
            v_ch = hsv[:, :, 2]
            mask = (
                (s_ch >= SAT_MIN) &
                (v_ch >= VAL_MIN) &
                (v_ch <= VAL_MAX)
            ).astype(np.uint8) * 255
        else:
            mask = None

        # 2D H-S histogram
        hist = cv2.calcHist(
            [hsv.astype(np.uint8)],
            [0, 1],
            mask,
            [self.h_bins, self.s_bins],
            [0, 180, 0, 256],
        )

        hist = hist.flatten().astype(np.float32)
        total = hist.sum() + 1e-8
        return hist / total  # L1-normalise

    # ------------------------------------------------------------------
    # Dominant colour label
    # ------------------------------------------------------------------

    def _dominant_colour(self, region: np.ndarray) -> Tuple[str, Tuple[int, int, int]]:
        """Return (colour_label, (H, S, V)) of the most prominent colour in region."""
        if region is None or region.size == 0:
            return "unknown", (0, 0, 0)

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]
        h_ch = hsv[:, :, 0]

        # Chromatic pixels mask
        chroma_mask = (s_ch >= SAT_MIN) & (v_ch >= VAL_MIN) & (v_ch <= VAL_MAX)
        chroma_count = chroma_mask.sum()

        # Determine if achromatic region dominates
        total_pixels = region.shape[0] * region.shape[1]
        is_achromatic = chroma_count < total_pixels * 0.15

        if is_achromatic:
            mean_v = float(v_ch.mean())
            label = _hue_to_label(0, True, mean_v)
            return label, (0, 0, int(mean_v))

        # Peak hue bin among chromatic pixels
        h_vals = h_ch[chroma_mask]
        hist_1d = np.bincount(h_vals.flatten().astype(np.int32), minlength=180)
        # Smooth over ±1 bin to avoid noise
        hist_smooth = np.convolve(hist_1d, [0.25, 0.5, 0.25], mode="same")
        peak_bin = int(np.argmax(hist_smooth))
        bin_idx = int(peak_bin * self.h_bins / 180)

        # Median HSV of pixels near the dominant hue
        near_peak = np.abs(h_ch.astype(np.int32) - peak_bin) <= 10
        dominant_mask = chroma_mask & near_peak
        if dominant_mask.sum() == 0:
            dominant_mask = chroma_mask

        median_h = int(np.median(h_ch[dominant_mask]))
        median_s = int(np.median(s_ch[dominant_mask]))
        median_v = int(np.median(v_ch[dominant_mask]))

        label = _hue_to_label(bin_idx, False, float(median_v))
        return label, (median_h, median_s, median_v)

    # ------------------------------------------------------------------
    # Region extraction
    # ------------------------------------------------------------------

    def _get_upper_region(self, crop: np.ndarray) -> np.ndarray:
        """Return the upper 40% of the body crop (torso area)."""
        h = crop.shape[0]
        end = max(int(h * 0.40), 1)
        return crop[:end]

    def _get_lower_region(self, crop: np.ndarray) -> np.ndarray:
        """Return the lower 50% of the body crop (legs area)."""
        h = crop.shape[0]
        start = min(int(h * 0.50), h - 1)
        return crop[start:]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_upper(self, body_crop: np.ndarray) -> np.ndarray:
        """
        Extract HSV histogram for the upper garment region.
        Returns np.ndarray of shape (HIST_SIZE,).
        """
        region = self._get_upper_region(body_crop)
        return self._compute_hist(region)

    def extract_lower(self, body_crop: np.ndarray) -> np.ndarray:
        """
        Extract HSV histogram for the lower garment region.
        Returns np.ndarray of shape (HIST_SIZE,).
        """
        region = self._get_lower_region(body_crop)
        return self._compute_hist(region)

    def extract(self, body_crop: np.ndarray) -> ColorFeatures:
        """
        Full colour feature extraction for a body crop.

        Parameters
        ----------
        body_crop : np.ndarray
            BGR image of a detected person bounding box.

        Returns
        -------
        ColorFeatures dataclass with upper_hist, lower_hist, and colour labels.
        """
        if body_crop is None or body_crop.size == 0:
            return ColorFeatures(
                upper_hist=np.zeros(self.hist_size, dtype=np.float32),
                lower_hist=np.zeros(self.hist_size, dtype=np.float32),
                upper_label="unknown",
                lower_label="unknown",
                upper_dominant_hsv=(0, 0, 0),
                lower_dominant_hsv=(0, 0, 0),
            )

        upper_region = self._get_upper_region(body_crop)
        lower_region = self._get_lower_region(body_crop)

        upper_hist = self._compute_hist(upper_region)
        lower_hist = self._compute_hist(lower_region)

        upper_label, upper_hsv = self._dominant_colour(upper_region)
        lower_label, lower_hsv = self._dominant_colour(lower_region)

        return ColorFeatures(
            upper_hist=upper_hist,
            lower_hist=lower_hist,
            upper_label=upper_label,
            lower_label=lower_label,
            upper_dominant_hsv=upper_hsv,
            lower_dominant_hsv=lower_hsv,
        )

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    @staticmethod
    def hist_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Bhattacharyya-based histogram similarity in [0, 1].
        1.0 = identical distributions, 0.0 = completely different.
        """
        if hist1 is None or hist2 is None or hist1.size == 0 or hist2.size == 0:
            return 0.0

        h1 = hist1.astype(np.float32).reshape(-1, 1)
        h2 = hist2.astype(np.float32).reshape(-1, 1)

        # Normalise if not already
        s1 = h1.sum() + 1e-8
        s2 = h2.sum() + 1e-8
        h1_n = h1 / s1
        h2_n = h2 / s2

        # Bhattacharyya distance (lower = more similar)
        bhatt_dist = cv2.compareHist(h1_n, h2_n, cv2.HISTCMP_BHATTACHARYYA)

        # Convert distance [0, ~1.5] → similarity [0, 1]
        similarity = float(max(1.0 - bhatt_dist, 0.0))
        return round(similarity, 4)

    def compare_upper(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compare two upper-garment histograms."""
        return self.hist_similarity(hist1, hist2)

    def compare_lower(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compare two lower-garment histograms."""
        return self.hist_similarity(hist1, hist2)

    def compare(self, crop1: np.ndarray, crop2: np.ndarray, weights=(0.55, 0.45)) -> float:
        """
        Overall clothing colour similarity between two body crops.

        Upper garment weight: 0.55 (shirts are more distinctive than trousers)
        Lower garment weight: 0.45

        Returns
        -------
        float in [0, 1]
        """
        f1 = self.extract(crop1)
        f2 = self.extract(crop2)

        upper_sim = self.hist_similarity(f1.upper_hist, f2.upper_hist)
        lower_sim = self.hist_similarity(f1.lower_hist, f2.lower_hist)

        w_upper, w_lower = weights
        combined = w_upper * upper_sim + w_lower * lower_sim
        return float(round(combined, 4))

    def compare_with_text(
        self,
        crop: np.ndarray,
        upper_color_query: str,
        lower_color_query: str,
    ) -> float:
        """
        Compare a body crop against text colour descriptions (from IntakeRequest).
        Useful when no reference crop is available.

        Returns
        -------
        float in [0, 1] - 1.0 if both labels match exactly.
        """
        features = self.extract(crop)
        upper_match = float(features.upper_label.lower() == upper_color_query.lower().strip())
        lower_match = float(features.lower_label.lower() == lower_color_query.lower().strip())
        return round(0.55 * upper_match + 0.45 * lower_match, 4)
