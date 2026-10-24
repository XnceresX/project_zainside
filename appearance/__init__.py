"""
appearance/__init__.py
======================
Track P2: Appearance Pipeline public API.

Exports:
    AppearanceExtractor   - OSNet body Re-ID embedding extractor
    BodyQualityEstimator  - Body crop quality estimation
    ClothingColorExtractor - HSV clothing color histogram extraction
"""

from appearance.extractor import AppearanceExtractor
from appearance.quality import BodyQualityEstimator, estimate_body_quality
from appearance.color import ClothingColorExtractor

__all__ = [
    "AppearanceExtractor",
    "BodyQualityEstimator",
    "estimate_body_quality",
    "ClothingColorExtractor",
]
