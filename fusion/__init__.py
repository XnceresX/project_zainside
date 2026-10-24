"""
fusion/__init__.py
==================
Track P2: Multimodal Fusion Engine public API.

Exports:
    FusionEngine     - Rule-based quality-adaptive score fusion
    LearnedFusion    - MLP-based learned weight predictor
    ScoreCalibrator  - Platt scaling + isotonic regression calibration
    fuse_scores      - Convenience one-shot fusion function
"""

from fusion.engine import FusionEngine, fuse_scores
from fusion.calibration import ScoreCalibrator
from fusion.learned import LearnedFusion

__all__ = [
    "FusionEngine",
    "fuse_scores",
    "ScoreCalibrator",
    "LearnedFusion",
]
