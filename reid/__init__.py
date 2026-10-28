"""
reid package
============
Person Re-Identification & Cross-Camera Multi-Track Framework for SAFEPASS.
Adapted from https://github.com/meethu51/ReID-framework.git
"""

from reid.tracker import SimpleTracker, Track, Detection
from reid.global_id_manager import GlobalIDManager, GlobalIdentity
from reid.detector import PersonDetector, SimpleBlobDetector
from reid.embedding import ReIDFeatureExtractor, cosine_similarity, stub_embedding_for_crop
from reid.topology_handoff import HandoffZone, load_zones, zone_confidence, intersection_ratio

__all__ = [
    "SimpleTracker",
    "Track",
    "Detection",
    "GlobalIDManager",
    "GlobalIdentity",
    "PersonDetector",
    "SimpleBlobDetector",
    "ReIDFeatureExtractor",
    "cosine_similarity",
    "stub_embedding_for_crop",
    "HandoffZone",
    "load_zones",
    "zone_confidence",
    "intersection_ratio",
]
