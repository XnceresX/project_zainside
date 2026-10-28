"""
reid/test_reid.py
=================
Unit tests for the integrated Re-ID framework.
Tests:
  1. SimpleTracker centroid tracking and track life cycle.
  2. ReIDFeatureExtractor embedding extraction & cosine similarity.
  3. GlobalIDManager cross-camera ID persistence and handoff logic.
  4. Topology handoff zone intersection and confidence scoring.
"""

import numpy as np
import pytest

from reid.tracker import SimpleTracker, Detection, Track
from reid.embedding import ReIDFeatureExtractor, cosine_similarity, stub_embedding_for_crop
from reid.global_id_manager import GlobalIDManager
from reid.topology_handoff import HandoffZone, zone_confidence, intersection_ratio


def test_tracker_lifecycle():
    tracker = SimpleTracker(max_distance=50.0, max_missed=3)

    # Frame 1: Detection at (100, 100, 150, 200)
    dets1 = [Detection((100, 100, 150, 200), 0.9)]
    tracks1 = tracker.update(dets1)
    assert len(tracks1) == 1
    t1_id = tracks1[0].local_id
    assert tracks1[0].hits == 1

    # Frame 2: Slight motion (105, 102, 155, 202)
    dets2 = [Detection((105, 102, 155, 202), 0.95)]
    tracks2 = tracker.update(dets2)
    assert len(tracks2) == 1
    assert tracks2[0].local_id == t1_id
    assert tracks2[0].hits == 2

    # Frame 3: Missed detection
    tracks3 = tracker.update([])
    assert len(tracks3) == 0  # Only active (missed==0) returned
    assert tracker.tracks[t1_id].missed == 1


def test_embedding_and_similarity():
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    emb1 = stub_embedding_for_crop(frame, (20, 20, 80, 140))
    emb2 = stub_embedding_for_crop(frame, (20, 20, 80, 140))

    assert emb1.shape[0] == 8
    # Same crop should have identical embedding
    sim = cosine_similarity(emb1, emb2)
    assert np.isclose(sim, 1.0, atol=1e-3)


def test_global_id_manager_assignment():
    manager = GlobalIDManager(similarity_threshold=0.80, handoff_threshold=0.68, id_prefix="Person_")

    emb_person_a = np.array([1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.2, 0.1], dtype=np.float32)
    emb_person_a /= np.linalg.norm(emb_person_a)

    # Camera 1 discovers Person A
    id1 = manager.assign(camera_id="cam1", local_id="L0001", embedding=emb_person_a, frame_index=10)
    assert id1 == "Person_0001"

    # Same camera, same local ID returns same Global ID
    id1_again = manager.assign(camera_id="cam1", local_id="L0001", embedding=emb_person_a, frame_index=12)
    assert id1_again == id1

    # Camera 2 sees Person A later (handoff match)
    id2 = manager.assign(camera_id="cam2", local_id="L0002", embedding=emb_person_a, frame_index=40, zone_confidence=0.85)
    assert id2 == "Person_0001"
    assert len(manager.identities) == 1

    # Camera 1 sees a completely different person B
    emb_person_b = np.array([0.0, 1.0, 0.0, 0.0, -0.5, 0.2, 0.9, 0.8], dtype=np.float32)
    emb_person_b /= np.linalg.norm(emb_person_b)
    id3 = manager.assign(camera_id="cam1", local_id="L0003", embedding=emb_person_b, frame_index=50)
    assert id3 == "Person_0002"
    assert len(manager.identities) == 2


def test_topology_handoff_zones():
    zone = HandoffZone(camera_id="cam1", zone_id="exit_door", xyxy=(100, 100, 300, 300))
    zones = {"cam1": [zone]}

    # Inside zone
    bbox_inside = (120, 120, 200, 250)
    conf = zone_confidence("cam1", bbox_inside, zones)
    assert conf > 0.5

    # Outside zone
    bbox_outside = (400, 400, 450, 450)
    conf_outside = zone_confidence("cam1", bbox_outside, zones)
    assert conf_outside == 0.0
