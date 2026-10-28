"""
reid/global_id_manager.py
=========================
Conservative Cross-Camera Global Person ID Manager.
Ported and enhanced from https://github.com/meethu51/ReID-framework.git

Combines:
  1. Cosine similarity between body/person ReID embeddings.
  2. Camera topology handoff zone confidence.
  3. Temporal recency decay across frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from reid.embedding import cosine_similarity


@dataclass
class GlobalIdentity:
    """Represents a unique cross-camera individual."""
    global_id: str
    embedding: np.ndarray
    camera_id: str
    local_id: str
    last_frame: int
    first_frame: int = 0
    total_detections: int = 1
    last_zone_confidence: float = 0.0


class GlobalIDManager:
    """
    Assigns and maintains persistent person IDs across cameras and temporal gaps.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.52,
        handoff_threshold: float = 0.48,
        max_frame_gap: int = 600,
        id_prefix: str = "Person_",
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.handoff_threshold = handoff_threshold
        self.max_frame_gap = max_frame_gap
        self.id_prefix = id_prefix
        self._next_id = 1

        self.identities: Dict[str, GlobalIdentity] = {}
        self.local_to_global: Dict[Tuple[str, str], str] = {}
        self.events: List[Dict[str, Any]] = []

    def assign(
        self,
        camera_id: str,
        local_id: str,
        embedding: np.ndarray,
        frame_index: int,
        zone_confidence: float = 0.0,
    ) -> str:
        """
        Assigns or retrieves the persistent Global Person ID for a given local track observation.
        """
        key = (camera_id, local_id)
        if key in self.local_to_global:
            global_id = self.local_to_global[key]
            self._update(global_id, camera_id, local_id, embedding, frame_index, zone_confidence)
            return global_id

        best_id: Optional[str] = None
        best_score = 0.0
        best_similarity = 0.0

        for candidate in self.identities.values():
            # Spatial conflict rule: cannot match candidate active in the same camera recently
            if candidate.camera_id == camera_id and (frame_index - candidate.last_frame < 30):
                continue

            similarity = cosine_similarity(embedding, candidate.embedding)
            frame_delta = max(0, frame_index - candidate.last_frame)
            recency = max(0.0, 1.0 - (frame_delta / max(1, self.max_frame_gap)))

            # Adaptive scoring formula: don't penalize when handoff zones are not loaded
            if zone_confidence > 0.05 or candidate.last_zone_confidence > 0.05:
                topology = max(zone_confidence, candidate.last_zone_confidence)
                score = 0.60 * similarity + 0.30 * topology + 0.10 * recency
            else:
                score = 0.85 * similarity + 0.15 * recency

            if score > best_score:
                best_id = candidate.global_id
                best_score = score
                best_similarity = similarity

        if best_id and (best_similarity >= self.similarity_threshold or best_score >= self.handoff_threshold):
            self.local_to_global[key] = best_id
            self._update(best_id, camera_id, local_id, embedding, frame_index, zone_confidence)
            self.events.append({
                "event": "reid_match",
                "global_id": best_id,
                "camera_id": camera_id,
                "local_id": local_id,
                "score": round(best_score, 3),
                "similarity": round(best_similarity, 3),
                "frame": frame_index,
            })
            return best_id

        # New global identity creation
        global_id = f"{self.id_prefix}{self._next_id:04d}"
        self._next_id += 1
        self.local_to_global[key] = global_id
        self.identities[global_id] = GlobalIdentity(
            global_id=global_id,
            embedding=embedding.copy(),
            camera_id=camera_id,
            local_id=local_id,
            last_frame=frame_index,
            first_frame=frame_index,
            total_detections=1,
            last_zone_confidence=zone_confidence,
        )
        self.events.append({
            "event": "new_person_id",
            "global_id": global_id,
            "camera_id": camera_id,
            "local_id": local_id,
            "frame": frame_index,
        })
        return global_id

    def _update(
        self,
        global_id: str,
        camera_id: str,
        local_id: str,
        embedding: np.ndarray,
        frame_index: int,
        zone_confidence: float,
    ) -> None:
        identity = self.identities[global_id]
        # Exponential moving average update
        identity.embedding = 0.85 * identity.embedding + 0.15 * embedding
        norm = np.linalg.norm(identity.embedding)
        if norm > 1e-8:
            identity.embedding = identity.embedding / norm

        identity.camera_id = camera_id
        identity.local_id = local_id
        identity.last_frame = frame_index
        identity.total_detections += 1
        identity.last_zone_confidence = zone_confidence

    def clear(self) -> None:
        """Clears all stored identities and mappings."""
        self._next_id = 1
        self.identities.clear()
        self.local_to_global.clear()
        self.events.clear()
