"""
reid/tracker.py
===============
Lightweight Centroid and Motion Person Tracker per Camera view.
Ported and adapted from https://github.com/meethu51/ReID-framework.git
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class Detection:
    """Represents a person bounding box detection."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float = 1.0
    label: str = "person"


@dataclass
class Track:
    """Represents a tracked person in a specific camera view."""
    local_id: str
    bbox: Tuple[int, int, int, int]
    age: int = 0
    missed: int = 0
    hits: int = 1
    history: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> int:
        return max(1, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(1, self.bbox[3] - self.bbox[1])


def box_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
    return float(iou)


class SimpleTracker:
    """
    Per-camera centroid and IoU tracker maintaining stable local track IDs across frames.
    """

    def __init__(self, max_distance: float = 120.0, max_missed: int = 15, min_hits: int = 1) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.min_hits = min_hits
        self._next_id = 1
        self.tracks: Dict[str, Track] = {}

    def update(self, detections: List[Detection]) -> List[Track]:
        unmatched = set(range(len(detections)))
        used_tracks: Set[str] = set()

        # Match existing tracks to closest detections using centroid distance & IoU
        for track in list(self.tracks.values()):
            best_index: Optional[int] = None
            best_cost = float("inf")
            tx, ty = track.center

            for index in unmatched:
                d_box = detections[index].bbox
                dx1, dy1, dx2, dy2 = d_box
                cx, cy = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
                dist = hypot(cx - tx, cy - ty)
                iou = box_iou(track.bbox, d_box)

                # Composite cost: distance penalty minus IoU bonus
                cost = dist - (iou * 80.0)

                if dist < self.max_distance and cost < best_cost:
                    best_cost = cost
                    best_index = index

            if best_index is None:
                track.missed += 1
                continue

            detection = detections[best_index]
            # Smooth bbox transition (0.75 new + 0.25 previous)
            px1, py1, px2, py2 = track.bbox
            nx1, ny1, nx2, ny2 = detection.bbox
            smooth_box = (
                int(0.25 * px1 + 0.75 * nx1),
                int(0.25 * py1 + 0.75 * ny1),
                int(0.25 * px2 + 0.75 * nx2),
                int(0.25 * py2 + 0.75 * ny2),
            )
            track.bbox = smooth_box
            track.age += 1
            track.missed = 0
            track.hits += 1
            track.history.append(track.center)
            unmatched.remove(best_index)
            used_tracks.add(track.local_id)

        # Create new tracks for unmatched detections
        for index in unmatched:
            local_id = f"L{self._next_id:04d}"
            self._next_id += 1
            track = Track(local_id=local_id, bbox=detections[index].bbox)
            track.history.append(track.center)
            self.tracks[local_id] = track
            used_tracks.add(local_id)

        # Evict dead tracks
        for local_id in list(self.tracks):
            if self.tracks[local_id].missed > self.max_missed:
                del self.tracks[local_id]

        # Only return confirmed active tracks (hits >= min_hits and not currently missed)
        return [
            track for track in self.tracks.values()
            if track.missed == 0 and track.hits >= self.min_hits
        ]
