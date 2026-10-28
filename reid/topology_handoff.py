"""
reid/topology_handoff.py
========================
Manual and topological handoff-zone helpers for cross-camera Re-ID.
Ported and adapted from https://github.com/meethu51/ReID-framework.git
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class HandoffZone:
    """A spatial handoff region in a camera view."""
    camera_id: str
    zone_id: str
    xyxy: Tuple[int, int, int, int]


def load_zones(path: Optional[str]) -> Dict[str, List[HandoffZone]]:
    """Loads handoff zone definitions from a JSON file."""
    if not path or not Path(path).exists():
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        zone_items = data.get("zones", [])
        zones: Dict[str, List[HandoffZone]] = {}
        for item in zone_items:
            if item.get("type", "rect") != "rect":
                continue
            camera_id = str(item["camera_id"])
            zone = HandoffZone(
                camera_id=camera_id,
                zone_id=str(item.get("zone_id", "zone")),
                xyxy=tuple(int(v) for v in item["xyxy"]),
            )
            zones.setdefault(camera_id, []).append(zone)
        return zones
    except Exception as e:
        print(f"[HandoffZones] Warning: failed to parse {path}: {e}")
        return {}


def intersection_ratio(bbox: Tuple[int, int, int, int], zone: HandoffZone) -> float:
    """Calculates intersection area ratio relative to bbox area."""
    x1, y1, x2, y2 = bbox
    zx1, zy1, zx2, zy2 = zone.xyxy
    ix1, iy1 = max(x1, zx1), max(y1, zy1)
    ix2, iy2 = min(x2, zx2), min(y2, zy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(1, (x2 - x1) * (y2 - y1))
    return float(inter / area)


def center_inside(bbox: Tuple[int, int, int, int], zone: HandoffZone) -> bool:
    """Returns True if the bbox center falls strictly inside the handoff zone."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    zx1, zy1, zx2, zy2 = zone.xyxy
    return (zx1 <= cx <= zx2) and (zy1 <= cy <= zy2)


def zone_confidence(
    camera_id: str,
    bbox: Tuple[int, int, int, int],
    zones: Dict[str, List[HandoffZone]],
) -> float:
    """
    Computes a topological handoff confidence score [0.0, 1.0] indicating
    if the person is currently in a known camera transition/handoff region.
    """
    scores = []
    for zone in zones.get(camera_id, []):
        overlap = intersection_ratio(bbox, zone)
        center_bonus = 0.35 if center_inside(bbox, zone) else 0.0
        scores.append(min(1.0, overlap + center_bonus))
    return float(max(scores, default=0.0))
