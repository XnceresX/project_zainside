"""
contract/schema.py
==================
Shared Data Contract for SAFEPASS.
This file defines the frozen schemas used across all four tracks (P1, P2, P3, P4).
Rule: Any modifications must be agreed upon by all team members.
"""

from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field


class QualityMetrics(BaseModel):
    """Quality estimation scores for a detected person / face crop (P1 & P2)."""
    blur_score: float = Field(..., ge=0.0, le=1.0, description="Normalized Laplacian variance: 0 (blurry) to 1 (sharp)")
    occlusion_score: float = Field(..., ge=0.0, le=1.0, description="0 (fully visible) to 1 (heavily occluded)")
    face_detected: bool = Field(..., description="True if valid face landmarks/crop present")
    face_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Face detector confidence score")
    recommended_face_weight: float = Field(0.5, ge=0.0, le=1.0, description="Dynamic weight for face in fusion")
    recommended_appearance_weight: float = Field(0.5, ge=0.0, le=1.0, description="Dynamic weight for appearance in fusion")
    recommended_color_weight: float = Field(0.0, ge=0.0, le=1.0, description="Dynamic weight for clothing color in fusion")


class ModalitySimilarities(BaseModel):
    """Normalized cosine similarities across modalities."""
    face_similarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Face embedding cosine similarity")
    appearance_similarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="OSNet body appearance cosine similarity")
    color_similarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="HSV clothing color histogram similarity")
    gait_similarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional gait similarity")
    fused_score: float = Field(..., ge=0.0, le=1.0, description="Quality-weighted fused score")


class Candidate(BaseModel):
    """A single candidate detection evaluated across feeds."""
    case_id: str
    candidate_id: str
    camera_id: str
    camera_name: str
    timestamp: str = Field(..., description="ISO 8601 formatted timestamp string")
    frame_crop_path: Optional[str] = None
    face_score: Optional[float] = None
    appearance_score: Optional[float] = None
    color_score: Optional[float] = None
    q_face: float = Field(1.0, description="Face quality score [0.0 - 1.0]")
    q_body: float = Field(1.0, description="Body appearance quality score [0.0 - 1.0]")
    fused_score: float = Field(..., description="Quality-weighted fused score [0.0 - 1.0]")
    topology_score: Optional[float] = Field(None, description="Spatial transition likelihood")
    is_spatially_valid: bool = Field(True, description="True if motion satisfies topology constraints")


class CameraNode(BaseModel):
    """Metadata for a CCTV node in the venue topology graph (P3)."""
    camera_id: str
    camera_name: str
    zone_name: str
    connected_cameras: List[str] = Field(default_factory=list, description="Physically reachable adjacent camera IDs")
    transit_bounds: Dict[str, Tuple[int, int]] = Field(
        default_factory=dict,
        description="Map of destination_camera_id -> (min_transit_seconds, max_transit_seconds)"
    )


class SpatialPrediction(BaseModel):
    """Predicted next search regions based on camera transition graph (P3)."""
    current_camera_id: str
    candidate_next_cameras: List[str] = Field(..., description="Ordered list of high-probability adjacent cameras")
    transition_probabilities: List[float] = Field(..., description="Probabilities corresponding to next cameras")
    estimated_arrival_window_seconds: Tuple[int, int] = Field(..., description="Min and max transit window (seconds)")


class TrajectoryPoint(BaseModel):
    """Chronological waypoint for trajectory reconstruction."""
    camera_id: str
    camera_name: str
    timestamp: str
    confidence: float
    snapshot_url: Optional[str] = None


class SearchCaseResult(BaseModel):
    """Complete search query response for operator dashboard (P4)."""
    case_id: str
    reported_person_name: str
    report_timestamp: str
    reference_photo_path: Optional[str] = None
    last_seen_camera_id: str
    status: str = Field("active", description="active | resolved | pending_review")
    candidates: List[Candidate] = Field(default_factory=list)
    reconstructed_trajectory: List[TrajectoryPoint] = Field(default_factory=list)
    predicted_next_zone: Optional[SpatialPrediction] = None


class IntakeRequest(BaseModel):
    """Case intake data submission model (P4)."""
    person_name: str
    guardian_name: str
    guardian_contact: str
    age_category: str = Field("child", description="child | teen | adult | elderly")
    upper_clothing_color: str
    lower_clothing_color: str
    last_seen_camera_id: str
    notes: Optional[str] = None
    consent_confirmed: bool = Field(True, description="DPDP Act 2023 guardian consent confirmation")
