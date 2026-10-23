"""
topology/__init__.py
====================
Venue camera topology, learned transition model, spatiotemporal pruning and
blind-spot forecasting (Track P3).

Provides:
    - VenueTopologyGraph     : NetworkX camera graph with path-based transit windows
    - LogNormalTravelTime    : travel-time distribution used by the learned model
    - TrajectorySimulator    : seeded synthetic walkers + sighting logs + look-alikes
    - TransitionLearner      : learns P(Cj | Ci) and travel-time distributions from logs
    - SpatiotemporalPruner   : teleportation rejection and topology scoring of Candidates
    - BlindSpotForecaster    : next-camera SpatialPrediction with momentum
    - TrajectoryReconstructor: ordered TrajectoryPoints + blind-spot intervals
"""

from topology.graph import LogNormalTravelTime, VenueTopologyGraph, to_seconds
from topology.simulator import SimulationResult, SimulatorConfig, Sighting, TrajectorySimulator, Visit
from topology.learner import TransitionLearner
from topology.pruner import PairScore, SpatiotemporalPruner
from topology.forecaster import BlindSpotForecaster
from topology.trajectory import BlindSpotInterval, ReconstructionResult, TrajectoryReconstructor

__all__ = [
    "VenueTopologyGraph",
    "LogNormalTravelTime",
    "to_seconds",
    "TrajectorySimulator",
    "SimulatorConfig",
    "SimulationResult",
    "Sighting",
    "Visit",
    "TransitionLearner",
    "SpatiotemporalPruner",
    "PairScore",
    "BlindSpotForecaster",
    "TrajectoryReconstructor",
    "ReconstructionResult",
    "BlindSpotInterval",
]
