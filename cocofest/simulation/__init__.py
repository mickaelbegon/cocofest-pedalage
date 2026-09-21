"""Lightweight simulation setup, validation and launch planning (no solver imports)."""
from .config import SimulationConfig
from .capabilities import CapabilityRegistry, ConfigurationError, ValidationIssue
from .launch import LaunchPlan, build_launch_plan
from .independent_arms import (
    ArmRunResult,
    ArmRuntimeParameters,
    IndependentArmConfig,
    IndependentArmCoordinator,
    ParametricArmSolver,
)

__all__ = ["SimulationConfig", "CapabilityRegistry", "ConfigurationError", "ValidationIssue",
           "LaunchPlan", "build_launch_plan", "ArmRunResult", "ArmRuntimeParameters",
           "IndependentArmConfig", "IndependentArmCoordinator", "ParametricArmSolver"]
