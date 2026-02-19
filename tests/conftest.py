"""Pytest configuration and fixtures."""

import pytest


def pytest_configure(config):
    """Rebuild Pydantic models with forward references at test collection time."""
    # Import HealthCategory first to make it available
    from core.cognitive.reflexion.schemas import HealthCategory

    # Now import AutonomousDecision and rebuild it
    from core.psyche.schema import AutonomousDecision

    AutonomousDecision.model_rebuild(_types_namespace={"HealthCategory": HealthCategory})
