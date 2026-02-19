"""Pit crew specialist system for Lilly development.

This module provides specialized Claude Code agents for subsystem development,
coordination, and reporting.
"""

from core.pit_crew.schemas import (
    HubTicket,
    ProposedChange,
    SpecialistReport,
    SpecialistType,
)
from core.pit_crew.spawner import spawn_specialist

__all__ = [
    "HubTicket",
    "ProposedChange",
    "SpecialistReport",
    "SpecialistType",
    "spawn_specialist",
]
