"""Reflexion phase for self-monitoring and autonomous modification.

This module implements Phase 5 of Lilly's cognitive loop: Reflexion.
It enables self-monitoring of cognitive health metrics and autonomous
modification of parameters when thresholds are crossed.

The Reflexion phase runs after Integration and allows Lilly to:
- Monitor prediction accuracy, integration success, and coherence
- Classify system health into categories (THRIVING, STABLE, STRESSED, CRITICAL)
- Apply tiered modifications (RUNTIME, CONFIG, PROMPT) based on confidence
- Track modifications with revert conditions for safety
- Persist reflexion entries to Psyche with FOLLOWS and MODIFIED relationships

Exports:
    HealthCategory: Enum of health states (THRIVING through CRITICAL)
    HealthSignal: Single metric measurement with baseline and trend
    HealthAssessment: Container for all health signals with worst_category property
    ModificationTier: Enum of modification types with confidence thresholds
    Modification: A proposed parameter change with rationale and revert condition
    ReflexionResult: Output from reflexion phase with assessment and modifications
    ReflexionEntry: Persistent record for graph storage
    HealthSignalCollector: Aggregates metrics from Psyche and cognitive state
    HealthAssessor: Categorizes signals into THRIVING/STABLE/STRESSED/CRITICAL
    ModificationEngine: Proposes and applies 3-tier autonomous modifications
    ReflexionJournal: Persists reflexion entries to Psyche with relationships
    ReflexionPhase: Orchestrates the full reflexion cycle
    ReflexionEngine: Static methods for corrective experiment proposals
"""

from __future__ import annotations

from core.cognitive.reflexion.schemas import (
    HealthCategory,
    HealthSignal,
    HealthAssessment,
    ModificationTier,
    Modification,
    ReflexionResult,
    ReflexionEntry,
)
from core.cognitive.reflexion.signals import HealthSignalCollector
from core.cognitive.reflexion.assessment import HealthAssessor
from core.cognitive.reflexion.modifications import ModificationEngine
from core.cognitive.reflexion.journal import ReflexionJournal
from core.cognitive.reflexion.phase import ReflexionPhase
from core.cognitive.reflexion.engine import ReflexionEngine

__all__ = [
    "HealthCategory",
    "HealthSignal",
    "HealthAssessment",
    "ModificationTier",
    "Modification",
    "ReflexionResult",
    "ReflexionEntry",
    "HealthSignalCollector",
    "HealthAssessor",
    "ModificationEngine",
    "ReflexionJournal",
    "ReflexionPhase",
    "ReflexionEngine",
]
