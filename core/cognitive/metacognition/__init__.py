"""Metacognition phase for bird's-eye pattern detection.

This module provides a local alternative to cloud-based Letta agents,
running Gemma-2-9B-IT to analyze patterns across cognitive cycles.
"""

from core.cognitive.metacognition.buffer import CycleSummary, MetacognitionBuffer
from core.cognitive.metacognition.memory import MetacognitionMemory
from core.cognitive.metacognition.phase import MetacognitionPhase

__all__ = [
    "CycleSummary",
    "MetacognitionBuffer",
    "MetacognitionMemory",
    "MetacognitionPhase",
]
