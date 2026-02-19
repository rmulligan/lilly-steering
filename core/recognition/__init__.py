"""Recognition Signal Loop for external grounding via Ryan's feedback.

This module implements external recognition signals (👍/👎/🤔) that:
- Provide ground truth for emergence detection
- Influence steering through preference learning
- Enable SAE feature attribution for approved behaviors

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    RECOGNITION SIGNAL LOOP                               │
    │                                                                          │
    │  Ryan's Signal (👍/👎/🤔)                                                │
    │       ↓                                                                  │
    │  [RecognitionSignal] ─→ [ValenceEvent] ─→ [PreferenceLearner]           │
    │       │                      │                    │                      │
    │       │                      ↓                    ↓                      │
    │       │              AffectiveResonator    LearnedPreference             │
    │       │                      │                    │                      │
    │       ↓                      ↓                    ↓                      │
    │  [SAE Feature Attribution]   Steering Bias   Intuition (via dreams)     │
    │       │                                                                  │
    │       ↓                                                                  │
    │  ApprovedFeatureTracker ─→ Emergence fitness signal                     │
    └─────────────────────────────────────────────────────────────────────────┘
"""

from core.recognition.schema import RecognitionSignal, SignalType, RecognitionStats
from core.recognition.file_watcher import RecognitionFileWatcher
from core.recognition.translator import translate_to_valence_event, SIGNAL_VALENCE_MAP
from core.recognition.feature_tracker import ApprovedFeatureTracker, ApprovedFeaturePattern
from core.recognition.discord_analyzer import DiscordSentimentAnalyzer

__all__ = [
    "RecognitionSignal",
    "SignalType",
    "RecognitionStats",
    "RecognitionFileWatcher",
    "translate_to_valence_event",
    "SIGNAL_VALENCE_MAP",
    "ApprovedFeatureTracker",
    "ApprovedFeaturePattern",
    "DiscordSentimentAnalyzer",
]
