"""Inquiry goal detection and management for progressive thinking."""
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from core.cognitive.stage import CognitiveStage


@dataclass
class InquiryGoal:
    """An active inquiry that Lilly is pursuing through dialectical stages."""

    question: str
    emerged_from: str
    stage: CognitiveStage = CognitiveStage.QUESTION
    stage_cycles: int = 0
    insights: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uid: Optional[str] = None  # Set when persisted to graph

    def with_update(self, **kwargs) -> "InquiryGoal":
        """Create a new goal with updated fields (immutable update pattern)."""
        return InquiryGoal(
            question=kwargs.get("question", self.question),
            emerged_from=kwargs.get("emerged_from", self.emerged_from),
            stage=kwargs.get("stage", self.stage),
            stage_cycles=kwargs.get("stage_cycles", self.stage_cycles),
            insights=kwargs.get("insights", self.insights.copy()),
            created_at=kwargs.get("created_at", self.created_at),
            uid=kwargs.get("uid", self.uid),
        )

    def with_uid(self, uid: str) -> "InquiryGoal":
        """Return a copy with the given uid set."""
        return self.with_update(uid=uid)

    def add_insight(self, insight: str) -> "InquiryGoal":
        """Add an insight and return updated goal."""
        new_insights = self.insights.copy()
        new_insights.append(insight)
        return self.with_update(insights=new_insights)

    def increment_cycle(self) -> "InquiryGoal":
        """Increment stage cycle counter."""
        return self.with_update(stage_cycles=self.stage_cycles + 1)


# Goal detection signals with confidence scores
GOAL_SIGNALS: List[Tuple[str, float]] = [
    (r"I wonder (if|why|how|what|whether)", 0.8),
    (r"I don't understand", 0.9),
    (r"This contradicts", 0.95),
    (r"What does it mean (to|that|for|when)", 0.85),
    (r"I assumed .* but", 0.9),
    (r"How can .* be", 0.75),
    (r"Why (do|does|is|are|would|should)", 0.7),
    (r"What if .* (is|are|were|was) (wrong|different|true)", 0.85),
]

GOAL_THRESHOLD = 0.7


def extract_goal_question(thought: str) -> Optional[str]:
    """Extract the core question from a thought."""
    # Try specific patterns first
    patterns = [
        r"I wonder (if|whether|why|how|what) ([^.?!]+[.?!]?)",
        r"(What does it mean [^.?!]+[.?!]?)",
        r"(How can [^.?!]+[.?!]?)",
        r"(Why [^.?!]+[.?!]?)",
        r"I don't understand (why|how|what) ([^.?!]+[.?!]?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, thought, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) >= 2:
                return f"{groups[0]} {groups[1]}".strip()
            return match.group(0).strip()

    # Fallback: find sentence with signal words
    sentences = re.split(r'[.!?]+', thought)
    signal_words = ["wonder", "understand", "contradict", "mean", "assume", "why", "how", "what"]
    for sentence in sentences:
        if any(word in sentence.lower() for word in signal_words):
            return sentence.strip()

    # Last resort: use truncated thought
    return thought[:100].strip() if thought else None


def detect_emerging_goal(
    thought: str,
    current_goal: Optional[InquiryGoal],
) -> Optional[InquiryGoal]:
    """Detect if a new inquiry goal should emerge from this thought."""
    # Don't spawn new goals while one is active (unless in COMMIT stage)
    if current_goal is not None and current_goal.stage != CognitiveStage.COMMIT:
        return None

    # Check for goal-worthy signals
    max_confidence = 0.0
    for pattern, confidence in GOAL_SIGNALS:
        if re.search(pattern, thought, re.IGNORECASE):
            if confidence > max_confidence:
                max_confidence = confidence

    # Only create goal if confidence exceeds threshold
    if max_confidence < GOAL_THRESHOLD:
        return None

    # Extract the question from the thought
    question = extract_goal_question(thought)
    if not question:
        return None

    return InquiryGoal(
        question=question,
        emerged_from=thought[:200],  # Truncate for storage
        stage=CognitiveStage.QUESTION,
        stage_cycles=0,
        insights=[],
    )
