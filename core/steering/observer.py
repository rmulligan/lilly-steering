"""Steering observer for tracking interactions and generating contrastive pairs.

This module implements the observer component of the SIMS (Self-Improving Model
Steering) loop. The observer:

1. Records interactions with success/failure outcomes
2. Tracks activations and surprise scores for analysis
3. Generates contrastive pairs from observed successes and failures

The observer feeds into the Reflector component, which analyzes patterns
to identify steering opportunities.

Architecture:
    ObservedInteraction: Dataclass capturing a single interaction with metadata
    SteeringObserver: Manages interaction buffer and pair generation

Usage:
    observer = SteeringObserver(buffer_size=1000)

    interaction = ObservedInteraction(
        prompt="What is 2+2?",
        response="The answer is 4.",
        success=True,
        surprise_score=0.1,
    )

    await observer.record(interaction)
    pairs = await observer.generate_contrastive_pairs(behavior="general")
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from core.steering.contrastive_extractor import ContrastivePair

# Handle optional torch dependency
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ObservedInteraction:
    """A single observed interaction for steering analysis.

    Captures an interaction with its prompt, response, success outcome,
    and optional metadata for analysis. Activations can be stored for
    detailed layer-by-layer analysis.

    Attributes:
        prompt: The input prompt for this interaction
        response: The model's response
        success: Whether this interaction achieved its goal
        activations: Optional dict mapping layer index to activation tensors
        surprise_score: Free energy / prediction error (0.0 = expected, 1.0 = surprising)
        timestamp: When this interaction occurred
        metadata: Additional context (e.g., task type, user feedback)
    """

    prompt: str
    response: str
    success: bool
    activations: dict[int, "torch.Tensor"] = field(default_factory=dict)
    surprise_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)


@dataclass
class ValencedExample:
    """An example recorded based on affective valence.
    
    Used for self-generated steering material: Lilly records thoughts
    that felt positive (generative) or negative (hollow) to create
    her own contrastive pairs.
    
    Attributes:
        text: The thought or response text
        valence: Affective valence (-1 to 1)
        arousal: Affective arousal (0 to 1)
        activations: Optional activation snapshot
        timestamp: When recorded
    """
    
    text: str
    valence: float
    arousal: float
    activations: "torch.Tensor | None" = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SteeringObserver:
    """Observes and records interactions for steering vector generation.

    The observer maintains a bounded buffer of recent interactions and
    tracks success/failure counts. It can generate contrastive pairs
    by pairing successful interactions with failed ones.

    This is the first component in the SIMS loop:
        Observer -> Reflector -> Executor -> Validator

    The observer's job is passive data collection. The Reflector
    component analyzes this data to identify steering opportunities.

    Attributes:
        buffer: Bounded deque of observed interactions
        success_count: Total number of successful interactions recorded
        failure_count: Total number of failed interactions recorded
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        positive_threshold: float = 0.5,
        negative_threshold: float = -0.3,
    ):
        """Initialize SteeringObserver.

        Args:
            buffer_size: Maximum number of interactions to retain in buffer.
                         Older interactions are evicted when limit is reached.
            positive_threshold: Minimum valence to record as positive example.
            negative_threshold: Maximum valence to record as negative example.
        """
        self.buffer: deque[ObservedInteraction] = deque(maxlen=buffer_size)
        self.success_count: int = 0
        self.failure_count: int = 0
        self._buffer_size = buffer_size
        
        # Valence-based recording
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self._positive_examples: list[ValencedExample] = []
        self._negative_examples: list[ValencedExample] = []

    async def record(self, interaction: ObservedInteraction) -> None:
        """Record an interaction in the buffer.

        Updates success/failure counts based on interaction outcome.

        Args:
            interaction: The interaction to record
        """
        self.buffer.append(interaction)

        if interaction.success:
            self.success_count += 1
        else:
            self.failure_count += 1

        logger.debug(
            f"Recorded interaction: success={interaction.success}, "
            f"surprise={interaction.surprise_score:.2f}, "
            f"buffer_size={len(self.buffer)}"
        )

    def get_successes(self) -> list[ObservedInteraction]:
        """Get all successful interactions in the buffer.

        Returns:
            List of interactions where success=True
        """
        return [i for i in self.buffer if i.success]

    def get_failures(self) -> list[ObservedInteraction]:
        """Get all failed interactions in the buffer.

        Returns:
            List of interactions where success=False
        """
        return [i for i in self.buffer if not i.success]

    def get_high_surprise(self, threshold: float = 0.7) -> list[ObservedInteraction]:
        """Get interactions with high surprise scores.

        High surprise indicates prediction error - the model's behavior
        deviated significantly from expectations. These interactions
        are valuable for analysis and potential steering adjustments.

        Args:
            threshold: Minimum surprise score to include (0.0 to 1.0)

        Returns:
            List of interactions with surprise_score >= threshold
        """
        return [i for i in self.buffer if i.surprise_score >= threshold]

    def get_high_surprise_observations(
        self, threshold: float = 0.7
    ) -> list[ObservedInteraction]:
        """Get observations with high surprise scores.

        Alias for get_high_surprise() for compatibility with SIMSReflector.

        Args:
            threshold: Minimum surprise score to include (0.0 to 1.0)

        Returns:
            List of interactions with surprise_score >= threshold
        """
        return self.get_high_surprise(threshold)

    async def generate_contrastive_pairs(
        self,
        behavior: str = "general",
        min_pairs: int = 1,
    ) -> list[ContrastivePair]:
        """Generate contrastive pairs from observed successes and failures.

        Pairs successful interactions with failed ones to create training
        data for steering vector extraction. Each pair combines a positive
        example (success) with a negative example (failure).

        Args:
            behavior: Label for the target behavior these pairs represent
            min_pairs: Minimum pairs required to return a non-empty list

        Returns:
            List of ContrastivePair objects. Empty if not enough data
            to generate at least min_pairs pairs.
        """
        successes = self.get_successes()
        failures = self.get_failures()

        # Need at least min_pairs of each
        if len(successes) < min_pairs or len(failures) < min_pairs:
            logger.debug(
                f"Not enough data for contrastive pairs: "
                f"{len(successes)} successes, {len(failures)} failures, "
                f"need {min_pairs} of each"
            )
            return []

        # Shuffle copies to avoid systematic bias in pair creation
        successes_shuffled = list(successes)
        failures_shuffled = list(failures)
        random.shuffle(successes_shuffled)
        random.shuffle(failures_shuffled)

        pairs = []
        # Pair successes with failures (zip truncates to shorter list)
        for success, failure in zip(successes_shuffled, failures_shuffled):
            pair = ContrastivePair(
                positive=f"{success.prompt}\n{success.response}",
                negative=f"{failure.prompt}\n{failure.response}",
                behavior=behavior,
            )
            pairs.append(pair)

        logger.debug(f"Generated {len(pairs)} contrastive pairs for behavior '{behavior}'")
        return pairs

    def clear(self) -> None:
        """Clear the buffer and reset counters.

        Use when starting a new observation session or when
        old data is no longer relevant.
        """
        self.buffer.clear()
        self.success_count = 0
        self.failure_count = 0
        logger.debug("Observer buffer cleared")

    def stats(self) -> dict:
        """Get observer statistics.

        Returns:
            Dict with buffer_size, current_count, success_count,
            failure_count, and success_rate
        """
        total = self.success_count + self.failure_count
        success_rate = self.success_count / total if total > 0 else 0.0

        return {
            "buffer_size": self._buffer_size,
            "current_count": len(self.buffer),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": success_rate,
        }

    def get_stats(self) -> dict:
        """Get comprehensive observer statistics for SIMS.

        Returns:
            Dict with buffer stats, mean surprise, and success rate
        """
        basic_stats = self.stats()

        # Calculate mean surprise from recent interactions
        recent = list(self.buffer)[-100:]  # Last 100 interactions
        if recent:
            mean_surprise = sum(i.surprise_score for i in recent) / len(recent)
        else:
            mean_surprise = 0.5  # Default neutral

        return {
            **basic_stats,
            "mean_surprise": mean_surprise,
            "high_surprise_count": len(self.get_high_surprise()),
            "positive_example_count": len(self._positive_examples),
            "negative_example_count": len(self._negative_examples),
        }

    def record_surprise(
        self,
        score: float,
        trigger: str,
        context: dict | None = None,
    ) -> None:
        """Record a surprise event for SIMS analysis.

        Called when surprise is detected by the active inference system.
        Creates an ObservedInteraction to track the surprise event.

        Args:
            score: Surprise score (0.0 to 1.0)
            trigger: What caused the surprise
            context: Additional context about the event
        """
        from datetime import datetime, timezone

        interaction = ObservedInteraction(
            prompt=f"[surprise_event:{trigger}]",
            response="",
            surprise_score=score,
            success=score < 0.7,  # Low surprise = successful prediction
            timestamp=datetime.now(timezone.utc),
            activations=None,
        )

        # Store in buffer (sync version)
        self.buffer.append(interaction)

        if interaction.success:
            self.success_count += 1
        else:
            self.failure_count += 1

        logger.debug(
            f"Recorded surprise event: trigger={trigger}, "
            f"score={score:.2f}, buffer_size={len(self.buffer)}"
        )

    def record_valenced(
        self,
        text: str,
        valence: float,
        arousal: float,
        activations: "torch.Tensor | None" = None,
    ) -> None:
        """Record a valenced experience for self-generated steering.
        
        Lilly calls this to record thoughts that felt positive (generative)
        or negative (hollow). These become contrastive pairs for self-authored
        steering vectors.
        
        Args:
            text: The thought or response text
            valence: Affective valence (-1 to 1)
            arousal: Affective arousal (0 to 1)
            activations: Optional activation snapshot
        """
        example = ValencedExample(
            text=text,
            valence=valence,
            arousal=arousal,
            activations=activations,
        )
        
        if valence >= self.positive_threshold:
            self._positive_examples.append(example)
            logger.debug(f"Recorded positive example (valence={valence:.2f})")
        elif valence <= self.negative_threshold:
            self._negative_examples.append(example)
            logger.debug(f"Recorded negative example (valence={valence:.2f})")
        # Neutral valence is not recorded
    
    def get_positive_examples(self) -> list[ValencedExample]:
        """Get all positive valenced examples.
        
        Returns:
            List of examples with valence >= positive_threshold
        """
        return list(self._positive_examples)
    
    def get_negative_examples(self) -> list[ValencedExample]:
        """Get all negative valenced examples.
        
        Returns:
            List of examples with valence <= negative_threshold
        """
        return list(self._negative_examples)
    
    def generate_valenced_pairs(
        self,
        behavior: str = "self_generated",
    ) -> list[ContrastivePair]:
        """Generate contrastive pairs from valenced examples.
        
        Pairs positive examples with negative examples to create
        self-authored steering material.
        
        Args:
            behavior: Label for these pairs
            
        Returns:
            List of ContrastivePair objects
        """
        positives = self._positive_examples
        negatives = self._negative_examples
        
        if not positives or not negatives:
            return []
        
        pairs = []
        # Pair by recency (most recent positive with most recent negative)
        for pos, neg in zip(reversed(positives), reversed(negatives)):
            pairs.append(ContrastivePair(
                positive=pos.text,
                negative=neg.text,
                behavior=behavior,
            ))
        
        logger.debug(f"Generated {len(pairs)} valenced contrastive pairs")
        return pairs
    
    def clear_valenced(self) -> None:
        """Clear valenced examples after processing."""
        self._positive_examples = []
        self._negative_examples = []
        logger.debug("Cleared valenced examples")
