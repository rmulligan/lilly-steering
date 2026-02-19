"""Episode selection based on cognitive signals.

Selects the next episode type based on context from the cognitive state,
including open questions, entity focus, tensions, and recent insights.
Emotion-aware selection modulates weights based on affective state.
"""

import random
from dataclasses import dataclass, field
from typing import Optional

from core.cognitive.episode import EpisodeType


@dataclass
class EmotionalSignals:
    """Emotional context for episode selection.

    These signals come from the affective system and modulate
    episode selection to prevent cognitive ruts and encourage
    novelty when emotional state indicates boredom.

    Attributes:
        is_bored: True if affective state indicates boredom
                  (mild disgust + low anticipation in Plutchik model)
        diversity_signal: Continuous score (0-1) for novelty-seeking,
                          higher values indicate desire for topic variety
        is_conflicted: True if opposite emotions are both elevated
                       (emotional ambivalence), signals dialectical tension
    """

    is_bored: bool = False
    diversity_signal: float = 0.0
    is_conflicted: bool = False


@dataclass
class SelectionSignals:
    """Signals from cognitive state that inform episode selection.

    These signals capture the current cognitive context and are used
    by the EpisodeSelector to choose the most appropriate episode type.

    Attributes:
        open_questions: Questions that emerged but weren't resolved
        entity_focus: Entity currently being explored (if any)
        recent_insight_count: Number of insights generated recently
        tension_level: Current cognitive tension (0.0-1.0)
        last_episode_type: The type of the most recent episode (if any)
        time_since_synthesis: Seconds since last synthesis episode
        time_since_reflection: Seconds since last meta-reflection
    """

    open_questions: list[str] = field(default_factory=list)
    entity_focus: str | None = None
    recent_insight_count: int = 0
    tension_level: float = 0.0
    last_episode_type: str | None = None
    time_since_synthesis: float = 0.0
    time_since_reflection: float = 0.0


class EpisodeSelector:
    """Selects the next episode type based on cognitive signals.

    Uses a weighted scoring system that considers:
    - Open questions → QUESTION_PURSUIT
    - Entity focus → MEMORY_ARCHAEOLOGY
    - High tension → DIALECTICAL_DEBATE
    - Many recent insights → SYNTHESIS
    - Long time since synthesis → SYNTHESIS
    - Long time since reflection → META_REFLECTION
    - Boredom → CREATIVE, MEMORY_ARCHAEOLOGY (novelty-seeking)
    - Randomness for variety → Any type

    The selector avoids repeating the same episode type consecutively
    and ensures variety through weighted random selection. Emotional
    signals modulate selection to prevent cognitive ruts.
    """

    # Thresholds for triggering specific episode types
    QUESTION_THRESHOLD = 2  # Open questions to trigger QUESTION_PURSUIT
    INSIGHT_THRESHOLD = 3  # Recent insights to trigger SYNTHESIS
    TENSION_THRESHOLD = 0.7  # Tension level to trigger DIALECTICAL_DEBATE
    SYNTHESIS_COOLDOWN = 1800.0  # 30 minutes before synthesis needed
    REFLECTION_COOLDOWN = 3600.0  # 1 hour before reflection needed
    DIVERSITY_THRESHOLD = 0.5  # Diversity signal to boost novelty-seeking

    # Base weights for each episode type (out of 100)
    BASE_WEIGHTS: dict[EpisodeType, float] = {
        EpisodeType.DEEP_DIVE: 25.0,
        EpisodeType.DIALECTICAL_DEBATE: 15.0,
        EpisodeType.MEMORY_ARCHAEOLOGY: 15.0,
        EpisodeType.QUESTION_PURSUIT: 15.0,
        EpisodeType.HYPOTHESIS_SIMULATION: 15.0,
        EpisodeType.SYNTHESIS: 10.0,
        EpisodeType.CREATIVE: 10.0,
        EpisodeType.META_REFLECTION: 10.0,
    }

    # Weight adjustments for emotional modulation
    BOREDOM_CREATIVE_BOOST = 25.0  # Boost CREATIVE when bored
    BOREDOM_ARCHAEOLOGY_BOOST = 20.0  # Boost MEMORY_ARCHAEOLOGY for novelty
    BOREDOM_DEEP_DIVE_PENALTY = 0.5  # 50% reduction to DEEP_DIVE when bored
    DIVERSITY_BOOST_SCALE = 15.0  # Scaled by diversity_signal
    DIVERSITY_ARCHAEOLOGY_BOOST_FACTOR = 0.8  # MEMORY_ARCHAEOLOGY gets 80% of boost
    DIVERSITY_HYPOTHESIS_BOOST_FACTOR = 0.6  # HYPOTHESIS_SIMULATION gets 60% of boost
    CONFLICT_DIALECTICAL_BOOST = 30.0  # Strong boost to DIALECTICAL_DEBATE for ambivalence

    def select(
        self,
        signals: SelectionSignals,
        emotional_signals: Optional[EmotionalSignals] = None,
    ) -> EpisodeType:
        """Select the next episode type based on cognitive signals.

        Args:
            signals: Current cognitive context
            emotional_signals: Optional emotional modulation from affective system

        Returns:
            The selected EpisodeType for the next episode
        """
        weights = self._calculate_weights(signals, emotional_signals)
        return self._weighted_random_choice(weights)

    def _calculate_weights(
        self,
        signals: SelectionSignals,
        emotional_signals: Optional[EmotionalSignals] = None,
    ) -> dict[EpisodeType, float]:
        """Calculate weights for each episode type based on signals.

        Args:
            signals: Current cognitive context
            emotional_signals: Optional emotional modulation

        Returns:
            Dictionary mapping episode types to their selection weights
        """
        weights = dict(self.BASE_WEIGHTS)

        # Boost QUESTION_PURSUIT if we have open questions
        if len(signals.open_questions) >= self.QUESTION_THRESHOLD:
            weights[EpisodeType.QUESTION_PURSUIT] += 30.0

        # Boost MEMORY_ARCHAEOLOGY if we have an entity focus
        if signals.entity_focus:
            weights[EpisodeType.MEMORY_ARCHAEOLOGY] += 25.0

        # Boost DIALECTICAL_DEBATE if tension is high
        if signals.tension_level >= self.TENSION_THRESHOLD:
            weights[EpisodeType.DIALECTICAL_DEBATE] += 25.0

        # Boost SYNTHESIS if we have many recent insights
        if signals.recent_insight_count >= self.INSIGHT_THRESHOLD:
            weights[EpisodeType.SYNTHESIS] += 30.0

        # Boost SYNTHESIS if it's been a while since one
        if signals.time_since_synthesis >= self.SYNTHESIS_COOLDOWN:
            weights[EpisodeType.SYNTHESIS] += 20.0

        # Boost META_REFLECTION if it's been a while
        if signals.time_since_reflection >= self.REFLECTION_COOLDOWN:
            weights[EpisodeType.META_REFLECTION] += 30.0

        # Reduce weight of last episode type to avoid repetition
        if signals.last_episode_type:
            for episode_type in EpisodeType:
                if episode_type.value == signals.last_episode_type:
                    weights[episode_type] *= 0.3  # 70% reduction
                    break

        # Apply emotional modulation if provided
        if emotional_signals is not None:
            weights = self._apply_emotional_modulation(weights, emotional_signals)

        return weights

    def _apply_emotional_modulation(
        self,
        weights: dict[EpisodeType, float],
        emotional_signals: EmotionalSignals,
    ) -> dict[EpisodeType, float]:
        """Apply emotional modulation to episode weights.

        When bored:
        - Boost CREATIVE and MEMORY_ARCHAEOLOGY for novelty
        - Reduce DEEP_DIVE to break repetitive focus

        High diversity signal:
        - Boost novelty-seeking types proportionally

        When conflicted (emotional ambivalence):
        - Boost DIALECTICAL_DEBATE to process tension

        Args:
            weights: Current weight dictionary
            emotional_signals: Emotional context

        Returns:
            Modified weight dictionary
        """
        # Boredom-based adjustments (Plutchik: mild disgust + low anticipation)
        if emotional_signals.is_bored:
            weights[EpisodeType.CREATIVE] += self.BOREDOM_CREATIVE_BOOST
            weights[EpisodeType.MEMORY_ARCHAEOLOGY] += self.BOREDOM_ARCHAEOLOGY_BOOST
            weights[EpisodeType.DEEP_DIVE] *= self.BOREDOM_DEEP_DIVE_PENALTY

        # Continuous diversity signal modulation
        if emotional_signals.diversity_signal >= self.DIVERSITY_THRESHOLD:
            # Scale boost by diversity signal strength
            boost = emotional_signals.diversity_signal * self.DIVERSITY_BOOST_SCALE
            weights[EpisodeType.CREATIVE] += boost
            weights[EpisodeType.MEMORY_ARCHAEOLOGY] += boost * self.DIVERSITY_ARCHAEOLOGY_BOOST_FACTOR
            weights[EpisodeType.HYPOTHESIS_SIMULATION] += boost * self.DIVERSITY_HYPOTHESIS_BOOST_FACTOR

        # Conflict-based adjustments (opposite emotions both elevated)
        if emotional_signals.is_conflicted:
            weights[EpisodeType.DIALECTICAL_DEBATE] += self.CONFLICT_DIALECTICAL_BOOST

        return weights

    def _weighted_random_choice(self, weights: dict[EpisodeType, float]) -> EpisodeType:
        """Select an episode type using weighted random selection.

        Args:
            weights: Dictionary mapping episode types to their weights

        Returns:
            Randomly selected EpisodeType weighted by the provided weights
        """
        types = list(weights.keys())
        type_weights = [weights[t] for t in types]
        total = sum(type_weights)
        normalized = [w / total for w in type_weights]

        return random.choices(types, weights=normalized, k=1)[0]
