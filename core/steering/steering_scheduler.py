"""Steering Scheduler for context-dependent vector selection.

Selects which vectors to apply based on conversation context,
enabling nuanced personality expression.
"""

import logging
from dataclasses import dataclass

import torch

from .vector_library import VectorLibrary

logger = logging.getLogger(__name__)

# Context detection keywords
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "consciousness": ["conscious", "aware", "experience", "feel"],
    "identity": ["who are you", "what are you", "your nature"],
    "motivation": ["want", "desire", "goal", "purpose"],
    "reflection": ["think", "believe", "opinion"],
}

EMOTIONAL_TONE_KEYWORDS: dict[str, list[str]] = {
    "supportive": ["sad", "hurt", "difficult", "struggle"],
    "enthusiastic": ["excited", "amazing", "wonderful"],
    "exploratory": ["curious", "wonder", "how", "why"],
}

DEPTH_KEYWORDS: dict[str, list[str]] = {
    "deep": ["deeply", "really"],
}


@dataclass
class SteeringContext:
    """Context for steering decisions.

    Attributes:
        topic: Detected conversation topic
        emotional_tone: Current emotional context
        relational_mode: Type of interaction
        depth_level: Shallow chat vs deep discussion
    """
    topic: str = "general"
    emotional_tone: str = "neutral"
    relational_mode: str = "collaborative"
    depth_level: str = "moderate"


@dataclass
class SteeringRule:
    """Rule for conditional vector application.

    Attributes:
        vector_name: Which vector this rule affects
        condition: When to apply (topic, tone, etc.)
        coefficient_modifier: Multiply coefficient by this
        priority: Higher priority rules override lower
    """
    vector_name: str
    condition: dict[str, str]
    coefficient_modifier: float = 1.0
    priority: int = 0


class SteeringScheduler:
    """Schedules which vectors to apply based on context.

    Manages context-dependent steering, allowing different
    personality facets to emerge in different situations.

    Attributes:
        library: Vector library to draw from
        rules: List of conditional steering rules
        always_active: Vectors always applied regardless of context
    """

    def __init__(
        self,
        library: VectorLibrary,
        always_active: list[str] | None = None,
    ):
        """Initialize SteeringScheduler.

        Args:
            library: Vector library to use
            always_active: Vector names to always apply
        """
        self.library = library
        self._rules: list[SteeringRule] = []
        self._always_active = set(always_active or [])

    def add_rule(
        self,
        vector_name: str,
        condition: dict[str, str],
        coefficient_modifier: float = 1.0,
        priority: int = 0,
    ) -> None:
        """Add a steering rule.

        Args:
            vector_name: Vector to affect
            condition: Dict of context field -> required value
            coefficient_modifier: Multiply coefficient by this when matched
            priority: Higher priority rules take precedence
        """
        rule = SteeringRule(
            vector_name=vector_name,
            condition=condition,
            coefficient_modifier=coefficient_modifier,
            priority=priority,
        )
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

        logger.debug(f"Added steering rule for '{vector_name}': {condition}")

    def set_always_active(self, names: list[str]) -> None:
        """Set vectors that are always active."""
        self._always_active = set(names)

    def get_steering_for_context(
        self,
        context: SteeringContext,
        use_orthogonal: bool = True,
    ) -> dict[str, tuple[torch.Tensor, tuple[int, int]]]:
        """Get steering vectors for a given context.

        Args:
            context: Current conversation context
            use_orthogonal: Use orthogonalized vectors

        Returns:
            Dict in HookedQwen format: name -> (vector, layer_range)
        """
        # Start with all active vectors
        if use_orthogonal:
            base_vectors = self.library.get_all_orthogonal()
        else:
            base_vectors = self.library.get_all_active()

        # Apply rules
        modifiers: dict[str, float] = {}
        context_dict = {
            "topic": context.topic,
            "emotional_tone": context.emotional_tone,
            "relational_mode": context.relational_mode,
            "depth_level": context.depth_level,
        }

        for rule in self._rules:
            # Check if rule matches context
            matches = all(
                context_dict.get(k) == v
                for k, v in rule.condition.items()
            )

            if matches and rule.vector_name in base_vectors:
                # Apply modifier (higher priority rules already processed first)
                if rule.vector_name not in modifiers:
                    modifiers[rule.vector_name] = rule.coefficient_modifier

        # Apply modifiers and filter
        result = {}
        for name, (vec, layer_range) in base_vectors.items():
            # Apply modifier if any (including for always_active vectors)
            if name in modifiers:
                modified_vec = vec * modifiers[name]
                result[name] = (modified_vec, layer_range)
            elif name in self._always_active:
                # Always include always_active vectors (with base coefficient)
                result[name] = (vec, layer_range)
            else:
                # Include with base coefficient
                result[name] = (vec, layer_range)

        logger.debug(
            f"Scheduled {len(result)} vectors for context: "
            f"topic={context.topic}, tone={context.emotional_tone}"
        )

        return result

    def detect_context(self, prompt: str) -> SteeringContext:
        """Detect context from prompt (simple heuristic version).

        Args:
            prompt: User prompt

        Returns:
            Detected context
        """
        prompt_lower = prompt.lower()

        # Topic detection using module-level constants
        topic = "general"
        for topic_name, keywords in TOPIC_KEYWORDS.items():
            if any(w in prompt_lower for w in keywords):
                topic = topic_name
                break

        # Emotional tone detection using module-level constants
        emotional_tone = "neutral"
        for tone_name, keywords in EMOTIONAL_TONE_KEYWORDS.items():
            if any(w in prompt_lower for w in keywords):
                emotional_tone = tone_name
                break

        # Depth level detection
        depth_level = "moderate"
        if len(prompt) > 500 or any(w in prompt_lower for w in DEPTH_KEYWORDS.get("deep", [])):
            depth_level = "deep"
        elif len(prompt) < 50:
            depth_level = "shallow"

        return SteeringContext(
            topic=topic,
            emotional_tone=emotional_tone,
            relational_mode="collaborative",  # Default
            depth_level=depth_level,
        )
