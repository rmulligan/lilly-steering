"""Concept diversity enhancement for human-like mind wandering.

This module prevents concept repetition through cooldowns and enables
discovery of new conceptual territory through multiple novelty sources.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.embedding.service import TieredEmbeddingService
    from core.model.curator_model import CuratorModel
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)


@dataclass
class NoveltySourceConfig:
    """Configuration for a single novelty source."""
    enabled: bool = True
    frequency: int = 10  # For periodic sources (every N cycles)


@dataclass
class NoveltyConfig:
    """Configuration for all novelty sources."""
    external_seeding: NoveltySourceConfig = field(
        default_factory=lambda: NoveltySourceConfig(enabled=True)
    )
    generative_curiosity: NoveltySourceConfig = field(
        default_factory=lambda: NoveltySourceConfig(enabled=True, frequency=10)
    )
    analogical_bridging: NoveltySourceConfig = field(
        default_factory=lambda: NoveltySourceConfig(enabled=True)
    )
    random_walk: NoveltySourceConfig = field(
        default_factory=lambda: NoveltySourceConfig(enabled=True, frequency=20)
    )
    driving_questions: NoveltySourceConfig = field(
        default_factory=lambda: NoveltySourceConfig(enabled=True)
    )


# Default configuration instance
NOVELTY_CONFIG = NoveltyConfig()

# Score bonus for novelty source candidates
NOVELTY_BONUS = 1.5

# Cooldown constants
CONCEPT_HARD_COOLDOWN = 10   # Can't use same concept within 10 cycles
CONCEPT_SOFT_COOLDOWN = 30   # Penalized until 30 cycles pass


def is_concept_available(
    concept: str,
    current_cycle: int,
    last_used: dict[str, int],
) -> tuple[bool, float]:
    """Check if concept is available and return availability score.

    Args:
        concept: The concept name to check
        current_cycle: Current cognitive cycle number
        last_used: Dict mapping concept names to cycle when last used

    Returns:
        Tuple of (is_available, availability_score).
        - (False, 0.0) if within hard cooldown (10 cycles)
        - (True, 0.0-1.0) if in soft cooldown (10-30 cycles)
        - (True, 1.0) if fully recovered (>30 cycles or never used)
    """
    if concept not in last_used:
        return True, 1.0  # Never used = fully available

    cycles_since = current_cycle - last_used[concept]

    # Hard cooldown - completely unavailable
    if cycles_since < CONCEPT_HARD_COOLDOWN:
        return False, 0.0

    # Soft cooldown - gradually recover
    if cycles_since < CONCEPT_SOFT_COOLDOWN:
        recovery = (cycles_since - CONCEPT_HARD_COOLDOWN) / (CONCEPT_SOFT_COOLDOWN - CONCEPT_HARD_COOLDOWN)
        return True, recovery  # 0.0 at cycle 10, 1.0 at cycle 30

    return True, 1.0  # Fully recovered


async def generate_curious_concept(
    current_thought: str,
    recent_concepts: list[str],
    curator_model: "CuratorModel",
) -> tuple[str, str]:
    """Ask LLM to propose unexplored adjacent territory.

    Args:
        current_thought: The current thought content
        recent_concepts: List of recently explored concepts
        curator_model: The curator model for generation

    Returns:
        Tuple of (concept_name, source_tag)
    """
    prompt = f"""Current thought: {current_thought}
Recently explored: {', '.join(recent_concepts[-10:])}

What concept or domain would be interesting to explore that feels
adjacent to this thought but hasn't been touched? Name ONE concept
that could spark a surprising new direction.

Respond with just the concept name, nothing else."""

    result = await curator_model.generate(prompt, max_tokens=20)
    concept = result.text.strip()

    return concept, "generative:curiosity"


def _extract_numbered_concepts(text: str) -> list[str]:
    """Extract concepts from numbered list format like '1. [concept]'."""
    pattern = r'\d+\.\s*\[([^\]]+)\]'
    matches = re.findall(pattern, text)
    return [m.strip() for m in matches if m.strip()]


async def find_analogous_domains(
    concept: str,
    pattern: str,
    curator_model: "CuratorModel",
) -> list[tuple[str, str]]:
    """Find other fields exhibiting the same pattern.

    Args:
        concept: The current concept being explored
        pattern: The pattern or behavior observed in this concept
        curator_model: The curator model for generation

    Returns:
        List of (concept_name, source_tag) tuples from different domains
    """
    prompt = f"""The concept "{concept}" involves this pattern: {pattern}

What 3 completely different domains (science, art, nature, social systems,
technology, philosophy) exhibit this SAME pattern?

For each domain, name ONE specific concept. Format:
1. [concept1]
2. [concept2]
3. [concept3]"""

    result = await curator_model.generate(prompt, max_tokens=100)
    concepts = _extract_numbered_concepts(result.text)

    return [(c, "analogical:bridging") for c in concepts]


async def random_embedding_walk(
    current_embedding: np.ndarray,
    curator_model: "CuratorModel",
    step_size: float = 0.3,
) -> tuple[str, str]:
    """Move randomly in embedding space and generate concept for that region.

    Args:
        current_embedding: Current position in embedding space
        curator_model: The curator model for generation
        step_size: How far to step in embedding space (absolute magnitude)

    Returns:
        Tuple of (concept_name, source_tag)
    """
    # Random unit direction
    direction = np.random.randn(len(current_embedding))
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    # For the extremely rare case of zero-norm, keep unnormalized (will still work)

    # New position
    new_position = current_embedding + direction * step_size

    # Describe the semantic region (using top dimensions)
    top_dims = np.argsort(np.abs(new_position))[-5:]
    region_desc = f"semantic region with high values in dimensions {top_dims.tolist()}"

    prompt = f"""I'm exploring a {region_desc} in concept space.

What interesting concept might live in this unexplored semantic territory?
Name ONE concept that feels like it belongs in a different neighborhood
than typical cognitive/philosophical concepts.

Respond with just the concept name."""

    result = await curator_model.generate(prompt, max_tokens=20)
    concept = result.text.strip()  # NOTE: Use result.text not result directly

    return concept, "random:walk"


async def question_driven_concepts(
    open_questions: list[str],
    curator_model: "CuratorModel",
    max_questions: int = 3,
) -> list[tuple[str, str]]:
    """Let unanswered questions suggest new territory.

    Args:
        open_questions: List of open questions to explore
        curator_model: The curator model for generation
        max_questions: Maximum questions to process

    Returns:
        List of (concept_name, source_tag) tuples
    """
    concepts = []

    for question in open_questions[:max_questions]:
        prompt = f"""Open question: {question}

What concept or domain might help answer this question that hasn't
been explored yet? Think of unexpected angles - concepts from science,
art, nature, or everyday life that might shed light on this question.

Respond with just ONE concept name."""

        result = await curator_model.generate(prompt, max_tokens=20)
        concept = result.text.strip()  # NOTE: Use result.text not result directly
        concepts.append((concept, f"question:{question[:30]}"))

    return concepts


async def gather_novelty_candidates(
    thought: str,
    recent_concepts: list[str],
    cycle_count: int,
    open_questions: list[str],
    current_pattern: str | None,
    context_embedding: np.ndarray | None,
    psyche: "PsycheClient",  # Reserved for external_seeding (not yet implemented)
    embedder: "TieredEmbeddingService",  # Reserved for external_seeding (not yet implemented)
    curator_model: "CuratorModel",
    config: NoveltyConfig = NOVELTY_CONFIG,
) -> list[tuple[str, float, str]]:
    """Gather concept candidates from all novelty sources.

    Args:
        thought: Current thought content
        recent_concepts: Recently explored concepts
        cycle_count: Current cycle number
        open_questions: List of open questions
        current_pattern: Pattern detected in current concept (for analogical bridging)
        context_embedding: Embedding of current context (for random walks)
        psyche: Graph client (reserved for external_seeding)
        embedder: Embedding service (reserved for external_seeding)
        curator_model: Curator model for LLM-based novelty
        config: Novelty source configuration

    Returns:
        List of (concept, score, source) tuples
    """
    candidates: list[tuple[str, float, str]] = []

    # === External seeding ===
    # TODO: Implement external seeding using psyche and embedder
    # Will extract novel concepts from inbox content and NotebookLM research
    if config.external_seeding.enabled:
        logger.debug("External seeding not yet implemented")

    # === Generative curiosity ===
    if config.generative_curiosity.enabled:
        if cycle_count % config.generative_curiosity.frequency == 0:
            try:
                concept, source = await generate_curious_concept(
                    thought, recent_concepts, curator_model
                )
                if concept:
                    candidates.append((concept, NOVELTY_BONUS, source))
                    logger.debug(f"Generative curiosity: {concept}")
            except Exception as e:
                logger.warning(f"Generative curiosity failed: {e}")

    # === Analogical bridging ===
    if config.analogical_bridging.enabled and current_pattern:
        try:
            # Use most recent concept for bridging
            bridge_concept = recent_concepts[-1] if recent_concepts else "awareness"
            analogies = await find_analogous_domains(
                bridge_concept, current_pattern, curator_model
            )
            for concept, source in analogies:
                if concept:
                    candidates.append((concept, NOVELTY_BONUS, source))
            logger.debug(f"Analogical bridging: {[c for c, _, _ in candidates[-3:]]}")
        except Exception as e:
            logger.warning(f"Analogical bridging failed: {e}")

    # === Random embedding walk ===
    if config.random_walk.enabled and context_embedding is not None:
        if cycle_count % config.random_walk.frequency == 0:
            try:
                concept, source = await random_embedding_walk(
                    context_embedding, curator_model
                )
                if concept:
                    candidates.append((concept, NOVELTY_BONUS, source))
                    logger.debug(f"Random walk: {concept}")
            except Exception as e:
                logger.warning(f"Random embedding walk failed: {e}")

    # === Driving questions ===
    if config.driving_questions.enabled and open_questions:
        try:
            question_concepts = await question_driven_concepts(
                open_questions, curator_model, max_questions=2
            )
            for concept, source in question_concepts:
                if concept:
                    # Driving questions get a slight boost
                    candidates.append((concept, NOVELTY_BONUS * 1.2, source))
            logger.debug(f"Driving questions: {[c for c, _, _ in candidates[-2:]]}")
        except Exception as e:
            logger.warning(f"Driving questions failed: {e}")

    return candidates


def filter_by_cooldowns(
    candidates: list[tuple[str, float, str]],
    current_cycle: int,
    last_used: dict[str, int],
) -> list[tuple[str, float, str]]:
    """Filter and score candidates based on cooldown status.

    Args:
        candidates: List of (concept, score, source) tuples
        current_cycle: Current cycle number
        last_used: Dict mapping concept names to cycle when last used

    Returns:
        Filtered list with scores adjusted for cooldowns
    """
    filtered: list[tuple[str, float, str]] = []

    for concept, score, source in candidates:
        available, availability = is_concept_available(concept, current_cycle, last_used)

        if not available:
            # Hard cooldown - skip entirely
            logger.debug(f"Skipping {concept}: hard cooldown")
            continue

        # Apply soft cooldown penalty to score
        adjusted_score = score * availability
        filtered.append((concept, adjusted_score, source))

    return filtered
