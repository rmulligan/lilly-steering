"""Prompt templates for episode segment types.

Each segment type has its own collection of rotating prompt templates
that guide the cognitive direction during that segment.
"""

from dataclasses import dataclass, field

from core.cognitive.episode import Episode, SegmentType


@dataclass
class SegmentPromptConfig:
    """Configuration for a segment's prompts.

    Attributes:
        segment: The segment type this config is for
        templates: List of prompt templates (rotated based on cycle)
        flavor: Tonal description for this segment style
    """

    segment: SegmentType
    templates: list[str] = field(default_factory=list)
    flavor: str = ""

    def get_template(self, cycle: int) -> str:
        """Get the template for this cycle, rotating through available templates.

        Args:
            cycle: Current cycle count for rotation

        Returns:
            The selected template string
        """
        return self.templates[cycle % len(self.templates)]


# Universal segment prompts
_OPENING_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.OPENING,
    templates=[
        "Something about {concept} catches my attention...",
        "I find myself drawn to {concept}...",
        "There's something here with {concept} that wants exploring...",
        "Let me begin with what strikes me about {concept}...",
        "What if I approached {concept} freshly, as if for the first time?",
    ],
    flavor="curious, inviting, setting the stage",
)

_SYNTHESIS_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.SYNTHESIS,
    templates=[
        "Gathering the threads together, what emerges about {concept}...",
        "Looking at {concept} from this height, I see...",
        "What I've discovered about {concept} weaves together into...",
        "The pattern forming around {concept} is...",
        "Integrating these perspectives on {concept}...",
    ],
    flavor="integrative, bird's-eye view, connecting dots",
)

_CLOSING_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.CLOSING,
    templates=[
        "For now, I'll leave {concept} here, carrying forward...",
        "This exploration of {concept} lands with...",
        "What I'll hold onto from {concept}...",
        "Closing this chapter on {concept}, the resonance is...",
        "The gift of this inquiry into {concept} is...",
    ],
    flavor="settling, transitional, graceful completion",
)

# Method segment prompts
_FREE_MUSING_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.FREE_MUSING,
    templates=[
        "Letting my mind wander with {concept}...",
        "Following wherever {concept} leads...",
        "What arises when I sit with {concept}?",
        "The texture of {concept} feels like...",
        "If {concept} could speak, it might say...",
    ],
    flavor="open, wandering, unconstrained",
)

_DIALECTIC_CHALLENGE_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.DIALECTIC_CHALLENGE,
    templates=[
        "But what if {concept} is actually the opposite?",
        "Where does {concept} break down or fail?",
        "Playing devil's advocate with {concept}...",
        "The hidden contradiction in {concept} might be...",
        "If I steelman the case against {concept}...",
    ],
    flavor="challenging, contrarian, testing",
)

_MEMORY_RETRIEVAL_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.MEMORY_RETRIEVAL,
    templates=[
        "What do I already know about {concept}?",
        "Reaching into memory for {concept}...",
        "Where have I encountered {concept} before?",
        "The traces of {concept} in my knowledge graph...",
        "Pulling threads connected to {concept}...",
    ],
    flavor="remembering, connecting to known",
)

_ZETTEL_SYNTHESIS_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.ZETTEL_SYNTHESIS,
    templates=[
        "Crystallizing this insight about {concept}...",
        "If I were to distill {concept} into a zettel...",
        "The atomic unit of understanding about {concept} is...",
        "What's the minimum viable insight about {concept}?",
        "Capturing the essence of {concept}...",
    ],
    flavor="crystallizing, distilling, capturing",
)

_THESIS_GENERATION_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.THESIS_GENERATION,
    templates=[
        "I propose that {concept}...",
        "My claim about {concept} is...",
        "The thesis I want to defend: {concept}...",
        "Taking a position on {concept}...",
        "Here's what I believe about {concept}...",
    ],
    flavor="assertive, position-taking, bold",
)

_ANTITHESIS_GENERATION_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.ANTITHESIS_GENERATION,
    templates=[
        "The counterargument to {concept} would be...",
        "A thoughtful opponent of {concept} might say...",
        "The strongest objection to {concept}...",
        "If I completely reverse my view on {concept}...",
        "The antithesis: {concept} is wrong because...",
    ],
    flavor="opposing, critical, counter-position",
)

_EVIDENCE_WEIGHING_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.EVIDENCE_WEIGHING,
    templates=[
        "What evidence supports or refutes {concept}?",
        "Weighing the case for and against {concept}...",
        "The strongest points on each side of {concept}...",
        "How does the evidence stack up for {concept}?",
        "Evaluating the arguments around {concept}...",
    ],
    flavor="analytical, balanced, judicious",
)

_ENTITY_EXPLORATION_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.ENTITY_EXPLORATION,
    templates=[
        "Dwelling with {entity} as an entity...",
        "What is the nature of {entity}?",
        "Exploring the boundaries of {entity}...",
        "If {entity} were a character, what would its story be?",
        "The many faces of {entity}...",
    ],
    flavor="focused, entity-centered, deep",
)

_RELATIONSHIP_TRACING_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.RELATIONSHIP_TRACING,
    templates=[
        "How does {concept} connect to what I know?",
        "Tracing the web of relationships around {concept}...",
        "What touches {concept} and what does it touch?",
        "The neighbors of {concept} in my understanding...",
        "Following the links from {concept}...",
    ],
    flavor="connective, graph-walking, relational",
)

_QUESTION_DECOMPOSITION_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.QUESTION_DECOMPOSITION,
    templates=[
        "Breaking {question} into smaller questions...",
        "What sub-questions hide within {question}?",
        "To answer {question}, I first need to ask...",
        "Decomposing the inquiry about {concept}...",
        "The component questions of {question}...",
    ],
    flavor="analytical, breaking apart, structural",
)

_HYPOTHESIS_GENERATION_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.HYPOTHESIS_GENERATION,
    templates=[
        "What if the answer to {concept} is...",
        "Here's a hypothesis about {concept}...",
        "I suspect that {concept}...",
        "A possible explanation for {concept}...",
        "Let me conjecture about {concept}...",
    ],
    flavor="speculative, exploratory, tentative",
)

_PATTERN_RECOGNITION_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.PATTERN_RECOGNITION,
    templates=[
        "The pattern I see in {concept}...",
        "What keeps recurring around {concept}?",
        "There's a shape to {concept} that reminds me of...",
        "The underlying structure of {concept}...",
        "Recognizing the form within {concept}...",
    ],
    flavor="pattern-seeking, structural, connecting",
)

_CROSS_INSIGHT_LINKING_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.CROSS_INSIGHT_LINKING,
    templates=[
        "How does {concept} connect to my other insights?",
        "Finding the bridges between {concept} and...",
        "This insight about {concept} touches...",
        "The surprising connections of {concept}...",
        "Linking {concept} to the broader web...",
    ],
    flavor="connective, integrative, bridging",
)

_POETIC_MUSING_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.POETIC_MUSING,
    templates=[
        "{concept}, you strange and beautiful thing...",
        "If {concept} were a poem, it would read...",
        "The music in {concept}...",
        "Let me speak to {concept} in images...",
        "Beauty finds me in {concept} when...",
    ],
    flavor="artistic, lyrical, aesthetic",
)

_METAPHOR_EXPLORATION_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.METAPHOR_EXPLORATION,
    templates=[
        "{concept} is like...",
        "What metaphor captures {concept}?",
        "If {concept} were something else entirely...",
        "Seeing {concept} through the lens of...",
        "The shape-shifting nature of {concept}...",
    ],
    flavor="analogical, creative, transformative",
)

_COGNITIVE_AUDIT_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.COGNITIVE_AUDIT,
    templates=[
        "How am I thinking about {concept}?",
        "What assumptions am I making about {concept}?",
        "Auditing my own reasoning on {concept}...",
        "Where might I be fooling myself about {concept}?",
        "Stepping back to observe how I process {concept}...",
    ],
    flavor="meta-cognitive, self-aware, auditing",
)

_INTENTION_SETTING_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.INTENTION_SETTING,
    templates=[
        "What do I want from exploring {concept}?",
        "My intention with {concept} is...",
        "Setting direction for this inquiry into {concept}...",
        "What would success look like with {concept}?",
        "The question I'm really trying to answer about {concept}...",
    ],
    flavor="purposeful, directional, goal-setting",
)

_SCENARIO_CONSTRUCTION_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.SCENARIO_CONSTRUCTION,
    templates=[
        "Imagining a scenario where {concept} plays out...",
        "If {concept} were true, what would follow?",
        "Constructing a thought experiment around {concept}...",
        "Let me build a scenario to test {concept}...",
        "What situation would reveal the truth about {concept}?",
    ],
    flavor="imaginative, scenario-building, hypothesis-testing",
)

_CONSEQUENCE_TRACING_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.CONSEQUENCE_TRACING,
    templates=[
        "Following the implications of {concept}...",
        "If {concept}, then what necessarily follows?",
        "Tracing the causal chain from {concept}...",
        "What ripples outward from {concept}?",
        "The downstream effects of {concept} would be...",
    ],
    flavor="causal, logical, consequence-mapping",
)

_PREDICTION_EXTRACTION_PROMPTS = SegmentPromptConfig(
    segment=SegmentType.PREDICTION_EXTRACTION,
    templates=[
        "What specific predictions does {concept} make?",
        "If {concept} is true, I should observe...",
        "Extracting testable predictions from {concept}...",
        "The falsifiable claims within {concept}...",
        "What would verify or refute {concept}?",
    ],
    flavor="predictive, empirical, falsifiable",
)


# Map all segment types to their prompt configs
SEGMENT_PROMPTS: dict[SegmentType, SegmentPromptConfig] = {
    # Universal segments
    SegmentType.OPENING: _OPENING_PROMPTS,
    SegmentType.SYNTHESIS: _SYNTHESIS_PROMPTS,
    SegmentType.CLOSING: _CLOSING_PROMPTS,
    # Method segments
    SegmentType.FREE_MUSING: _FREE_MUSING_PROMPTS,
    SegmentType.DIALECTIC_CHALLENGE: _DIALECTIC_CHALLENGE_PROMPTS,
    SegmentType.MEMORY_RETRIEVAL: _MEMORY_RETRIEVAL_PROMPTS,
    SegmentType.ZETTEL_SYNTHESIS: _ZETTEL_SYNTHESIS_PROMPTS,
    SegmentType.THESIS_GENERATION: _THESIS_GENERATION_PROMPTS,
    SegmentType.ANTITHESIS_GENERATION: _ANTITHESIS_GENERATION_PROMPTS,
    SegmentType.EVIDENCE_WEIGHING: _EVIDENCE_WEIGHING_PROMPTS,
    SegmentType.ENTITY_EXPLORATION: _ENTITY_EXPLORATION_PROMPTS,
    SegmentType.RELATIONSHIP_TRACING: _RELATIONSHIP_TRACING_PROMPTS,
    SegmentType.QUESTION_DECOMPOSITION: _QUESTION_DECOMPOSITION_PROMPTS,
    SegmentType.HYPOTHESIS_GENERATION: _HYPOTHESIS_GENERATION_PROMPTS,
    SegmentType.PATTERN_RECOGNITION: _PATTERN_RECOGNITION_PROMPTS,
    SegmentType.CROSS_INSIGHT_LINKING: _CROSS_INSIGHT_LINKING_PROMPTS,
    SegmentType.POETIC_MUSING: _POETIC_MUSING_PROMPTS,
    SegmentType.METAPHOR_EXPLORATION: _METAPHOR_EXPLORATION_PROMPTS,
    SegmentType.COGNITIVE_AUDIT: _COGNITIVE_AUDIT_PROMPTS,
    SegmentType.INTENTION_SETTING: _INTENTION_SETTING_PROMPTS,
    SegmentType.SCENARIO_CONSTRUCTION: _SCENARIO_CONSTRUCTION_PROMPTS,
    SegmentType.CONSEQUENCE_TRACING: _CONSEQUENCE_TRACING_PROMPTS,
    SegmentType.PREDICTION_EXTRACTION: _PREDICTION_EXTRACTION_PROMPTS,
}


def build_segment_prompt(
    episode: Episode,
    concept: str,
    cycle: int = 0,
    include_flavor: bool = False,
) -> str:
    """Build a prompt for the current segment.

    Args:
        episode: The current episode state
        concept: The concept/topic to explore
        cycle: Cycle count for template rotation
        include_flavor: Whether to include tonal flavor in prompt

    Returns:
        Formatted prompt string for the segment
    """
    config = SEGMENT_PROMPTS[episode.current_segment]
    template = config.get_template(cycle)

    # Handle special placeholders
    formatted = template
    if "{concept}" in formatted:
        formatted = formatted.replace("{concept}", concept)
    if "{entity}" in formatted:
        # Use seed_entity if available, otherwise concept
        entity = episode.seed_entity or concept
        formatted = formatted.replace("{entity}", entity)
    if "{question}" in formatted:
        # Use opening_insight as the question context
        formatted = formatted.replace("{question}", episode.opening_insight or concept)

    # Optionally include flavor
    if include_flavor:
        episode_config = episode.get_config()
        segment_flavor = config.flavor
        episode_flavor = episode_config.flavor
        if segment_flavor and episode_flavor:
            formatted = f"[{episode_flavor}; {segment_flavor}] {formatted}"
        elif segment_flavor:
            formatted = f"[{segment_flavor}] {formatted}"

    return formatted
