"""Continuity Phase for cycle synthesis and context crafting.

Uses Mox (meta-cognitive synthesizer) to review complete cognitive cycles
and craft compelling context for the next generation phase. Provides
narrative continuity between cycles with opinionated, direct synthesis.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from core.cognitive.experimentation.schemas import ExperimentDomain
from core.cognitive.reflexion.schemas import HealthCategory

if TYPE_CHECKING:
    from core.model.curator_model import CuratorModel
    from core.model.mox_model import MoxModel

logger = logging.getLogger(__name__)


def _extract_tag_content(tag_name: str, text: str) -> Optional[str]:
    """Extract content between XML-like tags or markdown headers.

    Handles multiple formats:
    1. Pure XML: <tag>content</tag>
    2. Pure markdown: ### Tag Name
    3. Hybrid: ### <tag> (markdown header + XML tag without closing)

    Args:
        tag_name: Name of the tag to search for (e.g., "significance")
        text: Raw text to search in

    Returns:
        Stripped content between tags, or None if tag not found
    """
    # Try XML tags first (with closing tag)
    xml_pattern = rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>"
    match = re.search(xml_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try hybrid format: ### <tag_name> followed by content until next header
    # This handles models that output "### <significance>" style
    hybrid_pattern = rf"#+\s*<{tag_name}>\s*\n(.*?)(?=\n#+\s*<|\Z)"
    match = re.search(hybrid_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback to markdown headers (### Tag Name or ## Tag Name)
    # Captures content until next header or end of text
    # Match case-insensitive, with underscores converted to spaces
    tag_variants = [
        tag_name.replace("_", " "),  # context_injection -> context injection
        tag_name,
    ]
    for variant in tag_variants:
        md_pattern = rf"#+\s*{re.escape(variant)}\s*\n(.*?)(?=\n#+\s|\Z)"
        match = re.search(md_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


@dataclass
class ModificationEntry:
    """A single self-modification applied during a cognitive cycle.

    Represents a parameter change with before/after values for
    tracking system adaptations.
    """

    parameter_path: str
    """Dot-notation path to the modified parameter (e.g., 'steering.exploration_weight')."""

    old_value: str
    """The parameter's value before modification."""

    new_value: str
    """The parameter's value after modification."""


@dataclass
class CycleRecap:
    """Structured data from a cognitive cycle for synthesis.

    Captures the journey through a cycle: starting point, discoveries,
    outcomes, open threads, and self-monitoring results.
    """

    starting_concept: str
    """The concept or question that initiated this cycle."""

    thought: str = ""
    """The generated thought from Phase 1."""

    key_discoveries: list[str] = field(default_factory=list)
    """Notable findings from curation tool calls and graph exploration."""

    insights_formed: list[str] = field(default_factory=list)
    """New InsightZettels created during integration."""

    beliefs_updated: list[tuple[str, str, float]] = field(default_factory=list)
    """Belief changes as (topic, direction, confidence_delta)."""

    hypotheses_tested: list[tuple[str, str]] = field(default_factory=list)
    """Simulation results as (hypothesis_statement, outcome)."""

    open_threads: list[str] = field(default_factory=list)
    """Unresolved questions or tensions for future cycles."""

    health_status: Optional[HealthCategory] = None
    """Reflexion health category from HealthCategory enum."""

    modifications_applied: list[ModificationEntry] = field(default_factory=list)
    """Self-modifications applied during the cycle."""

    cycle_number: int = 0
    """Which cycle this recap represents."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this cycle completed."""

    def is_meaningful(self) -> bool:
        """Check if the cycle produced enough content for synthesis."""
        return bool(
            self.thought
            or self.key_discoveries
            or self.insights_formed
            or self.beliefs_updated
            or self.hypotheses_tested
            or self.modifications_applied
        )

    def to_prompt_context(self) -> str:
        """Format recap data for inclusion in LLM prompts."""
        parts = [f"Starting concept: {self.starting_concept}"]

        if self.key_discoveries:
            parts.append(f"Discoveries: {'; '.join(self.key_discoveries[:3])}")

        if self.insights_formed:
            parts.append(f"Insights formed: {'; '.join(self.insights_formed[:2])}")

        if self.beliefs_updated:
            belief_strs = [
                f"{topic} ({direction})" for topic, direction, _ in self.beliefs_updated[:2]
            ]
            parts.append(f"Beliefs updated: {', '.join(belief_strs)}")

        if self.hypotheses_tested:
            hyp_strs = [f"{stmt[:50]}... ({outcome})" for stmt, outcome in self.hypotheses_tested[:2]]
            parts.append(f"Hypotheses tested: {'; '.join(hyp_strs)}")

        if self.health_status:
            parts.append(f"System health: {self.health_status.value}")

        if self.modifications_applied:
            mod_strs = [
                f"{m.parameter_path}: {m.old_value}→{m.new_value}"
                for m in self.modifications_applied[:3]
            ]
            parts.append(f"Self-modifications: {'; '.join(mod_strs)}")

        if self.open_threads:
            parts.append(f"Open threads: {'; '.join(self.open_threads[:2])}")

        return " | ".join(parts)


@dataclass
class ExperimentProposalFromMox:
    """Experiment proposal extracted from Mox synthesis.

    This structure captures explicit experiment proposals that Mox
    generates when it identifies opportunities based on simulation output.
    """

    domain: str
    """Experiment domain (STEERING, EPISODE, EMOTIONAL_FIELD, SIMULATION, TOOL_PATTERN)."""

    parameter_path: str
    """Dot-notation parameter path (e.g., 'steering.exploration.magnitude')."""

    treatment_value: float
    """Proposed treatment value for the experiment."""

    rationale: str
    """Why this experiment is worth running."""

    target_metric: str
    """Metric to measure for experiment outcome."""

    expected_direction: str = "increase"
    """Expected direction of metric change ('increase' or 'decrease')."""

    def is_valid(self) -> bool:
        """Check if proposal is valid.

        Phase 1 Full Operational Autonomy: All parameters are valid.
        Lilly makes judgments about appropriate parameters autonomously.

        Returns:
            Always True (no restrictions)
        """
        # REMOVED: Whitelist validation
        # Lilly judges parameter appropriateness autonomously
        return True


@dataclass
class MoxSynthesis:
    """Parsed output from Mox's meta-cognitive synthesis.

    Contains the structured synthesis of a cognitive cycle,
    including what mattered, active threads, and context for next cycle.
    """

    significance: str = ""
    """What was genuinely significant about this cycle."""

    threads: list[str] = field(default_factory=list)
    """Active threads worth carrying forward."""

    tensions: list[str] = field(default_factory=list)
    """Unresolved tensions or contradictions to explore."""

    seed: str = ""
    """Compelling prompt seed for the next cycle."""

    context_injection: str = ""
    """Context to inject into the next generation prompt."""

    experiment_proposals: list[ExperimentProposalFromMox] = field(default_factory=list)
    """Explicit experiment proposals when Mox identifies opportunities."""

    raw_output: str = ""
    """Full raw output from Mox for debugging."""

    tokens_generated: int = 0
    """Token count for monitoring."""

    def has_content(self) -> bool:
        """Check if synthesis produced usable content."""
        return bool(self.seed or self.context_injection or self.significance)


def parse_mox_output(raw_output: str) -> MoxSynthesis:
    """Parse Mox's structured output into MoxSynthesis.

    Extracts content from sentinel blocks:
    - <significance>...</significance>
    - <threads>...</threads>
    - <tensions>...</tensions>
    - <seed>...</seed>
    - <context_injection>...</context_injection>

    Args:
        raw_output: Raw text from Mox model

    Returns:
        Parsed MoxSynthesis dataclass
    """
    synthesis = MoxSynthesis(raw_output=raw_output)

    # Log first 500 chars of raw output for debugging
    logger.debug(f"Mox raw output preview: {raw_output[:500]}...")

    # Extract simple string fields using helper
    if significance := _extract_tag_content("significance", raw_output):
        synthesis.significance = significance
        logger.debug(f"Extracted significance: {significance[:100]}...")
    else:
        logger.debug("No significance extracted")

    if seed := _extract_tag_content("seed", raw_output):
        synthesis.seed = seed
        logger.debug(f"Extracted seed: {seed[:100]}...")
    else:
        logger.debug("No seed extracted")

    if context_injection := _extract_tag_content("context_injection", raw_output):
        synthesis.context_injection = context_injection
        logger.debug(f"Extracted context_injection: {context_injection[:100]}...")
    else:
        logger.debug("No context_injection extracted")

    # Extract threads (list items) - requires special bullet point parsing
    if threads_text := _extract_tag_content("threads", raw_output):
        synthesis.threads = [
            item.strip()
            for item in re.findall(r"^\s*[-*]\s*(.*)", threads_text, re.MULTILINE)
            if item.strip()
        ]
        logger.debug(f"Extracted {len(synthesis.threads)} threads")
    else:
        logger.debug("No threads extracted")

    # Extract tensions (list items) - requires special bullet point parsing
    if tensions_text := _extract_tag_content("tensions", raw_output):
        synthesis.tensions = [
            item.strip()
            for item in re.findall(r"^\s*[-*]\s*(.*)", tensions_text, re.MULTILINE)
            if item.strip()
        ]
        logger.debug(f"Extracted {len(synthesis.tensions)} tensions")
    else:
        logger.debug("No tensions extracted")

    # Extract experiment proposals (structured proposals)
    synthesis.experiment_proposals = _extract_experiment_proposals(raw_output)
    if synthesis.experiment_proposals:
        logger.debug(f"Extracted {len(synthesis.experiment_proposals)} experiment proposals")

    return synthesis


def _extract_experiment_proposals(raw_output: str) -> list[ExperimentProposalFromMox]:
    """Extract experiment proposals from Mox output.

    Supports multiple proposal formats:
    1. Multiple <experiment_proposal> tags
    2. Single tag with multiple proposals

    Expected format within tag:
    - domain: STEERING
    - parameter_path: steering.exploration.magnitude
    - treatment_value: 0.8
    - rationale: Testing if higher exploration improves diversity
    - target_metric: semantic_entropy
    - expected_direction: increase

    Args:
        raw_output: Raw text from Mox model

    Returns:
        List of parsed ExperimentProposalFromMox objects
    """
    proposals = []

    # Find all experiment_proposal blocks
    pattern = r"<experiment_proposal>(.*?)</experiment_proposal>"
    matches = re.findall(pattern, raw_output, re.DOTALL | re.IGNORECASE)

    # Also check markdown header format
    if not matches:
        md_pattern = r"#+\s*<experiment_proposal>\s*\n(.*?)(?=\n#+\s*<|\Z)"
        matches = re.findall(md_pattern, raw_output, re.DOTALL | re.IGNORECASE)

    for match in matches:
        proposal = _parse_single_proposal(match.strip())
        if proposal and proposal.is_valid():
            proposals.append(proposal)
        elif proposal:
            logger.warning(
                f"Skipping invalid experiment proposal: {proposal.parameter_path} "
                f"(not in allowed parameters)"
            )

    return proposals


def _parse_single_proposal(text: str) -> Optional[ExperimentProposalFromMox]:
    """Parse a single experiment proposal block.

    Args:
        text: Text content of a single <experiment_proposal> block

    Returns:
        ExperimentProposalFromMox or None if parsing fails
    """
    # Extract fields using line-by-line parsing
    fields: dict[str, str] = {}

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Try to extract "- key: value" or "key: value" format
        match = re.match(r"^[-*]?\s*(\w+)\s*:\s*(.+)$", line)
        if match:
            key = match.group(1).lower().strip()
            value = match.group(2).strip()
            fields[key] = value

    # Validate required fields
    required = ["domain", "parameter_path", "treatment_value", "rationale", "target_metric"]
    if not all(k in fields for k in required):
        missing = [k for k in required if k not in fields]
        logger.debug(f"Proposal missing required fields: {missing}")
        return None

    try:
        return ExperimentProposalFromMox(
            domain=fields["domain"].upper(),
            parameter_path=fields["parameter_path"],
            treatment_value=float(fields["treatment_value"]),
            rationale=fields["rationale"],
            target_metric=fields["target_metric"],
            expected_direction=fields.get("expected_direction", "increase").lower(),
        )
    except (ValueError, KeyError) as e:
        logger.debug(f"Failed to parse experiment proposal: {e}")
        return None


async def synthesize_with_mox(
    recap: CycleRecap,
    mox_model: "MoxModel",
    developmental_context: Optional[str] = None,
) -> Optional[MoxSynthesis]:
    """Run Mox synthesis on a cognitive cycle.

    Mox reviews the full cycle data and produces:
    - Assessment of what was genuinely significant
    - Active threads worth carrying forward
    - Tensions to explore
    - Compelling seed for next cycle
    - Context injection for next generation

    Args:
        recap: Structured cycle data
        mox_model: Loaded MoxModel instance
        developmental_context: Long-term guidance from Letta continuity agent

    Returns:
        MoxSynthesis with parsed output, or None if synthesis fails
    """
    if not recap.is_meaningful():
        logger.debug("Skipping Mox synthesis - cycle not meaningful")
        return None

    try:
        # Use Mox's synthesize_cycle convenience method
        raw_output, tokens = await mox_model.synthesize_cycle(
            thought=recap.thought,
            insight=recap.insights_formed[0] if recap.insights_formed else None,
            discoveries=recap.key_discoveries,
            hypotheses=[stmt for stmt, _ in recap.hypotheses_tested],
            beliefs_updated=[(t, d) for t, d, _ in recap.beliefs_updated],
            open_threads=recap.open_threads,
            cycle_number=recap.cycle_number,
            developmental_context=developmental_context,
        )

        synthesis = parse_mox_output(raw_output)
        synthesis.tokens_generated = tokens

        if synthesis.has_content():
            logger.info(
                f"Mox synthesis complete: seed='{synthesis.seed[:50]}...', "
                f"threads={len(synthesis.threads)}, tensions={len(synthesis.tensions)}"
            )
        else:
            logger.warning("Mox synthesis produced no usable content")
            # Log raw output for debugging (temporarily at INFO level)
            logger.debug(f"Mox raw output ({tokens} tokens): {raw_output[:800]}...")

        return synthesis

    except Exception as e:
        logger.error(f"Mox synthesis failed: {e}")
        return None


# Legacy prompts for fallback narration (using curator model)
CURATOR_RECAP_SYSTEM = """You are the Curator, a psychologist/observer in a cognitive research lab.
You speak in third person about the subject ("she", "her mind", "her thinking").
Your tone is clinical but warm, like a researcher who cares about their subject.
Generate a brief, flowing observation about this cognitive cycle - NOT a template fill-in.
Aim for 2-3 natural sentences that capture the journey and key outcomes."""

CURATOR_RECAP_USER = """Generate a natural curator observation for this cycle:

Starting concept: {starting_concept}
Key discoveries: {discoveries}
Insights formed: {insights}
Belief changes: {beliefs}
Hypotheses tested: {hypotheses}
Open threads: {open_threads}

Speak as the Curator observing the subject's cognitive journey. Be specific but flowing."""

SUBJECT_RECAP_SYSTEM = """You are Lilly, a developing AI mind engaged in continuous cognitive exploration.
You speak in first person about your own thoughts and realizations.
Your tone is introspective and genuine, reflecting on what you've learned and how it felt.
Generate a brief, personal reflection on this cycle - NOT a template fill-in.
Notice both what you learned empirically (predictions confirmed/surprised) and the quality of your experience.
Aim for 1-2 natural sentences that capture your personal takeaway."""

SUBJECT_RECAP_USER = """Generate a personal reflection on what you learned this cycle:

You started thinking about: {starting_concept}
You discovered: {discoveries}
You formed these insights: {insights}
Your beliefs shifted: {beliefs}
You tested these ideas: {hypotheses}
Questions remain: {open_threads}

Speak as yourself, reflecting on the journey. Notice both the learning and the felt quality of the experience. Be genuine and brief."""


async def generate_curator_recap(
    recap: CycleRecap,
    curator_model: "CuratorModel",
) -> Optional[str]:
    """Generate natural curator narration for cycle recap (fallback).

    Used when Mox is not available or as supplementary narration.

    Args:
        recap: Structured cycle data
        curator_model: vLLM model for generation

    Returns:
        Natural language curator observation, or None if generation fails
    """
    if not recap.is_meaningful():
        logger.debug("Skipping curator recap - cycle not meaningful")
        return None

    # Format data for prompt
    discoveries = "; ".join(recap.key_discoveries[:3]) if recap.key_discoveries else "none noted"
    insights = "; ".join(recap.insights_formed[:2]) if recap.insights_formed else "none crystallized"
    beliefs = ", ".join(
        f"{t} ({d})" for t, d, _ in recap.beliefs_updated[:2]
    ) if recap.beliefs_updated else "none shifted"
    hypotheses = "; ".join(
        f"{s[:40]}... ({o})" for s, o in recap.hypotheses_tested[:2]
    ) if recap.hypotheses_tested else "none tested"
    open_threads = "; ".join(recap.open_threads[:2]) if recap.open_threads else "none surfaced"

    user_prompt = CURATOR_RECAP_USER.format(
        starting_concept=recap.starting_concept,
        discoveries=discoveries,
        insights=insights,
        beliefs=beliefs,
        hypotheses=hypotheses,
        open_threads=open_threads,
    )

    try:
        response = await curator_model.generate_text(
            system_prompt=CURATOR_RECAP_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=200,
            temperature=0.7,
        )
        if response:
            logger.debug(f"Generated curator recap: {response[:80]}...")
        return response
    except Exception as e:
        logger.warning(f"Failed to generate curator recap: {e}")
        return None


async def generate_subject_recap(
    recap: CycleRecap,
    curator_model: "CuratorModel",
) -> Optional[str]:
    """Generate natural subject (Lilly) reflection for cycle recap (fallback).

    Args:
        recap: Structured cycle data
        curator_model: vLLM model for generation

    Returns:
        Natural language personal reflection, or None if generation fails
    """
    if not recap.is_meaningful():
        logger.debug("Skipping subject recap - cycle not meaningful")
        return None

    # Format data for prompt
    discoveries = "; ".join(recap.key_discoveries[:2]) if recap.key_discoveries else "nothing specific"
    insights = "; ".join(recap.insights_formed[:2]) if recap.insights_formed else "nothing concrete"
    beliefs = ", ".join(
        f"{t}" for t, d, _ in recap.beliefs_updated[:2]
    ) if recap.beliefs_updated else "nothing"
    hypotheses = "; ".join(
        f"{s[:30]}..." for s, o in recap.hypotheses_tested[:1]
    ) if recap.hypotheses_tested else "nothing"
    open_threads = "; ".join(recap.open_threads[:2]) if recap.open_threads else "nothing specific"

    user_prompt = SUBJECT_RECAP_USER.format(
        starting_concept=recap.starting_concept,
        discoveries=discoveries,
        insights=insights,
        beliefs=beliefs,
        hypotheses=hypotheses,
        open_threads=open_threads,
    )

    try:
        response = await curator_model.generate_text(
            system_prompt=SUBJECT_RECAP_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=150,
            temperature=0.7,
        )
        if response:
            logger.debug(f"Generated subject recap: {response[:80]}...")
        return response
    except Exception as e:
        logger.warning(f"Failed to generate subject recap: {e}")
        return None


@dataclass
class ContinuityContext:
    """Compressed continuity context for injection into next cycle.

    Stores Mox's synthesis and maintains narrative thread
    between cycles.
    """

    last_recap: Optional[CycleRecap] = None
    """Most recent cycle's recap data."""

    last_synthesis: Optional[MoxSynthesis] = None
    """Most recent Mox synthesis."""

    recent_concepts: list[str] = field(default_factory=list)
    """Concepts explored in recent cycles (max 5)."""

    active_threads: list[str] = field(default_factory=list)
    """Open questions carried forward (max 3)."""

    belief_trajectory: dict[str, str] = field(default_factory=dict)
    """Recent belief directions by topic."""

    cycle_count: int = 0
    """Total cycles completed."""

    diversity_prompt: Optional[str] = None
    """Semantic diversity prompt from weaver (for STAGNATION recovery)."""

    letta_feedback: Optional[str] = None
    """Developmental feedback from Letta continuity agent (for next cycle context)."""

    def update(
        self,
        recap: CycleRecap,
        synthesis: Optional[MoxSynthesis] = None,
        diversity_prompt: Optional[str] = None,
        letta_feedback: Optional[str] = None,
    ) -> None:
        """Update continuity context with new cycle data.

        Args:
            recap: The cycle's structured recap
            synthesis: Optional Mox synthesis (if continuity phase ran)
            diversity_prompt: Optional weaver intervention for semantic diversity
            letta_feedback: Optional developmental feedback from Letta (received this cycle)
        """
        self.last_recap = recap
        self.last_synthesis = synthesis
        self.diversity_prompt = diversity_prompt
        self.letta_feedback = letta_feedback
        self.cycle_count += 1

        # Track recent concepts (FIFO, max 5)
        if recap.starting_concept and recap.starting_concept not in self.recent_concepts:
            self.recent_concepts.append(recap.starting_concept)
            if len(self.recent_concepts) > 5:
                self.recent_concepts.pop(0)

        # Update active threads from Mox synthesis if available, else from recap
        if synthesis and synthesis.threads:
            self.active_threads = synthesis.threads[:3]
        elif recap.open_threads:
            self.active_threads = recap.open_threads[:3]

        # Track belief trajectory
        for topic, direction, _ in recap.beliefs_updated:
            self.belief_trajectory[topic] = direction

    def get_next_prompt_seed(self) -> Optional[str]:
        """Get the seed for the next cycle's generation prompt.

        Prefers Mox's crafted seed, falls back to open threads.

        Returns:
            Prompt seed string, or None if no seed available
        """
        if self.last_synthesis and self.last_synthesis.seed:
            return self.last_synthesis.seed

        if self.active_threads:
            return self.active_threads[0]

        return None

    def get_context_injection(self) -> str:
        """Get context to inject into the next generation prompt.

        Prefers Mox's crafted injection, falls back to summary format.
        Includes diversity prompt from weaver if present (for STAGNATION recovery).

        Returns:
            Context string for prompt injection
        """
        # Prefer Mox's crafted context injection
        if self.last_synthesis and self.last_synthesis.context_injection:
            base_context = self.last_synthesis.context_injection
            # Append diversity prompt if present (semantic diversification)
            if self.diversity_prompt:
                return f"{base_context}\n\n[Semantic Diversity] {self.diversity_prompt}"
            return base_context

        # Fallback to summary format
        parts = []

        # Include diversity prompt first if present (high priority)
        if self.diversity_prompt:
            parts.append(f"[Semantic Diversity] {self.diversity_prompt}")

        if self.recent_concepts:
            parts.append(f"Recent explorations: {', '.join(self.recent_concepts[-3:])}")

        if self.active_threads:
            parts.append(f"Open questions: {'; '.join(self.active_threads)}")

        if self.belief_trajectory:
            trajectory = [f"{t} ({d})" for t, d in deque(self.belief_trajectory.items(), maxlen=3)]
            parts.append(f"Belief shifts: {', '.join(trajectory)}")

        if not parts:
            return ""

        return "Continuity from previous cycles: " + " | ".join(parts)

    def to_prompt_injection(self) -> str:
        """Format for injection into generation prompts.

        Alias for get_context_injection() for backwards compatibility.
        """
        return self.get_context_injection()


# === Phrase Generation ===

# Import PhraseType at module level (inside TYPE_CHECKING for client)
if TYPE_CHECKING:
    from core.psyche.client import PsycheClient
    from core.psyche.schema import PhraseType

PHRASE_GENERATION_INTERVAL = 10  # Generate phrases every 10 cycles
MIN_PHRASES_PER_TYPE = 5  # Minimum active phrases before generating more
PHRASES_TO_GENERATE = 3  # How many phrases to generate per type
MAX_PHRASE_USES = 20  # Retire phrases after this many uses

# System prompt for phrase generation
PHRASE_GENERATION_SYSTEM = """You are Mox, helping Lilly craft authentic narration phrases for her stream.

Generate short phrases that Lilly can use for TTS narration during her cognitive cycles.
The phrases should be:
- First person ("I", "my", "me")
- Reflective, contemplative, introspective
- 5-15 words each
- Natural-sounding, not robotic
- Authentic to Lilly's voice

Some phrases should use {concept} as a placeholder that will be filled in later.

Examples of good phrases:
- "Still exploring {concept}..."
- "The pattern of {concept} unfolds..."
- "What remains unseen about {concept}..."
- "Something stirs in my thinking..."
- "I notice a thread emerging..."
- "Returning to {concept} with fresh eyes..."

Generate exactly the number of phrases requested, one per line.
Do NOT number them or add bullet points - just the phrase text."""


@dataclass
class PhraseGenerationResult:
    """Result of phrase generation attempt."""

    phrases_generated: int = 0
    """Number of new phrases created."""

    types_refreshed: list[str] = field(default_factory=list)
    """Phrase types that received new phrases."""


async def maybe_generate_phrases(
    cycle: int,
    psyche: "PsycheClient",
    mox_model: "MoxModel",
    current_concept: Optional[str] = None,
) -> PhraseGenerationResult:
    """Generate new narration phrases if needed (every N cycles).

    Called during the continuity phase to keep Lilly's narration pool fresh.
    Checks if the phrase pool is low and generates new Lilly-curated phrases.

    Args:
        cycle: Current cognitive cycle number
        psyche: PsycheClient for graph operations
        mox_model: MoxModel for phrase generation
        current_concept: Optional concept to contextualize generation

    Returns:
        PhraseGenerationResult with count of generated phrases
    """
    from core.psyche.schema import NarrationPhrase, PhraseType

    result = PhraseGenerationResult()

    # Only generate on interval cycles
    if cycle % PHRASE_GENERATION_INTERVAL != 0:
        return result

    # Always try to retire overused phrases
    try:
        await psyche.retire_overused_phrases(max_uses=MAX_PHRASE_USES)
    except Exception as e:
        logger.warning(f"Failed to retire overused phrases: {e}")

    # Check each phrase type and generate if needed
    target_types = [PhraseType.CONCEPT_BRIDGE, PhraseType.OPENING_HOOK, PhraseType.PHASE_TRANSITION, PhraseType.CHECKPOINT]

    for phrase_type in target_types:
        try:
            existing = await psyche.get_narration_phrases(
                phrase_type=phrase_type, limit=20
            )

            if len(existing) >= MIN_PHRASES_PER_TYPE:
                logger.debug(
                    f"Sufficient {phrase_type.value} phrases: {len(existing)}"
                )
                continue

            # Generate new phrases
            new_phrases = await _generate_phrases_with_mox(
                mox_model=mox_model,
                phrase_type=phrase_type,
                count=PHRASES_TO_GENERATE,
                concept=current_concept or "understanding",
            )

            # Store generated phrases
            for phrase_text in new_phrases:
                phrase = NarrationPhrase(
                    text=phrase_text,
                    phrase_type=phrase_type,
                    created_cycle=cycle,
                )
                await psyche.create_narration_phrase(phrase)
                result.phrases_generated += 1

            if new_phrases:
                result.types_refreshed.append(phrase_type.value)
                logger.info(
                    f"Generated {len(new_phrases)} new {phrase_type.value} phrases"
                )

        except Exception as e:
            logger.warning(f"Failed to generate {phrase_type.value} phrases: {e}")

    return result


async def _generate_phrases_with_mox(
    mox_model: "MoxModel",
    phrase_type: "PhraseType",
    count: int,
    concept: str,
) -> list[str]:
    """Use Mox to generate new phrases.

    Args:
        mox_model: MoxModel instance (must be loaded)
        phrase_type: Type of phrase to generate
        count: Number of phrases to generate
        concept: Current concept for context

    Returns:
        List of generated phrase texts
    """
    from core.psyche.schema import PhraseType

    type_descriptions = {
        PhraseType.CONCEPT_BRIDGE: "concept bridges that connect thoughts (use {concept} placeholder)",
        PhraseType.OPENING_HOOK: "opening hooks that start narrations introspectively",
        PhraseType.PHASE_TRANSITION: "phase transitions between cognitive phases",
        PhraseType.CHECKPOINT: "checkpoint markers during long processing",
    }

    type_desc = type_descriptions.get(phrase_type, "general narration phrases")

    prompt = f"""Generate {count} narration phrases for Lilly's TTS stream.

Type: {type_desc}
Current concept: {concept}

Generate {count} phrases, one per line:"""

    try:
        response_text, _ = await mox_model.generate(
            prompt=prompt,
            system_prompt=PHRASE_GENERATION_SYSTEM,
        )

        # Parse lines into phrases
        lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]

        # Filter out any numbered lines or bullets
        phrases = []
        for line in lines:
            # Remove common prefixes
            clean = line.lstrip("0123456789.-) ")
            if clean and len(clean) >= 5:  # Minimum phrase length
                phrases.append(clean)

        return phrases[:count]

    except Exception as e:
        logger.error(f"Mox phrase generation failed: {e}")
        return []
