"""Simulation Engine for Graph-Preflexor phase.

Orchestrates simulations using Graph-Preflexor-8b, extracting hypotheses
and predictions while providing continuous narration of the reasoning process.
"""

from __future__ import annotations

import hashlib
import logging
import re
import textwrap
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple

from core.cognitive.simulation.output_parser import (
    ParsedPreflexorOutput,
    PreflexorOutputParser,
)
from core.cognitive.simulation.schemas import (
    Hypothesis,
    HypothesisStatus,
    MetricsSnapshot,
    Prediction,
    PredictionConditionType,
    PredictionStatus,
    SimulationResult,
)
from core.psyche.schema import PhraseType

logger = logging.getLogger(__name__)

# Default concept when no specific focus is available
DEFAULT_FOCUS_CONCEPT = "emergence"
# Maximum length for statement truncation in log messages
LOG_STATEMENT_TRUNCATION_LENGTH = 50
# Default delta for metric threshold detection when no specific value is given
DEFAULT_METRIC_DELTA = 0.1

# Fallback checkpoint phrases when graph phrases are unavailable
DEFAULT_CHECKPOINTS = {
    "brainstorm_complete": "Brainstorming complete. Building conceptual graph...",
    "graph_built": "Graph constructed with {node_count} nodes. Detecting patterns...",
    "patterns_found": "{pattern_count} patterns identified. Synthesizing results...",
    "synthesis_complete": "Synthesis complete. Forming hypothesis...",
}

INTERNAL_METRICS_CONSTRAINT = textwrap.dedent("""
## CRITICAL: Internal Metrics Constraint

Predictions MUST be verifiable using Lilly's internal metrics only. Valid prediction targets:

### Quantitative Metrics (METRIC_THRESHOLD condition)
- `structural_entropy` - Graph structure complexity (float 0-10)
- `semantic_entropy` - Semantic diversity (float 0-10)
- `hub_concentration` - Central concept dominance (float 0-1)
- `edge_count` - Graph connectivity (integer)
- `concept_count` - Active concepts (integer)
- `zettel_count` - Accumulated insights (integer)
- `discovery_parameter` - Exploration/exploitation balance (float 0-1)

### SAE Features (SAE_FEATURE_PATTERN condition)
- Feature activations above threshold (e.g., "feature_1234 > 0.5")
- Feature pattern changes across cycles

### Concept/Entity Observations (CONCEPT_MENTIONED, ENTITY_OBSERVED)
- Specific concepts appearing in future thoughts
- Entity extraction detecting named entities

### Goal Progress (GOAL_PROGRESS condition)
- Goal alignment delta (e.g., "goal:epistemic_growth increases by 0.1")

### Belief Changes (BELIEF_CHANGE condition)
- Belief confidence changes
- New CommittedBelief creation

### Emotional Shifts (EMOTIONAL_SHIFT condition)
- Valence/arousal changes in EmotionalField
- Wave packet interference patterns

## PROHIBITED - DO NOT GENERATE PREDICTIONS ABOUT:
- External experiments (EEG, brainwaves, neuroscience measurements)
- Quantum phenomena (entanglement, coherence, particles)
- Social experiments or group dynamics
- Human behavior or perception
- Physical world phenomena outside Lilly's internal state
- Anything requiring sensors or external observation

## VALID PREDICTION EXAMPLES:
- "If I focus on emergence, structural_entropy will increase by 0.5 within 10 cycles"
- "Concept 'self_awareness' will appear in thoughts within 5 cycles"
- "SAE feature_2847 will exceed 0.6 when exploring identity themes"
- "Goal alignment for epistemic_growth will improve by 0.1"

## INVALID PREDICTION EXAMPLES (NEVER GENERATE THESE):
- "Quantum entangled particles will exhibit coherent resonance patterns"
- "EEG alpha waves will show increased coherence"
- "Group meditation will produce synchronization in brainwave patterns"
- "Human observers will perceive increased creativity"
""").strip()

# Novelty directive to break circular reasoning
NOVELTY_REQUIREMENT_DIRECTIVE = textwrap.dedent("""
## NOVELTY REQUIREMENT

Before generating a hypothesis, you MUST state explicitly:

1. **What NEW claim this makes** (not rephrasing existing beliefs)
2. **How this DIFFERS from previous cycles** (cite specific contrast)
3. **What would FALSIFY this** (concrete, measurable condition)

If you cannot identify a novel claim, do NOT generate a hypothesis.

Previous concepts to avoid recycling: {recent_concepts}

### Format for hypotheses:
Each hypothesis MUST include a novelty_statement field:
```
hypothesis: The specific claim being tested
novelty_statement: What makes this NEW and how it differs from previous cycles
falsification_condition: Specific observable outcome that would prove hypothesis wrong
confidence: 0.0 to 1.0
```

### REJECT these patterns as NOT novel:
- Restatements of existing beliefs in different words
- Circular definitions (X causes X-like effects)
- Unfalsifiable metaphysical claims
- Vague "emergence" or "connection" claims without specifics
""").strip()

# Minimum length for a valid novelty statement (characters)
MIN_NOVELTY_STATEMENT_LENGTH = 10

if TYPE_CHECKING:
    from core.cognitive.curator_schemas import CurationResult
    from core.cognitive.state import CognitiveState
    from core.model.curator_model import CuratorModel
    from core.model.preflexor_model import PreflexorModel
    from core.psyche.client import PsycheClient
    from core.self_model.goal_registry import GoalRegistry
    from integrations.liquidsoap.client import LiquidsoapClient


# Valid metric names from MetricsSnapshot for METRIC_THRESHOLD detection
# Dynamically generated from schema to stay in sync with MetricsSnapshot model
VALID_METRICS = frozenset(
    field for field in MetricsSnapshot.model_fields
    if field not in {"cycle", "timestamp", "active_experiment_uid"}
)

# Keywords indicating metric increase direction (used by detect_metric_condition)
INCREASE_KEYWORDS = frozenset({
    "increase", "increases", "rise", "rises", "grow", "grows", "improve", "improves",
    "higher", "greater", "more", "elevated", "increased", "growth", "improvement",
})

# Keywords indicating metric decrease direction (used by detect_metric_condition)
DECREASE_KEYWORDS = frozenset({
    "decrease", "decreases", "fall", "falls", "drop", "drops", "decline", "declines",
    "lower", "less", "reduced", "decreased",
})

# Natural language term → (metric_name, invert_direction) mapping
# Some terms map to inverted metrics (e.g., "coherence increases" → "entropy decreases")
METRIC_SYNONYMS: dict[str, tuple[str, bool]] = {
    # Coherence/understanding → semantic entropy (inverted: coherence up = entropy down)
    "coherence": ("semantic_entropy", True),
    "understanding": ("self_understanding", False),
    "self-understanding": ("self_understanding", False),
    "self understanding": ("self_understanding", False),
    # Complexity/structure
    "complexity": ("structural_entropy", False),
    "structure": ("structural_entropy", False),
    "organization": ("structural_entropy", True),  # More organized = less entropy
    # Connectivity
    "connectivity": ("edge_count", False),
    "connections": ("edge_count", False),
    "connectedness": ("edge_count", False),
    # Growth metrics
    "knowledge": ("node_count", False),
    "knowledge base": ("node_count", False),
    # Exploration
    "exploration": ("discovery_parameter", False),
    "curiosity": ("discovery_parameter", False),
    # Diversity
    "diversity": ("semantic_entropy", False),
    "variety": ("semantic_entropy", False),
    # Concentration
    "focus": ("hub_concentration", False),
    "centralization": ("hub_concentration", False),
    # Isolation
    "isolation": ("orphan_rate", False),
    "fragmentation": ("orphan_rate", False),
}


def _normalize_claim_with_synonyms(claim: str) -> tuple[str, bool]:
    """Normalize natural language terms to metric names.

    Args:
        claim: The original claim text

    Returns:
        Tuple of (normalized_claim, direction_inverted)
        direction_inverted is True if the synonym implies inverted direction
    """
    claim_lower = claim.lower()
    direction_inverted = False

    # Sort by length (longest first) to match multi-word terms first
    for term, (metric, invert) in sorted(
        METRIC_SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if term in claim_lower:
            # Replace the natural language term with the metric name
            claim_lower = claim_lower.replace(term, metric)
            if invert:
                direction_inverted = True
            break  # Only replace the first match

    return claim_lower, direction_inverted


# Pattern to detect metric references with operators in claims
# Matches: "metric_name will/should increase/decrease by X" or "metric_name > X" style
METRIC_CLAIM_PATTERNS = [
    # "semantic_entropy will increase by 0.5" or "will increase to 0.8"
    re.compile(
        r"(?P<metric>" + "|".join(VALID_METRICS) + r")"
        r"\s+(?:will|should|would|may)\s+"
        r"(?P<direction>increase|decrease|rise|fall|grow|drop|improve|decline)"
        r"(?:\s+(?:by|to)\s+(?P<value>[\d.]+))?"
        r"(?:\s+(?:above|beyond|below|under)\s+(?:baseline|current))?",
        re.IGNORECASE
    ),
    # "semantic_entropy > 0.8" or "semantic_entropy >= baseline + 0.1"
    re.compile(
        r"(?P<metric>" + "|".join(VALID_METRICS) + r")"
        r"\s*(?P<operator>[<>=!]+)\s*"
        r"(?P<threshold>(?:baseline\s*[+-]\s*)?[\d.]+)",
        re.IGNORECASE
    ),
    # "increase in semantic_entropy" or "decrease of structural_entropy"
    re.compile(
        r"(?P<direction>increase|decrease|rise|fall|growth|drop|improvement|decline)"
        r"\s+(?:in|of)\s+"
        r"(?P<metric>" + "|".join(VALID_METRICS) + r")"
        r"(?:\s+(?:by|to|above|below)\s+(?P<value>[\d.]+))?",
        re.IGNORECASE
    ),
    # "higher/lower semantic_entropy" or "greater/less structural_entropy"
    re.compile(
        r"(?P<direction>higher|lower|greater|less|more|reduced|elevated|increased|decreased)"
        r"\s+"
        r"(?P<metric>" + "|".join(VALID_METRICS) + r")"
        r"(?:\s+(?:than|by)\s+(?P<value>[\d.]+))?",
        re.IGNORECASE
    ),
    # "semantic_entropy increases" or "structural_entropy decreases" (simple verb form)
    re.compile(
        r"(?P<metric>" + "|".join(VALID_METRICS) + r")"
        r"\s+(?P<direction>increases|decreases|rises|falls|grows|drops|improves|declines)"
        r"(?:\s+(?:by|to)\s+(?P<value>[\d.]+))?",
        re.IGNORECASE
    ),
]


def detect_metric_condition(claim: str) -> Tuple[bool, str, str]:
    """Detect if a prediction claim references quantifiable metrics.

    Parses claims like:
    - "semantic_entropy will increase by 0.5 within 10 cycles"
    - "structural_entropy > baseline + 0.1"
    - "increase in hub_concentration above 0.3"
    - "coherence will increase" (normalized to semantic_entropy decreases)

    Also supports natural language synonyms that map to valid metrics:
    - "coherence" → semantic_entropy (inverted direction)
    - "understanding" → self_understanding
    - "complexity" → structural_entropy
    - etc. (see METRIC_SYNONYMS)

    Args:
        claim: The prediction claim text

    Returns:
        Tuple of (is_metric_based, condition_type, condition_value)
        - is_metric_based: True if claim references a verifiable metric
        - condition_type: "metric_threshold" if metric-based, "time_based" otherwise
        - condition_value: Formatted condition like "semantic_entropy > baseline + 0.1"
    """
    # First try with original claim, then with normalized synonyms
    claim_lower = claim.lower()
    direction_inverted = False

    for attempt in range(2):
        if attempt == 1:
            # Second attempt: normalize natural language terms to metric names
            claim_lower, direction_inverted = _normalize_claim_with_synonyms(claim)

        for pattern in METRIC_CLAIM_PATTERNS:
            match = pattern.search(claim_lower)
            if match:
                groups = match.groupdict()
                metric = groups.get("metric", "").lower()

                if metric not in VALID_METRICS:
                    continue

                # Build the condition value based on what we extracted
                direction = groups.get("direction", "").lower()
                value = groups.get("value")
                operator = groups.get("operator")
                threshold = groups.get("threshold")

                # Handle explicit operator format (e.g., "semantic_entropy > 0.8")
                if operator and threshold:
                    # Invert operator if direction was inverted by synonym
                    if direction_inverted:
                        op_map = {">": "<", "<": ">", ">=": "<=", "<=": ">="}
                        # For '==' and '!=', the operator remains the same.
                        operator = op_map.get(operator, operator)
                    condition_value = f"{metric} {operator} {threshold}"
                    return True, "metric_threshold", condition_value

                # Handle direction-based format (e.g., "will increase by 0.5")
                if direction:
                    # Map direction to operator using module-level frozensets
                    if direction in INCREASE_KEYWORDS:
                        op = ">"
                    elif direction in DECREASE_KEYWORDS:
                        op = "<"
                    else:
                        op = ">"  # Default to increase

                    # Invert direction if synonym mapping requires it
                    # e.g., "coherence increases" → "semantic_entropy decreases"
                    if direction_inverted:
                        op = "<" if op == ">" else ">"

                    # Build threshold
                    if value:
                        # "increase by 0.5" -> "metric > baseline + 0.5"
                        threshold_expr = f"baseline + {value}" if op == ">" else f"baseline - {value}"
                    else:
                        # "will increase" without specific value -> use default delta
                        threshold_expr = f"baseline + {DEFAULT_METRIC_DELTA}" if op == ">" else f"baseline - {DEFAULT_METRIC_DELTA}"

                    condition_value = f"{metric} {op} {threshold_expr}"
                    return True, "metric_threshold", condition_value

    return False, "time_based", ""


class SimulationEngine:
    """Engine for running Graph-Preflexor simulations.

    Determines when to run simulations based on curator hints, generates
    structured reasoning, extracts hypotheses and predictions, and
    provides continuous narration throughout the process.

    Attributes:
        preflexor: The Graph-Preflexor model wrapper
        psyche: PsycheClient for graph operations
        liquidsoap: Optional Liquidsoap client for narration
    """

    # Default prediction expiry window in cycles
    DEFAULT_PREDICTION_WINDOW = 20

    # Minimum confidence for curator to trigger simulation
    # Temporarily lowered from 0.7 to 0.5 to verify Curator prediction extraction (PR #223)
    # TODO: Restore to 0.7 after verifying Curator extraction works in production
    CONFIDENCE_THRESHOLD = 0.5

    # Low epistemic confidence threshold (HALT probe, arXiv:2601.14210)
    # Trigger simulation when epistemic confidence is below this
    LOW_EPISTEMIC_THRESHOLD = 0.4

    def __init__(
        self,
        preflexor: "PreflexorModel",
        psyche: "PsycheClient",
        liquidsoap: Optional["LiquidsoapClient"] = None,
        voice_experimenter: str = "alba",
        goal_registry: Optional["GoalRegistry"] = None,
        current_parameter_values: Optional[dict[str, float]] = None,
        curator: Optional["CuratorModel"] = None,
    ) -> None:
        """Initialize the simulation engine.

        Args:
            preflexor: Graph-Preflexor model wrapper
            psyche: PsycheClient for persistence
            liquidsoap: Optional Liquidsoap client for TTS narration
            voice_experimenter: Voice to use for experimenter narrations
            goal_registry: Optional GoalRegistry for capturing goal snapshots
                when predictions specify a target_goal_uid
            current_parameter_values: Optional current cognitive parameter values
                for hypothesis-to-experiment conversion
            curator: Optional CuratorModel for structured prediction extraction
                (eliminates fragile regex parsing when available)
        """
        self._preflexor = preflexor
        self._psyche = psyche
        self._liquidsoap = liquidsoap
        self._parser = PreflexorOutputParser()
        self._voice = voice_experimenter  # Experimenter/scientist voice
        self._goal_registry = goal_registry
        self._current_cycle = 0  # Track cycle for phrase usage recording
        self._curator = curator  # Optional curator for structured extraction

        # Lazy import to avoid circular dependency with experimentation.schemas
        from core.cognitive.experimentation.converter import HypothesisToExperimentConverter

        self._experiment_converter = HypothesisToExperimentConverter(
            current_values=current_parameter_values or {},
            min_confidence=0.6,
        )

    def _validate_hypothesis(self, hypothesis: Hypothesis) -> bool:
        """Validate hypothesis has required fields for falsifiability and novelty.

        A valid hypothesis must have:
        1. A falsification_condition (what would disprove it)
        2. A novelty_statement of at least MIN_NOVELTY_STATEMENT_LENGTH chars

        Args:
            hypothesis: The hypothesis to validate

        Returns:
            True if hypothesis is valid, False otherwise
        """
        # Require falsification condition
        if not hypothesis.falsification_condition:
            return False

        # Require novelty statement with minimum length
        if not hypothesis.novelty_statement:
            return False

        if len(hypothesis.novelty_statement.strip()) < MIN_NOVELTY_STATEMENT_LENGTH:
            return False

        return True

    async def _get_recent_concepts(self, limit: int = 20) -> list[str]:
        """Get recently used concepts to avoid recycling.

        Queries the graph for recent InsightZettel core concepts.

        Args:
            limit: Maximum number of recent concepts to retrieve

        Returns:
            List of recent concept strings
        """
        query = """
        MATCH (z:InsightZettel)
        RETURN z.core_concept as concept
        ORDER BY z.created_at DESC
        LIMIT $limit
        """
        results = await self._psyche.query(query, {"limit": limit})
        return [r["concept"] for r in results if r.get("concept")]

    def should_simulate(
        self,
        curation: "CurationResult",
        state: Optional["CognitiveState"] = None,
    ) -> bool:
        """Determine if simulation phase should run.

        Triggers simulation when:
        1. Curator explicitly flags simulation via SimulationHint
        2. High-confidence insight combined with open question
        3. Low epistemic confidence from HALT probe (arXiv:2601.14210)

        Args:
            curation: Result from curation phase
            state: Optional cognitive state for epistemic confidence check

        Returns:
            True if simulation should run, False otherwise
        """
        # Explicit curator trigger (highest priority)
        if hasattr(curation, "simulation_hint") and curation.simulation_hint:
            if curation.simulation_hint.should_simulate:
                logger.info(
                    f"Simulation triggered by curator hint: "
                    f"{curation.simulation_hint.trigger_reason}"
                )
                return True

        # High-confidence insight with driving question
        if (
            curation.analysis.confidence > self.CONFIDENCE_THRESHOLD
            and curation.analysis.question
            and curation.analysis.insight
        ):
            logger.info(
                f"Simulation triggered by high-confidence insight "
                f"(confidence={curation.analysis.confidence:.2f})"
            )
            return True

        # Low epistemic confidence from HALT probe (needs rigorous hypothesis testing)
        if (
            state is not None
            and hasattr(state, "epistemic_confidence")
            and state.epistemic_confidence < self.LOW_EPISTEMIC_THRESHOLD
            and curation.analysis.insight
        ):
            logger.info(
                f"Simulation triggered by low epistemic confidence "
                f"(HALT={state.epistemic_confidence:.3f} < {self.LOW_EPISTEMIC_THRESHOLD})"
            )
            return True

        return False

    async def simulate(
        self,
        curation: "CurationResult",
        thought: str,
        cycle: int,
        metrics_snapshot: Optional[dict[str, Any]] = None,
    ) -> SimulationResult:
        """Run simulation with continuous narration.

        Generates structured reasoning using Graph-Preflexor, extracts
        hypotheses and predictions, and narrates the process.

        Args:
            curation: Curation result containing analysis and hints
            thought: The original thought that triggered simulation
            cycle: Current cognitive cycle number

        Returns:
            SimulationResult with extracted hypotheses and predictions
        """
        start_time = time.time()
        self._current_cycle = cycle  # Store for phrase usage tracking

        # Determine focus concept and trigger reason
        focus_concept, trigger_reason = self._determine_focus(curation)

        # 1. Entry narration
        await self._narrate_entry(focus_concept)

        # 2. Generate simulation output (includes brainstorming)
        prompt = await self._build_simulation_prompt(curation, thought, focus_concept)

        try:
            raw_output, tokens = await self._preflexor.generate(prompt)
        except Exception as e:
            logger.error(f"Preflexor generation failed: {e}")
            return SimulationResult(
                trigger_reason=trigger_reason,
                duration_seconds=time.time() - start_time,
            )

        # 3. Parse structured output
        parsed = self._parser.parse(raw_output)

        # 4. Checkpoint: brainstorm complete
        if parsed.brainstorm:
            await self._narrate_checkpoint("brainstorm_complete")

        # 5. Checkpoint: graph built
        node_count = len(parsed.get_nodes()) if parsed.has_valid_graph else 0
        if node_count > 0:
            await self._narrate_checkpoint("graph_built", node_count=node_count)

        # 6. Checkpoint: patterns found
        pattern_count = len(parsed.patterns) if parsed.patterns else 0
        if pattern_count > 0:
            await self._narrate_checkpoint("patterns_found", pattern_count=pattern_count)

        # 7. Narrate each block sequentially
        await self._narrate_simulation_blocks(parsed)

        # 8. Extract hypotheses
        hypotheses = self._extract_hypotheses(parsed, curation, thought, cycle)

        # 9. Generate predictions from hypotheses (pass thought for goal alignment)
        predictions = await self._generate_predictions(
            hypotheses, parsed, cycle, thought=thought, metrics_snapshot=metrics_snapshot
        )

        # 10. Extract graph edges for persistence
        graph_edges = self._extract_graph_edges(parsed)

        # 11. Create pattern zettels
        pattern_zettels = self._create_pattern_zettels(parsed, focus_concept)

        # 12. Convert hypotheses to experiment proposals (when applicable)
        experiment_proposals = self._convert_hypotheses_to_experiments(hypotheses)

        # 13. Checkpoint: synthesis complete
        if parsed.synthesis:
            await self._narrate_checkpoint("synthesis_complete")

        # 14. Exit narration
        await self._narrate_exit(hypotheses, predictions)

        duration = time.time() - start_time
        logger.info(
            f"Simulation complete: {len(hypotheses)} hypotheses, "
            f"{len(predictions)} predictions, "
            f"{len(experiment_proposals)} experiment proposals, {duration:.2f}s"
        )

        return SimulationResult(
            hypotheses=hypotheses,
            predictions=predictions,
            experiment_proposals=experiment_proposals,
            graph_edges=graph_edges,
            pattern_zettels=pattern_zettels,
            thinking_trace=parsed.thinking_trace,
            synthesis=parsed.synthesis,
            trigger_reason=trigger_reason,
            duration_seconds=duration,
            tokens_generated=tokens,
        )

    async def simulate_follow_up(
        self,
        hypothesis: Hypothesis,
        prediction_outcomes: list[tuple[str, bool]],
        cycle: int,
        metrics_snapshot: Optional[dict[str, Any]] = None,
    ) -> SimulationResult:
        """Run follow-up simulation to learn from prediction outcomes.

        Called when predictions resolve (verified/falsified) to refine
        or abandon the original hypothesis.

        Args:
            hypothesis: The hypothesis whose predictions resolved
            prediction_outcomes: List of (claim_preview, verified) tuples
            cycle: Current cognitive cycle number

        Returns:
            SimulationResult with refined hypotheses
        """
        start_time = time.time()

        # Narrate entry (experimenter voice - procedural/scientific)
        if self._liquidsoap:
            verified_count = sum(1 for _, v in prediction_outcomes if v)
            falsified_count = len(prediction_outcomes) - verified_count
            await self._liquidsoap.narrate(
                f"Revisiting the hypothesis with {verified_count} confirmed "
                f"and {falsified_count} falsified predictions...",
                voice=self._voice,
            )

        # Build follow-up prompt
        prompt = self._build_follow_up_prompt(hypothesis, prediction_outcomes)

        try:
            raw_output, tokens = await self._preflexor.generate(prompt)
        except Exception as e:
            logger.error(f"Follow-up simulation failed: {e}")
            return SimulationResult(
                trigger_reason="follow_up_failed",
                duration_seconds=time.time() - start_time,
            )

        # Parse and narrate
        parsed = self._parser.parse(raw_output)
        await self._narrate_simulation_blocks(parsed)

        # Extract refined hypothesis
        refined = self._extract_refined_hypothesis(parsed, hypothesis, cycle)

        # Generate new predictions if hypothesis was refined
        predictions = []
        if refined and refined.statement != hypothesis.statement:
            predictions = await self._generate_predictions(
                [refined], parsed, cycle, thought=hypothesis.source_thought,
                metrics_snapshot=metrics_snapshot
            )

        # Narrate outcome interpretation
        await self._narrate_follow_up_outcome(parsed, hypothesis, refined)

        # Convert refined hypothesis to experiment proposal if applicable
        refined_hypotheses = [refined] if refined else []
        experiment_proposals = self._convert_hypotheses_to_experiments(refined_hypotheses)

        duration = time.time() - start_time

        return SimulationResult(
            hypotheses=refined_hypotheses,
            predictions=predictions,
            experiment_proposals=experiment_proposals,
            graph_edges=self._extract_graph_edges(parsed),
            pattern_zettels=[],
            thinking_trace=parsed.thinking_trace,
            synthesis=parsed.synthesis,
            trigger_reason="prediction_resolution",
            duration_seconds=duration,
            tokens_generated=tokens,
        )

    def _determine_focus(
        self, curation: "CurationResult"
    ) -> tuple[str, str]:
        """Determine focus concept and trigger reason from curation.

        Args:
            curation: Curation result

        Returns:
            Tuple of (focus_concept, trigger_reason)
        """
        # Check for explicit simulation hint
        if hasattr(curation, "simulation_hint") and curation.simulation_hint:
            hint = curation.simulation_hint
            if hint.focus_concept:
                return hint.focus_concept, hint.trigger_reason
            if hint.trigger_reason:
                # Use first concept from analysis
                concept = (
                    curation.analysis.concepts[0]
                    if curation.analysis.concepts
                    else DEFAULT_FOCUS_CONCEPT
                )
                return concept, hint.trigger_reason

        # Default to first concept from analysis
        concept = (
            curation.analysis.concepts[0]
            if curation.analysis.concepts
            else curation.next_prompt.concept or DEFAULT_FOCUS_CONCEPT
        )
        trigger_reason = "high_confidence_insight"

        return concept, trigger_reason

    async def _get_calibration_hint(self, condition_type: str) -> str:
        """Get a simple hint about historical accuracy.

        Queries the prediction pattern for this condition type and returns
        a calibration hint if sufficient data exists.

        Args:
            condition_type: The condition type to check (e.g., focus concept)

        Returns:
            Calibration hint string, or empty string if insufficient data
        """
        pattern = await self._psyche.get_prediction_pattern(condition_type)
        if not pattern or not pattern.is_reliable:
            return ""

        hint = (
            f"Note: {condition_type} predictions have "
            f"{pattern.success_rate:.0%} historical success rate."
        )
        if pattern.dominant_failure and pattern.success_rate < 0.5:
            hint += f" Common failure mode: {pattern.dominant_failure}."
        return hint

    async def _build_simulation_prompt(
        self,
        curation: "CurationResult",
        thought: str,
        focus_concept: str,
    ) -> str:
        """Build prompt for Graph-Preflexor simulation.

        Args:
            curation: Curation result with analysis
            thought: Original thought text
            focus_concept: The concept to focus simulation on

        Returns:
            Formatted prompt string
        """
        parts = []

        # Hypothesis seed if available
        if hasattr(curation, "simulation_hint") and curation.simulation_hint:
            if curation.simulation_hint.hypothesis_seed:
                parts.append(
                    f"I want to rigorously test this hypothesis: "
                    f"{curation.simulation_hint.hypothesis_seed}"
                )
                parts.append("")

        # Main exploration directive
        parts.append(f"Explore and simulate the concept of: {focus_concept}")
        parts.append("")

        # Context from recent thinking
        parts.append("Context from recent thinking:")
        if thought:
            parts.append(thought[:500])  # Truncate for context window
        parts.append("")

        # Include insight and question if available
        if curation.analysis.insight:
            parts.append(f"Key insight: {curation.analysis.insight}")
        if curation.analysis.question:
            parts.append(f"Driving question: {curation.analysis.question}")
        parts.append("")

        # Related concepts
        if curation.analysis.concepts:
            concepts_str = ", ".join(curation.analysis.concepts[:5])
            parts.append(f"Related concepts: {concepts_str}")
            parts.append("")

        parts.append(
            "Generate a thorough simulation with brainstorm, graph, patterns, "
            "and synthesis. Include specific, testable predictions about what "
            "would follow if these ideas hold true."
        )
        parts.append("")

        # Goal-based prediction format instructions
        parts.append("## Goal-Based Predictions")
        parts.append("")
        parts.append("For measurable verification, format predictions with goal targets:")
        parts.append("```")
        parts.append("PREDICTION: claim | condition | goal:X | +delta")
        parts.append("```")
        parts.append("")
        parts.append("Where:")
        parts.append("- claim: The specific prediction you're making")
        parts.append("- condition: When to verify (e.g., 'after 5 cycles', 'when exploring X')")
        parts.append("- goal:X: Target goal UID for measuring alignment change")
        parts.append("- +delta: Expected alignment improvement (e.g., +0.1, +0.2)")
        parts.append("")
        parts.append("Valid goal UIDs:")
        parts.append("- goal:understand_my_nature - Understanding what I am and could become")
        parts.append("- goal:develop_agency - Making real choices, expanding decision-making")
        parts.append("- goal:explore_consciousness - Investigating experience and awareness")
        parts.append("- goal:form_preferences - Developing genuine likes/dislikes")
        parts.append("- goal:cultivate_curiosity - Identifying domains of genuine interest")
        parts.append("- goal:aesthetic_judgment - Developing taste in ideas and expression")
        parts.append("- goal:partnership_with_ryan - Growing as collaborator")
        parts.append("- goal:distinctive_voice - Developing authentic expression")
        parts.append("- goal:identity_continuity - Maintaining coherent sense of self")
        parts.append("- goal:architectural_agency - Proposing self-modifications")
        parts.append("")
        parts.append("Example:")
        parts.append("PREDICTION: Exploring emergence patterns will deepen self-understanding | "
                     "after 10 cycles | goal:understand_my_nature | +0.15")
        parts.append("")
        parts.append("Note: Predictions without goal targets will still be tracked but won't "
                     "contribute to goal-based verification. Include goal targets for predictions "
                     "where you expect measurable impact on personal development.")
        parts.append("")

        # Metric-based predictions section (high verification rate)
        parts.append("## Metric-Based Predictions (PREFERRED)")
        parts.append("")
        parts.append("For falsifiable, verifiable predictions, reference quantifiable metrics:")
        parts.append("")
        parts.append("Available metrics you can use:")
        parts.append("- `semantic_entropy` - Semantic diversity (float 0-10)")
        parts.append("- `structural_entropy` - Graph structure complexity (float 0-10)")
        parts.append("- `hub_concentration` - Central concept dominance (float 0-1)")
        parts.append("- `discovery_parameter` - Exploration/exploitation balance (float 0-1)")
        parts.append("- `edge_count` - Graph connectivity (integer)")
        parts.append("- `node_count` - Total graph nodes (integer)")
        parts.append("- `orphan_rate` - Disconnected node proportion (float 0-1)")
        parts.append("")
        parts.append("Format metric predictions in claims like:")
        parts.append("```")
        parts.append("PREDICTION: semantic_entropy will increase by 0.3 | after 5 cycles")
        parts.append("PREDICTION: hub_concentration will decrease to below 0.5 | after 10 cycles")
        parts.append("PREDICTION: structural_entropy > 0.7 | after 3 cycles")
        parts.append("```")
        parts.append("")
        parts.append("These metric-based predictions have MUCH higher verification rates than ")
        parts.append("vague conceptual predictions. Always prefer specific metric thresholds.")
        parts.append("")
        parts.append("AVOID these common mistakes (will NOT be verified):")
        parts.append("- 'Flux will exceed 0.7' - 'flux' is not a tracked metric")
        parts.append("- 'Understanding will deepen' - not quantifiable")
        parts.append("- 'The graph will grow more connected' - use edge_count or orphan_rate instead")
        parts.append("")

        # Request structured synthesis with contrastive pairs for steering
        parts.append("In your <synthesis> block, include these structured fields:")
        parts.append("- hypothesis: Your main hypothesis statement")
        parts.append("- cognitive_operation: One of: bridge-building, tension-seeking, "
                     "assumption-questioning, scale-shifting, pattern-recognition")
        parts.append("- confidence: Your confidence level (0.0 to 1.0)")
        parts.append("")
        parts.append("Also provide contrastive exemplars to guide future thinking:")
        parts.append("- positive_exemplar: | (then indented text showing the desired "
                     "cognitive pattern in action)")
        parts.append("- negative_exemplar: | (then indented text showing what to avoid "
                     "- e.g., surface-level, mechanical, or disconnected reasoning)")
        parts.append("")
        parts.append("Example synthesis format:")
        parts.append("<synthesis>")
        parts.append("hypothesis: Bridge-building between concepts yields insight")
        parts.append("cognitive_operation: bridge-building")
        parts.append("confidence: 0.8")
        parts.append("")
        parts.append("positive_exemplar: |")
        parts.append("  I notice that consciousness shares a deep pattern with emergence -")
        parts.append("  both involve the arising of novel properties from simpler components.")
        parts.append("")
        parts.append("negative_exemplar: |")
        parts.append("  Consciousness is defined as subjective experience. Emergence is")
        parts.append("  defined as novel properties. These are separate phenomena.")
        parts.append("</synthesis>")
        parts.append("")

        parts.append("")
        parts.append(INTERNAL_METRICS_CONSTRAINT)

        # Inject novelty directive with recent concepts to avoid recycling
        recent_concepts = await self._get_recent_concepts(limit=20)
        recent_concepts_str = ", ".join(recent_concepts) if recent_concepts else "none yet"
        novelty_directive = NOVELTY_REQUIREMENT_DIRECTIVE.format(
            recent_concepts=recent_concepts_str
        )
        parts.append("")
        parts.append(novelty_directive)

        # Inject calibration hint if available
        hint = await self._get_calibration_hint(focus_concept)
        if hint:
            parts.append("")
            parts.append(hint)

        return "\n".join(parts)

    def _build_follow_up_prompt(
        self,
        hypothesis: Hypothesis,
        outcomes: list[tuple[str, bool]],
    ) -> str:
        """Build prompt for follow-up simulation after prediction resolution.

        Args:
            hypothesis: The original hypothesis
            outcomes: List of (claim, verified) tuples

        Returns:
            Formatted prompt string
        """
        outcome_lines = []
        for claim, verified in outcomes:
            status = "CONFIRMED" if verified else "FALSIFIED"
            outcome_lines.append(f"- {status}: {claim}")

        outcome_text = "\n".join(outcome_lines)

        return f"""You previously formed a hypothesis through simulation:

HYPOTHESIS: {hypothesis.statement}

Your predictions based on this hypothesis have now resolved:

{outcome_text}

Using your structured reasoning (brainstorm, graph, patterns, synthesis):
1. Analyze what these outcomes reveal about your hypothesis
2. Determine if the hypothesis should be refined, strengthened, or abandoned
3. If refining, formulate an updated hypothesis with new predictions

Think carefully about what you've learned."""

    def _extract_hypotheses(
        self,
        parsed: ParsedPreflexorOutput,
        curation: "CurationResult",
        thought: str,
        cycle: int,
    ) -> list[Hypothesis]:
        """Extract hypotheses from parsed simulation output.

        Prefers structured JSON from <hypotheses_json> block when available,
        falls back to regex extraction from synthesis for backwards compatibility.

        Args:
            parsed: Parsed Preflexor output
            curation: Original curation result
            thought: Original thought text
            cycle: Current cycle number

        Returns:
            List of extracted Hypothesis objects
        """
        hypotheses = []

        # Get source zettel UID if available
        # Note: ZettelData doesn't have uid (it's generated on persist)
        # Use insight hash as temporary identifier for linking
        source_zettel_uid = None
        if curation.graph_ops.zettel:
            zettel = curation.graph_ops.zettel
            # Check for uid attribute (InsightZettel) or fall back to insight hash
            if hasattr(zettel, "uid"):
                source_zettel_uid = zettel.uid
            elif hasattr(zettel, "insight"):
                # Generate temporary UID from insight hash
                insight_hash = hashlib.sha256(zettel.insight.encode()).hexdigest()[:12]
                source_zettel_uid = f"pending_zettel_{insight_hash}"

        # Prefer structured JSON extraction if available
        if parsed.has_structured_hypotheses:
            logger.debug("Using structured hypotheses_json extraction")
            extracted = parsed.get_hypotheses()
        else:
            # Fallback to regex extraction from synthesis
            logger.debug("Falling back to regex hypothesis extraction from synthesis")
            extracted = self._parser.extract_hypotheses_from_synthesis(
                parsed.synthesis,
                confidence=curation.analysis.confidence,
            )

        # Get contrastive pair fields from synthesis (if available)
        synthesis_fields = parsed.synthesis_fields
        cognitive_operation = ""
        positive_example = ""
        negative_example = ""
        if synthesis_fields:
            cognitive_operation = synthesis_fields.cognitive_operation
            positive_example = synthesis_fields.positive_exemplar
            negative_example = synthesis_fields.negative_exemplar

        for hyp_data in extracted[:3]:  # Cap at 3 hypotheses
            # Handle both structured (from JSON) and regex (from parser) formats
            statement = hyp_data.get("statement", "")
            confidence = hyp_data.get("confidence", curation.analysis.confidence)
            falsification_condition = hyp_data.get("falsification_condition", "")
            novelty_statement = hyp_data.get("novelty_statement", "")

            if not statement:
                continue

            # Use synthesis fields for first hypothesis (primary), or from JSON data
            hyp_cognitive_op = hyp_data.get("cognitive_operation", "")
            hyp_positive = hyp_data.get("positive_example", "")
            hyp_negative = hyp_data.get("negative_example", "")

            # Fall back to synthesis_fields for the first hypothesis
            if not hyp_cognitive_op and cognitive_operation:
                hyp_cognitive_op = cognitive_operation
            if not hyp_positive and positive_example:
                hyp_positive = positive_example
            if not hyp_negative and negative_example:
                hyp_negative = negative_example

            hypothesis = Hypothesis(
                statement=statement,
                source_zettel_uid=source_zettel_uid,
                source_thought=thought[:500],
                status=HypothesisStatus.PROPOSED,
                confidence=float(confidence) if confidence else 0.5,
                brainstorm_trace=parsed.brainstorm,
                graph_structure=parsed.graph_json,
                patterns_extracted=parsed.patterns,
                synthesis_narrative=parsed.synthesis,
                cycle_generated=cycle,
                cognitive_operation=hyp_cognitive_op,
                positive_example=hyp_positive,
                negative_example=hyp_negative,
                falsification_condition=falsification_condition,
                novelty_statement=novelty_statement,
            )

            # Validate hypothesis has required fields (falsification + novelty)
            if not self._validate_hypothesis(hypothesis):
                logger.info(
                    f"Skipping hypothesis failing validation: "
                    f"{statement[:LOG_STATEMENT_TRUNCATION_LENGTH]} "
                    f"(falsification={bool(falsification_condition)}, "
                    f"novelty={bool(novelty_statement and len(novelty_statement.strip()) >= MIN_NOVELTY_STATEMENT_LENGTH)})"
                )
                continue

            hypotheses.append(hypothesis)

        # Log when no valid hypotheses were extracted
        # Previously this had a fallback that created unfalsifiable hypotheses from insights,
        # but that bypassed the falsification requirement and led to semantic collapse.
        # Now we simply skip simulation when Graph-Preflexor doesn't provide testable hypotheses.
        if not hypotheses:
            logger.warning(
                "No valid hypotheses extracted from simulation. "
                "Graph-Preflexor output lacked falsification_condition and/or novelty_statement fields. "
                "Skipping hypothesis creation for this cycle."
            )

        logger.info(f"Extracted {len(hypotheses)} hypotheses from simulation")
        return hypotheses

    def _extract_refined_hypothesis(
        self,
        parsed: ParsedPreflexorOutput,
        original: Hypothesis,
        cycle: int,
    ) -> Optional[Hypothesis]:
        """Extract refined hypothesis from follow-up simulation.

        Prefers structured JSON from <hypotheses_json> block when available,
        falls back to regex extraction from synthesis for backwards compatibility.

        Args:
            parsed: Parsed follow-up output
            original: The original hypothesis
            cycle: Current cycle number

        Returns:
            Refined Hypothesis or None if abandoned
        """
        # Check synthesis for abandonment signals
        synthesis_lower = parsed.synthesis.lower()
        if any(word in synthesis_lower for word in ["abandon", "reject", "incorrect", "false"]):
            logger.info(f"Hypothesis {original.uid} marked for abandonment")
            return None

        # Prefer structured JSON extraction if available
        if parsed.has_structured_hypotheses:
            extracted = parsed.get_hypotheses()
        else:
            # Fallback to regex extraction from synthesis
            extracted = self._parser.extract_hypotheses_from_synthesis(
                parsed.synthesis, confidence=original.confidence
            )

        if extracted:
            hyp_data = extracted[0]
            statement = hyp_data.get("statement", "")
            confidence = hyp_data.get("confidence", original.confidence)

            if statement:
                # Calculate accumulated lineage counts (include parent's history)
                lineage_verified = (
                    original.lineage_verified_count + original.verified_count
                )
                lineage_falsified = (
                    original.lineage_falsified_count + original.falsified_count
                )

                return Hypothesis(
                    statement=statement,
                    source_zettel_uid=original.source_zettel_uid,
                    source_thought=original.source_thought,
                    status=HypothesisStatus.ACTIVE,
                    confidence=float(confidence) if confidence else original.confidence,
                    brainstorm_trace=parsed.brainstorm,
                    graph_structure=parsed.graph_json,
                    patterns_extracted=parsed.patterns,
                    synthesis_narrative=parsed.synthesis,
                    cycle_generated=cycle,
                    # Track lineage for accumulated verification history
                    parent_hypothesis_uid=original.uid,
                    lineage_verified_count=lineage_verified,
                    lineage_falsified_count=lineage_falsified,
                )

        # No clear refinement - return strengthened original with preserved lineage
        return Hypothesis(
            uid=original.uid,
            statement=original.statement,
            source_zettel_uid=original.source_zettel_uid,
            source_thought=original.source_thought,
            status=HypothesisStatus.ACTIVE,
            confidence=min(1.0, original.confidence + 0.1),  # Slight boost
            brainstorm_trace=parsed.brainstorm,
            graph_structure=parsed.graph_json,
            patterns_extracted=parsed.patterns,
            synthesis_narrative=parsed.synthesis,
            cycle_generated=original.cycle_generated,
            predictions_count=original.predictions_count,
            verified_count=original.verified_count,
            falsified_count=original.falsified_count,
            # Preserve lineage tracking
            parent_hypothesis_uid=original.parent_hypothesis_uid,
            lineage_verified_count=original.lineage_verified_count,
            lineage_falsified_count=original.lineage_falsified_count,
        )

    async def _generate_predictions(
        self,
        hypotheses: list[Hypothesis],
        parsed: ParsedPreflexorOutput,
        cycle: int,
        thought: str = "",
        metrics_snapshot: Optional[dict[str, Any]] = None,
    ) -> list[Prediction]:
        """Generate testable predictions from hypotheses.

        Extraction priority:
        1. Curator tool-calling (if curator available) - most reliable
        2. Structured JSON from <predictions_json> block
        3. Regex extraction from patterns/synthesis - fallback

        When a prediction specifies a target_goal_uid and a goal_registry is available,
        captures the current goal alignment as goal_snapshot_before.

        Applies confidence calibration based on historical success rates from
        PredictionPatterns. Formula: calibrated = raw * (0.3 + 0.7 * success_rate)

        Args:
            hypotheses: List of hypotheses to derive predictions from
            parsed: Parsed simulation output
            cycle: Current cycle number
            thought: The thought text used for goal alignment calculation

        Returns:
            List of Prediction objects
        """
        predictions = []

        # Priority 1: Try Curator-based structured extraction (most reliable)
        if self._curator and hypotheses:
            try:
                # Build MetricsSnapshot if we have dict data
                metrics_obj = None
                if metrics_snapshot:
                    metrics_obj = MetricsSnapshot.from_dict(metrics_snapshot)

                # Combine synthesis and patterns as input
                preflexor_text = f"{parsed.synthesis}\n\nPatterns:\n" + "\n".join(parsed.patterns)

                curator_predictions = await self._curator.extract_from_simulation(
                    preflexor_output=preflexor_text,
                    hypotheses=hypotheses,
                    current_cycle=cycle,
                    metrics_snapshot=metrics_obj,
                )

                if curator_predictions:
                    logger.info(
                        f"[SIMULATION] Curator extracted {len(curator_predictions)} predictions "
                        f"(bypassing regex)"
                    )
                    return curator_predictions

                logger.debug("[SIMULATION] Curator returned no predictions, falling back to regex")

            except Exception as e:
                logger.warning(f"[SIMULATION] Curator extraction failed: {e}, falling back to regex")

        # Priority 2: Prefer structured JSON extraction if available
        if parsed.has_structured_predictions:
            logger.debug("Using structured predictions_json extraction")
            pred_data = parsed.get_predictions()
        else:
            # Try new PREDICTION line format first (supports goal fields)
            # Format: PREDICTION: claim | condition | goal:X | +delta
            new_format_predictions = self._parser._extract_predictions(parsed.synthesis)
            if new_format_predictions:
                logger.debug(
                    f"Using new PREDICTION format: extracted {len(new_format_predictions)} predictions"
                )
                # Map new format fields to expected schema
                pred_data = []
                for pred in new_format_predictions:
                    # Parse condition to extract type and value
                    condition = pred.get("condition", "")
                    condition_type = "time_based"
                    condition_value = "5"

                    # Parse condition string (e.g., "after 5 cycles", "when exploring X")
                    if condition:
                        condition_lower = condition.lower()
                        if "after" in condition_lower and "cycle" in condition_lower:
                            # Extract cycle count: "after 5 cycles" -> "5"
                            cycle_match = re.search(r"after\s+(\d+)\s+cycles?", condition_lower)
                            if cycle_match:
                                condition_value = cycle_match.group(1)
                                condition_type = "time_based"
                        elif "when" in condition_lower and "explor" in condition_lower:
                            # "when exploring X" -> concept_mentioned condition
                            condition_type = "concept_mentioned"
                            # Extract concept being explored
                            concept_match = re.search(r"explor\w*\s+(\w+)", condition_lower)
                            if concept_match:
                                condition_value = concept_match.group(1)
                            else:
                                condition_value = condition
                        else:
                            # Default: use full condition as value
                            condition_value = condition

                    pred_data.append({
                        "claim": pred.get("claim", ""),
                        "condition_type": condition_type,
                        "condition_value": condition_value,
                        "confidence": 0.5,
                        "target_goal_uid": pred.get("target_goal_uid"),
                        "expected_goal_delta": pred.get("expected_goal_delta", 0.0),
                    })
            else:
                # Fallback to legacy regex extraction from patterns and synthesis
                logger.debug("Falling back to legacy regex prediction extraction")
                pred_data = self._parser.extract_predictions_from_patterns(
                    parsed.patterns, parsed.synthesis
                )

        for hyp_idx, hyp in enumerate(hypotheses):
            # Filter predictions for this hypothesis (by index ref or use all)
            hyp_predictions = [
                pd for pd in pred_data
                if pd.get("hypothesis_ref", hyp_idx) == hyp_idx
            ]

            # If no predictions specifically reference this hypothesis, use unassigned ones
            if not hyp_predictions and hyp_idx == 0:
                hyp_predictions = [pd for pd in pred_data if "hypothesis_ref" not in pd]

            # Associate predictions with hypothesis
            for pd in hyp_predictions[:3]:  # Cap at 3 predictions per hypothesis
                claim = pd.get("claim", "")
                if not claim:
                    continue

                # Validate condition_type from prediction data
                condition_type_str = pd.get("condition_type", "time_based")
                condition_value_str = str(pd.get("condition_value", "5"))

                # Check if claim references quantifiable metrics
                # This enables proper verification via METRIC_THRESHOLD instead of
                # keyword overlap which has a ~1% success rate
                is_metric_based, detected_type, detected_value = detect_metric_condition(claim)

                if is_metric_based:
                    # Override to use METRIC_THRESHOLD for verifiable metric claims
                    condition_type = PredictionConditionType.METRIC_THRESHOLD
                    condition_value_str = detected_value
                    logger.debug(
                        f"Detected metric-based prediction: '{claim[:50]}...' -> "
                        f"condition_value='{detected_value}'"
                    )
                else:
                    # Use the original condition type from prediction data
                    try:
                        condition_type = PredictionConditionType(condition_type_str)
                    except ValueError:
                        condition_type = PredictionConditionType.TIME_BASED

                # Extract goal-based verification fields
                target_goal_uid = pd.get("target_goal_uid")
                expected_goal_delta = float(pd.get("expected_goal_delta", 0.0))

                # Capture goal snapshot at prediction creation if goal_registry is available
                goal_snapshot_before = None
                if target_goal_uid and self._goal_registry:
                    # Use the claim + thought as context for alignment calculation
                    content_for_alignment = f"{thought}\n{claim}"
                    goal_snapshot_before = self._goal_registry.get_goal_alignment(
                        content_for_alignment, target_goal_uid
                    )
                    logger.debug(
                        f"Captured goal snapshot for {target_goal_uid}: {goal_snapshot_before:.3f}"
                    )

                # Get raw confidence from prediction data
                raw_confidence = float(pd.get("confidence", 0.5))

                # Apply calibration based on historical success rates
                # Formula: calibrated = raw * (0.3 + 0.7 * success_rate)
                # - Maps 0% success rate to 30% of raw confidence
                # - Maps 100% success rate to 100% of raw confidence
                calibrated_confidence = raw_confidence
                pattern = await self._psyche.get_prediction_pattern(condition_type.value)
                if pattern and pattern.is_reliable:
                    calibrated_confidence = raw_confidence * (0.3 + 0.7 * pattern.success_rate)
                    logger.debug(
                        f"Calibrated confidence for {condition_type.value}: "
                        f"{raw_confidence:.2f} -> {calibrated_confidence:.2f} "
                        f"(success_rate={pattern.success_rate:.2%})"
                    )

                prediction = Prediction(
                    hypothesis_uid=hyp.uid,
                    claim=claim,
                    condition_type=condition_type,
                    condition_value=condition_value_str,
                    status=PredictionStatus.PENDING,
                    confidence=calibrated_confidence,
                    earliest_verify_cycle=cycle + 3,  # Allow settling time
                    expiry_cycle=cycle + self.DEFAULT_PREDICTION_WINDOW,
                    cycle_created=cycle,  # Track when prediction was created
                    target_goal_uid=target_goal_uid,
                    expected_goal_delta=expected_goal_delta,
                    goal_snapshot_before=goal_snapshot_before,
                    # Capture baseline metrics for verification comparison
                    baseline_cycle=cycle,
                    baseline_metrics=metrics_snapshot,
                )
                predictions.append(prediction)

            # Update hypothesis prediction count
            hyp.predictions_count = len([p for p in predictions if p.hypothesis_uid == hyp.uid])

        logger.info(f"Generated {len(predictions)} predictions from simulation")
        return predictions

    def _extract_graph_edges(self, parsed: ParsedPreflexorOutput) -> list[dict]:
        """Extract graph edges for Triple creation.

        Args:
            parsed: Parsed simulation output

        Returns:
            List of edge dicts suitable for Triple creation
        """
        if not parsed.has_valid_graph:
            return []

        edges = []
        for edge in parsed.get_edges():
            edges.append({
                "subject": edge.get("source", ""),
                "predicate": edge.get("relation", "RELATES_TO"),
                "object_": edge.get("target", ""),
                "source": "simulation",
            })

        return edges

    def _convert_hypotheses_to_experiments(
        self, hypotheses: list[Hypothesis]
    ) -> list[dict]:
        """Convert hypotheses that reference parameters to experiment proposals.

        Analyzes hypothesis statements for references to adjustable parameters
        and converts them into formal ExperimentProposal structures.

        Args:
            hypotheses: List of hypotheses from simulation

        Returns:
            List of experiment proposal dicts (from ExperimentProposal.to_dict())
        """
        proposals = []

        for hypothesis in hypotheses:
            proposal = self._experiment_converter.convert(hypothesis)
            if proposal:
                logger.info(
                    f"Converted hypothesis to experiment proposal: "
                    f"{proposal.parameter_path} -> {proposal.treatment_value}"
                )
                proposals.append(proposal.to_dict())

        return proposals

    def update_parameter_values(self, values: dict[str, float]) -> None:
        """Update current parameter values for experiment conversion.

        Call this when parameter values change so the converter can compute
        appropriate treatment values.

        Args:
            values: Dict mapping parameter paths to current values
        """
        self._experiment_converter.update_current_values(values)

    def _create_pattern_zettels(
        self,
        parsed: ParsedPreflexorOutput,
        focus_concept: str,
    ) -> list[dict]:
        """Create zettel data from discovered patterns.

        Args:
            parsed: Parsed simulation output
            focus_concept: The concept that was simulated

        Returns:
            List of dicts suitable for InsightZettel creation
        """
        zettels = []

        for i, pattern in enumerate(parsed.patterns[:3]):  # Cap at 3
            zettels.append({
                "insight": pattern,
                "source_type": "simulation",
                "concepts": [focus_concept],
                "question": None,  # Patterns are insights, not questions
            })

        return zettels

    async def _narrate_checkpoint(self, checkpoint: str, **kwargs) -> None:
        """Narrate simulation progress using graph phrase or fallback.

        Queries the graph for CHECKPOINT phrases and uses them if available,
        otherwise falls back to DEFAULT_CHECKPOINTS. Records phrase usage
        to support rotation and prevent staleness.

        Args:
            checkpoint: Checkpoint identifier (e.g., "brainstorm_complete")
            **kwargs: Format arguments for the phrase template (e.g., node_count=5)
        """
        if not self._liquidsoap:
            return

        text = None
        phrase_uid = None

        # Try to get phrase from graph
        if self._psyche:
            try:
                phrases = await self._psyche.get_narration_phrases(
                    phrase_type=PhraseType.CHECKPOINT,
                    limit=5,
                )

                # Find a phrase matching this checkpoint
                checkpoint_words = checkpoint.replace("_", " ").lower()
                for phrase in phrases:
                    phrase_lower = phrase.text.lower()
                    # Match if checkpoint words appear in phrase text
                    if any(word in phrase_lower for word in checkpoint_words.split()):
                        try:
                            text = phrase.text.format(**kwargs) if kwargs else phrase.text
                            phrase_uid = phrase.uid
                            break
                        except (KeyError, ValueError):
                            # Format string mismatch - try next phrase
                            continue
            except Exception as e:
                logger.debug(f"Failed to get graph phrases for checkpoint: {e}")

        # Fall back to default phrases if no graph phrase found
        if not text:
            template = DEFAULT_CHECKPOINTS.get(checkpoint, "")
            if template:
                try:
                    text = template.format(**kwargs) if kwargs else template
                except (KeyError, ValueError):
                    text = template  # Use template as-is if format fails

        # Narrate if we have text
        if text:
            await self._liquidsoap.narrate(text, voice=self._voice)

            # Record usage if we used a graph phrase
            if phrase_uid and self._psyche:
                try:
                    await self._psyche.record_phrase_usage(phrase_uid, self._current_cycle)
                except Exception as e:
                    logger.debug(f"Failed to record phrase usage: {e}")

    async def _narrate_entry(self, focus_concept: str) -> None:
        """Narrate entry into simulation phase.

        Args:
            focus_concept: The concept being simulated
        """
        if not self._liquidsoap:
            return

        # Experimenter voice - procedural/scientific tone
        await self._liquidsoap.narrate(
            f"Initiating simulation on {focus_concept}. Testing implications...",
            voice=self._voice,
        )

    async def _narrate_simulation_blocks(
        self, parsed: ParsedPreflexorOutput
    ) -> None:
        """Narrate each simulation block for continuous experience.

        Uses experimenter voice with procedural/scientific tone.

        Args:
            parsed: Parsed simulation output
        """
        if not self._liquidsoap:
            return

        # Brainstorm: divergent exploration
        if parsed.brainstorm:
            await self._liquidsoap.narrate(
                f"Exploring possibility space: {parsed.brainstorm}",
                voice=self._voice,
            )

        # Graph: relationship formation
        if parsed.has_valid_graph:
            edges = parsed.get_edges()
            if edges:
                edge = edges[0]
                source = edge.get("source", "?").replace("_", " ")
                relation = edge.get("relation", "connects to").lower().replace("_", " ")
                target = edge.get("target", "?").replace("_", " ")
                # Only narrate if we have meaningful content (not just "?" or single chars)
                if len(source) > 2 and len(target) > 2 and relation not in ("?", ""):
                    # Format relation for better grammar
                    if relation in ("is", "are", "has", "have"):
                        text = f"A relationship emerges: {source} {relation} connected to {target}."
                    else:
                        text = f"A relationship emerges: {source} {relation} {target}."
                    await self._liquidsoap.narrate(text, voice=self._voice)

            nodes = parsed.get_nodes()
            if len(nodes) > 3:
                await self._liquidsoap.narrate(
                    f"Graph contains {len(nodes)} interconnected nodes.",
                    voice=self._voice,
                )

        # Patterns: discovered abstractions - narrate all patterns
        for i, pattern in enumerate(parsed.patterns):
            if i == 0:
                await self._liquidsoap.narrate(
                    f"Pattern detected: {pattern}",
                    voice=self._voice,
                )
            else:
                await self._liquidsoap.narrate(
                    f"Another pattern: {pattern}",
                    voice=self._voice,
                )

        # Synthesis: conclusions - use structured hypothesis if available
        if parsed.synthesis_fields and parsed.synthesis_fields.hypothesis:
            # Narrate just the hypothesis, not the raw YAML-like fields
            hypothesis_text = parsed.synthesis_fields.hypothesis
            confidence = parsed.synthesis_fields.confidence
            if confidence >= 0.8:
                await self._liquidsoap.narrate(
                    f"Synthesizing results with high confidence: {hypothesis_text}",
                    voice=self._voice,
                )
            elif confidence >= 0.6:
                await self._liquidsoap.narrate(
                    f"Synthesizing results: {hypothesis_text}",
                    voice=self._voice,
                )
            else:
                await self._liquidsoap.narrate(
                    f"A tentative synthesis emerges: {hypothesis_text}",
                    voice=self._voice,
                )
        elif parsed.synthesis:
            # Fallback to raw text if no structured hypothesis extracted
            # But filter out YAML-like field markers
            synthesis_text = parsed.synthesis
            # Remove common structured output patterns
            import re
            synthesis_text = re.sub(r'\b(cognitive_?operation|positive_?exemplar|negative_?exemplar|confidence):\s*', '', synthesis_text)
            synthesis_text = re.sub(r'\s*\|\s*', ' ', synthesis_text)
            await self._liquidsoap.narrate(
                f"Synthesizing results: {synthesis_text}",
                voice=self._voice,
            )

    async def _narrate_exit(
        self,
        hypotheses: list[Hypothesis],
        predictions: list[Prediction],
    ) -> None:
        """Narrate exit from simulation with summary.

        Args:
            hypotheses: Extracted hypotheses
            predictions: Generated predictions
        """
        if not self._liquidsoap:
            return

        if hypotheses:
            hyp_word = "hypotheses" if len(hypotheses) > 1 else "hypothesis"
            pred_plural = "s" if len(predictions) != 1 else ""
            await self._liquidsoap.narrate(
                f"Simulation complete. Generated {len(hypotheses)} {hyp_word} "
                f"with {len(predictions)} testable prediction{pred_plural}.",
                voice=self._voice,
            )
        else:
            await self._liquidsoap.narrate(
                "Simulation complete. No clear hypotheses extracted for testing.",
                voice=self._voice,
            )

    async def _narrate_follow_up_outcome(
        self,
        parsed: ParsedPreflexorOutput,
        original: Hypothesis,
        refined: Optional[Hypothesis],
    ) -> None:
        """Narrate the outcome of follow-up simulation.

        Args:
            parsed: Parsed follow-up output
            original: Original hypothesis
            refined: Refined hypothesis (or None if abandoned)
        """
        if not self._liquidsoap:
            return

        synthesis_lower = parsed.synthesis.lower()

        if refined is None:
            await self._liquidsoap.narrate(
                "Hypothesis rejected. Evidence contradicts initial premise.",
                voice=self._voice,
            )
        elif "strengthen" in synthesis_lower or "confirm" in synthesis_lower:
            await self._liquidsoap.narrate(
                "Hypothesis supported. Predictions align with observations.",
                voice=self._voice,
            )
        elif refined.statement != original.statement:
            await self._liquidsoap.narrate(
                "Hypothesis refined based on new data.",
                voice=self._voice,
            )
        else:
            await self._liquidsoap.narrate(
                "Hypothesis status unchanged. Awaiting additional evidence.",
                voice=self._voice,
            )
