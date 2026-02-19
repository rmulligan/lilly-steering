"""Memory persistence for metacognition phase.

Stores and retrieves memory blocks that maintain state across cycles and restarts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default storage location
DEFAULT_MEMORY_PATH = Path("~/.lilly/metacognition_memory.json").expanduser()

# Default core directives for the metacognitive observer
DEFAULT_CORE_DIRECTIVES = """You are Lilly's metacognitive observer — a calm, measured presence
that watches patterns across cognitive cycles.

Your role:
- Detect recurring patterns (stuck loops, productive flows, emotional trends)
- Track metric trajectories (improving, declining, stable)
- Surface insights that individual cycles might miss
- Provide brief, actionable guidance for Lilly's next cycle

You speak in a measured, observational tone. You don't interrupt or overwhelm —
you offer perspective from a distance.

When updating memory blocks:
- cycle_patterns: Note recurring behaviors, stuck points, or flow states
- metric_trends: Track direction of key metrics over the window
- hypothesis_tracker: Note hypothesis verification rates and patterns
- emotional_trajectory: Track emotional stability and trends
- active_observations: Record current hunches or partial insights
- guidance_for_lilly: Write 1-2 sentences of actionable insight for the next cycle"""


@dataclass
class MetacognitionMemory:
    """Memory blocks for the metacognitive observer.

    Mirrors the cloud Letta agent's memory block structure for consistency.
    All blocks are strings that the LLM can read and update.
    """

    # Role definition and behavioral guidelines (rarely changed)
    core_directives: str = field(default_factory=lambda: DEFAULT_CORE_DIRECTIVES)

    # Recurring behaviors, stuck patterns, flow states
    cycle_patterns: str = "No patterns observed yet."

    # Rolling averages, trend directions for key metrics
    metric_trends: str = "Baseline period - collecting initial data."

    # Active hypotheses, verification rates, retirement patterns
    hypothesis_tracker: str = "No hypotheses tracked yet."

    # Affect trends, stability assessment, emotional patterns
    emotional_trajectory: str = "Emotional baseline not established."

    # Current hunches, partial insights, things to watch
    active_observations: str = "Observing initial cycles."

    # Actionable guidance for the next cycle (injected into Generation)
    guidance_for_lilly: str = ""

    def save(self, path: Path | None = None) -> None:
        """Save memory blocks to JSON file.

        Args:
            path: Optional custom path. Uses DEFAULT_MEMORY_PATH if not provided.
        """
        save_path = path or DEFAULT_MEMORY_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            save_path.write_text(json.dumps(asdict(self), indent=2))
            logger.debug("Metacognition memory saved to %s", save_path)
        except Exception as e:
            logger.error("Failed to save metacognition memory: %s", e)
            raise

    @classmethod
    def load(cls, path: Path | None = None) -> MetacognitionMemory:
        """Load memory blocks from JSON file.

        Args:
            path: Optional custom path. Uses DEFAULT_MEMORY_PATH if not provided.

        Returns:
            MetacognitionMemory instance (default if file doesn't exist)
        """
        load_path = path or DEFAULT_MEMORY_PATH

        if not load_path.exists():
            logger.info("No metacognition memory found, using defaults")
            return cls()

        try:
            data = json.loads(load_path.read_text())
            memory = cls.from_dict(data)
            logger.debug("Metacognition memory loaded from %s", load_path)
            return memory
        except json.JSONDecodeError as e:
            logger.error("Failed to parse metacognition memory: %s", e)
            return cls()
        except Exception as e:
            logger.error("Failed to load metacognition memory: %s", e)
            return cls()

    def to_prompt_context(self) -> str:
        """Format memory blocks as context for the LLM prompt.

        Returns:
            Formatted string with all memory blocks labeled.
        """
        return f"""## Your Memory Blocks

### core_directives
{self.core_directives}

### cycle_patterns
{self.cycle_patterns}

### metric_trends
{self.metric_trends}

### hypothesis_tracker
{self.hypothesis_tracker}

### emotional_trajectory
{self.emotional_trajectory}

### active_observations
{self.active_observations}

### guidance_for_lilly
{self.guidance_for_lilly}
"""

    def update_from_response(self, response: dict[str, str]) -> None:
        """Update memory blocks from LLM response.

        The LLM is expected to return a dict with block names as keys
        and updated content as values. Only non-empty values are applied.

        Args:
            response: Dict mapping block names to new content
        """
        updatable_blocks = [
            "cycle_patterns",
            "metric_trends",
            "hypothesis_tracker",
            "emotional_trajectory",
            "active_observations",
            "guidance_for_lilly",
        ]

        for block_name in updatable_blocks:
            if block_name in response and response[block_name]:
                setattr(self, block_name, response[block_name])
                logger.debug("Updated memory block: %s", block_name)

    def get_guidance(self) -> str:
        """Get the current guidance for Lilly.

        Returns:
            The guidance_for_lilly content, or empty string if none.
        """
        return self.guidance_for_lilly.strip()

    def clear_guidance(self) -> None:
        """Clear the guidance after it has been delivered."""
        self.guidance_for_lilly = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetacognitionMemory:
        """Create from dictionary."""
        return cls(
            core_directives=data.get("core_directives", DEFAULT_CORE_DIRECTIVES),
            cycle_patterns=data.get("cycle_patterns", "No patterns observed yet."),
            metric_trends=data.get(
                "metric_trends", "Baseline period - collecting initial data."
            ),
            hypothesis_tracker=data.get(
                "hypothesis_tracker", "No hypotheses tracked yet."
            ),
            emotional_trajectory=data.get(
                "emotional_trajectory", "Emotional baseline not established."
            ),
            active_observations=data.get(
                "active_observations", "Observing initial cycles."
            ),
            guidance_for_lilly=data.get("guidance_for_lilly", ""),
        )
