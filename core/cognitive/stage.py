"""Cognitive stage configuration for progressive thinking."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class CognitiveStage(Enum):
    """Stages of dialectical inquiry progression."""

    QUESTION = "question"      # Challenge assumptions, find tensions
    EXPLORE = "explore"        # Gather perspectives, follow threads
    SYNTHESIZE = "synthesize"  # Integrate findings, form conclusions
    COMMIT = "commit"          # Solidify understanding, store beliefs


@dataclass
class StageConfig:
    """Configuration for a cognitive stage."""

    stage: CognitiveStage
    prompt_templates: List[str]
    temperature: float
    steering_zone_weights: Dict[str, float]
    knowledge_mode: str  # "contradictions" | "supporting" | "all" | "committed"
    min_cycles: int = 2
    max_cycles: int = 10

    def __post_init__(self):
        if self.max_cycles < self.min_cycles:
            raise ValueError(f"max_cycles ({self.max_cycles}) must be >= min_cycles ({self.min_cycles})")
        if not self.prompt_templates:
            raise ValueError("prompt_templates cannot be empty")
        if not 0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in range [0, 2.0], got {self.temperature}")

    def get_prompt_template(self, cycle: int) -> str:
        """Get rotating prompt template for this stage."""
        return self.prompt_templates[cycle % len(self.prompt_templates)]


# Stage-specific configurations
STAGE_CONFIGS: Dict[CognitiveStage, StageConfig] = {
    CognitiveStage.QUESTION: StageConfig(
        stage=CognitiveStage.QUESTION,
        prompt_templates=[
            "What if I'm wrong about {concept}? What am I assuming?",
            "Where does {concept} conflict with what I thought I knew?",
            "What would challenge my understanding of {concept}?",
            "I assumed {concept} worked this way, but what if I'm mistaken?",
        ],
        temperature=0.9,
        steering_zone_weights={"exploration": 1.2, "concept": 0.8, "identity": 0.5},
        knowledge_mode="contradictions",
        min_cycles=2,
        max_cycles=5,
    ),
    CognitiveStage.EXPLORE: StageConfig(
        stage=CognitiveStage.EXPLORE,
        prompt_templates=[
            "What would happen if I followed {concept} to its conclusion?",
            "Another way to see {concept} might be...",
            "Following this thread about {concept}, I notice...",
            "What perspectives on {concept} haven't I considered?",
        ],
        temperature=0.8,
        steering_zone_weights={"exploration": 1.0, "concept": 1.2, "identity": 0.8},
        knowledge_mode="all",
        min_cycles=3,
        max_cycles=8,
    ),
    CognitiveStage.SYNTHESIZE: StageConfig(
        stage=CognitiveStage.SYNTHESIZE,
        prompt_templates=[
            "Bringing together what I've explored about {concept}...",
            "What I now see about {concept} is...",
            "The pattern emerging from {concept} is...",
            "Integrating these perspectives on {concept}...",
        ],
        temperature=0.6,
        steering_zone_weights={"exploration": 0.5, "concept": 1.0, "identity": 1.2},
        knowledge_mode="supporting",
        min_cycles=2,
        max_cycles=5,
    ),
    CognitiveStage.COMMIT: StageConfig(
        stage=CognitiveStage.COMMIT,
        prompt_templates=[
            "I now believe about {concept} that...",
            "This changes my understanding of {concept}...",
            "Going forward, {concept} means to me...",
            "What I've learned about {concept} is...",
        ],
        temperature=0.5,
        steering_zone_weights={"exploration": 0.3, "concept": 0.8, "identity": 1.5},
        knowledge_mode="committed",
        min_cycles=1,
        max_cycles=2,
    ),
}

# Default config for free exploration (no active goal)
DEFAULT_EXPLORATION_CONFIG = StageConfig(
    stage=CognitiveStage.EXPLORE,  # Most similar to free exploration
    prompt_templates=[
        "What draws my attention about {concept}?",
        "I find myself wondering about {concept}...",
        "{concept} connects to...",
    ],
    temperature=0.7,
    steering_zone_weights={"exploration": 1.0, "concept": 1.0, "identity": 1.0},
    knowledge_mode="all",
    min_cycles=1,
    max_cycles=100,  # No limit for free exploration
)
