"""Stage-aware prompt building and steerer adjustment for progressive thinking."""
from typing import Optional, TYPE_CHECKING

from core.cognitive.goal import InquiryGoal
from core.cognitive.stage import StageConfig

if TYPE_CHECKING:
    from core.steering.hierarchical import HierarchicalSteerer


def build_stage_prompt(
    goal: Optional[InquiryGoal],
    config: StageConfig,
    concept: Optional[str] = None,
) -> str:
    """Build a prompt appropriate for the current stage.

    Args:
        goal: Active inquiry goal (if any)
        config: Stage configuration with prompt templates
        concept: Concept for free exploration (used when no goal)

    Returns:
        Formatted prompt string
    """
    # Determine the concept to explore
    if goal is not None:
        # Goal-directed: use the goal's question as the concept
        exploration_concept = goal.question
        cycle = goal.stage_cycles
    else:
        # Free exploration: use provided concept
        exploration_concept = concept or "my experience"
        cycle = 0

    # Get the rotating template for this cycle
    template = config.get_prompt_template(cycle)

    # Format the template with the concept
    return template.format(concept=exploration_concept)


def adjust_steerer_for_stage(
    steerer: "HierarchicalSteerer",
    config: StageConfig,
) -> None:
    """Adjust steerer zone magnitudes based on stage weights.

    Modifies the steerer in place, scaling each zone's vector by the
    corresponding weight from the stage configuration.

    Args:
        steerer: The hierarchical steerer to adjust
        config: Stage configuration with steering_zone_weights
    """
    for zone_name, weight in config.steering_zone_weights.items():
        if zone_name in steerer.vectors:
            # Scale the zone's vector by the weight
            steerer.vectors[zone_name] = steerer.vectors[zone_name] * weight
