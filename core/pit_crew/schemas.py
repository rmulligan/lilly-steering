"""Schemas for pit crew specialist system."""

from enum import Enum

from pydantic import BaseModel, Field


class SpecialistType(str, Enum):
    """Available specialist types."""
    # Phase 1-2: Deep specialists
    STEERING_QD = "steering.qd"
    SIMULATION_VERIFICATION = "simulation.verification"
    SUBSTRATE_CONSOLIDATION = "substrate.consolidation"
    REFLEXION_HEALTH = "reflexion.health"

    # Phase 3: Broad specialists
    ORCHESTRATOR_PHASES = "orchestrator.phases"
    PSYCHE_DB = "psyche.db"
    AUDIO_STREAMING = "audio.streaming"
    DOCS_CARTOGRAPHER = "docs.cartographer"

    # Innovation/Research
    INNOVATION_RESEARCH = "innovation.research"


class ProposedChange(BaseModel):
    """Single proposed PR from specialist."""
    title: str
    files: list[str]
    changes: str
    tests: str
    risk: str = Field(..., pattern="^(Low|Medium|High)$")
    dependencies: str | None = None


class SpecialistReport(BaseModel):
    """Structured report returned by specialist."""
    specialist: str
    date: str
    context: str
    findings: str
    proposed_changes: list[ProposedChange] = Field(default_factory=list, max_length=3)
    metrics: list[str] = Field(default_factory=list)
    cross_agent_question: str | None = None
    confidence: str = Field(..., pattern="^(low|med|high)$")
    references: list[str] = Field(default_factory=list)


class HubTicket(BaseModel):
    """Letta hub ticket entry."""
    owner: str
    type: str = Field(..., pattern="^(decision|todo|incident|metric|pr)$")
    status: str = Field(..., pattern="^(open|resolved)$")
    confidence: str = Field(..., pattern="^(low|med|high)$")
    refs: str
    summary: str
    next: str
