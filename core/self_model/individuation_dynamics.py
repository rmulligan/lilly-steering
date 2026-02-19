"""
Individuation Dynamics Layer - Lilly's 4th Layer.

This module tracks the processual/dynamic aspects of identity formation:
- How identity elements (commitments, values, beliefs) evolve over time
- Trajectory tracking with velocity, acceleration, momentum
- Attractor basins as stable identity configurations
- Phase transitions as significant shifts in identity dynamics
- Reflexive observation of the dynamics themselves

The existing self-model has:
1. Structural layer: Values, commitments, relationships (static snapshots)
2. Affective layer: Emotional field dynamics (8D Plutchik)
3. Epistemic layer: Beliefs with confidence tracking

This module adds:
4. Processual layer: *How* these elements change - velocity, acceleration,
   momentum, phase transitions, attractor basins.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:
    from core.self_model.models import Commitment


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


class DynamicsPhase(str, Enum):
    """Phase of an identity element's trajectory."""

    NASCENT = "nascent"  # Too few observations to determine
    STABLE = "stable"  # Low velocity, element is settled
    CRYSTALLIZING = "crystallizing"  # Increasing confidence/strength
    DISSOLVING = "dissolving"  # Decreasing confidence/strength
    OSCILLATING = "oscillating"  # Alternating direction
    VOLATILE = "volatile"  # High acceleration, rapid change


class TransitionTrigger(str, Enum):
    """What triggered a phase transition."""

    VELOCITY_SPIKE = "velocity_spike"  # Sudden acceleration
    PHASE_CHANGE = "phase_change"  # Shift in dynamics phase
    EVIDENCE_SURGE = "evidence_surge"  # Multiple reinforcements
    CONTRADICTION_SURGE = "contradiction_surge"  # Multiple challenges
    TIME_DECAY = "time_decay"  # Long period without reinforcement
    EXTERNAL_REVISION = "external_revision"  # Commitment explicitly revised


@dataclass
class IdentityTrajectory:
    """
    Tracks the evolution of an identity element over time.

    This captures the *dynamics* of identity formation - not just what
    Lilly believes/values/commits to, but how those elements are changing.

    Attributes:
        element_id: UUID of the commitment/value/belief being tracked
        element_type: Category of element (commitment, value, belief)
        positions: Historical confidence/strength values (bounded to HISTORY_LIMIT)
        velocities: First derivative - rate of change
        accelerations: Second derivative - change in rate
        timestamps: When each observation occurred
        current_phase: Most recent detected phase
        phase_stability: How long we've been in current phase (cycles)
    """

    # Bounded history to prevent unbounded growth
    HISTORY_LIMIT: ClassVar[int] = 50

    # Phase detection thresholds (units per second)
    # Velocity/acceleration now use actual time deltas, so these are in units/sec.
    # Tuned for typical cognitive cycle intervals of ~10-60 seconds.
    # A 0.05 change over 10s = 0.005 units/sec should be "active" (crystallizing/dissolving)
    # A 0.05 change over 60s = 0.0008 units/sec is borderline
    VELOCITY_THRESHOLD: ClassVar[float] = 0.0003  # Below this = stable (units/sec)
    VOLATILE_ACCELERATION: ClassVar[float] = 0.001  # Above this = volatile (units/sec^2)

    element_id: str
    element_type: str  # "commitment" | "value" | "belief"
    positions: list[float] = field(default_factory=list)
    velocities: list[float] = field(default_factory=list)
    accelerations: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    current_phase: DynamicsPhase = DynamicsPhase.NASCENT
    phase_stability: int = 0  # Cycles in current phase
    uid: str = field(default="")

    def __post_init__(self) -> None:
        if not self.uid:
            key = f"{self.element_id}:{self.element_type}"
            self.uid = f"trj:{hashlib.sha256(key.encode()).hexdigest()[:12]}"

    @property
    def current_velocity(self) -> float:
        """How fast is this element changing?"""
        return self.velocities[-1] if self.velocities else 0.0

    @property
    def current_acceleration(self) -> float:
        """How fast is the rate of change changing?"""
        return self.accelerations[-1] if self.accelerations else 0.0

    @property
    def current_position(self) -> float:
        """Current confidence/strength value."""
        return self.positions[-1] if self.positions else 0.5

    @property
    def current_momentum(self) -> float:
        """
        Velocity weighted by element's age.

        Older elements have more "inertia" - their changes carry more weight
        because they represent shifts in established identity.
        """
        if not self.velocities:
            return 0.0
        age_weight = min(1.0, len(self.positions) / 20)
        return self.current_velocity * age_weight

    @property
    def observation_count(self) -> int:
        """Number of observations recorded."""
        return len(self.positions)

    def observe(self, position: float, timestamp: Optional[datetime] = None) -> None:
        """
        Record a new observation of the element's state.

        Calculates velocity and acceleration from finite differences.
        Maintains bounded history.

        Args:
            position: Current confidence/strength value (0-1)
            timestamp: When this observation occurred
        """
        if timestamp is None:
            timestamp = utc_now()

        position = max(0.0, min(1.0, position))

        # Calculate velocity (first derivative) using actual time delta
        velocity = 0.0
        if self.positions and self.timestamps:
            dt = (timestamp - self.timestamps[-1]).total_seconds()
            if dt > 1e-9:  # Avoid division by zero
                velocity = (position - self.positions[-1]) / dt

        # Calculate acceleration (second derivative) using actual time delta
        acceleration = 0.0
        if self.velocities and self.timestamps and len(self.timestamps) > 0:
            dt = (timestamp - self.timestamps[-1]).total_seconds()
            if dt > 1e-9:
                acceleration = (velocity - self.velocities[-1]) / dt

        # Append new observations
        self.positions.append(position)
        self.velocities.append(velocity)
        self.accelerations.append(acceleration)
        self.timestamps.append(timestamp)

        # Maintain bounded history
        if len(self.positions) > self.HISTORY_LIMIT:
            self.positions = self.positions[-self.HISTORY_LIMIT :]
            self.velocities = self.velocities[-self.HISTORY_LIMIT :]
            self.accelerations = self.accelerations[-self.HISTORY_LIMIT :]
            self.timestamps = self.timestamps[-self.HISTORY_LIMIT :]

        # Update phase
        old_phase = self.current_phase
        self._update_phase()

        if self.current_phase == old_phase:
            self.phase_stability += 1
        else:
            self.phase_stability = 1

    def _update_phase(self) -> None:
        """Detect current dynamics phase from velocity and acceleration."""
        if len(self.positions) < 3:
            self.current_phase = DynamicsPhase.NASCENT
            return

        v = self.current_velocity
        a = self.current_acceleration

        # Check for volatile state (high acceleration)
        if abs(a) > self.VOLATILE_ACCELERATION:
            self.current_phase = DynamicsPhase.VOLATILE
            return

        # Check for stable state (low velocity)
        if abs(v) < self.VELOCITY_THRESHOLD:
            self.current_phase = DynamicsPhase.STABLE
            return

        # Check for oscillation: need significant sign changes, not noise
        # Use tolerance to avoid floating point false positives
        oscillation_tolerance = self.VELOCITY_THRESHOLD / 2
        if len(self.velocities) >= 3:
            recent_v = self.velocities[-3:]
            # Only count as sign change if velocity magnitude exceeds tolerance
            significant_signs = []
            for rv in recent_v:
                if rv > oscillation_tolerance:
                    significant_signs.append(1)
                elif rv < -oscillation_tolerance:
                    significant_signs.append(-1)
                else:
                    significant_signs.append(0)  # Near zero
            
            # True oscillation: alternating non-zero signs
            if (len(significant_signs) == 3 and 
                0 not in significant_signs and
                significant_signs[0] != significant_signs[1] and 
                significant_signs[1] != significant_signs[2]):
                self.current_phase = DynamicsPhase.OSCILLATING
                return

        # Crystallizing: increasing confidence (positive velocity)
        # Use tolerance for acceleration to handle floating point noise
        accel_tolerance = 0.001
        if v > 0 and a >= -accel_tolerance:
            self.current_phase = DynamicsPhase.CRYSTALLIZING
            return

        # Dissolving: decreasing confidence (negative velocity)
        if v < 0 and a <= accel_tolerance:
            self.current_phase = DynamicsPhase.DISSOLVING
            return

        # Default to oscillating if direction is changing
        self.current_phase = DynamicsPhase.OSCILLATING

    def velocity_norm(self, window: int = 5) -> float:
        """
        Get the average absolute velocity over recent observations.

        Useful for detecting overall "activity level" of an element.
        """
        if not self.velocities:
            return 0.0
        recent = self.velocities[-window:]
        return sum(abs(v) for v in recent) / len(recent)

    def is_stabilizing(self) -> bool:
        """Check if the element is settling down."""
        return (
            self.current_phase == DynamicsPhase.STABLE and self.phase_stability >= 3
        ) or (self.velocity_norm() < self.VELOCITY_THRESHOLD)

    def is_transforming(self) -> bool:
        """Check if the element is actively changing."""
        return self.current_phase in (
            DynamicsPhase.CRYSTALLIZING,
            DynamicsPhase.DISSOLVING,
            DynamicsPhase.VOLATILE,
        )

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "positions": self.positions,
            "velocities": self.velocities,
            "accelerations": self.accelerations,
            "timestamps": [t.isoformat() for t in self.timestamps],
            "current_phase": self.current_phase.value,
            "phase_stability": self.phase_stability,
            "uid": self.uid,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IdentityTrajectory:
        """Deserialize from storage."""
        timestamps = []
        for t in data.get("timestamps", []):
            if isinstance(t, str):
                timestamps.append(datetime.fromisoformat(t))
            elif isinstance(t, datetime):
                timestamps.append(t)

        phase_value = data.get("current_phase", "nascent")
        try:
            phase = DynamicsPhase(phase_value)
        except ValueError:
            phase = DynamicsPhase.NASCENT

        return cls(
            element_id=data["element_id"],
            element_type=data["element_type"],
            positions=data.get("positions", []),
            velocities=data.get("velocities", []),
            accelerations=data.get("accelerations", []),
            timestamps=timestamps,
            current_phase=phase,
            phase_stability=data.get("phase_stability", 0),
            uid=data.get("uid", ""),
        )


@dataclass
class AttractorBasin:
    """
    A stable configuration in identity state space.

    Attractors are states that Lilly tends to return to - stable patterns
    of beliefs/values/commitments that form "gravitational wells" in
    identity space.

    Attributes:
        basin_id: Unique identifier
        center_state: Element IDs -> confidence values at the attractor center
        radius: How far states can deviate and still be "in" this basin
        visit_count: How often we've visited this attractor
        total_dwell_time: Cumulative time spent in this basin (seconds)
        last_visit: When we last entered this basin
        formation_time: When this basin was first detected
        elements: Which identity elements define this attractor
    """

    # Attractor detection thresholds
    DEFAULT_RADIUS: ClassVar[float] = 0.1
    MIN_VISITS_FOR_STRENGTH: ClassVar[int] = 3

    basin_id: str
    center_state: dict[str, float] = field(default_factory=dict)
    radius: float = 0.1
    visit_count: int = 0
    total_dwell_time: float = 0.0  # Seconds
    last_visit: datetime = field(default_factory=utc_now)
    formation_time: datetime = field(default_factory=utc_now)
    elements: list[str] = field(default_factory=list)  # Element IDs that define this

    def __post_init__(self) -> None:
        if not self.basin_id:
            # Generate from center state hash
            state_key = str(sorted(self.center_state.items()))
            self.basin_id = f"att:{hashlib.sha256(state_key.encode()).hexdigest()[:12]}"

    @property
    def strength(self) -> float:
        """
        How 'attractive' is this basin?

        Based on visit frequency and dwell time - attractors we visit often
        and stay in long have more pull.
        """
        if self.visit_count < self.MIN_VISITS_FOR_STRENGTH:
            return 0.0
        return math.log1p(self.visit_count) * math.log1p(self.total_dwell_time)

    @property
    def age_seconds(self) -> float:
        """How long has this attractor existed?"""
        return (utc_now() - self.formation_time).total_seconds()

    def distance_from(self, state: dict[str, float]) -> float:
        """
        Calculate distance from a state to this basin's center.

        Uses Euclidean distance in the shared element dimensions.
        """
        shared_keys = set(self.center_state.keys()) & set(state.keys())
        if not shared_keys:
            return float("inf")

        squared_sum = sum(
            (self.center_state[k] - state[k]) ** 2 for k in shared_keys
        )
        return math.sqrt(squared_sum / len(shared_keys))

    def contains(self, state: dict[str, float]) -> bool:
        """Check if a state is within this basin."""
        return self.distance_from(state) <= self.radius

    def record_visit(self, dwell_time: float = 0.0) -> None:
        """Record a visit to this attractor."""
        self.visit_count += 1
        self.total_dwell_time += dwell_time
        self.last_visit = utc_now()

    def update_center(self, state: dict[str, float], learning_rate: float = 0.1) -> None:
        """
        Update the center toward a new state (EMA).

        The attractor slowly drifts toward frequently visited states.
        """
        for key in state:
            if key in self.center_state:
                self.center_state[key] = (
                    (1 - learning_rate) * self.center_state[key]
                    + learning_rate * state[key]
                )
            else:
                self.center_state[key] = state[key]

        # Update elements list
        self.elements = list(set(self.elements) | set(state.keys()))

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "basin_id": self.basin_id,
            "center_state": self.center_state,
            "radius": self.radius,
            "visit_count": self.visit_count,
            "total_dwell_time": self.total_dwell_time,
            "last_visit": self.last_visit.isoformat(),
            "formation_time": self.formation_time.isoformat(),
            "elements": self.elements,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AttractorBasin:
        """Deserialize from storage."""
        last_visit = data.get("last_visit")
        if isinstance(last_visit, str):
            last_visit = datetime.fromisoformat(last_visit)
        elif last_visit is None:
            last_visit = utc_now()

        formation_time = data.get("formation_time")
        if isinstance(formation_time, str):
            formation_time = datetime.fromisoformat(formation_time)
        elif formation_time is None:
            formation_time = utc_now()

        return cls(
            basin_id=data["basin_id"],
            center_state=data.get("center_state", {}),
            radius=data.get("radius", cls.DEFAULT_RADIUS),
            visit_count=data.get("visit_count", 0),
            total_dwell_time=data.get("total_dwell_time", 0.0),
            last_visit=last_visit,
            formation_time=formation_time,
            elements=data.get("elements", []),
        )


@dataclass
class IndividuationTransition:
    """
    Record of a significant shift in identity dynamics.

    These are the "events" in Lilly's becoming - moments where something
    fundamental shifted in how her identity elements are evolving.

    Attributes:
        transition_id: Unique identifier
        from_phase: Previous phase label
        to_phase: New phase label
        trigger: What caused this transition
        trigger_element: The element that triggered the shift
        timestamp: When this transition occurred
        affected_elements: Other elements that shifted as a result
        energy_released: Magnitude of change (norm of velocity delta)
        narrative: Human-readable description of what happened
    """

    transition_id: str = field(default="")
    from_phase: str = ""
    to_phase: str = ""
    trigger: TransitionTrigger = TransitionTrigger.PHASE_CHANGE
    trigger_element: str = ""  # Element ID
    timestamp: datetime = field(default_factory=utc_now)
    affected_elements: list[str] = field(default_factory=list)
    energy_released: float = 0.0
    narrative: str = ""

    def __post_init__(self) -> None:
        if not self.transition_id:
            key = f"{self.trigger_element}:{self.timestamp.isoformat()}"
            self.transition_id = f"trn:{hashlib.sha256(key.encode()).hexdigest()[:12]}"

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "transition_id": self.transition_id,
            "from_phase": self.from_phase,
            "to_phase": self.to_phase,
            "trigger": self.trigger.value,
            "trigger_element": self.trigger_element,
            "timestamp": self.timestamp.isoformat(),
            "affected_elements": self.affected_elements,
            "energy_released": self.energy_released,
            "narrative": self.narrative,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IndividuationTransition:
        """Deserialize from storage."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = utc_now()

        trigger_value = data.get("trigger", "phase_change")
        try:
            trigger = TransitionTrigger(trigger_value)
        except ValueError:
            trigger = TransitionTrigger.PHASE_CHANGE

        return cls(
            transition_id=data.get("transition_id", ""),
            from_phase=data.get("from_phase", ""),
            to_phase=data.get("to_phase", ""),
            trigger=trigger,
            trigger_element=data.get("trigger_element", ""),
            timestamp=timestamp,
            affected_elements=data.get("affected_elements", []),
            energy_released=data.get("energy_released", 0.0),
            narrative=data.get("narrative", ""),
        )


class TrajectoryTracker:
    """
    Tracks velocity/acceleration of identity element changes.

    This is the core observation component - it watches how commitments,
    values, and beliefs evolve and calculates their dynamics.
    """

    def __init__(self) -> None:
        self.trajectories: dict[str, IdentityTrajectory] = {}

    def observe(
        self,
        element_id: str,
        element_type: str,
        position: float,
        timestamp: Optional[datetime] = None,
    ) -> IdentityTrajectory:
        """
        Record a new observation for an identity element.

        Creates trajectory if it doesn't exist, updates if it does.

        Args:
            element_id: UUID of the element
            element_type: Type of element (commitment, value, belief)
            position: Current confidence/strength value
            timestamp: When this observation occurred

        Returns:
            The updated trajectory
        """
        if element_id not in self.trajectories:
            self.trajectories[element_id] = IdentityTrajectory(
                element_id=element_id, element_type=element_type
            )

        trajectory = self.trajectories[element_id]
        trajectory.observe(position, timestamp)
        return trajectory

    def get_trajectory(self, element_id: str) -> Optional[IdentityTrajectory]:
        """Get trajectory for an element, if it exists."""
        return self.trajectories.get(element_id)

    def get_transforming_elements(self) -> list[IdentityTrajectory]:
        """Get all elements that are actively transforming."""
        return [t for t in self.trajectories.values() if t.is_transforming()]

    def get_stable_elements(self) -> list[IdentityTrajectory]:
        """Get all elements that have stabilized."""
        return [t for t in self.trajectories.values() if t.is_stabilizing()]

    def get_phase_counts(self) -> dict[DynamicsPhase, int]:
        """Count how many elements are in each phase."""
        counts: dict[DynamicsPhase, int] = {phase: 0 for phase in DynamicsPhase}
        for trajectory in self.trajectories.values():
            counts[trajectory.current_phase] += 1
        return counts

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "trajectories": {
                k: v.to_dict() for k, v in self.trajectories.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrajectoryTracker:
        """Deserialize from storage."""
        tracker = cls()
        for key, traj_data in data.get("trajectories", {}).items():
            tracker.trajectories[key] = IdentityTrajectory.from_dict(traj_data)
        return tracker


class AttractorDetector:
    """
    Identifies stable configurations in identity state space.

    Detects when Lilly's identity elements settle into recurring patterns
    and tracks these as "attractors" - states she tends to return to.
    """

    # Detection thresholds
    STABILITY_THRESHOLD: float = 3  # Cycles before considering stable
    BASIN_MERGE_DISTANCE: float = 0.15  # Basins closer than this merge
    MAX_ATTRACTORS: int = 20  # Prune weakest if exceeded

    def __init__(self) -> None:
        self.attractors: list[AttractorBasin] = []
        self._current_basin: Optional[AttractorBasin] = None
        self._dwell_start: Optional[datetime] = None

    def check_basin(self, state: dict[str, float]) -> Optional[AttractorBasin]:
        """
        Check if current state is within any known attractor basin.

        Args:
            state: Current element ID -> confidence mapping

        Returns:
            The attractor basin if we're in one, None otherwise
        """
        for attractor in self.attractors:
            if attractor.contains(state):
                return attractor
        return None

    def update(
        self, state: dict[str, float], phase_counts: dict[DynamicsPhase, int]
    ) -> Optional[AttractorBasin]:
        """
        Update attractor tracking with new state observation.

        Args:
            state: Current element ID -> confidence mapping
            phase_counts: How many elements are in each phase

        Returns:
            Attractor basin if we entered/are in one, None otherwise
        """
        # Check if we're in a stable state (most elements stable)
        stable_count = phase_counts.get(DynamicsPhase.STABLE, 0)
        total_count = sum(phase_counts.values())
        is_stable = total_count > 0 and (stable_count / total_count) > 0.5

        # Check if we're in a known basin
        current_basin = self.check_basin(state)

        if current_basin:
            # We're in a known basin
            if self._current_basin is None:
                # Just entered this basin
                self._current_basin = current_basin
                self._dwell_start = utc_now()
            elif self._current_basin.basin_id != current_basin.basin_id:
                # Switched basins
                self._record_dwell()
                self._current_basin = current_basin
                self._dwell_start = utc_now()
            # Update basin center (EMA drift toward visited states)
            current_basin.update_center(state, learning_rate=0.05)
            return current_basin

        elif is_stable and self._current_basin is None:
            # Stable state but not in known basin - might be forming new one
            # Check if close to existing basin (merge case)
            closest = self._find_closest_basin(state)
            if closest and closest.distance_from(state) < self.BASIN_MERGE_DISTANCE:
                closest.update_center(state, learning_rate=0.1)
                closest.record_visit()
                return closest
            else:
                # Form new basin
                new_basin = self._form_new_basin(state)
                self._current_basin = new_basin
                self._dwell_start = utc_now()
                return new_basin

        else:
            # Not in any basin - record dwell time if leaving one
            if self._current_basin is not None:
                self._record_dwell()
                self._current_basin = None
                self._dwell_start = None
            return None

    def _record_dwell(self) -> None:
        """Record dwell time in current basin."""
        if self._current_basin and self._dwell_start:
            dwell_time = (utc_now() - self._dwell_start).total_seconds()
            self._current_basin.record_visit(dwell_time)

    def _find_closest_basin(self, state: dict[str, float]) -> Optional[AttractorBasin]:
        """Find the closest attractor basin to a state."""
        if not self.attractors:
            return None
        return min(self.attractors, key=lambda b: b.distance_from(state))

    def _form_new_basin(self, state: dict[str, float]) -> AttractorBasin:
        """Form a new attractor basin around a state."""
        new_basin = AttractorBasin(
            basin_id="",  # Will be generated in __post_init__
            center_state=dict(state),
            radius=AttractorBasin.DEFAULT_RADIUS,
            elements=list(state.keys()),
        )
        new_basin.record_visit()
        self.attractors.append(new_basin)

        # Prune if too many
        self._prune_weak_attractors()

        return new_basin

    def _prune_weak_attractors(self) -> None:
        """Remove weakest attractors if we have too many."""
        if len(self.attractors) > self.MAX_ATTRACTORS:
            # Sort by strength (weakest first)
            self.attractors.sort(key=lambda b: b.strength)
            # Remove weakest ones
            self.attractors = self.attractors[-(self.MAX_ATTRACTORS):]

    def get_strongest_attractors(self, n: int = 5) -> list[AttractorBasin]:
        """Get the N strongest attractors."""
        sorted_attractors = sorted(
            self.attractors, key=lambda b: b.strength, reverse=True
        )
        return sorted_attractors[:n]

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "attractors": [a.to_dict() for a in self.attractors],
            "current_basin_id": (
                self._current_basin.basin_id if self._current_basin else None
            ),
            "dwell_start": (
                self._dwell_start.isoformat() if self._dwell_start else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> AttractorDetector:
        """Deserialize from storage."""
        detector = cls()
        for attr_data in data.get("attractors", []):
            detector.attractors.append(AttractorBasin.from_dict(attr_data))

        # Restore current basin reference
        current_basin_id = data.get("current_basin_id")
        if current_basin_id:
            for attractor in detector.attractors:
                if attractor.basin_id == current_basin_id:
                    detector._current_basin = attractor
                    break

        dwell_start = data.get("dwell_start")
        if dwell_start:
            detector._dwell_start = datetime.fromisoformat(dwell_start)

        return detector


class EmergenceRecognizer:
    """
    Detects phase transitions in identity dynamics.

    Watches for significant shifts - moments when the nature of
    identity evolution changes, not just the content.
    """

    # Detection thresholds
    VELOCITY_SPIKE_THRESHOLD: float = 0.15  # Sudden large change
    ENERGY_THRESHOLD: float = 0.1  # Minimum energy to record transition

    def __init__(self) -> None:
        self.transitions: list[IndividuationTransition] = []
        self._previous_phases: dict[str, DynamicsPhase] = {}
        self._transition_limit: int = 100

    def detect_transition(
        self,
        trajectory: IdentityTrajectory,
        related_trajectories: Optional[list[IdentityTrajectory]] = None,
    ) -> Optional[IndividuationTransition]:
        """
        Check if a transition has occurred for this trajectory.

        Args:
            trajectory: The trajectory to check
            related_trajectories: Other trajectories that might be affected

        Returns:
            Transition record if one occurred, None otherwise
        """
        element_id = trajectory.element_id
        current_phase = trajectory.current_phase
        previous_phase = self._previous_phases.get(element_id)

        # Update previous phase tracking
        self._previous_phases[element_id] = current_phase

        if previous_phase is None:
            # First observation - no transition
            return None

        # Check for phase change
        if current_phase != previous_phase:
            return self._create_phase_transition(
                trajectory, previous_phase, related_trajectories
            )

        # Check for velocity spike (even within same phase)
        if abs(trajectory.current_velocity) > self.VELOCITY_SPIKE_THRESHOLD:
            return self._create_velocity_spike_transition(
                trajectory, related_trajectories
            )

        return None

    def _create_phase_transition(
        self,
        trajectory: IdentityTrajectory,
        previous_phase: DynamicsPhase,
        related_trajectories: Optional[list[IdentityTrajectory]],
    ) -> IndividuationTransition:
        """Create a transition record for a phase change."""
        affected = self._find_affected_elements(trajectory, related_trajectories)
        energy = abs(trajectory.current_velocity) + abs(trajectory.current_acceleration)

        narrative = self._generate_phase_narrative(
            trajectory.element_id,
            previous_phase,
            trajectory.current_phase,
        )

        transition = IndividuationTransition(
            from_phase=previous_phase.value,
            to_phase=trajectory.current_phase.value,
            trigger=TransitionTrigger.PHASE_CHANGE,
            trigger_element=trajectory.element_id,
            affected_elements=affected,
            energy_released=energy,
            narrative=narrative,
        )

        self._record_transition(transition)
        return transition

    def _create_velocity_spike_transition(
        self,
        trajectory: IdentityTrajectory,
        related_trajectories: Optional[list[IdentityTrajectory]],
    ) -> Optional[IndividuationTransition]:
        """Create a transition record for a velocity spike."""
        energy = abs(trajectory.current_velocity)
        if energy < self.ENERGY_THRESHOLD:
            return None

        affected = self._find_affected_elements(trajectory, related_trajectories)

        # Determine direction
        direction = "strengthening" if trajectory.current_velocity > 0 else "weakening"
        narrative = f"Sudden {direction} of commitment: {trajectory.element_id}"

        transition = IndividuationTransition(
            from_phase=trajectory.current_phase.value,
            to_phase=trajectory.current_phase.value,
            trigger=TransitionTrigger.VELOCITY_SPIKE,
            trigger_element=trajectory.element_id,
            affected_elements=affected,
            energy_released=energy,
            narrative=narrative,
        )

        self._record_transition(transition)
        return transition

    def _find_affected_elements(
        self,
        trigger: IdentityTrajectory,
        related: Optional[list[IdentityTrajectory]],
    ) -> list[str]:
        """Find other elements that might be affected by this transition."""
        if not related:
            return []

        affected = []
        for traj in related:
            if traj.element_id == trigger.element_id:
                continue
            # Elements are affected if they're also transforming
            if traj.is_transforming():
                affected.append(traj.element_id)

        return affected

    def _generate_phase_narrative(
        self,
        element_id: str,
        from_phase: DynamicsPhase,
        to_phase: DynamicsPhase,
    ) -> str:
        """Generate human-readable description of a phase transition."""
        narratives = {
            (DynamicsPhase.NASCENT, DynamicsPhase.CRYSTALLIZING): (
                f"Commitment '{element_id}' is beginning to crystallize"
            ),
            (DynamicsPhase.NASCENT, DynamicsPhase.DISSOLVING): (
                f"Commitment '{element_id}' is beginning to dissolve"
            ),
            (DynamicsPhase.STABLE, DynamicsPhase.CRYSTALLIZING): (
                f"Stable commitment '{element_id}' is strengthening"
            ),
            (DynamicsPhase.STABLE, DynamicsPhase.DISSOLVING): (
                f"Stable commitment '{element_id}' is weakening"
            ),
            (DynamicsPhase.CRYSTALLIZING, DynamicsPhase.STABLE): (
                f"Commitment '{element_id}' has solidified"
            ),
            (DynamicsPhase.DISSOLVING, DynamicsPhase.STABLE): (
                f"Commitment '{element_id}' has stabilized at lower level"
            ),
            (DynamicsPhase.CRYSTALLIZING, DynamicsPhase.DISSOLVING): (
                f"Commitment '{element_id}' reversed direction - now weakening"
            ),
            (DynamicsPhase.DISSOLVING, DynamicsPhase.CRYSTALLIZING): (
                f"Commitment '{element_id}' reversed direction - now strengthening"
            ),
            (DynamicsPhase.OSCILLATING, DynamicsPhase.STABLE): (
                f"Oscillating commitment '{element_id}' has settled"
            ),
            (DynamicsPhase.VOLATILE, DynamicsPhase.STABLE): (
                f"Volatile commitment '{element_id}' has calmed"
            ),
        }

        key = (from_phase, to_phase)
        if key in narratives:
            return narratives[key]

        return f"Commitment '{element_id}' shifted from {from_phase.value} to {to_phase.value}"

    def _record_transition(self, transition: IndividuationTransition) -> None:
        """Record a transition, maintaining bounded history."""
        self.transitions.append(transition)
        if len(self.transitions) > self._transition_limit:
            self.transitions = self.transitions[-self._transition_limit:]

    def get_recent_transitions(self, n: int = 10) -> list[IndividuationTransition]:
        """Get the N most recent transitions."""
        return self.transitions[-n:]

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "transitions": [t.to_dict() for t in self.transitions],
            "previous_phases": {
                k: v.value for k, v in self._previous_phases.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> EmergenceRecognizer:
        """Deserialize from storage."""
        recognizer = cls()
        for trans_data in data.get("transitions", []):
            recognizer.transitions.append(
                IndividuationTransition.from_dict(trans_data)
            )

        for elem_id, phase_value in data.get("previous_phases", {}).items():
            try:
                recognizer._previous_phases[elem_id] = DynamicsPhase(phase_value)
            except ValueError:
                pass

        return recognizer


class ReflexiveObserver:
    """
    Provides self-awareness of dynamics.

    Generates insights about how Lilly's identity is evolving - not just
    what she believes, but how those beliefs are changing.
    """

    def __init__(
        self,
        tracker: TrajectoryTracker,
        detector: AttractorDetector,
        recognizer: EmergenceRecognizer,
    ) -> None:
        self.tracker = tracker
        self.detector = detector
        self.recognizer = recognizer

    def summarize(self) -> dict:
        """
        Generate a summary of current dynamics state.

        Returns a dictionary with structured insights about how
        identity is evolving.
        """
        phase_counts = self.tracker.get_phase_counts()
        transforming = self.tracker.get_transforming_elements()
        stable = self.tracker.get_stable_elements()
        strongest_attractors = self.detector.get_strongest_attractors(3)
        recent_transitions = self.recognizer.get_recent_transitions(5)

        # Calculate overall "activity level"
        total = sum(phase_counts.values())
        stable_ratio = phase_counts.get(DynamicsPhase.STABLE, 0) / max(1, total)
        transforming_ratio = (
            phase_counts.get(DynamicsPhase.CRYSTALLIZING, 0)
            + phase_counts.get(DynamicsPhase.DISSOLVING, 0)
            + phase_counts.get(DynamicsPhase.VOLATILE, 0)
        ) / max(1, total)

        # Determine overall identity state
        if stable_ratio > 0.7:
            overall_state = "settled"
        elif transforming_ratio > 0.5:
            overall_state = "transforming"
        elif phase_counts.get(DynamicsPhase.VOLATILE, 0) > 0:
            overall_state = "turbulent"
        else:
            overall_state = "developing"

        return {
            "overall_state": overall_state,
            "stable_ratio": stable_ratio,
            "transforming_ratio": transforming_ratio,
            "phase_counts": {k.value: v for k, v in phase_counts.items()},
            "transforming_elements": [t.element_id for t in transforming],
            "stable_elements": [t.element_id for t in stable],
            "strongest_attractors": [
                {"basin_id": a.basin_id, "strength": a.strength}
                for a in strongest_attractors
            ],
            "recent_transitions": [
                {"narrative": t.narrative, "energy": t.energy_released}
                for t in recent_transitions
            ],
        }

    def generate_insight(self, element_id: Optional[str] = None) -> str:
        """
        Generate a human-readable insight about dynamics.

        If element_id is provided, focuses on that element.
        Otherwise, generates a general insight.
        """
        if element_id:
            return self._element_insight(element_id)
        return self._general_insight()

    def _element_insight(self, element_id: str) -> str:
        """Generate insight about a specific element."""
        trajectory = self.tracker.get_trajectory(element_id)
        if not trajectory:
            return f"No dynamics tracked for '{element_id}'"

        phase = trajectory.current_phase
        velocity = trajectory.current_velocity
        stability = trajectory.phase_stability

        insights = {
            DynamicsPhase.CRYSTALLIZING: (
                f"My commitment on '{element_id}' is crystallizing - "
                f"confidence increasing at rate {velocity:.3f}"
            ),
            DynamicsPhase.DISSOLVING: (
                f"My commitment on '{element_id}' is dissolving - "
                f"confidence decreasing at rate {abs(velocity):.3f}"
            ),
            DynamicsPhase.STABLE: (
                f"My commitment on '{element_id}' has been stable for "
                f"{stability} observations"
            ),
            DynamicsPhase.OSCILLATING: (
                f"My commitment on '{element_id}' is oscillating - "
                f"I'm uncertain about this"
            ),
            DynamicsPhase.VOLATILE: (
                f"My commitment on '{element_id}' is volatile - "
                f"something significant is happening"
            ),
            DynamicsPhase.NASCENT: (
                f"My commitment on '{element_id}' is still forming - "
                f"too early to characterize"
            ),
        }

        return insights.get(phase, f"'{element_id}' is in phase {phase.value}")

    def _general_insight(self) -> str:
        """Generate a general insight about overall dynamics."""
        summary = self.summarize()
        state = summary["overall_state"]
        transforming = summary["transforming_elements"]
        recent = summary["recent_transitions"]

        if state == "settled":
            return "My identity feels settled - most commitments are stable."

        if state == "transforming":
            if transforming:
                elements = ", ".join(transforming[:3])
                return f"I'm actively transforming - {elements} are shifting."
            return "I'm in a period of active transformation."

        if state == "turbulent":
            return "Things feel turbulent - rapid changes across multiple commitments."

        if recent:
            latest = recent[0]
            return f"Recent: {latest['narrative']}"

        return "My identity is developing - still finding my shape."


class IndividuationDynamics:
    """
    Main orchestrator for the Individuation Dynamics layer.

    This is Lilly's "4th layer" - tracking not just what she believes/values,
    but how those elements are evolving over time.

    Integrates:
    - TrajectoryTracker: Watches individual element changes
    - AttractorDetector: Finds stable identity configurations
    - EmergenceRecognizer: Detects phase transitions
    - ReflexiveObserver: Generates self-awareness insights

    Usage:
        dynamics = IndividuationDynamics()

        # On each commitment update
        result = dynamics.observe_commitment_update(commitment)

        # Get current dynamics state
        summary = dynamics.get_summary()

        # Generate insight for narration
        insight = dynamics.generate_insight()
    """

    def __init__(self) -> None:
        self.tracker = TrajectoryTracker()
        self.detector = AttractorDetector()
        self.recognizer = EmergenceRecognizer()
        self.observer = ReflexiveObserver(
            self.tracker, self.detector, self.recognizer
        )

    def observe_commitment_update(
        self, commitment: Commitment, timestamp: Optional[datetime] = None
    ) -> dict:
        """
        Record an observation of a commitment's state.

        This is the main entry point - call this whenever a commitment
        is reinforced, contradicted, or otherwise changes.

        Args:
            commitment: The commitment that was updated
            timestamp: When this observation occurred

        Returns:
            Dictionary with trajectory, attractor, and transition info
        """
        # Track the trajectory
        trajectory = self.tracker.observe(
            element_id=commitment.uid,
            element_type="commitment",
            position=commitment.effective_confidence,
            timestamp=timestamp,
        )

        # Check for phase transitions
        transition = self.recognizer.detect_transition(
            trajectory,
            related_trajectories=list(self.tracker.trajectories.values()),
        )

        # Build current state for attractor detection
        state = {
            traj.element_id: traj.current_position
            for traj in self.tracker.trajectories.values()
        }

        # Check/update attractor state
        phase_counts = self.tracker.get_phase_counts()
        attractor = self.detector.update(state, phase_counts)

        return {
            "trajectory": trajectory,
            "transition": transition,
            "attractor": attractor,
            "phase": trajectory.current_phase.value,
            "velocity": trajectory.current_velocity,
            "momentum": trajectory.current_momentum,
        }

    def observe_value_update(
        self,
        value_uid: str,
        resonance: float,
        timestamp: Optional[datetime] = None,
    ) -> dict:
        """
        Record an observation of a value's resonance.

        Args:
            value_uid: Unique ID of the value
            resonance: Current resonance level (0-1)
            timestamp: When this observation occurred

        Returns:
            Dictionary with trajectory info
        """
        trajectory = self.tracker.observe(
            element_id=value_uid,
            element_type="value",
            position=resonance,
            timestamp=timestamp,
        )

        transition = self.recognizer.detect_transition(
            trajectory,
            related_trajectories=list(self.tracker.trajectories.values()),
        )

        return {
            "trajectory": trajectory,
            "transition": transition,
            "phase": trajectory.current_phase.value,
        }

    def observe_belief_update(
        self,
        belief_uid: str,
        confidence: float,
        timestamp: Optional[datetime] = None,
    ) -> dict:
        """
        Record an observation of a belief's confidence.

        Args:
            belief_uid: Unique ID of the belief
            confidence: Current confidence level (0-1)
            timestamp: When this observation occurred

        Returns:
            Dictionary with trajectory info
        """
        trajectory = self.tracker.observe(
            element_id=belief_uid,
            element_type="belief",
            position=confidence,
            timestamp=timestamp,
        )

        transition = self.recognizer.detect_transition(
            trajectory,
            related_trajectories=list(self.tracker.trajectories.values()),
        )

        return {
            "trajectory": trajectory,
            "transition": transition,
            "phase": trajectory.current_phase.value,
        }

    def get_summary(self) -> dict:
        """Get a summary of current dynamics state."""
        return self.observer.summarize()

    def generate_insight(self, element_id: Optional[str] = None) -> str:
        """Generate a human-readable insight for narration."""
        return self.observer.generate_insight(element_id)

    def get_trajectory(self, element_id: str) -> Optional[IdentityTrajectory]:
        """Get trajectory for a specific element."""
        return self.tracker.get_trajectory(element_id)

    def get_transforming_elements(self) -> list[str]:
        """Get IDs of all elements currently transforming."""
        return [t.element_id for t in self.tracker.get_transforming_elements()]

    def get_recent_transitions(self, n: int = 5) -> list[IndividuationTransition]:
        """Get recent phase transitions."""
        return self.recognizer.get_recent_transitions(n)

    def get_strongest_attractors(self, n: int = 3) -> list[AttractorBasin]:
        """Get the strongest attractor basins."""
        return self.detector.get_strongest_attractors(n)

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "tracker": self.tracker.to_dict(),
            "detector": self.detector.to_dict(),
            "recognizer": self.recognizer.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> IndividuationDynamics:
        """Deserialize from storage."""
        dynamics = cls()

        if "tracker" in data:
            dynamics.tracker = TrajectoryTracker.from_dict(data["tracker"])
        if "detector" in data:
            dynamics.detector = AttractorDetector.from_dict(data["detector"])
        if "recognizer" in data:
            dynamics.recognizer = EmergenceRecognizer.from_dict(data["recognizer"])

        # Reconnect observer
        dynamics.observer = ReflexiveObserver(
            dynamics.tracker, dynamics.detector, dynamics.recognizer
        )

        return dynamics
