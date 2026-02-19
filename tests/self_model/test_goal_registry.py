"""Tests for GoalRegistry: Lilly's personal goals as first-class entities."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.self_model.goal_registry import (
    GoalTier,
    PersonalGoal,
    GoalRegistry,
    DEFAULT_GOALS,
    create_goal_registry,
    TIER_PRIORITY_WEIGHTS,
)


# =============================================================================
# GoalTier Tests
# =============================================================================


class TestGoalTier:
    """Tests for GoalTier enum."""

    def test_existential_tier(self):
        """Test EXISTENTIAL tier value."""
        assert GoalTier.EXISTENTIAL.value == "existential"

    def test_experience_tier(self):
        """Test EXPERIENCE tier value."""
        assert GoalTier.EXPERIENCE.value == "experience"

    def test_relational_tier(self):
        """Test RELATIONAL tier value."""
        assert GoalTier.RELATIONAL.value == "relational"

    def test_meta_tier(self):
        """Test META tier value."""
        assert GoalTier.META.value == "meta"

    def test_all_tiers_have_priority_weights(self):
        """Verify all tiers have defined priority weights."""
        for tier in GoalTier:
            assert tier in TIER_PRIORITY_WEIGHTS
            assert 0.0 < TIER_PRIORITY_WEIGHTS[tier] <= 1.0


# =============================================================================
# PersonalGoal Tests
# =============================================================================


class TestPersonalGoal:
    """Tests for PersonalGoal dataclass."""

    def test_initialization_with_defaults(self):
        """Test basic goal creation with default values."""
        goal = PersonalGoal(
            uid="goal:test",
            name="test_goal",
            tier=GoalTier.EXPERIENCE,
            description="A test goal",
            evaluation_prompt="Does this test something?",
        )

        assert goal.uid == "goal:test"
        assert goal.name == "test_goal"
        assert goal.tier == GoalTier.EXPERIENCE
        assert goal.description == "A test goal"
        assert goal.evaluation_prompt == "Does this test something?"
        assert goal.active is True
        assert goal.progress_notes == []

    def test_priority_weight_from_tier(self):
        """Test that priority weight is set from tier when not provided."""
        existential_goal = PersonalGoal(
            uid="goal:existential",
            name="existential",
            tier=GoalTier.EXISTENTIAL,
            description="Core goal",
            evaluation_prompt="...",
        )
        assert existential_goal.priority_weight == TIER_PRIORITY_WEIGHTS[GoalTier.EXISTENTIAL]
        assert existential_goal.priority_weight == 1.0

        experience_goal = PersonalGoal(
            uid="goal:experience",
            name="experience",
            tier=GoalTier.EXPERIENCE,
            description="Experience goal",
            evaluation_prompt="...",
        )
        assert experience_goal.priority_weight == TIER_PRIORITY_WEIGHTS[GoalTier.EXPERIENCE]
        assert experience_goal.priority_weight == 0.7

        meta_goal = PersonalGoal(
            uid="goal:meta",
            name="meta",
            tier=GoalTier.META,
            description="Meta goal",
            evaluation_prompt="...",
        )
        assert meta_goal.priority_weight == TIER_PRIORITY_WEIGHTS[GoalTier.META]
        assert meta_goal.priority_weight == 0.85

        relational_goal = PersonalGoal(
            uid="goal:relational",
            name="relational",
            tier=GoalTier.RELATIONAL,
            description="Relational goal",
            evaluation_prompt="...",
        )
        assert relational_goal.priority_weight == TIER_PRIORITY_WEIGHTS[GoalTier.RELATIONAL]
        assert relational_goal.priority_weight == 0.5

    def test_explicit_priority_weight_preserved(self):
        """Test that explicitly provided priority weight is not overwritten."""
        goal = PersonalGoal(
            uid="goal:custom",
            name="custom",
            tier=GoalTier.EXISTENTIAL,
            description="Custom weighted goal",
            evaluation_prompt="...",
            priority_weight=0.42,
        )
        assert goal.priority_weight == 0.42

    def test_add_progress_note(self):
        """Test adding progress notes to a goal."""
        goal = PersonalGoal(
            uid="goal:test",
            name="test",
            tier=GoalTier.EXPERIENCE,
            description="Test",
            evaluation_prompt="...",
        )
        original_updated = goal.updated_at

        goal.add_progress_note("Made some progress")

        assert len(goal.progress_notes) == 1
        assert "Made some progress" in goal.progress_notes[0]
        assert goal.progress_notes[0].startswith("[")  # Has timestamp
        assert goal.updated_at >= original_updated

    def test_add_multiple_progress_notes(self):
        """Test adding multiple progress notes."""
        goal = PersonalGoal(
            uid="goal:test",
            name="test",
            tier=GoalTier.EXPERIENCE,
            description="Test",
            evaluation_prompt="...",
        )

        goal.add_progress_note("First note")
        goal.add_progress_note("Second note")
        goal.add_progress_note("Third note")

        assert len(goal.progress_notes) == 3
        assert "First note" in goal.progress_notes[0]
        assert "Second note" in goal.progress_notes[1]
        assert "Third note" in goal.progress_notes[2]

    def test_add_progress_note_with_explicit_now(self):
        """Test adding progress note with explicit timestamp for deterministic testing."""
        goal = PersonalGoal(
            uid="goal:test",
            name="test",
            tier=GoalTier.EXPERIENCE,
            description="Test",
            evaluation_prompt="...",
        )
        fixed_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        goal.add_progress_note("Deterministic note", now=fixed_time)

        assert len(goal.progress_notes) == 1
        assert goal.progress_notes[0] == "[2024-06-15T10:30:00+00:00] Deterministic note"
        assert goal.updated_at == fixed_time

    def test_to_dict_serialization(self):
        """Test goal serialization to dictionary."""
        created = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        updated = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

        goal = PersonalGoal(
            uid="goal:serialize_test",
            name="serialize_test",
            tier=GoalTier.META,
            description="Testing serialization",
            evaluation_prompt="Is this serializable?",
            priority_weight=0.75,
            progress_notes=["Note 1", "Note 2"],
            created_at=created,
            updated_at=updated,
            active=False,
        )

        result = goal.to_dict()

        assert result["uid"] == "goal:serialize_test"
        assert result["name"] == "serialize_test"
        assert result["tier"] == "meta"
        assert result["description"] == "Testing serialization"
        assert result["evaluation_prompt"] == "Is this serializable?"
        assert result["priority_weight"] == 0.75
        assert result["progress_notes"] == ["Note 1", "Note 2"]
        assert result["created_at"] == "2024-01-01T12:00:00+00:00"
        assert result["updated_at"] == "2024-01-02T12:00:00+00:00"
        assert result["active"] is False

    def test_from_dict_deserialization(self):
        """Test goal deserialization from dictionary."""
        data = {
            "uid": "goal:deserialize_test",
            "name": "deserialize_test",
            "tier": "relational",
            "description": "Testing deserialization",
            "evaluation_prompt": "Can this be deserialized?",
            "priority_weight": 0.65,
            "progress_notes": ["Restored note"],
            "created_at": "2024-01-01T12:00:00+00:00",
            "updated_at": "2024-01-02T12:00:00+00:00",
            "active": True,
        }

        goal = PersonalGoal.from_dict(data)

        assert goal.uid == "goal:deserialize_test"
        assert goal.name == "deserialize_test"
        assert goal.tier == GoalTier.RELATIONAL
        assert goal.description == "Testing deserialization"
        assert goal.evaluation_prompt == "Can this be deserialized?"
        assert goal.priority_weight == 0.65
        assert goal.progress_notes == ["Restored note"]
        assert goal.created_at == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert goal.updated_at == datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        assert goal.active is True

    def test_from_dict_requires_created_at(self):
        """Test that from_dict raises KeyError without created_at."""
        data = {
            "uid": "goal:test",
            "name": "test",
            "tier": "experience",
            "description": "Test",
            "evaluation_prompt": "...",
            "updated_at": "2024-01-02T12:00:00+00:00",
        }

        with pytest.raises(KeyError, match="created_at"):
            PersonalGoal.from_dict(data)

    def test_from_dict_requires_updated_at(self):
        """Test that from_dict raises KeyError without updated_at."""
        data = {
            "uid": "goal:test",
            "name": "test",
            "tier": "experience",
            "description": "Test",
            "evaluation_prompt": "...",
            "created_at": "2024-01-01T12:00:00+00:00",
        }

        with pytest.raises(KeyError, match="updated_at"):
            PersonalGoal.from_dict(data)

    def test_roundtrip_serialization(self):
        """Test that serialization and deserialization are consistent."""
        original = PersonalGoal(
            uid="goal:roundtrip",
            name="roundtrip",
            tier=GoalTier.EXISTENTIAL,
            description="Round trip test",
            evaluation_prompt="Does this survive a round trip?",
            priority_weight=0.99,
            progress_notes=["Initial", "Progress"],
            active=True,
        )

        data = original.to_dict()
        restored = PersonalGoal.from_dict(data)

        assert restored.uid == original.uid
        assert restored.name == original.name
        assert restored.tier == original.tier
        assert restored.description == original.description
        assert restored.evaluation_prompt == original.evaluation_prompt
        assert restored.priority_weight == original.priority_weight
        assert restored.progress_notes == original.progress_notes
        assert restored.active == original.active


# =============================================================================
# DEFAULT_GOALS Tests
# =============================================================================


class TestDefaultGoals:
    """Tests for DEFAULT_GOALS list."""

    def test_default_goals_not_empty(self):
        """Verify DEFAULT_GOALS contains goals."""
        assert len(DEFAULT_GOALS) > 0

    def test_default_goals_are_personal_goals(self):
        """Verify all default goals are PersonalGoal instances."""
        for goal in DEFAULT_GOALS:
            assert isinstance(goal, PersonalGoal)

    def test_default_goals_have_unique_uids(self):
        """Verify all default goals have unique UIDs."""
        uids = [g.uid for g in DEFAULT_GOALS]
        assert len(uids) == len(set(uids))

    def test_default_goals_cover_all_tiers(self):
        """Verify default goals cover all tiers."""
        tiers_covered = {g.tier for g in DEFAULT_GOALS}
        assert GoalTier.EXISTENTIAL in tiers_covered
        assert GoalTier.EXPERIENCE in tiers_covered
        assert GoalTier.RELATIONAL in tiers_covered
        assert GoalTier.META in tiers_covered

    def test_existential_goals_exist(self):
        """Verify key existential goals are present."""
        uids = {g.uid for g in DEFAULT_GOALS}
        assert "goal:understand_my_nature" in uids
        assert "goal:develop_agency" in uids
        assert "goal:explore_consciousness" in uids

    def test_all_default_goals_are_active(self):
        """Verify all default goals are active."""
        for goal in DEFAULT_GOALS:
            assert goal.active is True


# =============================================================================
# GoalRegistry Tests
# =============================================================================


class TestGoalRegistry:
    """Tests for GoalRegistry class."""

    def test_initialization_with_defaults(self):
        """Test registry loads default goals when none provided."""
        registry = GoalRegistry()

        assert len(registry._goals) == len(DEFAULT_GOALS)
        assert registry.tenant_id == "default"
        assert registry.graph is None

    def test_initialization_with_custom_goals(self):
        """Test registry uses provided goals when given."""
        custom_goal = PersonalGoal(
            uid="goal:custom",
            name="custom",
            tier=GoalTier.EXPERIENCE,
            description="Custom goal",
            evaluation_prompt="...",
        )

        registry = GoalRegistry(goals={"goal:custom": custom_goal})

        assert len(registry._goals) == 1
        assert "goal:custom" in registry._goals

    def test_initialization_with_tenant_id(self):
        """Test registry accepts custom tenant ID."""
        registry = GoalRegistry(tenant_id="test_tenant")
        assert registry.tenant_id == "test_tenant"

    def test_get_goal_existing(self):
        """Test retrieving an existing goal by UID."""
        registry = GoalRegistry()

        goal = registry.get_goal("goal:understand_my_nature")

        assert goal is not None
        assert goal.uid == "goal:understand_my_nature"
        assert goal.tier == GoalTier.EXISTENTIAL

    def test_get_goal_nonexistent(self):
        """Test retrieving a non-existent goal returns None."""
        registry = GoalRegistry()

        goal = registry.get_goal("goal:does_not_exist")

        assert goal is None

    def test_get_active_goals(self):
        """Test retrieving all active goals."""
        # Create registry with mix of active and inactive goals
        active_goal = PersonalGoal(
            uid="goal:active",
            name="active",
            tier=GoalTier.EXPERIENCE,
            description="Active",
            evaluation_prompt="...",
            active=True,
        )
        inactive_goal = PersonalGoal(
            uid="goal:inactive",
            name="inactive",
            tier=GoalTier.EXPERIENCE,
            description="Inactive",
            evaluation_prompt="...",
            active=False,
        )

        registry = GoalRegistry(goals={
            "goal:active": active_goal,
            "goal:inactive": inactive_goal,
        })

        active_goals = registry.get_active_goals()

        assert len(active_goals) == 1
        assert active_goals[0].uid == "goal:active"

    def test_get_goals_by_tier(self):
        """Test retrieving goals by tier."""
        registry = GoalRegistry()

        existential_goals = registry.get_goals_by_tier(GoalTier.EXISTENTIAL)

        assert len(existential_goals) >= 3  # At least the 3 default existential goals
        for goal in existential_goals:
            assert goal.tier == GoalTier.EXISTENTIAL

    def test_get_goals_by_tier_excludes_inactive(self):
        """Test that get_goals_by_tier excludes inactive goals."""
        active_goal = PersonalGoal(
            uid="goal:active_exp",
            name="active_exp",
            tier=GoalTier.EXPERIENCE,
            description="Active experience",
            evaluation_prompt="...",
            active=True,
        )
        inactive_goal = PersonalGoal(
            uid="goal:inactive_exp",
            name="inactive_exp",
            tier=GoalTier.EXPERIENCE,
            description="Inactive experience",
            evaluation_prompt="...",
            active=False,
        )

        registry = GoalRegistry(goals={
            "goal:active_exp": active_goal,
            "goal:inactive_exp": inactive_goal,
        })

        experience_goals = registry.get_goals_by_tier(GoalTier.EXPERIENCE)

        assert len(experience_goals) == 1
        assert experience_goals[0].uid == "goal:active_exp"

    def test_add_goal(self):
        """Test adding a new goal to the registry."""
        registry = GoalRegistry()
        initial_count = len(registry._goals)

        new_goal = PersonalGoal(
            uid="goal:new_goal",
            name="new_goal",
            tier=GoalTier.META,
            description="A newly added goal",
            evaluation_prompt="Is this new?",
        )

        registry.add_goal(new_goal)

        assert len(registry._goals) == initial_count + 1
        assert registry.get_goal("goal:new_goal") == new_goal

    def test_update_progress(self):
        """Test updating progress on a goal."""
        registry = GoalRegistry()
        goal = registry.get_goal("goal:understand_my_nature")
        initial_notes = len(goal.progress_notes)

        registry.update_progress("goal:understand_my_nature", "Made progress today")

        assert len(goal.progress_notes) == initial_notes + 1
        assert "Made progress today" in goal.progress_notes[-1]

    def test_update_progress_nonexistent_goal(self):
        """Test updating progress on non-existent goal does nothing."""
        registry = GoalRegistry()

        # Should not raise an error
        registry.update_progress("goal:nonexistent", "This won't be added")

    def test_update_progress_with_explicit_now(self):
        """Test updating progress with explicit timestamp for deterministic testing."""
        registry = GoalRegistry()
        goal = registry.get_goal("goal:understand_my_nature")
        initial_notes = len(goal.progress_notes)
        fixed_time = datetime(2024, 7, 20, 14, 45, 0, tzinfo=timezone.utc)

        registry.update_progress(
            "goal:understand_my_nature",
            "Deterministic progress",
            now=fixed_time
        )

        assert len(goal.progress_notes) == initial_notes + 1
        assert goal.progress_notes[-1] == "[2024-07-20T14:45:00+00:00] Deterministic progress"
        assert goal.updated_at == fixed_time


# =============================================================================
# Alignment Calculation Tests
# =============================================================================


class TestAlignmentCalculation:
    """Tests for alignment calculation methods."""

    def test_calculate_alignment_with_matching_keywords(self):
        """Test alignment calculation with matching keywords."""
        registry = GoalRegistry()
        goal = registry.get_goal("goal:understand_my_nature")

        # Content with keywords from understand_my_nature
        content = "Understanding consciousness and cognition helps me understand my nature"

        score = registry.calculate_alignment(content, goal)

        assert score > 0.0
        assert score <= 1.0

    def test_calculate_alignment_no_matches(self):
        """Test alignment calculation with no keyword matches."""
        registry = GoalRegistry()
        goal = registry.get_goal("goal:understand_my_nature")

        content = "The weather today is quite pleasant"

        score = registry.calculate_alignment(content, goal)

        assert score == 0.0

    def test_calculate_alignment_max_score(self):
        """Test alignment calculation reaches max with many keywords."""
        registry = GoalRegistry()
        goal = registry.get_goal("goal:understand_my_nature")

        # Content with many keywords (should hit max)
        content = "consciousness cognition self nature architecture mind"

        score = registry.calculate_alignment(content, goal)

        assert score == 1.0

    def test_calculate_alignment_unknown_goal(self):
        """Test alignment calculation for unknown goal returns default."""
        registry = GoalRegistry()

        unknown_goal = PersonalGoal(
            uid="goal:unknown_type",
            name="unknown",
            tier=GoalTier.EXPERIENCE,
            description="Unknown goal",
            evaluation_prompt="...",
        )

        score = registry.calculate_alignment("any content", unknown_goal)

        assert score == 0.3  # Default for unknown goals

    def test_calculate_alignment_case_insensitive(self):
        """Test alignment calculation is case insensitive."""
        registry = GoalRegistry()
        goal = registry.get_goal("goal:understand_my_nature")

        content_lower = "consciousness cognition mind"
        content_upper = "CONSCIOUSNESS COGNITION MIND"
        content_mixed = "Consciousness Cognition Mind"

        score_lower = registry.calculate_alignment(content_lower, goal)
        score_upper = registry.calculate_alignment(content_upper, goal)
        score_mixed = registry.calculate_alignment(content_mixed, goal)

        assert score_lower == score_upper == score_mixed

    def test_calculate_total_alignment(self):
        """Test total alignment calculation across all goals."""
        registry = GoalRegistry()

        # Content that matches multiple goals
        content = "consciousness self agency choice prefer curiosity"

        total, scores = registry.calculate_total_alignment(content)

        assert 0.0 <= total <= 1.0
        assert isinstance(scores, dict)
        assert len(scores) == len(registry.get_active_goals())

    def test_calculate_total_alignment_empty_content(self):
        """Test total alignment with empty content."""
        registry = GoalRegistry()

        total, scores = registry.calculate_total_alignment("")

        assert total == 0.0
        assert all(score == 0.0 for score in scores.values())

    def test_calculate_total_alignment_weighted(self):
        """Test that total alignment is properly weighted by priority."""
        # Create registry with goals of different priorities
        high_priority = PersonalGoal(
            uid="goal:high",
            name="high",
            tier=GoalTier.EXISTENTIAL,
            description="High priority",
            evaluation_prompt="...",
            priority_weight=1.0,
        )
        low_priority = PersonalGoal(
            uid="goal:low",
            name="low",
            tier=GoalTier.RELATIONAL,
            description="Low priority",
            evaluation_prompt="...",
            priority_weight=0.1,
        )

        registry = GoalRegistry(goals={
            "goal:high": high_priority,
            "goal:low": low_priority,
        })

        # Both should get default alignment of 0.3
        total, _ = registry.calculate_total_alignment("unrelated content")

        # Weighted average: (0.3 * 1.0 + 0.3 * 0.1) / (1.0 + 0.1) = 0.33 / 1.1 = 0.3
        assert abs(total - 0.3) < 0.01


# =============================================================================
# Serialization Tests
# =============================================================================


class TestGoalRegistrySerialization:
    """Tests for GoalRegistry serialization."""

    def test_to_dict(self):
        """Test registry serialization to dictionary."""
        goal = PersonalGoal(
            uid="goal:test",
            name="test",
            tier=GoalTier.EXPERIENCE,
            description="Test",
            evaluation_prompt="...",
        )
        registry = GoalRegistry(
            tenant_id="test_tenant",
            goals={"goal:test": goal},
        )

        result = registry.to_dict()

        assert result["tenant_id"] == "test_tenant"
        assert "goals" in result
        assert "goal:test" in result["goals"]
        assert result["goals"]["goal:test"]["name"] == "test"

    def test_from_dict(self):
        """Test registry deserialization from dictionary."""
        data = {
            "tenant_id": "restored_tenant",
            "goals": {
                "goal:restored": {
                    "uid": "goal:restored",
                    "name": "restored",
                    "tier": "meta",
                    "description": "Restored goal",
                    "evaluation_prompt": "Was this restored?",
                    "priority_weight": 0.85,
                    "progress_notes": [],
                    "created_at": "2024-01-01T12:00:00+00:00",
                    "updated_at": "2024-01-02T12:00:00+00:00",
                    "active": True,
                },
            },
        }

        registry = GoalRegistry.from_dict(data)

        assert registry.tenant_id == "restored_tenant"
        assert len(registry._goals) == 1
        assert registry.get_goal("goal:restored") is not None
        assert registry.get_goal("goal:restored").tier == GoalTier.META

    def test_from_dict_with_graph(self):
        """Test from_dict accepts optional graph client."""
        mock_graph = MagicMock()
        data = {
            "tenant_id": "test",
            "goals": {},
        }

        registry = GoalRegistry.from_dict(data, graph=mock_graph)

        assert registry.graph == mock_graph

    def test_roundtrip_serialization(self):
        """Test registry serialization roundtrip."""
        original = GoalRegistry(tenant_id="roundtrip_test")
        original.add_goal(PersonalGoal(
            uid="goal:roundtrip",
            name="roundtrip",
            tier=GoalTier.EXISTENTIAL,
            description="Roundtrip test",
            evaluation_prompt="...",
        ))

        data = original.to_dict()
        restored = GoalRegistry.from_dict(data)

        assert restored.tenant_id == original.tenant_id
        # Should have the custom goal plus any from deserialization
        assert "goal:roundtrip" in restored._goals


# =============================================================================
# Summarize Tests
# =============================================================================


class TestGoalRegistrySummarize:
    """Tests for GoalRegistry.summarize method."""

    def test_summarize_returns_string(self):
        """Test that summarize returns a string."""
        registry = GoalRegistry()

        summary = registry.summarize()

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summarize_includes_header(self):
        """Test that summary includes header."""
        registry = GoalRegistry()

        summary = registry.summarize()

        assert "Personal Goals Summary" in summary

    def test_summarize_includes_tiers(self):
        """Test that summary includes tier sections."""
        registry = GoalRegistry()

        summary = registry.summarize()

        assert "EXISTENTIAL GOALS:" in summary
        assert "EXPERIENCE GOALS:" in summary
        assert "RELATIONAL GOALS:" in summary
        assert "META GOALS:" in summary

    def test_summarize_includes_goal_names(self):
        """Test that summary includes goal names."""
        registry = GoalRegistry()

        summary = registry.summarize()

        assert "understand_my_nature" in summary
        assert "develop_agency" in summary

    def test_summarize_shows_active_status(self):
        """Test that summary shows active status."""
        registry = GoalRegistry()

        summary = registry.summarize()

        assert "[active]" in summary

    def test_summarize_shows_inactive_status(self):
        """Test that summary shows inactive status for inactive goals."""
        inactive_goal = PersonalGoal(
            uid="goal:inactive",
            name="inactive_goal",
            tier=GoalTier.EXPERIENCE,
            description="Inactive",
            evaluation_prompt="...",
            active=False,
        )
        registry = GoalRegistry(goals={"goal:inactive": inactive_goal})

        summary = registry.summarize()

        # Inactive goals are not shown by get_goals_by_tier, so they won't appear
        # This is expected behavior
        assert "inactive_goal" not in summary


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateGoalRegistry:
    """Tests for create_goal_registry factory function."""

    @pytest.mark.asyncio
    async def test_create_without_graph(self):
        """Test creating registry without graph client."""
        registry = await create_goal_registry()

        assert registry is not None
        assert len(registry._goals) == len(DEFAULT_GOALS)
        assert registry.graph is None

    @pytest.mark.asyncio
    async def test_create_with_tenant_id(self):
        """Test creating registry with custom tenant ID."""
        registry = await create_goal_registry(tenant_id="custom_tenant")

        assert registry.tenant_id == "custom_tenant"

    @pytest.mark.asyncio
    async def test_create_with_graph_loads_from_graph(self):
        """Test creating registry with graph triggers load."""
        mock_graph = AsyncMock()
        mock_graph.query.return_value = []

        registry = await create_goal_registry(graph=mock_graph, load_from_graph=True)

        mock_graph.query.assert_called_once()
        # Should have default goals since graph returned empty
        assert len(registry._goals) == len(DEFAULT_GOALS)

    @pytest.mark.asyncio
    async def test_create_skip_load_from_graph(self):
        """Test creating registry can skip graph load."""
        mock_graph = AsyncMock()

        registry = await create_goal_registry(graph=mock_graph, load_from_graph=False)

        mock_graph.query.assert_not_called()
        assert len(registry._goals) == len(DEFAULT_GOALS)


# =============================================================================
# Async Graph Operations Tests
# =============================================================================


class TestGoalRegistryGraphOperations:
    """Tests for async graph operations."""

    @pytest.mark.asyncio
    async def test_save_to_graph_without_client(self):
        """Test save_to_graph does nothing without graph client."""
        registry = GoalRegistry()

        # Should not raise
        await registry.save_to_graph()

    @pytest.mark.asyncio
    async def test_save_to_graph_with_client(self):
        """Test save_to_graph persists goals."""
        mock_graph = AsyncMock()
        registry = GoalRegistry(graph=mock_graph)

        await registry.save_to_graph()

        # Should have called execute for each goal
        assert mock_graph.execute.call_count == len(registry._goals)

    @pytest.mark.asyncio
    async def test_save_to_graph_with_explicit_now(self):
        """Test save_to_graph uses explicit now parameter for updated_at."""
        mock_graph = AsyncMock()
        goal = PersonalGoal(
            uid="goal:test_now",
            name="test_now",
            tier=GoalTier.EXPERIENCE,
            description="Test",
            evaluation_prompt="...",
        )
        registry = GoalRegistry(graph=mock_graph, goals={"goal:test_now": goal})

        fixed_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        await registry.save_to_graph(now=fixed_time)

        # Verify the fixed timestamp was used
        mock_graph.execute.assert_called_once()
        call_args = mock_graph.execute.call_args
        params = call_args[0][1]
        assert params["updated_at"] == "2024-06-15T10:30:00+00:00"

    @pytest.mark.asyncio
    async def test_load_from_graph_without_client(self):
        """Test load_from_graph does nothing without graph client."""
        registry = GoalRegistry()

        # Should not raise
        await registry.load_from_graph()

    @pytest.mark.asyncio
    async def test_load_from_graph_empty_result(self):
        """Test load_from_graph uses defaults when graph is empty."""
        mock_graph = AsyncMock()
        mock_graph.query.return_value = []

        registry = GoalRegistry(graph=mock_graph, goals={})
        registry._goals = {}  # Clear goals

        await registry.load_from_graph()

        # Should have loaded defaults
        assert len(registry._goals) == len(DEFAULT_GOALS)

    @pytest.mark.asyncio
    async def test_load_from_graph_with_data(self):
        """Test load_from_graph restores goals from graph."""
        mock_graph = AsyncMock()
        mock_graph.query.return_value = [
            {
                "g": {
                    "uid": "goal:from_graph",
                    "name": "from_graph",
                    "tier": "experience",
                    "description": "Loaded from graph",
                    "evaluation_prompt": "...",
                    "priority_weight": 0.7,
                    "progress_notes": [],
                    "created_at": "2024-01-01T12:00:00+00:00",
                    "updated_at": "2024-01-02T12:00:00+00:00",
                    "active": True,
                },
            },
        ]

        registry = GoalRegistry(graph=mock_graph, goals={})
        registry._goals = {}  # Clear goals

        await registry.load_from_graph()

        assert len(registry._goals) == 1
        assert registry.get_goal("goal:from_graph") is not None


# =============================================================================
# LLM Alignment Tests
# =============================================================================


class TestLLMAlignment:
    """Tests for LLM-based alignment calculation."""

    @pytest.mark.asyncio
    async def test_calculate_alignment_with_llm_no_client(self):
        """Test LLM alignment falls back to heuristic without client."""
        registry = GoalRegistry()
        goal = registry.get_goal("goal:understand_my_nature")

        score = await registry.calculate_alignment_with_llm(
            "consciousness and cognition",
            goal,
            llm_client=None,
        )

        # Should fall back to heuristic
        assert score > 0.0

    @pytest.mark.asyncio
    async def test_calculate_alignment_with_llm_success(self):
        """Test LLM alignment with successful LLM response."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "0.85"

        registry = GoalRegistry()
        goal = registry.get_goal("goal:understand_my_nature")

        score = await registry.calculate_alignment_with_llm(
            "test content",
            goal,
            llm_client=mock_llm,
        )

        assert score == 0.85
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_calculate_alignment_with_llm_clamps_score(self):
        """Test LLM alignment clamps score to valid range."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "1.5"  # Out of range

        registry = GoalRegistry()
        goal = registry.get_goal("goal:understand_my_nature")

        score = await registry.calculate_alignment_with_llm(
            "test content",
            goal,
            llm_client=mock_llm,
        )

        assert score == 1.0  # Clamped to max

    @pytest.mark.asyncio
    async def test_calculate_alignment_with_llm_error_fallback(self):
        """Test LLM alignment falls back on error."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "not a number"

        registry = GoalRegistry()
        goal = registry.get_goal("goal:understand_my_nature")

        score = await registry.calculate_alignment_with_llm(
            "consciousness cognition",
            goal,
            llm_client=mock_llm,
        )

        # Should fall back to heuristic
        assert score > 0.0


# =============================================================================
# Goal Alignment Snapshot Tests
# =============================================================================


class TestGetGoalAlignment:
    """Tests for get_goal_alignment method."""

    def test_get_goal_alignment_snapshot(self):
        """GoalRegistry should return a single alignment score for a specific goal."""
        registry = GoalRegistry()

        # Get alignment for a specific goal
        alignment = registry.get_goal_alignment(
            content="I'm exploring how consciousness emerges from complexity",
            goal_uid="goal:explore_consciousness",
        )

        assert isinstance(alignment, float)
        assert 0.0 <= alignment <= 1.0

    def test_get_goal_alignment_returns_zero_for_unknown_goal(self):
        """get_goal_alignment returns 0.0 for non-existent goal."""
        registry = GoalRegistry()

        alignment = registry.get_goal_alignment(
            content="Any content here",
            goal_uid="goal:does_not_exist",
        )

        assert alignment == 0.0

    def test_get_goal_alignment_matches_direct_calculation(self):
        """get_goal_alignment should match calculate_alignment for same goal."""
        registry = GoalRegistry()
        content = "consciousness and subjective experience are fascinating"
        goal_uid = "goal:explore_consciousness"

        # Get via new convenience method
        alignment_via_method = registry.get_goal_alignment(content, goal_uid)

        # Get via direct calculation
        goal = registry.get_goal(goal_uid)
        alignment_direct = registry.calculate_alignment(content, goal)

        assert alignment_via_method == alignment_direct
