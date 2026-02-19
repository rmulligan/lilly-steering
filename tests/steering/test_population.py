"""Tests for population-based steering vectors."""
import numpy as np
import pytest
from datetime import datetime


def test_steering_vector_entry_creation():
    """SteeringVectorEntry stores vector with metadata."""
    from core.steering.population import SteeringVectorEntry

    vector = np.random.randn(4096).astype(np.float32)
    entry = SteeringVectorEntry(
        name="curious",
        vector=vector,
        description="Drives exploratory behavior",
    )

    assert entry.name == "curious"
    assert entry.vector.shape == (4096,)
    assert entry.description == "Drives exploratory behavior"
    assert entry.staleness == 0.0
    assert entry.selection_count == 0
    assert entry.birth_cycle == 0
    assert isinstance(entry.created_at, datetime)


def test_steering_vector_entry_staleness_increment():
    """Staleness increments on selection."""
    from core.steering.population import SteeringVectorEntry

    entry = SteeringVectorEntry(
        name="test",
        vector=np.zeros(4096, dtype=np.float32),
    )

    assert entry.staleness == 0.0
    entry.record_selection()
    assert entry.staleness == 1.0
    assert entry.selection_count == 1

    entry.record_selection()
    assert entry.staleness == 2.0
    assert entry.selection_count == 2


def test_steering_vector_entry_staleness_decay():
    """Staleness decays each cycle."""
    from core.steering.population import SteeringVectorEntry

    entry = SteeringVectorEntry(
        name="test",
        vector=np.zeros(4096, dtype=np.float32),
    )
    entry.staleness = 10.0

    entry.apply_decay(decay_rate=0.95)
    assert abs(entry.staleness - 9.5) < 0.01


def test_steering_vector_entry_staleness_penalty():
    """Staleness penalty is capped at 0.5."""
    from core.steering.population import SteeringVectorEntry

    entry = SteeringVectorEntry(
        name="test",
        vector=np.zeros(4096, dtype=np.float32),
    )

    # Low staleness = low penalty
    entry.staleness = 10.0
    assert entry.staleness_penalty == 0.1  # 10 / 100

    # High staleness = capped penalty
    entry.staleness = 100.0
    assert entry.staleness_penalty == 0.5  # Capped

    entry.staleness = 200.0
    assert entry.staleness_penalty == 0.5  # Still capped


# =============================================================================
# VectorPopulation Tests
# =============================================================================


def test_vector_population_initialization():
    """VectorPopulation initializes with given d_model."""
    from core.steering.population import VectorPopulation

    pop = VectorPopulation(d_model=4096, max_size=10)

    assert pop.d_model == 4096
    assert pop.max_size == 10
    assert len(pop) == 0


def test_vector_population_add_vector():
    """Can add vectors to population."""
    from core.steering.population import VectorPopulation

    pop = VectorPopulation(d_model=4096, max_size=10)
    vector = np.random.randn(4096).astype(np.float32)

    pop.add("curious", vector, description="Exploratory")

    assert len(pop) == 1
    assert "curious" in pop
    entry = pop.get("curious")
    assert entry is not None
    assert entry.name == "curious"


def test_vector_population_add_duplicate_raises():
    """Adding duplicate name raises ValueError."""
    from core.steering.population import VectorPopulation

    pop = VectorPopulation(d_model=4096, max_size=10)
    vector = np.zeros(4096, dtype=np.float32)

    pop.add("curious", vector)

    with pytest.raises(ValueError, match="already exists"):
        pop.add("curious", vector)


def test_vector_population_wrong_dimension_raises():
    """Adding wrong dimension vector raises ValueError."""
    from core.steering.population import VectorPopulation

    pop = VectorPopulation(d_model=4096, max_size=10)
    wrong_vector = np.zeros(2048, dtype=np.float32)

    with pytest.raises(ValueError, match="dimension"):
        pop.add("test", wrong_vector)


def test_vector_population_iteration():
    """Can iterate over population entries."""
    from core.steering.population import VectorPopulation

    pop = VectorPopulation(d_model=4096, max_size=10)
    pop.add("a", np.zeros(4096, dtype=np.float32))
    pop.add("b", np.ones(4096, dtype=np.float32))

    names = [entry.name for entry in pop]
    assert set(names) == {"a", "b"}


def test_vector_population_apply_decay_all():
    """apply_decay_all decays all entries."""
    from core.steering.population import VectorPopulation

    pop = VectorPopulation(d_model=4096, max_size=10)
    pop.add("a", np.zeros(4096, dtype=np.float32))
    pop.add("b", np.zeros(4096, dtype=np.float32))

    # Set initial staleness
    pop.get("a").staleness = 10.0
    pop.get("b").staleness = 20.0

    pop.apply_decay_all(decay_rate=0.9)

    assert abs(pop.get("a").staleness - 9.0) < 0.01
    assert abs(pop.get("b").staleness - 18.0) < 0.01


def test_vector_population_max_size_enforced():
    """Population respects max_size limit."""
    from core.steering.population import VectorPopulation

    pop = VectorPopulation(d_model=4096, max_size=2)
    pop.add("a", np.zeros(4096, dtype=np.float32))
    pop.add("b", np.zeros(4096, dtype=np.float32))

    with pytest.raises(ValueError, match="max_size"):
        pop.add("c", np.zeros(4096, dtype=np.float32))


# =============================================================================
# AffinityMatrix Tests
# =============================================================================


def test_affinity_matrix_initialization():
    """AffinityMatrix initializes empty."""
    from core.steering.population import AffinityMatrix

    matrix = AffinityMatrix()
    assert len(matrix) == 0


def test_affinity_matrix_get_default():
    """Unknown prompt×vector pairs return default affinity."""
    from core.steering.population import AffinityMatrix

    matrix = AffinityMatrix(default_affinity=0.5)

    affinity = matrix.get("unknown_prompt", "unknown_vector")
    assert affinity == 0.5


def test_affinity_matrix_set_and_get():
    """Can set and retrieve affinities."""
    from core.steering.population import AffinityMatrix

    matrix = AffinityMatrix()

    matrix.set("tensions?", "curious", 0.82)
    matrix.set("tensions?", "skeptical", 0.45)

    assert matrix.get("tensions?", "curious") == 0.82
    assert matrix.get("tensions?", "skeptical") == 0.45


def test_affinity_matrix_update_ema():
    """update() uses EMA blending."""
    from core.steering.population import AffinityMatrix

    matrix = AffinityMatrix(default_affinity=0.5, ema_alpha=0.1)

    # First update from default
    matrix.update("prompt", "vector", 1.0)
    # EMA: 0.9 * 0.5 + 0.1 * 1.0 = 0.55
    assert abs(matrix.get("prompt", "vector") - 0.55) < 0.01

    # Second update
    matrix.update("prompt", "vector", 1.0)
    # EMA: 0.9 * 0.55 + 0.1 * 1.0 = 0.595
    assert abs(matrix.get("prompt", "vector") - 0.595) < 0.01


def test_affinity_matrix_get_top_vectors():
    """get_top_vectors returns highest affinity vectors for a prompt."""
    from core.steering.population import AffinityMatrix

    matrix = AffinityMatrix()
    matrix.set("prompt", "a", 0.3)
    matrix.set("prompt", "b", 0.9)
    matrix.set("prompt", "c", 0.6)

    top = matrix.get_top_vectors("prompt", top_k=2)
    assert top == [("b", 0.9), ("c", 0.6)]


def test_affinity_matrix_get_top_prompts():
    """get_top_prompts returns highest affinity prompts for a vector."""
    from core.steering.population import AffinityMatrix

    matrix = AffinityMatrix()
    matrix.set("p1", "vector", 0.3)
    matrix.set("p2", "vector", 0.9)
    matrix.set("p3", "vector", 0.6)

    top = matrix.get_top_prompts("vector", top_k=2)
    assert top == [("p2", 0.9), ("p3", 0.6)]


def test_affinity_matrix_decay_all():
    """decay_all reduces all affinities toward default."""
    from core.steering.population import AffinityMatrix

    matrix = AffinityMatrix(default_affinity=0.5, decay_rate=0.9)
    matrix.set("prompt", "vector", 1.0)

    matrix.decay_all()
    # Decay toward default: 0.9 * 1.0 + 0.1 * 0.5 = 0.95
    assert abs(matrix.get("prompt", "vector") - 0.95) < 0.01


# =============================================================================
# PromptEntry Tests
# =============================================================================


def test_prompt_entry_creation():
    """PromptEntry stores prompt template with metadata."""
    from core.steering.population import PromptEntry

    entry = PromptEntry(
        key="tensions",
        template="What tensions do I feel about {concept}?",
        description="Explores internal conflicts",
    )

    assert entry.key == "tensions"
    assert "{concept}" in entry.template
    assert entry.staleness == 0.0


def test_prompt_population_add_and_get():
    """PromptPopulation manages prompt templates."""
    from core.steering.population import PromptPopulation

    pop = PromptPopulation(max_size=10)
    pop.add(
        key="tensions",
        template="What tensions do I feel about {concept}?",
    )

    assert len(pop) == 1
    entry = pop.get("tensions")
    assert entry is not None
    assert entry.key == "tensions"


def test_prompt_population_format():
    """PromptEntry can format with concept."""
    from core.steering.population import PromptEntry

    entry = PromptEntry(
        key="connect",
        template="How does {concept} connect to what I know?",
    )

    formatted = entry.format(concept="curiosity")
    assert formatted == "How does curiosity connect to what I know?"
