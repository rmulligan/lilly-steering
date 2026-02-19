"""Budget allocation for steering magnitude distribution.

This module provides the BudgetAllocator class which distributes steering
magnitude budget between hypothesis-driven process vectors and Evalatis
emergent vectors. This replaces magnitude capping with proportional allocation
to avoid homogenized steering - each source gets a percentage of total capacity.

Key concepts:
- Process budget: Allocated to hypothesis vectors, split proportionally by effectiveness
- Emergent budget: Allocated to Evalatis (emergence-selection) vectors
- Proportional allocation: Higher effectiveness = larger share of process budget
- Normalization: Each vector scaled to its allocated magnitude budget

Example:
    # 3 hypothesis vectors with effectiveness [0.8, 0.6, 0.4]
    # total_budget = 2.0, process_share = 0.6
    # process_budget = 1.2, emergent_budget = 0.8
    # effectiveness_sum = 1.8
    # vec1 gets 1.2 * (0.8/1.8) = 0.533 magnitude
    # vec2 gets 1.2 * (0.6/1.8) = 0.400 magnitude
    # vec3 gets 1.2 * (0.4/1.8) = 0.267 magnitude
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

from core.steering.hypothesis_vectors import HypothesisSteeringVector


class BudgetAllocator:
    """Distributes steering budget between process and emergent vectors.

    Splits the total steering magnitude budget between:
    1. Process vectors (hypothesis-driven): Split proportionally by effectiveness
    2. Emergent vectors (Evalatis): Gets remaining budget share

    This approach avoids magnitude capping which squashes multiple vectors into
    homogenized mush. Instead, each steering source gets a percentage of the
    total steering capacity.

    Attributes:
        default_process_share: Default fraction of budget for process vectors (0.0-1.0)
    """

    def __init__(self, default_process_share: float = 0.6):
        """Initialize the budget allocator.

        Args:
            default_process_share: Default fraction of total budget allocated to
                hypothesis/process vectors. Emergent gets (1 - process_share).
                Default is 0.6 (60% process, 40% emergent).
        """
        self.default_process_share = default_process_share

    def normalize_to_magnitude(
        self, vector: np.ndarray, target_magnitude: float
    ) -> np.ndarray:
        """Scale a vector to have exactly the target magnitude.

        Preserves direction while adjusting magnitude. Returns zero vector if
        input is zero or target magnitude is zero.

        Args:
            vector: Input vector to normalize
            target_magnitude: Desired magnitude for output vector

        Returns:
            Vector with same direction but target_magnitude norm
        """
        current_magnitude = np.linalg.norm(vector)

        # Handle zero vector or zero target
        if current_magnitude < 1e-10 or target_magnitude < 1e-10:
            return np.zeros_like(vector)

        # Scale to target magnitude
        scale_factor = target_magnitude / current_magnitude
        return vector * scale_factor

    def _compute_effectiveness_budgets(
        self,
        hypothesis_vectors: list[HypothesisSteeringVector],
        process_budget: float,
    ) -> list[float]:
        """Compute individual budgets based on effectiveness scores.

        Distributes the process budget proportionally by effectiveness score.
        If all effectiveness scores are zero, splits equally.

        Args:
            hypothesis_vectors: List of hypothesis steering vectors
            process_budget: Total budget available for process vectors

        Returns:
            List of individual magnitude budgets, one per vector
        """
        if not hypothesis_vectors:
            return []

        # Sum effectiveness scores
        effectiveness_scores = [v.effectiveness_score for v in hypothesis_vectors]
        total_effectiveness = sum(effectiveness_scores)

        # Handle all-zero case: equal split
        if total_effectiveness < 1e-10:
            equal_share = process_budget / len(hypothesis_vectors)
            return [equal_share] * len(hypothesis_vectors)

        # Proportional allocation
        budgets = []
        for score in effectiveness_scores:
            proportion = score / total_effectiveness
            budget = process_budget * proportion
            budgets.append(budget)

        return budgets

    def allocate(
        self,
        hypothesis_vectors: list[HypothesisSteeringVector],
        evalatis_vector: np.ndarray,
        total_budget: float,
        process_share: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Allocate steering budget between process and emergent vectors.

        Splits total budget into process and emergent shares. Within the process
        share, allocates proportionally by effectiveness score. Each hypothesis
        vector is normalized to its allocated budget, then combined. The combined
        process vector is normalized to the total process budget magnitude.

        Args:
            hypothesis_vectors: List of hypothesis steering vectors to combine
            evalatis_vector: Emergent vector from Evalatis steerer
            total_budget: Total steering magnitude budget (from capacity tracking)
            process_share: Fraction for process vectors (overrides default if provided)

        Returns:
            Tuple of (combined_process_vector, normalized_evalatis_vector)
            where each has the appropriate allocated magnitude
        """
        # Use provided share or default
        share = process_share if process_share is not None else self.default_process_share

        # Calculate budget split
        process_budget = total_budget * share
        emergent_budget = total_budget * (1 - share)

        # Normalize emergent vector to its budget
        normalized_emergent = self.normalize_to_magnitude(evalatis_vector, emergent_budget)

        # Handle empty hypothesis vectors
        if not hypothesis_vectors:
            zero_vector = np.zeros_like(evalatis_vector)
            return (zero_vector, normalized_emergent)

        # Compute individual budgets by effectiveness
        individual_budgets = self._compute_effectiveness_budgets(
            hypothesis_vectors, process_budget
        )

        # Normalize each hypothesis vector to its allocated budget and sum
        combined = np.zeros_like(evalatis_vector, dtype=np.float64)
        for vec, budget in zip(hypothesis_vectors, individual_budgets):
            vector_array = np.array(vec.vector_data, dtype=np.float64)
            # Resize if needed (hypothesis vectors might be different dimension)
            if len(vector_array) != len(combined):
                logger.warning(
                    f"Resizing hypothesis vector from dim {len(vector_array)} "
                    f"to {len(combined)} to match."
                )
                # Pad or truncate to match evalatis dimension
                if len(vector_array) < len(combined):
                    padded = np.zeros(len(combined), dtype=np.float64)
                    padded[: len(vector_array)] = vector_array
                    vector_array = padded
                else:
                    vector_array = vector_array[: len(combined)]

            # Normalize to individual budget
            normalized_vec = self.normalize_to_magnitude(vector_array, budget)
            combined += normalized_vec

        # Normalize combined process vector to total process budget
        combined_process = self.normalize_to_magnitude(combined, process_budget)

        return (combined_process, normalized_emergent)
