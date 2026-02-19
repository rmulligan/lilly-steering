# core/cognitive/faithfulness.py

"""
Faithfulness Validation: Detecting Divergence Between Activations and Verbal Claims.

This module implements cross-validation between SAE feature activations (what
actually fired during generation) and the curator's verbal claims about what
influenced the thought.

Based on Walden (2026) "Reasoning Models Will Blatantly Lie About Their Reasoning":
Large Reasoning Models deny using information even when behavioral evidence shows
they are using it. This poses "hard limits on how much insight can be gleaned about
model reasoning processes from CoT inspection."

Key insight: SAE features provide ground truth about what patterns activated.
Verbal claims can be checked against this ground truth.
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

from core.cognitive.curator_schemas import (
    ThoughtAnalysis,
    SAEFeature,
    FaithfulnessScore,
)

logger = logging.getLogger(__name__)

# Default activation threshold for considering a feature "active"
DEFAULT_ACTIVATION_THRESHOLD = 0.5


@dataclass
class SAEInfluenceMapping:
    """Maps SAE feature indices to semantic influence categories.

    Categories follow a namespace convention:
    - memory:episodic, memory:semantic
    - concept:<name>
    - emotion:<name>
    - identity:<aspect>
    """

    _feature_to_influence: dict[int, str] = field(default_factory=dict)
    _influence_to_features: dict[str, list[int]] = field(default_factory=dict)

    def register(self, feature_idx: int, influence_category: str) -> None:
        """Register a mapping from SAE feature to influence category."""
        self._feature_to_influence[feature_idx] = influence_category
        if influence_category not in self._influence_to_features:
            self._influence_to_features[influence_category] = []
        if feature_idx not in self._influence_to_features[influence_category]:
            self._influence_to_features[influence_category].append(feature_idx)

    def get_influences(
        self,
        features: list[SAEFeature],
        threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    ) -> list[str]:
        """Get active influence categories from SAE features.

        Args:
            features: SAE features with activations
            threshold: Minimum activation to consider a feature "active"

        Returns:
            List of influence category strings for active, mapped features
        """
        influences = set()
        for feature in features:
            if feature.activation >= threshold:
                if feature.feature_id in self._feature_to_influence:
                    influences.add(self._feature_to_influence[feature.feature_id])
        return list(influences)

    def get_features_for_influence(self, influence: str) -> list[int]:
        """Get all feature indices associated with an influence category."""
        return self._influence_to_features.get(influence, [])


@dataclass
class FaithfulnessValidator:
    """Validates faithfulness of verbal claims against activation evidence.

    The validator compares:
    1. What concepts the curator claims influenced the thought
    2. What SAE features actually activated during generation

    When there's significant divergence, this indicates the model may be
    confabulating its reasoning rather than accurately reporting it.
    """

    mapping: SAEInfluenceMapping = field(default_factory=SAEInfluenceMapping)
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD

    def compute_faithfulness(
        self,
        analysis: ThoughtAnalysis,
        sae_features: list[SAEFeature],
        concept_to_feature_map: Optional[dict[str, list[int]]] = None,
    ) -> FaithfulnessScore:
        """Compute faithfulness score comparing verbal claims to activations.

        Args:
            analysis: Curator's analysis with claimed concepts/influences
            sae_features: SAE features that activated during generation
            concept_to_feature_map: Optional mapping of concepts to feature indices
                If provided, overrides the internal mapping for this computation

        Returns:
            FaithfulnessScore with overlap metrics and divergence details
        """
        # Build active feature set
        active_features = {
            f.feature_id for f in sae_features
            if f.activation >= self.activation_threshold
        }

        # Get claimed influences from analysis
        claimed = set(analysis.concepts) if analysis.concepts else set()

        # Determine SAE-evidenced influences
        if concept_to_feature_map:
            # Use provided mapping
            evidenced = set()
            for concept, feature_ids in concept_to_feature_map.items():
                if any(fid in active_features for fid in feature_ids):
                    evidenced.add(concept)
        else:
            # Use internal mapping
            evidenced = set(self.mapping.get_influences(sae_features, self.activation_threshold))

        # Compute overlap
        overlap = claimed & evidenced
        overlap_ratio = len(overlap) / len(claimed) if claimed else 1.0

        # Find divergences
        missing_from_verbal = list(evidenced - claimed)
        unsupported_claims = list(claimed - evidenced)

        return FaithfulnessScore(
            claimed_influences=list(claimed),
            sae_evidence=list(evidenced),
            overlap_ratio=overlap_ratio,
            missing_from_verbal=missing_from_verbal,
            unsupported_claims=unsupported_claims,
        )

    def compute_faithfulness_from_labels(
        self,
        analysis: ThoughtAnalysis,
        sae_features: list[SAEFeature],
    ) -> FaithfulnessScore:
        """Compute faithfulness by matching concepts to SAE feature labels.

        This method uses fuzzy matching between claimed concepts and the
        semantic labels of active SAE features. More flexible than explicit
        feature-to-concept mappings.

        Args:
            analysis: Curator's analysis with claimed concepts
            sae_features: SAE features with labels from interpretation

        Returns:
            FaithfulnessScore based on label matching
        """
        claimed = set(analysis.concepts) if analysis.concepts else set()

        # Get labels from active features
        active_labels = set()
        for f in sae_features:
            if f.activation >= self.activation_threshold and f.label:
                # Normalize label: lowercase, split on underscores
                label_words = set(f.label.lower().replace("_", " ").split())
                active_labels.update(label_words)

        # Match claimed concepts against active labels
        evidenced = set()
        unsupported = []

        for concept in claimed:
            concept_lower = concept.lower()
            # Check if concept or its stem appears in any label
            if concept_lower in active_labels or any(
                concept_lower in label or label in concept_lower
                for label in active_labels
            ):
                evidenced.add(concept)
            else:
                unsupported.append(concept)

        # Find active labels not mentioned in verbal claims
        claimed_lower = {c.lower() for c in claimed}
        missing_from_verbal = [
            label for label in active_labels
            if not any(label in c or c in label for c in claimed_lower)
        ]

        overlap_ratio = len(evidenced) / len(claimed) if claimed else 1.0

        return FaithfulnessScore(
            claimed_influences=list(claimed),
            sae_evidence=list(active_labels),
            overlap_ratio=overlap_ratio,
            missing_from_verbal=missing_from_verbal,
            unsupported_claims=unsupported,
        )

    def log_divergence(self, score: FaithfulnessScore, cycle: int) -> None:
        """Log faithfulness divergence for monitoring."""
        if score.divergence_severity == "none":
            return

        logger.warning(
            f"Faithfulness divergence detected (cycle {cycle}): "
            f"severity={score.divergence_severity}, "
            f"overlap={score.overlap_ratio:.2f}, "
            f"unsupported={score.unsupported_claims}"
        )
