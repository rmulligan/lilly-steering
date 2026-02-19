"""Graph-aware position hints for retrieved context.

Implements the RePo (Context Re-Positioning) pattern for adding
semantic proximity metadata to help the LLM better attend to related context.

Position hints are lightweight bracketed annotations that provide semantic
anchors without overwhelming the narrative flow:
    "The boundary between self and other is permeable" [concept: consciousness, highly related]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)


@dataclass
class PositionHint:
    """A single position hint annotation.

    Hints provide graph-based proximity signals for retrieved insights,
    helping the LLM understand relationships between context items.
    """

    hint_type: str  # "concept", "sae", "lineage", "score"
    detail: str  # e.g., "consciousness", "parent", "highly related"

    def format(self) -> str:
        """Format as bracketed annotation.

        Returns:
            Formatted string like "[concept: consciousness]" or "[highly related]"
        """
        if self.hint_type == "score":
            return f"[{self.detail}]"
        return f"[{self.hint_type}: {self.detail}]"


@dataclass
class AnnotatedInsight:
    """An insight with its position hints.

    Combines the original insight tuple data with computed graph-aware hints
    for richer prompt context.
    """

    uid: str
    text: str
    score: float
    hints: list[PositionHint] = field(default_factory=list)

    def format_with_hints(self, max_hints: int = 2) -> str:
        """Format insight text with bracketed hint annotations.

        Args:
            max_hints: Maximum number of hints to include (default 2)

        Returns:
            Formatted string like:
            '"The boundary is permeable" [concept: consciousness, highly related]'
        """
        if not self.hints:
            return f'"{self.text}"'

        # Take top hints up to max
        selected_hints = self.hints[:max_hints]
        hint_str = ", ".join(h.format().strip("[]") for h in selected_hints)
        return f'"{self.text}" [{hint_str}]'


@dataclass
class AnnotatedContext:
    """Position hints computed for a RetrievedContext."""

    semantic: list[AnnotatedInsight] = field(default_factory=list)
    activated: list[AnnotatedInsight] = field(default_factory=list)
    questions: list[AnnotatedInsight] = field(default_factory=list)


class PositionHintComputer:
    """Computes graph-aware position hints for retrieved context.

    Uses multiple proximity signals:
    - Score bands: high similarity (>0.85) gets [highly related]
    - Shared concept: matching exploration concepts get [concept: X]
    - SAE overlap: shared SAE features (>2) get [SAE-linked]
    - Lineage: EMERGED_FROM relationships get [lineage: parent/child]

    Priority order: lineage > concept > SAE > score
    """

    # Score threshold for "highly related" hint
    HIGH_SCORE_THRESHOLD = 0.85

    # Minimum SAE feature overlap for "SAE-linked" hint
    MIN_SAE_OVERLAP = 2

    def __init__(self, psyche: Optional["PsycheClient"] = None):
        """Initialize the position hint computer.

        Args:
            psyche: PsycheClient for lineage queries. If None, lineage hints disabled.
        """
        self.psyche = psyche

    async def annotate_insights(
        self,
        insights: list[tuple[str, str, float]],
        current_concept: str,
        current_sae_features: Optional[list[int]] = None,
        zettel_metadata: Optional[dict[str, dict]] = None,
        lineage_map: Optional[dict[str, set[str]]] = None,
    ) -> list[AnnotatedInsight]:
        """Add position hints to a list of insights.

        Args:
            insights: List of (uid, text, score) tuples
            current_concept: Current exploration concept
            current_sae_features: Active SAE feature indices
            zettel_metadata: Optional dict mapping uid -> {concept, sae_feature_indices}
            lineage_map: Optional dict mapping uid -> set of parent uids

        Returns:
            List of AnnotatedInsight with computed position hints
        """
        annotated = []

        for uid, text, score in insights:
            hints: list[PositionHint] = []
            metadata = zettel_metadata.get(uid, {}) if zettel_metadata else {}

            # Priority 1: Lineage hints (highest value)
            if lineage_map and uid in lineage_map:
                parent_uids = lineage_map[uid]
                if parent_uids:
                    hints.append(PositionHint("lineage", "parent"))

            # Priority 2: Concept match (don't add if same as current - redundant)
            zettel_concept = metadata.get("concept", "")
            if zettel_concept and zettel_concept.lower() != current_concept.lower():
                # Only add if there's an interesting shared concept relationship
                # (In practice, we might want concept similarity, but for now just note it)
                hints.append(PositionHint("concept", zettel_concept))

            # Priority 3: SAE overlap
            if current_sae_features:
                zettel_sae = set(metadata.get("sae_feature_indices", []))
                overlap = len(zettel_sae.intersection(set(current_sae_features)))
                if overlap >= self.MIN_SAE_OVERLAP:
                    hints.append(PositionHint("SAE-linked", ""))

            # Priority 4: Score band (lowest priority, add if high score)
            if score >= self.HIGH_SCORE_THRESHOLD:
                hints.append(PositionHint("score", "highly related"))

            annotated.append(AnnotatedInsight(
                uid=uid,
                text=text,
                score=score,
                hints=hints,
            ))

        return annotated

    async def compute_for_context(
        self,
        semantic_insights: list[tuple[str, str, float]],
        activated_insights: list[tuple[str, str, float]],
        open_questions: list[tuple[str, str, float]],
        current_concept: str,
        current_sae_features: Optional[list[int]] = None,
    ) -> AnnotatedContext:
        """Compute position hints for a full RetrievedContext.

        This is the main entry point for computing all position hints.
        It fetches zettel metadata and lineage information in batch.

        Args:
            semantic_insights: (uid, text, score) tuples from semantic retrieval
            activated_insights: (uid, text, score) tuples from SAE activation
            open_questions: (uid, text, score) tuples from open questions
            current_concept: Current exploration concept
            current_sae_features: Active SAE feature indices

        Returns:
            AnnotatedContext with hints for all insight types
        """
        # Collect all UIDs for batch fetching
        all_uids: set[str] = set()
        for uid, _, _ in semantic_insights:
            all_uids.add(uid)
        for uid, _, _ in activated_insights:
            all_uids.add(uid)
        for uid, _, _ in open_questions:
            all_uids.add(uid)

        # Fetch metadata and lineage in batch
        zettel_metadata: dict[str, dict] = {}
        lineage_map: dict[str, set[str]] = {}

        if self.psyche and all_uids:
            try:
                # Fetch zettel metadata (concept, sae_feature_indices)
                zettel_metadata = await self._fetch_zettel_metadata_batch(list(all_uids))

                # Fetch lineage relationships
                lineage_map = await self._fetch_lineage_batch(list(all_uids))
            except Exception as e:
                logger.warning(f"Failed to fetch position hint metadata: {e}")

        # Annotate each insight type
        annotated_semantic = await self.annotate_insights(
            semantic_insights,
            current_concept,
            current_sae_features,
            zettel_metadata,
            lineage_map,
        )

        # Activated insights already imply SAE relationship, so skip SAE hints
        annotated_activated = await self.annotate_insights(
            activated_insights,
            current_concept,
            None,  # Skip SAE hints for activated insights (redundant)
            zettel_metadata,
            lineage_map,
        )

        annotated_questions = await self.annotate_insights(
            open_questions,
            current_concept,
            current_sae_features,
            zettel_metadata,
            lineage_map,
        )

        return AnnotatedContext(
            semantic=annotated_semantic,
            activated=annotated_activated,
            questions=annotated_questions,
        )

    async def _fetch_zettel_metadata_batch(
        self, uids: list[str]
    ) -> dict[str, dict]:
        """Fetch concept and SAE features for multiple zettels.

        Args:
            uids: List of zettel UIDs

        Returns:
            Dict mapping uid -> {concept, sae_feature_indices}
        """
        if not self.psyche or not uids:
            return {}

        cypher = """
        MATCH (z:InsightZettel)
        WHERE z.uid IN $uids
        RETURN z.uid as uid, z.concept as concept, z.sae_feature_indices as sae_feature_indices
        """
        try:
            results = await self.psyche.query(cypher, {"uids": uids})
            return {
                r["uid"]: {
                    "concept": r.get("concept", ""),
                    "sae_feature_indices": r.get("sae_feature_indices") or [],
                }
                for r in results
            }
        except Exception as e:
            logger.warning(f"Batch metadata fetch failed: {e}")
            return {}

    async def _fetch_lineage_batch(self, uids: list[str]) -> dict[str, set[str]]:
        """Fetch EMERGED_FROM relationships for multiple zettels.

        Args:
            uids: List of zettel UIDs

        Returns:
            Dict mapping child_uid -> set of parent_uids
        """
        if not self.psyche or not uids:
            return {}

        cypher = """
        MATCH (child:InsightZettel)-[:EMERGED_FROM]->(parent:InsightZettel)
        WHERE child.uid IN $uids OR parent.uid IN $uids
        RETURN child.uid as child_uid, parent.uid as parent_uid
        """
        try:
            results = await self.psyche.query(cypher, {"uids": uids})
            lineage: dict[str, set[str]] = {}
            for r in results:
                child_uid = r["child_uid"]
                parent_uid = r["parent_uid"]
                if child_uid not in lineage:
                    lineage[child_uid] = set()
                lineage[child_uid].add(parent_uid)
            return lineage
        except Exception as e:
            logger.warning(f"Batch lineage fetch failed: {e}")
            return {}
