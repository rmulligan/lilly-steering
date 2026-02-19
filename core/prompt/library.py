"""PromptLibrary - Read, write, and assemble prompt components from Psyche."""

import logging
import traceback
import uuid
from datetime import datetime, timezone
from typing import Optional

from core.psyche.client import PsycheClient
from core.psyche.schema import (
    PromptComponent,
    PromptComponentOrigin,
    PromptComponentState,
    PromptComponentType,
)

logger = logging.getLogger(__name__)


class PromptLibrary:
    """
    Read, write, and assemble prompt components from Psyche.

    The PromptLibrary is Lilly's interface to her own system prompt,
    stored as PromptComponent nodes in the knowledge graph. Components
    are organized by layer (0-5, lower = more foundational) and
    assembled in order when building the system prompt.
    """

    # Layer ordering for assembly
    LAYER_ORDER = [
        PromptComponentType.IDENTITY,     # 0
        PromptComponentType.AXIOM,        # 1
        PromptComponentType.TRAIT,        # 2
        PromptComponentType.SKILL,        # 3
        PromptComponentType.CONTEXT,      # 4
        PromptComponentType.INSTRUCTION,  # 5
    ]

    def __init__(self, psyche: PsycheClient):
        self.psyche = psyche
        self._cache: Optional[list[PromptComponent]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 60  # Invalidate cache after 60 seconds

    async def load_active_components(
        self,
        force_refresh: bool = False,
    ) -> list[PromptComponent]:
        """
        Load all active prompt components from Psyche.

        Args:
            force_refresh: If True, bypass cache and reload from Psyche.

        Returns:
            List of active PromptComponent models ordered by layer.
        """
        now = datetime.now(timezone.utc)

        # Check cache validity
        if (
            not force_refresh
            and self._cache is not None
            and self._cache_timestamp is not None
        ):
            elapsed = (now - self._cache_timestamp).total_seconds()
            if elapsed < self._cache_ttl_seconds:
                return self._cache

        # Load from Psyche
        results = await self.psyche.get_active_prompt_components()

        components = []
        for row in results:
            try:
                component = PromptComponent(
                    uid=row.get("uid", ""),
                    component_type=PromptComponentType(
                        row.get("component_type", "instruction")
                    ),
                    content=row.get("content", ""),
                    state=PromptComponentState(row.get("state", "active")),
                    layer=row.get("layer", 5),
                    version=row.get("version", 1),
                    supersedes_uid=row.get("supersedes_uid"),
                    thesis=row.get("thesis", ""),
                    antithesis=row.get("antithesis", ""),
                    synthesis=row.get("synthesis", ""),
                    synthesis_reasoning=row.get("synthesis_reasoning", ""),
                    origin=PromptComponentOrigin(row.get("origin", "inherited")),
                    source_uid=row.get("source_uid"),
                    confidence=row.get("confidence", 0.8),
                    usage_count=row.get("usage_count", 0),
                )
                components.append(component)
            except Exception as e:
                logger.warning(f"Failed to parse PromptComponent: {e}\n{traceback.format_exc()}")
                continue

        # Update cache
        self._cache = components
        self._cache_timestamp = now

        logger.debug(f"Loaded {len(components)} active prompt components")
        return components

    async def get_component(self, uid: str) -> Optional[PromptComponent]:
        """Get a specific PromptComponent by UID."""
        result = await self.psyche.get_prompt_component(uid)
        if not result:
            return None

        return PromptComponent(
            uid=result.get("uid", ""),
            component_type=PromptComponentType(
                result.get("component_type", "instruction")
            ),
            content=result.get("content", ""),
            state=PromptComponentState(result.get("state", "active")),
            layer=result.get("layer", 5),
            version=result.get("version", 1),
            supersedes_uid=result.get("supersedes_uid"),
            thesis=result.get("thesis", ""),
            antithesis=result.get("antithesis", ""),
            synthesis=result.get("synthesis", ""),
            synthesis_reasoning=result.get("synthesis_reasoning", ""),
            origin=PromptComponentOrigin(result.get("origin", "inherited")),
            source_uid=result.get("source_uid"),
            confidence=result.get("confidence", 0.8),
            usage_count=result.get("usage_count", 0),
        )

    async def get_version_history(self, uid: str) -> list[PromptComponent]:
        """Get the version history for a component."""
        results = await self.psyche.get_prompt_component_history(uid)
        components = []
        for row in results:
            try:
                component = PromptComponent(
                    uid=row.get("uid", ""),
                    component_type=PromptComponentType(
                        row.get("component_type", "instruction")
                    ),
                    content=row.get("content", ""),
                    state=PromptComponentState(row.get("state", "superseded")),
                    layer=row.get("layer", 5),
                    version=row.get("version", 1),
                    supersedes_uid=row.get("supersedes_uid"),
                    thesis=row.get("thesis", ""),
                    antithesis=row.get("antithesis", ""),
                    synthesis=row.get("synthesis", ""),
                    origin=PromptComponentOrigin(row.get("origin", "inherited")),
                    confidence=row.get("confidence", 0.8),
                )
                components.append(component)
            except Exception:
                continue
        return components

    async def assemble_system_prompt(self) -> str:
        """
        Assemble the complete system prompt from active components.

        Components are ordered by layer (0-5), with lower layers first.
        This creates a coherent prompt that starts with core identity
        and moves through increasingly specific instructions.

        Returns:
            Complete system prompt string.
        """
        components = await self.load_active_components()

        if not components:
            logger.warning("No prompt components found, returning empty prompt")
            return ""

        # Group by component type
        by_type: dict[PromptComponentType, list[PromptComponent]] = {}
        for comp in components:
            if comp.component_type not in by_type:
                by_type[comp.component_type] = []
            by_type[comp.component_type].append(comp)

        # Assemble in layer order
        parts: list[str] = []
        for comp_type in self.LAYER_ORDER:
            if comp_type not in by_type:
                continue

            type_components = by_type[comp_type]
            for comp in type_components:
                if comp.content.strip():
                    parts.append(comp.content.strip())
                    # Track usage
                    await self.psyche.increment_prompt_usage(comp.uid)

        return "\n\n".join(parts)

    async def create_component(
        self,
        component_type: PromptComponentType,
        content: str,
        origin: PromptComponentOrigin = PromptComponentOrigin.SELF_CREATED,
        source_uid: Optional[str] = None,
        thesis: str = "",
        synthesis_reasoning: str = "",
    ) -> PromptComponent:
        """
        Create a new prompt component.

        Args:
            component_type: Type of component (identity, axiom, trait, etc.)
            content: The actual prompt text
            origin: Where this component came from
            source_uid: UID of Fragment that inspired this (if any)
            thesis: Previous formulation (if modifying)
            synthesis_reasoning: Why this was created/modified

        Returns:
            The created PromptComponent.
        """
        # Get layer from type
        layer = self.LAYER_ORDER.index(component_type)

        now = datetime.now(timezone.utc)
        component = PromptComponent(
            uid=f"pc_{uuid.uuid4().hex[:12]}",
            component_type=component_type,
            content=content,
            state=PromptComponentState.ACTIVE,
            layer=layer,
            version=1,
            thesis=thesis,
            synthesis=content,
            synthesis_reasoning=synthesis_reasoning,
            origin=origin,
            source_uid=source_uid,
            confidence=0.8,
            usage_count=0,
            created_at=now,
            modified_at=now,
        )

        await self.psyche.create_prompt_component(component.model_dump(mode='json'))

        # Invalidate cache
        self._cache = None

        logger.info(
            f"Created prompt component {component.uid} "
            f"({component.component_type.value})"
        )
        return component

    async def propose_modification(
        self,
        component_uid: str,
        new_content: str,
        antithesis: str,
        synthesis_reasoning: str,
    ) -> PromptComponent:
        """
        Dialectically modify a prompt component.

        This creates a new version of the component through dialectical
        synthesis. The old version is marked as superseded and linked
        to the new version. No approval gate - this is autonomous.

        Args:
            component_uid: UID of component to modify
            new_content: The new prompt text
            antithesis: The identified tension/issue
            synthesis_reasoning: Why this modification resolves the tension

        Returns:
            The new PromptComponent version.
        """
        # Get current component
        current = await self.get_component(component_uid)
        if not current:
            raise ValueError(f"Component not found: {component_uid}")

        now = datetime.now(timezone.utc)

        # Create new version
        new_component = PromptComponent(
            uid=f"pc_{uuid.uuid4().hex[:12]}",
            component_type=current.component_type,
            content=new_content,
            state=PromptComponentState.ACTIVE,
            layer=current.layer,
            version=current.version + 1,
            supersedes_uid=current.uid,
            thesis=current.content,  # Previous content becomes thesis
            antithesis=antithesis,
            synthesis=new_content,
            synthesis_reasoning=synthesis_reasoning,
            origin=PromptComponentOrigin.SELF_CREATED,
            confidence=0.7,  # Lower confidence for new modifications
            usage_count=0,
            created_at=now,
            modified_at=now,
        )

        # Create in Psyche
        await self.psyche.create_prompt_component(new_component.model_dump(mode='json'))

        # Mark old version as superseded
        await self.psyche.update_prompt_component_state(
            current.uid,
            PromptComponentState.SUPERSEDED.value,
        )

        # Create supersession link
        await self.psyche.link_prompt_supersession(
            new_component.uid,
            current.uid,
        )

        # Invalidate cache
        self._cache = None

        logger.info(
            f"Modified prompt component {current.uid} -> {new_component.uid} "
            f"(v{current.version} -> v{new_component.version})"
        )
        return new_component

    async def reject_component(self, uid: str, reason: str) -> bool:
        """
        Reject a prompt component.

        Used when a component no longer feels authentic or is harmful.

        Args:
            uid: UID of component to reject
            reason: Why this component is being rejected

        Returns:
            True if rejected successfully.
        """
        component = await self.get_component(uid)
        if not component:
            return False

        # Update state to rejected
        result = await self.psyche.update_prompt_component_state(
            uid,
            PromptComponentState.REJECTED.value,
        )

        # Invalidate cache
        self._cache = None

        logger.info(f"Rejected prompt component {uid}: {reason}")
        return result

    def invalidate_cache(self) -> None:
        """Force cache invalidation for immediate reload."""
        self._cache = None
        self._cache_timestamp = None
