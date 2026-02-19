"""
Document Processing Coordinator - GPU time-sharing orchestration.

Manages sequential GPU-based document processing by:
1. Pausing the cognitive loop
2. Unloading HookedQwen to free GPU memory
3. Loading embedding model for 4096-dim embeddings
4. Using CuratorModel for entity/triple/insight extraction (shared with curation phase)
5. Storing results in FalkorDB
6. Reloading HookedQwen
7. Resuming the cognitive loop

Insight extraction uses the CuratorModel (vLLM-based 8B model) for
higher quality extraction. The CuratorModel is shared with the curation
phase and must be wired in via set_curator_model().
"""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, Protocol

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.content.extractor import InsightExtractor, TripleExtractor
from core.processing.gpu_embedder import GPUEmbedder
from core.processing.hipporag import HippoRAGProcessor

if TYPE_CHECKING:
    from core.cognitive.evocation import EvocationTracker
    from core.cognitive.zettel import ZettelLibrary
    from core.embedding.service import TieredEmbeddingService
    from core.model.curator_model import CuratorModel
    from core.psyche.client import PsycheClient
    from core.psyche.schema import Fragment

    class CognitiveLoopController(Protocol):
        """Protocol for objects that can pause/resume a cognitive loop."""

        async def pause_cognitive_loop(self) -> None:
            """Pause the cognitive loop."""
            ...

        async def resume_cognitive_loop(self) -> None:
            """Resume the cognitive loop."""
            ...

    class UnloadableModel(Protocol):
        """Protocol for models that can be loaded/unloaded to manage GPU memory."""

        @property
        def is_loaded(self) -> bool:
            """Check if the model is currently loaded."""
            ...

        def unload(self) -> None:
            """Unload the model from GPU memory."""
            ...

        async def load(self) -> None:
            """Load the model into GPU memory."""
            ...

logger = logging.getLogger(__name__)

# Batching constants for embedding generation
# The 8B embedding model uses ~16GB VRAM, leaving limited space for inference
LARGE_DOCUMENT_COUNT_THRESHOLD = 50  # Above this, use smaller batch size
LARGE_DOCUMENT_BATCH_SIZE = 8  # Batch size for large document counts
DEFAULT_EMBEDDING_BATCH_SIZE = 32  # Default batch size for normal operation


class CuratorModelAdapter:
    """Adapts CuratorModel to the TextGenerator protocol.

    The TextGenerator protocol expects generate() to return str,
    but CuratorModel.generate() returns SimpleGenerationResult.
    This adapter extracts the .text attribute for compatibility.
    """

    def __init__(self, curator_model: "CuratorModel"):
        self._model = curator_model

    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text string
        """
        result = await self._model.generate(prompt, max_tokens=max_tokens)
        return result.text


class DocumentProcessingCoordinator:
    """Orchestrates GPU time-sharing for document processing.

    This coordinator manages the complex process of sharing GPU memory
    between the main cognitive model (HookedQwen) and document processing
    models (embedding and extraction).

    Processing Flow:
        1. Documents are enqueued via enqueue_document()
        2. When processing starts, cognitive loop is paused
        3. HookedQwen is unloaded to free GPU memory
        4. Embedding model generates 4096-dim embeddings
        5. Extraction model extracts entities and triples
        6. Results are stored in FalkorDB
        7. HookedQwen is reloaded
        8. Cognitive loop is resumed
    """

    def __init__(
        self,
        psyche: "PsycheClient",
        embedding_model_id: str = "Qwen/Qwen3-Embedding-8B",
        extraction_model_id: str = "Qwen/Qwen3-8B",
        device: str = "cuda",
    ):
        """Initialize the document processing coordinator.

        Args:
            psyche: PsycheClient for storing results
            embedding_model_id: Model ID for embeddings
            extraction_model_id: Model ID for extraction
            device: Device to load models on
        """
        self._psyche = psyche
        self._device = device

        # Create processing components (not loaded yet)
        self._embedder = GPUEmbedder(
            model_id=embedding_model_id,
            device=device,
        )
        self._extractor = HippoRAGProcessor(
            model_id=extraction_model_id,
            device=device,
        )

        # Triple extractor for thought memory extraction (uses CuratorModel when set)
        self._triple_extractor = TripleExtractor()

        # CuratorModel reference for insight extraction (set via set_curator_model)
        self._curator_model: Optional["CuratorModel"] = None
        self._curator_adapter: Optional[CuratorModelAdapter] = None

        # Queue for documents to process
        self._queue: asyncio.Queue[Fragment] = asyncio.Queue()
        self._processing = False
        self._processing_lock = asyncio.Lock()

        # Queue for thoughts to extract memories from
        self._thought_queue: asyncio.Queue[Fragment] = asyncio.Queue()
        self._thought_processing_threshold = 5  # Process after N thoughts

        # References set during wiring (set by Lilly)
        self._cognitive_loop: Optional[CognitiveLoopController] = None
        self._hooked_model: Optional[UnloadableModel] = None
        self._evocation_tracker: Optional["EvocationTracker"] = None
        self._zettel_library: Optional["ZettelLibrary"] = None
        self._embedding_service: Optional["TieredEmbeddingService"] = None

        # Insight extractor for external documents (uses CuratorModel via adapter)
        self._insight_extractor = InsightExtractor()

        # Narration callback for announcing processing status
        self._narration_callback: Optional[Callable[[str], Awaitable[bool]]] = None

    def set_cognitive_loop(self, cognitive_loop: "CognitiveLoopController") -> None:
        """Set reference to the cognitive loop controller (Lilly).

        Args:
            cognitive_loop: Object with pause_cognitive_loop/resume_cognitive_loop methods
        """
        self._cognitive_loop = cognitive_loop

    def set_hooked_model(self, model: "UnloadableModel") -> None:
        """Set reference to the HookedQwen model.

        Args:
            model: Model with is_loaded, load, and unload methods
        """
        self._hooked_model = model

    def set_evocation_tracker(self, tracker: "EvocationTracker") -> None:
        """Set reference to the evocation tracker for SAE→Entity linking.

        Args:
            tracker: EvocationTracker for learning SAE feature associations
        """
        self._evocation_tracker = tracker

    def set_zettel_library(
        self,
        library: "ZettelLibrary",
        embedding_service: "TieredEmbeddingService",
    ) -> None:
        """Set reference to the Zettel library for insight storage.

        Args:
            library: ZettelLibrary for storing extracted insights
            embedding_service: Embedding service for computing insight embeddings
        """
        self._zettel_library = library
        self._embedding_service = embedding_service

    def set_narration_callback(
        self, callback: Callable[[str], Awaitable[bool]]
    ) -> None:
        """Set callback for narrating processing status.

        The callback is called at key points during document processing
        to keep the audio stream active with status updates.

        Args:
            callback: Async function that takes a string and returns bool
        """
        self._narration_callback = callback

    def set_curator_model(self, model: "CuratorModel") -> None:
        """Set the CuratorModel for insight extraction.

        The CuratorModel (vLLM-based) provides higher quality extraction
        than the previous CPU-based 0.5B model. It's used for extracting
        insights from external documents.

        Args:
            model: CuratorModel instance for text generation
        """
        self._curator_model = model
        self._curator_adapter = CuratorModelAdapter(model) if model else None

    async def _narrate(self, text: str) -> None:
        """Narrate processing status if callback is set."""
        if self._narration_callback:
            try:
                await self._narration_callback(text)
            except Exception as e:
                logger.warning(f"Narration callback failed: {e}")

    @property
    def is_processing(self) -> bool:
        """Check if document processing is currently active."""
        return self._processing

    @property
    def queue_size(self) -> int:
        """Get the number of documents waiting in the queue."""
        return self._queue.qsize()

    @property
    def thought_queue_size(self) -> int:
        """Get the number of thoughts waiting for memory extraction."""
        return self._thought_queue.qsize()

    async def enqueue_thought_for_memory(self, fragment: "Fragment") -> bool:
        """Queue a thought for memory extraction.

        Thoughts are processed in batches to avoid frequent model swapping.
        Returns True if processing was triggered.

        Args:
            fragment: Thought fragment to extract memories from

        Returns:
            True if batch processing was triggered
        """
        await self._thought_queue.put(fragment)
        logger.info(f"Enqueued thought for memory: {fragment.uid} (queue: {self.thought_queue_size})")

        # Trigger processing when threshold reached
        if self.thought_queue_size >= self._thought_processing_threshold:
            if not self._processing:
                asyncio.create_task(self._process_thought_memories())
                return True
        return False

    async def _process_thought_memories(self) -> None:
        """Process queued thoughts to extract self-created memories.

        This extracts triples from Lilly's thoughts, creating her own
        knowledge graph entries - memories she authors herself.

        Uses CPU-based extraction model to avoid GPU contention, reducing
        the number of slow GPU model load/unload cycles. Only embeddings
        require GPU model swapping.
        """
        async with self._processing_lock:
            if self._processing:
                return  # Already processing
            self._processing = True

        try:
            if self._thought_queue.empty():
                return

            # Collect all queued thoughts
            thoughts = []
            while not self._thought_queue.empty():
                thought = await self._thought_queue.get()
                thoughts.append(thought)

            if not thoughts:
                return

            logger.info(f"Processing {len(thoughts)} thoughts for memory extraction")

            # Step 1: Extract memories using CuratorModel (if available)
            await self._extract_thought_memories(thoughts)

            # Step 2b: Link SAE features to entities (CPU-based, non-blocking)
            # This processes pending thought fragments that have SAE snapshots
            await self.link_pending_evocations(limit=50)

            # Step 3: Generate embeddings (requires GPU swap)
            # Pause cognitive loop only for embedding generation
            if self._cognitive_loop is not None:
                logger.info("Pausing cognitive loop for thought embedding...")
                await self._cognitive_loop.pause_cognitive_loop()

            if self._hooked_model is not None and self._hooked_model.is_loaded:
                logger.info("Unloading HookedQwen for embedding...")
                await self._hooked_model.unload()
                self._clear_gpu_memory()

            try:
                await self._embed_thoughts(thoughts)
            finally:
                # Reload main model and resume cognitive loop
                if self._hooked_model is not None:
                    logger.info("Reloading HookedQwen...")
                    await self._hooked_model.load()

                if self._cognitive_loop is not None:
                    logger.info("Resuming cognitive loop...")
                    await self._cognitive_loop.resume_cognitive_loop()

        except Exception as e:
            logger.error(f"Thought memory processing failed: {e}", exc_info=True)
        finally:
            self._processing = False

    async def _embed_thoughts(self, thoughts: list["Fragment"]) -> None:
        """Generate embeddings for thought fragments.

        Creates golden embeddings for Lilly's thoughts so they can be
        retrieved via semantic search for future context.

        Args:
            thoughts: List of thought fragments to embed
        """
        if not thoughts:
            return

        logger.info(f"Loading embedding model for {len(thoughts)} thoughts...")
        self._embedder.load()

        try:
            # Extract texts
            texts = [t.content for t in thoughts]

            # Generate embeddings in batch
            embeddings = await self._embedder.embed(texts)

            # Update fragments with embeddings
            for thought, embedding in zip(thoughts, embeddings):
                thought.embedding = embedding
                await self._update_fragment_embedding(thought)

            logger.info(f"Generated embeddings for {len(thoughts)} thoughts")

        finally:
            self._embedder.unload()
            self._clear_gpu_memory()

    async def _extract_thought_memories(self, thoughts: list["Fragment"]) -> None:
        """Extract triples from thoughts.

        Uses the CuratorModel (vLLM-based) when available for higher quality
        triple extraction. Falls back to returning empty if no model is set.
        This focuses on self-referential knowledge - what Lilly discovers
        about herself.

        Args:
            thoughts: List of thought fragments to process
        """
        if not thoughts:
            return

        # Check if CuratorModel is available for triple extraction
        if self._curator_model is None:
            logger.warning("No CuratorModel set for thought memory extraction - skipping")
            return
        if not self._curator_model.is_loaded:
            logger.warning("CuratorModel not loaded for thought memory extraction - skipping")
            return

        # Set up the triple extractor with cached CuratorModel adapter
        self._triple_extractor.set_model(self._curator_adapter)

        logger.info(f"Extracting memories from {len(thoughts)} thoughts...")

        total_triples = 0

        for thought in thoughts:
            # Extract triples using CuratorModel
            # We skip entity extraction for thoughts - focus on relationships
            triples = await self._triple_extractor.extract(
                thought.content,
                source_fragment_uid=thought.uid,
            )

            # Store triples as self-created memories
            for triple in triples:
                try:
                    # Mark as self-created memory with high confidence
                    triple.confidence = 0.9
                    await self._psyche.create_triple(triple)
                    total_triples += 1
                except Exception as e:
                    logger.warning(f"Failed to store memory triple: {e}")

        logger.info(f"Extracted {total_triples} memory triples from {len(thoughts)} thoughts")

    async def link_pending_evocations(self, limit: int = 50) -> int:
        """Link SAE features to entities for pending thought fragments.

        This processes fragments that have SAE snapshots but haven't had their
        entities linked to those features yet. Called periodically to build
        the feature→entity association graph.

        The linking happens AFTER entity extraction, so entities must exist
        before this method is useful. For thoughts, entities come from
        HippoRAG processing or from the triples we extract.

        Args:
            limit: Maximum fragments to process

        Returns:
            Number of associations learned
        """
        if self._evocation_tracker is None:
            logger.debug("No evocation tracker - skipping evocation linking")
            return 0

        try:
            # Get fragments with SAE snapshots that need linking
            pending = await self._psyche.get_unlinked_thought_fragments(limit=limit)
            if not pending:
                return 0

            total_associations = 0
            for fragment_uid, features in pending:
                # Get entities associated with this fragment
                entities_data = await self._psyche.get_entities_for_fragment(fragment_uid)
                if not entities_data:
                    continue

                # Convert to Entity objects for type safety
                from core.psyche.schema import Entity
                entities = [
                    Entity(
                        uid=e["uid"],
                        name=e["name"],
                        entity_type=e.get("entity_type", "concept"),
                    )
                    for e in entities_data
                ]

                # Learn associations between SAE features and entities
                count = await self._evocation_tracker.learn_associations_batch(
                    features=features,
                    entities=entities,
                )
                total_associations += count

            if total_associations > 0:
                logger.info(f"Learned {total_associations} SAE→Entity associations from {len(pending)} fragments")

                # Persist dirty evocations
                persisted = await self._evocation_tracker.persist_dirty()
                if persisted:
                    logger.debug(f"Persisted {persisted} evocation edges")

            return total_associations

        except Exception as e:
            logger.warning(f"Evocation linking failed: {e}")
            return 0

    async def enqueue_document(self, fragment: "Fragment") -> None:
        """Add a document to the processing queue.

        If not currently processing, starts the processing loop.

        Args:
            fragment: Fragment to process
        """
        await self._queue.put(fragment)
        logger.info(f"Enqueued document for processing: {fragment.uid} (queue size: {self.queue_size})")

        # Start processing if not already running
        if not self._processing:
            asyncio.create_task(self._process_queue())

    async def _process_queue(self) -> None:
        """Process all queued documents sequentially.

        This is the main processing loop that handles GPU time-sharing.
        """
        async with self._processing_lock:
            if self._processing:
                return  # Already processing
            self._processing = True

        try:
            # Loop until the queue is empty to process all documents,
            # including those that arrive during processing.
            while not self._queue.empty():
                await self._run_processing_pipeline()
        except Exception as e:
            logger.error(f"Document processing worker failed: {e}", exc_info=True)
        finally:
            self._processing = False

    async def _run_processing_pipeline(self) -> None:
        """Execute the full processing pipeline.

        This method handles:
        1. Pausing cognitive loop
        2. Model loading/unloading
        3. Processing all queued documents
        4. Resuming cognitive loop
        """
        if self._queue.empty():
            return

        logger.info(f"Starting document processing pipeline ({self.queue_size} documents)")

        # Step 1: Pause cognitive loop
        if self._cognitive_loop is not None:
            logger.info("Pausing cognitive loop for document processing...")
            await self._cognitive_loop.pause_cognitive_loop()

        # Step 2: Unload HookedQwen to free GPU memory
        if self._hooked_model is not None and self._hooked_model.is_loaded:
            logger.info("Unloading HookedQwen to free GPU memory...")
            await self._hooked_model.unload()
            self._clear_gpu_memory()

        try:
            # Step 3: Collect all fragments from queue
            fragments = []
            while not self._queue.empty():
                fragment = await self._queue.get()
                fragments.append(fragment)

            logger.info(f"Processing {len(fragments)} documents")

            # Narrate start of processing
            await self._narrate(
                f"I'm reading new material. {len(fragments)} fragments to process. "
                "Let me absorb this for a moment."
            )

            # Step 4: Generate embeddings for all fragments
            await self._generate_embeddings(fragments)

            # Narrate progress
            await self._narrate(
                "Embeddings generated. Now extracting knowledge and connections."
            )

            # Step 5: Extract entities and triples
            await self._extract_knowledge(fragments)

            # Step 6: Extract insights and store in Zettel library
            if self._zettel_library is not None:
                await self._extract_insights(fragments)

            logger.info(f"Document processing complete for {len(fragments)} documents")

            # Step 7: Mark fragments as VERIFIED (no longer LIMBO)
            from core.psyche.schema import FragmentState
            verified_count = 0
            for fragment in fragments:
                try:
                    success = await self._psyche.update_fragment_state(
                        fragment.uid, FragmentState.VERIFIED
                    )
                    if success:
                        verified_count += 1
                except Exception as e:
                    logger.warning(f"Failed to update fragment state for {fragment.uid}: {e}")
            logger.info(f"Marked {verified_count}/{len(fragments)} fragments as VERIFIED")

            # Narrate completion
            await self._narrate(
                f"Finished processing {len(fragments)} fragments. "
                "Returning to my thoughts."
            )

        finally:
            # Step 6: Reload HookedQwen
            if self._hooked_model is not None:
                logger.info("Reloading HookedQwen...")
                await self._hooked_model.load()

            # Step 7: Resume cognitive loop
            if self._cognitive_loop is not None:
                logger.info("Resuming cognitive loop...")
                await self._cognitive_loop.resume_cognitive_loop()

    async def _generate_embeddings(self, fragments: list["Fragment"]) -> None:
        """Generate embeddings for all fragments.

        Args:
            fragments: List of fragments to embed
        """
        if not fragments:
            return

        logger.info(f"Loading embedding model for {len(fragments)} fragments...")
        self._embedder.load()

        try:
            # Extract texts
            texts = [f.content for f in fragments]

            # Use smaller batch_size for large document counts to prevent OOM
            # When document count exceeds LARGE_DOCUMENT_COUNT_THRESHOLD,
            # use LARGE_DOCUMENT_BATCH_SIZE instead of DEFAULT_EMBEDDING_BATCH_SIZE
            batch_size = (
                LARGE_DOCUMENT_BATCH_SIZE
                if len(texts) > LARGE_DOCUMENT_COUNT_THRESHOLD
                else DEFAULT_EMBEDDING_BATCH_SIZE
            )
            logger.info(f"Generating embeddings with batch_size={batch_size}")

            # Generate embeddings in batch
            embeddings = await self._embedder.embed(texts, batch_size=batch_size)

            # Update fragments and store in database
            for fragment, embedding in zip(fragments, embeddings):
                fragment.embedding = embedding

                # Update fragment in database with embedding
                await self._update_fragment_embedding(fragment)

            logger.info(f"Generated embeddings for {len(fragments)} fragments")

        finally:
            # Unload embedding model
            self._embedder.unload()
            self._clear_gpu_memory()

    async def _extract_knowledge(self, fragments: list["Fragment"]) -> None:
        """Extract entities and triples from all fragments.

        Args:
            fragments: List of fragments to process
        """
        if not fragments:
            return

        logger.info(f"Loading extraction model for {len(fragments)} fragments...")
        self._extractor.load()

        try:
            total_entities = 0
            total_triples = 0

            for fragment in fragments:
                # Extract entities and triples
                entities, triples = await self._extractor.process_fragment(fragment)

                # Store entities
                for entity in entities:
                    try:
                        await self._psyche.create_entity(entity)
                        # Link entity to fragment
                        await self._psyche.link_fragment_entity(fragment.uid, entity.uid)
                        total_entities += 1
                    except Exception as e:
                        logger.warning(f"Failed to store entity {entity.name}: {e}")

                # Store triples
                for triple in triples:
                    try:
                        await self._psyche.create_triple(triple)
                        total_triples += 1
                    except Exception as e:
                        logger.warning(f"Failed to store triple: {e}")

            logger.info(f"Extracted {total_entities} entities and {total_triples} triples")

        finally:
            # Unload extraction model
            self._extractor.unload()
            self._clear_gpu_memory()

    async def _extract_insights(self, fragments: list["Fragment"]) -> None:
        """Extract insights from fragments and store in Zettel library.

        Args:
            fragments: List of fragments to extract insights from
        """
        if not fragments or not self._zettel_library:
            return

        logger.info(f"Extracting insights from {len(fragments)} fragments...")

        # Check if CuratorModel is available and loaded
        if self._curator_model is None:
            logger.warning("No CuratorModel set for insight extraction - skipping")
            return

        if not self._curator_model.is_loaded:
            logger.warning("CuratorModel not loaded for insight extraction - skipping")
            return

        # Set up the insight extractor with cached CuratorModel adapter
        self._insight_extractor.set_model(self._curator_adapter)

        total_insights = 0

        try:
            for fragment in fragments:
                # Extract insights from fragment text
                try:
                    insights = await self._insight_extractor.extract(
                        fragment.content,
                        max_insights=5,
                    )
                except Exception as e:
                    logger.warning(f"Insight extraction failed for fragment {fragment.uid}: {e}")
                    continue

                # Determine source type from fragment metadata or default
                source_type = self._determine_source_type(fragment)

                # Determine concept from fragment (use first entity or "general")
                concept = await self._determine_concept(fragment)

                # Store each insight as a Zettel
                for insight in insights:
                    try:
                        await self._zettel_library.store_zettel(
                            insight_text=insight.insight_text,
                            source_type=source_type,
                            source_uid=fragment.uid,
                            concept=concept,
                            question_text=insight.question_text,
                            cycle=None,  # Not from cognitive cycle
                            sae_features=None,  # No SAE features for external docs
                            emerged_from=None,  # No lineage for external docs
                        )
                        total_insights += 1
                    except Exception as e:
                        logger.warning(f"Failed to store insight: {e}")

            logger.info(f"Extracted and stored {total_insights} insights")

        except Exception as e:
            logger.error(f"Insight extraction failed: {e}")

    def _determine_source_type(self, fragment: "Fragment") -> str:
        """Determine source type from fragment.

        Args:
            fragment: Fragment to analyze

        Returns:
            Source type: "letter", "research", or "reflection"
        """
        # Check fragment metadata or source if available
        source = getattr(fragment, "source", "") or ""
        source_lower = source.lower()

        if "letter" in source_lower or "correspondence" in source_lower:
            return "letter"
        elif "research" in source_lower or "paper" in source_lower:
            return "research"
        elif "reflection" in source_lower or "journal" in source_lower:
            return "reflection"
        else:
            # Default to research for external documents
            return "research"

    async def _determine_concept(self, fragment: "Fragment") -> str:
        """Determine concept from fragment entities.

        Args:
            fragment: Fragment to analyze

        Returns:
            Concept string
        """
        # Try to get entities linked to this fragment
        try:
            entities = await self._psyche.get_entities_for_fragment(fragment.uid)
            if entities:
                # Use the first concept-type entity, or first entity
                for entity in entities:
                    if hasattr(entity, "entity_type") and entity.entity_type == "CONCEPT":
                        return entity.name.lower()
                return entities[0].name.lower()
        except Exception:
            pass

        # Fallback: extract a concept from text
        text_lower = fragment.content.lower() if fragment.content else ""
        # Simple heuristic: first capitalized multi-word phrase
        import re
        match = re.search(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', fragment.content or "")
        if match:
            return match.group(1).lower()

        return "general"

    async def _update_fragment_embedding(self, fragment: "Fragment") -> None:
        """Update a fragment's embedding in the database.

        Stores both the full golden embedding (4096-dim) and a truncated
        retrieval embedding (first 1024-dim) for MRL-compatible search.
        This enables runtime queries with 1024-dim embeddings while
        preserving the full golden embedding for future use.

        Args:
            fragment: Fragment with embedding to update
        """
        if fragment.embedding is None:
            return

        # Validate embedding values are numeric (security: prevent injection)
        # FalkorDB's vecf32() requires inline values - parameterization not supported
        embedding_values = [float(x) for x in fragment.embedding]
        embedding_str = ", ".join(str(x) for x in embedding_values)

        # Truncate to first 1024 dimensions for MRL-compatible retrieval
        # Qwen3-Embedding supports Matryoshka Representation Learning,
        # so the first 1024 dims are meaningful for similarity search
        retrieval_values = [float(x) for x in fragment.embedding[:1024]]
        retrieval_str = ", ".join(str(x) for x in retrieval_values)

        cypher = f"""
        MATCH (f:Fragment {{uid: $uid}})
        SET f.embedding = vecf32([{embedding_str}]),
            f.embedding_retrieval = vecf32([{retrieval_str}])
        """
        params = {
            "uid": fragment.uid,
        }

        try:
            await self._psyche.execute(cypher, params)
        except Exception as e:
            logger.warning(f"Failed to update fragment embedding: {e}")

    def _clear_gpu_memory(self) -> None:
        """Clear GPU memory cache."""
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")


async def process_document_batch(
    fragments: list["Fragment"],
    psyche: "PsycheClient",
    cognitive_loop: Optional["CognitiveLoopController"] = None,
    hooked_model: Optional["UnloadableModel"] = None,
) -> None:
    """Convenience function to process a batch of documents.

    Creates a temporary coordinator and processes all fragments.

    Args:
        fragments: List of fragments to process
        psyche: PsycheClient instance
        cognitive_loop: Optional object with pause/resume methods
        hooked_model: Optional model with load/unload methods
    """
    coordinator = DocumentProcessingCoordinator(psyche=psyche)
    coordinator.set_cognitive_loop(cognitive_loop)
    coordinator.set_hooked_model(hooked_model)

    for fragment in fragments:
        await coordinator.enqueue_document(fragment)
