"""
HippoRAG Processor - GPU-based knowledge extraction.

Implements HippoRAG-style entity and triple extraction using
Qwen3-8B for high-quality extraction.

HippoRAG is based on:
- Entity extraction with named entity recognition
- Triple extraction (subject-predicate-object relationships)
- Optional Personalized PageRank for entity linking (future)
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import uuid
from typing import TYPE_CHECKING, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from core.psyche.schema import Entity, Triple, Fragment
    from core.psyche.client import PsycheClient
    from core.embedding.service import TieredEmbeddingService

logger = logging.getLogger(__name__)

# Default extraction model - Qwen3-8B for high-quality extraction
DEFAULT_EXTRACTION_MODEL = "Qwen/Qwen3-8B"

# Prompt templates using Qwen3 chat format
ENTITY_PROMPT = """<|im_start|>system
You are an expert entity extraction assistant. Extract named entities from text and return them as a JSON array. Be thorough and precise.
<|im_end|>
<|im_start|>user
Extract all named entities from the following text. Return a JSON array with this exact format:
[{{"name": "Entity Name", "type": "TYPE", "description": "Brief description"}}]

Valid types: PERSON, ORGANIZATION, CONCEPT, PLACE, THING, EVENT, WORK

Text to analyze:
{text}

Return ONLY the JSON array, nothing else.
<|im_end|>
<|im_start|>assistant
["""

TRIPLE_PROMPT = """<|im_start|>system
You are an expert knowledge extraction assistant. Extract factual relationships from text as subject-predicate-object triples. Be thorough and precise.
<|im_end|>
<|im_start|>user
Extract all factual relationships from the following text as triples. Return a JSON array with this exact format:
[{{"subject": "Subject Entity", "predicate": "relationship verb", "object": "Object Entity"}}]

Rules:
- Subject and object should be specific named entities or concepts from the text
- Predicate should be a clear verb or relationship phrase
- Only extract explicit facts stated in the text, not inferences
- Include all meaningful relationships

Text to analyze:
{text}

Return ONLY the JSON array, nothing else.
<|im_end|>
<|im_start|>assistant
["""


def _extract_json_array(text: str) -> list[dict]:
    """Extract JSON array from model output.

    The model output may start with '[' already due to prompt priming.

    Args:
        text: Model output text

    Returns:
        Parsed JSON array or empty list if parsing fails
    """
    text = text.strip()

    # Handle case where response starts without '[' (we primed it)
    if not text.startswith('['):
        text = '[' + text

    # Find the end of the JSON array
    bracket_count = 0
    end_pos = -1
    for i, char in enumerate(text):
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0:
                end_pos = i
                break

    if end_pos == -1:
        # Try to close the array if it's truncated
        text = text.rstrip(',\n ') + ']'
        end_pos = len(text) - 1

    json_str = text[:end_pos + 1]

    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
        logger.warning(f"JSON parsed but not a list: {type(result)}")
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"HippoRAG JSON parse failed: {e}")
        logger.debug(f"Failed JSON text: {json_str[:500]}")
        return []


class HippoRAGProcessor:
    """GPU-based knowledge extraction using HippoRAG approach.

    Uses Qwen3-8B for high-quality entity and triple extraction.
    Model is loaded on-demand and can be explicitly unloaded to free
    GPU memory for other processing.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_EXTRACTION_MODEL,
        device: str = "cuda",
        max_new_tokens: int = 2048,
        max_input_length: int = 6000,
    ):
        """Initialize the HippoRAG processor.

        Args:
            model_id: HuggingFace model ID for extraction
            device: Device to load model on ("cuda" or "cpu")
            max_new_tokens: Maximum tokens to generate
            max_input_length: Maximum input text length (chars)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch and transformers required for HippoRAGProcessor")

        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length

        self._model = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the extraction model to GPU."""
        if self._model is not None:
            logger.warning("Model already loaded")
            return

        logger.info(f"Loading extraction model: {self.model_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )

        logger.info(f"Extraction model loaded on {self.device}")

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if self._model is None:
            return

        logger.info("Unloading extraction model...")

        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None

        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Extraction model unloaded")

    async def _generate(self, prompt: str) -> str:
        """Generate text from prompt asynchronously.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        loop = asyncio.get_running_loop()

        def _run_generation():
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Deterministic for extraction
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            new_tokens = outputs[0][inputs.input_ids.shape[1]:]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return await loop.run_in_executor(None, _run_generation)

    async def extract_entities(
        self,
        text: str,
        source_fragment_uid: Optional[str] = None,
    ) -> list["Entity"]:
        """Extract named entities from text.

        Args:
            text: Text to extract entities from
            source_fragment_uid: Optional UID of source fragment

        Returns:
            List of Entity objects
        """
        from core.psyche.schema import Entity

        if not text or len(text.strip()) < 10:
            return []

        # Truncate text to max input length
        truncated_text = text[:self.max_input_length]
        prompt = ENTITY_PROMPT.format(text=truncated_text)
        response = await self._generate(prompt)

        logger.debug(f"Entity extraction response: {response[:300]}...")

        entities = []
        for item in _extract_json_array(response):
            if not isinstance(item, dict):
                continue

            name = item.get("name", "").strip()
            if not name:
                continue

            entity = Entity(
                uid=f"entity:{uuid.uuid4().hex[:12]}",
                name=name,
                entity_type=item.get("type", "CONCEPT").upper(),
                description=item.get("description"),
            )
            entities.append(entity)

        if not entities:
            logger.warning(f"Entity extraction yielded 0 entities from {len(text)} chars")
        else:
            logger.info(f"Extracted {len(entities)} entities from text ({len(text)} chars)")
        return entities

    async def extract_triples(
        self,
        text: str,
        source_fragment_uid: Optional[str] = None,
    ) -> list["Triple"]:
        """Extract subject-predicate-object triples from text.

        Args:
            text: Text to extract triples from
            source_fragment_uid: Optional UID of source fragment

        Returns:
            List of Triple objects
        """
        from core.psyche.schema import Triple

        if not text or len(text.strip()) < 10:
            return []

        # Truncate text to max input length
        truncated_text = text[:self.max_input_length]
        prompt = TRIPLE_PROMPT.format(text=truncated_text)
        response = await self._generate(prompt)

        logger.debug(f"Triple extraction response: {response[:300]}...")

        triples = []
        for item in _extract_json_array(response):
            if not isinstance(item, dict):
                continue

            subject = item.get("subject", "").strip()
            predicate = item.get("predicate", "").strip()
            obj = item.get("object", "").strip()

            if not (subject and predicate and obj):
                continue

            triple = Triple(
                uid=f"triple:{uuid.uuid4().hex[:12]}",
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=0.85,  # Higher confidence with Qwen3-8B
                source_fragment_uid=source_fragment_uid,
            )
            triples.append(triple)

        if not triples:
            logger.warning(f"Triple extraction yielded 0 triples from {len(text)} chars")
        else:
            logger.info(f"Extracted {len(triples)} triples from text ({len(text)} chars)")
        return triples

    async def process_text(
        self,
        text: str,
        source_fragment_uid: Optional[str] = None,
    ) -> tuple[list["Entity"], list["Triple"]]:
        """Extract both entities and triples from text.

        Args:
            text: Text to process
            source_fragment_uid: Optional UID of source fragment

        Returns:
            Tuple of (entities, triples)
        """
        entities = await self.extract_entities(text, source_fragment_uid)
        triples = await self.extract_triples(text, source_fragment_uid)
        return entities, triples

    async def process_fragment(
        self,
        fragment: "Fragment",
    ) -> tuple[list["Entity"], list["Triple"]]:
        """Process a fragment to extract knowledge.

        Args:
            fragment: Fragment to process

        Returns:
            Tuple of (entities, triples)
        """
        return await self.process_text(
            fragment.content,
            source_fragment_uid=fragment.uid,
        )


# Module-level processor instance for reuse
_processor: Optional[HippoRAGProcessor] = None


async def process_passage(
    text: str,
    psyche: "PsycheClient",
    embedder: Optional["TieredEmbeddingService"] = None,
    source_uid: Optional[str] = None,
) -> tuple[list["Entity"], list["Triple"]]:
    """Convenience function to process a passage and persist to psyche.

    This function is designed to be called from the cognitive orchestrator
    during the integration phase. It:
    1. Extracts entities and triples from the text
    2. Persists them to the knowledge graph via psyche

    Note: Model is loaded on first use and kept warm for efficiency.
    Call shutdown_processor() to release GPU memory when done.

    Args:
        text: Text passage to process
        psyche: PsycheClient for graph persistence
        embedder: Optional embedder for entity embeddings (not currently used)
        source_uid: Optional source fragment UID

    Returns:
        Tuple of (entities, triples) that were extracted and persisted
    """
    global _processor

    if not text or len(text.strip()) < 20:
        logger.debug("Text too short for HippoRAG processing")
        return [], []

    # Lazy load processor
    if _processor is None:
        _processor = HippoRAGProcessor()

    # Load model if needed
    if not _processor.is_loaded:
        logger.info("Loading HippoRAG model for passage processing...")
        _processor.load()

    # Extract entities and triples
    entities, triples = await _processor.process_text(text, source_uid)

    # Persist to psyche
    for entity in entities:
        try:
            await psyche.upsert_entity(entity)
        except Exception as e:
            logger.warning(f"Failed to persist entity {entity.name}: {e}")

    for triple in triples:
        try:
            await psyche.create_triple(triple)
        except Exception as e:
            logger.warning(f"Failed to persist triple {triple.subject}->{triple.predicate}->{triple.object}: {e}")

    logger.info(f"HippoRAG processed passage: {len(entities)} entities, {len(triples)} triples")
    return entities, triples


def shutdown_processor() -> None:
    """Shutdown the HippoRAG processor and release GPU memory.

    Call this when HippoRAG processing is complete and you need to
    free GPU memory for other models.
    """
    global _processor

    if _processor is not None and _processor.is_loaded:
        _processor.unload()
        _processor = None
        logger.info("HippoRAG processor shutdown complete")
