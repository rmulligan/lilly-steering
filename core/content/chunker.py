"""
Content chunker for Lilly's cognitive system.

Splits parsed documents into fragments suitable for storage
in the knowledge graph and RAG retrieval.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from core.content.parser import ParsedDocument
from core.psyche.schema import Fragment, FragmentState
from core.utils.token_utils import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata about a chunk's position in the original document."""

    document_title: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 0
    source_file: Optional[str] = None


class ContentChunker:
    """
    Split documents into fragments for the knowledge graph.

    Uses semantic chunking that respects:
    - Paragraph boundaries
    - Page boundaries (if available)
    - Sentence boundaries
    - Token limits

    Attributes:
        max_tokens: Maximum tokens per chunk (default 500)
        overlap_tokens: Tokens to overlap between chunks (default 50)
        min_tokens: Minimum tokens for a chunk (default 50)
    """

    def __init__(
        self,
        max_tokens: int = 500,
        overlap_tokens: int = 50,
        min_tokens: int = 50,
    ):
        """
        Initialize the chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Tokens to overlap between chunks for context
            min_tokens: Minimum chunk size (smaller merged with previous)
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        # Split on double newlines or markdown paragraph separators
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence boundary detection
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_page_chunks(self, text: str) -> list[tuple[Optional[int], str]]:
        """
        Extract page chunks from pymupdf4llm output.

        PyMuPDF4LLM adds page markers like:
        -----
        or
        [Page N]

        Returns:
            List of (page_number, content) tuples
        """
        # Check for pymupdf4llm page separator pattern
        page_pattern = r"(?:^|\n)-----+\s*(?:\n|$)"
        parts = re.split(page_pattern, text)

        if len(parts) <= 1:
            # No page separators found
            return [(None, text)]

        result = []
        for i, part in enumerate(parts):
            content = part.strip()
            if content:
                result.append((i + 1, content))

        return result

    def chunk(
        self,
        document: ParsedDocument,
        source: str = "unknown",
        source_file: Optional[str] = None,
    ) -> list[Fragment]:
        """
        Chunk a document into fragments.

        Args:
            document: ParsedDocument to chunk
            source: Source identifier (e.g., "inbox", "research")
            source_file: Original filename

        Returns:
            List of Fragment objects ready for storage
        """
        text = document.content
        if not text.strip():
            return []

        fragments = []
        chunk_index = 0

        # Extract page chunks if available
        page_chunks = self._extract_page_chunks(text)

        for page_num, page_content in page_chunks:
            # Split page into paragraphs
            paragraphs = self._split_into_paragraphs(page_content)

            current_chunk: list[str] = []
            current_tokens = 0

            for para in paragraphs:
                para_tokens = count_tokens(para)

                # If single paragraph exceeds max, split by sentences
                if para_tokens > self.max_tokens:
                    # Flush current chunk first
                    if current_chunk:
                        fragments.append(
                            self._create_fragment(
                                content="\n\n".join(current_chunk),
                                source=source,
                                chunk_index=chunk_index,
                                page_number=page_num,
                                source_file=source_file,
                            )
                        )
                        chunk_index += 1
                        current_chunk = []
                        current_tokens = 0

                    # Split large paragraph by sentences
                    sentences = self._split_into_sentences(para)
                    sent_chunk: list[str] = []
                    sent_tokens = 0

                    for sent in sentences:
                        sent_tok = count_tokens(sent)

                        if sent_tokens + sent_tok <= self.max_tokens:
                            sent_chunk.append(sent)
                            sent_tokens += sent_tok
                        else:
                            # Save current sentence chunk
                            if sent_chunk:
                                fragments.append(
                                    self._create_fragment(
                                        content=" ".join(sent_chunk),
                                        source=source,
                                        chunk_index=chunk_index,
                                        page_number=page_num,
                                        source_file=source_file,
                                    )
                                )
                                chunk_index += 1

                            # Start new chunk with overlap
                            overlap_sents = sent_chunk[-2:] if len(sent_chunk) > 2 else []
                            sent_chunk = overlap_sents + [sent]
                            sent_tokens = sum(count_tokens(s) for s in sent_chunk)

                    # Flush remaining sentences
                    if sent_chunk:
                        fragments.append(
                            self._create_fragment(
                                content=" ".join(sent_chunk),
                                source=source,
                                chunk_index=chunk_index,
                                page_number=page_num,
                                source_file=source_file,
                            )
                        )
                        chunk_index += 1

                elif current_tokens + para_tokens <= self.max_tokens:
                    # Add paragraph to current chunk
                    current_chunk.append(para)
                    current_tokens += para_tokens

                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        fragments.append(
                            self._create_fragment(
                                content="\n\n".join(current_chunk),
                                source=source,
                                chunk_index=chunk_index,
                                page_number=page_num,
                                source_file=source_file,
                            )
                        )
                        chunk_index += 1

                        # Add overlap from end of previous chunk
                        if len(current_chunk) > 1:
                            overlap_para = current_chunk[-1]
                            if count_tokens(overlap_para) <= self.overlap_tokens:
                                current_chunk = [overlap_para, para]
                                current_tokens = count_tokens(overlap_para) + para_tokens
                            else:
                                current_chunk = [para]
                                current_tokens = para_tokens
                        else:
                            current_chunk = [para]
                            current_tokens = para_tokens
                    else:
                        current_chunk = [para]
                        current_tokens = para_tokens

            # Flush remaining content for this page
            if current_chunk:
                content = "\n\n".join(current_chunk)
                # Merge small final chunks with previous if possible
                if (
                    current_tokens < self.min_tokens
                    and fragments
                    and count_tokens(fragments[-1].content) + current_tokens
                    <= self.max_tokens * 1.2
                ):
                    # Merge with previous fragment
                    prev = fragments[-1]
                    merged_content = prev.content + "\n\n" + content
                    fragments[-1] = Fragment(
                        uid=prev.uid,
                        content=merged_content,
                        source=prev.source,
                        state=FragmentState.LIMBO,
                        resonance=0.5,
                        confidence=0.8,
                        created_at=prev.created_at,
                    )
                else:
                    fragments.append(
                        self._create_fragment(
                            content=content,
                            source=source,
                            chunk_index=chunk_index,
                            page_number=page_num,
                            source_file=source_file,
                        )
                    )
                    chunk_index += 1

        # Update total_chunks in metadata
        total = len(fragments)
        logger.info(f"Chunked document into {total} fragments")

        return fragments

    def _create_fragment(
        self,
        content: str,
        source: str,
        chunk_index: int,
        page_number: Optional[int] = None,
        source_file: Optional[str] = None,
    ) -> Fragment:
        """Create a Fragment with unique ID and metadata."""
        # Generate unique ID
        uid = f"frag_{uuid.uuid4().hex[:12]}"

        # Include source info in the source field
        source_info = source
        if source_file:
            source_info = f"{source}:{source_file}"
        if page_number:
            source_info = f"{source_info}:p{page_number}"

        return Fragment(
            uid=uid,
            content=content,
            source=source_info,
            state=FragmentState.LIMBO,
            resonance=0.5,
            confidence=0.8,
            created_at=datetime.now(timezone.utc),
        )


def chunk_document(
    document: ParsedDocument,
    source: str = "unknown",
    max_tokens: int = 500,
) -> list[Fragment]:
    """
    Convenience function to chunk a document.

    Args:
        document: ParsedDocument to chunk
        source: Source identifier
        max_tokens: Maximum tokens per chunk

    Returns:
        List of Fragment objects
    """
    chunker = ContentChunker(max_tokens=max_tokens)
    return chunker.chunk(document, source=source)
