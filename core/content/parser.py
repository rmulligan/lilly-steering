"""
Content parser for Lilly's cognitive system.

Converts various document formats to structured text for processing.
Uses PyMuPDF4LLM for PDF conversion to clean markdown.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pymupdf4llm

logger = logging.getLogger(__name__)


# MIME type constants
MIME_PDF = "application/pdf"
MIME_TEXT = "text/plain"
MIME_CSV = "text/csv"
MIME_HTML = "text/html"
MIME_MARKDOWN = "text/markdown"
MIME_URL = "text/x-url"  # For URL files or direct URL content


@dataclass
class ParsedDocument:
    """Result of parsing a document."""

    content: str
    source_mime_type: str
    output_format: str  # "markdown" or "text"
    page_count: Optional[int] = None
    title: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    parsed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        """Character count."""
        return len(self.content)


class ContentParser:
    """
    Parse various document formats to structured text.

    Supports:
    - PDF: Converted to markdown with PyMuPDF4LLM
    - Plain text: Passed through with minimal processing
    - CSV: Converted to markdown table
    - HTML: Stripped of tags (basic)

    Attributes:
        page_chunks: Whether to preserve page boundaries (default True)
    """

    def __init__(self, page_chunks: bool = True):
        """
        Initialize the parser.

        Args:
            page_chunks: Include page separators in output
        """
        self.page_chunks = page_chunks

    async def parse_pdf(self, content: bytes) -> ParsedDocument:
        """
        Convert PDF to markdown using PyMuPDF4LLM.

        This produces clean markdown suitable for LLM processing,
        preserving structure, headings, lists, and tables.

        Args:
            content: PDF file bytes

        Returns:
            ParsedDocument with markdown content
        """
        loop = asyncio.get_running_loop()

        def _parse() -> tuple[str, int]:
            # Open PDF from bytes
            doc = pymupdf4llm.pymupdf.open(stream=content, filetype="pdf")
            page_count = len(doc)

            # Convert to markdown
            result = pymupdf4llm.to_markdown(
                doc,
                page_chunks=self.page_chunks,
                write_images=False,  # Don't write images to disk
                show_progress=False,
            )

            doc.close()

            # Handle page_chunks=True returning list of dicts
            if isinstance(result, list):
                # Each item is a dict with 'text' key (and optionally 'metadata')
                md_text = "\n\n".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in result
                )
            else:
                md_text = result

            return md_text, page_count

        try:
            md_content, page_count = await loop.run_in_executor(None, _parse)

            return ParsedDocument(
                content=md_content,
                source_mime_type=MIME_PDF,
                output_format="markdown",
                page_count=page_count,
            )

        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise ValueError(f"PDF parsing failed: {e}") from e

    async def parse_text(self, content: str | bytes) -> ParsedDocument:
        """
        Process plain text content.

        Minimal processing - preserves content as-is with whitespace normalization.

        Args:
            content: Text content (str or UTF-8 bytes)

        Returns:
            ParsedDocument with text content
        """
        if isinstance(content, bytes):
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = content.decode("latin-1")
        else:
            text = content

        # Normalize whitespace
        text = text.strip()

        return ParsedDocument(
            content=text,
            source_mime_type=MIME_TEXT,
            output_format="text",
        )

    async def parse_csv(self, content: str | bytes) -> ParsedDocument:
        """
        Convert CSV to markdown table.

        Args:
            content: CSV content (str or UTF-8 bytes)

        Returns:
            ParsedDocument with markdown table
        """
        if isinstance(content, bytes):
            text = content.decode("utf-8")
        else:
            text = content

        lines = text.strip().split("\n")
        if not lines:
            return ParsedDocument(
                content="",
                source_mime_type=MIME_CSV,
                output_format="markdown",
            )

        # Build markdown table
        md_lines = []

        # Header row
        header = lines[0].split(",")
        md_lines.append("| " + " | ".join(h.strip() for h in header) + " |")
        md_lines.append("| " + " | ".join("---" for _ in header) + " |")

        # Data rows
        for line in lines[1:]:
            cells = line.split(",")
            md_lines.append("| " + " | ".join(c.strip() for c in cells) + " |")

        return ParsedDocument(
            content="\n".join(md_lines),
            source_mime_type=MIME_CSV,
            output_format="markdown",
        )

    async def parse_html(self, content: str | bytes) -> ParsedDocument:
        """
        Convert HTML to plain text.

        Basic tag stripping - for more complex HTML, consider
        using a dedicated library like BeautifulSoup.

        Args:
            content: HTML content (str or UTF-8 bytes)

        Returns:
            ParsedDocument with text content
        """
        import re

        if isinstance(content, bytes):
            text = content.decode("utf-8")
        else:
            text = content

        # Remove script and style elements
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)

        # Replace block elements with newlines
        text = re.sub(r"<(?:p|div|br|h[1-6])[^>]*>", "\n", text, flags=re.IGNORECASE)

        # Remove remaining tags
        text = re.sub(r"<[^>]+>", "", text)

        # Decode entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')

        # Normalize whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = text.strip()

        return ParsedDocument(
            content=text,
            source_mime_type=MIME_HTML,
            output_format="text",
        )

    async def parse_url(self, content: str | bytes) -> ParsedDocument:
        """
        Fetch and parse content from a URL.

        Uses trafilatura for intelligent content extraction that removes
        boilerplate (nav, ads, footers) and preserves article structure.

        Args:
            content: URL string (str or UTF-8 bytes)

        Returns:
            ParsedDocument with extracted text content
        """
        from integrations.web_content_fetcher import fetch_web_content

        if isinstance(content, bytes):
            url = content.decode("utf-8").strip()
        else:
            url = content.strip()

        # Handle .url files (Windows Internet Shortcut format)
        if url.startswith("[InternetShortcut]"):
            import re
            match = re.search(r"URL=(.+)", url)
            if match:
                url = match.group(1).strip()
            else:
                return ParsedDocument(
                    content=f"Could not extract URL from shortcut file",
                    source_mime_type=MIME_URL,
                    output_format="text",
                    metadata={"error": "Invalid URL shortcut format"},
                )

        result = await fetch_web_content(url)

        if not result.success:
            logger.warning(f"Failed to fetch URL {url}: {result.error}")
            return ParsedDocument(
                content=f"Failed to fetch content from {url}: {result.error}",
                source_mime_type=MIME_URL,
                output_format="text",
                metadata={"url": url, "error": result.error},
            )

        return ParsedDocument(
            content=result.text or "",
            source_mime_type=MIME_URL,
            output_format="text",
            title=result.title,
            metadata={
                "url": url,
                "word_count": result.word_count,
            },
        )

    async def parse(
        self, content: bytes | str, mime_type: str
    ) -> ParsedDocument:
        """
        Parse content based on MIME type.

        Routes to appropriate parser based on the content type.

        Args:
            content: Content to parse (bytes or str)
            mime_type: MIME type of the content

        Returns:
            ParsedDocument with parsed content
        """
        if mime_type == MIME_PDF:
            if isinstance(content, str):
                raise ValueError("PDF content must be bytes")
            return await self.parse_pdf(content)

        elif mime_type in (MIME_TEXT, MIME_MARKDOWN):
            return await self.parse_text(content)

        elif mime_type == MIME_CSV:
            return await self.parse_csv(content)

        elif mime_type == MIME_HTML:
            return await self.parse_html(content)

        elif mime_type == MIME_URL:
            return await self.parse_url(content)

        else:
            # Check if content is a URL or contains URL
            text_content = content.decode("utf-8") if isinstance(content, bytes) else content
            text_stripped = text_content.strip()
            
            # Detect URL content by prefix or .url file format
            if (text_stripped.startswith("http://") or 
                text_stripped.startswith("https://") or
                text_stripped.startswith("[InternetShortcut]") or
                mime_type in ("text/uri-list", "application/x-url", "application/internet-shortcut")):
                return await self.parse_url(content)
            
            # Fallback: try to parse as text
            logger.warning(f"Unknown MIME type {mime_type}, attempting text parse")
            return await self.parse_text(content)


# Re-export for backwards compatibility
from core.utils.token_utils import estimate_tokens

__all__ = ["ContentParser", "ParsedDocument", "estimate_tokens"]
