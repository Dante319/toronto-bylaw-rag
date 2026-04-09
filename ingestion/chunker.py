# ingestion/chunk.py
from dataclasses import dataclass, field
from ingestion.parser import ParsedDocument
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str          # e.g. "short_term_rental::547-3::0"
    domain: str
    section_id: str
    section_title: str
    parent_section: str | None
    text: str
    source_file: str
    page: int | None
    char_start: int = 0


def _approx_token_count(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def _split_into_chunks(text: str, max_tokens: int = 512, overlap_tokens: int = 64) -> list[str]:
    """
    Split text into overlapping chunks at sentence boundaries.
    Avoids cutting mid-sentence.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    overlap_buffer = []

    for sentence in sentences:
        s_len = _approx_token_count(sentence)

        if current_len + s_len > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep last N tokens worth of sentences for overlap
            overlap_buffer = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + _approx_token_count(s) > overlap_tokens:
                    break
                overlap_buffer.insert(0, s)
                overlap_len += _approx_token_count(s)
            current_chunk = overlap_buffer.copy()
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += s_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_document(doc: ParsedDocument, max_tokens: int = 512, overlap_tokens: int = 64) -> list[Chunk]:
    """
    Convert a ParsedDocument into a flat list of Chunks.
    Short sections stay as single chunks; long sections are split.
    """
    chunks = []

    for section in doc.sections:
        text = section["text"].strip()
        if not text:
            continue

        if _approx_token_count(text) <= max_tokens:
            # Section fits in one chunk
            chunk_texts = [text]
        else:
            chunk_texts = _split_into_chunks(text, max_tokens, overlap_tokens)

        for i, chunk_text in enumerate(chunk_texts):
            chunk_id = f"{doc.domain}::{section['id']}::{i}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                domain=doc.domain,
                section_id=section["id"],
                section_title=section.get("title", ""),
                parent_section=section.get("parent"),
                text=chunk_text,
                source_file=doc.source_file,
                page=section.get("page"),
            ))

    logger.info(f"  → {len(chunks)} chunks from {len(doc.sections)} sections ({doc.domain})")
    return chunks