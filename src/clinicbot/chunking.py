"""Step 1 of RAG: split documents into chunks.

Strategy used here ("structure-aware chunking"):
  1. Split each Markdown file on its headings (#, ##, ###), so one chunk = one topic.
  2. If a section is still longer than CHUNK_MAX_CHARS, split it further on
     paragraph boundaries, carrying CHUNK_OVERLAP_CHARS of text into the next piece.
  3. Prefix every chunk with its "breadcrumb" (doc title > section > subsection).
     This is called a *contextual chunk header*: a chunk that just says
     "Rs. 500" means nothing, but "Appointments Policy > Cancellation: ... Rs. 500"
     embeds (and reads) much better.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from clinicbot.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS

HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def split_markdown_sections(markdown: str) -> list[tuple[list[str], str]]:
    """Return (heading_path, body) pairs, one per heading section."""
    sections: list[tuple[list[str], str]] = []
    heading_path: list[str] = []
    body_lines: list[str] = []

    def flush():
        body = "\n".join(body_lines).strip()
        if body:
            sections.append((list(heading_path), body))
        body_lines.clear()

    for line in markdown.splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            # Keep only the parent headings above this level, then add this one.
            heading_path[:] = heading_path[: level - 1] + [match.group(2).strip()]
        else:
            body_lines.append(line)
    flush()
    return sections


def split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Pack paragraphs into pieces of at most ~max_chars, overlapping neighbours."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            pieces.append(current)
            # Start the next piece with the tail of the previous one (the overlap).
            current = current[-overlap:] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        pieces.append(current)
    return pieces


def chunk_file(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section_idx, (heading_path, body) in enumerate(
        split_markdown_sections(path.read_text(encoding="utf-8"))
    ):
        breadcrumb = " > ".join(heading_path) or path.stem
        for piece_idx, piece in enumerate(
            split_long_text(body, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS)
        ):
            chunks.append(
                Chunk(
                    # Stable ids mean re-running ingestion overwrites, not duplicates.
                    id=f"{path.stem}:{section_idx}:{piece_idx}",
                    text=f"{breadcrumb}\n\n{piece}",
                    metadata={"source": path.name, "section": breadcrumb},
                )
            )
    return chunks


def chunk_directory(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        chunks.extend(chunk_file(path))
    return chunks
