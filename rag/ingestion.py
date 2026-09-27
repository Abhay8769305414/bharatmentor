"""
BharatMentor — RAG Ingestion Pipeline
======================================
Ingests knowledge documents from data/knowledge/.

Supported formats: .md, .txt, .pdf (PDF requires PyMuPDF)

For Markdown: section-aware chunking using ## headings as boundaries.
For plain text: sentence-boundary chunking.
For PDF: per-page text extraction (requires PyMuPDF).

Run:
    python -m rag.ingestion

Output (written to data/index/):
    chunks.json      — all chunks with metadata
    embeddings.npy   — numpy array (only if sentence-transformers installed)

Phase 2 baseline uses TF-IDF retrieval.
The chunks.json file is the single source of truth for both TF-IDF and
semantic retrieval.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"
INDEX_DIR = ROOT / "data" / "index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"

# Chunking — for plain text fallback only
CHUNK_SIZE = 600   # characters
CHUNK_OVERLAP = 100


# ── Markdown ingestion (section-aware) ────────────────────────────────────────

def ingest_markdown(md_path: Path) -> list[dict[str, Any]]:
    """
    Parse a Markdown file, splitting on ## headings.
    Each section becomes one or more chunks, preserving section name.

    A chunk dict contains:
      chunk_id  : int (assigned later)
      document  : str (file stem, e.g. "mechanics")
      section   : str (heading text, e.g. "Newton's Second Law")
      text      : str (section content)
      source    : str (filename)
    """
    raw = md_path.read_text(encoding="utf-8")
    doc_name = md_path.stem

    # Split on level-2 headings (## Section: ...)
    # Pattern: line starting with ##
    parts = re.split(r"^##\s+", raw, flags=re.MULTILINE)
    # If the markdown file did not start with ##, parts[0] is preamble (metadata / doc title)
    if not raw.lstrip().startswith("##") and parts:
        parts = parts[1:]

    chunks: list[dict[str, Any]] = []

    for part in parts:
        if not part.strip():
            continue

        lines = part.strip().splitlines()
        if not lines:
            continue

        # First line is the section title (comes after the ## that was split on)
        section_title = lines[0].strip()
        # Strip any remaining leading # characters (shouldn't happen, but be safe)
        section_title = re.sub(r"^#+\s*", "", section_title)
        # Strip "Section: " prefix if present
        section_title = re.sub(r"^Section:\s*", "", section_title)
        section_title = section_title.strip()

        # Skip document-level headers (h1) or empty section names
        if not section_title or len(section_title) < 3:
            continue

        # Skip if the "body" is just document metadata (preamble before first ##)
        # Heuristic: if there's no real sentence content, skip
        body = "\n".join(lines[1:]).strip()
        if not body or len(body) < 20:
            continue

        # Clean up the body
        body = _clean_text(body)

        # If section is very long, sub-chunk it
        sub_chunks = _split_long_text(body, max_chars=600, overlap=80)

        for i, sub in enumerate(sub_chunks):
            chunks.append({
                "document": doc_name,
                "section": section_title,
                "sub_chunk": i,
                "text": sub.strip(),
                "source": md_path.name,
            })

    return chunks


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into chunks ≤ max_chars with overlap at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    chunks = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if current_len + len(sent) > max_chars and current:
            chunks.append(" ".join(current))
            # Overlap: keep last N chars worth
            tail: list[str] = []
            tail_len = 0
            for s in reversed(current):
                if tail_len + len(s) <= overlap:
                    tail.insert(0, s)
                    tail_len += len(s)
                else:
                    break
            current = tail
            current_len = tail_len
        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks


# ── Plain text ingestion ───────────────────────────────────────────────────────

def ingest_text(txt_path: Path) -> list[dict[str, Any]]:
    """Simple chunking for .txt files — no section metadata."""
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    doc_name = txt_path.stem
    raw = _clean_text(raw)
    sub_chunks = _split_long_text(raw, max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    return [
        {
            "document": doc_name,
            "section": f"Chunk {i}",
            "sub_chunk": i,
            "text": sub.strip(),
            "source": txt_path.name,
        }
        for i, sub in enumerate(sub_chunks)
        if sub.strip()
    ]


# ── PDF ingestion (optional) ───────────────────────────────────────────────────

def ingest_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """Per-page text extraction using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF not installed — skipping %s. pip install PyMuPDF", pdf_path.name)
        return []

    doc_name = pdf_path.stem
    chunks: list[dict[str, Any]] = []
    doc = fitz.open(str(pdf_path))

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if not text.strip():
            continue
        text = _clean_text(text)
        sub_chunks = _split_long_text(text, max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for i, sub in enumerate(sub_chunks):
            if sub.strip():
                chunks.append({
                    "document": doc_name,
                    "section": f"Page {page_num}",
                    "sub_chunk": i,
                    "text": sub.strip(),
                    "source": pdf_path.name,
                })
    doc.close()
    return chunks


# ── Directory ingestion ───────────────────────────────────────────────────────

def ingest_directory(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[dict[str, Any]]:
    """
    Ingest all supported documents from knowledge_dir.
    Assigns sequential chunk_id across all documents.
    """
    all_chunks: list[dict[str, Any]] = []

    md_files = sorted(knowledge_dir.glob("**/*.md"))
    txt_files = sorted(knowledge_dir.glob("**/*.txt"))
    pdf_files = sorted(knowledge_dir.glob("**/*.pdf"))

    # Filter README files from txt
    txt_files = [f for f in txt_files if f.stem.upper() != "README"]

    total_files = len(md_files) + len(txt_files) + len(pdf_files)
    if total_files == 0:
        logger.error("No supported documents found in %s", knowledge_dir)
        logger.error("Supported: .md, .txt, .pdf")
        return []

    logger.info("Found: %d .md, %d .txt, %d .pdf files", len(md_files), len(txt_files), len(pdf_files))

    for f in md_files:
        logger.info("Ingesting Markdown: %s", f.name)
        chunks = ingest_markdown(f)
        logger.info("  → %d chunks from %d sections", len(chunks), len({c["section"] for c in chunks}))
        all_chunks.extend(chunks)

    for f in txt_files:
        logger.info("Ingesting text: %s", f.name)
        chunks = ingest_text(f)
        logger.info("  → %d chunks", len(chunks))
        all_chunks.extend(chunks)

    for f in pdf_files:
        logger.info("Ingesting PDF: %s", f.name)
        chunks = ingest_pdf(f)
        logger.info("  → %d chunks", len(chunks))
        all_chunks.extend(chunks)

    # Assign sequential chunk_id
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i

    logger.info("Total chunks: %d", len(all_chunks))
    return all_chunks


# ── Embedding (optional — requires sentence-transformers) ─────────────────────

def embed_chunks(chunks: list[dict[str, Any]]) -> np.ndarray | None:
    """
    Generate sentence-transformer embeddings.
    Returns None if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from app.config import get_settings
        settings = get_settings()
        logger.info("Loading embedding model: %s", settings.embedding_model)
        model = SentenceTransformer(settings.embedding_model)
        texts = [c["text"] for c in chunks]
        logger.info("Embedding %d chunks…", len(texts))
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        return np.array(embeddings, dtype=np.float32)
    except ImportError:
        logger.warning("sentence-transformers not installed — skipping embedding generation.")
        logger.warning("TF-IDF retrieval will be used. pip install sentence-transformers for semantic retrieval.")
        return None


def save_index(chunks: list[dict[str, Any]], embeddings: np.ndarray | None) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info("Saved chunks.json: %d chunks", len(chunks))

    if embeddings is not None:
        np.save(str(EMBEDDINGS_FILE), embeddings)
        logger.info("Saved embeddings.npy: shape %s", embeddings.shape)
    else:
        logger.info("No embeddings saved — TF-IDF will be used for retrieval.")


def run_ingestion(skip_embeddings: bool = False) -> list[dict[str, Any]]:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = ingest_directory(KNOWLEDGE_DIR)
    if not chunks:
        logger.error("No chunks produced — ingestion failed.")
        return []

    embeddings = None if skip_embeddings else embed_chunks(chunks)
    save_index(chunks, embeddings)

    print(f"\n[OK] Ingestion complete: {len(chunks)} chunks indexed.")
    sections = sorted({c["section"] for c in chunks})
    print(f"   Sections ({len(sections)}): {sections}")
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    run_ingestion(skip_embeddings=True)  # Phase 2 baseline: TF-IDF only
