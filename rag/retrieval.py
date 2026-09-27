"""
BharatMentor — RAG Retrieval
==============================
Primary retriever: TF-IDF over chunks.json (Phase 2 baseline).
Secondary: semantic embeddings (when embeddings.npy exists and sentence-transformers installed).

TF-IDF is the default and is reproducible without any model downloads.

Out-of-domain detection: a simple score threshold.
If max score < LOW_SCORE_THRESHOLD, the query may be out-of-domain.
This is documented as a limitation — it is not a sophisticated confidence model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CHUNKS_FILE = ROOT / "data" / "index" / "chunks.json"
EMBEDDINGS_FILE = ROOT / "data" / "index" / "embeddings.npy"

# Out-of-domain threshold: below this TF-IDF score, warn that results may be irrelevant.
# This is a heuristic, not a calibrated confidence estimate.
LOW_SCORE_THRESHOLD = 0.05

# ── Module-level cache ────────────────────────────────────────────────────────
_chunks: list[dict[str, Any]] | None = None
_tfidf_vectorizer = None
_tfidf_matrix = None

_embed_chunks: list[dict[str, Any]] | None = None
_embeddings: np.ndarray | None = None
_embed_model = None


# ── TF-IDF retrieval (primary for Phase 2) ────────────────────────────────────

def _build_tfidf_index() -> None:
    """Load chunks.json and fit TF-IDF vectorizer. Cached after first call."""
    global _chunks, _tfidf_vectorizer, _tfidf_matrix

    if _chunks is not None:
        return  # already loaded

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"chunks.json not found. Run: python -m rag.ingestion\n"
            f"  Expected: {CHUNKS_FILE}"
        )

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        _chunks = json.load(f)

    if not _chunks:
        raise ValueError("chunks.json is empty — re-run ingestion.")

    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [c["text"] for c in _chunks]
    _tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),   # unigrams + bigrams improve recall
        sublinear_tf=True,    # apply log normalization to term frequency
    )
    _tfidf_matrix = _tfidf_vectorizer.fit_transform(texts)
    logger.info("TF-IDF index built: %d chunks, vocab size %d",
                len(_chunks), len(_tfidf_vectorizer.vocabulary_))


def retrieve_tfidf(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    TF-IDF retrieval. Returns top_k chunks sorted by cosine similarity.
    Each result includes a 'score' and an 'out_of_domain' flag.
    """
    _build_tfidf_index()

    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    query_vec = _tfidf_vectorizer.transform([query])
    scores = sk_cosine(query_vec, _tfidf_matrix)[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = dict(_chunks[idx])
        chunk["score"] = float(scores[idx])
        chunk["retriever"] = "tfidf"
        chunk["out_of_domain"] = bool(scores[idx] < LOW_SCORE_THRESHOLD)
        results.append(chunk)

    if results:
        logger.debug("TF-IDF top score for '%s': %.4f (ood=%s)",
                     query, results[0]["score"], results[0]["out_of_domain"])

    return results


# ── Semantic retrieval (Phase 3+) ──────────────────────────────────────────────

def _build_semantic_index() -> None:
    global _embed_chunks, _embeddings, _embed_model

    if _embed_chunks is not None:
        return

    if not CHUNKS_FILE.exists() or not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            "Semantic index not found. Run: python -m rag.ingestion (without skip_embeddings)."
        )

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        _embed_chunks = json.load(f)
    _embeddings = np.load(str(EMBEDDINGS_FILE)).astype(np.float32)

    try:
        from sentence_transformers import SentenceTransformer
        from app.config import get_settings
        settings = get_settings()
        _embed_model = SentenceTransformer(settings.embedding_model)
        logger.info("Semantic index loaded: %d chunks, shape %s", len(_embed_chunks), _embeddings.shape)
    except ImportError:
        raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")


def retrieve_semantic(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Semantic retrieval using sentence-transformer embeddings."""
    _build_semantic_index()

    query_vec = _embed_model.encode([query])[0].astype(np.float32)
    # Normalised cosine similarity
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    c_norms = _embeddings / (np.linalg.norm(_embeddings, axis=1, keepdims=True) + 1e-10)
    scores = c_norms @ q_norm

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        chunk = dict(_embed_chunks[idx])
        chunk["score"] = float(scores[idx])
        chunk["retriever"] = "semantic"
        chunk["out_of_domain"] = bool(scores[idx] < 0.3)  # semantic threshold differs
        results.append(chunk)

    return results


# ── Unified retrieval entry point ─────────────────────────────────────────────

def retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Retrieve top_k relevant chunks for a query.

    Strategy:
    1. Try TF-IDF (always available once chunks.json exists).
    2. If semantic embeddings available, use semantic retrieval instead.
    3. If nothing is indexed, return empty list with a clear error.

    Returns list of chunk dicts with fields:
        document, section, text, score, retriever, out_of_domain
    """
    # Try semantic first if index exists
    if EMBEDDINGS_FILE.exists():
        try:
            return retrieve_semantic(query, top_k)
        except Exception as exc:
            logger.warning("Semantic retrieval failed, falling back to TF-IDF: %s", exc)

    # TF-IDF baseline
    try:
        return retrieve_tfidf(query, top_k)
    except FileNotFoundError as exc:
        logger.error("No index found: %s", exc)
        return []
    except Exception as exc:
        logger.error("TF-IDF retrieval error: %s", exc, exc_info=True)
        return []


def reset_cache() -> None:
    """Reset module-level cache. Useful in tests."""
    global _chunks, _tfidf_vectorizer, _tfidf_matrix
    global _embed_chunks, _embeddings, _embed_model
    _chunks = _tfidf_vectorizer = _tfidf_matrix = None
    _embed_chunks = _embeddings = _embed_model = None
