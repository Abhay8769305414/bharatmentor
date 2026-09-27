"""
BharatMentor — RAG Unit & Integration Tests
===========================================
Tests:
  - Markdown ingestion & section-aware chunking
  - TF-IDF indexing & retrieval
  - Out-of-domain score thresholding
  - Evaluation metric functions (Recall@k, MRR)
"""

import pytest
from pathlib import Path

from rag.ingestion import ingest_markdown, ingest_directory, INDEX_DIR
from rag.retrieval import retrieve, reset_cache, _build_tfidf_index
from experiments.rag.evaluate import section_match, recall_at_k, reciprocal_rank, run_evaluation


class TestRAGIngestion:
    def test_mechanics_md_ingested(self, tmp_path):
        corpus_path = Path("data/knowledge/mechanics.md")
        assert corpus_path.exists(), "mechanics.md must exist in data/knowledge/"
        chunks = ingest_markdown(corpus_path)
        assert len(chunks) > 10
        sections = {c["section"] for c in chunks}
        assert "Newton's First Law" in sections
        assert "Newton's Second Law" in sections
        assert "Newton's Third Law" in sections
        assert "Scalars and Vectors" in sections

    def test_preamble_skipped(self):
        corpus_path = Path("data/knowledge/mechanics.md")
        chunks = ingest_markdown(corpus_path)
        sections = {c["section"] for c in chunks}
        # Preamble header should not be a section
        assert not any("Newtonian Mechanics" in s for s in sections)


class TestRAGRetrieval:
    def setup_method(self):
        reset_cache()

    def test_retrieve_second_law(self):
        results = retrieve("What is Newton's second law of motion?", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0
        top_section = results[0]["section"]
        assert top_section == "Newton's Second Law"

    def test_retrieve_friction(self):
        results = retrieve("How does friction work and what is static friction?", top_k=3)
        assert isinstance(results, list)
        top_sections = [r["section"] for r in results]
        assert "Friction" in top_sections

    def test_out_of_domain_flag(self):
        results = retrieve("What is the recipe for chocolate cake and baking temperature?", top_k=3)
        assert isinstance(results, list)
        top_score = results[0]["score"] if results else 0.0
        # OOD queries should have low score or out_of_domain flag
        assert top_score < 0.35 or any(r.get("out_of_domain") for r in results)


class TestRAGEvaluationMetrics:
    def test_section_match(self):
        assert section_match("Newton's Second Law", "Newton's Second Law")
        assert section_match("Newton's Second Law ", "newton's second law")
        assert not section_match("Newton's First Law", "Newton's Second Law")

    def test_recall_at_k(self):
        results = [
            {"section": "Work and Energy"},
            {"section": "Kinematics"},
            {"section": "Friction"},
        ]
        assert recall_at_k(results, "Work and Energy", k=1) is True
        assert recall_at_k(results, "Kinematics", k=1) is False
        assert recall_at_k(results, "Kinematics", k=2) is True
        assert recall_at_k(results, "Friction", k=3) is True
        assert recall_at_k(results, "Momentum", k=3) is False

    def test_reciprocal_rank(self):
        results = [
            {"section": "Work and Energy"},
            {"section": "Kinematics"},
            {"section": "Friction"},
        ]
        assert reciprocal_rank(results, "Work and Energy") == 1.0
        assert reciprocal_rank(results, "Kinematics") == 0.5
        assert reciprocal_rank(results, "Friction") == pytest.approx(1.0 / 3)
        assert reciprocal_rank(results, "Nonexistent") == 0.0
