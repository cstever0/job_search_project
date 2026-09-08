"""Unit tests for the Personal RAG hybrid retrieval system."""

import shutil
from pathlib import Path
import pytest

from src.models.profile import CareerChunk
from src.rag.bm25_retriever import BM25CareerRetriever
from src.rag.hybrid_retriever import HybridCareerRetriever
from src.rag.vector_store import CareerVectorStore


@pytest.fixture
def sample_career_chunks() -> list[CareerChunk]:
    return [
        CareerChunk(
            chunk_id="chunk_001",
            source_document="resume.pdf",
            section="work_experience",
            job_title="Senior ML Engineer",
            organization="Federal Analytics Corp",
            skills=["Python", "PyTorch", "Hugging Face", "BERT", "AWS"],
            date="2022 - Present",
            content="Built transformer-based BERT pipelines for federal document classification using PyTorch and Hugging Face on AWS.",
        ),
        CareerChunk(
            chunk_id="chunk_002",
            source_document="resume.pdf",
            section="work_experience",
            job_title="Data Engineer",
            organization="Data Corp",
            skills=["SQL", "PostgreSQL", "Docker", "ETL", "Python"],
            date="2020 - 2022",
            content="Designed scalable ETL data pipelines using PostgreSQL and Docker, optimizing SQL query execution by 35%.",
        ),
        CareerChunk(
            chunk_id="chunk_003",
            source_document="projects.txt",
            section="projects",
            project_name="Legal Contract NLP",
            skills=["Python", "NLP", "ChromaDB", "RAG"],
            date="2023",
            content="Implemented retrieval-augmented generation (RAG) using LangChain, ChromaDB, and Python across 50,000 legal contracts.",
        ),
        CareerChunk(
            chunk_id="chunk_004",
            source_document="resume.pdf",
            section="education",
            education="Master of Science in Computer Science",
            skills=["Python", "Machine Learning"],
            date="2022",
            content="Master of Science in Computer Science at George Washington University, specializing in Machine Learning and NLP.",
        ),
    ]


@pytest.fixture
def temp_vector_store(tmp_path: Path, sample_career_chunks: list[CareerChunk]) -> CareerVectorStore:
    store = CareerVectorStore(persist_dir=tmp_path / "vec_test")
    store.add_chunks(sample_career_chunks)
    return store


def test_bm25_exact_keyword_matching(sample_career_chunks: list[CareerChunk]):
    retriever = BM25CareerRetriever(sample_career_chunks)

    # Search exact technical acronym
    results = retriever.search("PostgreSQL", n_results=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "chunk_002"
    assert "PostgreSQL" in top_chunk.content

    # Search PyTorch
    results_pytorch = retriever.search("PyTorch", n_results=2)
    assert len(results_pytorch) > 0
    assert results_pytorch[0][0].chunk_id == "chunk_001"


def test_vector_store_semantic_search(temp_vector_store: CareerVectorStore):
    # Search with semantically related phrase that doesn't use exact words
    results = temp_vector_store.search("natural language deep learning neural networks", n_results=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    # Should match either chunk_001 (BERT/PyTorch) or chunk_003 (NLP/RAG) or chunk_004 (Master CS NLP)
    assert top_chunk.chunk_id in ("chunk_001", "chunk_003", "chunk_004")
    assert score > 0.0


def test_hybrid_retrieval_and_rrf(temp_vector_store: CareerVectorStore):
    retriever = HybridCareerRetriever(temp_vector_store)
    results = retriever.search("transformer text classification", n_results=3)

    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "chunk_001"
    assert 0.0 <= score <= 1.0


def test_evidence_grounding_positive_and_negative(temp_vector_store: CareerVectorStore):
    retriever = HybridCareerRetriever(temp_vector_store)

    # Known qualification supported by candidate
    evidence_pos = retriever.retrieve_evidence_for_qualification("Experience with PyTorch and deep learning models")
    assert evidence_pos.supports_match is True
    assert evidence_pos.provenance == "CANDIDATE EVIDENCE"
    assert "PyTorch" in evidence_pos.evidence_text
    assert evidence_pos.chunk_id == "chunk_001"

    # Qualification not present in candidate resume
    evidence_neg = retriever.retrieve_evidence_for_qualification(
        "Commercial helicopter pilot license and flight hours", threshold=0.35
    )
    assert evidence_neg.supports_match is False
    assert evidence_neg.provenance == "No evidence found."
