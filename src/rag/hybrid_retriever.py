"""Hybrid retriever combining BM25 keyword matching and dense vector similarity."""

from __future__ import annotations

from typing import Optional
from src.models.match import RetrievedEvidence
from src.models.profile import CareerChunk
from src.rag.bm25_retriever import BM25CareerRetriever
from src.rag.vector_store import CareerVectorStore


class HybridCareerRetriever:
    """Combines BM25 and dense vector search using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        vector_store: CareerVectorStore,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.bm25 = BM25CareerRetriever(vector_store.chunks)

    def refresh_bm25_index(self) -> None:
        """Synchronize BM25 index with current vector store chunks."""
        self.bm25.index_chunks(self.vector_store.chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        bm25_candidates: int = 20,
        dense_candidates: int = 20,
    ) -> list[tuple[CareerChunk, float]]:
        """Perform hybrid search over indexed career documents.

        Returns list of (CareerChunk, normalized_score) sorted descending.
        """
        if not self.vector_store.chunks or not query.strip():
            return []

        # Retrieve top candidates from both methods
        bm25_results = self.bm25.search(query, n_results=bm25_candidates)
        dense_results = self.vector_store.search(query, n_results=dense_candidates)

        # Build rank lookups
        bm25_ranks: dict[str, int] = {chunk.chunk_id: rank + 1 for rank, (chunk, _) in enumerate(bm25_results)}
        dense_ranks: dict[str, int] = {chunk.chunk_id: rank + 1 for rank, (chunk, _) in enumerate(dense_results)}
        dense_sims: dict[str, float] = {chunk.chunk_id: sim for chunk, sim in dense_results}
        bm25_scores: dict[str, float] = {chunk.chunk_id: score for chunk, score in bm25_results}

        # Combine unique candidates
        candidate_chunks: dict[str, CareerChunk] = {}
        for chunk, _ in bm25_results:
            candidate_chunks[chunk.chunk_id] = chunk
        for chunk, _ in dense_results:
            candidate_chunks[chunk.chunk_id] = chunk

        # Compute RRF score
        rrf_scores: dict[str, float] = {}
        for chunk_id in candidate_chunks:
            score = 0.0
            if chunk_id in bm25_ranks:
                score += self.bm25_weight / (self.rrf_k + bm25_ranks[chunk_id])
            if chunk_id in dense_ranks:
                score += self.dense_weight / (self.rrf_k + dense_ranks[chunk_id])
            rrf_scores[chunk_id] = score

        max_possible_rrf = (self.bm25_weight / (self.rrf_k + 1)) + (self.dense_weight / (self.rrf_k + 1))

        # Detect skills in query
        from src.data_ingestion.semantic_chunker import CareerChunker
        query_skills = {s.lower() for s in CareerChunker.extract_skills_from_text(query)}

        ranked: list[tuple[CareerChunk, float]] = []
        for chunk_id, chunk in candidate_chunks.items():
            rrf_norm = min(1.0, rrf_scores[chunk_id] / max_possible_rrf)
            cos_sim = max(0.0, dense_sims.get(chunk_id, 0.0))
            has_bm25 = chunk_id in bm25_scores and bm25_scores[chunk_id] > 0

            # Skill overlap boost
            chunk_skills_lower = {s.lower() for s in chunk.skills}
            skill_overlap = bool(query_skills and (query_skills & chunk_skills_lower))

            # Absolute calibration: if no keyword match and low semantic similarity, suppress score
            if not has_bm25 and cos_sim < 0.40:
                calibrated = cos_sim * 0.5
            elif has_bm25:
                base = max(rrf_norm, cos_sim, 0.45)
                calibrated = min(1.0, base + (0.2 if skill_overlap else 0.0))
            else:
                calibrated = 0.5 * rrf_norm + 0.5 * cos_sim
                if skill_overlap:
                    calibrated = min(1.0, calibrated + 0.15)

            ranked.append((chunk, round(calibrated, 4)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:n_results]

    def retrieve_evidence_for_qualification(
        self, qualification: str, threshold: float = 0.35
    ) -> RetrievedEvidence:
        """Find the most relevant evidence in career documents for a qualification.

        If no evidence exceeds the threshold, returns 'No evidence found.'
        """
        top_matches = self.search(qualification, n_results=1)
        if not top_matches:
            return RetrievedEvidence(
                qualification=qualification,
                evidence_text="No evidence found in uploaded career documents.",
                source_document="None",
                section="None",
                chunk_id="N/A",
                similarity_score=0.0,
                supports_match=False,
                provenance="No evidence found.",
            )

        best_chunk, score = top_matches[0]
        # Also check whether chunk has direct keyword overlap or high semantic similarity
        query_tokens = BM25CareerRetriever.tokenize(qualification)
        chunk_tokens = set(BM25CareerRetriever.tokenize(best_chunk.content + " " + " ".join(best_chunk.skills)))
        token_overlap = any(t in chunk_tokens for t in query_tokens if len(t) > 3)

        if score >= threshold and (token_overlap or score >= 0.48):
            return RetrievedEvidence(
                qualification=qualification,
                evidence_text=best_chunk.content,
                source_document=best_chunk.source_document,
                section=best_chunk.section,
                chunk_id=best_chunk.chunk_id,
                similarity_score=score,
                supports_match=True,
                provenance="CANDIDATE EVIDENCE",
            )
        else:
            return RetrievedEvidence(
                qualification=qualification,
                evidence_text=f"No supporting evidence found in career documents (confidence {score:.2f} below threshold {threshold:.2f}).",
                source_document=best_chunk.source_document,
                section=best_chunk.section,
                chunk_id=best_chunk.chunk_id,
                similarity_score=score,
                supports_match=False,
                provenance="No evidence found.",
            )
