"""BM25 keyword retriever for technical skills and exact terminology matching."""

from __future__ import annotations

import re
from typing import Optional
from rank_bm25 import BM25Plus

from src.models.profile import CareerChunk

STOP_WORDS: set[str] = {
    "a", "an", "the", "in", "on", "at", "to", "for", "with", "by", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "of", "and", "or", "as", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "shall", "should", "may", "might", "must",
    "can", "could", "this", "that", "these", "those", "it", "its", "they",
}


class BM25CareerRetriever:
    """Keyword retriever using BM25Plus, optimized for exact technical terms."""

    def __init__(self, chunks: Optional[list[CareerChunk]] = None):
        self.chunks: list[CareerChunk] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: Optional[BM25Plus] = None
        if chunks:
            self.index_chunks(chunks)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize text preserving technical identifiers (e.g. C++, CI/CD, Node.js)."""
        if not text:
            return []
        text_lower = text.lower()
        tokens = re.findall(r"[a-z0-9+#.\-_/]+", text_lower)
        cleaned_tokens = [
            t.strip(".-_/") for t in tokens if len(t.strip(".-_/")) > 1 and t.strip(".-_/") not in STOP_WORDS
        ]
        return cleaned_tokens

    def index_chunks(self, chunks: list[CareerChunk]) -> None:
        """Build the BM25Plus index over the provided career chunks."""
        self.chunks = list(chunks)
        self.tokenized_corpus = []

        for c in self.chunks:
            skills_str = " ".join(c.skills)
            full_text = f"{c.section} {c.job_title or ''} {c.organization or ''} {skills_str} {c.content}"
            self.tokenized_corpus.append(self.tokenize(full_text))

        if self.tokenized_corpus:
            self.bm25 = BM25Plus(self.tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, n_results: int = 5) -> list[tuple[CareerChunk, float]]:
        """Search chunks by BM25 keyword matching score.

        Returns list of (CareerChunk, bm25_score) sorted descending.
        """
        if not self.bm25 or not self.chunks or not query.strip():
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        results: list[tuple[CareerChunk, float]] = []

        for chunk, score in zip(self.chunks, scores):
            if score > 0.0:
                results.append((chunk, float(score)))

        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n_results]
