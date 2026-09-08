"""Vector store wrapper for personal career chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import numpy as np

from config import VECTOR_DB_DIR
from src.models.profile import CareerChunk
from src.rag.embeddings import EmbeddingManager


class CareerVectorStore:
    """Manages indexing and similarity search over CareerChunk embeddings."""

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or VECTOR_DB_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_manager = EmbeddingManager()
        self.chunks: list[CareerChunk] = []
        self.embeddings: list[list[float]] = []
        self._load_persisted()

    def _get_storage_file(self) -> Path:
        return self.persist_dir / "career_chunks.json"

    def _load_persisted(self) -> None:
        """Load persisted chunks from disk if available."""
        file_path = self._get_storage_file()
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.chunks = [CareerChunk(**item["chunk"]) for item in data]
                self.embeddings = [item["embedding"] for item in data]
            except Exception:
                self.chunks = []
                self.embeddings = []

    def _save_persisted(self) -> None:
        """Save indexed chunks and embeddings to disk."""
        file_path = self._get_storage_file()
        data = [
            {"chunk": chunk.model_dump(), "embedding": emb}
            for chunk, emb in zip(self.chunks, self.embeddings)
        ]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_chunks(self, chunks: list[CareerChunk]) -> None:
        """Add new career chunks, compute their dense embeddings, and save."""
        if not chunks:
            return

        texts = [f"{c.section.upper()}: {c.content}" for c in chunks]
        new_embeddings = self.embedding_manager.embed_documents(texts)

        # Remove existing duplicates by chunk_id
        existing_ids = {c.chunk_id: idx for idx, c in enumerate(self.chunks)}
        for chunk, emb in zip(chunks, new_embeddings):
            if chunk.chunk_id in existing_ids:
                idx = existing_ids[chunk.chunk_id]
                self.chunks[idx] = chunk
                self.embeddings[idx] = emb
            else:
                self.chunks.append(chunk)
                self.embeddings.append(emb)

        self._save_persisted()

    def clear(self) -> None:
        """Clear all stored chunks and embeddings."""
        self.chunks = []
        self.embeddings = []
        file_path = self._get_storage_file()
        if file_path.exists():
            file_path.unlink()

    def count(self) -> int:
        """Return total number of indexed career chunks."""
        return len(self.chunks)

    def search(
        self, query: str, n_results: int = 5, min_similarity: float = 0.0
    ) -> list[tuple[CareerChunk, float]]:
        """Search chunks by cosine similarity to the query embedding.

        Returns list of (CareerChunk, cosine_similarity_score) sorted descending.
        """
        if not self.chunks or not query.strip():
            return []

        query_emb = self.embedding_manager.embed_query(query)
        q_vec = np.array(query_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return [(c, 0.0) for c in self.chunks[:n_results]]

        results: list[tuple[CareerChunk, float]] = []
        for chunk, emb in zip(self.chunks, self.embeddings):
            doc_vec = np.array(emb, dtype=np.float32)
            d_norm = np.linalg.norm(doc_vec)
            if d_norm == 0:
                sim = 0.0
            else:
                sim = float(np.dot(q_vec, doc_vec) / (q_norm * d_norm))

            if sim >= min_similarity:
                results.append((chunk, sim))

        # Sort descending by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n_results]
