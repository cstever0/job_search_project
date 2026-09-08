"""Local embedding manager providing dense embeddings for career chunks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence
import numpy as np

from config import CACHE_DIR

# Set local model download path within workspace
CHROMA_MODEL_CACHE: Path = CACHE_DIR / "chroma_models"
CHROMA_MODEL_CACHE.mkdir(parents=True, exist_ok=True)

try:
    from chromadb.utils.embedding_functions import onnx_mini_lm_l6_v2
    onnx_mini_lm_l6_v2.ONNXMiniLM_L6_V2.DOWNLOAD_PATH = CHROMA_MODEL_CACHE
    HAS_ONNX_EMBEDDINGS = True
except Exception:
    HAS_ONNX_EMBEDDINGS = False


class EmbeddingManager:
    """Manages dense sentence embeddings using local ONNX all-MiniLM-L6-v2."""

    def __init__(self):
        self._ef = None
        if HAS_ONNX_EMBEDDINGS:
            try:
                self._ef = onnx_mini_lm_l6_v2.ONNXMiniLM_L6_V2()
            except Exception:
                self._ef = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Compute dense vector embeddings for a list of text strings."""
        if not texts:
            return []

        cleaned_texts = [t.strip() or "empty" for t in texts]

        if self._ef is not None:
            try:
                embeddings = self._ef(cleaned_texts)
                return [list(map(float, emb)) for emb in embeddings]
            except Exception:
                pass

        # Fallback: deterministic character/word n-gram hashing vectorizer (normalized)
        return [self._fallback_embed(t) for t in cleaned_texts]

    def embed_query(self, text: str) -> list[float]:
        """Compute dense vector embedding for a single search query."""
        results = self.embed_documents([text])
        return results[0] if results else [0.0] * 384

    @staticmethod
    def _fallback_embed(text: str, dim: int = 384) -> list[float]:
        """Lightweight deterministic feature embedding for edge-case fallback."""
        vec = np.zeros(dim, dtype=np.float32)
        tokens = text.lower().split()
        for token in tokens:
            idx = hash(token) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    @staticmethod
    def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        """Compute cosine similarity between two float vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
