"""Data ingestion package for extracting and chunking career documents."""

from src.data_ingestion.document_parser import DocumentParser
from src.data_ingestion.semantic_chunker import CareerChunker

__all__ = ["DocumentParser", "CareerChunker"]
