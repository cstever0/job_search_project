"""Document parser for extracting and cleaning text from PDF, DOCX, and TXT files."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import BinaryIO, Union
import pypdf
import docx


class DocumentParser:
    """Extracts and normalizes text from multiple document formats (PDF, DOCX, TXT)."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    @classmethod
    def is_supported(cls, filename: str) -> bool:
        """Check if the file format is supported."""
        return Path(filename).suffix.lower() in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Normalize whitespace, remove non-printable characters, preserve paragraphs."""
        if not text:
            return ""
        # Normalize non-breaking spaces and special unicode quotes/dashes
        text = text.replace("\u00a0", " ").replace("\u200b", "")
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace excessive whitespace within lines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

        # Group multiple empty lines into at most two newlines
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @classmethod
    def extract_text_from_pdf(cls, file_obj: Union[str, Path, BinaryIO, bytes]) -> str:
        """Extract plain text from a PDF file or binary stream."""
        if isinstance(file_obj, (str, Path)):
            with open(file_obj, "rb") as f:
                reader = pypdf.PdfReader(f)
                pages = [page.extract_text() or "" for page in reader.pages]
        elif isinstance(file_obj, bytes):
            stream = io.BytesIO(file_obj)
            reader = pypdf.PdfReader(stream)
            pages = [page.extract_text() or "" for page in reader.pages]
        else:
            reader = pypdf.PdfReader(file_obj)
            pages = [page.extract_text() or "" for page in reader.pages]

        raw_text = "\n".join(pages)
        return cls.clean_text(raw_text)

    @classmethod
    def extract_text_from_docx(cls, file_obj: Union[str, Path, BinaryIO, bytes]) -> str:
        """Extract plain text from a Word (.docx) document."""
        if isinstance(file_obj, bytes):
            stream = io.BytesIO(file_obj)
            doc = docx.Document(stream)
        elif isinstance(file_obj, (str, Path)):
            doc = docx.Document(str(file_obj))
        else:
            doc = docx.Document(file_obj)

        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Also include text from tables if any
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)

        raw_text = "\n".join(paragraphs)
        return cls.clean_text(raw_text)

    @classmethod
    def extract_text_from_txt(cls, file_obj: Union[str, Path, BinaryIO, bytes]) -> str:
        """Extract plain text from a TXT file, handling encoding fallbacks."""
        if isinstance(file_obj, (str, Path)):
            try:
                with open(file_obj, "r", encoding="utf-8") as f:
                    raw_text = f.read()
            except UnicodeDecodeError:
                with open(file_obj, "r", encoding="latin-1") as f:
                    raw_text = f.read()
        elif isinstance(file_obj, bytes):
            try:
                raw_text = file_obj.decode("utf-8")
            except UnicodeDecodeError:
                raw_text = file_obj.decode("latin-1")
        else:
            data = file_obj.read()
            if isinstance(data, bytes):
                try:
                    raw_text = data.decode("utf-8")
                except UnicodeDecodeError:
                    raw_text = data.decode("latin-1")
            else:
                raw_text = str(data)

        return cls.clean_text(raw_text)

    @classmethod
    def parse_document(cls, filename: str, file_content: Union[str, Path, BinaryIO, bytes]) -> str:
        """Extract and clean text based on file extension."""
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return cls.extract_text_from_pdf(file_content)
        elif ext == ".docx":
            return cls.extract_text_from_docx(file_content)
        elif ext == ".txt":
            return cls.extract_text_from_txt(file_content)
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: {cls.SUPPORTED_EXTENSIONS}")
