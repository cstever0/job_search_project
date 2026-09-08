"""Semantic chunker for career documents.

Chunks documents into meaningful career units (e.g., one work position,
one academic project, one degree entry, one skills section, one certification)
rather than arbitrary fixed-size character splits.
"""

from __future__ import annotations

import re
import uuid
from typing import Optional
from src.models.profile import CareerChunk

# Common technical keywords to tag chunks
COMMON_SKILL_PATTERNS: list[str] = [
    r"\bpython\b",
    r"\br\b",
    r"\bsql\b",
    r"\bpostgres(?:ql)?\b",
    r"\bmysql\b",
    r"\bmongodb\b",
    r"\bpytorch\b",
    r"\btensorflow\b",
    r"\bkeras\b",
    r"\bscikit-learn\b",
    r"\bsklearn\b",
    r"\bpandas\b",
    r"\bnumpy\b",
    r"\bspark\b",
    r"\bpyspark\b",
    r"\bhadoop\b",
    r"\baws\b",
    r"\bazure\b",
    r"\bgcp\b",
    r"\bdocker\b",
    r"\bkubernetes\b",
    r"\bbert\b",
    r"\bgpt\b",
    r"\btransformers?\b",
    r"\bllms?\b",
    r"\bnlp\b",
    r"\bcomputer vision\b",
    r"\bdeep learning\b",
    r"\bmachine learning\b",
    r"\bdata science\b",
    r"\bflask\b",
    r"\bfastapi\b",
    r"\bdjango\b",
    r"\breact\b",
    r"\bnode(?:\.js)?\b",
    r"\bjavascript\b",
    r"\btypescript\b",
    r"\bgit\b",
    r"\bgithub\b",
    r"\bci/cd\b",
    r"\btableau\b",
    r"\bpower bi\b",
    r"\blinux\b",
    r"\bstatistics\b",
    r"\bdata pipelines?\b",
    r"\betl\b",
]

# Section header patterns
SECTION_PATTERNS: dict[str, str] = {
    "work_experience": r"^(?:(?:work|professional|relevant|employment)\s+experience|experience|employment history|work history)\b",
    "projects": r"^(?:(?:academic|technical|personal|selected)\s+projects|projects|portfolio)\b",
    "education": r"^(?:education|academic background|academic qualifications|degrees)\b",
    "skills": r"^(?:technical skills|skills|core competencies|competencies|technologies|tools & technologies)\b",
    "certifications": r"^(?:certifications?|licenses?|training|professional development)\b",
    "preferences": r"^(?:career interests|career goals|objective|professional summary|summary)\b",
}


class CareerChunker:
    """Chunks career documents into semantic career units with rich metadata."""

    @staticmethod
    def extract_skills_from_text(text: str) -> list[str]:
        """Extract recognized technical skill keywords present in the text."""
        skills: set[str] = set()
        lower_text = text.lower()
        for pattern in COMMON_SKILL_PATTERNS:
            match = re.search(pattern, lower_text)
            if match:
                skill_name = match.group(0).strip()
                # Clean and standardize common terms
                standardized = {
                    "sklearn": "scikit-learn",
                    "postgres": "postgresql",
                    "node": "node.js",
                    "transformers": "transformer models",
                    "transformer": "transformer models",
                    "llms": "large language models",
                    "llm": "large language models",
                }.get(skill_name, skill_name)
                skills.add(standardized.title() if len(standardized) > 3 else standardized.upper())
        return sorted(list(skills))

    @classmethod
    def identify_section(cls, line: str) -> Optional[str]:
        """Determine if a line is a section heading."""
        cleaned = line.strip().lower()
        # Remove markdown heading hashes or trailing colons
        cleaned = re.sub(r"^#+\s*", "", cleaned)
        cleaned = re.sub(r"[:\-_]+$", "", cleaned).strip()

        if len(cleaned) > 50:
            return None

        for section_name, pattern in SECTION_PATTERNS.items():
            if re.match(pattern, cleaned, re.IGNORECASE):
                return section_name
        return None

    @classmethod
    def _split_into_sections(cls, text: str) -> dict[str, list[str]]:
        """Group document lines under their detected career sections."""
        lines = text.split("\n")
        sections: dict[str, list[str]] = {}
        current_section = "general"
        sections[current_section] = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            detected = cls.identify_section(line_str)
            if detected:
                current_section = detected
                if current_section not in sections:
                    sections[current_section] = []
            else:
                sections[current_section].append(line_str)

        return sections

    @classmethod
    def chunk_document(cls, text: str, source_document: str) -> list[CareerChunk]:
        """Convert document text into a list of semantically rich CareerChunk objects."""
        if not text or not text.strip():
            return []

        sections = cls._split_into_sections(text)
        chunks: list[CareerChunk] = []
        chunk_idx = 1

        for section_name, lines in sections.items():
            if not lines:
                continue

            section_text = "\n".join(lines)

            if section_name in ("work_experience", "projects"):
                # Sub-chunk by distinct job positions or distinct projects
                sub_units = cls._split_positions_or_projects(lines)
                for unit in sub_units:
                    unit_text = "\n".join(unit).strip()
                    if len(unit_text) < 15:
                        continue

                    # Extract entity names if discernible
                    first_line = unit[0]
                    job_title = None
                    org = None
                    proj_name = None

                    if section_name == "work_experience":
                        # Pattern like "Senior ML Engineer | Google | 2021 - Present"
                        parts = [p.strip() for p in re.split(r"\||–|-{2,}| at ", first_line) if p.strip()]
                        if len(parts) >= 2:
                            job_title = parts[0]
                            org = parts[1]
                        else:
                            job_title = first_line[:60]
                    else:
                        proj_name = first_line.split(":")[0].split("|")[0].strip()[:60]

                    # Extract dates
                    date_match = re.search(
                        r"\b(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?\d{4}\s*(?:-|–|to)\s*(?:Present|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?\d{4})\b",
                        unit_text,
                        re.IGNORECASE,
                    )
                    date_str = date_match.group(0) if date_match else None

                    chunks.append(
                        CareerChunk(
                            chunk_id=f"{source_document}_{chunk_idx:03d}",
                            source_document=source_document,
                            section=section_name,
                            job_title=job_title,
                            organization=org,
                            project_name=proj_name,
                            skills=cls.extract_skills_from_text(unit_text),
                            date=date_str,
                            content=unit_text,
                        )
                    )
                    chunk_idx += 1

            elif section_name == "education":
                # Split education entries
                edu_units = cls._split_education_entries(lines)
                for unit in edu_units:
                    unit_text = "\n".join(unit).strip()
                    if len(unit_text) < 15:
                        continue

                    degree_match = re.search(
                        r"\b(Ph\.?D\.?|Master['’]s|Bachelor['’]s|B\.?S\.?|M\.?S\.?|B\.?A\.?|Associate)\b[^\n,]*",
                        unit_text,
                        re.IGNORECASE,
                    )
                    edu_degree = degree_match.group(0) if degree_match else None

                    date_match = re.search(r"\b(19\d{2}|20\d{2})\b", unit_text)
                    date_str = date_match.group(0) if date_match else None

                    chunks.append(
                        CareerChunk(
                            chunk_id=f"{source_document}_{chunk_idx:03d}",
                            source_document=source_document,
                            section=section_name,
                            education=edu_degree,
                            skills=cls.extract_skills_from_text(unit_text),
                            date=date_str,
                            content=unit_text,
                        )
                    )
                    chunk_idx += 1

            elif section_name == "skills":
                # Skills section: keep coherent groups (e.g. languages, frameworks, cloud)
                skill_subunits = cls._split_skills_blocks(lines)
                for unit in skill_subunits:
                    unit_text = "\n".join(unit).strip()
                    if not unit_text:
                        continue
                    chunks.append(
                        CareerChunk(
                            chunk_id=f"{source_document}_{chunk_idx:03d}",
                            source_document=source_document,
                            section=section_name,
                            skills=cls.extract_skills_from_text(unit_text),
                            content=unit_text,
                        )
                    )
                    chunk_idx += 1

            else:
                # General / Certifications / Preferences
                # Break into logical paragraph chunks
                para_units = cls._split_into_paragraphs(lines)
                for unit in para_units:
                    unit_text = "\n".join(unit).strip()
                    if len(unit_text) < 15:
                        continue
                    chunks.append(
                        CareerChunk(
                            chunk_id=f"{source_document}_{chunk_idx:03d}",
                            source_document=source_document,
                            section=section_name,
                            skills=cls.extract_skills_from_text(unit_text),
                            content=unit_text,
                        )
                    )
                    chunk_idx += 1

        # If no chunks were created (e.g. very unstructured document), fallback to paragraph chunks
        if not chunks:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for p in paragraphs:
                chunks.append(
                    CareerChunk(
                        chunk_id=f"{source_document}_{chunk_idx:03d}",
                        source_document=source_document,
                        section="general",
                        skills=cls.extract_skills_from_text(p),
                        content=p,
                    )
                )
                chunk_idx += 1

        return chunks

    @classmethod
    def _split_positions_or_projects(cls, lines: list[str]) -> list[list[str]]:
        """Segment a list of lines into distinct position or project entries."""
        units: list[list[str]] = []
        current: list[str] = []

        # Indicators of a new position or project header
        date_pattern = re.compile(
            r"\b(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?\d{4}\s*(?:-|–|to)\s*(?:Present|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?\d{4})\b",
            re.IGNORECASE,
        )

        for line in lines:
            # If line has a date range or starts with a bold/major delimiter and is not a bullet
            is_bullet = line.startswith(("-", "*", "•", "–"))
            has_date = bool(date_pattern.search(line))

            if current and has_date and not is_bullet:
                units.append(current)
                current = [line]
            elif current and not is_bullet and len(line) < 60 and ("|" in line or " - " in line or "@" in line):
                # Another header indicator
                units.append(current)
                current = [line]
            else:
                current.append(line)

        if current:
            units.append(current)

        return units if units else [lines]

    @classmethod
    def _split_education_entries(cls, lines: list[str]) -> list[list[str]]:
        """Segment education lines into discrete degree entries."""
        units: list[list[str]] = []
        current: list[str] = []
        degree_indicator = re.compile(
            r"\b(University|College|Institute|Ph\.?D\.?|Master|Bachelor|B\.?S\.?|M\.?S\.?|B\.?A\.?)\b",
            re.IGNORECASE,
        )

        for line in lines:
            if current and degree_indicator.search(line) and not line.startswith(("-", "*", "•")):
                units.append(current)
                current = [line]
            else:
                current.append(line)

        if current:
            units.append(current)
        return units if units else [lines]

    @classmethod
    def _split_skills_blocks(cls, lines: list[str]) -> list[list[str]]:
        """Group skills by category lines (e.g. 'Languages:', 'Frameworks:', etc.)."""
        units: list[list[str]] = []
        current: list[str] = []

        for line in lines:
            if ":" in line and not line.startswith(("-", "*", "•")):
                if current:
                    units.append(current)
                current = [line]
            else:
                current.append(line)

        if current:
            units.append(current)
        return units if units else [lines]

    @classmethod
    def _split_into_paragraphs(cls, lines: list[str]) -> list[list[str]]:
        """Group lines into paragraph units."""
        units: list[list[str]] = []
        current: list[str] = []

        for line in lines:
            current.append(line)
            if len("\n".join(current)) > 400:
                units.append(current)
                current = []

        if current:
            units.append(current)
        return units if units else [lines]
