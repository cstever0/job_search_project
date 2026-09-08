"""Data models for candidate profile, career documents, and semantic chunks."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class CareerChunk(BaseModel):
    """A semantically coherent career unit extracted from personal documents.

    Represents one work position, one project, one education entry,
    one skills group, or one certification.
    """

    chunk_id: str = Field(description="Unique identifier for this career chunk")
    source_document: str = Field(description="Filename of the uploaded document")
    section: str = Field(
        description="Career section type: work_experience, projects, education, skills, certifications, preferences"
    )
    job_title: Optional[str] = Field(default=None, description="Job title if from work experience")
    organization: Optional[str] = Field(default=None, description="Company, university, or organization")
    project_name: Optional[str] = Field(default=None, description="Project name if from academic/personal projects")
    skills: list[str] = Field(default_factory=list, description="Specific skills mentioned in this chunk")
    education: Optional[str] = Field(default=None, description="Degree or academic program if applicable")
    date: Optional[str] = Field(default=None, description="Date range or graduation year if applicable")
    content: str = Field(description="Cleaned, full textual content of this career unit")

    def to_citation(self) -> str:
        """Format a human-readable citation of where this evidence originated."""
        parts = [f"Doc: {self.source_document}", f"Section: {self.section.replace('_', ' ').title()}"]
        if self.job_title and self.organization:
            parts.append(f"{self.job_title} @ {self.organization}")
        elif self.project_name:
            parts.append(f"Project: {self.project_name}")
        elif self.education:
            parts.append(f"Degree: {self.education}")
        if self.date:
            parts.append(f"({self.date})")
        parts.append(f"[Chunk ID: {self.chunk_id}]")
        return " | ".join(parts)


class CandidateProfile(BaseModel):
    """Structured candidate profile capturing preferences and career parameters."""

    name: str = Field(default="Candidate", description="Full name")
    target_roles: list[str] = Field(
        default_factory=lambda: ["Data Scientist", "Machine Learning Engineer", "AI Engineer"],
        description="Preferred job titles/roles",
    )
    career_interests: list[str] = Field(
        default_factory=lambda: [
            "Machine learning",
            "Natural language processing",
            "Data analytics",
            "AI systems development",
        ],
        description="Core career interests and domains",
    )
    career_goals: str = Field(
        default="To design, evaluate, and deploy scalable AI and data science solutions that deliver real-world impact.",
        description="Long-term career objective or statement",
    )
    technical_skills: list[str] = Field(
        default_factory=list, description="List of technical skills supported by career documents"
    )
    preferred_locations: list[str] = Field(
        default_factory=lambda: ["Washington, DC", "Remote", "Anywhere (Geographically Flexible)"],
        description="Preferred locations (informational only; not scored)",
    )
    work_modes: list[str] = Field(
        default_factory=lambda: ["Remote", "Hybrid", "On-site"],
        description="Accepted work styles",
    )
    salary_expectation: Optional[float] = Field(
        default=95000.0, description="Target minimum annual salary"
    )
    job_types: list[str] = Field(
        default_factory=lambda: ["Full-Time", "Permanent"],
        description="Preferred employment types (Full-Time, Permanent, etc.)",
    )
    education_level: str = Field(
        default="Master's Degree", description="Highest level of completed or in-progress education"
    )
    years_of_experience: float = Field(
        default=2.0, description="Estimated total years of professional/academic technical experience"
    )
    clearance: Optional[str] = Field(
        default=None, description="Security clearance if held (e.g., Secret, Top Secret, None)"
    )
