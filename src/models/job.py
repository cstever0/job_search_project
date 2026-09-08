"""Normalized job schema and related data models."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class NormalizedJob(BaseModel):
    """Normalized representation of a job posting.

    Preserves strict separation between EMPLOYER REQUIREMENTS and
    O*NET OCCUPATIONAL INFORMATION.
    """

    job_id: str = Field(description="Unique job identifier")
    title: str = Field(description="Official job title")
    employer: str = Field(description="Employer / agency / department name")
    location: str = Field(default="Unspecified", description="City, State or Location description")
    is_remote: bool = Field(default=False, description="Whether job is explicitly marked remote")
    salary_min: Optional[float] = Field(default=None, description="Minimum salary/remuneration")
    salary_max: Optional[float] = Field(default=None, description="Maximum salary/remuneration")
    salary_type: str = Field(default="Per Year", description="Salary rate interval (e.g. Per Year, Per Hour)")
    job_type: str = Field(default="Full-Time", description="Work schedule / appointment type")

    # Employer Postings Details (Strictly Employer Requirement)
    description: str = Field(default="", description="Summary and overview text from employer")
    responsibilities: list[str] = Field(default_factory=list, description="Explicit duties listed by employer")
    required_qualifications: list[str] = Field(
        default_factory=list, description="Mandatory qualifications stated in employer job announcement"
    )
    preferred_qualifications: list[str] = Field(
        default_factory=list, description="Desired/preferred qualifications stated by employer"
    )
    required_skills: list[str] = Field(
        default_factory=list, description="Specific technical skills explicitly required by employer"
    )
    preferred_skills: list[str] = Field(
        default_factory=list, description="Specific technical skills preferred by employer"
    )
    experience_requirements: list[str] = Field(
        default_factory=list, description="Experience requirements stated by employer"
    )
    education_requirements: list[str] = Field(
        default_factory=list, description="Education requirements stated by employer"
    )
    mandatory_disqualifiers: list[str] = Field(
        default_factory=list, description="Explicit hard disqualifiers like security clearance or mandatory license"
    )

    # O*NET Occupational Enrichment (Strictly O*NET Occupational Information)
    onet_code: Optional[str] = Field(default=None, description="O*NET SOC code, e.g. 15-1252.00")
    onet_title: Optional[str] = Field(default=None, description="O*NET standard occupation title")
    onet_description: Optional[str] = Field(default=None, description="O*NET occupational description")
    onet_occupational_skills: list[str] = Field(
        default_factory=list, description="O*NET occupational skills (NOT employer requirements)"
    )
    onet_knowledge: list[str] = Field(
        default_factory=list, description="O*NET knowledge areas (NOT employer requirements)"
    )
    onet_abilities: list[str] = Field(
        default_factory=list, description="O*NET abilities (NOT employer requirements)"
    )
    onet_technology_skills: list[str] = Field(
        default_factory=list, description="O*NET technology skills (NOT employer requirements)"
    )
    onet_tasks: list[str] = Field(
        default_factory=list, description="O*NET typical tasks (NOT employer requirements)"
    )

    # Provenance & Metadata
    application_url: str = Field(default="", description="Direct URL to apply on USAJOBS or official site")
    source: str = Field(default="USAJOBS", description="Source of the job posting")
    source_url: str = Field(default="", description="Link to source announcement")
    raw_data: dict[str, Any] = Field(default_factory=dict, description="Original raw response from API")

    @property
    def salary_display(self) -> str:
        """Format salary range for human-readable display."""
        if self.salary_min is not None and self.salary_max is not None:
            if self.salary_min == self.salary_max:
                return f"${self.salary_min:,.0f} {self.salary_type}"
            return f"${self.salary_min:,.0f} - ${self.salary_max:,.0f} {self.salary_type}"
        elif self.salary_min is not None:
            return f"From ${self.salary_min:,.0f} {self.salary_type}"
        elif self.salary_max is not None:
            return f"Up to ${self.salary_max:,.0f} {self.salary_type}"
        return "Salary not disclosed"
