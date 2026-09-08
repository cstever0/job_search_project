"""Data models for match evaluation, scoring breakdown, evidence, and human feedback."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field

from config import (
    WEIGHT_CAREER_INTEREST,
    WEIGHT_EDUCATION,
    WEIGHT_EXPERIENCE,
    WEIGHT_JOB_TYPE,
    WEIGHT_SALARY,
    WEIGHT_SKILLS,
    WEIGHT_TOTAL,
)
from src.models.job import NormalizedJob


class RetrievedEvidence(BaseModel):
    """Specific evidence retrieved from the candidate's personal documents."""

    qualification: str = Field(description="Job requirement or skill evaluated")
    evidence_text: str = Field(description="Text snippet extracted from personal career documents")
    source_document: str = Field(description="Document where evidence was found")
    section: str = Field(description="Career section name (e.g. Work Experience, Projects)")
    chunk_id: str = Field(description="Identifier of the supporting CareerChunk")
    similarity_score: float = Field(default=0.0, description="Hybrid retrieval relevance score")
    supports_match: bool = Field(default=True, description="True if evidence affirmatively supports match")
    provenance: str = Field(
        default="CANDIDATE EVIDENCE",
        description="Data provenance label: CANDIDATE EVIDENCE or No evidence found",
    )


class ScoreBreakdown(BaseModel):
    """Component score breakdown strictly conforming to the 100-point fixed model.

    Skills: 35%
    Experience: 25%
    Education: 15%
    Career Interest: 15%
    Salary: 5%
    Job Type: 5%
    Total: 100%
    """

    skills_score: float = Field(default=0.0, ge=0.0, le=WEIGHT_SKILLS)
    skills_max: float = Field(default=WEIGHT_SKILLS)

    experience_score: float = Field(default=0.0, ge=0.0, le=WEIGHT_EXPERIENCE)
    experience_max: float = Field(default=WEIGHT_EXPERIENCE)

    education_score: float = Field(default=0.0, ge=0.0, le=WEIGHT_EDUCATION)
    education_max: float = Field(default=WEIGHT_EDUCATION)

    career_interest_score: float = Field(default=0.0, ge=0.0, le=WEIGHT_CAREER_INTEREST)
    career_interest_max: float = Field(default=WEIGHT_CAREER_INTEREST)

    salary_score: float = Field(default=0.0, ge=0.0, le=WEIGHT_SALARY)
    salary_max: float = Field(default=WEIGHT_SALARY)

    job_type_score: float = Field(default=0.0, ge=0.0, le=WEIGHT_JOB_TYPE)
    job_type_max: float = Field(default=WEIGHT_JOB_TYPE)

    overall_score: float = Field(default=0.0, ge=0.0, le=WEIGHT_TOTAL)

    ineligible_flag: bool = Field(default=False, description="True if an explicit mandatory qualification is missing")
    ineligible_reason: Optional[str] = Field(
        default=None, description="Explanation if marked ineligible/disqualified"
    )

    def calculate_total(self) -> float:
        """Compute the overall score as the exact sum of the six components."""
        total = round(
            self.skills_score
            + self.experience_score
            + self.education_score
            + self.career_interest_score
            + self.salary_score
            + self.job_type_score,
            2,
        )
        self.overall_score = min(WEIGHT_TOTAL, max(0.0, total))
        return self.overall_score

    def verify_sum(self) -> bool:
        """Verify that overall_score equals sum of components within floating-point tolerance."""
        component_sum = (
            self.skills_score
            + self.experience_score
            + self.education_score
            + self.career_interest_score
            + self.salary_score
            + self.job_type_score
        )
        return abs(self.overall_score - component_sum) < 0.05


class JobMatchEvaluation(BaseModel):
    """Complete evidence-grounded evaluation for a single job."""

    job: NormalizedJob
    breakdown: ScoreBreakdown
    matched_qualifications: list[RetrievedEvidence] = Field(default_factory=list)
    missing_required_qualifications: list[str] = Field(default_factory=list)
    missing_preferred_qualifications: list[str] = Field(default_factory=list)
    candidate_strengths: list[str] = Field(default_factory=list)
    retrieved_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    onet_occupational_info: dict[str, Any] = Field(default_factory=dict)
    skills_to_improve_competitiveness: list[str] = Field(
        default_factory=list,
        description="O*NET occupational skills not explicitly required by employer, suggested for career growth",
    )
    ranking_explanation: str = Field(
        default="", description="Transparent explanation of why this job received its score"
    )
    user_feedback: Optional[str] = Field(default=None, description="Human feedback label if recorded")


class HumanFeedback(BaseModel):
    """User feedback for a recommended job."""

    job_id: str
    job_title: str
    employer: str
    status: str = Field(description="Interested, Not Interested, Good Match, Poor Match, Applied")
    notes: str = Field(default="")
    overall_score: float = Field(default=0.0)
    created_at: str
    updated_at: str
