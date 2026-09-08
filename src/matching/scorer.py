"""Evidence-grounded scoring engine enforcing the fixed 100-point matching model."""

from __future__ import annotations

import re
from typing import Optional

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
from src.models.match import RetrievedEvidence, ScoreBreakdown
from src.models.profile import CandidateProfile
from src.rag.hybrid_retriever import HybridCareerRetriever


class JobScorer:
    """Calculates deterministic component scores strictly adhering to the 100-point system.

    Component Weights:
    - Skills: 35%
    - Experience: 25%
    - Education: 15%
    - Career Interest: 15%
    - Salary: 5%
    - Job Type: 5%
    Total: 100%
    """

    def __init__(self, retriever: HybridCareerRetriever):
        self.retriever = retriever

    def score_job(
        self,
        job: NormalizedJob,
        profile: CandidateProfile,
    ) -> tuple[ScoreBreakdown, list[RetrievedEvidence]]:
        """Compute the 6 component scores grounded in candidate career evidence.

        Returns (ScoreBreakdown, list of RetrievedEvidence).
        """
        all_evidence: list[RetrievedEvidence] = []

        # 1. SKILLS SCORE (0.0 to 35.0)
        skills_score, skills_evidence = self._score_skills(job)
        all_evidence.extend(skills_evidence)

        # 2. EXPERIENCE SCORE (0.0 to 25.0)
        exp_score, exp_evidence = self._score_experience(job, profile)
        all_evidence.extend(exp_evidence)

        # 3. EDUCATION SCORE (0.0 to 15.0)
        edu_score, edu_evidence = self._score_education(job, profile)
        all_evidence.extend(edu_evidence)

        # 4. CAREER INTEREST SCORE (0.0 to 15.0)
        interest_score = self._score_career_interest(job, profile)

        # 5. SALARY SCORE (0.0 to 5.0 - Soft Preference)
        salary_score = self._score_salary(job, profile)

        # 6. JOB TYPE SCORE (0.0 to 5.0)
        job_type_score = self._score_job_type(job, profile)

        # Check mandatory disqualifiers (e.g. strict clearance or license)
        ineligible_flag, ineligible_reason = self._check_mandatory_disqualifiers(job, profile)

        # Construct and calculate breakdown
        breakdown = ScoreBreakdown(
            skills_score=round(skills_score, 2),
            skills_max=WEIGHT_SKILLS,
            experience_score=round(exp_score, 2),
            experience_max=WEIGHT_EXPERIENCE,
            education_score=round(edu_score, 2),
            education_max=WEIGHT_EDUCATION,
            career_interest_score=round(interest_score, 2),
            career_interest_max=WEIGHT_CAREER_INTEREST,
            salary_score=round(salary_score, 2),
            salary_max=WEIGHT_SALARY,
            job_type_score=round(job_type_score, 2),
            job_type_max=WEIGHT_JOB_TYPE,
            ineligible_flag=ineligible_flag,
            ineligible_reason=ineligible_reason,
        )

        # Guarantee component scores mathematically sum to overall_score
        breakdown.calculate_total()
        assert breakdown.verify_sum(), "Component scores must strictly sum to overall score"

        return breakdown, all_evidence

    def _score_skills(
        self, job: NormalizedJob
    ) -> tuple[float, list[RetrievedEvidence]]:
        """Score skills (max 35.0) based on required and preferred employer skills."""
        evidence_list: list[RetrievedEvidence] = []

        req_skills = job.required_skills
        pref_skills = job.preferred_skills

        # If job has no explicit skills listed, evaluate duties as skill proxies
        if not req_skills and not pref_skills:
            test_items = job.responsibilities[:3] or [job.title]
            matched_count = 0
            for item in test_items:
                ev = self.retriever.retrieve_evidence_for_qualification(item)
                evidence_list.append(ev)
                if ev.supports_match:
                    matched_count += 1
            ratio = matched_count / max(1, len(test_items))
            return round(ratio * WEIGHT_SKILLS, 2), evidence_list

        req_weight = 0.70 * WEIGHT_SKILLS  # 24.5 points
        pref_weight = 0.30 * WEIGHT_SKILLS  # 10.5 points

        req_points = 0.0
        if req_skills:
            req_unit = req_weight / len(req_skills)
            for skill in req_skills:
                ev = self.retriever.retrieve_evidence_for_qualification(f"Technical skill proficiency in {skill}")
                evidence_list.append(ev)
                if ev.supports_match:
                    # Scale by retrieval confidence
                    req_points += req_unit * min(1.0, max(0.5, ev.similarity_score))
        else:
            req_points = req_weight * 0.75  # neutral if none explicitly specified

        pref_points = 0.0
        if pref_skills:
            pref_unit = pref_weight / len(pref_skills)
            for skill in pref_skills:
                ev = self.retriever.retrieve_evidence_for_qualification(f"Experience with {skill}")
                evidence_list.append(ev)
                if ev.supports_match:
                    pref_points += pref_unit * min(1.0, max(0.5, ev.similarity_score))
        else:
            pref_points = pref_weight * 0.5

        total_skills = min(WEIGHT_SKILLS, max(0.0, req_points + pref_points))
        return round(total_skills, 2), evidence_list

    def _score_experience(
        self, job: NormalizedJob, profile: CandidateProfile
    ) -> tuple[float, list[RetrievedEvidence]]:
        """Score experience (max 25.0) evaluating tenure and relevance of work chunks."""
        evidence_list: list[RetrievedEvidence] = []
        work_chunks = [c for c in self.retriever.vector_store.chunks if c.section == "work_experience"]

        if not work_chunks:
            # Fallback to general career chunks
            work_chunks = self.retriever.vector_store.chunks

        # Evaluate against job responsibilities
        key_duties = job.responsibilities[:2] or [f"Experience relevant to {job.title}"]
        duty_matches = 0
        for duty in key_duties:
            ev = self.retriever.retrieve_evidence_for_qualification(duty)
            evidence_list.append(ev)
            if ev.supports_match:
                duty_matches += 1

        relevance_factor = duty_matches / max(1, len(key_duties))

        # Tenure factor based on candidate profile years of experience
        # 1-2 years -> ~0.7, 3+ years -> 1.0
        tenure_factor = min(1.0, profile.years_of_experience / 3.0)

        # Blend relevance (60%) and tenure (40%)
        blended = (0.60 * relevance_factor + 0.40 * tenure_factor)
        exp_score = min(WEIGHT_EXPERIENCE, max(0.0, blended * WEIGHT_EXPERIENCE))
        return round(exp_score, 2), evidence_list

    def _score_education(
        self, job: NormalizedJob, profile: CandidateProfile
    ) -> tuple[float, list[RetrievedEvidence]]:
        """Score education (max 15.0) checking degree level and field of study."""
        evidence_list: list[RetrievedEvidence] = []
        edu_query = f"Degree in {job.title} Computer Science Data Science engineering STEM"
        ev = self.retriever.retrieve_evidence_for_qualification(edu_query)
        evidence_list.append(ev)

        # Baseline based on candidate declared education level
        cand_edu = profile.education_level.lower()
        if "ph" in cand_edu or "doctor" in cand_edu:
            degree_score = 1.0
        elif "master" in cand_edu or "m.s." in cand_edu:
            degree_score = 0.95
        elif "bachelor" in cand_edu or "b.s." in cand_edu:
            degree_score = 0.85
        else:
            degree_score = 0.70

        # Adjust with document grounding
        if ev.supports_match:
            final_factor = min(1.0, 0.5 * degree_score + 0.5 * max(0.8, ev.similarity_score))
        else:
            final_factor = degree_score * 0.8

        edu_score = min(WEIGHT_EDUCATION, max(0.0, final_factor * WEIGHT_EDUCATION))
        return round(edu_score, 2), evidence_list

    def _score_career_interest(
        self, job: NormalizedJob, profile: CandidateProfile
    ) -> float:
        """Score career interest (max 15.0) matching job title and duties with interests."""
        job_title_lower = job.title.lower()
        job_desc_lower = (job.description + " " + " ".join(job.responsibilities)).lower()

        # 1. Target roles match (0.50 of interest score)
        role_match = 0.0
        for role in profile.target_roles:
            r_lower = role.lower()
            if r_lower in job_title_lower:
                role_match = 1.0
                break
            # Partial match (e.g. "Data" or "Engineer" or "Scientist")
            words = [w for w in r_lower.split() if len(w) > 3]
            matched_words = sum(1 for w in words if w in job_title_lower)
            if words and matched_words > 0:
                role_match = max(role_match, matched_words / len(words))

        # 2. Career interests & goals match (0.50 of interest score)
        interest_match = 0.0
        if profile.career_interests:
            matched_interests = 0
            for interest in profile.career_interests:
                i_lower = interest.lower()
                if i_lower in job_desc_lower or i_lower in job_title_lower:
                    matched_interests += 1
            interest_match = min(1.0, matched_interests / max(1, len(profile.career_interests) // 2))
        else:
            interest_match = 0.75

        blended = 0.55 * role_match + 0.45 * interest_match
        interest_score = min(WEIGHT_CAREER_INTEREST, max(0.0, blended * WEIGHT_CAREER_INTEREST))
        return round(interest_score, 2)

    def _score_salary(self, job: NormalizedJob, profile: CandidateProfile) -> float:
        """Score salary (max 5.0) as a soft preference."""
        if profile.salary_expectation is None or profile.salary_expectation <= 0:
            return WEIGHT_SALARY  # No salary requirement set -> full points

        # If job has no disclosed salary, give a neutral score (3.5 / 5.0)
        if job.salary_min is None and job.salary_max is None:
            return round(0.70 * WEIGHT_SALARY, 2)

        # Use maximum offered or minimum offered
        offered = job.salary_max or job.salary_min or 0.0
        if job.salary_type == "Per Hour":
            # Normalize hourly to approximate annual (2080 hours)
            offered = offered * 2080

        expected = profile.salary_expectation

        if offered >= expected:
            # Exceeds or meets expectation -> full 5.0 points
            return WEIGHT_SALARY
        else:
            # Soft scaling down (never automatic rejection or 0 unless drastically lower)
            ratio = offered / max(1.0, expected)
            scaled = min(WEIGHT_SALARY, max(1.0, ratio * WEIGHT_SALARY))
            return round(scaled, 2)

    def _score_job_type(self, job: NormalizedJob, profile: CandidateProfile) -> float:
        """Score job type / work schedule (max 5.0)."""
        if not profile.job_types:
            return WEIGHT_JOB_TYPE

        cand_types_lower = [t.lower() for t in profile.job_types]
        job_type_lower = job.job_type.lower()

        # Direct match (e.g. full-time, permanent)
        if any(t in job_type_lower for t in cand_types_lower):
            return WEIGHT_JOB_TYPE

        # Default standard full-time
        if "full-time" in job_type_lower:
            return round(0.90 * WEIGHT_JOB_TYPE, 2)

        return round(0.50 * WEIGHT_JOB_TYPE, 2)

    def _check_mandatory_disqualifiers(
        self, job: NormalizedJob, profile: CandidateProfile
    ) -> tuple[bool, Optional[str]]:
        """Inspect if job contains a hard prerequisite the candidate clearly lacks."""
        for disq in job.mandatory_disqualifiers:
            disq_lower = disq.lower()
            if "commercial pilot" in disq_lower:
                return True, "Job requires active Commercial Pilot License with Instrument Rating."
            if "top secret" in disq_lower and (not profile.clearance or "top secret" not in profile.clearance.lower()):
                return True, "Job mandates active Top Secret security clearance."

        return False, None
