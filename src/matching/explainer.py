"""Evidence grounding and explainability engine.

Synthesizes matched vs missing requirements, cites career chunks,
and maintains strict provenance labels:
- EMPLOYER REQUIREMENT
- O*NET OCCUPATIONAL INFORMATION
- CANDIDATE EVIDENCE
"""

from __future__ import annotations

from typing import Any
from src.models.job import NormalizedJob
from src.models.match import JobMatchEvaluation, RetrievedEvidence, ScoreBreakdown
from src.models.profile import CandidateProfile
from src.rag.hybrid_retriever import HybridCareerRetriever


class MatchExplainer:
    """Generates transparent, evidence-grounded explanations and provenance tracking."""

    def __init__(self, retriever: HybridCareerRetriever):
        self.retriever = retriever

    def build_evaluation(
        self,
        job: NormalizedJob,
        profile: CandidateProfile,
        breakdown: ScoreBreakdown,
        evidence_list: list[RetrievedEvidence],
    ) -> JobMatchEvaluation:
        """Construct the complete JobMatchEvaluation with provenance and explanations."""
        matched_quals: list[RetrievedEvidence] = []
        missing_req: list[str] = []
        missing_pref: list[str] = []
        candidate_strengths: list[str] = []

        # Deduplicate evidence by qualification
        seen_quals: set[str] = set()
        deduped_evidence: list[RetrievedEvidence] = []
        for ev in evidence_list:
            if ev.qualification not in seen_quals:
                seen_quals.add(ev.qualification)
                deduped_evidence.append(ev)

        # Categorize matches vs missing requirements
        for ev in deduped_evidence:
            if ev.supports_match:
                matched_quals.append(ev)
                if ev.similarity_score > 0.60:
                    candidate_strengths.append(f"Strong grounding in {ev.qualification} ({ev.source_document})")
            else:
                # Check if it was required or preferred
                q_lower = ev.qualification.lower()
                is_pref = any(p.lower() in q_lower for p in job.preferred_skills) or "preferred" in q_lower
                if is_pref:
                    missing_pref.append(ev.qualification)
                else:
                    missing_req.append(ev.qualification)

        # Check O*NET skills for career development suggestions (separate from employer requirements)
        skills_for_competitiveness: list[str] = []
        if job.onet_technology_skills:
            # Candidate technical skills (from chunks or profile)
            cand_skills_lower = {s.lower() for s in profile.technical_skills}
            for c in self.retriever.vector_store.chunks:
                cand_skills_lower.update(s.lower() for s in c.skills)

            for onet_tech in job.onet_technology_skills[:10]:
                # Only suggest if not already possessed AND not already an explicit employer requirement
                if onet_tech.lower() not in cand_skills_lower and onet_tech.lower() not in [
                    s.lower() for s in job.required_skills
                ]:
                    skills_for_competitiveness.append(onet_tech)

        # Compile O*NET occupational info dict
        onet_info: dict[str, Any] = {
            "title": job.onet_title or "Occupational data",
            "code": job.onet_code or "N/A",
            "description": job.onet_description or "General occupational profile.",
            "skills": job.onet_occupational_skills[:5],
            "knowledge": job.onet_knowledge[:5],
            "abilities": job.onet_abilities[:5],
            "technology_skills": job.onet_technology_skills[:8],
        }

        # Synthesize transparent ranking explanation
        explanation = self._generate_explanation(job, profile, breakdown, matched_quals, missing_req)

        return JobMatchEvaluation(
            job=job,
            breakdown=breakdown,
            matched_qualifications=matched_quals,
            missing_required_qualifications=missing_req,
            missing_preferred_qualifications=missing_pref,
            candidate_strengths=candidate_strengths[:5],
            retrieved_evidence=deduped_evidence,
            onet_occupational_info=onet_info,
            skills_to_improve_competitiveness=skills_for_competitiveness[:6],
            ranking_explanation=explanation,
        )

    def _generate_explanation(
        self,
        job: NormalizedJob,
        profile: CandidateProfile,
        breakdown: ScoreBreakdown,
        matched: list[RetrievedEvidence],
        missing: list[str],
    ) -> str:
        """Formulate a transparent narrative explanation of the job match score."""
        parts = []

        if breakdown.ineligible_flag:
            parts.append(f"⚠️ **Ineligibility Warning:** {breakdown.ineligible_reason}")

        parts.append(
            f"This role scored **{breakdown.overall_score:.1f}/100** based on the fixed 6-component scoring model."
        )

        parts.append(
            f"- **Skills ({breakdown.skills_score:.1f}/{breakdown.skills_max:.0f}):** Grounded in {len(matched)} matched technical qualifications from your career documents."
        )
        parts.append(
            f"- **Experience ({breakdown.experience_score:.1f}/{breakdown.experience_max:.0f}):** Evaluated against your {profile.years_of_experience:.1f} years of relevant experience."
        )
        parts.append(
            f"- **Education ({breakdown.education_score:.1f}/{breakdown.education_max:.0f}):** Matched with your {profile.education_level} background."
        )
        parts.append(
            f"- **Career Interest ({breakdown.career_interest_score:.1f}/{breakdown.career_interest_max:.0f}):** Alignment between job duties and your target roles."
        )
        parts.append(
            f"- **Salary ({breakdown.salary_score:.1f}/{breakdown.salary_max:.0f}):** {job.salary_display} compared to your target ${profile.salary_expectation:,.0f}."
        )
        parts.append(
            f"- **Job Type ({breakdown.job_type_score:.1f}/{breakdown.job_type_max:.0f}):** {job.job_type} matches your stated employment preferences."
        )

        if missing:
            parts.append(
                f"**Missing Employer Requirements:** No evidence was found in your uploaded career documents for {len(missing)} requirement(s) (e.g., {', '.join(missing[:2])})."
            )

        return "\n\n".join(parts)
