"""Market analytics module summarizing most sought-after skills, education levels, and candidate gaps."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional
from src.models.job import NormalizedJob
from src.models.profile import CandidateProfile


class MarketSkillsEducationAnalyzer:
    """Aggregates and summarizes in-demand skills and education credentials from jobs."""

    @classmethod
    def get_candidate_skills(cls, profile: CandidateProfile, chunks: Optional[list[Any]] = None) -> set[str]:
        """Aggregate all technical skills demonstrated by candidate across profile and chunks."""
        skills: set[str] = {s.strip().title() for s in profile.technical_skills if s.strip()}
        if chunks:
            for c in chunks:
                for s in getattr(c, "skills", []):
                    if s.strip():
                        skills.add(s.strip().title())
        return skills

    @classmethod
    def analyze_skills(cls, jobs: list[NormalizedJob], candidate_skills: set[str]) -> dict[str, Any]:
        """Compute frequency of required vs preferred employer skills and O*NET technology skills."""
        employer_req_counter: Counter[str] = Counter()
        employer_pref_counter: Counter[str] = Counter()
        onet_tech_counter: Counter[str] = Counter()
        onet_competency_counter: Counter[str] = Counter()

        for job in jobs:
            # Employer Required Skills
            for skill in job.required_skills:
                norm_skill = skill.strip().title()
                employer_req_counter[norm_skill] += 1

            # Employer Preferred Skills
            for skill in job.preferred_skills:
                norm_skill = skill.strip().title()
                employer_pref_counter[norm_skill] += 1

            # O*NET Technology Skills
            for tech in job.onet_technology_skills:
                norm_tech = tech.strip().title()
                onet_tech_counter[norm_tech] += 1

            # O*NET Occupational Competencies
            for comp in job.onet_occupational_skills:
                norm_comp = comp.strip().title()
                onet_competency_counter[norm_comp] += 1

        # Combine all employer skills
        all_employer_skills = employer_req_counter + employer_pref_counter

        # Build candidate gap breakdown
        cand_skills_lower = {s.lower() for s in candidate_skills}

        possessed_skills: list[dict[str, Any]] = []
        missing_skill_gaps: list[dict[str, Any]] = []

        for skill, count in all_employer_skills.most_common():
            is_possessed = skill.lower() in cand_skills_lower
            item = {
                "skill": skill,
                "demand_count": count,
                "percentage": round((count / max(1, len(jobs))) * 100, 1),
                "status": "Possessed" if is_possessed else "Missing",
            }
            if is_possessed:
                possessed_skills.append(item)
            else:
                missing_skill_gaps.append(item)

        coverage_pct = round(
            (len(possessed_skills) / max(1, len(possessed_skills) + len(missing_skill_gaps))) * 100, 1
        )

        return {
            "top_employer_required": employer_req_counter.most_common(12),
            "top_employer_preferred": employer_pref_counter.most_common(10),
            "top_onet_tech_skills": onet_tech_counter.most_common(12),
            "top_onet_competencies": onet_competency_counter.most_common(8),
            "candidate_possessed_skills": possessed_skills,
            "candidate_skill_gaps": missing_skill_gaps,
            "candidate_skill_coverage_pct": coverage_pct,
        }

    @classmethod
    def analyze_education(cls, jobs: list[NormalizedJob], profile: CandidateProfile) -> dict[str, Any]:
        """Categorize and count education degree level and discipline requirements."""
        degree_levels: Counter[str] = Counter()
        disciplines: Counter[str] = Counter()

        level_patterns = [
            ("Ph.D. / Doctorate", [r"\bph\.?d\b", r"\bdoctorate\b", r"\bdoctoral\b"]),
            ("Master's Degree", [r"\bmaster['’]s\b", r"\bm\.?s\.?\b", r"\bm\.?a\.?\b", r"\bgraduate degree\b"]),
            ("Bachelor's Degree", [r"\bbachelor['’]s\b", r"\bb\.?s\.?\b", r"\bb\.?a\.?\b", r"\bundergraduate\b"]),
            ("Associate / High School", [r"\bassociate\b", r"\bhigh school\b", r"\bged\b"]),
        ]

        discipline_patterns = [
            ("Computer Science", [r"\bcomputer science\b", r"\bcs\b"]),
            ("Data Science / Analytics", [r"\bdata science\b", r"\bdata analytics\b", r"\banalytics\b"]),
            ("Software / Computer Engineering", [r"\bcomputer engineering\b", r"\bsoftware engineering\b"]),
            ("Mathematics / Statistics", [r"\bmathematics\b", r"\bmath\b", r"\bstatistics\b", r"\bapplied statistics\b"]),
            ("Information Technology / IS", [r"\binformation technology\b", r"\binformation systems\b", r"\bit\b", r"\bmis\b"]),
            ("General Engineering / STEM", [r"\bengineering\b", r"\bstem\b", r"\bquantitative discipline\b"]),
        ]

        for job in jobs:
            comb_edu_text = " ".join(job.education_requirements + job.required_qualifications + [job.description]).lower()

            # Identify degree levels
            found_level = False
            for label, pats in level_patterns:
                if any(re.search(p, comb_edu_text) for p in pats):
                    degree_levels[label] += 1
                    found_level = True
                    break  # Take highest mentioned level

            if not found_level:
                degree_levels["Bachelor's Degree or Equivalent"] += 1

            # Identify disciplines
            for disc_label, pats in discipline_patterns:
                if any(re.search(p, comb_edu_text) for p in pats):
                    disciplines[disc_label] += 1

        # Candidate alignment evaluation
        cand_edu = profile.education_level.lower()
        if "ph" in cand_edu or "doctor" in cand_edu:
            cand_tier = "Doctoral level (qualifies for 100% of educational tiers)"
        elif "master" in cand_edu or "m.s." in cand_edu:
            cand_tier = "Master's level (exceeds Bachelor's requirements and meets graduate postings)"
        else:
            cand_tier = "Bachelor's level (meets standard professional entry requirements)"

        return {
            "degree_level_distribution": degree_levels.most_common(),
            "top_disciplines": disciplines.most_common(8),
            "candidate_education_level": profile.education_level,
            "candidate_alignment_note": cand_tier,
        }

    @classmethod
    def generate_market_summary(
        cls,
        jobs: list[NormalizedJob],
        profile: CandidateProfile,
        chunks: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """Produce the comprehensive market skills and education summary report."""
        if not jobs:
            return {}

        cand_skills = cls.get_candidate_skills(profile, chunks)
        skill_analysis = cls.analyze_skills(jobs, cand_skills)
        edu_analysis = cls.analyze_education(jobs, profile)

        top_skill = skill_analysis["top_employer_required"][0][0] if skill_analysis["top_employer_required"] else "Python"
        top_edu = edu_analysis["degree_level_distribution"][0][0] if edu_analysis["degree_level_distribution"] else "Bachelor's Degree"

        return {
            "total_jobs_analyzed": len(jobs),
            "skills": skill_analysis,
            "education": edu_analysis,
            "most_sought_after_skill": top_skill,
            "most_common_education": top_edu,
            "candidate_skills_count": len(cand_skills),
        }
