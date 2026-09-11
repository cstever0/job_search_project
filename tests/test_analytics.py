"""Unit tests for the MarketSkillsEducationAnalyzer module."""

import pytest
from src.matching.analytics import MarketSkillsEducationAnalyzer
from src.models.job import NormalizedJob
from src.models.profile import CandidateProfile, CareerChunk


@pytest.fixture
def sample_test_jobs() -> list[NormalizedJob]:
    return [
        NormalizedJob(
            job_id="j1",
            title="AI & Machine Learning Specialist",
            employer="Agency A",
            required_skills=["Python", "PyTorch", "NLP", "Machine Learning"],
            preferred_skills=["Docker", "AWS", "SQL"],
            education_requirements=["Master's degree in Computer Science or related STEM discipline."],
            onet_technology_skills=["Python", "PyTorch", "TensorFlow", "Hugging Face"],
            onet_occupational_skills=["Mathematics", "Complex Problem Solving"],
        ),
        NormalizedJob(
            job_id="j2",
            title="Data Scientist",
            employer="Agency B",
            required_skills=["Python", "SQL", "Statistics", "Pandas"],
            preferred_skills=["Machine Learning", "Git"],
            education_requirements=["Bachelor's degree in Statistics or Mathematics."],
            onet_technology_skills=["Python", "R", "SQL", "Apache Spark"],
            onet_occupational_skills=["Critical Thinking", "Mathematics"],
        ),
        NormalizedJob(
            job_id="j3",
            title="Computer Scientist",
            employer="Agency C",
            required_skills=["Python", "PyTorch", "BERT", "Linux"],
            preferred_skills=["Docker", "Git"],
            education_requirements=["Ph.D. or Master's in Computer Science."],
            onet_technology_skills=["Python", "C++", "Java", "Docker"],
            onet_occupational_skills=["Programming", "Systems Analysis"],
        ),
    ]


@pytest.fixture
def sample_candidate() -> CandidateProfile:
    return CandidateProfile(
        name="Cory Stever",
        education_level="Master's Degree (Computer Science - AI, In Progress)",
        technical_skills=["Python", "PyTorch", "SQL", "BERT", "React", "Flask"],
    )


def test_skills_analysis_frequencies(sample_test_jobs, sample_candidate):
    candidate_skills = MarketSkillsEducationAnalyzer.get_candidate_skills(sample_candidate)
    analysis = MarketSkillsEducationAnalyzer.analyze_skills(sample_test_jobs, candidate_skills)

    top_req = dict(analysis["top_employer_required"])
    assert top_req.get("Python") == 3
    assert top_req.get("Pytorch") == 2

    # Check candidate possessed vs missing
    possessed = [item["skill"].lower() for item in analysis["candidate_possessed_skills"]]
    assert "python" in possessed
    assert "pytorch" in possessed

    gaps = [item["skill"].lower() for item in analysis["candidate_skill_gaps"]]
    assert "linux" in gaps or "statistics" in gaps

    assert 0.0 < analysis["candidate_skill_coverage_pct"] <= 100.0


def test_education_analysis_distribution(sample_test_jobs, sample_candidate):
    analysis = MarketSkillsEducationAnalyzer.analyze_education(sample_test_jobs, sample_candidate)
    dist = dict(analysis["degree_level_distribution"])

    # Postings contain Ph.D., Master's, and Bachelor's
    assert any("Master" in k or "Ph.D." in k or "Bachelor" in k for k in dist)
    assert len(analysis["top_disciplines"]) > 0
    assert "Master's level" in analysis["candidate_alignment_note"]


def test_market_summary_generation(sample_test_jobs, sample_candidate):
    summary = MarketSkillsEducationAnalyzer.generate_market_summary(sample_test_jobs, sample_candidate)
    assert summary["total_jobs_analyzed"] == 3
    assert summary["most_sought_after_skill"] == "Python"
    assert "skills" in summary
    assert "education" in summary
