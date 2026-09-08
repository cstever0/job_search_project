"""Unit tests for the scoring model, verifying strict mathematical sum and guardrails."""

import pytest
from src.matching.scorer import JobScorer
from src.models.job import NormalizedJob
from src.models.profile import CandidateProfile, CareerChunk
from src.rag.hybrid_retriever import HybridCareerRetriever
from src.rag.vector_store import CareerVectorStore


@pytest.fixture
def mock_retriever(tmp_path) -> HybridCareerRetriever:
    store = CareerVectorStore(persist_dir=tmp_path / "vec_score_test")
    chunks = [
        CareerChunk(
            chunk_id="chk_01",
            source_document="resume.pdf",
            section="work_experience",
            job_title="ML Engineer",
            organization="GovTech",
            skills=["Python", "PyTorch", "NLP", "BERT", "AWS"],
            content="Built deep learning and NLP models in Python and PyTorch.",
        ),
        CareerChunk(
            chunk_id="chk_02",
            source_document="resume.pdf",
            section="education",
            education="Master of Science in Computer Science",
            skills=["Python", "Machine Learning"],
            content="Master of Science in Computer Science with thesis on Machine Learning.",
        ),
    ]
    store.add_chunks(chunks)
    return HybridCareerRetriever(store)


@pytest.fixture
def candidate_profile() -> CandidateProfile:
    return CandidateProfile(
        name="Alex Mercer",
        target_roles=["Data Scientist", "Machine Learning Engineer"],
        career_interests=["Machine Learning", "NLP", "Predictive Modeling"],
        education_level="Master's Degree",
        years_of_experience=3.0,
        salary_expectation=100000.0,
        job_types=["Full-Time"],
        clearance=None,
    )


def test_scorer_six_components_mathematically_equal_overall_score(mock_retriever, candidate_profile):
    """VERIFY: The six component scores must mathematically equal the overall score."""
    scorer = JobScorer(mock_retriever)

    test_jobs = [
        NormalizedJob(
            job_id="job_1",
            title="Senior Machine Learning Engineer",
            employer="AI Federal Labs",
            salary_min=120000.0,
            salary_max=150000.0,
            job_type="Full-Time",
            required_skills=["Python", "PyTorch", "NLP"],
            preferred_skills=["AWS", "Docker"],
            responsibilities=["Build NLP pipelines", "Deploy machine learning models"],
        ),
        NormalizedJob(
            job_id="job_2",
            title="Junior Helpdesk Technician",
            employer="Agency X",
            salary_min=45000.0,
            salary_max=55000.0,
            job_type="Part-Time",
            required_skills=["Windows Support", "Hardware"],
            preferred_skills=[],
            responsibilities=["Fix desktop hardware and printers"],
        ),
        NormalizedJob(
            job_id="job_3",
            title="General Specialist",
            employer="Agency Y",
            # Missing salary, missing skills
            salary_min=None,
            salary_max=None,
            job_type="Full-Time",
            required_skills=[],
            preferred_skills=[],
            responsibilities=[],
        ),
    ]

    for job in test_jobs:
        breakdown, evidence = scorer.score_job(job, candidate_profile)

        # 1. Check components are within theoretical bounds
        assert 0.0 <= breakdown.skills_score <= 35.0
        assert 0.0 <= breakdown.experience_score <= 25.0
        assert 0.0 <= breakdown.education_score <= 15.0
        assert 0.0 <= breakdown.career_interest_score <= 15.0
        assert 0.0 <= breakdown.salary_score <= 5.0
        assert 0.0 <= breakdown.job_type_score <= 5.0

        # 2. Strict mathematical sum guarantee
        computed_sum = round(
            breakdown.skills_score
            + breakdown.experience_score
            + breakdown.education_score
            + breakdown.career_interest_score
            + breakdown.salary_score
            + breakdown.job_type_score,
            2,
        )

        assert abs(breakdown.overall_score - computed_sum) < 0.01, (
            f"Overall score {breakdown.overall_score} does not match sum of components {computed_sum}"
        )
        assert breakdown.verify_sum()


def test_scorer_mandatory_disqualification_flag(mock_retriever, candidate_profile):
    scorer = JobScorer(mock_retriever)
    disqualified_job = NormalizedJob(
        job_id="job_pilot",
        title="Commercial Flight Inspector",
        employer="FAA",
        job_type="Full-Time",
        mandatory_disqualifiers=["Commercial pilot certificate with instrument rating required"],
    )

    breakdown, _ = scorer.score_job(disqualified_job, candidate_profile)
    assert breakdown.ineligible_flag is True
    assert breakdown.ineligible_reason is not None
    assert "Commercial Pilot" in breakdown.ineligible_reason
    # Scores are still calculated deterministically
    assert breakdown.verify_sum()
