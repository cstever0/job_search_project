"""End-to-end integration test verifying the complete MVP pipeline."""

from pathlib import Path
import pytest

from src.data_ingestion.semantic_chunker import CareerChunker
from src.feedback.feedback_store import FeedbackStore
from src.matching.ranker import JobRanker
from src.matching.scorer import JobScorer
from src.models.profile import CandidateProfile
from src.rag.hybrid_retriever import HybridCareerRetriever
from src.rag.vector_store import CareerVectorStore
from src.services.onet_service import ONetService
from src.services.usajobs_service import USAJobsService


def test_complete_mvp_pipeline(tmp_path: Path):
    # 1. Ingest sample career document into Personal RAG
    sample_resume = """
    ALEX MERCER
    Washington, DC | alex.mercer@example.com

    WORK EXPERIENCE
    Senior Machine Learning Engineer | Federal Analytics Corp | 2022 - Present
    - Developed BERT transformer models using PyTorch and Hugging Face on AWS.
    - Designed scalable data ingestion pipelines with PostgreSQL and Docker.

    EDUCATION
    Master of Science in Computer Science | George Washington University | 2020 - 2022
    - Focus on NLP and Machine Learning.

    TECHNICAL SKILLS
    - Python, PyTorch, SQL, PostgreSQL, AWS, Docker, BERT, Machine Learning, NLP
    """

    chunks = CareerChunker.chunk_document(sample_resume, "resume.txt")
    assert len(chunks) >= 3

    vector_store = CareerVectorStore(persist_dir=tmp_path / "vec_e2e")
    vector_store.add_chunks(chunks)
    assert vector_store.count() >= 3

    retriever = HybridCareerRetriever(vector_store)

    # 2. Candidate Profile Setup
    profile = CandidateProfile(
        name="Alex Mercer",
        target_roles=["Machine Learning Engineer", "Data Scientist"],
        career_interests=["Machine Learning", "NLP"],
        education_level="Master's Degree",
        years_of_experience=3.0,
        salary_expectation=115000.0,
        job_types=["Full-Time"],
        clearance="Secret Eligible",
    )

    # 3. Retrieve jobs from USAJOBS service
    usajobs_service = USAJobsService()
    jobs = usajobs_service.search_jobs(keyword="Machine Learning")
    assert len(jobs) > 0

    # 4. Enrich with O*NET and Rank
    onet_service = ONetService()
    ranker = JobRanker(retriever, onet_service)
    ranked_evaluations = ranker.evaluate_and_rank(jobs, profile, enrich_with_onet=True)

    assert len(ranked_evaluations) == len(jobs)

    # Verify descending sort order
    for i in range(len(ranked_evaluations) - 1):
        if not ranked_evaluations[i].breakdown.ineligible_flag and not ranked_evaluations[i+1].breakdown.ineligible_flag:
            assert ranked_evaluations[i].breakdown.overall_score >= ranked_evaluations[i+1].breakdown.overall_score

    # 5. Inspect top recommendation for evidence grounding
    top_eval = ranked_evaluations[0]
    breakdown = top_eval.breakdown

    # MATHEMATICAL PROPERTY: components sum strictly to total
    component_sum = round(
        breakdown.skills_score
        + breakdown.experience_score
        + breakdown.education_score
        + breakdown.career_interest_score
        + breakdown.salary_score
        + breakdown.job_type_score,
        2,
    )
    assert abs(breakdown.overall_score - component_sum) < 0.01

    # EVIDENCE CITATIONS: verified against career chunks
    assert len(top_eval.matched_qualifications) > 0
    first_match = top_eval.matched_qualifications[0]
    assert first_match.supports_match is True
    assert first_match.provenance == "CANDIDATE EVIDENCE"
    assert first_match.source_document == "resume.txt"
    assert first_match.chunk_id.startswith("resume.txt_")

    # STRICT SEPARATION: O*NET info separated from employer requirements
    assert top_eval.job.onet_code is not None
    assert len(top_eval.job.onet_occupational_skills) > 0
    # Employer requirements only contain employer stated requirements
    assert "Mathematics & Statistics" not in top_eval.job.required_skills

    # 6. Human Feedback Recording
    feedback_file = tmp_path / "e2e_feedback.json"
    feedback_store = FeedbackStore(file_path=feedback_file)
    fb = feedback_store.record_feedback(
        job_id=top_eval.job.job_id,
        job_title=top_eval.job.title,
        employer=top_eval.job.employer,
        status="Interested",
        notes="High score and strong alignment with PyTorch and NLP experience.",
        overall_score=breakdown.overall_score,
    )
    assert fb.status == "Interested"
    assert feedback_store.get_feedback(top_eval.job.job_id) is not None
