"""Demonstration script searching and evaluating jobs for Cory Stever."""

import json
from pathlib import Path
from config import PROFILE_FILE, VECTOR_DB_DIR
from src.matching.ranker import JobRanker
from src.models.profile import CandidateProfile
from src.rag.hybrid_retriever import HybridCareerRetriever
from src.rag.vector_store import CareerVectorStore
from src.services.onet_service import ONetService
from src.services.usajobs_service import USAJobsService


def run_cory_job_search():
    print("=== Running Job Search & Matching for Cory Stever ===")

    # 1. Load Profile
    with open(PROFILE_FILE, "r", encoding="utf-8") as f:
        profile = CandidateProfile(**json.load(f))
    print(f"Candidate: {profile.name}")
    print(f"Target Roles: {profile.target_roles}")
    print(f"Salary Expectation: ${profile.salary_expectation:,.0f}")
    print(f"Education: {profile.education_level}\n")

    # 2. Initialize Components
    vector_store = CareerVectorStore(VECTOR_DB_DIR)
    retriever = HybridCareerRetriever(vector_store)
    usajobs_service = USAJobsService()
    onet_service = ONetService()
    ranker = JobRanker(retriever, onet_service)

    # 3. Retrieve and Rank Jobs
    jobs = usajobs_service.search_jobs(keyword="AI Data Scientist Machine Learning")
    print(f"Retrieved {len(jobs)} jobs from USAJOBS service.")

    evaluations = ranker.evaluate_and_rank(jobs, profile, enrich_with_onet=True)

    print("\n" + "=" * 80)
    print("🏆 RANKED JOB RECOMMENDATIONS FOR CORY STEVER")
    print("=" * 80)

    for rank, ev in enumerate(evaluations, start=1):
        job = ev.job
        b = ev.breakdown
        ineligible_str = f" [⚠️ INELIGIBLE: {b.ineligible_reason}]" if b.ineligible_flag else ""
        print(f"\nRank #{rank}: {job.title} - {b.overall_score:.1f}/100{ineligible_str}")
        print(f"   Employer: {job.employer} | Location: {job.location}")
        print(f"   Salary: {job.salary_display} | Type: {job.job_type}")
        print(
            f"   Scores: Skills: {b.skills_score:.1f}/35 | Exp: {b.experience_score:.1f}/25 | "
            f"Edu: {b.education_score:.1f}/15 | Int: {b.career_interest_score:.1f}/15 | "
            f"Sal: {b.salary_score:.1f}/5 | JobType: {b.job_type_score:.1f}/5"
        )
        print(f"   Mathematical Identity Check: {b.verify_sum()} (Sum = {b.overall_score:.1f})")

        print(f"   Matched Qualifications ({len(ev.matched_qualifications)}):")
        for m in ev.matched_qualifications[:3]:
            print(f"     ✅ {m.qualification} -> {m.source_document} [{m.chunk_id}] (Score: {m.similarity_score:.2f})")

        if ev.missing_required_qualifications:
            print(f"   Missing Requirements ({len(ev.missing_required_qualifications)}):")
            for req in ev.missing_required_qualifications[:2]:
                print(f"     ❌ {req}")

        if ev.skills_to_improve_competitiveness:
            print(f"   O*NET Competitiveness Suggestions:")
            print(f"     📈 {', '.join(ev.skills_to_improve_competitiveness[:4])}")

        print("-" * 80)


if __name__ == "__main__":
    run_cory_job_search()
