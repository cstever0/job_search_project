"""Main Streamlit application entrypoint for Personal AI Job Search and Matching Application."""

from __future__ import annotations

import json
from pathlib import Path
import streamlit as st

from config import (
    DATA_DIR,
    FEEDBACK_FILE,
    PROFILE_FILE,
    PROJECT_ROOT,
    UPLOADS_DIR,
    VECTOR_DB_DIR,
)
from src.data_ingestion.semantic_chunker import CareerChunker
from src.feedback.feedback_store import FeedbackStore
from src.matching.ranker import JobRanker
from src.models.profile import CandidateProfile
from src.rag.hybrid_retriever import HybridCareerRetriever
from src.rag.vector_store import CareerVectorStore
from src.services.onet_service import ONetService
from src.services.usajobs_service import USAJobsService
from src.ui.tabs.details_tab import render_details_tab
from src.ui.tabs.evidence_tab import render_evidence_tab
from src.ui.tabs.profile_tab import render_profile_tab
from src.ui.tabs.results_tab import render_results_tab
from src.ui.tabs.search_tab import render_search_tab
from src.ui.tabs.settings_tab import render_settings_tab
from src.ui.tabs.skills_education_tab import render_skills_education_tab

st.set_page_config(
    page_title="Personal AI Job Search & Matching",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_saved_profile() -> CandidateProfile:
    """Load saved candidate profile from disk if available."""
    if PROFILE_FILE.exists():
        try:
            with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CandidateProfile(**data)
        except Exception:
            pass
    return CandidateProfile()


def save_profile(profile: CandidateProfile) -> None:
    """Persist candidate profile to disk."""
    try:
        with open(PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(profile.model_dump(), f, indent=2)
    except Exception:
        pass


def initialize_app_state() -> None:
    """Initialize persistent services and state in Streamlit session_state."""
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = CareerVectorStore(VECTOR_DB_DIR)

    if "retriever" not in st.session_state:
        st.session_state["retriever"] = HybridCareerRetriever(st.session_state["vector_store"])

    if "usajobs_service" not in st.session_state:
        st.session_state["usajobs_service"] = USAJobsService()

    if "onet_service" not in st.session_state:
        st.session_state["onet_service"] = ONetService()

    if "job_ranker" not in st.session_state:
        st.session_state["job_ranker"] = JobRanker(
            st.session_state["retriever"],
            st.session_state["onet_service"],
        )

    if "feedback_store" not in st.session_state:
        st.session_state["feedback_store"] = FeedbackStore(FEEDBACK_FILE)

    if "profile" not in st.session_state:
        st.session_state["profile"] = load_saved_profile()

    if "evaluations" not in st.session_state:
        st.session_state["evaluations"] = []

    if "sample_file_path" not in st.session_state:
        st.session_state["sample_file_path"] = PROJECT_ROOT / "data" / "sample_data" / "sample_resume.txt"

    # Pre-populate sample resume if vector store is empty for smooth first-run demonstration
    if st.session_state["vector_store"].count() == 0:
        sample_path = st.session_state["sample_file_path"]
        if sample_path.exists():
            sample_text = sample_path.read_text(encoding="utf-8")
            chunks = CareerChunker.chunk_document(sample_text, "sample_resume.txt")
            st.session_state["vector_store"].add_chunks(chunks)
            st.session_state["retriever"].refresh_bm25_index()


def main() -> None:
    """Main application layout and tab router."""
    initialize_app_state()

    profile: CandidateProfile = st.session_state["profile"]
    retriever: HybridCareerRetriever = st.session_state["retriever"]
    usajobs_service: USAJobsService = st.session_state["usajobs_service"]
    job_ranker: JobRanker = st.session_state["job_ranker"]
    feedback_store: FeedbackStore = st.session_state["feedback_store"]
    evaluations = st.session_state.get("evaluations", [])

    # Sidebar
    with st.sidebar:
        st.title("🎯 Personal AI Job Matcher")
        st.caption("Evidence-Grounded Federal Job Search")

        st.markdown("---")
        st.markdown(f"**Candidate:** `{profile.name}`")
        st.markdown(f"**Roles:** `{', '.join(profile.target_roles[:2])}`")
        st.markdown(f"**Indexed Chunks:** `{retriever.vector_store.count()}`")
        st.markdown(f"**Evaluated Jobs:** `{len(evaluations)}`")

        st.markdown("---")
        st.markdown("### 🏛️ Core Pillars")
        st.markdown("1. **USAJOBS**: Federal opportunities")
        st.markdown("2. **O*NET**: Occupational enrichment")
        st.markdown("3. **Personal RAG**: BM25 + Vector grounding")

        st.markdown("---")
        st.markdown("### ⚖️ Fixed Weights (100%)")
        st.caption(
            "• Skills: 35%\n"
            "• Experience: 25%\n"
            "• Education: 15%\n"
            "• Career Interest: 15%\n"
            "• Salary: 5%\n"
            "• Job Type: 5%"
        )

        st.markdown("---")
        if st.button("🔄 Reload Sample Data", use_container_width=True):
            sample_path = st.session_state["sample_file_path"]
            if sample_path.exists():
                text = sample_path.read_text(encoding="utf-8")
                chunks = CareerChunker.chunk_document(text, "sample_resume.txt")
                retriever.vector_store.clear()
                retriever.vector_store.add_chunks(chunks)
                retriever.refresh_bm25_index()
                st.success("Sample data reset!")
                st.rerun()

    # Main Tabs
    tab_titles = [
        "👤 Profile & Career Data",
        "🔍 Job Search",
        "🏆 Recommended Jobs",
        "🔬 Job Details & Evidence",
        "📊 In-Demand Skills & Education",
        "🔎 Personal RAG Explorer",
        "⚙️ Settings & System Status",
    ]

    tabs = st.tabs(tab_titles)

    with tabs[0]:
        updated_profile = render_profile_tab(profile, retriever)
        if updated_profile != profile:
            st.session_state["profile"] = updated_profile
            save_profile(updated_profile)

    with tabs[1]:
        render_search_tab(usajobs_service, job_ranker, profile)

    with tabs[2]:
        render_results_tab(evaluations, feedback_store)

    with tabs[3]:
        render_details_tab(evaluations, feedback_store)

    with tabs[4]:
        jobs_to_analyze = (
            [e.job for e in evaluations]
            if evaluations
            else usajobs_service.search_jobs(keyword="AI Data Scientist Machine Learning")
        )
        render_skills_education_tab(jobs_to_analyze, profile, retriever.vector_store.chunks)

    with tabs[5]:
        render_evidence_tab(retriever)

    with tabs[6]:
        render_settings_tab(feedback_store)


if __name__ == "__main__":
    main()
