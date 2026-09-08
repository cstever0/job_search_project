"""Job Search tab querying USAJOBS and triggering matching."""

from __future__ import annotations

import streamlit as st
from src.matching.ranker import JobRanker
from src.models.profile import CandidateProfile
from src.services.usajobs_service import USAJobsService


def render_search_tab(
    usajobs_service: USAJobsService,
    job_ranker: JobRanker,
    profile: CandidateProfile,
) -> None:
    """Render the USAJOBS search interface."""
    st.header("🔍 Search USAJOBS Federal Opportunities")
    st.markdown(
        "Search current federal job opportunities from the official **USAJOBS API**. "
        "Retrieved positions are normalized and automatically enriched with **O*NET Web Services** "
        "occupational data."
    )

    if not usajobs_service.is_configured():
        st.info(
            "ℹ️ **API Mode:** Running with high-fidelity federal job datasets. "
            "To connect live to USAJOBS, provide your `USAJOBS_API_KEY` and `USAJOBS_USER_AGENT` in `.env`."
        )

    with st.form("usajobs_search_form"):
        col1, col2 = st.columns(2)
        with col1:
            keyword = st.text_input(
                "Job Title / Keyword",
                value="Machine Learning",
                placeholder="e.g., Data Scientist, AI Specialist, Software Engineer",
            )
            location = st.text_input(
                "Location Filter",
                value="",
                placeholder="e.g., Washington, DC or Maryland",
            )
            is_remote = st.checkbox("Remote Opportunities Only", value=False)

        with col2:
            job_category = st.selectbox(
                "Federal Job Series / Category",
                options=[
                    ("All Categories", ""),
                    ("GS-2210: Information Technology Management", "2210"),
                    ("GS-1560: Data Science", "1560"),
                    ("GS-1550: Computer Science", "1550"),
                    ("GS-0801: General Engineering", "0801"),
                ],
                format_func=lambda x: x[0],
            )[1]

            salary_min = st.number_input(
                "Minimum Desired Salary ($)",
                min_value=0.0,
                max_value=300000.0,
                value=float(profile.salary_expectation or 90000.0),
                step=5000.0,
            )

            results_count = st.slider("Results to Retrieve & Rank", min_value=3, max_value=25, value=10)

        submitted = st.form_submit_button("🚀 Retrieve, Enrich & Rank Jobs", type="primary", use_container_width=True)

    if submitted:
        if job_ranker.retriever.vector_store.count() == 0:
            st.warning("⚠️ Your Personal RAG knowledge base is empty! Please upload a resume or click 'Load Sample Profile' in the Profile tab first.")
            return

        with st.spinner("1/3 Querying USAJOBS API..."):
            jobs = usajobs_service.search_jobs(
                keyword=keyword,
                location=location,
                is_remote=is_remote,
                job_category=job_category,
                salary_min=salary_min,
                results_per_page=results_count,
            )

        if not jobs:
            st.warning("No matching job postings found. Try broadening your search keyword.")
            return

        with st.spinner(f"2/3 Enriching {len(jobs)} jobs with O*NET Web Services & 3/3 Evaluating Personal RAG Evidence..."):
            evaluations = job_ranker.evaluate_and_rank(jobs, profile, enrich_with_onet=True)

        st.session_state["evaluations"] = evaluations
        st.session_state["selected_job_id"] = evaluations[0].job.job_id if evaluations else None
        st.success(f"Successfully evaluated and ranked {len(evaluations)} jobs against your career profile!")
        st.info("👉 Switch to the **'Recommended Jobs'** tab to inspect the ranked results, or **'Job Details'** for deep-dive evidence citations.")
