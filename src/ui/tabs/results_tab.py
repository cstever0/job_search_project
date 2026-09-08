"""Recommended Jobs tab showing ranked opportunities."""

from __future__ import annotations

import streamlit as st
from src.feedback.feedback_store import FeedbackStore
from src.models.match import JobMatchEvaluation
from src.ui.components import render_feedback_badge


def render_results_tab(
    evaluations: list[JobMatchEvaluation],
    feedback_store: FeedbackStore,
) -> None:
    """Render the ranked job recommendations list."""
    st.header("🏆 Recommended Jobs (Ranked by Evidence Match)")
    st.markdown(
        "Jobs are ranked from highest overall match score to lowest using the **fixed 100-point matching model**. "
        "Every score is calculated deterministically and grounded in your Personal RAG evidence."
    )

    if not evaluations:
        st.info("No jobs evaluated yet. Run a search in the **'Job Search'** tab to view ranked recommendations.")
        return

    st.caption(f"Displaying **{len(evaluations)}** evaluated positions:")

    for rank, eval_obj in enumerate(evaluations, start=1):
        job = eval_obj.job
        breakdown = eval_obj.breakdown

        existing_feedback = feedback_store.get_feedback(job.job_id)
        feedback_status = existing_feedback.status if existing_feedback else None

        with st.container():
            col_rank, col_title, col_score, col_btn = st.columns([0.8, 4.5, 2.2, 1.5])

            with col_rank:
                st.markdown(f"### #{rank}")

            with col_title:
                st.markdown(f"**{job.title}**")
                meta_line = f"🏢 {job.employer} | 📍 {job.location} | 💰 {job.salary_display} | ⏱️ {job.job_type}"
                st.caption(meta_line)
                if feedback_status:
                    st.markdown(render_feedback_badge(feedback_status), unsafe_allow_html=True)
                if breakdown.ineligible_flag:
                    st.markdown(
                        f'<span style="background-color: #FEE2E2; color: #991B1B; padding: 2px 6px; '
                        f'border-radius: 4px; font-size: 0.75rem; font-weight: 600;">'
                        f'⚠️ Ineligible: {breakdown.ineligible_reason}</span>',
                        unsafe_allow_html=True,
                    )

            with col_score:
                st.metric(label="Match Score", value=f"{breakdown.overall_score:.1f} / 100")
                st.caption(
                    f"Skills: {breakdown.skills_score:.0f}/35 | Exp: {breakdown.experience_score:.0f}/25 | "
                    f"Edu: {breakdown.education_score:.0f}/15 | Int: {breakdown.career_interest_score:.0f}/15"
                )

            with col_btn:
                if st.button("Inspect Match", key=f"inspect_{job.job_id}", use_container_width=True):
                    st.session_state["selected_job_id"] = job.job_id
                    st.session_state["active_tab_index"] = 3  # Switch to Job Details
                    st.rerun()

            st.divider()
