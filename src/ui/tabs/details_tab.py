"""Job Details tab displaying deep-dive evidence citations and provenance."""

from __future__ import annotations

import streamlit as st
from src.feedback.feedback_store import FeedbackStore, VALID_FEEDBACK_STATUSES
from src.models.match import JobMatchEvaluation
from src.ui.components import (
    render_feedback_badge,
    render_provenance_badge,
    render_score_breakdown_card,
)


def render_details_tab(
    evaluations: list[JobMatchEvaluation],
    feedback_store: FeedbackStore,
) -> None:
    """Render the full evidence-grounded match analysis for a selected job."""
    st.header("🔬 Job Details & Evidence-Grounded Evaluation")

    if not evaluations:
        st.info("No evaluated jobs available. Search for jobs first in the **'Job Search'** tab.")
        return

    # Job selector
    job_map = {e.job.job_id: e for e in evaluations}
    selected_id = st.session_state.get("selected_job_id", evaluations[0].job.job_id)
    if selected_id not in job_map:
        selected_id = evaluations[0].job.job_id

    job_options = [
        (e.job.job_id, f"#{idx+1} {e.job.title} - {e.breakdown.overall_score:.1f}/100 ({e.job.employer})")
        for idx, e in enumerate(evaluations)
    ]

    selected_tuple = st.selectbox(
        "Select Job to Inspect:",
        options=job_options,
        index=[opt[0] for opt in job_options].index(selected_id),
        format_func=lambda x: x[1],
    )
    current_eval = job_map[selected_tuple[0]]
    job = current_eval.job
    breakdown = current_eval.breakdown

    # Main Job Header
    st.markdown("---")
    col_hdr, col_link = st.columns([3.5, 1.5])
    with col_hdr:
        st.subheader(f"{job.title}")
        st.markdown(
            f"**Employer:** {job.employer} | **Location:** {job.location} | **Type:** {job.job_type}"
        )
        st.markdown(f"**Salary:** {job.salary_display}")

    with col_link:
        if job.application_url:
            st.link_button("🌐 Apply on USAJOBS", job.application_url, use_container_width=True)
        st.caption(f"Source: **{job.source}**")

    # Score breakdown card
    render_score_breakdown_card(breakdown)

    # Narrative explanation
    st.markdown("### 📝 Ranking Explanation")
    st.markdown(current_eval.ranking_explanation)

    # Human Feedback section
    st.markdown("### 💬 Your Feedback & Tracking")
    existing_fb = feedback_store.get_feedback(job.job_id)

    with st.form(f"feedback_form_{job.job_id}"):
        col_st, col_nt = st.columns([1.5, 3])
        with col_st:
            current_status = existing_fb.status if existing_fb else "Interested"
            status = st.selectbox(
                "Status",
                options=list(VALID_FEEDBACK_STATUSES),
                index=list(VALID_FEEDBACK_STATUSES).index(current_status),
            )
        with col_nt:
            notes = st.text_input("Notes / Application Status", value=existing_fb.notes if existing_fb else "")

        if st.form_submit_button("💾 Save Feedback", type="primary"):
            saved = feedback_store.record_feedback(
                job_id=job.job_id,
                job_title=job.title,
                employer=job.employer,
                status=status,
                notes=notes,
                overall_score=breakdown.overall_score,
            )
            st.success(f"Feedback recorded: {saved.status}!")
            st.rerun()

    st.markdown("---")

    # Candidate Strengths
    if current_eval.candidate_strengths:
        st.markdown("### 🌟 Candidate Strengths")
        for strength in current_eval.candidate_strengths:
            st.markdown(f"- ✅ {strength}")

    # Matched Qualifications with Full Provenance
    st.markdown("### 🎯 Matched Employer Qualifications & Evidence Citations")
    st.caption("Each qualification is traceable back to verified excerpts in your personal career documents:")

    if current_eval.matched_qualifications:
        for ev in current_eval.matched_qualifications:
            with st.container():
                st.markdown(
                    f"**Requirement:** `{ev.qualification}` &nbsp;&nbsp; {render_provenance_badge(ev.provenance)}",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"📁 Source: **{ev.source_document}** | Section: **{ev.section.replace('_', ' ').title()}** | "
                    f"Chunk ID: `{ev.chunk_id}` | Confidence: `{ev.similarity_score:.2f}`"
                )
                st.markdown(f"> *\"{ev.evidence_text}\"*")
                st.divider()
    else:
        st.write("No strong direct qualification matches identified.")

    # Missing Employer Requirements
    col_req, col_pref = st.columns(2)
    with col_req:
        st.markdown("### ⚠️ Missing Job Requirements")
        st.markdown(render_provenance_badge("EMPLOYER REQUIREMENT"), unsafe_allow_html=True)
        st.caption("Stated in the employer's official job announcement with no supporting evidence in your career documents:")
        if current_eval.missing_required_qualifications:
            for q in current_eval.missing_required_qualifications:
                st.markdown(f"- ❌ **{q}**")
                st.caption("Status: `No evidence found in uploaded career documents.`")
        else:
            st.write("No mandatory requirements missing!")

    with col_pref:
        st.markdown("### 💡 Missing Preferred Qualifications")
        st.markdown(render_provenance_badge("EMPLOYER REQUIREMENT"), unsafe_allow_html=True)
        st.caption("Desired or preferred qualifications stated by the employer that were not found:")
        if current_eval.missing_preferred_qualifications:
            for p in current_eval.missing_preferred_qualifications:
                st.markdown(f"- ⚪ {p}")
        else:
            st.write("All preferred qualifications matched or none specified.")

    st.markdown("---")

    # O*NET Occupational Information (STRICTLY SEPARATED)
    st.markdown("### 🌐 O*NET Occupational Information & Career Development")
    st.markdown(render_provenance_badge("O*NET OCCUPATIONAL INFORMATION"), unsafe_allow_html=True)
    st.caption(
        "Notice: O*NET information reflects general federal occupational taxonomy from the Department of Labor. "
        "It is **strictly separated** from the employer's explicit job requirements."
    )

    onet = current_eval.onet_occupational_info
    if onet and onet.get("title"):
        st.markdown(f"**Standard O*NET Occupation:** {onet.get('title')} (SOC `{onet.get('code')}`)")
        st.write(onet.get("description", ""))

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("##### Typical Occupational Knowledge & Skills")
            for sk in onet.get("skills", [])[:4]:
                st.markdown(f"- 🧠 {sk}")
            for kn in onet.get("knowledge", [])[:3]:
                st.markdown(f"- 📖 {kn}")

        with col_t2:
            st.markdown("##### Common Technology Skills in this Occupation")
            tech_tags = " ".join([f"`{t}`" for t in onet.get("technology_skills", [])[:8]])
            st.markdown(tech_tags if tech_tags else "General computing tools.")

    # Skills That Could Improve Competitiveness (Separated from Missing Job Requirements!)
    if current_eval.skills_to_improve_competitiveness:
        st.markdown("#### 🚀 Skills That Could Improve Competitiveness")
        st.caption(
            "These are O*NET occupational skills commonly used in this profession that the employer did *not* "
            "strictly mandate, but which could strengthen your candidate profile:"
        )
        for s in current_eval.skills_to_improve_competitiveness:
            st.markdown(f"- 📈 **{s}** *(O*NET recommendation - NOT an employer disqualifier)*")
