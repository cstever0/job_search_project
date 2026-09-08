"""Profile & Career Data management tab."""

from __future__ import annotations

import streamlit as st
from config import PROFILE_FILE, UPLOADS_DIR
from src.data_ingestion.document_parser import DocumentParser
from src.data_ingestion.semantic_chunker import CareerChunker
from src.models.profile import CandidateProfile
from src.rag.hybrid_retriever import HybridCareerRetriever


def render_profile_tab(
    profile: CandidateProfile, retriever: HybridCareerRetriever
) -> CandidateProfile:
    """Render the Profile and Career Data view."""
    st.header("👤 Profile & Personal Career Data")
    st.markdown(
        "Upload your resume, CV, and career documents (**PDF, DOCX, TXT**). "
        "The system extracts and semantically chunks your experience into meaningful units "
        "to power your **Personal RAG Knowledge Base**."
    )

    # Quick demo loader
    sample_file = st.session_state.get("sample_file_path")
    col_demo, _ = st.columns([2, 3])
    with col_demo:
        if st.button("🚀 Load Sample Profile & Resume (Alex Mercer)", use_container_width=True):
            if sample_file and sample_file.exists():
                text = sample_file.read_text(encoding="utf-8")
                chunks = CareerChunker.chunk_document(text, "sample_resume.txt")
                retriever.vector_store.clear()
                retriever.vector_store.add_chunks(chunks)
                retriever.refresh_bm25_index()

                profile.name = "Alex Mercer"
                profile.target_roles = ["Data Scientist", "Machine Learning Engineer", "AI Specialist"]
                profile.career_interests = ["NLP", "Transformers", "RAG Systems", "Machine Learning"]
                profile.education_level = "Master's Degree"
                profile.years_of_experience = 3.5
                profile.salary_expectation = 110000.0
                profile.job_types = ["Full-Time"]
                profile.clearance = "Secret Eligible (U.S. Citizen)"
                profile.technical_skills = ["Python", "PyTorch", "SQL", "Docker", "AWS", "BERT"]
                st.success("✅ Sample career profile and 7 career chunks loaded into Personal RAG!")
                st.rerun()

    st.markdown("---")

    # Document Uploader
    st.subheader("📄 Upload Career Documents")
    uploaded_files = st.file_uploader(
        "Upload CV / Resume / Academic Projects / Transcripts / Portfolio",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("📥 Ingest & Rebuild Personal RAG Index", type="primary"):
            new_chunks_count = 0
            for file in uploaded_files:
                save_path = UPLOADS_DIR / file.name
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())

                text = DocumentParser.parse_document(file.name, file.getvalue())
                chunks = CareerChunker.chunk_document(text, file.name)
                retriever.vector_store.add_chunks(chunks)
                new_chunks_count += len(chunks)

            retriever.refresh_bm25_index()
            st.success(f"Successfully processed {len(uploaded_files)} file(s) into {new_chunks_count} semantic career chunks!")
            st.rerun()

    # Ingestion Stats
    total_chunks = retriever.vector_store.count()
    st.info(f"📚 **Personal RAG Status:** Currently indexing **{total_chunks}** semantic career chunks.")

    if total_chunks > 0:
        with st.expander("🔍 Inspect Indexed Career Chunks & Metadata", expanded=False):
            for c in retriever.vector_store.chunks:
                st.markdown(f"**Chunk ID:** `{c.chunk_id}` | **Section:** `{c.section}` | **Doc:** `{c.source_document}`")
                if c.job_title and c.organization:
                    st.caption(f"Role: {c.job_title} @ {c.organization} ({c.date or 'N/A'})")
                elif c.education:
                    st.caption(f"Education: {c.education} ({c.date or 'N/A'})")
                if c.skills:
                    st.caption(f"Detected Skills: {', '.join(c.skills)}")
                st.text(c.content)
                st.divider()

    st.markdown("---")

    # Candidate Profile & Preferences Form
    st.subheader("🎯 Career Preferences & Parameters")
    st.caption("These parameters ground the fixed 6-component scoring model (Skills 35%, Experience 25%, Education 15%, Career Interest 15%, Salary 5%, Job Type 5%).")

    with st.form("profile_preferences_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Candidate Name", value=profile.name)
            target_roles_str = st.text_input(
                "Target Job Roles (comma-separated)",
                value=", ".join(profile.target_roles),
                help="Used for the 15% Career Interest score component.",
            )
            career_interests_str = st.text_input(
                "Career Interests / Focus Domains (comma-separated)",
                value=", ".join(profile.career_interests),
            )
            career_goals = st.text_area("Career Goals", value=profile.career_goals, height=80)

        with col2:
            edu_level = st.selectbox(
                "Highest Education Level",
                options=["Bachelor's Degree", "Master's Degree", "Ph.D. / Doctorate", "Associate Degree", "Other"],
                index=1 if "master" in profile.education_level.lower() else 0,
            )
            yoe = st.number_input(
                "Years of Technical Experience",
                min_value=0.0,
                max_value=40.0,
                value=float(profile.years_of_experience),
                step=0.5,
            )
            salary = st.number_input(
                "Target Annual Salary ($ USD)",
                min_value=0.0,
                max_value=500000.0,
                value=float(profile.salary_expectation or 100000.0),
                step=5000.0,
                help="Treated as a soft preference (5% weight). Lower salaries scale smoothly and never automatically reject.",
            )
            job_types_str = st.text_input(
                "Preferred Job Types (comma-separated)",
                value=", ".join(profile.job_types),
                help="E.g., Full-Time, Permanent (5% weight).",
            )
            clearance = st.text_input(
                "Security Clearance Held (if any)",
                value=profile.clearance or "None",
            )

        st.markdown(
            "📍 *Note on Location: Because you are geographically flexible, location is excluded from the numerical score (0% weight).*"
        )

        submitted = st.form_submit_button("💾 Save Profile Preferences", type="primary")
        if submitted:
            profile.name = name
            profile.target_roles = [r.strip() for r in target_roles_str.split(",") if r.strip()]
            profile.career_interests = [i.strip() for i in career_interests_str.split(",") if i.strip()]
            profile.career_goals = career_goals
            profile.education_level = edu_level
            profile.years_of_experience = yoe
            profile.salary_expectation = salary
            profile.job_types = [t.strip() for t in job_types_str.split(",") if t.strip()]
            profile.clearance = clearance if clearance.strip() and clearance.lower() != "none" else None
            st.success("✅ Profile preferences successfully saved!")

    return profile
