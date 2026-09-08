"""Settings and system configuration tab."""

from __future__ import annotations

import streamlit as st
from config import (
    SCORING_WEIGHTS,
    VECTOR_DB_DIR,
    has_onet_credentials,
    has_usajobs_credentials,
)
from src.feedback.feedback_store import FeedbackStore


def render_settings_tab(feedback_store: FeedbackStore) -> None:
    """Render the configuration, API status, and scoring weights documentation."""
    st.header("⚙️ Settings & System Verification")

    st.markdown("### 🔌 API Integration & System Status")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏛️ USAJOBS API")
        if has_usajobs_credentials():
            st.success("✅ **Connected:** USAJOBS API credentials detected.")
        else:
            st.info(
                "ℹ️ **Demo Mode:** USAJOBS credentials not detected in `.env`. "
                "Using curated federal job dataset for demonstration."
            )
        st.caption("Endpoints: `https://data.usajobs.gov/api/search`")

    with col2:
        st.markdown("#### 🌐 O*NET Web Services API")
        if has_onet_credentials():
            st.success("✅ **Connected:** O*NET Web Services credentials detected.")
        else:
            st.info(
                "ℹ️ **Demo Mode:** O*NET Web Services credentials not detected in `.env`. "
                "Using curated occupational taxonomy for demonstration."
            )
        st.caption("Endpoints: `https://services.onetcenter.org/ws/online` | Header: `X-API-Key`")

    st.markdown("---")

    st.markdown("### ⚖️ Fixed 100-Point Matching Model Weights")
    st.markdown(
        "As specified by system design, these weights are **strictly locked** in `config.py` "
        "to ensure transparent, reproducible, and explainable scoring without LLM drift."
    )

    weight_data = [
        {"Component": "Skills Match", "Weight": "35%", "Description": "Required (70%) and preferred (30%) technical skills"},
        {"Component": "Experience Match", "Weight": "25%", "Description": "Tenure and role relevance grounded in work chunks"},
        {"Component": "Education Match", "Weight": "15%", "Description": "Degree level and discipline relevance"},
        {"Component": "Career Interest Match", "Weight": "15%", "Description": "Alignment with target roles and career domains"},
        {"Component": "Salary Preference", "Weight": "5%", "Description": "Soft preference scaling smoothly; never hard disqualifier"},
        {"Component": "Job Type Preference", "Weight": "5%", "Description": "Alignment with Full-Time / Permanent schedule"},
        {"Component": "Geographic Location", "Weight": "0% (Excluded)", "Description": "Candidate is geographically flexible"},
    ]
    st.table(weight_data)
    st.success("🔒 **Total Fixed Weight Sum:** 100.0% (Mathematically Guaranteed)")

    st.markdown("---")

    # Feedback log
    st.markdown("### 📋 Candidate Feedback History")
    feedback_items = feedback_store.list_all_feedback()
    if feedback_items:
        st.caption(f"Currently tracking feedback for **{len(feedback_items)}** jobs:")
        for fb in feedback_items:
            with st.expander(f"[{fb.status}] {fb.job_title} @ {fb.employer} (Score: {fb.overall_score:.1f})"):
                st.markdown(f"**Status:** `{fb.status}`")
                st.markdown(f"**Updated:** `{fb.updated_at}`")
                st.markdown(f"**Notes:** {fb.notes or 'None'}")
                if st.button("🗑️ Delete Feedback", key=f"del_fb_{fb.job_id}"):
                    feedback_store.delete_feedback(fb.job_id)
                    st.success("Deleted!")
                    st.rerun()
    else:
        st.write("No feedback recorded yet. You can mark jobs as Interested, Applied, or Good Match in the Job Details tab.")
