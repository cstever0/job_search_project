"""Reusable Streamlit visual components, badges, and score display cards."""

from __future__ import annotations

import streamlit as st
from src.models.match import ScoreBreakdown


def render_provenance_badge(provenance_type: str) -> str:
    """Return colored HTML badge indicating the provenance of information."""
    provenance_upper = provenance_type.upper()
    if "EMPLOYER" in provenance_upper:
        return (
            '<span style="background-color: #1E3A8A; color: #DBEAFE; padding: 3px 8px; '
            'border-radius: 4px; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.5px;">'
            '🏢 EMPLOYER REQUIREMENT</span>'
        )
    elif "O*NET" in provenance_upper:
        return (
            '<span style="background-color: #064E3B; color: #D1FAE5; padding: 3px 8px; '
            'border-radius: 4px; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.5px;">'
            '🌐 O*NET OCCUPATIONAL INFORMATION</span>'
        )
    elif "CANDIDATE" in provenance_upper:
        return (
            '<span style="background-color: #581C87; color: #F3E8FF; padding: 3px 8px; '
            'border-radius: 4px; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.5px;">'
            '📄 CANDIDATE EVIDENCE</span>'
        )
    else:
        return (
            '<span style="background-color: #7F1D1D; color: #FEE2E2; padding: 3px 8px; '
            'border-radius: 4px; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.5px;">'
            '❌ No evidence found.</span>'
        )


def render_feedback_badge(status: str) -> str:
    """Return colored HTML pill badge for user feedback status."""
    color_map = {
        "Interested": ("#2563EB", "#EFF6FF"),
        "Good Match": ("#059669", "#ECFDF5"),
        "Applied": ("#7C3AED", "#F5F3FF"),
        "Poor Match": ("#D97706", "#FFFBEB"),
        "Not Interested": ("#DC2626", "#FEF2F2"),
    }
    bg, fg = color_map.get(status, ("#4B5563", "#F3F4F6"))
    return (
        f'<span style="background-color: {fg}; color: {bg}; border: 1px solid {bg}; '
        f'padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">'
        f'● {status}</span>'
    )


def render_score_breakdown_card(breakdown: ScoreBreakdown) -> None:
    """Render the fixed 6-component score card ensuring mathematical transparency."""
    st.markdown("### 📊 100-Point Fixed Matching Breakdown")

    col_score, col_math = st.columns([1, 2])
    with col_score:
        st.metric(
            label="Overall Match Score",
            value=f"{breakdown.overall_score:.1f} / 100",
            delta=None,
        )
        # Visual progress bar
        progress = min(1.0, max(0.0, breakdown.overall_score / 100.0))
        st.progress(progress)

    with col_math:
        st.caption("Mathematical Identity: 6 Fixed Component Weights (Sum = 100.0)")
        st.markdown(
            f"""
            - **Skills:** `{breakdown.skills_score:.1f} / 35` (35%)
            - **Experience:** `{breakdown.experience_score:.1f} / 25` (25%)
            - **Education:** `{breakdown.education_score:.1f} / 15` (15%)
            - **Career Interest:** `{breakdown.career_interest_score:.1f} / 15` (15%)
            - **Salary:** `{breakdown.salary_score:.1f} / 5` (5% soft preference)
            - **Job Type:** `{breakdown.job_type_score:.1f} / 5` (5%)
            """
        )

    if breakdown.ineligible_flag:
        st.error(f"⚠️ **Ineligibility Warning:** {breakdown.ineligible_reason}")
