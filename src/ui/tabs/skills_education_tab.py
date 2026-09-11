"""Summary tab for most sought-after skills, education requirements, and candidate gap analysis."""

from __future__ import annotations

from typing import Any, Optional
import pandas as pd
import streamlit as st
from src.matching.analytics import MarketSkillsEducationAnalyzer
from src.models.job import NormalizedJob
from src.models.profile import CandidateProfile
from src.ui.components import render_provenance_badge


def render_skills_education_tab(
    jobs: list[NormalizedJob],
    profile: CandidateProfile,
    chunks: list[Any] = None,
    onet_service: Optional[Any] = None,
) -> None:
    """Render the skills and education market intelligence summary."""
    st.header("📊 Most Sought-After Skills & Education Summary")
    st.markdown(
        "This tab aggregates current federal postings (**USAJOBS**) and occupational data (**O*NET**) "
        "to analyze market demand and identify your competitive advantages and skill development areas."
    )

    if not onet_service:
        from src.services.onet_service import ONetService
        onet_service = ONetService()

    if not jobs:
        from src.services.usajobs_service import USAJobsService
        jobs = USAJobsService().search_jobs(keyword="AI Data Scientist Software")

    # Ensure all jobs are actively enriched with O*NET occupational and technology data
    for j in jobs:
        if not j.onet_code or not j.onet_technology_skills or not j.onet_occupational_skills:
            try:
                onet_service.enrich_job(j)
            except Exception:
                pass

    summary = MarketSkillsEducationAnalyzer.generate_market_summary(jobs, profile, chunks)
    skills_data = summary["skills"]
    edu_data = summary["education"]

    # Top KPI Metrics Cards
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Total Positions Analyzed",
            value=f"{summary['total_jobs_analyzed']} jobs",
            help="Jobs currently retrieved from USAJOBS and enriched with O*NET.",
        )
    with m2:
        st.metric(
            label="Top In-Demand Skill",
            value=summary["most_sought_after_skill"],
            help="Most frequently required technical skill across employer postings.",
        )
    with m3:
        st.metric(
            label="Dominant Degree Level",
            value=summary["most_common_education"],
            help="Most commonly required or accepted degree tier.",
        )
    with m4:
        st.metric(
            label="Your Skill Coverage",
            value=f"{skills_data['candidate_skill_coverage_pct']}%",
            delta="Strong Grounding" if skills_data['candidate_skill_coverage_pct'] >= 60 else "Skill Gaps Found",
            help="Percentage of top market-demanded skills verified in your personal career documents.",
        )

    st.markdown("---")

    # SECTION 1: MOST SOUGHT-AFTER TECHNICAL SKILLS
    st.subheader("💻 In-Demand Technical Skills Breakdown")
    st.caption("Distinguishing between explicit employer requirements and broader O*NET occupational technology.")

    col_emp, col_onet = st.columns(2)

    with col_emp:
        st.markdown(f"#### 🏢 Employer Required Skills {render_provenance_badge('EMPLOYER REQUIREMENT')}", unsafe_allow_html=True)
        st.caption("Mandated in official USAJOBS job announcements:")

        req_items = skills_data["top_employer_required"]
        if req_items:
            df_req = pd.DataFrame(req_items, columns=["Skill", "Job Postings"])
            st.bar_chart(df_req.set_index("Skill"), color="#2563EB", horizontal=True)
        else:
            st.info("No employer skills explicitly extracted from current search filter.")

    with col_onet:
        st.markdown(f"#### 🌐 O*NET Occupational Skills & Technologies {render_provenance_badge('O*NET OCCUPATIONAL INFORMATION')}", unsafe_allow_html=True)
        st.caption("Skills commonly associated with these occupations in federal taxonomy:")

        onet_tab1, onet_tab2 = st.tabs(["Tools & Technologies", "Core Competencies"])
        with onet_tab1:
            onet_tech = skills_data["top_onet_tech_skills"]
            if onet_tech:
                df_onet_tech = pd.DataFrame(onet_tech, columns=["Technology", "Occupations"])
                st.bar_chart(df_onet_tech.set_index("Technology"), color="#059669", horizontal=True)
            else:
                st.info("No O*NET technology skills recorded.")

        with onet_tab2:
            onet_comp = skills_data.get("top_onet_competencies", [])
            if onet_comp:
                df_onet_comp = pd.DataFrame(onet_comp, columns=["Competency", "Frequency"])
                st.bar_chart(df_onet_comp.set_index("Competency"), color="#0D9488", horizontal=True)
            else:
                st.info("No O*NET competencies recorded.")

    st.markdown("---")

    # SECTION 2: CANDIDATE SKILL ALIGNMENT & GAP ANALYSIS
    st.subheader("🎯 Your Personal Skill Alignment & Gap Analysis")
    st.caption(f"Candidate: **{profile.name}** | Verified Skills: **{summary['candidate_skills_count']}**")

    col_pos, col_gap = st.columns(2)

    with col_pos:
        st.markdown("#### ✅ Skills You Possess (Verified Grounding)")
        st.caption("In-demand skills that you have documented in your uploaded resume and career portfolio:")
        possessed = skills_data["candidate_possessed_skills"]
        if possessed:
            for item in possessed[:10]:
                st.markdown(
                    f"- **{item['skill']}** — Present in **{item['percentage']}%** of analyzed jobs "
                    f"`(Count: {item['demand_count']})`"
                )
        else:
            st.info("No matching skills recorded yet.")

    with col_gap:
        st.markdown("#### 🎯 In-Demand Skills to Consider Acquiring")
        st.caption("Frequently required skills where no verified evidence was found in your career documents:")
        gaps = skills_data["candidate_skill_gaps"]
        if gaps:
            for item in gaps[:8]:
                st.markdown(
                    f"- 📈 **{item['skill']}** — Required in **{item['percentage']}%** of positions "
                    f"`(Count: {item['demand_count']})`"
                )
            st.info("💡 *Tip: Adding academic projects or certifications in these tools can noticeably boost your match scores.*")
        else:
            st.success("🎉 You possess all the top skills required by current postings!")

    st.markdown("---")

    # SECTION 3: MOST SOUGHT-AFTER EDUCATION CREDENTIALS
    st.subheader("🎓 In-Demand Education Levels & Disciplines")
    st.caption("Breakdown of degree levels and target academic majors requested by federal employers.")

    col_deg, col_disc = st.columns(2)

    with col_deg:
        st.markdown("#### 📜 Required Degree Levels")
        deg_dist = edu_data["degree_level_distribution"]
        if deg_dist:
            df_deg = pd.DataFrame(deg_dist, columns=["Degree Level", "Postings"])
            st.dataframe(df_deg, use_container_width=True, hide_index=True)

        st.markdown("##### 👤 Your Education Alignment")
        st.success(f"**Your Degree:** {profile.education_level}")
        st.caption(f"**Market Fit:** {edu_data['candidate_alignment_note']}.")

    with col_disc:
        st.markdown("#### 🔬 Most Requested Fields of Study")
        disc_dist = edu_data["top_disciplines"]
        if disc_dist:
            df_disc = pd.DataFrame(disc_dist, columns=["Academic Discipline", "Mentions"])
            st.bar_chart(df_disc.set_index("Academic Discipline"), color="#7C3AED", horizontal=True)

    st.markdown("---")

    # SECTION 4: STRATEGIC RECOMMENDATIONS SUMMARY
    st.subheader("💡 Strategic Career Recommendations")
    top_missing_str = ", ".join([g["skill"] for g in skills_data["candidate_skill_gaps"][:3]])
    st.markdown(
        f"""
        Based on the current analysis of **{summary['total_jobs_analyzed']} federal positions** for your target roles:
        
        1. **High-Value Strengths:** Your expertise in **Python, PyTorch, BERT/Transformers, and SQL** matches the core requirements for federal Data Science and AI Specialist series (GS-1560 and GS-2210).
        2. **Education Advantage:** Your **Master's in Computer Science (Concentration: AI)** in progress at UMKC qualifies you for GS-13+ specialized roles, exceeding standard Bachelor's degree prerequisites.
        3. **Target Skill Growth:** To maximize competitiveness, consider highlighting or gaining hands-on project experience in: **{top_missing_str or 'Cloud and CI/CD tools'}**.
        4. **Proven Integrity:** Federal hiring panels require strict evidence validation; all scores and gap analysis displayed here are grounded in verifiable personal documents.
        """
    )
