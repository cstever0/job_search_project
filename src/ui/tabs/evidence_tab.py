"""Personal RAG Evidence Explorer tab."""

from __future__ import annotations

import streamlit as st
from src.rag.hybrid_retriever import HybridCareerRetriever


def render_evidence_tab(retriever: HybridCareerRetriever) -> None:
    """Render the interactive Personal RAG knowledge base explorer."""
    st.header("🔎 Personal RAG Knowledge Base Explorer")
    st.markdown(
        "Directly test how your personal career documents are retrieved using **Hybrid Retrieval** "
        "(BM25 Keyword Matching + Dense Vector Similarity via Reciprocal Rank Fusion)."
    )

    total_chunks = retriever.vector_store.count()
    if total_chunks == 0:
        st.warning("⚠️ No documents indexed yet. Load or upload your resume in the **'Profile'** tab first.")
        return

    st.caption(f"Searching across **{total_chunks}** semantically indexed career units.")

    query = st.text_input(
        "Enter a qualification, technical skill, or experience query to test retrieval:",
        value="Transformer BERT text classification and PyTorch pipelines",
        placeholder="e.g. SQL data pipelines, AWS cloud deployments, Computer Science degree",
    )

    col_k, col_thresh = st.columns(2)
    with col_k:
        top_k = st.slider("Top Chunks to Retrieve", min_value=1, max_value=10, value=3)
    with col_thresh:
        thresh = st.slider("Evidence Support Threshold", min_value=0.1, max_value=0.8, value=0.35, step=0.05)

    if query.strip():
        evidence = retriever.retrieve_evidence_for_qualification(query, threshold=thresh)
        st.markdown("### 🏆 Top Retrieved Evidence")
        if evidence.supports_match:
            st.success(f"✅ **Match Supported!** (Confidence: {evidence.similarity_score:.2f})")
        else:
            st.error(f"❌ **{evidence.provenance}**")

        st.markdown(f"**Document:** `{evidence.source_document}` | **Section:** `{evidence.section}` | **Chunk ID:** `{evidence.chunk_id}`")
        st.info(f"\"{evidence.evidence_text}\"")

        st.markdown("---")
        st.markdown("### 🔬 Detailed Hybrid Search Candidates (BM25 + Dense Embeddings)")
        results = retriever.search(query, n_results=top_k)

        for rank, (chunk, score) in enumerate(results, start=1):
            with st.expander(f"Rank #{rank} - Chunk `{chunk.chunk_id}` (Hybrid Score: {score:.3f})"):
                st.markdown(f"**Source Document:** `{chunk.source_document}`")
                st.markdown(f"**Section:** `{chunk.section}`")
                if chunk.job_title and chunk.organization:
                    st.markdown(f"**Role:** {chunk.job_title} @ {chunk.organization}")
                elif chunk.project_name:
                    st.markdown(f"**Project:** {chunk.project_name}")
                if chunk.skills:
                    st.markdown(f"**Detected Skills:** {', '.join(chunk.skills)}")
                st.text(chunk.content)
