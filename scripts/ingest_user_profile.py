"""Script to ingest Cory Stever's resume and career information into the Personal RAG knowledge base."""

from pathlib import Path
from config import PROFILE_FILE, UPLOADS_DIR, VECTOR_DB_DIR
from src.data_ingestion.document_parser import DocumentParser
from src.data_ingestion.semantic_chunker import CareerChunker
from src.models.profile import CandidateProfile
from src.rag.hybrid_retriever import HybridCareerRetriever
from src.rag.vector_store import CareerVectorStore


def ingest_cory_profile():
    print("=== Ingesting Cory Stever's Career Documents ===")

    # 1. Initialize Vector Store
    vector_store = CareerVectorStore(VECTOR_DB_DIR)
    vector_store.clear()

    # 2. Ingest Resume
    resume_path = UPLOADS_DIR / "cory_stever_resume.txt"
    resume_text = resume_path.read_text(encoding="utf-8")
    resume_chunks = CareerChunker.chunk_document(resume_text, "cory_stever_resume.txt")
    print(f"Extracted {len(resume_chunks)} semantic chunks from resume.")

    # 3. Ingest Career Information & Project Supplement
    info_path = UPLOADS_DIR / "cory_stever_career_info.txt"
    info_text = info_path.read_text(encoding="utf-8")
    info_chunks = CareerChunker.chunk_document(info_text, "cory_stever_career_info.txt")
    print(f"Extracted {len(info_chunks)} semantic chunks from career info supplement.")

    all_chunks = resume_chunks + info_chunks
    vector_store.add_chunks(all_chunks)
    print(f"Successfully indexed total {vector_store.count()} chunks into Personal RAG Vector Store.")

    # 4. Build CandidateProfile
    profile = CandidateProfile(
        name="Cory Stever",
        target_roles=["AI Engineer", "Data Scientist", "Web Developer"],
        career_interests=[
            "Artificial Intelligence",
            "Machine Learning",
            "Natural Language Processing",
            "Web Development",
            "Big Data",
            "Agentic AI",
            "Deep Learning",
        ],
        career_goals="Short-term: Learn the industry. Long-term: Become a high level employee and AI technical leader.",
        education_level="Master's Degree (Computer Science - AI, In Progress)",
        years_of_experience=2.0,
        salary_expectation=60000.0,
        job_types=["Full-Time"],
        preferred_locations=["Kansas City, MO", "Any", "Remote", "Hybrid", "On-site"],
        work_modes=["Remote", "Hybrid", "On-site"],
        clearance=None,
        technical_skills=[
            "Python",
            "JavaScript",
            "React.js",
            "Redux",
            "Flask",
            "SQLAlchemy",
            "SQL",
            "PostgreSQL",
            "Node.js",
            "Express.js",
            "AWS",
            "Git",
            "HTML",
            "CSS",
            "R",
            "BERT",
            "Transformers",
            "LoRA/PEFT",
            "Hugging Face",
            "NLP",
            "Deep Learning",
            "Machine Learning",
            "Agentic AI",
            "AdamW",
        ],
    )

    import json
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(profile.model_dump(), f, indent=2)
    print(f"Saved CandidateProfile to {PROFILE_FILE}.")

    # 5. Verify Hybrid Retriever
    retriever = HybridCareerRetriever(vector_store)
    test_queries = [
        "React and Redux full stack web application",
        "BERT transformer fine-tuning with LoRA and Hugging Face",
        "Agentic AI and O*NET dataset integration",
        "Python and Flask backend RESTful APIs",
        "Graduate coursework in Machine Learning at University of Missouri Kansas City",
    ]
    print("\n--- Verifying Hybrid Retrieval on Cory's Evidence ---")
    for q in test_queries:
        ev = retriever.retrieve_evidence_for_qualification(q)
        status = "✅ MATCH" if ev.supports_match else "❌ NO MATCH"
        print(f"{status} | Query: '{q}'")
        print(f"   Score: {ev.similarity_score:.2f} | Chunk: {ev.chunk_id} | Doc: {ev.source_document}")
        print(f"   Excerpt: {ev.evidence_text[:90]}...\n")


if __name__ == "__main__":
    ingest_cory_profile()
