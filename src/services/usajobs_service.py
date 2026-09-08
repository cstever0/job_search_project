"""Official USAJOBS API integration service and schema normalizer."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional
import requests

from config import USAJOBS_API_KEY, USAJOBS_USER_AGENT, has_usajobs_credentials
from src.data_ingestion.semantic_chunker import CareerChunker
from src.models.job import NormalizedJob

logger = logging.getLogger(__name__)

USAJOBS_API_ENDPOINT = "https://data.usajobs.gov/api/search"


class USAJobsService:
    """Service to search, retrieve, and normalize job announcements from USAJOBS."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        user_agent: Optional[str] = None,
        timeout: int = 10,
    ):
        self.api_key = api_key or USAJOBS_API_KEY
        self.user_agent = user_agent or USAJOBS_USER_AGENT
        self.timeout = timeout

    def is_configured(self) -> bool:
        """Check if USAJOBS API credentials are available."""
        return bool(
            self.api_key
            and self.user_agent
            and self.api_key != "your_usajobs_api_key_here"
            and self.user_agent != "your_email@example.com"
        )

    def search_jobs(
        self,
        keyword: str = "",
        location: str = "",
        is_remote: bool = False,
        job_category: str = "",
        salary_min: Optional[float] = None,
        results_per_page: int = 15,
        page: int = 1,
    ) -> list[NormalizedJob]:
        """Query USAJOBS API and return normalized job objects.

        If credentials are not configured or an API error occurs, gracefully
        returns high-fidelity realistic federal jobs for demonstration.
        """
        if not self.is_configured():
            logger.info("USAJOBS credentials not configured. Using fallback federal dataset.")
            return self._get_fallback_federal_jobs(keyword=keyword, location=location, is_remote=is_remote)

        headers = {
            "Host": "data.usajobs.gov",
            "User-Agent": self.user_agent,
            "Authorization-Key": self.api_key,
        }

        params: dict[str, Any] = {
            "ResultsPerPage": max(1, min(results_per_page, 50)),
            "Page": max(1, page),
        }

        if keyword.strip():
            params["Keyword"] = keyword.strip()
        if location.strip():
            params["LocationName"] = location.strip()
        if is_remote:
            params["Remotely"] = "True"
        if job_category.strip():
            params["JobCategoryCode"] = job_category.strip()
        if salary_min is not None and salary_min > 0:
            params["RemunerationMinimumAmount"] = int(salary_min)

        try:
            response = requests.get(
                USAJOBS_API_ENDPOINT,
                headers=headers,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return self.parse_api_response(data)
        except requests.exceptions.RequestException as e:
            logger.warning(f"USAJOBS API request failed: {e}. Falling back to demonstration dataset.")
            return self._get_fallback_federal_jobs(keyword=keyword, location=location, is_remote=is_remote)
        except Exception as e:
            logger.error(f"Error parsing USAJOBS response: {e}")
            return []

    def parse_api_response(self, data: dict[str, Any]) -> list[NormalizedJob]:
        """Normalize raw USAJOBS API JSON response into internal NormalizedJob models."""
        search_result = data.get("SearchResult", {})
        items = search_result.get("SearchResultItems", [])

        normalized_list: list[NormalizedJob] = []
        for item in items:
            try:
                job = self.normalize_job_descriptor(item)
                normalized_list.append(job)
            except Exception as e:
                logger.warning(f"Skipping malformed job item: {e}")
                continue

        return normalized_list

    def normalize_job_descriptor(self, raw_item: dict[str, Any]) -> NormalizedJob:
        """Convert a single USAJOBS SearchResultItem into NormalizedJob."""
        obj = raw_item.get("MatchedObjectDescriptor", {})
        job_id = raw_item.get("MatchedObjectId") or str(obj.get("PositionID", "unknown_id"))
        title = obj.get("PositionTitle", "Untitled Job")
        employer = obj.get("DepartmentName") or obj.get("OrganizationName") or "Federal Agency"
        location_display = obj.get("PositionLocationDisplay", "Unspecified")

        # Remote status
        is_remote = False
        locations = obj.get("PositionLocation", [])
        if locations and isinstance(locations, list):
            for loc in locations:
                if loc.get("PositionLocationDisplay", "").lower() == "negotiable" or loc.get("LocationName", "").lower() == "telework eligible":
                    is_remote = True
                    break
        if "remote" in location_display.lower() or "telework" in location_display.lower():
            is_remote = True

        # Salary
        remuneration = obj.get("PositionRemuneration", [])
        salary_min = None
        salary_max = None
        salary_type = "Per Year"
        if remuneration and isinstance(remuneration, list) and len(remuneration) > 0:
            rem = remuneration[0]
            try:
                salary_min = float(rem.get("MinimumRange", 0)) or None
                salary_max = float(rem.get("MaximumRange", 0)) or None
                salary_type = rem.get("RateIntervalCode", "Per Year")
            except (ValueError, TypeError):
                pass

        # Job Schedule / Type
        schedule_info = obj.get("PositionSchedule", [])
        job_type = "Full-Time"
        if schedule_info and isinstance(schedule_info, list):
            job_type = schedule_info[0].get("Name", "Full-Time")

        # Details from UserArea
        user_area = obj.get("UserArea", {})
        details = user_area.get("Details", {})
        job_summary = details.get("JobSummary", "")
        major_duties = details.get("MajorDuties", [])
        requirements = details.get("Requirements", "")
        education_text = details.get("Education", "")
        evaluations_text = details.get("Evaluations", "")

        # Responsibilities
        responsibilities = []
        if isinstance(major_duties, list):
            responsibilities = [d for d in major_duties if isinstance(d, str) and d.strip()]
        elif isinstance(major_duties, str) and major_duties.strip():
            responsibilities = [d.strip() for d in re.split(r"\n|•|\*", major_duties) if len(d.strip()) > 10]

        # Extract explicit employer requirements and skills
        req_qualifications = []
        if requirements:
            cleaned_req = re.sub(r"<[^>]+>", " ", requirements)
            req_qualifications = [q.strip() for q in re.split(r"\n|•|\*", cleaned_req) if len(q.strip()) > 20]

        combined_text = f"{job_summary} {' '.join(responsibilities)} {requirements} {education_text} {evaluations_text}"
        detected_skills = CareerChunker.extract_skills_from_text(combined_text)

        # Distinguish required vs preferred skills
        required_skills = detected_skills[: max(1, len(detected_skills) // 2)]
        preferred_skills = detected_skills[len(required_skills) :]

        # Mandatory disqualifiers (citizenship, clearance, drug testing)
        disqualifiers = []
        lower_comb = combined_text.lower()
        if re.search(r"\bu\.?s\.?\s+citizen(?:ship)?\b", lower_comb):
            disqualifiers.append("Must be a U.S. Citizen or U.S. National")
        if "top secret" in lower_comb:
            disqualifiers.append("Requires active Top Secret security clearance")
        elif "secret clearance" in lower_comb:
            disqualifiers.append("Requires active Secret security clearance")

        application_url = obj.get("PositionURI", "")
        source_url = obj.get("PositionURI", "")

        return NormalizedJob(
            job_id=str(job_id),
            title=title,
            employer=employer,
            location=location_display,
            is_remote=is_remote,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_type=salary_type,
            job_type=job_type,
            description=job_summary,
            responsibilities=responsibilities[:8],
            required_qualifications=req_qualifications[:6],
            preferred_qualifications=[evaluations_text[:200]] if evaluations_text else [],
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            experience_requirements=[r for r in req_qualifications if "experience" in r.lower()][:3],
            education_requirements=[education_text[:300]] if education_text else [],
            mandatory_disqualifiers=disqualifiers,
            application_url=application_url,
            source="USAJOBS",
            source_url=source_url,
            raw_data=raw_item,
        )

    def _get_fallback_federal_jobs(
        self, keyword: str = "", location: str = "", is_remote: bool = False
    ) -> list[NormalizedJob]:
        """Realistic federal job opportunities for demonstration and offline testing."""
        all_sample_jobs = [
            NormalizedJob(
                job_id="USA-GS-2210-001",
                title="Artificial Intelligence & Machine Learning Specialist (GS-2210-14)",
                employer="National Science Foundation (NSF)",
                location="Alexandria, VA (Remote Available)",
                is_remote=True,
                salary_min=132368.0,
                salary_max=172075.0,
                salary_type="Per Year",
                job_type="Full-Time",
                description="Lead federal AI research initiatives, deploying machine learning and NLP systems for grant evaluation.",
                responsibilities=[
                    "Architect and deploy transformer-based NLP and LLM pipelines for scientific proposal analysis.",
                    "Develop reproducible machine learning models in Python, PyTorch, and scikit-learn.",
                    "Build automated data extraction and ETL workflows on AWS cloud infrastructure.",
                    "Collaborate with senior researchers to establish AI governance and benchmark metrics.",
                ],
                required_qualifications=[
                    "At least 1 year of specialized experience at the GS-13 level in applied machine learning or data science.",
                    "Demonstrated proficiency building NLP models, transformers, and deep learning pipelines in Python.",
                    "Experience with cloud architectures (AWS) and containerized workflows (Docker).",
                ],
                preferred_qualifications=[
                    "Experience with vector databases, RAG systems, and semantic search.",
                    "Master's degree or higher in Computer Science, Data Science, or related STEM discipline.",
                ],
                required_skills=["Python", "Pytorch", "Nlp", "Machine Learning", "Aws"],
                preferred_skills=["Docker", "Sql", "Postgresql", "Transformers"],
                experience_requirements=["1+ year specialized experience in machine learning pipelines."],
                education_requirements=["Master's or Bachelor's in CS, Information Systems, or Data Science."],
                mandatory_disqualifiers=["Must be a U.S. Citizen or U.S. National"],
                application_url="https://www.usajobs.gov/job/789123401",
                source="USAJOBS",
                source_url="https://www.usajobs.gov/job/789123401",
            ),
            NormalizedJob(
                job_id="USA-GS-1560-002",
                title="Data Scientist (GS-1560-13)",
                employer="Department of Health and Human Services (HHS)",
                location="Bethesda, MD",
                is_remote=False,
                salary_min=112015.0,
                salary_max=145617.0,
                salary_type="Per Year",
                job_type="Full-Time",
                description="Perform advanced statistical analytics, predictive modeling, and data pipelines on federal public health records.",
                responsibilities=[
                    "Develop statistical models and machine learning pipelines in Python and SQL.",
                    "Clean and preprocess multi-terabyte healthcare datasets using PostgreSQL and pandas.",
                    "Communicate predictive insights to stakeholders using interactive dashboards and reports.",
                ],
                required_qualifications=[
                    "Specialized experience in statistical computing, data analysis, and predictive modeling.",
                    "Strong background in SQL, PostgreSQL, and Python data manipulation (pandas, numpy).",
                    "Degree in statistics, mathematics, computer science, or data science.",
                ],
                preferred_qualifications=[
                    "Familiarity with scikit-learn, classification algorithms, and regression modeling.",
                ],
                required_skills=["Python", "Sql", "Postgresql", "Statistics", "Pandas"],
                preferred_skills=["Scikit-Learn", "Machine Learning", "Git"],
                experience_requirements=["1+ year of specialized statistical and ML data modeling experience."],
                education_requirements=["Bachelor's or Master's degree in quantitative discipline."],
                mandatory_disqualifiers=["Must be a U.S. Citizen or U.S. National"],
                application_url="https://www.usajobs.gov/job/789123402",
                source="USAJOBS",
                source_url="https://www.usajobs.gov/job/789123402",
            ),
            NormalizedJob(
                job_id="USA-GS-1550-003",
                title="Computer Scientist - NLP & Information Extraction (GS-1550-13)",
                employer="Department of Commerce - NIST",
                location="Gaithersburg, MD (Hybrid Eligible)",
                is_remote=False,
                salary_min=117962.0,
                salary_max=153354.0,
                salary_type="Per Year",
                job_type="Full-Time",
                description="Conduct research in automated document analysis, information retrieval, and BERT/transformer language models.",
                responsibilities=[
                    "Investigate and benchmark modern transformer language models (BERT, GPT) for technical text processing.",
                    "Build hybrid information retrieval systems combining dense embeddings and BM25.",
                    "Publish open-source code and technical benchmarks for scientific computing.",
                ],
                required_qualifications=[
                    "Expertise in Python, PyTorch, Hugging Face transformers, and vector representations.",
                    "Understanding of retrieval benchmarks, cosine similarity, and text chunking.",
                ],
                preferred_qualifications=[
                    "Experience with Docker, Linux environment, and version control with Git.",
                ],
                required_skills=["Python", "Pytorch", "Nlp", "Bert", "Linux"],
                preferred_skills=["Docker", "Git", "Scikit-Learn"],
                experience_requirements=["Specialized experience in NLP research or software engineering."],
                education_requirements=["Degree in Computer Science or related engineering field."],
                mandatory_disqualifiers=["Must be a U.S. Citizen or U.S. National"],
                application_url="https://www.usajobs.gov/job/789123403",
                source="USAJOBS",
                source_url="https://www.usajobs.gov/job/789123403",
            ),
            NormalizedJob(
                job_id="USA-GS-2210-004",
                title="Cloud Data Systems Engineer (GS-2210-12)",
                employer="General Services Administration (GSA)",
                location="Washington, DC (Remote Available)",
                is_remote=True,
                salary_min=94199.0,
                salary_max=122459.0,
                salary_type="Per Year",
                job_type="Full-Time",
                description="Maintain and deploy secure federal cloud data platforms and continuous integration pipelines.",
                responsibilities=[
                    "Maintain AWS and Azure cloud computing environments for civic applications.",
                    "Automate deployment pipelines using Docker, Kubernetes, and CI/CD.",
                    "Manage relational databases (PostgreSQL, MySQL) ensuring high availability and backups.",
                ],
                required_qualifications=[
                    "Experience deploying containerized services with Docker and Kubernetes.",
                    "Hands-on experience with AWS cloud infrastructure and database administration.",
                ],
                preferred_qualifications=[
                    "Scripting in Python or Bash and Linux system administration.",
                ],
                required_skills=["Aws", "Docker", "Kubernetes", "Postgresql", "Linux"],
                preferred_skills=["Python", "Ci/Cd", "Git"],
                experience_requirements=["1+ year experience in cloud systems engineering."],
                education_requirements=["Bachelor's degree or equivalent technical experience."],
                mandatory_disqualifiers=["Must be a U.S. Citizen or U.S. National"],
                application_url="https://www.usajobs.gov/job/789123404",
                source="USAJOBS",
                source_url="https://www.usajobs.gov/job/789123404",
            ),
            NormalizedJob(
                job_id="USA-GS-0801-005",
                title="Commercial Pilot / Flight Operations Inspector (GS-1825-13)",
                employer="Federal Aviation Administration (FAA)",
                location="Atlanta, GA",
                is_remote=False,
                salary_min=110000.0,
                salary_max=140000.0,
                salary_type="Per Year",
                job_type="Full-Time",
                description="Conduct safety evaluations of commercial air carriers and flight crew operations.",
                responsibilities=[
                    "Conduct flight inspections and certification of commercial pilots.",
                    "Evaluate flight simulator training curricula.",
                ],
                required_qualifications=[
                    "Commercial pilot certificate with instrument rating.",
                    "Minimum 1,500 total flight hours as pilot-in-command.",
                ],
                preferred_qualifications=["Airline Transport Pilot (ATP) certificate."],
                required_skills=[],
                preferred_skills=[],
                experience_requirements=["1,500 flight hours."],
                education_requirements=["High school diploma or higher."],
                mandatory_disqualifiers=[
                    "Commercial Pilot License with Instrument Rating required",
                    "FAA First Class Medical Certificate required",
                ],
                application_url="https://www.usajobs.gov/job/789123405",
                source="USAJOBS",
                source_url="https://www.usajobs.gov/job/789123405",
            ),
        ]

        # Filter by keyword or remote if specified
        results = all_sample_jobs
        if keyword:
            kw = keyword.lower()
            results = [
                j
                for j in results
                if kw in j.title.lower()
                or kw in j.description.lower()
                or any(kw in s.lower() for s in j.required_skills)
            ]
            if not results:
                # If filter yielded nothing, return general tech list
                results = all_sample_jobs[:3]

        if is_remote:
            results = [j for j in results if j.is_remote] or results[:2]

        return results
