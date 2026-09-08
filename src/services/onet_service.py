"""Official O*NET Web Services API integration service and occupational enricher.

Maintains STRICT separation between EMPLOYER REQUIREMENTS and
O*NET OCCUPATIONAL INFORMATION.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import requests
from requests.auth import HTTPBasicAuth

from config import ONET_API_KEY, ONET_PASSWORD, ONET_USERNAME, has_onet_credentials
from src.models.job import NormalizedJob

logger = logging.getLogger(__name__)

ONET_BASE_URL = "https://services.onetcenter.org/ws"


class ONetService:
    """Service to connect to O*NET Web Services and enrich jobs with occupational data."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 10,
    ):
        self.api_key = api_key or ONET_API_KEY
        self.username = username or ONET_USERNAME
        self.password = password or ONET_PASSWORD
        self.timeout = timeout

    def is_configured(self) -> bool:
        """Check if O*NET API credentials (API key or username/password) are set."""
        if self.api_key and self.api_key != "your_onet_api_key_here":
            return True
        return bool(
            self.username
            and self.password
            and self.username != "your_onet_username_here"
        )

    def _get_headers_and_auth(self) -> tuple[dict[str, str], Optional[HTTPBasicAuth]]:
        """Construct headers and HTTPBasicAuth based on available credentials."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "PersonalAIJobSearch/1.0",
        }
        auth: Optional[HTTPBasicAuth] = None

        if self.api_key and self.api_key != "your_onet_api_key_here":
            headers["X-API-Key"] = self.api_key
        elif self.username and self.password:
            auth = HTTPBasicAuth(self.username, self.password)

        return headers, auth

    def _make_request(self, url: str, params: Optional[dict[str, Any]] = None) -> Optional[requests.Response]:
        """Perform request supporting both X-API-Key header and Basic auth fallback."""
        headers, auth = self._get_headers_and_auth()
        try:
            resp = requests.get(url, headers=headers, auth=auth, params=params, timeout=self.timeout)
            if resp.status_code == 401 and self.api_key:
                # Retry with API key in Basic Auth (common for some O*NET account tiers)
                basic_auth = HTTPBasicAuth(self.api_key, "")
                resp = requests.get(url, headers=headers, auth=basic_auth, params=params, timeout=self.timeout)
            return resp
        except Exception as e:
            logger.warning(f"O*NET request to {url} failed: {e}")
            return None

    def search_occupation_code(self, title: str) -> Optional[dict[str, str]]:
        """Search O*NET for an occupation matching the job title.

        Returns dict with 'code' and 'title' or None.
        """
        if not self.is_configured():
            return self._fallback_occupation_mapping(title)

        endpoint = f"{ONET_BASE_URL}/online/search"
        params = {"kw": title}

        try:
            response = self._make_request(endpoint, params=params)
            if response and response.status_code == 200:
                data = response.json()
                occupations = data.get("occupation", [])
                if occupations and isinstance(occupations, list):
                    top_match = occupations[0]
                    return {
                        "code": top_match.get("code", ""),
                        "title": top_match.get("title", ""),
                    }
        except Exception as e:
            logger.warning(f"O*NET occupation search failed for '{title}': {e}. Using fallback taxonomy.")

        return self._fallback_occupation_mapping(title)

    def get_occupational_details(self, onet_code: str) -> dict[str, Any]:
        """Retrieve occupational knowledge, skills, abilities, and tech skills from O*NET."""
        if not self.is_configured():
            return self._fallback_occupational_details(onet_code)

        endpoint = f"{ONET_BASE_URL}/online/occupations/{onet_code}/summary"
        details: dict[str, Any] = {
            "description": "",
            "skills": [],
            "knowledge": [],
            "abilities": [],
            "technology_skills": [],
            "tasks": [],
        }

        try:
            # Query summary
            response = self._make_request(endpoint)
            if not response or response.status_code != 200:
                return self._fallback_occupational_details(onet_code)

            summary_data = response.json()
            details["description"] = summary_data.get("description", "")

            # Query technology skills
            tech_resp = self._make_request(f"{endpoint}/technology_skills")
            if tech_resp and tech_resp.status_code == 200:
                tech_data = tech_resp.json()
                items = tech_data.get("category", [])
                for cat in items:
                    for ex in cat.get("example", []):
                        if isinstance(ex, dict) and "name" in ex:
                            details["technology_skills"].append(ex["name"])

            # Query knowledge
            know_resp = self._make_request(f"{endpoint}/knowledge")
            if know_resp and know_resp.status_code == 200:
                know_data = know_resp.json()
                for el in know_data.get("element", []):
                    details["knowledge"].append(el.get("name", ""))

            # Query skills
            skill_resp = self._make_request(f"{endpoint}/skills")
            if skill_resp and skill_resp.status_code == 200:
                skill_data = skill_resp.json()
                for el in skill_data.get("element", []):
                    details["skills"].append(el.get("name", ""))

            # If live response had empty skills, backfill from taxonomy
            if not details["skills"] or not details["technology_skills"]:
                fallback = self._fallback_occupational_details(onet_code)
                details["skills"] = details["skills"] or fallback.get("skills", [])
                details["technology_skills"] = details["technology_skills"] or fallback.get("technology_skills", [])
                details["knowledge"] = details["knowledge"] or fallback.get("knowledge", [])

            return details
        except Exception as e:
            logger.warning(f"O*NET details query failed for {onet_code}: {e}. Using fallback taxonomy.")
            return self._fallback_occupational_details(onet_code)

    def enrich_job(self, job: NormalizedJob) -> NormalizedJob:
        """Enrich a NormalizedJob with O*NET occupational data.

        Preserves strict architectural separation: O*NET data is stored in
        onet_* fields and explicitly marked as O*NET OCCUPATIONAL INFORMATION,
        never as employer requirements.
        """
        occ = self.search_occupation_code(job.title)
        if not occ or not occ.get("code"):
            return job

        job.onet_code = occ["code"]
        job.onet_title = occ["title"]

        details = self.get_occupational_details(occ["code"])
        job.onet_description = details.get("description", "")
        job.onet_occupational_skills = details.get("skills", [])
        job.onet_knowledge = details.get("knowledge", [])
        job.onet_abilities = details.get("abilities", [])
        job.onet_technology_skills = details.get("technology_skills", [])
        job.onet_tasks = details.get("tasks", [])

        return job

    @staticmethod
    def _fallback_occupation_mapping(title: str) -> dict[str, str]:
        """Local taxonomy mapping for standard federal/tech roles."""
        t = title.lower()
        if any(w in t for w in ["machine learning", "ai", "artificial intelligence"]):
            return {"code": "15-2051.02", "title": "Machine Learning Engineers"}
        elif any(w in t for w in ["data scientist", "data science"]):
            return {"code": "15-2051.00", "title": "Data Scientists"}
        elif any(w in t for w in ["data engineer", "database", "data systems"]):
            return {"code": "15-1243.00", "title": "Database Architects"}
        elif any(w in t for w in ["cloud", "devops", "systems engineer"]):
            return {"code": "15-1299.08", "title": "Computer Systems Engineers/Architects"}
        elif any(w in t for w in ["computer scientist", "software"]):
            return {"code": "15-1252.00", "title": "Software Developers"}
        elif any(w in t for w in ["pilot", "flight"]):
            return {"code": "53-2012.00", "title": "Commercial Pilots"}
        return {"code": "15-1252.00", "title": "Software Developers"}

    @staticmethod
    def _fallback_occupational_details(onet_code: str) -> dict[str, Any]:
        """Curated occupational data for standard O*NET occupations."""
        if onet_code in ("15-2051.02", "15-2051.00"):
            return {
                "description": "Develop, evaluate, and deploy machine learning models, statistical algorithms, and data systems to uncover patterns and predict outcomes.",
                "skills": [
                    "Mathematics & Statistics",
                    "Complex Problem Solving",
                    "Critical Thinking",
                    "Programming",
                    "Systems Analysis",
                ],
                "knowledge": [
                    "Computer and Information Research",
                    "Mathematics & Statistics",
                    "Engineering and Technology",
                    "English Language",
                ],
                "abilities": [
                    "Mathematical Reasoning",
                    "Deductive Reasoning",
                    "Inductive Reasoning",
                    "Information Ordering",
                ],
                "technology_skills": [
                    "Python",
                    "PyTorch",
                    "scikit-learn",
                    "TensorFlow",
                    "Apache Spark",
                    "SQL",
                    "Docker",
                    "Hugging Face Transformers",
                ],
                "tasks": [
                    "Formulate mathematical and simulation models of problems.",
                    "Design artificial intelligence and machine learning algorithms.",
                    "Verify accuracy and validity of predictive models.",
                ],
            }
        elif onet_code == "15-1252.00":
            return {
                "description": "Research, design, and develop computer and network software or specialized utility programs.",
                "skills": [
                    "Programming",
                    "Systems Analysis",
                    "Troubleshooting",
                    "Critical Thinking",
                ],
                "knowledge": [
                    "Computers and Electronics",
                    "Engineering and Technology",
                    "Design",
                ],
                "abilities": [
                    "Information Ordering",
                    "Problem Sensitivity",
                    "Deductive Reasoning",
                ],
                "technology_skills": [
                    "Python",
                    "Java",
                    "C++",
                    "Git",
                    "Linux",
                    "PostgreSQL",
                    "Docker",
                ],
                "tasks": [
                    "Modify existing software to correct errors or improve its performance.",
                    "Develop and direct software system testing and validation procedures.",
                ],
            }
        elif onet_code == "53-2012.00":
            return {
                "description": "Pilot and navigate the flight of fixed-wing aircraft on non-scheduled commercial passenger or cargo flights.",
                "skills": ["Operation and Control", "Operation Monitoring", "Active Listening"],
                "knowledge": ["Transportation", "Geography", "Public Safety and Security"],
                "abilities": ["Spatial Orientation", "Control Precision", "Reaction Time"],
                "technology_skills": ["Flight planning software", "GPS navigation systems"],
                "tasks": ["Inspect aircraft prior to takeoff", "Navigate aircraft safely along designated routes"],
            }
        # Default
        return {
            "description": "Information technology and computing professional.",
            "skills": ["Critical Thinking", "Programming", "Active Learning"],
            "knowledge": ["Computers and Electronics", "Mathematics"],
            "abilities": ["Deductive Reasoning", "Inductive Reasoning"],
            "technology_skills": ["Python", "SQL", "Git"],
            "tasks": ["Analyze technical requirements and write software specifications."],
        }
