"""Central configuration module for Personal AI Job Search and Matching Application.

Loads settings from .env, defines data directory paths, and locks the
human-designed fixed matching weights.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
UPLOADS_DIR: Final[Path] = DATA_DIR / "uploads"
VECTOR_DB_DIR: Final[Path] = DATA_DIR / "vector_db"
FEEDBACK_FILE: Final[Path] = DATA_DIR / "feedback.json"
PROFILE_FILE: Final[Path] = DATA_DIR / "profile.json"
CACHE_DIR: Final[Path] = PROJECT_ROOT / ".cache"

# Ensure local directories exist
for directory in (DATA_DIR, UPLOADS_DIR, VECTOR_DB_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Ensure cache env vars point inside workspace for sandboxed execution
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)
os.environ["CHROMA_CACHE_DIR"] = str(CACHE_DIR / "chroma")

# Load .env file
load_dotenv(PROJECT_ROOT / ".env")

# API Keys and Credentials
USAJOBS_API_KEY: str = os.getenv("USAJOBS_API_KEY", "").strip()
USAJOBS_USER_AGENT: str = os.getenv("USAJOBS_USER_AGENT", "").strip()
ONET_API_KEY: str = os.getenv("ONET_API_KEY", "").strip()
ONET_USERNAME: str = os.getenv("ONET_USERNAME", "").strip()
ONET_PASSWORD: str = os.getenv("ONET_PASSWORD", "").strip()
APP_ENV: str = os.getenv("APP_ENV", "development").strip()
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()

# ==============================================================================
# FIXED WEIGHTED SCORING SYSTEM (Total: 100%)
# As specified by the human-designed matching model:
# Skills: 35%, Experience: 25%, Education: 15%, Career Interest: 15%, Salary: 5%, Job Type: 5%
# Location is 0% (user is geographically flexible).
# ==============================================================================
WEIGHT_SKILLS: Final[float] = 35.0
WEIGHT_EXPERIENCE: Final[float] = 25.0
WEIGHT_EDUCATION: Final[float] = 15.0
WEIGHT_CAREER_INTEREST: Final[float] = 15.0
WEIGHT_SALARY: Final[float] = 5.0
WEIGHT_JOB_TYPE: Final[float] = 5.0
WEIGHT_TOTAL: Final[float] = 100.0

# Verify mathematically at import time
assert (
    round(
        WEIGHT_SKILLS
        + WEIGHT_EXPERIENCE
        + WEIGHT_EDUCATION
        + WEIGHT_CAREER_INTEREST
        + WEIGHT_SALARY
        + WEIGHT_JOB_TYPE,
        6,
    )
    == WEIGHT_TOTAL
), "Fixed weights must sum precisely to 100.0"

SCORING_WEIGHTS: Final[dict[str, float]] = {
    "skills": WEIGHT_SKILLS,
    "experience": WEIGHT_EXPERIENCE,
    "education": WEIGHT_EDUCATION,
    "career_interest": WEIGHT_CAREER_INTEREST,
    "salary": WEIGHT_SALARY,
    "job_type": WEIGHT_JOB_TYPE,
}


def has_usajobs_credentials() -> bool:
    """Return True if USAJOBS API key and user agent are configured."""
    return bool(USAJOBS_API_KEY and USAJOBS_USER_AGENT and USAJOBS_API_KEY != "your_usajobs_api_key_here")


def has_onet_credentials() -> bool:
    """Return True if O*NET credentials (API key or username/password) are configured."""
    if ONET_API_KEY and ONET_API_KEY != "your_onet_api_key_here":
        return True
    return bool(ONET_USERNAME and ONET_PASSWORD and ONET_USERNAME != "your_onet_username_here")
