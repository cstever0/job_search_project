"""Unit tests for USAJOBS service, query generation, and normalization."""

import pytest
from src.models.job import NormalizedJob
from src.services.usajobs_service import USAJobsService


SAMPLE_USAJOBS_RAW_ITEM = {
    "MatchedObjectId": "12345678",
    "MatchedObjectDescriptor": {
        "PositionTitle": "Data Scientist",
        "PositionID": "12345678",
        "PositionURI": "https://www.usajobs.gov/job/12345678",
        "DepartmentName": "Department of the Treasury",
        "OrganizationName": "Internal Revenue Service",
        "PositionLocationDisplay": "Washington, DC",
        "PositionLocation": [
            {"LocationName": "Washington, DC", "PositionLocationDisplay": "Washington, DC"}
        ],
        "PositionSchedule": [{"Code": "1", "Name": "Full-Time"}],
        "PositionRemuneration": [
            {"MinimumRange": "105000", "MaximumRange": "140000", "RateIntervalCode": "Per Year"}
        ],
        "UserArea": {
            "Details": {
                "JobSummary": "Serve as a Data Scientist conducting predictive modeling using Python and SQL.",
                "MajorDuties": [
                    "Develop statistical models and machine learning pipelines in Python.",
                    "Manage PostgreSQL databases and build automated data pipelines.",
                ],
                "Requirements": "Must possess 1 year specialized experience in machine learning. Must be a U.S. Citizen or U.S. National.",
                "Education": "Bachelor's degree or higher in Computer Science, Data Science, or Mathematics.",
                "Evaluations": "Candidates will be evaluated on Python programming and statistical reasoning.",
            }
        },
    },
}


def test_usajobs_normalization():
    service = USAJobsService()
    job = service.normalize_job_descriptor(SAMPLE_USAJOBS_RAW_ITEM)

    assert isinstance(job, NormalizedJob)
    assert job.job_id == "12345678"
    assert job.title == "Data Scientist"
    assert "Treasury" in job.employer
    assert job.location == "Washington, DC"
    assert job.salary_min == 105000.0
    assert job.salary_max == 140000.0
    assert job.salary_type == "Per Year"
    assert job.job_type == "Full-Time"
    assert job.application_url == "https://www.usajobs.gov/job/12345678"

    # Skills detection from duties and summary
    all_skills_lower = [s.lower() for s in job.required_skills + job.preferred_skills]
    assert "python" in all_skills_lower
    assert "sql" in all_skills_lower

    # Disqualification flag
    assert any("U.S. Citizen" in d for d in job.mandatory_disqualifiers)


def test_usajobs_missing_fields_handling():
    service = USAJobsService()
    minimal_raw = {
        "MatchedObjectId": "999",
        "MatchedObjectDescriptor": {
            "PositionTitle": "General Analyst",
        },
    }
    job = service.normalize_job_descriptor(minimal_raw)
    assert job.job_id == "999"
    assert job.title == "General Analyst"
    assert job.salary_min is None
    assert job.salary_max is None
    assert job.salary_display == "Salary not disclosed"
    assert job.responsibilities == []
    assert job.mandatory_disqualifiers == []


def test_usajobs_fallback_search():
    service = USAJobsService(api_key="", user_agent="")
    jobs = service.search_jobs(keyword="Machine Learning")
    assert len(jobs) > 0
    assert all(isinstance(j, NormalizedJob) for j in jobs)
    assert any("Machine Learning" in j.title or "AI" in j.title for j in jobs)
