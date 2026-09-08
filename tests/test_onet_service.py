"""Unit tests for O*NET Web Services integration and strict separation guardrail."""

import pytest
from src.models.job import NormalizedJob
from src.services.onet_service import ONetService


def test_onet_occupation_mapping():
    service = ONetService()
    occ = service.search_occupation_code("Senior Machine Learning Engineer")
    assert occ is not None
    assert "code" in occ
    assert "15-2051" in occ["code"]  # Data Scientist / ML Engineer SOC group


def test_onet_enrichment_strict_separation():
    service = ONetService()
    job = NormalizedJob(
        job_id="test_01",
        title="Data Scientist",
        employer="Federal Agency",
        required_skills=["Python", "SQL"],
        preferred_skills=["Docker"],
        required_qualifications=["Must have Bachelor's in CS"],
    )

    enriched = service.enrich_job(job)

    # Verify O*NET fields are populated
    assert enriched.onet_code is not None
    assert enriched.onet_title is not None
    assert len(enriched.onet_occupational_skills) > 0
    assert len(enriched.onet_technology_skills) > 0

    # STRICT GUARDRAIL: Employer requirements must NOT be polluted or overwritten by O*NET data
    assert enriched.required_skills == ["Python", "SQL"]
    assert enriched.preferred_skills == ["Docker"]
    assert enriched.required_qualifications == ["Must have Bachelor's in CS"]

    # Verify that an O*NET skill is in onet_occupational_skills, not employer required_skills
    for skill in enriched.onet_occupational_skills:
        # The employer's required_skills must only contain what the employer specified
        assert skill not in enriched.required_skills or skill in ["Python", "SQL"]


def test_onet_api_key_configuration():
    service = ONetService(api_key="test_sample_key_123")
    assert service.is_configured() is True
    headers, auth = service._get_headers_and_auth()
    assert headers.get("X-API-Key") == "test_sample_key_123"
    assert auth is None
