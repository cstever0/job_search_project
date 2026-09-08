"""Unit tests for the human feedback store."""

import pytest
from src.feedback.feedback_store import FeedbackStore, VALID_FEEDBACK_STATUSES


def test_feedback_crud_operations(tmp_path):
    store_file = tmp_path / "test_feedback.json"
    store = FeedbackStore(file_path=store_file)

    # 1. Record feedback
    fb1 = store.record_feedback(
        job_id="job_101",
        job_title="Data Scientist",
        employer="HHS",
        status="Interested",
        notes="Strong alignment with NLP and PyTorch experience.",
        overall_score=87.5,
    )

    assert fb1.job_id == "job_101"
    assert fb1.status == "Interested"

    # 2. Retrieve feedback
    retrieved = store.get_feedback("job_101")
    assert retrieved is not None
    assert retrieved.notes == "Strong alignment with NLP and PyTorch experience."

    # 3. Update feedback to Applied
    fb2 = store.record_feedback(
        job_id="job_101",
        job_title="Data Scientist",
        employer="HHS",
        status="Applied",
        notes="Application submitted on USAJOBS portal.",
        overall_score=87.5,
    )
    assert fb2.status == "Applied"
    assert store.get_feedback("job_101").status == "Applied"

    # 4. List feedback
    all_fb = store.list_all_feedback()
    assert len(all_fb) == 1

    # 5. Invalid status rejection
    with pytest.raises(ValueError):
        store.record_feedback(
            job_id="job_102",
            job_title="Analyst",
            employer="DoD",
            status="MaybeLater",  # Not a valid status
        )

    # 6. Delete feedback
    assert store.delete_feedback("job_101") is True
    assert store.get_feedback("job_101") is None
