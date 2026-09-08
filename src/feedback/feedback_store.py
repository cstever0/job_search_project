"""Persistent local storage for human feedback on job recommendations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import FEEDBACK_FILE
from src.models.match import HumanFeedback

VALID_FEEDBACK_STATUSES = {
    "Interested",
    "Not Interested",
    "Good Match",
    "Poor Match",
    "Applied",
}


class FeedbackStore:
    """Manages local storage of candidate feedback without altering fixed scoring weights."""

    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = file_path or FEEDBACK_FILE
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, HumanFeedback] = {}
        self._load()

    def _load(self) -> None:
        """Load feedback entries from disk."""
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._cache = {
                    item["job_id"]: HumanFeedback(**item)
                    for item in data.values()
                }
            except Exception:
                self._cache = {}

    def _save(self) -> None:
        """Save feedback entries to disk."""
        data = {job_id: fb.model_dump() for job_id, fb in self._cache.items()}
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def record_feedback(
        self,
        job_id: str,
        job_title: str,
        employer: str,
        status: str,
        notes: str = "",
        overall_score: float = 0.0,
    ) -> HumanFeedback:
        """Record or update user feedback for a recommended job."""
        if status not in VALID_FEEDBACK_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Must be one of: {VALID_FEEDBACK_STATUSES}"
            )

        now_str = datetime.now(timezone.utc).isoformat()
        existing = self._cache.get(job_id)
        created_at = existing.created_at if existing else now_str

        fb = HumanFeedback(
            job_id=job_id,
            job_title=job_title,
            employer=employer,
            status=status,
            notes=notes,
            overall_score=overall_score,
            created_at=created_at,
            updated_at=now_str,
        )

        self._cache[job_id] = fb
        self._save()
        return fb

    def get_feedback(self, job_id: str) -> Optional[HumanFeedback]:
        """Retrieve feedback for a specific job, or None."""
        return self._cache.get(job_id)

    def list_all_feedback(self) -> list[HumanFeedback]:
        """Return all recorded feedback entries sorted by update time descending."""
        items = list(self._cache.values())
        items.sort(key=lambda x: x.updated_at, reverse=True)
        return items

    def delete_feedback(self, job_id: str) -> bool:
        """Delete feedback for a given job."""
        if job_id in self._cache:
            del self._cache[job_id]
            self._save()
            return True
        return False
