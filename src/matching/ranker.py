"""Job ranker evaluating and sorting candidate jobs by match score."""

from __future__ import annotations

from src.matching.explainer import MatchExplainer
from src.matching.scorer import JobScorer
from src.models.job import NormalizedJob
from src.models.match import JobMatchEvaluation
from src.models.profile import CandidateProfile
from src.rag.hybrid_retriever import HybridCareerRetriever
from src.services.onet_service import ONetService


class JobRanker:
    """Evaluates a batch of jobs and ranks them from highest match score to lowest."""

    def __init__(
        self,
        retriever: HybridCareerRetriever,
        onet_service: ONetService,
    ):
        self.retriever = retriever
        self.onet_service = onet_service
        self.scorer = JobScorer(retriever)
        self.explainer = MatchExplainer(retriever)

    def evaluate_and_rank(
        self,
        jobs: list[NormalizedJob],
        profile: CandidateProfile,
        enrich_with_onet: bool = True,
    ) -> list[JobMatchEvaluation]:
        """Score, enrich, and rank jobs descending by overall match score."""
        evaluations: list[JobMatchEvaluation] = []

        for job in jobs:
            # 1. Enrich with O*NET occupational data if requested and not yet enriched
            if enrich_with_onet and not job.onet_code:
                try:
                    job = self.onet_service.enrich_job(job)
                except Exception:
                    pass

            # 2. Score job using fixed matching model
            breakdown, evidence_list = self.scorer.score_job(job, profile)

            # 3. Build comprehensive evaluation & explanation
            eval_obj = self.explainer.build_evaluation(job, profile, breakdown, evidence_list)
            evaluations.append(eval_obj)

        # 4. Sort descending by overall match score
        # Keep non-ineligible jobs prioritized over ineligible ones at the same score
        evaluations.sort(
            key=lambda e: (
                not e.breakdown.ineligible_flag,
                e.breakdown.overall_score,
            ),
            reverse=True,
        )

        return evaluations
