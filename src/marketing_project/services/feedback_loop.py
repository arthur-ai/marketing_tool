"""
Feedback loop service for learning from user feedback and post performance.

Leverages existing ApprovalManager to track which posts get approved/rejected
and learns from high-performing posts.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("marketing_project.services.feedback_loop")


class FeedbackLoopService:
    """Service for collecting and learning from feedback."""

    def __init__(self):
        self._feedback_data: Dict[str, Dict[str, Any]] = {}

    async def record_feedback(
        self,
        job_id: str,
        feedback_type: str,
        rating: Optional[int] = None,
        comments: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record user feedback on a generated post.

        Args:
            job_id: Job ID of the post
            feedback_type: Type of feedback (approval, rejection, rating, etc.)
            rating: Optional rating (1-5 or similar)
            comments: Optional feedback comments
            metadata: Additional metadata

        Returns:
            Feedback record
        """
        feedback_id = f"feedback_{job_id}_{int(datetime.now(timezone.utc).timestamp())}"

        feedback_record = {
            "feedback_id": feedback_id,
            "job_id": job_id,
            "feedback_type": feedback_type,
            "rating": rating,
            "comments": comments,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self._feedback_data[feedback_id] = feedback_record

        logger.info(f"Recorded feedback {feedback_id} for job {job_id}")

        # Learn from feedback
        await self._learn_from_feedback(job_id, feedback_record)

        return feedback_record

    async def _learn_from_feedback(self, job_id: str, feedback: Dict[str, Any]) -> None:
        """
        Learn from feedback to improve future generations.

        Args:
            job_id: Job ID
            feedback: Feedback record
        """
        # TODO: Implement learning logic
        # - Track which posts get approved vs rejected
        # - Identify patterns in high-performing posts
        # - Adjust prompts based on feedback
        # - Update quality thresholds

        logger.debug(f"Learning from feedback for job {job_id}")

    async def get_feedback_stats(
        self, days: int = 30, platform: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get feedback statistics.

        Args:
            days: Number of days to look back
            platform: Optional platform filter

        Returns:
            Feedback statistics
        """
        cutoff_date = datetime.now(timezone.utc).timestamp() - (days * 24 * 60 * 60)

        recent_feedback = [
            f
            for f in self._feedback_data.values()
            if datetime.fromisoformat(f["created_at"]).timestamp() >= cutoff_date
        ]

        if platform:
            recent_feedback = [
                f
                for f in recent_feedback
                if f.get("metadata", {}).get("platform") == platform
            ]

        # Calculate stats
        total_feedback = len(recent_feedback)
        approval_count = sum(
            1 for f in recent_feedback if f["feedback_type"] == "approval"
        )
        rejection_count = sum(
            1 for f in recent_feedback if f["feedback_type"] == "rejection"
        )

        ratings = [f["rating"] for f in recent_feedback if f.get("rating")]
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        return {
            "total_feedback": total_feedback,
            "approval_count": approval_count,
            "rejection_count": rejection_count,
            "approval_rate": (
                approval_count / total_feedback if total_feedback > 0 else 0.0
            ),
            "average_rating": avg_rating,
            "days": days,
            "platform": platform,
        }

    async def get_high_performing_patterns(
        self, platform: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Identify patterns from high-performing posts.

        Args:
            platform: Optional platform filter
            limit: Maximum number of patterns to return

        Returns:
            List of identified patterns
        """
        # TODO: Implement pattern analysis
        # - Analyze approved posts
        # - Identify common characteristics
        # - Extract best practices

        return []


# Singleton instance
_feedback_service: Optional[FeedbackLoopService] = None


def get_feedback_service() -> FeedbackLoopService:
    """Get the singleton feedback loop service instance."""
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = FeedbackLoopService()
    return _feedback_service
