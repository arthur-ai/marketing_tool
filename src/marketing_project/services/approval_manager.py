"""
Approval Manager Service.

Manages approval requests for human-in-the-loop review of non-deterministic agent outputs.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from pydantic import BaseModel
from sqlalchemy import delete, func, select, update

from marketing_project.models.db_models import ApprovalModel, ApprovalSettingsModel
from marketing_project.services.database import get_database_manager
from marketing_project.services.redis_manager import get_redis_manager


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for datetime and other non-serializable objects."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    # Handle Pydantic BaseModel instances
    if isinstance(obj, BaseModel):
        try:
            return obj.model_dump(mode="json")
        except (TypeError, ValueError):
            # Fallback to regular model_dump if mode='json' fails
            return obj.model_dump()
    raise TypeError(f"Type {type(obj)} not serializable")


from marketing_project.models.approval_models import (
    ApprovalDecisionRequest,
    ApprovalRequest,
    ApprovalSettings,
    ApprovalStats,
)

logger = logging.getLogger(__name__)

# Redis key for storing approval settings
APPROVAL_SETTINGS_KEY = "approval:settings"
# Redis key prefix for pipeline context
PIPELINE_CONTEXT_KEY_PREFIX = "pipeline:context:"
# Redis key prefix for approval requests
APPROVAL_KEY_PREFIX = "approval:request:"
# Redis key for approval list
APPROVAL_LIST_KEY = "approval:list"
# Redis key prefix for job approvals mapping
JOB_APPROVAL_KEY_PREFIX = "approval:job:"


class ApprovalManager:
    """
    Manages approval requests and their lifecycle.

    Settings are persisted in PostgreSQL database (source of truth) with Redis as cache layer.
    This allows settings to be shared between API and worker processes with persistence.
    """

    def __init__(self, settings: Optional[ApprovalSettings] = None):
        self.settings = settings or ApprovalSettings()
        self._redis_manager = get_redis_manager()

    async def get_redis(self) -> Optional[redis.Redis]:
        """Get Redis client from RedisManager."""
        try:
            return await self._redis_manager.get_redis()
        except Exception as e:
            logger.warning(
                f"Failed to get Redis connection: {e}. Settings will only be stored in memory."
            )
            return None

    async def load_settings_from_db(self) -> Optional[ApprovalSettings]:
        """Load settings from database (source of truth)."""
        try:
            db_manager = get_database_manager()
            if not db_manager.is_initialized:
                return None

            async with db_manager.get_session() as session:
                result = await session.execute(
                    select(ApprovalSettingsModel)
                    .order_by(ApprovalSettingsModel.id.desc())
                    .limit(1)
                )
                settings_model = result.scalar_one_or_none()

                if settings_model:
                    settings_dict = settings_model.to_dict()
                    return ApprovalSettings(**settings_dict)
        except Exception as e:
            logger.warning(f"Failed to load settings from database: {e}")
        return None

    async def load_settings_from_redis(self) -> Optional[ApprovalSettings]:
        """Load settings from Redis (cache fallback)."""
        try:

            async def get_operation(redis_client: redis.Redis):
                return await redis_client.get(APPROVAL_SETTINGS_KEY)

            settings_json = await self._redis_manager.execute(get_operation)
            if settings_json:
                settings_dict = json.loads(settings_json)
                return ApprovalSettings(**settings_dict)
        except Exception as e:
            logger.warning(f"Failed to load settings from Redis: {e}")
        return None

    async def load_settings(self) -> Optional[ApprovalSettings]:
        """
        Load settings from database (source of truth), fallback to Redis cache.

        Returns:
            ApprovalSettings if found, None otherwise
        """
        # Try database first (source of truth)
        settings = await self.load_settings_from_db()
        if settings:
            # Update Redis cache
            await self.save_settings_to_redis(settings)
            return settings

        # Fallback to Redis cache
        settings = await self.load_settings_from_redis()
        return settings

    async def save_settings_to_db(self, settings: ApprovalSettings) -> bool:
        """
        Save settings to database (source of truth).

        Returns:
            True if successful, False otherwise
        """
        try:
            db_manager = get_database_manager()
            if not db_manager.is_initialized:
                return False

            async with db_manager.get_session() as session:
                # Get existing settings (there should only be one row)
                result = await session.execute(
                    select(ApprovalSettingsModel)
                    .order_by(ApprovalSettingsModel.id.desc())
                    .limit(1)
                )
                settings_model = result.scalar_one_or_none()

                if settings_model:
                    # Update existing
                    settings_model.require_approval = settings.require_approval
                    settings_model.approval_agents = settings.approval_agents
                    settings_model.auto_approve_threshold = (
                        str(settings.auto_approve_threshold)
                        if settings.auto_approve_threshold
                        else None
                    )
                    settings_model.timeout_seconds = settings.timeout_seconds
                else:
                    # Create new
                    settings_model = ApprovalSettingsModel(
                        require_approval=settings.require_approval,
                        approval_agents=settings.approval_agents,
                        auto_approve_threshold=(
                            str(settings.auto_approve_threshold)
                            if settings.auto_approve_threshold
                            else None
                        ),
                        timeout_seconds=settings.timeout_seconds,
                    )
                    session.add(settings_model)

                await session.commit()
                logger.info(
                    f"Saved approval settings to database: require_approval={settings.require_approval}"
                )
                return True
        except Exception as e:
            logger.error(f"Failed to save settings to database: {e}")
            return False

    async def save_settings_to_redis(self, settings: ApprovalSettings):
        """Save settings to Redis (cache)."""
        try:
            settings_json = json.dumps(
                settings.model_dump(mode="json"), default=_json_serializer
            )

            async def set_operation(redis_client: redis.Redis):
                await redis_client.set(APPROVAL_SETTINGS_KEY, settings_json)

            await self._redis_manager.execute(set_operation)
            logger.debug(
                f"Saved approval settings to Redis cache: require_approval={settings.require_approval}"
            )
        except Exception as e:
            logger.warning(f"Failed to save settings to Redis cache: {e}")

    async def save_settings(self, settings: ApprovalSettings):
        """
        Save settings to database (source of truth) and update Redis cache.

        Args:
            settings: ApprovalSettings to save
        """
        # Save to database first (source of truth)
        db_success = await self.save_settings_to_db(settings)

        # Update Redis cache regardless of DB success (for backward compatibility)
        await self.save_settings_to_redis(settings)

        if not db_success:
            logger.warning(
                "Settings saved to Redis cache only (database not available)"
            )

    async def save_pipeline_context(
        self,
        job_id: str,
        context: Dict,
        step_name: str,
        step_number: int,
        step_result: Dict,
        original_content: Optional[Dict] = None,
    ):
        """
        Save pipeline context to Redis for resuming after approval.

        Args:
            job_id: Job ID
            context: Accumulated context from previous pipeline steps
            step_name: Name of the step that requires approval
            step_number: Step number that requires approval
            step_result: Result from the step that requires approval
            original_content: Original content JSON (for resume)
        """
        try:
            context_data = {
                "context": context,
                "last_step": step_name,
                "last_step_number": step_number,
                "step_result": step_result,
                "original_content": original_content,
                "content_type": context.get(
                    "content_type", "blog_post"
                ),  # Save content_type
                "output_content_type": context.get(
                    "output_content_type"
                ),  # Save output_content_type
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "job_id": job_id,
            }

            context_json = json.dumps(context_data, default=_json_serializer)
            context_key = f"{PIPELINE_CONTEXT_KEY_PREFIX}{job_id}"

            async def setex_operation(redis_client: redis.Redis):
                # Store with 24 hour TTL
                await redis_client.setex(context_key, 86400, context_json)

            await self._redis_manager.execute(setex_operation)
            logger.info(
                f"Saved pipeline context for job {job_id} at step {step_number} ({step_name})"
            )
        except Exception as e:
            logger.error(f"Failed to save pipeline context for job {job_id}: {e}")

    async def load_pipeline_context(self, job_id: str) -> Optional[Dict]:
        """
        Load pipeline context from Redis for resuming after approval.

        Args:
            job_id: Job ID

        Returns:
            Dictionary with context, last_step, last_step_number, step_result, or None if not found
        """
        try:
            context_key = f"{PIPELINE_CONTEXT_KEY_PREFIX}{job_id}"

            async def get_operation(redis_client: redis.Redis):
                return await redis_client.get(context_key)

            context_json = await self._redis_manager.execute(get_operation)

            if context_json:
                context_data = json.loads(context_json)
                logger.info(
                    f"Loaded pipeline context for job {job_id} from step {context_data.get('last_step_number')}"
                )
                return context_data
            else:
                logger.warning(f"No pipeline context found for job {job_id}")
                return None
        except Exception as e:
            logger.error(f"Failed to load pipeline context for job {job_id}: {e}")
            return None

    async def create_approval_request(
        self,
        job_id: str,
        agent_name: str,
        step_name: str,
        input_data: Dict,
        output_data: Dict,
        confidence_score: Optional[float] = None,
        suggestions: Optional[List[str]] = None,
        pipeline_step: Optional[str] = None,
    ) -> ApprovalRequest:
        """
        Create a new approval request.

        Writes to PostgreSQL (source of truth) and caches in Redis.

        Args:
            job_id: ID of the parent processing job
            agent_name: Name of the agent that generated the output
            step_name: Pipeline step name
            input_data: Input provided to the agent
            output_data: Output generated by the agent
            confidence_score: Agent's confidence in output (0-1)
            suggestions: Optional suggestions for reviewer
            pipeline_step: Actual pipeline step name for retry (defaults to agent_name)

        Returns:
            Created approval request
        """
        approval_id = str(uuid.uuid4())

        if pipeline_step is None:
            pipeline_step = agent_name

        approval = ApprovalRequest(
            id=approval_id,
            job_id=job_id,
            agent_name=agent_name,
            step_name=step_name,
            pipeline_step=pipeline_step,
            input_data=input_data,
            output_data=output_data,
            confidence_score=confidence_score,
            suggestions=suggestions,
        )

        # Check auto-approval threshold
        if (
            self.settings.auto_approve_threshold is not None
            and confidence_score is not None
            and confidence_score >= self.settings.auto_approve_threshold
        ):
            logger.info(
                f"Auto-approving {approval_id} (confidence: {confidence_score:.2f} >= {self.settings.auto_approve_threshold})"
            )
            approval.status = "approved"
            approval.reviewed_at = datetime.now(timezone.utc)
            approval.user_comment = "Auto-approved based on confidence threshold"

        # Persist to PostgreSQL (source of truth)
        try:
            db_manager = get_database_manager()
            if db_manager.is_initialized:
                async with db_manager.get_session() as session:
                    db_approval = ApprovalModel(
                        approval_id=approval_id,
                        job_id=job_id,
                        agent_name=agent_name,
                        step_name=step_name,
                        pipeline_step=pipeline_step,
                        status=approval.status,
                        input_data=input_data,
                        output_data=output_data,
                        confidence_score=(
                            str(confidence_score)
                            if confidence_score is not None
                            else None
                        ),
                        user_comment=approval.user_comment,
                        reviewed_at=approval.reviewed_at,
                    )
                    session.add(db_approval)
                    await session.flush()
                logger.debug(f"Saved approval {approval_id} to PostgreSQL")
        except Exception as e:
            logger.warning(f"Failed to save approval {approval_id} to PostgreSQL: {e}")

        # Cache in Redis (write-through, 7-day TTL)
        await self._cache_approval_in_redis(approval)

        logger.info(
            f"Created approval request {approval_id} for job {job_id}, agent {agent_name}"
        )

        return approval

    async def _cache_approval_in_redis(self, approval: ApprovalRequest):
        """Write-through cache: store approval in Redis with 7-day TTL."""
        try:
            approval_key = f"{APPROVAL_KEY_PREFIX}{approval.id}"
            approval_json = approval.model_dump_json()

            async def save_operation(redis_client: redis.Redis):
                await redis_client.setex(approval_key, 604800, approval_json)

            await self._redis_manager.execute(save_operation)
            logger.debug(f"Cached approval {approval.id} in Redis")
        except Exception as e:
            logger.warning(f"Failed to cache approval {approval.id} in Redis: {e}")

    async def _load_approval_from_redis_cache(
        self, approval_id: str
    ) -> Optional[ApprovalRequest]:
        """Load approval from Redis cache. Returns None on miss or error."""
        approval_key = f"{APPROVAL_KEY_PREFIX}{approval_id}"
        try:

            async def get_operation(redis_client: redis.Redis):
                return await redis_client.get(approval_key)

            approval_json = await self._redis_manager.execute(get_operation)
            if not approval_json:
                return None
            return ApprovalRequest.model_validate_json(approval_json)
        except Exception as e:
            logger.debug(f"Redis cache miss for approval {approval_id}: {e}")
            return None

    async def _load_approval_from_db(
        self, approval_id: str
    ) -> Optional[ApprovalRequest]:
        """Load approval from PostgreSQL. Returns None if not found."""
        try:
            db_manager = get_database_manager()
            if not db_manager.is_initialized:
                return None
            async with db_manager.get_session() as session:
                result = await session.execute(
                    select(ApprovalModel).where(
                        ApprovalModel.approval_id == approval_id
                    )
                )
                db_approval = result.scalar_one_or_none()
                if db_approval:
                    return db_approval.to_approval_request()
        except Exception as e:
            logger.warning(
                f"Failed to load approval {approval_id} from PostgreSQL: {e}"
            )
        return None

    async def _load_all_approvals_from_redis(self):
        """Deprecated: kept for backward compatibility. Use list_approvals() instead."""
        pass  # No-op — approvals now live in PostgreSQL

    async def get_approval(self, approval_id: str) -> Optional[ApprovalRequest]:
        """Get an approval request by ID. Checks Redis cache first, then PostgreSQL."""
        # Fast path: Redis cache
        approval = await self._load_approval_from_redis_cache(approval_id)
        if approval:
            return approval

        # Cache miss: load from PostgreSQL and backfill cache
        approval = await self._load_approval_from_db(approval_id)
        if approval:
            await self._cache_approval_in_redis(approval)
        return approval

    async def list_approvals(
        self,
        job_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ApprovalRequest]:
        """
        List approval requests with optional filters. Reads from PostgreSQL.

        Args:
            job_id: Filter by job ID
            status: Filter by status (pending, approved, rejected, modified)

        Returns:
            List of approval requests, sorted newest first
        """
        try:
            db_manager = get_database_manager()
            if not db_manager.is_initialized:
                return []

            async with db_manager.get_session() as session:
                stmt = select(ApprovalModel).order_by(ApprovalModel.created_at.desc())
                if job_id:
                    stmt = stmt.where(ApprovalModel.job_id == job_id)
                if status:
                    stmt = stmt.where(ApprovalModel.status == status)
                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [row.to_approval_request() for row in rows]
        except Exception as e:
            logger.error(f"Failed to list approvals from PostgreSQL: {e}")
            return []

    async def decide_approval(
        self,
        approval_id: str,
        decision: ApprovalDecisionRequest,
    ) -> ApprovalRequest:
        """
        Make a decision on an approval request.

        Updates PostgreSQL (source of truth), refreshes Redis cache, and publishes
        a pub/sub notification so any waiting `wait_for_approval` calls unblock.

        Args:
            approval_id: ID of approval request
            decision: Approval decision details

        Returns:
            Updated approval request

        Raises:
            ValueError: If approval not found or invalid decision
        """
        approval = await self.get_approval(approval_id)
        if not approval:
            raise ValueError(f"Approval {approval_id} not found")

        if approval.status != "pending":
            raise ValueError(
                f"Approval {approval_id} already decided: {approval.status}"
            )

        # Handle keyword selection for seo_keywords step
        keyword_selection_used = False
        if approval.pipeline_step == "seo_keywords":
            # Allow keyword selection, approval, or rejection
            if decision.selected_keywords or decision.main_keyword:
                # Use main_keyword from decision if provided, otherwise from selected_keywords
                main_keyword = (
                    decision.main_keyword
                    or decision.selected_keywords.get("main_keyword")
                    if decision.selected_keywords
                    else None
                )

                # Filter keywords based on selection
                filtered_output = self.filter_selected_keywords(
                    approval.output_data,
                    decision.selected_keywords or {},
                    main_keyword=main_keyword,
                )
                approval.modified_output = filtered_output
                approval.status = "approved"  # Treating selection as approval
                keyword_selection_used = True
                logger.info(
                    f"Filtered keywords for approval {approval_id}: "
                    f"main_keyword={filtered_output.get('main_keyword', 'N/A')}, "
                    f"primary={len(filtered_output.get('primary_keywords', []))}, "
                    f"secondary={len(filtered_output.get('secondary_keywords', []))}, "
                    f"lsi={len(filtered_output.get('lsi_keywords', []))}"
                )
            else:
                # For seo_keywords without keyword selection, validate modify decision
                if decision.decision == "modify" and not decision.modified_output:
                    raise ValueError(
                        "seo_keywords step requires keyword selection or modified_output when using modify decision. "
                        "Please provide selected_keywords, modified_output, or use decision='approve'/'reject'."
                    )
        else:
            # Validate modified output if decision is modify (and not keyword selection)
            if (
                decision.decision == "modify"
                and not decision.modified_output
                and not decision.selected_keywords
            ):
                raise ValueError("Modified output required when decision is 'modify'")

        # Update approval status (skip if keyword selection was used, as status is already set)
        if not keyword_selection_used:
            if decision.decision == "approve":
                approval.status = "approved"
            elif decision.decision == "reject":
                approval.status = "rejected"
            elif decision.decision == "modify":
                approval.status = "modified"
                approval.modified_output = decision.modified_output
            elif decision.decision == "rerun":
                approval.status = "rerun"
                # Store comment as user_guidance for the rerun (will be used in API handler)

        approval.reviewed_at = datetime.now(timezone.utc)
        approval.user_comment = decision.comment
        approval.reviewed_by = decision.reviewed_by

        # Persist decision to PostgreSQL (source of truth)
        try:
            db_manager = get_database_manager()
            if db_manager.is_initialized:
                async with db_manager.get_session() as session:
                    await session.execute(
                        update(ApprovalModel)
                        .where(ApprovalModel.approval_id == approval_id)
                        .values(
                            status=approval.status,
                            modified_output=approval.modified_output,
                            user_comment=approval.user_comment,
                            reviewed_by=approval.reviewed_by,
                            reviewed_at=approval.reviewed_at,
                            retry_job_id=approval.retry_job_id,
                            retry_count=approval.retry_count,
                        )
                    )
        except Exception as e:
            logger.error(
                f"Failed to persist decision for approval {approval_id} to PostgreSQL: {e}"
            )
            raise

        # Update Redis cache
        await self._cache_approval_in_redis(approval)

        # Notify any waiting cross-process listeners via pub/sub
        try:

            async def publish_operation(redis_client: redis.Redis):
                await redis_client.publish(
                    f"approval:decided:{approval_id}", approval.status
                )

            await self._redis_manager.execute(publish_operation)
        except Exception as e:
            logger.warning(f"Failed to publish approval decision notification: {e}")

        logger.info(f"Approval {approval_id} decided: {decision.decision}")
        return approval

    async def execute_rejection_with_retry(
        self,
        approval: "ApprovalRequest",
        job_manager,
        user_comment: str = "",
        reviewed_by: str = "system",
    ) -> Optional[str]:
        """
        Create a retry job for a rejected or expired approval.

        Extracted from the API layer so that both the decide_approval endpoint and
        the expire_stale_approvals cron produce identical outcomes on rejection.

        Args:
            approval: The ApprovalRequest being rejected
            job_manager: JobManager instance for job creation/queuing
            user_comment: User or system comment to pass as guidance to the retry worker
            reviewed_by: Who/what made the decision (username or "system")

        Returns:
            retry_job_id if a retry job was created, None otherwise
        """
        from marketing_project.models.user_context import UserContext

        pipeline_step = approval.pipeline_step
        input_data = approval.input_data

        # Get accumulated context from Redis so the retry can pick up where it left off
        context_data = await self.load_pipeline_context(approval.job_id)
        context = context_data.get("context", {}) if context_data else {}

        # Load the waiting job to inherit user_id, session_id, and content_id
        approval_job = await job_manager.get_job(approval.job_id)
        session_id = None
        if approval_job and approval_job.metadata:
            session_id = approval_job.metadata.get("session_id")
            if not session_id:
                original_job_id = approval_job.metadata.get("original_job_id")
                if original_job_id:
                    root_job = await job_manager.get_job(original_job_id)
                    if root_job and root_job.metadata:
                        session_id = root_job.metadata.get("session_id")

        parent_user_id = approval_job.user_id if approval_job else None
        if not parent_user_id and approval_job and approval_job.metadata:
            original_job_id = approval_job.metadata.get("original_job_id")
            if original_job_id:
                root_job = await job_manager.get_job(original_job_id)
                parent_user_id = root_job.user_id if root_job else None

        parent_user_context = None
        source_job = approval_job
        if source_job and source_job.metadata and parent_user_id:
            triggered_by = source_job.metadata.get("triggered_by_user_id")
            if triggered_by == parent_user_id:
                parent_user_context = UserContext(
                    user_id=parent_user_id,
                    username=source_job.metadata.get("triggered_by_username"),
                    email=source_job.metadata.get("triggered_by_email"),
                )

        retry_job_id = str(uuid.uuid4())
        retry_content_id = approval_job.content_id if approval_job else approval.job_id
        retry_metadata = {
            "original_job_id": approval.job_id,
            "approval_id": approval.id,
            "step_name": pipeline_step,
            "retry_attempt": approval.retry_count + 1,
        }
        if session_id:
            retry_metadata["session_id"] = session_id

        await job_manager.create_job(
            job_id=retry_job_id,
            job_type=f"retry_step_{pipeline_step}",
            content_id=retry_content_id,
            metadata=retry_metadata,
            user_id=parent_user_id,
            user_context=parent_user_context,
        )

        # Link the retry job onto the waiting job so get_job_chain can find it
        if approval_job:
            approval_job.metadata["retry_job_id"] = retry_job_id
            await job_manager._save_job(approval_job)

        await job_manager.submit_to_arq(
            retry_job_id,
            "retry_step_job",
            pipeline_step,
            input_data,
            context,
            retry_job_id,
            approval.id,
            user_guidance=user_comment,
        )

        # Update approval with retry linkage
        approval.retry_job_id = retry_job_id
        approval.retry_count += 1
        # Persist retry linkage to PostgreSQL
        try:
            db_manager = get_database_manager()
            if db_manager.is_initialized:
                async with db_manager.get_session() as session:
                    await session.execute(
                        update(ApprovalModel)
                        .where(ApprovalModel.approval_id == approval.id)
                        .values(
                            retry_job_id=retry_job_id,
                            retry_count=approval.retry_count,
                        )
                    )
        except Exception as e:
            logger.warning(
                f"Failed to update retry linkage for approval {approval.id}: {e}"
            )
        await self._cache_approval_in_redis(approval)

        logger.info(
            f"Created retry job {retry_job_id} for approval {approval.id} "
            f"(step: {pipeline_step}, attempt: {approval.retry_count}, by: {reviewed_by})"
        )
        return retry_job_id

    def filter_selected_keywords(
        self,
        output_data: Dict,
        selected_keywords: Dict[str, List[str]],
        main_keyword: Optional[str] = None,
    ) -> Dict:
        """
        Filter SEO keywords based on user selection.

        Args:
            output_data: Original SEOKeywordsResult output_data
            selected_keywords: Dict with 'primary', 'secondary', 'lsi' keyword lists
            main_keyword: Main keyword selection (required for seo_keywords step)

        Returns:
            Filtered dict maintaining SEOKeywordsResult structure

        Raises:
            ValueError: If selected keywords don't exist in original data or no keywords selected
        """
        # Extract main_keyword from selected_keywords if not provided directly
        if not main_keyword and selected_keywords:
            main_keyword = selected_keywords.get("main_keyword")

        # Validate main_keyword is required
        if not main_keyword:
            raise ValueError(
                "Main keyword is required. Please select one keyword as the main focus."
            )

        # Check if main_keyword exists in any category (allows promotion from any category)
        original_primary = output_data.get("primary_keywords", [])
        original_secondary = output_data.get("secondary_keywords", []) or []
        original_lsi = output_data.get("lsi_keywords", []) or []
        original_long_tail = output_data.get("long_tail_keywords", []) or []

        # Allow promotion from any category
        if (
            main_keyword not in original_primary
            and main_keyword not in original_secondary
            and main_keyword not in original_lsi
            and main_keyword not in original_long_tail
        ):
            raise ValueError(
                f"Selected main keyword '{main_keyword}' must exist in one of the original keyword categories "
                f"(primary, secondary, LSI, or long-tail)."
            )

        # Validate at least one keyword is selected across all types
        total_selected = (
            len(selected_keywords.get("primary", []))
            + len(selected_keywords.get("secondary", []))
            + len(selected_keywords.get("lsi", []))
        )
        if total_selected == 0 and not main_keyword:
            raise ValueError(
                "At least one keyword must be selected across all types (primary, secondary, LSI)"
            )

        filtered = output_data.copy()

        # Set main_keyword first
        filtered["main_keyword"] = main_keyword

        # Filter primary keywords
        if "primary" in selected_keywords:
            selected_primary = selected_keywords["primary"]
            # Ensure main_keyword is included in primary keywords (even if promoted from another category)
            if main_keyword not in selected_primary:
                selected_primary = [main_keyword] + [
                    kw for kw in selected_primary if kw != main_keyword
                ]
            # Validate all selected keywords exist (allow main_keyword even if promoted)
            invalid = [
                kw
                for kw in selected_primary
                if kw != main_keyword and kw not in original_primary
            ]
            if invalid:
                raise ValueError(
                    f"Selected primary keywords not found in original: {invalid}"
                )
            filtered["primary_keywords"] = selected_primary
        else:
            # If no primary selection, ensure main_keyword is in the list (even if promoted)
            if main_keyword not in filtered.get("primary_keywords", []):
                filtered["primary_keywords"] = [main_keyword] + [
                    kw
                    for kw in filtered.get("primary_keywords", [])
                    if kw != main_keyword
                ]

        # Filter secondary keywords (exclude main_keyword if it was promoted from here)
        if "secondary" in selected_keywords:
            original_secondary = output_data.get("secondary_keywords", []) or []
            selected_secondary = selected_keywords["secondary"]
            # Remove main_keyword from secondary if it was promoted (should not be in both)
            selected_secondary = [kw for kw in selected_secondary if kw != main_keyword]
            # Validate all selected keywords exist
            invalid = [kw for kw in selected_secondary if kw not in original_secondary]
            if invalid:
                raise ValueError(
                    f"Selected secondary keywords not found in original: {invalid}"
                )
            filtered["secondary_keywords"] = (
                selected_secondary if selected_secondary else None
            )
        else:
            # Remove main_keyword from secondary if it was promoted
            if main_keyword in filtered.get("secondary_keywords", []):
                filtered["secondary_keywords"] = [
                    kw
                    for kw in filtered.get("secondary_keywords", [])
                    if kw != main_keyword
                ]

        # Filter LSI keywords (exclude main_keyword if it was promoted from here)
        if "lsi" in selected_keywords:
            original_lsi = output_data.get("lsi_keywords", []) or []
            selected_lsi = selected_keywords["lsi"]
            # Remove main_keyword from LSI if it was promoted (should not be in both)
            selected_lsi = [kw for kw in selected_lsi if kw != main_keyword]
            # Validate all selected keywords exist
            invalid = [kw for kw in selected_lsi if kw not in original_lsi]
            if invalid:
                raise ValueError(
                    f"Selected LSI keywords not found in original: {invalid}"
                )
            filtered["lsi_keywords"] = selected_lsi if selected_lsi else None
        else:
            # Remove main_keyword from LSI if it was promoted
            if main_keyword in filtered.get("lsi_keywords", []):
                filtered["lsi_keywords"] = [
                    kw for kw in filtered.get("lsi_keywords", []) if kw != main_keyword
                ]

        # Handle long_tail_keywords if present in selection (exclude main_keyword if it was promoted from here)
        if "long_tail" in selected_keywords:
            original_long_tail = output_data.get("long_tail_keywords", []) or []
            selected_long_tail = selected_keywords["long_tail"]
            # Remove main_keyword from long_tail if it was promoted (should not be in both)
            selected_long_tail = [kw for kw in selected_long_tail if kw != main_keyword]
            # Validate all selected keywords exist
            invalid = [kw for kw in selected_long_tail if kw not in original_long_tail]
            if invalid:
                raise ValueError(
                    f"Selected long-tail keywords not found in original: {invalid}"
                )
            filtered["long_tail_keywords"] = (
                selected_long_tail if selected_long_tail else None
            )
        else:
            # Remove main_keyword from long_tail if it was promoted
            if main_keyword in filtered.get("long_tail_keywords", []):
                filtered["long_tail_keywords"] = [
                    kw
                    for kw in filtered.get("long_tail_keywords", [])
                    if kw != main_keyword
                ]

        # Filter metadata to match selected keywords
        # Primary keywords metadata
        if "primary_keywords_metadata" in output_data and filtered.get(
            "primary_keywords"
        ):
            original_metadata = output_data.get("primary_keywords_metadata", [])
            if isinstance(original_metadata, list):
                selected_primary_set = set(filtered["primary_keywords"])
                filtered["primary_keywords_metadata"] = (
                    [
                        meta
                        for meta in original_metadata
                        if isinstance(meta, dict)
                        and meta.get("keyword") in selected_primary_set
                    ]
                    if original_metadata
                    else None
                )

        # Secondary keywords metadata
        if "secondary_keywords_metadata" in output_data and filtered.get(
            "secondary_keywords"
        ):
            original_metadata = output_data.get("secondary_keywords_metadata", [])
            if isinstance(original_metadata, list):
                selected_secondary_set = set(filtered["secondary_keywords"])
                filtered["secondary_keywords_metadata"] = (
                    [
                        meta
                        for meta in original_metadata
                        if isinstance(meta, dict)
                        and meta.get("keyword") in selected_secondary_set
                    ]
                    if original_metadata
                    else None
                )

        # Long-tail keywords metadata
        if "long_tail_keywords_metadata" in output_data and filtered.get(
            "long_tail_keywords"
        ):
            original_metadata = output_data.get("long_tail_keywords_metadata", [])
            if isinstance(original_metadata, list):
                selected_long_tail_set = set(filtered["long_tail_keywords"])
                filtered["long_tail_keywords_metadata"] = (
                    [
                        meta
                        for meta in original_metadata
                        if isinstance(meta, dict)
                        and meta.get("keyword") in selected_long_tail_set
                    ]
                    if original_metadata
                    else None
                )

        # Filter keyword_difficulty Dict to match selected keywords
        if "keyword_difficulty" in output_data:
            original_difficulty = output_data.get("keyword_difficulty")
            if isinstance(original_difficulty, dict):
                # Filter to only include selected keywords
                all_selected = set(filtered.get("primary_keywords", []))
                if filtered.get("secondary_keywords"):
                    all_selected.update(filtered["secondary_keywords"])
                if filtered.get("long_tail_keywords"):
                    all_selected.update(filtered["long_tail_keywords"])

                filtered["keyword_difficulty"] = (
                    {
                        kw: score
                        for kw, score in original_difficulty.items()
                        if kw in all_selected
                    }
                    if original_difficulty
                    else None
                )
            # If it's a string (old format), keep it as is for backward compatibility
            elif isinstance(original_difficulty, str):
                filtered["keyword_difficulty"] = original_difficulty

        # Filter keyword_density_analysis to match selected keywords
        if "keyword_density_analysis" in output_data:
            original_density_analysis = output_data.get("keyword_density_analysis", [])
            if isinstance(original_density_analysis, list):
                all_selected = set(filtered.get("primary_keywords", []))
                if filtered.get("secondary_keywords"):
                    all_selected.update(filtered["secondary_keywords"])

                filtered["keyword_density_analysis"] = (
                    [
                        analysis
                        for analysis in original_density_analysis
                        if isinstance(analysis, dict)
                        and analysis.get("keyword") in all_selected
                    ]
                    if original_density_analysis
                    else None
                )

        # Also update legacy keyword_density for backward compatibility
        if "keyword_density" in output_data and filtered.get("primary_keywords"):
            original_density = output_data.get("keyword_density")
            if isinstance(original_density, dict):
                all_selected = set(filtered.get("primary_keywords", []))
                if filtered.get("secondary_keywords"):
                    all_selected.update(filtered["secondary_keywords"])

                filtered["keyword_density"] = (
                    {
                        kw: density
                        for kw, density in original_density.items()
                        if kw in all_selected
                    }
                    if original_density
                    else None
                )

        # Filter keyword_clusters to only include clusters with selected keywords
        if "keyword_clusters" in output_data:
            original_clusters = output_data.get("keyword_clusters", [])
            if isinstance(original_clusters, list):
                all_selected = set(filtered.get("primary_keywords", []))
                if filtered.get("secondary_keywords"):
                    all_selected.update(filtered["secondary_keywords"])
                if filtered.get("lsi_keywords"):
                    all_selected.update(filtered["lsi_keywords"])
                if filtered.get("long_tail_keywords"):
                    all_selected.update(filtered["long_tail_keywords"])

                filtered["keyword_clusters"] = (
                    [
                        cluster
                        for cluster in original_clusters
                        if isinstance(cluster, dict)
                        and any(
                            kw in all_selected for kw in cluster.get("keywords", [])
                        )
                    ]
                    if original_clusters
                    else None
                )

        return filtered

    async def wait_for_approval(
        self,
        approval_id: str,
        timeout: Optional[float] = None,
    ) -> ApprovalRequest:
        """
        Wait for an approval to be decided.

        Uses Redis pub/sub so this works correctly across multiple API processes.

        Args:
            approval_id: ID of approval request
            timeout: Optional timeout in seconds (auto-rejects on expiry)

        Returns:
            Approved/rejected/modified approval request

        Raises:
            ValueError: If approval not found
        """
        approval = await self.get_approval(approval_id)
        if not approval:
            raise ValueError(f"Approval {approval_id} not found")

        # Already decided — return immediately
        if approval.status != "pending":
            return approval

        # Subscribe to the decision channel and block until notified
        channel = f"approval:decided:{approval_id}"
        deadline = asyncio.get_event_loop().time() + (timeout or 86400)

        redis_client = await self.get_redis()
        if redis_client is None:
            # No Redis — fall back to DB polling every second
            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(1.0)
                approval = await self.get_approval(approval_id)
                if approval and approval.status != "pending":
                    return approval
        else:
            pubsub = redis_client.pubsub()
            try:
                await pubsub.subscribe(channel)
                while True:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        msg = await asyncio.wait_for(
                            pubsub.get_message(ignore_subscribe_messages=True),
                            timeout=min(remaining, 1.0),
                        )
                    except asyncio.TimeoutError:
                        msg = None
                    if msg:
                        # Decision was made — reload from DB for fresh state
                        approval = await self.get_approval(approval_id)
                        if approval:
                            return approval
            finally:
                try:
                    await pubsub.unsubscribe(channel)
                    await pubsub.close()
                except Exception:
                    pass

        # Timeout reached — auto-reject
        logger.warning(
            f"Approval {approval_id} timed out after {timeout}s, auto-rejecting"
        )
        decision = ApprovalDecisionRequest(
            decision="reject",
            comment=f"Auto-rejected after {timeout}s timeout",
        )
        return await self.decide_approval(approval_id, decision)

    async def get_stats(self) -> ApprovalStats:
        """Get approval statistics from PostgreSQL (always accurate, no stale data)."""
        try:
            db_manager = get_database_manager()
            if db_manager.is_initialized:
                async with db_manager.get_session() as session:
                    # Single aggregate query for counts
                    count_result = await session.execute(
                        select(
                            ApprovalModel.status, func.count(ApprovalModel.id)
                        ).group_by(ApprovalModel.status)
                    )
                    counts = {row[0]: row[1] for row in count_result.fetchall()}

                    # Average review time for decided approvals
                    reviewed_result = await session.execute(
                        select(
                            ApprovalModel.created_at, ApprovalModel.reviewed_at
                        ).where(ApprovalModel.reviewed_at.isnot(None))
                    )
                    reviewed_rows = reviewed_result.fetchall()
                    avg_review_time = None
                    if reviewed_rows:
                        times = [
                            (r.reviewed_at - r.created_at).total_seconds()
                            for r in reviewed_rows
                            if r.reviewed_at and r.created_at
                        ]
                        if times:
                            avg_review_time = sum(times) / len(times)

                    total = sum(counts.values())
                    pending = counts.get("pending", 0)
                    approved = counts.get("approved", 0)
                    rejected = counts.get("rejected", 0)
                    modified = counts.get("modified", 0)
                    rerun = counts.get("rerun", 0)
                    decided = approved + rejected + modified + rerun
                    approval_rate = (
                        (approved + modified) / decided if decided > 0 else 0.0
                    )

                    return ApprovalStats(
                        total_requests=total,
                        pending=pending,
                        approved=approved,
                        rejected=rejected,
                        modified=modified,
                        rerun=rerun,
                        avg_review_time_seconds=avg_review_time,
                        approval_rate=approval_rate,
                    )
        except Exception as e:
            logger.error(f"Failed to compute approval stats from PostgreSQL: {e}")

        return ApprovalStats(
            total_requests=0,
            pending=0,
            approved=0,
            rejected=0,
            modified=0,
            rerun=0,
            avg_review_time_seconds=None,
            approval_rate=0.0,
        )

    async def delete_all_approvals(self) -> int:
        """
        Delete all approvals from PostgreSQL and Redis cache.

        This will:
        - Count and DELETE all rows from the approvals table
        - Scan and delete all approval:request:* Redis cache keys
        - Delete all pipeline contexts (pipeline:context:*)

        Returns:
            Number of approvals deleted from PostgreSQL
        """
        deleted_count = 0
        try:
            # Delete from PostgreSQL (source of truth)
            db_manager = get_database_manager()
            if db_manager.is_initialized:
                async with db_manager.get_session() as session:
                    count_result = await session.execute(
                        select(func.count(ApprovalModel.id))
                    )
                    deleted_count = count_result.scalar() or 0
                    await session.execute(delete(ApprovalModel))
                logger.info(f"Deleted {deleted_count} approvals from PostgreSQL")
        except Exception as e:
            logger.error(
                f"Failed to delete approvals from PostgreSQL: {e}", exc_info=True
            )
            raise

        # Clear Redis cache keys (approval:request:*, pipeline:context:*)
        try:

            async def scan_and_delete(redis_client: redis.Redis, pattern: str):
                keys = []
                async for key in redis_client.scan_iter(match=pattern):
                    keys.append(key)
                if keys:
                    batch_size = 100
                    for i in range(0, len(keys), batch_size):
                        await redis_client.delete(*keys[i : i + batch_size])
                return len(keys)

            async def cleanup_redis(redis_client: redis.Redis):
                approval_keys = await scan_and_delete(
                    redis_client, f"{APPROVAL_KEY_PREFIX}*"
                )
                context_keys = await scan_and_delete(
                    redis_client, f"{PIPELINE_CONTEXT_KEY_PREFIX}*"
                )
                logger.info(
                    f"Deleted {approval_keys} approval cache keys and "
                    f"{context_keys} pipeline contexts from Redis"
                )

            await self._redis_manager.execute(cleanup_redis)
        except Exception as e:
            logger.warning(f"Failed to clear approval Redis cache keys: {e}")

        logger.info(f"delete_all_approvals: total deleted = {deleted_count}")
        return deleted_count

    async def cleanup(self):
        """Cleanup Redis connections."""
        # RedisManager cleanup is handled globally
        # This method is here for consistency with other managers
        pass


# Global approval manager instance
_approval_manager: Optional[ApprovalManager] = None


async def get_approval_manager(
    settings: Optional[ApprovalSettings] = None, reload_from_db: bool = True
) -> ApprovalManager:
    """
    Get or create the global approval manager instance.

    Args:
        settings: Optional settings to use (only used if manager doesn't exist)
        reload_from_db: If True, reload settings from database/Redis if manager already exists

    Returns:
        ApprovalManager instance
    """
    global _approval_manager
    if _approval_manager is None:
        # Try to load from database first (source of truth), fallback to Redis
        temp_manager = ApprovalManager()
        loaded_settings = await temp_manager.load_settings()
        if loaded_settings:
            _approval_manager = ApprovalManager(loaded_settings)
            logger.info("Loaded approval settings from database/Redis")
        else:
            _approval_manager = ApprovalManager(settings)
            logger.info(
                "Using default or provided approval settings (database/Redis not available or empty)"
            )
    elif reload_from_db:
        # Reload settings from database/Redis to ensure we have the latest
        loaded_settings = await _approval_manager.load_settings()
        if loaded_settings:
            _approval_manager.settings = loaded_settings
            logger.debug("Reloaded approval settings from database/Redis")
    return _approval_manager


def get_approval_manager_sync() -> ApprovalManager:
    """
    Synchronous version for backwards compatibility.

    Note: This will not reload from Redis. Use async version when possible.
    """
    global _approval_manager
    if _approval_manager is None:
        _approval_manager = ApprovalManager()
    return _approval_manager


async def set_approval_settings(settings: ApprovalSettings):
    """Update approval settings and persist to database (source of truth) and Redis cache."""
    manager = await get_approval_manager(reload_from_db=False)
    manager.settings = settings
    await manager.save_settings(settings)
    logger.info(
        f"Updated approval settings: require_approval={settings.require_approval}, agents={settings.approval_agents}"
    )


def set_approval_settings_sync(settings: ApprovalSettings):
    """
    Synchronous version for backwards compatibility.

    Creates an event loop if needed. Prefer async version.
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, we need to use a different approach
            # Schedule the coroutine
            asyncio.create_task(set_approval_settings(settings))
        else:
            loop.run_until_complete(set_approval_settings(settings))
    except RuntimeError:
        # No event loop, create one
        asyncio.run(set_approval_settings(settings))
