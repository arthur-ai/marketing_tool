"""
SQLAlchemy Database Models.

Defines database tables for configuration storage.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from marketing_project.services.database import Base


class ApprovalSettingsModel(Base):
    """Database model for approval settings."""

    __tablename__ = "approval_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    require_approval = Column(Boolean, default=False, nullable=False)
    approval_agents = Column(JSONB, nullable=False)  # List of strings
    auto_approve_threshold = Column(Float, nullable=True)
    timeout_seconds = Column(Integer, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "require_approval": self.require_approval,
            "approval_agents": (
                self.approval_agents
                if isinstance(self.approval_agents, list)
                else (
                    json.loads(self.approval_agents)
                    if isinstance(self.approval_agents, str)
                    else []
                )
            ),
            "auto_approve_threshold": (
                float(self.auto_approve_threshold)
                if self.auto_approve_threshold
                else None
            ),
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class BrandKitConfigModel(Base):
    """Database model for brand kit configuration."""

    __tablename__ = "design_kit_config"  # Table name kept for backward compatibility

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String, nullable=False, unique=True, index=True)
    config_data = Column(JSONB, nullable=False)  # Full DesignKitConfig as JSON
    is_active = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        config = (
            self.config_data
            if isinstance(self.config_data, dict)
            else (
                json.loads(self.config_data)
                if isinstance(self.config_data, str)
                else {}
            )
        )
        return {
            "version": self.version,
            "config_data": config,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# Backward-compatibility alias
DesignKitConfigModel = BrandKitConfigModel


class InternalDocsConfigModel(Base):
    """Database model for internal docs configuration."""

    __tablename__ = "internal_docs_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String, nullable=False, unique=True, index=True)
    config_data = Column(JSONB, nullable=False)  # Full InternalDocsConfig as JSON
    is_active = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        config = (
            self.config_data
            if isinstance(self.config_data, dict)
            else (
                json.loads(self.config_data)
                if isinstance(self.config_data, str)
                else {}
            )
        )
        return {
            "version": self.version,
            "config_data": config,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class PipelineSettingsModel(Base):
    """Database model for pipeline settings."""

    __tablename__ = "pipeline_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    settings_data = Column(JSONB, nullable=False)  # Full pipeline settings as JSON
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        settings = (
            self.settings_data
            if isinstance(self.settings_data, dict)
            else (
                json.loads(self.settings_data)
                if isinstance(self.settings_data, str)
                else {}
            )
        )
        return {
            "settings_data": settings,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class UserSettingsModel(Base):
    """Per-user pipeline and approval settings (overrides global defaults)."""

    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, unique=True, index=True)

    # Pipeline preferences (None = use global default)
    disabled_steps = Column(JSONB, nullable=True)  # List[str] | None

    # Approval preferences (None = use global default)
    require_approval = Column(Boolean, nullable=True)
    approval_agents = Column(JSONB, nullable=True)  # List[str] | None
    auto_approve_threshold = Column(Float, nullable=True)
    approval_timeout_seconds = Column(Integer, nullable=True)

    # LLM preferences (None = use global default)
    preferred_model = Column(String, nullable=True)
    preferred_temperature = Column(Float, nullable=True)

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "user_id": self.user_id,
            "disabled_steps": (
                self.disabled_steps
                if isinstance(self.disabled_steps, list)
                else (
                    json.loads(self.disabled_steps)
                    if isinstance(self.disabled_steps, str)
                    else None
                )
            ),
            "require_approval": self.require_approval,
            "approval_agents": (
                self.approval_agents
                if isinstance(self.approval_agents, list)
                else (
                    json.loads(self.approval_agents)
                    if isinstance(self.approval_agents, str)
                    else None
                )
            ),
            "auto_approve_threshold": (
                float(self.auto_approve_threshold)
                if self.auto_approve_threshold is not None
                else None
            ),
            "approval_timeout_seconds": self.approval_timeout_seconds,
            "preferred_model": self.preferred_model,
            "preferred_temperature": (
                float(self.preferred_temperature)
                if self.preferred_temperature is not None
                else None
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class JobModel(Base):
    """Database model for job storage and tracking."""

    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(
        String, nullable=False, unique=True, index=True
    )  # UUID job identifier
    job_type = Column(
        String, nullable=False, index=True
    )  # blog, release_notes, transcript, etc.
    status = Column(
        String, nullable=False, index=True
    )  # pending, queued, processing, completed, failed, etc.
    content_id = Column(String, nullable=False)  # Content identifier
    user_id = Column(
        String, nullable=True, index=True
    )  # User ID who triggered the job (nullable for backward compatibility)
    progress = Column(Integer, default=0, nullable=False)  # 0-100
    current_step = Column(String, nullable=True)  # Current step name
    result = Column(JSONB, nullable=True)  # Job result data
    error = Column(Text, nullable=True)  # Error message if failed
    job_metadata = Column(
        JSONB, nullable=False, default={}
    )  # Additional metadata (renamed from 'metadata' to avoid SQLAlchemy reserved name)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True, index=True)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Indexes for common queries
    __table_args__ = (
        Index("idx_jobs_status_created", "status", "created_at"),
        Index("idx_jobs_type_status", "job_type", "status"),
        Index("idx_jobs_user_id", "user_id"),
    )

    def to_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        """Convert model to dictionary matching Job Pydantic model."""
        # Parse status string to JobStatus enum value
        from marketing_project.services.job_manager import JobStatus

        try:
            status_enum = JobStatus(self.status)
        except ValueError:
            # If status doesn't match enum, use as-is (will be handled by Pydantic)
            status_enum = self.status

        return {
            "id": self.job_id,
            "type": self.job_type,
            "status": status_enum,
            "content_id": self.content_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "current_step": self.current_step,
            "result": (
                self.result
                if isinstance(self.result, dict)
                else (
                    json.loads(self.result)
                    if isinstance(self.result, str)
                    else self.result
                )
            ),
            "error": self.error,
            "metadata": (
                self.job_metadata
                if isinstance(self.job_metadata, dict)
                else (
                    json.loads(self.job_metadata)
                    if isinstance(self.job_metadata, str)
                    else {}
                )
            ),
        }


class ApprovalModel(Base):
    """Database model for approval requests (human-in-the-loop review)."""

    __tablename__ = "approvals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    approval_id = Column(String, nullable=False, unique=True, index=True)
    job_id = Column(String, nullable=False, index=True)
    agent_name = Column(String, nullable=False)
    step_name = Column(String, nullable=False)
    pipeline_step = Column(String, nullable=True)
    status = Column(String, nullable=False, default="pending", index=True)
    input_data = Column(JSONB, nullable=True)
    output_data = Column(JSONB, nullable=True)
    modified_output = Column(JSONB, nullable=True)
    confidence_score = Column(Float, nullable=True)
    user_comment = Column(Text, nullable=True)
    reviewed_by = Column(String, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    retry_job_id = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_approvals_job_id_status", "job_id", "status"),
        Index("idx_approvals_status_created", "status", "created_at"),
    )

    def to_approval_request(self):
        """Convert to ApprovalRequest Pydantic model."""
        from marketing_project.models.approval_models import ApprovalRequest

        return ApprovalRequest(
            id=self.approval_id,
            job_id=self.job_id,
            agent_name=self.agent_name,
            step_name=self.step_name,
            pipeline_step=self.pipeline_step,
            status=self.status,
            input_data=self.input_data or {},
            output_data=self.output_data or {},
            modified_output=self.modified_output,
            confidence_score=self.confidence_score,
            user_comment=self.user_comment,
            reviewed_by=self.reviewed_by,
            retry_count=self.retry_count or 0,
            retry_job_id=self.retry_job_id,
            created_at=self.created_at,
            reviewed_at=self.reviewed_at,
        )


class StepResultModel(Base):
    """Database model for pipeline step results."""

    __tablename__ = "step_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, nullable=False)  # Job that executed this step
    root_job_id = Column(String, nullable=False)  # Root job in the chain
    execution_context_id = Column(String, nullable=True)  # Resume cycle (0, 1, 2…)
    step_number = Column(Integer, nullable=False)
    relative_step_number = Column(Integer, nullable=True)  # Within context (1-indexed)
    step_name = Column(String, nullable=False)
    status = Column(
        String, nullable=False, default="success"
    )  # success / failed / skipped
    result = Column(JSONB, nullable=True)  # Full LLM output
    input_snapshot = Column(JSONB, nullable=True)
    context_keys_used = Column(JSONB, nullable=True)
    execution_time = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("job_id", "step_name", name="uq_step_results_job_step"),
        Index("idx_step_results_job_id", "job_id"),
        Index("idx_step_results_root_job_id", "root_job_id"),
    )
