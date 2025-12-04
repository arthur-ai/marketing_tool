"""
SQLAlchemy Database Models.

Defines database tables for configuration storage.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, Column, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from marketing_project.services.database import Base


class ApprovalSettingsModel(Base):
    """Database model for approval settings."""

    __tablename__ = "approval_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    require_approval = Column(Boolean, default=False, nullable=False)
    approval_agents = Column(JSONB, nullable=False)  # List of strings
    auto_approve_threshold = Column(String, nullable=True)  # Optional float as string
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


class DesignKitConfigModel(Base):
    """Database model for design kit configuration."""

    __tablename__ = "design_kit_config"

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
    )

    def to_dict(self) -> Dict[str, Any]:
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
