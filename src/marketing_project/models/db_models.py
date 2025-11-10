"""
SQLAlchemy Database Models.

Defines database tables for configuration storage.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
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
