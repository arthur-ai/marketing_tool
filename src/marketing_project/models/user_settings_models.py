"""
Per-user settings Pydantic models.

Users can override global pipeline and approval settings. Any field left as
None falls back to the active global default at job creation time.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, validator

# Valid pipeline step names (from config/pipeline.yml)
VALID_PIPELINE_STEPS = [
    "AnalyzeContent",
    "ExtractSEOKeywords",
    "GenerateMarketingBrief",
    "GenerateArticle",
    "OptimizeSEO",
    "suggested_links",
    "FormatContent",
]

# Valid approval agent names (from approval_models.py defaults)
VALID_APPROVAL_AGENTS = [
    "content_pipeline",
    "article_generation",
    "marketing_brief",
    "seo_keywords",
    "seo_optimization",
    "content_formatting",
    "transcript_preprocessing_approval",
    "blog_post_preprocessing_approval",
    "suggested_links",
    "social_media_marketing_brief",
    "social_media_angle_hook",
    "social_media_post_generation",
]

VALID_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-5.1",
    "o1-preview",
    "o1-mini",
]


class UserSettings(BaseModel):
    """
    Per-user settings that override global pipeline and approval defaults.

    All fields are optional. A None value means "use the global default".
    Set a field explicitly to override the global setting for this user's jobs.
    """

    # Pipeline step preferences
    disabled_steps: Optional[List[str]] = Field(
        None,
        description=(
            "Pipeline steps to skip for this user's jobs. "
            f"Valid values: {VALID_PIPELINE_STEPS}"
        ),
    )

    # Approval preferences
    require_approval: Optional[bool] = Field(
        None,
        description="Whether pipeline steps require human approval. None = use global setting.",
    )
    approval_agents: Optional[List[str]] = Field(
        None,
        description=(
            "Specific agents that require approval for this user's jobs. "
            "None = use global setting. "
            f"Valid values: {VALID_APPROVAL_AGENTS}"
        ),
    )
    auto_approve_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score threshold (0.0–1.0) above which steps are auto-approved.",
    )
    approval_timeout_seconds: Optional[int] = Field(
        None,
        gt=0,
        description="Seconds before a pending approval is auto-rejected. None = no timeout.",
    )

    # LLM preferences
    preferred_model: Optional[str] = Field(
        None,
        description=f"OpenAI model to use for this user's jobs. Valid values: {VALID_MODELS}",
    )
    preferred_temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0.0–2.0). None = use global default.",
    )

    @validator("disabled_steps")
    def validate_disabled_steps(cls, v):
        if v is not None:
            invalid = set(v) - set(VALID_PIPELINE_STEPS)
            if invalid:
                raise ValueError(
                    f"Unknown pipeline step(s): {sorted(invalid)}. "
                    f"Valid steps are: {VALID_PIPELINE_STEPS}"
                )
        return v

    @validator("approval_agents")
    def validate_approval_agents(cls, v):
        if v is not None:
            invalid = set(v) - set(VALID_APPROVAL_AGENTS)
            if invalid:
                raise ValueError(
                    f"Unknown approval agent(s): {sorted(invalid)}. "
                    f"Valid agents are: {VALID_APPROVAL_AGENTS}"
                )
        return v

    @validator("preferred_model")
    def validate_preferred_model(cls, v):
        if v is not None and v not in VALID_MODELS:
            raise ValueError(f"Invalid model '{v}'. Must be one of: {VALID_MODELS}")
        return v


class UserSettingsResponse(UserSettings):
    """UserSettings with server-populated metadata fields."""

    user_id: str
    created_at: datetime
    updated_at: datetime


class ResolvedUserSettings(BaseModel):
    """
    Fully-resolved settings for a specific user, stored in job.metadata['user_settings'].

    This is the merged result of global defaults + user overrides, computed
    once at job creation time so the ARQ worker uses a stable snapshot.
    """

    disabled_steps: List[str]
    require_approval: bool
    approval_agents: List[str]
    auto_approve_threshold: Optional[float]
    approval_timeout_seconds: Optional[int]
    preferred_model: str
    preferred_temperature: float
