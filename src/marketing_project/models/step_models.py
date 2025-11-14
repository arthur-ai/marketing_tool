"""
Step execution models for individual pipeline step triggers.

This module defines Pydantic models for executing individual pipeline steps.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StepExecutionRequest(BaseModel):
    """Request model for executing a single pipeline step."""

    content: Dict[str, Any] = Field(
        ..., description="Input content for the step (e.g., blog post, article)"
    )
    context: Dict[str, Any] = Field(
        ...,
        description="Context dictionary containing all required inputs for the step",
    )


class StepExecutionResponse(BaseModel):
    """Response model for step execution."""

    step_name: str = Field(..., description="Name of the executed step")
    job_id: str = Field(..., description="Job ID for tracking the execution")
    status: str = Field(..., description="Initial job status")
    message: str = Field(..., description="Response message")


class StepInfo(BaseModel):
    """Information about a pipeline step."""

    step_name: str = Field(..., description="Name of the step")
    step_number: int = Field(..., description="Step number in the pipeline")
    description: Optional[str] = Field(None, description="Step description")


class StepListResponse(BaseModel):
    """Response model for listing available pipeline steps."""

    steps: List[StepInfo] = Field(..., description="List of available pipeline steps")


class StepRequirementsResponse(BaseModel):
    """Response model for step requirements."""

    step_name: str = Field(..., description="Name of the step")
    step_number: int = Field(..., description="Step number in the pipeline")
    required_context_keys: List[str] = Field(
        ..., description="List of required context keys"
    )
    descriptions: Dict[str, str] = Field(
        ...,
        description="Human-readable descriptions for each required context key",
    )
