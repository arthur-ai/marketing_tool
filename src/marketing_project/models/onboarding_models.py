"""
Pydantic schemas for onboarding example (quick-start template) API endpoints.
"""

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class OnboardingExampleResponse(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    job_type: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    display_order: int = 0
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


_INPUT_DATA_MAX_BYTES = 65_536  # 64 KB


def _validate_input_data_size(v: Dict[str, Any]) -> Dict[str, Any]:
    if len(json.dumps(v)) > _INPUT_DATA_MAX_BYTES:
        raise ValueError(
            f"input_data must not exceed {_INPUT_DATA_MAX_BYTES} bytes when serialized"
        )
    return v


class OnboardingExampleCreateRequest(BaseModel):
    title: str = Field(..., max_length=200)
    description: Optional[str] = None
    job_type: str = Field(..., max_length=100)
    input_data: Dict[str, Any] = Field(default_factory=dict)
    display_order: int = 0
    is_active: bool = True

    @field_validator("input_data")
    @classmethod
    def input_data_size(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        return _validate_input_data_size(v)


class OnboardingExampleUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = None
    job_type: Optional[str] = Field(None, max_length=100)
    input_data: Optional[Dict[str, Any]] = None
    display_order: Optional[int] = None
    is_active: Optional[bool] = None

    @field_validator("input_data")
    @classmethod
    def input_data_size(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is not None:
            return _validate_input_data_size(v)
        return v


class OnboardingExamplesListResponse(BaseModel):
    examples: List[OnboardingExampleResponse]
    total: int
