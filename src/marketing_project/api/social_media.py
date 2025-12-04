"""
API endpoints for social media post preview, validation, and editing.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from marketing_project.services.platform_error_handler import PlatformErrorHandler
from marketing_project.services.social_media_pipeline import SocialMediaPipeline

logger = logging.getLogger("marketing_project.api.social_media")

router = APIRouter(prefix="/v1/social-media", tags=["social-media"])


class PostPreviewRequest(BaseModel):
    """Request for post preview generation."""

    content: str = Field(..., description="Post content to preview")
    platform: str = Field(..., description="Platform: linkedin, hackernews, or email")
    email_type: Optional[str] = Field(
        None, description="Email type if platform is email"
    )


class PostPreviewResponse(BaseModel):
    """Response with formatted preview."""

    preview: str = Field(..., description="Formatted preview HTML")
    character_count: int = Field(..., description="Character count")
    word_count: int = Field(..., description="Word count")
    platform_limits: Dict[str, Any] = Field(..., description="Platform-specific limits")
    warnings: list[str] = Field(default_factory=list, description="Preview warnings")


class PostValidationRequest(BaseModel):
    """Request for post validation."""

    content: str = Field(..., description="Post content to validate")
    platform: str = Field(..., description="Platform: linkedin, hackernews, or email")
    email_type: Optional[str] = Field(
        None, description="Email type if platform is email"
    )
    subject_line: Optional[str] = Field(None, description="Email subject line")


class PostValidationResponse(BaseModel):
    """Response with validation results."""

    is_valid: bool = Field(..., description="Whether content is valid")
    errors: list[str] = Field(default_factory=list, description="Validation errors")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: list[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )
    auto_fix_available: bool = Field(False, description="Whether auto-fix is available")
    auto_fixed_content: Optional[str] = Field(
        None, description="Auto-fixed content if available"
    )


class PostUpdateRequest(BaseModel):
    """Request to update a post."""

    job_id: str = Field(..., description="Job ID of the post to update")
    content: str = Field(..., description="Updated post content")
    platform: str = Field(..., description="Platform: linkedin, hackernews, or email")
    email_type: Optional[str] = Field(
        None, description="Email type if platform is email"
    )
    subject_line: Optional[str] = Field(None, description="Updated email subject line")


class PostUpdateResponse(BaseModel):
    """Response after updating a post."""

    success: bool = Field(..., description="Whether update was successful")
    message: str = Field(..., description="Update message")
    updated_content: Optional[str] = Field(None, description="Updated content")


@router.post("/preview", response_model=PostPreviewResponse)
async def preview_post(request: PostPreviewRequest):
    """
    Generate a formatted preview of a social media post.

    Args:
        request: Preview request with content and platform

    Returns:
        Formatted preview with metadata
    """
    try:
        # Load platform config
        pipeline = SocialMediaPipeline()
        try:
            platform_config = await pipeline._load_platform_config()
        except Exception:
            platform_config = {}
        platform_spec = platform_config.get(request.platform, {})

        # Get platform limits
        limits = {
            "max_characters": platform_spec.get("max_characters", 3000),
            "recommended_characters": platform_spec.get("recommended_characters", 2000),
            "max_hashtags": platform_spec.get("max_hashtags", 5),
        }

        # Format preview (basic markdown to HTML)
        import markdown

        preview_html = markdown.markdown(request.content)

        # Check for warnings
        warnings = []
        char_count = len(request.content)
        if char_count > limits["max_characters"]:
            warnings.append(
                f"Content exceeds maximum length by {char_count - limits['max_characters']} characters"
            )
        elif char_count > limits["recommended_characters"]:
            warnings.append(
                f"Content exceeds recommended length by {char_count - limits['recommended_characters']} characters"
            )

        # Count hashtags
        import re

        hashtags = re.findall(r"#\w+", request.content)
        if len(hashtags) > limits["max_hashtags"]:
            warnings.append(
                f"Too many hashtags ({len(hashtags)}), recommended max is {limits['max_hashtags']}"
            )

        word_count = len(request.content.split())

        return PostPreviewResponse(
            preview=preview_html,
            character_count=char_count,
            word_count=word_count,
            platform_limits=limits,
            warnings=warnings,
        )
    except Exception as e:
        logger.error(f"Failed to generate preview: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate preview: {str(e)}"
        )


@router.post("/validate", response_model=PostValidationResponse)
async def validate_post(request: PostValidationRequest):
    """
    Validate a social media post against platform-specific rules.

    Args:
        request: Validation request with content and platform

    Returns:
        Validation results with errors, warnings, and suggestions
    """
    try:
        errors = []
        warnings = []
        suggestions = []
        auto_fix_available = False
        auto_fixed_content = None

        # Load platform config
        pipeline = SocialMediaPipeline()
        try:
            platform_config = await pipeline._load_platform_config()
        except Exception:
            platform_config = {}
        platform_spec = platform_config.get(request.platform, {})

        # Validate character limits
        char_limit = platform_spec.get("max_characters", 3000)
        char_count = len(request.content)
        if char_count > char_limit:
            errors.append(
                f"Content exceeds {request.platform} limit of {char_limit} characters by {char_count - char_limit}"
            )
            # Try auto-fix
            is_platform_error, error_type, error_details = (
                PlatformErrorHandler.detect_platform_error(
                    Exception("Character limit exceeded"),
                    request.platform,
                    request.content,
                )
            )
            if is_platform_error:
                fixed, was_fixed = PlatformErrorHandler.auto_fix_content(
                    request.content, error_type, error_details
                )
                if was_fixed:
                    auto_fix_available = True
                    auto_fixed_content = fixed
        elif char_count > platform_spec.get("recommended_characters", 2000):
            warnings.append(
                f"Content exceeds recommended length for {request.platform}"
            )

        # Validate hashtags (LinkedIn specific)
        if request.platform == "linkedin":
            import re

            hashtags = re.findall(r"#\w+", request.content)
            max_hashtags = platform_spec.get("max_hashtags", 5)
            if len(hashtags) > max_hashtags:
                warnings.append(
                    f"Too many hashtags ({len(hashtags)}), recommended max is {max_hashtags}"
                )
                suggestions.append(f"Consider reducing to {max_hashtags} hashtags")

            # Check for invalid hashtags
            invalid_hashtags = [
                tag
                for tag in hashtags
                if not re.match(r"^#\w+$", tag)
                or not tag[1:].replace("_", "").isalnum()
            ]
            if invalid_hashtags:
                errors.append(
                    f"Invalid hashtags found: {', '.join(invalid_hashtags)}. Hashtags must be alphanumeric."
                )

        # Validate subject line (Email specific)
        if request.platform == "email" and request.subject_line:
            subject_length = len(request.subject_line)
            if subject_length > 60:
                warnings.append(
                    f"Subject line is {subject_length} characters, recommended max is 60 for mobile display"
                )
            elif subject_length < 10:
                warnings.append(
                    "Subject line is very short, consider making it more descriptive"
                )

        # Validate content quality
        if char_count < 50:
            warnings.append("Content is very short, consider adding more detail")
            suggestions.append("Add more context or details to improve engagement")

        is_valid = len(errors) == 0

        return PostValidationResponse(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            auto_fix_available=auto_fix_available,
            auto_fixed_content=auto_fixed_content,
        )
    except Exception as e:
        logger.error(f"Failed to validate post: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to validate post: {str(e)}"
        )


@router.post("/update", response_model=PostUpdateResponse)
async def update_post(request: PostUpdateRequest):
    """
    Update a social media post content.

    Args:
        request: Update request with job_id and updated content

    Returns:
        Update response with success status
    """
    try:
        from marketing_project.services.job_manager import get_job_manager

        job_manager = get_job_manager()
        job = await job_manager.get_job(request.job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Validate updated content
        validation_request = PostValidationRequest(
            content=request.content,
            platform=request.platform,
            email_type=request.email_type,
            subject_line=request.subject_line,
        )
        validation_result = await validate_post(validation_request)

        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Validation failed: {', '.join(validation_result.errors)}",
            )

        # Update job result with new content
        if not job.result:
            job.result = {}

        # Handle both direct result and nested pipeline_result structures
        result = job.result
        if isinstance(result, dict):
            # Check for nested pipeline_result structure
            if "pipeline_result" in result:
                pipeline_result = result["pipeline_result"]
                if isinstance(pipeline_result, dict):
                    pipeline_result["final_content"] = request.content
                    if request.platform == "email" and request.subject_line:
                        pipeline_result["subject_line"] = request.subject_line
            # Also update direct final_content if it exists
            if "final_content" in result:
                result["final_content"] = request.content
            else:
                # If no final_content exists, add it
                result["final_content"] = request.content

            # Update subject_line for email platform
            if request.platform == "email" and request.subject_line:
                if "pipeline_result" in result and isinstance(
                    result["pipeline_result"], dict
                ):
                    result["pipeline_result"]["subject_line"] = request.subject_line
                result["subject_line"] = request.subject_line

            # Update the job's result
            job.result = result

        # Save updated job
        await job_manager._save_job(job)

        logger.info(f"Updated post content for job {request.job_id}")

        return PostUpdateResponse(
            success=True,
            message="Post updated successfully",
            updated_content=request.content,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update post: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update post: {str(e)}")
