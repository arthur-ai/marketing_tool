"""
Tests for approval helper functions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.processors.approval_helper import (
    ApprovalRequiredException,
    check_and_create_approval_request,
    extract_confidence_score,
    prepare_approval_data,
)


def test_approval_required_exception():
    """Test ApprovalRequiredException initialization."""
    exc = ApprovalRequiredException(
        approval_id="test-approval-1",
        job_id="test-job-1",
        step_name="Step 1",
        step_number=1,
    )

    assert exc.approval_id == "test-approval-1"
    assert exc.job_id == "test-job-1"
    assert exc.step_name == "Step 1"
    assert exc.step_number == 1
    assert "Approval required" in str(exc)


@pytest.mark.asyncio
async def test_check_and_create_approval_request_approvals_disabled():
    """Test check_and_create_approval_request when approvals are disabled."""
    mock_manager = MagicMock()
    mock_manager.settings.require_approval = False
    mock_manager.settings.approval_agents = []

    with patch(
        "marketing_project.processors.approval_helper.get_approval_manager",
        return_value=AsyncMock(return_value=mock_manager),
    ):
        result = await check_and_create_approval_request(
            job_id="test-job-1",
            agent_name="seo_keywords",
            step_name="Step 1",
            step_number=1,
            input_data={},
            output_data={},
            context={},
        )

        assert result is False


@pytest.mark.asyncio
async def test_check_and_create_approval_request_agent_not_in_list():
    """Test check_and_create_approval_request when agent is not in approval list."""
    mock_manager = MagicMock()
    mock_manager.settings.require_approval = True
    mock_manager.settings.approval_agents = ["other_agent"]

    with patch(
        "marketing_project.processors.approval_helper.get_approval_manager",
        return_value=AsyncMock(return_value=mock_manager),
    ):
        result = await check_and_create_approval_request(
            job_id="test-job-1",
            agent_name="seo_keywords",
            step_name="Step 1",
            step_number=1,
            input_data={},
            output_data={},
            context={},
        )

        assert result is False


@pytest.mark.asyncio
async def test_check_and_create_approval_request_approval_required():
    """Test check_and_create_approval_request when approval is required."""
    mock_manager = MagicMock()
    mock_manager.settings.require_approval = True
    mock_manager.settings.approval_agents = ["seo_keywords"]
    mock_approval = MagicMock()
    mock_approval.id = "test-approval-1"
    mock_manager.create_approval_request = AsyncMock(return_value=mock_approval)
    mock_manager.save_pipeline_context = AsyncMock()

    with patch(
        "marketing_project.processors.approval_helper.get_approval_manager",
        return_value=AsyncMock(return_value=mock_manager),
    ):
        with pytest.raises(ApprovalRequiredException) as exc_info:
            await check_and_create_approval_request(
                job_id="test-job-1",
                agent_name="seo_keywords",
                step_name="Step 1",
                step_number=1,
                input_data={},
                output_data={},
                context={},
            )

        assert exc_info.value.approval_id == "test-approval-1"
        assert exc_info.value.job_id == "test-job-1"
        mock_manager.create_approval_request.assert_called_once()
        mock_manager.save_pipeline_context.assert_called_once()


def test_extract_confidence_score_with_confidence():
    """Test extract_confidence_score when confidence is present."""
    output = {"confidence": 0.85, "result": "test"}

    result = extract_confidence_score(output)

    assert result == 0.85


def test_extract_confidence_score_with_confidence_score():
    """Test extract_confidence_score with confidence_score field."""
    output = {"confidence_score": 0.75, "result": "test"}

    result = extract_confidence_score(output)

    assert result == 0.75


def test_extract_confidence_score_without_confidence():
    """Test extract_confidence_score when confidence is not present."""
    output = {"result": "test"}

    result = extract_confidence_score(output)

    assert result is None


def test_extract_confidence_score_nested():
    """Test extract_confidence_score with nested structure."""
    output = {"metadata": {"confidence": 0.9}, "result": "test"}

    result = extract_confidence_score(output)

    # Should return None if not at top level
    assert result is None or result == 0.9


def test_prepare_approval_data():
    """Test prepare_approval_data function."""
    input_data = {"content": {"title": "Test"}}
    output_data = {"keywords": ["test"]}
    context = {"previous_step": "result"}

    result = prepare_approval_data(input_data, output_data, context)

    assert result is not None
    assert "input" in result
    assert "output" in result
    assert "context" in result


def test_prepare_approval_data_with_confidence():
    """Test prepare_approval_data with confidence score."""
    input_data = {"content": {"title": "Test"}}
    output_data = {"keywords": ["test"], "confidence": 0.8}
    context = {}

    result = prepare_approval_data(
        input_data, output_data, context, confidence_score=0.8
    )

    assert result is not None
    assert "confidence_score" in result or "confidence" in result


def test_prepare_approval_data_with_suggestions():
    """Test prepare_approval_data with suggestions."""
    input_data = {"content": {"title": "Test"}}
    output_data = {"keywords": ["test"]}
    context = {}
    suggestions = ["suggestion1", "suggestion2"]

    result = prepare_approval_data(
        input_data, output_data, context, suggestions=suggestions
    )

    assert result is not None
    assert "suggestions" in result or suggestions in result.values()
