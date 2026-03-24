"""
Tests for approval helper functions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.processors.approval_helper import (
    ApprovalRequiredException,
    ApprovalResult,
    ApprovalStatus,
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
        new_callable=AsyncMock,
        return_value=mock_manager,
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

        assert isinstance(result, ApprovalResult)
        assert result.status == ApprovalStatus.NOT_REQUIRED
        assert result.can_continue


@pytest.mark.asyncio
async def test_check_and_create_approval_request_agent_not_in_list():
    """Test check_and_create_approval_request when agent is not in approval list."""
    mock_manager = MagicMock()
    mock_manager.settings.require_approval = True
    mock_manager.settings.approval_agents = ["other_agent"]

    with patch(
        "marketing_project.processors.approval_helper.get_approval_manager",
        new_callable=AsyncMock,
        return_value=mock_manager,
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

        assert isinstance(result, ApprovalResult)
        assert result.status == ApprovalStatus.NOT_REQUIRED
        assert result.can_continue


@pytest.mark.asyncio
async def test_check_and_create_approval_request_approval_required():
    """Test check_and_create_approval_request when approval is required."""
    mock_manager = MagicMock()
    mock_manager.settings.require_approval = True
    mock_manager.settings.approval_agents = ["seo_keywords"]
    mock_approval = MagicMock()
    mock_approval.id = "test-approval-1"
    mock_approval.status = "pending"  # Not "approved" so exception is raised
    mock_manager.create_approval_request = AsyncMock(return_value=mock_approval)
    mock_manager.save_pipeline_context = AsyncMock()

    with patch(
        "marketing_project.processors.approval_helper.get_approval_manager",
        new_callable=AsyncMock,
        return_value=mock_manager,
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

        assert isinstance(result, ApprovalResult)
        assert result.status == ApprovalStatus.REQUIRED
        assert result.requires_approval
        assert result.approval_id == "test-approval-1"
        assert result.job_id == "test-job-1"
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

    result = prepare_approval_data(input_data, output_data)

    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2
    input_result, output_result = result
    assert isinstance(input_result, dict)
    assert isinstance(output_result, dict)


def test_prepare_approval_data_with_confidence():
    """Test prepare_approval_data with confidence score."""
    input_data = {"content": {"title": "Test"}}
    output_data = {"keywords": ["test"], "confidence": 0.8}

    result = prepare_approval_data(input_data, output_data)

    assert result is not None
    assert isinstance(result, tuple)
    input_result, output_result = result
    assert isinstance(input_result, dict)
    assert isinstance(output_result, dict)


def test_prepare_approval_data_with_suggestions():
    """Test prepare_approval_data with suggestions."""
    input_data = {"content": {"title": "Test"}}
    output_data = {"keywords": ["test"]}

    result = prepare_approval_data(input_data, output_data)

    assert result is not None
    assert isinstance(result, tuple)
    input_result, output_result = result
    assert isinstance(input_result, dict)
    assert isinstance(output_result, dict)


@pytest.mark.asyncio
async def test_check_and_create_approval_request_passes_job_root_traceparent():
    """save_pipeline_context receives the __job_root_traceparent__ from context, not the live step span."""
    mock_manager = MagicMock()
    mock_manager.settings.require_approval = True
    mock_manager.settings.approval_agents = ["seo_keywords"]
    mock_approval = MagicMock()
    mock_approval.id = "test-approval-tp"
    mock_approval.status = "pending"
    mock_manager.create_approval_request = AsyncMock(return_value=mock_approval)
    mock_manager.save_pipeline_context = AsyncMock()

    traceparent_value = "00-aabbccddeeff00112233445566778899-0011223344556677-01"

    with patch(
        "marketing_project.processors.approval_helper.get_approval_manager",
        new_callable=AsyncMock,
        return_value=mock_manager,
    ):
        result = await check_and_create_approval_request(
            job_id="test-job-tp",
            agent_name="seo_keywords",
            step_name="Step 1",
            step_number=1,
            input_data={},
            output_data={},
            context={"__job_root_traceparent__": traceparent_value},
        )

    assert isinstance(result, ApprovalResult)
    assert result.status == ApprovalStatus.REQUIRED
    call_kwargs = mock_manager.save_pipeline_context.call_args
    assert call_kwargs is not None
    assert call_kwargs.kwargs.get("traceparent") == traceparent_value


@pytest.mark.asyncio
async def test_check_and_create_approval_request_falls_back_to_live_traceparent():
    """Falls back to get_current_traceparent() when __job_root_traceparent__ is absent."""
    mock_manager = MagicMock()
    mock_manager.settings.require_approval = True
    mock_manager.settings.approval_agents = ["seo_keywords"]
    mock_approval = MagicMock()
    mock_approval.id = "test-approval-fallback"
    mock_approval.status = "pending"
    mock_manager.create_approval_request = AsyncMock(return_value=mock_approval)
    mock_manager.save_pipeline_context = AsyncMock()

    with patch(
        "marketing_project.processors.approval_helper.get_approval_manager",
        new_callable=AsyncMock,
        return_value=mock_manager,
    ):
        with patch(
            "marketing_project.processors.approval_helper.get_current_traceparent",
            return_value="00-fallback00000000000000000000000-fallback00000000-01",
        ):
            result = await check_and_create_approval_request(
                job_id="test-job-fallback",
                agent_name="seo_keywords",
                step_name="Step 1",
                step_number=1,
                input_data={},
                output_data={},
                context={},  # no __job_root_traceparent__
            )
    assert isinstance(result, ApprovalResult)
    assert result.status == ApprovalStatus.REQUIRED

    call_kwargs = mock_manager.save_pipeline_context.call_args
    assert call_kwargs is not None
    assert (
        call_kwargs.kwargs.get("traceparent")
        == "00-fallback00000000000000000000000-fallback00000000-01"
    )
