"""
Tests for per-user approval settings in check_and_create_approval_request.

Verifies that user_settings in pipeline_context override the global approval
manager settings, with correct fallback when absent or malformed.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.processors.approval_helper import (
    ApprovalResult,
    ApprovalStatus,
    check_and_create_approval_request,
)


def _make_manager(
    require_approval=True, approval_agents=None, auto_approve_threshold=None
):
    """Build a mock ApprovalManager with configurable settings."""
    mock_manager = MagicMock()
    mock_manager.settings.require_approval = require_approval
    mock_manager.settings.approval_agents = approval_agents or ["seo_keywords"]
    mock_manager.settings.auto_approve_threshold = auto_approve_threshold
    mock_approval = MagicMock()
    mock_approval.id = "approval-123"
    mock_approval.status = "pending"
    mock_manager.create_approval_request = AsyncMock(return_value=mock_approval)
    mock_manager.save_pipeline_context = AsyncMock()
    return mock_manager


@pytest.fixture
def mock_manager_global_enabled():
    return _make_manager(require_approval=True, approval_agents=["seo_keywords"])


@pytest.mark.asyncio
async def test_user_require_approval_false_overrides_global_true(
    mock_manager_global_enabled,
):
    """Per-user require_approval=False skips gate even when global is True."""
    context = {
        "user_settings": {
            "require_approval": False,
            "approval_agents": ["seo_keywords"],
        }
    }

    with patch(
        "marketing_project.processors.approval_helper.get_approval_manager",
        new_callable=AsyncMock,
        return_value=mock_manager_global_enabled,
    ):
        result = await check_and_create_approval_request(
            job_id="job-1",
            agent_name="seo_keywords",
            step_name="seo_keywords",
            step_number=3,
            input_data={},
            output_data={},
            context=context,
        )

    assert result.status == ApprovalStatus.NOT_REQUIRED
    mock_manager_global_enabled.create_approval_request.assert_not_called()


@pytest.mark.asyncio
async def test_user_require_approval_true_overrides_global_false():
    """Per-user require_approval=True triggers gate even when global is False."""
    manager = _make_manager(require_approval=False, approval_agents=["seo_keywords"])
    context = {
        "user_settings": {
            "require_approval": True,
            "approval_agents": ["seo_keywords"],
        }
    }

    with patch(
        "marketing_project.processors.approval_helper.get_approval_manager",
        new_callable=AsyncMock,
        return_value=manager,
    ):
        result = await check_and_create_approval_request(
            job_id="job-2",
            agent_name="seo_keywords",
            step_name="seo_keywords",
            step_number=3,
            input_data={},
            output_data={},
            context=context,
        )

    assert result.status == ApprovalStatus.REQUIRED
    manager.create_approval_request.assert_called_once()


@pytest.mark.asyncio
async def test_no_user_settings_falls_back_to_global(mock_manager_global_enabled):
    """Empty context["user_settings"] falls back to global manager settings."""
    with patch(
        "marketing_project.processors.approval_helper.get_approval_manager",
        new_callable=AsyncMock,
        return_value=mock_manager_global_enabled,
    ):
        result = await check_and_create_approval_request(
            job_id="job-3",
            agent_name="seo_keywords",
            step_name="seo_keywords",
            step_number=3,
            input_data={},
            output_data={},
            context={},  # no user_settings key
        )

    # Global has require_approval=True and agent in list → REQUIRED
    assert result.status == ApprovalStatus.REQUIRED


@pytest.mark.asyncio
async def test_user_settings_none_falls_back_to_global(mock_manager_global_enabled):
    """context["user_settings"] = None falls back to global."""
    with patch(
        "marketing_project.processors.approval_helper.get_approval_manager",
        new_callable=AsyncMock,
        return_value=mock_manager_global_enabled,
    ):
        result = await check_and_create_approval_request(
            job_id="job-4",
            agent_name="seo_keywords",
            step_name="seo_keywords",
            step_number=3,
            input_data={},
            output_data={},
            context={"user_settings": None},
        )

    assert result.status == ApprovalStatus.REQUIRED


@pytest.mark.asyncio
async def test_corrupt_user_settings_falls_back_to_global(mock_manager_global_enabled):
    """Malformed user_settings dict (missing require_approval) falls back to global."""
    context = {"user_settings": {"bad_key": "garbage"}}  # missing require_approval

    with patch(
        "marketing_project.processors.approval_helper.get_approval_manager",
        new_callable=AsyncMock,
        return_value=mock_manager_global_enabled,
    ):
        result = await check_and_create_approval_request(
            job_id="job-5",
            agent_name="seo_keywords",
            step_name="seo_keywords",
            step_number=3,
            input_data={},
            output_data={},
            context=context,
        )

    # Falls back to global which has require_approval=True → REQUIRED
    assert result.status == ApprovalStatus.REQUIRED


@pytest.mark.asyncio
async def test_per_user_approval_agents_override():
    """Per-user approval_agents list takes precedence over global."""
    # Global has seo_keywords in agents, user does NOT
    manager = _make_manager(require_approval=True, approval_agents=["seo_keywords"])
    context = {
        "user_settings": {
            "require_approval": True,
            "approval_agents": ["article_generation"],  # seo_keywords NOT in user list
        }
    }

    with patch(
        "marketing_project.processors.approval_helper.get_approval_manager",
        new_callable=AsyncMock,
        return_value=manager,
    ):
        result = await check_and_create_approval_request(
            job_id="job-6",
            agent_name="seo_keywords",
            step_name="seo_keywords",
            step_number=3,
            input_data={},
            output_data={},
            context=context,
        )

    assert result.status == ApprovalStatus.NOT_REQUIRED
    manager.create_approval_request.assert_not_called()
