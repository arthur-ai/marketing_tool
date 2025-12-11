"""
Tests for approval manager keyword filtering methods.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.approval_manager import ApprovalManager


@pytest.fixture
def mock_redis_manager():
    """Mock Redis manager."""
    with patch("marketing_project.services.approval_manager.get_redis_manager") as mock:
        manager = MagicMock()
        manager.get_redis = AsyncMock(return_value=MagicMock())
        manager.execute = AsyncMock(return_value=None)
        mock.return_value = manager
        yield manager


@pytest.fixture
def approval_manager(mock_redis_manager):
    """Create an ApprovalManager instance."""
    return ApprovalManager()


def test_filter_selected_keywords_with_main_keyword(approval_manager):
    """Test filter_selected_keywords with main keyword."""
    output_data = {
        "main_keyword": "original",
        "primary_keywords": ["original", "keyword1", "keyword2"],
        "secondary_keywords": ["keyword3", "keyword4"],
        "lsi_keywords": ["keyword5"],
    }

    selected = {
        "primary": ["original", "keyword1"],
        "secondary": ["keyword3"],
    }

    filtered = approval_manager.filter_selected_keywords(
        output_data, selected, main_keyword="original"
    )

    assert filtered["main_keyword"] == "original"
    assert "keyword1" in filtered["primary_keywords"]
    assert "keyword3" in filtered["secondary_keywords"]


def test_filter_selected_keywords_promote_to_main(approval_manager):
    """Test filter_selected_keywords with promotion to main."""
    output_data = {
        "main_keyword": "original",
        "primary_keywords": ["original", "keyword1"],
        "secondary_keywords": ["keyword2", "promoted"],
    }

    selected = {
        "primary": ["original", "keyword1"],
        "secondary": ["keyword2"],
    }

    filtered = approval_manager.filter_selected_keywords(
        output_data, selected, main_keyword="promoted"
    )

    assert filtered["main_keyword"] == "promoted"
    assert "promoted" in filtered["primary_keywords"]


def test_filter_selected_keywords_invalid_main(approval_manager):
    """Test filter_selected_keywords with invalid main keyword."""
    output_data = {
        "main_keyword": "original",
        "primary_keywords": ["original", "keyword1"],
    }

    with pytest.raises(ValueError, match="must exist in one of the original"):
        approval_manager.filter_selected_keywords(
            output_data, {}, main_keyword="nonexistent"
        )


def test_filter_selected_keywords_no_selection(approval_manager):
    """Test filter_selected_keywords with no keywords selected."""
    output_data = {
        "main_keyword": "original",
        "primary_keywords": ["original", "keyword1"],
    }

    # When main_keyword is None, it should raise "Main keyword is required"
    with pytest.raises(ValueError, match="Main keyword is required"):
        approval_manager.filter_selected_keywords(output_data, {}, main_keyword=None)


def test_filter_selected_keywords_invalid_primary(approval_manager):
    """Test filter_selected_keywords with invalid primary keywords."""
    output_data = {
        "main_keyword": "original",
        "primary_keywords": ["original", "keyword1"],
    }

    selected = {
        "primary": ["nonexistent"],
    }

    with pytest.raises(ValueError, match="not found in original"):
        approval_manager.filter_selected_keywords(
            output_data, selected, main_keyword="original"
        )
