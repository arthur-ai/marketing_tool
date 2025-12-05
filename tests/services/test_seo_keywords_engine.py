"""
Tests for SEO Keywords engine modules.

Tests the composer, LLM engine, and local semantic engine.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.pipeline_steps import SEOKeywordsResult
from marketing_project.services.engines.composer import EngineComposer
from marketing_project.services.engines.seo_keywords.composer import SEOKeywordsComposer
from marketing_project.services.engines.seo_keywords.llm_engine import (
    LLMSEOKeywordsEngine,
)


@pytest.fixture
def mock_engine_composer():
    """Create a mock EngineComposer."""
    composer = MagicMock(spec=EngineComposer)
    composer.default_engine_type = "llm"
    composer.field_overrides = {}
    composer.execute_operation = AsyncMock()
    composer.get_engine_type_for_field = MagicMock(return_value="llm")
    return composer


@pytest.fixture
def seo_keywords_composer(mock_engine_composer):
    """Create a SEOKeywordsComposer instance."""
    return SEOKeywordsComposer(mock_engine_composer)


@pytest.fixture
def mock_plugin():
    """Create a mock plugin for LLM engine."""
    plugin = MagicMock()
    plugin._execute_step = AsyncMock(
        return_value=SEOKeywordsResult(
            main_keyword="test",
            primary_keywords=["test", "keyword"],
            search_intent="informational",
        )
    )
    return plugin


@pytest.fixture
def llm_engine(mock_plugin):
    """Create a LLMSEOKeywordsEngine instance."""
    return LLMSEOKeywordsEngine(mock_plugin)


class TestSEOKeywordsComposer:
    """Test SEOKeywordsComposer."""

    @pytest.mark.asyncio
    async def test_compose_result_llm_default(
        self, seo_keywords_composer, mock_engine_composer
    ):
        """Test compose_result with LLM as default engine."""
        content = {"title": "Test", "content": "Test content"}
        context = {}

        mock_result = SEOKeywordsResult(
            main_keyword="test",
            primary_keywords=["test"],
            search_intent="informational",
        )
        mock_engine_composer.execute_operation.return_value = mock_result

        result = await seo_keywords_composer.compose_result(
            content, context, pipeline=MagicMock()
        )

        assert isinstance(result, SEOKeywordsResult)
        assert result.main_keyword == "test"

    @pytest.mark.asyncio
    async def test_compose_result_with_field_overrides(
        self, seo_keywords_composer, mock_engine_composer
    ):
        """Test compose_result with field overrides."""
        mock_engine_composer.default_engine_type = "llm"
        mock_engine_composer.field_overrides = {
            "main_keyword": "local",
            "primary_keywords": "local",
        }

        content = {"title": "Test", "content": "Test content"}
        context = {}

        # Mock local engine result
        mock_engine_composer.execute_operation.return_value = "test_keyword"

        result = await seo_keywords_composer.compose_result(
            content, context, pipeline=MagicMock()
        )

        assert isinstance(result, SEOKeywordsResult)

    def test_get_fields_to_extract_with_overrides(
        self, seo_keywords_composer, mock_engine_composer
    ):
        """Test _get_fields_to_extract method with field overrides."""
        mock_engine_composer.default_engine_type = "llm"
        mock_engine_composer.field_overrides = {"main_keyword": "local_semantic"}
        mock_engine_composer.get_engine_type_for_field = MagicMock(
            return_value="local_semantic"
        )

        fields = seo_keywords_composer._get_fields_to_extract()

        assert isinstance(fields, list)
        assert "main_keyword" in fields

    @pytest.mark.asyncio
    async def test_extract_field_llm(self, seo_keywords_composer, mock_engine_composer):
        """Test _extract_field with LLM engine."""
        mock_engine_composer.get_engine_type_for_field.return_value = "llm"
        mock_engine_composer.execute_operation.return_value = "test_keyword"

        result = await seo_keywords_composer._extract_field(
            "main_keyword",
            {"title": "Test"},
            {},
            MagicMock(),
            {},
        )

        assert result == "test_keyword"

    @pytest.mark.asyncio
    async def test_extract_field_local(
        self, seo_keywords_composer, mock_engine_composer
    ):
        """Test _extract_field with local engine."""
        mock_engine_composer.get_engine_type_for_field.return_value = "local_semantic"
        mock_engine_composer.execute_operation.return_value = ["keyword1", "keyword2"]

        result = await seo_keywords_composer._extract_field(
            "primary_keywords",
            {"title": "Test", "content": "Test content"},
            {},
            None,
            {},
        )

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_extract_field_error_handling(
        self, seo_keywords_composer, mock_engine_composer
    ):
        """Test _extract_field error handling."""
        mock_engine_composer.get_engine_type_for_field.return_value = "llm"
        mock_engine_composer.execute_operation.side_effect = Exception("Error")

        result = await seo_keywords_composer._extract_field(
            "main_keyword",
            {"title": "Test"},
            {},
            MagicMock(),
            {},
        )

        # Should return None on error
        assert result is None


class TestLLMSEOKeywordsEngine:
    """Test LLMSEOKeywordsEngine."""

    def test_supports_operation(self, llm_engine):
        """Test supports_operation method."""
        assert llm_engine.supports_operation("extract_all") is True
        assert llm_engine.supports_operation("extract_main_keyword") is False

    @pytest.mark.asyncio
    async def test_execute_success(self, llm_engine, mock_plugin):
        """Test execute method success."""
        mock_pipeline = MagicMock()

        result = await llm_engine.execute(
            "extract_all",
            {"content": {"title": "Test", "content": "Test content"}},
            {},
            mock_pipeline,
        )

        assert isinstance(result, SEOKeywordsResult)
        assert result.main_keyword == "test"
        mock_plugin._execute_step.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_invalid_operation(self, llm_engine):
        """Test execute with invalid operation."""
        with pytest.raises(ValueError, match="only supports 'extract_all'"):
            await llm_engine.execute(
                "invalid_operation",
                {},
                {},
                MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_execute_missing_pipeline(self, llm_engine):
        """Test execute without pipeline."""
        with pytest.raises(ValueError, match="requires pipeline instance"):
            await llm_engine.execute(
                "extract_all",
                {},
                {},
                None,
            )
