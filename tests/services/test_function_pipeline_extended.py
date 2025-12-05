"""
Extended tests for function pipeline service methods.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.function_pipeline import FunctionPipeline


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    with patch("marketing_project.services.function_pipeline.AsyncOpenAI") as mock:
        mock_client = MagicMock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def function_pipeline(mock_openai):
    """Create a FunctionPipeline instance."""
    return FunctionPipeline()


class TestFunctionPipelineHelpers:
    """Test helper methods in FunctionPipeline."""

    def test_get_system_instruction(self, function_pipeline):
        """Test _get_system_instruction method."""
        instruction = function_pipeline._get_system_instruction("seo_keywords")

        assert isinstance(instruction, str)
        assert len(instruction) > 0

    def test_get_user_prompt(self, function_pipeline):
        """Test _get_user_prompt method."""
        context = {"input_content": {"title": "Test", "content": "Content"}}

        prompt = function_pipeline._get_user_prompt("seo_keywords", context)

        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_step_model(self, function_pipeline):
        """Test _get_step_model method."""
        model = function_pipeline._get_step_model("seo_keywords")

        assert isinstance(model, str)
        assert len(model) > 0

    def test_get_step_temperature(self, function_pipeline):
        """Test _get_step_temperature method."""
        temp = function_pipeline._get_step_temperature("seo_keywords")

        assert isinstance(temp, (int, float))
        assert 0 <= temp <= 2

    def test_get_step_max_retries(self, function_pipeline):
        """Test _get_step_max_retries method."""
        retries = function_pipeline._get_step_max_retries("seo_keywords")

        assert isinstance(retries, int)
        assert retries >= 0

    @pytest.mark.asyncio
    async def test_call_function(self, function_pipeline, mock_openai):
        """Test _call_function method."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"main_keyword": "test"}'
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await function_pipeline._call_function(
            prompt="Test prompt",
            system_instruction="Test instruction",
            response_model=None,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_step_with_plugin(self, function_pipeline):
        """Test _execute_step_with_plugin method."""
        with patch(
            "marketing_project.services.function_pipeline.get_plugin_registry"
        ) as mock_registry:
            mock_plugin = MagicMock()
            mock_plugin.step_name = "seo_keywords"
            mock_plugin.step_number = 1
            mock_plugin.execute = AsyncMock(return_value=MagicMock(main_keyword="test"))
            mock_registry.return_value.get_plugin.return_value = mock_plugin

            context = {"input_content": {"title": "Test", "content": "Content"}}

            result = await function_pipeline._execute_step_with_plugin(
                "seo_keywords", context, "test-job"
            )

            assert result is not None
