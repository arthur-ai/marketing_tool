"""
Tests for core API endpoints (analyze, pipeline).
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.core import router
from marketing_project.models import AnalyzeRequest, ContentContext, PipelineRequest


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def sample_content():
    """Sample content for testing."""
    from marketing_project.models import BlogPostContext

    return BlogPostContext(
        id="test_content_1",
        title="Test Article",
        content="This is a test article about marketing automation.",
        snippet="A test article snippet",
    )


@pytest.fixture
def analyze_request(sample_content):
    """Sample analyze request."""
    return AnalyzeRequest(content=sample_content)


@pytest.fixture
def pipeline_request(sample_content):
    """Sample pipeline request."""
    return PipelineRequest(content=sample_content)


class TestAnalyzeEndpoint:
    """Test the /analyze endpoint."""

    @patch("marketing_project.api.core.analyze_content_for_pipeline")
    def test_analyze_content_success(self, mock_analyze, client, analyze_request):
        """Test successful content analysis."""
        # Setup mocks
        mock_analyze.return_value = {
            "quality_score": 85,
            "seo_potential": "high",
            "recommendations": ["Add more keywords", "Improve structure"],
        }

        response = client.post("/analyze", json=analyze_request.model_dump(mode="json"))

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["content_id"] == "test_content_1"
        assert "analysis" in data
        assert data["analysis"]["quality_score"] == 85

    @patch("marketing_project.api.core.analyze_content_for_pipeline")
    def test_analyze_content_analysis_error(
        self, mock_analyze, client, analyze_request
    ):
        """Test content analysis error."""
        # Setup mocks
        mock_analyze.side_effect = Exception("Analysis failed")

        response = client.post("/analyze", json=analyze_request.model_dump(mode="json"))

        assert response.status_code == 500
        data = response.json()
        assert "Content analysis failed" in data["detail"]


class TestPipelineEndpoint:
    """Test the /pipeline endpoint."""

    @patch("marketing_project.api.core.process_blog_post")
    @pytest.mark.asyncio
    async def test_run_pipeline_success_blog(
        self,
        mock_process_blog,
        client,
        pipeline_request,
    ):
        """Test successful pipeline execution for blog posts.

        The pipeline now routes directly to deterministic processors based on content type.
        """
        import json

        # Setup mock processor response
        mock_process_blog.return_value = json.dumps(
            {
                "status": "success",
                "content_type": "blog_post",
                "blog_type": "tutorial",
                "metadata": {
                    "author": "Test Author",
                    "category": "tutorial",
                    "word_count": 100,
                },
                "pipeline_result": {
                    "seo_keywords": ["test", "marketing"],
                    "formatted_content": "Formatted content",
                },
                "validation": "passed",
                "processing_steps_completed": [
                    "validation",
                    "metadata_extraction",
                    "pipeline",
                ],
                "message": "Blog post processed successfully",
            }
        )

        response = client.post(
            "/pipeline", json=pipeline_request.model_dump(mode="json")
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["content_id"] == "test_content_1"
        assert "result" in data
        assert data["result"]["content_type"] == "blog_post"
        assert "metadata" in data["result"]
        assert "pipeline_result" in data["result"]

        # Verify the processor was called
        mock_process_blog.assert_called_once()

    @patch("marketing_project.api.core.process_blog_post")
    @pytest.mark.asyncio
    async def test_run_pipeline_processor_error(
        self,
        mock_process_blog,
        client,
        pipeline_request,
    ):
        """Test pipeline execution error from processor.

        The pipeline now routes directly to deterministic processors based on content type.
        """
        import json

        # Setup mock processor to return error
        mock_process_blog.return_value = json.dumps(
            {
                "status": "error",
                "error": "validation_failed",
                "message": "Content validation failed",
            }
        )

        response = client.post(
            "/pipeline", json=pipeline_request.model_dump(mode="json")
        )

        assert response.status_code == 400
        data = response.json()
        assert "Content validation failed" in data["detail"]

    @patch("marketing_project.api.core.process_blog_post")
    @pytest.mark.asyncio
    async def test_run_pipeline_execution_error(
        self,
        mock_process_blog,
        client,
        pipeline_request,
    ):
        """Test pipeline execution error when processor raises exception."""
        # Setup mock processor to raise exception
        mock_process_blog.side_effect = Exception("Pipeline execution failed")

        response = client.post(
            "/pipeline", json=pipeline_request.model_dump(mode="json")
        )

        assert response.status_code == 500
        data = response.json()
        assert "Pipeline execution failed" in data["detail"]
