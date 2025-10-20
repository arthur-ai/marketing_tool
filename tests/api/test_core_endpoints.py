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

        response = client.post("/analyze", json=analyze_request.dict())

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

        response = client.post("/analyze", json=analyze_request.dict())

        assert response.status_code == 500
        data = response.json()
        assert "Content analysis failed" in data["detail"]


class TestPipelineEndpoint:
    """Test the /pipeline endpoint."""

    @patch("marketing_project.api.core.get_marketing_orchestrator_agent")
    @patch("marketing_project.api.core.get_releasenotes_agent")
    @patch("marketing_project.api.core.get_blog_agent")
    @patch("marketing_project.api.core.get_transcripts_agent")
    @pytest.mark.asyncio
    async def test_run_pipeline_success(
        self,
        mock_transcripts_agent,
        mock_blog_agent,
        mock_releasenotes_agent,
        mock_orchestrator_agent,
        client,
        pipeline_request,
    ):
        """Test successful pipeline execution."""
        # Setup mock agents
        mock_orchestrator = AsyncMock()
        mock_orchestrator.run_async = AsyncMock(
            return_value={
                "status": "completed",
                "content_type": "blog_post",
                "processed": True,
            }
        )

        mock_transcripts_agent.return_value = AsyncMock()
        mock_blog_agent.return_value = AsyncMock()
        mock_releasenotes_agent.return_value = AsyncMock()
        mock_orchestrator_agent.return_value = mock_orchestrator

        response = client.post("/pipeline", json=pipeline_request.dict())

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["content_id"] == "test_content_1"
        assert "result" in data
        assert "processed_content" in data["result"]
        assert "stats" in data["result"]

        # Verify the orchestrator was called with a string prompt
        mock_orchestrator.run_async.assert_called_once()
        call_args = mock_orchestrator.run_async.call_args[0]
        assert isinstance(call_args[0], str)
        assert "test_content_1" in call_args[0]
        assert "blogpost" in call_args[0]

    @patch("marketing_project.api.core.get_marketing_orchestrator_agent")
    @patch("marketing_project.api.core.get_releasenotes_agent")
    @patch("marketing_project.api.core.get_blog_agent")
    @patch("marketing_project.api.core.get_transcripts_agent")
    @pytest.mark.asyncio
    async def test_run_pipeline_execution_error(
        self,
        mock_transcripts_agent,
        mock_blog_agent,
        mock_releasenotes_agent,
        mock_orchestrator_agent,
        client,
        pipeline_request,
    ):
        """Test pipeline execution error."""
        # Setup mock agents
        mock_orchestrator = AsyncMock()
        mock_orchestrator.run_async = AsyncMock(
            side_effect=Exception("Pipeline execution failed")
        )

        mock_transcripts_agent.return_value = AsyncMock()
        mock_blog_agent.return_value = AsyncMock()
        mock_releasenotes_agent.return_value = AsyncMock()
        mock_orchestrator_agent.return_value = mock_orchestrator

        response = client.post("/pipeline", json=pipeline_request.dict())

        assert response.status_code == 500
        data = response.json()
        assert "Pipeline execution failed" in data["detail"]
