"""
Tests for core API endpoints (analyze, pipeline).
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from marketing_project.models import AnalyzeRequest, PipelineRequest, ContentContext
from marketing_project.api.core import router


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
        snippet="A test article snippet"
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

    @patch('marketing_project.api.core.analyze_content_for_pipeline')
    def test_analyze_content_success(
        self, 
        mock_analyze, 
        client, 
        analyze_request
    ):
        """Test successful content analysis."""
        # Setup mocks
        mock_analyze.return_value = {
            "quality_score": 85,
            "seo_potential": "high",
            "recommendations": ["Add more keywords", "Improve structure"]
        }
        
        response = client.post("/analyze", json=analyze_request.dict())
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["content_id"] == "test_content_1"
        assert "analysis" in data
        assert data["analysis"]["quality_score"] == 85

    @patch('marketing_project.api.core.analyze_content_for_pipeline')
    def test_analyze_content_analysis_error(
        self, 
        mock_analyze, 
        client, 
        analyze_request
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

    @patch('marketing_project.api.core.run_marketing_project_pipeline')
    def test_run_pipeline_success(
        self, 
        mock_run_pipeline, 
        client, 
        pipeline_request
    ):
        """Test successful pipeline execution."""
        # Setup mocks
        mock_run_pipeline.return_value = {
            "steps": [
                {"name": "AnalyzeContent", "status": "completed"},
                {"name": "ExtractSEOKeywords", "status": "completed"},
                {"name": "GenerateMarketingBrief", "status": "completed"}
            ],
            "total_time": 45.2,
            "success": True
        }
        
        response = client.post("/pipeline", json=pipeline_request.dict())
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["content_id"] == "test_content_1"
        assert "result" in data
        assert len(data["result"]["steps"]) == 3

    @patch('marketing_project.api.core.run_marketing_project_pipeline')
    def test_run_pipeline_execution_error(
        self, 
        mock_run_pipeline, 
        client, 
        pipeline_request
    ):
        """Test pipeline execution error."""
        # Setup mocks
        mock_run_pipeline.side_effect = Exception("Pipeline execution failed")
        
        response = client.post("/pipeline", json=pipeline_request.dict())
        
        assert response.status_code == 500
        data = response.json()
        assert "Pipeline execution failed" in data["detail"]
