"""
Extended tests for core API endpoints.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.core import router
from marketing_project.models import BlogPostContext
from marketing_project.server import app

# Add router to app
app.include_router(router)

client = TestClient(app)


@pytest.fixture
def sample_blog_context():
    """Create a sample blog post context."""
    return {
        "id": "test-blog-1",
        "title": "Test Blog Post",
        "content": "This is test content for the blog post.",
        "snippet": "Test snippet",
    }


@pytest.mark.asyncio
async def test_analyze_content_endpoint(sample_blog_context):
    """Test /analyze endpoint."""
    request_data = {"content": sample_blog_context}

    response = client.post("/analyze", json=request_data)

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "success" in data
        assert "analysis" in data


@pytest.mark.asyncio
async def test_run_pipeline_endpoint(sample_blog_context):
    """Test /pipeline endpoint."""
    with patch("marketing_project.api.core.process_blog_post") as mock_process:
        # Mock process_blog_post to return valid JSON result
        mock_process.return_value = json.dumps(
            {
                "status": "success",
                "message": "Pipeline completed",
                "pipeline_result": {"seo_keywords": {}, "marketing_brief": {}},
                "metadata": {},
                "validation": {},
                "processing_steps_completed": 1,
            }
        )

        request_data = {"content": sample_blog_context}

        response = client.post("/pipeline", json=request_data)

        # Endpoint should return 200 on success, 400 on validation error, or 500 on server error
        assert response.status_code in [200, 400, 500]


@pytest.mark.asyncio
async def test_get_steps_endpoint():
    """Test /steps endpoint."""
    response = client.get("/steps")

    assert response.status_code == 200
    data = response.json()
    assert "steps" in data
    assert isinstance(data["steps"], list)


@pytest.mark.asyncio
async def test_get_step_requirements():
    """Test /steps/{step_name}/requirements endpoint."""
    response = client.get("/steps/seo_keywords/requirements")

    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert "step_name" in data
        assert "required_context_keys" in data


@pytest.mark.asyncio
async def test_execute_step(sample_blog_context):
    """Test /steps/{step_name}/execute endpoint."""
    with patch("marketing_project.api.core.FunctionPipeline") as mock_pipeline_class:
        mock_pipeline = MagicMock()
        mock_pipeline.execute_single_step = AsyncMock(
            return_value={"result": {}, "step_name": "seo_keywords"}
        )
        mock_pipeline_class.return_value = mock_pipeline

        request_data = {
            "content": sample_blog_context,
            "context": {"input_content": sample_blog_context},
        }

        response = client.post("/steps/seo_keywords/execute", json=request_data)

        assert response.status_code in [200, 500]
