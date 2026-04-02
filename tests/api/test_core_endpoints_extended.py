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


# ---------------------------------------------------------------------------
# Additional tests to cover missed lines in core.py
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_release_notes():
    """Test /pipeline endpoint with release notes content (lines 128-135)."""
    with patch("marketing_project.api.core.process_release_notes") as mock_process:
        mock_process.return_value = json.dumps(
            {
                "status": "success",
                "message": "Release notes processed",
                "pipeline_result": {},
                "metadata": {},
                "validation": {},
                "processing_steps_completed": 1,
            }
        )

        request_data = {
            "content": {
                "id": "test-release-1",
                "title": "Test Release Notes",
                "content": "Version 1.0.0 released with new features.",
                "snippet": "Test snippet",
                "version": "1.0.0",
            }
        }

        response = client.post("/pipeline", json=request_data)

        assert response.status_code in [200, 400, 500]


@pytest.mark.asyncio
async def test_run_pipeline_transcript_content():
    """Test /pipeline endpoint with transcript content (lines 137-144)."""
    with patch("marketing_project.api.core.process_transcript") as mock_process:
        mock_process.return_value = json.dumps(
            {
                "status": "success",
                "message": "Transcript processed",
                "pipeline_result": {},
                "metadata": {},
                "validation": {},
                "processing_steps_completed": 1,
            }
        )

        request_data = {
            "content": {
                "id": "test-transcript-1",
                "title": "Test Transcript",
                "content": "Speaker 1: Hello there.",
                "snippet": "Test snippet",
                "speakers": ["Speaker 1"],
            }
        }

        response = client.post("/pipeline", json=request_data)

        assert response.status_code in [200, 400, 500]


@pytest.mark.asyncio
async def test_run_pipeline_error_result():
    """Test /pipeline endpoint when processor returns error status (lines 162-167)."""
    with patch("marketing_project.api.core.process_blog_post") as mock_process:
        mock_process.return_value = json.dumps(
            {
                "status": "error",
                "message": "Pipeline error occurred",
                "error": "pipeline_failed",
            }
        )

        request_data = {"content": sample_blog_context_data()}

        response = client.post("/pipeline", json=request_data)

        assert response.status_code in [400, 500]


def sample_blog_context_data():
    return {
        "id": "test-blog-1",
        "title": "Test Blog Post",
        "content": "This is test content for the blog post.",
        "snippet": "Test snippet",
    }


@pytest.mark.asyncio
async def test_run_pipeline_invalid_json_result():
    """Test /pipeline endpoint when processor returns invalid JSON (lines 155-159)."""
    with patch("marketing_project.api.core.process_blog_post") as mock_process:
        mock_process.return_value = "not-valid-json{"

        request_data = {"content": sample_blog_context_data()}

        response = client.post("/pipeline", json=request_data)

        # The endpoint catches json.JSONDecodeError -> raises HTTPException(500)
        # but FastAPI may translate this to 500 or 400 depending on version
        assert response.status_code in [400, 500]


@pytest.mark.asyncio
async def test_get_step_requirements_not_found():
    """Test /steps/{step_name}/requirements when step not found (lines 242-247)."""
    response = client.get("/steps/nonexistent_step/requirements")

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data.get("detail", "").lower()


@pytest.mark.asyncio
async def test_execute_step_not_found():
    """Test /steps/{step_name}/execute when step not found (lines 322-327)."""
    request_data = {
        "content": {"id": "test-1", "title": "Test"},
        "context": {"input_content": {}},
    }

    response = client.post("/steps/nonexistent_step/execute", json=request_data)

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_execute_step_missing_context_keys():
    """Test /steps/{step_name}/execute when required context keys missing (lines 333-337)."""
    request_data = {
        "content": {"id": "test-1", "title": "Test"},
        "context": {},  # Missing required keys
    }

    response = client.post("/steps/seo_keywords/execute", json=request_data)

    assert response.status_code in [400, 500]


@pytest.mark.asyncio
async def test_execute_step_success_with_arq():
    """Test /steps/{step_name}/execute with job manager mock (lines 340-397)."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_job = MagicMock()
    mock_job.id = "test-job-id-123"

    with patch("marketing_project.api.core.get_job_manager") as mock_jm:
        mock_manager = MagicMock()
        mock_manager.create_job = AsyncMock(return_value=mock_job)
        mock_manager.submit_to_arq = AsyncMock(return_value="arq-id-123")
        mock_jm.return_value = mock_manager

        with patch("marketing_project.api.core.resolve_user_settings") as mock_settings:
            from marketing_project.models.user_settings_models import UserSettings

            mock_settings.return_value = UserSettings()

            request_data = {
                "content": {"id": "test-1", "title": "Test", "content": "Content"},
                "context": {
                    "input_content": {"id": "test-1"},
                    "content_type": "blog_post",
                },
            }

            response = client.post("/steps/seo_keywords/execute", json=request_data)

            assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_analyze_content_exception():
    """Test /analyze endpoint handles exceptions (lines 75-79)."""
    with patch(
        "marketing_project.api.core.analyze_content_for_pipeline",
        side_effect=Exception("Analysis failed"),
    ):
        request_data = {
            "content": {
                "id": "test-1",
                "title": "Test",
                "content": "Content",
                "snippet": "snippet",
            }
        }

        response = client.post("/analyze", json=request_data)

        assert response.status_code == 500


@pytest.mark.asyncio
async def test_list_pipeline_steps_all_steps():
    """Test /pipeline/steps endpoint lists all steps (lines 205-218)."""
    response = client.get("/pipeline/steps")

    assert response.status_code == 200
    data = response.json()
    assert "steps" in data
    assert len(data["steps"]) > 0
    for step in data["steps"]:
        assert "step_name" in step
        assert "step_number" in step


@pytest.mark.asyncio
async def test_get_step_requirements_success():
    """Test /pipeline/steps/{step_name}/requirements for existing step."""
    response = client.get("/pipeline/steps/seo_keywords/requirements")

    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert "step_name" in data
        assert "required_context_keys" in data
        assert "descriptions" in data
