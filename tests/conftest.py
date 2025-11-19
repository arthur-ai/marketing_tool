import sys
from pathlib import Path

import pytest

# Add the project root to the Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration for Marketing Project


# Dummy LLM to avoid any real API/network calls in plugin tasks
class DummyLLM:
    def __call__(self, prompt, **kwargs):
        if "transcript" in prompt.lower():
            return "Test transcript analysis result"
        elif "blog" in prompt.lower():
            return "Test blog post analysis result"
        elif "release" in prompt.lower():
            return "Test release notes analysis result"
        return "Dummy response"


@pytest.fixture(autouse=True)
def patch_llms(monkeypatch):
    # Mock LLM for any plugin tasks that might use it
    # This can be extended when needed for specific plugin tests
    pass


# Plugin test configuration
@pytest.fixture(scope="session")
def plugin_test_config():
    """Configuration for plugin tests."""
    return {
        "test_timeout": 30,  # seconds
        "max_retries": 3,
        "parallel_workers": 4,
        "coverage_threshold": 90,
    }


# Sample data fixtures for testing
@pytest.fixture
def sample_article_data():
    """Sample article data for testing."""
    return {
        "id": "test-article-1",
        "title": "Test Article Title",
        "content": "This is a test article content with some sample text for testing purposes. It contains various elements that can be analyzed.",
        "snippet": "A test article snippet for unit testing purposes.",
        "meta_description": "A test article for unit testing",
        "author": "Test Author",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "tags": ["test", "sample", "article"],
        "category": "testing",
        "status": "published",
    }


@pytest.fixture
def sample_article_structure():
    """Sample article structure for testing write_article_content and add_supporting_elements."""
    return {
        "title": "Test Article Title",
        "meta_description": "A test article for unit testing",
        "introduction": {
            "hook": "Compelling opening statement",
            "problem_statement": "What problem does this solve?",
            "solution_preview": "What will readers learn?",
            "value_proposition": "Why should they read this?",
            "word_count": 150,
        },
        "main_sections": [
            {
                "heading": "H2: Main Topic 1",
                "subheadings": ["H3: Getting Started", "H3: Key Concepts"],
                "key_points": ["Point 1", "Point 2"],
                "word_count": 300,
                "seo_keywords": ["keyword1", "keyword2"],
            },
            {
                "heading": "H2: Main Topic 2",
                "subheadings": ["H3: Best Practices", "H3: Common Mistakes"],
                "key_points": ["Point 3", "Point 4"],
                "word_count": 300,
                "seo_keywords": ["keyword3", "keyword4"],
            },
        ],
        "conclusion": {
            "summary": "Key takeaways summary",
            "call_to_action": "What should readers do next?",
            "related_topics": "Suggestions for further reading",
            "word_count": 100,
        },
        "word_count_target": 1500,
        "reading_time_estimate": "5-7 minutes",
        "seo_optimization": {
            "primary_keyword": "test keyword",
            "secondary_keywords": ["secondary1", "secondary2"],
            "keyword_density_target": "1-3%",
            "title_optimization": "Include primary keyword in title",
            "heading_optimization": "Include keywords in H2 and H3 tags",
        },
        "content_pillars": ["Main Topic 1", "Main Topic 2"],
        "target_audience": {"demographics": "test audience"},
    }


@pytest.fixture
def sample_seo_keywords():
    """Sample SEO keywords for testing."""
    return [
        {"keyword": "artificial intelligence", "score": 95, "search_volume": 10000},
        {"keyword": "machine learning", "score": 88, "search_volume": 8000},
        {"keyword": "AI technology", "score": 82, "search_volume": 6000},
        {"keyword": "automation", "score": 75, "search_volume": 5000},
        {"keyword": "data science", "score": 70, "search_volume": 4000},
    ]


@pytest.fixture
def sample_content_context():
    """Sample ContentContext for testing."""
    from marketing_project.models.content_models import BlogPostContext

    return BlogPostContext(
        id="test-content-1",
        title="Test Content Context",
        content="This is test content for ContentContext testing.",
        snippet="Test snippet for content context",
        author="Test Author",
        tags=["test", "context"],
        category="testing",
        created_at="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_marketing_brief():
    """Sample marketing brief data for testing."""
    return {
        "id": "test-brief-1",
        "title": "Test Marketing Brief",
        "executive_summary": "This is a test marketing brief for unit testing purposes.",
        "target_audience": {
            "primary": "Business professionals",
            "secondary": "Technical decision makers",
        },
        "goals": ["Increase brand awareness", "Generate leads", "Drive conversions"],
        "budget": 50000,
        "timeline": "3 months",
        "content_pillars": ["Education", "Thought Leadership", "Product Information"],
        "key_messages": ["Innovation", "Reliability", "Customer Success"],
        "success_metrics": {
            "engagement_rate": "Target: >5%",
            "lead_generation": "Target: >100 leads",
            "conversion_rate": "Target: >2%",
        },
        "distribution_channels": ["Website", "LinkedIn", "Email"],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_blog_post():
    """Sample blog post data for testing."""
    from marketing_project.models.content_models import BlogPostContext

    return BlogPostContext(
        id="test-blog-1",
        title="How to Use AI in Marketing",
        content="This is a comprehensive guide about AI in marketing. It covers various aspects of artificial intelligence and how it can be applied to marketing strategies.",
        snippet="A comprehensive guide about AI in marketing",
        author="Test Author",
        tags=["AI", "Marketing", "Technology"],
        category="Technology",
        word_count=500,
        reading_time="3 minutes",
        created_at="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_transcript():
    """Sample transcript data for testing."""
    from marketing_project.models.content_models import TranscriptContext

    return TranscriptContext(
        id="test-transcript-1",
        title="AI Marketing Discussion",
        content="Speaker 1: Welcome to our discussion about AI in marketing. Speaker 2: Thank you for having me. This is an exciting topic.",
        snippet="A discussion about AI in marketing",
        speakers=["Speaker 1", "Speaker 2"],
        duration=1800,  # 30 minutes in seconds (int, not string)
        transcript_type="podcast",
        created_at="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_app_context_transcript():
    """Sample AppContext with transcript for testing."""
    from marketing_project.core.models import AppContext
    from marketing_project.models.content_models import TranscriptContext

    transcript = TranscriptContext(
        id="test-transcript-1",
        title="AI Marketing Discussion",
        content="Speaker 1: Welcome to our discussion about AI in marketing. Speaker 2: Thank you for having me.",
        snippet="A discussion about AI in marketing",
        speakers=["Speaker 1", "Speaker 2"],
        duration=1800,  # 30 minutes in seconds (int, not string)
        transcript_type="podcast",
        created_at="2024-01-01T00:00:00Z",
    )

    return AppContext(
        content=transcript,
        labels={"topic": "AI", "industry": "Marketing"},
        content_type="transcript",
    )


@pytest.fixture
def sample_release_notes():
    """Sample release notes data for testing."""
    from marketing_project.models.content_models import ReleaseNotesContext

    return ReleaseNotesContext(
        id="test-release-1",
        title="Version 2.0.0 Release Notes",
        content="This is a major release with new features and improvements.",
        snippet="Major release with new features",
        version="2.0.0",
        features=["New dashboard", "Enhanced security"],
        bug_fixes=["Fixed login issue", "Resolved memory leak"],
        changes=["Added new features", "Fixed bugs"],
        created_at="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_app_context_release():
    """Sample AppContext with release notes for testing."""
    from marketing_project.core.models import AppContext
    from marketing_project.models.content_models import ReleaseNotesContext

    release_notes = ReleaseNotesContext(
        id="test-release-1",
        title="Version 2.0.0 Release Notes",
        content="This is a major release with new features and improvements.",
        snippet="Major release with new features",
        version="2.0.0",
        features=["New dashboard", "Enhanced security"],
        bug_fixes=["Fixed login issue", "Resolved memory leak"],
        created_at="2024-01-01T00:00:00Z",
    )

    return AppContext(
        content=release_notes,
        labels={"version": "2.0.0", "type": "major"},
        content_type="release_notes",
    )


@pytest.fixture
def sample_app_context_blog():
    """Sample AppContext with blog post for testing."""
    from marketing_project.core.models import AppContext
    from marketing_project.models.content_models import BlogPostContext

    blog_post = BlogPostContext(
        id="test-blog-1",
        title="How to Use AI in Marketing",
        content="This is a comprehensive guide about AI in marketing.",
        snippet="A comprehensive guide about AI in marketing",
        author="Test Author",
        tags=["AI", "Marketing"],
        category="Technology",
        created_at="2024-01-01T00:00:00Z",
    )

    return AppContext(
        content=blog_post,
        labels={"topic": "AI", "industry": "Marketing"},
        content_type="blog_post",
    )


@pytest.fixture
def sample_email():
    """Sample email data for testing."""
    from marketing_project.core.models import EmailContext

    return EmailContext(
        id="test-email-1",
        thread_id="thread-123",
        subject="Test Email Subject",
        snippet="This is a test email snippet",
        body="This is the full body of the test email with some content for testing purposes.",
    )


@pytest.fixture
def sample_style_guide():
    """Sample style guide for testing."""
    return {
        "heading_style": "title_case",  # title_case, sentence_case
        "list_style": "bullet",  # bullet, numbered
        "paragraph_spacing": "double",  # single, double
        "quote_style": "blockquote",  # blockquote, markdown, html
        "code_style": "fenced",  # fenced, indented, inline
        "link_style": "markdown",  # markdown, html
        "emphasis_style": "bold_italic",  # bold, italic, bold_italic
    }


@pytest.fixture
def sample_available_processors():
    """Sample available processors for testing.

    Note: The system now uses deterministic processors instead of agents.
    """
    return {
        "blog_processor": {
            "name": "Blog Processor",
            "capabilities": ["blog_post", "article"],
            "type": "processor",
        },
        "transcript_processor": {
            "name": "Transcript Processor",
            "capabilities": ["transcripts", "transcript_processing"],
            "type": "processor",
        },
        "releasenotes_processor": {
            "name": "Release Notes Processor",
            "capabilities": ["release_notes", "changelog"],
            "type": "processor",
        },
    }


@pytest.fixture
def function_pipeline():
    """Create a FunctionPipeline instance for testing."""
    from unittest.mock import AsyncMock, MagicMock

    from marketing_project.services.function_pipeline import FunctionPipeline

    pipeline = FunctionPipeline(model="gpt-4o-mini", temperature=0.7, lang="en")

    # Mock the OpenAI client to avoid real API calls
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = []
    mock_response.choices[0].message.content = '{"status": "success"}'
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    pipeline.client = mock_client

    return pipeline


@pytest.fixture
async def job_manager():
    """Create a JobManager instance for testing."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from marketing_project.services.job_manager import JobManager

    manager = JobManager()

    # Mock Redis manager
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.setex = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)
    mock_redis.sadd = AsyncMock(return_value=1)
    mock_redis.smembers = AsyncMock(return_value=set())

    with patch.object(manager, "get_redis", return_value=mock_redis):
        yield manager


@pytest.fixture
def plugin_registry():
    """Create a PluginRegistry instance for testing."""
    from marketing_project.plugins.registry import PluginRegistry

    registry = PluginRegistry()
    return registry


@pytest.fixture
def mock_plugin_registry():
    """Create a mocked PluginRegistry for testing."""
    from unittest.mock import MagicMock

    from marketing_project.plugins.registry import PluginRegistry

    registry = PluginRegistry()
    # Don't auto-discover, just return empty registry for testing
    return registry


@pytest.fixture
def mock_retry_service():
    """Create a mock retry service for testing."""
    from unittest.mock import AsyncMock

    service = AsyncMock()
    service.retry_step = AsyncMock()
    return service


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual functions")
    config.addinivalue_line("markers", "integration: Integration tests between plugins")
    config.addinivalue_line("markers", "performance: Performance and efficiency tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "plugin: Plugin-specific tests")
