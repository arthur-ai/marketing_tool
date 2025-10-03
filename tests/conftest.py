import pytest
import sys
from pathlib import Path

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
        "coverage_threshold": 90
    }

# Sample data fixtures for testing
@pytest.fixture
def sample_article_data():
    """Sample article data for testing."""
    return {
        'id': 'test-article-1',
        'title': 'Test Article Title',
        'content': 'This is a test article content with some sample text for testing purposes. It contains various elements that can be analyzed.',
        'snippet': 'A test article snippet for unit testing purposes.',
        'meta_description': 'A test article for unit testing',
        'author': 'Test Author',
        'created_at': '2024-01-01T00:00:00Z',
        'updated_at': '2024-01-01T00:00:00Z',
        'tags': ['test', 'sample', 'article'],
        'category': 'testing',
        'status': 'published'
    }

@pytest.fixture
def sample_article_structure():
    """Sample article structure for testing write_article_content and add_supporting_elements."""
    return {
        'title': 'Test Article Title',
        'meta_description': 'A test article for unit testing',
        'introduction': {
            'hook': 'Compelling opening statement',
            'problem_statement': 'What problem does this solve?',
            'solution_preview': 'What will readers learn?',
            'value_proposition': 'Why should they read this?',
            'word_count': 150
        },
        'main_sections': [
            {
                'heading': 'H2: Main Topic 1',
                'subheadings': ['H3: Getting Started', 'H3: Key Concepts'],
                'key_points': ['Point 1', 'Point 2'],
                'word_count': 300,
                'seo_keywords': ['keyword1', 'keyword2']
            },
            {
                'heading': 'H2: Main Topic 2',
                'subheadings': ['H3: Best Practices', 'H3: Common Mistakes'],
                'key_points': ['Point 3', 'Point 4'],
                'word_count': 300,
                'seo_keywords': ['keyword3', 'keyword4']
            }
        ],
        'conclusion': {
            'summary': 'Key takeaways summary',
            'call_to_action': 'What should readers do next?',
            'related_topics': 'Suggestions for further reading',
            'word_count': 100
        },
        'word_count_target': 1500,
        'reading_time_estimate': '5-7 minutes',
        'seo_optimization': {
            'primary_keyword': 'test keyword',
            'secondary_keywords': ['secondary1', 'secondary2'],
            'keyword_density_target': '1-3%',
            'title_optimization': 'Include primary keyword in title',
            'heading_optimization': 'Include keywords in H2 and H3 tags'
        },
        'content_pillars': ['Main Topic 1', 'Main Topic 2'],
        'target_audience': {'demographics': 'test audience'}
    }

@pytest.fixture
def sample_seo_keywords():
    """Sample SEO keywords for testing."""
    return [
        {'keyword': 'artificial intelligence', 'score': 95, 'search_volume': 10000},
        {'keyword': 'machine learning', 'score': 88, 'search_volume': 8000},
        {'keyword': 'AI technology', 'score': 82, 'search_volume': 6000},
        {'keyword': 'automation', 'score': 75, 'search_volume': 5000},
        {'keyword': 'data science', 'score': 70, 'search_volume': 4000}
    ]

@pytest.fixture
def sample_content_context():
    """Sample ContentContext for testing."""
    from marketing_project.core.models import BlogPostContext
    
    return BlogPostContext(
        id='test-content-1',
        title='Test Content Context',
        content='This is test content for ContentContext testing.',
        snippet='Test snippet for content context',
        author='Test Author',
        tags=['test', 'context'],
        category='testing',
        created_at='2024-01-01T00:00:00Z'
    )

@pytest.fixture
def sample_marketing_brief():
    """Sample marketing brief data for testing."""
    return {
        'id': 'test-brief-1',
        'title': 'Test Marketing Brief',
        'executive_summary': 'This is a test marketing brief for unit testing purposes.',
        'target_audience': {
            'primary': 'Business professionals',
            'secondary': 'Technical decision makers'
        },
        'goals': ['Increase brand awareness', 'Generate leads', 'Drive conversions'],
        'budget': 50000,
        'timeline': '3 months',
        'content_pillars': ['Education', 'Thought Leadership', 'Product Information'],
        'key_messages': ['Innovation', 'Reliability', 'Customer Success'],
        'success_metrics': {
            'engagement_rate': 'Target: >5%',
            'lead_generation': 'Target: >100 leads',
            'conversion_rate': 'Target: >2%'
        },
        'distribution_channels': ['Website', 'LinkedIn', 'Email'],
        'created_at': '2024-01-01T00:00:00Z',
        'updated_at': '2024-01-01T00:00:00Z'
    }

@pytest.fixture
def sample_blog_post():
    """Sample blog post data for testing."""
    from marketing_project.core.models import BlogPostContext
    
    return BlogPostContext(
        id='test-blog-1',
        title='How to Use AI in Marketing',
        content='This is a comprehensive guide about AI in marketing. It covers various aspects of artificial intelligence and how it can be applied to marketing strategies.',
        snippet='A comprehensive guide about AI in marketing',
        author='Test Author',
        tags=['AI', 'Marketing', 'Technology'],
        category='Technology',
        word_count=500,
        reading_time='3 minutes',
        created_at='2024-01-01T00:00:00Z'
    )

@pytest.fixture
def sample_transcript():
    """Sample transcript data for testing."""
    from marketing_project.core.models import TranscriptContext
    
    return TranscriptContext(
        id='test-transcript-1',
        title='AI Marketing Discussion',
        content='Speaker 1: Welcome to our discussion about AI in marketing. Speaker 2: Thank you for having me. This is an exciting topic.',
        snippet='A discussion about AI in marketing',
        speakers=['Speaker 1', 'Speaker 2'],
        duration='30:00',
        transcript_type='podcast',
        timestamps={'00:00': 'Introduction', '15:00': 'Main discussion'},
        created_at='2024-01-01T00:00:00Z'
    )

@pytest.fixture
def sample_app_context_transcript():
    """Sample AppContext with transcript for testing."""
    from marketing_project.core.models import AppContext, TranscriptContext
    
    transcript = TranscriptContext(
        id='test-transcript-1',
        title='AI Marketing Discussion',
        content='Speaker 1: Welcome to our discussion about AI in marketing. Speaker 2: Thank you for having me.',
        snippet='A discussion about AI in marketing',
        speakers=['Speaker 1', 'Speaker 2'],
        duration='30:00',
        transcript_type='podcast',
        created_at='2024-01-01T00:00:00Z'
    )
    
    return AppContext(
        content=transcript,
        labels={'topic': 'AI', 'industry': 'Marketing'},
        content_type='transcript'
    )

@pytest.fixture
def sample_release_notes():
    """Sample release notes data for testing."""
    from marketing_project.core.models import ReleaseNotesContext
    
    return ReleaseNotesContext(
        id='test-release-1',
        title='Version 2.0.0 Release Notes',
        content='This is a major release with new features and improvements.',
        snippet='Major release with new features',
        version='2.0.0',
        features=['New dashboard', 'Enhanced security'],
        bug_fixes=['Fixed login issue', 'Resolved memory leak'],
        breaking_changes=['Removed deprecated API'],
        changes=['Added new features', 'Fixed bugs'],
        created_at='2024-01-01T00:00:00Z'
    )

@pytest.fixture
def sample_app_context_release():
    """Sample AppContext with release notes for testing."""
    from marketing_project.core.models import AppContext, ReleaseNotesContext
    
    release_notes = ReleaseNotesContext(
        id='test-release-1',
        title='Version 2.0.0 Release Notes',
        content='This is a major release with new features and improvements.',
        snippet='Major release with new features',
        version='2.0.0',
        features=['New dashboard', 'Enhanced security'],
        bug_fixes=['Fixed login issue', 'Resolved memory leak'],
        created_at='2024-01-01T00:00:00Z'
    )
    
    return AppContext(
        content=release_notes,
        labels={'version': '2.0.0', 'type': 'major'},
        content_type='release_notes'
    )

@pytest.fixture
def sample_app_context_blog():
    """Sample AppContext with blog post for testing."""
    from marketing_project.core.models import AppContext, BlogPostContext
    
    blog_post = BlogPostContext(
        id='test-blog-1',
        title='How to Use AI in Marketing',
        content='This is a comprehensive guide about AI in marketing.',
        snippet='A comprehensive guide about AI in marketing',
        author='Test Author',
        tags=['AI', 'Marketing'],
        category='Technology',
        created_at='2024-01-01T00:00:00Z'
    )
    
    return AppContext(
        content=blog_post,
        labels={'topic': 'AI', 'industry': 'Marketing'},
        content_type='blog_post'
    )

@pytest.fixture
def sample_email():
    """Sample email data for testing."""
    from marketing_project.core.models import EmailContext
    
    return EmailContext(
        id='test-email-1',
        thread_id='thread-123',
        subject='Test Email Subject',
        snippet='This is a test email snippet',
        body='This is the full body of the test email with some content for testing purposes.'
    )

@pytest.fixture
def sample_style_guide():
    """Sample style guide for testing."""
    return {
        'heading_style': 'title_case',  # title_case, sentence_case
        'list_style': 'bullet',  # bullet, numbered
        'paragraph_spacing': 'double',  # single, double
        'quote_style': 'blockquote',  # blockquote, markdown, html
        'code_style': 'fenced',  # fenced, indented, inline
        'link_style': 'markdown',  # markdown, html
        'emphasis_style': 'bold_italic'  # bold, italic, bold_italic
    }

@pytest.fixture
def sample_available_agents():
    """Sample available agents for testing - only includes actual agents from the codebase."""
    return {
        'article_generation_agent': {
            'name': 'Article Generation Agent',
            'capabilities': ['article_generation', 'content_creation'],
            'priority': 'high'
        },
        'blog_agent': {
            'name': 'Blog Agent',
            'capabilities': ['blog_post', 'article'],
            'priority': 'high'
        },
        'content_formatting_agent': {
            'name': 'Content Formatting Agent',
            'capabilities': ['content_formatting', 'formatting'],
            'priority': 'high'
        },
        'content_pipeline_agent': {
            'name': 'Content Pipeline Agent',
            'capabilities': ['content_pipeline', 'workflow'],
            'priority': 'high'
        },
        'internal_docs_agent': {
            'name': 'Internal Docs Agent',
            'capabilities': ['internal_docs', 'documentation'],
            'priority': 'high'
        },
        'marketing_agent': {
            'name': 'Marketing Agent',
            'capabilities': ['marketing', 'promotion'],
            'priority': 'high'
        },
        'marketing_brief_agent': {
            'name': 'Marketing Brief Agent',
            'capabilities': ['marketing_brief', 'brief_creation'],
            'priority': 'high'
        },
        'releasenotes_agent': {
            'name': 'Release Notes Agent',
            'capabilities': ['release_notes', 'changelog'],
            'priority': 'high'
        },
        'seo_keywords_agent': {
            'name': 'SEO Keywords Agent',
            'capabilities': ['seo_keywords', 'keyword_research'],
            'priority': 'high'
        },
        'seo_optimization_agent': {
            'name': 'SEO Optimization Agent',
            'capabilities': ['seo_optimization', 'seo'],
            'priority': 'high'
        },
        'transcripts_agent': {
            'name': 'Transcripts Agent',
            'capabilities': ['transcripts', 'transcript_processing'],
            'priority': 'high'
        }
    }

# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual functions")
    config.addinivalue_line("markers", "integration: Integration tests between plugins")
    config.addinivalue_line("markers", "performance: Performance and efficiency tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "plugin: Plugin-specific tests")