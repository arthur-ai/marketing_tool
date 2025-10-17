# API Reference

## Overview

The Arthur Marketing Generator provides a comprehensive API for content processing and analysis. All functions follow standardized patterns for input/output handling and error management.

## Core Models

### ContentContext

Base class for all content types.

```python
class ContentContext(BaseModel):
    id: str
    title: Optional[str] = None
    content: Optional[str] = None
    snippet: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    source_url: Optional[str] = None
```

### Specialized Content Types

#### BlogPostContext
```python
class BlogPostContext(ContentContext):
    author: Optional[str] = None
    tags: List[str] = []
    category: Optional[str] = None
    word_count: Optional[int] = None
```

#### TranscriptContext
```python
class TranscriptContext(ContentContext):
    speakers: List[str] = []
    duration: Optional[int] = None
    transcript_type: Optional[str] = None
```

#### ReleaseNotesContext
```python
class ReleaseNotesContext(ContentContext):
    version: Optional[str] = None
    release_date: Optional[datetime] = None
    changes: List[str] = []
    features: List[str] = []
    bug_fixes: List[str] = []
```

## Core Utilities

### ensure_content_context()

Ensures the input is a ContentContext object.

```python
def ensure_content_context(content: Union[ContentContext, Dict[str, Any]]) -> ContentContext:
    """
    Ensures the input is a ContentContext object. If it's a dictionary, it converts it.

    Args:
        content: Content context object or dictionary

    Returns:
        ContentContext: Content context object

    Raises:
        TypeError: If content type is not supported
    """
```

### create_standard_task_result()

Creates a standardized dictionary for task results.

```python
def create_standard_task_result(
    success: bool,
    task_name: str,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Creates a standardized dictionary for task results, compatible with Any Agent.

    Args:
        success: Whether the task was successful
        task_name: Name of the task
        data: Task-specific data
        error: Error message if unsuccessful
        metadata: Additional metadata

    Returns:
        Dict[str, Any]: Standardized task result
    """
```

### validate_content_for_processing()

Validates that content has the required structure for pipeline processing.

```python
def validate_content_for_processing(content: ContentContext) -> Dict[str, Any]:
    """
    Validates that the content has the required structure and basic information for pipeline processing.

    Args:
        content: Content context object

    Returns:
        Dict[str, Any]: Validation result with 'is_valid' (bool) and 'issues' (List[str])
    """
```

## Content Analysis Plugin

### analyze_content_for_pipeline()

Comprehensive content analysis for pipeline processing.

```python
def analyze_content_for_pipeline(content: Union[ContentContext, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyzes content for the new marketing pipeline workflow.

    Args:
        content: Content context object or dictionary

    Returns:
        Dict[str, Any]: Comprehensive content analysis for pipeline
    """
```

**Return Structure**:
```python
{
    "task_name": "analyze_content_for_pipeline",
    "success": True,
    "data": {
        "content_type": "blog_post",
        "content_quality": {
            "word_count": 1500,
            "has_title": True,
            "has_snippet": True,
            "readability_score": 75,
            "completeness_score": 85
        },
        "seo_potential": {
            "has_keywords": True,
            "title_optimization": "good",
            "content_structure": "excellent",
            "internal_linking_potential": "high"
        },
        "marketing_value": {
            "engagement_potential": "high",
            "conversion_potential": "medium",
            "shareability": "high",
            "target_audience_appeal": "excellent"
        },
        "processing_recommendations": [
            "Add more internal links",
            "Improve meta description"
        ],
        "pipeline_ready": True
    },
    "metadata": {
        "content_id": "test-1",
        "content_title": "Test Article",
        "content_type": "blogpost",
        "created_at": "2024-01-01T00:00:00",
        "has_snippet": True,
        "has_metadata_field": True
    }
}
```

## SEO Keywords Plugin

### extract_primary_keywords()

Extract main SEO keywords from content.

```python
def extract_primary_keywords(
    content: Union[ContentContext, Dict[str, Any]],
    max_keywords: int = 5
) -> Dict[str, Any]:
    """
    Extracts primary SEO keywords from content.

    Args:
        content: Content context object or dictionary
        max_keywords: Maximum number of keywords to extract

    Returns:
        Dict[str, Any]: Primary keywords with scores
    """
```

**Return Structure**:
```python
{
    "task_name": "extract_primary_keywords",
    "success": True,
    "data": {
        "keywords": [
            {
                "keyword": "marketing automation",
                "frequency": 15,
                "score": 8.5,
                "position": 1
            },
            {
                "keyword": "content strategy",
                "frequency": 12,
                "score": 7.8,
                "position": 2
            }
        ],
        "total_keywords": 2,
        "extraction_method": "tfidf"
    },
    "metadata": {
        "content_id": "test-1",
        "content_title": "Test Article",
        "max_keywords": 5,
        "extraction_time": "2024-01-01T00:00:00"
    }
}
```

### analyze_keyword_density()

Analyze keyword density and distribution.

```python
def analyze_keyword_density(
    content: Union[ContentContext, Dict[str, Any]],
    keywords: List[str]
) -> Dict[str, Any]:
    """
    Analyzes keyword density and distribution in content.

    Args:
        content: Content context object or dictionary
        keywords: List of keywords to analyze

    Returns:
        Dict[str, Any]: Keyword density analysis
    """
```

## Marketing Brief Plugin

### generate_brief_outline()

Generate structured marketing brief outline.

```python
def generate_brief_outline(
    content: Union[ContentContext, Dict[str, Any]],
    seo_keywords: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates a structured marketing brief outline based on content analysis.

    Args:
        content: Content context object or dictionary
        seo_keywords: List of SEO keywords from previous analysis

    Returns:
        Dict[str, Any]: Structured marketing brief outline
    """
```

**Return Structure**:
```python
{
    "task_name": "generate_brief_outline",
    "success": True,
    "data": {
        "title": "Marketing Automation Guide",
        "content_type": "blog_post",
        "executive_summary": "A comprehensive guide to marketing automation...",
        "target_audience": {
            "primary": "marketing professionals",
            "secondary": "business owners",
            "demographics": {
                "age_range": "25-45",
                "experience_level": "intermediate"
            }
        },
        "content_objectives": {
            "primary": "educate on marketing automation",
            "secondary": "generate leads",
            "tertiary": "establish thought leadership"
        },
        "key_messages": [
            "Marketing automation saves time",
            "ROI is measurable",
            "Implementation is straightforward"
        ],
        "seo_strategy": {
            "primary_keywords": ["marketing automation", "content strategy"],
            "secondary_keywords": ["lead generation", "email marketing"],
            "keyword_density_target": "1-3%",
            "meta_description_length": "150-160 characters"
        },
        "content_requirements": {
            "word_count": 1500,
            "sections": 5,
            "images": 3,
            "call_to_actions": 2
        },
        "success_metrics": {
            "engagement": "time_on_page > 3 minutes",
            "conversion": "lead_form_submissions > 5%",
            "sharing": "social_shares > 50"
        },
        "timeline": {
            "research": "2 days",
            "writing": "3 days",
            "review": "1 day",
            "publication": "1 day"
        },
        "resources_needed": [
            "content writer",
            "graphic designer",
            "SEO specialist"
        ],
        "distribution_channels": [
            "company blog",
            "LinkedIn",
            "email newsletter"
        ],
        "created_at": "2024-01-01T00:00:00"
    },
    "metadata": {
        "content_id": "test-1",
        "content_title": "Test Article",
        "content_type": "blogpost",
        "seo_keywords_count": 2
    }
}
```

## Article Generation Plugin

### generate_article_structure()

Generate article structure and outline.

```python
def generate_article_structure(
    marketing_brief: Union[Dict[str, Any], ContentContext],
    seo_keywords: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates article structure and outline based on marketing brief.

    Args:
        marketing_brief: Marketing brief dictionary or ContentContext
        seo_keywords: List of SEO keywords for optimization

    Returns:
        Dict[str, Any]: Article structure and outline
    """
```

**Return Structure**:
```python
{
    "task_name": "generate_article_structure",
    "success": True,
    "data": {
        "title": "Marketing Automation: Complete Guide",
        "meta_description": "Learn about marketing automation with our comprehensive guide...",
        "introduction": {
            "hook": "Compelling opening statement",
            "problem_statement": "What problem does this solve?",
            "solution_preview": "What will readers learn?",
            "value_proposition": "Why should they read this?",
            "word_count": 150
        },
        "main_sections": [
            {
                "heading": "H2: Getting Started with Marketing Automation",
                "subheadings": [
                    "H3: Getting Started",
                    "H3: Key Concepts"
                ],
                "key_points": [],
                "word_count": 300,
                "seo_keywords": ["marketing automation", "getting started"]
            }
        ],
        "conclusion": {
            "summary": "Key takeaways summary",
            "call_to_action": "What should readers do next?",
            "related_topics": "Suggestions for further reading",
            "word_count": 100
        },
        "word_count_target": 1500,
        "reading_time_estimate": "5-7 minutes",
        "seo_optimization": {
            "primary_keyword": "marketing automation",
            "secondary_keywords": ["content strategy", "lead generation"],
            "keyword_density_target": "1-3%",
            "title_optimization": "Include primary keyword in title",
            "heading_optimization": "Include keywords in H2 and H3 tags"
        },
        "content_pillars": ["Getting Started", "Best Practices", "Advanced Techniques"],
        "target_audience": {
            "primary": "marketing professionals",
            "secondary": "business owners"
        }
    },
    "metadata": {
        "content_pillars_count": 3,
        "main_sections_count": 3,
        "seo_keywords_count": 2
    }
}
```

## SEO Optimization Plugin

### optimize_title_tags()

Optimize title tags for SEO.

```python
def optimize_title_tags(
    article: Union[Dict[str, Any], ContentContext],
    keywords: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Optimizes title tags for better SEO performance.

    Args:
        article: Article dictionary with content or ContentContext
        keywords: List of SEO keywords for optimization

    Returns:
        Dict[str, Any]: Optimized title tags and recommendations
    """
```

**Return Structure**:
```python
{
    "task_name": "optimize_title_tags",
    "success": True,
    "data": {
        "current_title": "Marketing Automation Guide",
        "optimized_titles": [
            "Marketing Automation: Complete Guide",
            "How to Marketing Automation: Expert Tips",
            "Ultimate Marketing Automation Guide"
        ],
        "recommendations": [
            "Title length is optimal",
            "Consider adding power words to increase click-through rate"
        ],
        "seo_score": 85,
        "character_count": 45
    },
    "metadata": {
        "current_title_length": 45,
        "optimized_titles_count": 3,
        "seo_score": 85
    }
}
```

## Internal Docs Plugin

### analyze_content_gaps()

Analyze content for gaps that could be filled with internal documents.

```python
def analyze_content_gaps(
    article: Union[Dict[str, Any], ContentContext],
    existing_docs: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyzes content for gaps that could be filled with internal documents.

    Args:
        article: Article dictionary with content or ContentContext
        existing_docs: List of existing internal documents

    Returns:
        Dict[str, Any]: Content gap analysis results
    """
```

**Return Structure**:
```python
{
    "task_name": "analyze_content_gaps",
    "success": True,
    "data": {
        "content_gaps": [
            "Missing introduction section",
            "Missing best practices section"
        ],
        "missing_topics": [
            "for more information about implementation",
            "see also related tools"
        ],
        "unexplained_concepts": [
            "API",
            "SDK",
            "framework"
        ],
        "suggested_docs": [
            {
                "type": "glossary",
                "title": "Marketing Automation - Technical Glossary",
                "description": "Definitions and explanations for technical terms: API, SDK, framework",
                "priority": "high",
                "target_audience": "beginners"
            },
            {
                "type": "troubleshooting",
                "title": "Marketing Automation - Troubleshooting Guide",
                "description": "Common issues and solutions related to the main topic",
                "priority": "medium",
                "target_audience": "all"
            }
        ],
        "priority_level": "medium"
    },
    "metadata": {
        "content_gaps_count": 2,
        "unexplained_concepts_count": 3,
        "suggested_docs_count": 2,
        "priority_level": "medium"
    }
}
```

## Content Formatting Plugin

### apply_formatting_rules()

Apply consistent formatting rules to content.

```python
def apply_formatting_rules(
    article: Union[Dict[str, Any], ContentContext],
    style_guide: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Applies consistent formatting rules to content.

    Args:
        article: Article dictionary with content or ContentContext
        style_guide: Style guide configuration

    Returns:
        Dict[str, Any]: Formatted content with applied rules
    """
```

**Return Structure**:
```python
{
    "task_name": "apply_formatting_rules",
    "success": True,
    "data": {
        "content": "# Marketing Automation: Complete Guide\n\n## Introduction\n\nMarketing automation is...",
        "title": "Marketing Automation: Complete Guide",
        "meta_description": "Learn about marketing automation...",
        "formatting_applied": True,
        "style_guide_used": {
            "heading_style": "title_case",
            "list_style": "bullet",
            "paragraph_spacing": "double",
            "quote_style": "blockquote",
            "code_style": "fenced",
            "link_style": "markdown",
            "emphasis_style": "bold_italic"
        }
    },
    "metadata": {
        "style_guide_used": {
            "heading_style": "title_case",
            "list_style": "bullet"
        },
        "content_length": 1500,
        "formatting_applied": True
    }
}
```

## Error Handling

All API functions follow a consistent error handling pattern:

```python
{
    "task_name": "function_name",
    "success": False,
    "error": "Detailed error message",
    "timestamp": "2024-01-01T00:00:00",
    "data": {},
    "metadata": {}
}
```

Common error types:
- **ValidationError**: Input validation failed
- **ProcessingError**: Content processing failed
- **ConfigurationError**: Configuration issue
- **NetworkError**: Network-related error
- **TimeoutError**: Operation timed out

## Rate Limiting

API functions may be subject to rate limiting:

- **Default Rate Limit**: 100 requests per minute
- **Burst Limit**: 20 requests per second
- **Rate Limit Headers**: Included in response headers
- **Retry Logic**: Built-in retry with exponential backoff

## Authentication

API functions require authentication:

- **API Key**: Required for all requests
- **Bearer Token**: Alternative authentication method
- **Rate Limiting**: Per-API-key rate limiting
- **Usage Tracking**: Request tracking and monitoring

## SDKs and Libraries

Official SDKs are available for:

- **Python**: `pip install arthur-marketing-generator`
- **JavaScript**: `npm install @arthur/marketing-generator`
- **Go**: `go get github.com/arthur/marketing-generator`
- **Java**: Available in Maven Central

## Examples

### Python Example

```python
from arthur_marketing_generator import MarketingGenerator
from arthur_marketing_generator.models import BlogPostContext

# Initialize the generator
generator = MarketingGenerator(api_key="your-api-key")

# Create content context
content = BlogPostContext(
    id="example-1",
    title="Marketing Automation Guide",
    content="Marketing automation is a powerful tool...",
    author="John Doe",
    tags=["marketing", "automation"],
    category="tutorial"
)

# Run the full pipeline
result = generator.run_pipeline(content)

# Or run individual steps
analysis = generator.analyze_content(content)
keywords = generator.extract_keywords(content)
brief = generator.generate_brief(content, keywords)
```

### JavaScript Example

```javascript
import { MarketingGenerator } from '@arthur/marketing-generator';

// Initialize the generator
const generator = new MarketingGenerator({
  apiKey: 'your-api-key'
});

// Create content context
const content = {
  id: 'example-1',
  title: 'Marketing Automation Guide',
  content: 'Marketing automation is a powerful tool...',
  author: 'John Doe',
  tags: ['marketing', 'automation'],
  category: 'tutorial'
};

// Run the full pipeline
const result = await generator.runPipeline(content);

// Or run individual steps
const analysis = await generator.analyzeContent(content);
const keywords = await generator.extractKeywords(content);
const brief = await generator.generateBrief(content, keywords);
```
