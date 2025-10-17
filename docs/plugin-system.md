# Plugin System

## Overview

The Arthur Marketing Generator uses a modular plugin architecture that allows for easy extension and customization. Each plugin handles a specific aspect of content processing and follows standardized patterns for input/output handling and error management.

## Plugin Architecture

All plugins follow a consistent structure:

```
src/marketing_project/plugins/
├── plugin_name/
│   ├── __init__.py
│   └── tasks.py
```

Each plugin contains:
- **`__init__.py`**: Plugin initialization and exports
- **`tasks.py`**: Core plugin functions and logic

## Plugin Standards

All plugins follow these standards:

### 1. Function Signatures
```python
def task_function(
    content: Union[ContentContext, Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """Function docstring."""
    try:
        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Validate content
        validation = validate_content_for_processing(content_obj)
        if not validation['is_valid']:
            return create_standard_task_result(
                success=False,
                error=f"Validation failed: {', '.join(validation['issues'])}",
                task_name='task_function'
            )

        # Process content
        result_data = process_content(content_obj, **kwargs)

        # Return standardized result
        return create_standard_task_result(
            success=True,
            data=result_data,
            task_name='task_function',
            metadata=extract_content_metadata_for_pipeline(content_obj)
        )

    except Exception as e:
        logger.error(f"Error in task_function: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Task failed: {str(e)}",
            task_name='task_function'
        )
```

### 2. Standardized Return Format
```python
{
    "task_name": "task_function",
    "success": True/False,
    "timestamp": "ISO8601 timestamp",
    "data": { /* task-specific data */ },
    "error": "Error message (if success=False)",
    "metadata": { /* content metadata */ }
}
```

### 3. Error Handling
- All exceptions are caught and logged
- Standardized error messages
- Graceful degradation
- Validation before processing

## Available Plugins

### 1. Content Analysis Plugin

**Purpose**: Initial content analysis and quality assessment

**Location**: `plugins/content_analysis/`

**Key Functions**:
- `analyze_content_for_pipeline()` - Comprehensive content analysis
- `analyze_content_type()` - Determine content type
- `extract_content_metadata()` - Extract metadata
- `validate_content_structure()` - Validate content structure
- `calculate_basic_readability()` - Calculate readability scores
- `assess_content_completeness()` - Assess completeness
- `extract_potential_keywords()` - Extract potential keywords
- `assess_title_seo()` - Assess title SEO potential
- `assess_content_structure()` - Assess content structure quality
- `assess_linking_potential()` - Assess internal linking potential
- `assess_engagement_potential()` - Assess engagement potential
- `assess_conversion_potential()` - Assess conversion potential
- `assess_shareability()` - Assess shareability potential
- `assess_audience_appeal()` - Assess target audience appeal

**Input**: Raw content (text, documents, etc.)
**Output**: Comprehensive content analysis with quality metrics

### 2. SEO Keywords Plugin

**Purpose**: Extract and analyze SEO keywords for optimization

**Location**: `plugins/seo_keywords/`

**Key Functions**:
- `extract_primary_keywords()` - Extract main SEO keywords
- `extract_secondary_keywords()` - Extract supporting keywords and LSI terms
- `analyze_keyword_density()` - Analyze keyword density and distribution
- `generate_keyword_suggestions()` - Generate additional keyword suggestions
- `optimize_keyword_placement()` - Optimize keyword placement
- `calculate_keyword_scores()` - Calculate keyword relevance scores

**Input**: Content analysis results
**Output**: Structured keyword data with relevance scores

### 3. Marketing Brief Plugin

**Purpose**: Create comprehensive marketing brief and strategy

**Location**: `plugins/marketing_brief/`

**Key Functions**:
- `generate_brief_outline()` - Create structured marketing brief
- `define_target_audience()` - Define target audience based on content
- `set_content_objectives()` - Set content objectives and KPIs
- `create_content_strategy()` - Create comprehensive content strategy
- `analyze_competitor_content()` - Analyze competitor content
- `generate_content_calendar_suggestions()` - Suggest content calendar items

**Input**: Content analysis + SEO keywords
**Output**: Detailed marketing brief and strategy

### 4. Article Generation Plugin

**Purpose**: Generate high-quality article content

**Location**: `plugins/article_generation/`

**Key Functions**:
- `generate_article_structure()` - Create article structure and outline
- `write_article_content()` - Write main article content
- `add_supporting_elements()` - Add images, quotes, callouts
- `review_article_quality()` - Review and improve article quality
- `optimize_article_flow()` - Optimize article flow and transitions
- `add_call_to_actions()` - Add strategic CTAs

**Input**: Marketing brief + SEO keywords
**Output**: Structured article content

### 5. SEO Optimization Plugin

**Purpose**: Apply advanced SEO optimizations

**Location**: `plugins/seo_optimization/`

**Key Functions**:
- `optimize_title_tags()` - Optimize title tags for SEO
- `optimize_meta_descriptions()` - Optimize meta descriptions
- `optimize_headings()` - Optimize heading structure
- `optimize_content_structure()` - Optimize overall content structure
- `add_internal_links()` - Add strategic internal links
- `analyze_seo_performance()` - Analyze comprehensive SEO performance

**Input**: Generated article + SEO keywords
**Output**: SEO-optimized content

### 6. Internal Docs Plugin

**Purpose**: Suggest internal documents and cross-references

**Location**: `plugins/internal_docs/`

**Key Functions**:
- `analyze_content_gaps()` - Analyze content for missing information
- `suggest_related_docs()` - Suggest related internal documents
- `identify_cross_references()` - Identify cross-reference opportunities
- `generate_doc_suggestions()` - Generate specific document suggestions
- `create_content_relationships()` - Create content relationships
- `optimize_internal_linking()` - Optimize internal linking strategy

**Input**: SEO-optimized content
**Output**: Internal documentation suggestions

### 7. Content Formatting Plugin

**Purpose**: Final formatting and publication preparation

**Location**: `plugins/content_formatting/`

**Key Functions**:
- `apply_formatting_rules()` - Apply consistent formatting rules
- `optimize_readability()` - Optimize content for readability
- `add_visual_elements()` - Add visual elements and formatting
- `finalize_content()` - Finalize content for publication
- `validate_formatting()` - Validate formatting compliance
- `generate_publication_ready_content()` - Generate final publication-ready content

**Input**: SEO-optimized content + internal docs
**Output**: Publication-ready content

## Creating Custom Plugins

To create a custom plugin:

1. **Create Plugin Directory**:
   ```bash
   mkdir src/marketing_project/plugins/your_plugin_name
   ```

2. **Create `__init__.py`**:
   ```python
   """
   Your Plugin Name - Description
   """

   from .tasks import *
   ```

3. **Create `tasks.py`**:
   ```python
   """
   Your Plugin Name processing tasks.
   """

   import logging
   from typing import Dict, Any, Union
   from marketing_project.core.models import ContentContext
   from marketing_project.core.utils import (
       ensure_content_context, create_standard_task_result,
       validate_content_for_processing, extract_content_metadata_for_pipeline
   )

   logger = logging.getLogger("marketing_project.plugins.your_plugin_name")

   def your_task_function(content: Union[ContentContext, Dict[str, Any]]) -> Dict[str, Any]:
       """Your task function."""
       try:
           # Ensure content is a ContentContext object
           content_obj = ensure_content_context(content)

           # Validate content
           validation = validate_content_for_processing(content_obj)
           if not validation['is_valid']:
               return create_standard_task_result(
                   success=False,
                   error=f"Validation failed: {', '.join(validation['issues'])}",
                   task_name='your_task_function'
               )

           # Process content
           result_data = process_your_content(content_obj)

           # Return standardized result
           return create_standard_task_result(
               success=True,
               data=result_data,
               task_name='your_task_function',
               metadata=extract_content_metadata_for_pipeline(content_obj)
           )

       except Exception as e:
           logger.error(f"Error in your_task_function: {str(e)}")
           return create_standard_task_result(
               success=False,
               error=f"Task failed: {str(e)}",
               task_name='your_task_function'
           )
   ```

4. **Add to Pipeline Configuration**:
   ```yaml
   # In config/pipeline.yml
   pipelines:
     default:
       - AnalyzeContent
       - YourPluginName
       - FormatContent
   ```

5. **Create Agent Integration**:
   ```python
   # In src/marketing_project/agents/your_agent.py
   from marketing_project.plugins.your_plugin_name import tasks as your_tasks

   tools = [
       your_tasks.your_task_function,
       # ... other tools
   ]
   ```

## Plugin Development Best Practices

1. **Follow Standards**: Always follow the standardized function signatures and return formats
2. **Error Handling**: Include comprehensive error handling and logging
3. **Validation**: Validate inputs before processing
4. **Documentation**: Document all functions with clear docstrings
5. **Testing**: Write tests for all plugin functions
6. **Type Hints**: Use proper type hints for all parameters and return values
7. **Logging**: Use appropriate logging levels for different types of messages
8. **Performance**: Consider performance implications of your plugin functions
9. **Compatibility**: Ensure compatibility with the Any Agent framework
10. **Reusability**: Design functions to be reusable across different contexts

## Plugin Testing

Each plugin should include comprehensive tests:

```python
# tests/plugins/test_your_plugin.py
import pytest
from marketing_project.plugins.your_plugin_name import tasks
from marketing_project.core.models import ContentContext

def test_your_task_function():
    """Test your task function."""
    content = ContentContext(
        id="test-1",
        title="Test Content",
        content="This is test content."
    )

    result = tasks.your_task_function(content)

    assert isinstance(result, dict)
    assert result.get('success') is not None
    assert result.get('task_name') == 'your_task_function'
    assert 'data' in result
    assert 'metadata' in result
```

## Plugin Configuration

Plugins can be configured through:

1. **Environment Variables**: For global plugin settings
2. **Configuration Files**: For plugin-specific settings
3. **Function Parameters**: For runtime configuration
4. **Pipeline Configuration**: For pipeline-specific settings

## Monitoring and Debugging

Plugins include built-in monitoring capabilities:

- **Logging**: Comprehensive logging at appropriate levels
- **Metrics**: Performance and quality metrics
- **Error Tracking**: Detailed error information
- **Debugging**: Debug information for troubleshooting
- **Profiling**: Performance profiling capabilities

## Integration with Any Agent

All plugins are designed to work seamlessly with the Any Agent framework:

- **Function Calling**: Standardized function signatures for agent tool calling
- **Context Passing**: Proper context passing between agents and plugins
- **Error Handling**: Consistent error handling across the system
- **Type Safety**: Full type safety for better development experience
- **Documentation**: Auto-generated documentation from type hints
