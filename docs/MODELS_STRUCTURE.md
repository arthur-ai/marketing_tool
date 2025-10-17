# API Models Structure

## Overview

The API models have been reorganized into a modular folder structure for better maintainability and organization. The original `api_models.py` file has been split into multiple focused modules within the `models/` package.

## Folder Structure

```
src/marketing_project/models/
├── __init__.py              # Main exports and backward compatibility
├── content_models.py        # Content-related models
├── request_models.py        # API request models
├── response_models.py       # API response models
├── auth_models.py          # Authentication models
└── validation.py           # Validation helper functions
```

## Module Descriptions

### `content_models.py`
Contains all content-related Pydantic models:

- **`ContentType`** - Enumeration of supported content types
- **`ContentContext`** - Base content model with common fields
- **`BlogPostContext`** - Blog post specific model
- **`TranscriptContext`** - Transcript specific model
- **`ReleaseNotesContext`** - Release notes specific model

### `request_models.py`
Contains models for incoming API requests:

- **`AnalyzeRequest`** - Request model for content analysis
- **`PipelineRequest`** - Request model for pipeline execution
- **`WebhookRequest`** - Request model for webhook endpoints

### `response_models.py`
Contains models for API responses:

- **`APIResponse`** - Base response model
- **`ErrorResponse`** - Error response model
- **`ContentAnalysisResponse`** - Content analysis response
- **`PipelineResponse`** - Pipeline execution response
- **`HealthResponse`** - Health check response
- **`ContentSourceResponse`** - Content source information
- **`ContentSourceListResponse`** - List of content sources
- **`ContentFetchResponse`** - Content fetch response
- **`RateLimitResponse`** - Rate limiting information

### `auth_models.py`
Contains authentication-related models:

- **`APIKeyAuth`** - API key authentication model
- **`TokenResponse`** - Token response model

### `validation.py`
Contains validation helper functions:

- **`validate_content_length()`** - Validate content length limits
- **`validate_api_key_format()`** - Validate API key format

## Usage

### Importing Models

You can import models in several ways:

#### 1. Import from the main models package (recommended)
```python
from marketing_project.models import BlogPostContext, AnalyzeRequest, APIResponse
```

#### 2. Import from specific modules
```python
from marketing_project.models.content_models import BlogPostContext, ContentType
from marketing_project.models.request_models import AnalyzeRequest
from marketing_project.models.response_models import APIResponse
```

#### 3. Import everything from a module
```python
from marketing_project.models.content_models import *
```

### Backward Compatibility

The `__init__.py` file ensures backward compatibility by re-exporting all models. This means existing code that imports from `marketing_project.api_models` will continue to work without changes.

## Benefits of This Structure

### 1. **Better Organization**
- Related models are grouped together
- Easier to find specific model types
- Clear separation of concerns

### 2. **Improved Maintainability**
- Smaller, focused files are easier to maintain
- Changes to one model type don't affect others
- Better code organization and readability

### 3. **Enhanced Developer Experience**
- IDE autocomplete works better with smaller files
- Easier to navigate and understand the codebase
- Clear module boundaries

### 4. **Scalability**
- Easy to add new model categories
- Simple to extend existing model groups
- Better support for team development

## Migration Guide

### For Existing Code

No changes are required! The `__init__.py` file maintains backward compatibility:

```python
# This still works exactly as before
from marketing_project.api_models import BlogPostContext, AnalyzeRequest
```

### For New Code

Use the new import structure for better organization:

```python
# Recommended approach
from marketing_project.models import BlogPostContext, AnalyzeRequest

# Or import from specific modules
from marketing_project.models.content_models import BlogPostContext
from marketing_project.models.request_models import AnalyzeRequest
```

## Adding New Models

### 1. Determine the Category
Choose the appropriate module based on the model's purpose:
- Content-related → `content_models.py`
- API requests → `request_models.py`
- API responses → `response_models.py`
- Authentication → `auth_models.py`

### 2. Add the Model
Add your new model to the appropriate file:

```python
# In content_models.py
class NewContentType(ContentContext):
    """New content type model."""
    specific_field: str = Field(..., description="Specific field")
```

### 3. Update Exports
Add the new model to `__init__.py`:

```python
# In __init__.py
from .content_models import (
    ContentType,
    ContentContext,
    BlogPostContext,
    TranscriptContext,
    ReleaseNotesContext,
    NewContentType  # Add this
)

__all__ = [
    # ... existing exports ...
    "NewContentType",  # Add this
]
```

## Best Practices

### 1. **Keep Models Focused**
Each model should have a single, clear purpose.

### 2. **Use Descriptive Names**
Model names should clearly indicate their purpose and type.

### 3. **Add Comprehensive Documentation**
Include docstrings for all models and fields.

### 4. **Follow Pydantic Conventions**
Use proper field definitions with descriptions and validators.

### 5. **Maintain Backward Compatibility**
Always update `__init__.py` when adding new models.

## Example Usage

```python
from marketing_project.models import (
    BlogPostContext,
    AnalyzeRequest,
    ContentAnalysisResponse
)

# Create content
content = BlogPostContext(
    id="example-1",
    title="Marketing Automation Guide",
    content="This is a comprehensive guide...",
    author="John Doe",
    tags=["marketing", "automation"],
    category="tutorial"
)

# Create request
request = AnalyzeRequest(content=content)

# Process response
response = ContentAnalysisResponse(
    success=True,
    message="Analysis completed",
    data={"word_count": 1500, "readability_score": 75}
)
```

This modular structure provides a clean, maintainable foundation for the API models while preserving backward compatibility with existing code.
