# Per-Step Model Configuration

This document describes how to configure different models for each pipeline step.

## Overview

The per-step model configuration allows you to:
- Use cheaper models for simple steps (e.g., `gpt-4o-mini` for SEO Keywords)
- Use more powerful models for complex steps (e.g., `gpt-4o` for Article Generation)
- Optimize costs while maintaining quality
- Experiment with different models per step

## Configuration Models

### PipelineStepConfig

Configuration for a single step:

```python
from marketing_project.models.pipeline_steps import PipelineStepConfig

step_config = PipelineStepConfig(
    step_name="seo_keywords",
    model="gpt-4o-mini",  # Use cheaper model
    temperature=0.7,
    max_retries=2
)
```

### PipelineConfig

Configuration for the entire pipeline:

```python
from marketing_project.models.pipeline_steps import PipelineConfig, PipelineStepConfig

pipeline_config = PipelineConfig(
    default_model="gpt-5.1",
    default_temperature=0.7,
    default_max_retries=2,
    step_configs={
        "seo_keywords": PipelineStepConfig(
            step_name="seo_keywords",
            model="gpt-4o-mini",  # Cheaper for simple step
        ),
        "article_generation": PipelineStepConfig(
            step_name="article_generation",
            model="gpt-4o",  # More powerful for complex step
            temperature=0.8,
        ),
    }
)
```

## Usage

### In FunctionPipeline

```python
from marketing_project.services.function_pipeline import FunctionPipeline
from marketing_project.models.pipeline_steps import PipelineConfig

# Create pipeline with custom config
pipeline = FunctionPipeline(
    pipeline_config=PipelineConfig(
        default_model="gpt-5.1",
        step_configs={
            "seo_keywords": PipelineStepConfig(
                step_name="seo_keywords",
                model="gpt-4o-mini",
            ),
        }
    )
)

# Or update config after initialization
pipeline.pipeline_config = custom_config
```

### In API Requests

```python
# POST /api/v1/pipeline
{
    "content": {...},
    "pipeline_config": {
        "default_model": "gpt-5.1",
        "default_temperature": 0.7,
        "step_configs": {
            "seo_keywords": {
                "step_name": "seo_keywords",
                "model": "gpt-4o-mini"
            },
            "article_generation": {
                "step_name": "article_generation",
                "model": "gpt-4o"
            }
        }
    }
}
```

### Step Execution

```python
# POST /api/v1/pipeline/steps/{step_name}/execute
{
    "content": {...},
    "context": {...},
    "pipeline_config": {
        "default_model": "gpt-4o-mini",
        "step_configs": {
            "seo_keywords": {
                "step_name": "seo_keywords",
                "model": "gpt-4o-mini"
            }
        }
    }
}
```

## Model Selection

The system selects models in this order:
1. Step-specific config (if provided)
2. Pipeline default model
3. FunctionPipeline default ("gpt-5.1")

## Recommendations

### Cost Optimization

Use cheaper models for simpler steps:
- `gpt-4o-mini`: SEO Keywords, Suggested Links
- `gpt-4o`: Marketing Brief, SEO Optimization
- `gpt-5.1`: Article Generation, Content Formatting

### Performance Optimization

Use faster models where appropriate:
- `gpt-4o-mini`: Fast, good for simple tasks
- `gpt-4o`: Balanced speed/quality
- `gpt-5.1`: Best quality, slower

### Example Configuration

```python
pipeline_config = PipelineConfig(
    default_model="gpt-4o",  # Default for most steps
    step_configs={
        "seo_keywords": PipelineStepConfig(
            step_name="seo_keywords",
            model="gpt-4o-mini",  # Simple step
        ),
        "marketing_brief": PipelineStepConfig(
            step_name="marketing_brief",
            model="gpt-4o",  # Medium complexity
        ),
        "article_generation": PipelineStepConfig(
            step_name="article_generation",
            model="gpt-5.1",  # Complex step, needs best quality
            temperature=0.8,  # More creative
        ),
        "seo_optimization": PipelineStepConfig(
            step_name="seo_optimization",
            model="gpt-4o",  # Medium complexity
        ),
        "suggested_links": PipelineStepConfig(
            step_name="suggested_links",
            model="gpt-4o-mini",  # Simple matching task
        ),
        "content_formatting": PipelineStepConfig(
            step_name="content_formatting",
            model="gpt-4o-mini",  # Simple formatting
        ),
        "design_kit": PipelineStepConfig(
            step_name="design_kit",
            model="gpt-4o-mini",  # Simple extraction
        ),
    }
)
```

## Tracking

Model usage is tracked per step in pipeline metadata:
- Model used per step
- Cost per step-model combination
- Performance metrics

## Validation

The system validates:
- Model name format
- Model compatibility with step requirements
- Temperature ranges (0.0-2.0)
- Max retries (>= 0)

## Backward Compatibility

If no `pipeline_config` is provided:
- Uses default model for all steps
- Maintains existing behavior
- No breaking changes
