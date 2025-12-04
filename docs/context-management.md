# Context Management Strategy

This document describes the context management system that ensures zero data loss throughout pipeline execution.

## Overview

The context management system provides:
- **Zero data loss**: All step outputs are preserved with full history
- **Efficient access**: Context references enable lazy loading
- **Complete audit trail**: Full input/output snapshots for debugging
- **Versioning**: Support for multiple execution contexts (resume cycles)

## Architecture

### Context Registry

The `ContextRegistry` service manages all step outputs with the following features:

1. **Context Versioning**: Each step output is stored with version information
2. **Lazy Loading**: Context data is loaded only when requested
3. **Compression**: Optional compression for storage efficiency
4. **Reference System**: Context references enable efficient passing between steps

### Storage Structure

```
context_registry/
  {job_id}/
    context_0/              # Initial execution
      seo_keywords.json
      marketing_brief.json
      ...
    context_1/              # First resume after approval
      article_generation.json
      ...
```

## Usage

### Registering Step Outputs

Step outputs are automatically registered when saved via `StepResultManager`:

```python
from marketing_project.services.step_result_manager import get_step_result_manager

step_manager = get_step_result_manager()
await step_manager.save_step_result(
    job_id=job_id,
    step_number=1,
    step_name="seo_keywords",
    result_data=result_dict,
    input_snapshot=input_context,
    context_keys_used=["input_content"],
)
```

### Retrieving Context

Get context for a specific step:

```python
from marketing_project.services.context_registry import get_context_registry

context_registry = get_context_registry()
context_data = await context_registry.get_context(
    job_id=job_id,
    key="seo_keywords"
)
```

### Querying Multiple Context Keys

```python
contexts = await context_registry.query_context(
    job_id=job_id,
    keys=["seo_keywords", "marketing_brief"]
)
```

### Getting Full History

```python
history = await context_registry.get_full_history(job_id)
# Returns: {
#   "0": {
#     "seo_keywords": {...},
#     "marketing_brief": {...},
#   },
#   "1": {
#     "article_generation": {...},
#   }
# }
```

## Context References

Context references are lightweight pointers to stored context data:

```python
reference = context_registry.get_context_reference(
    job_id=job_id,
    step_name="seo_keywords"
)

# Resolve reference to actual data (lazy loading)
context_data = await context_registry.resolve_context(reference)
```

## Integration with Pipeline

The pipeline automatically:
1. Registers each step output after execution
2. Uses context references when available
3. Falls back to direct context passing for backward compatibility
4. Resolves missing context keys from registry on-demand

## Benefits

1. **No Data Loss**: Every step output is preserved
2. **Efficient Memory Usage**: Lazy loading prevents memory bloat
3. **Better Debugging**: Full input/output snapshots available
4. **Resume Support**: Complete context history enables pipeline resumption
5. **Audit Trail**: Track exactly what data was used by each step

## Migration

The system maintains backward compatibility:
- Existing code continues to work with direct context passing
- New features are opt-in via context registry
- Gradual migration path available
