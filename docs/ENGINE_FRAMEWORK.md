# Generic Engine Framework

## Overview

The Generic Engine Framework provides a composable system for mixing and matching different processing engines (LLM, local, hybrid, etc.) at the field level. This enables fine-grained control over which engine processes which field of a result model.

## Architecture

### Core Components

1. **Engine Interface** (`services/engines/base.py`)
   - Abstract base class that all engines must implement
   - Defines `execute(operation, inputs, context, pipeline)` method
   - Supports `supports_operation(operation)` for capability checking

2. **Engine Registry** (`services/engines/registry.py`)
   - Central registry for managing engines
   - Engines are registered by type (e.g., 'llm', 'local_semantic')
   - Provides `get_engine(engine_type)` for retrieval

3. **Engine Composer** (`services/engines/composer.py`)
   - Orchestrates multiple engines based on configuration
   - Supports default engine + field-level overrides
   - Executes appropriate engine for each field

### Configuration

Engines are configured using `EngineConfig`:

```python
{
    "default_engine": "llm",  # Default for all fields
    "field_overrides": {      # Override specific fields
        "keyword_density_analysis": "local_semantic",
        "search_intent": "local_semantic"
    }
}
```

## Creating New Engines

### 1. Implement the Engine Interface

```python
from marketing_project.services.engines.base import Engine

class MyCustomEngine(Engine):
    def supports_operation(self, operation: str) -> bool:
        return operation in ["operation1", "operation2"]

    async def execute(
        self,
        operation: str,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
        pipeline: Optional[Any] = None,
    ) -> Any:
        if operation == "operation1":
            return await self._do_operation1(inputs, context)
        elif operation == "operation2":
            return await self._do_operation2(inputs, context)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
```

### 2. Register the Engine

```python
from marketing_project.services.engines.registry import register_engine

engine = MyCustomEngine()
register_engine("my_custom", engine)
```

### 3. Use in Composer

```python
from marketing_project.services.engines.composer import EngineComposer

composer = EngineComposer(
    default_engine_type="llm",
    field_overrides={
        "some_field": "my_custom"
    }
)

result = await composer.execute_operation(
    "some_field",
    "operation1",
    inputs,
    context,
    pipeline
)
```

## Plugin-Specific Composers

For complex result models, create a plugin-specific composer:

```python
class MyPluginComposer:
    FIELD_TO_OPERATION = {
        "field1": "operation1",
        "field2": "operation2",
    }

    def __init__(self, composer: EngineComposer):
        self.composer = composer

    async def compose_result(self, content, context, pipeline):
        # Extract fields using appropriate engines
        # Merge into final result model
        pass
```

## Extensibility

The framework is designed to be extensible:

- **New engines**: Implement `Engine` interface and register
- **New operations**: Add methods to engines and update field-to-operation mappings
- **New plugins**: Create plugin-specific composers following the SEO keywords pattern

## Examples

### Example 1: SEO Keywords (Current Implementation)

- **LLM Engine**: Extracts all keywords via LLM
- **Local Semantic Engine**: Uses spaCy, YAKE, RAKE, TF-IDF, embeddings
- **Composer**: Mixes LLM and local based on field config

### Example 2: Article Generation (Future)

```python
# services/engines/article_generation/llm_engine.py
# services/engines/article_generation/local_engine.py
# Use composer with field_overrides for article_generation
```

## Best Practices

1. **Lazy Loading**: Load heavy models (spaCy, transformers) on first use
2. **Error Handling**: Gracefully handle missing dependencies
3. **Caching**: Cache engine instances in registry
4. **Operation Granularity**: Keep operations focused and composable
5. **Backward Compatibility**: Default to existing behavior (LLM) when config missing
