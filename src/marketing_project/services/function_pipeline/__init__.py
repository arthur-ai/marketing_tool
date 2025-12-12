"""
Function Pipeline Service - Modular Architecture

This package contains the refactored function pipeline service, split into focused modules:
- helpers: Prompt and configuration helpers
- tracing: OpenTelemetry tracing utilities
- llm_client: LLM calling logic
- approval: Approval integration
- orchestration: Pipeline orchestration utilities
- step_results: Step result saving utilities
"""

# Import FunctionPipeline from pipeline.py module
# The FunctionPipeline class is defined in pipeline.py within this package
from marketing_project.services.function_pipeline.pipeline import FunctionPipeline

__all__ = ["FunctionPipeline"]
