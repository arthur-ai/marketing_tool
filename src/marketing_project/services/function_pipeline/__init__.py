"""
Function Pipeline Service - Modular Architecture

This package contains the refactored function pipeline service, split into focused modules:
- helpers: Prompt and configuration helpers
- tracing: OpenTelemetry tracing utilities
- llm_client: LLM calling logic
- approval: Approval integration
- execution: Step execution orchestration
"""

from marketing_project.services.function_pipeline.pipeline import FunctionPipeline

__all__ = ["FunctionPipeline"]
