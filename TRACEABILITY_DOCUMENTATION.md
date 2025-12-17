# Complete Traceability Documentation

This document provides comprehensive documentation of all span types in the marketing tool pipeline, including their attributes, events, relationships, and traceability features.

## Table of Contents

1. [Overview](#overview)
2. [Span Type Hierarchy](#span-type-hierarchy)
3. [Job Root Spans](#job-root-spans)
4. [Pipeline Spans](#pipeline-spans)
5. [Step Execution Spans](#step-execution-spans)
6. [LLM-Related Spans](#llm-related-spans)
7. [Prompt and Context Spans](#prompt-and-context-spans)
8. [Schema and Parsing Spans](#schema-and-parsing-spans)
9. [Approval Spans](#approval-spans)
10. [Retry and Rerun Spans](#retry-and-rerun-spans)
11. [Social Media Pipeline Spans](#social-media-pipeline-spans)
12. [Common Attributes](#common-attributes)
13. [Span Relationships](#span-relationships)

---

## Overview

The marketing tool uses OpenTelemetry with OpenInference extensions for comprehensive observability. All spans follow OpenInference conventions and include:

- **Minimum Required Attributes**: Every span has `openinference.span.kind`, `input.value`, `input.mime_type`, `output.value`, `output.mime_type`, `duration_ms`, and `duration_seconds`
- **Never Blank**: All spans are guaranteed to have minimum metadata, with fallback values when data is unavailable
- **Comprehensive Metadata**: Each span type includes type-specific attributes for debugging, performance analysis, and business intelligence
- **Event Tracking**: Important milestones are tracked as span events
- **Span Links**: Related spans are linked to show relationships (e.g., approval → rerun)

---

## Span Type Hierarchy

```
job.{job_id} (Root Span)
└── pipeline.execute
    ├── pipeline.step_execution.{step_name}
    │   ├── pipeline.prompt_preparation.{step_name}
    │   ├── pipeline.context_building.{step_name}
    │   ├── pipeline.llm_call.{step_name}
    │   │   ├── function_pipeline.{step_name}
    │   │   │   ├── pipeline.schema_generation.{step_name}
    │   │   │   └── pipeline.result_parsing.{step_name}
    │   │   └── (retry attempts tracked within)
    │   └── pipeline.approval_check.{step_name}
    │       └── (if approval required)
    │           └── approval.rerun_decision (when rerun)
    └── step_retry.execute (if retry needed)
```

---

## Job Root Spans

**Span Name**: `job.{job_id}`
**OpenInference Kind**: `CHAIN`
**Created By**: `create_job_root_span()` in `tracing.py`
**When**: At the start of any job execution

### Purpose
Root span for the entire job lifecycle. Contains job-level metadata and aggregates all pipeline execution spans.

### Attributes

#### Job Identification
- `job.id` - Unique job identifier
- `job.type` - Type of job (e.g., "pipeline", "social_media")
- `job.status` - Current job status (pending, running, completed, failed)
- `job.progress` - Job progress percentage (0-100)
- `job.current_step` - Current step being executed

#### Job Timing
- `job.created_at` - ISO timestamp when job was created
- `job.started_at` - ISO timestamp when job started
- `job.completed_at` - ISO timestamp when job completed
- `job.queue_wait_time_seconds` - Time job waited in queue (if applicable)

#### User and Session
- `user.id` - User ID who created the job
- `session.id` - Session ID (propagated from parent job if this is a subjob, ensuring all jobs in a chain share the same session)

#### Content Information
- `metadata.content_type` - Type of input content (blog, email, etc.)
- `metadata.output_content_type` - Expected output content type
- `metadata.title` - Title of the content (if available)

#### Pipeline Configuration
- `metadata.pipeline_config` - JSON string of pipeline configuration
- `metadata.step_name` - Current step name
- `metadata.step_number` - Current step number

#### LLM Aggregated Metrics
- `llm.system` - AI product/vendor (always "openai" per OpenInference spec)
- `llm.provider` - Hosting provider (always "openai" per OpenInference spec)
- `llm.model_name` - Primary model used
- `llm.models_used` - JSON array of all models used (if multiple)
- `llm.token_count.total` - Total tokens used across all steps
- `llm.invocation_parameters` - JSON of invocation parameters (temperature, etc.)

#### Job Relationships
- `metadata.original_job_id` - Original job ID (for retries/reruns)
- `metadata.parent_job_id` - Parent job ID (for subjobs)
- `metadata.resume_job_id` - Job ID to resume from

#### Input/Output
- `input.value` - Job input content (JSON string)
- `input.mime_type` - "application/json"
- `output.value` - Final job result (JSON string)
- `output.mime_type` - "application/json"

#### Duration
- `duration_ms` - Total job duration in milliseconds
- `duration_seconds` - Total job duration in seconds

### Events
- None (root span, events tracked in child spans)

### Span Links
- None (this is the root span)

---

## Pipeline Spans

**Span Name**: `pipeline.execute`
**OpenInference Kind**: `CHAIN`
**Created By**: `pipeline.py` - `execute()` method
**When**: At the start of pipeline execution

### Purpose
Tracks the entire pipeline execution, including all steps, configuration, and business metrics.

### Attributes

#### Pipeline Identification
- `agentic.workflow_type` - Always "pipeline"
- `pipeline_type` - Always "function_pipeline"
- `content_type` - Type of input content
- `output_content_type` - Expected output content type
- `job_id` - Associated job ID

#### Pipeline Configuration
- `pipeline.template_version` - Version of templates used (e.g., "v1")
- `pipeline.config_version` - Version of pipeline configuration (e.g., "v1")
- `pipeline.enabled_steps` - JSON array of enabled step names
- `pipeline.optional_steps` - JSON array of optional step names
- `pipeline.total_steps` - Total number of steps in pipeline

#### Content Characteristics
- `content.word_count` - Word count of input content
- `content.character_count` - Character count of input content
- `content.has_images` - Boolean indicating if content has images
- `content.image_count` - Number of images in content
- `content.language` - Detected language (or "unknown")
- `content.complexity` - Content complexity ("simple", "medium", "complex")
- `content.readability_score` - Readability score (if available)
- `content.blog_category` - Blog category (if applicable)
- `content.blog_tags_count` - Number of blog tags (if applicable)
- `content.transcript_duration_seconds` - Transcript duration (if applicable)

#### Business Intelligence Metrics
- `business.steps_completed` - Number of steps completed
- `business.total_steps` - Total number of steps
- `business.steps_completed_rate` - Ratio of completed to total steps (0.0-1.0)
- `business.success_rate` - Ratio of successful to total steps (0.0-1.0)
- `business.failed_steps_count` - Number of failed steps

#### Execution Metrics
- `steps_completed` - Number of steps completed
- `execution_time_seconds` - Total execution time

#### Input/Output
- `input.value` - Pipeline input content (JSON string, never empty)
- `input.mime_type` - "application/json"
- `output.value` - Final pipeline result (JSON string, never empty)
- `output.mime_type` - "application/json"

#### Duration
- `duration_ms` - Pipeline duration in milliseconds
- `duration_seconds` - Pipeline duration in seconds

### Events
- `pipeline.started` - When pipeline execution begins
  - Attributes: `content_type`, `output_content_type`
- `pipeline.completed` - When pipeline completes successfully
  - Attributes: `steps_completed`, `execution_time`
- `pipeline.failed` - When pipeline fails
  - Attributes: `error_type`, `steps_completed`

### Span Links
- None (parent of all step execution spans)

---

## Step Execution Spans

**Span Name**: `pipeline.step_execution.{step_name}`
**OpenInference Kind**: `AGENT`
**Created By**: `pipeline.py` - `_execute_step_with_plugin()` method
**When**: At the start of each step execution

### Purpose
Tracks individual step execution, including dependencies, context flow, business metrics, and transformation metrics.

### Attributes

#### Step Identification
- `step_name` - Name of the step (e.g., "seo_keywords", "marketing_brief")
- `plugin_name` - Same as step_name (for compatibility)
- `step_number` - Execution order number
- `job_id` - Associated job ID

#### Step Dependencies
- `step.dependencies` - JSON array of required context keys
- `step.dependencies_available` - JSON array of available context keys
- `step.dependencies_missing` - JSON array of missing context keys
- `step.dependencies_satisfied` - Boolean indicating if all dependencies are met
- `step.execution_order` - Execution order number

#### Context Information
- `context_keys_available` - JSON array of all available context keys
- `context_keys_count` - Number of available context keys

#### Context Registry Metrics
- `context_registry.queries_count` - Number of context registry queries
- `context_registry.hits` - Number of successful context lookups
- `context_registry.misses` - Number of failed context lookups
- `context_registry.hit_rate` - Ratio of hits to queries (0.0-1.0)
- `context_registry.keys_resolved` - JSON array of resolved keys
- `context_registry.keys_missing` - JSON array of missing keys

#### Job Input Snapshot
- `job.original_input_snapshot` - JSON string containing:
  - `content_type` - Type of original input
  - `has_title` - Boolean indicating if title exists
  - `title` - Title preview (first 200 chars, if available)
  - `content_preview` - Content preview (first 500 chars, if available)
  - `content_size_bytes` - Size of content in bytes
  - `input_type` - Type of input object (if not dict)

#### Step Output Summary
- `step.final_output_summary` - JSON string containing:
  - `step_name` - Name of the step
  - `output_keys` - JSON array of output keys
  - `output_keys_count` - Number of output keys
  - `has_confidence_score` - Boolean indicating if confidence score exists
  - `output_size_bytes` - Size of output in bytes
  - `has_main_keyword` - Boolean (SEO steps)
  - `has_target_audience` - Boolean (marketing steps)
  - `has_key_messages` - Boolean (marketing steps)

#### Step-Specific Business Metrics

**SEO Keywords Steps:**
- `seo.main_keyword` - Main keyword (truncated to 100 chars)
- `seo.primary_keywords_count` - Number of primary keywords
- `seo.secondary_keywords_count` - Number of secondary keywords
- `seo.lsi_keywords_count` - Number of LSI keywords
- `seo.search_intent` - Search intent classification
- `seo.keyword_clusters_count` - Number of keyword clusters

**Marketing Brief Steps:**
- `marketing.target_audience_count` - Number of target audience segments
- `marketing.key_messages_count` - Number of key messages
- `marketing.has_competitive_angle` - Boolean indicating competitive angle
- `marketing.kpis_count` - Number of KPIs

**Article Generation Steps:**
- `article.word_count_actual` - Actual word count
- `article.word_count_target` - Target word count
- `article.sections_count` - Number of sections
- `article.has_cta` - Boolean indicating CTA presence
- `article.internal_links_count` - Number of internal links

#### Content Transformation Metrics
- `transformation.input_size_bytes` - Size of input in bytes
- `transformation.output_size_bytes` - Size of output in bytes
- `transformation.size_change_percent` - Percentage change in size

#### Quality Metrics
- `quality.confidence_score` - Confidence score (if available)
- `quality.relevance_score` - Relevance score (if available)
- `quality.readability_score` - Readability score (if available)
- `quality.keyword_density_score` - Keyword density score (if available)

#### Input/Output
- `input.value` - Pipeline context at step start (JSON string, never empty)
- `input.mime_type` - "application/json"
- `output.value` - Step result (JSON string, never empty)
- `output.mime_type` - "application/json"

#### Duration
- `duration_ms` - Step execution duration in milliseconds
- `duration_seconds` - Step execution duration in seconds

### Events
- `step.started` - When step execution begins
  - Attributes: `step_name`, `context_keys_count`
- `context.resolved` - When context keys are resolved from registry
  - Attributes: `resolved_keys`, `source`
- `approval.required` - When step requires approval (if applicable)
  - Attributes: `approval_id`
- `step.completed` - When step completes successfully
  - Attributes: `step_name`, `output_keys_count`

### Span Links
- Links to `pipeline.approval_check.{step_name}` span if approval is required
  - Link attributes: `relationship: "approval_required"`

---

## LLM-Related Spans

### 1. LLM Call Span (Wrapper)

**Span Name**: `pipeline.llm_call.{step_name}`
**OpenInference Kind**: `LLM`
**Created By**: `llm_client.py` - `_make_api_call()` method
**When**: Wraps the entire LLM call including retries

### Purpose
Tracks the complete LLM call lifecycle, including all retry attempts and final outcome.

### Attributes

#### LLM Identification
- `step_name` - Name of the step
- `step_number` - Step number
- `model` - Model name (e.g., "gpt-4-turbo-preview")
- `llm.model_name` - Model name
- `llm.system` - AI product/vendor (always "openai" per OpenInference spec)
- `llm.provider` - Hosting provider (always "openai" per OpenInference spec)
- `llm.model.family` - Model family (e.g., "gpt-4", "gpt-3.5")
- `llm.model.version` - Model version (e.g., "turbo", "preview")

#### Retry Tracking
- `retry_attempt` - Current retry attempt number (1-based)
- `retry.attempt_number` - Current retry attempt number
- `retry.max_attempts` - Maximum number of retry attempts
- `retry.all_attempts` - JSON array of all retry attempts with errors
- `all_retries_exhausted` - Boolean indicating if all retries failed

#### LLM Configuration
- `llm.structured_output` - Always `true` (we use structured output)
- `llm.streaming` - Always `false` (we don't use streaming)
- `temperature` - Temperature setting
- `attempt` - Current attempt number

#### Input/Output
- `input.value` - LLM call parameters (JSON string, never empty)
- `input.mime_type` - "application/json"
- `output.value` - LLM response or error (JSON string, never empty)
- `output.mime_type` - "application/json"

#### Duration
- `duration_ms` - Total LLM call duration (including retries) in milliseconds
- `duration_seconds` - Total LLM call duration in seconds

### Events
- `llm_call.started` - When LLM call begins
  - Attributes: `step_name`, `model`
- `retry.attempt` - On each retry attempt (if attempt > 1)
  - Attributes: `attempt`, `max_attempts`, `previous_errors`
- `retry.error` - When a retry attempt fails
  - Attributes: `attempt`, `error_type`, `will_retry`
- `retry.backoff` - When waiting before retry
  - Attributes: `backoff_seconds`, `next_attempt`
- `llm_call.completed` - When LLM call succeeds
  - Attributes: `attempt`, `total_attempts`
- `llm_call.failed` - When all retries exhausted
  - Attributes: `all_retries_exhausted`, `total_attempts`

### Span Links
- None (contains child spans)

---

### 2. Function Pipeline Span (Actual LLM Call)

**Span Name**: `function_pipeline.{step_name}`
**OpenInference Kind**: `LLM`
**Created By**: `llm_client.py` - `_make_api_call()` method
**When**: For each individual LLM API call (within retry loop)

### Purpose
Tracks a single LLM API call with full OpenInference LLM attributes.

### Attributes

#### LLM Identification
- `step_name` - Name of the step
- `step_number` - Step number
- `model` - Model name
- `llm.model_name` - Model name
- `llm.system` - AI product/vendor (always "openai" per OpenInference spec)
- `llm.provider` - Hosting provider (always "openai" per OpenInference spec)
- `llm.model.family` - Model family (extracted from model name)
- `llm.model.version` - Model version (extracted from model name)
- `llm.structured_output` - Always `true`
- `llm.streaming` - Always `false`
- `temperature` - Temperature setting
- `attempt` - Current attempt number
- `job_id` - Associated job ID

#### LLM Messages (OpenInference Format)
- `llm.input_messages.0.message.role` - Role of first message (e.g., "system", "user")
- `llm.input_messages.0.message.content` - Content of first message
- `llm.input_messages.1.message.role` - Role of second message
- `llm.input_messages.1.message.content` - Content of second message
- (Additional messages indexed as needed)

#### LLM Response Format
- `llm.response_format` - Response format name (Pydantic model name)
- `llm.response_format.schema` - JSON schema of response format

#### LLM Invocation Parameters
- `llm.invocation_parameters` - JSON string of invocation parameters:
  - `temperature` - Temperature value
  - `model` - Model name

#### LLM Token Counts
- `llm.token_count.prompt` - Number of prompt tokens
- `llm.token_count.completion` - Number of completion tokens
- `llm.token_count.total` - Total tokens (prompt + completion)

#### Performance Metrics
- `llm.response_time_ms` - API call response time in milliseconds

#### Content Type
- `content_type` - Content type (if available in context)

#### Input/Output
- `input.value` - Full context dictionary (JSON string, never empty)
- `input.mime_type` - "application/json"
- `output.value` - Parsed result or error (JSON string, never empty)
- `output.mime_type` - "application/json"

#### Content Size Metrics
- `content.input_size_bytes` - Size of input in bytes
- `content.input_token_estimate` - Estimated input tokens (~4 chars per token)
- `content.output_size_bytes` - Size of output in bytes
- `content.output_token_estimate` - Estimated output tokens

#### Duration
- `duration_ms` - API call duration in milliseconds
- `duration_seconds` - API call duration in seconds

### Events
- None (events tracked in parent `pipeline.llm_call` span)

### Span Links
- None (child of `pipeline.llm_call` span)

---

## Prompt and Context Spans

### 1. Prompt Preparation Span

**Span Name**: `pipeline.prompt_preparation.{step_name}`
**OpenInference Kind**: `TOOL`
**Created By**: `helpers.py` - `get_user_prompt()` method
**When**: When preparing a prompt from a template

### Purpose
Tracks prompt template rendering, including template complexity and variable usage.

### Attributes

#### Template Information
- `step_name` - Name of the step
- `template_name` - Name of the template file
- `template_language` - Template language (e.g., "en")
- `prompt.template_version` - Version of template (e.g., "v1")
- `prompt.template_complexity` - Template complexity (length of template source)

#### Template Variables
- `llm.prompt_template.variables` - JSON string of template variables used
- `context_keys_used` - JSON array of context keys used in template
- `prompt.variable_count` - Number of template variables

#### Prompt Metrics
- `prompt_length` - Length of rendered prompt
- `prompt.rendered_length` - Length of rendered prompt
- `prompt.estimated_tokens` - Estimated token count (~4 chars per token)
- `prompt.has_conditional_logic` - Boolean indicating if template has conditionals

#### Template Content
- `llm.prompt_template.template` - Full template source code
- `llm.prompt_template.version` - Template version

#### Input/Output
- `input.value` - Template name and context (JSON string, never empty)
  - Contains: `template_name`, `template_language`, `context`
- `input.mime_type` - "application/json"
- `output.value` - Rendered prompt string (never empty)
- `output.mime_type` - "text/plain"

#### Duration
- `duration_ms` - Prompt preparation duration in milliseconds
- `duration_seconds` - Prompt preparation duration in seconds

### Events
- `prompt_preparation.started` - When prompt preparation begins
  - Attributes: `step_name`, `template_name`
- `prompt_preparation.completed` - When prompt preparation completes
  - Attributes: `prompt_length`

### Span Links
- None (child of `pipeline.step_execution` span)

---

### 2. Context Building Span

**Span Name**: `pipeline.context_building.{step_name}`
**OpenInference Kind**: `TOOL`
**Created By**: `llm_client.py` - `_build_context_messages()` method
**When**: When building context messages for LLM call

### Purpose
Tracks context building, including context registry lookups and message construction.

### Attributes

#### Context Information
- `step_name` - Name of the step
- `context_keys_count` - Number of context keys available
- `context_size_bytes` - Size of context in bytes
- `job_id` - Associated job ID

#### Context Registry Usage
- `agentic.context_registry_used` - Boolean indicating if context registry was used

#### Context Registry Performance Metrics
- `context_registry.queries_count` - Number of context registry queries
- `context_registry.hits` - Number of successful lookups
- `context_registry.misses` - Number of failed lookups
- `context_registry.hit_rate` - Ratio of hits to queries (0.0-1.0)
- `context_registry.keys_resolved` - JSON array of resolved keys
- `context_registry.keys_missing` - JSON array of missing keys

#### Input/Output
- `input.value` - Context dictionary (JSON string, never empty)
- `input.mime_type` - "application/json"
- `output.value` - Built messages array (JSON string, never empty)
- `output.mime_type` - "application/json"

#### Duration
- `duration_ms` - Context building duration in milliseconds
- `duration_seconds` - Context building duration in seconds

### Events
- `context_building.started` - When context building begins
  - Attributes: `context_keys_count`
- `context_building.completed` - When context building completes
  - Attributes: `uses_context_registry`

### Span Links
- None (child of `pipeline.step_execution` span)

---

## Schema and Parsing Spans

### 1. Schema Generation Span

**Span Name**: `pipeline.schema_generation.{step_name}`
**OpenInference Kind**: `TOOL`
**Created By**: `llm_client.py` - `_make_api_call()` method
**When**: When generating JSON schema from Pydantic model

### Purpose
Tracks schema generation from Pydantic models, including complexity metrics.

### Attributes

#### Schema Information
- `step_name` - Name of the step
- `response_model_name` - Name of the Pydantic model

#### Schema Metrics
- `schema.complexity_score` - Calculated complexity score (field_count * (nested_depth + 1) * 10)
- `schema.field_count` - Total number of fields in schema
- `schema.nested_depth` - Maximum nesting depth
- `schema.required_fields_count` - Number of required fields
- `schema.optional_fields_count` - Number of optional fields
- `schema_complexity` - Legacy complexity metric (length of schema JSON)

#### Input/Output
- `input.value` - Response model information (JSON string, never empty)
  - Contains: `response_model_name`, `schema`
- `input.mime_type` - "application/json"
- `output.value` - Generated JSON schema (JSON string, never empty)
- `output.mime_type` - "application/json"

#### Duration
- `duration_ms` - Schema generation duration in milliseconds
- `duration_seconds` - Schema generation duration in seconds

### Events
- None

### Span Links
- None (child of `function_pipeline` span)

---

### 2. Result Parsing Span

**Span Name**: `pipeline.result_parsing.{step_name}`
**OpenInference Kind**: `TOOL`
**Created By**: `llm_client.py` - `_make_api_call()` method
**When**: When parsing LLM response into Pydantic model

### Purpose
Tracks parsing of LLM response, including validation and error handling.

### Attributes

#### Parsing Information
- `step_name` - Name of the step
- `response_model_name` - Name of the Pydantic model

#### Parsing Metrics
- `parsing_success` - Boolean indicating if parsing succeeded
- `result_keys_count` - Number of keys in parsed result (if successful)

#### Input/Output
- `input.value` - Raw response to parse (JSON string, never empty)
  - Contains: `response_model_name`, `raw_response`
- `input.mime_type` - "application/json"
- `output.value` - Parsed result or error (JSON string, never empty)
- `output.mime_type` - "application/json"

#### Duration
- `duration_ms` - Parsing duration in milliseconds
- `duration_seconds` - Parsing duration in seconds

### Events
- None

### Span Links
- None (child of `function_pipeline` span)

---

## Approval Spans

**Span Name**: `pipeline.approval_check.{step_name}`
**OpenInference Kind**: `GUARDRAIL`
**Created By**: `approval.py` - `check_approval()` function
**When**: After step execution, before proceeding to next step

### Purpose
Tracks approval checks, including decision times, auto-approval reasons, and manual review requirements.

### Attributes

#### Approval Identification
- `step_name` - Name of the step
- `step_number` - Step number
- `job_id` - Associated job ID
- `approval_id` - Approval ID (if approval required)
- `content_type` - Content type (if available)

#### Approval Workflow Metrics
- `approval.decision_time_seconds` - Time taken to make approval decision
- `approval.requires_manual_review` - Boolean indicating if manual review is required
- `approval.auto_approval_reason` - Reason for auto-approval (e.g., "high_confidence", "low_risk") or `null` if manual review
- `approval.confidence_score` - Confidence score (if available)

#### Input/Output
- `input.value` - Approval input data (JSON string, never empty)
  - Contains: `prompt`, `system_instruction`, `context_keys`, `original_content`
- `input.mime_type` - "application/json"
- `output.value` - Approval result (JSON string, never empty)
  - Contains: `approval_required`, `status`, `approval_id` (if required), `step_name`, `step_number`
- `output.mime_type` - "application/json"

#### Duration
- `duration_ms` - Approval check duration in milliseconds
- `duration_seconds` - Approval check duration in seconds

### Events
- `approval_check.started` - When approval check begins
  - Attributes: `step_name`, `step_number`
- `approval.required` - When approval is required
  - Attributes: `approval_id`, `confidence_score`
- `approval.auto_approved` - When auto-approved
  - Attributes: `status`, `confidence_score`
- `approval_check.failed` - When approval check fails (error)
  - Attributes: `error_type`

### Span Links
- None (linked from `pipeline.step_execution` span)

### Error Attributes (if error occurs)
- `error` - Always `true`
- `error.type` - Exception type name
- `error.message` - Error message
- `error.category` - Error category (e.g., "network", "validation", "api_error")
- `error.is_retryable` - Boolean indicating if error is retryable
- `error.recovery_action` - Suggested recovery action
- `error.user_visible` - Boolean indicating if error should be shown to user
- `error.requires_manual_intervention` - Boolean indicating if manual intervention needed
- `error.context.*` - Additional error context attributes

---

## Retry and Rerun Spans

### 1. Step Retry Span

**Span Name**: `step_retry.execute`
**OpenInference Kind**: `AGENT`
**Created By**: `step_retry_service.py` - `execute_step_retry()` function
**When**: When manually retrying a failed step

### Purpose
Tracks manual step retries, including user guidance and retry context.

### Attributes

#### Retry Identification
- `step_name` - Name of the step being retried
- `job_id` - Associated job ID
- `has_user_guidance` - Boolean indicating if user provided guidance

#### Retry Context
- `execution_time` - Execution time of retry
- `status` - Retry status ("success" or "error")

#### Quality Metrics (if result available)
- `quality.confidence_score` - Confidence score (if available)
- `quality.relevance_score` - Relevance score (if available)
- `quality.readability_score` - Readability score (if available)

#### Input/Output
- `input.value` - Retry input data (JSON string, never empty)
  - Contains: `input_data`, `context`, `user_guidance`
- `input.mime_type` - "application/json"
- `output.value` - Retry result or error (JSON string, never empty)
- `output.mime_type` - "application/json"

#### Duration
- `duration_ms` - Retry execution duration in milliseconds
- `duration_seconds` - Retry execution duration in seconds

### Events
- `step_retry.started` - When retry begins
  - Attributes: `step_name`, `has_user_guidance`
- `step_retry.completed` - When retry succeeds
  - Attributes: `step_name`, `execution_time`
- `step_retry.failed` - When retry fails
  - Attributes: `error_type`

### Span Links
- None (standalone retry operation)

### Error Attributes (if error occurs)
- Same error attributes as approval spans

---

### 2. Rerun Decision Span

**Span Name**: `approval.rerun_decision`
**OpenInference Kind**: `AGENT`
**Created By**: `api/approvals.py` - `rerun_decision()` function
**When**: When user decides to rerun a step after approval

### Purpose
Tracks rerun decisions after approval, including user guidance and decision context.

### Attributes

#### Rerun Identification
- `approval_id` - Approval ID
- `step_name` - Name of the step being rerun
- `job_id` - Associated job ID
- `retry_attempt` - Retry attempt number (approval.retry_count + 1)
- `has_user_guidance` - Boolean indicating if user provided guidance

#### Input/Output
- `input.value` - Rerun input data (JSON string, never empty)
  - Contains: `approval_id`, `step_name`, `input_data`, `user_guidance`
- `input.mime_type` - "application/json"
- `output.value` - Rerun result (JSON string, never empty)
- `output.mime_type` - "application/json"

#### Duration
- `duration_ms` - Rerun execution duration in milliseconds
- `duration_seconds` - Rerun execution duration in seconds

### Events
- `rerun.started` - When rerun begins
  - Attributes: `approval_id`, `retry_attempt`
- `rerun.completed` - When rerun succeeds
  - Attributes: `approval_id`, `step_name`
- `rerun.failed` - When rerun fails
  - Attributes: `error_type`

### Span Links
- Links to approval span (if available)
  - Link attributes: `relationship: "rerun_from_approval"`

### Error Attributes (if error occurs)
- Same error attributes as approval spans

---

## Social Media Pipeline Spans

**Span Name**: `social_media_pipeline.{step_name}`
**OpenInference Kind**: `LLM`
**Created By**: `social_media_pipeline.py` - `_make_api_call()` method
**When**: For social media content generation LLM calls

### Purpose
Tracks LLM calls for social media content generation, similar to function_pipeline spans but for social media specific workflows.

### Attributes

#### LLM Identification
- `step_name` - Name of the step
- `step_number` - Step number
- `model` - Model name
- `llm.model_name` - Model name
- `llm.system` - AI product/vendor (always "openai" per OpenInference spec)
- `llm.provider` - Hosting provider (always "openai" per OpenInference spec)
- `llm.model.family` - Model family (extracted from model name)
- `llm.model.version` - Model version (extracted from model name)
- `llm.structured_output` - Always `true`
- `llm.streaming` - Always `false`
- `temperature` - Temperature setting
- `job_id` - Associated job ID

#### Social Media Specific
- `platform` - Social media platform (e.g., "twitter", "linkedin")
- `social_media_platform` - Same as platform

#### Content Type
- `content_type` - Content type (if available in context)

#### LLM Messages (OpenInference Format)
- Same as `function_pipeline` spans

#### LLM Response Format
- Same as `function_pipeline` spans

#### LLM Invocation Parameters
- Same as `function_pipeline` spans

#### LLM Token Counts
- Same as `function_pipeline` spans

#### Input/Output
- `input.value` - Full context dictionary (JSON string, never empty)
- `input.mime_type` - "application/json"
- `output.value` - Parsed result or error (JSON string, never empty)
- `output.mime_type` - "application/json"

#### Content Size Metrics
- Same as `function_pipeline` spans

#### Duration
- `duration_ms` - API call duration in milliseconds
- `duration_seconds` - API call duration in seconds

### Events
- None (events tracked at higher level)

### Span Links
- None (part of social media pipeline workflow)

---

## Common Attributes

All spans include these minimum required attributes:

### OpenInference Span Kind
- `openinference.span.kind` - Always set to one of:
  - `CHAIN` - Pipeline orchestration
  - `AGENT` - Step execution, retries, reruns
  - `LLM` - LLM API calls
  - `TOOL` - Prompt preparation, context building, schema generation, parsing
  - `GUARDRAIL` - Approval checks

### Input/Output (Always Present)
- `input.value` - Input data (JSON string, never empty - defaults to `{}` if no input)
- `input.mime_type` - MIME type (typically "application/json" or "text/plain")
- `output.value` - Output data (JSON string, never empty - defaults to `{}` if no output)
- `output.mime_type` - MIME type (typically "application/json" or "text/plain")

### Duration (Always Present)
- `duration_ms` - Duration in milliseconds
- `duration_seconds` - Duration in seconds

### Content Size Metrics (When Applicable)
- `content.input_size_bytes` - Size of input in bytes
- `content.input_token_estimate` - Estimated input tokens (~4 chars per token)
- `content.output_size_bytes` - Size of output in bytes
- `content.output_token_estimate` - Estimated output tokens

### Error Attributes (When Error Occurs)
- `error` - Always `true`
- `error.type` - Exception type name
- `error.message` - Error message
- `error.category` - Error category (network, validation, api_error, authentication, timeout, etc.)
- `error.is_retryable` - Boolean indicating if error is retryable
- `error.recovery_action` - Suggested recovery action (retry, retry_with_backoff, fail)
- `error.user_visible` - Boolean indicating if error should be shown to user
- `error.requires_manual_intervention` - Boolean indicating if manual intervention needed
- `error.context.*` - Additional error context attributes

---

## Span Relationships

### Parent-Child Relationships

1. **Job Root → Pipeline Execute**
   - `job.{job_id}` is parent of `pipeline.execute`

2. **Pipeline Execute → Step Execution**
   - `pipeline.execute` is parent of all `pipeline.step_execution.{step_name}` spans

3. **Step Execution → Prompt Preparation**
   - `pipeline.step_execution.{step_name}` is parent of `pipeline.prompt_preparation.{step_name}`

4. **Step Execution → Context Building**
   - `pipeline.step_execution.{step_name}` is parent of `pipeline.context_building.{step_name}`

5. **Step Execution → LLM Call**
   - `pipeline.step_execution.{step_name}` is parent of `pipeline.llm_call.{step_name}`

6. **LLM Call → Function Pipeline**
   - `pipeline.llm_call.{step_name}` is parent of `function_pipeline.{step_name}`

7. **Function Pipeline → Schema Generation**
   - `function_pipeline.{step_name}` is parent of `pipeline.schema_generation.{step_name}`

8. **Function Pipeline → Result Parsing**
   - `function_pipeline.{step_name}` is parent of `pipeline.result_parsing.{step_name}`

9. **Step Execution → Approval Check**
   - `pipeline.step_execution.{step_name}` is parent of `pipeline.approval_check.{step_name}`

### Span Links (Non-Parent-Child Relationships)

1. **Step Execution → Approval Check**
   - When approval is required, `pipeline.step_execution.{step_name}` links to `pipeline.approval_check.{step_name}`
   - Link attributes: `relationship: "approval_required"`

2. **Rerun Decision → Approval**
   - `approval.rerun_decision` links to the approval span that triggered it
   - Link attributes: `relationship: "rerun_from_approval"`

### Trace Flow Example

```
job.abc123
└── pipeline.execute
    ├── pipeline.step_execution.seo_keywords
    │   ├── pipeline.prompt_preparation.seo_keywords
    │   ├── pipeline.context_building.seo_keywords
    │   ├── pipeline.llm_call.seo_keywords
    │   │   └── function_pipeline.seo_keywords
    │   │       ├── pipeline.schema_generation.seo_keywords
    │   │       └── pipeline.result_parsing.seo_keywords
    │   └── pipeline.approval_check.seo_keywords
    │       └── (link) approval.rerun_decision (if rerun)
    └── pipeline.step_execution.marketing_brief
        └── ... (similar structure)
```

---

## Traceability Features

### 1. End-to-End Traceability
- Every operation is traced from job creation to completion
- All spans are linked in a parent-child hierarchy
- Job ID is propagated to all spans for correlation
- **Session ID is propagated through entire job chains** - all jobs (root and subjobs) in a chain share the same `session.id`, enabling grouping of all related traces

### 2. Context Propagation
- Pipeline context flows through all step execution spans
- Original job input is captured in step execution spans
- Context registry lookups are tracked for dependency analysis

### 3. Performance Analysis
- Duration metrics at every level (job, pipeline, step, LLM call)
- Response time tracking for LLM calls
- Content size metrics for input/output analysis
- Token count tracking for cost analysis

### 4. Business Intelligence
- Step completion rates
- Success/failure rates
- Quality metrics (confidence scores, relevance, readability)
- Step-specific business metrics (SEO keywords, marketing metrics, article metrics)

### 5. Error Tracking
- Comprehensive error categorization
- Retry tracking with attempt history
- Error recovery suggestions
- User-visible vs. internal error classification

### 6. Approval Workflow Tracking
- Approval decision times
- Auto-approval reasons
- Manual review requirements
- Rerun tracking with user guidance

### 7. Dependency Analysis
- Step dependencies (required vs. available keys)
- Context registry performance (hits, misses, hit rate)
- Context resolution tracking

### 8. Content Analysis
- Content characteristics (word count, complexity, language)
- Content transformation metrics (input/output sizes)
- Template complexity and variable usage

---

## Best Practices for Using Traces

### 1. Filtering Traces
- Filter by `job.id` to see complete job lifecycle
- Filter by `step_name` to see all executions of a specific step
- Filter by `error: true` to find all failures
- Filter by `approval.requires_manual_review: true` to find steps requiring review

### 2. Performance Analysis
- Compare `duration_ms` across similar steps
- Analyze `llm.response_time_ms` for LLM performance
- Track `context_registry.hit_rate` for context efficiency
- Monitor `business.success_rate` for pipeline health

### 3. Business Intelligence
- Aggregate `seo.*` metrics for SEO step analysis
- Aggregate `marketing.*` metrics for marketing step analysis
- Track `quality.confidence_score` trends
- Monitor `business.steps_completed_rate` for pipeline efficiency

### 4. Debugging
- Use `job.original_input_snapshot` to see what entered the pipeline
- Use `step.final_output_summary` to see what was produced
- Check `error.category` and `error.recovery_action` for error handling
- Review `retry.all_attempts` for retry patterns

### 5. Cost Analysis
- Sum `llm.token_count.total` across all spans for total cost
- Group by `llm.model_name` to see model usage
- Track `content.input_token_estimate` vs. actual tokens for accuracy

---

## Conclusion

This comprehensive traceability system provides:

- **Complete Visibility**: Every operation is traced with detailed metadata
- **Never Blank**: All spans have minimum required attributes with fallback values
- **OpenInference Compliant**: Follows OpenInference conventions for LLM observability
- **Business Intelligence**: Rich metrics for analysis and optimization
- **Error Tracking**: Comprehensive error categorization and recovery suggestions
- **Performance Analysis**: Detailed timing and size metrics at every level

All spans are designed to provide maximum observability while maintaining performance and following industry standards for distributed tracing.
