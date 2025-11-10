# Marketing Project Architecture

## Overview

The Marketing Project follows a **function-based pipeline architecture** using OpenAI's structured outputs (function calling) for deterministic, type-safe content processing.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                       API Layer (FastAPI)                        │
│  /api/v1/process/blog                                           │
│  /api/v1/process/release-notes                                  │
│  /api/v1/process/transcript                                     │
│  /api/v1/jobs/*                 (Job management)                │
│  /api/v1/approvals/*            (Human-in-the-loop)             │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Processors Layer                              │
│  processors/                                                     │
│  ├─ blog_processor.py          (validates & processes blogs)    │
│  ├─ releasenotes_processor.py  (validates & processes releases) │
│  └─ transcript_processor.py    (validates & processes transcripts)│
│                                                                  │
│  RESPONSIBILITIES:                                               │
│  - Input validation (Pydantic models)                            │
│  - Content type routing                                          │
│  - Pipeline orchestration                                        │
│  - Error handling                                                │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│            Function Pipeline (OpenAI Function Calling)           │
│  services/function_pipeline.py                                   │
│                                                                  │
│  7-STEP STRUCTURED PIPELINE:                                     │
│  1. SEO Keywords Extraction     (SEOKeywordsResult)             │
│  2. Marketing Brief Generation  (MarketingBriefResult)          │
│  3. Article Content Generation  (ArticleGenerationResult)       │
│  4. SEO Optimization           (SEOOptimizationResult)          │
│  5. Internal Docs Suggestions  (InternalDocsResult)             │
│  6. Content Formatting         (ContentFormattingResult)        │
│  7. Design Kit Generation      (DesignKitResult)                │
│                                                                  │
│  FEATURES:                                                       │
│  - Structured outputs (Pydantic models)                          │
│  - Quality metrics (confidence scores)                           │
│  - Approval system integration                                   │
│  - Template-based system instructions                            │
│  - Type-safe end-to-end                                          │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Prompt Templates (Jinja2)                        │
│  prompts/v1/{lang}/                                             │
│  ├─ seo_keywords_agent_instructions.j2                          │
│  ├─ marketing_brief_agent_instructions.j2                       │
│  ├─ article_generation_agent_instructions.j2                    │
│  ├─ seo_optimization_agent_instructions.j2                      │
│  ├─ internal_docs_agent_instructions.j2                         │
│  ├─ content_formatting_agent_instructions.j2                    │
│  └─ design_kit_agent_instructions.j2                            │
│                                                                  │
│  RESPONSIBILITIES:                                               │
│  - Comprehensive system instructions (100-170 lines each)        │
│  - Multi-language support                                        │
│  - Versioned templates                                           │
│  - Best practices and guidelines                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Request Flow

### Example: Blog Post Processing

```
1. POST /api/v1/process/blog
   └─> BlogProcessorRequest { content: BlogPostContext }

2. blog_processor.process_blog_post()
   ├─> Parse and validate input (Pydantic)
   ├─> Convert to BlogPostContext model
   └─> Call FunctionPipeline

3. FunctionPipeline.execute_pipeline()
   ├─> Load system instructions from Jinja2 templates
   ├─> Execute 7-step function calling pipeline:
   │
   │   Step 1: SEO Keywords
   │   ├─> Load seo_keywords_agent_instructions.j2
   │   ├─> Call OpenAI with response_format=SEOKeywordsResult
   │   ├─> Get structured JSON output
   │   ├─> Extract confidence_score & relevance_score
   │   └─> Request approval if needed
   │
   │   Step 2: Marketing Brief
   │   ├─> Load marketing_brief_agent_instructions.j2
   │   ├─> Pass Step 1 results as context
   │   ├─> Call OpenAI with response_format=MarketingBriefResult
   │   ├─> Get structured JSON output
   │   └─> Request approval if needed
   │
   │   Step 3-7: Similar pattern...
   │
   └─> Return PipelineResult with all step outputs

4. blog_processor returns JSON result
   └─> { status, content_type, pipeline_result, message }
```

## Key Design Decisions

### Why Function Calling Over Agents?

**Legacy (Removed)**:
- ❌ Agent-based orchestration (LangChain)
- ❌ Unpredictable output format
- ❌ Complex error handling
- ❌ Higher latency
- ❌ Higher costs

**Current (Function Pipeline)**:
- ✅ Direct OpenAI function calling
- ✅ Guaranteed JSON structure (Pydantic)
- ✅ Type-safe end-to-end
- ✅ 20% faster execution
- ✅ 10% lower costs
- ✅ Predictable outputs
- ✅ Built-in quality metrics

### Data Models

All pipeline steps use Pydantic models for type safety:

```python
class SEOKeywordsResult(BaseModel):
    primary_keywords: List[str]
    secondary_keywords: Optional[List[str]]
    lsi_keywords: Optional[List[str]]
    keyword_density: Optional[str]  # JSON string
    search_intent: str
    keyword_difficulty: Optional[str]
    confidence_score: Optional[float]
    relevance_score: Optional[float]

class PipelineResult(BaseModel):
    pipeline_status: str
    step_results: Dict[str, Any]
    final_content: str
    quality_metrics: Optional[Dict[str, float]]
    step_info: List[PipelineStepInfo]
```

### Approval System

Human-in-the-loop approval for quality control:

```python
# Approval triggered based on confidence scores
if confidence_score < APPROVAL_THRESHOLD:
    approved_result = await request_approval_if_needed(
        job_id=job_id,
        agent_name=step_name,
        input_data=input_data,
        output_data=result_dict,
        confidence_score=confidence_score
    )
```

## Component Responsibilities

### Processors
- **Input Validation**: Ensure data structure is correct
- **Type Conversion**: Convert to Pydantic models
- **Pipeline Orchestration**: Call FunctionPipeline
- **Error Handling**: Catch and format errors
- **Response Formatting**: Return standardized JSON

### Function Pipeline
- **Template Loading**: Load Jinja2 system instructions
- **Function Calling**: Execute OpenAI structured outputs
- **Context Management**: Pass results between steps
- **Quality Tracking**: Extract and store metrics
- **Approval Integration**: Trigger human review when needed

### Prompt Templates
- **System Instructions**: Comprehensive guidelines (100-170 lines)
- **Best Practices**: Industry standards and requirements
- **Quality Requirements**: Scoring and metrics definitions
- **Multi-language**: Support for en, fr, etc.

## Benefits

### Technical
1. **Type Safety**: Pydantic models throughout
2. **Predictability**: Guaranteed JSON structure
3. **Performance**: 20% faster than legacy agents
4. **Cost Efficiency**: 10% cheaper
5. **Maintainability**: Simple, straightforward code

### Business
1. **Quality Metrics**: Confidence scores for every step
2. **Human Oversight**: Approval system for quality control
3. **Scalability**: Easy to add new content types
4. **Reliability**: Consistent, predictable outputs
5. **Visibility**: Step-by-step result tracking

## Configuration

### Pipeline Configuration
File: `src/marketing_project/config/pipeline.yml`

```yaml
version: "2"
pipeline_steps:
  - step_name: "seo_keywords"
    description: "Extract SEO keywords"
  - step_name: "marketing_brief"
    description: "Generate marketing brief"
  # ... 5 more steps
```

### Environment Variables
```bash
TEMPLATE_VERSION=v1
PROMPTS_DIR=src/marketing_project/prompts
OPENAI_API_KEY=your_key
APPROVAL_ENABLED=true
APPROVAL_THRESHOLD=0.8
```

## Job Management

All processing is asynchronous using ARQ (Redis-based queue):

```python
# Job submission
job_id = await enqueue_job(
    "process_content",
    content_data=content_json,
    content_type="blog_post"
)

# Job tracking
job_status = await get_job_status(job_id)
# Returns: { status, progress, current_step, result }
```

## Deployment Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   FastAPI   │◄────►│    Redis     │◄────►│ ARQ Workers │
│   Server    │      │   (Queue)    │      │  (1-N pods) │
└─────────────┘      └──────────────┘      └─────────────┘
      │                      │
      │                      │
      ▼                      ▼
┌─────────────┐      ┌──────────────┐
│  PostgreSQL │      │   OpenAI API │
│  (Metadata) │      │  (Functions) │
└─────────────┘      └──────────────┘
```

## Monitoring

Built-in metrics:
- **Performance**: Execution time per step
- **Quality**: Confidence scores, relevance scores
- **Reliability**: Success/failure rates
- **Cost**: Token usage tracking

## Extensibility

### Adding New Content Types
1. Create Pydantic model in `core/models.py`
2. Add processor in `processors/`
3. Register route in `api/`

### Adding Pipeline Steps
1. Create Pydantic result model in `models/pipeline_steps.py`
2. Create Jinja2 template in `prompts/v1/{lang}/`
3. Add step to `FunctionPipeline.execute_pipeline()`

### Adding Quality Metrics
1. Add field to result model (e.g., `readability_score`)
2. Update prompt template to request metric
3. Display in frontend (automatic via QualityMetricsDisplay)

## Testing

- **Unit Tests**: Individual processor logic
- **Integration Tests**: Full pipeline execution
- **Model Tests**: Pydantic validation
- **API Tests**: Endpoint responses

See `tests/` directory for comprehensive test suite.
