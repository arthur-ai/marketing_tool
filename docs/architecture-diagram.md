# Marketing Project Architecture

## Content Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        CONTENT SOURCES                         │
├─────────────────────────────────────────────────────────────────┤
│  File Sources    │  API Sources    │  Database Sources         │
│  - Local files   │  - REST APIs    │  - SQL databases          │
│  - Directory     │  - Webhooks     │  - MongoDB                │
│    watching      │  - RSS feeds    │  - Redis                  │
│  - Uploads       │  - Social Media │                          │
│                  │                 │  Web Scraping Sources     │
│                  │                 │  - BeautifulSoup          │
│                  │                 │  - Selenium               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONTENT SOURCE MANAGER                      │
├─────────────────────────────────────────────────────────────────┤
│  • Fetches content from all sources                            │
│  • Converts to ContentContext models                           │
│  • Handles caching, health checks, error recovery             │
│  • Provides unified interface for content access              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONTENT CONTEXT MODELS                      │
├─────────────────────────────────────────────────────────────────┤
│  • TranscriptContext    • BlogPostContext    • ReleaseNotesContext │
│  • Proper Pydantic validation and type safety                  │
│  • Automatic content type detection                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENTS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │ MARKETING ORCHESTRATOR │    │ CONTENT PIPELINE ORCHESTRATOR │ │
│  │                     │    │                                 │ │
│  │ Routes content to:  │    │ Manages 8-step workflow:       │ │
│  │ • Transcripts Agent │    │ 1. AnalyzeContent              │ │
│  │ • Blog Agent        │    │ 2. ExtractSEOKeywords          │ │
│  │ • Release Notes     │    │ 3. GenerateMarketingBrief      │ │
│  │   Agent             │    │ 4. GenerateArticle             │ │
│  │                     │    │ 5. OptimizeSEO                 │ │
│  │                     │    │ 6. SuggestInternalDocs         │ │
│  │                     │    │ 7. FormatContent               │ │
│  │                     │    │ 8. ApplyDesignKit              │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SPECIALIZED AGENTS                          │
├─────────────────────────────────────────────────────────────────┤
│  Transcripts Agent    │  Blog Agent           │  Release Notes  │
│  • Process podcasts   │  • Process articles   │  • Process      │
│  • Extract speakers   │  • Extract metadata   │    releases     │
│  • Generate summaries │  • Generate tags      │  • Extract      │
│                       │  • Optimize content   │    features     │
│                       │                       │  • Generate     │
│                       │                       │    summaries    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSED CONTENT                           │
├─────────────────────────────────────────────────────────────────┤
│  • Enhanced content with marketing insights                    │
│  • SEO optimizations applied                                  │
│  • Professional formatting and design                         │
│  • Ready for publication or further processing                │
└─────────────────────────────────────────────────────────────────┘
```

## Key Architectural Principles

### 1. **Central Orchestration**
- **Orchestrator agents** are the central routing points
- They decide which specialized agents to use based on content type
- They manage the complete workflow for their respective pipelines

### 2. **Content Source Abstraction**
- **Content Source Manager** provides unified interface to all sources
- Handles different source types transparently
- Converts all content to standardized `ContentContext` models

### 3. **Type Safety and Validation**
- All content is converted to proper Pydantic models
- Automatic content type detection and routing
- Full type safety throughout the pipeline

### 4. **Separation of Concerns**
- **Content Sources**: Fetch and normalize content
- **Orchestrators**: Route and coordinate processing
- **Specialized Agents**: Handle specific content types or tasks
- **Models**: Ensure data integrity and type safety

### 5. **Extensibility**
- Easy to add new content sources
- Easy to add new specialized agents
- Easy to modify orchestration logic
- Configuration-driven setup

## Pipeline Types

### Marketing Project Pipeline
- **Purpose**: Process different content types with specialized agents
- **Orchestrator**: `MarketingOrchestratorAgent`
- **Flow**: Content → Orchestrator → Specialized Agent → Processed Content

### Content Analysis Pipeline
- **Purpose**: Comprehensive 8-step content analysis and generation
- **Orchestrator**: `ContentPipelineAgent`
- **Flow**: Content → Orchestrator → 8-Step Workflow → Enhanced Content

## Benefits of This Architecture

1. **Single Responsibility**: Each component has a clear, focused role
2. **Loose Coupling**: Components can be modified independently
3. **High Cohesion**: Related functionality is grouped together
4. **Easy Testing**: Each component can be tested in isolation
5. **Scalability**: Easy to add new sources, agents, or workflows
6. **Maintainability**: Clear separation makes code easier to understand and modify
