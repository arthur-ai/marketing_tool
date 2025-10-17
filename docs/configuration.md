# Configuration

## Overview

The Arthur Marketing Generator is highly configurable through YAML configuration files, environment variables, and runtime parameters. This document covers all configuration options and their usage.

## Pipeline Configuration

### Main Configuration File

The primary configuration is stored in `src/marketing_project/config/pipeline.yml`:

```yaml
version: "2"

pipelines:
  default:
    - AnalyzeContent
    - ExtractSEOKeywords
    - GenerateMarketingBrief
    - GenerateArticle
    - OptimizeSEO
    - SuggestInternalDocs
    - FormatContent

# Context passing configuration for better data flow
context_passing:
  AnalyzeContent:
    output: "content_analysis_result"
    next_input: "content_analysis_result"
    data_structure: "Dict[str, Any]"
  ExtractSEOKeywords:
    input: "content_analysis_result"
    output: "seo_keywords_result"
    next_input: "seo_keywords_result"
    data_structure: "Dict[str, Any]"
  # ... (continued for all pipeline steps)

sub_pipelines:
  content_analysis:
    - AnalyzeContentType
    - ExtractContentMetadata
    - ValidateContentStructure
  seo_keywords:
    - ExtractPrimaryKeywords
    - ExtractSecondaryKeywords
    - AnalyzeKeywordDensity
    - GenerateKeywordSuggestions
  # ... (detailed sub-pipeline steps)

branching:
  AnalyzeContent:
    blog_post: content_analysis
    release_notes: content_analysis
    transcript: content_analysis
    email: content_analysis
    other: content_analysis
```

### Pipeline Configuration Options

#### `version`
- **Type**: String
- **Default**: "2"
- **Description**: Configuration file version

#### `pipelines`
- **Type**: Dictionary
- **Description**: Defines available pipelines and their steps

#### `context_passing`
- **Type**: Dictionary
- **Description**: Defines data flow between pipeline steps
- **Options**:
  - `input`: Input data source
  - `output`: Output data destination
  - `next_input`: Next step input source
  - `data_structure`: Expected data structure type

#### `sub_pipelines`
- **Type**: Dictionary
- **Description**: Detailed steps for each major pipeline phase

#### `branching`
- **Type**: Dictionary
- **Description**: Content type routing to appropriate analysis

## Environment Variables

### Required Variables

```bash
# API Configuration
ARTHUR_API_KEY=your-api-key-here
ARTHUR_API_URL=https://api.arthur.com/v1

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/arthur
REDIS_URL=redis://localhost:6379/0

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST_LIMIT=20
```

### Optional Variables

```bash
# Content Processing
MAX_CONTENT_LENGTH=10000
MIN_CONTENT_LENGTH=100
DEFAULT_LANGUAGE=en

# SEO Configuration
DEFAULT_KEYWORD_DENSITY=0.02
MAX_KEYWORDS_PER_CONTENT=10
MIN_KEYWORD_SCORE=0.5

# Marketing Configuration
DEFAULT_TARGET_AUDIENCE=general
DEFAULT_CONTENT_CATEGORY=tutorial
DEFAULT_WORD_COUNT=1500

# Formatting Configuration
DEFAULT_HEADING_STYLE=title_case
DEFAULT_LIST_STYLE=bullet
DEFAULT_PARAGRAPH_SPACING=double

# Performance Configuration
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30
CACHE_TTL=3600

# Monitoring Configuration
ENABLE_METRICS=true
METRICS_PORT=8080
HEALTH_CHECK_PORT=8081
```

## Plugin Configuration

### Content Analysis Plugin

```yaml
content_analysis:
  readability:
    min_score: 60
    max_score: 100
    target_score: 75

  quality_checks:
    min_word_count: 100
    max_word_count: 10000
    require_title: true
    require_snippet: false

  seo_analysis:
    check_keywords: true
    check_meta_tags: true
    check_heading_structure: true
    check_internal_links: true
```

### SEO Keywords Plugin

```yaml
seo_keywords:
  extraction:
    method: "tfidf"  # tfidf, textrank, yake
    max_keywords: 10
    min_keyword_length: 3
    max_keyword_length: 20

  scoring:
    frequency_weight: 0.4
    position_weight: 0.3
    length_weight: 0.2
    uniqueness_weight: 0.1

  filtering:
    min_frequency: 2
    min_score: 0.5
    exclude_stopwords: true
    exclude_numbers: false
```

### Marketing Brief Plugin

```yaml
marketing_brief:
  target_audience:
    default_demographics:
      age_range: "25-45"
      experience_level: "intermediate"
      industry: "technology"

  content_objectives:
    default_primary: "educate"
    default_secondary: "generate_leads"
    default_tertiary: "establish_authority"

  success_metrics:
    engagement:
      min_time_on_page: 180  # seconds
      min_bounce_rate: 0.7
    conversion:
      min_conversion_rate: 0.05
      min_lead_quality: 0.8
    sharing:
      min_social_shares: 10
      min_email_forwards: 5
```

### Article Generation Plugin

```yaml
article_generation:
  structure:
    min_sections: 3
    max_sections: 8
    min_words_per_section: 200
    max_words_per_section: 800

  content:
    default_tone: "professional"
    default_style: "informative"
    use_active_voice: true
    sentence_length:
      min: 10
      max: 25
      target: 15

  seo_integration:
    keyword_density_target: 0.02
    title_keyword_placement: "beginning"
    heading_keyword_placement: "natural"
```

### SEO Optimization Plugin

```yaml
seo_optimization:
  title_tags:
    min_length: 30
    max_length: 60
    target_length: 50
    include_primary_keyword: true
    include_brand: false

  meta_descriptions:
    min_length: 120
    max_length: 160
    target_length: 150
    include_call_to_action: true
    include_primary_keyword: true

  headings:
    use_h1_once: true
    h2_keyword_placement: "natural"
    h3_keyword_placement: "optional"
    heading_hierarchy: "logical"

  content_structure:
    min_paragraphs: 3
    max_paragraphs: 20
    paragraph_length:
      min: 50
      max: 200
      target: 100
```

### Internal Docs Plugin

```yaml
internal_docs:
  gap_analysis:
    check_technical_terms: true
    check_missing_sections: true
    check_unexplained_concepts: true
    min_gap_score: 0.3

  suggestions:
    max_suggestions: 5
    priority_levels: ["high", "medium", "low"]
    target_audiences: ["beginners", "intermediate", "advanced", "all"]

  cross_references:
    min_relevance_score: 0.7
    max_suggestions: 3
    include_related_topics: true
```

### Content Formatting Plugin

```yaml
content_formatting:
  style_guide:
    heading_style: "title_case"  # title_case, sentence_case, all_caps
    list_style: "bullet"  # bullet, numbered, dash
    paragraph_spacing: "double"  # single, double, custom
    quote_style: "blockquote"  # blockquote, italic, bold
    code_style: "fenced"  # fenced, inline, indented
    link_style: "markdown"  # markdown, html, plain
    emphasis_style: "bold_italic"  # bold, italic, bold_italic

  readability:
    target_grade_level: 8
    max_sentence_length: 25
    min_sentence_length: 10
    use_transition_words: true
    break_long_paragraphs: true

  visual_elements:
    add_headers: true
    add_lists: true
    add_quotes: true
    add_code_blocks: true
    add_images: false
    add_tables: true
```

## Agent Configuration

### Agent Settings

```yaml
agents:
  content_pipeline_agent:
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 4000
    timeout: 30

  seo_keywords_agent:
    model: "gpt-3.5-turbo"
    temperature: 0.5
    max_tokens: 2000
    timeout: 20

  marketing_brief_agent:
    model: "gpt-4"
    temperature: 0.8
    max_tokens: 3000
    timeout: 25

  article_generation_agent:
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 4000
    timeout: 30

  seo_optimization_agent:
    model: "gpt-3.5-turbo"
    temperature: 0.6
    max_tokens: 2000
    timeout: 20

  internal_docs_agent:
    model: "gpt-3.5-turbo"
    temperature: 0.5
    max_tokens: 2000
    timeout: 20

  content_formatting_agent:
    model: "gpt-3.5-turbo"
    temperature: 0.4
    max_tokens: 2000
    timeout: 20
```

### Model Configuration

```yaml
models:
  gpt-4:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    max_retries: 3
    retry_delay: 1

  gpt-3.5-turbo:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    max_retries: 3
    retry_delay: 1

  claude-3:
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: "https://api.anthropic.com/v1"
    max_retries: 3
    retry_delay: 1
```

## Logging Configuration

### Log Levels

```yaml
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: json  # json, text, structured
  handlers:
    console:
      enabled: true
      level: INFO
    file:
      enabled: true
      level: DEBUG
      path: "logs/arthur.log"
      max_size: "10MB"
      backup_count: 5
    syslog:
      enabled: false
      level: WARNING
      host: "localhost"
      port: 514
```

### Log Categories

```yaml
loggers:
  marketing_project.agents:
    level: DEBUG
    handlers: [console, file]

  marketing_project.plugins:
    level: INFO
    handlers: [console, file]

  marketing_project.core:
    level: INFO
    handlers: [console, file]

  marketing_project.services:
    level: WARNING
    handlers: [console, file]
```

## Database Configuration

### PostgreSQL

```yaml
database:
  postgresql:
    host: "localhost"
    port: 5432
    database: "arthur"
    username: "arthur_user"
    password: "${DATABASE_PASSWORD}"
    ssl_mode: "prefer"
    pool_size: 10
    max_overflow: 20
    pool_timeout: 30
    pool_recycle: 3600
```

### Redis

```yaml
cache:
  redis:
    host: "localhost"
    port: 6379
    database: 0
    password: "${REDIS_PASSWORD}"
    max_connections: 20
    socket_timeout: 5
    socket_connect_timeout: 5
    retry_on_timeout: true
```

## Monitoring Configuration

### Metrics

```yaml
monitoring:
  metrics:
    enabled: true
    port: 8080
    path: "/metrics"
    interval: 60  # seconds

  health_checks:
    enabled: true
    port: 8081
    path: "/health"
    interval: 30  # seconds

  alerts:
    enabled: true
    webhook_url: "${ALERT_WEBHOOK_URL}"
    thresholds:
      error_rate: 0.05
      response_time: 5.0
      memory_usage: 0.8
      cpu_usage: 0.8
```

### Performance Monitoring

```yaml
performance:
  profiling:
    enabled: false
    sample_rate: 0.1
    output_dir: "profiles"

  tracing:
    enabled: false
    jaeger_endpoint: "http://localhost:14268/api/traces"
    sample_rate: 0.1

  benchmarks:
    enabled: false
    output_dir: "benchmarks"
    interval: 3600  # seconds
```

## Security Configuration

### Authentication

```yaml
security:
  authentication:
    method: "api_key"  # api_key, jwt, oauth2
    api_key_header: "X-API-Key"
    jwt_secret: "${JWT_SECRET}"
    jwt_expiry: 3600  # seconds

  authorization:
    enabled: true
    default_role: "user"
    roles:
      admin:
        permissions: ["read", "write", "delete", "admin"]
      user:
        permissions: ["read", "write"]
      viewer:
        permissions: ["read"]

  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst_limit: 20
    per_ip: true
    per_user: true
```

### Data Protection

```yaml
data_protection:
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
    key: "${ENCRYPTION_KEY}"

  anonymization:
    enabled: false
    fields: ["email", "phone", "ssn"]
    method: "hash"  # hash, mask, remove

  retention:
    logs: 90  # days
    metrics: 30  # days
    content: 365  # days
    user_data: 730  # days
```

## Runtime Configuration

### Application Settings

```yaml
application:
  name: "Arthur Marketing Generator"
  version: "1.0.0"
  environment: "production"  # development, staging, production
  debug: false
  host: "0.0.0.0"
  port: 8000

  workers: 4
  worker_class: "uvicorn.workers.UvicornWorker"
  worker_connections: 1000
  max_requests: 1000
  max_requests_jitter: 100
  timeout: 30
  keepalive: 2
```

### Feature Flags

```yaml
features:
  content_analysis:
    enabled: true
    experimental_features: false

  seo_optimization:
    enabled: true
    advanced_analysis: true

  marketing_brief:
    enabled: true
    competitor_analysis: false

  article_generation:
    enabled: true
    ai_writing: true

  internal_docs:
    enabled: true
    cross_reference: true

  content_formatting:
    enabled: true
    visual_elements: true
```

## Configuration Validation

The system validates all configuration on startup:

```python
# Example validation
def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration file."""
    required_sections = [
        "pipelines", "context_passing", "sub_pipelines"
    ]

    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing required section: {section}")

    # Validate pipeline steps
    for pipeline_name, steps in config["pipelines"].items():
        for step in steps:
            if step not in AVAILABLE_STEPS:
                raise ConfigurationError(f"Unknown pipeline step: {step}")

    return True
```

## Configuration Examples

### Development Configuration

```yaml
# config/development.yml
version: "2"
environment: "development"
debug: true
log_level: "DEBUG"

pipelines:
  default:
    - AnalyzeContent
    - ExtractSEOKeywords
    - GenerateMarketingBrief

# ... (minimal configuration for development)
```

### Production Configuration

```yaml
# config/production.yml
version: "2"
environment: "production"
debug: false
log_level: "INFO"

pipelines:
  default:
    - AnalyzeContent
    - ExtractSEOKeywords
    - GenerateMarketingBrief
    - GenerateArticle
    - OptimizeSEO
    - SuggestInternalDocs
    - FormatContent

# ... (full configuration for production)
```

### Testing Configuration

```yaml
# config/testing.yml
version: "2"
environment: "testing"
debug: true
log_level: "DEBUG"

pipelines:
  test:
    - AnalyzeContent
    - ExtractSEOKeywords

# ... (minimal configuration for testing)
```

## Configuration Management

### Environment-Specific Configs

```bash
# Load different configs based on environment
export ARTHUR_ENV=production
export ARTHUR_CONFIG_PATH=config/production.yml

# Or use environment variables
export ARTHUR_PIPELINE_STEPS="AnalyzeContent,ExtractSEOKeywords"
export ARTHUR_LOG_LEVEL=INFO
```

### Configuration Overrides

```python
# Override configuration at runtime
config = load_config("config/production.yml")
config.override({
    "pipelines": {
        "custom": ["AnalyzeContent", "FormatContent"]
    }
})
```

### Configuration Validation

```python
# Validate configuration before use
try:
    config = load_config("config/production.yml")
    validate_config(config)
    print("Configuration is valid")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    sys.exit(1)
```
