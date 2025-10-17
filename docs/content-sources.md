# Content Source Integration

The Marketing Project now includes a comprehensive content source integration system that allows you to automatically fetch and process content from various sources.

## Overview

The content source system provides:

- **Multiple Source Types**: File, API, Database, Web Scraping, Webhooks, RSS, and Social Media
- **Automatic Content Processing**: Content is automatically converted to the appropriate format and processed through the marketing pipeline
- **Real-time Monitoring**: Directory watching, polling, and webhook support
- **Caching and Performance**: Built-in caching and rate limiting
- **Health Monitoring**: Automatic health checks and error recovery

## Supported Content Sources

### 1. File Sources
- **Local Files**: Read content from local directories
- **Directory Watching**: Monitor directories for new files
- **Upload Handling**: Process uploaded files
- **Supported Formats**: JSON, Markdown, YAML, TXT, HTML

### 2. API Sources
- **REST APIs**: Fetch content from REST endpoints
- **Authentication**: Bearer tokens, API keys, Basic auth
- **Rate Limiting**: Built-in rate limiting and retry logic
- **Pagination**: Support for paginated responses

### 3. Database Sources
- **SQL Databases**: PostgreSQL, MySQL, SQLite
- **NoSQL Databases**: MongoDB, Redis
- **Query Support**: Custom SQL queries and MongoDB queries
- **Connection Pooling**: Efficient database connections

### 4. Web Scraping Sources
- **BeautifulSoup**: For static content scraping
- **Selenium**: For JavaScript-heavy sites
- **CSS Selectors**: Flexible content extraction
- **Robots.txt Compliance**: Respects website scraping policies

### 5. Webhook Sources
- **Real-time Updates**: Receive content via webhooks
- **Signature Verification**: Secure webhook handling
- **Event Filtering**: Process only specific events

### 6. RSS Sources
- **Feed Parsing**: Parse RSS and Atom feeds
- **Content Extraction**: Extract full content from feeds
- **Multiple Feeds**: Support for multiple feed sources

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# File sources (enabled by default)
CONTENT_DIR=content/
DATA_DIR=data/

# API sources (optional)
CONTENT_API_URL=https://api.example.com
CONTENT_API_KEY=your_api_key_here

# Database sources (optional)
CONTENT_DATABASE_URL=sqlite:///./content.db

# Web scraping sources (optional)
SCRAPE_URLS=https://example.com/blog,https://news.example.com

# Content source polling
CONTENT_POLLING_ENABLED=true
CONTENT_POLLING_INTERVAL=300
CONTENT_BATCH_SIZE=10

# Content caching
CONTENT_CACHE_ENABLED=true
CONTENT_CACHE_TTL=300
CONTENT_CACHE_MAX_SIZE=1000
```

### Pipeline Configuration

The `src/marketing_project/config/pipeline.yml` file includes content source configuration:

```yaml
content_sources:
  enabled: true
  default_sources:
    - name: "local_content"
      type: "file"
      enabled: true
      config:
        file_paths: ["content/", "data/"]
        file_patterns: ["content/**/*.md", "content/**/*.json"]
        watch_directory: true
```

## Usage

### Basic Usage

The content source system is automatically integrated into both pipelines:

```python
# Marketing Project Pipeline
result = await run_marketing_project_pipeline(prompts_dir, lang)
content_manager = result["content_manager"]
processed_content = result["processed_content"]

# Content Analysis Pipeline
result = await run_content_analysis_pipeline(prompts_dir, lang)
content_manager = result["content_manager"]
processed_content = result["processed_content"]
```

### Manual Content Source Management

```python
from marketing_project.services.content_source_factory import ContentSourceManager
from marketing_project.core.content_sources import FileSourceConfig, ContentSourceType
from marketing_project.core.models import ContentContext

# Create manager
manager = ContentSourceManager()

# Add file source
file_config = FileSourceConfig(
    name="my_files",
    source_type=ContentSourceType.FILE,
    file_paths=["my_content/"],
    watch_directory=True
)
await manager.add_source_from_config(file_config)

# Fetch content as ContentContext models
content_models: List[ContentContext] = await manager.fetch_content_as_models()

# Search content models
search_results: List[ContentContext] = await manager.search_content_models("marketing automation")

# Get content by type
blog_posts: List[ContentContext] = await manager.get_content_models_by_type("blog_post")

# Get statistics
stats = await manager.get_source_statistics()
```

## Content Format

Content sources automatically convert data to proper `ContentContext` models:

### BaseContentContext
All content types inherit from `BaseContentContext`:
```python
class BaseContentContext(BaseModel):
    id: str
    title: str
    content: str
    snippet: str
    created_at: Optional[datetime] = None
    source_url: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)
```

### Specific Content Types

**TranscriptContext** - For podcasts, videos, meetings:
```python
class TranscriptContext(BaseContentContext):
    speakers: List[str] = Field(default_factory=list)
    duration: Optional[str] = None
    transcript_type: str = "podcast"
    timestamps: Optional[Dict[str, str]] = None
```

**BlogPostContext** - For articles, blog posts:
```python
class BlogPostContext(BaseContentContext):
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    word_count: Optional[int] = None
    reading_time: Optional[str] = None
```

**ReleaseNotesContext** - For software releases:
```python
class ReleaseNotesContext(BaseContentContext):
    version: str
    release_date: Optional[datetime] = None
    changes: List[str] = Field(default_factory=list)
    breaking_changes: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    bug_fixes: List[str] = Field(default_factory=list)
```

### Content Type Detection

The system automatically detects content types based on available fields and creates the appropriate model:

- **Transcripts**: Contains `speakers`, `transcript_type`, `duration` → `TranscriptContext`
- **Blog Posts**: Contains `author`, `tags`, `category` → `BlogPostContext`
- **Release Notes**: Contains `version`, `changes`, `features` → `ReleaseNotesContext`

## Advanced Features

### Caching

Content is automatically cached to improve performance:

```python
# Enable caching
manager.set_cache_ttl(300)  # 5 minutes

# Fetch with cache
results = await manager.fetch_content_with_cache(use_cache=True)

# Clear cache
manager.clear_cache()
```

### Health Monitoring

Monitor source health and restart failed sources:

```python
# Check all sources
health_status = await manager.health_check_all()

# Restart failed sources
restart_results = await manager.restart_failed_sources()
```

### Content Filtering

Filter content by type or search terms:

```python
# Get only blog posts
blog_posts = await manager.get_content_by_type("blog_post")

# Search content
search_results = await manager.search_content("AI marketing")
```

## Error Handling

The system includes comprehensive error handling:

- **Retry Logic**: Automatic retry for failed requests
- **Circuit Breaker**: Prevents cascading failures
- **Graceful Degradation**: Continues processing even if some sources fail
- **Detailed Logging**: Comprehensive logging for debugging

## Performance Optimization

- **Async Processing**: All operations are asynchronous
- **Connection Pooling**: Efficient database and HTTP connections
- **Rate Limiting**: Prevents overwhelming external services
- **Batch Processing**: Process multiple items efficiently

## Monitoring and Logging

The system provides detailed monitoring:

- **Source Statistics**: Track content counts and processing times
- **Health Checks**: Monitor source availability
- **Error Tracking**: Track and report errors
- **Performance Metrics**: Monitor processing performance

## Examples

### Example 1: Local File Processing

```python
# Content is automatically processed from the content/ directory
# Files are watched for changes and processed in real-time
```

### Example 2: API Integration

```python
# Configure API source in .env
CONTENT_API_URL=https://api.example.com
CONTENT_API_KEY=your_key

# Content is automatically fetched and processed
```

### Example 3: Database Integration

```python
# Configure database source
CONTENT_DATABASE_URL=postgresql://user:pass@localhost/content_db

# Content is automatically queried and processed
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Install missing dependencies with `pip install -r requirements.txt`
2. **Permission Errors**: Ensure proper file/directory permissions
3. **Connection Errors**: Check network connectivity and credentials
4. **Rate Limiting**: Adjust rate limits in source configuration

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("marketing_project.services").setLevel(logging.DEBUG)
```

## Contributing

To add new content source types:

1. Create a new source class inheriting from `ContentSource`
2. Implement required methods: `initialize()`, `fetch_content()`, `health_check()`
3. Add configuration model in `content_sources.py`
4. Update the factory in `content_source_factory.py`
5. Add tests and documentation

## Dependencies

The content source system requires additional dependencies:

```
aiofiles>=23.0.0
aiohttp>=3.8.0
aiosqlite>=0.19.0
asyncpg>=0.29.0
aiomysql>=0.2.0
motor>=3.3.0
redis>=4.5.0
feedparser>=6.0.0
selenium>=4.15.0
watchdog>=3.0.0
```

Install with: `pip install -r requirements.txt`
