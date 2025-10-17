# Getting Started

## Overview

This guide will help you get up and running with the Arthur Marketing Generator quickly. You'll learn how to install, configure, and use the system to analyze and generate marketing content.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **pip** or **poetry** for package management
- **Git** for version control
- **API Keys** for AI models (OpenAI, Anthropic, etc.)
- **Database** (PostgreSQL recommended)
- **Redis** (for caching)

## Installation

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/arthur/marketing-generator.git
cd marketing-generator

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Option 2: Using poetry

```bash
# Clone the repository
git clone https://github.com/arthur/marketing-generator.git
cd marketing-generator

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Option 3: Using Docker

```bash
# Clone the repository
git clone https://github.com/arthur/marketing-generator.git
cd marketing-generator

# Build the Docker image
docker build -t arthur-marketing-generator .

# Run the container
docker run -p 8000:8000 arthur-marketing-generator
```

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/arthur
REDIS_URL=redis://localhost:6379/0

# Application Configuration
ARTHUR_ENV=development
LOG_LEVEL=INFO
```

### 2. Database Setup

```bash
# Create the database
createdb arthur

# Run migrations
python -m marketing_project.migrations upgrade

# Or using Alembic directly
alembic upgrade head
```

### 3. Redis Setup

```bash
# Start Redis server
redis-server

# Or using Docker
docker run -d -p 6379:6379 redis:alpine
```

## Quick Start

### 1. Basic Usage

```python
from marketing_project import MarketingGenerator
from marketing_project.models import BlogPostContext

# Initialize the generator
generator = MarketingGenerator()

# Create content
content = BlogPostContext(
    id="example-1",
    title="Marketing Automation Guide",
    content="Marketing automation is a powerful tool for businesses...",
    author="John Doe",
    tags=["marketing", "automation"],
    category="tutorial"
)

# Run the full pipeline
result = generator.run_pipeline(content)
print(result)
```

### 2. Individual Steps

```python
# Analyze content
analysis = generator.analyze_content(content)
print(f"Content quality: {analysis['data']['content_quality']}")

# Extract keywords
keywords = generator.extract_keywords(content)
print(f"Keywords: {keywords['data']['keywords']}")

# Generate marketing brief
brief = generator.generate_brief(content, keywords['data'])
print(f"Brief: {brief['data']['executive_summary']}")

# Generate article
article = generator.generate_article(brief['data'], keywords['data'])
print(f"Article: {article['data']['content']}")
```

### 3. Custom Pipeline

```python
# Create custom pipeline
custom_pipeline = [
    "AnalyzeContent",
    "ExtractSEOKeywords",
    "GenerateArticle",
    "FormatContent"
]

# Run custom pipeline
result = generator.run_pipeline(content, pipeline=custom_pipeline)
```

## Examples

### Example 1: Blog Post Analysis

```python
from marketing_project import MarketingGenerator
from marketing_project.models import BlogPostContext

# Initialize
generator = MarketingGenerator()

# Create blog post content
blog_post = BlogPostContext(
    id="blog-1",
    title="10 Marketing Automation Tips for Small Businesses",
    content="""
    Marketing automation can revolutionize how small businesses reach their customers.
    Here are 10 essential tips to get started:

    1. Start with email marketing
    2. Use lead scoring
    3. Personalize your messages
    4. Track customer behavior
    5. Automate social media posting
    6. Use chatbots for customer service
    7. Implement drip campaigns
    8. Analyze your results
    9. A/B test your campaigns
    10. Continuously optimize

    By following these tips, you can create a powerful marketing automation system
    that drives results for your business.
    """,
    author="Jane Smith",
    tags=["marketing", "automation", "small business"],
    category="tutorial"
)

# Run analysis
result = generator.analyze_content(blog_post)

# Print results
print("Content Analysis Results:")
print(f"Word count: {result['data']['content_quality']['word_count']}")
print(f"Readability score: {result['data']['content_quality']['readability_score']}")
print(f"SEO potential: {result['data']['seo_potential']['has_keywords']}")
print(f"Marketing value: {result['data']['marketing_value']['engagement_potential']}")
```

### Example 2: SEO Optimization

```python
# Extract keywords
keywords = generator.extract_keywords(blog_post)
print("Keywords found:")
for keyword in keywords['data']['keywords']:
    print(f"- {keyword['keyword']}: {keyword['score']}")

# Generate SEO-optimized content
seo_result = generator.optimize_seo(blog_post, keywords['data'])
print("SEO optimizations:")
for rec in seo_result['data']['recommendations']:
    print(f"- {rec}")
```

### Example 3: Marketing Brief Generation

```python
# Generate marketing brief
brief = generator.generate_brief(blog_post, keywords['data'])
print("Marketing Brief:")
print(f"Target audience: {brief['data']['target_audience']}")
print(f"Content objectives: {brief['data']['content_objectives']}")
print(f"Success metrics: {brief['data']['success_metrics']}")
```

### Example 4: Article Generation

```python
# Generate article structure
article = generator.generate_article(brief['data'], keywords['data'])
print("Generated Article:")
print(f"Title: {article['data']['title']}")
print(f"Meta description: {article['data']['meta_description']}")
print(f"Word count target: {article['data']['word_count_target']}")
print(f"Main sections: {len(article['data']['main_sections'])}")
```

## API Usage

### REST API

Start the API server:

```bash
python -m marketing_project.server
```

Make API calls:

```bash
# Analyze content
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "id": "example-1",
    "title": "Marketing Automation Guide",
    "content": "Marketing automation is...",
    "author": "John Doe",
    "tags": ["marketing", "automation"]
  }'

# Run full pipeline
curl -X POST http://localhost:8000/api/v1/pipeline \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "id": "example-1",
    "title": "Marketing Automation Guide",
    "content": "Marketing automation is...",
    "author": "John Doe",
    "tags": ["marketing", "automation"]
  }'
```

### GraphQL API

```graphql
# Query content analysis
query {
  analyzeContent(input: {
    id: "example-1"
    title: "Marketing Automation Guide"
    content: "Marketing automation is..."
    author: "John Doe"
    tags: ["marketing", "automation"]
  }) {
    success
    data {
      contentQuality {
        wordCount
        readabilityScore
        completenessScore
      }
      seoPotential {
        hasKeywords
        titleOptimization
        contentStructure
      }
      marketingValue {
        engagementPotential
        conversionPotential
        shareability
      }
    }
  }
}
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

plugins:
  content_analysis:
    readability:
      min_score: 50
      target_score: 70
  seo_keywords:
    extraction:
      max_keywords: 5
      min_score: 0.3
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

plugins:
  content_analysis:
    readability:
      min_score: 60
      target_score: 75
  seo_keywords:
    extraction:
      max_keywords: 10
      min_score: 0.5
```

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_plugins.py

# Run with coverage
pytest --cov=marketing_project

# Run integration tests
pytest tests/integrations/
```

### Test Configuration

```python
# tests/conftest.py
import pytest
from marketing_project import MarketingGenerator
from marketing_project.models import BlogPostContext

@pytest.fixture
def generator():
    return MarketingGenerator()

@pytest.fixture
def sample_content():
    return BlogPostContext(
        id="test-1",
        title="Test Article",
        content="This is a test article.",
        author="Test Author",
        tags=["test"],
        category="tutorial"
    )
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Ensure the package is installed
pip install -e .

# Check Python path
python -c "import marketing_project; print(marketing_project.__file__)"
```

#### 2. Database Connection Issues

```bash
# Check database connection
python -c "from marketing_project.core.database import check_connection; check_connection()"

# Run migrations
alembic upgrade head
```

#### 3. API Key Issues

```bash
# Check environment variables
echo $OPENAI_API_KEY

# Test API connection
python -c "from marketing_project.core.openai_client import test_connection; test_connection()"
```

#### 4. Redis Connection Issues

```bash
# Check Redis connection
redis-cli ping

# Test Redis connection
python -c "from marketing_project.core.cache import test_connection; test_connection()"
```

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export LOG_LEVEL=DEBUG
```

### Performance Issues

```python
# Enable profiling
from marketing_project.core.profiler import enable_profiling
enable_profiling()

# Check memory usage
from marketing_project.core.monitoring import get_memory_usage
print(get_memory_usage())
```

## Next Steps

Now that you have the Arthur Marketing Generator up and running:

1. **Explore the Examples**: Check out the examples in the `examples/` directory
2. **Read the Documentation**: Dive deeper into the [API Reference](api-reference.md)
3. **Configure for Your Needs**: Customize the [Configuration](configuration.md) for your use case
4. **Build Custom Plugins**: Learn how to [create custom plugins](plugin-system.md#creating-custom-plugins)
5. **Deploy to Production**: Follow the [deployment guide](deployment.md)

## Support

If you need help:

1. **Check the Documentation**: Most questions are answered in the docs
2. **Search Issues**: Look for similar issues in the GitHub repository
3. **Create an Issue**: Open a new issue with detailed information
4. **Join the Community**: Participate in discussions and get help from other users

## Contributing

We welcome contributions! See [Contributing](contributing.md) for guidelines on:

- Code style and standards
- Testing requirements
- Documentation updates
- Pull request process
- Issue reporting

Happy coding! 🚀
