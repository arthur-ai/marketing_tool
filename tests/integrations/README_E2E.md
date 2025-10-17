# End-to-End Integration Tests

This directory contains comprehensive end-to-end integration tests for the Marketing Project pipeline.

## Overview

The E2E tests validate the complete content processing pipeline using real content from the `/content/` folder and test both the original 3-agent pipeline and the new 8-step content analysis pipeline.

## Test Coverage

### ✅ Content Source Integration
- File source content loading from `/content/` folder
- Content type detection and routing
- Multiple content source support (file, API, database, web scraping)

### ✅ Pipeline Execution
- **Original 3-Agent Pipeline**: transcripts, blog, release notes
- **New 8-Step Pipeline**: Complete content analysis workflow
  1. AnalyzeContent
  2. ExtractSEOKeywords
  3. GenerateMarketingBrief
  4. GenerateArticle
  5. OptimizeSEO
  6. SuggestInternalDocs
  7. FormatContent
  8. ApplyDesignKit

### ✅ Agent Initialization
- All 12 individual agents can be created
- Content pipeline orchestrator agent
- Proper tool integration and configuration

### ✅ Data Flow Validation
- Content flows correctly between pipeline steps
- Context passing between agents
- Output structure validation

### ✅ Error Handling
- Graceful error handling and recovery
- Invalid configuration handling
- Missing content handling

## Prerequisites

### Required Environment Variables
```bash
# Copy environment file
cp env.example .env

# Add your OpenAI API key
OPENAI_API_KEY=your_actual_openai_key_here
```

### Required Files
- `content/example_blog_post.json` - Blog post content for testing
- `content/example_transcript.md` - Podcast transcript for testing
- `prompts/v1/en/*.j2` - Prompt templates
- `config/pipeline.yml` - Pipeline configuration

### Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Running the Tests

### Option 1: Run E2E tests only
```bash
# Run all E2E tests
pytest -m e2e -v

# Run with detailed output
pytest -m e2e -v -s

# Run specific E2E test
pytest -m e2e -v -k "content_source"
```

### Option 2: Run all integration tests (including E2E)
```bash
# Run all integration tests
pytest -m integration -v

# Run with detailed output
pytest -m integration -v -s
```

### Option 3: Run specific test file
```bash
# Run the E2E test file directly
pytest tests/integrations/test_e2e_full_pipeline.py -v

# Run with detailed output
pytest tests/integrations/test_e2e_full_pipeline.py -v -s
```

### Option 4: Run with coverage
```bash
# Run E2E tests with coverage
pytest -m e2e -v --cov=src --cov-report=html

# Run all tests with coverage
pytest --cov=src --cov-report=html
```

### Option 5: Run only unit tests (exclude E2E and integration)
```bash
# Run only unit tests
pytest -m "not e2e and not integration" -v
```

## Test Structure

### `test_e2e_full_pipeline.py`
Main E2E test file containing:

- **`TestE2EFullPipeline`** class with individual test methods
- **`test_complete_e2e_workflow`** - Complete end-to-end workflow test
- Comprehensive fixtures for test setup
- Error handling and validation

### Test Methods

1. **`test_content_source_loading`** - Validates content source integration
2. **`test_original_3_agent_pipeline`** - Tests the original 3-agent pipeline
3. **`test_content_analysis_pipeline_8_steps`** - Tests the new 8-step pipeline
4. **`test_individual_agent_initialization`** - Tests all individual agents
5. **`test_pipeline_step_execution`** - Tests pipeline execution with real content
6. **`test_error_handling_and_recovery`** - Tests error handling
7. **`test_content_type_detection`** - Tests content type detection
8. **`test_pipeline_configuration_loading`** - Tests configuration loading
9. **`test_complete_e2e_workflow`** - Complete end-to-end workflow validation

## Expected Output

When tests pass, you should see:
```
$ pytest -m e2e -v
============================= test session starts ==============================
platform darwin -- Python 3.12.0, pytest-7.4.0, pluggy-1.0.0
rootdir: /path/to/arthur-marketing-generator
collected 9 items

tests/integrations/test_e2e_full_pipeline.py::TestE2EFullPipeline::test_content_source_loading PASSED
tests/integrations/test_e2e_full_pipeline.py::TestE2EFullPipeline::test_original_3_agent_pipeline PASSED
tests/integrations/test_e2e_full_pipeline.py::TestE2EFullPipeline::test_content_analysis_pipeline_8_steps PASSED
tests/integrations/test_e2e_full_pipeline.py::TestE2EFullPipeline::test_individual_agent_initialization PASSED
tests/integrations/test_e2e_full_pipeline.py::TestE2EFullPipeline::test_pipeline_step_execution PASSED
tests/integrations/test_e2e_full_pipeline.py::TestE2EFullPipeline::test_error_handling_and_recovery PASSED
tests/integrations/test_e2e_full_pipeline.py::TestE2EFullPipeline::test_content_type_detection PASSED
tests/integrations/test_e2e_full_pipeline.py::TestE2EFullPipeline::test_pipeline_configuration_loading PASSED
tests/integrations/test_e2e_full_pipeline.py::test_complete_e2e_workflow PASSED

============================== 9 passed in 45.67s ==============================
```

## Troubleshooting

### Common Issues

1. **Missing OpenAI API Key**
   ```
   ❌ .env file not found. Please create it from env.example
   ```
   **Solution**: Copy `env.example` to `.env` and add your `OPENAI_API_KEY`

2. **Missing Content Files**
   ```
   ❌ Missing required content files: ['content/example_blog_post.json']
   ```
   **Solution**: Ensure content files exist in the `/content/` directory

3. **Missing Prompt Templates**
   ```
   ❌ prompts/v1/en directory not found
   ```
   **Solution**: Ensure prompt templates are in the correct location

4. **Agent Initialization Failures**
   ```
   Failed to create seo_keywords agent: OpenAI API key not found
   ```
   **Solution**: Verify your `OPENAI_API_KEY` is correctly set in `.env`

### Debug Mode

For detailed debugging, run with verbose output:
```bash
python run_e2e_test.py --verbose --no-capture
```

This will show all log output and help identify where the pipeline might be failing.

## Test Data

The tests use real content from:
- `content/example_blog_post.json` - Marketing automation blog post
- `content/example_transcript.md` - AI in marketing podcast transcript

These files are processed through the complete pipeline to validate:
- Content type detection
- Pipeline routing
- Agent execution
- Output generation
- Data flow between steps

## Performance

The E2E tests typically take 2-5 minutes to complete, depending on:
- OpenAI API response times
- Number of content items processed
- System performance

For faster testing during development, you can run specific test methods:
```bash
pytest tests/integrations/test_e2e_full_pipeline.py::TestE2EFullPipeline::test_content_source_loading -v
```
