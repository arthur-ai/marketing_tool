# SEO Keywords Hybrid Engine Implementation

## Overview

This document describes the SEO Keywords hybrid engine implementation, which allows mixing LLM and local semantic processing for different fields of the `SEOKeywordsResult` model.

## Where SEO Extraction Happens

- **Plugin**: `src/marketing_project/plugins/seo_keywords/tasks.py`
- **Engines**: `src/marketing_project/services/engines/seo_keywords/`
- **Model**: `src/marketing_project/models/pipeline_steps.py` (SEOKeywordsResult)

## What Was Wrapped

The existing LLM-based extraction in `SEOKeywordsPlugin._execute_step()` was wrapped in `LLMSEOKeywordsEngine`, maintaining backward compatibility.

## New Components

### 1. Generic Engine Framework
- `services/engines/base.py` - Engine interface
- `services/engines/registry.py` - Engine registry
- `services/engines/composer.py` - Engine composer

### 2. SEO-Specific Engines
- `services/engines/seo_keywords/llm_engine.py` - LLM-based extraction
- `services/engines/seo_keywords/local_semantic_engine.py` - Local + semantic processing
- `services/engines/seo_keywords/composer.py` - SEO-specific composer
- `services/engines/seo_keywords/seo_metrics_provider.py` - SEO metrics stub

### 3. Configuration
- `EngineConfig` model in `models/pipeline_steps.py`
- Frontend settings UI in `components/settings/seo-keywords-engine-settings.tsx`

## Available Operations

The Local Semantic Engine supports these operations:

- `extract_main_keyword` - Extract main keyword (1-3 words)
- `extract_primary_keywords` - Extract primary keywords (3-5)
- `extract_secondary_keywords` - Extract secondary keywords (5-10)
- `extract_lsi_keywords` - Extract LSI keywords (semantic neighbors)
- `extract_long_tail_keywords` - Extract long-tail keywords (3-6 words)
- `calculate_density` - Calculate keyword density analysis
- `classify_intent` - Classify search intent (zero-shot)
- `cluster_keywords` - Cluster keywords by semantic similarity
- `get_seo_metrics` - Get SEO metrics (difficulty, volume)

## Configuration Examples

### Example 1: LLM Only (Default)
```json
{
  "default_engine": "llm",
  "field_overrides": {}
}
```

### Example 2: Local Semantic Only
```json
{
  "default_engine": "local_semantic",
  "field_overrides": {}
}
```

### Example 3: Mixed Mode
```json
{
  "default_engine": "llm",
  "field_overrides": {
    "keyword_density_analysis": "local_semantic",
    "search_intent": "local_semantic",
    "keyword_clusters": "local_semantic"
  }
}
```

## Implementation Details

### Local Semantic Engine Processing

1. **Preprocessing** (syntok + UDPipe)
   - Parse document into title, headings, body
   - Extract tokens, noun chunks, sentences

2. **Candidate Generation**
   - YAKE: Statistical phrase extraction
   - RAKE: Stopword-separated keyphrases
   - TF-IDF: N-gram extraction

3. **Semantic Refinement** (Hugging Face)
   - Embed document and candidates
   - Use cosine similarity for ranking
   - Remove near-duplicates

4. **Field Population**
   - Each field uses appropriate operation
   - Operations can depend on other fields

5. **Density & Placement** (UDPipe)
   - Count occurrences (exact + lemma-based)
   - Calculate density
   - Record placement locations

6. **Intent Classification** (Zero-shot)
   - Classify main keyword
   - Labels: informational, transactional, navigational, commercial

## Dependencies

Required packages (added to `requirements.in`):
- `syntok>=1.4.4` - Tokenization and sentence segmentation
- `ufal.udpipe>=1.2.0` - Lemmatization, POS tagging, and dependency parsing
- `yake>=0.4.8`
- `rake-nltk>=1.0.6`

Post-installation:
The UDPipe English model is automatically downloaded on first use and cached at `~/.udpipe/models/`.
If you need to download it manually:
```bash
mkdir -p ~/.udpipe/models
wget -O ~/.udpipe/models/english-ewt-ud-2.5-191206.udpipe \
  https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/english-ewt-ud-2.5-191206.udpipe
```

## Performance Considerations

- **LLM Mode**: Slower, higher quality, requires API calls
- **Local Semantic Mode**: Faster, no API calls, good quality
- **Mixed Mode**: Balance between speed and quality

## Usage

### Backend

```python
from marketing_project.models.pipeline_steps import EngineConfig

# In pipeline config
engine_config = EngineConfig(
    default_engine="llm",
    field_overrides={
        "keyword_density_analysis": "local_semantic"
    }
)
```

### Frontend

1. Navigate to Settings → SEO Keywords Engine tab
2. Select default engine
3. Add field-level overrides as needed
4. Save settings

## Testing

See:
- `tests/services/test_engines_framework.py` - Framework tests
- `tests/services/test_seo_keywords_engines.py` - Engine tests
- `tests/integration/test_seo_keywords_hybrid.py` - Integration tests

## Future Enhancements

- Real SEO metrics API integration
- Caching for local semantic processing
- Additional engine types (hybrid, cached_llm)
- Performance optimizations
