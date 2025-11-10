# Blog Post Process - Complete Step Review

## Overview

The Blog Post process consists of **1 initial validation step + 7 pipeline steps** that transform raw blog post content into a fully optimized, formatted, and design-ready publication.

---

## Process Flow

```
Input Submission → Validation → Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6 → Step 7 → Final Output
```

---

## Step 0: Input Submission & Validation

### What It Is
Initial blog post submission and validation before processing begins.

### Position in Process
**Entry point** - Before any pipeline steps run.

### Input Required
Blog post data as `BlogPostContext` model with:
- **Required fields:**
  - `id` (str): Unique identifier
  - `title` (str): Blog post title
  - `content` (str): Full blog post content
  - `snippet` (str): Short preview/summary
- **Optional fields:**
  - `author` (str): Author name
  - `tags` (List[str]): Content tags
  - `category` (str): Content category
  - `word_count` (int): Word count
  - `reading_time` (str): Estimated reading time
  - `created_at` (datetime): Creation timestamp
  - `source_url` (str): Source URL
  - `metadata` (Dict[str, str]): Additional metadata

### Output
- **On Success:** Validated `BlogPostContext` object passed to pipeline
- **On Error:** Error response with validation details

### Approvals Required
❌ **No approvals** - This is a validation step only.

---

## Step 1: SEO Keywords Extraction

### What It Is
Extracts comprehensive SEO keywords from the blog post content, including primary keywords, secondary keywords, LSI (Latent Semantic Indexing) keywords, analyzes keyword density, and determines search intent.

### Position in Process
**Step 1** - First pipeline step. Runs immediately after validation.

### Input Required
- Blog post title
- Blog post content (first 2000 chars)
- Category
- Tags

### Output
`SEOKeywordsResult` containing:
- `main_keyword` (str): **The single most important keyword** - primary focus for content
- `primary_keywords` (List[str]): 3-5 main keywords (includes main_keyword)
- `secondary_keywords` (List[str]): 5-10 supporting keywords
- `lsi_keywords` (List[str]): Semantic keywords for context
- `keyword_density` (Dict[str, float]): Keyword frequency analysis per keyword
- `search_intent` (str): informational/transactional/navigational/commercial
- `keyword_difficulty` (str): easy/medium/hard
- `confidence_score` (float): 0-1, AI confidence in analysis
- `relevance_score` (float): 0-100, keyword relevance score

### Approvals Required
✅ **Yes** - If approvals are enabled and `seo_keywords` is in the approval agents list.

#### Approval UI Features:
- **SEO Analysis Metrics Display:**
  - **Search Intent:** Chip showing user search intent (informational/transactional/navigational/commercial)
  - **Keyword Difficulty:** Color-coded chip (green=easy, yellow=medium, red=hard)
  - **Relevance Score:** Progress bar and percentage (color-coded: green ≥70%, yellow ≥50%, red <50%)
  - **Confidence Score:** Progress bar and percentage (color-coded: green ≥70%, yellow ≥50%, red <50%)
  - **Keyword Density Analysis:** Table showing density for each primary keyword with:
    - Visual progress bars for each keyword
    - Exact percentage values
    - "Main" badge for the main keyword
    - Note about optimal density (1-2% per keyword)
- **Main Keyword Selection (Required):**
  - Radio buttons for selecting ONE main keyword from primary keywords
  - Auto-selects AI-suggested main keyword by default
  - Shows "AI Suggested" badge for AI-recommended keyword
  - Displays keyword density percentage for each option
  - **Required** - User must select a main keyword to proceed
- **Supporting Keywords Selection:**
  - Primary keywords with checkboxes (select/deselect all) - main keyword always included
  - Secondary keywords with checkboxes (select/deselect all)
  - LSI keywords with checkboxes (select/deselect all)
  - Shows count of selected vs total keywords
- **Decision Options:**
  - **Approve:** Use all keywords as-is (with selected main keyword)
  - **Modify:** Select main keyword and supporting keywords to keep (special keyword selection UI)
  - **Reject:** Reject the entire keyword set (triggers auto-regeneration with user guidance if enabled)
- **Display Info:**
  - All analysis metrics displayed prominently
  - Confidence score as percentage with progress bar
  - Suggestions for reviewer
  - Output shown in tabs (Output/Input/Raw JSON)

---

## Step 2: Marketing Brief Generation

### What It Is
Creates a comprehensive marketing strategy and brief that defines target audience, messaging, content strategy, KPIs, and distribution channels.

### Position in Process
**Step 2** - Runs after SEO keywords extraction. Uses keywords from Step 1.

### Input Required
- **Content type** (str): Content type (e.g., "blog_post")
- **Main keyword** (str): The single main keyword from Step 1 (selected by user)

**Note:** No longer uses blog post title or content summary - focuses solely on the main keyword and content type.

### Output
`MarketingBriefResult` containing:
- `target_audience` (List[str]): Target audience personas with demographics
- `key_messages` (List[str]): 3-5 core messaging points
- `content_strategy` (str): Recommended content strategy and approach
- `kpis` (List[str]): Key performance indicators to track
- `distribution_channels` (List[str]): Recommended distribution channels
- `tone_and_voice` (str): Recommended tone and voice
- `competitive_angle` (str): Unique competitive positioning
- `confidence_score` (float): 0-1, AI confidence in brief quality
- `strategy_alignment_score` (float): 0-100, alignment with business objectives

### Approvals Required
✅ **Yes** - If approvals are enabled and `marketing_brief` is in the approval agents list.

#### Approval UI Features:
- **Decision Options:**
  - **Approve:** Use marketing brief as-is
  - **Modify:** Edit the brief JSON and submit modified version
  - **Reject:** Reject the brief - triggers auto-regeneration with user guidance (if `auto_retry` enabled)
    - User can provide feedback/guidance in the rejection comment
    - System automatically regenerates the brief incorporating the feedback
    - If `auto_retry` is disabled, job fails
- **User Guidance on Retry:**
  - When rejecting with feedback, the user's comment is passed to the LLM as guidance
  - LLM regenerates the brief incorporating the user's feedback
  - Supports iterative refinement until approved
- **Display Info:**
  - Confidence score as percentage with progress bar
  - Strategy alignment score
  - Suggestions for reviewer
  - Output shown in tabs (Output/Input/Raw JSON)
  - Markdown rendering for readable display

---

## Step 3: Article Generation/Enhancement

### What It Is
Enhances the original blog post content to be more engaging and valuable, creating an improved version with better structure, compelling title, strong opening hook, and clear takeaways.

### Position in Process
**Step 3** - Runs after marketing brief. Uses keywords from Step 1 and brief from Step 2.

### Input Required
- Original blog post title
- Original blog post content
- **Main keyword** from Step 1 (single focus keyword)
- Supporting keywords from Step 1 (other primary keywords excluding main)
- Target audience from Step 2
- Key messages from Step 2
- Tone and voice from Step 2

### Output
`ArticleGenerationResult` containing:
- `article_title` (str): Generated article title optimized for clicks and engagement
- `article_content` (str): Generated article content created from scratch based on marketing brief
- `outline` (List[str]): Article structure outline with section headers
- `call_to_action` (str): Recommended call-to-action
- `hook` (str): Opening hook to capture attention
- `key_takeaways` (List[str]): Main takeaways from the content
- `word_count` (int): Word count of generated article content
- `confidence_score` (float): 0-1, AI confidence in article quality
- `readability_score` (float): 0-100, Flesch-Kincaid readability score
- `engagement_score` (float): 0-100, Predicted engagement potential

### Approvals Required
✅ **Yes** - If approvals are enabled and `article_generation` is in the approval agents list.

#### Approval UI Features:
- **Decision Options:**
  - **Approve:** Use enhanced article as-is
  - **Modify:** Edit the article content JSON and submit modified version
  - **Reject:** Reject the article (job fails)
- **Display Info:**
  - Confidence score as percentage with progress bar
  - Readability score
  - Engagement score
  - Suggestions for reviewer
  - Output shown in tabs (Output/Input/Raw JSON)
  - Markdown rendering for readable display

---

## Step 4: SEO Optimization

### What It Is
Optimizes the enhanced content for maximum search engine visibility, creating SEO-optimized versions, meta tags, descriptions, URL slugs, schema markup, and Open Graph tags.

### Position in Process
**Step 4** - Runs after article generation. Uses article content from Step 3, keywords from Step 1, and marketing brief from Step 2.

### Input Required
- Article title from Step 3
- Full article content from Step 3 (complete, not truncated)
- **Main keyword** from Step 1 (single focus keyword)
- Primary and secondary keywords from Step 1
- Search intent from Step 1
- Target audience from Step 2
- Tone and voice from Step 2
- Content strategy from Step 2
- Key messages from Step 2
- Article outline and key takeaways from Step 3

### Output
`SEOOptimizationResult` containing:
- `optimized_content` (str): SEO-optimized version of the full article content (complete, not truncated)
- `meta_title` (str): SEO meta title (50-60 characters)
- `meta_description` (str): SEO meta description (150-160 characters)
- `slug` (str): URL-friendly slug
- `alt_texts` (Dict[str, str]): Alt text suggestions for images
- `schema_markup` (str): JSON-LD schema markup as JSON string
- `canonical_url` (str): Recommended canonical URL
- `og_tags` (Dict[str, str]): Open Graph tags for social sharing
- `confidence_score` (float): 0-1, AI confidence in SEO optimization quality
- `seo_score` (float): 0-100, Overall SEO optimization score
- `keyword_optimization_score` (float): 0-100, Keyword integration and density score
- `header_structure` (Dict): H1-H3 hierarchy analysis with validation results
- `keyword_map` (Dict): Primary and related keywords with placement locations in content
- `readability_optimization` (Dict): Readability analysis including score, grade level, and active voice percentage
- `modification_report` (List[str]): Summary of changes made during SEO optimization

### Approvals Required
✅ **Yes** - If approvals are enabled and `seo_optimization` is in the approval agents list.

#### Approval UI Features:
- **Decision Options:**
  - **Approve:** Use SEO optimizations as-is
  - **Modify:** Edit the SEO optimization JSON and submit modified version
  - **Reject:** Reject the SEO optimizations (job fails)
- **Display Info:**
  - Confidence score as percentage with progress bar
  - SEO score
  - Keyword optimization score
  - Suggestions for reviewer
  - Output shown in tabs (Output/Input/Raw JSON)
  - Markdown rendering for readable display

---

## Step 5: Internal Documentation Suggestions

### What It Is
Analyzes the content and suggests internal documentation strategy, including related documents to link to, internal linking opportunities, documentation gaps, topic clusters, and pillar content opportunities.

### Position in Process
**Step 5** - Runs after SEO optimization. Uses optimized content from Step 4, keywords from Step 1, article structure from Step 3, and SEO analysis from Step 4.

### Input Required
- Full optimized content from Step 4 (complete, not truncated)
- Article title from Step 3
- Content category
- **Keywords from Step 1**: main_keyword, primary_keywords, secondary_keywords, search_intent
- **Article outline from Step 3**: for section-level linking opportunities
- **Key takeaways from Step 3**: for pillar content connections
- **Header structure from Step 4**: to understand content hierarchy
- **Keyword map from Step 4**: to optimize anchor text with primary keywords

### Output
`InternalDocsResult` containing:
- `related_docs` (List[Dict[str, str]]): Related internal documents with title and URL
- `internal_links` (List[Dict[str, str]]): Suggested internal links with anchor text and target URL
- `documentation_gaps` (List[str]): Missing documentation topics identified
- `topic_clusters` (List[str]): Related topic clusters for content grouping
- `pillar_content_suggestions` (List[str]): Suggested pillar content pieces
- `confidence_score` (float): 0-1, AI confidence in documentation suggestions
- `relevance_score` (float): 0-100, Relevance of suggested internal links

### Approvals Required
❌ **No approvals** - This step does not require approval by default (not in default approval_agents list).

#### If Approvals Were Enabled:
- **Decision Options:**
  - **Approve:** Use suggestions as-is
  - **Modify:** Edit the suggestions JSON and submit modified version
  - **Reject:** Reject the suggestions (job fails)

---

## Step 6: Content Formatting

### What It Is
Formats the content for publication, creating clean semantic HTML, markdown versions, proper section structure, table of contents, and reading time estimation.

### Position in Process
**Step 6** - Runs after internal docs suggestions. Uses optimized content from Step 4 and preserves all SEO optimizations during formatting.

### Input Required
- Full optimized content from Step 4
- **Header structure from Step 4** (CRITICAL): Preserve exact H1-H3 hierarchy
- **Keyword map from Step 4** (CRITICAL): Preserve keyword placement locations
- **Readability optimization from Step 4** (CRITICAL): Maintain grade level and active voice percentage
- **Meta title and meta description from Step 4**: Include in formatted HTML
- **Schema markup from Step 4**: Preserve in formatted output
- **Article outline from Step 3**: Validate section organization

### Output
`ContentFormattingResult` containing:
- `formatted_html` (str): HTML-formatted content ready for publication
- `formatted_markdown` (str): Markdown-formatted content
- `sections` (List[Dict[str, str]]): Content sections with headings and content
- `reading_time` (int): Estimated reading time in minutes
- `table_of_contents` (List[Dict[str, str]]): Table of contents with anchors
- `formatting_notes` (List[str]): Formatting recommendations or notes
- `confidence_score` (float): 0-1, AI confidence in formatting quality
- `accessibility_score` (float): 0-100, Accessibility compliance score
- `formatting_quality_score` (float): 0-100, Overall formatting quality

### Approvals Required
✅ **Yes** - If approvals are enabled and `content_formatting` is in the approval agents list.

#### Approval UI Features:
- **Decision Options:**
  - **Approve:** Use formatted content as-is
  - **Modify:** Edit the formatting JSON and submit modified version
  - **Reject:** Reject the formatting (job fails)
- **Display Info:**
  - Confidence score as percentage with progress bar
  - Accessibility score
  - Formatting quality score
  - Suggestions for reviewer
  - Output shown in tabs (Output/Input/Raw JSON)
  - HTML rendering for preview

---

## Step 7: Design Kit Application

### What It Is
Recommends design elements for the content, including visual components, color schemes, typography, layout suggestions, hero image concepts, and accessibility considerations.

### Position in Process
**Step 7** - Final pipeline step. Runs after content formatting. Uses comprehensive data from all previous steps for data-driven design recommendations.

### Input Required
- Article title from Step 3
- Content type (blog_post)
- Category
- **Tone and voice from Step 2**
- **Target audience from Step 2**: for design choices and color preferences
- **Keywords from Step 1**: main_keyword, primary_keywords for alt text suggestions
- **Header structure from Step 4**: for visual hierarchy recommendations
- **Readability optimization from Step 4**: for typography decisions
- **Key takeaways from Step 3**: for visual emphasis suggestions
- **Article outline from Step 3**: for layout suggestions
- **Meta title and meta description from Step 4**: for social sharing card design
- **Alt texts from Step 4**: build upon existing suggestions

### Output
`DesignKitResult` containing:
- `visual_components` (List[Dict[str, str]]): Visual elements to add (images, charts, infographics)
- `color_scheme` (Dict[str, str]): Recommended color palette with hex codes
- `typography` (Dict[str, str]): Typography recommendations (fonts, sizes)
- `layout_suggestions` (List[str]): Layout and spacing recommendations
- `hero_image_concept` (str): Hero image concept or description
- `accessibility_notes` (List[str]): Accessibility recommendations
- `confidence_score` (float): 0-1, AI confidence in design recommendations
- `design_quality_score` (float): 0-100, Overall design quality assessment
- `brand_consistency_score` (float): 0-100, Brand consistency score

### Approvals Required
❌ **No approvals** - This step does not require approval by default (not in default approval_agents list).

#### If Approvals Were Enabled:
- **Decision Options:**
  - **Approve:** Use design recommendations as-is
  - **Modify:** Edit the design JSON and submit modified version
  - **Reject:** Reject the design recommendations (job fails)

---

## Final Output

After all steps complete, the pipeline returns a `PipelineResult` containing:

- `pipeline_status`: "completed" or "completed_with_warnings" or "waiting_for_approval" or "failed"
- `step_results`: Dictionary with all step outputs
- `quality_warnings`: List of any quality issues identified
- `final_content`: Final formatted HTML content (from Step 6)
- `metadata`: Execution metadata including:
  - `job_id`: Job identifier
  - `content_id`: Content identifier
  - `content_type`: "blog_post"
  - `title`: Original title
  - `steps_completed`: Number of steps completed
  - `execution_time_seconds`: Total execution time
  - `total_tokens_used`: Total tokens consumed
  - `model`: Model used (gpt-4o-mini)
  - `completed_at`: Completion timestamp
  - `step_info`: Detailed info about each step execution

---

## Approval System Configuration

### Default Approval Settings
- **Require Approval:** `False` (disabled by default)
- **Approval Agents:**
  - `seo_keywords`
  - `marketing_brief`
  - `article_generation`
  - `seo_optimization`
  - `content_formatting`
- **Auto-Approve Threshold:** `None` (no auto-approval)
- **Timeout:** `None` (no timeout)

### Approval Workflow

1. **When Approval Required:**
   - Pipeline stops at that step
   - Job status changes to `WAITING_FOR_APPROVAL`
   - Approval request created in database
   - Pipeline context saved to Redis for resume

2. **Approval Decision:**
   - **Approve:** Pipeline resumes from next step
   - **Modify:** Modified output used, pipeline resumes from next step
   - **Reject:**
     - If `auto_retry=True` (default): Automatically regenerates step with user guidance
       - User's rejection comment is passed to LLM as guidance
       - Step is retried with the feedback incorporated
       - Pipeline continues after successful regeneration
     - If `auto_retry=False`: Job marked as `FAILED`, pipeline stops

3. **Resume After Approval:**
   - Pipeline resumes from the step after approval
   - Uses approved/modified output from previous step
   - Continues with remaining steps

---

## Approval UI Overview

### Common Approval UI Elements

1. **Header Section:**
   - Step name badge (color-coded by step type)
   - Approval status badge (pending/approved/rejected/modified)
   - Back button to approvals list

2. **Confidence Score Display:**
   - Percentage display
   - Visual progress bar
   - Info alert styling

3. **Suggestions Alert:**
   - List of suggestions for reviewer
   - Warning-style alert

4. **Content Tabs:**
   - **Output Tab:** Formatted display of generated output (Markdown rendering)
   - **Input Tab:** JSON view of input data (syntax highlighted)
   - **Raw JSON Tab:** Raw JSON of output data (syntax highlighted)

5. **Decision Actions:**
   - **Approve Button:** Green/primary styling
   - **Modify Button:** Orange/warning styling
   - **Reject Button:** Red/error styling
   - Comment field for reviewer notes
   - Modified output field (for modify decision)

6. **Special Features by Step:**
   - **SEO Keywords:**
     - **Metrics Display:** Comprehensive SEO analysis metrics (search intent, difficulty, relevance, confidence, keyword density)
     - **Main Keyword Selection:** Required radio button selection for single main keyword
     - **Keyword Density Visualization:** Visual bars showing density for each primary keyword
     - **AI Suggestion Indicator:** Badge showing which keyword was AI-suggested
     - **Supporting Keywords:** Checkboxes for selecting additional primary/secondary/LSI keywords
   - **Marketing Brief:**
     - Auto-regeneration on rejection with user guidance
     - User feedback incorporated into LLM prompt for iterative improvement
   - **Article Generation:** Enhanced markdown preview
   - **Content Formatting:** HTML preview

---

## Summary Table

| Step | Name | Approvals | Input Source | Key Output |
|------|------|-----------|--------------|------------|
| 0 | Validation | ❌ | User submission | Validated BlogPostContext |
| 1 | SEO Keywords | ✅ | Original content | Main keyword + primary/secondary/LSI keywords |
| 2 | Marketing Brief | ✅ | Content type + main keyword (Step 1) | Target audience, strategy, KPIs |
| 3 | Article Generation | ✅ | Steps 1-2 output | Enhanced title & content |
| 4 | SEO Optimization | ✅ | Steps 1,3 output | Optimized content, meta tags |
| 5 | Internal Docs | ❌ | Steps 1, 3, 4 output (full content, keywords, structure) | Internal linking suggestions with keyword-optimized anchor text |
| 6 | Content Formatting | ✅ | Steps 3-4 output (preserves SEO optimizations) | HTML/Markdown formatted content with SEO metadata |
| 7 | Design Kit | ❌ | Steps 1-4 output (comprehensive data-driven) | Visual components, colors, typography aligned with audience and structure |

---

## Notes

- Steps run **sequentially** - each step must complete before the next begins
- Results from previous steps are **passed forward** as context
- If a step requires approval, the pipeline **stops** and waits for human review
- After approval, the pipeline **resumes** from the next step
- **Rejection with auto-retry:** When a step is rejected with `auto_retry=True` (default), the step is automatically regenerated with user guidance
- **Main Keyword Focus:** Step 1 now identifies a single `main_keyword` that becomes the primary focus for all subsequent steps
- **User Guidance:** Rejection comments are passed to the LLM as guidance for regeneration, enabling iterative refinement
- All steps use **OpenAI structured outputs** with Pydantic models for type safety
- Default model: **gpt-4o-mini** with temperature **0.7**
