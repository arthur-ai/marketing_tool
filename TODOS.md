# TODOS

## Audit _llm_exclude_fields models for Anthropic schema complexity

**What:** Evaluate the 6 remaining models that use `_llm_exclude_fields`
(lines 74, 273, 306, 338, 751, 971 in `pipeline_steps.py` — SEOKeywordsResult and others)
and apply the `BlogPostPreprocessingApprovalLLMResult` inheritance pattern to any at risk
of hitting Anthropic's schema complexity limit.

**Why:** The `_llm_exclude_fields` bandaid is structurally unsound — the same "Schema is
too complex" error will recur on the next field addition to any of these models.
`BlogPostPreprocessingApprovalResult` hit the limit after two prior patch attempts; the
other models are one or two fields away from the same failure.

**How to apply:** Follow the pattern established in this PR — create a `*LLMResult` base
class with only the fields the LLM actually needs to fill in, have the full result inherit
from it, update the plugin's `response_model` and `execute()` upgrade logic.

**Depends on:** This PR (adding-full-tests) — establishes the pattern and confirms it works.

**Context:** The inheritance refactor reduced the schema sent to Anthropic from 21 fields
(10 Optional) to 13 fields (3 Optional) — a 5x reduction in grammar complexity. Other
models using `_llm_exclude_fields` are currently under the limit but accumulate fields over time.
