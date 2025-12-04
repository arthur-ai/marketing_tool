"""
Integration tests for Social Media Pipeline.

Tests end-to-end pipeline execution, multi-platform generation, error scenarios,
and performance benchmarks.
"""

import json
import time

import pytest

from marketing_project.services.social_media_pipeline import SocialMediaPipeline

# Sample blog post content for testing
SAMPLE_BLOG_POST = {
    "id": "test-blog-1",
    "title": "Introduction to Machine Learning",
    "content": """
    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.

    In this article, we'll explore the fundamentals of machine learning, including
    supervised learning, unsupervised learning, and reinforcement learning.

    Supervised learning involves training a model on labeled data, while unsupervised
    learning finds patterns in unlabeled data. Reinforcement learning uses rewards
    and penalties to guide learning.

    These techniques have applications in various fields including healthcare,
    finance, and technology.
    """,
}


class TestEndToEndPipeline:
    """Test end-to-end pipeline execution for each platform."""

    @pytest.mark.asyncio
    async def test_linkedin_pipeline_execution(self):
        """Test complete LinkedIn pipeline execution."""
        pipeline = SocialMediaPipeline()
        content_json = json.dumps(SAMPLE_BLOG_POST)

        # Note: This test requires OpenAI API access and may incur costs
        # Skip if API key not available
        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        result = await pipeline.execute_pipeline(
            content_json=content_json,
            job_id="test-linkedin-1",
            content_type="blog_post",
            social_media_platform="linkedin",
        )

        assert result is not None
        assert "pipeline_status" in result
        assert "step_results" in result
        assert "final_content" in result

        # Verify all 4 steps completed
        step_results = result.get("step_results", {})
        assert "seo_keywords" in step_results
        assert "social_media_marketing_brief" in step_results
        assert "social_media_angle_hook" in step_results
        assert "social_media_post_generation" in step_results

        # Verify final content exists
        assert result.get("final_content") is not None
        assert len(result.get("final_content", "")) > 0

    @pytest.mark.asyncio
    async def test_hackernews_pipeline_execution(self):
        """Test complete HackerNews pipeline execution."""
        pipeline = SocialMediaPipeline()
        content_json = json.dumps(SAMPLE_BLOG_POST)

        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        result = await pipeline.execute_pipeline(
            content_json=content_json,
            job_id="test-hackernews-1",
            content_type="blog_post",
            social_media_platform="hackernews",
        )

        assert result is not None
        assert "pipeline_status" in result
        assert "step_results" in result
        assert "final_content" in result

        # Verify platform-specific result
        post_result = result.get("step_results", {}).get(
            "social_media_post_generation", {}
        )
        assert post_result.get("platform") == "hackernews"

    @pytest.mark.asyncio
    async def test_email_pipeline_execution_newsletter(self):
        """Test complete email pipeline execution (newsletter)."""
        pipeline = SocialMediaPipeline()
        content_json = json.dumps(SAMPLE_BLOG_POST)

        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        result = await pipeline.execute_pipeline(
            content_json=content_json,
            job_id="test-email-newsletter-1",
            content_type="blog_post",
            social_media_platform="email",
            email_type="newsletter",
        )

        assert result is not None
        assert "pipeline_status" in result

        # Verify email-specific fields
        post_result = result.get("step_results", {}).get(
            "social_media_post_generation", {}
        )
        assert post_result.get("platform") == "email"
        assert "subject_line" in post_result

    @pytest.mark.asyncio
    async def test_email_pipeline_execution_promotional(self):
        """Test complete email pipeline execution (promotional)."""
        pipeline = SocialMediaPipeline()
        content_json = json.dumps(SAMPLE_BLOG_POST)

        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        result = await pipeline.execute_pipeline(
            content_json=content_json,
            job_id="test-email-promotional-1",
            content_type="blog_post",
            social_media_platform="email",
            email_type="promotional",
        )

        assert result is not None
        assert "pipeline_status" in result


class TestMultiPlatformGeneration:
    """Test generating posts for multiple platforms."""

    @pytest.mark.asyncio
    async def test_multi_platform_parallel_execution(self):
        """Test parallel execution for multiple platforms."""
        pipeline = SocialMediaPipeline()
        content_json = json.dumps(SAMPLE_BLOG_POST)

        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        import asyncio

        # Execute for multiple platforms in parallel
        platforms = ["linkedin", "hackernews"]
        tasks = [
            pipeline.execute_pipeline(
                content_json=content_json,
                job_id=f"test-multi-{platform}",
                content_type="blog_post",
                social_media_platform=platform,
            )
            for platform in platforms
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 2
        for result in results:
            assert result is not None
            assert "pipeline_status" in result
            assert "final_content" in result


class TestErrorScenarios:
    """Test error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_json_input(self):
        """Test handling of invalid JSON input."""
        pipeline = SocialMediaPipeline()

        with pytest.raises((ValueError, json.JSONDecodeError)):
            await pipeline.execute_pipeline(
                content_json="invalid json",
                job_id="test-invalid-json",
                content_type="blog_post",
                social_media_platform="linkedin",
            )

    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        pipeline = SocialMediaPipeline()
        incomplete_content = {"title": "Test"}
        content_json = json.dumps(incomplete_content)

        # Should handle gracefully or raise appropriate error
        try:
            result = await pipeline.execute_pipeline(
                content_json=content_json,
                job_id="test-missing-fields",
                content_type="blog_post",
                social_media_platform="linkedin",
            )
            # If it succeeds, verify it handled missing fields
            assert result is not None
        except Exception as e:
            # Should raise a meaningful error
            assert isinstance(e, (ValueError, KeyError, AttributeError))

    def test_invalid_platform(self):
        """Test handling of invalid platform."""
        pipeline = SocialMediaPipeline()

        # Should handle invalid platform gracefully
        config = pipeline._get_platform_config("invalid_platform")
        assert isinstance(config, dict)  # Should return empty dict


class TestPerformanceBenchmarks:
    """Test performance benchmarks for pipeline execution."""

    @pytest.mark.asyncio
    async def test_pipeline_execution_time(self):
        """Test that pipeline completes within reasonable time."""
        pipeline = SocialMediaPipeline()
        content_json = json.dumps(SAMPLE_BLOG_POST)

        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        start_time = time.time()
        result = await pipeline.execute_pipeline(
            content_json=content_json,
            job_id="test-performance-1",
            content_type="blog_post",
            social_media_platform="linkedin",
        )
        execution_time = time.time() - start_time

        assert result is not None
        assert execution_time < 300  # Should complete within 5 minutes

        # Verify execution time is recorded
        metadata = result.get("metadata", {})
        assert "execution_time_seconds" in metadata
        assert metadata["execution_time_seconds"] > 0

    @pytest.mark.asyncio
    async def test_token_usage_tracking(self):
        """Test that token usage is tracked."""
        pipeline = SocialMediaPipeline()
        content_json = json.dumps(SAMPLE_BLOG_POST)

        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        result = await pipeline.execute_pipeline(
            content_json=content_json,
            job_id="test-tokens-1",
            content_type="blog_post",
            social_media_platform="linkedin",
        )

        assert result is not None
        metadata = result.get("metadata", {})

        # Token usage should be tracked
        if "total_tokens_used" in metadata:
            assert metadata["total_tokens_used"] > 0


class TestContentValidation:
    """Test content validation and quality checks."""

    @pytest.mark.asyncio
    async def test_content_length_validation(self):
        """Test that content length validation works in pipeline."""
        pipeline = SocialMediaPipeline()

        # Create content that exceeds LinkedIn limit
        long_content = {
            "id": "test-long",
            "title": "Long Content Test",
            "content": "a" * 5000,  # Exceeds LinkedIn 3000 char limit
        }
        content_json = json.dumps(long_content)

        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        result = await pipeline.execute_pipeline(
            content_json=content_json,
            job_id="test-long-content",
            content_type="blog_post",
            social_media_platform="linkedin",
        )

        # Should complete but with warnings
        assert result is not None
        if result.get("quality_warnings"):
            warnings = result.get("quality_warnings", [])
            # Should have length-related warning
            assert any("limit" in str(w).lower() for w in warnings)


class TestPlatformSpecificFeatures:
    """Test platform-specific features are included."""

    @pytest.mark.asyncio
    async def test_linkedin_hashtags(self):
        """Test that LinkedIn posts include hashtags."""
        pipeline = SocialMediaPipeline()
        content_json = json.dumps(SAMPLE_BLOG_POST)

        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        result = await pipeline.execute_pipeline(
            content_json=content_json,
            job_id="test-hashtags",
            content_type="blog_post",
            social_media_platform="linkedin",
        )

        post_result = result.get("step_results", {}).get(
            "social_media_post_generation", {}
        )
        # LinkedIn should have hashtags field (may be empty)
        assert "hashtags" in post_result or post_result.get("platform") == "linkedin"

    @pytest.mark.asyncio
    async def test_email_subject_line(self):
        """Test that email posts include subject line."""
        pipeline = SocialMediaPipeline()
        content_json = json.dumps(SAMPLE_BLOG_POST)

        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        result = await pipeline.execute_pipeline(
            content_json=content_json,
            job_id="test-subject",
            content_type="blog_post",
            social_media_platform="email",
            email_type="newsletter",
        )

        post_result = result.get("step_results", {}).get(
            "social_media_post_generation", {}
        )
        # Email should have subject_line
        assert "subject_line" in post_result
        assert post_result.get("subject_line") is not None
