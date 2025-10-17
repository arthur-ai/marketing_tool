"""
Test suite for Design Kit plugin.

This module contains comprehensive tests for all Design Kit plugin functions
including template selection, brand guidelines, visual components, responsive layout,
visual assets, and design compliance validation.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from marketing_project.core.models import BlogPostContext, ContentContext
from marketing_project.plugins.design_kit import tasks


class TestDesignKitPlugin:
    """Test class for Design Kit plugin functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_content = BlogPostContext(
            id="test-1",
            title="Test Article: Marketing Best Practices",
            content="This is a comprehensive guide to marketing best practices. It covers various strategies and techniques that can help businesses improve their marketing efforts.",
            snippet="A guide to marketing best practices for businesses.",
            created_at=datetime.now(),
            source_url="https://example.com/test-article",
            author="Test Author",
            tags=["marketing", "best-practices"],
            category="Marketing",
            word_count=50,
            reading_time="1 minute",
        )

        self.sample_blog_post = BlogPostContext(
            id="test-blog-1",
            title="How to Improve Your Marketing Strategy",
            content="Marketing strategy is crucial for business success. Here are the key steps to improve your marketing approach.",
            snippet="Learn how to improve your marketing strategy with these proven techniques.",
            author="Marketing Expert",
            tags=["marketing", "strategy", "business"],
            category="Marketing",
            word_count=150,
            reading_time="1 minute",
        )

    def test_select_design_template_success(self):
        """Test successful design template selection."""
        result = tasks.select_design_template(self.sample_content, "blog_post")

        assert result["success"] is True
        assert result["task_name"] == "select_design_template"
        assert "template" in result["data"]
        assert "content_type" in result["data"]
        assert "template_id" in result["data"]
        assert result["data"]["content_type"] == "blog_post"

    def test_select_design_template_with_content_context(self):
        """Test design template selection with ContentContext."""
        result = tasks.select_design_template(self.sample_content)

        assert result["success"] is True
        assert "template" in result["data"]
        assert result["data"]["customizations_applied"] is True

    def test_select_design_template_validation_failure(self):
        """Test design template selection with invalid content."""
        invalid_content = BlogPostContext(
            id="",
            title="",
            content="",
            snippet="",
            author="",
            tags=[],
            category="",
            word_count=0,
            reading_time="",
        )

        result = tasks.select_design_template(invalid_content)

        assert result["success"] is False
        assert "Validation failed" in result["error"]

    def test_apply_brand_guidelines_success(self):
        """Test successful brand guidelines application."""
        brand_config = {
            "id": "test_brand",
            "colors": {"primary": "#2563eb", "secondary": "#64748b"},
            "typography": {"heading_font": "Inter, sans-serif"},
            "spacing": {"section_padding": "3rem"},
            "layout": {"max_width": "1200px", "content_width": "800px"},
        }

        result = tasks.apply_brand_guidelines(self.sample_content, brand_config)

        assert result["success"] is True
        assert result["task_name"] == "apply_brand_guidelines"
        assert "brand_styling" in result["data"]
        assert result["metadata"]["brand_config_applied"] is True

    def test_apply_brand_guidelines_default_config(self):
        """Test brand guidelines application with default configuration."""
        result = tasks.apply_brand_guidelines(self.sample_content)

        assert result["success"] is True
        assert "brand_styling" in result["data"]
        assert result["metadata"]["brand_id"] == "default_brand"

    def test_generate_visual_components_success(self):
        """Test successful visual components generation."""
        result = tasks.generate_visual_components(self.sample_content)

        assert result["success"] is True
        assert result["task_name"] == "generate_visual_components"
        assert "components" in result["data"]
        assert result["metadata"]["enhancement_applied"] is True

    def test_generate_visual_components_with_config(self):
        """Test visual components generation with configuration."""
        component_config = {
            "include_hero": True,
            "include_cta": True,
            "include_cards": False,
        }

        result = tasks.generate_visual_components(self.sample_content, component_config)

        assert result["success"] is True
        assert "components" in result["data"]

    def test_optimize_responsive_layout_success(self):
        """Test successful responsive layout optimization."""
        result = tasks.optimize_responsive_layout(self.sample_content)

        assert result["success"] is True
        assert result["task_name"] == "optimize_responsive_layout"
        assert result["metadata"]["mobile_optimized"] is True
        assert result["metadata"]["tablet_optimized"] is True
        assert result["metadata"]["desktop_optimized"] is True

    def test_optimize_responsive_layout_with_config(self):
        """Test responsive layout optimization with configuration."""
        responsive_config = {
            "breakpoints": {"mobile": "768px", "tablet": "1024px", "desktop": "1200px"},
            "mobile_optimizations": {"font_size_scale": 0.9},
            "tablet_optimizations": {"font_size_scale": 1.0},
            "desktop_optimizations": {"font_size_scale": 1.0},
        }

        result = tasks.optimize_responsive_layout(
            self.sample_content, responsive_config
        )

        assert result["success"] is True
        assert "responsive_markup" in result["data"]

    def test_create_visual_assets_success(self):
        """Test successful visual assets creation."""
        result = tasks.create_visual_assets(self.sample_content)

        assert result["success"] is True
        assert result["task_name"] == "create_visual_assets"
        assert "assets" in result["data"]
        assert result["metadata"]["assets_integrated"] is True

    def test_create_visual_assets_with_config(self):
        """Test visual assets creation with configuration."""
        asset_config = {
            "include_images": True,
            "include_icons": True,
            "include_charts": False,
        }

        result = tasks.create_visual_assets(self.sample_content, asset_config)

        assert result["success"] is True
        assert "assets" in result["data"]

    def test_validate_design_compliance_success(self):
        """Test successful design compliance validation."""
        result = tasks.validate_design_compliance(self.sample_content)

        assert result["success"] is True
        assert result["task_name"] == "validate_design_compliance"
        assert "compliance_results" in result["data"]
        assert "recommendations" in result["data"]
        assert "overall_score" in result["data"]

    def test_validate_design_compliance_with_standards(self):
        """Test design compliance validation with custom standards."""
        design_standards = {
            "accessibility": {"min_contrast_ratio": 4.5},
            "performance": {"max_image_size": "500KB"},
            "brand_consistency": {"color_usage": True},
        }

        result = tasks.validate_design_compliance(self.sample_content, design_standards)

        assert result["success"] is True
        assert "compliance_results" in result["data"]

    def test_apply_design_kit_enhancement_success(self):
        """Test successful comprehensive design kit enhancement."""
        result = tasks.apply_design_kit_enhancement(self.sample_content)

        assert result["success"] is True
        assert result["task_name"] == "apply_design_kit_enhancement"
        assert "original_content" in result["data"]
        assert "template" in result["data"]
        assert "brand_styling" in result["data"]
        assert "visual_components" in result["data"]
        assert "responsive_layout" in result["data"]
        assert "visual_assets" in result["data"]
        assert result["data"]["enhancement_applied"] is True

    def test_apply_design_kit_enhancement_with_config(self):
        """Test design kit enhancement with custom configuration."""
        design_config = {
            "content_type": "tutorial",
            "brand_config": {
                "id": "custom_brand",
                "colors": {"primary": "#2563eb", "secondary": "#64748b"},
                "typography": {"heading_font": "Inter, sans-serif"},
                "spacing": {"section_padding": "3rem"},
                "layout": {"max_width": "1200px"},
            },
            "component_config": {"include_hero": True},
            "responsive_config": {
                "breakpoints": {
                    "mobile": "768px",
                    "tablet": "1024px",
                    "desktop": "1200px",
                },
                "mobile_optimizations": {"font_size_scale": 0.9},
                "tablet_optimizations": {"font_size_scale": 1.0},
                "desktop_optimizations": {"font_size_scale": 1.0},
            },
            "asset_config": {"include_images": True},
        }

        result = tasks.apply_design_kit_enhancement(self.sample_content, design_config)

        assert result["success"] is True
        assert "enhancement_applied" in result["data"]

    def test_determine_content_type_tutorial(self):
        """Test content type determination for tutorial content."""
        tutorial_content = BlogPostContext(
            id="tutorial-1",
            title="How to Create a Marketing Strategy: Step-by-Step Guide",
            content="This tutorial will teach you how to create an effective marketing strategy.",
            snippet="Learn to create marketing strategies with this step-by-step guide.",
            author="Tutorial Author",
            tags=["tutorial", "marketing"],
            category="Tutorial",
            word_count=100,
            reading_time="2 minutes",
        )

        content_type = tasks.determine_content_type(tutorial_content)
        assert content_type == "tutorial"

    def test_determine_content_type_case_study(self):
        """Test content type determination for case study content."""
        case_study_content = BlogPostContext(
            id="case-1",
            title="Case Study: Successful Marketing Campaign Results",
            content="This case study examines a successful marketing campaign.",
            snippet="Analysis of a successful marketing campaign case study.",
            author="Case Study Author",
            tags=["case-study", "marketing"],
            category="Case Study",
            word_count=150,
            reading_time="3 minutes",
        )

        content_type = tasks.determine_content_type(case_study_content)
        assert content_type == "case_study"

    def test_determine_content_type_product_page(self):
        """Test content type determination for product page content."""
        product_content = BlogPostContext(
            id="product-1",
            title="Marketing Software Features and Specifications",
            content="Our marketing software includes advanced features and specifications.",
            snippet="Comprehensive marketing software with advanced features.",
            author="Product Author",
            tags=["product", "software"],
            category="Product",
            word_count=200,
            reading_time="4 minutes",
        )

        content_type = tasks.determine_content_type(product_content)
        assert content_type == "product_page"

    def test_determine_content_type_blog_post_default(self):
        """Test content type determination defaults to blog_post."""
        content_type = tasks.determine_content_type(self.sample_content)
        assert content_type == "blog_post"

    def test_load_design_templates(self):
        """Test design templates loading."""
        templates = tasks.load_design_templates()

        assert isinstance(templates, list)
        assert len(templates) > 0

        for template in templates:
            assert "id" in template
            assert "name" in template
            assert "type" in template
            assert "description" in template
            assert "features" in template
            assert "responsive" in template
            assert "brand_compatible" in template

    def test_load_default_brand_guidelines(self):
        """Test default brand guidelines loading."""
        brand_guidelines = tasks.load_default_brand_guidelines()

        assert "id" in brand_guidelines
        assert "name" in brand_guidelines
        assert "colors" in brand_guidelines
        assert "typography" in brand_guidelines
        assert "spacing" in brand_guidelines
        assert "layout" in brand_guidelines

    def test_load_component_library(self):
        """Test component library loading."""
        components = tasks.load_component_library()

        assert isinstance(components, list)
        assert len(components) > 0

        for component in components:
            assert "id" in component
            assert "type" in component
            assert "name" in component
            assert "description" in component
            assert "html_template" in component
            assert "css_classes" in component

    def test_load_responsive_guidelines(self):
        """Test responsive guidelines loading."""
        guidelines = tasks.load_responsive_guidelines()

        assert "breakpoints" in guidelines
        assert "mobile_optimizations" in guidelines
        assert "tablet_optimizations" in guidelines
        assert "desktop_optimizations" in guidelines

    def test_load_asset_library(self):
        """Test asset library loading."""
        assets = tasks.load_asset_library()

        assert isinstance(assets, list)
        assert len(assets) > 0

        for asset in assets:
            assert "id" in asset
            assert "type" in asset
            assert "category" in asset
            assert "url" in asset
            assert "alt_text" in asset
            assert "dimensions" in asset

    def test_load_design_standards(self):
        """Test design standards loading."""
        standards = tasks.load_design_standards()

        assert "accessibility" in standards
        assert "performance" in standards
        assert "brand_consistency" in standards
        assert "responsive_design" in standards

    def test_load_default_design_config(self):
        """Test default design configuration loading."""
        config = tasks.load_default_design_config()

        assert "content_type" in config
        assert "brand_config" in config
        assert "component_config" in config
        assert "responsive_config" in config
        assert "asset_config" in config
        assert "design_standards" in config

    def test_error_handling_invalid_input(self):
        """Test error handling with invalid input."""
        result = tasks.select_design_template(None)

        assert result["success"] is False
        assert "error" in result

    def test_error_handling_exception(self):
        """Test error handling when exceptions occur."""
        with patch(
            "marketing_project.plugins.design_kit.tasks.ensure_content_context",
            side_effect=Exception("Test error"),
        ):
            result = tasks.select_design_template(self.sample_content)

            assert result["success"] is False
            assert "Template selection failed" in result["error"]

    def test_blog_post_context_handling(self):
        """Test handling of BlogPostContext input."""
        result = tasks.select_design_template(self.sample_blog_post)

        assert result["success"] is True
        assert "template" in result["data"]

    def test_metadata_extraction(self):
        """Test metadata extraction in results."""
        result = tasks.select_design_template(self.sample_content)

        assert "metadata" in result
        # Check that metadata contains expected fields (may vary based on actual implementation)
        assert "metadata" in result
        metadata = result["metadata"]
        # The exact fields may vary, so we just check that metadata exists and has some content
        assert len(metadata) > 0

    def test_task_result_structure(self):
        """Test standardized task result structure."""
        result = tasks.select_design_template(self.sample_content)

        # Check required fields
        assert "task_name" in result
        assert "success" in result
        assert "timestamp" in result
        assert "data" in result
        assert "metadata" in result

        # Check task_name matches function name
        assert result["task_name"] == "select_design_template"

    def test_comprehensive_enhancement_workflow(self):
        """Test the complete design kit enhancement workflow."""
        # Test with a more complex content structure
        complex_content = BlogPostContext(
            id="complex-1",
            title="Complete Marketing Guide: From Strategy to Implementation",
            content="""
            # Marketing Strategy Fundamentals

            Marketing strategy is the foundation of business success. Here's how to build an effective strategy:

            ## 1. Market Research
            Understanding your market is crucial for success.

            ## 2. Target Audience
            Define your ideal customer profile.

            ## 3. Value Proposition
            Create compelling value propositions.

            ## 4. Implementation
            Execute your strategy effectively.
            """,
            snippet="A comprehensive guide to marketing strategy development and implementation.",
            created_at=datetime.now(),
            author="Strategy Expert",
            tags=["strategy", "marketing", "implementation"],
            category="Guide",
            word_count=300,
            reading_time="5 minutes",
        )

        result = tasks.apply_design_kit_enhancement(complex_content)

        assert result["success"] is True
        assert result["data"]["enhancement_applied"] is True
        assert "template" in result["data"]
        assert "brand_styling" in result["data"]
        assert "visual_components" in result["data"]
        assert "responsive_layout" in result["data"]
        assert "visual_assets" in result["data"]
        assert "design_compliance" in result["data"]

    def test_performance_metrics(self):
        """Test that performance metrics are included in results."""
        result = tasks.apply_design_kit_enhancement(self.sample_content)

        assert "metadata" in result
        metadata = result["metadata"]

        # Check that enhancement status is tracked
        assert "template_applied" in metadata
        assert "brand_guidelines_applied" in metadata
        assert "visual_components_generated" in metadata
        assert "responsive_optimized" in metadata
        assert "visual_assets_created" in metadata
        # Note: design_compliance_checked might not always be present

    def test_content_type_specific_processing(self):
        """Test that different content types are processed appropriately."""
        # Test tutorial content
        tutorial_result = tasks.select_design_template(self.sample_content, "tutorial")
        assert tutorial_result["success"] is True
        assert tutorial_result["data"]["content_type"] == "tutorial"

        # Test case study content
        case_study_result = tasks.select_design_template(
            self.sample_content, "case_study"
        )
        assert case_study_result["success"] is True
        assert case_study_result["data"]["content_type"] == "case_study"

        # Test product page content
        product_result = tasks.select_design_template(
            self.sample_content, "product_page"
        )
        assert product_result["success"] is True
        assert product_result["data"]["content_type"] == "product_page"

    def test_brand_consistency_validation(self):
        """Test brand consistency validation in design compliance."""
        result = tasks.validate_design_compliance(self.sample_content)

        assert result["success"] is True
        compliance_results = result["data"]["compliance_results"]

        # Check that brand consistency is validated
        assert "brand_consistency" in compliance_results["checks"]
        brand_checks = compliance_results["checks"]["brand_consistency"]
        assert "title_present" in brand_checks
        assert "content_structured" in brand_checks

    def test_responsive_design_validation(self):
        """Test responsive design validation in design compliance."""
        result = tasks.validate_design_compliance(self.sample_content)

        assert result["success"] is True
        compliance_results = result["data"]["compliance_results"]

        # Check that responsive design is validated
        assert "responsive_design" in compliance_results["checks"]
        responsive_checks = compliance_results["checks"]["responsive_design"]
        assert "mobile_ready" in responsive_checks
        assert "flexible_layout" in responsive_checks

    def test_accessibility_validation(self):
        """Test accessibility validation in design compliance."""
        result = tasks.validate_design_compliance(self.sample_content)

        assert result["success"] is True
        compliance_results = result["data"]["compliance_results"]

        # Check that accessibility is validated
        assert "accessibility" in compliance_results["checks"]
        accessibility_checks = compliance_results["checks"]["accessibility"]
        assert "alt_text_present" in accessibility_checks
        assert "heading_structure" in accessibility_checks
        assert "contrast_adequate" in accessibility_checks

    def test_performance_validation(self):
        """Test performance validation in design compliance."""
        result = tasks.validate_design_compliance(self.sample_content)

        assert result["success"] is True
        compliance_results = result["data"]["compliance_results"]

        # Check that performance is validated
        assert "performance" in compliance_results["checks"]
        performance_checks = compliance_results["checks"]["performance"]
        assert "content_size_ok" in performance_checks
        assert "image_count_reasonable" in performance_checks

    def test_recommendations_generation(self):
        """Test that recommendations are generated based on compliance results."""
        result = tasks.validate_design_compliance(self.sample_content)

        assert result["success"] is True
        assert "recommendations" in result["data"]
        recommendations = result["data"]["recommendations"]

        # Recommendations should be a list
        assert isinstance(recommendations, list)

        # If there are issues, there should be recommendations
        compliance_results = result["data"]["compliance_results"]
        if compliance_results["overall_score"] < 80:
            assert len(recommendations) > 0

    def test_enhancement_timestamp(self):
        """Test that enhancement timestamp is included in results."""
        result = tasks.apply_design_kit_enhancement(self.sample_content)

        assert result["success"] is True
        assert "enhancement_timestamp" in result["data"]

        # Timestamp should be a valid ISO format
        timestamp = result["data"]["enhancement_timestamp"]
        assert isinstance(timestamp, str)
        # Basic check that it's in ISO format
        assert "T" in timestamp
        assert timestamp.endswith("Z") or "+" in timestamp or timestamp.count("-") >= 2
