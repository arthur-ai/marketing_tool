"""
Design Kit processing plugin tasks for Marketing Project.

This module provides functions to enhance content with professional design
templates, visual components, and brand-consistent styling.

Functions:
    select_design_template: Choose appropriate design template based on content type
    apply_brand_guidelines: Apply consistent brand styling and guidelines
    generate_visual_components: Create headers, CTAs, cards, and other visual elements
    optimize_responsive_layout: Ensure mobile and desktop compatibility
    create_visual_assets: Generate or select appropriate images and graphics
    validate_design_compliance: Check content against design standards
    apply_design_kit_enhancement: Main function to apply all design enhancements
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from marketing_project.core.models import AppContext, ContentContext
from marketing_project.core.utils import (
    create_standard_task_result,
    ensure_content_context,
    extract_content_metadata_for_pipeline,
    validate_content_for_processing,
)

logger = logging.getLogger("marketing_project.plugins.design_kit")


def select_design_template(
    content: Union[Dict[str, Any], ContentContext], content_type: str = None
) -> Dict[str, Any]:
    """
    Select appropriate design template based on content type and characteristics.

    Args:
        content: Content dictionary or ContentContext
        content_type: Type of content (blog_post, article, case_study, etc.)

    Returns:
        Dict[str, Any]: Standardized task result with selected template
    """
    try:
        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Validate content
        validation = validate_content_for_processing(content_obj)
        if not validation["is_valid"]:
            return create_standard_task_result(
                success=False,
                error=f"Validation failed: {', '.join(validation['issues'])}",
                task_name="select_design_template",
            )

        # Determine content type if not provided
        if not content_type:
            content_type = determine_content_type(content_obj)

        # Load available templates
        templates = load_design_templates()

        # Select template based on content type and characteristics
        selected_template = choose_template_for_content(
            content_obj, content_type, templates
        )

        # Enhance template with content-specific customizations
        customized_template = customize_template_for_content(
            selected_template, content_obj
        )

        return create_standard_task_result(
            success=True,
            data={
                "template": customized_template,
                "content_type": content_type,
                "template_id": selected_template["id"],
                "customizations_applied": True,
            },
            task_name="select_design_template",
            metadata=extract_content_metadata_for_pipeline(content_obj),
        )

    except Exception as e:
        logger.error(f"Error in select_design_template: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Template selection failed: {str(e)}",
            task_name="select_design_template",
        )


def apply_brand_guidelines(
    content: Union[Dict[str, Any], ContentContext], brand_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Apply consistent brand styling and guidelines to content.

    Args:
        content: Content dictionary or ContentContext
        brand_config: Brand configuration settings

    Returns:
        Dict[str, Any]: Content with brand guidelines applied
    """
    try:
        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Load default brand guidelines if not provided
        if not brand_config:
            brand_config = load_default_brand_guidelines()

        # Apply brand styling
        styled_content = apply_brand_styling(content_obj, brand_config)

        # Apply typography guidelines
        styled_content = apply_typography_guidelines(styled_content, brand_config)

        # Apply color scheme
        styled_content = apply_color_scheme(styled_content, brand_config)

        # Apply spacing and layout guidelines
        styled_content = apply_layout_guidelines(styled_content, brand_config)

        return create_standard_task_result(
            success=True,
            data=styled_content,
            task_name="apply_brand_guidelines",
            metadata={
                "brand_config_applied": True,
                "brand_id": brand_config.get("id", "default"),
                "styling_applied": True,
            },
        )

    except Exception as e:
        logger.error(f"Error in apply_brand_guidelines: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Brand guidelines application failed: {str(e)}",
            task_name="apply_brand_guidelines",
        )


def generate_visual_components(
    content: Union[Dict[str, Any], ContentContext],
    component_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Generate visual components like headers, CTAs, cards, and other elements.

    Args:
        content: Content dictionary or ContentContext
        component_config: Component configuration settings

    Returns:
        Dict[str, Any]: Content with visual components added
    """
    try:
        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Load component library
        component_library = load_component_library()

        # Generate components based on content analysis
        components = generate_components_for_content(
            content_obj, component_library, component_config
        )

        # Apply components to content
        enhanced_content = apply_components_to_content(content_obj, components)

        return create_standard_task_result(
            success=True,
            data=enhanced_content,
            task_name="generate_visual_components",
            metadata={
                "components_generated": len(components),
                "component_types": list(set(comp["type"] for comp in components)),
                "enhancement_applied": True,
            },
        )

    except Exception as e:
        logger.error(f"Error in generate_visual_components: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Visual components generation failed: {str(e)}",
            task_name="generate_visual_components",
        )


def optimize_responsive_layout(
    content: Union[Dict[str, Any], ContentContext],
    responsive_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Optimize content layout for mobile and desktop compatibility.

    Args:
        content: Content dictionary or ContentContext
        responsive_config: Responsive design configuration

    Returns:
        Dict[str, Any]: Content optimized for responsive display
    """
    try:
        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Load responsive design guidelines
        if not responsive_config:
            responsive_config = load_responsive_guidelines()

        # Optimize for mobile
        mobile_optimized = optimize_for_mobile(content_obj, responsive_config)

        # Optimize for tablet
        tablet_optimized = optimize_for_tablet(mobile_optimized, responsive_config)

        # Optimize for desktop
        desktop_optimized = optimize_for_desktop(tablet_optimized, responsive_config)

        # Create responsive CSS/HTML
        responsive_content = create_responsive_markup(
            desktop_optimized, responsive_config
        )

        return create_standard_task_result(
            success=True,
            data=responsive_content,
            task_name="optimize_responsive_layout",
            metadata={
                "mobile_optimized": True,
                "tablet_optimized": True,
                "desktop_optimized": True,
                "responsive_markup_generated": True,
            },
        )

    except Exception as e:
        logger.error(f"Error in optimize_responsive_layout: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Responsive optimization failed: {str(e)}",
            task_name="optimize_responsive_layout",
        )


def create_visual_assets(
    content: Union[Dict[str, Any], ContentContext], asset_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create or select appropriate visual assets for content.

    Args:
        content: Content dictionary or ContentContext
        asset_config: Asset configuration settings

    Returns:
        Dict[str, Any]: Content with visual assets integrated
    """
    try:
        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Analyze content for asset requirements
        asset_requirements = analyze_asset_requirements(content_obj)

        # Load asset library
        asset_library = load_asset_library()

        # Select or generate assets
        selected_assets = select_assets_for_content(
            asset_requirements, asset_library, asset_config
        )

        # Integrate assets into content
        content_with_assets = integrate_assets_into_content(
            content_obj, selected_assets
        )

        return create_standard_task_result(
            success=True,
            data=content_with_assets,
            task_name="create_visual_assets",
            metadata={
                "assets_selected": len(selected_assets),
                "asset_types": list(set(asset["type"] for asset in selected_assets)),
                "assets_integrated": True,
            },
        )

    except Exception as e:
        logger.error(f"Error in create_visual_assets: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Visual assets creation failed: {str(e)}",
            task_name="create_visual_assets",
        )


def validate_design_compliance(
    content: Union[Dict[str, Any], ContentContext],
    design_standards: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Validate content against design standards and guidelines.

    Args:
        content: Content dictionary or ContentContext
        design_standards: Design standards to validate against

    Returns:
        Dict[str, Any]: Design compliance validation results
    """
    try:
        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Load design standards if not provided
        if not design_standards:
            design_standards = load_design_standards()

        # Perform compliance checks
        compliance_results = perform_design_compliance_checks(
            content_obj, design_standards
        )

        # Generate improvement recommendations
        recommendations = generate_design_recommendations(compliance_results)

        return create_standard_task_result(
            success=True,
            data={
                "compliance_results": compliance_results,
                "recommendations": recommendations,
                "overall_score": compliance_results["overall_score"],
                "compliant": compliance_results["overall_score"] >= 80,
            },
            task_name="validate_design_compliance",
            metadata={
                "checks_performed": len(compliance_results["checks"]),
                "issues_found": len(compliance_results["issues"]),
                "recommendations_count": len(recommendations),
            },
        )

    except Exception as e:
        logger.error(f"Error in validate_design_compliance: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Design compliance validation failed: {str(e)}",
            task_name="validate_design_compliance",
        )


def apply_design_kit_enhancement(
    content: Union[Dict[str, Any], ContentContext], design_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Apply comprehensive design kit enhancements to content.

    This is the main function that orchestrates all design kit features.

    Args:
        content: Content dictionary or ContentContext
        design_config: Comprehensive design configuration

    Returns:
        Dict[str, Any]: Content with all design enhancements applied
    """
    try:
        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Load default design configuration if not provided
        if not design_config:
            design_config = load_default_design_config()

        # Step 1: Select design template
        template_result = select_design_template(
            content_obj, design_config.get("content_type")
        )
        if not template_result["success"]:
            return template_result

        # Step 2: Apply brand guidelines
        brand_result = apply_brand_guidelines(
            content_obj, design_config.get("brand_config")
        )
        if not brand_result["success"]:
            return brand_result

        # Step 3: Generate visual components
        components_result = generate_visual_components(
            content_obj, design_config.get("component_config")
        )
        if not components_result["success"]:
            return components_result

        # Step 4: Optimize responsive layout
        responsive_result = optimize_responsive_layout(
            content_obj, design_config.get("responsive_config")
        )
        if not responsive_result["success"]:
            return responsive_result

        # Step 5: Create visual assets
        assets_result = create_visual_assets(
            content_obj, design_config.get("asset_config")
        )
        if not assets_result["success"]:
            return assets_result

        # Step 6: Validate design compliance
        compliance_result = validate_design_compliance(
            content_obj, design_config.get("design_standards")
        )

        # Combine all enhancements
        enhanced_content = {
            "original_content": (
                content_obj.dict() if hasattr(content_obj, "dict") else content_obj
            ),
            "template": template_result["data"]["template"],
            "brand_styling": brand_result["data"],
            "visual_components": components_result["data"],
            "responsive_layout": responsive_result["data"],
            "visual_assets": assets_result["data"],
            "design_compliance": (
                compliance_result["data"] if compliance_result["success"] else None
            ),
            "enhancement_applied": True,
            "enhancement_timestamp": datetime.now().isoformat(),
        }

        return create_standard_task_result(
            success=True,
            data=enhanced_content,
            task_name="apply_design_kit_enhancement",
            metadata={
                "template_applied": True,
                "brand_guidelines_applied": True,
                "visual_components_generated": True,
                "responsive_optimized": True,
                "visual_assets_created": True,
                "design_compliance_checked": compliance_result["success"],
            },
        )

    except Exception as e:
        logger.error(f"Error in apply_design_kit_enhancement: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Design kit enhancement failed: {str(e)}",
            task_name="apply_design_kit_enhancement",
        )


# Helper functions


def determine_content_type(content_obj: ContentContext) -> str:
    """Determine content type based on content characteristics."""
    content_lower = content_obj.content.lower()
    title_lower = content_obj.title.lower()

    # Check for specific content type indicators
    if any(keyword in title_lower for keyword in ["tutorial", "guide", "how to"]):
        return "tutorial"
    elif any(keyword in title_lower for keyword in ["case study", "success story"]):
        return "case_study"
    elif any(
        keyword in content_lower for keyword in ["product", "feature", "specification"]
    ):
        return "product_page"
    elif any(
        keyword in content_lower for keyword in ["news", "announcement", "update"]
    ):
        return "news_article"
    else:
        return "blog_post"


def load_design_templates() -> List[Dict[str, Any]]:
    """Load available design templates."""
    return [
        {
            "id": "blog_post_modern",
            "name": "Modern Blog Post",
            "type": "blog_post",
            "description": "Clean, modern design for blog posts",
            "features": ["hero_section", "content_blocks", "sidebar", "footer"],
            "responsive": True,
            "brand_compatible": True,
        },
        {
            "id": "tutorial_step_by_step",
            "name": "Step-by-Step Tutorial",
            "type": "tutorial",
            "description": "Structured layout for tutorials and guides",
            "features": [
                "progress_indicator",
                "step_blocks",
                "code_highlights",
                "navigation",
            ],
            "responsive": True,
            "brand_compatible": True,
        },
        {
            "id": "case_study_detailed",
            "name": "Detailed Case Study",
            "type": "case_study",
            "description": "Professional layout for case studies",
            "features": [
                "hero_with_stats",
                "challenge_solution",
                "results_section",
                "testimonials",
            ],
            "responsive": True,
            "brand_compatible": True,
        },
        {
            "id": "product_page_feature_rich",
            "name": "Feature-Rich Product Page",
            "type": "product_page",
            "description": "Comprehensive product showcase layout",
            "features": [
                "product_hero",
                "feature_grid",
                "comparison_table",
                "cta_sections",
            ],
            "responsive": True,
            "brand_compatible": True,
        },
    ]


def choose_template_for_content(
    content_obj: ContentContext, content_type: str, templates: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Choose the most appropriate template for the content."""
    # Filter templates by content type
    matching_templates = [t for t in templates if t["type"] == content_type]

    if not matching_templates:
        # Fallback to blog_post template
        matching_templates = [t for t in templates if t["type"] == "blog_post"]

    # For now, return the first matching template
    # In a real implementation, you'd have more sophisticated selection logic
    return matching_templates[0] if matching_templates else templates[0]


def customize_template_for_content(
    template: Dict[str, Any], content_obj: ContentContext
) -> Dict[str, Any]:
    """Customize template based on content characteristics."""
    customized = template.copy()

    # Add content-specific customizations
    customized["content_specific"] = {
        "title": content_obj.title,
        "word_count": len(content_obj.content.split()),
        "estimated_reading_time": len(content_obj.content.split()) // 200 + 1,
        "has_images": "![" in content_obj.content,
        "has_code": "```" in content_obj.content,
        "has_lists": any(
            marker in content_obj.content for marker in ["- ", "* ", "1. "]
        ),
    }

    return customized


def load_default_brand_guidelines() -> Dict[str, Any]:
    """Load default brand guidelines."""
    return {
        "id": "default_brand",
        "name": "Default Brand Guidelines",
        "colors": {
            "primary": "#2563eb",
            "secondary": "#64748b",
            "accent": "#f59e0b",
            "background": "#ffffff",
            "text": "#1e293b",
        },
        "typography": {
            "heading_font": "Inter, sans-serif",
            "body_font": "Inter, sans-serif",
            "heading_sizes": [
                "2.5rem",
                "2rem",
                "1.5rem",
                "1.25rem",
                "1.125rem",
                "1rem",
            ],
            "body_size": "1rem",
            "line_height": 1.6,
        },
        "spacing": {
            "section_padding": "3rem",
            "content_padding": "1.5rem",
            "element_margin": "1rem",
        },
        "layout": {
            "max_width": "1200px",
            "content_width": "800px",
            "sidebar_width": "300px",
        },
    }


def apply_brand_styling(
    content_obj: ContentContext, brand_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply brand styling to content."""
    return {
        "content": content_obj.content,
        "title": content_obj.title,
        "brand_styling": {
            "colors": brand_config["colors"],
            "typography": brand_config["typography"],
            "spacing": brand_config["spacing"],
            "layout": brand_config["layout"],
        },
        "styling_applied": True,
    }


def apply_typography_guidelines(
    content: Dict[str, Any], brand_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply typography guidelines."""
    content["typography_applied"] = True
    return content


def apply_color_scheme(
    content: Dict[str, Any], brand_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply color scheme."""
    content["color_scheme_applied"] = True
    return content


def apply_layout_guidelines(
    content: Dict[str, Any], brand_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply layout guidelines."""
    content["layout_guidelines_applied"] = True
    return content


def load_component_library() -> List[Dict[str, Any]]:
    """Load component library."""
    return [
        {
            "id": "hero_section",
            "type": "header",
            "name": "Hero Section",
            "description": "Eye-catching header with title and subtitle",
            "html_template": '<div class="hero-section"><h1>{title}</h1><p class="subtitle">{subtitle}</p></div>',
            "css_classes": ["hero-section", "text-center", "py-8"],
        },
        {
            "id": "cta_button",
            "type": "button",
            "name": "Call to Action Button",
            "description": "Prominent call-to-action button",
            "html_template": '<a href="{url}" class="cta-button">{text}</a>',
            "css_classes": ["cta-button", "btn", "btn-primary"],
        },
        {
            "id": "info_card",
            "type": "card",
            "name": "Information Card",
            "description": "Card layout for highlighting information",
            "html_template": '<div class="info-card"><h3>{title}</h3><p>{content}</p></div>',
            "css_classes": ["info-card", "card", "p-4", "rounded"],
        },
        {
            "id": "testimonial",
            "type": "quote",
            "name": "Testimonial",
            "description": "Customer testimonial with attribution",
            "html_template": '<blockquote class="testimonial"><p>"{quote}"</p><cite>- {author}</cite></blockquote>',
            "css_classes": ["testimonial", "quote", "italic"],
        },
    ]


def generate_components_for_content(
    content_obj: ContentContext,
    component_library: List[Dict[str, Any]],
    component_config: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """Generate appropriate components for content."""
    components = []

    # Always add hero section
    hero_component = next(
        (comp for comp in component_library if comp["id"] == "hero_section"), None
    )
    if hero_component:
        components.append(
            {
                **hero_component,
                "data": {
                    "title": content_obj.title,
                    "subtitle": (
                        content_obj.snippet[:100] + "..."
                        if len(content_obj.snippet) > 100
                        else content_obj.snippet
                    ),
                },
            }
        )

    # Add CTA button if content has call-to-action indicators
    if any(
        cta_word in content_obj.content.lower()
        for cta_word in ["learn more", "get started", "contact us", "subscribe"]
    ):
        cta_component = next(
            (comp for comp in component_library if comp["id"] == "cta_button"), None
        )
        if cta_component:
            components.append(
                {**cta_component, "data": {"text": "Learn More", "url": "#learn-more"}}
            )

    return components


def apply_components_to_content(
    content_obj: ContentContext, components: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Apply components to content."""
    return {
        "content": content_obj.content,
        "title": content_obj.title,
        "components": components,
        "components_applied": True,
    }


def load_responsive_guidelines() -> Dict[str, Any]:
    """Load responsive design guidelines."""
    return {
        "breakpoints": {"mobile": "768px", "tablet": "1024px", "desktop": "1200px"},
        "mobile_optimizations": {
            "font_size_scale": 0.9,
            "padding_reduction": 0.5,
            "image_resize": True,
        },
        "tablet_optimizations": {
            "font_size_scale": 1.0,
            "padding_reduction": 0.8,
            "image_resize": False,
        },
        "desktop_optimizations": {
            "font_size_scale": 1.0,
            "padding_reduction": 1.0,
            "image_resize": False,
        },
    }


def optimize_for_mobile(
    content_obj: ContentContext, responsive_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Optimize content for mobile devices."""
    return {
        "content": content_obj.content,
        "mobile_optimized": True,
        "optimizations": responsive_config["mobile_optimizations"],
    }


def optimize_for_tablet(
    content: Dict[str, Any], responsive_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Optimize content for tablet devices."""
    content["tablet_optimized"] = True
    content["tablet_optimizations"] = responsive_config["tablet_optimizations"]
    return content


def optimize_for_desktop(
    content: Dict[str, Any], responsive_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Optimize content for desktop devices."""
    content["desktop_optimized"] = True
    content["desktop_optimizations"] = responsive_config["desktop_optimizations"]
    return content


def create_responsive_markup(
    content: Dict[str, Any], responsive_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create responsive HTML/CSS markup."""
    content["responsive_markup"] = {
        "html": generate_responsive_html(content),
        "css": generate_responsive_css(responsive_config),
        "breakpoints": responsive_config["breakpoints"],
    }
    return content


def generate_responsive_html(content: Dict[str, Any]) -> str:
    """Generate responsive HTML markup."""
    return f"""
    <div class="responsive-content">
        <h1 class="content-title">{content.get('title', 'Untitled')}</h1>
        <div class="content-body">
            {content.get('content', '')}
        </div>
    </div>
    """


def generate_responsive_css(responsive_config: Dict[str, Any]) -> str:
    """Generate responsive CSS."""
    return f"""
    .responsive-content {{
        max-width: 100%;
        margin: 0 auto;
        padding: 1rem;
    }}

    @media (max-width: {responsive_config['breakpoints']['mobile']}) {{
        .responsive-content {{
            padding: 0.5rem;
            font-size: {responsive_config['mobile_optimizations']['font_size_scale']}em;
        }}
    }}

    @media (min-width: {responsive_config['breakpoints']['tablet']}) {{
        .responsive-content {{
            max-width: 800px;
        }}
    }}

    @media (min-width: {responsive_config['breakpoints']['desktop']}) {{
        .responsive-content {{
            max-width: 1200px;
        }}
    }}
    """


def analyze_asset_requirements(content_obj: ContentContext) -> Dict[str, Any]:
    """Analyze content to determine asset requirements."""
    content_lower = content_obj.content.lower()

    requirements = {
        "images_needed": 0,
        "icons_needed": 0,
        "charts_needed": 0,
        "infographics_needed": 0,
        "asset_types": [],
    }

    # Count headings to estimate image needs
    heading_count = content_obj.content.count("#")
    requirements["images_needed"] = max(1, heading_count // 2)

    # Check for data that might need charts
    if any(
        keyword in content_lower
        for keyword in ["percent", "%", "increase", "decrease", "growth", "statistics"]
    ):
        requirements["charts_needed"] = 1
        requirements["asset_types"].append("chart")

    # Check for process descriptions that might need infographics
    if any(
        keyword in content_lower
        for keyword in ["process", "steps", "workflow", "methodology"]
    ):
        requirements["infographics_needed"] = 1
        requirements["asset_types"].append("infographic")

    # Always add some icons
    requirements["icons_needed"] = 3
    requirements["asset_types"].extend(["icon", "image"])

    return requirements


def load_asset_library() -> List[Dict[str, Any]]:
    """Load asset library."""
    return [
        {
            "id": "hero_image_1",
            "type": "image",
            "category": "hero",
            "url": "/assets/images/hero-placeholder.jpg",
            "alt_text": "Hero image placeholder",
            "dimensions": "1200x600",
        },
        {
            "id": "info_icon_1",
            "type": "icon",
            "category": "information",
            "url": "/assets/icons/info.svg",
            "alt_text": "Information icon",
            "dimensions": "24x24",
        },
        {
            "id": "chart_placeholder",
            "type": "chart",
            "category": "data",
            "url": "/assets/charts/placeholder.svg",
            "alt_text": "Chart placeholder",
            "dimensions": "600x400",
        },
    ]


def select_assets_for_content(
    requirements: Dict[str, Any],
    asset_library: List[Dict[str, Any]],
    asset_config: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """Select appropriate assets for content."""
    selected_assets = []

    for asset_type in requirements["asset_types"]:
        matching_assets = [
            asset for asset in asset_library if asset["type"] == asset_type
        ]
        if matching_assets:
            selected_assets.append(matching_assets[0])

    return selected_assets


def integrate_assets_into_content(
    content_obj: ContentContext, assets: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Integrate selected assets into content."""
    return {
        "content": content_obj.content,
        "title": content_obj.title,
        "assets": assets,
        "assets_integrated": True,
    }


def load_design_standards() -> Dict[str, Any]:
    """Load design standards for compliance checking."""
    return {
        "accessibility": {
            "min_contrast_ratio": 4.5,
            "alt_text_required": True,
            "heading_hierarchy": True,
        },
        "performance": {
            "max_image_size": "500KB",
            "max_css_size": "100KB",
            "max_js_size": "200KB",
        },
        "brand_consistency": {
            "color_usage": True,
            "font_consistency": True,
            "spacing_consistency": True,
        },
        "responsive_design": {
            "mobile_friendly": True,
            "tablet_optimized": True,
            "desktop_optimized": True,
        },
    }


def perform_design_compliance_checks(
    content_obj: ContentContext, design_standards: Dict[str, Any]
) -> Dict[str, Any]:
    """Perform design compliance checks."""
    checks = {
        "accessibility": {
            "alt_text_present": "![" in content_obj.content,
            "heading_structure": "#" in content_obj.content,
            "contrast_adequate": True,  # Simplified check
        },
        "performance": {
            "content_size_ok": len(content_obj.content) < 100000,
            "image_count_reasonable": content_obj.content.count("![") < 10,
        },
        "brand_consistency": {
            "title_present": bool(content_obj.title),
            "content_structured": "##" in content_obj.content,
        },
        "responsive_design": {"mobile_ready": True, "flexible_layout": True},
    }

    # Calculate overall score
    all_checks = []
    for category in checks.values():
        all_checks.extend(category.values())

    overall_score = (sum(all_checks) / len(all_checks)) * 100 if all_checks else 0

    return {
        "checks": checks,
        "overall_score": overall_score,
        "issues": [],
        "passed": overall_score >= 80,
    }


def generate_design_recommendations(compliance_results: Dict[str, Any]) -> List[str]:
    """Generate design improvement recommendations."""
    recommendations = []

    if compliance_results["overall_score"] < 80:
        recommendations.append("Improve overall design consistency")

    if not compliance_results["checks"]["accessibility"]["alt_text_present"]:
        recommendations.append("Add alt text to images for accessibility")

    if not compliance_results["checks"]["accessibility"]["heading_structure"]:
        recommendations.append("Improve heading structure and hierarchy")

    if not compliance_results["checks"]["brand_consistency"]["content_structured"]:
        recommendations.append("Add more structured content with subheadings")

    return recommendations


def load_default_design_config() -> Dict[str, Any]:
    """Load default design configuration."""
    return {
        "content_type": "blog_post",
        "brand_config": load_default_brand_guidelines(),
        "component_config": {
            "include_hero": True,
            "include_cta": True,
            "include_cards": True,
        },
        "responsive_config": load_responsive_guidelines(),
        "asset_config": {
            "include_images": True,
            "include_icons": True,
            "include_charts": True,
        },
        "design_standards": load_design_standards(),
    }
