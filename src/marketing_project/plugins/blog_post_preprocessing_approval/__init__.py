"""
Blog Post Preprocessing Approval plugin for Marketing Project.

This plugin validates blog post preprocessing data (title, content, author, category, tags, etc.)
and requires approval before proceeding to SEO keywords extraction.
"""

from marketing_project.plugins.blog_post_preprocessing_approval.tasks import (
    BlogPostPreprocessingApprovalPlugin,
)

__all__ = ["BlogPostPreprocessingApprovalPlugin"]
