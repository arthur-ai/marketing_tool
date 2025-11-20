"""
Transcript Preprocessing Approval plugin for Marketing Project.

This plugin validates transcript preprocessing data (speakers, duration, content, etc.)
and requires approval before proceeding to SEO keywords extraction.
"""

from marketing_project.plugins.transcript_preprocessing_approval.tasks import (
    TranscriptPreprocessingApprovalPlugin,
)

__all__ = ["TranscriptPreprocessingApprovalPlugin"]
