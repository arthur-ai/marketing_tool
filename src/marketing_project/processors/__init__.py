"""
Deterministic content processors for Marketing Project.

This package contains deterministic processors that follow strict, predictable workflows
for processing different content types. Unlike agents, these processors do not use LLM-based
routing or decision-making at the top level, but may call agents for specific sub-tasks.

Processors:
- blog_processor: Processes blog posts and articles
- releasenotes_processor: Processes release notes and changelogs
- transcript_processor: Processes transcripts from videos, podcasts, meetings
"""

from marketing_project.processors.blog_processor import process_blog_post
from marketing_project.processors.releasenotes_processor import process_release_notes
from marketing_project.processors.transcript_processor import process_transcript

__all__ = [
    "process_blog_post",
    "process_release_notes",
    "process_transcript",
]
