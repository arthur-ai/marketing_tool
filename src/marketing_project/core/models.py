"""
Pydantic models for Marketing Project application context.

This module defines application context models that use the API content models.
The content models themselves (BlogPostContext, TranscriptContext, ReleaseNotesContext)
are now defined in marketing_project.models.content_models.

Classes:
    ContentContext: Union type that can hold any of the content types (from API models).
    AppContext: Represents the application context, including the content, labels, and extracted information.
    EmailContext: Legacy email context model (for backward compatibility).
"""

from typing import Dict, Union

from pydantic import BaseModel, Field

# Import API models for the Union type
from marketing_project.models.content_models import (
    BlogPostContext,
    ReleaseNotesContext,
    TranscriptContext,
)

# Union type for all content types (using API models)
ContentContext = Union[TranscriptContext, BlogPostContext, ReleaseNotesContext]


class AppContext(BaseModel):
    """
    Model representing the application context for Marketing Project.

    Attributes:
        content (ContentContext): The content context object (transcript, blog post, or release notes).
        labels (Dict[str, str]): Dictionary of labels associated with the content.
        content_type (str): Type of content being processed (transcript, blog_post, release_notes).
    """

    content: ContentContext
    labels: Dict[str, str] = Field(default_factory=dict)
    content_type: str = "transcript"  # transcript, blog_post, release_notes


# Legacy support - keeping EmailContext for backward compatibility
class EmailContext(BaseModel):
    """
    Model representing the structure of an email (legacy support).

    Attributes:
        id (str): Unique identifier for the email.
        thread_id (str): Identifier for the email thread.
        subject (str): Subject line of the email.
        snippet (str): Short snippet or preview of the email.
        body (str): Full body content of the email.
    """

    id: str
    thread_id: str
    subject: str
    snippet: str
    body: str
