"""
Pydantic models for Marketing Project application context.

This module defines data models for representing various content types and application context using Pydantic BaseModel classes.

Classes:
    BaseContentContext: Base model for all content types with common fields.
    TranscriptContext: Represents the structure of a transcript (podcast, video, meeting).
    BlogPostContext: Represents the structure of a blog post or article.
    ReleaseNotesContext: Represents the structure of software release notes.
    ContentContext: Union type that can hold any of the content types.
    AppContext: Represents the application context, including the content, labels, and extracted information.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class BaseContentContext(BaseModel):
    """
    Base model for all content types with common fields.

    Attributes:
        id (str): Unique identifier for the content.
        title (str): Title of the content.
        content (str): Full content text.
        snippet (str): Short snippet or preview of the content.
        created_at (Optional[datetime]): When the content was created.
        source_url (Optional[str]): URL where the content was found.
        metadata (Dict[str, str]): Additional metadata about the content.
    """

    id: str
    title: str
    content: str
    snippet: str
    created_at: Optional[datetime] = None
    source_url: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class TranscriptContext(BaseContentContext):
    """
    Model representing the structure of a transcript.

    Attributes:
        speakers (List[str]): List of speakers in the transcript.
        duration (Optional[str]): Duration of the audio/video content.
        transcript_type (str): Type of transcript (podcast, video, meeting, etc.).
        timestamps (Optional[Dict[str, str]]): Timestamp markers in the transcript.
    """

    speakers: List[str] = Field(default_factory=list)
    duration: Optional[str] = None
    transcript_type: str = "podcast"  # podcast, video, meeting, interview, etc.
    timestamps: Optional[Dict[str, str]] = None


class BlogPostContext(BaseContentContext):
    """
    Model representing the structure of a blog post or article.

    Attributes:
        author (Optional[str]): Author of the blog post.
        tags (List[str]): Tags associated with the blog post.
        category (Optional[str]): Category of the blog post.
        word_count (Optional[int]): Approximate word count of the content.
        reading_time (Optional[str]): Estimated reading time.
    """

    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    word_count: Optional[int] = None
    reading_time: Optional[str] = None


class ReleaseNotesContext(BaseContentContext):
    """
    Model representing the structure of software release notes.

    Attributes:
        version (str): Version number of the release.
        release_date (Optional[datetime]): When the release was published.
        changes (List[str]): List of changes in this release.
        breaking_changes (List[str]): List of breaking changes.
        features (List[str]): List of new features.
        bug_fixes (List[str]): List of bug fixes.
    """

    version: str
    release_date: Optional[datetime] = None
    changes: List[str] = Field(default_factory=list)
    breaking_changes: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    bug_fixes: List[str] = Field(default_factory=list)


# Union type for all content types
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
