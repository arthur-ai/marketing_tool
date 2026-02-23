"""
Version information for the marketing project.

This module attempts to read the version from Git tags, falling back to
a default version if Git is not available or tags cannot be read.
"""

import os
import subprocess
import sys
from pathlib import Path


def get_version_from_git():
    """
    Attempt to get the version from the latest Git tag.

    Returns:
        str: Version string (e.g., "0.1.0") or None if not available
    """
    try:
        # Get the root directory of the project (where .git is located)
        current_dir = Path(__file__).parent
        repo_root = None

        # Walk up the directory tree to find .git
        for parent in [current_dir] + list(current_dir.parents):
            if (parent / ".git").exists():
                repo_root = parent
                break

        if repo_root is None:
            return None

        # Fetch tags to ensure we have the latest
        try:
            subprocess.run(
                ["git", "fetch", "--tags", "--force"],
                cwd=repo_root,
                capture_output=True,
                check=False,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Get the latest tag matching v*.*.* pattern
        result = subprocess.run(
            ["git", "tag", "-l", "v*.*.*", "--sort=-version:refname"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and result.stdout.strip():
            latest_tag = result.stdout.strip().split("\n")[0]
            # Remove 'v' prefix if present
            version = latest_tag.lstrip("v")
            return version

        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return None


def get_version():
    """
    Get the current version of the project.

    Priority order:
    1. PROJECT_VERSION environment variable (for Docker/build-time setting)
    2. Git tags (if available)
    3. Fallback to default version

    Returns:
        str: Version string (e.g., "1.0.85")
    """
    # First, check for environment variable (useful for Docker builds)
    env_version = os.getenv("PROJECT_VERSION")
    if env_version:
        return env_version.strip()

    # Try to get version from Git
    git_version = get_version_from_git()
    if git_version:
        return git_version

    # Fallback to default version
    # Update this to match the latest release version
    return "2.0.0"


# Get the version when the module is imported
__version__ = get_version()
