#!/usr/bin/env python3
"""
Setup script for Marketing Project.
"""

import os
from pathlib import Path

from setuptools import find_packages, setup


# Read the README file (optional for Docker builds)
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Marketing Project - A production-ready marketing agentic project"


# Read requirements (optional for Docker builds)
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [
                line.strip() for line in fh if line.strip() and not line.startswith("#")
            ]
    except FileNotFoundError:
        return []


# Read dev requirements (optional for Docker builds)
def read_dev_requirements():
    try:
        with open("requirements-dev.txt", "r", encoding="utf-8") as fh:
            return [
                line.strip()
                for line in fh
                if line.strip()
                and not line.startswith("#")
                and not line.startswith("-r")
            ]
    except FileNotFoundError:
        return []


# Read version from _version.py
def read_version():
    """
    Read version from _version.py module.
    This allows the version to be dynamically determined from Git tags.
    """
    # Import the version module directly
    # We need to add src to the path first
    import sys

    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        from marketing_project._version import __version__

        return __version__
    except (ImportError, Exception):
        # Fallback to default version if import fails
        return "0.1.0"


setup(
    name="marketing-project",
    version=read_version(),
    author="Ibrahim Abouhashish",
    author_email="your-email@example.com",
    description="A marketing agentic project with extensible agents, plugins, and multi-locale support",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/marketing-project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements(),
    extras_require={
        "dev": read_dev_requirements(),
    },
    entry_points={
        "console_scripts": [
            "marketing-project=marketing_project.main:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
