import shutil
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_full_pipeline(monkeypatch, tmp_path):
    # Use real prompt templates from the prompts directory
    lang = "en"
    prompts_source_dir = (
        Path(__file__).parent.parent.parent
        / "src"
        / "marketing_project"
        / "prompts"
        / "v1"
        / lang
    )
    prompts_test_dir = tmp_path / lang
    prompts_test_dir.mkdir(parents=True)

    # Copy all real prompt templates to test directory
    if prompts_source_dir.exists():
        for template_file in prompts_source_dir.glob("*.j2"):
            shutil.copy2(template_file, prompts_test_dir / template_file.name)
    else:
        # Fallback: create minimal templates if source doesn't exist
        required_agents = [
            "transcripts_agent",
            "blog_agent",
            "releasenotes_agent",
            "marketing_orchestrator_agent",
        ]
        for name in required_agents:
            (prompts_test_dir / f"{name}_instructions.j2").write_text("Prompt")
            (prompts_test_dir / f"{name}_description.j2").write_text("Description")

    from marketing_project.runner import run_marketing_project_pipeline

    await run_marketing_project_pipeline(str(tmp_path), lang)
