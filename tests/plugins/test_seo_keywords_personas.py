"""
Unit tests for profound_personas_used field population in SEOKeywordsPlugin.

Tests the two distinct code paths in execute():
  1. _inject_profound_personas() — fetches personas and populates context
  2. Step 10.5 logic — reads context["_profound_personas"] and sets result field
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.plugins.seo_keywords.tasks import SEOKeywordsPlugin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plugin() -> SEOKeywordsPlugin:
    return SEOKeywordsPlugin()


def _persona_dict(name: str) -> dict:
    """Minimal dict as returned by ProfoundPersona.to_prompt_dict()."""
    return {
        "name": name,
        "pain_points": None,
        "motivations": None,
        "job_titles": [],
        "industries": [],
        "role_seniority": [],
    }


def _persona_dict_no_name() -> dict:
    """to_prompt_dict() output when the persona has an empty name."""
    return {
        "name": "",
        "pain_points": "Slow reporting",
        "motivations": "Save time",
        "job_titles": ["Analyst"],
        "industries": [],
        "role_seniority": [],
    }


# ---------------------------------------------------------------------------
# _inject_profound_personas unit tests
# ---------------------------------------------------------------------------


class TestInjectProfoundPersonas:
    """Unit tests for _inject_profound_personas()."""

    @pytest.mark.asyncio
    async def test_not_configured_does_not_set_context_key(self):
        """When Profound is not configured, context key must not be set."""
        plugin = _make_plugin()
        context: dict = {}
        content = {"title": "Test"}

        mock_client = MagicMock()
        mock_client.is_configured.return_value = False

        with patch(
            "marketing_project.plugins.seo_keywords.tasks.get_profound_client",
            new_callable=AsyncMock,
            return_value=(mock_client, None),
        ):
            await plugin._inject_profound_personas(context, content)

        assert "_profound_personas" not in context

    @pytest.mark.asyncio
    async def test_configured_but_no_category_id_does_not_set_key(self):
        """When configured but no category ID available, context key is not set."""
        plugin = _make_plugin()
        context: dict = {}
        content = {}  # no profound_category_id

        mock_client = MagicMock()
        mock_client.is_configured.return_value = True

        with patch(
            "marketing_project.plugins.seo_keywords.tasks.get_profound_client",
            new_callable=AsyncMock,
            return_value=(mock_client, None),  # no db_default_category_id either
        ):
            await plugin._inject_profound_personas(context, content)

        assert "_profound_personas" not in context

    @pytest.mark.asyncio
    async def test_configured_with_content_category_id_fetches_and_stores(self):
        """When content overrides category_id, personas are fetched and stored."""
        plugin = _make_plugin()
        context: dict = {}
        content = {"profound_category_id": "cat-abc"}

        mock_persona = MagicMock()
        mock_persona.to_prompt_dict.return_value = _persona_dict("Marketing Manager")

        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_client.get_category_personas = AsyncMock(return_value=[mock_persona])

        with patch(
            "marketing_project.plugins.seo_keywords.tasks.get_profound_client",
            new_callable=AsyncMock,
            return_value=(mock_client, None),
        ):
            await plugin._inject_profound_personas(context, content)

        assert "_profound_personas" in context
        assert context["_profound_personas"] == [_persona_dict("Marketing Manager")]
        mock_client.get_category_personas.assert_awaited_once_with("cat-abc")

    @pytest.mark.asyncio
    async def test_configured_with_db_default_category_id(self):
        """Uses db_default_category_id when content has no profound_category_id."""
        plugin = _make_plugin()
        context: dict = {}
        content = {}

        mock_persona = MagicMock()
        mock_persona.to_prompt_dict.return_value = _persona_dict("Engineer")

        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_client.get_category_personas = AsyncMock(return_value=[mock_persona])

        with patch(
            "marketing_project.plugins.seo_keywords.tasks.get_profound_client",
            new_callable=AsyncMock,
            return_value=(mock_client, "db-cat-99"),
        ):
            await plugin._inject_profound_personas(context, content)

        assert "_profound_personas" in context
        mock_client.get_category_personas.assert_awaited_once_with("db-cat-99")

    @pytest.mark.asyncio
    async def test_api_returns_empty_list_does_not_set_key(self):
        """When Profound returns no personas, context key is not set."""
        plugin = _make_plugin()
        context: dict = {}
        content = {"profound_category_id": "cat-empty"}

        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_client.get_category_personas = AsyncMock(return_value=[])

        with patch(
            "marketing_project.plugins.seo_keywords.tasks.get_profound_client",
            new_callable=AsyncMock,
            return_value=(mock_client, None),
        ):
            await plugin._inject_profound_personas(context, content)

        assert "_profound_personas" not in context

    @pytest.mark.asyncio
    async def test_exception_is_swallowed(self):
        """Exceptions from get_profound_client are swallowed silently."""
        plugin = _make_plugin()
        context: dict = {}
        content = {"profound_category_id": "cat-fail"}

        with patch(
            "marketing_project.plugins.seo_keywords.tasks.get_profound_client",
            side_effect=ConnectionError("Profound API unreachable"),
        ):
            # Must not raise
            await plugin._inject_profound_personas(context, content)

        assert "_profound_personas" not in context

    @pytest.mark.asyncio
    async def test_get_category_personas_exception_is_swallowed(self):
        """Exceptions from get_category_personas are swallowed silently."""
        plugin = _make_plugin()
        context: dict = {}
        content = {"profound_category_id": "cat-fail"}

        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_client.get_category_personas = AsyncMock(
            side_effect=RuntimeError("API error")
        )

        with patch(
            "marketing_project.plugins.seo_keywords.tasks.get_profound_client",
            new_callable=AsyncMock,
            return_value=(mock_client, None),
        ):
            await plugin._inject_profound_personas(context, content)

        assert "_profound_personas" not in context


# ---------------------------------------------------------------------------
# Step 10.5 logic: profound_personas_used field population
# ---------------------------------------------------------------------------


class TestProfoundPersonasUsedField:
    """
    Tests for step 10.5 in execute(): reading _profound_personas from context
    and writing profound_personas_used on the result.

    We patch _inject_profound_personas to directly control what lands in context,
    isolating step 10.5 from the Profound API.
    """

    def _make_minimal_seo_result(self):
        from marketing_project.models.pipeline_steps import SEOKeywordsResult

        return SEOKeywordsResult(
            main_keyword="cloud storage",
            primary_keywords=["cloud", "storage", "backup"],
            search_intent="informational",
        )

    def _patch_inject(self, context_mutation):
        """
        Returns a coroutine function that mutates context as specified.
        context_mutation: callable(context) called when _inject_profound_personas runs.
        """

        async def _fake_inject(self_plugin, context, content):
            context_mutation(context)

        return _fake_inject

    @pytest.mark.asyncio
    async def test_profound_not_configured_field_is_none(self):
        """When Profound is not configured, profound_personas_used is None."""
        plugin = _make_plugin()

        minimal_result = self._make_minimal_seo_result()

        with (
            patch.object(
                plugin,
                "_inject_profound_personas",
                new=AsyncMock(),  # does nothing — key never set
            ),
            patch.object(
                plugin,
                "_execute_keyword_steps",
                new=AsyncMock(return_value=minimal_result),
            ),
            patch.object(
                plugin,
                "_validate_and_fix",
                side_effect=lambda r, c: r,
            ),
            patch.object(
                plugin,
                "_calculate_derived_metrics",
                side_effect=lambda r, c: r,
            ),
            patch.object(
                plugin,
                "_validate_result",
                return_value=(True, []),
            ),
            patch.object(
                plugin,
                "_build_prompt_context",
                return_value={},
            ),
        ):
            result = await plugin.execute(
                context={"input_content": {"title": "Test"}},
                pipeline=MagicMock(),
                job_id=None,
            )

        assert result is not None
        assert result.profound_personas_used is None

    @pytest.mark.asyncio
    async def test_profound_configured_with_named_personas(self):
        """When Profound returns named personas, field contains their names."""
        plugin = _make_plugin()

        minimal_result = self._make_minimal_seo_result()

        async def inject_two_personas(context, content):
            context["_profound_personas"] = [
                _persona_dict("Marketing Manager"),
                _persona_dict("DevOps Engineer"),
            ]

        with (
            patch.object(
                plugin,
                "_inject_profound_personas",
                new=inject_two_personas,
            ),
            patch.object(
                plugin,
                "_execute_keyword_steps",
                new=AsyncMock(return_value=minimal_result),
            ),
            patch.object(
                plugin,
                "_validate_and_fix",
                side_effect=lambda r, c: r,
            ),
            patch.object(
                plugin,
                "_calculate_derived_metrics",
                side_effect=lambda r, c: r,
            ),
            patch.object(
                plugin,
                "_validate_result",
                return_value=(True, []),
            ),
            patch.object(
                plugin,
                "_build_prompt_context",
                return_value={},
            ),
        ):
            result = await plugin.execute(
                context={"input_content": {"title": "Test"}},
                pipeline=MagicMock(),
                job_id=None,
            )

        assert result.profound_personas_used == ["Marketing Manager", "DevOps Engineer"]

    @pytest.mark.asyncio
    async def test_all_personas_have_empty_names_field_is_none(self):
        """When all persona dicts have empty names, field collapses to None."""
        plugin = _make_plugin()

        minimal_result = self._make_minimal_seo_result()

        async def inject_nameless(context, content):
            context["_profound_personas"] = [
                _persona_dict_no_name(),
                _persona_dict_no_name(),
            ]

        with (
            patch.object(
                plugin,
                "_inject_profound_personas",
                new=inject_nameless,
            ),
            patch.object(
                plugin,
                "_execute_keyword_steps",
                new=AsyncMock(return_value=minimal_result),
            ),
            patch.object(
                plugin,
                "_validate_and_fix",
                side_effect=lambda r, c: r,
            ),
            patch.object(
                plugin,
                "_calculate_derived_metrics",
                side_effect=lambda r, c: r,
            ),
            patch.object(
                plugin,
                "_validate_result",
                return_value=(True, []),
            ),
            patch.object(
                plugin,
                "_build_prompt_context",
                return_value={},
            ),
        ):
            result = await plugin.execute(
                context={"input_content": {"title": "Test"}},
                pipeline=MagicMock(),
                job_id=None,
            )

        assert result.profound_personas_used is None

    @pytest.mark.asyncio
    async def test_empty_personas_list_in_context_field_is_none(self):
        """When _profound_personas is set but empty [], field is None."""
        plugin = _make_plugin()

        minimal_result = self._make_minimal_seo_result()

        async def inject_empty(context, content):
            context["_profound_personas"] = []

        with (
            patch.object(
                plugin,
                "_inject_profound_personas",
                new=inject_empty,
            ),
            patch.object(
                plugin,
                "_execute_keyword_steps",
                new=AsyncMock(return_value=minimal_result),
            ),
            patch.object(
                plugin,
                "_validate_and_fix",
                side_effect=lambda r, c: r,
            ),
            patch.object(
                plugin,
                "_calculate_derived_metrics",
                side_effect=lambda r, c: r,
            ),
            patch.object(
                plugin,
                "_validate_result",
                return_value=(True, []),
            ),
            patch.object(
                plugin,
                "_build_prompt_context",
                return_value={},
            ),
        ):
            result = await plugin.execute(
                context={"input_content": {"title": "Test"}},
                pipeline=MagicMock(),
                job_id=None,
            )

        assert result.profound_personas_used is None

    @pytest.mark.asyncio
    async def test_mixed_named_and_unnamed_personas(self):
        """Only personas with non-empty names appear in the field."""
        plugin = _make_plugin()

        minimal_result = self._make_minimal_seo_result()

        async def inject_mixed(context, content):
            context["_profound_personas"] = [
                _persona_dict("Product Manager"),
                _persona_dict_no_name(),  # empty name — filtered out
                _persona_dict("Designer"),
            ]

        with (
            patch.object(
                plugin,
                "_inject_profound_personas",
                new=inject_mixed,
            ),
            patch.object(
                plugin,
                "_execute_keyword_steps",
                new=AsyncMock(return_value=minimal_result),
            ),
            patch.object(
                plugin,
                "_validate_and_fix",
                side_effect=lambda r, c: r,
            ),
            patch.object(
                plugin,
                "_calculate_derived_metrics",
                side_effect=lambda r, c: r,
            ),
            patch.object(
                plugin,
                "_validate_result",
                return_value=(True, []),
            ),
            patch.object(
                plugin,
                "_build_prompt_context",
                return_value={},
            ),
        ):
            result = await plugin.execute(
                context={"input_content": {"title": "Test"}},
                pipeline=MagicMock(),
                job_id=None,
            )

        assert result.profound_personas_used == ["Product Manager", "Designer"]


# ---------------------------------------------------------------------------
# _inject_profound_personas + step 10.5 integration path
# ---------------------------------------------------------------------------


class TestInjectAndFieldIntegration:
    """
    End-to-end path: _inject_profound_personas populates context,
    step 10.5 reads it — but using real _inject_profound_personas
    backed by mocked Profound API.
    """

    @pytest.mark.asyncio
    async def test_full_path_not_configured(self):
        """Real _inject_profound_personas with unconfigured Profound → field is None."""
        plugin = _make_plugin()

        from marketing_project.models.pipeline_steps import SEOKeywordsResult

        minimal_result = SEOKeywordsResult(
            main_keyword="ai tools",
            primary_keywords=["ai", "tools", "automation"],
            search_intent="commercial",
        )

        mock_client = MagicMock()
        mock_client.is_configured.return_value = False

        with (
            patch(
                "marketing_project.plugins.seo_keywords.tasks.get_profound_client",
                new_callable=AsyncMock,
                return_value=(mock_client, None),
            ),
            patch.object(
                plugin,
                "_execute_keyword_steps",
                new=AsyncMock(return_value=minimal_result),
            ),
            patch.object(
                plugin,
                "_validate_and_fix",
                side_effect=lambda r, c: r,
            ),
            patch.object(
                plugin,
                "_calculate_derived_metrics",
                side_effect=lambda r, c: r,
            ),
            patch.object(
                plugin,
                "_validate_result",
                return_value=(True, []),
            ),
            patch.object(
                plugin,
                "_build_prompt_context",
                return_value={},
            ),
        ):
            result = await plugin.execute(
                context={"input_content": {}},
                pipeline=MagicMock(),
                job_id=None,
            )

        assert result.profound_personas_used is None

    @pytest.mark.asyncio
    async def test_full_path_configured_with_personas(self):
        """Real _inject_profound_personas with configured Profound → names in field."""
        plugin = _make_plugin()

        from marketing_project.models.pipeline_steps import SEOKeywordsResult

        minimal_result = SEOKeywordsResult(
            main_keyword="data pipelines",
            primary_keywords=["data", "pipelines", "etl"],
            search_intent="informational",
        )

        mock_persona_a = MagicMock()
        mock_persona_a.to_prompt_dict.return_value = _persona_dict("Data Engineer")
        mock_persona_b = MagicMock()
        mock_persona_b.to_prompt_dict.return_value = _persona_dict("Analytics Lead")

        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_client.get_category_personas = AsyncMock(
            return_value=[mock_persona_a, mock_persona_b]
        )

        with (
            patch(
                "marketing_project.plugins.seo_keywords.tasks.get_profound_client",
                new_callable=AsyncMock,
                return_value=(mock_client, "cat-data"),
            ),
            patch.object(
                plugin,
                "_execute_keyword_steps",
                new=AsyncMock(return_value=minimal_result),
            ),
            patch.object(
                plugin,
                "_validate_and_fix",
                side_effect=lambda r, c: r,
            ),
            patch.object(
                plugin,
                "_calculate_derived_metrics",
                side_effect=lambda r, c: r,
            ),
            patch.object(
                plugin,
                "_validate_result",
                return_value=(True, []),
            ),
            patch.object(
                plugin,
                "_build_prompt_context",
                return_value={},
            ),
        ):
            result = await plugin.execute(
                context={"input_content": {"profound_category_id": "cat-data"}},
                pipeline=MagicMock(),
                job_id=None,
            )

        assert result.profound_personas_used == ["Data Engineer", "Analytics Lead"]
