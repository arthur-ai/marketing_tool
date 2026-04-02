"""
Tests for DELETE /content/{content_type}/{file_id} endpoint.

Covers: owner delete, non-owner 403, admin delete-any, unknown file 404,
invalid content_type 400, meta sidecar removed with content, non-UUID 400.
"""

import json
import os
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from marketing_project.api.upload import router
from marketing_project.middleware.keycloak_auth import get_current_user
from tests.utils.keycloak_test_helpers import create_user_context

OWNER_ID = "owner-user-111"
OTHER_ID = "other-user-222"


@pytest.fixture
def owner_user():
    return create_user_context(user_id=OWNER_ID, roles=[])


@pytest.fixture
def other_user():
    return create_user_context(user_id=OTHER_ID, roles=[])


@pytest.fixture
def admin_user():
    return create_user_context(user_id="admin-999", roles=["admin"])


@pytest.fixture
def app_as_owner(owner_user):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: owner_user
    return TestClient(app)


@pytest.fixture
def app_as_other(other_user):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: other_user
    return TestClient(app)


@pytest.fixture
def app_as_admin(admin_user):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: admin_user
    return TestClient(app)


@pytest.fixture
def fake_upload_dir(tmp_path):
    return str(tmp_path / "uploads")


@pytest.fixture
def fake_content_dir(tmp_path):
    return str(tmp_path / "content")


def _make_meta_and_content(upload_dir, content_dir, content_type, file_id, uploaded_by):
    """Create both meta sidecar and a dummy content file."""
    os.makedirs(upload_dir, exist_ok=True)
    meta_path = os.path.join(upload_dir, f"{file_id}.meta.json")
    with open(meta_path, "w") as mf:
        json.dump(
            {
                "file_id": file_id,
                "filename": "test.json",
                "uploaded_by": uploaded_by,
                "content_type": content_type,
            },
            mf,
        )

    content_type_dir = os.path.join(content_dir, f"{content_type}s")
    os.makedirs(content_type_dir, exist_ok=True)
    content_file = os.path.join(content_type_dir, f"{file_id}_test.json")
    Path(content_file).write_text('{"title": "test"}')

    return meta_path, content_file


class TestDeleteContentItem:
    def test_owner_can_delete(self, app_as_owner, fake_upload_dir, fake_content_dir):
        file_id = str(uuid.uuid4())
        meta_path, content_file = _make_meta_and_content(
            fake_upload_dir, fake_content_dir, "blog_post", file_id, OWNER_ID
        )

        with (
            patch("marketing_project.api.upload.UPLOAD_DIR", fake_upload_dir),
            patch("marketing_project.api.upload.CONTENT_DIR", fake_content_dir),
        ):
            resp = app_as_owner.delete(f"/content/blog_post/{file_id}")

        assert resp.status_code == 200
        assert not os.path.exists(meta_path)
        assert not os.path.exists(content_file)

    def test_non_owner_gets_403(self, app_as_other, fake_upload_dir, fake_content_dir):
        file_id = str(uuid.uuid4())
        meta_path, _ = _make_meta_and_content(
            fake_upload_dir, fake_content_dir, "blog_post", file_id, OWNER_ID
        )

        with (
            patch("marketing_project.api.upload.UPLOAD_DIR", fake_upload_dir),
            patch("marketing_project.api.upload.CONTENT_DIR", fake_content_dir),
        ):
            resp = app_as_other.delete(f"/content/blog_post/{file_id}")

        assert resp.status_code == 403
        assert os.path.exists(meta_path)

    def test_admin_can_delete_any(
        self, app_as_admin, fake_upload_dir, fake_content_dir
    ):
        file_id = str(uuid.uuid4())
        meta_path, content_file = _make_meta_and_content(
            fake_upload_dir, fake_content_dir, "transcript", file_id, OWNER_ID
        )

        with (
            patch("marketing_project.api.upload.UPLOAD_DIR", fake_upload_dir),
            patch("marketing_project.api.upload.CONTENT_DIR", fake_content_dir),
        ):
            resp = app_as_admin.delete(f"/content/transcript/{file_id}")

        assert resp.status_code == 200
        assert not os.path.exists(meta_path)
        assert not os.path.exists(content_file)

    def test_unknown_file_returns_404(
        self, app_as_owner, fake_upload_dir, fake_content_dir
    ):
        file_id = str(uuid.uuid4())
        os.makedirs(fake_upload_dir, exist_ok=True)

        with (
            patch("marketing_project.api.upload.UPLOAD_DIR", fake_upload_dir),
            patch("marketing_project.api.upload.CONTENT_DIR", fake_content_dir),
        ):
            resp = app_as_owner.delete(f"/content/blog_post/{file_id}")

        assert resp.status_code == 404

    def test_invalid_content_type_returns_400(self, app_as_owner):
        file_id = str(uuid.uuid4())
        resp = app_as_owner.delete(f"/content/invalid_type/{file_id}")
        assert resp.status_code == 400

    def test_non_uuid_file_id_returns_400(self, app_as_owner):
        resp = app_as_owner.delete("/content/blog_post/not-a-uuid-at-all")
        assert resp.status_code == 400

    def test_meta_deleted_with_content(
        self, app_as_owner, fake_upload_dir, fake_content_dir
    ):
        """Both meta sidecar and content file are removed on success."""
        file_id = str(uuid.uuid4())
        meta_path, content_file = _make_meta_and_content(
            fake_upload_dir, fake_content_dir, "release_notes", file_id, OWNER_ID
        )

        with (
            patch("marketing_project.api.upload.UPLOAD_DIR", fake_upload_dir),
            patch("marketing_project.api.upload.CONTENT_DIR", fake_content_dir),
        ):
            resp = app_as_owner.delete(f"/content/release_notes/{file_id}")

        assert resp.status_code == 200
        assert not os.path.exists(meta_path), "meta sidecar should be deleted"
        assert not os.path.exists(content_file), "content file should be deleted"

    def test_no_content_file_still_deletes_meta(
        self, app_as_owner, fake_upload_dir, fake_content_dir
    ):
        """If content file is missing (e.g. S3-only), meta is still cleaned up."""
        file_id = str(uuid.uuid4())
        os.makedirs(fake_upload_dir, exist_ok=True)
        # Write meta only, no content file
        meta_path = os.path.join(fake_upload_dir, f"{file_id}.meta.json")
        with open(meta_path, "w") as mf:
            json.dump(
                {
                    "file_id": file_id,
                    "uploaded_by": OWNER_ID,
                    "content_type": "blog_post",
                },
                mf,
            )

        with (
            patch("marketing_project.api.upload.UPLOAD_DIR", fake_upload_dir),
            patch("marketing_project.api.upload.CONTENT_DIR", fake_content_dir),
        ):
            resp = app_as_owner.delete(f"/content/blog_post/{file_id}")

        assert resp.status_code == 200
        assert not os.path.exists(meta_path)
