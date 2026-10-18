"""Offline tests for scripts/download_data.py (no network, no huggingface_hub needed)."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "download_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dd = _load_module()


def test_revision_is_pinned_to_a_full_commit_sha():
    assert len(dd.PINNED_REVISION) == 40
    int(dd.PINNED_REVISION, 16)


@pytest.mark.parametrize("declared", [None, "", "cc-by-nc-4.0", "Apache-2.0"])
def test_unreviewed_license_is_refused(declared):
    with pytest.raises(dd.DownloadError, match=r"decisions\.md"):
        dd.check_license(declared)


def test_reviewed_license_is_accepted():
    dd.check_license("apache-2.0")


def test_manifest_lists_files_with_sha256_and_skips_cache_and_itself(tmp_path):
    (tmp_path / "01").mkdir()
    (tmp_path / "01" / "qa_01.json").write_bytes(b"[]")
    (tmp_path / "README.md").write_bytes(b"hello")
    (tmp_path / ".cache" / "huggingface").mkdir(parents=True)
    (tmp_path / ".cache" / "huggingface" / "lock").write_bytes(b"x")
    (tmp_path / dd.MANIFEST_NAME).write_bytes(b"{}")

    manifest = dd.build_manifest(tmp_path, repo_id="r", revision="abc", license_id="apache-2.0")

    assert manifest["revision"] == "abc"
    assert manifest["license"] == "apache-2.0"
    assert manifest["files"] == [
        {"path": "01/qa_01.json", "bytes": 2, "sha256": hashlib.sha256(b"[]").hexdigest()},
        {"path": "README.md", "bytes": 5, "sha256": hashlib.sha256(b"hello").hexdigest()},
    ]


def test_manifest_refuses_empty_download(tmp_path):
    with pytest.raises(dd.DownloadError, match="no files"):
        dd.build_manifest(tmp_path, repo_id="r", revision="abc", license_id="apache-2.0")
