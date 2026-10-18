"""Download the EverMemBench dataset from Hugging Face into data/raw/ (gitignored).

    uv run --extra data python scripts/download_data.py

The revision is pinned: the HF repo is named "Dynamic" and its data files have
been re-uploaded more than once, so an unpinned download could silently change
the corpus under an existing `corpus_id`. The pinned revision is the first one
that declares a license (apache-2.0); see docs/decisions.md D-002.

After downloading, a MANIFEST.json (repo, revision, license, per-file sha256) is
written next to the data so later stages can record exactly what they read.
The raw data must never be committed (CLAUDE.md §2.3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

REPO_ID = "EverMind-AI/EverMemBench-Dynamic"
# "Set license to apache-2.0", 2026-07-16. Older revisions declare no license.
PINNED_REVISION = "a6b210a32248e841967b7b64a64281d2ff3f669d"
EXPECTED_LICENSE = "apache-2.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "evermembench"

# Current per-topic format (0N/, has evidence locations `R`) and the legacy
# per-batch format (dataset/NNN/, includes the no-evidence F_HL questions).
# The older JSON files at the repo root use a different schema without
# evidence locations, so they are skipped unless --all is given.
DEFAULT_PATTERNS = ["README.md", "0*/*.json", "dataset/*.json"]
MANIFEST_NAME = "MANIFEST.json"


class DownloadError(Exception):
    pass


def check_license(declared: str | None, expected: str = EXPECTED_LICENSE) -> None:
    """Refuse to download a revision whose declared license we have not reviewed."""
    if declared != expected:
        raise DownloadError(
            f"dataset card declares license {declared!r}, expected {expected!r}. "
            "Review the license and update docs/decisions.md before downloading this revision."
        )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(data_dir: Path, *, repo_id: str, revision: str, license_id: str) -> dict:
    files = sorted(
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.name != MANIFEST_NAME and ".cache" not in p.relative_to(data_dir).parts
    )
    if not files:
        raise DownloadError(f"no files found under {data_dir}")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "license": license_id,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": [
            {
                "path": p.relative_to(data_dir).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256_of(p),
            }
            for p in files
        ],
    }


def download(revision: str, out_dir: Path, patterns: list[str] | None) -> Path:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        raise DownloadError(
            "huggingface_hub is not installed. Run: "
            "uv run --extra data python scripts/download_data.py"
        ) from None

    info = HfApi().dataset_info(REPO_ID, revision=revision)
    if info.sha != revision:
        raise DownloadError(f"requested revision {revision}, hub resolved {info.sha}")
    card = info.card_data.to_dict() if info.card_data is not None else {}
    license_id = card.get("license")
    check_license(license_id)

    target = out_dir / revision[:8]
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=target,
        allow_patterns=patterns,
    )
    manifest = build_manifest(target, repo_id=REPO_ID, revision=revision, license_id=license_id)
    (target / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download EverMemBench into data/raw/.")
    parser.add_argument(
        "--revision",
        default=PINNED_REVISION,
        help="HF commit sha (default: the pinned, license-reviewed revision)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument(
        "--all", action="store_true", help="download every file, not just the default subset"
    )
    args = parser.parse_args(argv)

    try:
        target = download(args.revision, args.out, None if args.all else DEFAULT_PATTERNS)
    except DownloadError as exc:
        print(f"download_data: {exc}", file=sys.stderr)
        return 1
    print(f"downloaded {REPO_ID}@{args.revision[:8]} to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
