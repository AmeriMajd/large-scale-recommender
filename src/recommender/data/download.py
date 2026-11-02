"""Download and safely extract MovieLens 25M."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
REQUIRED_FILES = {"ratings.csv", "movies.csv", "tags.csv", "links.csv"}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive(archive: zipfile.ZipFile) -> None:
    """Reject path traversal and archives missing required source tables."""
    names = set(archive.namelist())
    available_files = {Path(name).name for name in names if not name.endswith("/")}
    missing = REQUIRED_FILES - available_files
    if missing:
        raise ValueError(f"MovieLens archive is missing files: {sorted(missing)}")

    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path in dataset archive: {name}")


def download_dataset(destination_root: Path, force: bool = False) -> Path:
    """Download MovieLens 25M and return the extracted dataset directory."""
    destination_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = destination_root / "ml-25m"
    if dataset_dir.exists() and not force:
        missing = REQUIRED_FILES - {path.name for path in dataset_dir.iterdir()}
        if missing:
            raise FileExistsError(
                f"{dataset_dir} exists but is incomplete: missing {sorted(missing)}"
            )
        return dataset_dir

    if dataset_dir.exists() and force:
        shutil.rmtree(dataset_dir)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        archive_path = Path(temp_file.name)

    try:
        print(f"Downloading {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, archive_path)
        digest = sha256(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            _validate_archive(archive)
            archive.extractall(destination_root)

        metadata = {
            "source_url": DATASET_URL,
            "downloaded_at_utc": datetime.now(UTC).isoformat(),
            "archive_sha256": digest,
        }
        (dataset_dir / "download_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        archive_path.unlink(missing_ok=True)

    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw"),
        help="Parent directory for the ml-25m dataset",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = download_dataset(args.destination, force=args.force)
    print(f"Dataset ready at {dataset_dir}")


if __name__ == "__main__":
    main()
