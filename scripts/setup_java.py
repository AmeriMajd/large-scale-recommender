"""Install a project-local Eclipse Temurin 17 JDK for Spark."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

DOWNLOAD_URL = (
    "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/"
    "jdk/hotspot/normal/eclipse"
)


def _download(url: str, destination: Path) -> str:
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "recsys-setup/1.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def install_java(tools_dir: Path, force: bool = False) -> Path:
    """Install Temurin under ``tools_dir/java`` and return JAVA_HOME."""
    java_home = tools_dir / "java"
    java_binary = java_home / "bin" / "java"
    if java_binary.is_file() and not force:
        return java_home

    tools_dir.mkdir(parents=True, exist_ok=True)
    if java_home.exists():
        shutil.rmtree(java_home)

    with tempfile.TemporaryDirectory(prefix="recsys-java-") as temp_name:
        temp_dir = Path(temp_name)
        archive_path = temp_dir / "temurin.tar.gz"
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        print("Downloading project-local Eclipse Temurin 17...")
        digest = _download(DOWNLOAD_URL, archive_path)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(extract_dir, filter="data")

        roots = [path for path in extract_dir.iterdir() if path.is_dir()]
        if len(roots) != 1 or not (roots[0] / "bin" / "java").is_file():
            raise ValueError("Unexpected Temurin archive structure")
        shutil.move(str(roots[0]), java_home)

    metadata = {
        "source_url": DOWNLOAD_URL,
        "installed_at_utc": datetime.now(UTC).isoformat(),
        "archive_sha256": digest,
    }
    (java_home / "recsys_install_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return java_home


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-dir", type=Path, default=Path(".tools"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    java_home = install_java(args.tools_dir, force=args.force)
    print(f"JAVA_HOME={java_home.resolve()}")


if __name__ == "__main__":
    main()
