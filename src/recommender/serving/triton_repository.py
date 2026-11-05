"""Export the latest SavedModels into Triton's versioned repository layout."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from recommender.config import load_config

MODEL_NAMES = ("user_tower", "wide_and_deep", "dcn")


def _pointer(path: Path) -> tuple[str, Path]:
    value = json.loads(path.read_text(encoding="utf-8"))
    run_dir = Path(value["run_dir"])
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    return str(value["run_id"]), run_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_repository(repository: Path = Path("triton-models")) -> dict:
    """Copy three SavedModels to version 1 and return an auditable manifest."""
    config = load_config()
    retrieval_id, retrieval_dir = _pointer(
        config.retrieval.model_dir / "latest.json"
    )
    ranking_id, ranking_dir = _pointer(config.ranking.model_dir / "latest.json")
    sources = {
        "user_tower": retrieval_dir / "user_tower",
        "wide_and_deep": ranking_dir / "wide_and_deep",
        "dcn": ranking_dir / "dcn",
    }
    models = {}
    for name, source in sources.items():
        config_path = repository / name / "config.pbtxt"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing Triton config: {config_path}")
        saved_model = source / "saved_model.pb"
        if not saved_model.is_file():
            raise FileNotFoundError(f"missing SavedModel: {saved_model}")
        destination = repository / name / "1" / "model.savedmodel"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
        models[name] = {
            "source": str(source),
            "destination": str(destination),
            "saved_model_sha256": _sha256(saved_model),
        }
    manifest = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "repository": str(repository),
        "triton_container": "nvcr.io/nvidia/tritonserver:25.02-py3",
        "retrieval_run_id": retrieval_id,
        "ranking_run_id": ranking_id,
        "models": models,
    }
    (repository / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    print(json.dumps(export_repository(), indent=2))


if __name__ == "__main__":
    main()
