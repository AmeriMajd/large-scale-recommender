import json
from pathlib import Path
from types import SimpleNamespace

from recommender.serving import triton_repository


def _model(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "saved_model.pb").write_bytes(b"saved-model")
    variables = path / "variables"
    variables.mkdir()
    (variables / "variables.index").write_bytes(b"index")


def test_committed_model_configs_match_repository_contract() -> None:
    for name in triton_repository.MODEL_NAMES:
        config = Path("triton-models", name, "config.pbtxt").read_text(
            encoding="utf-8"
        )
        assert config.startswith(f'name: "{name}"')
        assert 'platform: "tensorflow_savedmodel"' in config
        assert "max_batch_size: 0" in config
        assert "\n+" not in config

    user_tower = Path(
        "triton-models/user_tower/config.pbtxt"
    ).read_text(encoding="utf-8")
    assert 'name: "user_id" data_type: TYPE_INT64' in user_tower
    assert 'name: "embedding" data_type: TYPE_FP32' in user_tower

    for name in ("wide_and_deep", "dcn"):
        ranker = Path("triton-models", name, "config.pbtxt").read_text(
            encoding="utf-8"
        )
        assert ranker.count("data_type:") == 13
        assert 'name: "logit" data_type: TYPE_FP32' in ranker


def test_export_repository_creates_versioned_models(
    tmp_path: Path, monkeypatch
) -> None:
    retrieval = tmp_path / "retrieval" / "run-r"
    ranking = tmp_path / "ranking" / "run-k"
    _model(retrieval / "user_tower")
    _model(ranking / "wide_and_deep")
    _model(ranking / "dcn")
    for base, run_id, run_dir in (
        (tmp_path / "retrieval", "run-r", retrieval),
        (tmp_path / "ranking", "run-k", ranking),
    ):
        (base / "latest.json").write_text(
            json.dumps({"run_id": run_id, "run_dir": str(run_dir)}),
            encoding="utf-8",
        )
    repository = tmp_path / "repository"
    for name in triton_repository.MODEL_NAMES:
        model_dir = repository / name
        model_dir.mkdir(parents=True)
        (model_dir / "config.pbtxt").write_text(
            f'name: "{name}"\n', encoding="utf-8"
        )
    config = SimpleNamespace(
        retrieval=SimpleNamespace(model_dir=tmp_path / "retrieval"),
        ranking=SimpleNamespace(model_dir=tmp_path / "ranking"),
    )
    monkeypatch.setattr(triton_repository, "load_config", lambda: config)

    manifest = triton_repository.export_repository(repository)

    assert manifest["retrieval_run_id"] == "run-r"
    assert manifest["ranking_run_id"] == "run-k"
    for name in triton_repository.MODEL_NAMES:
        assert (
            repository / name / "1" / "model.savedmodel" / "saved_model.pb"
        ).is_file()
