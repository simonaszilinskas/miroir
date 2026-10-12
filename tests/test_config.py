from __future__ import annotations

from pathlib import Path

import pytest

from miroir.config import Settings, eval_result_path, eval_set_path
from miroir.records.model import Collection


def test_export_paths_use_folder_names_on_disk() -> None:
    settings = Settings(exports_dir=Path("/exports"))
    assert settings.export_jsonl(Collection.ROSE_VALLAND) == Path(
        "/exports/rose-valand/rose-valand.jsonl"
    )


def test_settings_read_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MIROIR_DATA_DIR", str(tmp_path))
    assert Settings().records_parquet == tmp_path / "records.parquet"


def test_secrets_use_their_usual_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    settings = Settings()
    assert settings.openrouter_api_key is not None
    assert settings.openrouter_api_key.get_secret_value() == "sk-test"
    assert settings.hf_token is not None
    assert "hf-test" not in repr(settings)


def test_eval_result_path_follows_the_rule() -> None:
    path = eval_result_path("R1", "A", "dinov2-vitl14")
    assert path.parts[-4:] == ("eval", "results", "R1", "A-dinov2-vitl14.json")


def test_sealed_half_opens_only_in_r4() -> None:
    assert eval_set_path("T1", "tune", round_id="R1").name == "tune.jsonl"
    assert eval_set_path("T1", "sealed", round_id="R4").name == "sealed.jsonl"
    with pytest.raises(PermissionError):
        eval_set_path("T1", "sealed", round_id="R2")


@pytest.mark.parametrize("bad", ["../x", "a/b", ""])
def test_path_parts_are_checked(bad: str) -> None:
    with pytest.raises(ValueError, match="path component"):
        eval_result_path("R1", "A", bad)


def test_round_ids_are_checked() -> None:
    with pytest.raises(ValueError, match="round"):
        eval_result_path("round1", "A", "x")
