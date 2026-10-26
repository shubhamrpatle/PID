import inspect
import os
import getpass
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import warnings

import huggingface_hub
import pytest
from huggingface_hub.utils import filter_repo_objects


SCRIPT = Path(__file__).resolve().parents[1] / "upload_cluster_runs.sh"
BODY = SCRIPT.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
API_SIGNATURE = inspect.signature(huggingface_hub.HfApi.upload_large_folder)


@pytest.mark.parametrize("args,expected", [
    ([], ["CaseStudyRef/RefWave-Cluster-Runs", str(Path(os.environ.get("WORK_ROOT", Path.cwd() / "refwave-assets")) / "runs")]),
    (["other-user/backup", "/tmp/other runs"], ["other-user/backup", "/tmp/other runs"]),
])
def test_shell_defaults_and_overrides_without_uploading(args, expected):
    prefix = SCRIPT.read_text().split('if [[ ! -d "$SOURCE" ]]', 1)[0]
    result = subprocess.run(
        ["bash", "-c", prefix + '\nprintf "%s\\n" "$REPO_ID" "$SOURCE"\n', str(SCRIPT), *args],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.splitlines() == expected


@pytest.fixture
def upload(tmp_path, monkeypatch):
    source = tmp_path / "runs"
    source.mkdir()
    api = Mock()
    api.whoami.return_value = {"name": "test-user"}
    api.repo_info.return_value = SimpleNamespace(private=False)
    factory = Mock(return_value=api)
    monkeypatch.setattr(huggingface_hub, "HfApi", factory)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    monkeypatch.setenv("HF_TOKEN", "test-placeholder-not-a-real-token")

    def run(repo="test-user/public-runs", workers="4"):
        monkeypatch.setattr("sys.argv", ["-", repo, str(source), workers])
        exec(compile(BODY, str(SCRIPT), "exec"), {"__name__": "__main__"})

    return SimpleNamespace(source=source, api=api, factory=factory, run=run)


def test_upload_keeps_checkpoints_and_logs_without_exposing_token(upload, capsys):
    upload.run()
    upload.api.create_repo.assert_called_once_with(
        repo_id="test-user/public-runs", repo_type="dataset", private=False, exist_ok=True
    )
    kwargs = upload.api.upload_large_folder.call_args.kwargs
    API_SIGNATURE.bind(object(), **kwargs)
    assert kwargs["folder_path"] == upload.source
    assert kwargs["private"] is False
    assert kwargs["num_workers"] == 4
    assert "delete_patterns" not in kwargs
    keep = [
        "run/stage1/checkpoint-0002500.pt", "run/stage1/best.pt", "run/stage1/last.pt",
        "run/stage2/optimizer_rank0.pt", "run/stage2/model.safetensors",
        "run/console.log", "run/wandb/offline-run/files/config.yaml", "run/metrics.json",
    ]
    skip = [
        ".env", "run/.env.local", "run/.netrc", "run/private.key", "run/id_rsa",
        ".cache/refwave-upload/destination.json", "run/.cache/temp", ".git/config",
        "run/checkpoint.pt.tmp", "run/.checkpoint-10.partial/model.pt",
    ]
    assert list(filter_repo_objects(keep + skip, ignore_patterns=kwargs["ignore_patterns"])) == keep
    assert "test-placeholder-not-a-real-token" not in capsys.readouterr().out


def test_existing_public_repository_needs_no_visibility_change(upload):
    upload.run()
    upload.api.update_repo_settings.assert_not_called()
    upload.api.upload_large_folder.assert_called_once()


def test_existing_private_repository_is_made_public(upload, capsys):
    upload.api.repo_info.side_effect = [SimpleNamespace(private=True), SimpleNamespace(private=False)]
    upload.run()
    upload.api.update_repo_settings.assert_called_once_with(
        repo_id="test-user/public-runs", repo_type="dataset", private=False
    )
    upload.api.upload_large_folder.assert_called_once()
    assert "including all existing files and history" in capsys.readouterr().out


def test_failed_visibility_change_blocks_upload(upload):
    upload.api.repo_info.return_value = SimpleNamespace(private=True)
    with pytest.raises(SystemExit, match="still private"):
        upload.run()
    upload.api.upload_large_folder.assert_not_called()
    assert not (upload.source / ".cache/refwave-upload/destination.json").exists()


def test_same_destination_can_resume_but_different_one_is_rejected(upload):
    upload.run()
    upload.run()
    assert upload.api.upload_large_folder.call_count == 2
    with pytest.raises(SystemExit, match="another repository"):
        upload.run(repo="test-user/another-repository")
    assert upload.api.upload_large_folder.call_count == 2
    marker = upload.source / ".cache/refwave-upload/destination.json"
    assert json.loads(marker.read_text())["repo_id"] == "test-user/public-runs"


def test_invalid_worker_count_fails_before_network(upload):
    with pytest.raises(SystemExit, match="at least 1"):
        upload.run(workers="0")
    upload.factory.assert_not_called()


def test_failed_upload_does_not_print_complete(upload, capsys):
    upload.api.upload_large_folder.side_effect = RuntimeError("Simulated network interruption")
    with pytest.raises(RuntimeError, match="Simulated"):
        upload.run()
    assert "UPLOAD COMPLETE" not in capsys.readouterr().out


def test_hidden_prompt_token_is_not_printed_or_saved(upload, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN")
    secret = "placeholder-hidden-token-not-a-real-secret"
    monkeypatch.setattr(getpass, "getpass", lambda *args, **kwargs: secret)
    with patch("builtins.open", return_value=io.StringIO()):
        upload.run()
    upload.factory.assert_called_once_with(token=secret)
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    for path in upload.source.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text()


def test_visible_password_fallback_is_rejected(upload, monkeypatch):
    monkeypatch.delenv("HF_TOKEN")

    def insecure_prompt(*args, **kwargs):
        warnings.warn("Cannot disable terminal echo", getpass.GetPassWarning)
        raise AssertionError("Visible input must never be reached")

    monkeypatch.setattr(getpass, "getpass", insecure_prompt)
    with patch("builtins.open", return_value=io.StringIO()):
        with pytest.raises(SystemExit, match="Hidden token entry is unavailable"):
            upload.run()
    upload.factory.assert_not_called()


def test_missing_terminal_without_token_is_rejected(upload, monkeypatch):
    monkeypatch.delenv("HF_TOKEN")
    with patch("builtins.open", side_effect=OSError("No terminal")):
        with pytest.raises(SystemExit, match="upload was not started"):
            upload.run()
    upload.factory.assert_not_called()
