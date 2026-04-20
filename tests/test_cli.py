"""Tests for CLI modality dispatch (Task B11)."""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pet_annotation.cli import cli


@pytest.fixture()
def params_file(tmp_path: Path) -> Path:
    """Minimal params.yaml fixture for CLI tests."""
    db_path = tmp_path / "test.db"
    content = textwrap.dedent(f"""
        database:
          path: "{db_path}"
          data_root: "{tmp_path}"

        annotation:
          batch_size: 4
          max_concurrent: 2
          max_daily_tokens: 1000000
          review_sampling_rate: 0.15
          low_confidence_threshold: 0.70
          primary_model: "qwen2.5-vl-72b"
          schema_version: "1.0"

        models:
          qwen2.5-vl-72b:
            provider: "openai_compat"
            base_url: "https://example.com"
            model_name: "qwen2.5-vl-72b-instruct"
            accounts:
              - key_env: "QWEN_API_KEY_1"
                rpm: 60
                tpm: 100000
            timeout: 60
            max_retries: 3

        dpo:
          min_pairs_per_release: 500
    """)
    p = tmp_path / "params.yaml"
    p.write_text(content)
    return p


class TestExportModalityDispatch:
    def test_export_audio_wires_audio_labels(self, params_file: Path, tmp_path: Path) -> None:
        """export --format=audio --modality=audio calls export_audio_labels once."""
        runner = CliRunner()
        out_path = str(tmp_path / "audio_labels.jsonl")
        with patch(
            "pet_annotation.export.to_audio_labels.export_audio_labels",
            return_value=0,
        ) as mock_export:
            result = runner.invoke(
                cli,
                [
                    "export",
                    "--format=audio",
                    "--modality=audio",
                    f"--output={out_path}",
                    f"--params={params_file}",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_export.assert_called_once_with(Path(out_path))

    def test_export_format_modality_mismatch_sft_audio(self, params_file: Path) -> None:
        """export --format=sft --modality=audio returns non-zero exit."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "export",
                "--format=sft",
                "--modality=audio",
                f"--params={params_file}",
            ],
        )
        assert result.exit_code != 0

    def test_export_format_modality_mismatch_dpo_audio(self, params_file: Path) -> None:
        """export --format=dpo --modality=audio returns non-zero exit."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "export",
                "--format=dpo",
                "--modality=audio",
                f"--params={params_file}",
            ],
        )
        assert result.exit_code != 0

    def test_export_format_modality_mismatch_audio_vision(self, params_file: Path) -> None:
        """export --format=audio --modality=vision returns non-zero exit."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "export",
                "--format=audio",
                "--modality=vision",
                f"--params={params_file}",
            ],
        )
        assert result.exit_code != 0

    def test_modality_default_is_vision(self, params_file: Path, tmp_path: Path) -> None:
        """export --format=sft without --modality uses vision path (export_sharegpt called)."""
        runner = CliRunner()
        out_path = str(tmp_path / "sft.jsonl")
        with patch(
            "pet_annotation.export.to_sharegpt.export_sharegpt",
            return_value=0,
        ) as mock_sft, patch(
            "pet_annotation.store.AnnotationStore",
        ):
            result = runner.invoke(
                cli,
                [
                    "export",
                    "--format=sft",
                    f"--output={out_path}",
                    f"--params={params_file}",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_sft.assert_called_once()


class TestAnnotateModalityDispatch:
    def test_annotate_audio_rejected(self, params_file: Path) -> None:
        """annotate --modality=audio exits with code 1 and error message."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["annotate", "--modality=audio", f"--params={params_file}"],
        )
        assert result.exit_code == 1
        assert "not yet implemented" in result.output.lower() or "audio" in result.output.lower()


class TestCheckModalityDispatch:
    def test_check_audio_rejected(self, params_file: Path) -> None:
        """check --modality=audio exits with code 1."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["check", "--modality=audio", f"--params={params_file}"],
        )
        assert result.exit_code == 1


class TestLsImportModalityDispatch:
    def test_ls_import_audio_rejected_or_raises(self, params_file: Path) -> None:
        """ls-import --modality=audio surfaces NotImplementedError as non-zero exit."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "ls-import",
                "--modality=audio",
                "--ls-key=fake-key",
                f"--params={params_file}",
                "--ls-url=http://localhost:8080",
            ],
        )
        # NotImplementedError should surface as non-zero exit (ClickException or exception)
        assert result.exit_code != 0

    def test_ls_import_vision_passes_modality(self, params_file: Path) -> None:
        """ls-import --modality=vision passes modality=vision to import_needs_review."""
        runner = CliRunner()
        with patch(
            "pet_annotation.human_review.import_to_ls.import_needs_review",
            return_value=0,
        ) as mock_import, patch(
            "pet_annotation.human_review.ls_auth.get_ls_session",
            return_value=None,
        ), patch(
            "pet_annotation.store.AnnotationStore",
        ):
            result = runner.invoke(
                cli,
                [
                    "ls-import",
                    "--modality=vision",
                    "--ls-key=fake-key",
                    f"--params={params_file}",
                    "--ls-url=http://localhost:8080",
                ],
            )
        assert result.exit_code == 0, result.output
        call_kwargs = mock_import.call_args
        # modality="vision" should be passed
        assert call_kwargs.kwargs.get("modality") == "vision" or (
            len(call_kwargs.args) >= 5 and call_kwargs.args[4] == "vision"
        )


class TestLsExportModalityDispatch:
    def test_ls_export_audio_rejected(self, params_file: Path) -> None:
        """ls-export --modality=audio exits with code 1 (not implemented)."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "ls-export",
                "--modality=audio",
                "--ls-key=fake-key",
                f"--params={params_file}",
                "--ls-url=http://localhost:8080",
            ],
        )
        assert result.exit_code != 0

    def test_ls_export_vision_works(self, params_file: Path) -> None:
        """ls-export --modality=vision calls export_reviewed (vision path)."""
        runner = CliRunner()
        with patch(
            "pet_annotation.human_review.export_from_ls.export_reviewed",
            return_value=0,
        ), patch(
            "pet_annotation.human_review.ls_auth.get_ls_session",
            return_value=None,
        ), patch(
            "pet_annotation.store.AnnotationStore",
        ):
            result = runner.invoke(
                cli,
                [
                    "ls-export",
                    "--modality=vision",
                    "--ls-key=fake-key",
                    f"--params={params_file}",
                    "--ls-url=http://localhost:8080",
                ],
            )
        assert result.exit_code == 0, result.output
