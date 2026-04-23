"""Tests for CLI --annotator dispatch — llm / classifier / rule / human."""

from click.testing import CliRunner

from pet_annotation.cli import cli


def test_cli_annotate_llm_audio_routes_to_llm_table(tmp_path, monkeypatch):
    """annotate --annotator=llm --dry-run must exit 0 and echo dispatch info."""
    runner = CliRunner()
    db = tmp_path / "t.db"
    result = runner.invoke(
        cli,
        ["annotate", "--annotator", "llm", "--modality", "audio", "--db", str(db), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "dispatch=llm" in result.output.lower()


def test_cli_rejects_unknown_annotator():
    """annotate --annotator=vlm must fail with non-zero exit code."""
    runner = CliRunner()
    result = runner.invoke(cli, ["annotate", "--annotator", "vlm", "--modality", "vision"])
    assert result.exit_code != 0
    output = result.output.lower()
    assert "invalid" in output or "unknown" in output or "not one of" in output


def test_cli_annotate_classifier_dry_run(tmp_path):
    """annotate --annotator=classifier --dry-run exits 0 and echoes classifier dispatch info."""
    runner = CliRunner()
    db = tmp_path / "t.db"
    result = runner.invoke(
        cli,
        [
            "annotate",
            "--annotator",
            "classifier",
            "--modality",
            "vision",
            "--db",
            str(db),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dispatch=classifier" in result.output.lower()


def test_cli_annotate_rule_dry_run(tmp_path):
    """annotate --annotator=rule --dry-run exits 0 and echoes rule dispatch info."""
    runner = CliRunner()
    db = tmp_path / "t.db"
    result = runner.invoke(
        cli,
        [
            "annotate",
            "--annotator",
            "rule",
            "--modality",
            "vision",
            "--db",
            str(db),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dispatch=rule" in result.output.lower()


def test_cli_annotate_human_dry_run(tmp_path):
    """annotate --annotator=human --dry-run exits 0 and echoes dispatch=human info."""
    runner = CliRunner()
    db = tmp_path / "t.db"
    result = runner.invoke(
        cli,
        [
            "annotate",
            "--annotator",
            "human",
            "--modality",
            "vision",
            "--db",
            str(db),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dispatch=human" in result.output.lower()
