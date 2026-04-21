"""CLI entry point for pet-annotation."""

from __future__ import annotations

import click

from pet_annotation.config import load_config, setup_logging


@click.group()
def cli():
    """pet-annotation: VLM annotation, quality check, and training data export."""
    setup_logging()


@cli.command()
@click.option("--batch-size", default=None, type=int, help="Override params.yaml batch_size")
@click.option("--dry-run", is_flag=True, help="Print plan without calling APIs")
@click.option("--params", default="params.yaml", show_default=True, type=click.Path())
@click.option("--db", default=None, type=click.Path(), help="Path to SQLite database (overrides params.yaml)")
@click.option(
    "--annotator",
    type=click.Choice(["llm", "classifier", "rule", "human"]),
    required=True,
    help="Annotator paradigm to dispatch (required)",
)
@click.option(
    "--modality",
    type=click.Choice(["vision", "audio", "sensor", "multimodal"]),
    default="vision",
    show_default=True,
    help="Modality of the samples being annotated",
)
def annotate(batch_size, dry_run, params, db, annotator, modality):
    """Batch annotate pending samples using the specified annotator paradigm."""
    if dry_run:
        click.echo(f"dispatch={annotator} modality={modality} dry_run=True")
        return

    from pathlib import Path

    from pet_annotation.store import AnnotationStore

    config = load_config(Path(params)) if params and Path(params).exists() else None
    db_path = db or (str(config.database.path) if config else "data/annotation.db")

    if batch_size and config:
        config.annotation.batch_size = batch_size

    store = AnnotationStore(db_path)
    store.init_schema()

    click.echo(f"dispatch={annotator} modality={modality}")
    click.echo(f"Annotator paradigm '{annotator}' pipeline not yet wired — use dry-run for now.")


@cli.command(name="export")
@click.option("--format", "fmt", type=click.Choice(["sft", "dpo"]), required=True)
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--params", default="params.yaml", type=click.Path(exists=True))
@click.option(
    "--annotator",
    type=click.Choice(["llm", "classifier", "rule", "human"]),
    required=True,
    help="Annotator paradigm to export from (required)",
)
def export_cmd(fmt, output, params, annotator):
    """Export training data from the specified annotator paradigm table."""
    from pathlib import Path

    from pet_annotation.store import AnnotationStore

    config = load_config(Path(params))
    store = AnnotationStore(str(config.database.path))
    store.init_schema()

    click.echo(f"export fmt={fmt} annotator={annotator}")
    click.echo("Export pipeline not yet wired for 4-table schema — contributions welcome.")


@cli.command()
@click.option("--params", default="params.yaml", type=click.Path(exists=True))
def stats(params):
    """Print annotation progress statistics."""
    from pathlib import Path

    from pet_annotation.store import AnnotationStore

    config = load_config(Path(params))
    store = AnnotationStore(str(config.database.path))
    store.init_schema()

    for table in ("llm_annotations", "classifier_annotations", "rule_annotations", "human_annotations"):
        count = store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        click.echo(f"  {table}: {count}")


if __name__ == "__main__":
    cli()
