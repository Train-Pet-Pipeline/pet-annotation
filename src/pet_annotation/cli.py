"""CLI entry point for pet-annotation."""

from __future__ import annotations

import asyncio

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
@click.option(
    "--db",
    default=None,
    type=click.Path(),
    help="Path to SQLite database (overrides params.yaml)",
)
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
@click.option(
    "--pet-data-db",
    default=None,
    type=click.Path(),
    help="Path to pet-data SQLite (overrides annotation.pet_data_db_path in params.yaml)",
)
def annotate(batch_size, dry_run, params, db, annotator, modality, pet_data_db):
    """Batch annotate pending samples using the specified annotator paradigm."""
    from pathlib import Path

    from pet_annotation.store import AnnotationStore

    config = load_config(Path(params)) if params and Path(params).exists() else None
    db_path = db or (str(config.database.path) if config else "data/annotation.db")

    if batch_size and config:
        config.annotation.batch_size = batch_size

    if annotator == "human":
        click.echo(
            "Annotator paradigm 'human' not yet wired (Label Studio integration in Subagent D)"
        )
        return

    if dry_run:
        if annotator == "llm":
            llm_count = len(config.llm.annotators) if config else 0
            click.echo(
                f"dispatch=llm modality={modality} dry_run=True "
                f"configured_annotators={llm_count}"
            )
        elif annotator == "classifier":
            cls_count = len(config.classifier.annotators) if config else 0
            click.echo(
                f"dispatch=classifier modality={modality} dry_run=True "
                f"configured_annotators={cls_count}"
            )
        elif annotator == "rule":
            rule_count = len(config.rule.annotators) if config else 0
            click.echo(
                f"dispatch=rule modality={modality} dry_run=True "
                f"configured_annotators={rule_count}"
            )
        return

    # Actual dispatch via AnnotationOrchestrator (llm / classifier / rule)
    from pet_annotation.teacher.orchestrator import AnnotationOrchestrator

    store = AnnotationStore(db_path)
    store.init_schema()

    pet_data_db_path = (
        pet_data_db
        or (config.annotation.pet_data_db_path if config else "/data/pet-data/pet_data.db")
    )

    orch = AnnotationOrchestrator(config, store, pet_data_db_path)
    stats = asyncio.run(orch.run())
    click.echo(
        f"processed={stats['processed']} skipped={stats['skipped']} failed={stats['failed']}"
    )


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

    tables = (
        "llm_annotations",
        "classifier_annotations",
        "rule_annotations",
        "human_annotations",
    )
    for table in tables:
        count = store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        click.echo(f"  {table}: {count}")


if __name__ == "__main__":
    cli()
