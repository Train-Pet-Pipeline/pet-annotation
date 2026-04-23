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
        elif annotator == "human":
            human_count = len(config.human.annotators) if config else 0
            click.echo(
                f"dispatch=human modality={modality} dry_run=True "
                f"configured_annotators={human_count}"
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
    # --annotator selects exactly one paradigm to run (symmetric to --dry-run output).
    stats = asyncio.run(orch.run(paradigms=[annotator]))
    click.echo(
        f"processed={stats['processed']} skipped={stats['skipped']} failed={stats['failed']}"
    )


@cli.command(name="export")
@click.option("--format", "fmt", type=click.Choice(["sft", "dpo"]), required=True)
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--params", default="params.yaml", type=click.Path(exists=True))
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
    help="Annotator paradigm to export from (required)",
)
def export_cmd(fmt, output, params, db, annotator):
    """Export training data from the specified annotator paradigm table.

    Emits JSONL to --output file (or stdout if omitted).
    """
    import sys
    from pathlib import Path

    from pet_annotation.export.sft_dpo import to_dpo_pairs, to_sft_samples
    from pet_annotation.store import AnnotationStore

    config = load_config(Path(params))
    db_path = db or str(config.database.path)
    store = AnnotationStore(db_path)
    store.init_schema()

    output_path = Path(output) if output else None

    if fmt == "sft":
        samples = to_sft_samples(store, annotator_type=annotator, output_path=output_path)
        count = len(samples)
        if output_path is None:
            for s in samples:
                import json as _json
                sys.stdout.write(_json.dumps(s, ensure_ascii=False) + "\n")
    else:  # dpo
        pairs = to_dpo_pairs(store, annotator_type=annotator, output_path=output_path)
        count = len(pairs)
        if output_path is None:
            for p in pairs:
                import json as _json
                sys.stdout.write(_json.dumps(p, ensure_ascii=False) + "\n")

    destination = str(output_path) if output_path else "stdout"
    click.echo(f"exported {count} {fmt} samples to {destination}", err=True)


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
