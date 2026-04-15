"""CLI entry point for pet-annotation."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click

from pet_annotation.config import load_config, setup_logging


@click.group()
def cli():
    """pet-annotation: VLM annotation, quality check, and training data export."""
    setup_logging()


@cli.command()
@click.option("--batch-size", default=None, type=int, help="Override params.yaml batch_size")
@click.option("--dry-run", is_flag=True, help="Print plan without calling APIs")
@click.option("--params", default="params.yaml", type=click.Path(exists=True))
def annotate(batch_size, dry_run, params):
    """Batch annotate pending frames using configured models."""
    from pet_annotation.store import AnnotationStore
    from pet_annotation.teacher.orchestrator import AnnotationOrchestrator

    config = load_config(Path(params))
    if batch_size:
        config.annotation.batch_size = batch_size

    if dry_run:
        store = AnnotationStore(db_path=Path(config.database.path))
        pending = store.fetch_pending_frames(limit=9999)
        click.echo(f"Pending frames: {len(pending)}")
        click.echo(f"Models: {list(config.models.keys())}")
        click.echo(f"Primary: {config.annotation.primary_model}")
        store.close()
        return

    store = AnnotationStore(db_path=Path(config.database.path))
    try:
        orch = AnnotationOrchestrator(config=config, store=store)
        stats = asyncio.run(orch.run())
        click.echo(json.dumps(stats, indent=2))
    finally:
        store.close()


@cli.command()
@click.option("--params", default="params.yaml", type=click.Path(exists=True))
def check(params):
    """Run quality check on auto_checked annotations."""
    from pet_annotation.quality.auto_check import run_auto_check
    from pet_annotation.store import AnnotationStore

    config = load_config(Path(params))
    store = AnnotationStore(db_path=Path(config.database.path))
    try:
        stats = run_auto_check(
            store,
            config.annotation.review_sampling_rate,
            config.annotation.low_confidence_threshold,
            config.annotation.primary_model,
        )
        click.echo(json.dumps(stats, indent=2))
    finally:
        store.close()


@cli.command(name="export")
@click.option("--format", "fmt", type=click.Choice(["sft", "dpo", "audio"]), required=True)
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--params", default="params.yaml", type=click.Path(exists=True))
def export_cmd(fmt, output, params):
    """Export training data in the specified format."""
    from pet_annotation.store import AnnotationStore

    config = load_config(Path(params))
    store = AnnotationStore(db_path=Path(config.database.path))

    try:
        if fmt == "sft":
            from pet_annotation.export.to_sharegpt import export_sharegpt
            out = Path(output) if output else Path("exports/sft_train.jsonl")
            count = export_sharegpt(store, out, config.annotation.schema_version)
            click.echo(f"Exported {count} SFT records to {out}")

        elif fmt == "dpo":
            from pet_annotation.dpo.generate_pairs import generate_cross_model_pairs
            from pet_annotation.export.to_dpo_pairs import export_dpo_pairs
            pairs = generate_cross_model_pairs(
                store, config.annotation.primary_model, config.annotation.schema_version
            )
            out = Path(output) if output else Path("exports/dpo_pairs.jsonl")
            count = export_dpo_pairs(pairs, out, config.annotation.schema_version)
            click.echo(f"Exported {count} DPO pairs to {out}")

        elif fmt == "audio":
            click.echo("Audio label export not yet implemented")
    finally:
        store.close()


@cli.command()
@click.option("--params", default="params.yaml", type=click.Path(exists=True))
def stats(params):
    """Print annotation progress and token usage statistics."""
    from pet_annotation.store import AnnotationStore

    config = load_config(Path(params))
    store = AnnotationStore(db_path=Path(config.database.path))
    try:
        status_counts = store.get_status_counts()
        model_stats = store.get_model_stats()
        click.echo("=== Frame Status ===")
        for row in status_counts:
            click.echo(f"  {row['annotation_status']}: {row['count']}")
        click.echo("\n=== Model Stats ===")
        for row in model_stats:
            click.echo(
                f"  {row['model_name']}: {row['annotation_count']} annotations, "
                f"{row['total_tokens_sum'] or 0} tokens"
            )
    finally:
        store.close()


if __name__ == "__main__":
    cli()
