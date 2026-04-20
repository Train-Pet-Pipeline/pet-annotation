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
@click.option(
    "--modality",
    type=click.Choice(["vision", "audio"]),
    default="vision",
    show_default=True,
    help="Annotation modality",
)
def annotate(batch_size, dry_run, params, modality):
    """Batch annotate pending frames using configured models."""
    if modality == "audio":
        raise click.ClickException("Audio pipeline not yet implemented for this command")

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
@click.option(
    "--modality",
    type=click.Choice(["vision", "audio"]),
    default="vision",
    show_default=True,
    help="Annotation modality",
)
def check(params, modality):
    """Run quality check on auto_checked annotations."""
    if modality == "audio":
        raise click.ClickException("Audio pipeline not yet implemented for this command")

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
@click.option(
    "--modality",
    type=click.Choice(["vision", "audio"]),
    default="vision",
    show_default=True,
    help="Annotation modality",
)
def export_cmd(fmt, output, params, modality):
    """Export training data in the specified format."""
    # Validate format/modality consistency
    if fmt == "sft" and modality == "audio":
        raise click.ClickException("SFT export is vision-only; use --modality=vision")
    if fmt == "dpo" and modality == "audio":
        raise click.ClickException("DPO audio export is not yet implemented; use --modality=vision")
    if fmt == "audio" and modality == "vision":
        raise click.ClickException(
            "Format/modality mismatch: --format=audio requires --modality=audio"
        )

    if fmt == "audio" and modality == "audio":
        from pet_annotation.export.to_audio_labels import export_audio_labels
        from pet_annotation.store import AnnotationStore
        config = load_config(Path(params))
        store = AnnotationStore(db_path=Path(config.database.path))
        out = Path(output) if output else Path("exports/audio_labels.jsonl")
        try:
            count = export_audio_labels(store, out)
            click.echo(f"Exported {count} audio labels to {out}")
        finally:
            store.close()
        return

    from pet_annotation.store import AnnotationStore

    config = load_config(Path(params))
    store = AnnotationStore(db_path=Path(config.database.path))

    try:
        if fmt == "sft":
            from pet_annotation.export.to_sharegpt import export_sharegpt
            out = Path(output) if output else Path("exports/sft_train.jsonl")
            count = export_sharegpt(
                store, out, config.annotation.schema_version,
                data_root=config.database.data_root,
            )
            click.echo(f"Exported {count} SFT records to {out}")

        elif fmt == "dpo":
            from pet_annotation.dpo.generate_pairs import generate_cross_model_pairs
            from pet_annotation.export.to_dpo_pairs import export_dpo_pairs
            pairs = generate_cross_model_pairs(
                store, config.annotation.primary_model, config.annotation.schema_version
            )
            out = Path(output) if output else Path("exports/dpo_pairs.jsonl")
            count = export_dpo_pairs(
                pairs, out, config.annotation.schema_version,
                data_root=config.database.data_root,
            )
            click.echo(f"Exported {count} DPO pairs to {out}")
    finally:
        store.close()


@cli.command(name="ls-import")
@click.option("--ls-url", default="http://localhost:8080", help="Label Studio URL")
@click.option("--ls-key", envvar="LABEL_STUDIO_API_KEY", default=None, help="Label Studio API key")
@click.option("--ls-email", envvar="LABEL_STUDIO_ADMIN_EMAIL", default=None, help="LS admin email")
@click.option(
    "--ls-password", envvar="LABEL_STUDIO_ADMIN_PASSWORD",
    default=None, help="LS admin password",
)
@click.option("--params", default="params.yaml", type=click.Path(exists=True))
@click.option(
    "--modality",
    type=click.Choice(["vision", "audio"]),
    default="vision",
    show_default=True,
    help="Annotation modality",
)
def ls_import(ls_url, ls_key, ls_email, ls_password, params, modality):
    """Import needs_review annotations into Label Studio for human review."""
    from pet_annotation.human_review.import_to_ls import import_needs_review
    from pet_annotation.human_review.ls_auth import get_ls_session
    from pet_annotation.store import AnnotationStore

    if not ls_key and not (ls_email and ls_password):
        click.echo(
            "Error: provide --ls-key or both --ls-email and --ls-password "
            "(or set LABEL_STUDIO_API_KEY / LABEL_STUDIO_ADMIN_EMAIL + "
            "LABEL_STUDIO_ADMIN_PASSWORD env vars)",
            err=True,
        )
        raise SystemExit(1)

    session = get_ls_session(ls_url, api_key=ls_key, email=ls_email, password=ls_password)

    config = load_config(Path(params))
    store = AnnotationStore(db_path=Path(config.database.path))
    try:
        try:
            count = import_needs_review(
                store, ls_url, session,
                data_root=config.database.data_root,
                modality=modality,
            )
        except NotImplementedError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Imported {count} tasks into Label Studio")
    finally:
        store.close()


@cli.command(name="ls-export")
@click.option("--ls-url", default="http://localhost:8080", help="Label Studio URL")
@click.option("--ls-key", envvar="LABEL_STUDIO_API_KEY", default=None, help="Label Studio API key")
@click.option("--ls-email", envvar="LABEL_STUDIO_ADMIN_EMAIL", default=None, help="LS admin email")
@click.option(
    "--ls-password", envvar="LABEL_STUDIO_ADMIN_PASSWORD",
    default=None, help="LS admin password",
)
@click.option("--params", default="params.yaml", type=click.Path(exists=True))
@click.option(
    "--modality",
    type=click.Choice(["vision", "audio"]),
    default="vision",
    show_default=True,
    help="Annotation modality",
)
def ls_export(ls_url, ls_key, ls_email, ls_password, params, modality):
    """Pull reviewed annotations from Label Studio back to DB."""
    from pet_annotation.human_review.export_from_ls import export_reviewed
    from pet_annotation.human_review.ls_auth import get_ls_session
    from pet_annotation.store import AnnotationStore

    if modality == "audio":
        raise click.ClickException("Audio pipeline not yet implemented for this command")

    if not ls_key and not (ls_email and ls_password):
        click.echo(
            "Error: provide --ls-key or both --ls-email and --ls-password "
            "(or set LABEL_STUDIO_API_KEY / LABEL_STUDIO_ADMIN_EMAIL + "
            "LABEL_STUDIO_ADMIN_PASSWORD env vars)",
            err=True,
        )
        raise SystemExit(1)

    session = get_ls_session(ls_url, api_key=ls_key, email=ls_email, password=ls_password)

    config = load_config(Path(params))
    store = AnnotationStore(db_path=Path(config.database.path))
    try:
        count = export_reviewed(store, ls_url, session)
        click.echo(f"Updated {count} annotations from Label Studio")
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
