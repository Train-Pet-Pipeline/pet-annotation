"""Entry-point target for pet-infra's plugin discovery.

pet-infra scans ``[project.entry-points."pet_infra.plugins"]`` and calls the
registered callable (named ``register_all``, matching pet-infra's own
convention — see ``pet_infra._register``) at CLI startup to trigger the
``@DATASETS.register_module`` side-effects in plugin modules.
"""

from __future__ import annotations


def register_all() -> None:
    """Import (or reload) pet-annotation plugin modules to trigger registration side-effects."""
    import importlib

    from pet_annotation.datasets import audio_annotations, vision_annotations

    importlib.reload(vision_annotations)
    importlib.reload(audio_annotations)
