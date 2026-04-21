"""Entry-point target for pet-infra's plugin discovery.

pet-infra scans ``[project.entry-points."pet_infra.plugins"]`` and calls the
registered callable (named ``register_all``, matching pet-infra's own
convention — see ``pet_infra._register``) at CLI startup to trigger the
``@DATASETS.register_module`` side-effects in plugin modules.
"""

from __future__ import annotations

try:
    import pet_infra  # noqa: F401
except ImportError as e:
    raise ImportError(
        "pet-annotation requires pet-infra to be installed first. "
        "Install via 'pip install pet-infra @ "
        "git+https://github.com/Train-Pet-Pipeline/pet-infra@<tag>' "
        "using the tag pinned in pet-infra/docs/compatibility_matrix.yaml."
    ) from e


def register_all() -> None:
    """Import pet-annotation plugin modules to trigger registration side-effects."""
    from pet_annotation.datasets import (
        classifier_annotations,  # noqa: F401
        human_annotations,  # noqa: F401
        llm_annotations,  # noqa: F401
        rule_annotations,  # noqa: F401
    )
