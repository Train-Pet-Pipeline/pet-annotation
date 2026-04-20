"""Label Studio labeling-config templates by modality."""

from __future__ import annotations

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent


def template_for(modality: str) -> str:
    """Return the labeling-config XML for the given modality.

    Args:
        modality: One of "vision" or "audio".

    Returns:
        The XML labeling config as a string.

    Raises:
        ValueError: If no template exists for the given modality.
    """
    path = _TEMPLATE_DIR / f"{modality}.xml"
    if not path.exists():
        raise ValueError(f"No LS template for modality={modality!r}")
    return path.read_text()
