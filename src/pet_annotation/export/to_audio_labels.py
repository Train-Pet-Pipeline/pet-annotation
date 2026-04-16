"""Export audio classification labels for the audio CNN training pipeline.

Post-launch feature: requires audio segment annotations (not part of VLM
annotation pipeline). Will export audio clip paths with their classification
labels in CSV format for pet-train audio CNN training.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_audio_labels(output_path: Path) -> int:
    """Export audio labels to CSV format.

    Args:
        output_path: Path to write the labels file.

    Returns:
        Number of labels exported.

    Raises:
        NotImplementedError: Audio annotation pipeline is a post-launch feature.
    """
    raise NotImplementedError(
        "Audio labeling export is a post-launch feature. "
        "The audio CNN training pipeline currently uses pre-labeled datasets "
        "from pet-data, not VLM-generated audio annotations. "
        "See: https://github.com/Train-Pet-Pipeline/pet-annotation/issues "
        "for tracking."
    )
