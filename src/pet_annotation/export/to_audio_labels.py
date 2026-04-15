"""Export audio classification labels for the audio CNN training pipeline."""
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
    """
    raise NotImplementedError("Audio labeling pipeline not yet defined")
