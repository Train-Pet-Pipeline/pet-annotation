"""Tests for LS import dispatch by modality."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pet_annotation.human_review.import_to_ls import import_needs_review


@patch("pet_annotation.human_review.import_to_ls.template_for")
def test_import_needs_review_default_modality_uses_vision_template(
    mock_template_for, monkeypatch
):
    """Default modality is vision; template_for('vision') is consulted."""
    mock_template_for.return_value = "<View><Image/></View>"
    store = MagicMock()
    store.fetch_needs_review_annotations.return_value = []
    session = MagicMock()
    # GET /api/projects returns empty list → project creation path
    session.get.return_value.json.return_value = {"results": []}
    session.get.return_value.raise_for_status = MagicMock()
    session.post.return_value.json.return_value = {"id": 1}
    session.post.return_value.raise_for_status = MagicMock()
    session.post.return_value.ok = True

    import_needs_review(store, "http://ls", session, data_root="/data")

    mock_template_for.assert_called_with("vision")


@patch("pet_annotation.human_review.import_to_ls.template_for")
def test_import_needs_review_audio_raises_not_implemented(mock_template_for):
    """Audio modality is not yet wired; must raise NotImplementedError."""
    store = MagicMock()
    session = MagicMock()
    with pytest.raises(NotImplementedError, match="audio review flow"):
        import_needs_review(store, "http://ls", session, modality="audio")


@patch("pet_annotation.human_review.import_to_ls.template_for")
def test_import_needs_review_vision_modality_explicit_uses_vision_template(
    mock_template_for,
):
    """Explicit modality='vision' also uses vision template."""
    mock_template_for.return_value = "<View><Image/></View>"
    store = MagicMock()
    store.fetch_needs_review_annotations.return_value = []
    session = MagicMock()
    session.get.return_value.json.return_value = {"results": []}
    session.get.return_value.raise_for_status = MagicMock()
    session.post.return_value.json.return_value = {"id": 1}
    session.post.return_value.raise_for_status = MagicMock()
    session.post.return_value.ok = True

    import_needs_review(store, "http://ls", session, modality="vision")
    mock_template_for.assert_called_with("vision")


def test_import_needs_review_unknown_modality_raises_value_error():
    """Unknown modality should raise ValueError (from template_for or dispatch)."""
    store = MagicMock()
    session = MagicMock()
    with pytest.raises((ValueError, NotImplementedError)):
        import_needs_review(store, "http://ls", session, modality="sensor")
