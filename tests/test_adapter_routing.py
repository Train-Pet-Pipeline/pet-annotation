"""Tests for adapter.py annotator_type routing."""

from datetime import datetime

import pytest
from pet_schema import HumanAnnotation, LLMAnnotation

from pet_annotation.adapter import route_annotation_to_store


def _base(**kw):
    """Return base annotation kwargs."""
    return dict(
        annotation_id="a",
        target_id="t",
        annotator_id="x",
        modality="vision",
        schema_version="2.1.0",
        created_at=datetime(2026, 4, 21),
        storage_uri=None,
        **kw,
    )


@pytest.fixture
def store(tmp_path):
    """Fresh AnnotationStore with schema initialised."""
    from pet_annotation.store import AnnotationStore

    db = tmp_path / "ann.db"
    s = AnnotationStore(str(db))
    s.init_schema()
    return s


def test_route_llm(store):
    """LLMAnnotation routes to insert_llm."""
    ann = LLMAnnotation(**_base(prompt_hash="h", raw_response="r", parsed_output={}))
    route_annotation_to_store(ann, store)
    assert len(store.fetch_llm_by_target("t")) == 1


def test_route_human(store):
    """HumanAnnotation routes to insert_human."""
    ann = HumanAnnotation(
        annotation_id="a",
        target_id="t",
        annotator_id="alice",
        modality="vision",
        schema_version="2.1.0",
        created_at=datetime(2026, 4, 21),
        storage_uri=None,
        reviewer="alice",
        decision="accept",
        notes=None,
    )
    route_annotation_to_store(ann, store)
    assert len(store.fetch_human_by_target("t")) == 1


def test_route_unknown_type_fails(store):
    """Unknown annotator_type raises ValueError with diagnostic message."""

    class FakeAnn:
        annotator_type = "vlm"  # 旧名应 fail-fast

    with pytest.raises(ValueError, match="unknown annotator_type"):
        route_annotation_to_store(FakeAnn(), store)
