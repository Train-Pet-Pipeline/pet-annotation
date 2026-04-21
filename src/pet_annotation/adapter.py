"""Route annotations to the correct store method by annotator_type — not modality."""

from __future__ import annotations

from pet_schema import (  # noqa: F401
    Annotation,
    ClassifierAnnotation,
    HumanAnnotation,
    LLMAnnotation,
    RuleAnnotation,
)

_ROUTES = {
    "llm": "insert_llm",
    "classifier": "insert_classifier",
    "rule": "insert_rule",
    "human": "insert_human",
}


def route_annotation_to_store(ann, store) -> None:
    """Dispatch *ann* to the matching store insert method based on annotator_type.

    Args:
        ann: An Annotation object with an ``annotator_type`` attribute.
        store: An AnnotationStore instance.

    Raises:
        ValueError: If ``ann.annotator_type`` is not one of the 4 known paradigms.
    """
    t = getattr(ann, "annotator_type", None)
    if not isinstance(t, str) or t not in _ROUTES:
        raise ValueError(f"unknown annotator_type: {t!r} (valid: {list(_ROUTES)})")
    getattr(store, _ROUTES[t])(ann)
