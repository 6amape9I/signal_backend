from __future__ import annotations

import pytest

from signal_backend.inference.model_input import build_model_input


def test_build_model_input_prefers_explicit_model_input() -> None:
    assert build_model_input("Title", "Text", "  Direct input  ") == "Direct input"


def test_build_model_input_combines_title_and_text() -> None:
    assert build_model_input("Title", "Text", None) == "Title\n\nText"


def test_build_model_input_uses_title_only_when_text_missing() -> None:
    assert build_model_input("Title", "   ", None) == "Title"


def test_build_model_input_uses_text_only_when_title_missing() -> None:
    assert build_model_input("   ", "Text", None) == "Text"


def test_build_model_input_raises_for_empty_payload() -> None:
    with pytest.raises(ValueError, match="Provide non-empty"):
        build_model_input("  ", None, "   ")