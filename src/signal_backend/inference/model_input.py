from __future__ import annotations


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def build_model_input(
    title: str | None,
    text: str | None,
    model_input: str | None,
) -> str:
    direct_input = _normalize_text(model_input)
    if direct_input is not None:
        return direct_input

    normalized_title = _normalize_text(title)
    normalized_text = _normalize_text(text)
    if normalized_title and normalized_text:
        return f"{normalized_title}\n\n{normalized_text}"
    if normalized_title:
        return normalized_title
    if normalized_text:
        return normalized_text
    raise ValueError("Provide non-empty 'model_input' or at least one of 'title'/'text'.")