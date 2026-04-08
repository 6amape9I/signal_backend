from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


REQUIRED_STRING_FIELDS = (
    "record_id",
    "category_teacher_final",
    "title",
    "text_clean",
    "model_input",
)


class DatasetRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    record_id: str
    category_teacher_final: str
    title: str
    text_clean: str
    model_input: str

    project: str | None = None
    project_nick: str | None = None
    type: str | None = None
    category_teacher_raw: str | None = None
    body: str | None = None
    publish_date: str | None = None
    publish_date_t: int | float | None = None
    fronturl: str | None = None
    picture: str | None = None
    badge: Any | None = None

    @field_validator(*REQUIRED_STRING_FIELDS, mode="before")
    @classmethod
    def validate_required_string(cls, value: Any, info: Any) -> str:
        if not isinstance(value, str):
            raise ValueError(f"Field '{info.field_name}' must be a string.")

        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"Field '{info.field_name}' must not be empty.")

        return cleaned

