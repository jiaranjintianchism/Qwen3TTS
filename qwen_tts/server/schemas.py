from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    ok: bool = True
    model_loaded: bool
    ready: bool
    warmed_up: bool
    model_path: str
    model_type: str | None = None
    sample_rate: int | None = None
    inflight_requests: int = 0
    queued_requests: int = 0
    max_concurrency: int = 3
    max_queue_size: int = 0


class MetadataResponse(BaseModel):
    model_path: str
    model_type: str | None = None
    supported_languages: list[str] = Field(default_factory=list)
    supported_speakers: list[str] = Field(default_factory=list)
    sample_rate: int | None = None


class CustomVoiceRequest(BaseModel):
    text: str = Field(min_length=1)
    speaker: str = Field(min_length=1)
    language: str | None = None
    instruct: str | None = None
    audio_format: Literal["wav", "pcm16"] = "wav"

    @field_validator("text", "speaker")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty")
        return cleaned

    @field_validator("language", "instruct")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class StreamStartEvent(BaseModel):
    type: Literal["started"] = "started"
    sample_rate: int
    audio_format: Literal["pcm16"] = "pcm16"


class StreamAudioEvent(BaseModel):
    type: Literal["audio"] = "audio"
    seq: int
    data: str
    sample_rate: int


class StreamCompleteEvent(BaseModel):
    type: Literal["completed"] = "completed"
    chunks: int


class StreamErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    detail: str
