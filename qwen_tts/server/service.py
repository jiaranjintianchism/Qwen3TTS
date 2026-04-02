from __future__ import annotations

import asyncio
import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import soundfile as sf
import torch
from fastapi import HTTPException

from qwen_tts import Qwen3TTSModel

from .schemas import CustomVoiceRequest, HealthResponse, MetadataResponse


def _dtype_from_str(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _float_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    clipped = np.clip(mono, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    return pcm.tobytes()


def _float_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    buffer = io.BytesIO()
    sf.write(buffer, mono, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@dataclass(slots=True)
class ServerSettings:
    model_path: str
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    flash_attn: bool = True
    max_workers: int = 2
    max_concurrency: int = 1
    use_compile: bool = False
    compile_mode: str = "reduce-overhead"
    decode_window_frames: int = 80
    emit_every_frames: int = 8
    overlap_samples: int = 0
    preload_model: bool = True
    warmup_enabled: bool = True
    warmup_text: str = "Hello, this is a warmup request."
    warmup_speaker: str = "Ryan"
    max_queue_size: int = 8


class CustomVoiceService:
    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self._model: Qwen3TTSModel | None = None
        self._load_lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._state_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_workers,
            thread_name_prefix="qwen3-customvoice-server",
        )
        self._inflight_requests = 0
        self._queued_requests = 0
        self._warmed_up = False
        self._last_load_error: str | None = None
        self._startup_ms: float | None = None

    def _load_model(self) -> Qwen3TTSModel:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            model = Qwen3TTSModel.from_pretrained(
                self.settings.model_path,
                device_map=self.settings.device,
                dtype=_dtype_from_str(self.settings.dtype),
                attn_implementation="flash_attention_2" if self.settings.flash_attn else "eager",
            )
            model_type = getattr(model.model, "tts_model_type", None)
            if model_type != "custom_voice":
                raise RuntimeError(
                    f"CustomVoice server expects a custom_voice model, got tts_model_type={model_type!r}"
                )
            if self.settings.use_compile:
                model.enable_streaming_optimizations(
                    decode_window_frames=self.settings.decode_window_frames,
                    use_compile=True,
                    compile_mode=self.settings.compile_mode,
                )
            self._model = model
            self._last_load_error = None
            return model

    async def ensure_model(self) -> Qwen3TTSModel:
        try:
            return await asyncio.to_thread(self._load_model)
        except Exception as exc:  # noqa: BLE001
            self._last_load_error = str(exc)
            raise

    async def startup(self) -> None:
        if not self.settings.preload_model:
            return
        start = perf_counter()
        model = await self.ensure_model()
        self._startup_ms = (perf_counter() - start) * 1000.0
        if self.settings.warmup_enabled:
            await asyncio.to_thread(
                model.generate_custom_voice,
                self.settings.warmup_text,
                self.settings.warmup_speaker,
                None,
                None,
            )
            self._warmed_up = True

    async def health(self) -> HealthResponse:
        model = self._model
        ready = model is not None and (self._warmed_up or not self.settings.warmup_enabled)
        return HealthResponse(
            model_loaded=model is not None,
            ready=ready,
            warmed_up=self._warmed_up,
            model_path=self.settings.model_path,
            model_type=getattr(model.model, "tts_model_type", None) if model is not None else None,
            sample_rate=getattr(model.model, "sample_rate", None) if model is not None else None,
            inflight_requests=self._inflight_requests,
            queued_requests=self._queued_requests,
            max_concurrency=self.settings.max_concurrency,
            max_queue_size=self.settings.max_queue_size,
        )

    async def metadata(self) -> MetadataResponse:
        model = await self.ensure_model()
        return MetadataResponse(
            model_path=self.settings.model_path,
            model_type=getattr(model.model, "tts_model_type", None),
            supported_languages=model.get_supported_languages() or [],
            supported_speakers=model.get_supported_speakers() or [],
            sample_rate=getattr(model.model, "sample_rate", None),
        )

    async def _acquire_slot(self) -> None:
        async with self._state_lock:
            if self._queued_requests >= self.settings.max_queue_size:
                raise HTTPException(status_code=429, detail="Server is busy, queue is full")
            self._queued_requests += 1
        try:
            await self._semaphore.acquire()
        except Exception:
            async with self._state_lock:
                self._queued_requests -= 1
            raise
        async with self._state_lock:
            self._queued_requests -= 1
            self._inflight_requests += 1

    async def _release_slot(self) -> None:
        async with self._state_lock:
            self._inflight_requests = max(0, self._inflight_requests - 1)
        self._semaphore.release()

    async def synthesize(self, request: CustomVoiceRequest) -> tuple[bytes, str]:
        await self._acquire_slot()
        try:
            model = await self.ensure_model()
            wavs, sample_rate = await asyncio.to_thread(
                model.generate_custom_voice,
                request.text,
                request.speaker,
                request.language,
                request.instruct,
            )
            audio = wavs[0]
            if request.audio_format == "pcm16":
                return _float_to_pcm16_bytes(audio), "audio/pcm"
            return _float_to_wav_bytes(audio, sample_rate), "audio/wav"
        finally:
            await self._release_slot()

    async def stream_chunks(self, request: CustomVoiceRequest):
        await self._acquire_slot()
        try:
            model = await self.ensure_model()
            queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue(maxsize=8)
            loop = asyncio.get_running_loop()

            def worker() -> None:
                try:
                    seq = 0
                    for chunk, sr in model.stream_generate_custom_voice(
                        text=request.text,
                        speaker=request.speaker,
                        language=request.language,
                        instruct=request.instruct,
                        emit_every_frames=self.settings.emit_every_frames,
                        decode_window_frames=self.settings.decode_window_frames,
                        overlap_samples=self.settings.overlap_samples,
                    ):
                        seq += 1
                        payload = {
                            "seq": seq,
                            "data": base64.b64encode(_float_to_pcm16_bytes(chunk)).decode("ascii"),
                            "sample_rate": sr,
                        }
                        asyncio.run_coroutine_threadsafe(queue.put(("audio", payload)), loop).result()
                    asyncio.run_coroutine_threadsafe(queue.put(("completed", {"chunks": seq})), loop).result()
                except Exception as exc:  # noqa: BLE001
                    asyncio.run_coroutine_threadsafe(queue.put(("error", str(exc))), loop).result()
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop).result()

            future = loop.run_in_executor(self._executor, worker)
            try:
                while True:
                    event_type, payload = await queue.get()
                    if event_type == "done":
                        break
                    yield event_type, payload
            finally:
                await future
        finally:
            await self._release_slot()
