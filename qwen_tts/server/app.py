from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from .schemas import (
    CustomVoiceRequest,
    HealthResponse,
    StreamAudioEvent,
    StreamCompleteEvent,
    StreamErrorEvent,
    StreamStartEvent,
)
from .service import CustomVoiceService, ServerSettings


def create_app(settings: ServerSettings) -> FastAPI:
    service = CustomVoiceService(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await service.startup()
        yield

    app = FastAPI(title="Qwen3-TTS CustomVoice Server", version="0.2.0", lifespan=lifespan)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return await service.health()

    @app.get("/readyz", response_model=HealthResponse)
    async def readyz() -> HealthResponse:
        return await service.health()

    @app.get("/v1/metadata")
    async def metadata():
        return await service.metadata()

    @app.post("/v1/tts")
    async def tts(request: CustomVoiceRequest) -> Response:
        audio, media_type = await service.synthesize(request)
        return Response(content=audio, media_type=media_type)

    @app.websocket("/v1/tts/stream")
    async def tts_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            request = CustomVoiceRequest.model_validate(await websocket.receive_json())
            started = False
            async for event_type, payload in service.stream_chunks(request):
                if event_type == "audio":
                    if not started:
                        started = True
                        await websocket.send_json(
                            StreamStartEvent(sample_rate=payload["sample_rate"]).model_dump()
                        )
                    await websocket.send_json(StreamAudioEvent(**payload).model_dump())
                elif event_type == "completed":
                    await websocket.send_json(StreamCompleteEvent(chunks=payload["chunks"]).model_dump())
                elif event_type == "error":
                    await websocket.send_json(StreamErrorEvent(detail=str(payload)).model_dump())
                    break
        except WebSocketDisconnect:
            return
        except Exception as exc:  # noqa: BLE001
            await websocket.send_json(StreamErrorEvent(detail=str(exc)).model_dump())
        finally:
            await websocket.close()

    return app
