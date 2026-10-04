"""One local browser conversation per model instance."""

from contextlib import asynccontextmanager
from pathlib import Path
import json
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def create_app(runtime):
    @asynccontextmanager
    async def lifespan(app):
        await runtime.load()
        yield
        await runtime.close()

    app = FastAPI(lifespan=lifespan)
    client = Path(__file__).parent / "client"
    app.mount("/static", StaticFiles(directory=client), name="static")

    @app.get("/")
    async def index():
        return FileResponse(client / "index.html")

    @app.get("/config")
    async def config():
        return runtime.metadata()

    @app.websocket("/stream")
    async def stream(socket: WebSocket):
        await socket.accept()
        session = None
        try:
            session = await runtime.create_session()
            await socket.send_json(dict(type="ready", **runtime.metadata()))
            while True:
                message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    if json.loads(message["text"]).get("type") == "stop":
                        break
                    raise ValueError("Unknown control message")
                payload = message.get("bytes")
                if payload is None or len(payload) > 64000:
                    raise ValueError("Expected at most two seconds of PCM16 per packet")
                started = time.perf_counter()
                batch = await session.push_pcm16(payload)
                processing_seconds = time.perf_counter() - started
                # Publish each native frame in order. The model has already
                # cleared unplayed speech on an interrupt.
                for pcm, event in zip(batch.pcm_frames, batch.frame_events, strict=True):
                    await socket.send_json({**event, "type": "event"})
                    await socket.send_bytes(pcm)
                await socket.send_json(
                    dict(
                        type="ack", samples=len(payload) // 2, processing_seconds=processing_seconds
                    )
                )
        except WebSocketDisconnect:
            pass
        except Exception as error:
            try:
                await socket.send_json(dict(type="error", message=str(error)))
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            if session is not None:
                await session.close()
            try:
                await socket.close()
            except (RuntimeError, WebSocketDisconnect):
                pass

    return app
