import argparse
import asyncio
import contextlib
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    TypeAdapter,
    ValidationError,
)

from .game import SCENARIOS, DoomSession


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Keys(Command):
    type: Literal["keys"]
    keys: list[
        Literal[
            "KeyW", "KeyS", "KeyA", "KeyD", "ArrowLeft", "ArrowRight", "Space", "KeyE"
        ]
    ] = Field(max_length=8)


class Mode(Command):
    type: Literal["mode"]
    value: Literal["manual", "copilot", "autopilot"]


class Pause(Command):
    type: Literal["pause"]
    value: StrictBool


class Scenario(Command):
    type: Literal["scenario"]
    value: Literal["defend_the_center", "basic", "deadly_corridor"]


class Settings(Command):
    type: Literal["settings"]
    hz: float = Field(ge=1, le=10)
    tics: int = Field(strict=True, ge=1, le=12)


class Simple(Command):
    type: Literal["reset", "step", "takeover", "claim"]


COMMAND = TypeAdapter(
    Annotated[
        Keys | Mode | Pause | Scenario | Settings | Simple,
        Field(discriminator="type"),
    ]
)


def create_app(session: DoomSession):
    @asynccontextmanager
    async def lifespan(app):
        session.start()
        yield
        session.close()

    app = FastAPI(title="Doom / typed controls", lifespan=lifespan)
    app.state.controller = None
    static = Path(__file__).with_name("static")

    @app.get("/")
    async def index():
        return FileResponse(
            static / "index.html", headers={"Cache-Control": "no-store"}
        )

    @app.get("/app.js")
    async def javascript():
        return FileResponse(
            static / "app.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/style.css")
    async def stylesheet():
        return FileResponse(static / "style.css", media_type="text/css")

    @app.get("/api/state")
    async def state():
        payload, _ = session.snapshot()
        return payload

    @app.get("/api/frame")
    async def frame():
        _, jpeg = session.snapshot()
        return Response(
            jpeg or b"",
            media_type="image/jpeg",
            status_code=200 if jpeg else 503,
            headers={"Cache-Control": "no-store"},
        )

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        origin = ws.headers.get("origin")
        if origin and urlsplit(origin).netloc != ws.headers.get("host"):
            await ws.close(code=1008, reason="Use the app from the same origin")
            return
        await ws.accept()
        client_id = str(uuid.uuid4())
        if app.state.controller is None:
            app.state.controller = client_id
        send_lock = asyncio.Lock()

        async def send_json(value):
            async with send_lock:
                await ws.send_json(value)

        async def publish():
            last_frame = -1
            while True:
                payload, jpeg = session.snapshot()
                payload = {
                    **payload,
                    "type": "state",
                    "is_controller": app.state.controller == client_id,
                }
                async with send_lock:
                    await ws.send_json(payload)
                    if jpeg is not None and payload.get("frame_id") != last_frame:
                        # Each binary JPEG is preceded by state with its frame id.
                        await ws.send_bytes(jpeg)
                        last_frame = payload.get("frame_id")
                await asyncio.sleep(0.05)

        publisher = asyncio.create_task(publish())
        try:
            while True:
                raw = await ws.receive_text()
                if len(raw) > 2048:
                    await send_json({"type": "error", "message": "Command too large"})
                    continue
                try:
                    command = COMMAND.validate_json(raw).model_dump()
                except ValidationError:
                    await send_json({"type": "error", "message": "Invalid command"})
                    continue
                if command["type"] == "claim":
                    app.state.controller = client_id
                    command = {"type": "takeover"}
                if app.state.controller != client_id:
                    await send_json(
                        {
                            "type": "error",
                            "message": "This tab is spectating. Take control first.",
                        }
                    )
                    continue
                if not session.command(command):
                    await send_json(
                        {"type": "error", "message": "Controller is busy; try again"}
                    )
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            publisher.cancel()
            with contextlib.suppress(
                asyncio.CancelledError, WebSocketDisconnect, RuntimeError
            ):
                await publisher
            if app.state.controller == client_id:
                app.state.controller = None
                session.command({"type": "disconnect"})

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Play Doom in the browser with typed Gemma controls"
    )
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--scenario", choices=tuple(SCENARIOS), default="defend_the_center"
    )
    parser.add_argument(
        "--no-model", action="store_true", help="manual-only mode without loading CUDA"
    )
    parser.add_argument(
        "--no-compile", action="store_true",
        help="use the eager decoder instead of the default compiled decoder",
    )
    args = parser.parse_args()
    if not args.no_model:
        from ..runtime import configure_gpu

        configure_gpu(args.gpu)
    import uvicorn

    session = DoomSession(
        gpu=args.gpu, model_enabled=not args.no_model, scenario=args.scenario,
        compile_decoder=not args.no_compile,
    )
    uvicorn.run(
        create_app(session),
        host=args.host,
        port=args.port,
        ws_max_size=4096,
        log_level="info",
    )


if __name__ == "__main__":
    main()
