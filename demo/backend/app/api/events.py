"""GET /events: Server-Sent Events stream of well changes."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api import deps
from app.services.events import Broadcaster

router = APIRouter()
KEEPALIVE_S = 15.0


def frame(event: dict) -> str:
    """One SSE frame: the event name and a JSON payload."""
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


@router.get("/events")
async def stream(request: Request, events: Broadcaster = Depends(deps.events)) -> StreamingResponse:
    """Emits well_created, well_located, well_ready, well_failed and well_scored as they happen (`: keep-alive` every 15 s).

    Events are not replayed: a client that connects late reads the current state from GET /wells.
    """
    queue = events.subscribe()

    async def gen():
        try:
            yield ": connected\n\n"
            while not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(queue.get(), KEEPALIVE_S)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield frame(event)
        finally:
            events.unsubscribe(queue)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})
