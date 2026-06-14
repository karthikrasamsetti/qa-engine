"""FastAPI entrypoint.

Exposes:
  POST /runs                  -> start a graph run in the background, return run_id
  GET  /runs/{run_id}/stream  -> SSE stream of trajectory events
  POST /runs/{run_id}/resume  -> (Stage B) resume a HITL-paused run

The SSE endpoint disables proxy buffering so events arrive live.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.graph.builder import graph
from app.streaming.events import emitter

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Intelligent QA Engine")

# Dev CORS: lock down before deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    raw_input: str
    target_url: str = ""


class RunResponse(BaseModel):
    run_id: str


async def _execute_run(run_id: str, raw_input: str, target_url: str) -> None:
    """Drive the graph and ensure the stream is closed when done."""
    config = {"configurable": {"thread_id": run_id}}
    initial: dict = {
        "run_id": run_id,
        "raw_input": raw_input,
        "target_url": target_url,
        "status": "running",
    }
    try:
        await graph.ainvoke(initial, config=config)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the stream
        logging.exception("run %s failed", run_id)
        await emitter.emit(run_id, "System", 0, "error", f"Run failed: {exc}")
    finally:
        await emitter.close(run_id)


@app.post("/runs", response_model=RunResponse)
async def start_run(req: RunRequest) -> RunResponse:
    run_id = uuid.uuid4().hex
    asyncio.create_task(_execute_run(run_id, req.raw_input, req.target_url))
    return RunResponse(run_id=run_id)


@app.get("/runs/{run_id}/stream")
async def stream_run(run_id: str) -> EventSourceResponse:
    async def event_generator():
        async for event in emitter.subscribe(run_id):
            yield {"event": event.type, "data": event.model_dump_json()}

    return EventSourceResponse(
        event_generator(),
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
