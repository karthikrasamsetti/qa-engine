"""FastAPI entrypoint.

Exposes:
  POST /runs                  -> start a graph run in the background, return run_id
  GET  /runs/{run_id}/stream  -> SSE stream of trajectory events
  POST /runs/{run_id}/resume  -> resume a HITL-paused run with human clarification
"""
from __future__ import annotations

import sys
import asyncio

# Playwright spawns subprocesses; the default SelectorEventLoop on Windows
# cannot do that.  Switch to ProactorEventLoop before anything else touches
# asyncio so uvicorn inherits the correct policy.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import logging
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.graph.builder import graph
from app.observability.audit import audit_log
from app.streaming.events import emitter

logging.basicConfig(level=logging.INFO)

# Wire the audit log into the emitter so every trajectory event is persisted
# to data/audit/<run_id>.jsonl without any changes to individual nodes.
emitter.set_audit(audit_log.write_event)

app = FastAPI(title="Intelligent QA Engine")

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


class ResumeRequest(BaseModel):
    response: str


async def _execute_run(run_id: str, raw_input: str, target_url: str) -> None:
    """Drive the graph and manage the event stream lifecycle.

    If the graph is interrupted for HITL the stream stays open — the resume
    endpoint will continue emitting events and close the stream when done.
    """
    config = {"configurable": {"thread_id": run_id}}
    initial: dict = {
        "run_id": run_id,
        "raw_input": raw_input,
        "target_url": target_url,
        "status": "running",
    }
    try:
        result = await graph.ainvoke(initial, config=config)
    except Exception as exc:
        logging.exception("run %s failed", run_id)
        await emitter.emit(run_id, "System", 0, "error", f"Run failed: {exc}")
        await emitter.close(run_id)
        return

    # LangGraph puts __interrupt__ in the result when a node called interrupt().
    # Leave the stream open — the resume endpoint will close it when the run ends.
    if isinstance(result, dict) and "__interrupt__" in result:
        return

    await emitter.close(run_id)


async def _execute_resume(run_id: str, hitl_response: str) -> None:
    """Resume a HITL-paused graph run with the human's clarification."""
    config = {"configurable": {"thread_id": run_id}}
    try:
        result = await graph.ainvoke(Command(resume=hitl_response), config=config)
    except Exception as exc:
        logging.exception("resume %s failed", run_id)
        await emitter.emit(run_id, "System", 0, "error", f"Resume failed: {exc}")
        await emitter.close(run_id)
        return

    if isinstance(result, dict) and "__interrupt__" in result:
        return  # another interrupt — stream stays open

    await emitter.close(run_id)


@app.post("/runs", response_model=RunResponse)
async def start_run(req: RunRequest) -> RunResponse:
    run_id = uuid.uuid4().hex
    asyncio.create_task(_execute_run(run_id, req.raw_input, req.target_url))
    return RunResponse(run_id=run_id)


@app.post("/runs/{run_id}/resume", response_model=RunResponse)
async def resume_run(run_id: str, req: ResumeRequest) -> RunResponse:
    asyncio.create_task(_execute_resume(run_id, req.response))
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


@app.get("/runs/{run_id}/cost")
async def get_run_cost(run_id: str) -> dict:
    from app.llm.cost_store import cost_store
    return {
        "run_id": run_id,
        "total_cost_usd": cost_store.get(run_id),
        "llm_calls": cost_store.calls(run_id),
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
