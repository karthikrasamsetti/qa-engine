import sys
import asyncio

# Must be set before uvicorn imports asyncio internals or creates any event loop.
# --reload spawns the worker before app.main is imported, so the policy line in
# main.py runs too late.  Setting it here, at process start, guarantees the
# Proactor loop is used — which supports subprocess creation (needed by Playwright).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
