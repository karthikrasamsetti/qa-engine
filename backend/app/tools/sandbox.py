"""Docker-based Playwright script runner.

Pipes the generated script into an isolated container via stdin — no host bind
mounts — so path handling is identical on Linux, Mac, and Windows Docker Desktop.

Graceful degradation: if Docker is unreachable the function returns a structured
error result rather than raising; callers receive {passed: False, error: "..."}.
"""
from __future__ import annotations

import asyncio
import logging
import re

logger = logging.getLogger(__name__)

# Playwright Python image shipped with Chromium and all browsers pre-installed.
_IMAGE = "mcr.microsoft.com/playwright/python:v1.61.0-noble"
_DEFAULT_TIMEOUT = 120  # seconds

# ---------------------------------------------------------------------------
# URL substitution — fix scaffolder-invented placeholder addresses
# ---------------------------------------------------------------------------

# Matches page.goto('<url>') when <url> looks like a placeholder:
#   example.com, SOME_URL, ALL_CAPS_URL, localhost:NNNN
_GOTO_PLACEHOLDER_RE = re.compile(
    r"""(page\.goto\(\s*['"])"""
    r"""((?:https?://)?"""
    r"""(?:example\.com|[A-Z][A-Z0-9_]{2,}URL[A-Z0-9_]*|[A-Z_]{4,}|localhost:\d{2,5})"""
    r"""(?:/[^'"]*)?)"""
    r"""(['"])""",
)

# Rewrites localhost / 127.0.0.1 to the Docker-Desktop host gateway so a
# script running inside a container can reach services on the host machine.
_LOCALHOST_RE = re.compile(r"(https?://)(?:localhost|127\.0\.0\.1)(:\d+)?")


def _rewrite_for_container(url: str) -> str:
    """Return *url* with localhost/127.0.0.1 replaced by host.docker.internal.

    Only affects the URL used inside the Docker container — the original
    target_url stored in state is never modified.
    """
    if not url:
        return url
    return _LOCALHOST_RE.sub(
        lambda m: f"{m.group(1)}host.docker.internal{m.group(2) or ''}",
        url,
    )


def substitute_url(script: str, target_url: str) -> str:
    """Replace scaffolder-invented placeholder URLs in page.goto() calls."""
    if not target_url:
        return script
    return _GOTO_PLACEHOLDER_RE.sub(
        lambda m: f"{m.group(1)}{target_url}{m.group(3)}",
        script,
    )


# ---------------------------------------------------------------------------
# Failure classification — locator error vs genuine assertion failure
# ---------------------------------------------------------------------------

# These match Playwright-level "can't find / can't reach the element" failures.
_LOCATOR_PATTERNS: list[re.Pattern] = [
    re.compile(r"TimeoutError", re.IGNORECASE),
    re.compile(r"strict mode violation", re.IGNORECASE),
    re.compile(r"resolved to \d+ elements", re.IGNORECASE),
    re.compile(r"ElementHandle.*not attached", re.IGNORECASE),
    re.compile(r"Element is not attached", re.IGNORECASE),
    re.compile(r"locator\.(click|fill|type|select_option|check|uncheck|hover|press)\b", re.IGNORECASE),
    re.compile(r"Error: page\.locator", re.IGNORECASE),
    re.compile(r"Target closed", re.IGNORECASE),
]

# These match test-logic failures that self-heal must NOT attempt to fix.
_ASSERTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"AssertionError", re.IGNORECASE),
    re.compile(r"Locator expected", re.IGNORECASE),  # expect().to_... messages
    re.compile(r"Expected.*to have", re.IGNORECASE),
    re.compile(r"Expected.*Received", re.IGNORECASE),
]


def is_locator_failure(text: str) -> bool:
    """True when *text* describes a locator/selector fault rather than a test assertion.

    Assertion failures (wrong values, wrong URLs) are genuine bugs; self-heal
    must leave them untouched.  Locator failures (element not found, ambiguous
    selector) are selector staleness and can be healed automatically.
    """
    if any(p.search(text) for p in _ASSERTION_PATTERNS):
        return False
    return any(p.search(text) for p in _LOCATOR_PATTERNS)


def is_assertion_failure(text: str) -> bool:
    """True when *text* contains an explicit test assertion failure.

    Distinct from is_locator_failure: a sandbox-level timeout or Docker error
    is neither — callers should handle the three cases separately rather than
    treating 'not locator' as 'assertion'.
    """
    return any(p.search(text) for p in _ASSERTION_PATTERNS)


# ---------------------------------------------------------------------------
# Docker availability check
# ---------------------------------------------------------------------------

async def _docker_available() -> bool:
    """Return True when the Docker CLI responds to 'docker version'."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "version",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        return proc.returncode == 0
    except (FileNotFoundError, asyncio.TimeoutError, OSError):
        return False


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

async def run_script(
    script: str,
    target_url: str = "",
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict:
    """Run *script* in an isolated Playwright Docker container.

    Substitutes placeholder URLs, pipes the script via stdin (no bind mounts),
    captures stdout+stderr, and returns::

        {
            "passed":      bool,
            "logs":        str,   # full pytest output
            "error":       str | None,   # extracted failure fragment
            "screenshots": list,  # future: base64-encoded PNGs
            "exit_code":   int,
        }
    """
    if not await _docker_available():
        msg = (
            "Docker is not available on this host. "
            "Install Docker Desktop and ensure the daemon is running."
        )
        logger.error("Sandbox: %s", msg)
        return {"passed": False, "logs": "", "error": msg, "screenshots": [], "exit_code": -1}

    # Rewrite localhost/127.0.0.1 → host.docker.internal so the script can
    # reach host services from inside the container.  Two passes:
    #   1. substitute_url() replaces scaffolder placeholder URLs in page.goto() calls.
    #   2. _rewrite_for_container() rewrites the full script text so assertions like
    #      expect(page).to_have_url("http://localhost:PORT/…") also use the container
    #      host — otherwise navigation and assertion use different hostnames and the
    #      test always fails even when the page loaded correctly.
    # The original target_url stored in state is never modified.
    container_url = _rewrite_for_container(target_url)
    patched = substitute_url(script, container_url)
    patched = _rewrite_for_container(patched)

    # The bash pipeline:
    #   1. pip install deps (quiet, stderr suppressed — already in image but
    #      pytest-playwright may not be; adding takes ~5 s on cache hit)
    #   2. cat > /script.py  ← reads script from container stdin
    #   3. pytest runs from the written file
    bash_cmd = (
        "pip install pytest pytest-playwright -q 2>/dev/null"
        " && cat > /script.py"
        " && python -m pytest /script.py -v --tb=short --no-header 2>&1"
    )
    cmd = [
        "docker", "run", "--rm", "-i",
        "--network", "bridge",   # default; use host network only when explicitly needed
        "--memory", "512m",
        "--cpus", "1",
        "--security-opt", "no-new-privileges",
        _IMAGE,
        "bash", "-c", bash_cmd,
    ]

    logger.info("Sandbox: spawning container (timeout=%ds, url=%r)", timeout, target_url)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except (FileNotFoundError, OSError) as exc:
        logger.error("Sandbox: container launch failed — %s", exc)
        return {
            "passed": False, "logs": "", "screenshots": [], "exit_code": -1,
            "error": f"Container launch failed: {exc}",
        }

    try:
        stdout_bytes, _ = await asyncio.wait_for(
            proc.communicate(input=patched.encode()),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        msg = f"Sandbox timed out after {timeout}s."
        logger.warning("Sandbox: %s", msg)
        return {"passed": False, "logs": "", "error": msg, "screenshots": [], "exit_code": -1}

    logs = stdout_bytes.decode(errors="replace")
    passed = proc.returncode == 0
    error = None if passed else _extract_error(logs)

    logger.info("Sandbox: exit_code=%d passed=%s", proc.returncode, passed)
    return {
        "passed": passed,
        "logs": logs,
        "error": error,
        "screenshots": [],   # future: mount a volume and collect PNGs here
        "exit_code": proc.returncode,
    }


def _extract_error(logs: str) -> str:
    """Return the most useful failure fragment from pytest output (≤ 60 lines)."""
    lines = logs.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("FAILED") or "Error" in line or "assert" in line.lower():
            return "\n".join(lines[i: i + 60])
    return logs[-3000:] if len(logs) > 3000 else logs
