"""Per-run LLM cost accumulator (in-memory).

Each completed complete() call adds its cost here.  The GET /runs/{id}/cost
endpoint reads from this store.  Not persisted across restarts; swap to Redis
or Postgres for multi-worker / durable deployments.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _RunCost:
    total_usd: float = 0.0
    calls: int = 0


class CostStore:
    def __init__(self) -> None:
        self._data: dict[str, _RunCost] = {}

    def add(self, run_id: str, cost_usd: float) -> float:
        """Add cost for one LLM call; return the new running total."""
        if run_id not in self._data:
            self._data[run_id] = _RunCost()
        entry = self._data[run_id]
        entry.total_usd += cost_usd
        entry.calls += 1
        return entry.total_usd

    def get(self, run_id: str) -> float:
        """Return accumulated cost for a run (0.0 if unknown)."""
        return self._data.get(run_id, _RunCost()).total_usd

    def calls(self, run_id: str) -> int:
        return self._data.get(run_id, _RunCost()).calls


# Process-wide singleton.
cost_store = CostStore()
