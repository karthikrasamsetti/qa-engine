// Mirrors the backend's frozen contracts: TrajectoryEvent + run lifecycle.
// Keep these field names in sync with app/streaming/events.py.

export type EventType =
  | "thought"
  | "action"
  | "tool_call"
  | "tool_result"
  | "decision"
  | "error"
  | "hitl_request"
  | "complete";

export interface TrajectoryEvent {
  run_id: string;
  ts: number;
  agent: string;
  phase: number;
  type: EventType;
  message: string;
  data: Record<string, unknown>;
}

export type RunStatus = "idle" | "running" | "paused_hitl" | "done" | "failed";

export interface StartRunResponse {
  run_id: string;
}

// --- API calls (same-origin via Vite proxy in dev) ---

export async function startRun(
  rawInput: string,
  targetUrl: string
): Promise<string> {
  const res = await fetch("/runs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ raw_input: rawInput, target_url: targetUrl }),
  });
  if (!res.ok) throw new Error(`startRun failed: ${res.status}`);
  const json = (await res.json()) as StartRunResponse;
  return json.run_id;
}

export async function resumeRun(
  runId: string,
  response: string
): Promise<void> {
  // NOTE: backend resume field is `response` (not `hitl_response`).
  const res = await fetch(`/runs/${runId}/resume`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ response }),
  });
  if (!res.ok) throw new Error(`resumeRun failed: ${res.status}`);
}

export interface CostSummary {
  total_cost_usd: number;
  llm_calls: number;
}

export async function fetchCost(runId: string): Promise<CostSummary | null> {
  try {
    const res = await fetch(`/runs/${runId}/cost`);
    if (!res.ok) return null;
    return (await res.json()) as CostSummary;
  } catch {
    return null;
  }
}
