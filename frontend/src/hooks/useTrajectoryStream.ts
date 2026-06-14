import { useCallback, useEffect, useRef, useState } from "react";
import type { RunStatus, TrajectoryEvent } from "../lib/api";

interface StreamState {
  events: TrajectoryEvent[];
  status: RunStatus;
  hitlEvent: TrajectoryEvent | null; // latest unanswered hitl_request
}

/**
 * Opens an SSE connection to /runs/{runId}/stream and accumulates events.
 *
 * The backend emits named SSE events (event: thought, event: decision, …) whose
 * `data` is the JSON-serialized TrajectoryEvent. We listen with a generic
 * handler across all known types so one parser covers them all.
 *
 * Reconnect: EventSource auto-reconnects on transient drops. We only hard-close
 * on a terminal event (complete/failed) or when the caller resets.
 */
export function useTrajectoryStream(runId: string | null) {
  const [state, setState] = useState<StreamState>({
    events: [],
    status: "idle",
    hitlEvent: null,
  });
  const esRef = useRef<EventSource | null>(null);

  const reset = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
    setState({ events: [], status: "idle", hitlEvent: null });
  }, []);

  // Called by the UI after a successful resume so the HITL box disappears
  // and the run shows as running again until the next event arrives.
  const clearHitl = useCallback(() => {
    setState((s) => ({ ...s, hitlEvent: null, status: "running" }));
  }, []);

  useEffect(() => {
    if (!runId) return;

    setState({ events: [], status: "running", hitlEvent: null });

    const es = new EventSource(`/runs/${runId}/stream`);
    esRef.current = es;

    const handle = (raw: MessageEvent) => {
      let evt: TrajectoryEvent;
      try {
        evt = JSON.parse(raw.data) as TrajectoryEvent;
      } catch {
        return; // ignore malformed / ping comments
      }

      setState((s) => {
        const next: StreamState = {
          ...s,
          events: [...s.events, evt],
        };
        if (evt.type === "hitl_request") {
          next.status = "paused_hitl";
          next.hitlEvent = evt;
        } else if (evt.type === "complete") {
          next.status = "done";
        } else if (evt.type === "error") {
          next.status = "failed";
        }
        return next;
      });

      if (evt.type === "complete" || evt.type === "error") {
        es.close();
        esRef.current = null;
      }
    };

    // The backend tags each event with its type as the SSE event name.
    const types = [
      "thought",
      "action",
      "tool_call",
      "tool_result",
      "decision",
      "error",
      "hitl_request",
      "complete",
    ];
    types.forEach((t) => es.addEventListener(t, handle as EventListener));
    // Fallback for any untyped messages.
    es.onmessage = handle;

    es.onerror = () => {
      // Browser auto-reconnects; we just note we're not done yet. If the run
      // has already finished, the close above already fired.
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [runId]);

  return { ...state, reset, clearHitl };
}
