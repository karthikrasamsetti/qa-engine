import { useCallback, useEffect, useState } from "react";
import { InputPanel } from "./components/InputPanel";
import { TrajectoryPanel } from "./components/TrajectoryPanel";
import { OutputPanel } from "./components/OutputPanel";
import { phaseLabel } from "./components/eventStyles";
import { useTrajectoryStream } from "./hooks/useTrajectoryStream";
import { startRun, resumeRun, fetchCost, type CostSummary, type RunStatus } from "./lib/api";

const STATUS_STYLE: Record<RunStatus, { label: string; color: string }> = {
  idle: { label: "idle", color: "#788396" },
  running: { label: "running", color: "#5fa8e6" },
  paused_hitl: { label: "paused · awaiting human", color: "#c98bdb" },
  done: { label: "done", color: "#5fc97a" },
  failed: { label: "failed", color: "#e0606b" },
};

export default function App() {
  const [runId, setRunId] = useState<string | null>(null);
  const { events, status, hitlEvent, reset, clearHitl } = useTrajectoryStream(runId);
  const [cost, setCost] = useState<CostSummary | null>(null);

  // The current phase is the highest phase number seen in the stream.
  const currentPhase = events.reduce((max, e) => (e.phase > max ? e.phase : max), 0);

  const handleRun = useCallback(async (story: string, targetUrl: string) => {
    reset();
    setCost(null);
    try {
      const id = await startRun(story, targetUrl);
      setRunId(id);
    } catch (err) {
      console.error(err);
      alert("Could not start the run. Is the backend running on :8000?");
    }
  }, [reset]);

  const handleResume = useCallback(async (answer: string) => {
    if (!runId) return;
    clearHitl();
    try {
      await resumeRun(runId, answer);
    } catch (err) {
      console.error(err);
    }
  }, [runId, clearHitl]);

  const handleReset = useCallback(() => {
    setRunId(null);
    setCost(null);
    reset();
  }, [reset]);

  // Refresh cost when the run reaches a terminal/paused state.
  useEffect(() => {
    if (!runId) return;
    if (status === "done" || status === "paused_hitl" || status === "failed") {
      fetchCost(runId).then(setCost);
    }
  }, [runId, status]);

  const s = STATUS_STYLE[status];

  return (
    <div className="flex flex-col h-screen" style={{ background: "var(--bg)" }}>
      {/* Header */}
      <header
        className="flex items-center justify-between px-5 py-3"
        style={{ borderBottom: "1px solid var(--border)", background: "var(--panel)" }}
      >
        <div className="flex items-baseline gap-3">
          <span className="text-[15px] font-semibold tracking-tight">Intelligent QA Engine</span>
          <span className="mono text-xs" style={{ color: "var(--text-dim)" }}>
            multi-agent test authoring
          </span>
        </div>
        <div className="flex items-center gap-4 mono text-xs">
          {cost && (
            <span style={{ color: "var(--text-dim)" }}>
              {cost.llm_calls} calls · ${cost.total_cost_usd.toFixed(4)}
            </span>
          )}
          <span
            className="flex items-center gap-1.5 rounded-full px-3 py-1"
            style={{ background: "var(--panel-2)", border: "1px solid var(--border)", color: s.color }}
          >
            <span style={{ fontSize: "8px" }}>●</span>
            {s.label}
          </span>
        </div>
      </header>

      {/* Phase rail — signature element: fills as the run advances */}
      <div className="flex items-stretch" style={{ background: "var(--panel)", borderBottom: "1px solid var(--border)" }}>
        {[1, 2, 3, 4, 5].map((p) => {
          const reached = currentPhase >= p;
          const active = currentPhase === p && status === "running";
          return (
            <div
              key={p}
              className="flex-1 px-4 py-2"
              style={{
                borderRight: p < 5 ? "1px solid var(--border)" : "none",
                opacity: reached ? 1 : 0.35,
              }}
            >
              <div className="flex items-center gap-2">
                <span
                  className={active ? "pulse" : ""}
                  style={{
                    width: "7px", height: "7px", borderRadius: "50%",
                    background: reached ? "#5fa8e6" : "#3a4452",
                    display: "inline-block",
                  }}
                />
                <span className="mono text-[10px] uppercase tracking-wider" style={{ color: "var(--text-dim)" }}>
                  Phase {p}
                </span>
              </div>
              <div className="mt-1 text-[11px]" style={{ color: reached ? "var(--text)" : "var(--text-dim)" }}>
                {phaseLabel(p)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Three panels */}
      <main className="flex flex-1 overflow-hidden">
        <section className="w-[26%] min-w-[300px]" style={{ borderRight: "1px solid var(--border)", background: "var(--panel)" }}>
          <InputPanel status={status} onRun={handleRun} onReset={handleReset} />
        </section>
        <section className="flex-1" style={{ background: "var(--bg)" }}>
          <TrajectoryPanel events={events} status={status} hitlEvent={hitlEvent} onResume={handleResume} />
        </section>
        <section className="w-[32%] min-w-[320px]" style={{ borderLeft: "1px solid var(--border)", background: "var(--panel)" }}>
          <OutputPanel events={events} />
        </section>
      </main>
    </div>
  );
}
