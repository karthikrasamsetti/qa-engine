import { useMemo, useState } from "react";
import type { TrajectoryEvent } from "../lib/api";

interface Props {
  events: TrajectoryEvent[];
}

// Pull the most recent script / report out of the event stream as later phases
// populate them. Stage C ships the panels; Phases 3 & 5 fill them with data.
function deriveOutputs(events: TrajectoryEvent[]) {
  let script = "";
  let report: Record<string, unknown> | null = null;
  let verdict: Record<string, unknown> | null = null;

  for (const e of events) {
    if (typeof e.data?.script === "string") script = e.data.script as string;
    if (e.type === "complete" && e.data?.report) report = e.data.report as Record<string, unknown>;
    if (e.data?.verdict) verdict = e.data.verdict as Record<string, unknown>;
  }
  return { script, report, verdict };
}

export function OutputPanel({ events }: Props) {
  const [tab, setTab] = useState<"code" | "report">("code");
  const { script, report, verdict } = useMemo(() => deriveOutputs(events), [events]);

  return (
    <div className="flex flex-col h-full">
      <div className="flex" style={{ borderBottom: "1px solid var(--border)" }}>
        {(["code", "report"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className="px-4 py-2.5 text-xs mono uppercase tracking-wider transition-colors"
            style={{
              color: tab === t ? "var(--text)" : "var(--text-dim)",
              borderBottom: tab === t ? "2px solid #2563eb" : "2px solid transparent",
            }}
          >
            {t === "code" ? "Playwright script" : "Report"}
          </button>
        ))}
      </div>

      <div className="console flex-1 overflow-y-auto p-4 mono text-[13px]">
        {tab === "code" && (
          script ? (
            <pre className="whitespace-pre-wrap leading-relaxed" style={{ color: "#9db2d4" }}>{script}</pre>
          ) : (
            <div style={{ color: "var(--text-dim)" }}>
              The generated Playwright script will appear here once Phase 3
              (Scaffolder ↔ Critic) is implemented.
            </div>
          )
        )}

        {tab === "report" && (
          report ? (
            <pre className="whitespace-pre-wrap leading-relaxed" style={{ color: "var(--text)" }}>
              {JSON.stringify(report, null, 2)}
            </pre>
          ) : verdict ? (
            <div>
              <div className="mb-2 text-xs uppercase tracking-wide" style={{ color: "var(--text-dim)" }}>
                Latest INVEST verdict
              </div>
              <pre className="whitespace-pre-wrap leading-relaxed" style={{ color: "var(--text)" }}>
                {JSON.stringify(verdict, null, 2)}
              </pre>
            </div>
          ) : (
            <div style={{ color: "var(--text-dim)" }}>
              The stakeholder report will appear here once Phase 5 (Synthesis)
              is implemented.
            </div>
          )
        )}
      </div>
    </div>
  );
}
