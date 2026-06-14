import { useEffect, useRef, useState } from "react";
import type { RunStatus, TrajectoryEvent } from "../lib/api";
import { TYPE_STYLE, agentColor } from "./eventStyles";

interface Props {
  events: TrajectoryEvent[];
  status: RunStatus;
  hitlEvent: TrajectoryEvent | null;
  onResume: (answer: string) => void;
}

function timeOf(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("en-US", { hour12: false }) + "." +
    String(d.getMilliseconds()).padStart(3, "0");
}

export function TrajectoryPanel({ events, status, hitlEvent, onResume }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const [answer, setAnswer] = useState("");

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length, hitlEvent]);

  return (
    <div className="flex flex-col h-full">
      <div
        className="flex items-center gap-2 px-4 py-2.5 text-xs mono"
        style={{ borderBottom: "1px solid var(--border)", color: "var(--text-dim)" }}
      >
        <span style={{ color: "#5fc97a" }}>●</span>
        trajectory
        {status === "running" && <span className="pulse">— streaming</span>}
      </div>

      <div className="console flex-1 overflow-y-auto px-4 py-3 mono text-[13px] leading-relaxed">
        {events.length === 0 && (
          <div style={{ color: "var(--text-dim)" }}>
            Awaiting run. Submit a story to watch the agents reason in real time.
          </div>
        )}

        {events.map((e, i) => {
          const t = TYPE_STYLE[e.type];
          return (
            <div key={i} className="event-row flex gap-3 py-1">
              <span className="shrink-0 tabular-nums" style={{ color: "#48515f" }}>
                {timeOf(e.ts)}
              </span>
              <span
                className="shrink-0 w-12 text-right"
                style={{ color: t.color }}
                title={e.type}
              >
                {t.label}
              </span>
              <span className="shrink-0" style={{ color: t.color }}>
                {t.glyph}
              </span>
              <span className="flex flex-col">
                <span style={{ color: agentColor(e.agent), fontWeight: 500 }}>
                  {e.agent}
                </span>
                <span style={{ color: e.type === "thought" ? "var(--text-dim)" : "var(--text)" }}>
                  {e.message}
                </span>
              </span>
            </div>
          );
        })}

        {hitlEvent && (
          <div
            className="event-row my-3 rounded-md p-3"
            style={{ background: "#231a2e", border: "1px solid #4a2f5e" }}
          >
            <div className="mb-2 text-xs uppercase tracking-wide" style={{ color: "#c98bdb" }}>
              Human clarification needed
            </div>
            <div className="mb-3 text-[13px]" style={{ color: "var(--text)" }}>
              {(hitlEvent.data?.question as string) ?? hitlEvent.message}
            </div>
            <textarea
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              placeholder="Rewrite or clarify the story…"
              spellCheck={false}
              className="w-full resize-none rounded p-2 text-[13px] outline-none"
              style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--text)", minHeight: "70px" }}
            />
            <button
              onClick={() => { onResume(answer); setAnswer(""); }}
              disabled={!answer.trim()}
              className="mt-2 rounded px-4 py-1.5 text-sm font-medium disabled:opacity-40"
              style={{ background: "#9333ea", color: "white" }}
            >
              Submit & resume
            </button>
          </div>
        )}

        <div ref={endRef} />
      </div>
    </div>
  );
}
