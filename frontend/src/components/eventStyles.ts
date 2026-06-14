import type { EventType } from "../lib/api";

// Per-event-type visual treatment, read like log levels.
export const TYPE_STYLE: Record<
  EventType,
  { label: string; color: string; glyph: string }
> = {
  thought: { label: "think", color: "#7c8aa0", glyph: "·" },
  action: { label: "act", color: "#9db2d4", glyph: "›" },
  tool_call: { label: "tool", color: "#5fa8e6", glyph: "⇄" },
  tool_result: { label: "result", color: "#56c2b3", glyph: "✓" },
  decision: { label: "decide", color: "#e0a458", glyph: "◆" },
  error: { label: "error", color: "#e0606b", glyph: "✕" },
  hitl_request: { label: "human", color: "#c98bdb", glyph: "?" },
  complete: { label: "done", color: "#5fc97a", glyph: "■" },
};

// Stable color per agent so handoffs are legible at a glance.
const AGENT_COLORS = [
  "#6ea8fe",
  "#56c2b3",
  "#e0a458",
  "#c98bdb",
  "#7ed492",
  "#e0808a",
  "#9db2d4",
  "#d4b86a",
];

const agentColorCache = new Map<string, string>();
let nextColor = 0;

export function agentColor(agent: string): string {
  if (!agentColorCache.has(agent)) {
    agentColorCache.set(agent, AGENT_COLORS[nextColor % AGENT_COLORS.length]);
    nextColor++;
  }
  return agentColorCache.get(agent)!;
}

export function phaseLabel(phase: number): string {
  const names: Record<number, string> = {
    1: "Context & Validation",
    2: "Generation & Mapping",
    3: "Script & Reflection",
    4: "Execution & Healing",
    5: "Reporting & Memory",
  };
  return names[phase] ?? "System";
}
