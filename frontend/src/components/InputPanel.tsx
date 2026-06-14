import { useState } from "react";
import type { RunStatus } from "../lib/api";

interface Props {
  status: RunStatus;
  onRun: (story: string, targetUrl: string) => void;
  onReset: () => void;
}

const SAMPLE =
  "As a registered user, I want to reset my password via an email link so that I can regain access to my account. Acceptance criteria: clicking Forgot Password sends a reset email within 1 minute; the link expires after 24 hours; submitting a new password logs me in.";

export function InputPanel({ status, onRun, onReset }: Props) {
  const [story, setStory] = useState(SAMPLE);
  const [url, setUrl] = useState("http://localhost:3000");
  const running = status === "running" || status === "paused_hitl";

  return (
    <div className="flex flex-col gap-3 p-4 h-full">
      <div className="text-xs uppercase tracking-wider mono" style={{ color: "var(--text-dim)" }}>
        User story / Jira ID
      </div>
      <textarea
        value={story}
        onChange={(e) => setStory(e.target.value)}
        disabled={running}
        spellCheck={false}
        className="flex-1 resize-none rounded-md p-3 text-sm leading-relaxed outline-none disabled:opacity-60"
        style={{
          background: "var(--panel-2)",
          border: "1px solid var(--border)",
          color: "var(--text)",
          minHeight: "180px",
        }}
        placeholder="Paste a user story or a Jira ticket ID…"
      />
      <div className="text-xs uppercase tracking-wider mono" style={{ color: "var(--text-dim)" }}>
        Target URL
      </div>
      <input
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        disabled={running}
        spellCheck={false}
        className="rounded-md px-3 py-2 text-sm outline-none mono disabled:opacity-60"
        style={{
          background: "var(--panel-2)",
          border: "1px solid var(--border)",
          color: "var(--text)",
        }}
      />
      <div className="flex gap-2 pt-1">
        <button
          onClick={() => onRun(story, url)}
          disabled={running || !story.trim()}
          className="flex-1 rounded-md py-2.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40"
          style={{ background: "#2563eb", color: "white" }}
        >
          {running ? "Running…" : "Run analysis"}
        </button>
        <button
          onClick={onReset}
          className="rounded-md px-4 py-2.5 text-sm transition-colors"
          style={{ background: "var(--panel-2)", border: "1px solid var(--border)", color: "var(--text-dim)" }}
        >
          Clear
        </button>
      </div>
    </div>
  );
}
