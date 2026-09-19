"use client";

import { useEffect, useState } from "react";

import { Highlight } from "@/components/Highlight";
import { api } from "@/lib/api";
import { useDebounced } from "@/lib/hooks";
import type { Frame } from "@/lib/replay";
import type { DecisionRow, MemoryRow, Persona } from "@/lib/types";

interface Props {
  runId: number;
  agent: Persona;
  agents: Persona[];
  colors: Record<string, string>;
  frame: Frame | undefined;
  tick: number;
  regex: RegExp | null;
}

const SOURCE_LABEL: Record<string, string> = {
  heard: "heard",
  overheard: "overheard",
  said: "said",
  event: "saw",
  sighting: "noticed",
  conversation: "talked",
};

function minutes(hhmm: string) {
  const [h, m] = hhmm.split(":").map(Number);
  return h * 60 + m;
}

function currentSlot(schedule: Persona["schedule"], simTime: string | undefined) {
  if (!simTime) return -1;
  const match = simTime.match(/(\d+):(\d+) (AM|PM)/);
  if (!match) return -1;
  let h = Number(match[1]) % 12;
  if (match[3] === "PM") h += 12;
  const now = h * 60 + Number(match[2]);
  let idx = 0;
  schedule.forEach((s, i) => {
    if (minutes(s[0]) <= now) idx = i;
  });
  return idx;
}

export default function AgentPanel({ runId, agent, agents, colors, frame, tick, regex }: Props) {
  const [memories, setMemories] = useState<MemoryRow[]>([]);
  const [decision, setDecision] = useState<DecisionRow | null>(null);
  const [error, setError] = useState<string | null>(null);
  const settledTick = useDebounced(tick, 350);
  const names = Object.fromEntries(agents.map((a) => [a.id, a.name]));
  const state = frame?.agents[agent.id];
  const slot = currentSlot(agent.schedule, frame?.simTime);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.memories(runId, agent.id, settledTick, 40), api.decisions(runId, agent.id, settledTick, 1)])
      .then(([m, d]) => {
        if (cancelled) return;
        setMemories(m);
        setDecision(d[0] ?? null);
        setError(null);
      })
      .catch((e) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [runId, agent.id, settledTick]);

  return (
    <div className="scrollbar-thin h-full space-y-4 overflow-y-auto p-3 text-sm">
      <div>
        <div className="flex items-center gap-2">
          <span className="inline-block h-3.5 w-3.5 rounded-full" style={{ background: colors[agent.id] }} />
          <h2 className="text-lg font-semibold">{agent.name}</h2>
        </div>
        <p className="text-ink-2">{agent.role}</p>
        <p className="mt-1 text-xs text-ink-2">{agent.personality}</p>
        <p className="mt-1 text-xs text-ink-3">Interests: {agent.interests.join(", ")}</p>
      </div>

      <div className="rounded-lg border border-line bg-surface px-3 py-2">
        <div className="text-[11px] text-ink-3">Right now · {frame?.simTime ?? "—"}</div>
        <div className="font-semibold">{state?.location ?? "—"}</div>
        <div className="text-ink-2">{state?.activity}</div>
      </div>

      <section>
        <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-3">Temperament (mock LLM only)</h3>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          {Object.entries(agent.traits).map(([k, v]) => (
            <div key={k} className="text-xs">
              <div className="flex justify-between text-ink-2">
                <span>{k}</span>
                <span className="tabular">{v.toFixed(2)}</span>
              </div>
              <div className="h-1.5 rounded bg-surface-3">
                <div className="h-1.5 rounded bg-ink-3" style={{ width: `${v * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-3">Schedule</h3>
        <ol className="space-y-0.5 text-xs">
          {agent.schedule.map(([time, loc, act], i) => (
            <li key={time} className={`flex gap-2 rounded px-1 ${i === slot ? "bg-surface-2 font-semibold" : "text-ink-2"}`}>
              <span className="w-11 tabular">{time}</span>
              <span className="w-24">{loc}</span>
              <span className="truncate">{act}</span>
            </li>
          ))}
        </ol>
      </section>

      <section>
        <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-3">Knows at the start</h3>
        <ul className="space-y-0.5 text-xs text-ink-2">
          {Object.entries(agent.relationships).map(([id, [label, closeness]]) => (
            <li key={id} className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-2 rounded-full" style={{ background: colors[id] }} />
              <span className="font-semibold text-ink">{names[id]}</span> {label}
              <span className="ml-auto tabular text-ink-3">{closeness.toFixed(2)}</span>
            </li>
          ))}
        </ul>
      </section>

      {error && <p className="text-xs text-critical">{error}</p>}

      <section>
        <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-3">Latest LLM call</h3>
        {!decision && <p className="text-xs text-ink-3">No model calls yet (following the schedule on autopilot).</p>}
        {decision && (
          <div className="space-y-1.5 text-xs">
            <div className="text-ink-2">
              {decision.kind === "decide" ? "Decision" : "Conversation reply"} at tick {decision.tick}
              {decision.error && <span className="text-critical"> · {decision.error}</span>}
            </div>
            {decision.parsed && (
              <pre className="whitespace-pre-wrap rounded-md bg-surface-2 p-2 font-mono text-[11px] leading-snug">
                {JSON.stringify(decision.parsed, null, 2)}
              </pre>
            )}
            <details>
              <summary className="cursor-pointer text-ink-2">Memories it retrieved ({decision.retrieved_memories.length})</summary>
              <ul className="mt-1 space-y-1">
                {decision.retrieved_memories.map((m) => (
                  <li key={m.id} className="rounded bg-surface-2 px-2 py-1">
                    <Highlight text={m.text} regex={regex} />
                  </li>
                ))}
              </ul>
            </details>
            <details>
              <summary className="cursor-pointer text-ink-2">Full prompt (exactly what the agent saw)</summary>
              <pre className="mt-1 max-h-80 overflow-auto whitespace-pre-wrap rounded-md bg-surface-2 p-2 font-mono text-[11px] leading-snug">
                {decision.prompt}
              </pre>
            </details>
          </div>
        )}
      </section>

      <section>
        <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-3">Recent memories</h3>
        <ul className="space-y-1 text-xs">
          {memories.map((m) => (
            <li key={m.id} className="flex gap-2">
              <span className="w-16 shrink-0 text-ink-3">{SOURCE_LABEL[m.source_type] ?? m.source_type}</span>
              <span className="text-ink-2">
                <Highlight text={m.text} regex={regex} />
              </span>
            </li>
          ))}
          {memories.length === 0 && <li className="text-ink-3">No memories yet.</li>}
        </ul>
      </section>
    </div>
  );
}
