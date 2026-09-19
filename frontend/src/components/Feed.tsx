"use client";

import { useMemo, useState } from "react";

import { Highlight } from "@/components/Highlight";
import type { EventRow, Persona } from "@/lib/types";
import type { Replay } from "@/lib/replay";

interface Props {
  replay: Replay;
  tick: number;
  agents: Persona[];
  colors: Record<string, string>;
  regex: RegExp | null;
  strongRegex: RegExp | null;
  selectedAgent: string | null;
}

type Item =
  | { kind: "conversation"; id: string; tick: number; simTime: string; location: string; lines: EventRow[] }
  | { kind: "event"; id: string; tick: number; simTime: string; location: string; event: EventRow };

const MAX_ITEMS = 80;

export default function Feed({ replay, tick, agents, colors, regex, strongRegex, selectedAgent }: Props) {
  const [filter, setFilter] = useState<"all" | "memes" | "agent">("all");
  const names = useMemo(() => Object.fromEntries(agents.map((a) => [a.id, a.name])), [agents]);

  const items = useMemo(() => {
    const out: Item[] = [];
    const byConversation = new Map<string, Item & { kind: "conversation" }>();
    const test = regex ? new RegExp(regex.source, "i") : null;
    for (const u of replay.utterances) {
      if (u.tick > tick) break;
      if (filter === "memes" && !(test && test.test(u.text ?? ""))) continue;
      if (filter === "agent" && selectedAgent && u.agent_id !== selectedAgent && !u.data.audience?.includes(selectedAgent))
        continue;
      const cid = u.data.conversation_id ?? String(u.id);
      let conv = byConversation.get(cid);
      if (!conv) {
        conv = { kind: "conversation", id: cid, tick: u.tick, simTime: u.sim_time, location: u.location ?? "", lines: [] };
        byConversation.set(cid, conv);
        out.push(conv);
      }
      conv.lines.push(u);
    }
    if (filter === "all") {
      for (const e of replay.worldEvents) {
        if (e.tick > tick) break;
        out.push({ kind: "event", id: `e${e.id}`, tick: e.tick, simTime: e.sim_time, location: e.location ?? "", event: e });
      }
    }
    out.sort((a, b) => b.tick - a.tick || (a.kind === "event" ? -1 : 1));
    return out.slice(0, MAX_ITEMS);
  }, [replay, tick, filter, regex, selectedAgent]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex gap-1 border-b border-line px-3 py-2 text-xs">
        {(
          [
            ["all", "Everything"],
            ["memes", "Lines with detected phrases"],
            ["agent", selectedAgent ? `${names[selectedAgent]} only` : "Selected agent"],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            disabled={key === "agent" && !selectedAgent}
            onClick={() => setFilter(key)}
            className={`rounded-full px-2.5 py-1 ${
              filter === key ? "bg-ink text-[var(--surface)]" : "text-ink-2 hover:bg-surface-2 disabled:opacity-40"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="scrollbar-thin min-h-0 flex-1 space-y-2 overflow-y-auto px-3 py-3">
        {items.length === 0 && <p className="text-sm text-ink-3">Nothing yet at this point in time.</p>}
        {items.map((item) =>
          item.kind === "event" ? (
            <div key={item.id} className="rounded-lg bg-mark/60 px-3 py-2 text-xs text-mark-ink">
              <div className="mb-0.5 font-semibold">
                ⚡ {item.location} · {item.simTime}
              </div>
              {item.event.text}
            </div>
          ) : (
            <div key={item.id} className="rounded-lg border border-line bg-surface px-3 py-2">
              <div className="mb-1 flex items-center gap-1.5 text-[11px] text-ink-3">
                <span>{item.location}</span>·<span className="tabular">{item.simTime}</span>
                {item.lines[0].data.kind === "react" && <span>· said out loud</span>}
              </div>
              <ul className="space-y-1">
                {item.lines.map((line) => (
                  <li key={line.id} className="flex gap-2 text-[13px] leading-snug">
                    <span
                      className="mt-1.5 inline-block h-2 w-2 shrink-0 rounded-full"
                      style={{ background: colors[line.agent_id ?? ""] }}
                      aria-hidden
                    />
                    <span>
                      <span className="font-semibold">{names[line.agent_id ?? ""]}</span>
                      {line.data.audience?.length === 1 && line.data.kind === "talk" && (
                        <span className="text-ink-3"> → {names[line.data.audience[0]]}</span>
                      )}
                      <span className="text-ink-3">: </span>
                      <Highlight text={line.text ?? ""} regex={regex} strong={strongRegex} />
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ),
        )}
      </div>
    </div>
  );
}
