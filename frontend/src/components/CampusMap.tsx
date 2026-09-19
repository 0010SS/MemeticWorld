"use client";

import { useMemo } from "react";

import type { EventRow, Persona, WorldLayout } from "@/lib/types";
import type { Frame } from "@/lib/replay";

interface Props {
  world: WorldLayout;
  agents: Persona[];
  colors: Record<string, string>;
  frame: Frame | undefined;
  activeEvents: EventRow[];
  selectedAgent: string | null;
  onSelectAgent: (id: string | null) => void;
  phraseRegex: RegExp | null;
  transitionMs: number;
}

const COLS = 4;
const BUBBLE_W = 190;
const BUBBLE_H = 54;

function slotPosition(rect: [number, number, number, number], index: number) {
  const [x, y, w] = rect;
  const cellW = (w - 24) / COLS;
  const col = index % COLS;
  const row = Math.floor(index / COLS);
  return { x: x + 12 + cellW * (col + 0.5), y: y + 84 + row * 48 };
}

function truncate(s: string, n: number) {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

export default function CampusMap({
  world,
  agents,
  colors,
  frame,
  activeEvents,
  selectedAgent,
  onSelectAgent,
  phraseRegex,
  transitionMs,
}: Props) {
  const rects = useMemo(() => Object.fromEntries(world.locations.map((l) => [l.name, l.rect])), [world]);
  const centers = useMemo(
    () =>
      Object.fromEntries(world.locations.map((l) => [l.name, { x: l.rect[0] + l.rect[2] / 2, y: l.rect[1] + l.rect[3] / 2 }])),
    [world],
  );
  const names = useMemo(() => Object.fromEntries(agents.map((a) => [a.id, a.name])), [agents]);

  // Agent positions: each location lays its occupants out in a small grid, in cast order.
  const positions = useMemo(() => {
    const out: Record<string, { x: number; y: number }> = {};
    const byLocation: Record<string, string[]> = {};
    for (const a of agents) {
      const loc = frame?.agents[a.id]?.location ?? a.schedule[0][1];
      (byLocation[loc] ??= []).push(a.id);
    }
    for (const [loc, ids] of Object.entries(byLocation)) {
      const rect = rects[loc] ?? ([0, 0, 100, 100] as [number, number, number, number]);
      ids.forEach((id, i) => (out[id] = slotPosition(rect, i)));
    }
    return out;
  }, [agents, frame, rects]);

  // Speech: the last line each agent said this tick; lines with a meme phrase first; at most 3 bubbles.
  const bubbles = useMemo(() => {
    if (!frame) return [];
    const lastBySpeaker = new Map<string, EventRow>();
    for (const u of frame.utterances) if (u.agent_id) lastBySpeaker.set(u.agent_id, u);
    const lines = [...lastBySpeaker.values()].sort((a, b) => {
      const am = phraseRegex ? Number(new RegExp(phraseRegex.source, "i").test(a.text ?? "")) : 0;
      const bm = phraseRegex ? Number(new RegExp(phraseRegex.source, "i").test(b.text ?? "")) : 0;
      return bm - am;
    });
    const placed: { x: number; y: number; line: EventRow; hasMeme: boolean }[] = [];
    for (const line of lines.slice(0, 3)) {
      const p = positions[line.agent_id!];
      if (!p) continue;
      let x = Math.min(Math.max(p.x - BUBBLE_W / 2, 4), 1000 - BUBBLE_W - 4);
      let y = p.y - 22 - BUBBLE_H;
      for (let guard = 0; guard < 6; guard++) {
        const hit = placed.find((b) => Math.abs(b.x - x) < BUBBLE_W && Math.abs(b.y - y) < BUBBLE_H + 4);
        if (!hit) break;
        y = hit.y - BUBBLE_H - 6;
        if (y < 2) {
          y = p.y + 30;
          x = Math.min(x + 40, 1000 - BUBBLE_W - 4);
        }
      }
      const hasMeme = phraseRegex ? new RegExp(phraseRegex.source, "i").test(line.text ?? "") : false;
      placed.push({ x, y: Math.max(2, y), line, hasMeme });
    }
    return placed;
  }, [frame, positions, phraseRegex]);

  const conversationLinks = useMemo(() => {
    if (!frame) return [];
    const seen = new Set<string>();
    const links: { a: string; b: string }[] = [];
    for (const u of frame.utterances) {
      const other = u.data.audience?.[0];
      if (u.data.kind !== "talk" || !u.agent_id || !other) continue;
      const key = [u.agent_id, other].sort().join("|");
      if (!seen.has(key)) {
        seen.add(key);
        links.push({ a: u.agent_id, b: other });
      }
    }
    return links;
  }, [frame]);

  const eventsByLocation = useMemo(() => {
    const out: Record<string, EventRow[]> = {};
    for (const e of activeEvents) (out[e.location ?? ""] ??= []).push(e);
    return out;
  }, [activeEvents]);

  const [vx, vy, vw, vh] = world.viewBox;

  return (
    <svg
      viewBox={`${vx} ${vy} ${vw} ${vh}`}
      className="w-full h-auto select-none"
      role="img"
      aria-label="Campus map with agents"
      onClick={() => onSelectAgent(null)}
    >
      {/* walkways */}
      {world.edges.map(([a, b]) => (
        <line
          key={`${a}-${b}`}
          x1={centers[a].x}
          y1={centers[a].y}
          x2={centers[b].x}
          y2={centers[b].y}
          stroke="var(--surface-3)"
          strokeWidth={10}
          strokeLinecap="round"
        />
      ))}

      {/* buildings */}
      {world.locations.map((loc) => {
        const [x, y, w, h] = loc.rect;
        const events = eventsByLocation[loc.name] ?? [];
        const fresh = events.some((e) => frame && e.tick === frame.tick);
        return (
          <g key={loc.name}>
            <title>{`${loc.name}: ${loc.description}`}</title>
            <rect
              x={x}
              y={y}
              width={w}
              height={h}
              rx={14}
              fill={loc.name === "Quad" ? "var(--lawn)" : "var(--surface)"}
              stroke={fresh ? "var(--ink-2)" : "var(--border)"}
              strokeWidth={fresh ? 2 : 1}
            />
            <text x={x + 14} y={y + 24} fontSize={14} fontWeight={600} fill="var(--ink)">
              {loc.name}
            </text>
            {events.length > 0 && (
              <g>
                <title>{events.map((e) => e.text).join("\n")}</title>
                <rect x={x + 8} y={y + 34} width={w - 16} height={20} rx={10} fill="var(--mark)" />
                <text x={x + 18} y={y + 48} fontSize={11} fill="var(--mark-ink)">
                  {`⚡ ${truncate(events[events.length - 1].text ?? "", 36)}`}
                </text>
              </g>
            )}
          </g>
        );
      })}

      {/* who is talking to whom this tick */}
      {conversationLinks.map(({ a, b }) =>
        positions[a] && positions[b] ? (
          <line
            key={`${a}-${b}`}
            x1={positions[a].x}
            y1={positions[a].y}
            x2={positions[b].x}
            y2={positions[b].y}
            stroke="var(--ink-2)"
            strokeWidth={2}
            strokeLinecap="round"
            opacity={0.55}
          />
        ) : null,
      )}

      {/* agents (only once the log has loaded, so nobody glides in from a default spot) */}
      {frame && agents.map((a) => {
        const p = positions[a.id];
        if (!p) return null;
        const selected = selectedAgent === a.id;
        const activity = frame?.agents[a.id]?.activity ?? "";
        return (
          <g
            key={a.id}
            className="agent-token cursor-pointer"
            style={{ transform: `translate(${p.x}px, ${p.y}px)`, transitionDuration: `${transitionMs}ms` }}
            onClick={(ev) => {
              ev.stopPropagation();
              onSelectAgent(selected ? null : a.id);
            }}
          >
            <title>{`${a.name}: ${activity}`}</title>
            <circle r={22} fill="transparent" />
            {selected && <circle r={19} fill="none" stroke="var(--ink)" strokeWidth={2} />}
            <circle r={14} fill={colors[a.id]} stroke="var(--surface)" strokeWidth={2} />
            <text y={30} textAnchor="middle" fontSize={11} fontWeight={selected ? 700 : 500} fill="var(--ink-2)">
              {a.name}
            </text>
          </g>
        );
      })}

      {/* speech bubbles, on top */}
      {bubbles.map(({ x, y, line, hasMeme }) => (
        <foreignObject key={line.id} x={x} y={y} width={BUBBLE_W} height={BUBBLE_H} style={{ overflow: "visible" }}>
          <div
            className={`max-h-full rounded-xl border px-2 py-1 text-[10.5px] leading-snug shadow-sm ${
              hasMeme ? "border-[var(--mark-ink)]/40 bg-mark text-mark-ink" : "border-line bg-[var(--bubble)] text-ink"
            }`}
            style={{ display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}
          >
            <span className="font-semibold">{names[line.agent_id!]}: </span>
            {line.text}
          </div>
        </foreignObject>
      ))}
    </svg>
  );
}
