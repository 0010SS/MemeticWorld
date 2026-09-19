"use client";

import { useMemo, useState } from "react";

import { useWidth } from "@/lib/hooks";
import type { MemeDetail, MemeUse, Persona } from "@/lib/types";

interface Props {
  meme: MemeDetail;
  agents: Persona[];
  colors: Record<string, string>;
  tick: number;
  totalTicks: number;
  dayTicks: { tick: number; day: number }[];
  onSeek: (tick: number) => void;
}

const ROW_H = 28;
const LEFT = 84;
const RIGHT = 16;
const TOP = 22;

const ROLE_LABEL: Record<MemeUse["role"], string> = {
  origin: "first use (originator)",
  independent: "independent first use",
  adoption: "adoption: reused in a new conversation after hearing it",
  echo: "echo: repeated back in the same conversation",
  reuse: "reuse",
};

/**
 * Swimlane transmission diagram: one row per agent, time left to right.
 * Dots are uses; arrows run from the use that exposed an agent to that agent's adoption.
 * Heavier arrows = the adopter's prompt contained a memory quoting the phrase.
 */
export default function CascadeChart({ meme, agents, colors, tick, totalTicks, dayTicks, onSeek }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>(700);
  const [hover, setHover] = useState<{ use: MemeUse; x: number; y: number } | null>(null);
  const names = useMemo(() => Object.fromEntries(agents.map((a) => [a.id, a.name])), [agents]);
  const row = useMemo(() => Object.fromEntries(agents.map((a, i) => [a.id, i])), [agents]);
  const height = TOP + agents.length * ROW_H + 8;
  const plotW = width - LEFT - RIGHT;
  const x = (t: number) => LEFT + ((t + 0.5) / Math.max(1, totalTicks)) * plotW;
  const y = (agent: string) => TOP + (row[agent] ?? 0) * ROW_H + ROW_H / 2;
  const useTick = useMemo(() => Object.fromEntries(meme.uses.map((u) => [u.event_id, u.tick])), [meme]);
  const roles = useMemo(() => Object.fromEntries(meme.nodes.map((n) => [n.id, n.role])), [meme]);

  return (
    <div ref={ref} className="relative">
      <svg width={width} height={height} className="block" onMouseLeave={() => setHover(null)}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--ink-2)" />
          </marker>
        </defs>

        {/* rows */}
        {agents.map((a, i) => (
          <g key={a.id}>
            <line x1={LEFT} x2={width - RIGHT} y1={TOP + i * ROW_H + ROW_H / 2} y2={TOP + i * ROW_H + ROW_H / 2} stroke="var(--grid)" />
            <circle cx={10} cy={TOP + i * ROW_H + ROW_H / 2} r={4} fill={colors[a.id]} />
            <text x={20} y={TOP + i * ROW_H + ROW_H / 2 + 4} fontSize={12} fill="var(--ink-2)" fontWeight={roles[a.id] === "originator" ? 700 : 400}>
              {a.name}
            </text>
          </g>
        ))}

        {/* days */}
        {dayTicks.map((d) => (
          <g key={d.day}>
            <line x1={x(d.tick) - 0.5 * (plotW / totalTicks)} x2={x(d.tick) - 0.5 * (plotW / totalTicks)} y1={TOP - 6} y2={height - 6} stroke="var(--axis)" />
            <text x={x(d.tick) - 0.5 * (plotW / totalTicks) + 4} y={12} fontSize={10} fill="var(--ink-3)">
              Day {d.day}
            </text>
          </g>
        ))}

        {/* transmission arrows (only once they've happened) */}
        {meme.edges.map((e) => {
          const from = useTick[e.exposure_event_id] ?? e.tick;
          const x1 = x(from);
          const y1 = y(e.source);
          const x2 = x(e.tick);
          const y2 = y(e.target);
          const mid = Math.max(x1 + 12, (x1 + x2) / 2);
          const confirmed = e.evidence === "memory";
          return (
            <path
              key={`${e.source}-${e.target}`}
              d={`M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2 - 7} ${y2}`}
              fill="none"
              stroke="var(--ink-2)"
              strokeWidth={confirmed ? 2 : 1}
              opacity={e.tick <= tick ? (confirmed ? 0.9 : 0.45) : 0.08}
              markerEnd="url(#arrow)"
            />
          );
        })}

        {/* playhead */}
        <line x1={x(tick)} x2={x(tick)} y1={TOP - 6} y2={height - 6} stroke="var(--ink)" strokeWidth={1.5} opacity={0.6} />

        {/* uses */}
        {meme.uses.map((u) => {
          const cx = x(u.tick);
          const cy = y(u.speaker);
          const future = u.tick > tick;
          const big = u.role === "origin" || u.role === "adoption";
          return (
            <g
              key={u.event_id}
              opacity={future ? 0.18 : 1}
              className="cursor-pointer"
              onMouseEnter={() => setHover({ use: u, x: cx, y: cy })}
              onClick={() => onSeek(u.tick)}
            >
              <circle cx={cx} cy={cy} r={12} fill="transparent" />
              {u.role === "echo" ? (
                <circle cx={cx} cy={cy} r={4} fill="var(--surface)" stroke={colors[u.speaker]} strokeWidth={2} />
              ) : (
                <circle cx={cx} cy={cy} r={big ? 6 : 4} fill={colors[u.speaker]} stroke="var(--surface)" strokeWidth={2} />
              )}
              {u.role === "origin" && <circle cx={cx} cy={cy} r={9.5} fill="none" stroke="var(--ink)" strokeWidth={1.5} />}
            </g>
          );
        })}
      </svg>

      {hover && (
        <div
          className="pointer-events-none absolute z-10 w-64 rounded-md border border-line bg-surface px-2.5 py-2 text-xs shadow-md"
          style={{ left: Math.min(hover.x + 12, width - 270), top: Math.max(0, hover.y - 10) }}
        >
          <div className="mb-1 text-[13px] leading-snug text-ink">“{hover.use.text}”</div>
          <div className="text-ink-2">
            <span className="font-semibold">{names[hover.use.speaker]}</span> · {hover.use.sim_time} · {hover.use.location}
          </div>
          <div className="text-ink-3">{ROLE_LABEL[hover.use.role]}</div>
        </div>
      )}

      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-2">
        <span className="flex items-center gap-1.5">
          <svg width={20} height={20} aria-hidden>
            <circle cx={10} cy={10} r={5} fill="var(--ink-3)" />
            <circle cx={10} cy={10} r={8.5} fill="none" stroke="var(--ink)" strokeWidth={1.5} />
          </svg>
          origin
        </span>
        <span className="flex items-center gap-1.5">
          <svg width={14} height={14} aria-hidden>
            <circle cx={7} cy={7} r={5} fill="var(--ink-3)" />
          </svg>
          adoption
        </span>
        <span className="flex items-center gap-1.5">
          <svg width={14} height={14} aria-hidden>
            <circle cx={7} cy={7} r={3.5} fill="var(--surface)" stroke="var(--ink-3)" strokeWidth={2} />
          </svg>
          echo (same conversation)
        </span>
        <span className="flex items-center gap-1.5">
          <svg width={26} height={10} aria-hidden>
            <line x1={0} x2={26} y1={5} y2={5} stroke="var(--ink-2)" strokeWidth={2} />
          </svg>
          confirmed via retrieved memory
        </span>
        <span className="flex items-center gap-1.5">
          <svg width={26} height={10} aria-hidden>
            <line x1={0} x2={26} y1={5} y2={5} stroke="var(--ink-2)" strokeWidth={1} opacity={0.5} />
          </svg>
          heard it earlier (exposure only)
        </span>
      </div>
    </div>
  );
}
