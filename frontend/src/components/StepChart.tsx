"use client";

import { useMemo, useState } from "react";

import { useWidth } from "@/lib/hooks";

export interface StepSeries {
  id: string;
  label: string;
  color: string; // a text/ink token: aggregate curves are not agents, so they don't borrow agent colors
  points: { tick: number; value: number }[]; // cumulative, sorted by tick
}

interface Props {
  series: StepSeries[];
  totalTicks: number;
  yLabel: string;
  playhead?: number;
  dayTicks?: { tick: number; day: number }[];
  tickLabel?: (tick: number) => string;
  height?: number;
  yMax?: number;
}

const PAD = { top: 12, right: 16, bottom: 22, left: 36 };
const LABEL_ROOM = 130; // space for end-of-line labels when there are several series

function valueAt(points: { tick: number; value: number }[], tick: number) {
  let v = 0;
  for (const p of points) {
    if (p.tick > tick) break;
    v = p.value;
  }
  return v;
}

function niceTicks(max: number) {
  const step = max <= 5 ? 1 : max <= 10 ? 2 : max <= 25 ? 5 : max <= 50 ? 10 : Math.ceil(max / 5 / 10) * 10;
  const out = [];
  for (let v = 0; v <= max; v += step) out.push(v);
  return out;
}

/** Cumulative step lines over simulation time, with a crosshair that snaps to the tick under the pointer. */
export default function StepChart({ series, totalTicks, yLabel, playhead, dayTicks = [], tickLabel, height = 170, yMax }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>(500);
  const [hover, setHover] = useState<number | null>(null);
  const right = series.length > 1 ? LABEL_ROOM : PAD.right;
  const plotW = width - PAD.left - right;
  const plotH = height - PAD.top - PAD.bottom;
  const maxV = yMax ?? Math.max(1, ...series.flatMap((s) => s.points.map((p) => p.value)));
  const ticks = niceTicks(maxV);
  const top = Math.max(maxV, ticks[ticks.length - 1]);
  const x = (t: number) => PAD.left + (t / Math.max(1, totalTicks)) * plotW;
  const y = (v: number) => PAD.top + plotH - (v / top) * plotH;

  const paths = useMemo(
    () =>
      series.map((s) => {
        let d = `M ${x(0)} ${y(0)}`;
        let last = 0;
        for (const p of s.points) {
          d += ` H ${x(p.tick)} V ${y(p.value)}`;
          last = p.value;
        }
        d += ` H ${x(totalTicks)}`;
        return { ...s, d, last };
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [series, totalTicks, width, top],
  );

  return (
    <div ref={ref} className="relative">
      <svg
        width={width}
        height={height}
        className="block"
        onPointerMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const t = Math.round(((e.clientX - r.left - PAD.left) / plotW) * totalTicks);
          setHover(t >= 0 && t <= totalTicks ? t : null);
        }}
        onPointerLeave={() => setHover(null)}
      >
        {ticks.map((v) => (
          <g key={v}>
            <line x1={PAD.left} x2={width - right} y1={y(v)} y2={y(v)} stroke={v === 0 ? "var(--axis)" : "var(--grid)"} />
            <text x={PAD.left - 6} y={y(v) + 3.5} fontSize={10} textAnchor="end" fill="var(--ink-3)" className="tabular">
              {v}
            </text>
          </g>
        ))}
        {dayTicks.map((d) => (
          <text key={d.day} x={x(d.tick) + 2} y={height - 6} fontSize={10} fill="var(--ink-3)">
            Day {d.day}
          </text>
        ))}
        <text x={PAD.left} y={PAD.top - 2} fontSize={10} fill="var(--ink-3)">
          {yLabel}
        </text>
        {paths.map((s) => (
          <g key={s.id}>
            <path d={`${s.d} V ${y(0)} H ${x(0)} Z`} fill={s.color} opacity={0.08} />
            <path d={s.d} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
            {series.length > 1 && (
              <text x={width - right + 6} y={y(s.last) + 4} fontSize={11} fill="var(--ink-2)">
                {s.label}
              </text>
            )}
          </g>
        ))}
        {playhead !== undefined && (
          <line x1={x(playhead)} x2={x(playhead)} y1={PAD.top} y2={PAD.top + plotH} stroke="var(--ink)" strokeWidth={1.5} opacity={0.5} />
        )}
        {hover !== null && (
          <line x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={PAD.top + plotH} stroke="var(--ink-3)" strokeWidth={1} />
        )}
      </svg>
      {hover !== null && (
        <div
          className="pointer-events-none absolute z-10 rounded-md border border-line bg-surface px-2 py-1.5 text-xs shadow"
          style={{ left: Math.min(x(hover) + 10, width - 170), top: 4 }}
        >
          <div className="mb-0.5 text-ink-3">{tickLabel ? tickLabel(hover) : `tick ${hover}`}</div>
          {series.map((s) => (
            <div key={s.id} className="flex items-center gap-2">
              <svg width={12} height={6} aria-hidden>
                <line x1={0} x2={12} y1={3} y2={3} stroke={s.color} strokeWidth={2} />
              </svg>
              <span className="font-semibold tabular text-ink">{valueAt(s.points, hover)}</span>
              <span className="text-ink-2">{s.label}</span>
            </div>
          ))}
        </div>
      )}
      {series.length > 1 && (
        <div className="mt-1 flex gap-4 text-[11px] text-ink-2">
          {series.map((s) => (
            <span key={s.id} className="flex items-center gap-1.5">
              <svg width={16} height={6} aria-hidden>
                <line x1={0} x2={16} y1={3} y2={3} stroke={s.color} strokeWidth={2} />
              </svg>
              {s.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
