"use client";

import { useMemo, useRef, useState } from "react";

import { useWidth } from "@/lib/hooks";
import type { Replay } from "@/lib/replay";
import { dayStartTicks } from "@/lib/replay";

interface Props {
  replay: Replay;
  totalTicks: number;
  tick: number;
  onSeek: (tick: number) => void;
  playing: boolean;
  onTogglePlay: () => void;
  speed: number;
  onSpeed: (speed: number) => void;
  live: boolean;
  follow: boolean;
  onFollow: (follow: boolean) => void;
  marks: number[]; // ticks where the selected meme was used
}

const STRIP_H = 40;
const SPEEDS = [1, 2, 4, 8, 16];

export default function Timeline(props: Props) {
  const { replay, totalTicks, tick, onSeek, playing, onTogglePlay, speed, onSpeed, live, follow, onFollow, marks } = props;
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const dragging = useRef(false);

  const counts = useMemo(() => replay.frames.map((f) => (f ? f.utterances.length : 0)), [replay]);
  const maxCount = Math.max(4, ...counts);
  const days = useMemo(() => dayStartTicks(replay), [replay]);
  const span = Math.max(1, totalTicks);
  const barW = width / span;
  const x = (t: number) => (t + 0.5) * barW;
  const frame = replay.frames[tick];
  const hoverFrame = hover !== null ? replay.frames[hover] : undefined;

  const tickAt = (clientX: number, el: HTMLElement) => {
    const r = el.getBoundingClientRect();
    const t = Math.floor(((clientX - r.left) / r.width) * span);
    return Math.max(0, Math.min(replay.maxTick, t));
  };

  return (
    <div className="px-4 pb-3 pt-2">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <button
          onClick={() => onSeek(0)}
          className="rounded-md border border-line px-2 py-1 text-sm hover:bg-surface-2"
          aria-label="Back to start"
        >
          ⏮
        </button>
        <button
          onClick={onTogglePlay}
          className="w-20 rounded-md bg-ink px-3 py-1 text-sm font-medium text-[var(--surface)] hover:opacity-90"
        >
          {playing ? "Pause" : tick >= replay.maxTick && !live ? "Replay" : "Play"}
        </button>
        <label className="flex items-center gap-1 text-xs text-ink-2">
          Speed
          <select
            value={speed}
            onChange={(e) => onSpeed(Number(e.target.value))}
            className="rounded-md border border-line bg-surface px-1 py-1 text-xs"
          >
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s} tick/s
              </option>
            ))}
          </select>
        </label>
        <div className="ml-2 text-sm font-semibold tabular">{frame?.simTime ?? "—"}</div>
        <div className="text-xs text-ink-3 tabular">
          tick {Math.max(0, tick)} / {Math.max(0, replay.maxTick)}
          {totalTicks > replay.maxTick + 1 ? ` (of ${totalTicks})` : ""}
        </div>
        {live && (
          <label className="ml-auto flex items-center gap-1.5 text-xs text-ink-2">
            <input type="checkbox" checked={follow} onChange={(e) => onFollow(e.target.checked)} />
            Follow live
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[var(--good)]" aria-hidden />
          </label>
        )}
      </div>

      <div
        ref={ref}
        className="relative cursor-pointer touch-none"
        style={{ height: STRIP_H + 14 }}
        onPointerDown={(e) => {
          dragging.current = true;
          e.currentTarget.setPointerCapture(e.pointerId);
          onSeek(tickAt(e.clientX, e.currentTarget));
        }}
        onPointerMove={(e) => {
          const t = tickAt(e.clientX, e.currentTarget);
          setHover(t);
          if (dragging.current) onSeek(t);
        }}
        onPointerUp={() => (dragging.current = false)}
        onPointerLeave={() => {
          setHover(null);
          dragging.current = false;
        }}
      >
        <svg width={width} height={STRIP_H + 14} className="block">
          <line x1={0} x2={width} y1={STRIP_H} y2={STRIP_H} stroke="var(--axis)" strokeWidth={1} />
          {counts.map((c, t) =>
            c > 0 ? (
              <rect
                key={t}
                x={t * barW + (barW > 3 ? 0.5 : 0)}
                width={Math.max(1, barW - (barW > 3 ? 1 : 0))}
                y={STRIP_H - (c / maxCount) * (STRIP_H - 6)}
                height={(c / maxCount) * (STRIP_H - 6)}
                fill={t <= tick ? "var(--ink-3)" : "var(--grid)"}
              />
            ) : null,
          )}
          {days.map((d) => (
            <g key={d.day}>
              <line x1={d.tick * barW} x2={d.tick * barW} y1={0} y2={STRIP_H + 14} stroke="var(--axis)" />
              <text x={d.tick * barW + 4} y={STRIP_H + 12} fontSize={10} fill="var(--ink-3)">
                Day {d.day}
              </text>
            </g>
          ))}
          {marks.map((t, i) => (
            <circle key={i} cx={x(t)} cy={5} r={3.5} fill="var(--ink)" stroke="var(--surface)" strokeWidth={1.5} />
          ))}
          <line x1={x(tick)} x2={x(tick)} y1={0} y2={STRIP_H} stroke="var(--ink)" strokeWidth={2} />
          {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={0} y2={STRIP_H} stroke="var(--ink-3)" />}
        </svg>
        {hover !== null && hoverFrame && (
          <div
            className="pointer-events-none absolute -top-9 z-10 whitespace-nowrap rounded-md border border-line bg-surface px-2 py-1 text-xs shadow"
            style={{ left: Math.min(Math.max(0, x(hover) - 70), width - 160) }}
          >
            <span className="font-semibold tabular">{counts[hover]}</span>{" "}
            <span className="text-ink-2">lines · {hoverFrame.simTime}</span>
          </div>
        )}
      </div>
      <input
        type="range"
        min={0}
        max={Math.max(0, replay.maxTick)}
        value={Math.max(0, tick)}
        onChange={(e) => onSeek(Number(e.target.value))}
        className="sr-only"
        aria-label="Simulation time"
      />
      <p className="mt-1 text-[11px] text-ink-3">
        Bars: lines spoken per tick. {marks.length > 0 && "Dots: uses of the selected phrase."} Click or drag to scrub.
      </p>
    </div>
  );
}
