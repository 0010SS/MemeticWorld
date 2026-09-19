"use client";

import { useMemo } from "react";

import CascadeChart from "@/components/CascadeChart";
import { Highlight, phraseRegex } from "@/components/Highlight";
import { TierBadge } from "@/components/MemeBoard";
import StepChart from "@/components/StepChart";
import type { Gloss, MemeDetail, Persona } from "@/lib/types";

interface Props {
  meme: MemeDetail;
  agents: Persona[];
  colors: Record<string, string>;
  tick: number;
  totalTicks: number;
  dayTicks: { tick: number; day: number }[];
  tickLabel: (tick: number) => string;
  gloss?: Gloss;
  onSeek: (tick: number) => void;
  onClose: () => void;
}

export default function MemeSpotlight({ meme, agents, colors, tick, totalTicks, dayTicks, tickLabel, gloss, onSeek, onClose }: Props) {
  const names = Object.fromEntries(agents.map((a) => [a.id, a.name]));
  const regex = useMemo(() => phraseRegex([meme.phrase]), [meme.phrase]);
  const usersSeries = useMemo(
    () => [
      {
        id: "users",
        label: "agents who have used it",
        color: "var(--ink-2)",
        points: meme.timeline.map((p) => ({ tick: p.tick, value: p.cumulative_users })),
      },
    ],
    [meme],
  );
  const pastUses = meme.uses.filter((u) => u.tick <= tick);
  const confirmedEdges = meme.edges.filter((e) => e.evidence === "memory");

  return (
    <section className="border-t border-line px-4 py-4">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold">“{meme.phrase}”</h2>
            <TierBadge tier={meme.tier} />
          </div>
          <p className="mt-0.5 text-sm text-ink-2">
            First said by <span className="font-semibold">{names[meme.originator ?? ""]}</span> ({meme.uses[0]?.sim_time}),
            picked up by {meme.n_adopters} other{meme.n_adopters === 1 ? "" : "s"} in new conversations
            {confirmedEdges.length > 0 && <>, {confirmedEdges.length} of them with the quote in their retrieved memories</>}.
            {meme.depth > 1 && <> Longest chain: {meme.depth} hops.</>}
          </p>
          {gloss?.gloss && <p className="mt-1 text-sm text-ink">{gloss.gloss}</p>}
          {meme.control && (
            <p className="mt-1 text-xs text-ink-3">
              Control run (no speech memory): {meme.control.n_users} agents used it, {meme.control.n_adopters} adoptions.
            </p>
          )}
        </div>
        <button onClick={onClose} className="rounded-md border border-line px-2 py-1 text-xs text-ink-2 hover:bg-surface-2">
          Close
        </button>
      </div>

      <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-3">Who got it from whom</h3>
      <CascadeChart meme={meme} agents={agents} colors={colors} tick={tick} totalTicks={totalTicks} dayTicks={dayTicks} onSeek={onSeek} />

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div>
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-3">Agents who have used it</h3>
          <StepChart
            series={usersSeries}
            totalTicks={totalTicks}
            yLabel="agents"
            playhead={tick}
            dayTicks={dayTicks}
            tickLabel={tickLabel}
            yMax={agents.length}
            height={150}
          />
          {meme.variants.length > 0 && (
            <div className="mt-3">
              <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-3">Longer forms</h3>
              <ul className="flex flex-wrap gap-1.5 text-xs">
                {meme.variants.map((v) => (
                  <li key={v.phrase} className="rounded-full border border-line px-2 py-0.5 text-ink-2">
                    “{v.phrase}” <span className="tabular text-ink-3">×{v.n_uses}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
        <div>
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-3">
            Uses so far ({pastUses.length} of {meme.uses.length})
          </h3>
          <ul className="scrollbar-thin max-h-56 space-y-1 overflow-y-auto pr-1 text-[13px]">
            {pastUses
              .slice()
              .reverse()
              .map((u) => (
                <li key={u.event_id} className="flex gap-2">
                  <span className="mt-1.5 inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: colors[u.speaker] }} />
                  <span>
                    <button onClick={() => onSeek(u.tick)} className="text-[11px] text-ink-3 tabular hover:underline">
                      {u.sim_time}
                    </button>{" "}
                    <span className="font-semibold">{names[u.speaker]}</span>
                    <span className="text-ink-3"> ({u.role})</span>: <Highlight text={u.text} regex={regex} />
                  </span>
                </li>
              ))}
            {pastUses.length === 0 && <li className="text-ink-3">Not said yet at this point. Scrub forward.</li>}
          </ul>
        </div>
      </div>
    </section>
  );
}
