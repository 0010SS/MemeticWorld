"use client";

import Link from "next/link";

import type { Gloss, Meme, MemeReport, Persona, Tier } from "@/lib/types";

interface Props {
  report: MemeReport | null;
  loading: boolean;
  agents: Persona[];
  colors: Record<string, string>;
  selectedMemeId: number | null;
  onSelect: (id: number | null) => void;
  runId: number;
  glosses: Record<string, Gloss>;
  onGloss: () => void;
  glossing: boolean;
  tick: number;
}

export const TIER_INFO: Record<Tier, { icon: string; label: string; help: string }> = {
  strong: {
    icon: "●",
    label: "Strong",
    help: "One originator, at least one adoption where the adopter's prompt held a memory quoting it, and no spread in the control run.",
  },
  suggestive: {
    icon: "◐",
    label: "Suggestive",
    help: "Spread to others after exposure, but without memory-confirmed adoption (or with a second independent coiner).",
  },
  baseline: {
    icon: "○",
    label: "Baseline",
    help: "Also spreads in the control run where agents can't remember speech: probably ordinary language or a shared model habit.",
  },
};

export function TierBadge({ tier }: { tier: Tier }) {
  const info = TIER_INFO[tier];
  return (
    <span title={info.help} className="inline-flex items-center gap-1 rounded-full border border-line px-2 py-0.5 text-[11px] text-ink-2">
      <span aria-hidden>{info.icon}</span>
      {info.label}
    </span>
  );
}

function Stat({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2">
      <div className="text-[11px] text-ink-3">{label}</div>
      <div className="text-xl font-semibold">{value}</div>
      {sub && <div className="text-[11px] text-ink-3">{sub}</div>}
    </div>
  );
}

function UsersMeter({ meme, agents, colors, tick }: { meme: Meme; agents: Persona[]; colors: Record<string, string>; tick: number }) {
  const firstUse = new Map<string, number>();
  for (const u of meme.uses) if (!firstUse.has(u.speaker)) firstUse.set(u.speaker, u.tick);
  return (
    <div className="flex gap-[2px]" aria-label={`${meme.n_users} of ${agents.length} agents have used it`}>
      {agents.map((a) => {
        const t = firstUse.get(a.id);
        const used = t !== undefined && t <= tick;
        return (
          <span
            key={a.id}
            title={`${a.name}${t === undefined ? " never used it" : used ? " has used it" : " uses it later"}`}
            className="h-2 w-3 rounded-sm"
            style={{ background: used ? colors[a.id] : "var(--surface-3)" }}
          />
        );
      })}
    </div>
  );
}

export default function MemeBoard(props: Props) {
  const { report, loading, agents, colors, selectedMemeId, onSelect, runId, glosses, onGloss, glossing, tick } = props;
  const names = Object.fromEntries(agents.map((a) => [a.id, a.name]));

  if (!report) {
    return <p className="p-4 text-sm text-ink-3">{loading ? "Analyzing the conversation log…" : "No analysis yet."}</p>;
  }
  const { summary } = report;
  const byId = new Map(report.memes.map((m) => [m.id, m]));

  return (
    <div className="scrollbar-thin h-full space-y-4 overflow-y-auto p-3">
      <div className="grid grid-cols-2 gap-2">
        <Stat label="Candidate phrases" value={summary.n_memes} sub={`${report.n_utterances.toLocaleString()} lines analyzed`} />
        <Stat label="Strong evidence" value={summary.n_strong} sub="memory-confirmed, single origin" />
        <Stat
          label="Adoptions"
          value={summary.total_adopters}
          sub={`${summary.total_confirmed} confirmed via retrieved memory`}
        />
        <Stat
          label="Control baseline"
          value={report.control_run_id ? `run #${report.control_run_id}` : "none"}
          sub={report.control_run_id ? `${summary.n_baseline} phrases demoted` : "start a run with a control to baseline"}
        />
      </div>
      {report.control_run_id && (
        <Link href={`/compare?a=${runId}&b=${report.control_run_id}`} className="block text-xs text-ink-2 underline">
          Compare with the control run →
        </Link>
      )}

      <section>
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-semibold">Detected phrases</h3>
          <button
            onClick={onGloss}
            disabled={glossing || report.memes.length === 0}
            className="rounded-md border border-line px-2 py-1 text-xs text-ink-2 hover:bg-surface-2 disabled:opacity-50"
            title="Ask the analysis LLM what each top phrase means. Agents never see this."
          >
            {glossing ? "Explaining…" : "Explain top phrases"}
          </button>
        </div>
        {report.memes.length === 0 && (
          <p className="text-sm text-ink-3">
            Nothing has spread yet. Phrases show up here once someone reuses a phrase they heard from someone else,
            in a different conversation.
          </p>
        )}
        <ul className="space-y-2">
          {report.memes.map((m) => {
            const gloss = glosses[m.phrase];
            const selected = m.id === selectedMemeId;
            const notYet = m.first_tick > tick;
            return (
              <li key={m.id}>
                <button
                  onClick={() => onSelect(selected ? null : m.id)}
                  className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                    selected ? "border-ink bg-surface-2" : "border-line bg-surface hover:bg-surface-2"
                  } ${notYet ? "opacity-50" : ""}`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="font-semibold">“{m.phrase}”</span>
                    <TierBadge tier={m.tier} />
                  </div>
                  <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-2">
                    <span className="flex items-center gap-1">
                      <span className="inline-block h-2 w-2 rounded-full" style={{ background: colors[m.originator ?? ""] }} />
                      from {names[m.originator ?? ""] ?? "?"}
                    </span>
                    <span className="tabular">
                      {m.n_adopters} adopter{m.n_adopters === 1 ? "" : "s"}
                      {m.n_confirmed > 0 && ` (${m.n_confirmed} confirmed)`}
                    </span>
                    <span className="tabular">{m.n_uses} uses</span>
                    {m.control && <span className="tabular text-ink-3">control: {m.control.n_users} users</span>}
                  </div>
                  <div className="mt-1.5 flex items-center justify-between">
                    <UsersMeter meme={m} agents={agents} colors={colors} tick={tick} />
                    <span className="text-[11px] text-ink-3">{m.uses[0]?.sim_time}</span>
                  </div>
                  {gloss?.gloss && (
                    <p className="mt-1.5 text-xs text-ink-2">
                      {gloss.gloss}
                      {gloss.ordinary_language && <span className="text-ink-3"> (analyst: ordinary language)</span>}
                    </p>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      </section>

      {report.groups.length > 0 && (
        <section>
          <h3 className="mb-1 text-sm font-semibold">Competing names for the same incident</h3>
          <p className="mb-2 text-xs text-ink-3">
            Convergence means one phrasing takes most of the references. Share is by number of uses.
          </p>
          <ul className="space-y-3">
            {report.groups.map((g) => {
              const members = g.meme_ids.map((id) => byId.get(id)).filter(Boolean) as Meme[];
              const total = members.reduce((s, m) => s + m.n_uses, 0) || 1;
              return (
                <li key={g.key} className="rounded-lg border border-line bg-surface px-3 py-2">
                  <div className="mb-1.5 text-xs text-ink-2" title={g.example_text ?? ""}>
                    {g.example_text ? `${g.example_text.slice(0, 80)}${g.example_text.length > 80 ? "…" : ""}` : g.key}
                  </div>
                  <div className="flex h-2.5 w-full gap-[2px] overflow-hidden rounded">
                    {members.map((m, i) => (
                      <span
                        key={m.id}
                        title={`“${m.phrase}”: ${m.n_uses} uses`}
                        className="h-full"
                        style={{
                          width: `${(m.n_uses / total) * 100}%`,
                          background: i === 0 ? "var(--ink-2)" : "var(--axis)",
                        }}
                      />
                    ))}
                  </div>
                  <ul className="mt-1.5 space-y-0.5 text-xs">
                    {members.map((m, i) => (
                      <li key={m.id} className="flex items-center justify-between">
                        <span className="flex items-center gap-1.5">
                          <span
                            className="inline-block h-2 w-3 rounded-sm"
                            style={{ background: i === 0 ? "var(--ink-2)" : "var(--axis)" }}
                          />
                          “{m.phrase}”
                        </span>
                        <span className="tabular text-ink-2">{Math.round((m.n_uses / total) * 100)}%</span>
                      </li>
                    ))}
                  </ul>
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
}
