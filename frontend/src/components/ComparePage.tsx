"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import StepChart from "@/components/StepChart";
import ThemeToggle from "@/components/ThemeToggle";
import { api } from "@/lib/api";
import type { CompareResult, Run } from "@/lib/types";

function Stat({ label, a, b }: { label: string; a: number; b: number }) {
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2">
      <div className="text-[11px] text-ink-3">{label}</div>
      <div className="flex items-baseline gap-3">
        <span className="text-2xl font-semibold">{a}</span>
        <span className="text-sm text-ink-2">vs {b} in control</span>
      </div>
    </div>
  );
}

function defaultPair(runs: Run[]): [number | null, number | null] {
  const params = typeof window !== "undefined" ? new URLSearchParams(window.location.search) : null;
  const a = Number(params?.get("a")) || null;
  const b = Number(params?.get("b")) || null;
  if (a && b) return [a, b];
  for (const full of runs.filter((r) => r.condition === "full" && r.current_tick > 0)) {
    const control = runs.find(
      (r) => r.condition === "no_speech_memory" && r.seed === full.seed && r.days === full.days && r.current_tick > 0,
    );
    if (control) return [full.id, control.id];
  }
  return [runs.find((r) => r.condition === "full")?.id ?? null, runs.find((r) => r.condition !== "full")?.id ?? null];
}

export default function ComparePage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [a, setA] = useState<number | null>(null);
  const [b, setB] = useState<number | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api
      .runs()
      .then((list) => {
        setRuns(list);
        const [x, y] = defaultPair(list);
        setA(x);
        setB(y);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (a == null || b == null) return;
    let cancelled = false;
    setLoading(true);
    window.history.replaceState(null, "", `?a=${a}&b=${b}`);
    api
      .compare(a, b)
      .then((r) => {
        if (!cancelled) {
          setResult(r);
          setError(null);
        }
      })
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [a, b]);

  const series = useMemo(() => {
    if (!result) return [];
    return [
      {
        id: "a",
        label: `run #${result.a.run_id} (${result.a.condition === "full" ? "full" : "control"})`,
        color: "var(--ink)",
        points: result.a.curve.map((p) => ({ tick: p.tick, value: p.adoptions })),
      },
      {
        id: "b",
        label: `run #${result.b.run_id} (${result.b.condition === "full" ? "full" : "control"})`,
        color: "var(--ink-3)",
        points: result.b.curve.map((p) => ({ tick: p.tick, value: p.adoptions })),
      },
    ];
  }, [result]);

  const totalTicks = result ? Math.max(result.a.total_ticks, result.b.total_ticks) : 1;
  const dayTicks = useMemo(() => {
    const perDay = 56; // 15-minute ticks from 8:00 to 22:00
    return Array.from({ length: Math.ceil(totalTicks / perDay) }, (_, i) => ({ tick: i * perDay, day: i + 1 }));
  }, [totalTicks]);

  const select = (value: number | null, onChange: (v: number) => void, label: string) => (
    <label className="flex items-center gap-2 text-sm">
      <span className="text-ink-2">{label}</span>
      <select
        value={value ?? ""}
        onChange={(e) => onChange(Number(e.target.value))}
        className="rounded-md border border-line bg-surface px-2 py-1.5"
      >
        {runs.map((r) => (
          <option key={r.id} value={r.id}>
            #{r.id} · {r.name}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <div className="min-h-screen">
      <header className="flex flex-wrap items-center gap-3 border-b border-line bg-surface px-4 py-2.5">
        <Link href="/" className="text-base font-bold">
          MemeticWorld
        </Link>
        <span className="text-sm text-ink-2">Full run vs control</span>
        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto max-w-5xl space-y-6 px-4 py-6">
        <section className="space-y-2">
          <h1 className="text-xl font-semibold">Does memory of what others said make phrases spread?</h1>
          <p className="max-w-3xl text-sm text-ink-2">
            Both runs share a seed, a cast and a schedule. In the <span className="font-semibold">control</span>, agents
            remember that they talked but never the words. Phrases that still &quot;spread&quot; there come from the
            model&apos;s own habits or from obvious descriptions of shared events. The gap between the two runs is the
            transmission effect.
          </p>
          <div className="flex flex-wrap gap-4 pt-2">
            {select(a, setA, "Run")}
            {select(b, setB, "against")}
          </div>
        </section>

        {error && <p className="text-sm text-critical">{error}</p>}
        {loading && <p className="text-sm text-ink-3">Analyzing both runs…</p>}

        {result && (
          <>
            <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Stat label="Candidate phrases" a={result.a.summary.n_memes} b={result.b.summary.n_memes} />
              <Stat label="Adoptions (agent picked up a phrase)" a={result.a.summary.total_adopters} b={result.b.summary.total_adopters} />
              <Stat label="Most agents sharing one phrase" a={result.a.summary.max_users} b={result.b.summary.max_users} />
              <Stat label="Lines spoken" a={result.a.n_utterances} b={result.b.n_utterances} />
            </section>

            <section className="rounded-xl border border-line bg-surface p-4">
              <h2 className="mb-1 text-sm font-semibold">Cumulative adoptions over time</h2>
              <p className="mb-3 text-xs text-ink-3">
                Each step is one agent reusing, in a new conversation, a phrase it had heard from someone else.
              </p>
              <StepChart
                series={series}
                totalTicks={totalTicks}
                yLabel="adoptions"
                dayTicks={dayTicks}
                tickLabel={(t) => `Day ${Math.floor(t / 56) + 1}, tick ${t % 56}`}
                height={220}
              />
            </section>

            <section className="rounded-xl border border-line bg-surface p-4">
              <h2 className="mb-3 text-sm font-semibold">Top phrases from either run, traced in both</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-line text-left text-xs text-ink-3">
                      <th className="py-2 pr-3 font-medium">Phrase</th>
                      <th className="py-2 pr-3 text-right font-medium">Adopters #{result.a.run_id}</th>
                      <th className="py-2 pr-3 text-right font-medium">Adopters #{result.b.run_id}</th>
                      <th className="py-2 pr-3 text-right font-medium">Users #{result.a.run_id}</th>
                      <th className="py-2 pr-3 text-right font-medium">Users #{result.b.run_id}</th>
                      <th className="py-2 text-right font-medium">Independent coiners #{result.a.run_id}</th>
                    </tr>
                  </thead>
                  <tbody className="tabular">
                    {result.rows.map((row) => (
                      <tr key={row.phrase} className="border-b border-line/60">
                        <td className="py-1.5 pr-3">
                          “{row.phrase}” <span className="text-xs text-ink-3">found in #{row.found_in === "a" ? result.a.run_id : result.b.run_id}</span>
                        </td>
                        <td className="py-1.5 pr-3 text-right font-semibold">{row.a.n_adopters}</td>
                        <td className="py-1.5 pr-3 text-right">{row.b.n_adopters}</td>
                        <td className="py-1.5 pr-3 text-right">{row.a.n_users}</td>
                        <td className="py-1.5 pr-3 text-right">{row.b.n_users}</td>
                        <td className="py-1.5 text-right">{row.a.n_independent}</td>
                      </tr>
                    ))}
                    {result.rows.length === 0 && (
                      <tr>
                        <td colSpan={6} className="py-3 text-ink-3">
                          No phrases passed the filters in either run.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}
