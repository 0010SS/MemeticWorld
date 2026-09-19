"use client";

import { useState } from "react";

import type { NewRun } from "@/lib/api";
import type { Run } from "@/lib/types";

interface Props {
  runs: Run[];
  selectedId: number | null;
  onSelect: (id: number) => void;
  onCreate: (req: NewRun) => Promise<void>;
  onStop: (id: number) => void;
  isMock: boolean;
}

const STATUS_DOT: Record<Run["status"], string> = {
  pending: "var(--ink-3)",
  running: "var(--good)",
  finished: "var(--ink-2)",
  stopped: "var(--ink-3)",
  failed: "var(--critical)",
  interrupted: "var(--critical)",
};

export function runLabel(r: Run) {
  return `#${r.id} · ${r.name}`;
}

export default function RunControls({ runs, selectedId, onSelect, onCreate, onStop, isMock }: Props) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState<NewRun>({
    days: 2,
    seed: 7,
    condition: "full",
    with_control: true,
    tick_delay: isMock ? 0.4 : 0,
    name: "",
  });
  const selected = runs.find((r) => r.id === selectedId);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await onCreate({ ...form, name: form.name || undefined });
      setOpen(false);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="relative flex flex-wrap items-center gap-2">
      <select
        value={selectedId ?? ""}
        onChange={(e) => onSelect(Number(e.target.value))}
        className="max-w-72 rounded-md border border-line bg-surface px-2 py-1.5 text-sm"
        aria-label="Run"
      >
        {runs.length === 0 && <option value="">No runs yet</option>}
        {runs.map((r) => (
          <option key={r.id} value={r.id}>
            {runLabel(r)} · {r.status}
          </option>
        ))}
      </select>

      {selected && (
        <span className="flex items-center gap-1.5 text-xs text-ink-2" title={selected.error ?? ""}>
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: STATUS_DOT[selected.status] }} />
          {selected.status}
          {selected.live && (
            <span className="tabular">
              {" "}
              · tick {selected.current_tick}/{selected.total_ticks} · {selected.stats.calls ?? 0} LLM calls
            </span>
          )}
          {selected.condition === "no_speech_memory" && <span className="rounded bg-surface-3 px-1.5">control</span>}
        </span>
      )}
      {selected?.live && selected.status !== "pending" && (
        <button onClick={() => onStop(selected.id)} className="rounded-md border border-line px-2 py-1 text-xs hover:bg-surface-2">
          Stop
        </button>
      )}

      <button
        onClick={() => setOpen((o) => !o)}
        className="rounded-md bg-ink px-3 py-1.5 text-sm font-medium text-[var(--surface)] hover:opacity-90"
      >
        New run
      </button>

      {open && (
        <div className="absolute right-0 top-11 z-30 w-80 space-y-3 rounded-xl border border-line bg-surface p-4 text-sm shadow-xl">
          <div className="grid grid-cols-2 gap-3">
            <label className="space-y-1">
              <span className="text-xs text-ink-2">Days</span>
              <input
                type="number"
                min={1}
                max={14}
                value={form.days}
                onChange={(e) => setForm({ ...form, days: Number(e.target.value) })}
                className="w-full rounded-md border border-line bg-surface px-2 py-1"
              />
            </label>
            <label className="space-y-1">
              <span className="text-xs text-ink-2">Seed</span>
              <input
                type="number"
                value={form.seed}
                onChange={(e) => setForm({ ...form, seed: Number(e.target.value) })}
                className="w-full rounded-md border border-line bg-surface px-2 py-1"
              />
            </label>
          </div>
          <fieldset className="space-y-1">
            <legend className="text-xs text-ink-2">Condition</legend>
            <label className="flex items-start gap-2">
              <input
                type="radio"
                checked={form.condition === "full"}
                onChange={() => setForm({ ...form, condition: "full" })}
                className="mt-1"
              />
              <span>
                Full <span className="text-xs text-ink-3">agents remember what others said</span>
              </span>
            </label>
            <label className="flex items-start gap-2">
              <input
                type="radio"
                checked={form.condition === "no_speech_memory"}
                onChange={() => setForm({ ...form, condition: "no_speech_memory", with_control: false })}
                className="mt-1"
              />
              <span>
                Control <span className="text-xs text-ink-3">they remember talking, not the words</span>
              </span>
            </label>
          </fieldset>
          {form.condition === "full" && (
            <label className="flex items-start gap-2">
              <input
                type="checkbox"
                checked={form.with_control}
                onChange={(e) => setForm({ ...form, with_control: e.target.checked })}
                className="mt-1"
              />
              <span>
                Also run a control with the same seed{" "}
                <span className="text-xs text-ink-3">(runs after this one, doubles the cost)</span>
              </span>
            </label>
          )}
          <label className="block space-y-1">
            <span className="text-xs text-ink-2">Pause between ticks (s), for watching live</span>
            <input
              type="number"
              min={0}
              max={10}
              step={0.1}
              value={form.tick_delay}
              onChange={(e) => setForm({ ...form, tick_delay: Number(e.target.value) })}
              className="w-full rounded-md border border-line bg-surface px-2 py-1"
            />
          </label>
          <label className="block space-y-1">
            <span className="text-xs text-ink-2">Name (optional)</span>
            <input
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              className="w-full rounded-md border border-line bg-surface px-2 py-1"
            />
          </label>
          {!isMock && (
            <p className="text-xs text-ink-3">
              Uses the configured model. A 2-day run is roughly 1–2k LLM calls; replies are cached per seed.
            </p>
          )}
          {error && <p className="text-xs text-critical">{error}</p>}
          <div className="flex justify-end gap-2">
            <button onClick={() => setOpen(false)} className="rounded-md px-3 py-1.5 text-ink-2 hover:bg-surface-2">
              Cancel
            </button>
            <button
              onClick={submit}
              disabled={busy}
              className="rounded-md bg-ink px-3 py-1.5 font-medium text-[var(--surface)] disabled:opacity-50"
            >
              {busy ? "Starting…" : "Start"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
