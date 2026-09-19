"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import AgentPanel from "@/components/AgentPanel";
import CampusMap from "@/components/CampusMap";
import Feed from "@/components/Feed";
import { phraseRegex } from "@/components/Highlight";
import MemeBoard from "@/components/MemeBoard";
import MemeSpotlight from "@/components/MemeSpotlight";
import RunControls from "@/components/RunControls";
import ThemeToggle from "@/components/ThemeToggle";
import Timeline from "@/components/Timeline";
import { API_URL, api, loadAllEvents, streamEvents, type NewRun, type WorldInfo } from "@/lib/api";
import { activeWorldEvents, agentColors, buildReplay, dayStartTicks } from "@/lib/replay";
import type { EventRow, Gloss, MemeDetail, MemeReport, Run } from "@/lib/types";

type Tab = "memes" | "feed" | "agent";

function pickDefaultRun(runs: Run[]): number | null {
  const fromUrl = typeof window !== "undefined" ? Number(new URLSearchParams(window.location.search).get("run")) : 0;
  if (fromUrl && runs.some((r) => r.id === fromUrl)) return fromUrl;
  const preferred = runs.find((r) => r.condition === "full" && (r.status === "finished" || r.live));
  return (preferred ?? runs[0])?.id ?? null;
}

export default function Dashboard() {
  const [info, setInfo] = useState<WorldInfo | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [runId, setRunId] = useState<number | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [events, setEvents] = useState<EventRow[]>([]);
  const [loadingRun, setLoadingRun] = useState(false);
  const [tick, setTick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [follow, setFollow] = useState(true);
  const [report, setReport] = useState<MemeReport | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [memePhrase, setMemePhrase] = useState<string | null>(null);
  const [memeDetail, setMemeDetail] = useState<MemeDetail | null>(null);
  const [agentId, setAgentId] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("memes");
  const [glosses, setGlosses] = useState<Record<string, Gloss>>({});
  const [glossing, setGlossing] = useState(false);

  // --- loading ------------------------------------------------------------------------------

  const refreshRuns = useCallback(async () => {
    try {
      const list = await api.runs();
      setRuns(list);
      setBackendError(null);
      setRun((prev) => {
        const fresh = prev && list.find((r) => r.id === prev.id);
        return fresh ? { ...prev, ...fresh } : prev;
      });
      return list;
    } catch (e) {
      setBackendError(String(e));
      return [];
    }
  }, []);

  useEffect(() => {
    api
      .world()
      .then(setInfo)
      .catch((e) => setBackendError(String(e)));
    refreshRuns().then((list) => setRunId((prev) => prev ?? pickDefaultRun(list)));
  }, [refreshRuns]);

  const loadReport = useCallback(async (id: number) => {
    setReportLoading(true);
    try {
      setReport(await api.memes(id));
    } catch {
      /* analysis is best-effort while a run is young */
    } finally {
      setReportLoading(false);
    }
  }, []);

  // Load a run: its config, the whole event log, then (if still running) stream the rest.
  useEffect(() => {
    if (runId == null) return;
    let cancelled = false;
    let stopStream: (() => void) | null = null;
    setLoadingRun(true);
    setEvents([]);
    setReport(null);
    setMemePhrase(null);
    setMemeDetail(null);
    setGlosses({});
    setPlaying(false);
    (async () => {
      const r = await api.run(runId);
      if (cancelled) return;
      setRun(r);
      const log = await loadAllEvents(runId);
      if (cancelled) return;
      setEvents(log);
      setTick(Math.max(0, buildReplay(log).maxTick));
      setLoadingRun(false);
      loadReport(runId);
      if (r.live) {
        stopStream = streamEvents(
          runId,
          log.length ? log[log.length - 1].id : 0,
          (batch) => setEvents((prev) => [...prev, ...batch]),
          () => {
            refreshRuns();
            loadReport(runId);
          },
        );
      }
    })().catch((e) => {
      if (!cancelled) {
        setBackendError(String(e));
        setLoadingRun(false);
      }
    });
    return () => {
      cancelled = true;
      stopStream?.();
    };
  }, [runId, loadReport, refreshRuns]);

  // Poll the run list while anything is running; refresh the analysis every so often.
  const anyLive = runs.some((r) => r.live);
  useEffect(() => {
    if (!anyLive) return;
    const id = setInterval(refreshRuns, 4000);
    return () => clearInterval(id);
  }, [anyLive, refreshRuns]);

  useEffect(() => {
    if (runId == null || !run?.live) return;
    const id = setInterval(() => loadReport(runId), 15000);
    return () => clearInterval(id);
  }, [runId, run?.live, loadReport]);

  // --- replay -------------------------------------------------------------------------------

  const replay = useMemo(() => buildReplay(events), [events]);
  const agents = useMemo(() => run?.config?.agents ?? info?.agents ?? [], [run, info]);
  const world = run?.config?.world ?? info?.world;
  const colors = useMemo(() => agentColors(agents), [agents]);
  const days = useMemo(() => dayStartTicks(replay), [replay]);
  const totalTicks = run?.total_ticks ?? Math.max(1, replay.maxTick + 1);
  const live = !!run?.live;

  useEffect(() => {
    if (live && follow && !playing) setTick(Math.max(0, replay.maxTick));
  }, [live, follow, playing, replay.maxTick]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setTick((t) => Math.min(t + 1, replay.maxTick)), 1000 / speed);
    return () => clearInterval(id);
  }, [playing, speed, replay.maxTick]);

  useEffect(() => {
    if (playing && tick >= replay.maxTick && !live) setPlaying(false);
  }, [playing, tick, replay.maxTick, live]);

  const togglePlay = () => {
    if (playing) return setPlaying(false);
    if (tick >= replay.maxTick && !live) setTick(0);
    setFollow(false);
    setPlaying(true);
  };

  const seek = (t: number) => {
    setTick(Math.max(0, Math.min(t, replay.maxTick)));
    if (live && t < replay.maxTick) setFollow(false);
  };

  // --- memes --------------------------------------------------------------------------------

  // ?meme=<phrase> selects a phrase on load, and the URL follows the selection (shareable links).
  const initialMeme = useRef<string | null>(
    typeof window !== "undefined" ? new URLSearchParams(window.location.search).get("meme") : null,
  );
  useEffect(() => {
    const wanted = initialMeme.current;
    if (!report || !wanted) return;
    initialMeme.current = null;
    if (report.memes.some((m) => m.phrase === wanted)) setMemePhrase(wanted);
  }, [report]);
  useEffect(() => {
    if (runId == null) return;
    const params = new URLSearchParams({ run: String(runId) });
    if (memePhrase) params.set("meme", memePhrase);
    window.history.replaceState(null, "", `?${params}`);
  }, [runId, memePhrase]);

  const memeId = report?.memes.find((m) => m.phrase === memePhrase)?.id ?? null;
  useEffect(() => {
    if (runId == null || memeId == null) {
      setMemeDetail(null);
      return;
    }
    let cancelled = false;
    api.meme(runId, memeId).then((d) => !cancelled && setMemeDetail(d)).catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [runId, memeId, report]);

  const allPhrases = useMemo(() => phraseRegex(report?.memes.map((m) => m.phrase) ?? []), [report]);
  const selectedPhrase = useMemo(() => (memePhrase ? phraseRegex([memePhrase]) : null), [memePhrase]);

  const explain = async () => {
    if (runId == null) return;
    setGlossing(true);
    try {
      setGlosses(await api.gloss(runId));
    } catch (e) {
      setBackendError(String(e));
    } finally {
      setGlossing(false);
    }
  };

  const createRun = async (req: NewRun) => {
    const created = await api.createRun(req);
    await refreshRuns();
    setRunId(created.run_ids[0]);
  };

  const stopRun = async (id: number) => {
    try {
      await api.stopRun(id);
      refreshRuns();
    } catch (e) {
      setBackendError(String(e));
    }
  };

  const frame = replay.frames[tick];
  const active = useMemo(() => activeWorldEvents(replay, tick), [replay, tick]);
  const selectedAgent = agents.find((a) => a.id === agentId) ?? null;
  const tickLabel = (t: number) => replay.frames[t]?.simTime ?? `tick ${t}`;
  const transitionMs = playing ? Math.max(120, Math.min(800, 900 / speed)) : 450;
  const memeMarks = memeDetail ? memeDetail.uses.map((u) => u.tick) : [];

  // --- render -------------------------------------------------------------------------------

  return (
    <div className="flex min-h-screen flex-col lg:h-screen">
      <header className="flex flex-wrap items-center gap-3 border-b border-line bg-surface px-4 py-2.5">
        <div className="mr-2">
          <h1 className="text-base font-bold leading-tight">MemeticWorld</h1>
          <p className="text-[11px] text-ink-3">Do memes emerge from ordinary conversation between AI agents?</p>
        </div>
        <RunControls
          runs={runs}
          selectedId={runId}
          onSelect={setRunId}
          onCreate={createRun}
          onStop={stopRun}
          isMock={info?.llm_provider === "mock"}
        />
        <div className="ml-auto flex items-center gap-2">
          {info && (
            <span className="rounded-md bg-surface-2 px-2 py-1 text-[11px] text-ink-2" title="Configured in .env">
              model: {info.model}
              {info.llm_provider === "mock" && " (offline mock)"}
            </span>
          )}
          <Link href="/compare" className="rounded-md border border-line px-2 py-1.5 text-xs text-ink-2 hover:bg-surface-2">
            Full vs control
          </Link>
          <ThemeToggle />
        </div>
      </header>

      {backendError && (
        <div className="border-b border-line bg-mark px-4 py-2 text-sm text-mark-ink">
          Can&apos;t reach the backend at <code>{API_URL}</code> ({backendError}). Start it with{" "}
          <code>cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000</code>.
        </div>
      )}
      {run?.status === "failed" && run.error && (
        <div className="border-b border-line bg-surface-2 px-4 py-2 text-sm">
          <span className="font-semibold text-critical">Run failed:</span> {run.error}
        </div>
      )}

      {!world || (runs.length === 0 && !backendError) ? (
        <div className="flex flex-1 items-center justify-center p-8 text-center">
          <div className="max-w-md space-y-2">
            <h2 className="text-lg font-semibold">No runs yet</h2>
            <p className="text-sm text-ink-2">
              Start one with <span className="font-semibold">New run</span>. Eight students go about two days on campus;
              every conversation is logged, and the analysis looks for phrases that spread from one agent to others.
            </p>
          </div>
        </div>
      ) : (
        <main className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_420px]">
          <section className="scrollbar-thin min-h-0 overflow-y-auto">
            <div className="relative px-4 pt-3">
              {loadingRun && <div className="absolute right-6 top-5 text-xs text-ink-3">Loading run…</div>}
              <CampusMap
                world={world}
                agents={agents}
                colors={colors}
                frame={frame}
                activeEvents={active}
                selectedAgent={agentId}
                onSelectAgent={(id) => {
                  setAgentId(id);
                  if (id) setTab("agent");
                }}
                phraseRegex={allPhrases}
                transitionMs={transitionMs}
              />
            </div>
            <Timeline
              replay={replay}
              totalTicks={totalTicks}
              tick={tick}
              onSeek={seek}
              playing={playing}
              onTogglePlay={togglePlay}
              speed={speed}
              onSpeed={setSpeed}
              live={live}
              follow={follow}
              onFollow={setFollow}
              marks={memeMarks}
            />
            {memeDetail && memeDetail.phrase === memePhrase && (
              <MemeSpotlight
                meme={memeDetail}
                agents={agents}
                colors={colors}
                tick={tick}
                totalTicks={totalTicks}
                dayTicks={days}
                tickLabel={tickLabel}
                gloss={glosses[memeDetail.phrase]}
                onSeek={seek}
                onClose={() => setMemePhrase(null)}
              />
            )}
            {!memePhrase && report && report.memes.length > 0 && (
              <p className="px-4 pb-4 text-xs text-ink-3">
                Pick a phrase in the <span className="font-semibold">Phrases</span> panel to see who picked it up from whom.
              </p>
            )}
          </section>

          <aside className="flex min-h-[70vh] flex-col border-t border-line bg-page lg:min-h-0 lg:border-l lg:border-t-0">
            <nav className="flex border-b border-line bg-surface text-sm">
              {(
                [
                  ["memes", "Phrases"],
                  ["feed", "Conversations"],
                  ["agent", selectedAgent ? selectedAgent.name : "Agent"],
                ] as [Tab, string][]
              ).map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => setTab(key)}
                  className={`flex-1 border-b-2 px-3 py-2.5 ${
                    tab === key ? "border-ink font-semibold" : "border-transparent text-ink-2 hover:bg-surface-2"
                  }`}
                >
                  {label}
                </button>
              ))}
            </nav>
            <div className="min-h-0 flex-1">
              {tab === "memes" && run && (
                <MemeBoard
                  report={report}
                  loading={reportLoading}
                  agents={agents}
                  colors={colors}
                  selectedMemeId={memeId}
                  onSelect={(id) => setMemePhrase(report?.memes.find((m) => m.id === id)?.phrase ?? null)}
                  runId={run.id}
                  glosses={glosses}
                  onGloss={explain}
                  glossing={glossing}
                  tick={tick}
                />
              )}
              {tab === "feed" && (
                <Feed
                  replay={replay}
                  tick={tick}
                  agents={agents}
                  colors={colors}
                  regex={allPhrases}
                  strongRegex={selectedPhrase}
                  selectedAgent={agentId}
                />
              )}
              {tab === "agent" &&
                (selectedAgent && run ? (
                  <AgentPanel
                    runId={run.id}
                    agent={selectedAgent}
                    agents={agents}
                    colors={colors}
                    frame={frame}
                    tick={tick}
                    regex={allPhrases}
                  />
                ) : (
                  <p className="p-4 text-sm text-ink-3">Click an agent on the map to see what they saw, remembered and decided.</p>
                ))}
            </div>
          </aside>
        </main>
      )}
    </div>
  );
}
