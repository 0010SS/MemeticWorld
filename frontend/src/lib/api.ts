import type {
  CompareResult,
  DecisionRow,
  EventRow,
  Gloss,
  MemeDetail,
  MemeReport,
  MemoryRow,
  Persona,
  Run,
  WorldLayout,
} from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export interface WorldInfo {
  world: WorldLayout;
  agents: Persona[];
  conditions: Record<string, string>;
  model: string;
  llm_provider: string;
}

export interface NewRun {
  name?: string;
  days: number;
  condition: "full" | "no_speech_memory";
  seed: number;
  tick_delay: number;
  with_control: boolean;
}

export const api = {
  world: () => request<WorldInfo>("/api/world"),
  runs: () => request<Run[]>("/api/runs"),
  run: (id: number) => request<Run>(`/api/runs/${id}`),
  createRun: (body: NewRun) =>
    request<{ run_ids: number[]; runs: Run[] }>("/api/runs", { method: "POST", body: JSON.stringify(body) }),
  stopRun: (id: number) => request<{ ok: boolean }>(`/api/runs/${id}/stop`, { method: "POST" }),
  deleteRun: (id: number) => request<{ ok: boolean }>(`/api/runs/${id}`, { method: "DELETE" }),
  events: (id: number, afterId = 0, limit = 20000) =>
    request<EventRow[]>(`/api/runs/${id}/events?after_id=${afterId}&limit=${limit}`),
  memes: (id: number) => request<MemeReport>(`/api/runs/${id}/memes`),
  meme: (id: number, memeId: number) => request<MemeDetail>(`/api/runs/${id}/memes/${memeId}`),
  gloss: (id: number, top = 8) =>
    request<Record<string, Gloss>>(`/api/runs/${id}/memes/gloss?top=${top}`, { method: "POST" }),
  memories: (id: number, agent: string, maxTick: number, limit = 40) =>
    request<MemoryRow[]>(`/api/runs/${id}/agents/${agent}/memories?max_tick=${maxTick}&limit=${limit}`),
  decisions: (id: number, agent: string, maxTick: number, limit = 1) =>
    request<DecisionRow[]>(`/api/runs/${id}/agents/${agent}/decisions?max_tick=${maxTick}&limit=${limit}`),
  compare: (a: number, b: number) => request<CompareResult>(`/api/compare?a=${a}&b=${b}`),
};

/** Load the whole replay log in pages. */
export async function loadAllEvents(id: number): Promise<EventRow[]> {
  const out: EventRow[] = [];
  const page = 20000;
  for (;;) {
    const batch = await api.events(id, out.length ? out[out.length - 1].id : 0, page);
    out.push(...batch);
    if (batch.length < page) return out;
  }
}

export function streamEvents(
  id: number,
  afterId: number,
  onEvents: (events: EventRow[]) => void,
  onEnd: () => void,
): () => void {
  const source = new EventSource(`${API_URL}/api/runs/${id}/stream?after_id=${afterId}`);
  let buffer: EventRow[] = [];
  const flush = setInterval(() => {
    if (buffer.length) {
      onEvents(buffer);
      buffer = [];
    }
  }, 400);
  source.onmessage = (msg) => buffer.push(JSON.parse(msg.data) as EventRow);
  source.addEventListener("end", () => {
    if (buffer.length) onEvents(buffer);
    buffer = [];
    source.close();
    clearInterval(flush);
    onEnd();
  });
  return () => {
    source.close();
    clearInterval(flush);
  };
}
