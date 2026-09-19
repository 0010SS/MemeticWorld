// Shapes returned by the FastAPI backend (see backend/app/api/routes.py).

export type EventType = "tick" | "day_start" | "world_event" | "move" | "utterance" | "action" | "run_end";

export interface EventRow {
  id: number;
  run_id: number;
  tick: number;
  sim_time: string;
  type: EventType;
  agent_id: string | null;
  location: string | null;
  text: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: Record<string, any>;
}

export interface Persona {
  id: string;
  name: string;
  role: string;
  personality: string;
  traits: Record<string, number>;
  interests: string[];
  relationships: Record<string, [string, number]>;
  schedule: [string, string, string][];
}

export interface LocationDef {
  name: string;
  description: string;
  rect: [number, number, number, number];
}

export interface WorldLayout {
  locations: LocationDef[];
  edges: [string, string][];
  viewBox: [number, number, number, number];
}

export interface RunStats {
  calls?: number;
  cache_hits?: number;
  failures?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
}

export interface Run {
  id: number;
  name: string;
  condition: "full" | "no_speech_memory";
  seed: number;
  days: number;
  status: "pending" | "running" | "finished" | "stopped" | "failed" | "interrupted";
  current_tick: number;
  total_ticks: number;
  model: string;
  stats: RunStats;
  error: string | null;
  created_at: string;
  updated_at: string;
  live: boolean;
  config?: {
    sim: Record<string, unknown>;
    agents: Persona[];
    world: WorldLayout;
  };
}

export interface MemeUse {
  event_id: number;
  tick: number;
  sim_time: string;
  speaker: string;
  location: string;
  text: string;
  conversation_id: string;
  role: "origin" | "independent" | "echo" | "adoption" | "reuse";
}

export interface MemeEdge {
  source: string;
  target: string;
  tick: number;
  evidence: "memory" | "exposure";
  memory_id: number | null;
  exposure_event_id: number;
  adoption_event_id: number;
}

export interface Spread {
  n_uses: number;
  n_users: number;
  n_adopters: number;
  n_independent: number;
  n_confirmed: number;
}

export type Tier = "strong" | "suggestive" | "baseline";

export interface Meme {
  id: number;
  phrase: string;
  uses: MemeUse[];
  n_uses: number;
  users: string[];
  n_users: number;
  originator: string | null;
  independent: string[];
  adopters: string[];
  n_adopters: number;
  n_confirmed: number;
  edges: MemeEdge[];
  first_tick: number;
  last_tick: number;
  depth: number;
  score: number;
  tier: Tier;
  control: Spread | null;
  variants: { phrase: string; n_uses: number }[];
  referent: string | null;
}

export interface CascadeNode {
  id: string;
  name: string;
  role: "originator" | "independent" | "adopter" | "echo" | "exposed" | "untouched";
  first_use_tick: number | null;
  adoption_tick: number | null;
  exposed_tick: number | null;
}

export interface MemeDetail extends Meme {
  nodes: CascadeNode[];
  timeline: { tick: number; uses: number; cumulative_users: number; cumulative_adopters: number }[];
}

export interface ReferentGroup {
  key: string;
  topic: string | null;
  example_text: string | null;
  meme_ids: number[];
  leader: string;
  leader_share: number;
}

export interface MemeReport {
  memes: Meme[];
  control_run_id: number | null;
  summary: {
    n_memes: number;
    n_strong: number;
    n_baseline: number;
    total_adopters: number;
    total_confirmed: number;
    max_users: number;
    max_depth: number;
  };
  groups: ReferentGroup[];
  curve: { tick: number; adoptions: number }[];
  social: { a: string; b: string; utterances: number }[];
  n_utterances: number;
}

export interface Gloss {
  gloss: string | null;
  refers_to: string | null;
  ordinary_language: boolean | null;
  error?: string;
}

export interface MemoryRow {
  id: number;
  agent_id: string;
  tick: number;
  sim_time: string;
  text: string;
  importance: number;
  source_type: string;
  source_event_id: number | null;
}

export interface DecisionRow {
  id: number;
  tick: number;
  agent_id: string;
  kind: "decide" | "reply";
  prompt: string;
  raw_output: string | null;
  parsed: Record<string, unknown> | null;
  error: string | null;
  retrieved_memories: MemoryRow[];
}

export interface CompareSide {
  run_id: number;
  condition: string;
  summary: MemeReport["summary"];
  curve: { tick: number; adoptions: number }[];
  total_ticks: number;
  n_utterances: number;
}

export interface CompareResult {
  a: CompareSide;
  b: CompareSide;
  rows: { phrase: string; found_in: "a" | "b"; a: Spread; b: Spread }[];
}
