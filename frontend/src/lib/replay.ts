import type { EventRow, Persona } from "./types";

/** Everything the UI needs to draw one tick. Built from the event log; the log is the source of truth. */
export interface Frame {
  tick: number;
  simTime: string;
  day: number;
  agents: Record<string, { location: string; activity: string }>;
  utterances: EventRow[];
  worldEvents: EventRow[]; // fired during this tick
  actions: EventRow[];
}

export interface Replay {
  frames: Frame[]; // index = tick, only complete ticks (those with a snapshot)
  maxTick: number; // -1 when nothing has happened yet
  utterances: EventRow[];
  worldEvents: EventRow[];
  ended: boolean;
}

export function buildReplay(events: EventRow[]): Replay {
  const frames: Frame[] = [];
  const utterances: EventRow[] = [];
  const worldEvents: EventRow[] = [];
  let pending: { utterances: EventRow[]; worldEvents: EventRow[]; actions: EventRow[] } = {
    utterances: [],
    worldEvents: [],
    actions: [],
  };
  let ended = false;
  for (const e of events) {
    switch (e.type) {
      case "utterance":
        pending.utterances.push(e);
        utterances.push(e);
        break;
      case "world_event":
        pending.worldEvents.push(e);
        worldEvents.push(e);
        break;
      case "action":
        pending.actions.push(e);
        break;
      case "tick":
        frames[e.tick] = { tick: e.tick, simTime: e.sim_time, day: e.data.day, agents: e.data.agents, ...pending };
        pending = { utterances: [], worldEvents: [], actions: [] };
        break;
      case "run_end":
        ended = true;
        break;
    }
  }
  return { frames, maxTick: frames.length - 1, utterances, worldEvents, ended };
}

export function activeWorldEvents(replay: Replay, tick: number): EventRow[] {
  return replay.worldEvents.filter((e) => e.tick <= tick && tick <= (e.data.until_tick ?? e.tick));
}

/** Fixed agent -> color slot, by the agent's position in the cast (never by rank or activity). */
export function agentColors(agents: Persona[]): Record<string, string> {
  return Object.fromEntries(agents.map((a, i) => [a.id, `var(--agent-${(i % 8) + 1})`]));
}

export function dayStartTicks(replay: Replay): { tick: number; day: number }[] {
  const out: { tick: number; day: number }[] = [];
  let last = 0;
  for (const f of replay.frames) {
    if (f && f.day !== last) {
      out.push({ tick: f.tick, day: f.day });
      last = f.day;
    }
  }
  return out;
}

/** "Day 2, 12:15 PM" -> "12:15 PM" */
export function clockPart(simTime: string): string {
  const i = simTime.indexOf(", ");
  return i >= 0 ? simTime.slice(i + 2) : simTime;
}
