# MemeticWorld

A hackathon MVP. Eight LLM student agents live on a small campus, follow schedules, run into each
other, talk, and remember. Everything is logged. A separate analysis layer looks for phrases that
one agent coins and others later reuse, and shows how they spread. The research question:
**do memes, slang, and in-jokes emerge from ordinary interaction, without anyone asking for them?**

## The one rule that matters: never prompt for memes

The whole result is worthless if agents are told to produce the phenomenon we measure.

- Agent-facing text (system prompt, decision/reply prompts, personas, relationship labels,
  schedule activities, location descriptions, event text) must never ask for or hint at slang,
  memes, catchphrases, nicknames, running jokes, "conventions", virality, or "creative language".
- **No example utterances in prompts.** Any example line gets copied by every agent and then
  shows up in the analysis as a fake meme. JSON formats describe fields in words only.
- Personas describe temperament ("quick to laugh", "deadpan") and interests, nothing memetic.
- Event text is plain description. Don't hand agents a ready-made catchphrase.
- `backend/tests/test_prompts.py` renders every agent-facing string and fails on banned terms.
  If it fails, **rephrase the prompt; never weaken the banned list.**
- The analysis LLM prompt (`analysis/semantics.py`, "gloss") may talk about memes freely because
  agents never see it. Keep that separation.

## Run it

```bash
# backend (Python 3.11+; uv or plain venv both fine)
cd backend
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/uvicorn app.main:app --reload --port 8000

# frontend (Node 20+)
cd frontend
npm install
npm run dev            # http://localhost:3000, talks to http://localhost:8000

# tests (always offline: mock LLM, hash embeddings, temp DB)
cd backend && .venv/bin/python -m pytest -q

# headless runs and analysis (same SQLite file as the API, so CLI runs show up in the UI)
cd backend
.venv/bin/python -m app.cli run --days 2 --seed 7 --control   # full run, then same-seed control
.venv/bin/python -m app.cli list
.venv/bin/python -m app.cli memes <run_id>
.venv/bin/python -m app.cli trace <run_id> "some phrase"
.venv/bin/python -m app.cli compare <full_id> <control_id>
```

Config lives in a repo-root `.env` (copy `.env.example`). **With no API key the backend runs in
mock mode**: a rule-based fake LLM plus hash embeddings, no network. Use it for UI and
pipeline work. Mock runs say nothing about real LLM behavior; never present them as results.
Any OpenAI-compatible endpoint works (OpenAI, OpenRouter, Ollama, vLLM, LM Studio). If the
provider has no `/embeddings`, set `EMBED_PROVIDER=hash`.

## Architecture

```
backend/app/
  main.py                 FastAPI app (CORS, router)
  config.py               env -> Settings (the only place env vars are read)
  cli.py                  headless run / list / memes / trace / compare
  api/routes.py           REST + SSE under /api
  db/database.py          SQLite schema + plain query functions (all SQL lives here or in analysis loaders)
  llm/client.py           OpenAI-compatible chat + embeddings, retries, SQLite cache, JSON repair
  llm/mock.py             offline fake LLM (test double only)
  llm/prompts.py          every agent-facing prompt template (see the rule above)
  memory/store.py         per-agent memory stream; batched embed + persist once per tick
  memory/retrieval.py     score = 0.6 similarity + 0.3 recency + 0.1 importance, top 5
  simulation/world.py     Simulation: the tick loop, perception, conversations, conditions
  simulation/agent.py     Agent dataclass + pydantic Decision/Reply (validation of model output)
  simulation/personas.py  the 8-student cast, schedules, starting relationships
  simulation/campus.py    location graph + SVG layout for the map
  simulation/events.py    recurring world incidents (tray drop, squirrel, printer jam...)
  simulation/scheduler.py clock, schedule lookup, BFS next hop
  simulation/runner.py    create/execute runs (shared by API and CLI), stop, liveness
  analysis/meme_detector.py  n-gram candidates, exposure/adoption tracing, tiers, control baselining
  analysis/semantics.py      referents (which incident a phrase is about), analyst-LLM glosses
  analysis/propagation.py    cascades, adoption curves, social graph, full-vs-control comparison
  analysis/pipeline.py       analyze_run(): load -> pair with control -> detect -> referents
frontend/src/
  components/Dashboard.tsx   page state: run loading, SSE streaming, playback, selection
  components/CampusMap.tsx   SVG campus, agents, speech bubbles, events
  components/Timeline.tsx    scrubber with per-tick activity strip
  components/MemeBoard.tsx   phrase leaderboard, competing names per incident
  components/MemeSpotlight.tsx + CascadeChart.tsx + StepChart.tsx   who got it from whom
  components/AgentPanel.tsx  memories, latest prompt, retrieved memories, raw model output
  components/ComparePage.tsx full vs control
  lib/replay.ts              event log -> per-tick frames (the UI derives everything from the log)
  lib/api.ts, lib/types.ts   backend client + response types (keep in sync with routes.py)
```

### One tick (15 campus minutes; days run 08:00–22:00)

1. World events fire at their locations (`events.py`).
2. Each agent perceives **only its own location**: who is here, what they're visibly doing, what's
   happening. New events and newly seen people become memories.
3. Agents that are alone with nothing new follow their schedule on autopilot (no LLM call). The rest
   embed a query, retrieve ~5 memories, and make one LLM call returning
   `{action: MOVE|TALK|REACT|CONTINUE_ACTIVITY|IDLE, target, activity, utterance, reason, importance}`.
   Bad JSON gets one repair retry, then falls back to the schedule.
4. Resolve: moves first (one graph edge per tick), then TALKs pair into conversations in random
   order (a busy/absent target turns the line into a REACT), then REACTs. Conversations are up to
   4 turns, and **each turn is a separate LLM call with only that speaker's memories**. Never
   generate both sides in one call, because that fakes transmission.
5. Memories are embedded and saved in one batch; a `tick` snapshot event is logged.

### Conditions (the experiment)

- `full`: speech is stored verbatim in memory (`Maya said to me at the Quad: "..."`). This quote
  is the **only** channel a phrase can travel through. Never paraphrase it.
- `no_speech_memory` (control): agents remember *that* they talked, never *what* was said.
  Same seed means the same events and the same starting point.

### How the detector reads a phrase

For each 1–4-gram (stopword-bounded, not made only of world vocabulary):
exposure = heard it from someone (addressee or overhearer); **originator** = first user with no
prior exposure; **echo** = repeated in the same conversation (not adoption); **adopter** = used
it in a *different* conversation after exposure; **confirmed** = the adopting utterance's
prompt contained a retrieved memory quoting the phrase (causal evidence via logged
`memory_ids`). Tiers: `strong` (single origin + confirmed adoption + no spread in control),
`suggestive`, `baseline` (also spreads in the same-seed control run, so it's probably a model habit).
The API pairs a full run with the latest same-seed, same-length control automatically.

## Data model (SQLite, `backend/data/memeticworld.db`)

- `runs`: config snapshot (cast, world, sim settings), status, progress, LLM stats, heartbeat.
- `events`: **the replay log and source of truth.** Types: `tick` (all agent positions and
  activities), `day_start`, `world_event`, `move`, `utterance` (speaker, text, conversation_id,
  audience, overheard_by, memory_ids retrieved for it), `action` (LLM decision + reason), `run_end`.
- `memories`: per-agent, with `source_type` (event/sighting/heard/overheard/said/conversation),
  `source_event_id` (provenance), embedding blob.
- `decisions`: every LLM call: full prompt, retrieved memory ids, raw output, parsed JSON, error.
- `llm_cache`, `embedding_cache`: reruns with the same prompts and seed are free.

The frontend never stores derived state. It rebuilds frames from `events`. If you need
something new in the UI, log it as event data rather than adding a side table.

## API (all under `/api`)

`GET health`, `GET world`, `GET runs`, `POST runs {days, seed, condition, tick_delay, with_control}`,
`GET runs/{id}`, `POST runs/{id}/stop`, `DELETE runs/{id}`, `GET runs/{id}/events?after_id=`,
`GET runs/{id}/stream?after_id=` (SSE; ends with `event: end`),
`GET runs/{id}/agents/{agent}/memories?max_tick=`, `GET runs/{id}/agents/{agent}/decisions?max_tick=`,
`GET runs/{id}/memes`, `GET runs/{id}/memes/{meme_id}`, `GET runs/{id}/phrase?q=`,
`POST runs/{id}/memes/gloss`, `GET compare?a=&b=`.
If you change a response shape, update `frontend/src/lib/types.ts` in the same change.

## Common changes

- **Add/edit an agent**: `simulation/personas.py`. Keep it generic, keep schedules overlapping
  (shared lecture, lunch, evenings). Locations must exist in `campus.py`. The UI has 8 validated
  colors; a 9th agent needs a color decision first (`globals.css`).
- **Add a location**: `campus.py` (description, rect in the 1000×640 map, edges). Check that the
  graph stays connected (`tests/test_world.py`).
- **Add a world event**: `events.py`. Short, concrete, recurring, a little absurd. Plain
  description only.
- **Change a prompt**: `llm/prompts.py`, then run `pytest tests/test_prompts.py`. Changing prompts
  invalidates the LLM cache (the key includes the prompt). That's expected.
- **Tune the detector**: constants at the top of `meme_detector.py`. Add a case to
  `tests/test_detector.py` with known ground truth before changing behavior.
- **New experimental condition**: add to `CONDITIONS` in `world.py` and branch on it where
  memories are written. Keep everything else identical so comparisons stay fair.

## Conventions

- Python: plain functions and dataclasses, no frameworks beyond FastAPI/pydantic. SQL lives in
  `db/database.py` (analysis loaders are the one exception). All env vars go through `config.py`.
  Keep modules small and don't add abstraction layers.
- Every model output goes through a pydantic model (`Decision`, `Reply`, `Gloss`). Never trust raw JSON.
- Determinism: all randomness uses seeded `random.Random` instances owned by the simulation.
  Don't call `random.*` module functions.
- Frontend: Next.js 16 App Router + Tailwind v4 (read `frontend/AGENTS.md`: this Next.js version
  differs from older docs). Client components fetch from `NEXT_PUBLIC_API_URL`. Colors are CSS tokens in
  `globals.css`. **Agent colors belong to agents** (fixed by cast order) and must not be reused
  for UI chrome, tiers, or aggregate series. Use ink tokens for those. Text never takes a series
  color; a colored dot beside it carries identity.
- Tests must stay offline and fast (~1s). No real API calls in tests.

## Gotchas

- `LLM_CACHE=true` + same seed + same prompts means identical replies. To get a genuinely new sample,
  change the seed or set `LLM_CACHE=false`.
- A run started from the CLI is visible in the UI, but only the process that started it can stop it.
  A run whose process died shows as `interrupted` after 3 minutes without a heartbeat.
- Rough cost: a 2-day run is ~1–2k chat calls (~1k tokens each) plus embeddings, and ~10–20 minutes
  with a fast model at `LLM_MAX_CONCURRENCY=8`. Add a control run and it doubles.
- `backend/data/` (the DB) and `.env` are git-ignored. Don't commit either.
