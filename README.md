# MemeWorld

MemeWorld is a research prototype for **cultural transmission and changing interpretations**, built on top of
[Generative Agents](https://github.com/joonspk-research/generative_agents) (Park et al., 2023).

There are two world modes. The new **campus commons** connects consequential projects,
private measurements, local speech, versioned records, participant turnover and changed
operating conditions. Its first experiment addresses **RQ2: How do shared records affect
continuity and adaptation?** Four matched configurations cross records/no-records with
stable/changed conditions. All participants are agents; the interface is an observer.

Read the [implementation and research contract](docs/COMMONS_IMPLEMENTATION.md) for the
question-to-mechanism mapping, action rules, measures, commands and limitations. The
[research review](docs/MEMEWORLD_RESEARCH_AND_DESIGN_REVIEW.md) gives the broader proposal.
Task success and repeated words are not automatically evidence of semantic change.

In the original **latent-event mode**, eight students live ordinary routines on a Homewood-style campus. The simulator injects hidden, recurring
*latent event structures* that agents never see (a small mistake that cascades, two mistakes that cancel out, an
independent coincidence, a beneficial failure). Agents perceive fragments of these events, form **lossy symbolic
memories**, retrieve them stochastically, reflect, and talk.

In that original mode, an external observer asks: do agents spontaneously invent and converge on expressions that stand for those
hidden structures? It also measures how the answer changes when a single selective pressure is changed (memory
quality, emotion, social reward, prestige, conformity).

```
latent event → partial perception → lossy symbolic memory → retrieval/reflection
             → social interaction → symbol invention/reuse → cultural transmission
```

There is **no meme state inside agents**: no `known_memes`, no meanings, no fitness. A "meme" exists only as
language that agents happen to reuse, and the analyzer finds it afterwards.

## Four separated layers

| Layer | Code | Can see |
|---|---|---|
| **WORLD**: campus graph, clock, routines, hidden latent events | `backend/simulation/` | everything |
| **AGENT**: GA persona, perception, memory stream, retrieval, reflection, planning, conversation | `backend/agents/`, `backend/memory/` | only its own observations and memories |
| **EXPERIMENT CONTROLLER**: config and optional modules | `configs/`, `backend/modules/` | re-weights cognition; never specifies culture |
| **OBSERVER**: meme analyzer, probes, ground-truth evaluation | `backend/analysis/` | the finished run's logs; never imported by agent code (enforced by a test) |

## What comes from Generative Agents

The upstream code is vendored **unmodified** in `third_party/generative_agents/` (commit `fe05a71`) and imported
through `backend/ga_compat.py`. MemeWorld reuses:
- the memory stream: `AssociativeMemory` / `ConceptNode` and the GA on-disk format
- `Scratch` and its identity stable set
- GA's retrieval scoring functions and weights
- the reflection trigger, plus the focal-point and insight prompts
- the poignancy prompts
- `decide_to_talk`
- the relationship-summary and iterative-conversation prompt functions, called directly
- GA's prompt and retry helpers
- the character sprites

All of GA's OpenAI calls are routed to Claude. Every deviation is logged in **[docs/DECISIONS.md](docs/DECISIONS.md)**.

## Quick start (Windows CMD)

```cmd
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt

REM Offline commons run and descriptive research measures
.venv\Scripts\python.exe -X utf8 -m backend.cli run --config configs\commons_smoke.yaml --analyze

REM Original latent-event smoke run
.venv\Scripts\python.exe -X utf8 -m backend.cli run --config configs\smoke.yaml --analyze

REM Observer interface at http://127.0.0.1:8765
.venv\Scripts\python.exe -X utf8 -m backend.cli serve
```

In another CMD window, run tests or replay a recording. Replace `YOUR_RUN_ID` with
the run directory printed by the run command. Existing recordings are never overwritten.

```cmd
.venv\Scripts\python.exe -X utf8 -m pytest -q tests
.venv\Scripts\python.exe -X utf8 -m backend.cli replay runs\YOUR_RUN_ID
```

For an actual model run, `commons.yaml` uses the configured, locally authenticated
Claude CLI. The mock is a hand-written software test policy and supplies no research
evidence about language-model culture.

```cmd
.venv\Scripts\python.exe -X utf8 -m backend.cli run --config configs\commons.yaml --analyze
```

Any config value can be overridden with `--set`, e.g. `--set memory.encoding_noise=0.6 simulation_days=2`.

## Experimental modes (they differ only in config)

The commons configurations use `world.mode: commons`. Their shared base fixes the
population, resources, project calendar, turnover on day 3 and comparison boundary
on day 5. Each condition runs eight working days by default.

| Config | Shared records | Outdoor conditions change |
|---|---|---|
| `commons.yaml` | Yes | Yes |
| `commons_records_stable.yaml` | Yes | No |
| `commons_no_records_change.yaml` | No | Yes |
| `commons_no_records_stable.yaml` | No | No |
| `commons_smoke.yaml` | Yes | Yes; shortened offline software exercise |

The following configurations retain the original latent-event world:

| Config | Change relative to `default.yaml` |
|---|---|
| `baseline.yaml` | GA-style cognition, lossy memory (noise 0.3), no optional modules |
| `perfect_memory.yaml` | verbatim encoding (no LLM rewrite), almost no decay or forgetting, deterministic retrieval |
| `high_noise.yaml` | gist-only encoding, fast decay, small capacity, retrieval τ = 1.5 |
| `social_reward.yaml` | emotion + social reward (utility from improving others' mood) |
| `event_rich.yaml` | latent events about twice as frequent |

Prestige and conformity are available as `modules.prestige_bias` and `modules.conformity`.

## What a run writes (`runs/<run_id>/`)

| File | Content |
|---|---|
| `config.resolved.yaml` | the exact configuration |
| `manifest.json` | population, world, groups, stats |
| `trace.jsonl` | full causal trace: world events and beats, observations (with attention probabilities), encoded, merged and forgotten memories (with prompts), decisions (prompt, response, retrieved memories and scores), utterances (with retrieved memories and context), exposures (actual listeners only), conversations, reflections (with evidence), moves, invitations |
| `frames.jsonl` | per-tick state for replay |
| `events.jsonl` | hidden ground truth (simulator only) |
| `llm_calls.jsonl` | every prompt and response (used for replay) |
| `memory_meta.json` | simulator-only memory provenance |
| `agents_final/<id>/associative_memory/` | final memory streams in upstream GA format |
| `analysis.json`, `analysis_llm_calls.jsonl` | observer output |

Commons runs additionally write `commons_events.jsonl`, `commons_final.json`, and
daily `checkpoints/tick-*/` containing world and personal-memory snapshots. These are
inspection snapshots, not a resume API. Replay re-executes the recording from its
beginning. Commons analysis is deterministic and does not produce observer LLM calls
or use the original four-family meaning probes.

## UI

**Campus**
- a top-down map with GA sprites, animated semantic movement, active events and speech bubbles with listener
  lines
- play, pause, speed and rewind controls, plus buttons that jump to the first meme use and the first cross-group
  transmission
- click an agent to see identity, personality, relationships, routine, current activity, memories as of that
  tick, last retrieved memories with score components, reflections and conversations
- click any utterance to see its causal trace

**Culture**
- convention cards, adoption over time, a propagation network (confidence-weighted, with cross-group edges
  marked), lexical lineage, meaning over time, private agent interpretations, and a latent-event correspondence
  panel (debug only)

**World (commons runs)**
- a synchronized timeline of projects, kit custody, calibration, supplies and active work
- complete record versions with authorship and time-filtered read histories
- newcomer membership and persistent work across working days

For commons runs, the Culture page shows **Inheritance and adaptation**: task outcomes
before and after the common boundary, newcomer participation, record access, and the
implemented coverage of RQ1--RQ4. It explicitly marks semantic change as unevaluated.

**Trace**
- search utterances and expand the chain: world event → observation → memory → retrieval → utterance → listener
  memories → later reuse

**Research debug**
- this toggle reveals hidden event types; demo mode strips them *server-side*

## Layout

```
backend/
  ga_compat.py            bridge to upstream Generative Agents
  llm/                    client (record/replay), claude CLI / Anthropic / mock backends, embeddings
  simulation/             world.py, scheduler.py, latent_events.py, engine.py
  agents/                 profile.py, agent.py, perception.py, planner.py, conversation.py, ga_prompts.py
  memory/                 store.py, encoder.py, retrieval.py, reflection.py
  modules/                base.py, emotion.py, social_reward.py, prestige.py, conformity.py
  analysis/               candidates, transmission, semantics, lineage, probes, evaluation, pipeline
  tracing/logger.py       deterministic JSONL trace
  api/server.py           FastAPI
  prompts/                MemeWorld's own (non-GA) prompt templates
configs/                  experiment configs + population
frontend/                 index.html, app.js, styles.css
third_party/generative_agents/   vendored upstream (unmodified)
docs/DECISIONS.md         every deviation from GA / the plan
REPORT.md                 build report and first results
```
