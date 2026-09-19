# MemeWorld

MemeWorld is a **controlled simulation environment for studying how memes form**, built on top of
[Generative Agents](https://github.com/joonspk-research/generative_agents) (Park et al., 2023).

Eight students live ordinary routines on a Homewood-style campus. A hidden *world script*, generated before the
run, makes recurring *latent event structures* happen to them: a small slip that cascades, two mistakes that
cancel out, an independent coincidence, a beneficial failure. Each is told through one of many everyday
surface "skins". Agents never see the structure. They perceive fragments of it, form **lossy memories**,
retrieve them stochastically, reflect and talk.

An external observer then asks two questions: does a shared expression **emerge** (spread through exposure), and
is it **grounded** (does it track a hidden structure beyond chance)? It also measures how the answers change when
a single mechanism is switched on.

```
latent event → partial perception → lossy symbolic memory → retrieval/reflection
             → social interaction → symbol invention/reuse → cultural transmission
```

There is **no meme state inside agents**: no `known_memes`, no meanings, no fitness. A "meme" exists only as
language that agents happen to reuse, and the analyzer finds it afterwards. Agents are never told to invent
memes, slang or labels, never see hidden families, skins, circles or assignment, and never see observer output.

## Research direction

The implementation contract is **[docs/ONTOLOGY_V2.md](docs/ONTOLOGY_V2.md)**. It tests the hypothesis that a
shared expression forms only when three bottlenecks are passed: **NEED**, **LINK** and **WORDING** (see
*Mechanisms* below). As of 2026-09-19 the research direction has changed
([docs/research/MEMO_2026-09-19_research_landscape.md](docs/research/MEMO_2026-09-19_research_landscape.md)).
The primary study will ask how a community's existing ideas acquire new meanings when its circumstances change,
and when cultural memory helps or hinders that. It is being specified in `docs/ONTOLOGY_V3.md`. The v2
bottleneck factorial is now **calibration** and a secondary study.

## Four separated layers

| Layer | Code | Can see |
|---|---|---|
| **WORLD**: campus graph, clock, routines, and the pre-generated world script (structures × skins, referents, circles) | `backend/simulation/` | everything |
| **AGENT**: GA persona, perception with viewpoints, lossy memory, retrieval, reflection, reminding, open matters, dyadic and group conversation | `backend/agents/`, `backend/memory/` | only its own observations and memories |
| **EXPERIMENT CONTROLLER**: configs, mechanism switches, pressure modules, factorial designs | `configs/`, `backend/modules/`, `backend/experiment/` | re-weights cognition; never specifies culture |
| **OBSERVER**: candidates, typed provenance, emergence, grounding, funnel and manipulation checks, probes, outcomes | `backend/analysis/` | the finished run's logs; never imported by agent code (enforced by a test) |

## Mechanisms (ontology v2)

Every mechanism is **off** in `configs/default.yaml`, so `baseline.yaml` is the plain cognitive model: GA-style
cognition with partial perception, viewpoints and lossy memory. Each mechanism is switched on by one key and
leaves trace records for a manipulation check. Decision numbers refer to [docs/DECISIONS.md](docs/DECISIONS.md).

| Bottleneck | Mechanism | Key | See |
|---|---|---|---|
| NEED | recurring referents ("the night shuttle") | `latent_events.referents.enabled` | D53 |
| NEED | instances cast inside friend circles | `latent_events.assignment.mode` (`balanced`, `home`) | D54 |
| NEED | evening catch-ups between close friends | `conversation.catchup.enabled` | D47 |
| NEED | multi-party talk at meal tables | `conversation.group.enabled` | D63 |
| NEED | open matters cue conversation | `need.enabled` | D60 |
| LINK | spontaneous reminding after perception, conversation or overhearing | `reminding.enabled` | D48, D59 |
| LINK | an instance's private "inner" fact visible to bystanders | `latent_events.link_visibility` | D56 |
| WORDING | distinctive heard wording sticks verbatim | `memory.verbatim.enabled` | D49, D58 |
| WORDING | recently heard wording is in mind when speaking | `priming.enabled` | D61 |
| WORDING | down-weight seed facts and routine sightings in retrieval | `retrieval.source_weights` | D50 |

Other switches:
- `latent_events.structure`: `real`, `scrambled` or `none` (a calibration for grounding; D55).
- `latent_events.holdout_frac`: how often held-out skins are used (D55).
- `topology.mode: generated`: a generated friendship network (D62).
- `controls.planted_phrase`: the positive control (D65).
- The pressure modules `modules.emotion`, `social_reward`, `prestige_bias` and `conformity`. These are
  independent (D64).

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

## Quick start

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python numpy fastapi uvicorn pyyaml pytest anthropic

# offline smoke run (deterministic mock LLM), then analyze it
.venv/bin/python -m backend.cli run --config configs/smoke.yaml --out runs/smoke --analyze

# real run: Claude Haiku via the local `claude` CLI (or set llm.backend=anthropic with ANTHROPIC_API_KEY)
.venv/bin/python -m backend.cli run --config configs/baseline.yaml --analyze

# (re-)analyze a finished run; compare runs (refuses to mix observer specs)
.venv/bin/python -m backend.cli analyze runs/<run_id>
.venv/bin/python -m backend.cli compare runs/a runs/b

# deterministic replay from recorded LLM outputs (verifies the trace hash)
.venv/bin/python -m backend.cli replay runs/<run_id>

# UI: http://127.0.0.1:8765
.venv/bin/python -m backend.cli serve

# tests
.venv/bin/python -m pytest -q tests
```

Any config value can be overridden with `--set`, for example `--set memory.encoding_noise=0.6 simulation_days=2`.

`analyze` and `run --analyze` use the config's fixed observer, `analysis.observer: {backend, model}`. If that is
null, they use the agents' LLM. Explicit `--backend` / `--model` flags take precedence.

## Experiments: factorial designs

A design file in `configs/designs/*.yaml` contains:
- a base config;
- factors, where each level is a config overlay (crossed as a full factorial);
- extra control cells;
- seeds;
- a fixed observer;
- pre-registered outcome columns.

Every cell × seed runs with `seed = world_seed`, so all cells of a seed see the same world (common random
numbers).

```bash
D=configs/designs/bottleneck_factorial.yaml
.venv/bin/python -m backend.cli design expand $D                      # list cells and run dirs
.venv/bin/python -m backend.cli design run $D --parallel 4            # run + analyze what is missing (resumable)
.venv/bin/python -m backend.cli design run $D --only control --seed 1 # a pilot subset
.venv/bin/python -m backend.cli design status $D                      # missing/running/failed/finished/stale/analyzed
.venv/bin/python -m backend.cli design table $D --csv out.csv         # per-cell mean ± sd from outcomes.json
```

- Each cell and seed runs in its own subprocess, into `runs/<design>/<cell_id>/s<seed>`.
- Runs analysed with a different observer spec are marked `stale` and re-analysed, never pooled.
- `--backend mock` runs a whole design offline into `runs/<design>__mock/`.
- The runs root is `--runs-root`, else the env var `MEMEWORLD_RUNS_ROOT` (also used by `run`), else `runs/`.

Shipped designs:
- `bottleneck_factorial.yaml`: NEED × LINK × WORDING × real/scrambled structure, plus a no-events control and a
  planted-phrase control. That is 54 two-day runs, with Haiku agents and a Sonnet observer.
- `smoke_mock.yaml`: offline, a few seconds per run.

## Single-run conditions (`configs/*.yaml`)

| Config | Change relative to `default.yaml` |
|---|---|
| `baseline.yaml` | none: the plain cognitive model, with every NEED/LINK/WORDING mechanism and every module off |
| `perfect_memory.yaml` | verbatim encoding (no LLM rewrite), almost no decay or forgetting, deterministic retrieval |
| `high_noise.yaml` | gist-only encoding, fast decay, small capacity, retrieval τ = 1.5 |
| `social_reward.yaml` | social reward only. It no longer switches on emotion; add `modules.emotion: true` to combine them |
| `event_rich.yaml` | latent events about twice as frequent |
| `no_events.yaml` | negative control: an empty world script |
| `smoke.yaml` | 2-day offline mock run |

`scripts/run_experiments.sh <seed> <parallel>` runs these single-run conditions with one seed.
`modules.prestige_bias` (status source: `module_params.prestige_bias.scores_source`, one of `degree`, `random`
or `file`) and `modules.conformity` are also available.

## What a run writes (`runs/<run_id>/`)

| File | Content |
|---|---|
| `config.resolved.yaml` | the exact configuration |
| `manifest.json` | population, circles, topology metrics, condition (design/cell/levels), code version (git sha, prompt hashes), world-script hash, population lexicon, module manipulation checks, stats |
| `world_script.jsonl` | the whole pre-generated world, written at start (hidden ground truth) |
| `events.jsonl` | world-script instances as they are released (hidden ground truth) |
| `trace.jsonl` | full causal trace: world events and beats, observations, viewpoints, encoded, merged and forgotten memories (with prompts), memory links, reminding, wordings, priming, open matters, decisions, utterances (with retrieved memories), exposures (actual listeners only), dyadic and group conversations, reflections, moves, invitations |
| `frames.jsonl` | per-tick state for replay |
| `llm_calls.jsonl` | every prompt and response (used for replay) |
| `memory_meta.json` | simulator-only memory provenance, links and wordings |
| `agents_final/<id>/associative_memory/` | final memory streams in upstream GA format |
| `analysis.json`, `analysis_llm_calls.jsonl` | observer output: candidates, emergence, grounding, funnel, probes, plus the legacy evaluation |
| `outcomes.json` | the fixed per-run outcome vector: `n_emerged`, `n_grounded`, `max_emerged_adoption`, funnel, manipulation checks, per-mechanism validity, observer spec. Read by `compare` and `design table` |
| `design_cell.json`, `design_cell.log` | design runs only: runner status and log |

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

**Trace**
- search utterances and expand the chain: world event → observation → memory → retrieval → utterance → listener
  memories → later reuse

**Research debug**
- this toggle reveals hidden event types; demo mode strips them *server-side*, along with skins, schemas,
  circles, assignment and grounding

## Layout

```
backend/
  ga_compat.py            bridge to upstream Generative Agents
  llm/                    client (record/replay), claude CLI / Anthropic / mock backends, embeddings
  simulation/             engine.py, world.py, scheduler.py, world_script.py, structures.py, skins/ (E1-E4),
                          referents.py, circles.py, lexicon.py, rngs.py, latent_events.py, legacy_scenarios_v1.py
  agents/                 profile.py, agent.py, perception.py, viewpoint.py, planner.py, conversation.py,
                          group_conversation.py, topology.py, ga_prompts.py
  memory/                 store.py, encoder.py, retrieval.py, reflection.py, reminding.py, need.py, wording.py
  modules/                base.py, emotion.py, social_reward.py, prestige.py, conformity.py
  analysis/               candidates, provenance, emergence, grounding, funnel, outcomes, transmission,
                          semantics, lineage, probes, evaluation (legacy), compare, pipeline
  experiment/design.py    factorial designs: expand, run, status, table
  tracing/logger.py       deterministic JSONL trace
  api/server.py           FastAPI
  prompts/                MemeWorld's own (non-GA) prompt templates; never edited, only new versions added
configs/                  default + condition overlays, designs/, population/
frontend/                 index.html, app.js, styles.css
third_party/generative_agents/   vendored upstream (unmodified)
docs/ONTOLOGY_V2.md       v2 implementation contract
docs/DECISIONS.md         every deviation from GA / the plan (D1-D73)
docs/COGNITIVE_FRAMING.md cognitive-science framing of each component
docs/research/            research memo (current direction)
REPORT.md                 build report and first results (pre-v2)
```
