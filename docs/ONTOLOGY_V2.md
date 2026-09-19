# MemeWorld ontology v2 — implementation contract

Thesis: MemeWorld is a **controlled simulation environment for evaluating how memes form under
manipulable conditions**. The working hypothesis is that a shared expression for a recurring pattern
forms only when three bottlenecks are passed:

1. **NEED** — the same people repeatedly need to refer to the same thing;
2. **LINK** — someone connects separate instances as "the same kind of thing";
3. **WORDING** — a specific wording survives lossy memory and gets reused.

Every experimental pressure should act through one of these. v2 makes each bottleneck a set of
independently switchable mechanisms, keeps the world identical across conditions, and measures
**emergence** (a convention spreads through exposure) separately from **grounding** (it tracks
the hidden structure beyond chance).

Hard constraints (unchanged, enforced by tests):
- never tell agents to invent memes, slang, names or labels;
- never reveal hidden families, schemas, skins, circles or assignment to agents;
- never feed analyzer output back to agents;
- no meme state inside agents;
- WORLD / AGENT / EXPERIMENT CONTROLLER / OBSERVER layers stay separate.

Engineering rules for everyone:
- **Never edit an existing file under `backend/prompts/`.** Running jobs read them. Create a new
  versioned file instead.
- Do not touch anything under `runs/` except new `runs/dev_*` or `runs/test_*` directories.
- Every mechanism is **off by default** in `configs/default.yaml`, so "baseline" is the plain
  cognitive model. Each mechanism is switched on by exactly the config key listed below.
- Every mechanism writes **trace records**, so the observer can compute manipulation checks.
- Record non-GA, non-plan decisions in `docs/DECISIONS.md` as D51+. The integrator assigns final
  numbers; parallel agents write their entries into their own report.

---------------------------------------------------------------------------------------------------

## 1. World (item 1: pre-generated world script; items 4–7)

### 1.1 World script
The whole sequence of world events is generated **before** the simulation, from streams seeded
only by `world_seed` (null → `seed`). It is independent of agents' decisions, LLM outputs and
cognitive mechanisms, so every condition with the same `world_seed` sees exactly the same
events. This gives common random numbers across conditions.

- Module: `backend/simulation/world_script.py`
  - `generate(cfg, profiles: dict[str, AgentProfile], circles: dict[str, list[str]], clock) -> list[EventInstance]`
  - `save(instances, path)` / `load(path) -> list[EventInstance]`
- Random streams: `_seed_rng(world_seed, "<purpose>", ...)`, with one stream per purpose:
  - `"timing"`: whether an instance starts at a tick;
  - `"family"`;
  - `"skin"`;
  - `"cast"`;
  - `"slots"`;
  - `"referent"`;
  - `"holdout"`;
  - `"compose"`.

  The helper lives in `backend/simulation/rngs.py` (`seed_rng(*parts)`, crc32 of each part, the
  same scheme as `engine._seed_rng`).
- `"current"` locations resolve at generation time to the protagonist's **planned** routine
  position. The generator calls `scheduler.plan_day(agent_like, clock, seed_rng(cfg["seed"], "plan", aid, day))`,
  which is exactly what the engine calls, so plans match. Every beat therefore has a concrete
  location and arena, and all of the beat's movers are forced there at runtime.
- "Busy" is decided at generation time from already-scheduled instances that overlap in time,
  never from live conversation state.
- `latent_events.script_from: <path>` loads a previously saved script verbatim.
- The engine writes `world_script.jsonl` into the run dir at start, and `events.jsonl` (the same
  records) as instances are released.
- The LLM surface generator (`generator: llm`) is dropped in v2. A config asking for it must raise
  a clear error.

### 1.2 Structure schema × surface skin (item 5)
A **family** (E1–E4) is an abstract causal structure. A **skin** is one concrete wording of it.
All families share one shape, so shape carries no information about family:

| node | beat | offset | location | visibility | salience | involves |
|---|---|---|---|---|---|---|
| `n0` | 0 | 0 | P's planned position | all | 0.55 | P |
| `n0_private` | 0 | 0 | same | P only (see `link_visibility`) | 0.30 | P |
| `n1` | 1 | 2 | second actor's planned position (P's, if the actor is an NPC) | all | 0.50 | second actor (+P if mentioned) |
| `n2` | 2 | 4 | P's planned position | all | 0.55 | P (+second actor if mentioned) |

Second actor: `{S}` (a related person) in E1, E2 and E4; `{Q}` (an unrelated person) in E3. The
unrelatedness *is* E3's structure.

| family | n0 | n1 | n2 |
|---|---|---|---|
| E1 cascade | small slip by P | a consequence of that slip hits S | a further, otherwise unrelated failure follows |
| E2 cancelling | P makes a mistake | S (or the world) independently makes a second mistake | the two cancel out and nothing goes wrong |
| E3 coincidence | P does an unusual thing | an unrelated Q independently does the same unusual thing | P's outcome: the thing is noticed again or talked about; no causal link |
| E4 beneficial failure | P fails at something | the failure puts P in contact with S or an opportunity | P ends up better off because of it |

The private fact `n0_private` gives P's inner view (why P did it, or what P didn't notice). It is
never a label.

Public facts **must not** use narrator causal glue across beats ("because of", "which meant",
"thanks to", "as a result", "so"). The structure has to be inferred. Facts must also make sense
on their own, since perception is partial.

**Topic balance.** Topic domains are fixed in `backend/simulation/referents.py`:
- 8 train domains: `dining, coursework, lab, transit, gym, dorm, cafe, clubs`;
- 3 holdout domains: `library, quad, tech`.

Every family has exactly **one train skin per train domain** and **one holdout skin per holdout
domain**, 11 skins per family. So topic carries no information about family, and held-out skins
are also new topics.

**Skin format**: `backend/simulation/skins/e1.py` … `e4.py`, each defining a `SKINS: list[dict]`:
```python
{"key": "e1_dining",                  # "<family lower>_<domain>"
 "family": "E1",
 "domain": "dining",                  # one of the 11 domains; the {R} slot takes a referent from this domain
 "holdout": False,                    # True exactly for library, quad, tech
 "slots": {"X": ["...", "..."]},      # optional extra slot lists ({X}, {Y}); 2-4 options each
 "locations": {"n0": "Dining Hall"},  # optional: pin nodes to locations (Dorm, Dining Hall, Classroom, Library,
                                      #   Research Lab, Gym, Cafe, Quad); else the actor's planned position
 "facts": {"n0": "...{P}...{R}...", "n0_private": "...{P}...", "n1": "...{S}...", "n2": "...{P}..."}}
```
- Placeholders are only `{P}`, `{S}` (or `{Q}` in E3), `{R}`, `{X}` and `{Y}`.
- `{R}` appears exactly once, in `n0`, and may appear again in `n1`/`n2`. It must read naturally
  with **every** name in its domain's pool in `referents.CATEGORIES`, e.g. "{P} waited twenty
  minutes at {R}" or "{R} rejected {P}'s card". Never capitalize it at the start of a sentence;
  put a person first.
- Text is plain past tense, one sentence per fact, 8–25 words, everyday campus life.

**Lexical hygiene**, validated by `tests/test_world_v2.py`. Content bigrams are those where both
words are non-stopwords, excluding placeholders, referent names and population-lexicon bigrams.
- Content bigrams never repeat across skins of **different** families.
- Holdout skins share no content bigram with train skins of the same family.
- No skin contains a nickname-like phrase ("the X thing/incident/mix-up/saga") or a quoted label.

### 1.3 Structure controls (item 7)
`latent_events.structure`:
- `real` (default): a skin's four facts come from one skin of the drawn family.
- `scrambled`: each instance keeps the same shape and roles, but each node (`n0`+`n0_private`,
  `n1`, `n2`) is drawn from a **different random skin of any family**. The instance label is drawn
  uniformly from `families` and carries no structure (a positive-null calibration for grounding).
  `composed_from` records the source (family, skin) of each node.
- `none`: the same composition as `scrambled`, but every instance is labelled `E0` (plain mishaps).

`latent_events.link_visibility` (default 0.0): the probability that an instance's `n0_private`
fact is visible to everyone who is present, instead of P only.

### 1.4 Referents (item 6)
A **Referent** is a persistent world thing that recurs across instances and can be referred to
again ("the night shuttle", "the lounge oven").
- Referent categories are the topic domains. The pools live in `backend/simulation/referents.py`
  (written by the lead), with 5 names per domain.
- Referents are family-neutral: every family has a skin in every domain.
- Config `latent_events.referents: {enabled: false, reuse: 0.6}`.
- When disabled, each instance draws its domain's referent uniformly from the pool. Repeats happen
  only at chance, and nothing tracks them.
- When enabled, the protagonist's circle (or, with no circle, the whole population) keeps, per
  domain, the referents it has already met. With probability `reuse`, the instance reuses one of
  those (uniformly); otherwise it draws uniformly from the pool.
- Each instance records `referents: [{"id": "<domain>:<index>", "domain", "name", "reused": bool}]`.

### 1.5 Circles and assignment (item 4)
The population file gets a `circles:` section of **disjoint** circles, each with **3 or more**
members, validated at load. Agents in no circle form the free pool. For homewood8:
```yaml
circles:
  c_lab: [maya, dev, hana]
  c_dorm: [ethan, leo, jordan]
# free: priya, sofia
```

`latent_events.assignment: {mode: none, strength: 0.7}`:
- `none`: P is drawn uniformly from non-busy agents, as in v1 without clustering.
- `balanced`: each instance picks a circle round-robin (with a random offset from `world_seed`).
  With probability `strength`, P, and S where possible, are cast from it. This removes any
  family-to-circle correlation while keeping recurrence inside circles.
- `home`: families map to circles round-robin, rotated by `world_seed % n_circles`
  (counterbalanced across seeds). With probability `strength`, P and S are cast from the
  family's circle.

Every instance records `circle` (the id, or null) and `cast_from_home` (bool: was P actually cast
from that circle?).

### 1.6 Held-out split (item 7)
Skins with `holdout: True` are held-out surfaces. `latent_events.holdout_frac` (default 0.25) is
the probability that an instance, from day 1 on, uses a held-out skin. The old `holdout_from_day`
is removed.

### 1.7 EventInstance record
This is what `events.jsonl` and `world_script.jsonl` contain, one JSON object per line:
```
{id, latent_type ("E0"-"E4"), structure_mode, schema (family's schema id), skin (key, or null when composed),
 composed_from ([{"node","family","skin"}] or null), holdout, start_tick, generator: "script_v2",
 roles: {"P": {"agent","name"}, "S"|"Q": {...}},
 circle, cast_from_home, referents: [...],
 beats: [{"idx","tick","location","arena","facts":[{"id","text","salience","visibility","kind","involves"}],"movers"}],
 narrative}
```
Fact ids follow `"{event_id}.b{beat}.f{i}"`; `kind` is `"action"` or `"inner"` (for
`n0_private`). The engine consumes `beats`, `roles` and `movers` as before. The keys
`latent_type`, `schema`, `skin`, `composed_from`, `structure_mode`, `circle`, `cast_from_home`
and `narrative` are **hidden** and must be stripped by the API in demo mode.

---------------------------------------------------------------------------------------------------

## 2. Agent cognition (items 1, 4, 8)

### 2.1 Random-number substreams (item 1)
`Agent.stream(name) -> np.random.Generator` is created lazily as
`seed_rng(cfg["seed"], agent_id, name)` and cached. The names are:
- `perceive`, `ambient`, `encode`, `lens`, `verbatim`, `remind`, `react`, `reflect`, `talk`,
  `remark`, `need`, `prime`.

Mechanisms draw only from their own stream. For example, `encode()` uses `stream("lens")` for
lens sampling and `stream("verbatim")` for stickiness, whatever rng is passed in. Switching one
mechanism on therefore never changes another mechanism's draws. `agent.rng` remains an alias for
`stream("legacy")`.

### 2.2 Verbatim stickiness fixes (item 4)
In `memory.verbatim`, when `enabled: true`:
- **`exclude_self: true`**: only utterances by other people are candidates. Currently 17 of 60
  stuck phrases were the agent's own.
- **`distinctiveness: lexicon`** (the default when enabled): a phrase is distinctive if it
  contains a content word that is rare (Zipf ≤ `zipf_max`) **and** not in the population lexicon
  (`backend/simulation/lexicon.py`: routine activities, places, arenas, classes, clubs, habits,
  backgrounds). `zipf` keeps the old test.
- **Clean spans**: 2–3 word spans whose first and last tokens are content words, using a proper
  function-word and adverb list ("all", "those", "basically", "really" and so on can't sit at an
  edge). No names of people.
- Stuck phrases are stored as first-class **Wording** records (2.4). Setting
  `memory.verbatim.render_in_text: true` (the default) also quotes them in the description.

### 2.3 Linking (item 4 + 8)
`reminding: {enabled: false, on_sources: [perception, conversation, overheard], min_salience: 0.45, n_related: 4, n_salient: 3, min_age_minutes: 90}`
- Reminding can now fire after conversation and overheard memories, not just after perception.
- **Candidates may include earlier reminding thoughts**, so links compound: a third instance can
  attach to an existing "X reminded me of Y" thought and extend the chain.
- Every link, whether from reminding, the lens `association`, a merge, or a reflection citing
  evidence, is recorded as a **Link** (`MemoryMeta.links: [{"to": node_id, "mechanism", "reason"}]`)
  and traced as `{"type": "memory_link", agent, from, to, mechanism, reason}`.

### 2.4 Wording entity + production priming (item 8)
- `MemoryMeta.wordings: [{"phrase", "heard_from", "utterance_id", "tick", "self_produced": false}]`
  is filled by verbatim stickiness.
- Module `backend/memory/wording.py`:
  - `record(agent, node_id, phrases, obs)` stores them;
  - `recent_wordings(agent, now, window_hours, k, rng) -> list[str]` returns phrases from the
    agent's own memory nodes' wordings (sim-side sidecar of the agent's own memories, never the
    analyzer). It samples up to k weighted by recency and repetition.
- `priming: {enabled: false, window_hours: 24, max_phrases: 3}`. When enabled, the conversation
  context gets the line `Things {first} has heard people say lately: "a", "b".`
  `priming_line(agent, rng) -> str | None` lives in `wording.py`, and the integrator wires it into
  `conversation._context`. Traced as `{"type": "priming", agent, conversation_id, phrases}`.

### 2.5 Open matters (NEED, item 8)
`need: {enabled: false, min_importance: 6, half_life_hours: 24, max_open: 3, discuss_decay: 0.5}`
- Module `backend/memory/need.py`.
- `note(agent, obs, node)`: after encoding, an observation that the agent took part in, or whose
  memory has importance ≥ `min_importance`, becomes an **open matter**:
  `{"node_id", "text", "created", "strength": 1.0}` in `agent.open_matters`. This is ordinary
  agent state, not meme state.
- `focal(agent, now) -> list[str]`: the texts of the top `max_open` open matters by decayed
  strength. The integrator appends them as retrieval focal points in conversations and adds the
  context line `{first} still has on their mind: <text>`.
- `discussed(agent, retrieved_node_ids)`: multiplies strength by `discuss_decay` when the matter's
  node was retrieved into an utterance.
- Traced as `open_matter` and `open_matter_focal`.

---------------------------------------------------------------------------------------------------

## 3. Social (item 8)

- **Population lexicon**: `backend/simulation/lexicon.py` (already written by the lead).
- **Generated topology**: `backend/agents/topology.py`, used when
  `topology.mode: generated` (default `file`):
  `generate(profiles, cfg, rng) -> (circles, relationships, routine_additions)`.
  - Partition agents into `n_circles` disjoint circles.
  - Within-circle ties are dense (familiarity/affinity `within`); between-circle ties are sparse
    (`p_between`, familiarity `between`); there are `n_bridges` bridge agents with one extra
    strong tie into another circle.
  - With `shared_meals: true`, each circle gets a shared dinner slot (the same time and arena)
    added to its members' routines, so co-location agrees with the ties.
  - Record topology metrics (density, modularity, bridges) for the manifest.
- **Multi-party talk**: `backend/agents/group_conversation.py`,
  `run_group_conversation(conv_id, participants, bystanders, rng, *, topic="meal") -> dict`.
  - The record has the same shape as `run_conversation`, but with 3 or more `participants`.
  - Round-robin speakers from a random start; up to `conversation.group.max_utterances`.
  - A new prompt `backend/prompts/group_chat_v1.txt`, adapted from GA `iterative_convo`.
  - Listeners are all other participants plus overhearing bystanders.
  - Every participant encodes a conversation memory; bystanders' overheard observations are
    returned.
  - Config `conversation.group: {enabled: false, venues: [["Dining Hall", "Main Floor"]], windows: ["12:00-13:30", "18:00-19:30"], min_participants: 3, max_participants: 5, max_utterances: 8, prob: 0.5}`.
  - The integrator schedules it in `engine._conversations`.
- **Pressure modules are independent**:
  - `social_reward` no longer switches on `emotion`. It appraises partner valence itself with the
    lexicon and never adds mood lines. When `emotion` is on too, it uses emotion's valence.
  - `prestige_bias.scores_source: degree | random | file`.
  - Every module keeps `counters` (effects applied per hook) and exposes
    `manipulation_check() -> dict`.
  - `ModuleStack.manipulation_checks()` aggregates them.

---------------------------------------------------------------------------------------------------

## 4. Observer (item 3)

Everything here is observer-only and reads a finished run directory.

- **`backend/analysis/provenance.py`**: typed links per utterance.
  - `referent_event_ids`: events started before the utterance whose fact texts (world text plus
    any viewpoint renderings) or referent names share 2 or more content tokens with the utterance
    (non-stopword, non-name, Zipf < 5). Also the conversation's trigger events.
  - `direct_event_ids` / `carried_event_ids`: from the source types of retrieved nodes. Perception
    counts as direct; conversation, overheard, reminding and reflection count as carried.
  - **Grounding uses `referent_event_ids` only.**
- **`backend/analysis/emergence.py`**: exposure-conditioned adoption, per candidate.
  - The originator is the first speaker.
  - For every other agent, the first exposure tick is the first time they were a listener of a
    usage. Adopters used it after exposure; independent users used it with no prior exposure.
  - Outputs: `n_exposed`, `n_adopters`, `n_independent`, the adopter and independent rates, a
    Fisher exact p, and `in_lexicon`/`in_world_text` flags.
  - `emerged = n_adopters >= 2 and n_adopters > n_independent and not in_lexicon`.
- **`backend/analysis/grounding.py`**:
  - Each referent-linked usage gets a fractional family distribution.
  - Statistic: the weighted share of the post-hoc best family z. This replaces the `any()` hit.
  - Null: 1000 permutations of family labels **within strata (circle × day)**, keeping each
    usage's links fixed.
  - p-value, then Benjamini–Hochberg q across candidates.
  - `grounded = q < 0.1 and linked_usages >= 3 and linked_speakers >= 2`.
  - Also reports holdout generalization (use linked to a held-out instance of z) and probe accuracy
    against chance (0.25) for exposed vs unexposed agents.
- **`backend/analysis/funnel.py`**: the `scripts/diagnose_funnel.py` metrics moved into the
  observer on the `referent_event_ids` definition, plus **manipulation checks** counted from trace
  types, one per mechanism:
  - viewpoint drops;
  - lens distortions and associations;
  - verbatim stuck (self vs other);
  - reminding asked and linked, same-event vs cross-event;
  - catch-ups and group conversations;
  - priming lines;
  - open matters;
  - `cast_from_home` rate and referent reuse rate;
  - module counters.
- **`backend/analysis/outcomes.py`** writes `outcomes.json` per run: a fixed outcome vector that
  does not depend on the LLM classifier.
  ```
  {run_id, condition (from manifest), observer {backend, model, analysis_version},
   n_candidates, n_emerged, max_emerged_adoption, n_grounded, grounded:[...], emerged:[...],
   funnel:{...}, manipulation:{...}, validity:{mechanism: "active"|"inactive"|"off"}}
  ```
  A mechanism that is on in the config but never fired is marked `inactive`. The LLM classifier
  output is kept as an annotation only.
- `pipeline.analyze` calls all of these, puts `emergence`, `grounding` and `funnel` blocks in
  `analysis.json`, and writes `outcomes.json`.
- `compare` reads `outcomes.json` and refuses to mix runs with different observer specs.
- `analysis.observer: {backend: null, model: null}`: when set, these are used by both `analyze`
  and `run --analyze` instead of the agents' LLM.

---------------------------------------------------------------------------------------------------

## 5. Experiment control (item 2)

- **Design files**: `configs/designs/<name>.yaml`:
  ```yaml
  name: bottleneck_factorial
  base: configs/baseline.yaml
  seeds: [1, 2, 3]            # seed = world_seed = s for every cell -> common random numbers
  days: 2                     # optional override of simulation_days
  observer: {backend: claude_cli, model: sonnet}
  factors:                    # full factorial over levels; each level is a config overlay
    need:    {off: {}, on: {...}}
    link:    {off: {}, on: {...}}
    wording: {off: {}, on: {...}}
    structure: {real: {}, scrambled: {latent_events: {structure: scrambled}}}
  controls:                   # extra single cells (base + overlay), same seeds
    no_events: {latent_events: {event_rate: 0}}
  outcomes: [n_emerged, n_grounded, ...]   # pre-registered columns for the table
  ```
- **`backend/experiment/design.py`**:
  - `load_design`;
  - `expand(design) -> list[Cell]`, where a Cell is `{cell_id, levels, overrides, seed, run_dir}`
    and `run_dir = runs/<design>/<cell_id>/s<seed>`;
  - `status(design)`;
  - `table(design) -> rows`, read from `outcomes.json`.
- **CLI** (`backend/cli.py`):
  - `design expand <file>`;
  - `design run <file> [--parallel N] [--only cell] [--backend mock]`: runs the missing cells,
    then analyzes each with the design's observer spec;
  - `design status <file>`;
  - `design table <file> [--csv out]`.
- **Manifest** (`engine.write_manifest`, written by the integrator) records:
  - `condition: {design, cell, levels}`, taken from `cfg["_condition"]`, which the design runner
    puts into the config;
  - `code_version: {git_sha, dirty, prompt_hashes: {file: sha256}}`;
  - `world_script_sha256`;
  - `population_lexicon`;
  - `topology`;
  - `manipulation` (module counters).
- **Shipped designs**:
  - `configs/designs/bottleneck_factorial.yaml`: the first experiment, 2×2×2 bottlenecks × real
    or scrambled structure, plus a `no_events` control and a `planted` control;
  - `configs/designs/smoke_mock.yaml`: a tiny mock-backend design used by the tests.
- **Planted-convention positive control**:
  `controls.planted_phrase: null | {agent: maya, habit: "Maya has a habit of calling any mess \"a full pickle\"."}`
  appends the habit to that agent's profile habits at load. It is used only as a control cell, to
  check the observer can detect a convention that really does spread.

---------------------------------------------------------------------------------------------------

## 6. Ownership (parallel build)

| part | files owned (create or edit) | must NOT edit |
|---|---|---|
| W core (world) | `backend/simulation/world_script.py`, `rngs.py`, `referents.py`, `structures.py`, `skins/__init__.py`, `latent_events.py` (the EventInstance record, removing v1 scenario use), `configs/population/homewood8.yaml` (circles), `tests/test_world_v2.py` | engine.py, conversation.py, default.yaml |
| W skins E1–E4 | `backend/simulation/skins/eN.py` (one each) | everything else |
| C (cognition) | `backend/agents/agent.py` (stream), `backend/memory/encoder.py`, `reminding.py`, `retrieval.py`, `store.py`, `wording.py`, `need.py`, new prompt versions, `tests/test_cognition_v2.py` | engine.py, conversation.py, default.yaml |
| S (social) | `backend/agents/topology.py`, `group_conversation.py`, `profile.py`, `backend/modules/*`, new `group_chat_v1.txt`, `tests/test_social_v2.py` | engine.py, conversation.py, default.yaml, homewood8.yaml |
| O (observer) | `backend/analysis/*`, `scripts/diagnose_funnel.py`, `tests/test_observer_v2.py` | everything outside analysis/ |
| X (control) | `backend/experiment/*`, `backend/cli.py`, `backend/config.py`, `configs/designs/*`, `configs/*.yaml` except `default.yaml`, `tests/test_design.py` | engine.py, default.yaml |
| I (integrator, after the rest) | `backend/simulation/engine.py`, `backend/agents/conversation.py`, `backend/api/server.py`, `configs/default.yaml` (final values), `docs/DECISIONS.md`, `README.md`, fixes across files | — |

`configs/default.yaml` already contains every new key with its default (written by the lead). Read
your knobs from there and do not change their names.
