# Campus commons: research contract and implementation

This document links the first executable world extension to the questions in
[the research review](MEMEWORLD_RESEARCH_AND_DESIGN_REVIEW.md#7-research-questions-and-competing-hypotheses).
It describes an agent-only experiment, not a finding that semantic change occurs.

## Scope and question mapping

| Question | Implemented mechanism / instrument | Claim boundary |
|---|---|---|
| RQ1: When does an expression acquire a different shared interpretation? | Consequential sensor projects, local speech, recorded exposure, decisions with retrieved evidence, periodic memory snapshots | These establish observable context and consequences. Lexical repetition and task success do not establish semantic change; validated matched interpretation probes remain future work. |
| RQ2: How do shared records affect continuity and adaptation? | Versioned records with explicit reading; fresh participants; crossed records/no-records and stable/changed operating conditions; identical release and turnover schedules | This is the first executable experiment. Report performance and access descriptively before making causal or semantic claims. |
| RQ3: How much disagreement can a useful convention tolerate? | Indoor and outdoor projects can require different procedures; overlapping campus identities remain | A dedicated group-interpretation manipulation and matched probes are not yet supplied. |
| RQ4: Does history matter after present conditions are matched? | Deterministic recorded replay and memory snapshots preserve histories for inspection | Replay is not a matched-state counterfactual. Material-state matching and history-swap experiments remain future work. |

## First experiment

The primary factors are access to shared records and a change in outdoor operating
conditions. All four cells have the same newcomer schedule, daily project releases,
resource deliveries, initial population and per-seed physical calibration mapping.
The intervention tick is retained in stable controls as a common analysis boundary.

Two recurring requests ask for sensor kits operating at the Library (indoors) and
Quad (outdoors). Assembly, calibration, tests, carrying and delivery consume time.
Assembly and bench testing reserve exclusive stations. Components and test supplies
are finite and replenish on a calendar. A delivery that fails its measured tolerance
stays open and can be reworked.

Calibration settings A and B have a seed-specific physical mapping. A bench test
measures performance in the laboratory. Outdoor conditions can change, making that
test insufficient for the outdoor project while leaving indoor performance intact.
Agents see local conditions and their own measurements, never the correct setting,
treatment schedule, research questions, or other agents' private observations.

Records start empty. Agents choose their text and whether to write, read or revise.
The Library catalog exposes titles and version metadata, not unread content. An
explicit read returns exactly the requested version. Revisions require the version
the editor actually read; concurrent edits can conflict. Records are claims, not
world facts or automatically enforced instructions.

Scheduled newcomers have new identities and empty episodic memory. Departing agents
stop acting; their unfinished actions are canceled and carried objects remain in
the world. Shared facilities and records persist. Persona traits can be matched to
the departing participant, but relationships and episodic history are not inherited.

## World contract

- At each tick, calendar drivers advance, due actions complete, observations are
  delivered, and free agents choose one action from the same resulting snapshot.
- Simultaneous requests use a seed/tick/agent priority independent of LLM calls.
  Accepted operations reserve their objects and equipment until completion.
- Movement takes one tick per graph edge. Work survives day boundaries; there are
  no forced event movements or routine teleports in commons mode.
- Speech is an action. Actual local listeners are recorded. A record is a separate
  transmission channel with version-specific access logs.
- Invalid intentions return bounded feedback without modifying physical state.
- Provider failures invalidate the run. Strict replay cannot silently substitute a
  mock response for a missing recorded response.
- The legacy latent-event environment remains a separate mode. Its E1--E4 probes
  are not used to evaluate commons runs.

## Measures and limits

The observer reports projects released/completed, delivery attempts and failures,
completion latency, resources consumed, invalid intentions, record versions and
reads, newcomer participation and time to first successful delivery, and outcomes
before/after the common intervention boundary. Unfinished tasks and incomplete
newcomer acquisition remain visible rather than being discarded.

Access followed by a decision is an exposure association, not proof that the record
caused the decision. The randomized records treatment estimates availability's
total effect, including the time spent writing and reading. Separate equal-time
controls would be needed to isolate information content from opportunity cost.
Use independently replicated worlds as experimental units; do not treat utterances
or task attempts as independent world replications.

The offline mock is a software exercise with a hand-written policy. Its results
cannot support claims about autonomous language-model culture or the hypotheses.
Human evaluation, validated semantic probes, model replication and pilot-informed
sample sizing are still required before a substantive research claim.

## Deliberately deferred world features

Negotiated institutions, dynamic reputation, voluntary exhibitions, construction of
new facilities, richer project composition and general-purpose scheduling belong
to later stages of the world plan. They should be added when they enable a specified
question, with their own manipulation and validation, rather than bundled into the
first records-by-environment experiment.

## Research foundations

- [Concordia](https://arxiv.org/abs/2312.03664): language-mediated grounded actions.
- [GlossoGen](https://arxiv.org/abs/2609.01491): partial information, communication
  pressure and explicit negotiation; a 2026 preprint.
- [TerraLingua](https://arxiv.org/abs/2603.16910): persistent artifacts and ecological
  inheritance; a close precedent, not a novelty claim for our artifact system.
- [Lowe et al.](https://arxiv.org/abs/1903.05168): observed communication need not
  establish causal influence on behavior.
- [Zhou et al.](https://arxiv.org/abs/2403.05020): information boundaries matter in
  LLM social simulation.

## Running and inspecting the milestone

Use Windows CMD from the repository root. Setup uses the tested direct dependency
versions in `requirements-dev.txt`; transitive dependencies are not fully locked.

```cmd
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
.venv\Scripts\python.exe -X utf8 -m backend.cli run --config configs\commons_smoke.yaml --analyze
.venv\Scripts\python.exe -X utf8 -m backend.cli serve
```

Select the printed run in the observer at http://127.0.0.1:8765. **World** shows the
material state and archive at each tick; **Culture** shows the RQ2 measures and
question coverage. The run launcher leaves the duration from the selected config
unless the user explicitly supplies a day count.

The four main configuration files differ only in the two treatment booleans and
their descriptive run names. Run each cell with the same seed, and repeat the
complete set across independent seeds. This CMD command exercises the full set
offline; it does not run a scientific experiment with language models:

```cmd
for %C in (commons commons_records_stable commons_no_records_change commons_no_records_stable) do .venv\Scripts\python.exe -X utf8 -m backend.cli run --config configs\%C.yaml --set llm.backend=mock --analyze
```

For actual model runs, omit `llm.backend=mock` to use the configured Claude CLI.
An Anthropic API run also needs an API model identifier, not the CLI alias `haiku`.
Choose and record the model explicitly when designing the experiment. No live
provider experiment was needed to validate this implementation.

Replace the run names below with the directories printed by the run commands:

```cmd
.venv\Scripts\python.exe -X utf8 -m backend.cli replay runs\YOUR_RUN_ID
.venv\Scripts\python.exe -X utf8 -m backend.cli compare runs\RECORDS_CHANGE runs\RECORDS_STABLE runs\NO_RECORDS_CHANGE runs\NO_RECORDS_STABLE
.venv\Scripts\python.exe -X utf8 -m pytest -q tests
```

Files are written only to a new run directory. A provider error or replay miss
marks the run `failed`; the commons analyzer rejects failed or unfinished runs.
Snapshots preserve daily memories for later observer development but are not
complete resumable execution checkpoints. Pending actions and incomplete tasks
at the run horizon remain explicit in the final state and analysis.

## Implementation map and validation

| Component | File | Important validation |
|---|---|---|
| Pure world and action contracts | `backend/simulation/commons.py` | Travel duration, custody, exclusive capacity, costs, failure and recovery |
| Agent decision interface | `backend/agents/commons_planner.py`, `backend/prompts/commons_action_v1.txt` | Local observations only; no treatment schedule or physical answer |
| Memory and execution bridge | `backend/simulation/commons_runtime.py` | Actual speech/read exposure; fresh newcomer memory; daily snapshots |
| RQ2 descriptive observer | `backend/analysis/commons.py` | Correct denominators, unfinished outcomes, record versions, common phase boundary |
| Integration and replay | `backend/simulation/engine.py`, `backend/llm/client.py` | Byte-identical replay; hard failure on missing responses/provider errors |
| Inspection interface | `frontend/` | World timeline, record history, question coverage and claim limits |

The tests cover all four treatment cells. Additional contracts check that record
contents remain unavailable until read, edits cannot silently overwrite a changed
version, a departed member's resources stay in the world, and unrelated agent
activity cannot alter the exogenous schedule. The original environment's tests
remain part of the regression suite.

The initial implementation uses direct local speech with a pinned audience and
deterministic perception of explicitly addressed outcomes. It does not reuse the
legacy four-utterance conversation scheduler, probabilistic event attention, or
forced movement. Optional cognition modules are rejected in this mode until their
behavior across arrival/departure is defined and validated. Agents keep frozen
model parameters; adaptation occurs through observation, memory and records.
