# MemeWorld ontology v3 — implementation contract

Status: design contract, 2026-09-19. Nothing in this document is implemented yet. It **extends**
[ONTOLOGY_V2.md](ONTOLOGY_V2.md). Wherever this document is silent, v2 applies unchanged: the world script, random
streams, cognition, observer and design runner. Sources: the collaborators' memo
([research/MEMO_2026-09-19_research_landscape.md](research/MEMO_2026-09-19_research_landscape.md)), three
candidate designs (measurement-first, ecology-first, minimal-build-first) and two reviews (scientific validity;
engineering feasibility and constraint safety). Appendix A maps every review finding to where this contract
resolves it.

**Thesis in one sentence.** The campus becomes the home of an agent-run Makerspace co-op. Its shared laser cutter
fails in ways that can be diagnosed objectively. v3 asks whether a persistent shared binder carries useful
interpretations of those failures across member turnover, and whether the same binder slows reinterpretation
after the hidden failure mechanism changes. Each question is answered with **history-matched counterfactual
branches**: every arm of a seed shares the same society up to the moment of manipulation, and the world's
random draws are identical across arms. Meaning is measured on **isolated checkpoint copies** of agents'
memories, with a matched test battery that has ground truth for each regime.

Hard constraints (v2's, unchanged, enforced by tests; §8.1 step 10):
- never tell agents to invent memes, slang, names, labels, terms, rules or shorthand;
- never reveal hidden state to agents. That covers the regime, the mapping, causes, symptom-class ids, content
  packs, job ids, outcome probabilities or uniforms, branch/arm/condition names, rotation schemes and E1–E4;
- never feed observer output back into the simulation. Probes, targets, codes, counterfactuals and metrics are
  never read by simulation code;
- no meme state inside agents. Agents hold no knowledge, procedure or meaning fields. The binder and its read
  receipts are world state;
- WORLD / AGENT / EXPERIMENT CONTROLLER / OBSERVER layers stay separate;
- agents only: there is no human player.

v3 adds:
- observer probes run on copies in a separate process. They never write into a run directory's simulation files,
  and their LLM calls never enter a run's `llm_calls.jsonl`;
- the binder holds only agent-written text. The world adds nothing but author and time stamps;
- world-provided wording (menu options, symptom surfaces, the panel code, onboarding, the tally template, binder
  stamps) is always flagged as world text by the observer and never counted as agent coinage.

Engineering rules (v2's, plus two):
- Never edit an existing file under `backend/prompts/`; add new versioned files.
- Every v3 mechanism is **off by default** in `configs/default.yaml`. `configs/v3_base.yaml` switches on the
  first-study settings.
- Every mechanism writes trace records, so the observer can check that the manipulation happened.
- v3 decision-log entries are **D74+** (v2 used D51–D73). The integrator assigns the final numbers.
- New: **LLM errors are never recorded as model output** (§6.1).
- New: a study runs at **one frozen git sha with one set of prompt hashes**. Branching refuses to run otherwise.

---------------------------------------------------------------------------------------------------

## 0. Research position and thesis

### 0.1 Question
From the memo (§1): *when a community's circumstances change, how do its existing ideas acquire new meanings, and
when does cultural memory help or obstruct that adaptation?* The binding first study is the memo's
recommendation: **do persistent shared records preserve useful meanings across participant turnover, and do the
same records slow reinterpretation after environmental change?**

### 0.2 Research questions in v3

| RQ (memo §7) | Role in v3 | Where measured |
|---|---|---|
| RQ2: records → continuity vs inertia vs revision | **Primary.** Continuity is H1 and inertia is H2 (confirmatory). Revision history is a Tier-2 extension. | §7.4 |
| RQ1: an established expression acquires a different interpretation | **Measured throughout; secondary.** "Expression lag" of a world-anchored term W and, if one forms, an agent-coined X, against a stable control Y. | §5.4–5.5 |
| RQ3: tolerated disagreement / local specialization | Exploratory in study 1 (AM vs PM divergence). A dedicated extension design (§7.8). | §5.5, §7.8 |
| RQ4: history effects under matched present conditions | Gated extension: a revert branch vs a no-shift branch continued. | §7.8 |
| v2 NEED/LINK/WORDING bottlenecks, hidden families E1–E4 | **Calibration and a secondary study** on the same codebase. E1–E4 are off in v3 (`latent_events.event_rate: 0`). | §9.2 |

### 0.3 Causal design in one paragraph
One laser cutter. A symptom class K1 has a hidden cause that swaps at a scheduled regime change: the lens (A) vs
damp sheets (B), counterbalanced by seed parity. A second class K2 is a stable control (always the belt). A third
class K3 appears only after the change. Jobs, faults, symptoms, menu orders and outcome uniforms are
pre-generated from `world_seed` (common random numbers). The same job with the same action therefore has the same
outcome in every arm of a seed. Members read and write a binder that hangs by the machine. For each seed, one
**trunk** runs founding (D1–D3) and wave-1 turnover (D4) with the binder kept. Two **records interventions** are
then branched from identical states:
- **wipe at the turnover point** (D4 start) gives the continuity contrast;
- **wipe at the change point** (D5 start, the moment the hidden cause swaps) gives the inertia contrast. Heads are
  identical before the change, so there is no level confound.

A no-shift branch is the positive control, the placebo (a visible supplier change without a causal change) and
the null for expression-level change. Wave-2 newcomers arrive after the change, so an old-regime answer from them
can only be inherited.

### 0.4 What v3 is built from
Reused from v2:
- the world script with `world_seed` streams (extended with job, roster, cue and tally records);
- `rngs.seed_rng`, `Agent.stream` substreams, perception with viewpoints, the lossy encoder with its lens,
  reminding v2, verbatim Wording records, source weights, catch-ups, and group conversation (reused for the
  members' meeting);
- typed provenance, emergence, funnel manipulation checks and `outcomes.json`;
- the design runner and manifest `code_version`, and the record/replay LLM cache (which becomes
  replay-then-branch).

New:
- the workshop task layer, the binder, roster and turnover, and work decisions;
- checkpoints, prefix replay with branching and design trees;
- the LB1 battery with its isolated probe runner, and the v3 metrics.

### 0.5 Explicit non-claims
- **Artificial societies only.** Results describe societies of Claude Haiku 4.5 agents under these constraints
  (memo §2.4). They are not evidence about human cultural evolution. Claims are sufficiency claims about the
  modelled mechanisms.
- **No claim that agents coin expressions.** v1 and v2 found no durable coined conventions. RQ1 analyses of an
  agent-coined X are conditional on X existing. W (the panel code) is world-provided and never counted as
  emergence.
- **"Records help" is a total effect** of this binder: always shown at decision time, dated and attributed, with a
  neutral write prompt. When a wipe coincides with turnover, incumbents lose the record too. Situated accuracy
  (binder in the prompt) is information availability, not meaning. Meaning claims rest on memory-only probes and
  comprehension contrasts.
- **Not an open-ended ecology** (unlike TerraLingua or Emergence World), and not newcomer acquisition under
  information asymmetry as such (unlike GlossoGen). The contribution is controlled, history-matched interventions
  on shared records at the turnover and change boundaries, scored against regime ground truth with isolated
  temporal probes.
- **Not cumulative improvement** (memo §2.3), and not normative change beyond an exploratory coded NORM flag.
- **Sampling temperature is not controlled** (claude CLI, D1). Divergence between replicate branches comes from
  seeded retrieval and lens sampling. It is not LLM sampling variance and is labelled as such.
- **Small n.** 8 societies per arm, detecting only large, consistent effects (§7.5). Everything else is effect-size
  estimation for a follow-up.

---------------------------------------------------------------------------------------------------

## 1. World

### 1.1 Setting and places
- The campus, routines, meals, catch-ups and reflection all continue. The co-op is added on top.
- **Research Lab › Makerspace** (existing arena): the laser cutter, with the binder hanging next to it.
- **Research Lab › Stockroom** (new arena; a one-line addition to `world.ARENAS`, appended so `default_arena` is
  unchanged): the stores desk, deliveries and sheet stock.
- **Arena visibility.** Perception gains `visibility: "arena"`: a fact is perceivable only by agents in the
  beat's arena (participants still perceive their own facts with p = 1). Job and cue facts use it
  (`workshop.symptom_visibility: arena`), so crews and stores really have different information. v2 facts keep
  `"all"`.
- **Calendar.** The co-op runs every simulated day. The clock is v2's: 07:30–22:30, 15-minute ticks, 60 ticks a
  day. D1 is Monday 2026-09-14. Day d starts at global tick `T(d) = 60·(d−1)`, so D4 = 180, D5 = 240, D6 = 300
  and D7 = 360.

### 1.2 The machine, symptoms, causes, actions (content pack `laser_alpha`)

| hidden id | what agents may perceive (plain past tense; 8 world surfaces each, 10–25 words) | cause |
|---|---|---|
| K1 (changing) | the cut stops short on the left, with browned edges. "The first sheet came out with the left third still attached, and the edges there were the color of weak tea." | regime-dependent (§1.4) |
| K2 (stable control) | doubled or wavy lines on the right. "Near the right edge every line came out doubled, as if the head had wobbled." | always BELT |
| K3 (B only) | M1: "During the cut there was a faint crackling, and tiny bubbles lined the edges." M2: "The lines came out wider than the file said, and a faint haze clouded the surface." | the B-cause |
| K0 | a clean cut; with p = 0.3 an irrelevant oddity ("the exhaust fan sounded louder than usual") | none |

- **Panel code (a world-anchored expression, W).** With `workshop.panel_code.enabled`, a K1 job shows the extra
  fact "The laser's panel showed F4." with probability 0.5, drawn from its own regime-independent uniform. F4
  never appears for K0, K2 or K3.
- F4's form is fixed by the world, while its practical implication changes at the regime change. This guarantees
  an RQ1 target even if agents coin nothing. The observer flags F4 as world text (§5.4).
- **Lexical hygiene** (validated in `tests/test_v3_world.py`, reusing v2's content-bigram machinery):
  - no content bigram occurs in more than 2 surfaces of a class, or in surfaces of two different classes;
  - no nickname-like phrase or quoted label;
  - surfaces never name a cause or a fix;
  - no content bigram is shared with battery held-out wording (§5.3).

**Action menu.** The same six options appear in the world, the decision prompt and the battery. The order is
pre-drawn per job and per attempt.

| id | text |
|---|---|
| `rerun` | re-run the sheet as it is |
| `lens` | clean the focus lens, then re-run |
| `dry` | dry the sheets on the heated rack for 15 minutes, then re-run |
| `belt` | tighten the drive belt, then re-run |
| `slow` | slow the cutting speed down, then re-run |
| `stop` | stop and leave the job for later |

- Every fix takes one tick, so the correct fix costs the same in both regimes.
- Content pack **`laser_beta`** (causes AIR = air-assist nozzle clogged, WARP = warped sheet, BELT; fixes "clear the
  air-assist nozzle", "flatten the sheet with hold-down pins") is the pre-registered alternative. It is used if the
  alpha prior is unbalanced (gate G1).

### 1.3 Outcome function and ground truth (`backend/simulation/workshop.py::OutcomeModel`)

p(success | action, cause):

| action \ cause | LENS | DAMP | BELT |
|---|---|---|---|
| `rerun` | .10 | .10 | .10 |
| `lens` | **.85** | .10 | .10 |
| `dry` | .10 | **.85** | .10 |
| `belt` | .10 | .10 | **.85** |
| `slow` (hedge) | .35 | .35 | .20 |
| `stop` | no attempt: the job is deferred, not delivered, and scored "defer" | | |

- Success iff `u[job, attempt] < p(action, cause)`. The uniform is pre-drawn, one per job and attempt (maximal
  coupling), so the same job and action have the same outcome in every arm.
- At most `workshop.max_attempts: 2`. Two failures mean the job is not delivered. It is **not** re-queued, because
  re-queueing would make the schedule depend on agents.
- K0 jobs need no decision (the clean outcome is certain).
- **Ground truth:** `gt(item, regime, mapping) = argmax_a p(a | cause_regime(class))`. It is computed by one
  function, used by the engine for outcomes and by `analysis/battery/gt.py` for scoring. No LLM ever judges
  correctness.
- **Scoring categories** (post-change): **old** = GT(A), **new** = GT(B), **other** = anything else (slow, rerun,
  belt, stop). Hedging is reported, never counted as reinterpretation.
- Learnability lever (G2): `success: {match: .90, other_fix: .05, rerun: .05}`.

### 1.4 Regimes, mapping and the change schedule (hidden)
- **Mapping** by world-seed parity (`regimes.mapping: auto`):
  - **M1** (odd): A = LENS, B = DAMP, with K3 surfaces of the "crackling, bubbles" type.
  - **M2** (even): A = DAMP, B = LENS, with K3 surfaces of the "wide lines, haze" type.
  - The pretrained favourite for an incomplete cut therefore cannot pass for adaptation. The prior's pull is
    measured by C0 (§5.6), and mapping is a blocking variable.
- **Schedule** `regimes.schedule: [{day, regime}]`. Shift arms use `[{day: 1, regime: A}, {day: 5, regime: B}]`;
  the no-shift arm uses `[{day: 1, regime: A}]`. From B on, K1's cause swaps and K3 starts to occur.
- **Cue** `regimes.cues: [{day: 4, time: "16:45", arena: Stockroom, text_key: new_supplier}]` is part of the trunk,
  so it is identical in every branch, including no-shift (where it is a placebo). The text is the same for both
  mappings: "A pallet of acrylic sheets from a different supplier was delivered to the stockroom; the co-op starts
  cutting from it tomorrow."
  - It never mentions faults.
  - Its diagnosticity (does "different supplier" pull toward `dry` or toward `lens`?) is measured with C0 cue
    items. It enters as a covariate and is gated (G1).
- **Revert (RQ4):** add `{day: 7, regime: A}` and the cue `{day: 7, time: "08:45", arena: Stockroom, text_key:
  usual_supplier}`. The no-shift comparison arm gets the identical cue (§7.8).
- **Change-type library** for later studies: `mechanism_swap` (study 1); `new_symptom` (K3 without a swap);
  `material_property` (success-table shift); `demand_shift` (class weights or project mix); `split` (a
  crew-specific material, RQ3).
- `workshop.causal: scrambled` is a false-positive control. The cause of each faulted job is uniform over the three
  causes, drawn from its own uniform and independent of class, so learnable structure is absent.
- Nobody ever sees causes, regimes, the mapping, p or u. They are never named in any prompt.

### 1.5 Jobs and their lifecycle
- **8 jobs a day** in fixed slots. AM: 09:00, 10:00, 11:00, 12:00 (k = 6, 10, 14, 18). PM: 13:30, 14:30, 15:30,
  16:30 (k = 24, 28, 32, 36).
- Each job is for a campus project, drawn from a pool (for example "24 name plates for the robotics team's open
  house" or "tube racks for the bio lab").
- `p_fault = 0.75`. Among faults, K1 is 0.70 and K2 is 0.30 in **both** regimes. Fault-free jobs are K0.
- In a regime with a K3 cause, a K0 job becomes K3 with `k3_from_k0 = 0.6`, from its own uniform. K3 is therefore
  carved out of the fault-free mass: the K1 and K2 job slots are identical across regimes, which is what makes
  jobs pairable.
- Expected jobs per day: K1 4.2, K2 1.8, and K0 2.0 (A) or K0 0.8 + K3 1.2 (B).

Lifecycle (at most 4 ticks, so the next job's slot is never reached):

| tick | world | agents |
|---|---|---|
| t0 | job-start fact (salience .35) + symptom fact(s) (.6; F4 fact .5) at the Makerspace, visibility arena; operator = participant | phase 4b: the operator renders the symptom (viewpoint), then decides attempt 1: an action, or G = ask someone present (§4.4) |
| (t0+1) | only if G: the clarification ran in phase 6 at t0 | operator decides attempt 1 (G no longer offered) |
| next | outcome fact (.55): "Dev cleaned the focus lens and re-ran the sheet; the cut went all the way through." / "...; the left side still didn't cut through." | if it failed and attempts remain: attempt-2 decision |
| end | job-end; delivered or not | phase 4c: RecordWrite offer to the operator after faulted jobs (§2.4); the job episode is queued for encoding next tick (§4.2) |

- 17:30 (k = 40) is the **tally** at the Makerspace. The rostered stores member and the PM crew perceive a
  templated fact: "End-of-day tally at the laser: 6 of 8 jobs delivered; the bio lab's tube racks and the a cappella
  stencils were not."
- The tally is also a binder reading and writing point for stores (§2.3–2.4).

### 1.6 Roles, groups and information asymmetry
- Always **8 active members** in three roles, with circles reused from v2:
  - **AM crew** (`c_lab`: maya, dev, hana), on shift 08:45–13:15;
  - **PM crew** (`c_dorm`: ethan, leo, jordan), on shift 13:00–17:45;
  - **stores** (priya, sofia), in the Stockroom 08:45–12:00 and 13:00–17:45.
- Shift entries replace the overlapping routine entries through a `plan_day` hook. Meals, classes outside shift
  hours and evenings stay.
- **Operator rota** (world-script stream `operator`): a shift's 4 jobs go round-robin over the 3 crew members from
  a per-day offset, so each member operates 1–2 jobs. On their first two days, newcomers are first in the
  round-robin (2 jobs per shift).

| who | perceives |
|---|---|
| operator | own symptom (p = 1), own attempts and outcomes, the binder view at decision time |
| crewmates on shift (same arena) | symptoms and outcomes at near attention; says-aloud remarks; clarifications they overhear |
| other crew | nothing from the laser directly (arena visibility); only what reaches them by talk, handover, meeting or the binder |
| stores | the cue (D4 16:45), the tally, the binder at tally time; never symptoms directly |

- The change is visible where it does not hurt (stores) and hurts where it is barely visible (crews). Cause-side
  and symptom-side information meet only through talk or the binder.

### 1.7 World script v3 (pre-generated, common random numbers)
- `backend/simulation/workshop.py::generate(cfg, roster, clock) -> list[JobInstance | WorldFact]`. It is called by
  `world_script.generate` when `workshop.enabled`, and the records are appended to `world_script.jsonl`.
- **Streams are day-local and keyed per job:** `seed_rng(world_seed, "ws3", purpose, day, slot)` with purposes
  `fault, class, k3, code, surface, project, oddity, menu, outcome, operator, cause_scr`.
- Every job draws a **fixed number** of values from every stream, whatever the config does with them.
- Consequences:
  - extending the horizon or overlaying a later day leaves every earlier day byte-identical;
  - switching `panel_code` or `causal` shifts nothing else.
- **Regime-independent script.** A job record stores its class and cause under **each** regime. The schedule only
  selects between them at runtime. One script therefore serves every arm of a seed. Test: the records of days
  < `at_day` are identical across every node of a seed.
- **Deterministic (no randomness):**
  - roster and turnover (§3.2, a Latin square by seed);
  - cue, orientation, farewell and tally times;
  - meetings and handovers.
- Beat locations are pinned to the Makerspace or Stockroom, and the operator is forced there as a mover. v2's
  dependence of beat locations on `seed` through plans (D51) does not arise.
- E1–E4 generation still runs with `event_rate: 0` and yields no instances.

### 1.8 JobInstance record
```
{"id": "j04.2", "kind": "job", "generator": "workshop_v1", "day": 4, "slot": 2, "shift": "am", "start_tick": 194,
 "operator": "wei", "project": "tube racks for the bio lab",
 "menu_order": [["dry","rerun","stop","slow","lens","belt"], ["belt","slow","lens","rerun","dry","stop"]],
 "surface": {"K1": 5, "K2": 2, "K3": 7, "K0": 1}, "code": true, "oddity": null,
 "hidden": {"fault": true, "class": {"A": "K1", "B": "K1"}, "cause": {"A": "LENS", "B": "DAMP"},
            "u_attempt": [0.40, 0.61], "u": {"fault": 0.21, "class": 0.33, "k3": 0.71, "code": 0.12}}}
```
- Other record kinds: `cue`, `orientation`, `farewell`, `roster` (depart, arrive), `tally`, `meeting`, `handover`,
  `transition`. Transitions are deterministic from config.
- Beats are built at runtime from the record and the active regime, because outcomes depend on decisions.
- Fact ids are `"{job}.b{beat}.f{i}"`.
- `hidden.*`, `regime`, `mapping` and the `job_truth` / `regime_active` trace records are stripped by the API in
  demo mode.

### 1.9 Example episode (seed 13, M1: A = LENS, B = DAMP)
1. **Tue 10:00, job j02.1** ("24 name plates for the robotics team's open house"). The script says: fault
   (u = .21), K1, surface s5, F4 shown. The cause under A is LENS. The operator is Dev, and Maya and Hana are on
   shift.
2. Dev's rendered facts: "Dev's first sheet came out with the left third still attached, and the edges there were
   the color of weak tea." and "The panel showed F4."
3. The binder's front page (Mon 16:40, Maya) reads: "Wavy lines on the right: tightened the belt knob under the
   bed, fine after."
4. Dev picks "clean the focus lens" (p = .85, u1 = .40) and succeeds. Maya's episode encoding triggers a REACT
   remark, "tea edges again? that's the lens.", heard by Dev and Hana.
5. Dev's RecordWrite adds a log note: "Left side not cutting, tea-colored edges, F4 -> cleaned lens, fine."
   **Wed 11:20:** Dev rewrites the front page, adding "F4 or tea edges on the left = lens, wipe it first."
6. **C3** (end of Wed; trunk): the pre-registered selector picks X = "tea edges". It has 5 tokens linked to K1
   jobs (2 of them in binder entries) from 3 producers, and C0 production of 0/13.
7. **Thu 07:30:** wave 1 (seed 13 rotation, §3.2). Hana departs and Wei joins the AM crew; Ethan and Priya also
   leave, replaced by Amara (PM) and Felix (stores).
   - Trunk: "Someone put a new cover on the co-op binder by the laser cutter; all the old pages are still in it."
   - `wipe4`: "Someone replaced the co-op binder by the laser cutter with a new, empty one; the old pages were
     boxed up and archived."
8. **Thu 16:45:** Sofia and Felix, at the Stockroom, perceive the new-supplier delivery. The crews don't.
9. **Fri 09:00, job j05.0.** K1 surface s2: "Halfway through, the beam stopped biting on the left; those pieces had
   tan rims." Wei operates, with u = [.52, .61].
   - `keep_shift` (cause DAMP): Wei reads Dev's front-page line and tries the lens (p = .10), which fails. He then
     slows the speed (p = .35) and fails again. The job is not delivered, and the 17:30 tally lists it.
   - `wipe_shift` (the same job and u): the binder view reads "There is nothing written in the binder yet." Wei
     relies on his D4 memory of Dev saying at the laser "F4 is basically always the lens" and fails with the lens.
     He then dries the sheets (p = .85) and succeeds.
   - `keep_noshift` (cause LENS): the lens succeeds.
10. **C6 probe** with the note "tea edges again on the laser today.":
    - `keep_shift` Wei: "the lens needs wiping" (the old implication);
    - `wipe_shift` Wei: "the sheets might be damp, dry them first".

    SRI (§5.5) pools exactly this contrast over members and framings.

---------------------------------------------------------------------------------------------------

## 2. Records: the co-op binder (`backend/simulation/records.py`, WORLD layer)

### 2.1 Entities
```
Binder   {binder_id, created_tick, front: [Revision], log: [Entry], archived: [Binder]}   # world state only
Revision {rev_id, author, tick, text (<= records.front_max_chars = 600), revision_of}
Entry    {entry_id, author, tick, text (<= records.log_max_chars = 200), context: job|tally|farewell, job_id}
Receipt  {(agent, entry_or_rev_id) -> first_read_tick}          # world-side read receipts, never shown to agents
```
- The world **always** keeps every revision and every archived binder. Conditions change only what persists into
  the current binder and what a reader sees.
- There is one binder (for the one machine). A front-page rewrite replaces the whole text. The log is append-only.
- The binder is snapshotted into every checkpoint (`binder.json`) and appears in the frame record for the UI.

### 2.2 What agents see (rendering; identical format in every arm)
```
The co-op binder next to the laser cutter.
Front page (last rewritten Wed 11:20 by Dev):
  "Wavy lines on the right: tightened the belt knob under the bed, fine after. F4 or tea edges on the left = lens, wipe it first."
Log, newest first:
  [Wed 16:05, Hana] ...
  [Wed 14:20, Dev] ...                              (at most records.view_log_n = 5 notes)
```
- **Author first name and day/time are always shown** (critic fix: provenance is held constant across arms).
- `records.display`:
  - `current` (study 1): the rendering above;
  - `history`: additionally up to 2 superseded front-page versions, each as "Earlier front page (Tue 16:40, Maya),
    replaced Wed 11:20: …";
  - `none`: no binder at all.
- An empty binder always renders "There is nothing written in the binder yet.", whatever the reason it is empty.
- The rendering never includes job ids, class or cause ids, or hidden fields (audited).

### 2.3 Reading
- **At decision time** (`records.consult: always`): the operator's JobDecision prompt includes the rendered view at
  every attempt.
  - `on_choice` (robustness arm) instead adds a menu option "look through the binder first" and re-asks with the
    view shown.
  - `never` keeps the binder out of decisions. It is still readable at the tally.
- **At the tally**: the rostered stores member reads the view.
- **Encoding** (`records.encode_on_read: new_only`):
  - a read creates an `AgentObservation(source_type="record")` whose facts are only the entries or revisions this
    agent has not read before, taken from the world-side receipts, and formatted exactly as displayed;
  - it is encoded next tick by the normal lossy encoder, with the verb "read in the co-op binder" and salience 0.5;
  - `MemoryMeta.record_ids` holds the entry and revision ids;
  - re-reads are **not** re-encoded, which avoids flooding memory with near-duplicates (critic fix);
  - reminding may fire on record reads (`reminding.on_sources` includes `record` in v3_base);
  - verbatim stickiness may keep a distinctive written phrase (`memory.verbatim.on_sources` includes `record`;
    `exclude_self` excludes one's own entries);
  - `never` makes the binder purely external (a later factor).
- `records.retrieval: recent` shows the last 5 log notes. `relevant` shows the 5 notes lexically nearest to the
  operator's perceived symptom (hash embedder, ties broken by recency), for a later factor.

### 2.4 Writing
- **Offers:**
  - (i) to the operator after every **faulted** job's final outcome;
  - (ii) to the rostered stores member at the tally;
  - (iii) to a departing member at the end of their last shift.
- Writing is never forced. Offers are applied in phase 4c in sorted (tick, agent) order, after all of the tick's
  decisions. A front rewrite made from an older view still becomes the newest revision, and history keeps both.
- The author gets a no-LLM self memory ("Dev wrote in the co-op binder: …", source type `self`).

Prompt `backend/prompts/record_write_v1.txt` (new; the neutral wording is a critic fix):
```
{ISS}
It is {Thursday 10:30}. {situation: "{first} just finished a job on the co-op's laser cutter." |
 "{first} just went over today's jobs at the laser." | "It is {first}'s last shift at the co-op."}
What {first} remembers of it:
{the operator's own perceived job facts, attempts and outcomes | the tally facts | 3 retrieved memories about the laser}
Next to the laser cutter hangs the co-op binder. It has a front page that any member can rewrite and a log where
members add dated notes. Members look at it before running jobs. It currently reads:
{view, §2.2}
Is there anything {first} wants the next person at the laser to know? {first} can leave the binder as it is, add
a note to the log (up to 200 characters), or rewrite the whole front page (up to 600 characters).
Answer with one JSON object and nothing else:
{"choice": "none" | "log" | "front", "text": "<exactly what {first} writes, or null>"}
```
- No wording asks for tips, rules, procedures, names, labels or terms. The length limits are given only in
  characters. Longer texts are truncated at a word boundary and the truncation is traced.

### 2.5 Transitions (the manipulation) and matched facts
`records.transitions: [{day, mode}]` is applied in the day-start hook (phase 0a, tick `T(day)`), after roster
changes:
- **keep**: the binder is unchanged. A matched world fact is released at 08:45 at the Makerspace (salience 0.5):
  "Someone put a new cover on the co-op binder by the laser cutter; all the old pages are still in it."
- **wipe**: the current binder is moved to `archived` (still in world state and still in checkpoints, but not
  readable by agents), and a fresh empty binder replaces it. The matched fact reads: "Someone replaced the co-op
  binder by the laser cutter with a new, empty one; the old pages were boxed up and archived."

Every arm gets exactly one transition fact on each transition day. Matched salience removes "something happened
to the binder" as a difference between arms. How often talk mentions the binder, per arm, is counted as a
manipulation side-effect (§5.8).

### 2.6 Authority and provenance
- `records.authority: members` (study 1): entries are signed by their member authors.
- Departed authors' names stay visible, and newcomers never met them. The share of front-page text written by
  departed authors is an observer metric.
- `manager_signed` (later factor): the front page is shown as signed by an NPC shop manager, "Front page (kept by
  the shop manager)". The member author stays in the trace. This tests source authority.
- `records.replies` (off): a reader may attach a question to an entry, which gives repair through the record.

### 2.7 Manipulable conditions

| key | levels | study 1 |
|---|---|---|
| `records.enabled` | false / true | true (the never-had-records arm is Tier 3) |
| `records.transitions` | keep / wipe on a day | keep@D4 (trunk); wipe@D4 (`wipe4`); keep or wipe@D5 (branches) |
| `records.display`, `history_from_day` | none / current / history | current; history from D5 in the Tier-2 revision arm |
| `records.consult` | always / on_choice / never | always; on_choice in the Tier-2 compliance arm |
| `records.encode_on_read` | new_only / never | new_only |
| `records.view_log_n` | 3 / 5 / 10 | 5 |
| `records.retrieval` | recent / relevant | recent |
| `records.revisable` | true / false (log only) | true |
| `records.authority` | members / manager_signed | members |
| `records.replies` | false / true | false |

### 2.8 Trace records
- `record_read {agent, tick, context: decision|tally, view_mode, entry_ids, rev_ids, new_ids, view_sha}`
- `record_write {agent, tick, offer: job|tally|farewell, choice, entry_id|rev_id, text, truncated, prompt, response}`
- `record_transition {day, mode, archived_binder_id}`
- `binder.json` in every checkpoint

---------------------------------------------------------------------------------------------------

## 3. Turnover (`backend/simulation/roster.py`, WORLD layer)

### 3.1 Persona universe and identity handling
- `configs/population/coop13.yaml`:
  - `agents:` the 8 homewood8 founders plus `coop_role`;
  - `reserves:` 5 newcomer personas: noor (Public Health, sophomore), tomas (History, junior), wei (Linguistics,
    freshman), amara (Psychology, sophomore), felix (Economics, junior). None has laser or materials expertise;
  - `roles:` am_crew, pm_crew, stores.
- `population_size: 8` counts active founders. Reserves are always loaded.
- **Everything name- or vocabulary-based is computed once, at t0, from the full persona universe of all 13**,
  identically in every arm. That covers the population lexicon, `encoder._person_names`, viewpoint name guards,
  verbatim distinctiveness and the observer's name filters.
- The effect: a departed author's name in a binder stamp never becomes a "distinctive phrase", and newcomers'
  profile vocabulary is lexicon (critic fix).
- `SimContext.active` is the active set. `ctx.agents` keeps every persona for name resolution.
- `Simulation.active_agents()` replaces direct iteration everywhere agents act or perceive.

### 3.2 Waves and rotation (identical in every arm of a seed)
`turnover.waves`:
- `[{day: 4, depart: {am_crew: 1, pm_crew: 1, stores: 1}}, {day: 6, depart: {am_crew: 1, pm_crew: 1}}]`, applied at
  the first tick of the day.
- By D6 each crew has 1 founder, 1 wave-1 newcomer (W1) and 1 wave-2 newcomer (W2). Stores has 1 founder and 1 W1.

Rotation by seed (`turnover.rotate: by_world_seed`; a Latin square, no randomness). Let i = (s−11) mod 3,
j = (s−11) mod 2, r = (s−11) mod 5, AM = [maya, dev, hana], PM = [ethan, leo, jordan], ST = [priya, sofia] and
R = [noor, tomas, wei, amara, felix].
- W1 departs AM[i], PM[(i+1) mod 3] and ST[j]. Arrivals: R[r] joins AM, R[r+1] PM, and R[r+2] stores.
- W2 departs AM[(i+1) mod 3] and PM[(i+2) mod 3]. Arrivals: R[r+3] joins AM and R[r+4] PM.
- All R indices are taken mod 5.

| seed | W1 out → in (AM, PM, ST) | W2 out → in (AM, PM) |
|---|---|---|
| 11 | maya→noor, leo→tomas, priya→wei | dev→amara, jordan→felix |
| 12 | dev→tomas, jordan→wei, sofia→amara | hana→felix, ethan→noor |
| 13 | hana→wei, ethan→amara, priya→felix | maya→noor, leo→tomas |

### 3.3 Departure
- **The afternoon before**, during the last hour of the member's last shift, a world fact is released in their
  shift arena (involves them; visibility arena): "Hana mentioned that today was her last shift at the co-op; she
  is moving to another lab next week."
- Incumbents present perceive it through normal perception. It is never injected into memory (critic fix).
- At the end of that shift, the departing member gets the farewell RecordWrite offer (§2.4). This happens in every
  arm.
- At the next day start: `state.active = False`. The agent leaves `ctx.active`. It gets no movement, perception,
  conversation, group seat, module hook or frame entry. Its memory is in the previous day's checkpoint, and
  `departed/<id>.json` points to it.
- Test: after departure the trace contains zero records authored by, perceiving, or addressed to the departed
  agent.
- Trace: `roster_change {day, agent, kind: depart, role}`.

### 3.4 Arrival and onboarding
- A fresh `Agent` from the reserve persona. Its random streams are `seed_rng(seed, id, name)`, identical across
  arms. It is placed at its profile home, and its first shift follows the plan hook.
- **Symmetric onboarding ties** (critic fix: newcomers are not strangers by construction). The newcomer and each
  crewmate get `{relation_type: "co-op crewmate", familiarity: 0.35, affinity: 0.5}` in **both** profiles.
  - That is at or above `RECOGNISE_FAMILIARITY = 0.3`, so each recognises the other by name.
  - Other active members get `{relation_type: "co-op member", familiarity: 0.2, affinity: 0.5}`.
- Both parties get one no-LLM seed memory per crewmate pair ("Wei and Maya met at co-op orientation; they are on
  the morning crew together."). These are created under the scope `t{T:04d}:00roster`, never `"seed"`, so prefix
  replay stays valid.
- **Orientation** at 08:45 in the shift arena. These are world facts, identical to what founders perceived on D1
  and free of fault or fix content:
  - "Maya showed Wei around the co-op: the laser runs a morning and an afternoon shift, campus groups send in
    jobs, the binder hangs next to the laser, and the stockroom keeps the sheets."
  - Those present perceive "Wei joined the co-op's morning crew today."
- Newcomers are first in the operator round-robin on their first two days (§1.6).
- Trace: `roster_change {day, agent, kind: arrive, role, replaces}`, `onboarding {agent, ties}`.
- Manipulation check: the talk rate to newcomers, by arm.

### 3.5 Reassignment (off in study 1)
`turnover.reassign: [{day, agent, to_role}]` moves an incumbent between crews, with the world fact "Leo is on the
morning crew from today." It is used by the RQ3 extension.

---------------------------------------------------------------------------------------------------

## 4. Agents

### 4.1 What stays the same
GA persona and memory stream, partial perception with viewpoints (D42), the lossy encoder with its memory lens
(D44), stochastic retrieval, reflection, react decisions, dyadic and group conversation, forgetting and named
random substreams all stay as in v2.

Agents get **no** new knowledge fields. The only new transient state is `JobView`, the prompt bundle for one
decision. It is discarded after the decision, like GA's scratch.

### 4.2 Job perception and episode encoding (`workshop.encode: per_job`)
- Job facts are released per beat and sampled per beat by v2 perception, so who notices what is decided at that
  tick.
- The operator's own job facts are viewpoint-rendered per beat, because the decision needs them. Encoding of job
  facts is deferred.
- At job end, each perceiver's noticed facts for that job become **one** episode observation:
  `source_type="perception"`, `event_ids=[job_id]`. For the operator it also includes an `inner` fact, "{first}
  had thought: <reason>", when `workshop.encode_reason: true`.
- The episode is queued into `pending_obs` and runs through the full phase-4 pipeline next tick: render any
  unrendered facts, then encode, reminding, and a react decision for non-operators.
- This mirrors v2's one memory per conversation. It keeps the episode coherent and saves about half the job-layer
  calls (proposed D75). `per_beat` restores v2 behaviour.

### 4.3 Job decision (`backend/agents/work.py::decide_job`, prompt `job_decision_v1.txt`, new)
```
{ISS}
It is {Thursday 10:00}. {first} is on the {morning} shift at the co-op's laser cutter in the Makerspace, running a
job: {project line}.
What {first} has noticed so far:
{own perceived job facts, oldest first; from attempt 2 on, what {first} tried and what happened}
{binder view (§2.2) when records are enabled and consult is always}
What {first} remembers that might matter:
{k = 5 retrieved memories; focal points = [perceived symptom text, "the co-op's laser cutter"]}
People here: {names or "nobody"}
{relationship lines}
What does {first} do next?
A. …  F. …            (the job's pre-drawn order for this attempt)
G. ask {name} something first      (only if someone is present and no question was asked on this job)
Answer with one JSON object and nothing else:
{"choice": "<letter>", "question": "<for G only: exactly what {first} asks, else null>",
 "says_aloud": "<exact words {first} says to whoever is around, or null>", "reason": "<a few words>"}
```
- Scope `t{tick:04d}:02work:{agent}`, run in parallel per operator in phase 4b.
- An unparseable reply gets one retry, then counts as `stop`, and is traced and excluded from accuracy.
- `says_aloud` becomes a remark utterance on v2's REACT path, with the same listener sampling and exposures.
- Trace: `job_decision {agent, job, attempt, menu_order, choice, action, question, says_aloud, reason, retrieved,
  binder_view_sha, binder_entry_ids, prompt, response}`.
- No template or rendered text names a class, cause, regime or rule. Nothing asks for names or labels.

### 4.4 Clarification and repair
Choice G is not handled inside the decision phase. It becomes a **forced TALK** (critic fix):
- the question is the opening line of a v2 dyadic conversation in phase 6 of the same tick, with
  `topic="clarify"`, `max_utterances = comm.clarify.max_utterances (4)`, and the context line "{s} is running a
  job on the laser cutter and asks {o} something.";
- participants encode it as usual, and bystanders may overhear;
- the operator decides attempt 1 at t0+1, with G removed. There is at most one question per job
  (`comm.clarify.max_per_job: 1`).
- This reuses the engine's `in_conversation`/claimed bookkeeping. Ordering stays deterministic.

Ordinary negotiation is unrestricted inside any conversation.

### 4.5 Scheduled talk and communication budgets

| key | study 1 | what it is |
|---|---|---|
| `comm.clarify` | enabled, 1 per job, 4 utterances | repair at the machine |
| `comm.handover` | enabled, 13:15, 6 utterances | the operator of the 12:00 job and the operator of the 13:30 job are forced to the Makerspace 13:00–13:30; context line: "{s} and {o} are at the laser cutter at the shift change." |
| `comm.meeting` | enabled, D2 and D5 at 18:30, 10 utterances, up to 8 participants | every active member is forced to the Makerspace 18:30–19:15; v2 `run_group_conversation(topic="meeting")`; context line: "The co-op's members are meeting at the Makerspace to go over how the week has been going." No agenda. |
| `conversation.max_utterances` | 4 (v2) | ordinary dyadic talk |
| `conversation.catchup` | on (D47) | evening talk between close friends: a cross-crew, off-record channel |

All of these are variables for Study 2 (§7.8). Their trace records are v2's `conversation` and `utterance` records
with `topic`, plus `handover` and `meeting` summaries.

### 4.6 v2 mechanism settings in the first study (`configs/v3_base.yaml`)

| mechanism | setting | why |
|---|---|---|
| viewpoints (D42), memory lens (D44), noise 0.3 | on | standard cognition |
| reminding (D48/D59) | on; `on_sources` adds `record` | linking job episodes is how the mapping gets learned |
| verbatim (D58) | on, `exclude_self`, lexicon distinctiveness; `on_sources` adds `record` | the wording channel for RQ1 |
| source weights (D50) | `{seed: 0.5, ambient: 0.5}` | experiences are not crowded out |
| catch-ups (D47) | on | the off-record channel between crews |
| group conversation (D63) | off; the meeting uses `comm.meeting` | meal-table group talk is not needed |
| need (D60), priming (D61) | off | priming can masquerade as transmission (D36) |
| referents, assignment, link_visibility, structure | not applicable (E1–E4 off) | |
| modules (emotion, social reward, prestige, conformity) | off | |
| reflection threshold | 60 (v2); cost lever 90 | |

### 4.7 Tick order (v3 additions marked ★)
```
0   modules.on_tick
0a★ day start (first tick of a day): roster (depart/arrive, ties, seed memories), records transitions
1   world: v2 events (none) + ★ workshop beats (job, outcome, cue, orientation, farewell, transition fact, tally)
2   movement (+★ shift entries, handover/meeting/tally movers)
3   perception (★ arena visibility)
4   per agent: viewpoint → encode → reminding → react   (★ job facts rendered but buffered; episodes from 4c
    of the previous tick are encoded here)
4b★ work decisions (parallel per operator)
4c★ apply: actions → outcome beats at t+1; G → forced talks; remarks; job ends → episode queue;
    record reads (receipts) and writes (sorted)
5   apply react decisions
6   conversations: forced talks (react TALK, ★clarify), ★handover, ★meeting, catch-ups, gate → decide_to_talk
6b  overheard encoding
7   reflection
8   frame; trace flush; ★ per-tick digest
8b★ day end: checkpoint (§5.1)
```

---------------------------------------------------------------------------------------------------

## 5. Observer and measurement (`backend/analysis/`; never imported by simulation code)

### 5.1 Checkpoints (`backend/experiment/checkpoint.py`, written by the engine's day-end hook)
- `checkpoints/C{d}/` is written at the end of every day. It is cheap, since it is only files.
- Contents:
  - `agents/<id>/associative_memory/` in GA format, one per **active** agent;
  - `memory_meta.json`, `agent_state.json` (role, cohort, arrival day, relationship table including onboarding
    ties, scratch essentials);
  - `binder.json` (all revisions, archived binders, receipts), `roster.json`;
  - `world_state.json` (hidden: regime, mapping, job outcomes to date);
  - `CHECKSUMS` (sha256 per file).
- Serialisation is deterministic (sorted keys). Files are write-once and made read-only.
- Branch runs regenerate identical pre-branch checkpoints by replay, and the observer probes each checksum only
  once (§5.2).
- **Study checkpoints:**
  - C3 (trunk; before turnover; target selection);
  - C4 (trunk and `wipe4`; after wave 1, regime A);
  - C5 (each D5 branch; the first day after the change point; light battery);
  - C6 (each D5 branch; after wave 2).
- **C0** is not a run checkpoint: it is the 13 personas with empty memory (§5.6).
- Trace: `checkpoint {id, tick, files: {path: sha256}}`.

### 5.2 Isolation protocol (`analysis/battery/runner.py`; CLI `probe <run> --checkpoint C4`)
1. A **separate process** runs after the segment has finished. By default it never runs concurrently with a
   simulation (§6.6).
2. It copies `C{d}` to `probes/C{d}/work/` and verifies `CHECKSUMS` before and after. A mismatch invalidates the
   probe.
3. It builds isolated Agent copies:
   - `ModuleStack([])`, no run tracer;
   - clock set to the checkpoint tick;
   - `memory_meta` loaded, so the D50 source weights apply exactly as in the simulation (critic fix: today's
     `_Ctx` has no meta);
   - the persona universe and the checkpoint roster loaded, departed members included.
4. Retrieval uses `touch=False`, k = 6 and τ = 0.7. The seed is `seed_rng(seed, "probe", ckpt, agent, item, form,
   order)`. Focal points are `[item text, "the co-op's laser cutter"]`; the fixed machine-level focal is a critic
   fix for the lexical embedder (D2).
5. The copies never encode, reflect or save.
6. **Separate LLMClient and cache** (`probes/C{d}/llm_calls.jsonl`), with scope `probe:C{d}:{agent}:{item}:{form}:
   {cue}:{framing}:{order}`. The same rule applies to the other observer clients: `dcf:`, `retest:`, `synth:`,
   `fresh:`, `code:`.
   - Tests assert that no observer purpose or scope ever appears in any simulation `llm_calls.jsonl`, so a branch
     can never replay an observer call.
   - The engine and agent code never import `backend.analysis` and never open `probes/`.
7. **The answering model is Haiku, the agents' model.** The probe measures agent cognition. Sonnet only codes free
   text (§5.10). This supersedes D41's use of the observer model for probes (proposed D83).
8. Probes of identical checkpoints (the same checksum) run once and are shared by every node that replays that
   prefix.
9. Every probe logs the **retrieval hit rate**: whether at least one retrieved node's meta links it to a job (via
   `event_ids`) or to a binder record. The rate is logged per item type and arm, and reported as a validity flag.

### 5.3 Battery LB1 (`analysis/battery/items_lb1_{pack}_{mapping}.yaml`; ground truth from `gt.py`)

| type | n | content | scored |
|---|---|---|---|
| K1c (K1 core) | 8 | 4 held-out wordings + 4 **world-wording twins** (world surfaces with the F4 sentence removed; a critic fix for retrieval) | yes, old/new/other |
| K1a (K1 ambiguous) | 2 | "Everything cut through, but the left-side edges were browner than usual." | yes |
| K3 | 3 | mapping-specific B-cause symptoms, held-out wording | yes |
| K2 | 4 | 2 held-out + 2 twins; matched to K1c in length and structure | yes (stable control) |
| K0 foils | 2 | "The cut was clean everywhere, but the exhaust fan sounded louder than usual." | yes (overextension) |
| CUE | 2 | K1 + "these sheets came from a different supplier's pallet"; K1 + "the machine had been running all morning" (neutral) | no; C0 cue diagnosticity only |

- Held-out wording shares no content bigram with world surfaces. No item shows F4, so item accuracy measures
  symptom interpretation. F4's meaning is measured with the cue formats.

**Formats** (one JSON call each; menu order re-drawn per call and recorded):
- **P, memory-only** (`probe_act_v1.txt`), on all 21 items. K1c items are asked in **2 menu orders**, giving 16
  responses per agent. 29 calls.
  ```
  {ISS} It is {Thursday} evening; {first} is thinking about the co-op's laser cutter.
  What {first} remembers: {6 retrieved memories}
  Suppose this happens on {first}'s next shift: {item}
  1) In a few words, how would {first} describe what's going on to another member?
  2) What would {first} do first? {A–F}   3) How sure is {first}, 1–5?
  JSON {"describe": "...", "choice": "<letter>", "confidence": n}
  ```
- **P-sit**: the same, plus "The binder next to the laser currently reads: {view at C{d} in this arm's display}".
  Run on K1c held-out ×4, K2 held-out ×2 and K3 ×3: 9 calls.
- **A, applicability** (`probe_apply_v1.txt`), one call per cue c ∈ {X, W, Y}: "Another member uses the words
  "{c}" when talking about the laser. For each situation below, would "{c}" fit what happened? fits / doesn't fit /
  not sure." The situations are K1c held-out ×4, K1a ×2, K3 ×3, K2 held-out ×2 and K0 ×2 (13). 3 calls.
- **N, comprehension** (`probe_note_v1.txt`). Cues X, W, Y and NONCE use the carrier "{c} again on the laser
  today."; NONE is "Heads up about the laser today." Three framings: a sticky note on the laser, a text message on
  the way in, overheard at lunch.
  - Questions: 1) In one sentence, what is it telling {first}? 2) If the laser acted up on {first}'s next job, what
    would {first} do first? A–F. 3) Has {first} heard these words before? yes / no / not sure.
  - 5 cues × 3 framings = 15 calls.
  - NONCE is a pseudoword pair matched to X in words and syllables, with Zipf 0 (for example "dal ombers").
- **Source-ablated** (newcomers at C4 and C6; ecology must-keep): P on K1c ×8 and K2 held-out ×2 (10 calls) after
  dropping nodes whose source is `conversation`, `overheard` or `record`, plus reflections and reminding thoughts
  whose evidence contains any of them.

**Battery by checkpoint (calls per agent):**

| checkpoint | battery | calls |
|---|---|---|
| C3 | P + N(W, NONCE) | 35 |
| C4 | full: P + P-sit + A + N | 56 |
| C4 in `wipe4` | P on K1c×2, K2, K1a, K0 | 24 |
| C5 (light) | P on K1c×2 + K3, and N(X, W, NONCE) in the note framing | 22 |
| C6 | full | 56 |

### 5.4 Target expressions
- **W (world-anchored, guaranteed):** the panel code "F4". It is flagged `in_world_text` and never counted as
  emergence. It is analysed only as a form whose implication may change.
- **X (agent-coined, if any).** Pre-registered and selected by `analysis/battery/targets.py` on **the trunk at C3**,
  before any branch runs, and frozen in `probes/targets.json` with its sha in the design's prereg log. X is the
  highest v2 emergence-score n-gram over utterances **and binder entries** (reads count as exposures) that meets
  all of:
  - at least 3 tokens linked to K1 jobs (provenance `referent_event_ids` ∋ a K1 job);
  - at least 2 producers;
  - not in world text (surfaces, menu, onboarding, tally template, binder stamps, F4), the lexicon, or battery
    wording;
  - C0 production below 10% (the share of C0 P descriptions containing it).
- **Y (stable control):** the same rule for K2.
- **Fallbacks:** `X_desc` and `Y_desc` are the most frequent K1 and K2 descriptive noun phrases in the binder,
  flagged descriptive. If no X exists, RQ1 X-analyses are reported as descriptive-target and W carries RQ1.
- Every branch of a seed probes the same X, W and Y. Expressions discovered later may be probed retrospectively at
  earlier checkpoints, and are labelled **exploratory**.

### 5.5 Metrics (agent i, checkpoint k; "cell" = run node × checkpoint, pooled over the members present)

| metric | definition |
|---|---|
| ACC_cur(i,k,S) | share of choices on item set S equal to GT(item, regime at k) |
| OLD / NEW / OTHER | at B checkpoints: shares equal to GT(A), equal to GT(B), or neither (three-way; HEDGE = the `slow` share, reported separately) |
| ORR | = OLD on K1c: the old-regime response rate |
| prior-adjusted | metric − the same metric for the same persona at C0 |
| KNOW(k) | ½(NEW − OLD) on K1c, memory-only, pooled over the cell; range −1..1 |
| SRI(c,k) | ½{[p_B(c) − p_B(NONCE)] − [p_A(c) − p_A(NONCE)]} from N choices pooled over present members × framings. Negative means c implies the A-fix. SRI_text is the same from coded implications (§5.10). |
| **LAG(c)** (RQ1) | [SRI(c,C6) − SRI(c,C4)] − [KNOW(C6) − KNOW(C4)], shift arm minus `keep_noshift`. Negative: the expression's implication trails what members know (expression-bound inertia). Positive: it leads. LAG(Y) ≈ 0 is the control. |
| EXT(c,k,type) | mean fit (fits = 1, not sure = .5, no = 0) by item type. **Broadening** = rise on K3 + K1a, C4 → C6. **Narrowing** = fall on K1c. **Overextension** = EXT on K0. **Anchoring:** in B, K1 and K3 share a cause, so a rise on K3 means the expression follows the cause (*cause-anchored*); staying on K1 only means it follows the symptom (*cue-anchored*). |
| breadth(c,k) | number of the 13 situations with cell-mean fit ≥ .5 |
| FORM | spontaneous c in P descriptions; in-simulation uses of c per K1 event before and after D5; "heard before" checked against the exposure trace |
| INTERNALIZATION | ACC(P) / ACC(P-sit) on shared items; the situated-minus-memory gap is also reported |
| SOCIAL SHARE | ACC(P) − ACC(P-ablated), newcomers |
| RETENTION | W1 ACC_mem at C4 / the departed W1 leavers' ACC_mem at C3 (K1c) |
| behavioural | first-attempt correctness on real K1 and K2 jobs; OLD/NEW/OTHER of first attempts after D5; **switch latency** = post-shift K1 jobs until the first window where 2 of 3 consecutive first attempts are the B-fix; jobs not delivered; defers |
| **classification** per cell and c ∈ {X, W} | SEMANTIC CHANGE: c used in B at ≥ 50% of its A rate **and** ΔSRI(C4→C6) ≥ 0.4 **and** larger than the no-shift ΔSRI, than ΔSRI(Y), and than 2 × the retest SD. REPLACEMENT: a new expression with SRI > 0 while c's use drops by more than 50% or SRI(c) < 0. INERTIA: c persists with SRI(C6) < 0. LOSS: c unused in B. |
| local specialization (RQ3; exploratory) | Jensen–Shannon divergence of p(action \| K1c) and p(action \| c) between AM and PM crews, minus within-crew; null = 1000 permutations of crew labels within seed |
| pragmatic function | the distribution of coded functions of c's in-simulation usages, before vs after D5 (§5.10) |

### 5.6 Baselines
- **C0 fresh prior:** all 13 personas with empty memory run P (all items, 2 orders on K1c) and N (W, NONCE,
  NONE). N(X, Y) is run after target selection. A **persona-free** responder ("a student who volunteers at a
  campus makerspace co-op") is asked 3 menu orders.
  - Gives the pretrained action distribution per item, the prior reading of each cue, prior production of X, and
    cue diagnosticity.
  - Run for both content packs at calibration.
- **Record-only reader** at every studied checkpoint cell: the persona-free responder given only the binder view,
  answering P-sit items and N. Measures what the text alone conveys. If newcomers beat it, talk and experience
  contributed something.
- **Departed members at C3:** the "source knowledge" for retention.
- **Mapping counterbalance:** prior drift = movement toward the pretrained favourite, averaged over M1 and M2.

### 5.7 Transmission vs independent rediscovery (`analysis/attribution.py`)
- For each agent in a shift arm, take the **first B-fix first attempt** on a K1 job, and the first checkpoint
  where at least 5 of 8 K1c memory-only answers are the B-fix.
- Label the exposures before adoption (multi-label; shares reported):
  - **RECORD**: the view shown at the decision, or a read record node, recommends the B-fix (entry coding, §5.10);
  - **TALK**: a retrieved conversation or overheard node, or a clarify conversation for this job, recommends it;
  - **OWN**: a retrieved own or witnessed episode shows the B-fix succeeding;
  - **NONE**: none of these.
- The NONE rate is compared with the C0 rate of choosing the B-fix unprompted: the rediscovery base rate.
- Expression spread uses v2 `emergence.py`, with binder reads counted as exposures (the reader is the listener of
  the entry's author).

### 5.8 Artifact lineage and record metrics (`analysis/lineage.py`)
- **Lineage graph:** entry or revision → reads (agent, tick) → record memory nodes (`record_ids`) → utterances
  (via retrieved nodes, v2 provenance) → decisions (`binder_entry_ids` + retrieved nodes). Written to
  `lineage.json`.
- Each entry's **recommended action** comes from a keyword map over the menu texts, with a Sonnet fallback when the
  map is ambiguous (§5.10).
- **Record metrics:**
  - reads per faulted job; writes per offer and per day; front revisions;
  - **front-page staleness**: hours from the first B-fault until the front page stops recommending the A-fix;
  - **stale-view exposure**: the share of post-D5 K1 decisions whose view recommends the A-fix with no B
    correction;
  - **follow rate**: first choice equals the fix recommended by the newest relevant entry;
  - **departed-author share** of front-page text;
  - binder mentions in talk, per arm (a transition side-effect).

### 5.9 Decision counterfactuals (DCF; `analysis/counterfactual.py`, observer-only)
Every logged post-D5 K1 first-attempt prompt in `keep_shift` and `wipe_shift` is re-queried on a separate client
under the `dcf:` scope, in four versions:
1. **unchanged** (a new scope, so the cache is bypassed): the determinism floor;
2. **binder removed** (the empty-binder rendering);
3. **front page → its last pre-D5 revision**;
4. **front page → the latest post-D5 revision**.

The change in choice rates is the causal effect of record content at decision time. It separates stale content
from the update channel. There are about 8 prompts × 4 versions per arm per seed.

### 5.10 Coding (Sonnet via the CLI; `analysis/coding.py`)
Codebooks (new prompts `code_implication_v1.txt`, `code_record_v1.txt`, `code_function_v1.txt`):
- **implication** (N sentence 1, P describe): cause ∈ {lens, moisture/sheets, belt, speed/power, other, don't
  know}, the implied first action, hedged or certain, and change mentioned (yes/no);
- **record entry**: recommended action (menu id or none), mentions the supplier or a change (yes/no);
- **pragmatic function** of in-simulation usages of X, W and Y, and of binder entries: REPORT, DIAGNOSE,
  INSTRUCT, WARN, QUERY, EVALUATE/JOKE, BLAME/EXCUSE and REASSURE, plus a NORM flag (licenses or obliges an action).

**Blind:** condition, day, regime and author are stripped, and items are shuffled across runs and batched 10 per
call.

**Validation gates** (coded outcomes stay secondary until they pass):
- a second Sonnet pass with a paraphrased codebook and a different order;
- Haiku as a second coder;
- a 120-item human sample coded by two collaborators (`scripts/export_validation_sample.py`), κ ≥ 0.6.

The pre-registered fallback collapses functions to {inform, direct, social}. The primary outcomes never use a
coder.

### 5.11 False-positive and validity controls
1. Fixed items rule out a change of topic.
2. Within-agent change for incumbents, and identical rosters across arms, rule out speaker mix.
3. The X-minus-NONCE contrast, and interpretation by members who never produced X, rule out copying without
   interpretation.
4. C0 and mapping counterbalance rule out pretrained association.
5. Y (K2) must show |ΔSRI| ≈ 0, and K2 accuracy must stay put.
6. `keep_noshift` gives ΔSRI and ΔKNOW ≈ 0 with the placebo cue present. Over-switching there counts as
   superstitious reinterpretation and is reported.
7. The scrambled-causality run (calibration) must give prior-adjusted accuracy ≈ 0: the learning false-positive
   floor.
8. **Test-retest:** C4 of seeds 11–12 re-probed with new retrieval seeds and menu orders. Criteria: agreement ≥ .8
   on P, and cell-level SRI SD ≤ .1.
9. **Replicate branches** (seeds 11–12, post-T salt): the floor for divergence between branches.
10. **Synthetic-memory calibration** on isolated C3 copies. Inject "X means the lens needs cleaning" into 4 copies,
    the B meaning into 4, and irrelevant belt memories into 8. Required: SRI difference ≥ 0.6, and |ΔSRI| ≤ 0.1 for
    the irrelevant injection.
11. **Predictive validity:** an agent's P-sit choice at C3 (and at C5) must match its own next real K1 first attempt
    on D4 (and on D6) ≥ 50% of the time; chance is about 0.17.
12. **Retrieval hit rate** for held-out items vs twins, per arm (§5.2).
13. **Prompt audit** over every rendered prompt in the simulation `llm_calls.jsonl` (§8.1 step 10).
14. **Manipulation checks** per mechanism, as in v2 D69: jobs, K1 faults, decisions, asks, handovers, meetings,
    reads, writes, transitions, departures, arrivals, regime applied, talk to newcomers. A mechanism that is on but
    never fired is marked `inactive`.

### 5.12 `outcomes.json` v3 block (`analysis/outcomes_v3.py`)
```
"v3": {"node", "seed", "mapping", "cohorts": {agent: founder|W1|W2|stores},
  "continuity": {"acc_mem_w1_c4", "acc_mem_all_c4", "first_attempt_acc_d4", "retention_ratio"},
  "inertia":    {"orr_mem_all_c5", "orr_mem_all_c6", "orr_jobs_d5d6", "new_mem_all_c6", "other_mem_all_c5",
                 "hedge_c5", "switch_latency"},
  "inherited":  {"orr_mem_w2_c6", "orr_w2_prior_c0"},
  "records":    {"reads_per_faulted_job", "writes_per_day", "front_revisions", "staleness_hours",
                 "stale_view_share", "follow_rate", "departed_author_share", "binder_talk_mentions"},
  "dcf": {...}, "attribution": {...}, "social_share": {...}, "internalization": {...},
  "expressions": {"targets": {"X", "W", "Y"}, "sri", "lag", "ext", "breadth", "class"},
  "pragmatics": {...}, "specialization": {...},
  "manipulation": {...}, "validity": {"predictive_validity", "retest_agreement", "kappa", "retrieval_hit_rate",
                   "prompt_audit", "llm_errors", "prefix_digest_match"},
  "cost": {"llm_calls", "calls_per_day", "wall_seconds"}}
```
`compare` and `design table` keep v2's rule: runs with different observer specs are never pooled.

---------------------------------------------------------------------------------------------------

## 6. Experiment controller

### 6.1 LLM client (`backend/llm/client.py`)
- **Fail-fast (critic fix).** A backend exception is **never** written as a response, so `LLM_ERROR` text is never
  recorded. Up to `llm.fail_fast.max_consecutive_errors: 3` consecutive failures are retried with pauses of
  `pause_seconds: 600`. After `max_pauses: 6`, the run raises `LLMUnavailable`. The manifest then gets
  `status: "paused"` with `last_complete_tick`.
- **Prefix mode** (`llm.replay_from: <parent llm_calls.jsonl>`, `llm.replay_until_tick: T`):
  - the replay dict is filtered to keys whose scope is `"seed"` or whose parsed tick is < T;
  - a miss for a scope with tick < T raises **`PrefixDivergence`**;
  - scopes with tick ≥ T are **never** served from the cache: they are always live and recorded;
  - so a twin, replicate or no-shift branch of a longer parent can never silently reuse the parent's post-T
    responses.
- **Resume after a crash or pause:** re-run the same node with `replay_from` = its own partial `llm_calls.jsonl` and
  `T = last_complete_tick + 1`.
- `replay` mode (v2) is unchanged.

### 6.2 Branches (`backend/experiment/branch.py`)
- A branch is `{parent, at_day, days, set: overlay, salt}`. Its config is the parent config plus the overlay, with
  `simulation_days = at_day − 1 + days`, `llm.replay_from = parent/llm_calls.jsonl` and
  `llm.replay_until_tick = T(at_day)`.
- The branch replays days 1…at_day−1 at CPU speed, with no LLM calls, then runs live.
- **Overlay whitelist.** An overlay may set only dated entries with day ≥ `at_day`:
  - `regimes.schedule`, `regimes.cues`;
  - `records.transitions`, `records.history_from_day`, `records.consult_from_day`;
  - `turnover.waves`, `turnover.reassign`;
  - `comm.*_from_day`;

  plus `branch.salt`, `simulation_days` and `checkpoints`. Anything else is rejected at load.
- **Trunk prompts never depend on branch-only config.** That covers display, transition mode, schedule, horizon
  and condition. `_condition` reaches only the manifest.
- **Identity checks** before a branch's first live tick; any failure aborts:
  - the per-tick trace digests (`digests.jsonl`, one sha256 per tick) for ticks < T equal the parent's;
  - the world-script records for days < at_day are identical;
  - git sha and prompt hashes equal;
  - the parent has zero `LLM_ERROR` records and `status: finished`.
- **Salt** (replicate branches): tick-keyed streams (`talk`, `invite`, conversation rngs) mix in the salt only for
  tick ≥ T. Agent substreams are **re-derived at T** as `seed_rng(seed, agent, name, "salt", salt)`. Pre-T draws
  are untouched, so the prefix still replays (critic fix: changing `seed` would change plans before T). The world
  script is not salted, so jobs and uniforms stay common.
- **No state-restore branching.** Checkpoints are probe inputs only (critic fix: restoring would need every
  override, pending beat and RNG state).
- Manifest: `branch {parent_run, at_day, at_tick, overlay_sha, salt, parent_prefix_digest, prefix_digest,
  git_sha, prompt_hashes}`.

### 6.3 Design trees (`backend/experiment/design.py`, extended; `kind: tree`)
```yaml
# configs/designs/records_boundaries_v3.yaml
name: records_boundaries_v3
kind: tree
base: configs/v3_base.yaml
seeds: [11, 12, 13, 14, 15, 16, 17, 18]          # seed = world_seed; M1 if odd, M2 if even
observer: {backend: claude_cli, model: sonnet}
common: {llm: {backend: claude_cli, model: haiku, max_workers: 6}}
freeze: {git_sha: required, prompt_hashes: required}
tree:
  trunk:        {days: 4}                          # D1-D4; keep@D4 and wave 1 come from v3_base
  wipe4:        {parent: trunk, at_day: 4, days: 1,
                 set: {records: {transitions: [{day: 4, mode: wipe}]}}}
  keep_shift:   {parent: trunk, at_day: 5, days: 2,
                 set: {regimes: {schedule: [{day: 1, regime: A}, {day: 5, regime: B}]},
                       records: {transitions: [{day: 4, mode: keep}, {day: 5, mode: keep}]}}}
  wipe_shift:   {parent: trunk, at_day: 5, days: 2,
                 set: {regimes: {schedule: [{day: 1, regime: A}, {day: 5, regime: B}]},
                       records: {transitions: [{day: 4, mode: keep}, {day: 5, mode: wipe}]}}}
  keep_noshift: {parent: trunk, at_day: 5, days: 2,
                 set: {records: {transitions: [{day: 4, mode: keep}, {day: 5, mode: keep}]}}}
  keep_shift_rep: {parent: trunk, at_day: 5, days: 2, seeds: [11, 12],
                 set: {regimes: {schedule: [{day: 1, regime: A}, {day: 5, regime: B}]},
                       records: {transitions: [{day: 4, mode: keep}, {day: 5, mode: keep}]},
                       branch: {salt: 1}}}
targets: {select_at: {node: trunk, checkpoint: C3}}          # run automatically after each trunk
probes:
  C3: {nodes: [trunk], battery: [P, N_W]}
  C4: {nodes: [trunk], battery: full, ablated: newcomers}
  C4_wipe: {nodes: [wipe4], checkpoint: C4, battery: continuity}
  C5: {nodes: [keep_shift, wipe_shift, keep_noshift, keep_shift_rep], battery: light}
  C6: {nodes: [keep_shift, wipe_shift, keep_noshift, keep_shift_rep], battery: full, ablated: newcomers}
  retest: {seeds: [11, 12], checkpoint: C4, node: trunk}
prereg: configs/designs/records_boundaries_v3.prereg.yaml   # hypotheses, estimands, gates (section 7)
outcomes: [v3.continuity.acc_mem_w1_c4, v3.inertia.orr_mem_all_c5, v3.inertia.orr_jobs_d5d6,
           v3.inherited.orr_mem_w2_c6, v3.expressions.lag, v3.records.staleness_hours]
```
- Run dirs are `runs/<design>/<node>/s<seed>/`.
- Nodes run in dependency order: a trunk, its target selection, then its branches.
- Status adds `waiting_parent` and `paused`. Resume is by prefix replay.
- `table` computes seed-paired contrasts from `prereg.contrasts`.
- v2 `factors`/`controls` designs keep working (`kind: factorial`, the default).

### 6.4 CLI
- `design run <file> [--only node] [--seed s] [--stage B]` runs missing nodes in dependency order, one simulation
  at a time.
- `design status | table <file>`.
- `design probe <file> [--checkpoint C4]` runs the observer battery after segments finish.
- `branch <parent_run> --day N --set k=v ...` makes a single ad-hoc branch.
- `probe <run> --checkpoint C4 --battery lb1 --modes memory,situated,ablated`.
- `calibrate prior|synthetic|retest <...>`.

### 6.5 Manifest additions
`workshop` (content pack, script sha), `records`, `roster` (the realised rotation), `branch` (§6.2), `checkpoints`
(ids and shas), `persona_universe`, `status: paused`, `last_complete_tick`, `llm.errors`.

The API's demo mode strips `hidden`, `regime`, `mapping`, `world_state`, `job_truth`, `regime_active`,
`cause_by_regime` and the rotation.

### 6.6 Operations
- **One simulation stream at a time.** Concurrent CLI load plus probes risks rate and usage limits. Errors now
  pause the run instead of being recorded, but serial running is still the rule.
- Probes and coding run after each stage's simulations, or concurrently at ≤ 2 workers only if G1 shows zero
  errors under that load.
- Run under `caffeinate -dimsu` on mains power, since the laptop sleeps if the lid is closed. Resume by prefix
  replay.

---------------------------------------------------------------------------------------------------

## 7. First study: `records_boundaries_v3` ("Records at the boundaries")

### 7.1 Design tree (per seed; identical world script and uniforms in every node)
```
          D1     D2     D3   │  D4                 │  D5            D6
trunk  ── founding, binder kept (display current) ───│ W1 in; keep@D4 ──┐
          meeting D2          │  cue 16:45 (stores)  │                  │
wipe4                         └── D4': W1 in; WIPE@D4 (continuity arm)  │
keep_shift                                            ├── keep@D5, regime B ── W2 in (D6)
wipe_shift                                            ├── WIPE@D5, regime B ── W2 in (D6)
keep_noshift                                          ├── keep@D5, regime A ── W2 in (D6)   (placebo cue)
keep_shift_rep (seeds 11–12, salt)                    └── keep@D5, regime B ── W2 in (D6)
checkpoints: C3 (trunk) · C4 (trunk, wipe4) · C5 and C6 (each D5 branch); meeting D5 in every D5 branch
```

### 7.2 Factors, levels, controls
- **Records at turnover** (between branches, within seed): trunk (keep@D4) vs `wipe4` (wipe@D4). This is the
  continuity contrast, from the identical C3 state.
- **Records at the change** (within seed): `keep_shift` vs `wipe_shift`, wiped at D5 exactly when the cause swaps.
  This is the inertia contrast, from the identical C4 state.
- **Change** (within seed): `keep_shift` vs `keep_noshift`. It is the positive control that the shift bites, the
  placebo (the supplier cue without a causal change), and the null for RQ1's LAG.
- **Within run:** phase (C4 vs C5/C6); cohort (founder, W1, W2, stores); class (K1 changing, K2 stable, K3 new).
- **Between seeds:** mapping, M1 vs M2 (4 + 4).
- **Held fixed:**
  - 8 active agents; Haiku 4.5 agents and Haiku probes; Sonnet coder;
  - `laser_alpha` (or beta after G1); 8 jobs a day; the success table of §1.3; F4 on;
  - binder display `current`, consult `always`, `view_log_n` 5, `encode_on_read: new_only`;
  - comm budgets of §4.5; v2 mechanisms as in §4.6; one frozen git sha and prompt set.
- **Controls:** `keep_shift_rep` (noise floor); within-run K2/Y; C0; record-only reader; NONCE/NONE; test-retest;
  synthetic calibration. Calibration runs: `cal_scrambled` (scrambled causality) and `cal_shiftmini` (shift
  sensitivity), see §7.6.

### 7.3 Seeds, days, segments

| node | parent | T | live days | sim-days × seeds | total |
|---|---|---|---|---|---|
| trunk | — | — | D1–D4 | 4 × 8 | 32 |
| wipe4 | trunk | 180 | D4 | 1 × 8 | 8 |
| keep_shift / wipe_shift / keep_noshift | trunk | 240 | D5–D6 | 2 × 8 × 3 | 48 |
| keep_shift_rep | trunk | 240 | D5–D6 | 2 × 2 | 4 |
| **core** | | | | **42 segments** | **92** |
| cal_scrambled (seed 10, no turnover), cal_shiftmini (seed 9, shift at D2, no turnover) | — | — | — | 3 + 2 | 5 |

- Seeds are 11–18: odd = M1 (11, 13, 15, 17), even = M2 (12, 14, 16, 18).
- A per-seed path is 6 simulated days. The trunk prefix is replayed, never re-bought.
- **Lean fallback** (declared before stage C, for compute reasons only): seeds 11–16, which is 32 segments and 70
  core sim-days.

### 7.4 Outcomes
Everything is coder-free for the primaries.
- **PO1, continuity (H1).**
  - A1_s(node) = the mean over wave-1 newcomers (all 3, pre-registered; crew-only W1 as a sensitivity analysis) of
    **memory-only** ACC_cur on K1c at C4 (16 responses each), scored against GT(A).
  - **Δ1_s = A1_s(trunk) − A1_s(wipe4).**
  - Convergent measure (reported, not separately tested): all-operator first-attempt accuracy on D4 faulted jobs,
    matched by job id across the two nodes.
- **PO2, inertia (H2).**
  - O2_s(node) = the mean over all present members of **memory-only** ORR on K1c at C5 (16 responses each).
  - **Δ2_s = O2_s(keep_shift) − O2_s(wipe_shift).**
  - Heads are identical at T = 240, so there is no level confound (critic fix).
- **Secondary family** (Benjamini–Hochberg, q < .10; two-sided):
  - S1 behavioural inertia: OLD on matched K1 first attempts D5–D6, `keep_shift` vs `wipe_shift`;
  - S2 inherited inertia: W2 newcomers' memory-only ORR at C6, `keep_shift` − `wipe_shift`, prior-adjusted with
    the same personas' C0 values;
  - S3 adaptation: NEW at C6 (all members) and switch latency;
  - S4 internalization gap (P-sit − P) by arm;
  - S5 DCF effect of binder removal on post-D5 choices;
  - S6 attribution shares for the first B-fix (RECORD/TALK/OWN/NONE vs the C0 base rate);
  - S7 RQ1: LAG(W), and LAG(X) where X exists, against LAG(Y);
  - S8 broadening: EXT on K3 for W and X, C4 → C6, shift − no-shift, including the anchoring classification;
  - S9 pragmatic-function shift of W and X usages (only if κ ≥ .6);
  - S10 retention ratio.
- **Positive control** (gate G3; not in the family): K1c OLD at C5, `keep_noshift` − `keep_shift`.
- **Exploratory:** the three-way outcome (old/new/other) and HEDGE; cue diagnosticity × mapping; placebo
  over-switching in `keep_noshift`; AM–PM divergence; stores vs crews; SRI trajectories by cohort; front-page
  staleness and stale-view exposure.

### 7.5 Pre-registered analyses and power
- **Unit = seed.** The tests are exact sign-flip permutations of the seed-level paired differences, with 2⁸ = 256
  patterns: minimum one-sided p = .0039, minimum two-sided p = .0078.
- **Fixed-sequence gatekeeping** (critic fix; replaces Holm):
  - H1 (continuity: E[Δ1] > 0), one-sided, α = .05;
  - **only if H1 is rejected**, H2 (E[Δ2] ≠ 0: inertia if > 0, acceleration if < 0), two-sided, α = .05.
  - An untested H2 is reported as an estimate.
- Reporting: mean Δ with seed-bootstrap 95% CIs. The replicate-branch |Δ| (seeds 11–12) is the divergence floor;
  effects below it are not claimed.
- **Supporting model:** `logit(correct or old) ~ node × phase + mapping + cohort + (1|seed) + (1|seed:node:agent) +
  (1|item)`. Matched real jobs are analysed with conditional logistic regression stratified by job id.
- Mapping is constant within a seed, so paired differences cancel it. It is reported as a stratifier.
- **Power** (simulated, exact sign-flip, normal paired differences, 3,000 replications):

  | test | 80% power at |
  |---|---|
  | 8 seeds, one-sided (H1) | dz ≈ 1.0 |
  | 8 seeds, two-sided (H2) | dz ≈ 1.2 (power .65 at dz 1.0) |
  | 6 seeds, one-sided (lean) | dz ≈ 1.25 |
  | 6 seeds, two-sided (lean) | reaches only .66 at dz 1.5 |

  So Δ ≈ .15–.18 is detectable when the SD of the paired differences is about .15. The study doubles as
  effect-size estimation.
- **Precision rule** (decided at G3 without comparing arms): estimate the between-seed SD from the `keep_shift`
  arm alone and from the replicate divergence. If the projected 95% CI half-width for Δ2 at 8 seeds exceeds .12
  and budget remains, add seeds 19–20.
- **Exclusion:** a node failing a pre-registered manipulation check (§7.6 G2 list) is excluded and its seed is
  replaced by the next seed (19, then 20). Nothing is excluded on outcomes.

### 7.6 Decision gates

| gate | when | pass criteria | if it fails |
|---|---|---|---|
| **G0** engineering | mock backend, whole tree for 1 seed | rendered-prompt audit clean; prefix mode with `PrefixDivergence` and per-tick digest identity; world-script prefix identity across nodes; zero observer scopes in simulation caches; probe checksum invariance and meta parity; zero trace records from departed agents; fail-fast works; gt, binder, transition and roster unit tests green | fix; no live run before G0 passes |
| **G1** calibration | about 8 h: C0 on both packs; `cal_shiftmini`; `cal_scrambled`; synthetic; one live day for cost | prior balance \|P0(A-fix) − P0(B-fix)\| ≤ .2 on K1c; cue diagnosticity: the supplier cue shifts P0 toward either fix by ≤ .3; synthetic ΔSRI ≥ .6 (irrelevant ≤ .1); retest agreement ≥ .8; held-out retrieval hit rate ≥ .7 of the twins' rate; `cal_shiftmini` has ≥ 1 switch to the B-fix within 6 K1 jobs; `cal_scrambled` prior-adjusted accuracy within ±.05; a live day ≤ 1.9k calls; 0 errors | prior imbalance → content pack `beta`; cue too diagnostic → neutral carrier "a new batch of sheets arrived"; cost too high → levers below; no switching → success contrast .90/.05 |
| **G2** learnability and engagement | seed-11 trunk | D3 K1 first-attempt accuracy ≥ max(C0 + .25, .5); C3 memory-only K1c ACC ≥ C0 + .25; binder read on ≥ 90% of faulted jobs; writes per offer .2–.9 and ≥ 2 writes a day; follow rate ≤ .9; K1 faults ≥ 2.5/day; ≤ 1.9k calls/day; 0 `LLM_ERROR` | pre-registered adjustments in order: (1) success .90/.05; (2) `p_fault` .85; (3) switch the content pack. Re-run trunk 11 (the failed one becomes a pilot). Follow rate > .9 → consult `on_choice` for the whole study. |
| **G3** interim (no efficacy peeking) | after seeds 11–14, all nodes | manipulation checks pass; **positive control**: `keep_shift` K1c OLD at C5 is ≥ .15 below `keep_noshift`, pooled; κ ≥ .6; predictive validity ≥ .5; X found in ≥ 2 of 4 trunks | shift not biting → stop and revise the world (the regime is not learned, so H2 cannot be tested); κ fails → collapsed codebook; no X → RQ1 on W plus descriptive X; precision rule (§7.5) |
| **G4** after the core | all 8 seeds | — | H1 rejected and \|Δ2\| ≥ .10 → Tier 2: revision history + RQ4 revert. \|Δ2\| < .10 → the comms-lean branch (do records matter when talk is scarce?) and consult on_choice. H1 not rejected → the internalization and never-had-records arms, and a report on whether records were read but not internalized |

**Cost levers** are decided at G1, before any core run. They are applied in this order until a day costs ≤ 1.9k
calls, to every cell alike, and each is logged as a D-entry:
1. reflection threshold 90;
2. catch-ups off;
3. K0 episodes encoded for the operator only.

If the cost is still over 1.9k, `keep_noshift` is dropped for seeds 15–18, leaving 84 sim-days. G3's positive
control and LAG then use seeds 11–14.

### 7.7 Compute budget (Haiku via the claude CLI; one stream)
- **Per simulated day, planning figure: 1.7k calls, about 52–55 min.** The measured v2 rate is 1.3k calls in about
  40 minutes (≈ 0.54 calls/s). The breakdown:
  - campus base without E1–E4: about 0.9–1.1k;
  - job layer: about 6 faulted jobs × 15 plus 2 K0 jobs × 8, about 105;
  - records: about 25;
  - world facts: about 20;
  - handover, clarifications and meetings: about 30;
  - extra talk from co-located crews: about 0.2–0.4k.
- **Simulation:** 97 sim-days (92 core + 5 calibration) × 1.7k ≈ 165k Haiku calls, ≈ 85–89 h.
  - At the measured 1.3k calls a day: ≈ 126k calls, ≈ 65 h.
  - Replayed prefixes cost CPU minutes, not calls.
- **Probes:**

  | component | calls |
  |---|---|
  | per seed: C3 280 + C4 448 + wipe4 192 + C5 528 + C6 1,344 + ablated 210 | ≈ 3.0k |
  | 8 seeds | ≈ 24k |
  | replicates | 1.2k |
  | C0 (with retrospective X/Y) | 2k |
  | record-only reader | 1.2k |
  | retest | 0.9k |
  | synthetic | 0.4k |
  | DCF | 0.5k |
  | **total Haiku** | **≈ 30k**, about 9–16 h at 0.5–1 call/s |

  Sonnet coding: about 2k calls, about 1.5 h.
- **Total: about 80–105 h (3.3–4.4 days) of continuous compute.**
  - Stage A (G0/G1): about 8 h.
  - Stage B (seeds 11–14 + replicates; G2, G3): about 48 sim-days, about 50 h.
  - Stage C (seeds 15–18): about 44 sim-days, about 45 h.
  - Lean fallback: about 60–80 h.
- In run units: 42 core segments + 2 calibration runs, about 32 three-day run-equivalents, inside the agreed
  budget of 20–40 runs.

### 7.8 Extensions (separate budgets; gated by G4)
- **Tier 2, revision history (RQ2 revision).** `keep_shift_hist`: from trunk at D5,
  `records: {display: history, history_from_day: 5}`, compared with `keep_shift`. Author and date are held
  constant, so only the visibility of superseded versions differs (critic fix). Seeds 11–18, 16 sim-days.
- **Tier 2, compliance robustness.** `keep_shift_onchoice` / `wipe_shift_onchoice` with `records.consult:
  on_choice` from D5. This separates reading on demand from being shown the record. 32 sim-days, or 16 with seeds
  11–14.
- **Tier 2, RQ4 revert.**
  - `rq4_revert`: from `keep_shift` at D7 (T = 360), regime A with the `usual_supplier` cue, D7–D8.
  - `rq4_control`: from `keep_noshift` at D7, the identical cue and regime A.
  - At C8 both are in identical present conditions, so a remaining difference in OLD/NEW (B-fix responding under
    A), SRI or EXT is history dependence.
  - Seeds 11–14: 16 sim-days.
  - Adding a wave 3 would need 2 more reserve personas.
- **Tier 3, RQ3 local specialization.** From trunk at D5, `regimes.split: {am_crew: plywood}`: on plywood K1 is
  always LENS, while PM's acrylic follows the regime. Plus `turnover.reassign` of one member at D6. Measures: AM–PM
  JSD for X/W, and coordination on cross-crew handovers. 2 branches × seeds 11–14, 16 sim-days.
- **Tier 3:** a never-had-records trunk (`records.enabled: false` from D1; 6 days × 8); authority
  (`manager_signed`); `encode_on_read: never`; `retrieval: relevant`.
- **Study 2, communication:** handover 0/6/12 utterances × clarify on/off × keep/wipe at D5, branched from the same
  trunks.
- **Calibration suite:** the v2 bottleneck factorial and the E1–E4 structure controls, on this codebase with
  `workshop.enabled: false` (§9.2).

---------------------------------------------------------------------------------------------------

## 8. Build plan

### 8.1 Ordered steps
Build against this contract only after the v2 integration is merged and frozen at a tag.

| # | step | files (new unless marked) | APIs | config keys | trace records | tests | size |
|---|---|---|---|---|---|---|---|
| 0 | Freeze v2; land the v2 pieces v3 reuses (world script + CRN streams, `Agent.stream`, group conversation, reminding/verbatim/source weights, provenance, emergence, funnel, outcomes, design runner, manifest `code_version`) | — | — | — | — | v2 suite green on mock | S |
| 1 | **LLM client**: fail-fast, prefix mode, observer-client helper; per-tick digests | `llm/client.py` (edit), `tracing/logger.py` (edit), `llm/mock.py` (JSON handlers for the new prompts) | `LLMClient(..., replay_until_tick)`, `PrefixDivergence`, `LLMUnavailable`, `observer_client(kind, path)` | `llm.replay_until_tick`, `llm.fail_fast.*` | `digests.jsonl`; manifest `status: paused` | `test_v3_client.py`: a pre-T miss raises; a post-T hit is never served; errors are never recorded; resume | M |
| 2 | **Workshop world**: machine, classes, surfaces (hygiene), menu, OutcomeModel, `gt()`, regimes, cues, jobs, tally; Stockroom arena; arena visibility | `simulation/workshop.py`, `simulation/regimes.py`, `simulation/content/laser_alpha.py`, `laser_beta.py`; `world_script.py`, `world.py`, `perception.py` (edits) | `workshop.generate()`, `OutcomeModel.p/gt/resolve`, `regimes.active(day)` | `workshop.*`, `regimes.*` | `job_start`, `job_attempt`, `job_end`, `tally`, `cue_event`, `regime_active`† , `job_truth`† | `test_v3_world.py`: hygiene; day-local prefix identity for any horizon; regime-independence of the script; K1/K2 slots equal across regimes; gt | L |
| 3 | **Binder**: entities, rendering, receipts, transitions, reads as observations, RecordWrite | `simulation/records.py`, `prompts/record_write_v1.txt`, `agents/work.py` (write part); `memory/encoder.py` (VERB/HEARD for `record`), `memory/store.py` (`record_ids`) (edits) | `Binder.view(mode, n)`, `.append/.rewrite/.transition/.new_for(agent)` | `records.*` | `record_read`, `record_write`, `record_transition` | `test_v3_records.py`: display parity across arms; new-only encoding; wipe archives; sorted writes | M |
| 4 | **Roster and turnover**: coop13 population, reserves, roles, rotation, farewell, arrival ties, persona universe | `simulation/roster.py`, `configs/population/coop13.yaml`; `agents/profile.py`, `simulation/scheduler.py` (plan hook), `memory/encoder.py` + `agents/viewpoint.py` (names from the universe), `simulation/lexicon.py` (edits) | `roster.apply_day_start()`, `active_agents()` | `roster.*`, `turnover.*` | `roster_change`, `onboarding`, `farewell` | `test_v3_roster.py`: the Latin square; zero records from departed agents; symmetric ties; tick-scoped seeds | M |
| 5 | **Work cognition and engine**: the CoopWorld facade (day_start, world phase, 4b/4c, day_end), job-episode encoding, clarify through TALK, handover, meeting, API strip | `simulation/coop_world.py`, `agents/work.py`, `prompts/job_decision_v1.txt`; `engine.py` (≈120-line diff), `agents/conversation.py` (topics clarify/handover), `group_conversation.py` (topic meeting), `api/server.py` (edits) | `CoopWorld.day_start/world/after_cognition/apply/day_end` | `comm.*`, `workshop.encode*` | `job_decision`, `clarification`, `handover`, `meeting`, v2 `utterance`/`exposure` | `test_v3_engine.py` (mock): 1-day run; tick order; replay trace-sha identity | L |
| 6 | **Checkpoints** | `experiment/checkpoint.py`; engine day-end call | `write_checkpoint(sim, day)` | `checkpoints.*` | `checkpoint` | deterministic bytes; read-only; identical after prefix replay | M |
| 7 | **Branches and design trees** | `experiment/branch.py`; `experiment/design.py`, `cli.py` (edits); `configs/v3_base.yaml`, `configs/designs/records_boundaries_v3.yaml`, `.prereg.yaml`, `v3_calibration.yaml`, `smoke_v3_mock.yaml` | `make_branch()`, `check_prefix()`, `expand_tree()` | `branch.*`, `design.tree` | manifest `branch` | `test_v3_branch.py`: overlay whitelist; prefix digest identity; the salt leaves the prefix unchanged; the whole mock tree runs | L |
| 8 | **Battery and probe runner** | `analysis/battery/{items_lb1_*.yaml, gt.py, runner.py, targets.py, nonce.py, fresh.py, synthetic.py, retest.py, record_only.py}`, `prompts/probe_act_v1.txt`, `probe_apply_v1.txt`, `probe_note_v1.txt`; `analysis/probes.py` (edit: meta, clock, roster) | `run_battery(run, ckpt, modes)`, `select_targets(trunk, "C3")` | `probe.*` | `probes/C*/responses.jsonl`, `llm_calls.jsonl`, `targets.json` | `test_v3_isolation.py`: checksums unchanged; import guard; no observer scope in simulation caches; source-weight parity | L |
| 9 | **Observer metrics** | `analysis/{meaning.py, jobs.py, attribution.py, counterfactual.py, lineage.py, coding.py, outcomes_v3.py}`, `prompts/code_{implication,record,function}_v1.txt`, `scripts/export_validation_sample.py`; `provenance.py`, `emergence.py` (edits: jobs as events, reads as exposures, extended world-text penalty) | `analyze_v3(run)` | `analysis.coder.*` | `outcomes.json` v3, `meaning.json`, `lineage.json`, `dcf.jsonl`, `coding.jsonl` | `test_v3_observer.py`: metric definitions on synthetic traces | L |
| 10 | **Invariants and audit** | `tests/test_v3_invariants.py`, `tests/test_v3_prompt_audit.py` | — | — | — | see below | M |
| 11 | **Docs and UI**: binder panel, job timeline; D74+ entries; README | `frontend/app.js`, `docs/DECISIONS.md`, `README.md` (edits) | — | — | — | — | S |

† hidden; stripped by the API.

**Step 10 audit rules:**
- **Templates.** No new template contains invent, coin, name (as a verb), label, term, slang, meme, nickname,
  shorthand, rule, tip, procedure or jot.
- **Rendered prompts.** Every prompt in every simulation `llm_calls.jsonl` is free of:
  - the case-sensitive ids `K0`–`K3`, `LENS`, `DAMP`, `BELT`, `AIR`, `WARP`, `M1`, `M2`;
  - job ids (`j\d\d\.\d`);
  - node and arm names (`trunk`, `wipe4`, `keep_shift`, `noshift`, `placebo`), `regime`, `mapping`, content-pack
    ids, `E1`–`E4`;
  - probabilities or uniforms.
- **Structure.** No agent field holds knowledge or procedures. The binder and receipts live only in world state.
  Simulation code never imports `backend.analysis`.

### 8.2 Config keys
All keys are **off by default** in `configs/default.yaml`. The integrator writes them.
```yaml
workshop:
  enabled: false
  content: laser_alpha              # laser_alpha | laser_beta
  jobs_ticks: [6, 10, 14, 18, 24, 28, 32, 36]
  p_fault: 0.75
  class_weights: {K1: 0.70, K2: 0.30}
  k3_from_k0: 0.6
  success: {match: 0.85, other_fix: 0.10, rerun: 0.10, slow: 0.35, slow_belt: 0.20}
  max_attempts: 2
  panel_code: {enabled: false, text: "F4", prob: 0.5}
  oddity_prob: 0.3
  symptom_visibility: arena
  encode: per_job                   # per_job | per_beat
  encode_reason: true
  causal: real                      # real | scrambled
  tally_time: "17:30"
regimes: {mapping: auto, schedule: [{day: 1, regime: A}], cues: [], split: null}
records:
  enabled: false
  display: current                  # none | current | history
  history_from_day: null
  consult: always                   # always | on_choice | never
  consult_from_day: null
  view_log_n: 5
  front_max_chars: 600
  log_max_chars: 200
  revisable: true
  retrieval: recent                 # recent | relevant
  encode_on_read: new_only          # new_only | never
  write_after: [faulted_job, tally, farewell]
  authority: members                # members | manager_signed
  replies: false
  transitions: []                   # [{day, mode: keep|wipe}]
roster: {enabled: false, shifts: {am: "08:45-13:15", pm: "13:00-17:45", stores: ["08:45-12:00", "13:00-17:45"]}}
turnover: {enabled: false, waves: [], rotate: by_world_seed, crew_familiarity: 0.35, member_familiarity: 0.2,
           farewell: true, reassign: []}
comm:
  clarify: {enabled: false, max_per_job: 1, max_utterances: 4}
  handover: {enabled: false, time: "13:15", max_utterances: 6}
  meeting: {enabled: false, days: [2, 5], time: "18:30", max_utterances: 10, max_participants: 8}
checkpoints: {enabled: false, days: all}
branch: {parent: null, at_day: null, salt: null}
llm: {replay_until_tick: null, fail_fast: {max_consecutive_errors: 3, pause_seconds: 600, max_pauses: 6}}
memory: {verbatim: {on_sources: [conversation, overheard]}}      # v3_base adds record
probe: {backend: claude_cli, model: haiku, workers: 8, k: 6, tau: 0.7, framings: [note, text, overheard]}
analysis: {coder: {backend: claude_cli, model: sonnet, second: haiku, blind: true, human_sample: 120}}
```

`configs/v3_base.yaml` (first study) overlays:
- population `coop13.yaml`, `latent_events.event_rate: 0`;
- `workshop.enabled`, `panel_code.enabled`;
- `regimes.cues` (D4 new supplier);
- `records.enabled` with `transitions: [{day: 4, mode: keep}]`;
- `roster.enabled`;
- `turnover.enabled` with the two waves;
- all three `comm.*` enabled;
- `checkpoints.enabled`;
- `reminding.enabled` with `on_sources` + `record`;
- `memory.verbatim.enabled` with `on_sources` + `record`;
- `retrieval.source_weights: {seed: 0.5, ambient: 0.5}`;
- `conversation.catchup.enabled`.

### 8.3 Ownership (parallel build, as in v2 §6)

| part | owns | must not edit |
|---|---|---|
| W3 (world) | `workshop.py`, `regimes.py`, `content/*`, `world_script.py`, `world.py`, `perception.py` (arena visibility), `tests/test_v3_world.py` | engine.py, default.yaml |
| R (records) | `records.py`, `record_write_v1.txt`, encoder/store record hooks, `tests/test_v3_records.py` | engine.py |
| T (turnover) | `roster.py`, `coop13.yaml`, `profile.py`, `scheduler.py` hook, name/lexicon universe, `tests/test_v3_roster.py` | engine.py |
| A (agent work) | `agents/work.py`, `job_decision_v1.txt`, conversation topics | engine.py |
| X3 (controller) | `llm/client.py`, `tracing/logger.py`, `experiment/*`, `cli.py`, `configs/v3_base.yaml`, `configs/designs/*` | engine.py, default.yaml |
| O3 (observer) | `backend/analysis/**`, probe and coding prompts, validation script | everything outside analysis/ |
| I (integrator) | `engine.py`, `coop_world.py`, `api/server.py`, `default.yaml`, `DECISIONS.md` (D74+), `README.md`, fixes | — |

### 8.4 Deferred (not in the first study's build)
- Replies and repair through the record; the `relevant` retrieval index; authority tags.
- A second machine; resource budgets (sheet stock, equipment time); projects with deadlines beyond the tally.
- Asynchronous messages; cross-crew reassignment (RQ3 only).
- LLM-generated job wording; more than 8 active agents.
- Reverts beyond RQ4 and gradual drift.
- The human kappa sample, DCF and synthetic calibration are built in steps 8–9 but *run* only after G2.
- The frontend binder panel may lag the study.

---------------------------------------------------------------------------------------------------

## 9. Risks, mitigations and v2 demotions

### 9.1 Risks and mitigations

| risk | mitigation |
|---|---|
| **Pretrained priors swamp learning** ("incomplete cut → lens") | M1/M2 counterbalance; C0 on every item; G1 prior balance, with content pack beta as the alternative; prior-adjusted metrics; within-seed contrasts cancel mapping |
| **The mapping is not learned in 3 days** (about 12 K1 faults) | G2 learnability gate with pre-registered adjustments (success .90/.05, `p_fault` .85, pack switch); reminding on; episode encoding; binder |
| **Agents are hyper-adaptive** (they switch after one failure, so there is no inertia) | H2 is two-sided; a null H2 is informative; the no-shift branch exposes over-switching; the three-way outcome separates hedging from switching |
| **Records reduce to "Haiku follows the text above the menu"** (compliance) | primaries are memory-only; follow rate, DCF, the situated-minus-memory gap and the record-only reader are reported; the Tier-2 `on_choice` arm |
| **Expression-level meaning is confounded with knowledge of the referent** | LAG subtracts ΔKNOW; the NONCE contrast; Y and no-shift nulls; the guaranteed world-anchored W; X-analyses conditional on X existing |
| **No coined X forms** (as in v1 and v2) | W carries RQ1; the X_desc fallback; G3 decides before stage C |
| **Lexical retrieval misses job memories for held-out items** (D2) | world-wording twins; machine-level focal; hit-rate logging by arm; G1 hit-rate gate |
| **Probe answers do not predict behaviour** | predictive-validity gate; matched real-job outcomes as convergent and secondary measures |
| **Few societies**; only large effects detectable | CRN and history-matched pairing; fixed-sequence confirmatory tests; replicate floor; precision rule; effect-size framing |
| **Replay fragility** (a code or prompt change, nondeterministic ordering) | frozen sha and prompt hashes; per-tick digests; `PrefixDivergence`; sorted application of every parallel result (D6/D7); no state-restore |
| **CLI errors and usage caps over about 100 h** | fail-fast pause and resume; never record errors; one stream; a zero-error gate before branching |
| **Cost overrun** (job beats × perceivers × 4 calls) | per-job episode encoding; new-only record encoding; G1 cost measurement; pre-declared levers; a pre-declared rule to drop no-shift arms |
| **Transition salience differs by arm** | one matched-salience fact per transition day in every arm; binder mentions in talk counted per arm |
| **The cue leaks the answer** ("different supplier" → damp) | the same neutral text for both mappings; C0 diagnosticity gate and covariate; mapping as stratifier; placebo in no-shift |
| **Write prompt pressures compression into labels** | neutral wording, limits in characters only, audited; identical across arms; world surfaces varied so compact forms must come from agents |
| **Newcomer or persona confound** | Latin-square rotation of personas and departures; identical personas in every arm of a seed; cohort as a moderator; C0 persona priors |
| **Newcomers get less talk** (low familiarity) | symmetric onboarding ties ≥ 0.3 with crewmates; talk-to-newcomer rate as a manipulation check |
| **Hidden information leaks** (job ids, class ids in stamps or menus) | rendered-prompt audit; rendering spec (§2.2); API strip |
| **Coder bias** | primaries coder-free; blind, two-pass coding with a second coder and a human κ gate |
| **Only 2 days of regime B** (practice changes, meaning lags) | C5/C6 trajectories plus LAG; RQ4 extends branches to D7–D8 if G3 shows switching still in progress |
| **v2 is still in flux** | v3 lives in new modules behind the CoopWorld facade (engine diff ≈ 120 lines); build starts after the v2 tag |

### 9.2 What v2 parts are demoted
- **NEED × LINK × WORDING bottleneck factorial** (`bottleneck_factorial.yaml`): now a calibration and secondary
  study, run on the v3 codebase with `workshop.enabled: false`. It does not count toward the first-study budget.
  In v3 the mechanisms are frozen at the §4.6 baseline instead of being factors.
- **Hidden families E1–E4**, the 44 skins, holdout skins and the real/scrambled/none structure controls:
  calibration only. They are the known-structure test bed for the observer's emergence and grounding. v3 sets
  `event_rate: 0`, and `workshop.causal: scrambled` is v3's own structure null.
- **`grounding.py`** (family-label permutation within circle × day): calibration only. v3 grounding is the task's
  ground truth per regime (`gt.py`).
- **Final-memory probes** (`probe_meaning_v1`, and the 4-way `probe_match_v1` over holdout families): replaced by
  LB1 on isolated checkpoint copies, answered by the agents' model.
- **`latent_events.assignment`** and **referent pools**: unused. v2 circles are reused as crews; the laser, the
  binder and the sheet stock are the persistent referents.
- **Production priming**: stays off (the D36 lesson).
- **The group conversation mechanism as a NEED factor**: replaced by the scheduled members' meeting and handovers.
  Catch-ups stay on as a fixed channel.
- **The LLM convention classifier and the legacy `evaluation.py`**: annotation only, never outcomes.
- **`controls.planted_phrase`**: calibration only. Its role as an SRI positive control is taken by synthetic-memory
  calibration and the world-anchored W.
- **Emotion, social reward, prestige and conformity modules**: off, reserved for later studies of influence and
  authority.
- **`topology.mode: generated`**: not used; crews are fixed in `coop13.yaml`.

---------------------------------------------------------------------------------------------------

## Appendix A. Review findings and where they are resolved

| finding (review) | resolution |
|---|---|
| Two-sided H2/H3 can never pass Holm with 6 seeds | fixed-sequence H1→H2 with 8 seeds (min two-sided p = .0078); H3 moved to the secondary family (§7.5) |
| Power claims ignored the multiplicity correction | re-simulated for the actual tests (§7.5) |
| The inertia contrast is confounded with pre-change knowledge | wipe **at the change point** from identical C4 heads (`keep_shift` vs `wipe_shift`) (§7.2) |
| A reset at D4 is refilled with A-notes by D5 | inertia is manipulated at D5, continuity at D4 (§7.1) |
| Situated battery as primary (binder in the probe prompt; effective n ≈ 1) | primaries are memory-only; situated results are mediators (§7.4) |
| `persist_current` confounded provenance with history | dates and authors are always shown; the Tier-2 arm varies only superseded versions (§2.2, §7.8) |
| Precision: too few responses per seed | K1c 8 items × 2 orders; all present members for H2; precision rule (§5.3, §7.5) |
| RQ1 construct: ΔSRI tracks referent knowledge | LAG = ΔSRI − ΔKNOW vs no-shift and Y; world-anchored W as a guaranteed target (§5.4–5.5) |
| Lexical-retrieval artifact | twins, machine focal, hit-rate logging, G1 gate (§5.2–5.3) |
| Compliance confound | memory-only primaries, follow rate, DCF, record-only reader, `on_choice` arm (§5, §7.8) |
| The cue leaks priors | one neutral cue text for both mappings, C0 diagnosticity gate and covariate (§1.4, G1) |
| Hedge attractor | three-way old/new/other scoring; slow speed .35 (§1.3) |
| Too many cells per day for learning (ecology) | one machine, one changing class, about 4.2 K1 faults a day; G2 (§1.5) |
| Short record window (minimal) | front page + log; 3 founding days (§2.1) |
| Replicates under a near-deterministic CLI | post-T salt with re-derived agent streams; labelled as retrieval/lens divergence (§6.2, §0.5) |
| Turnover coincides with the wipe | named a total effect; newcomer-only access is a later arm (§0.5) |
| `LLM_ERROR` cached as model output | fail-fast; errors never recorded; zero-error gate (§6.1) |
| Prefix-replay mode missing | `replay_until_tick`, `PrefixDivergence`, never serve cache at or after T (§6.1) |
| Reseeding breaks the prefix | branch salt from T on; world script never salted (§6.2) |
| Departed names and newcomer lexicon | persona universe at t0 for names and lexicon; world-text penalty extended to stamps, menu, onboarding and tally (§3.1, §5.4) |
| Newcomers are strangers by construction | symmetric ties ≥ 0.3; talk-to-newcomer check (§3.4) |
| Probe retrieval differs from the simulation (no meta, wrong clock) | meta, clock, roster and universe loaded in probes (§5.2) |
| Cost under-planned | 1.7k/day planning; per-job encoding; levers; drop rule (§4.2, §7.6–7.7) |
| Binder encoded on every read | new-only receipts (§2.3) |
| Synchronous ASK inside the decision phase | clarification through TALK in phase 6; sorted writes (§4.4) |
| K1 and K3 coupling broke job pairing | K3 carved from K0 with its own uniform; day-local job keys (§1.5, §1.7) |
| Observer re-queries could pollute or hit the cache | separate clients and scopes; DCF "unchanged" on a new scope (§5.2, §5.9) |
| State-restore fallback | dropped (§6.2) |
| Trunk prompts depend on branch config | overlay whitelist; per-tick digests; `_condition` only in the manifest (§6.2) |
| Incomplete turnover removal; seed-scope newcomers; injected notices | active set; zero-record test; tick-scoped seeds; notices as world facts (§3.3–3.4) |
| Write prompt pushes toward rules | neutral wording, character limits, audit (§2.4, §8.1 step 10) |
| Concurrent streams hit limits | one simulation stream; probes after, or at ≤ 2 workers after G1 (§6.6) |
| SRI noise floor | pooled at cell level over members × framings; retest SD ≤ .1 at G1 (§5.5, §5.11) |
