# MemeWorld: build report (overnight autonomous build, 2026-09-19)

## TL;DR

- **The simulator, analyzer and UI are built and tested** (16 tests). Everything runs on top of the original
  Generative Agents code, which is vendored unmodified and imported.
- **All five experimental conditions ran for 3 simulated days** with the same seed (42), 8 agents and Claude
  Haiku 4.5. That is about 1.4–2.1k LLM calls per run, with 0 errors. Results are in **§Results** below.
- **Main finding: over 3 days, no durable convention for a hidden event type emerged in any condition.** What
  did emerge:
  - One genuine local convention, "**the frisbee analogy**" (event-rich run). It is a named explanatory trick for
    conditional probability, invented by Maya and reused by 4 agents over 3 days. It refers to a shared episode,
    not to a latent event.
  - Event nicknames ("the backpack mix-up") and gossip chains that drift as they pass between agents.
  - A treatment artifact: the emotion module's own prompt wording ("a bit restless") became the most repeated
    word in the first social-reward run. After the fix, the re-run (`runs/social_reward_v2_s42`) has 0 uses of
    "restless", and its top candidates are ordinary vocabulary again.
- **Every choice that is not GA-native or not in your plan** is logged in
  **[docs/DECISIONS.md](docs/DECISIONS.md)**.

## What was built

| Plan item | Status | Where |
|---|---|---|
| Agent ontology: profile, relationships, routine, state, memory stream, reflections | ✅ | `backend/agents/profile.py`, `agent.py`, `configs/population/homewood8.yaml` |
| No meme state in agents | ✅ enforced by tests | `tests/test_invariants.py` |
| Experimental modules as swappable hooks | ✅ emotion, social reward, prestige, conformity; memory noise and capacity via config | `backend/modules/` |
| Campus graph, time, schedules, semantic movement | ✅ | `backend/simulation/world.py`, `scheduler.py` |
| 8 students, 7 overlapping groups, 4 bridge agents (Maya, Hana, Jordan, Sofia) | ✅ | population yaml |
| Routines with jitter, detours, event-forced deviations, conversation delays, invitations, reactive re-planning | ✅ | `scheduler.py`, `engine.py` |
| Hidden latent events E1–E4, diverse surfaces, held-out surfaces | ✅ 33 scenario scripts plus slot pools | `backend/simulation/latent_events.py` |
| Partial perception | ✅ fact-level; attention, participation, relationship, salience, room, busy | `backend/agents/perception.py` |
| Lossy symbolic memory (pre-filter, generalization, noise-dependent reconstruction, merge) | ✅ | `backend/memory/encoder.py` |
| Decay and forgetting | ✅ | `store.py`, `encoder.py` |
| Stochastic retrieval, P ∝ exp(s/τ) | ✅ GA scoring + Gumbel-top-k | `backend/memory/retrieval.py` |
| Constrained reflection | ✅ GA prompts + plan's pattern questions; never asks for names | `backend/memory/reflection.py` |
| Decision loop with structured actions | ✅ MOVE / TALK / REACT / CONTINUE (IDLE ≈ routine) | `backend/agents/planner.py` |
| 1–4 turn conversations on GA's own prompts | ✅ | `backend/agents/conversation.py` |
| Exposure logging (actual hearers only, including overhearing) | ✅ | trace `exposure` records |
| Config-driven runs, exact config stored | ✅ | `configs/*.yaml`, `config.resolved.yaml` |
| Analyzer: candidates, grouping, adoption, transmission, semantics, lineage | ✅ | `backend/analysis/` |
| Private meaning probes and generalization multiple-choice probe | ✅ | `backend/analysis/probes.py` |
| Generalization test on held-out surfaces | ✅ | `latent_events.holdout_from_day` |
| Ground-truth precision, recall, generalization, drift, lift | ✅ | `backend/analysis/evaluation.py` |
| Full causal trace | ✅ | `trace.jsonl`; `/api/runs/{id}/chain/{utterance}` |
| Frontend: campus, agent inspector, Research Debug Mode | ✅ | `frontend/` |
| Culture dashboard (all 7 panels) | ✅ | `frontend/app.js` |
| Replay: seeded world, stored LLM outputs, play/pause/speed/rewind/jump | ✅ | `cli replay`; UI timeline |
| 5 experimental modes, differing only in config | ✅ | `configs/` |
| Cross-condition comparison | ✅ (extra) | `cli compare`; Runs tab |

### MVP acceptance criteria (§37)

| # | Criterion | Status |
|---|---|---|
| 1 | ≥ 6 agents with distinct profiles | ✅ 8 |
| 2 | Explicit relationships and overlapping routines | ✅ |
| 3 | Movement through a Homewood-style campus | ✅ |
| 4 | Local-only perception | ✅ |
| 5 | Lossy memory | ✅ |
| 6 | Stochastic retrieval | ✅ tested |
| 7 | Private reflection | ✅ |
| 8 | Natural conversation | ✅ |
| 9 | Unnamed recurring latent classes | ✅ |
| 10 | No labels in any prompt | ✅ checked on every recorded prompt (tests + the real integration run) |
| 11 | No meme state in agents | ✅ |
| 12 | Full logging | ✅ |
| 13 | Candidate detection | ✅ |
| 14 | Propagation visualization | ✅ |
| 15 | Private probes | ✅ |
| 16 | Ground-truth comparison | ✅ |
| 17 | Deterministic replay | ✅ tested |
| 18 | Module toggles without core-code changes | ✅ tested |

## How Generative Agents is used

The upstream repo (`joonspk-research/generative_agents@fe05a71`) was sparse-cloned into
`third_party/generative_agents/` and is **not edited**. `backend/ga_compat.py` stubs its OpenAI/`utils`
dependencies, routes every LLM call to Claude, and exposes the upstream modules to MemeWorld.

GA-native pieces in the loop:
- memory stream (`AssociativeMemory` / `ConceptNode`, including its on-disk format)
- `Scratch` identity set
- retrieval scoring functions and weights
- reflection trigger, focal-point prompt, insight prompt
- poignancy prompts
- `decide_to_talk`
- relationship summary and iterative conversation (called directly)
- the sprites

What is new is exactly what the plan asked for: partial perception, lossy encoding, stochastic retrieval,
latent events, modules, observer, replay. The complete list is in DECISIONS.md §0.

## What the real-LLM runs showed (qualitative; not an experimental result)

From the integration run and about 35 ticks of baseline with Claude Haiku 4.5:

- **Partial perception works as intended.** In one E3 event, Dev noticed Sofia's umbrella on a sunny day
  (p = 0.30) but not her reason. Sofia remembered the dare. Ethan, elsewhere, saw only the anonymous third umbrella
  carrier, then *started a conversation* about it with Sofia, who explained the dare.
- **Memories are reconstructive and diverge.** For example, Leo encoded an overheard remark as "Maya … sounding
  frustrated … seemed like she'd made a scheduling mistake."
- **Rumors spread and mutate.** Priya's backpack mix-up (E1) became campus gossip within an hour. Leo later
  confabulated that the backpack had been *his*; Jordan relayed "Maya mentioned Priya taking her backpack"; Ethan
  asked Priya about a backpack with blue duct tape.
- **Reflection abstracts the hidden structure without being asked.** Priya's reflections included "Priya is
  experiencing a cascade of complications from the backpack mix-up" and "…a ripple effect impacting multiple
  people". The word *cascade* arose in private reflection, not from any label. Whether such words become shared
  conventions is exactly what the full runs need to show.
- **No hidden label appeared** in any of the ~1,300 recorded prompts checked (E1–E4, family names, scenario keys, event
  ids, "meme", "slang", "invent").

On the smoke runs (mock LLM), the analyzer surfaces the mock's own templated phrases. That is expected and
confirms the pipeline end to end; it is **not** a finding.


## Results (seed 42, 3 days, 8 agents, Claude Haiku 4.5 via CLI)

Produced with `python -m backend.cli compare runs/*_s42` (also shown in the UI's Runs tab). "Conventions" are
candidates the observer's LLM classifier labels as locally specific. Where a run has none, the pooled metrics use
its top-3 candidates.

| condition | events | convs | utterances | reflections | candidates | LLM conventions | max adoption | mean depth | cross-group edges | coherence | alignment | lift | top expression |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 14 | 82 | 356 | 216 | 44 | 0 | 0.50 | 1.0 | 0 | 0.34 | 0.53 | 0.99 | might've |
| perfect memory | 27 | 73 | 332 | 234 | 41 | 0 | 0.75 | 1.0 | 5 | 0.29 | 0.40 | 0.75 | how's your lunch |
| high noise | 16 | 67 | 297 | 180 | 41 | 0 | 0.38 | 1.33 | 0 | 0.40 | 0.72 | 1.48 | bayes theorem |
| social reward (v1, flawed) | 19 | 84 | 355 | 342 | 45 | 0 | 1.00 | 1.0 | 4 | 0.34 | 0.76 | 1.85 | restless ⚠ |
| social reward (v2, fixed) | 20 | 87 | 378 | 324 | 41 | 0 | 0.50 | 1.0 | 2 | 0.38 | 0.50 | 1.45 | swamped |
| event rich | 42 | 83 | 386 | 288 | 41 | **1** | 0.50 | 1.0 | 1 | 0.37 | 0.48 | 2.25 | frisbee analogy |

**How to read this.** With one seed per condition, differences between rows are anecdotes, not effects. The
alignment and lift columns also inherit the provenance over-attribution caveat (D26). The qualitative findings
matter more:

1. **One real local convention: "the frisbee analogy"** (event-rich run; Culture tab → "frisbee analogy"; Trace
   tab → search "frisbee analogy").
   - **Origin:** on day 1, Maya explains conditional probability to Ethan with a frisbee example before his
     frisbee game.
   - **Spread:** Ethan passes it on to Leo, and on day 2 Leo coaches Ethan: "That frisbee thing Maya mentioned
     last time? Might help to visualize it that way."
   - **Named as a thing:** on day 3, Ethan says "remember that frisbee analogy you gave me?" and Maya answers
     "Oh yeah, the frisbee analogy! I'm glad that stuck with you."
   - **Is it a meme?** It fits the plan's definition of a meme (reused language with a locally specific
     referent). It is **not** a symbol for a hidden event type: it never appeared on held-out events, and the
     generalization-probe accuracy was 0.125 (chance is 0.25). Its E1 "alignment" of 0.48 is an artifact of
     provenance over-attribution.
   - **Private probes:** all 8 agents, *including the 4 who used it*, say they have never heard the expression.
     With baseline-level memory noise (0.3), the exact wording did not survive in memory; the gist ("Maya's
     frisbee thing") did, and it was regenerated from context. This is exactly the phrase-copying versus
     concept-memory distinction the probes were designed to show.
   - **Analyzer flaw exposed:** variant grouping merged the literal "frisbee game" uses into the candidate, so it
     credits Ethan as inventor. The analogy was Maya's.
2. **Event nicknames without generalization.** "the backpack mix-up" (baseline), "extra crispy" (perfect memory;
   the wording comes from a held-out event's surface text), "the cape", "the pattern" (high noise). Each is used by
   2–3 agents about *one* incident. None is reused for a *different* instance of the same hidden type, which is the
   step from event reference to latent concept that did not happen within 3 days.
3. **Gossip mutates as it spreads (lossy memory at work).**
   - An E1 backpack incident became campus gossip within an hour.
   - Leo later confabulated that the backpack was *his*.
   - Dev's memory turned "Maya waited 40 minutes while he returned a backpack" into "a backpack mix-up at the
     counter".
   - Reflections abstracted the hidden structure without being asked: "Priya is experiencing a cascade of
     complications…", "…a ripple effect".
   - None of these abstractions was lexicalized into shared speech.
4. **Treatment artifact in social reward (v1).**
   - Emotion arousal saturated, so the mood line "X is feeling a bit restless" was injected into 349 of 385 chat
     and reaction prompts.
   - Agents repeated "restless" in 92 of 355 utterances (baseline: 0).
   - It spread to all 8 agents, so the observer's metrics show max adoption 1.0 and the highest lift. The LLM
     classifier correctly rejected it as ordinary language, but a naive frequency-based meme detector would have
     reported a successful meme.
   - **Lesson for the experiment design:** a treatment that adds words to prompts can masquerade as cultural
     transmission. The module now moves arousal toward event intensity and adds mood lines only for clearly
     non-neutral states (D36).
   - **v2 re-run (same seed, fixed module):**
     - Mood lines appear in 167 of 413 chat and reaction prompts instead of 349 of 385.
     - "restless" appears in 0 of 378 utterances.
     - The mood words that remain surface rarely: "excited" 9 times, "stressed" 4 times.
     - Social reward still has the most reflections of any condition (324, versus 216 in baseline).
     - Prosocial gratitude language recurs: "you're a lifesaver" is used by 3 agents, mostly about Jordan's
       help with probability. It reads as a reputational tag in the making, but the classifier did not accept it
       as a convention.
     - The discovery pass flagged "the 12-minute thing", a reference to the held-out E1 incubator-timer
       incident on day 3. Only one agent used it, so it is not a candidate.
5. **Memory conditions.**
   - Perfect memory had the most cross-group exposure edges (5) and the highest max adoption among the non-social
     conditions, but its most-adopted expressions are greetings ("how's your lunch").
   - High noise produced the fewest utterances and conversations.
   - All of this is suggestive only (one seed).

**What would make convergence more likely** (not done; these are selective-pressure or scale choices for you to
make):
- longer runs (a week or more)
- more seeds
- denser recurrence of the *same* latent type with more varied surfaces (`latent_events.generator: llm`)
- a larger population
- conformity or prestige turned on

## Problems found and fixed during the build

1. **CLI latency.** The `claude` CLI had extended thinking on by default, making calls 3–10× slower. It is now
   disabled.
2. **Scope bug.** A closure bug gave every conversation in a tick the same LLM-cache scope, which broke replay
   determinism. It is fixed, and replay is now bit-identical (tested).
3. **Parsing of davinci-era prompts.** Chat models add preambles and markdown to GA's davinci-style prompts, and
   one reflection stored "Based on the statements provided, here are 3 insights:" as an insight. Parsers are now
   tolerant (D5); prompt text is unchanged.
4. **Invisible overlay.** A hidden modal was intercepting all clicks in the UI (a CSS `[hidden]` bug). Fixed.
5. **Time parsing.** Unquoted YAML times like `10:00` parse as sexagesimal integers. The clock handles both forms
   now.
6. **Upstream recency bug.** GA's rank-based recency gives the *oldest* memory the highest score. MemeWorld uses
   time-based decay instead (D20).
7. **CLI hangs.** Occasional hangs are cut off by a 45 s timeout with retries.
8. **Memory reloading.** Reloading saved memories for the probes collided with ids left gapped by forgetting.
   This crashed the high-noise analysis and could silently drop a memory. Fixed, with a regression test; all
   analyses were re-run.
9. **Candidate ranking.** Common English words ("earlier", "sounds") dominated the candidate lists. A
   word-frequency prior (`wordfreq`) was added on the observer side only.
10. **Emotion-module wording leak** (Results §4). Fixed, and social reward is being re-run.
11. **Laptop sleep on battery.** It stalled the first batch of runs. They were re-run under `caffeinate` once the
    laptop was charging.

## Caveats you should know before interpreting results

- **Provenance over-attributes** (D26). A conversation memory inherits the events of *all* memories the speakers
  retrieved, so latent-event precision and recall are approximate.
- **Embeddings are lexical** (D2), so "semantic coherence" measures lexical-contextual similarity.
- **Temperature cannot be controlled** with the CLI backend (D1). Use `llm.backend=anthropic` with an API key for
  that.
- **One seed per condition** is a demo, not a study. For real inference, run several seeds:
  `scripts/run_experiments.sh <seed> 5`.
- **Surface repetition.** The template bank is finite, so world wording can repeat. The analyzer down-weights
  n-grams that appear verbatim in world text, but still inspect the top candidates by hand.
- **Throughput.** With the CLI backend a 3-day run takes about 1.5–2 hours and 1.5–2.5k LLM calls. Keep the
  machine awake and on power: `caffeinate -dimsu`.

## Deliverables

- Code: `backend/`, `frontend/`, `configs/`, `scripts/run_experiments.sh`, `tests/`
- Docs: `README.md`, `docs/DECISIONS.md`, this `REPORT.md`
- Sample runs:
  - `runs/dev_smoke`: mock, 2 days, analyzed; open it in the UI
  - `runs/dev_smoke2`, `runs/dev_smoke_replay`: determinism checks
  - `runs/partial_*`: interrupted real runs (unanalyzed)
