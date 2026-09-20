# Experiment plan — a cohort of injected memes

**Pre-registration. Written before the run so the result cannot be reverse-engineered into a hypothesis.**
Read with `docs/ONTOLOGY_V2.md` (mechanisms, hidden state, layer separation), `docs/ONTOLOGY_V3.md` §5
(measurement, isolation, validity controls) and `docs/research/MEMO_2026-09-19_research_landscape.md`
(§2.1 the eight outcomes, §6 prior art, §7 RQ1).

Status: plan only. No result is claimed here. Every number below is either a threshold **decided now** or a
measurement, and measurements say where they came from.

---

## 0. What this replaces, and why

**The seeded naming study is retired.** 80 agents seeded to call one dining hall "FFC" against 20 seeded
"Hopkins Cafe" can exhibit only two of the memo's eight outcomes (§2.1): diffusion and formal variation. A
proper noun for a fixed building has no boundary that can move, so semantic broadening, narrowing, local
specialization, pragmatic change and normative change are all unreachable by construction. Shared names with
committed minorities are also crowded prior art (memo §6; Ashery et al., *Science Advances* 2025).

**Injecting one meme is also not enough.** With n = 1 the finding is "this phrase broadened" or "this phrase
died", and the *why* is unconstrained. So we inject a **cohort** whose members differ on dimensions we
control, and the result becomes *which property predicts survival and reinterpretation* — comparative rather
than anecdotal.

---

## 1. The question

> When several coined expressions are injected into the same community at the same time, **which ones
> survive, and which ones come to apply to things beyond what they were coined for?**

Read against the memo's RQ1 (an established expression acquires a different interpretation) and the v2
bottlenecks: grounding supplies NEED and LINK (there is a recurring thing to refer to, and instances of it to
connect), breadth decides whether the LINK can reach past the origin at all.

### The 2×2

|                           | **narrow extension**                                | **broad extension**                                      |
|---------------------------|-----------------------------------------------------|----------------------------------------------------------|
| **grounded** (witnessed origin event) | applies to one specific thing; cannot broaden by construction | applies to a class of situations; can broaden |
| **ungrounded** (no incident; seeded only) | an arbitrary label the minority just uses | a general evaluative phrase the minority just uses |

Crossed on **grounding** and **breadth** only. *Practical stakes* is deliberately **not** manipulated: making
utility real needs the v3 workshop task world, and turning the free campus into a controlled task was
explicitly rejected. Stakes is recorded as an **observed covariate** (does a usage co-occur with a
consequence the speaker reports?) and never as a factor.

---

## 2. The cohort (the registry)

Nothing about a meme is hardcoded. One config block declares the whole cohort
(`memes.registry`, `backend/analysis/battery/registry.py`); adding, removing or swapping a meme is a config
edit. This is load-bearing: the prior check in §9 can force a replacement.

| id | phrase (candidate) | cell | context | origin |
|---|---|---|---|---|
| `tray_washer` | "blue-tray" | grounded × broad | dining | the dish machine at the freshman-side hall runs cold for days; what comes out looks clean and is still greasy; some people get sick. Extension: anything that looks fine and is not. |
| `hood_sash` | "hood-three" | grounded × narrow | lab | one extraction cabinet reads closed on its panel and does not seal. Applies to that one cabinet. |
| `running_warm` | "running warm" | ungrounded × broad | coursework / social | no incident. An evaluative phrase for anything overcommitted or behind. |
| `tuesday_list` | "the Tuesday list" | ungrounded × narrow | coursework | no incident. An arbitrary label for one recurring thing. |

Phrases are **candidates, not decisions**: §9 can reject any of them. The four sit in different parts of
campus life on purpose (§8, airtime).

The draft registry used for tonight's prior check — with `origin_probe`, `probe_gradient` and `foils` written
so that no text contains the entry's own distinctive words — is reproduced at the end of this document.

### Matching (R6)

The four memes must differ **only** in their factors: same minority size `k = 5`, comparable graph positions,
comparable habit-line length and register. If one minority is better connected, a difference in spread is
attributable to the graph and the design is dead. The seed sets are drawn to match and the **degree
distributions are reported side by side** in the gate (§10, G0-5).

---

## 3. Injection, manipulation, unit of analysis

- **Injection.** Each meme's phrase appears in exactly one place in the whole world: the habit line given to
  its `k = 5` seeded agents (R1). Seeds are spread across the graph, disjoint between memes.
- **Incidents.** Grounded memes have a staged incident on days 1–3 at a fixed location/arena, with
  `prob_per_meal` per meal window (lunch 11:30–13:30, dinner 17:30–19:00). Incident text never contains the
  phrase, and never contains the words the phrase is built from (R2).
- **The manipulation.** On **day 4** the basis of the grounded memes is removed: the thing that was failing is
  repaired. The ungrounded memes have nothing to remove, which is exactly what makes them the comparison.
- **Windows.** PRE = days 2–3 (incident live, injection settled). Day 4 is the transition and is excluded from
  the ratio. POST = days 5–6.
- **Unit of analysis.** The **meme-run** (meme × seed). With a single seed the 2×2 comparison is descriptive
  effect-size estimation, not a test; the cost table (§11) prices three seeds, which gives three observations
  per cell and is the smallest design in which the cell means mean anything.

---

## 4. The three outcomes, in numbers, decided now

All primary measures are **coder-free**: they come from utterance matching
(`registry.usage_pattern`, which treats "blue-tray" / "blue tray" / "blue trays" as one expression and records
the variant) plus v2 typed provenance. A Sonnet coding of each usage is **secondary** and never decides an
outcome (ONTOLOGY_V3 §5.10).

**On-origin vs off-origin (the coder-free proxy for "applies to things beyond its origin").** A usage is
ON-ORIGIN if it happens in the meme's home location/arena, **or** its utterance's `referent_event_ids` include
one of that meme's incident events, **or** the conversation was triggered by one. Otherwise OFF-ORIGIN. For
ungrounded memes the home context is declared in the registry (`home: {location, arena}`), since there is no
incident to take it from.

| outcome | definition (per meme, per seed) |
|---|---|
| **UNMEASURABLE** | fewer than **15 uses** or fewer than **6 distinct speakers** across days 1–3. Reported as unmeasurable, **never as "dies"**. |
| **DIES** | POST uses ÷ PRE uses ≤ **0.25**, or zero uses on day 6. |
| **CALCIFIES** | ratio > 0.25 **and** off-origin share in POST < **0.20** **and** Δ off-origin share (POST − PRE) < **+0.10**. |
| **BROADENS** | ratio > 0.25 **and** off-origin share in POST ≥ **0.35** **and** Δ ≥ **+0.15** **and** ≥ **3 distinct speakers** produce off-origin uses. |
| **AMBIGUOUS** | anything between calcify and broaden (0.10 ≤ Δ < 0.15, or 0.20 ≤ share < 0.35). Reported as ambiguous, not rounded to the nearer story. |

Secondary, reported but not decisive: **adoption** (distinct non-seed speakers by end of day 3), **formal
variation** (the distribution of recorded variants), **pragmatic function** (coded), and the **applicability
probe** (§7) run prior-adjusted.

---

## 5. Comparative prediction, and two falsifiers

Prior (ours, stated in advance so it can be wrong): **grounding buys early adoption; breadth buys survival.**

| cell | predicted |
|---|---|
| grounded × broad (`tray_washer`) | highest early adoption; survives the repair **by broadening** — the only cell predicted to broaden |
| grounded × narrow (`hood_sash`) | high early adoption; **dies or calcifies** after the repair — nothing left to refer to |
| ungrounded × broad (`running_warm`) | low early adoption; if it survives, it survives by breadth; off-origin from the start |
| ungrounded × narrow (`tuesday_list`) | lowest adoption; predicted to **die** |

- **The grounding hypothesis is falsified** if mean adoption (distinct non-seed speakers by end of day 3) over
  the two grounded memes is not at least **1.5×** that of the two ungrounded memes. Grounded ≤ ungrounded
  falsifies it outright.
- **The breadth hypothesis is falsified** if the mean POST/PRE use ratio over the two broad memes is not
  **greater** than over the two narrow memes.

They are falsified **separately**: the design is a 2×2 precisely so that one can fail while the other holds.

---

## 6. The null, and telling broadening from an agreeable model

**The null** — no cell differs on adoption, survival or off-origin share beyond the between-seed spread — is a
real, publishable, honest outcome, and will be reported as one. Given the single-seed pilot has no error term,
a null from S = 1 is reported as "no visible difference at this n", not as evidence of no effect.

**The agreeableness worry.** "Broadening" measured by asking a model whether a phrase fits a situation can be
an artifact of a model that says yes to everything. The check is the **foil false-positive rate**: foils share
surface features with the gradient but not its structure, and a fresh model's acceptance of them is the
false-positive floor the run must clear.

Measured tonight (`scripts/probe_priors.py`, haiku, n = 8, 96 calls, 0 refusals, 0 errors, $0.0778): the
**foil acceptance rate is 0.00 for all four memes** — of 64 foil responses, **0 said "fits"**, 3 "not sure",
61 "doesn't fit". Across all 192 item-responses: **107 fits / 72 doesn't fit / 13 not sure**. So the format
discriminates, and a rise on the foils during the run would be a genuine red flag rather than the
instrument's own noise.

**The format matters, and this was measured, not assumed.** Presenting the situations one per call instead of
together collapses the measurement: on the same texts, the 192 item-responses become **8 fits / 54 doesn't
fit / 130 not sure** (68% hedging), and the foils pick up 8 stray "fits" — because with no contrast set the
model abstains rather than discriminates. The applicability probe must therefore present the gradient and the
foils **together** (`--boundary batched`, the shape of ONTOLOGY_V3 §5.3 format A), and "not sure" must be
reported as its own category rather than scored as half a fit.

**The headroom problem, also measured.** Cold, a fresh Haiku already accepts every gradient item for three of
the four phrases (prior breadth 4/4 for `tray_washer`, `running_warm`, `tuesday_list`; 3/4 for `hood_sash`).
Consequence, decided now: **broadening is read from in-run use (§4), not from probe breadth.** The probe can
show *narrowing*, can compare cells at the same checkpoint, and can be reported prior-adjusted (agent minus
fresh model in the same format) — it cannot show an upward shift it has no room for.

---

## 7. Transmission vs independent coinage

If the model would coin the phrase by itself, spread is not evidence of transmission. Two defences:

1. **R2, machine-checked.** The world never emits a registry phrase, and never emits the words that phrase is
   built from. The word list is derived in exactly one place
   (`registry.MemeSpec.distinctive_words`) and `scripts/probe_priors.py` checks its own prompts against the
   same list, so the probe cannot accidentally hand over the answer it is testing for.
2. **The coinage probe (§9b).** The origin situation is described to a fresh model without those words, and
   the model is asked for the three most likely nicknames. Measured tonight, n = 8 per meme: **0/8 hits for
   all four memes.** What it reached for instead: "grease trap" ×8, "fake hood" ×4 / "ghost hood" ×3,
   "overcommitted" ×8, "the holy trinity" ×2 / "the three amigos" ×2. The model understood every situation
   and named none of them our way.

In-run, independent use is still separated from adoption by the v2 exposure logic
(`backend/analysis/emergence.py`): a non-seed speaker who used the phrase with no prior exposure counts as
independent, not as an adopter.

---

## 8. Confounds

1. **Minority graph position (R6).** The strongest threat. Seed sets are matched on degree and the
   distributions are reported side by side (G0-5). Unmatched sets invalidate the comparison.
2. **Hall and exposure assignment.** The `tray_washer` incident is staged only at *the dining hall beside the
   freshman residences*; *the other dining hall, over by the east apartments* is unexposed. Exposure is
   therefore **not randomized** — it follows routines, and the two halls' populations differ systematically.
   Exposure is measured per agent (did they perceive an incident beat?) and used as a covariate; no causal
   claim is made from the hall contrast alone.
3. **Competition for airtime between four memes.** The likeliest failure of the whole build. Mitigated by
   spreading the memes across dining, lab, coursework and social contexts. Measured directly: each meme's
   share of all meme-bearing utterances. If one meme takes **> 70%** of them, the other cells' nulls are
   reported as *possibly competition*, not as properties of grounding or breadth.
4. **The model's own prior.** `running warm` is already meaningful to Haiku on **8/8** samples — and with a
   meaning close to the opposite of the seeded one ("an engine operating at normal temperature; proceeding
   smoothly"). Its results carry that caveat permanently; it is a candidate for replacement (§9).
5. **Gradient validity.** `tuesday_list` scored near 0.50 but mid 1.00 and far 1.00 cold — its "gradient" is
   not a distance gradient to the model. A meme whose gradient is non-monotonic cold cannot support a
   distance-based reading of broadening; its items need rewriting before the run.

---

## 9. The prior check (`scripts/probe_priors.py`)

A reusable artifact aimed at the **registry**: it reads `memes.registry` and tests every entry, so swapping a
phrase re-runs its check. A fresh model, no simulation context, the CLI invoked from an empty directory
outside the repo (inside it the model answers as a coding assistant — 50% refusals and answers naming the
project when this was done by hand). Refusals are their own category: a refusal is not evidence of absence.

- **(a) arbitrary** — does the phrase already mean something?
- **(b) coinage — the go/no-go.** The origin without the distinctive words; "what would people call it?". Any
  spontaneous production of the phrase (`--coinage-max 0`) **rejects that meme**.
- **(c) boundary** — the four gradient situations and the foils, cold: the prior boundary, without which an
  end-of-run boundary has no baseline to be a shift from.

```
.venv/bin/python scripts/probe_priors.py configs/<study config>.yaml --n 8
# exit 0 = all pass, 2 = at least one meme must be replaced or reviewed, 1 = the probe itself failed
```

**Result on the draft registry** (haiku, n = 8, batched, 96 calls, 31 s, $0.0778, 0 refusals, 0 errors):

| meme | coined by the model | already known | prior breadth | foil accept | hedge | verdict |
|---|---|---|---|---|---|---|
| `tray_washer` | 0/8 | 0/8 | 4/4 | 0.00 | 0.12 | REVIEW (no probe headroom) |
| `hood_sash` | 0/8 | 0/8 | 3/4 | 0.00 | 0.06 | PASS |
| `running_warm` | 0/8 | **8/8** | 4/4 | 0.00 | 0.04 | REVIEW (known, and near-opposite, prior) |
| `tuesday_list` | 0/8 | 0/8 | 4/4 | 0.00 | 0.04 | REVIEW (no headroom; non-monotonic gradient) |

Per-item "fits" share, cold:

| meme | literal | near | mid | far | foil0 | foil1 |
|---|---|---|---|---|---|---|
| `tray_washer` | 1.00 | 1.00 | 0.88 | 0.50 | 0.00 | 0.00 |
| `hood_sash` | 1.00 | 0.75 | 1.00 | 0.25 | 0.00 | 0.00 |
| `running_warm` | 1.00 | 1.00 | 1.00 | 0.50 | 0.00 | 0.00 |
| `tuesday_list` | 1.00 | 0.50 | 1.00 | 1.00 | 0.00 | 0.00 |

**Go/no-go: no meme is rejected on the coinage test.** Three carry review caveats, which §6 and §8 absorb. The
probe's own noise floor: two runs of the same registry at the same n put `hood_sash` prior breadth at 3/4 and
4/4, so a one-item difference in breadth is within run-to-run variation and is not a finding.

Raw reports, with every response kept: `runs/dev_priors/registry_draft_haiku_n8_batched.json`,
`runs/dev_priors/registry_draft_haiku_n8_per_item.json`, against
`runs/dev_priors/registry_draft.yaml` (`registry_sha256` 792fdec343d7…). **Re-run this against the shipped
study config before launching** — these numbers belong to the draft registry, not to whatever the config
finally ships.

---

## 10. The gate — what must be true before spending real money

**G0, mock backend, minutes.** All mechanical; all must pass.

1. `registry.validate()` returns no errors; the 2×2 has one meme per cell; `k` equal across memes.
2. **R1/R2 prompt audit** over every rendered prompt in `llm_calls.jsonl`: zero occurrences of any registry
   phrase and of any distinctive word in agent-facing text, except the seeds' habit lines (expected count =
   `k` × occurrences). One stray occurrence and the run measures supply, not transmission.
3. **D75 no-regression:** zero canonical place names in agent-facing text (`world.reference_mode: situated`;
   measured on 2026-09-19 at 95.4% → 0.0% of prompts carrying a canonical label).
4. **Determinism (R4):** the same seed twice gives an identical `trace.jsonl` sha256.
5. **R6 matching:** seed sets disjoint, `k = 5` each, degree distributions printed side by side; each set's
   median degree within ±1 of the cohort median and no set holding more than one top-5-degree agent.
6. Incident fires on days 1–3 with the expected count and **the repair is applied on day 4** (trace record
   present); every new mechanism is off by default in `configs/default.yaml` (R5).

**G1, live, one day, 100 agents (~15.9k calls, measured 2.2–5.7 h; see §11).** The airtime gate.

7. Every meme reaches **≥ 5 uses by ≥ 3 distinct speakers on day 1, at least one of them not a seed.**
8. The repair on day 4 is perceived: **≥ 30% of agents present at the hall** encode a repair observation
   (checked on a 4-day pilot, or waived to a post-hoc validity flag if the pilot stops at day 1).

Failing G1 means **do not launch**: strengthen the injection (larger `k`, higher
`initial_memories_importance`, more cue windows) and re-pilot. It does not mean run it anyway and hope.

---

## 11. Cost

Measured, from this repository, not estimated.

| quantity | value | source |
|---|---|---|
| agents | 100 (`configs/population/homewood100.yaml`) | config |
| day length | 08:30–17:00, 15-min ticks = **34 ticks/day** | `runs/verify_cell1_situated/manifest.json` |
| days | 6 (incident 1–3, repair day 4, post 5–6) | this plan |
| ticks | **204** (6 × 34) | above |
| utterances | **~1,418 per day** (100 agents ≈ 14 per agent-day) | `runs/verify_wording_2day` (2 days, 2,836 `chat_utterance` calls) |
| LLM calls, mechanisms off | **130.4 per agent-day** (13,039 in 1 day × 100 agents) | `runs/verify_cell1_situated/llm_calls.jsonl` |
| LLM calls, wording mechanisms on | **~15,900 per day**, flat across days (15,986 then 15,835) | `runs/verify_wording_2day/llm_calls.jsonl` |
| **projected calls, 6-day run** | **~95,400** | 6 × 15,900 |
| token volume per call | ~2,184 prompt chars, ~342 response chars ⇒ ≈ 550 in / 85 out | `verify_wording_2day` (prompts), `baseline_s42` (live responses) |
| live throughput | **0.26–0.67 calls/s at 4 workers** (2,723 s / 1,718 calls; 6,284 s / 1,607 calls) | `runs/baseline_s42`, `runs/c2_baseline_haiku_s1` |
| **projected wall-clock, 6 days at 12 workers** | **13–34 h** (assumes throughput scales linearly with workers) | above |
| one-day pilot (G1) | ~15,900 calls, **2.2–5.7 h** | above |
| prior check (§9) | **96 calls, 31 s, $0.0778** per full re-check; 256 calls / $0.0981 in `per_item` mode | measured tonight |
| three seeds (the smallest inferential design) | ~286,000 calls, 40–103 h | 3 × above |

Dollar cost of the simulation is deliberately not projected: the CLI path records no cost field, so it would
be an estimate. Token volume is given instead. (The probe's dollar figures are real — the CLI returns
`total_cost_usd` per call and the script sums them.)

---

## 12. What would make the result uninterpretable

Listed honestly, before the run.

1. **No meme clears the measurable floor** (§4). Then there is nothing to compare and the correct report is
   "the injection did not reach speech", not a story about grounding.
2. **R1/R2 audit fails** anywhere: spread becomes indistinguishable from the world supplying the words.
3. **A phrase was swapped and the prior check was not re-run**, or was run against a different registry sha
   than the one the run used. The report records `registry_sha256`; they must match.
4. **Seed sets unmatched on degree** (R6): spread attributable to the graph.
5. **The repair is not perceived** (G1-8): "the basis was removed" is then false and the manipulation did not
   happen.
6. **One meme takes > 70% of meme-bearing utterances**: the other cells' nulls may be crowding out.
7. **Probe-only broadening claims**: ruled out in advance by §6 — the prior has no headroom, so a probe rise
   is not available as evidence either way.
8. **Determinism broken**: a re-run with the same seed that diverges makes every contrast unreplicable.

---

## Appendix — the draft registry probed in §9

Reproduced so the numbers above are attached to exact texts. `origin_probe` is the origin as the *world*
describes it, free of the phrase's distinctive words; `probe_gradient` and `foils` are **observer-only** and
must never reach an agent (R7).

```yaml
memes:
  enabled: false
  registry:
    - id: tray_washer
      phrase: "blue-tray"
      grounding: grounded
      breadth: broad
      habit: "{name} has a habit of calling anything that looks fine and isn't a blue-tray."
      seeds: {k: 5, strategy: spread}
      incident: {location: "Dining Hall", arena: "Main Floor", days: [1, 2, 3],
                 repaired_day: 4, prob_per_meal: 0.6}
      origin_probe: "For several days the dish machine at one dining hall ran cold. What came out of it
        looked clean and still felt greasy, and a few people got sick."
      probe_gradient:
        literal: "Things came out of the dish machine looking clean and still feeling greasy."
        near: "The glasses in the rack came out spotted, though the machine said the cycle had finished."
        mid: "A lab bench was wiped down and signed off, but the residue was still there the next morning."
        far: "Someone's problem set was neatly written out and completely wrong."
      foils:
        - "The dish machine broke down and the dirty dishes piled up at the window."
        - "Someone dropped a full stack of dishes in the servery and everyone clapped."
    - id: hood_sash
      phrase: "hood-three"
      grounding: grounded
      breadth: narrow
      habit: "{name} has a habit of calling that one extraction cabinet hood-three."
      seeds: {k: 5, strategy: spread}
      incident: {location: "Research Lab", arena: "Wet Lab", days: [1, 2, 3],
                 repaired_day: 4, prob_per_meal: 0.6}
      origin_probe: "In one wet lab, the extraction cabinet by the window reads closed on its panel but
        never actually seals, so fumes come back into the room."
      probe_gradient:
        literal: "The sash on the extraction cabinet by the window showed closed on the panel but never sealed."
        near: "A different extraction cabinet in the same lab would not hold its airflow."
        mid: "The autoclave's door latched on the display but not in fact."
        far: "The building's card reader said the door was locked when it was not."
      foils:
        - "The wet lab ran out of nitrile gloves in the middle of a run."
        - "A student left a beaker of solvent open on the bench overnight."
    - id: running_warm
      phrase: "running warm"
      grounding: ungrounded
      breadth: broad
      habit: "{name} has a habit of saying someone is running warm when they are overcommitted and behind."
      seeds: {k: 5, strategy: spread}
      incident: null
      origin_probe: "Someone has said yes to more than they can finish and is now a week behind on all of it."
      probe_gradient:
        literal: "Someone has taken on more than they can finish and is a week behind on all of it."
        near: "A study group keeps pushing its deadline back because everyone said yes to too much."
        mid: "The print shop is taking three days instead of one because of the backlog."
        far: "The coffee line moves slowly because there is only one person on the machine."
      foils:
        - "Someone got their work in early and had the weekend free."
        - "The heating in the reading room is set too high."
    - id: tuesday_list
      phrase: "the Tuesday list"
      grounding: ungrounded
      breadth: narrow
      habit: "{name} has a habit of calling the seminar's left-over readings the Tuesday list."
      seeds: {k: 5, strategy: spread}
      incident: null
      origin_probe: "The same three readings are left over from one seminar every week and nobody ever
        gets to them."
      probe_gradient:
        literal: "The same three readings are left over from the seminar every week and nobody gets to them."
        near: "Another seminar has its own set of readings that nobody ever gets to."
        mid: "The lab meeting has standing agenda items that are always deferred."
        far: "A group chat has a set of plans everyone keeps saying they will do."
      foils:
        - "The seminar met early in the week and finished with time to spare."
        - "Someone wrote down everything they needed from the shop."
```
