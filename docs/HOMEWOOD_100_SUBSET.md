# 100 agents: the connected campus cut the meme-cohort run uses

Prepared September 19, 2026; meals and the two-hall split added the same day (D76). A **100-person subset of the [500-person campus](HOMEWOOD_500_EXPERIMENT.md)**, chosen so a 100-agent run keeps enough of the designed social network to be worth running. Nothing about the 500-agent design, profiles, or seeds is changed; this is a selection on top of them, plus one deterministic pass that gives every selected agent a lunch and a dinner. No LLM run has been made at this size — every number below is an offline count of the population files.

Machine-readable provenance for everything here: [HOMEWOOD_100_SUBSET.json](HOMEWOOD_100_SUBSET.json), written by the same command that writes the population.

## The subset

| Group | Count |
|---|---:|
| First-year undergraduates | 20 |
| Sophomores | 20 |
| Juniors | 20 |
| Seniors | 20 |
| Instructional faculty | 9 |
| Campus staff | 11 |
| **Total** | **100** |

The category mix is proportional to the source: 80.0% students against 80.0% in the 500 (400/500), 9% faculty against 8.8% (44/500), 11% staff against 11.2% (56/500). Every agent keeps the biography, personality, and seed memory it has in the 500-agent files, byte for byte; only the meal entries of the routine are rewritten, by the pass described below.

- Profiles: [homewood100.yaml](../configs/population/homewood100.yaml)
- Seed memories carried over from the 500: [homewood100_initial_memories.yaml](../configs/population/homewood100_initial_memories.yaml), [homewood100_both_names_memories.yaml](../configs/population/homewood100_both_names_memories.yaml)

> **These two memory files belong to the retired naming study.** They seed every agent with a proper name for the dining hall ("FFC" / "Hopkins Cafe"). A meme-cohort run must not load them: a permanent naming habit for the same referent competes for the conversational airtime the injected phrases need, and it hands agents a label the world is otherwise careful never to supply. `configs/homewood100_memes.yaml` correctly points `initial_memories_file` at its own file; any new study config must do the same, or set it to `null`.

## Why not just take the first 100

`population_size: 100` loads the first 100 agents of the file. The 500-agent relationship graph was built over all 500 people, so a prefix keeps only the edges whose *both* endpoints happen to land inside it, and it inherits whatever cohort mix the file's shuffled order happens to give.

| | First 100 of homewood500 | This subset |
|---|---:|---:|
| Students / faculty / staff | 73 / 13 / 14 | **80 / 9 / 11** |
| Relationship edges kept (of 1,404) | 60 | **214** |
| Mean degree | 1.20 | **4.28** |
| Agents with no tie at all | 26 | **0** |
| Connected components | 41 (largest 14) | **2 (97 + 3)** |
| Groups with ≥ 2 members | 49 | 43 |
| Surviving group memberships | 217 (mean group 4.4) | **246 (mean group 5.7)** |

The prefix keeps slightly *more* group names because it scatters people thinly across many of them; the subset keeps fewer, fuller groups, which is what the designed structure was for.

## How it was selected

[`scripts/subsample_population.py`](../scripts/subsample_population.py) is deterministic; `--seed` only breaks exact ties.

1. **Label** comes from each agent's primary seed memory — the first `--split` label appearing in the text, otherwise the residual pool.
2. **Quotas.** Agents are bucketed by `(demographics.category, demographics.year, label)`. Each label's target is spread across its buckets by largest remainder, in proportion to the source bucket sizes, so the split is exact and the mix stays proportional.
3. **Snowball.** Growth starts from the densest group cluster (a `groups:` entry ranked by internal relationship edges per possible member pair, clusters of three or more) and repeatedly admits the candidate with the most ties *into the set so far*, tie-broken by shared group memberships, then source degree, then a seeded per-agent key. Only candidates whose bucket still has quota are admissible. A greedy snowball is sensitive to where it starts, so the growth is repeated once per candidate seed group in density order and the best result is kept.
4. **Split repair** swaps until every label matches its target. Here it is a no-op.
5. **Tie maximisation.** Best-improvement local search: swap a selected agent for an unselected agent *of the same bucket* whenever that strictly increases the retained edge count.
6. **Meals** (`--meals`, below) rewrites each selected agent's meal entries. It runs after selection and changes nobody's membership, so every number in the tables above is identical with and without it.

`--split "Hopkins Cafe=20"` is still passed, and it is now a **demographic** constraint rather than a naming one: in the 500-agent design only first-years carry that seed label, so the flag is what pins the cut at 20 agents per student year. Dropping it would change which 100 people are selected and therefore every count on this page. It says nothing about what any agent will call anything.

Printed summary of the committed files:

| | |
|---|---|
| Seed group | `academic_Philosophy` (best of 78 candidate seeds tried) |
| Split-repair swaps | 0 |
| Tie-maximising swaps | 7 |
| Admissions with no tie into the set | 1 |
| Categories | 80 student, 9 faculty, 11 staff |
| Years | 20 first-year, 20 sophomore, 20 junior, 20 senior, 9 faculty, 11 staff |
| Relationships | 214 of 1,404 retained (15.2%); first-100 baseline 60 |
| Mean degree | 4.28 (source 5.62; first-100 baseline 1.20) |
| Components | 2 — sizes 97 and 3 |
| Groups | 43 of 78 kept, 35 dropped below two members |

Edge mix in the subset: 114 acquaintance, 52 clubmate, 48 friend. Degree runs from 1 to 10, median 4, no isolates. 143 of the 214 edges cross cohorts.

## Everyone eats, and where is a manipulation

Before D76, 63 of the 100 had any routine step at a dining hall and the rest ate at the cafe, on the quad, or in a dorm lounge. **All 100 now take a midday and an evening meal at one of the campus's two dining halls.** A meal is where co-located conversation happens and where an incident sited in a hall is witnessed, so an agent with no meal cannot take part in either.

The campus has a second hall, `Nolans` ([world.py](../backend/simulation/world.py)), sited east beside the graduate flats. It is not a naming experiment: it is the **unexposed control venue**. People who eat there have to *hear* about a dining-sited incident; they cannot see it.

### The rule

Two facts already in each profile decide everything. There is no coin flip.

- **Where you live** sets both meals: `demographics.residence: on campus` → the hall beside the freshman residences, `off campus` → the east hall.
- **Where your working day happens** is the one exception: if your longest-dwell routine location is *at or next door to the other hall* (read off `WORLD_GRAPH`, so a new edge needs no new rule), you take the midday meal there and the evening meal at your own. Those agents are the bridge.

Residence was chosen over a random half because a convention needs sub-communities: the split has to run *along* the social graph and still be crossed by enough ties for a phrase to get over it. Both properties are measured, not assumed.

| | hall beside the freshman residences | the east hall (`Nolans`) |
|---|---:|---:|
| Eats the evening meal there | **53** | **47** |
| Eats the midday meal there | 64 | 36 |
| Mean degree | 3.83 | 4.79 |
| Degree min / median / max | 2 / 4 / 7 | 1 / 4 / 10 |
| Agents with ≥ 1 tie to the other hall | 52 of 53 | 41 of 47 |
| Internal components | 46, 2, 2, 1, 1, 1 | 43, 3, 1 |
| Composition | 53 students | 27 students, 11 staff, 9 faculty |

**99 of the 214 relationships (46.3%) cross the two halls.** Each side is one large component plus a few pairs, so both sides are real sub-communities rather than fragments, and 93 of the 100 agents have a tie across. No rebalancing was needed.

**Who has reason to visit both.** 11 agents, all campus staff, and it falls out of the rule rather than being placed by hand: they live off campus (evening meal at the east hall) and work at or beside the other hall (midday meal there). Eight of them are the dining-service staff, who are on the serving floor of the exposed hall from 08:30 to 16:30 whatever they eat — in the 500-agent design they are also the only staff with relationships to students, so they are the strongest carriers the population has.

### Times

Lunch starts land on the tick grid at 11:30, 11:45, 12:00, 12:15, 12:30, 12:45, 13:00; dinner at 17:30, 17:45, 18:00, 18:15, 18:30. A meal is 30 minutes, i.e. two 15-minute ticks at the table. Dining-service staff take the last two lunch slots, keeping the source generator's intent (a break after the service peak, not in the middle of the student wave) as far as a window that has to hold everyone's lunch allows.

Slots are **dealt round-robin** inside each sitting rather than drawn from a hash, because peak head-count in a room is not cosmetic: `perception.crowd_factor` (0.92) is charged per other person in the room, so a hall that seats its whole side at once leaves each diner almost blind to what happens there. Round-robin cuts the peak from 38 to 24.

| Peak in one room in any tick | midday | evening |
|---|---:|---:|
| hall beside the freshman residences | 24 | 22 |
| the east hall (`Nolans`) | 11 | 20 |

What this buys, counted over the whole roster: **agent-ticks spent inside a dining hall rise from 366 to 456 over the old 08:30–17:00 day (+25%), and to 720 over the 08:30–19:30 day the study config uses (+61% against the same window before)**, and the number of pairs a hall could seat rises from 219 to 356. That is the direct answer to whether this part helps or hurts a phrase reach speech.

At a peak of 24 the crowd term alone is 0.92²² ≈ 0.16. Anything that has to be *witnessed* at a meal, rather than said in a conversation (where participants attend with probability 1), is paying that. It is the floor the two windows allow at 15-minute ticks; widening a window or lengthening the day is the only way down from here.

### Where an incident can be sited

An incident sited where nobody goes produces no witnesses. Agents with at least one routine step in each room, out of 100:

| Place | Room | Agents |
|---|---|---:|
| Dining Hall | Main Floor | 64 |
| Classroom | Seminar Room | 57 |
| Nolans | Servery | 47 |
| Library | Study Tables | 45 |
| Research Lab | Wet Lab | 44 |
| Quad | Lawn | 36 |
| Classroom | Lecture Hall | 36 |
| Gym | Main Floor | 35 |
| Dorm | Lounge | 34 |
| Cafe | Counter | 25 |
| Research Lab | Dry Lab | 19 |
| Research Lab | Makerspace | 10 |
| Dorm | (five bedrooms) | 8–12 each |

Both sites the shipped registry uses exist and are well populated: `tray_washer` at `Dining Hall / Main Floor` reaches 64 agents' routines, `hood_sash` at `Research Lab / Wet Lab` reaches 44. The two ungrounded memes have no site by construction.

`Research Lab → Stockroom` and the Shuttle Stop, Museum, Theater, Apartments, Admin Building, Student Center, Auditorium, Engineering Hall, Science Hall and Athletic Center have **no** routine step at all in this population — agents reach them only by errand or by choosing to move — so an incident sited in any of them would be seen by nobody by design. Check this table before siting one.

## The second hall on the map

An agent standing at either hall is told "the dining hall …" and then whose side of campus it is on: **"the dining hall beside the freshman residences"** and **"the dining hall by the graduate flats"**. Matched wording is deliberate — nothing the world says marks one hall as the main one and the other as the alternative — and the second description deliberately avoids the design brief's draft wording ("the other dining hall, over by the east apartments"), which names the canonical place `Apartments`, runs to nine words, and carries a comma that would split the `reference.options()` MOVE list into two bogus options. All three are checked by `tests/test_reference.py`.

`Nolans` is drawn as the west wing of Scott-Bates Commons (the former Charles Commons, which houses the real Nolan's on 33rd), on N Charles Street beside The Charles. Its walking neighbours in `WORLD_GRAPH` are the real ones, `Apartments` and `Museum`, and the reciprocal edges are appended to the *end* of those two lists. All-pairs BFS over the 18 pre-existing places before and after: **0 of 324 shortest paths changed.**

The tile map is regenerated by `scripts/build_homewood_map.py` and the vector layer by `scripts/export_realmap.py`. The build is deterministic: two consecutive runs produce byte-identical `frontend/homewood_map.json`, `frontend/homewood_thumb.png` and `frontend/homewood_vector.json`, and adding the hall changed 254 tile cells, all inside its own footprint (x 405–425, y 372–384), leaving all 18 pre-existing place entries byte-identical. The one caveat is the PNGs: their pixels are reproducible but their file bytes depend on the installed Pillow version, so a different Pillow will show a binary diff with identical pixels.

## What the subset cannot keep

- **It is not one component, and no 100-agent cut can be.** In the source, only *Hopkins dining service* staff have any relationship outside their own department (their 120 ties to students). Every other staff department is a closed ring. An 11-staff share therefore cannot all reach the students: the sampler takes all 8 dining workers into the main component and the remaining 3 from one department (Student services), where they form a connected trio of their own. That is the structural optimum, not a sampling failure.
- **Majors concentrate.** 27 of the 80 students are Neuroscience and 14 Computer Science, against a much flatter spread in the 500. Anything that reads as a field effect in this subset is confounded with the selection.
- **Most groups are gone.** 35 of 78 groups fall below two members and are dropped. Club-wide dynamics are not represented.
- **Selection is not random.** The subset is chosen to be dense, so it over-represents the well-connected. It cannot be used to estimate anything about a typical member of the 500.
- **The hall split is confounded with cohort and role.** Residence is not independent of who you are: the exposed hall's 53 are all students and mostly first-years and sophomores, while the east hall holds every member of staff and faculty and most seniors. That is what makes it legible and graph-aligned, and it is also why "the exposed side adopted it more" can never be read as an exposure effect on its own — it is equally a cohort effect. Any comparison across halls has to be made within cohort, or with cohort in the model.
- **The bridge group is exactly the staff.** All 11 both-halls agents are campus staff, so "bridge" and "staff" cannot be separated in this population either.

## How it differs from the 500-agent design

| | 500 | 100 |
|---|---|---|
| Network | 1,404 edges, mean degree 5.62, all 78 groups | 214 edges, mean degree 4.28, 43 groups |
| Cohorts | 100 per student year, 44 faculty, 56 staff | 20 per student year, 9 faculty, 11 staff |
| Selection | every generated person | quota-constrained snowball on the same people |
| Meals | one lunch, scheduled per cohort; 63 of 100 pass a dining hall | **two meals, both at a dining hall, all 100** |
| Dining venues | one | **two, split 53 / 47 by residence** |

Everything else — biographies, personalities, rooms, disabled jitter, detours, invitations, and latent events — is inherited unchanged.

## Commands

Regenerate the subset (writes all three population files plus the summary; byte-identical every time):

```bash
.venv/bin/python scripts/subsample_population.py --n 100 --split "Hopkins Cafe=20" --seed 42 --meals \
    --out configs/population/homewood100.yaml --summary docs/HOMEWOOD_100_SUBSET.json
```

`--meals` is off by default, so the same command without it reproduces the pre-D76 population byte for byte.

Rebuild the campus map after any change to `ARENAS` or the place tables:

```bash
.venv/bin/python scripts/build_homewood_map.py     # needs Pillow
.venv/bin/python scripts/export_realmap.py
```

Offline checks of the population files:

```bash
.venv/bin/python -m pytest tests/test_homewood100_population.py tests/test_campus_places.py -q
```

`--set` values go through `yaml.safe_load`, which reads `12:30` as the integer 750 (YAML 1.1 base-60). Change clock times by editing the config, not with `--set`.

## What this can and cannot support

**It can support**, as offline accounting over a fixed, documented population:

- A per-agent exposure variable that is a property of the design rather than of chance co-location: who eats where, who bridges, and how many ties cross the halls.
- Comparisons of what happens at an exposed venue against an unexposed one with the same rooms, the same room description, and the same one arena, so a difference between them is not a difference in decor.
- Inspection of individual exchanges by cohort, hall and bridge status.

**It cannot support:**

- Any claim about Johns Hopkins, its students, or its staff. The population is synthetic.
- Inference from the 100 back to the 500. The subset is deliberately dense and major-concentrated; it is not a sample.
- Statistics across seeds. This is one seed and one population cut.
- Separating exposure from cohort, or bridging from staff status, without stratifying — see the confounds above.
- A throughput estimate from the mock backend.
