# 100 agents: a connected cut of the Hopkins Cafe / FFC naming experiment

Prepared September 19, 2026. A **100-person subset of the [500-person campus](HOMEWOOD_500_EXPERIMENT.md)**, chosen so a 100-agent run keeps both the requested **20/80 naming split** and enough of the designed social network to be worth running. Nothing about the 500-agent design, profiles, or seeds is changed; this is a selection on top of them. No LLM run has been made at this size — the numbers below are offline counts of the population files.

## The subset

| Group | Count | Initial preferred name |
|---|---:|---|
| First-year undergraduates | 20 | Hopkins Cafe |
| Sophomores | 20 | FFC |
| Juniors | 20 | FFC |
| Seniors | 20 | FFC |
| Instructional faculty | 9 | FFC |
| Campus staff | 11 | FFC |
| **Total** | **100** | **20 Hopkins Cafe / 80 FFC** |

The split is exact, and the category mix is proportional to the source: 80.0% students against 80.0% in the 500 (400/500), 9% faculty against 8.8% (44/500), 11% staff against 11.2% (56/500). Every agent keeps the biography, personality, routine, and seed memory it has in the 500-agent files, byte for byte.

- Profiles: [homewood100.yaml](../configs/population/homewood100.yaml)
- Primary naming seeds: [homewood100_initial_memories.yaml](../configs/population/homewood100_initial_memories.yaml)
- Both-names recognition control: [homewood100_both_names_memories.yaml](../configs/population/homewood100_both_names_memories.yaml)

## Why not just take the first 100

`population_size: 100` loads the first 100 agents of the file. The 500-agent relationship graph was built over all 500 people, so a prefix keeps only the edges whose *both* endpoints happen to land inside it, and it inherits whatever cohort mix the file's shuffled order happens to give.

| | First 100 of homewood500 | This subset |
|---|---:|---:|
| Naming split | 15 / 85 | **20 / 80** |
| Students / faculty / staff | 73 / 13 / 14 | **80 / 9 / 11** |
| Relationship edges kept (of 1,404) | 60 | **214** |
| Mean degree | 1.20 | **4.28** |
| Agents with no tie at all | 26 | **0** |
| Connected components | 41 (largest 14) | **2 (97 + 3)** |
| Hopkins Cafe ↔ FFC edges | 11 | **49** |
| Hopkins-side agents with ≥ 1 FFC-side tie | 8 of 15 | **20 of 20** |
| Groups with ≥ 2 members | 49 | 43 |
| Surviving group memberships | 217 (mean group 4.4) | **246 (mean group 5.7)** |

The prefix keeps slightly *more* group names because it scatters people thinly across many of them; the subset keeps fewer, fuller groups, which is what the designed structure was for. The two naming rows are the ones that matter most for this experiment: in the prefix, seven of the fifteen Hopkins-side agents have no relationship at all to anyone on the FFC side, so the contrast the study is about would depend entirely on who happens to share a room.

## How it was selected

[`scripts/subsample_population.py`](../scripts/subsample_population.py) is deterministic; `--seed` only breaks exact ties.

1. **Naming label** comes from each agent's primary seed memory — the first `--split` label appearing in the text, otherwise the residual pool. Classification reads the primary seed file, never the both-names control, where every text carries both names.
2. **Quotas.** Agents are bucketed by `(demographics.category, demographics.year, label)`. Each label's target is spread across its buckets by largest remainder, in proportion to the source bucket sizes, so the split is exact and the mix stays proportional.
3. **Snowball.** Growth starts from the densest group cluster (a `groups:` entry ranked by internal relationship edges per possible member pair, clusters of three or more) and repeatedly admits the candidate with the most ties *into the set so far*, tie-broken by shared group memberships, then source degree, then a seeded per-agent key. Only candidates whose bucket still has quota are admissible, so growth cannot drift off the split. A greedy snowball is sensitive to where it starts, so the growth is repeated once per candidate seed group in density order and the best result is kept.
4. **Split repair** swaps until every label matches its target, preferring a replacement of the same category and year. Here it is a no-op, because in the 500-agent design the name follows the cohort (only first-years carry "Hopkins Cafe"), so step 2 already pins the split. It exists for source populations whose cohorts mix names.
5. **Tie maximisation.** Best-improvement local search: swap a selected agent for an unselected agent *of the same bucket* whenever that strictly increases the retained edge count. Quotas, mix, and split are invariant under such a swap.

Relationships are kept only when both endpoints are inside; groups are restricted to selected members and dropped below two; circles are carried over from the source the same way (the 500-agent file has none, so this is a no-op).

Printed summary of the committed files:

| | |
|---|---|
| Seed group | `academic_Philosophy` (best of 78 candidate seeds tried) |
| Split-repair swaps | 0 |
| Tie-maximising swaps | 7 |
| Admissions with no tie into the set | 1 |
| Naming split | 20 Hopkins Cafe / 80 other |
| Categories | 80 student, 9 faculty, 11 staff |
| Years | 20 first-year, 20 sophomore, 20 junior, 20 senior, 9 faculty, 11 staff |
| Relationships | 214 of 1,404 retained (15.2%); first-100 baseline 60 |
| Mean degree | 4.28 (source 5.62; first-100 baseline 1.20) |
| Components | 2 — sizes 97 and 3 |
| Groups | 43 of 78 kept, 35 dropped below two members |

Edge mix in the subset: 114 acquaintance, 52 clubmate, 48 friend. Degree runs from 1 to 10, median 4, no isolates. 143 of the 214 edges cross cohorts.

## What the subset cannot keep

- **It is not one component, and no 100-agent cut can be.** In the source, only *Hopkins dining service* staff have any relationship outside their own department (their 120 ties to students). Every other staff department — Facilities, Library services, Student services, Campus safety, IT services, Levering dining, Laboratory support — is a closed ring. An 11-staff share therefore cannot all reach the students: the sampler takes all 8 dining workers into the main component and the remaining 3 from one department (Student services), where they form a connected trio of their own. That is the structural optimum, not a sampling failure. Lowering the staff share, or adding cross-department staff ties to the 500-agent generator, is the only way to change it.
- **Majors concentrate.** Density is bought by keeping classmates together: 27 of the 80 students are Neuroscience and 14 Computer Science, against a much flatter spread in the 500. Anything that reads as a field effect in this subset is confounded with the selection.
- **Most groups are gone.** 35 of 78 groups fall below two members and are dropped, and the remaining clubs are partial. Club-wide dynamics are not represented.
- **Selection is not random.** The subset is chosen to be dense, so it over-represents the well-connected. It cannot be used to estimate anything about a typical member of the 500.

## How it differs from the 500-agent design

| | 500 | 100 |
|---|---|---|
| Naming split | 100 / 400, exactly one quarter of students | 20 / 80, exactly one quarter of students |
| Network | 1,404 edges, mean degree 5.62, all 78 groups | 214 edges, mean degree 4.28, 43 groups |
| Cohorts | 100 per student year, 44 faculty, 56 staff | 20 per student year, 9 faculty, 11 staff |
| Selection | every generated person | quota-constrained snowball on the same people |
| Lunch destination | scheduling assumption per cohort | inherited unchanged; 53 of 100 lunch at the Dining Hall, 63 pass through it at some point |
| Backend in the shipped config | mock (structure inspection) | `claude_cli` / haiku, 12 workers |
| Mechanisms | plain cognition only | two cells: plain cognition, and the WORDING channel |

Everything else — biographies, personalities, routines, rooms, the 08:30–17:00 window, disabled jitter, detours, invitations, and latent events — is inherited unchanged.

## The two run configs

| Key | [`homewood100_naming.yaml`](../configs/homewood100_naming.yaml) | [`homewood100_naming_wording.yaml`](../configs/homewood100_naming_wording.yaml) |
|---|---|---|
| `memory.verbatim.enabled` | false | **true** |
| `priming.enabled` | false | **true** |
| `retrieval.source_weights` | `{}` | **`{seed: 0.5, ambient: 0.5}`** |
| `conversation.catchup.enabled` | false | **true**, `after: "16:00"` |

Everything else is identical and inherited: seed and `world_seed` 42, `population_size` 100, `simulation_days` 3, 08:30–17:00, `llm: {backend: claude_cli, model: haiku, max_workers: 12}`, `latent_events.event_rate` 0, `routine.jitter_minutes` 0, `routine.deviation_prob` 0, `conversation.invite_prob` 0, `topology.mode: file`, `analysis: {llm_classifier: true, probes: false}`. The wording cell is the same block as the `wording: on` level of [`configs/designs/bottleneck_factorial.yaml`](../configs/designs/bottleneck_factorial.yaml), so the comparison is a single named mechanism, not an assortment.

**One deliberate deviation from the default.** `conversation.catchup.after` defaults to `"17:00"`, but with `day_end: "17:00"` and 15-minute ticks the day's last tick is 16:45, so the default would make `catchup.enabled: true` a switch that does nothing. It is set to `"16:00"` here, which gives four late-afternoon ticks. A paired mock run confirms it fires: one day of the wording cell produces 354 conversations and 1,294 utterances with `after: "17:00"`, and 384 conversations and 1,420 utterances with `after: "16:00"`. It is a late-afternoon catch-up, not the evening one the mechanism was written for.

The recognition control from the 500 design is not shipped as a third config. To run it, point `initial_memories_file` at `configs/population/homewood100_both_names_memories.yaml`; the file is generated and covers exactly these 100 agents.

## Commands

Regenerate the subset (writes all three population files; byte-identical every time):

```bash
.venv/bin/python scripts/subsample_population.py --n 100 --split "Hopkins Cafe=20" --seed 42 \
    --out configs/population/homewood100.yaml
```

Run the two cells. They are independent and can run at the same time; each prints its run directory (`runs/<timestamp>_<run_name>_s42`) as it starts.

```bash
.venv/bin/python -m backend.cli run --config configs/homewood100_naming.yaml
.venv/bin/python -m backend.cli run --config configs/homewood100_naming_wording.yaml
```

A one-day pilot first, to confirm throughput and that the seeded names actually reach speech:

```bash
.venv/bin/python -m backend.cli run --config configs/homewood100_naming.yaml \
    --out runs/pilot_homewood100 --set simulation_days=1
```

`--set` values go through `yaml.safe_load`, which reads `12:30` as the integer 750 (YAML 1.1 base-60). Change clock times by editing the config, not with `--set`.

Count the names in a finished run:

```bash
.venv/bin/python scripts/analyze_dining_names.py runs/YOUR_RUN --out runs/YOUR_RUN/dining_names.json
```

Offline checks of the population files:

```bash
.venv/bin/python -m pytest tests/test_homewood100_population.py -q
```

## What this can and cannot support

**It can support**, as lexical accounting over a fixed, documented population:

- Per-cohort shares of utterances containing FFC only, Hopkins Cafe only, or both, over three days, with distinct-speaker counts alongside message counts — the competition metric the 500 design specifies.
- A within-subject comparison of those shares between plain cognition and the WORDING channel, with the same people, schedules, seeds, and model. The two cells share `world_seed`, so they see the same world.
- Whether the seeded first-year name reaches anyone outside its cohort at all: all 20 Hopkins-side agents have at least one relationship to an FFC-side agent, and 49 edges cross the naming line, so the opportunity is in the design rather than left to chance co-location.
- Inspection of individual exchanges by cohort, which the 500 design asks for before any claim about confusion, repair, or adoption.

**It cannot support:**

- Any claim about Johns Hopkins, its students, or its staff. The population is synthetic; the 20/80 split is a user-specified initial condition, not a measurement. See the source boundaries in the [500-agent guide](HOMEWOOD_500_EXPERIMENT.md).
- Inference from the 100 back to the 500. The subset is deliberately dense and major-concentrated; it is not a sample.
- Statistics across seeds. This is one seed and one population cut. Convergence or persistent division at n = 100 over three days is a single observation, and two cells at one seed cannot separate a mechanism effect from run-to-run variation. Repeat with several seeds before comparing cells.
- Anything about confusion, understanding, repair, or adoption from the name counts alone. [`scripts/analyze_dining_names.py`](../scripts/analyze_dining_names.py) counts lexical mentions; a quoted "they call it FFC" counts, and `no_target_alias` does not mean "no dining reference".
- A throughput estimate from the mock backend. A one-day mock run of the wording cell logged 384 conversations, 1,420 utterances, and roughly 18,700 LLM calls; mock replies are degenerate and inflate reflection, so treat that as an upper-bound smoke test, not a budget.
