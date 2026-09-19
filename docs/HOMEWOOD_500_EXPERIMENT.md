# 500 agents: Hopkins Cafe / FFC naming experiment

Prepared September 19, 2026. **500 fictional people, including students, faculty, and staff.** Profiles are generated; no 500-agent LLM experiment has been run. This is a seeded naming study, separate from the earlier spontaneous-emergence experiment.

## The population

| Group | Count | Initial preferred name |
|---|---:|---|
| First-year undergraduates | 100 | Hopkins Cafe |
| Sophomores | 100 | FFC |
| Juniors | 100 | FFC |
| Seniors | 100 | FFC |
| Instructional faculty | 44 | FFC |
| Campus staff | 56 | FFC |
| **Total** | **500** | **100 Hopkins Cafe / 400 FFC** |

Exactly **one quarter of students** start with “Hopkins Cafe”; three quarters of students and every faculty/staff agent start with “FFC.” Across the entire population, this is a 20/80 split.

**FFC stands for Fresh Food Cafe.** Both names refer to the same physical dining hall. The simulator's internal destination remains `Dining Hall`; its separate `Cafe` destination is Levering Cafe.

Start with the [readable 500-person roster](HOMEWOOD_500_ROSTER.md). The [full population YAML](../configs/population/homewood500.yaml) contains every biography, trait, interest, routine, relationship, and group. All names are fictional combinations, not copied from real student or employee records.

## What comes from public data, and what is a modeling choice

| Item | Public evidence | How this population uses it |
|---|---|---|
| Student–faculty ratio | Hopkins reports **9:1**. The 2025–26 CDS calculation uses **9,537 undergraduate-plus-graduate FTE students and 1,037 instructional faculty**, with stated exclusions. | 400 students / 44 faculty = **9.09:1**. An approximate benchmark, **not** a Homewood undergraduate census reconstruction. |
| Undergraduate scale | Admissions reports about **5,600 undergraduates**. | This is a reduced synthetic campus, not all of Hopkins or a random representative sample. Graduate students and medical-campus populations are outside this scenario. |
| Academic mix | Older Krieger and Whiting materials describe roughly **3,500** and **2,000** undergraduates, respectively. | Use **256 Arts & Sciences / 144 Engineering students**, about 64/36. Exact major counts are scenario choices. Each student has one primary field for bookkeeping; double majors are not modeled. |
| Organizations | Admissions reports **83%** involved in at least one organization. | 332 students have a club membership, balanced at 83 per class. Applying the same share to every year is an extra modeling choice. |
| First-year housing | Admissions reports **93%** of first-years living on campus. | 93 of 100 first-years are assigned on-campus housing. Other years' housing shares are assumptions. |
| Dining employees | JHU reported retaining **more than 150 hourly union dining workers** in its 2022 Homewood/Peabody dining transition. Its directory confirms varied dining occupations. | Include dining workers, but do **not** turn that multi-campus historical count into a current Hopkins Cafe staffing estimate. All 56 staff allocations below are modeling choices. |
| Personalities and language | No Hopkins-specific distribution of these traits or the requested naming split was established. | Synthetic, transparent trait assignments; user-specified starting names. No claim of measured student behavior. |

Sources: [2025–26 CDS, section I-2](https://oira.jhu.edu/wp-content/uploads/2025-2026-CDS-Johns-Hopkins-University-v2.pdf), [Admissions Fast Facts](https://apply.jhu.edu/fast-facts/), [Krieger 2024 strategic plan](https://krieger.jhu.edu/wp-content/uploads/2024/08/StrategicPlan_Aug24.pdf), [Whiting 2023 booklet](https://engineering.jhu.edu/wp-content/uploads/2023/05/HEEP-Booklet_Final.pdf), [official dining staff directory](https://studentaffairs.jhu.edu/dining/staff/). These sources differ in year and coverage; they supply design anchors, not jointly consistent population microdata.

**Historical correction:** the official JHU announcement dates the Fresh Food Cafe → Hopkins Cafe rename to **August 2022**. The present experiment's freshman-versus-continuing-student split is inspired by that real change; it is not a verified description of fall 2026 cohorts. Older students can inherit “FFC” from peers without having attended before the rename. [JHU announcement](https://hub.jhu.edu/2022/08/23/hopkins-dining-menu-upgrades-operational-changes/)

## Staff and individual differences

| Staff group | Count | Typical responsibilities |
|---|---:|---|
| Main dining hall | 8 | Shift supervision, cooking, serving, cashier, dishroom, utility work |
| Levering dining | 4 | Cooking, cashier, serving, utility work |
| Library | 10 | Reference help, circulation, digital collections |
| Facilities | 10 | Custodial and maintenance work |
| Student services | 12 | Advising, residential life, organizations, registration |
| Laboratory support | 6 | Wet lab, dry lab, makerspace support |
| Campus safety | 4 | Campus safety liaison work |
| IT support | 2 | User support |

Faculty have teaching and consultation routines; staff have work duties and breaks. Dining workers take breaks after peak lunch service. Faculty ages use broad rank-related ranges; student ages advance with class year. These are plausible synthetic ranges, not empirical age distributions or judgments about ability.

To keep the naming comparison readable, **each class has the same personality margins**:

| Characteristic | Distribution in each 100-student class |
|---|---|
| Sociability | 20 reserved / 60 moderate / 20 outgoing |
| Planning style | 20 structured / 60 flexible / 20 spontaneous |
| Speaking style | 20 brief / 60 balanced / 20 detailed |

The same proportions are rounded for faculty and staff. These are simple modeling categories, not validated psychological scores. Characteristics are shuffled independently of major, occupation, name, and initial terminology. Housing, club membership, and meal destination are separately shuffled within class years; they are not coupled through a common ordering. No race, nationality, gender, or religion is inferred from names.

The current engine gives the literal traits `reserved` and `outgoing` different conversation probabilities. Other traits enter language prompts rather than separate numerical behavioral models. The [audit file](../configs/population/homewood500_audit.json) records assignments so this can be checked.

## How the names enter the experiment

Every person receives **one ordinary initial memory** describing their own previous use of their assigned name. For example:

> In my recent everyday conversations, I have called the dining hall beside AMR III Hopkins Cafe when arranging meals.

The older-name group's corresponding memory says “FFC.” Neither primary seed presents the other name. The [seed file](../configs/population/homewood500_initial_memories.yaml) is separate from permanent biographies and personality. Its entries are loaded once through `initial_memories_file` and can subsequently be retrieved, forgotten, or supplemented by conversation memories.

This gives exactly the requested **initial assignments**. It does not force every generated utterance to obey the assignment. Check actual early speech before interpreting later changes. Pretrained models may already recognize the real-world aliases even when their simulation memories do not show both.

Do not prompt agents to be confused, teach freshmen, copy a majority, or agree. Possible outcomes include continued disagreement in wording, successful mutual understanding with two names, or a shift toward either name.

An optional [both-names control](../configs/homewood500_both_names.yaml) keeps the same people, preferred-term history, and schedules, but explicitly tells each agent that both names denote the same hall. It helps distinguish a naming preference from missing alias information. This is an information intervention, not proof of agents' pre-existing ignorance.

## Meetings and schedules

The profiles provide 1,404 explicit undirected relationships and 78 overlapping groups: same-year peer groups, mixed-year clubs, academic contacts, work groups, and regular customer contacts. There are also first-year/continuing-student acquaintances. Neutral shuffled IDs avoid putting all freshmen first in the engine's sorted processing order.

Lunches are spread across six half-hour starts, 11:00–13:30. As an explicit scheduling assumption, 85 first-years, 75 sophomores, 50 juniors, and 45 seniors visit the main dining hall; others use Levering or eat on the Quad. These are **not measured dining shares**. Different cohorts therefore have different contact opportunities; report opportunities alongside name use.

The scenario models **08:30–17:00 campus activity**, not complete residential life or employment shifts. `home` in the existing schema is used as a starting campus anchor. `demographics.residence` separately records housing; faculty and staff live off campus, not in their mapped workspaces. Rooms are coarse shared-space proxies. Routine jitter, random detours, invitations, and unrelated scripted latent events are disabled in this scenario to keep the comparison interpretable.

**Profiles loading successfully does not establish a realistic 500-person contact model.** The current map has few rooms and standing positions. Ambient perception and overhearing can reach many people in one arena; matching can favor earlier IDs. Before a full LLM study, introduce or validate bounded nearby contacts/dining tables, inspect occupancy and contact counts by cohort, and make sure work activities stay plausible. The validation report measures scheduled crowding; it is not a performance benchmark.

## What to measure

Keep **wording**, **understanding**, and **confusion** separate:

| Question | Simple evidence |
|---|---|
| Which name do they use? | Per-cohort shares of utterances mentioning FFC only, Hopkins Cafe only, or both; distinct speakers as well as message counts |
| Do they express confusion? | An explicit question or mistaken destination in a relevant exchange; annotate the conversation |
| Do they repair it? | The exchange supplies an explanation and the participants demonstrate agreement about the destination |
| Do they adopt a name? | An agent uses the other name independently in later, separate conversations; distinguish quotation or correction from their own usage |
| Do they understand both? | Optional private probes on copied states; do not feed the answers back into ongoing conversations |

The provided [name analyzer](../scripts/analyze_dining_names.py) implements **lexical usage counts only**. It recognizes FFC, Fresh Food Cafe/Café, and Hopkins Cafe/Café. The existing generic novelty detector skips three-letter single words, so it cannot be the sole analysis for this experiment.

The name analyzer does not infer confusion, understanding, or adoption. A quoted “they call it FFC” still counts as a lexical mention. Utterances without either alias are labeled `no_target_alias`, not “no dining reference.” The denominator for a naming share is alias-bearing utterances, with zero-denominator results left undefined. Agent counts with no mentions are retained.

Inspect a small, fixed sample of actual exchanges from each cohort, including unsuccessful or ambiguous ones. Later reuse after exposure is trace evidence; it alone does not prove that a particular speaker caused adoption. Successful communication while both names persist is a meaningful result.

## Files and checks

- [Readable roster](HOMEWOOD_500_ROSTER.md): all 500 people in one table.
- [Profiles](../configs/population/homewood500.yaml): complete loadable population.
- [Initial memories](../configs/population/homewood500_initial_memories.yaml): primary naming split.
- [Both-names memories](../configs/population/homewood500_both_names_memories.yaml): optional information control.
- [Audit metadata](../configs/population/homewood500_audit.json): counts, assignments, generator seed, and profile hash; observer-only.
- [Scenario config](../configs/homewood500_naming.yaml): uses the mock backend by default for structural inspection.
- [Validation report](HOMEWOOD_500_VALIDATION.json): offline checks and scheduled occupancy, with no inference about cultural outcomes.

Regenerate from the repository root:

```bash
.venv/bin/python scripts/generate_homewood500.py
```

After a separately chosen and completed run, count its names:

```bash
.venv/bin/python scripts/analyze_dining_names.py runs/YOUR_RUN --out runs/YOUR_RUN/dining_names.json
```

For a real comparison, keep the same population, schedules, model, and contact rules across the primary and information-control scenarios. Repeat with several seeds and inspect each run separately before aggregating. Mock output checks software wiring; it is not evidence of confusion or alignment.
