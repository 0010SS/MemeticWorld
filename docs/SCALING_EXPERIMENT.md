# Scaling MemeticWorld without making it hard to understand

September 19, 2026. Proposed plan based on the current local code; these expansions have not been implemented or tested in this task.

## Start here

**Try 12 agents in the existing campus first. Keep the same four action types and four event families. Expand one thing at a time.**

The question stays simple: **when agents experience and discuss everyday incidents, do they develop expressions that other agents reuse with a similar meaning?** A larger world is useful when it gives you a clearer comparison or a new interaction to inspect.

| What to expand | Current implementation | Small next step | Effort |
|---|---|---|---|
| Agents | 8 profiles | 12, then optionally 16 | Low–medium: profiles, relationships, schedules |
| Locations | 8 buildings/outdoor areas, each with rooms | One extra meeting room; later one new building | Medium: world, schedules, map |
| Actions | `CONTINUE`, `REACT`, `TALK`, `MOVE` | Richer everyday activities using these same actions | Low for descriptions; higher for new mechanics |
| Events | 4 hidden families, 33 scenario templates | More everyday examples within an existing family | Low–medium; changing frequency is config-only |

**Example of a readable comparison:** run the current eight students, then add four students with ordinary campus routines. Keep the model, duration, event rate, and action menu unchanged. Compare how widely expressions are reused and inspect their actual conversations.

## 1. Add more agents: 8 → 12 → 16

**Add people with a reason to meet someone.** Four isolated profiles will increase population without necessarily increasing useful interaction.

For the first four newcomers, use short profiles: two or three traits, one main interest, a routine, and a few relationships. For example, add a classmate, a lab partner, a club member, and a dorm neighbor. Give each one familiar contact and one weaker connection to an existing student. Let their routines overlap with those contacts.

Keep the original eight profiles and schedules unchanged for the first comparison. These new identities also change the community's composition, so describe the result as **this population expansion**, rather than a universal effect of agent count.

### What to edit

1. Copy [homewood8.yaml](../configs/population/homewood8.yaml) into a new `configs/population/homewood12.yaml`.
2. Append four complete profiles with unique IDs and names. Follow the existing fields and use existing locations, rooms, and sprites.
3. Add their relationships and memberships to the file's existing `relationships` and `groups` sections.
4. Create a separate experiment config with these overrides:

```yaml
# Proposed configs/scale_agents12.yaml; create the population file first.
extends: configs/baseline.yaml
run_name: scale_agents12
population: configs/population/homewood12.yaml
population_size: 12
```

**Changing `population_size` alone does not create agents.** The [loader](../backend/agents/profile.py) only selects entries already present in the population file.

Ordinary conversations require the **same location, same room, and overlapping time**. A group label alone does not create a meeting. Existing map rooms have roughly 8–12 standing positions; larger gatherings can overlap visually.

Move to 16 only after the 12-agent run is affordable and you can follow a few example conversations comfortably. Keep the existing conversation settings initially; their daily limit is not a strict cap on every conversation pathway, so check actual counts.

## 2. Add more locations: give each place one purpose

The existing locations are Dorm, Dining Hall, Classroom, Library, Research Lab, Gym, Cafe, and Quad. You already have enough variety for a first population expansion.

**First try one extra room inside an existing building**, such as a Club Room in the Dorm. A room is called an `arena` in the code. Schedule a few students there at the same time, replacing an existing social time slot rather than lengthening their day.

Later, add **Student Center → Common Room** as a ninth location. Its simple purpose: give students from different existing groups a shared meeting opportunity.

### What to edit

- [world.py](../backend/simulation/world.py): add a room to `ARENAS`. For a new building, also update `WORLD_GRAPH`, reciprocal connections, `HOMEWOOD_LABELS`, and `MAP_POS`.
- Population YAML: put visits in routines with the exact location and room names. An unused location changes little.
- [build_homewood_map.py](../scripts/build_homewood_map.py): add room/building geometry, entrances, walking space, and standing positions; regenerate [homewood_map.json](../frontend/homewood_map.json). The current map needs these entries in addition to backend coordinates.

For a comparison, use the **same population** with and without the new meeting place. Hold other routines steady except for the explicitly changed visit. This tests the new meeting arrangement; it does not isolate architecture from scheduling.

Keep the rest of the map fixed. Backend movement is currently semantic, so a longer drawn path does not establish a longer simulated travel time.

## 3. Add more actions: start with richer activities

You can make daily life richer while keeping the action menu small:

| Existing action | What it currently supports |
|---|---|
| `CONTINUE` | Carry on with the scheduled activity |
| `REACT` | A brief spoken reaction that others may hear |
| `TALK` | A conversation with a nearby person |
| `MOVE` | A temporary change of destination |

Add routine descriptions such as **working through a problem together**, **waiting for coffee**, or **practicing a club presentation** in the population YAML. Let questions, explanations, disagreements, and retelling occur within `TALK`.

These descriptions provide context; they do not implement task success, object manipulation, or learning mechanics. Likewise, a silent `REACT` currently produces no executed physical effect.

**Optional later addition: `HELP`.** Keep its meaning concrete: help a nearby agent with one unresolved task, changing that task from `unresolved` to `resolved`. This requires task state and validation, updates to the [reaction prompt](../backend/prompts/react_v1.txt), [planner](../backend/agents/planner.py), [engine](../backend/simulation/engine.py), and an action log. Adding the word to a prompt is insufficient. The current language analyzer would also need a separate measure if you want to study helping behavior itself.

For now, keep the four-action menu. Do not instruct agents to invent slang or spread phrases; that would change the emergence question.

## 4. Create more events: more examples before more categories

The simulator already has four hidden structures:

| Family | Simple meaning |
|---|---|
| E1 | A small mistake leads to several failures |
| E2 | Two mistakes cancel each other out |
| E3 | Unrelated people independently do the same unusual thing |
| E4 | An apparent failure leads to a useful outcome |

Keep these categories stable initially. Add variety through ordinary objects, activities, and settings in [latent_events.py](../backend/simulation/latent_events.py).

**Proposed E4 example:** a printer fails; a student switches to explaining their work on a whiteboard; they notice and correct an error. Write a few observable steps. Keep private information restricted to the relevant participants and check the resulting observations. Do not give the incident a nickname or tell every bystander its hidden causal explanation.

The smallest extension is to expand an existing template's `slots`. Next, add one ordinary scenario and one different **held-out scenario** within the same family. Held-out means an example saved for later evaluation: by default, day 3 uses those reserved templates. Keep that split intact so you can inspect reuse in a new situation.

### More frequent events are a separate experiment

Existing configs already provide:

| Config | `latent_events.event_rate` |
|---|---:|
| [no_events.yaml](../configs/no_events.yaml) | 0.00 |
| [baseline.yaml](../configs/baseline.yaml), via defaults | 0.12 |
| [event_rich.yaml](../configs/event_rich.yaml) | 0.25 |

The rate is a **world-wide probability of an event-start attempt per tick**, not an event count per student. Actual starts may be fewer when participants are busy or a scenario cannot fit before the day ends.

Compare frequency while holding templates fixed; compare new templates while holding frequency fixed. More events can also interrupt routines because scripted events move participants. More agents at the same global rate do not automatically receive the same event opportunity per person. Record actual events and conversations when interpreting either change.

Keep `generator: template` for an editable, inspectable event bank. Exact-time manual event scheduling would require additional implementation.

## Optional: comparing results without reading every conversation

**Choose one expansion and compare it with the baseline.** For example: eight versus twelve agents, or event rate 0.12 versus 0.25. Save combining expansions for after you understand each individual change.

Use the same model, memory settings, prompts outside the chosen change, analyzer model, and three-day duration. Rerun the baseline with the same code version as its comparison; the repository [decision log](DECISIONS.md) records earlier event-wording and analysis corrections.

Start with three fresh runs per condition as a **small exploratory pilot**, then decide whether more repeats are worth the cost. This is not a statistical sufficiency rule. Reuse a seed list such as 42, 43, 44, but do not assume matching seeds produce identical event sequences after changing the world or population; fresh LLM outputs can also differ.

Use a short result card:

| Measure | What to show |
|---|---|
| Reuse reach | Distinct users / total agents for an expression; 6/12 is more informative than “6 users” alone |
| Continued use | Whether it appears again in a later conversation, with the actual message |
| Meaning in context | A few uses, including a held-out incident if available, judged as similar / different / unclear |
| Interaction and cost | Actual events, conversations per agent, LLM calls, and elapsed time |

Inspect at most three candidate expressions per run using the same selection rule across conditions. Mark copied world wording and ordinary repeated language separately. If nothing qualifies, record **no clear convention observed**. Automated candidate counts are not confirmed shared meanings.

The existing `compare` command helps summarize runs; the short result card above still needs a small manual review.

> Reading the existing output: its adoption summary can include ordinary candidates when no convention is classified. Check the actual messages before calling something a convention.

For the demo, show **one expression, first observed use, actual listeners, and later reuse**. Keep the complete logs available behind that example. Exposure establishes a possible transmission path; it does not by itself prove influence or shared meaning.

## Before spending on larger runs

- Check unique agent IDs, valid room names, scheduled encounters, and map crowding.
- Use the mock backend for a short structural check; mock language is not evidence of emergence.
- After adding executable actions or event templates, run the existing invariant checks and inspect sample agent-visible observations for hidden labels or scripted catchphrases.
- Measure one small real run before scaling further. More workers may improve speed, but do not remove the work or cost of extra LLM calls.
- Save each configuration and its run logs separately. Keep expansion ideas as proposals until you have inspected the results.
