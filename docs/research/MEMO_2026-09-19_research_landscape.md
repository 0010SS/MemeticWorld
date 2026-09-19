# MemeWorld: Research Landscape, Scientific Positioning, and Design Proposal
Research and design discussion memo (prepared 19 September 2026; code snapshot reviewed: 7f4ed0c).
Audience: project collaborators. Agreed constraint: all participants inside the simulated world are autonomous agents; there is no human player.
Status: research synthesis and proposed design; not claims of implemented functionality or results.

> NOTE (lead): this copy was pasted by the user and is TRUNCATED after §8.1. The remaining sections
> (agent architecture and cultural mechanisms, measuring meaning, experimental program, validity,
> roadmap, pushback, alternatives, decisions, reading list) were not provided.
>
> DECISION (user, 2026-09-19):
> - The memo's research question is the primary first study.
> - The v2 bottleneck factorial (NEED / LINK / WORDING) becomes calibration and a secondary study.
> - Compute: Haiku agents and a Sonnet observer, both via the claude CLI.

## 1. Recommended research position
MemeWorld should become a controlled research environment for studying how socially transmitted meanings are
preserved, reinterpreted, and contested as a community's circumstances and membership change.

Central question: **When a community's circumstances change, how do its existing ideas acquire new meanings, and
when does cultural memory help or obstruct that adaptation?**

The strongest scientific opportunity is the combination of:
- consequential interaction in a persistent world;
- measurement of meaning through interpretation and behavior;
- interventions on transmission, memory, and environmental conditions;
- replicated evidence about the mechanisms that produce semantic change.

Excluding human players is appropriate for controlled experiments. It is an experimental constraint, not a
standalone novelty claim. The literature already includes emergent conventions, language transmission, shared
artifacts, resource-constrained agent ecologies, and prolonged simulations exhibiting local vocabulary. A
convincing contribution needs to answer a narrower question than whether agents can develop culture.

**Recommended first study:** investigate whether persistent shared records preserve useful meanings across
participant turnover, and whether those same records slow reinterpretation after environmental change. This
connects to the project's emphasis on lossy memory while introducing a clear long-horizon question.

Several close papers are recent preprints; distinguish their claims from replicated results.

## 2. Concepts and scope
### 2.1 A meme's expression and its meaning are different
An idea can survive while its wording changes; a phrase can remain stable while its interpretation changes.

| Phenomenon | Example | What it establishes |
|---|---|---|
| Diffusion | More agents repeat a phrase | Spread of a form or practice |
| Formal variation | A phrase becomes shorter or is paraphrased | Change in expression |
| Semantic broadening | A repair term starts applying to scheduling problems | Expansion of applicability |
| Semantic narrowing | A general warning comes to denote one specific failure | Restriction of applicability |
| Local specialization | Two groups infer different things from the same phrase | Community-dependent interpretation |
| Pragmatic change | A description becomes a warning, joke, excuse, or instruction | Change in communicative function |
| Normative change | A label comes to license an action or impose an obligation | Change in social consequences |
| Cumulative improvement | Later cohorts inherit and improve a useful procedure | Accumulating capability (with controls) |

These outcomes should not be collapsed into one success score.

### 2.2 Working definition
A meme is a socially transmitted pattern of interpretation or action, expressed through language or artifacts,
whose continuity can be investigated through exposure, reuse, and observable consequences. This is an analytical
definition and must not become a designer-supplied inventory inside agent state. First study: recurring
expressions and procedural ideas used to coordinate shared work. Narratives, rituals and ideologies come later.

### 2.3 Evolution does not automatically imply improvement
Possible outcomes include useful adaptation, arbitrary convention, fragmentation, forgetting, or maladaptive
persistence. Cumulative improvement needs evidence that socially inherited modifications support increasing
capability across learners or cohorts. Accumulating text or vocabulary is not enough. Distinguish retention from
reconstruction (cultural attraction theory: transformations during transmission plus selection; Claidière &
Sperber 2007).

### 2.4 Artificial societies are the immediate object of study
Pretrained LLMs contain human cultural knowledge, so local innovations may reflect interactions among pretrained
associations, prompts, memory and environment. Claims about human cultural evolution would need external
validation.

## 3. What the existing project does (summary)
Python simulation, FastAPI and a browser observer UI, YAML configuration, JSONL logs; reuses Generative Agents
memory, retrieval and reflection.
- Default: 8 campus agents, 3 simulated days, 15-minute ticks.
- Loop: world events → partial perception → lossy memory → retrieval and reflection → conversation or reaction →
  other agents' observations and memories.
- Four hidden event families.
- The observer finds recurring expressions, spread, context similarity, private meaning probes, and alignment
  with the hidden categories.

Foundations worth keeping:
- the world/agent/observer separation;
- partial perception;
- explicit memory transformations;
- exposure logging;
- no predefined meme inventory;
- configurable mechanisms;
- run logs and replay.

Omniscient versus information-asymmetric simulations differ substantially (Zhou et al. 2024).

### 3.3 Gaps relative to the clarified objective
| Current design | Limitation | Proposed change |
|---|---|---|
| Fixed hidden event categories | Mainly tests discovery of the experimenter's taxonomy | Retain as calibration; add situations whose practical significance changes |
| Events mainly supply conversational material | Shared terminology may provide little instrumental benefit | Give agents consequential tasks requiring information exchange |
| Short default horizon | Does not establish inheritance across replacement of memories or participants | Define horizons through task history, turnover and environmental change |
| Short conversations | Limits clarification and negotiation | Make communication budgets and repair opportunities experimental variables |
| Lexical candidate discovery | Misses ideas surviving through paraphrase or a change of medium | Add inference-, procedure- and artifact-level analysis |
| Context-similarity measures | Can confuse topic change with meaning change | Matched behavioral and interpretation probes |
| Final-memory probes | Weak evidence about intermediate meanings | Temporal checkpoints; probe isolated copies |
| Mood-oriented social reward | Does not establish the practical value of a convention | Observable task consequences; study mood separately |

The reported finding (no durable conventions for hidden categories in 3 days; one local analogy spread) may be
due to limited practical need. That is a hypothesis, not an established explanation. The run artifacts were not
available for audit.

## 4. Research landscape (condensed)
- Classical artificial language and culture: Steels 1995; Axelrod 1997.
- Human iterated learning: Kirby, Cornish & Smith 2008.
- Transmission perturbation and meaning: Brochhagen & Franke 2017.
- Neural emergent communication: Lazaridou et al.
- Embodied cultural transmission: Bhoopchand et al. 2023.
- Melting Pot: Leibo et al. 2021.
- Generative Agents: Park et al. 2023.
- Concordia: Vezhnevets et al. 2023. SOTOPIA: Zhou et al.
- Project Sid 2024; AgentSociety 2025; OASIS 2024.
- LLM transmission biases: Acerbi & Stubbersfield, PNAS 2023.
- LLM cultural dynamics and telephone-game attractors: Perez et al. 2024; Perez et al., ICLR 2025.
- LLM naming conventions and committed minorities: Ashery et al., Science Advances 2025.
- Referential games and iterated learning of categories: Kouwenhoven et al., COLING 2025; Imel & Zaslavsky, ICLR 2026.
- Collective invention and dynamic connectivity: Nisioti et al. 2024.
- Socially acquired significance of goods: Cross et al. 2026 (preprint).

Implications:
- Communication should matter for an observable task.
- Distinguish transmission from independent rediscovery.
- Participant replacement tests inheritance.
- Memory and network structure are mechanisms to manipulate, not novelties.
- Model priors, prompts and affordances all contribute.
- A focused system can contribute through strong measurement.

## 5. Closest recent competitors
- **TerraLingua** (March 2026 preprint): persistent ecology, resources, lifespans, reproduction, editable artifacts
  that outlive their creators, and an external AI anthropologist. Our distinction: a focused, replicated
  experiment on how particular interpretations change after controlled environmental change.
- **GlossoGen** (September 2026 preprint): language evolution under information asymmetry and communication
  pressure (the SaveVeyru scenario), with task success, structure, productivity and newcomer acquisition;
  explicit discussion matters. Lesson: allow ordinary clarification and negotiation, and treat "without explicit
  agreement" as a separate experiment. Our distinction: how an established expression changes applicability,
  implications or social function as the ecology changes.
- **Emergence World** (September 2026 preprint): long-running societies with tools, memory and institutions;
  local metaphors; an LLM assessment of outsider understandability; one trajectory per configuration. Our
  distinction: repeated controlled experiments linking a semantic transition to a mechanism, with matched
  interpretation tests and interventions.

## 6. Potential contributions (provisional)
| Candidate | Assessment |
|---|---|
| An agent-only world | Established design choice |
| More agents or a longer runtime | Weak alone |
| Emergent slang or shared names | Extensive precedent |
| Lossy memory affecting transmission | Established basis; isolate a specific effect |
| Persistent artifacts and inheritance | Close existing systems; needs a narrower causal question |
| Behavioral measurement of semantic transitions | Promising methodological contribution |
| Stable expressions repurposed after ecological change | Promising focused question |
| External memory preserving outdated interpretations | Promising mechanism question |
| Shared forms, different local meanings | Promising extension |
| Ideas moving from stories to rules to procedures | Ambitious extension |

Recommended package:
1. a small persistent environment in which established interpretations can become less useful as conditions
   change;
2. a protocol for measuring an expression's meaning at multiple times without contaminating the running society;
3. controlled evidence about how shared records affect semantic adaptation and inheritance;
4. an auditable dataset linking observations, messages, artifacts, decisions and outcomes.

## 7. Research questions and competing hypotheses
- **RQ1: When does an established expression acquire a different shared interpretation?**
  - Mechanisms: successful reuse in a new situation, misunderstanding followed by repair, influential exemplars,
    selective memory, explicit negotiation.
  - Observation: the form persists while its inferred consequences change on matched test situations.
  - Competing explanations: the topic distribution changed; different speakers now dominate; the phrase is copied
    without interpretation; shared pretrained association.
- **RQ2: How do shared records affect continuity and adaptation?**
  - Continuity: records improve acquisition and retention across turnover.
  - Inertia: records preserve an old interpretation after its practical basis changes.
  - Revision: records with accessible revision history help agents diagnose change.
  - Any of these may depend on retrieval, source authority, revisability and feedback frequency.
- **RQ3: How much disagreement can a useful convention tolerate?**
  - Groups may attach different implications to a shared expression and still coordinate, until a task exposes
    the difference.
  - Competing explanations: the groups are discussing different subjects; the expression has no effect on
    coordination.
- **RQ4: Does history leave an effect once present conditions are matched?**
  - Compare agents or communities in identical test situations after different histories.
  - Historical dependence: interpretations stay different even after material conditions return to an earlier
    state.
  - Alternative: interpretations track current conditions.
  - Control resources, goals and test information.

## 8. Recommended environment and tasks
### 8.1 Setting: an agent-run campus cooperative or workshop
Keep the campus as the spatial and social setting. Agents jointly operate a workshop or research cooperative with
shared equipment, projects, inspections and records, which gives ideas a reason to circulate and makes inaccurate
interpretations matter.

| Component | Initial design |
|---|---|
| Population | About 8-12 agents (a starting recommendation) |
| Social organization | A few groups with different responsibilities and information |
| Work | Maintain equipment, complete projects, allocate shared resources, inspect results |
| Resources | Equipment time, materials, project capacity, attention |
| Persistent records | Repair notes, incident reports, project boards, procedures, agreements |
| Communication | Direct conversation, local overhearing, messages, shared records |
| Turnover | Scheduled departures, newcomers, or reassignment |
| Environmental change | Changed material properties, failure mechanisms, demand, or task requirements |
| Participants | Agents only |

(Memo truncated here.)
