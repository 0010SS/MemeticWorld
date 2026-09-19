# MemeWorld: Research Landscape, Scientific Positioning, and Design Proposal

**Research and design discussion memo**

- **Prepared:** 19 September 2026
- **Project:** MemeWorld, in the MemeticWorld repository
- **Code snapshot reviewed:** 7f4ed0cf0f46bef1bf9c78b005ebddd88eb9f3ec
- **Audience:** Project collaborators and research colleagues
- **Agreed constraint:** All participants inside the simulated world are autonomous agents; there is no human player.
- **Document status:** Research synthesis and proposed design. Proposed mechanisms, experiments, and contributions are not claims of implemented functionality or demonstrated results.
- **Scope of this deliverable:** Documentation only. Creating this memo does not modify the simulation or implement the recommendations.

## Contents

1. [Recommended research position](#1-recommended-research-position)
2. [Concepts and scope](#2-concepts-and-scope)
3. [What the existing project does](#3-what-the-existing-project-does)
4. [Research landscape](#4-research-landscape)
5. [Closest recent competitors](#5-closest-recent-competitors)
6. [Potential contributions and novelty assessment](#6-potential-contributions-and-novelty-assessment)
7. [Research questions and competing hypotheses](#7-research-questions-and-competing-hypotheses)
8. [Recommended environment and tasks](#8-recommended-environment-and-tasks)
9. [Ontology and theoretical inspirations](#9-ontology-and-theoretical-inspirations)
10. [Agent architecture and cultural mechanisms](#10-agent-architecture-and-cultural-mechanisms)
11. [Measuring meaning and its evolution](#11-measuring-meaning-and-its-evolution)
12. [Experimental program](#12-experimental-program)
13. [Validity, reproducibility, and engineering priorities](#13-validity-reproducibility-and-engineering-priorities)
14. [Roadmap and decision gates](#14-roadmap-and-decision-gates)
15. [Pushback, failure modes, and scope limits](#15-pushback-failure-modes-and-scope-limits)
16. [Alternative settings and later extensions](#16-alternative-settings-and-later-extensions)
17. [Decisions for collaborators](#17-decisions-for-collaborators)
18. [Suggested project description](#18-suggested-project-description)
19. [Reading list](#19-reading-list)

## 1. Recommended research position

MemeWorld should become a controlled research environment for studying **how socially transmitted meanings are preserved, reinterpreted, and contested as a community's circumstances and membership change**.

The central question is:

> When a community's circumstances change, how do its existing ideas acquire new meanings, and when does cultural memory help or obstruct that adaptation?

The strongest scientific opportunity is the combination of:

1. Consequential interaction in a persistent world.
2. Measurement of meaning through interpretation and behavior.
3. Interventions on transmission, memory, and environmental conditions.
4. Replicated evidence about the mechanisms that produce semantic change.

The decision to exclude human players is appropriate for controlled experiments. It reduces variation from live human intervention and makes automated replication easier. It is an experimental constraint, not a standalone novelty claim.

The literature already includes emergent conventions, language transmission, shared artifacts, resource-constrained agent ecologies, and prolonged simulations exhibiting local vocabulary. A convincing contribution therefore needs to answer a narrower question than whether agents can develop culture.

**Recommended first study:** Investigate whether persistent shared records preserve useful meanings across participant turnover, and whether those same records slow reinterpretation after environmental change.

This connects directly to the project's existing emphasis on lossy memory while introducing a clear long-horizon question.

### Evidence and novelty boundaries

This memo draws on the earlier repository review and a focused review of primary research available through 19 September 2026. It covers major adjacent traditions and the closest systems identified during that review. It is not an exhaustive systematic review, a reproduction of the cited experiments, or proof of priority.

Several especially close papers are recent preprints. Their relevance to positioning is substantial, but their claims should be distinguished from independently replicated results. Literature descriptions below summarize the authors' reported methods and findings; proposed gaps and project recommendations are this memo's assessment.

## 2. Concepts and scope

### 2.1 A meme's expression and its meaning are different

The project is motivated by ideas changing over time. Repeated expressions are observable evidence, but an expression is not a complete representation of an idea.

An idea can survive while its wording changes. A phrase can also remain stable while its interpretation changes.

| Phenomenon | Example | What it establishes |
|---|---|---|
| Diffusion | More agents repeat a phrase | Spread of a form or practice |
| Formal variation | A phrase becomes shorter or is paraphrased | Change in expression |
| Semantic broadening | A repair term starts applying to scheduling problems | Expansion of applicability |
| Semantic narrowing | A general warning comes to denote one specific failure | Restriction of applicability |
| Local specialization | Two groups infer different things from the same phrase | Community-dependent interpretation |
| Pragmatic change | A description becomes a warning, joke, excuse, or instruction | Change in communicative function |
| Normative change | A label comes to license an action or impose an obligation | Change in social consequences |
| Cumulative improvement | Later cohorts inherit and improve a useful procedure | Accumulating capability, if appropriate controls support it |

These outcomes should not be collapsed into one success score.

### 2.2 Proposed working definition

For the initial research program:

> A meme is a socially transmitted pattern of interpretation or action, expressed through language or artifacts, whose continuity can be investigated through exposure, reuse, and observable consequences.

This is an analytical definition. It should not become a designer-supplied inventory of memes inside agent state.

The first study should focus on a tractable subset: **recurring expressions and procedural ideas used to coordinate shared work**. Narratives, rituals, and ideologies are possible later extensions.

### 2.3 Evolution does not automatically imply improvement

The system may produce useful adaptation, arbitrary convention, fragmentation, forgetting, or maladaptive persistence. These are all possible cultural outcomes.

A claim of cumulative improvement needs evidence that socially inherited modifications support increasing capability across learners or cohorts. Merely accumulating more text, artifacts, or distinctive vocabulary is insufficient.

Likewise, retention and reconstruction should be distinguished. Cultural attraction theory explicitly considers transformations during transmission alongside selection among alternatives. That is a useful foundation for a project built around memory compression and reinterpretation. [Claidiere & Sperber, 2007](https://www.dan.sperber.fr/wp-content/uploads/2007_claidiere_the-role-of-attraction-in-cultural-evolution.pdf).

### 2.4 Artificial societies are the immediate object of study

The most defensible initial claims concern the simulated agents and environments actually tested.

Pretrained LLMs already contain extensive human linguistic and cultural knowledge. An agent-only simulation does not begin from a culture-free state. Local innovations may reflect interactions among pretrained associations, prompts, memory, and the environment.

Human cultural theories can inspire hypotheses. Claims that the results explain human cultural evolution would require additional external validation.

## 3. What the existing project does

### 3.1 Current architecture

MemeWorld is a Python simulation with a FastAPI backend, a browser-based observer interface, YAML configuration, and JSONL experiment logs. It selectively reuses components from Generative Agents, including associative memory, retrieval machinery, and reflection prompts.

The default configuration contains eight campus agents, three simulated days, and 15-minute ticks. Agents follow routines, observe parts of events, encode memories, retrieve and reflect, move, converse, and sometimes overhear each other.

The current loop is approximately:

> World events -> partial perception -> lossy memory -> retrieval and reflection -> conversation or reaction -> other agents' observations and memories.

The world includes four hidden event families:

- A small failure that causes a cascade.
- Mistakes whose consequences cancel.
- Independent events that appear related.
- A failure that produces an unexpectedly beneficial outcome.

Agents encounter surface descriptions rather than the hidden family labels. The observer later searches for recurring expressions, estimates their spread and contextual similarities, asks private meaning questions, and compares expression use with the hidden categories.

Relevant implementation references:

- [Project overview and separation of responsibilities](../README.md)
- [Default configuration](../configs/default.yaml)
- [Simulation engine](../backend/simulation/engine.py)
- [Event templates](../backend/simulation/latent_events.py)
- [Memory and retrieval](../backend/memory/)
- [Observer analysis](../backend/analysis/)
- [Design decisions](DECISIONS.md)

### 3.2 Foundations worth keeping

| Existing feature | Research value |
|---|---|
| Separation of world, agents, and observer | Helps prevent evaluation labels from shaping the behavior being measured |
| Partial perception | Allows agents to have different experiences and information |
| Explicit memory transformations | Makes retention, abstraction, and forgetting available as experimental mechanisms |
| Exposure logging | Supports investigation of plausible transmission paths |
| No predefined meme inventory in agents | Avoids assuming the cultural units that the observer hopes to discover |
| Configurable mechanisms | Supports controlled comparisons once treatment definitions are isolated |
| Run logs and replay | Provides a foundation for auditability and later counterfactual experiments |

The separation between agent information and observer information is especially valuable. Research comparing omniscient and information-asymmetric LLM simulations has found substantial differences in apparent social success. [Zhou et al., 2024](https://arxiv.org/abs/2403.05020).

### 3.3 Gaps relative to the clarified research objective

| Current design | Limitation for studying semantic evolution | Proposed change |
|---|---|---|
| Fixed hidden event categories | Primarily tests discovery of the experimenter's taxonomy | Retain as calibration; add situations whose practical significance changes |
| Events primarily supply conversational material | Shared terminology may provide little instrumental benefit | Give agents consequential tasks requiring information exchange |
| Short default horizon | Does not establish inheritance across replacement of memories or participants | Define horizons through task history, turnover, and environmental changes |
| Short conversations | Limits some clarification and negotiation opportunities | Make communication budgets and repair opportunities explicit experimental variables |
| Lexical candidate discovery | Can miss an idea that survives through paraphrase or changes medium | Add inference, procedure, and artifact-level analysis |
| Context-similarity measures | Can confuse topic changes with meaning changes | Introduce matched behavioral and interpretation probes |
| Final-memory probes | Provide weak evidence about intermediate meanings | Save temporal checkpoints and probe isolated copies |
| Mood-oriented social reward | Does not directly establish practical value of a convention | Add observable task consequences; study mood separately |

The [repository report](../REPORT.md) describes a local analogy spreading but no durable conventions for the hidden categories over the reported three-day experiments. These are reported findings from that document. The original experiment artifacts were not available for an independent audit during the earlier review.

A plausible explanation is that the agents had limited practical need to develop such conventions. That is a hypothesis to investigate, not an established explanation for the reported outcome.

## 4. Research landscape

### 4.1 Major precedents and their implications

| Research tradition or system | What the cited work reports | Implication for MemeWorld |
|---|---|---|
| **Classical artificial language and culture** | Agents develop shared vocabularies, accommodate newcomers, and negotiate expanding meanings; cultural influence can create both local convergence and persistent differences between groups. [Steels, 1995](https://pubmed.ncbi.nlm.nih.gov/8925502/); [Axelrod, 1997](https://web.mit.edu/curhan/www/docs/Articles/15341_Readings/Culture_and_Identity/Axelrod-1997.pdf). | Emergent names, cultural diffusion, and group differentiation are established research topics. |
| **Human iterated learning** | Repeated transmission across learners can produce increasingly learnable and structured artificial languages without deliberate global design. [Kirby, Cornish & Smith, 2008](https://doi.org/10.1073/pnas.0707835105). | Memory bottlenecks and cohort turnover have strong precedents. |
| **Transmission perturbation and meaning** | Formal models investigate how observation errors affect vagueness, meaning deflation, and underspecified lexical meaning. [Brochhagen & Franke, 2017](https://escholarship.org/uc/item/098406s8). | Perception noise and semantic change already have a theoretical connection; the particular mechanism and evidence must be specified. |
| **Neural emergent communication** | Sender and receiver networks learn communication protocols through referential tasks with consequences for success. [Lazaridou et al.](https://arxiv.org/abs/1612.07182). | Grounding communication in tasks is established and should inform the baseline. |
| **Embodied cultural transmission** | Artificial agents acquire useful navigational knowledge through observing experts and retain it for subsequent behavior. [Bhoopchand et al., 2023](https://www.nature.com/articles/s41467-023-42875-2). | Inheritance should be assessed through useful retained behavior, not only text reuse. |
| **Social generalization benchmarks** | Melting Pot evaluates behavior with unfamiliar social partners across social dilemmas and coordination settings. [Leibo et al., 2021](https://proceedings.mlr.press/v139/leibo21a.html). | Newcomer and partner-transfer tests belong in the evaluation design. |
| **Generative Agents** | Memory, reflection, and planning support coherent individual behavior and emergent social coordination in a simulated town. [Park et al., 2023](https://arxiv.org/abs/2304.03442). | The memory-based social-agent architecture is an established foundation. |
| **Concordia and SOTOPIA** | Concordia supports grounded generative simulations; SOTOPIA evaluates interactive pursuit of social goals. [Vezhnevets et al., 2023](https://arxiv.org/abs/2312.03664); [Zhou et al.](https://arxiv.org/abs/2310.11667). | Grounded interaction and structured social evaluation should be treated as design foundations. |
| **Larger artificial societies** | Project Sid reports specialization, changing collective rules, and cultural transmission. AgentSociety and OASIS support larger social simulations, including diffusion and polarization. [Project Sid, 2024](https://arxiv.org/abs/2411.00114); [AgentSociety, 2025](https://arxiv.org/abs/2502.08691); [OASIS, 2024](https://arxiv.org/abs/2411.11581). | Population scale and the existence of agent societies provide limited distinctiveness on their own. |
| **LLM transmission biases** | Transmission-chain experiments find differential retention of content, including social and negative information. [Acerbi & Stubbersfield, PNAS 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10622889/). | Retelling is not neutral copying; pretrained content preferences must be considered. |
| **LLM cultural dynamics** | A population framework varies networks and information transformation. Telephone-game experiments identify characteristic changes and attractors in properties such as positivity, difficulty, and length. [Perez et al., 2024](https://arxiv.org/abs/2403.08882); [Perez et al., ICLR 2025](https://arxiv.org/abs/2407.04503). | Repeated paraphrasing and convergence of textual content already have direct precedents. |
| **LLM naming conventions** | Local coordination produces population-level conventions and collective biases; committed minorities can change established conventions. [Ashery et al., Science Advances 2025](https://www.lajello.com/papers/sciadv25emergent.pdf). | Spontaneous convention formation is insufficient as the sole contribution. |
| **LLM language structure and categorization** | Referential games produce structured vocabularies; iterated learning reshapes artificial color categories toward efficient compression, with model-dependent limitations. [Kouwenhoven et al., COLING 2025](https://aclanthology.org/2025.coling-main.667/); [Imel & Zaslavsky, ICLR 2026](https://arxiv.org/abs/2509.08093). | Category formation and learnable artificial language also have close precedents. |
| **Collective invention** | Groups of LLMs share discoveries in Little Alchemy 2; dynamic connectivity can outperform full connectivity in the tested setup. [Nisioti et al., 2024](https://arxiv.org/abs/2407.05377). | Social-network effects on innovation are already studied; connectivity needs a specific semantic hypothesis. |
| **Socially acquired significance** | A Concordia study examines how interaction changes status signaling and demand for goods, including synthetic goods. [Cross et al., 2026 preprint](https://arxiv.org/abs/2603.13220). | Social meaning extends beyond language, and some causal investigation of that broader phenomenon already exists. |

### 4.2 What this literature suggests

The field supports several useful design choices:

- Communication should matter for an observable task.
- Cultural transmission should be distinguished from independent rediscovery.
- Participant replacement is a meaningful test of inheritance.
- Memory and network structure are mechanisms to manipulate, not automatic novelties.
- Model priors, prompts, and environmental affordances can each contribute to observed behavior.
- A focused system can contribute through strong measurement and explanation without competing on population size.

The prospective gap is not that nobody has combined language and simulated society. It is a more specific question about **the causal dynamics of changing interpretations in persistent communities**, with sufficient controls to establish what changed and why.

## 5. Closest recent competitors

### 5.1 TerraLingua

**Status:** March 2026 preprint.

TerraLingua combines a persistent agent ecology, resource constraints, limited lifespans, reproduction, and editable artifacts that can survive their creators. An external AI anthropologist analyzes behavior, social organization, and artifact histories. The authors report norms, division of labor, governance attempts, and branching cultural artifacts.

**Overlap:** Resources, generations, shared records, artifact inheritance, and an external cultural observer.

**Possible distinction:** A focused, replicated experiment measuring how particular interpretations change after controlled environmental changes. That is a narrower objective than broad analysis of emergent social organization and artifact development.

**Design lesson:** Adding persistent documents or mortality is useful but does not itself establish priority. [TerraLingua](https://arxiv.org/abs/2603.16910).

### 5.2 GlossoGen

**Status:** September 2026 preprint.

GlossoGen studies language evolution in interactions where agents possess different information and face communication pressure. Its SaveVeyru scenario uses defined referents and evaluates task success, language structure, productivity, and acquisition by newcomers. Explicit discussion between rounds is an important factor in the reported results.

**Overlap:** Grounded communication, repeated interaction, emergent conventions, and newcomer learning.

**Possible distinction:** Investigating how an established expression changes applicability, implications, or social function as the ecology changes.

**Design lesson:** Allow ordinary clarification and negotiation. Whether conventions arise without explicit agreement should be a separate experiment, rather than an assumption about what counts as legitimate emergence. [GlossoGen](https://arxiv.org/abs/2609.01491).

### 5.3 Emergence World

**Status:** September 2026 preprint.

Emergence World studies prolonged operation of agent societies with tools, memory, and institutions. It reports shared metaphors and vocabulary acquiring local significance, and includes an LLM-based assessment of how understandable messages are to outsiders.

The authors state that each world configuration has one continuous trajectory, which establishes possible behaviors but limits inference about their recurrence.

**Overlap:** Long-running interaction, local vocabulary, semantic repurposing, and social consequences.

**Possible distinction:** Repeated controlled experiments that connect a semantic transition to a particular mechanism, with matched interpretation tests and interventions on transmission or memory.

**Design lesson:** Interesting histories need to be complemented by estimates of reliability and conditions of occurrence. [Emergence World](https://arxiv.org/abs/2609.17320).

## 6. Potential contributions and novelty assessment

The assessments in this table are provisional research-positioning judgments.

| Candidate contribution | Assessment | Evidence required |
|---|---|---|
| An agent-only world | Established design choice | Explain its experimental value without claiming novelty |
| More agents or longer runtime | Weak standalone contribution | Show a phenomenon requiring the added scale or temporal dependencies |
| Emergent slang or shared names | Extensive precedent | Identify a mechanism or capability beyond occurrence |
| Lossy memory influencing transmission | Established conceptual basis | Isolate a specific effect and distinguish it from model or prompt artifacts |
| Persistent artifacts and cultural inheritance | Close existing systems | Investigate a narrower causal interaction |
| Behavioral measurement of semantic transitions | Promising methodological contribution | Demonstrate construct validity, sensitivity, and false-positive controls |
| Stable expressions repurposed after ecological change | Promising focused question | Show matched-context changes, social transmission, and practical effects |
| External memory preserving outdated interpretations | Promising mechanism question | Manipulate memory access and environmental change across replicated worlds |
| Shared forms supporting different local meanings | Promising extension | Measure group-specific interpretations and cross-group coordination |
| Ideas moving from stories to rules to procedures | Ambitious extension | Establish continuity across media and verify actual effects of the resulting practice |

### Recommended contribution package

A coherent first paper could contribute:

1. A small persistent environment in which established interpretations can become less useful as conditions change.
2. A protocol for measuring expression meaning at multiple times without contaminating the running society.
3. Controlled evidence about how shared records affect semantic adaptation and inheritance.
4. An auditable dataset connecting observations, messages, artifacts, decisions, and outcomes.

This would be more defensible than a claim to have created the first artificial culture.

## 7. Research questions and competing hypotheses

### RQ1: When does an established expression acquire a different shared interpretation?

Potential mechanisms include successful reuse in a new situation, misunderstanding followed by repair, influential exemplars, selective memory, and explicit negotiation.

**Possible observation:** Agents continue using a form but change its inferred consequences on matched test situations.

**Competing explanations:** Topic distribution changed; different agents now dominate the conversation; the phrase is being copied without interpretation; or all agents independently draw on the same pretrained association.

### RQ2: How do shared records affect continuity and adaptation?

**Continuity hypothesis:** Records improve acquisition and retention across turnover.

**Inertia hypothesis:** Records preserve an old interpretation after its practical basis changes.

**Revision hypothesis:** Records with accessible revision history help agents diagnose why a convention changed and improve adaptation.

These effects may depend on retrieval, source authority, revisability, and the frequency of feedback. They need not appear in every setting.

### RQ3: How much disagreement can a useful convention tolerate?

Groups may attach different implications to a shared expression while coordinating adequately on some tasks.

**Possible observation:** Interpretation differs across groups, yet joint performance remains high until a task exposes the difference.

**Competing explanations:** Groups are merely discussing different subjects, or the shared expression has no effect on their coordination.

### RQ4: Does history leave an effect after present conditions are matched?

Compare agents or communities evaluated in identical test situations after different histories.

**Historical-dependence hypothesis:** Interpretations remain different even after material conditions return to an earlier state.

**Alternative:** Interpretations track current conditions with little durable influence from prior social history.

Different memories are part of the proposed mechanism. Other differences, such as resources, current goals, and test information, should be controlled when interpreting this comparison.

## 8. Recommended environment and tasks

### 8.1 Setting: an agent-run campus cooperative or workshop

The current campus can remain the spatial and social setting. Agents should jointly operate a workshop or research cooperative with shared equipment, projects, inspections, and records.

This provides a reason for ideas to circulate and for inaccurate interpretations to matter.

| Component | Initial design |
|---|---|
| Population | Approximately 8-12 agents; the number is a starting design recommendation, not an established optimum |
| Social organization | A few groups with different work responsibilities and information |
| Work | Maintain equipment, complete projects, allocate shared resources, inspect results |
| Resources | Equipment time, materials, project capacity, and attention |
| Persistent records | Repair notes, incident reports, project boards, procedures, agreements |
| Communication | Direct conversation, local overhearing, messages, and shared records |
| Turnover | Scheduled departures, newcomers, or reassignment |
| Environmental change | Changed material properties, failure mechanisms, demand, or task requirements |
| Participants | Agents only |

### 8.2 Give roles different information and constraints

Useful differences can come from what agents know and do:

- Operators observe performance and immediate symptoms.
- Maintainers inspect faults and repair history.
- Coordinators see deadlines, requests, and competing uses of equipment.
- Inspectors observe outcomes or conduct tests.

Roles should create interdependence. Avoid relying solely on personality biographies to produce diversity.

No role should automatically know the complete world state. The simulator should track which observations actually reached each participant.

### 8.3 Use a small set of consequential task families

**Diagnosis and repair.** Several faults produce overlapping symptoms. Different agents see different evidence. Choosing an inappropriate repair consumes resources or leaves the problem unresolved.

**Allocation and commitment.** Groups share equipment or materials. Their decisions create queues, obligations, delays, and opportunities for negotiation.

**Inspection and certification.** An agent must decide whether work is ready for another group to use. Local meanings of quality or completion can become consequential.

**Teaching and handover.** New participants inherit work from others. They can ask questions, observe demonstrations, or read records.

These task families should initially share a compact causal system. A large collection of unrelated scenarios would make it harder to determine what a cultural pattern generalizes over.

### 8.4 Actions should have verifiable effects

The world should support actions such as inspecting, repairing, testing, allocating, transferring, documenting, teaching, and requesting clarification.

Material outcomes should follow explicit simulation rules. Language models can choose actions and interpret observations; resource accounting and equipment transitions should remain inspectable.

A document declaring a repair successful must not make the repair successful. A proposed rule must be distinguished from an adopted rule and from actual enforcement.

If agents can create executable procedures, begin with compositions of permitted actions. Arbitrary tool invention would introduce a much larger engineering and measurement problem.

### 8.5 Give environmental changes a causal role

Examples include:

- A new material invalidates an old diagnostic cue.
- A repair method still works but now requires an additional inspection.
- Demand changes which allocation policy is useful.
- Two groups begin sharing equipment after developing separate practices.

The experimenter changes the environment, not the desired meaning. Agents may respond by revising an expression, replacing it, abandoning it, or continuing to use it unsuccessfully.

### 8.6 Illustrative cultural trajectory

The following is a hypothetical example, not a scripted target:

1. A failed repair involving a blue tray becomes a memorable episode.
2. Agents begin using “blue tray” for a kind of improvised repair.
3. The phrase broadens to mean proceeding without a usual inspection.
4. One group uses it approvingly; another uses it to criticize shortcuts.
5. New equipment makes the old shortcut unreliable.
6. A newcomer learns the phrase from an old record and applies it under the new conditions.

The research task is to establish which transitions occur, whether the interpretation is shared, how transmission happened, and what changed in behavior.

### 8.7 Define the horizon through dependencies

A convincing long-horizon experiment should include:

- Repeated opportunities to use and revise a practice.
- Enough intervening experience that immediate context is insufficient.
- Memory consolidation or forgetting.
- A change in circumstances.
- Participant replacement or a new cohort.
- Evaluation after those transitions.

Simulated calendar duration alone is not an adequate definition.

## 9. Ontology and theoretical inspirations

### 9.1 Separate material facts, beliefs, practices, and interpretations

| Layer | Entities or records | Design principle |
|---|---|---|
| Material world | Objects, quantities, equipment states, actions, outcomes | Record what happened independently of what agents say happened |
| Agent experience | Observations, reports, memories, uncertainty, plans | Preserve the possibility of incomplete or mistaken beliefs |
| Communicative activity | Speaker, audience, utterance, situation, response | Treat expression use as situated activity |
| Cultural artifacts | Stories, procedures, records, notices, agreements | Track versions, authorship, access, and use |
| Social arrangements | Commitments, permissions, expectations, sanctions, roles | Distinguish proposal, acceptance, and enforcement |
| Observer analysis | Candidate lineages, meaning hypotheses, supporting evidence | Keep analytical interpretations outside agent-accessible state |

The world can have a known causal structure without having one objectively correct community vocabulary.

An emerging distinction may cut across the designer's initial categories. A local term may also have several meanings, or a contested meaning. These should be representable outcomes.

### 9.2 Common ground and conversational repair

Clark and Brennan describe communication as collaborative activity that requires participants to establish sufficient shared understanding.

**Design implication:** Agents need opportunities to ask what another participant meant, give examples, demonstrate, correct, and acknowledge. These actions should consume some attention or time, but should not be prohibited by default.

**Research opportunity:** Test whether repair stabilizes meanings, enables reinterpretation, or reduces costly misunderstanding. The theory itself is established. [Clark & Brennan, 1991](https://web.stanford.edu/~clark/1990s/Clark%2C%20H.H.%20_%20Brennan%2C%20S.E.%20_Grounding%20in%20communication_%201991.pdf).

### 9.3 Boundary objects

Star and Griesemer analyze how shared objects can support cooperation among communities with different perspectives.

**Design implication:** A repair record, project board, or certification can remain recognizable across groups while serving different local purposes.

**Research opportunity:** Investigate when shared artifacts sustain coordination despite interpretive differences, and when those differences become consequential. Universal semantic convergence should not be the only success condition. [Star & Griesemer, 1989](https://criticalmanagement.uniud.it/fileadmin/user_upload/documents/Star__Griesemer_1989.pdf).

### 9.4 Institutional grammar

Crawford and Ostrom distinguish rules, norms, and shared strategies through structured descriptions of institutional statements.

**Design implication:** Track who a prescription concerns, what it asks or requires, when it applies, and what follows from violation. Keep those descriptions distinct from observed compliance.

**Research opportunity:** Measure whether an expression changes from describing an event to prescribing behavior, and whether that prescription is actually adopted. [Crawford & Ostrom, 1995](https://wiki.santafe.edu/images/a/ab/Crawfordandostrom1995.pdf).

### 9.5 Do not impose a universal meaning field

Avoid designing every cultural item around a single global mapping such as an expression equaling one hidden event class.

The observer can instead maintain:

- Related forms and paraphrases.
- Situations in which they are used.
- Group- and time-specific interpretation evidence.
- Proposed connections to procedures or artifacts.
- Confidence and alternative interpretations.

Agents may create their own glossaries or procedures. That is an observable act of cultural externalization, not a violation of the principle against designer-supplied meme state.

## 10. Agent architecture and cultural mechanisms

### 10.1 Minimal loop

The proposed loop is:

> Observe local conditions -> retrieve relevant experience -> interpret the current problem -> communicate or act -> receive consequences -> update memory and, optionally, shared records.

Keep frozen model parameters in the first study. This isolates changes carried through experience, memory, and artifacts. Parameter training could be studied later as a separate mechanism.

### 10.2 Distinguish mechanisms

| Mechanism | Example implementation | Measurement concern |
|---|---|---|
| Variation | Paraphrase, analogy, reinterpretation, procedural modification | Distinguish semantic changes from superficial wording |
| Selection | Reuse influenced by usefulness, source credibility, or local acceptance | Do not reward the observer's preferred meme directly |
| Retention | Memory, rehearsal, records, repeated practice | Measure which information remains accessible |
| Transmission | Conversation, teaching, observation, artifacts | Record actual access and opportunities to learn |
| Reconstruction | A learner infers a rule from examples | Compare inferred applicability and consequences |
| Recombination | A procedure combines parts of several earlier practices | Permit multiple ancestry hypotheses |
| Repair | Participants resolve a misunderstanding | Observe whether interpretation and behavior subsequently change |

### 10.3 Memory should retain epistemic distinctions

Where feasible, distinguish direct observation from hearsay, uncertain inference, and accepted procedure. Record provenance without requiring the agent to retain perfect access to it forever.

Compression can be a treatment, but it should be described precisely: what is removed, abstracted, or merged, and under what conditions.

Shared records should have explicit access rules, capacity or attention costs where relevant, and version history in the observer logs. Comparing records to oral transmission should account for differences in information availability and communication budget.

### 10.4 Social mechanisms should have observable foundations

Prestige may derive from visible competence or past success, rather than only a fixed status number. Trust can vary by source and domain. Conformity can affect what agents repeat without necessarily changing their private interpretation.

These are proposed options. They should be added only when they support a specified hypothesis.

## 11. Measuring meaning and its evolution

### 11.1 Operationalize meaning as a profile across situations

For an expression at a particular time, measure:

- Which situations the agent applies it to.
- What referent or category the agent selects.
- What consequence the agent infers.
- What action the agent chooses.
- How interpretation depends on speaker, audience, and community.

Conceptually:

> Meaning profile = distribution of interpretations and decisions for an expression across controlled contexts, conditional on an agent's cultural history.

This is a practical operationalization of selected aspects of meaning. It is not a complete philosophical account.

### 11.2 Preserve several types of evidence

| Claim | Stronger evidence | Insufficient evidence by itself |
|---|---|---|
| Social spread | Exposure precedes adoption and an exposure intervention changes adoption | The same phrase appears in several agents' outputs |
| Semantic change | Interpretations differ over time on matched test situations | Context embeddings move |
| Shared interpretation | Several agents show compatible inferences or useful complementary coordination | An observer writes a convincing definition |
| Behavioral relevance | Changing a message changes decisions in appropriate situations | Messages correlate with later actions |
| Inheritance | Newcomers acquire and use a practice after earlier participants leave | A document remains stored |
| Generalization | Appropriate use in genuinely unseen situations | Reuse on a template encountered earlier |
| Cumulative improvement | Later cohorts benefit from inherited modifications under suitable controls | More artifacts or more elaborate prose |

The distinction between correlation and communicative effect is established in emergent-communication evaluation. [Lowe et al., 2019](https://arxiv.org/abs/1903.05168).

### 11.3 Probe snapshots without influencing the society

At predefined checkpoints:

1. Save the state required to reconstruct each agent's perspective.
2. Create isolated copies for evaluation.
3. Present balanced, held-out situations.
4. Ask for referent choices, inferred implications, and actions.
5. Collect short explanations as supporting evidence.
6. Do not return probe outputs to the running agents.

Testing only final memories cannot reliably reconstruct what an expression meant earlier.

Probe contexts should control immediate information, goals, and payoffs. Otherwise a changed decision might reflect ordinary adaptation rather than a changed interpretation.

Useful message conditions include the original expression, an explicit paraphrase, an older interpretation, and an appropriately neutral comparison. Their purposes differ; not every condition needs to be used in every experiment.

### 11.4 Separate semantic change from confounds

| Potential confound | Control or diagnostic |
|---|---|
| Topic distribution changes | Compare interpretations on a fixed, balanced set of contexts |
| Different speakers dominate later | Track individuals and population composition separately |
| Word frequency changes | Use frequency-aware comparison and minimum evidence requirements |
| Independent rediscovery from model priors | Use agents without social exposure and unfamiliar world instances |
| Prompt or event wording supplies the phrase | Trace first appearances across all agent-visible inputs |
| Shared model tendencies | Replicate across model families and compare independent communities |
| Memory loss produces incoherence | Require stable shared uptake and examine consequences |
| Observer overinterpretation | Blind annotation, alternative judges, and auditable usage examples |
| Candidate selection favors apparent change | Separate discovery and evaluation data; freeze analysis before confirmatory runs |
| Action policy changes without lexical change | Combine behavioral tests with reference and inference tests |

Representation-based semantic-change measures can produce artifacts even in conditions designed to contain no change. Matched controls are therefore essential. [Dubossarsky et al., 2017](https://aclanthology.org/D17-1118.pdf).

Usage-pair annotation provides a complementary approach to judging whether expressions are used with related meanings. [Schlechtweg et al., 2018](https://arxiv.org/abs/1804.06517).

Human annotation of a small offline sample is compatible with having no human player. If human annotation is unavailable, report the limitation and strengthen objective task checks and judge-sensitivity analysis.

### 11.5 Treat transmission histories as evidence with uncertainty

Distinguish:

- Confirmed exposure.
- Information present in memory.
- Information retrieved for a decision.
- Inferred ancestry.
- Influence supported by an intervention.

A retrieved memory is evidence of availability, not proof of causation. A cultural item may combine several sources or be independently rediscovered.

The observer should therefore permit branching and merging histories and avoid presenting every inferred edge as a proven causal link.

### 11.6 Recommended outcomes

Report a small set of interpretable outcomes rather than one total culture score:

- Interpretation change on matched contexts.
- Within-group and between-group differences.
- Coordination success and misunderstanding costs.
- Acquisition by newcomers.
- Retention after turnover.
- Adaptation following environmental change.
- Dependence on particular records or communication paths.
- Failure, abandonment, and persistence of outdated interpretations.

## 12. Experimental program

### 12.1 Stage A: validate the environment and instruments

Before studying emergence:

- Confirm that the task is solvable with sufficiently informative communication.
- Confirm that different agents possess information useful to one another.
- Test whether the measurement system detects a deliberately introduced interpretation change.
- Include a condition in which interpretation is kept stable but topics or expression frequency change.
- Check that a seeded convention can be learned and used on held-out situations.

Seeded conventions are positive controls for measurement. They must not be counted as evidence of spontaneous emergence.

### 12.2 Stage B: establish an unseeded baseline

Allow agents to work without instructions to invent language, memes, or culture.

Inspect whether recurring forms and practices appear, whether they are used by multiple agents, and whether they affect decisions.

Predefine the discovery interval and stopping rule. A failure to develop a candidate within that interval is a result, not a reason to keep extending the same run until something interesting appears.

If no recurring practice appears, distinguish among:

- Inadequate task competence.
- Little practical need for communication.
- Excessively restrictive communication or memory.
- Sufficient ordinary language with no need for a local convention.
- Measurement failure.

### 12.3 Stage C: primary factorial experiment

Begin with a two-by-two design:

| Condition | Environmental change | Persistent shared records |
|---|---|---|
| A | No | No |
| B | No | Yes |
| C | Yes | No |
| D | Yes | Yes |

Use the same planned turnover schedule in all conditions.

The central estimand is the interaction between environmental change and records: whether access to inherited external memory changes semantic adaptation, retention, or task outcomes after the shift.

The record condition must specify what agents can write, revise, read, and transmit to newcomers. Account for its additional information and attention opportunities when interpreting effects.

Later, compare fixed records with editable records and visible revision history.

### 12.4 Stage D: mechanism tests

Possible follow-up interventions include:

- Removing access to a particular historical record.
- Providing a revised version of a record.
- Changing whether newcomers can ask experienced agents for clarification.
- Restricting or restoring a communication bridge between groups.
- Replaying an exposure with different wording while preserving relevant factual information.
- Returning the environment to earlier conditions and testing for persistent historical effects.

For an intervention after culture has formed, fork matched checkpoints where practical. Keep exogenous disturbances controlled while allowing real behavioral consequences to diverge.

### 12.5 Baselines answer different questions

| Baseline | Question |
|---|---|
| Individual learning without cultural inheritance | Is an effect dependent on social transmission? |
| Equivalent access to relevant raw observations | Does a benefit reflect interpretation or simply more information? |
| Transmission chain without consequential world interaction | What does the persistent task environment add? |
| Stable environment | Would comparable change occur without ecological pressure? |
| Restricted clarification | What role does repair play? |
| Artifacts only, teaching only, both, neither | Through which channels does inheritance occur? |
| Fresh agents from the same model | Could pretrained associations explain the result? |
| A simpler memory architecture | Is the proposed memory mechanism necessary? |

These comparisons should be prioritized by the research question. Running every possible ablation at once would make the initial project unnecessarily expensive.

### 12.6 Replication and inference

- Treat the independent world as the main experimental unit.
- Do not count thousands of connected utterances as thousands of independent replications.
- Use pilot variability to determine a defensible replication plan.
- Report effect sizes, uncertainty, and failure rates.
- Correct or explicitly account for multiple candidate and hypothesis tests.
- Separate exploratory discoveries from confirmatory analysis.
- Initially replicate with more than one model family in separate populations.
- Treat mixed-model populations as a later factor, because they introduce additional sources of variation.

Same-seed comparisons require care: a shared seed does not guarantee identical exogenous events if conditions consume randomness differently.

## 13. Validity, reproducibility, and engineering priorities

### 13.1 Repair issues that directly affect inference

The earlier code review identified the following research-relevant issues. They should be resolved and verified before using new runs for confirmatory claims.

| Issue | Scientific consequence | Relevant code |
|---|---|---|
| Event generation and social interaction share randomness | A treatment can unintentionally alter the events agents experience | [Simulation engine](../backend/simulation/engine.py) |
| Model exceptions can become ordinary output text | Failures can enter memory or reflection as apparent content | [LLM client](../backend/llm/client.py), [memory encoder](../backend/memory/encoder.py) |
| Lift uses incompatible weighting conventions | A high score can reflect the metric definition rather than category discrimination | [Evaluation](../backend/analysis/evaluation.py) |
| Final probes draw from the held-out bank also used during simulation | Evaluation is not guaranteed to test unseen experiences | [Probes](../backend/analysis/probes.py), [event scheduling](../backend/simulation/engine.py) |
| Probe answer parsing accepts the first A-D character | Ordinary response prefixes can be misread as the selected option | [Probes](../backend/analysis/probes.py) |
| Same-tick ordering can distort inferred transmission | Apparent inventors and inheritance edges can be wrong | [Run data](../backend/analysis/rundata.py), [transmission analysis](../backend/analysis/transmission.py) |
| Default observer embeddings are lexical hashes | Similarity should not be treated as a validated semantic measure | [Embeddings](../backend/llm/embeddings.py), [semantic analysis](../backend/analysis/semantics.py) |
| Some experimental presets change multiple parameters | Effects cannot be assigned to one mechanism without further controls | [Configuration presets](../configs/) |

Operational issues from the review also matter for sustained studies: the CLI backend's fixed /tmp working directory is unsuitable for the reviewed Windows setup, and the global Generative Agents compatibility bindings require care when isolating multiple experiments.

This memo does not claim those repairs have been implemented.

### 13.2 Make a run an auditable research object

A run should preserve:

- Code revision and dependency versions.
- Complete resolved configuration and population definition.
- World generator and evaluation-set versions.
- Exact prompts and model identifiers.
- Model responses, errors, retries, and fallback behavior.
- Separate randomness streams or a reproducible exogenous schedule.
- Ordered events, observations, exposures, and actions.
- Artifact contents, versions, access, and modifications.
- Agent checkpoints sufficient for temporal probes.
- Observer version and analysis settings.
- Cost and computational budget.

Replay of recorded outputs and generation of new counterfactual trajectories should be described separately. A changed intervention may require new model outputs; it cannot always reuse the original transcript.

### 13.3 Keep observer and agent changes separate

Changing an agent's retrieval representation changes the system under study. Changing the observer's representation changes the measurement instrument.

Give these separate configurations. Hold the observer fixed across a comparison and validate it on known examples and no-change controls.

### 13.4 Improve the research interface

Useful observer views would include:

- An expression's uses over time, with context.
- Interpretation evidence from different agents and groups.
- Artifact revision histories.
- Confirmed exposures and uncertain ancestry links.
- Decisions and outcomes associated with an expression.
- Side-by-side comparisons of experimental conditions.
- Failed or ambiguous candidate cases.

Every claimed transition should be traceable to underlying evidence. This is the most relevant form of interface polish for the research objective.

## 14. Roadmap and decision gates

| Phase | Deliverable | Decision gate |
|---|---|---|
| 0. Make current experiments trustworthy | Correct critical metrics, error handling, randomness, and evaluation separation | Controlled comparisons preserve their intended treatment |
| 1. Add consequential work | One compact task family with explicit world transitions and asymmetric information | Agents can solve the task, and communication matters |
| 2. Add temporal evaluation | Checkpoints, isolated probes, and validated meaning tests | Instruments distinguish known change from no-change controls |
| 3. Establish spontaneous behavior | Unseeded repeated work and documented candidate selection | Recurring practices have evidence of shared use or practical consequences |
| 4. Run the main study | Records-by-environment experiment with fixed turnover and replication | Effects can be estimated with uncertainty and alternative explanations assessed |
| 5. Test transfer and scope | New world instances, partners, and model families | Findings extend beyond one trajectory or configuration |
| 6. Extend the cultural scope | Group polysemy, institutionalization, or narrative transformation | Extension answers a new question without invalidating the measurement framework |

There is no justification yet for a precise runtime or monetary estimate. Use measured pilot call counts and token consumption to budget the study.

If resources are tight, reduce the number of mechanisms and task families before sacrificing replication of the central comparison.

## 15. Pushback, failure modes, and scope limits

### 15.1 “Agents invented slang” is an observation, not a sufficient explanation

The project needs evidence about interpretation, transmission, persistence, and consequences. Distinctive vocabulary can be an entry point into analysis without being the endpoint.

### 15.2 Do not define emergence so strictly that ordinary cultural mechanisms become forbidden

Agents should be allowed to agree on terminology, teach, write procedures, or maintain a glossary. The relevant constraint is that the experimenter does not predefine the target cultural outcome.

Explicit agreement, spontaneous reuse, and implicit learning can be compared as mechanisms.

### 15.3 More world detail does not automatically improve the experiment

Survival, reproduction, markets, governments, and religion each create new dependencies and confounds. Add them when a research question requires them.

A workshop with a small number of meaningful actions may produce stronger evidence than a visually elaborate town with weakly grounded consequences.

### 15.4 Do not optimize for maximum drift

Rapid change could reflect forgetting, instability, or measurement noise. Stable shared meanings may be highly functional.

Evaluate the relationship between continuity, adaptation, coordination, and error. Do not reward agents for the observer's semantic-change score.

### 15.5 Do not treat a stored artifact as an operating institution

A manifesto, policy, or procedure may be ignored. Its existence, interpretation, adoption, and enforcement are separate observations.

This distinction should be visible in the world state and analysis.

### 15.6 Avoid assuming model narratives reveal internal mechanisms

Explanations are observable outputs. They can support interpretation but do not substitute for behavioral evidence or interventions.

The same applies to an AI observer's confident account of a cultural lineage.

### 15.7 A null result can be useful

Examples include:

- Agents use ordinary language effectively without developing local conventions.
- Shared records improve continuity without increasing inertia.
- Apparent semantic drift disappears under matched-context probes.
- Newcomers fail to acquire a practice despite persistent artifacts.

These outcomes would constrain the theory and design. They should not be hidden by showcasing only successful trajectories.

### 15.8 Human-free participation does not imply human-free validation

There is no human player inside the world. Researchers still define experiments and may inspect or annotate outputs offline.

The simulation can be studied as an artificial system even if it does not closely reproduce human behavior. Human generalization should be a separately supported claim.

## 16. Alternative settings and later extensions

| Setting | Strength | Main limitation | Recommendation |
|---|---|---|---|
| Campus workshop or cooperative | Reuses the current setting; supports coordination, records, and turnover | Requires adding actual material consequences | Best initial extension |
| Synthetic scientific community | Makes explanatory ideas, competing theories, and evidence central | Harder to distinguish concept change from ordinary scientific learning | Strong alternative if ideas beyond phrases are the priority |
| Agent organization maintaining services | Operational meanings of completion, verification, and urgency have direct consequences | Familiar terminology and tools bring substantial pretrained associations | Useful applied extension |
| Survival ecology | Strong resource feedback and inheritance pressures | Close prior systems and many coupled mechanisms | Add only if survival is central to the hypothesis |
| Social-media world | Convenient diffusion and network experiments | Meaning may be weakly connected to nonlinguistic consequences | Useful comparison condition |

Later studies could examine:

- Oral stories becoming procedures or rules.
- Artifacts retaining meaning after all original authors leave.
- Brokers translating between communities.
- Practices that remain socially attractive despite reduced instrumental value.
- Competition between incompatible interpretations of a shared record.
- Whether recurrent misunderstandings lead to new distinctions or institutions.

Each would require its own account of what changes, what persists, and how continuity is established.

## 17. Decisions for collaborators

| Decision | Recommended starting position | Discussion question |
|---|---|---|
| Primary scientific object | Artificial cultural dynamics | Are human-culture claims an eventual goal or outside scope? |
| First cultural unit | Expressions and procedural ideas | Which important phenomena would this initial scope exclude? |
| Setting | Agent-run workshop within the campus | Does the group prefer coordination or theory formation as the central activity? |
| Main intervention | Environmental change crossed with shared records | Is cultural continuity versus adaptation the question the team most wants answered? |
| Horizon | Repeated tasks, memory change, and turnover | What dependence on history must the study demonstrate? |
| Agent learning | Frozen model with changing memory and artifacts | Is parameter learning necessary for the first claim? |
| Evaluation | Matched interpretation and action probes | What result would convince a skeptical reader that meaning changed? |
| Novelty claim | Mechanism and measurement | Which closest prior system provides the most useful comparison? |
| Research budget | Pilot-based and replication-focused | What computation can be committed to confirmatory runs? |

Before implementation, collaborators should agree on a one-page study specification containing the research question, treatment definitions, primary outcomes, controls, stopping rules, and criteria for interpreting a null result.

## 18. Suggested project description

> MemeWorld is a proposed experimental environment for studying semantic change in persistent societies of autonomous language-model agents. Agents collaborate on consequential tasks under partial information, maintain personal memories, and create shared records. As the environment and population change, the system tracks how expressions and procedural ideas are retained, reinterpreted, or abandoned. Controlled interventions and isolated behavioral probes distinguish changes in interpretation from changes in wording, topic, and population composition. The initial research program investigates how external cultural memory affects continuity across turnover and adaptation after environmental change.

This describes a proposed research direction. A paper reporting results should replace prospective language only after the corresponding experiments have been completed.

## 19. Reading list

The links below point to primary papers, author-hosted copies, or primary publication records. Dates indicate the cited work rather than webpage crawl dates. A venue is specified where it was verified; an arXiv link alone does not establish peer-reviewed publication status.

### Closest systems and immediate comparisons

1. Paolo et al. **TerraLingua: Emergence and Analysis of Open-endedness in LLM Ecologies.** 2026 preprint. [Paper](https://arxiv.org/abs/2603.16910).
2. Stengel-Eskin et al. **GlossoGen: Emergent Language in Complex Multi-Agent LLM Interactions.** 2026 preprint. [Paper](https://arxiv.org/abs/2609.01491).
3. Akkil et al. **Emergence World: Adversarial Stress-Testing of Long-Horizon Multi-Agent Systems.** 2026 preprint. [Paper](https://arxiv.org/abs/2609.17320).
4. Ashery, Aiello, and Baronchelli. **Emergent social conventions and collective bias in LLM populations.** Science Advances, 2025. [Author-hosted paper](https://www.lajello.com/papers/sciadv25emergent.pdf).
5. Perez et al. **When LLMs Play the Telephone Game: Cultural Attractors as Conceptual Tools to Evaluate LLMs in Multi-turn Settings.** ICLR 2025; arXiv version subsequently revised. [Paper](https://arxiv.org/abs/2407.04503).
6. Imel and Zaslavsky. **Evolution and compression in LLMs: On the emergence of human-aligned categorization.** ICLR 2026. [Paper](https://arxiv.org/abs/2509.08093).
7. Kouwenhoven, Peeperkorn, and Verhoef. **Searching for Structure: Investigating Emergent Communication with Large Language Models.** COLING 2025. [Paper](https://aclanthology.org/2025.coling-main.667/).
8. Cross et al. **A Generative Model of Conspicuous Consumption and Status Signaling.** 2026 preprint. [Paper](https://arxiv.org/abs/2603.13220).

### Cultural transmission and collective invention

9. Acerbi and Stubbersfield. **Large language models show human-like content biases in transmission chain experiments.** PNAS, 2023. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10622889/).
10. Perez et al. **Cultural evolution in populations of Large Language Models.** 2024. [Paper](https://arxiv.org/abs/2403.08882).
11. Nisioti et al. **Collective Innovation in Groups of Large Language Models.** 2024. [Paper](https://arxiv.org/abs/2407.05377).
12. Kirby, Cornish, and Smith. **Cumulative cultural evolution in the laboratory: An experimental approach to the origins of structure in human language.** PNAS, 2008. [Paper](https://doi.org/10.1073/pnas.0707835105).
13. Claidiere and Sperber. **The role of attraction in cultural evolution.** Journal of Cognition and Culture, 2007. [Author-hosted paper](https://www.dan.sperber.fr/wp-content/uploads/2007_claidiere_the-role-of-attraction-in-cultural-evolution.pdf).
14. Brochhagen and Franke. **Effects of transmission perturbation in the cultural evolution of language.** 2017. [Paper](https://escholarship.org/uc/item/098406s8).

### Simulation foundations

15. Park et al. **Generative Agents: Interactive Simulacra of Human Behavior.** 2023. [Paper](https://arxiv.org/abs/2304.03442).
16. Vezhnevets et al. **Generative agent-based modeling with actions grounded in physical, social, or digital space using Concordia.** 2023. [Paper](https://arxiv.org/abs/2312.03664).
17. Zhou et al. **SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents.** ICLR 2024. [Paper](https://arxiv.org/abs/2310.11667).
18. Altera.AL et al. **Project Sid: Many-agent simulations toward AI civilization.** 2024. [Paper](https://arxiv.org/abs/2411.00114).
19. Piao et al. **AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents Advances Understanding of Human Behaviors and Society.** 2025. [Paper](https://arxiv.org/abs/2502.08691).
20. Yang et al. **OASIS: Open Agent Social Interaction Simulations with One Million Agents.** 2024. [Paper](https://arxiv.org/abs/2411.11581).
21. Steels. **A self-organizing spatial vocabulary.** Artificial Life, 1995. [Primary publication record](https://pubmed.ncbi.nlm.nih.gov/8925502/).
22. Axelrod. **The Dissemination of Culture: A Model with Local Convergence and Global Polarization.** Journal of Conflict Resolution, 1997. [Paper](https://web.mit.edu/curhan/www/docs/Articles/15341_Readings/Culture_and_Identity/Axelrod-1997.pdf).
23. Lazaridou, Peysakhovich, and Baroni. **Multi-Agent Cooperation and the Emergence of (Natural) Language.** Initial preprint 2016. [Paper](https://arxiv.org/abs/1612.07182).
24. Bhoopchand et al. **Learning few-shot imitation as cultural transmission.** Nature Communications, 2023. [Paper](https://www.nature.com/articles/s41467-023-42875-2).
25. Leibo et al. **Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot.** ICML 2021. [Paper](https://proceedings.mlr.press/v139/leibo21a.html).

### Measurement and ontology

26. Lowe et al. **On the Pitfalls of Measuring Emergent Communication.** AAMAS 2019. [Paper](https://arxiv.org/abs/1903.05168).
27. Dubossarsky, Grossman, and Weinshall. **Outta Control: Laws of Semantic Change and Inherent Biases in Word Representation Models.** EMNLP 2017. [Paper](https://aclanthology.org/D17-1118.pdf).
28. Schlechtweg, Schulte im Walde, and Eckmann. **Diachronic Usage Relatedness (DURel): A Framework for the Annotation of Lexical Semantic Change.** NAACL 2018. [Paper](https://arxiv.org/abs/1804.06517).
29. Zhou et al. **Is this the real life? Is this just fantasy? The Misleading Success of Simulating Social Interactions With LLMs.** 2024. [Paper](https://arxiv.org/abs/2403.05020).
30. Clark and Brennan. **Grounding in communication.** 1991. [Author-hosted chapter](https://web.stanford.edu/~clark/1990s/Clark%2C%20H.H.%20_%20Brennan%2C%20S.E.%20_Grounding%20in%20communication_%201991.pdf).
31. Star and Griesemer. **Institutional Ecology, 'Translations' and Boundary Objects: Amateurs and Professionals in Berkeley's Museum of Vertebrate Zoology, 1907-39.** Social Studies of Science, 1989. [Paper](https://criticalmanagement.uniud.it/fileadmin/user_upload/documents/Star__Griesemer_1989.pdf).
32. Crawford and Ostrom. **A Grammar of Institutions.** American Political Science Review, 1995. [Paper](https://wiki.santafe.edu/images/a/ab/Crawfordandostrom1995.pdf).

### Repository materials

- [README](../README.md): project purpose, components, and experiment interface.
- [Build report](../REPORT.md): reported initial experiments and limitations.
- [Design decisions](DECISIONS.md): implementation choices and rationale.
- [Default configuration](../configs/default.yaml): the reviewed baseline setup.

This memo adds no new simulation results. The proposed scientific claims remain hypotheses until the corresponding measurements and controlled experiments are completed.
