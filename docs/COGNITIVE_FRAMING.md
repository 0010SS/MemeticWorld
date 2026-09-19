# MemeWorld: Cognitive-Science Framing

This document states, for every part of MemeWorld, which cognitive or cultural-evolution claim it
operationalizes, what the literature says, and where our implementation departs from it. Statements are
written so they can be lifted into a paper, poster or README. Decision numbers (D1, D39, ...) refer to
[DECISIONS.md](DECISIONS.md).

> **Citation note.** References were compiled from memory and have not yet been checked against the
> sources. Verify every citation (authors, year, venue, and that the paper says what we attribute to it)
> before any submission. The less certain ones are marked †.

---

## 1. Framing statement

**One-paragraph version.** Human groups develop shared expressions (nicknames, in-jokes, coined phrases)
for recurring situations. We ask which cognitive ingredients are sufficient for such conventions to
emerge, and whether they come to track the causal structure of the environment rather than its surface.
MemeWorld is a closed micro-society of eight language-model agents built on the Generative Agents
architecture (Park et al., 2023). The environment generates incidents from four hidden relational
categories: cascading failures, mistakes that cancel out, independent coincidences, and failures with
beneficial outcomes. Agents never see these categories. They perceive incidents partially and from their
own vantage point, store them in a reconstructive, lossy memory, retrieve them stochastically, reflect on
them, and talk about them. No agent is ever asked to coin or name anything. An external observer, never
visible to the agents, detects candidate conventions, traces their transmission, and tests whether they
align with the hidden categories beyond chance and generalize to surface-novel instances. Memory fidelity,
attention, event density and social biases are manipulated experimentally.

**Research question.** Under what cognitive and social constraints do shared lexical conventions emerge
in a small group, and do they come to encode latent *relational* structure (Gentner, 1983) rather than
surface features?

**Theoretical position.** Culture is treated as the population-level outcome of individual cognition
applied repeatedly during social transmission, as in the "epidemiology of representations" (Sperber,
1996) and iterated learning (Kirby, Cornish & Smith, 2008). A meme (Dawkins, 1976) is operationalized as
a *convention* in Lewis's (1969) sense: a regularity in expression that several members share and that
has acquired a local meaning.

---

## 2. The experiment as a cultural-transmission paradigm

| Design element | Cognitive-science counterpart | Statement |
|---|---|---|
| Closed group of 8 agents, 3 simulated days | Laboratory micro-societies (Baum et al., 2004); closed-group convention formation (Garrod & Doherty, 1994; Centola & Baronchelli, 2015) | Conventions are studied in a small, closed group with a fixed social network, the same unit used in laboratory work on emerging communication (Galantucci, 2005; Fay et al., 2010). |
| Retelling through conversation and memory | Serial reproduction (Bartlett, 1932); transmission chains (Mesoudi & Whiten, 2008); iterated learning (Kirby et al., 2008; Griffiths & Kalish, 2007) | Every retelling passes through a lossy memory bottleneck. Transmission-chain research shows that repeated reconstruction transforms content systematically toward the transmitters' prior expectations. |
| Four separate layers: world, agents, experiment controller, observer (D-layer rules) | Separation of experimenter knowledge from participant knowledge; blinding | The ground truth (hidden categories, event ids) exists only on the simulator side. Agents receive only concrete sentences. This mirrors a blind experiment, in which participants have no access to the manipulation. |
| Agents are never told to invent memes, slang or new words | Demand characteristics (Orne, 1962) | Instructions to "coin terms" would create conventions through demand rather than cognition. Every prompt is audited for this, and tests enforce it. |
| Observer / analyzer | Etic analysis, the outside observer's categories (Pike, 1967); field-linguistic elicitation | The analyzer is an outside observer. It infers conventions from usage and never feeds its results back to the agents. |
| Private probes after the run | Recall and recognition testing; elicited meaning | Each agent's final memory is probed privately for what an expression means and which of four incidents it fits. Answers never enter any memory. This separates *public use* from *private representation*. |
| Deterministic replay of the full trace | Reproducibility; auditing of stimuli and responses | Every perception, memory, retrieval and utterance is logged with provenance, so any cultural outcome can be traced back to the individual cognitive events behind it. |

---

## 3. The environment: hidden relational structure

**Statement.** The environment has a *latent causal structure* that agents must infer from sparse,
partial evidence, as in latent-cause models of learning (Gershman, Blei & Niv, 2010). Each hidden family
is a **relational category** (Gentner & Kurtz, 2005): its members share a causal or relational pattern,
not surface features. A spilled coffee and a mis-set incubator timer belong to the same category because
of how the outcome follows from the mistake.

| Family | Relational schema | Relevant literature |
|---|---|---|
| E1 cascade failure | One small mistake → chain of unrelated failures | Causal-chain reasoning; attribution of blame along chains |
| E2 cancelling mistakes | Two errors cancel, nothing goes wrong | Counterfactual "near misses"; norm theory: abnormal events call up alternatives (Kahneman & Miller, 1986; Roese, 1997) |
| E3 independent coincidence | Unrelated people do the same unusual thing | Coincidence detection as evidence of hidden causes (Griffiths & Tenenbaum, 2007) |
| E4 beneficial failure | An apparent failure has a good outcome | Upward/downward counterfactuals, "blessing in disguise" (Roese, 1997) |

**Surface variation and holdout.** Each family has several surface scripts, and from day 3 only held-out
scripts are used. A convention that is used spontaneously for a held-out instance demonstrates **far
transfer** (Barnett & Ceci, 2002): the expression is tied to the relation, not the surface.

**No world-supplied labels (D39).** Event text had contained ready-made names ("the whole backpack
mix-up", "'extra crispy'"). These act as experimenter-supplied labels and prime convergence (Orne, 1962).
They were removed so that any shared name must come from the agents.

**Asymmetric access to causes (D39).** Hidden causes (for example, "the alarm was set for PM") are
visible only to the people involved. Bystanders see only the outcomes. This implements the actor-observer
asymmetry in causal knowledge (Jones & Nisbett, 1972; see Malle, 2006, for qualifications): actors know
the situational causes of their own behaviour, while observers must infer them or learn them through
talk. Causal structure can therefore become shared only through communication, which is the process under
study.

---

## 4. Agent cognition, component by component

For each component: the claim it implements, how it is implemented, and where it departs from the
literature.

### 4.1 Architecture

**Statement.** Agents follow the Generative Agents architecture (Park et al., 2023): perception → memory
stream → retrieval → reflection → planning and dialogue. Its lineage runs through symbolic cognitive
architectures such as ACT-R (Anderson et al., 2004) and Soar (Laird, Newell & Rosenbloom, 1987), and it
fits the CoALA taxonomy of language-agent architectures (Sumers et al., 2024). We use the upstream
memory structures, scoring functions and prompt templates unchanged where possible. Each change is logged
as a departure.

**Individual differences.** Profiles give each agent traits, a communication style, habits, a routine and
relationships. These are thin: three trait words and one style phrase. See §9 for why that matters.

### 4.2 Attention and perception (D42, D45)

**Statement.** Perception is **selective and capacity-limited**. People notice only part of what happens
around them (inattentional blindness: Simons & Chabris, 1999). They notice less under high perceptual
load or divided attention (Lavie, 1995; Kahneman, 1973), and more when familiar people are involved.

| Mechanism | Implementation | Grounding |
|---|---|---|
| Fact-level partial perception | Each fact noticed independently; p = 0.45 × (0.5 + 0.5 × salience) | Selective attention (Broadbent, 1958; Treisman, 1964) |
| Other room of the building | × 0.35 | Spatial attention, reduced access |
| In a conversation | × 0.6 | Divided attention; dual-task costs (Kahneman, 1973) |
| Crowded place (D45) | × 0.92 per agent in the building beyond 2 | Perceptual load (Lavie, 1995) |
| Familiar people involved | + 0.25 × familiarity | Attentional capture by familiar and self-relevant stimuli (cocktail-party effect: Cherry, 1953; Moray, 1959) |
| Own involvement | p = 1 | Self-relevance; participants cannot miss their own actions |

**Viewpoints (D42).** People who witness the same event come away with different experiences of it
(Hastorf & Cantril, 1954). Each noticed fact is re-described from the observer's **vantage**:
participant, near, distracted or far. A participant's version is first-person, which corresponds to
field-perspective memory (Nigro & Neisser, 1983). Observers far away get glimpses; distracted observers
half-notice. People the observer does not know are described by appearance, not by name. This follows
face- and person-recognition models, in which unfamiliar people are recognized by appearance without
access to identity or name (Bruce & Young, 1986). The renderer may omit or blur but may not add causes or
motives. This keeps viewpoints a matter of *access*, not of invented interpretation.

**Departure.** Vantage has four discrete levels, and the LLM produces the re-description. Real
perceptual degradation is continuous and modality-specific.

### 4.3 Encoding: reconstructive, lossy memory (D43, D44)

**Statement.** Memory is **reconstructive**. People store meaning and a schema-shaped version, not a
recording (Bartlett, 1932). Surface wording decays quickly while gist is retained (Sachs, 1967).
Fuzzy-trace theory (Reyna & Brainerd, 1995) describes this as parallel verbatim and gist traces, with
the verbatim trace fading faster. `memory.encoding_noise` is a single parameter along this
verbatim-to-gist axis. At 0 memories are stored verbatim with no rewrite; at higher values more detail
is lost and wording is paraphrased.

| Encoding step | Grounding |
|---|---|
| Low-salience facts dropped with p = noise × (1 − salience) | Selective encoding; levels of processing (Craik & Lockhart, 1972) |
| Unfamiliar names become "a student" | Gist-level encoding of peripheral people; person recognition without identity (Bruce & Young, 1986) |
| Rewrite in the agent's own framing | Schema-driven reconstruction (Bartlett, 1932) |
| Importance rated by poignancy prompt (GA) | Emotional and personal significance modulates consolidation (McGaugh, 2004) |
| Participants encode their own actions as their own experience (D43) | Autobiographical memory is organized around the self (Conway & Pleydell-Pearce, 2000); self-reference effect (Rogers, Kuiper & Kirker, 1977) |

**Encoding variability: the "memory lens" (D44).** A single act of remembering depends on momentary
state: what was attended, current goals and the cues present. Stimulus-sampling theory (Estes, 1955) and
the encoding-variability principle formalize this. Each encoding samples a lens that sets:

- **Focus**: the people, the action, the scene, the feeling, the oddest detail, implications for one's
  plans, or who else was around. Perspective at encoding shapes what is retained (Pichert & Anderson, 1977).
- **Length and style.**
- **Fidelity jitter** around the condition's noise level.
- **Uncertainty**, with probability 0.3: metamemory, knowing that one is unsure.
- **One misremembered minor detail**, with probability 0.5 × noise. This covers misinformation and
  distortion (Loftus & Palmer, 1974; Loftus, 2005) and Schacter's (1999) "sins" of misattribution and bias.
- **Association**, with probability 0.35: a related older memory, retrieved stochastically, can colour
  the new one. Reactivated memories integrate new information (Hupbach et al., 2007) and are labile during
  reconsolidation (Nader, Schafe & LeDoux, 2000).

**Departure, stated honestly.** Through the CLI the model is close to deterministic: the same prompt
gave the same memory 3 of 4 times, and temperature cannot be set. Variability is therefore *imposed*
through sampled instructions, not an emergent property of the model. It is seeded and replayable. Its
range is a modelling assumption.

**Verbatim stickiness (D49, in progress).** Most wording fades, but distinctive wording tends to survive
verbatim. Distinctiveness enhances memory (von Restorff, 1933; Hunt, 2006), and certain phrasings are
more memorable than others (Danescu-Niculescu-Mizil et al., 2012). Each rare-word phrase in heard speech
survives verbatim with p = 0.3, raised by 0.2 for each existing memory that already contains it, up to
0.9. That rise models familiarity: wording already met before is easier to encode. This is the channel
through which exact expressions, and not just ideas, can be transmitted.

### 4.4 Consolidation, interference and forgetting

| Mechanism | Implementation | Grounding |
|---|---|---|
| Near-duplicate merge | New memory with cosine ≥ 0.92 folds into the old one (old wording kept, importance raised) | Schema assimilation; blending and interference between similar episodes |
| Capacity-based forgetting | Above 300 memories, drop the lowest importance × recency | Adaptive forgetting: memory tracks the probability that an item will be needed (Anderson & Schooler, 1991) |
| Recency decay | exp(−decay × hours since last access), reset when retrieved | Forgetting curve (Ebbinghaus, 1885). Our curve is exponential; empirical forgetting is closer to a power function (Wixted & Ebbesen, 1991). Retrieval refreshes the trace (testing effect: Roediger & Karpicke, 2006). |

### 4.5 Retrieval: cue-driven and stochastic

**Statement.** Retrieval is **cue-dependent** (encoding specificity: Tulving & Thomson, 1973) and
**probabilistic**. The same cue does not always bring back the same memories, as in the sampling models
SAM (Raaijmakers & Shiffrin, 1981) and MINERVA 2 (Hintzman, 1984). Scores combine relevance, recency and
importance, following the Generative Agents weighting, which approximates the rational-analysis need
probability (Anderson & Schooler, 1991). Memories are sampled with P ∝ exp(score/τ). This is Luce's
(1959) choice rule, and Gumbel-top-k draws k memories without replacement under it. At τ = 0 retrieval
reduces to GA's deterministic top-k. Top-k = 5 keeps retrieved context within typical working-memory
limits (Cowan, 2001).

**Source weights (D50, in progress).** Relationship facts from seeding and routine sightings are
down-weighted by half at retrieval, so that retrieved context is dominated by experiences, as it is for
people. The funnel analysis (§8) found that stable background facts made up about half of what agents
recalled before speaking.

### 4.6 Reflection: abstraction and schema induction

**Statement.** GA reflection is an offline abstraction step. Episodes accumulate until an importance
threshold is reached; the agent then asks itself high-level questions and forms insights citing evidence.
This parallels offline consolidation, in which general structure is extracted from episodes
(complementary learning systems: McClelland, McNaughton & O'Reilly, 1995), and schema induction from
compared examples (Gick & Holyoak, 1983). Our reflection prompts include neutral pattern-seeking
questions ("Have similar situations happened before?") but never ask for a label.

### 4.7 Spontaneous reminding (D48, in progress)

**Statement.** Linking two incidents requires that one *reminds* the agent of the other. Reminding is a
core mechanism of learning from experience (Schank, 1982; Ross, 1984). People rarely retrieve
structurally similar but surface-different cases spontaneously: retrieval is driven by surface
similarity, while inference is driven by structure (Gick & Holyoak, 1980; Gentner, Rattermann & Forbus,
1993). When a salient experience is encoded, the agent is shown a stochastic mix of its most related
earlier experiences and a few important older ones. It may say that one feels alike, or that nothing
comes to mind. A positive answer is stored as a thought linking the two memories.

**Departure.** Including important older memories regardless of similarity gives surface-different
analogues more access than human reminding does. We also use lexical embeddings, which on their own would
give them almost none. This is a deliberate thumb on the scale, to test whether the *downstream*
processes produce conventions once connections exist, and should be reported as such.

### 4.8 Dialogue and social transmission

| Mechanism | Grounding |
|---|---|
| GA conversation (iterative utterances conditioned on retrieved memories) | Conversation as the main vehicle of cultural transmission; gossip about third parties is a large share of talk (Dunbar, 2004) |
| Conversations started by salient observations | Sharing of emotional and unexpected events (Rimé, 2009) |
| Overhearing (p = 0.3 per utterance) encoded as "overheard" | Overhearers understand and retain less than addressees (Schober & Clark, 1989) |
| Evening catch-ups between close friends (D47, in progress) | Recurring shared settings for repeated reference to the same things; conventions form through repeated interaction in stable pairs and communities (Clark & Wilkes-Gibbs, 1986; Garrod & Doherty, 1994) |
| Hidden-type instances clustered in a "home" friend group (D46, in progress) | Conventions need a community that repeatedly faces the same referents; shared experience builds common ground (Clark & Marshall, 1981; Clark, 1996) |

**Expected dialogue mechanisms.** Repeated reference to the same thing shortens and standardizes
descriptions into "conceptual pacts" (Clark & Wilkes-Gibbs, 1986; Brennan & Clark, 1996). Interlocutors
align their word choices (interactive alignment: Pickering & Garrod, 2004). Retelling produces
**leveling, sharpening and assimilation** (Allport & Postman, 1947).

---

## 5. Social-cognitive modules (optional, config-toggled)

| Module | Claim operationalized | Implementation | Grounding |
|---|---|---|---|
| Emotion | Affect modulates attention, memory and sharing | Valence and arousal appraised from a lexicon; arousal raises memory importance; mild contagion between conversation partners; mood line in prompts only for clearly non-neutral states (D36) | Emotional memory enhancement (McGaugh, 2004); mood congruence (Bower, 1981); emotional contagion (Hatfield, Cacioppo & Rapson, 1994); emotional selection of memes (Heath, Bell & Sternberg, 2001); arousal drives sharing (Berger & Milkman, 2012) |
| Social reward | Lifting others' mood is rewarding and reinforces the behaviour | Reward = partner's change in valence; raises importance of rewarded conversations; prefers talking to people who seem down | Sharing and social connection as intrinsically rewarding (Tamir & Mitchell, 2012) |
| Prestige bias | People attend to and copy high-status individuals | Status from network centrality biases attention, importance and prompt status cues | Prestige-biased transmission (Henrich & Gil-White, 2001) |
| Conformity bias | Information from several distinct sources is weighted more | Memories corroborated by multiple distinct conversation partners get a log-boost at retrieval | Conformist transmission (Boyd & Richerson, 1985); conformity (Asch, 1956); social learning strategies (Morgan et al., 2012) |

**Lesson from the first social-reward run (D36).** Injecting a mood word into ~90% of prompts made
"restless" the most repeated word in the population, a treatment artifact. Stimuli that are presented
repeatedly get repeated back, the same concern as demand characteristics (Orne, 1962). Module outputs now
enter prompts only when clearly non-neutral, and the analyzer checks top candidates against the treatment
wording.

---

## 6. Measurement

| Measure | Statement | Grounding |
|---|---|---|
| Candidate conventions | An expression shared by at least 2 speakers and 3 uses, not repeating world wording, scored higher when rare in English (wordfreq prior, D37) or novel | Convention as a shared regularity (Lewis, 1969). The frequency prior separates coinages from ordinary vocabulary. |
| LLM classifier | Judges whether usage shows a local meaning beyond ordinary description | Stand-in for human coders; observer model held fixed across conditions (D41) |
| Transmission graph | Exposure → later use, with confidence from memory provenance | Transmission-chain analysis (Mesoudi & Whiten, 2008) |
| Complex contagion | Number of distinct sources from which a listener heard an expression | Adoption of conventions and behaviours often requires reinforcement from multiple contacts (Centola & Macy, 2007; Centola, 2010) |
| Latent alignment + permutation null (D40) | Does usage concentrate on one hidden family more than under random relabelling of events? | A neutral model is needed before inferring selection (Bentley, Hahn & Shennan, 2004) |
| No-events control (D41) | Same society with no incidents | Estimates how many "conventions" arise from shared routines and topics alone: the neutral baseline |
| Private probes | Meaning question and four-way incident-matching question per agent | Dissociation of public use from private representation; recall and recognition |
| Held-out spontaneous use | Use of the expression for surface-novel instances | Far transfer (Barnett & Ceci, 2002) |

---

## 7. Hypotheses by condition

| Condition | Manipulation | Hypothesis and rationale |
|---|---|---|
| baseline | Moderate noise (0.3), stochastic retrieval | Reference condition |
| perfect_memory | Verbatim encoding, deterministic retrieval, no forgetting | Preserves exact wording (more copying) but removes the compression that, in iterated learning, drives structure (Kirby et al., 2008). Predicts more verbatim reuse but *less* abstraction. |
| high_noise | Gist-only encoding, fast forgetting, τ = 1.5 | A strong bottleneck forces compression. Iterated learning predicts simplification, but too much loss destroys transmission. Predicts fewer surviving expressions; those that survive are shorter and more schematic. |
| event_rich | Event rate doubled | More instances per family give more occasions for repeated reference and reminding. Predicts more conventions (Clark & Wilkes-Gibbs, 1986). |
| social_reward | Emotion + social reward | Emotional content is selected for transmission (Heath et al., 2001). Predicts more retelling of emotionally charged incidents. |
| no_events | No incidents | Null model: any conventions reflect routine topics only |
| Haiku vs Sonnet agents | Agent model | Whether results depend on the language model rather than the architecture |

---

## 8. Interpreting the results so far

**Main result.** Across 9 finished runs (Haiku and Sonnet agents, 3 days each), no convention emerged
that was aligned with a hidden family. Of about 208 candidates in the seed-42 runs, none beat the
permutation null. The no-events control produced as much phrase spread as the runs with events. **This
null result is what the cognitive literature predicts for this configuration.** Each break in the
pipeline corresponds to a known constraint:

| Where the process stopped | Measured | Cognitive-science explanation |
|---|---|---|
| Incidents were discussed briefly, then dropped | 2–5 mentions by 2–3 speakers per incident; talk ended within about 2–21 hours | Names and conceptual pacts emerge from **repeated reference** to the same referent (Clark & Wilkes-Gibbs, 1986). Nothing was referred to often enough to need a name. |
| Same-family incidents were almost never connected | 1–19 utterances per run mention two different incidents; 0–6 reflections per run reference two incidents | **Surface-driven retrieval**: people rarely retrieve analogues that share only relational structure (Gick & Holyoak, 1980; Gentner et al., 1993). Our incidents were designed to differ on the surface. |
| Private insights stayed private | Reflections described patterns within a single incident but were rarely voiced | Reflection is individual. Without a shared referent and label, abstractions do not become common ground (Clark, 1996). Labels help people form and share relational categories (Gentner, 2003; Lupyan, Rakison & McClelland, 2007). |
| Exact wording rarely survived | 20–37% of distinctive phrases survived verbatim in listeners' memories; reuse 15–31% if kept, 1–7% if not | Verbatim form is lost quickly while meaning is kept (Sachs, 1967; Reyna & Brainerd, 1995). |
| Little multi-source exposure | Only 3–6% of phrases reached a listener from 2 or more speakers | Conventions behave like **complex contagions** that need reinforcement from several contacts (Centola & Macy, 2007). |
| Talk dominated by logistics and shared class vocabulary | 31–53% of utterances about schedules; top shared phrases were "bayes theorem", "conditional probability" | Unlike human talk, which is largely social and gossip (Dunbar, 2004). |

**Qualitative findings in the same terms.**

- **"The frisbee analogy"** (event-rich run) was an explanatory analogy coined by one agent and used by 4
  over 3 days. It was not tied to any hidden family. In private probes all agents, including those who
  had used it, said in their free-text answers that they had never heard the expression: the idea
  survived and the wording did not. This is a direct instance of gist-over-verbatim retention (Sachs, 1967).
- **Gossip mutated in retelling**, for example an agent later claimed the swapped backpack was his. This
  combines leveling and assimilation in rumour transmission (Allport & Postman, 1947) with source-monitoring
  errors (Johnson, Hashtroudi & Lindsay, 1993).
- **Treatment wording became the most "meme-like" expression** in the flawed social-reward run (D36):
  repeated stimuli get repeated back.

---

## 9. Theory-driven interventions (D46–D50, in progress)

These manipulations were designed from the funnel analysis above. They are implemented in the working
tree but not yet logged, tested or run. Each targets one break and is a cognitive hypothesis in its own
right.

| Change | Targets | Hypothesis |
|---|---|---|
| D46 Clustering: 70% of a family's instances fall within its "home" friend group | Too little repeated reference; no shared community | A community that repeatedly meets the same kind of situation develops shared ways of referring to it (Garrod & Doherty, 1994). |
| D47 Evening catch-ups between close friends (up to 6 turns) | Brief, logistics-dominated talk | A recurring setting for retelling the day increases repeated reference and gossip (Dunbar, 2004). |
| D48 Spontaneous reminding | Incidents never connected | Once incidents are linked in memory, the link can be voiced, and a shared shorthand can form (Schank, 1982; Ross, 1984). |
| D49 Verbatim stickiness of distinctive wording | Exact wording lost | Distinctive phrasing survives verbatim often enough to be copied (von Restorff, 1933; Hunt, 2006). |
| D50 Source weights at retrieval | Background facts crowding out experiences | Recall before speaking should be dominated by recent experiences and conversations, as it is for people. |

**Caution.** D46 and D48 make conventions more likely by construction. A positive result after them
supports "these processes are *sufficient*", not "conventions emerge unaided". Each should be run with
and without the change as an ablation, against the no-events null.

---

## 10. Validity: what LLM agents can and cannot tell us

1. **LLMs are not people.** LLMs reproduce some human effects but not others, and their behaviour depends
   on the prompt (Binz & Schulz, 2023; Aher, Arriaga & Kalai, 2023; Dillion et al., 2023). We use them as
   a *generative model of plausible behaviour*, conditioned on explicit cognitive constraints. Claims are
   about the sufficiency of the modelled constraints, not about human psychology directly.
2. **Homogeneity.** All agents share one model and thin profiles. Human populations differ in style and
   idiolect, which gives conventions variation to select from. Diversity here comes only from profiles,
   perception, memory sampling and the lens.
3. **Imposed stochasticity.** Encoding variability (D44) is sampled by us, because the model is close to
   deterministic. Its form and range are assumptions.
4. **Default register.** The models' helpful, varied register discourages repetition and catchphrases.
   This bias works *against* convention formation and should be reported.
5. **LLM populations can form conventions.** Recent work finds that populations of LLM agents in a naming
   game converge on shared conventions (Ashery, Aiello & Baronchelli, 2025†). There the task demands
   coordination; in MemeWorld nothing does. The contrast is the point.
6. **Scale and statistics.** 8 agents, 3 days, mostly one seed per condition. Human in-group slang forms
   over weeks. Group size also affects convention structure (Raviv, Meyer & Lev-Ari, 2019). Current
   comparisons across conditions are anecdotal until multiple seeds are run.
7. **Observer validity.** An LLM classifier stands in for human coders. The observer model is held fixed
   (D41), but its judgments have not been validated against human raters.
8. **Semantic similarity is lexical.** Hashed n-gram embeddings capture wording, not meaning. They
   underestimate relational similarity, which both limits and motivates D48.

---

## References

Aher, G., Arriaga, R. I., & Kalai, A. T. (2023). Using large language models to simulate multiple humans and replicate human subject studies. *ICML*.
Allport, G. W., & Postman, L. (1947). *The Psychology of Rumor*. Holt.
Anderson, J. R., Bothell, D., Byrne, M. D., Douglass, S., Lebiere, C., & Qin, Y. (2004). An integrated theory of the mind. *Psychological Review, 111*, 1036–1060.
Anderson, J. R., & Schooler, L. J. (1991). Reflections of the environment in memory. *Psychological Science, 2*, 396–408.
Asch, S. E. (1956). Studies of independence and conformity. *Psychological Monographs, 70*(9).
Ashery, A. F., Aiello, L. M., & Baronchelli, A. (2025). Emergent social conventions and collective bias in LLM populations. *Science Advances*. †
Barnett, S. M., & Ceci, S. J. (2002). When and where do we apply what we learn? A taxonomy for far transfer. *Psychological Bulletin, 128*, 612–637.
Bartlett, F. C. (1932). *Remembering*. Cambridge University Press.
Baum, W. M., Richerson, P. J., Efferson, C. M., & Paciotti, B. M. (2004). Cultural evolution in laboratory microsocieties including traditions of rule giving and rule following. *Evolution and Human Behavior, 25*, 305–326.
Bentley, R. A., Hahn, M. W., & Shennan, S. J. (2004). Random drift and culture change. *Proceedings of the Royal Society B, 271*, 1443–1450.
Berger, J., & Milkman, K. L. (2012). What makes online content viral? *Journal of Marketing Research, 49*, 192–205.
Binz, M., & Schulz, E. (2023). Using cognitive psychology to understand GPT-3. *PNAS, 120*(6).
Bower, G. H. (1981). Mood and memory. *American Psychologist, 36*, 129–148.
Boyd, R., & Richerson, P. J. (1985). *Culture and the Evolutionary Process*. University of Chicago Press.
Brennan, S. E., & Clark, H. H. (1996). Conceptual pacts and lexical choice in conversation. *JEP: Learning, Memory, and Cognition, 22*, 1482–1493.
Broadbent, D. E. (1958). *Perception and Communication*. Pergamon.
Bruce, V., & Young, A. (1986). Understanding face recognition. *British Journal of Psychology, 77*, 305–327.
Centola, D. (2010). The spread of behavior in an online social network experiment. *Science, 329*, 1194–1197.
Centola, D., & Baronchelli, A. (2015). The spontaneous emergence of conventions: An experimental study of cultural evolution. *PNAS, 112*, 1989–1994.
Centola, D., & Macy, M. (2007). Complex contagions and the weakness of long ties. *American Journal of Sociology, 113*, 702–734.
Cherry, E. C. (1953). Some experiments on the recognition of speech, with one and with two ears. *JASA, 25*, 975–979.
Clark, H. H. (1996). *Using Language*. Cambridge University Press.
Clark, H. H., & Marshall, C. R. (1981). Definite reference and mutual knowledge. In *Elements of Discourse Understanding*. Cambridge University Press.
Clark, H. H., & Wilkes-Gibbs, D. (1986). Referring as a collaborative process. *Cognition, 22*, 1–39.
Conway, M. A., & Pleydell-Pearce, C. W. (2000). The construction of autobiographical memories in the self-memory system. *Psychological Review, 107*, 261–288.
Cowan, N. (2001). The magical number 4 in short-term memory. *Behavioral and Brain Sciences, 24*, 87–114.
Craik, F. I. M., & Lockhart, R. S. (1972). Levels of processing. *Journal of Verbal Learning and Verbal Behavior, 11*, 671–684.
Danescu-Niculescu-Mizil, C., Cheng, J., Kleinberg, J., & Lee, L. (2012). You had me at hello: How phrasing affects memorability. *ACL*.
Dawkins, R. (1976). *The Selfish Gene*. Oxford University Press.
Dillion, D., Tandon, N., Gu, Y., & Gray, K. (2023). Can AI language models replace human participants? *Trends in Cognitive Sciences, 27*, 597–600.
Dunbar, R. I. M. (2004). Gossip in evolutionary perspective. *Review of General Psychology, 8*, 100–110.
Ebbinghaus, H. (1885). *Über das Gedächtnis*. Duncker & Humblot.
Estes, W. K. (1955). Statistical theory of spontaneous recovery and regression. *Psychological Review, 62*, 145–154.
Fay, N., Garrod, S., Roberts, L., & Swoboda, N. (2010). The interactive evolution of human communication systems. *Cognitive Science, 34*, 351–386.
Galantucci, B. (2005). An experimental study of the emergence of human communication systems. *Cognitive Science, 29*, 737–767.
Garrod, S., & Doherty, G. (1994). Conversation, co-ordination and convention. *Cognition, 53*, 181–215.
Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy. *Cognitive Science, 7*, 155–170.
Gentner, D. (2003). Why we're so smart. In D. Gentner & S. Goldin-Meadow (Eds.), *Language in Mind*. MIT Press.
Gentner, D., & Kurtz, K. J. (2005). Relational categories. In W. Ahn et al. (Eds.), *Categorization Inside and Outside the Laboratory*. APA.
Gentner, D., Rattermann, M. J., & Forbus, K. D. (1993). The roles of similarity in transfer. *Cognitive Psychology, 25*, 524–575.
Gershman, S. J., Blei, D. M., & Niv, Y. (2010). Context, learning, and extinction. *Psychological Review, 117*, 197–209.
Gick, M. L., & Holyoak, K. J. (1980). Analogical problem solving. *Cognitive Psychology, 12*, 306–355.
Gick, M. L., & Holyoak, K. J. (1983). Schema induction and analogical transfer. *Cognitive Psychology, 15*, 1–38.
Griffiths, T. L., & Kalish, M. L. (2007). Language evolution by iterated learning with Bayesian agents. *Cognitive Science, 31*, 441–480.
Griffiths, T. L., & Tenenbaum, J. B. (2007). From mere coincidences to meaningful discoveries. *Cognition, 103*, 180–226.
Hastorf, A. H., & Cantril, H. (1954). They saw a game: A case study. *Journal of Abnormal and Social Psychology, 49*, 129–134.
Hatfield, E., Cacioppo, J. T., & Rapson, R. L. (1994). *Emotional Contagion*. Cambridge University Press.
Heath, C., Bell, C., & Sternberg, E. (2001). Emotional selection in memes: The case of urban legends. *JPSP, 81*, 1028–1041.
Henrich, J., & Gil-White, F. J. (2001). The evolution of prestige. *Evolution and Human Behavior, 22*, 165–196.
Hintzman, D. L. (1984). MINERVA 2: A simulation model of human memory. *Behavior Research Methods, Instruments, & Computers, 16*, 96–101.
Hunt, R. R. (2006). The concept of distinctiveness in memory research. In R. R. Hunt & J. Worthen (Eds.), *Distinctiveness and Memory*. Oxford University Press.
Hupbach, A., Gomez, R., Hardt, O., & Nadel, L. (2007). Reconsolidation of episodic memories: A subtle reminder triggers integration of new information. *Learning & Memory, 14*, 47–53. †
Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring. *Psychological Bulletin, 114*, 3–28.
Jones, E. E., & Nisbett, R. E. (1972). The actor and the observer: Divergent perceptions of the causes of behavior. In *Attribution: Perceiving the Causes of Behavior*. General Learning Press.
Kahneman, D. (1973). *Attention and Effort*. Prentice-Hall.
Kahneman, D., & Miller, D. T. (1986). Norm theory: Comparing reality to its alternatives. *Psychological Review, 93*, 136–153.
Kirby, S., Cornish, H., & Smith, K. (2008). Cumulative cultural evolution in the laboratory. *PNAS, 105*, 10681–10686.
Laird, J. E., Newell, A., & Rosenbloom, P. S. (1987). SOAR: An architecture for general intelligence. *Artificial Intelligence, 33*, 1–64.
Lavie, N. (1995). Perceptual load as a necessary condition for selective attention. *JEP: Human Perception and Performance, 21*, 451–468.
Lewis, D. (1969). *Convention: A Philosophical Study*. Harvard University Press.
Loftus, E. F. (2005). Planting misinformation in the human mind. *Learning & Memory, 12*, 361–366.
Loftus, E. F., & Palmer, J. C. (1974). Reconstruction of automobile destruction. *Journal of Verbal Learning and Verbal Behavior, 13*, 585–589.
Luce, R. D. (1959). *Individual Choice Behavior*. Wiley.
Lupyan, G., Rakison, D. H., & McClelland, J. L. (2007). Language is not just for talking: Redundant labels facilitate learning of novel categories. *Psychological Science, 18*, 1077–1083.
Malle, B. F. (2006). The actor-observer asymmetry in attribution: A (surprising) meta-analysis. *Psychological Bulletin, 132*, 895–919.
McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review, 102*, 419–457.
McGaugh, J. L. (2004). The amygdala modulates the consolidation of memories of emotionally arousing experiences. *Annual Review of Neuroscience, 27*, 1–28.
Mesoudi, A., & Whiten, A. (2008). The multiple roles of cultural transmission experiments in understanding human cultural evolution. *Phil. Trans. R. Soc. B, 363*, 3489–3501.
Moray, N. (1959). Attention in dichotic listening. *Quarterly Journal of Experimental Psychology, 11*, 56–60.
Morgan, T. J. H., Rendell, L. E., Ehn, M., Hoppitt, W., & Laland, K. N. (2012). The evolutionary basis of human social learning. *Proceedings of the Royal Society B, 279*, 653–662.
Nader, K., Schafe, G. E., & LeDoux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature, 406*, 722–726.
Nigro, G., & Neisser, U. (1983). Point of view in personal memories. *Cognitive Psychology, 15*, 467–482.
Orne, M. T. (1962). On the social psychology of the psychological experiment. *American Psychologist, 17*, 776–783.
Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *UIST*.
Pichert, J. W., & Anderson, R. C. (1977). Taking different perspectives on a story. *Journal of Educational Psychology, 69*, 309–315.
Pickering, M. J., & Garrod, S. (2004). Toward a mechanistic psychology of dialogue. *Behavioral and Brain Sciences, 27*, 169–190.
Pike, K. L. (1967). *Language in Relation to a Unified Theory of the Structure of Human Behavior*. Mouton.
Raaijmakers, J. G. W., & Shiffrin, R. M. (1981). Search of associative memory. *Psychological Review, 88*, 93–134.
Raviv, L., Meyer, A., & Lev-Ari, S. (2019). Larger communities create more systematic languages. *Proceedings of the Royal Society B, 286*.
Reyna, V. F., & Brainerd, C. J. (1995). Fuzzy-trace theory: An interim synthesis. *Learning and Individual Differences, 7*, 1–75.
Rimé, B. (2009). Emotion elicits the social sharing of emotion. *Emotion Review, 1*, 60–85.
Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning. *Psychological Science, 17*, 249–255.
Roese, N. J. (1997). Counterfactual thinking. *Psychological Bulletin, 121*, 133–148.
Rogers, T. B., Kuiper, N. A., & Kirker, W. S. (1977). Self-reference and the encoding of personal information. *JPSP, 35*, 677–688.
Ross, B. H. (1984). Remindings and their effects in learning a cognitive skill. *Cognitive Psychology, 16*, 371–416.
Sachs, J. S. (1967). Recognition memory for syntactic and semantic aspects of connected discourse. *Perception & Psychophysics, 2*, 437–442.
Schacter, D. L. (1999). The seven sins of memory. *American Psychologist, 54*, 182–203.
Schank, R. C. (1982). *Dynamic Memory*. Cambridge University Press.
Schober, M. F., & Clark, H. H. (1989). Understanding by addressees and overhearers. *Cognitive Psychology, 21*, 211–232.
Simons, D. J., & Chabris, C. F. (1999). Gorillas in our midst. *Perception, 28*, 1059–1074.
Sperber, D. (1996). *Explaining Culture: A Naturalistic Approach*. Blackwell.
Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2024). Cognitive architectures for language agents. *Transactions on Machine Learning Research*.
Tamir, D. I., & Mitchell, J. P. (2012). Disclosing information about the self is intrinsically rewarding. *PNAS, 109*, 8038–8043.
Treisman, A. M. (1964). Selective attention in man. *British Medical Bulletin, 20*, 12–16.
Tulving, E., & Thomson, D. M. (1973). Encoding specificity and retrieval processes in episodic memory. *Psychological Review, 80*, 352–373.
von Restorff, H. (1933). Über die Wirkung von Bereichsbildungen im Spurenfeld. *Psychologische Forschung, 18*, 299–342.
Wixted, J. T., & Ebbesen, E. B. (1991). On the form of forgetting. *Psychological Science, 2*, 409–415.
