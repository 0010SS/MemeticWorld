# Decision log: deviations from Generative Agents and from the build plan

Every choice below is either **not native to Generative Agents** (Park et al., 2023,
`joonspk-research/generative_agents@fe05a71`) or **not specified in the build plan**.
Each entry gives the reason and, where relevant, the scientific risk.

## 0. What is reused from Generative Agents (GA-native)

The upstream code is vendored, unmodified, in `third_party/generative_agents/` and is imported through
`backend/ga_compat.py`.

| GA component | How MemeWorld uses it |
|---|---|
| `AssociativeMemory` / `ConceptNode` (memory stream) | subclassed by `backend/memory/store.py::MemoryStream`; nodes, keyword indices, embeddings, and the GA on-disk format (`agents_final/<id>/associative_memory/nodes.json`) |
| `Scratch` (identity stable set, short-term state) | one per agent; `get_str_iss()` is the persona description used in every prompt |
| `retrieve.py`: `normalize_dict_floats`, `extract_importance`, `extract_relevance`, `cos_sim`, GA weights `gw=[0.5,3,2]` | the scoring half of stochastic retrieval |
| Reflection trigger (`importance_trigger_curr/max`, `importance_ele_n`) | the reflection schedule |
| Prompt templates `generate_focal_pt_v1`, `insight_and_evidence_v1` | reflection |
| `poignancy_{event,chat,thought}_v1` (v3_ChatGPT) | the importance score of every memory, identical in every condition |
| `decide_to_talk_v2` | conversation initiation |
| `converse.generate_summarize_agent_relationship` (called directly) | relationship summary before a conversation |
| `run_gpt_generate_iterative_chat_utt` (`iterative_convo_v1`; called directly) | every conversational utterance |
| `gpt_structure.generate_prompt`, `safe_generate_response`, `ChatGPT_safe_generate_response` | prompt filling and retry wrappers |
| Character sprite sheets (`environment/.../assets/characters`) | agent sprites in the campus view |

## 1. Infrastructure

**D1. LLM backend.** Upstream calls OpenAI gpt-3.5-turbo and text-davinci-003. MemeWorld routes every upstream call
(`ChatGPT_request`, `GPT_request`, `GPT4_request`) to `backend/llm/client.py`.
- Default: Claude Haiku 4.5 through the locally authenticated `claude` CLI (no API key was available).
- Alternatives: the Anthropic API (`llm.backend: anthropic`) and a deterministic offline mock.
- Extended thinking is disabled in the CLI; it multiplied latency by 3–10×.
- The CLI exposes no temperature or stop parameters. Stop sequences are applied post hoc, and temperature is
  ignored in CLI mode. *Risk:* sampling temperature is not under experimental control with this backend. Use
  `anthropic` for that.

**D2. Embeddings.** Upstream uses OpenAI `text-embedding-ada-002`. MemeWorld's default is a local, deterministic
hashed n-gram embedder (word unigrams and bigrams plus character trigrams). `sentence_transformers` is optional.
*Risk:* relevance is mostly lexical, which affects retrieval relevance and the analyzer's semantic coherence.
Semantic scores should be read as *lexical-contextual* similarity.

**D3. Compatibility shim.** Upstream needs a git-ignored `utils.py` and the `openai` package. MemeWorld injects stub
modules and monkey-patches the request functions. It also makes template paths absolute (upstream resolves them
relative to the working directory) and silences upstream debug prints. No upstream file is edited.

**D4. Output cleanup.** Modern chat models wrap JSON in markdown fences, and upstream calls `json.loads` on the
raw text, so fences are stripped in the routed `ChatGPT_request`. The system prompts ask for "output only what is
asked" (chat) and "continue the text" (davinci-style completion).

**D5. Tolerant parsers.** Upstream validators were written for davinci and fail often on chat models, wasting
retries. For focal points, insights, decide-to-talk and poignancy, MemeWorld keeps the upstream template text and
retry wrapper but supplies a tolerant parser: it keeps numbered items, drops preambles and markdown, and extracts
the last yes/no. Prompt wording is unchanged.

**D6. Parallelism.** Within a tick, agents' independent cognition runs in a thread pool (encoding, reactions,
decide-to-talk, conversations between disjoint pairs, reflection). Upstream runs strictly sequentially. Results
are applied in a fixed order.

**D7. Determinism and replay.**
- Every LLM call is keyed by (deterministic job scope, prompt hash, occurrence) and recorded in
  `llm_calls.jsonl`.
- Trace records are buffered per job and flushed in sorted scope order.
- `python -m backend.cli replay runs/<id>` re-executes a run from the recorded outputs and checks the trace's
  SHA-256; this is verified in the tests.
- Fresh runs with a real LLM are *not* bit-reproducible, because the model samples.

## 2. Agent ontology and world

**D8. Profile → GA identity set.** The plan's `AgentProfile` fields are rendered into GA's identity stable set:
- `innate` = traits
- `learned` = demographics, background, interests, clubs and communication style
- `lifestyle` = habits
- `daily_plan_req` = routine

Relationships (type, familiarity, affinity) have no GA equivalent. They reach cognition through (a) seed
memories and (b) a one-line relationship statement added to conversation and reaction context.

**D9. Seed memories.** GA seeds memory from the persona description. MemeWorld seeds one thought per
non-stranger relationship, for example "Maya Chen and Priya Raman are roommates; they know each other very well
and get along well." Each has fixed importance 5 and no LLM call.

**D10. World model.** GA uses a Tiled tile map with sectors, arenas and game objects, plus A* pathfinding.
MemeWorld uses:
- a semantic location graph (the plan's eight locations) with arenas (rooms) inside each location
- BFS paths, recorded for animation only
- conversation and overhearing that require the same location *and* arena; events are visible across the whole
  location, with reduced attention in other rooms
- Homewood building names as display-only labels; agents see only the generic names
- 15-minute ticks from 07:30 to 22:30, with nights skipped

**D11. Planning.** GA generates wake-up hours, daily plans, hourly schedules and task decompositions with the
LLM. MemeWorld uses deterministic routines from the profile instead, with seeded jitter of ±20 minutes and an 8%
chance of a small detour. The plan explicitly asked to avoid LLM calls for routine behavior. Re-planning comes
from:
- event beats that force the protagonist somewhere
- reaction decisions (MOVE)
- invitations: after a conversation between agents with affinity ≥ 0.6, the partner follows the initiator for up
  to 3 ticks with probability 0.15; this is not LLM-negotiated

**D12. Reaction decision (non-GA prompt).** GA's `decide_to_react` only chooses between waiting and talking about
another persona. The plan requires structured actions, so a new prompt (`backend/prompts/react_v1.txt`) returns
`{"action": CONTINUE|REACT|TALK|MOVE, "target", "utterance", "reason"}`. It is invoked only when an agent notices
a fact with salience ≥ 0.5. A REACT with words becomes an utterance heard by people nearby: probability 0.9 in the
same room, 0.2 elsewhere in the location.

## 3. Perception and memory (the main architectural change the plan asked for)

**D13. Partial perception.** GA perceives all events within a vision radius, subject to attention bandwidth and
retention. MemeWorld perceives individual facts of a world event independently:
- `p = base_attention × (0.5 + 0.5·salience) × other_arena_factor × busy_factor + familiarity_weight × familiarity`
- modules can then adjust `p`
- participants always perceive facts about themselves
- facts with visibility restrictions, such as the hidden cause of a mistake, reach only the roles privy to them

**D14. Ambient perception without an LLM.** Co-located people's routine activities ("Leo Martinez is lifting
weights at the Gym.") are stored with 50% probability and fixed importance 1–2, without an LLM poignancy call or
event triple. This saves hundreds of calls per day.

**D15. Lossy symbolic encoding (new step; `backend/memory/encoder.py`).** GA stores perceived event descriptions
directly. MemeWorld adds:
1. A deterministic semantic pre-filter. Low-salience facts are dropped with probability
   `noise·(1−salience)`, and unfamiliar people are generalized ("Leo" → "a student") with probability
   `entity_generalization·noise`.
2. An LLM reconstruction (`backend/prompts/encode_memory_v1.txt`) whose fidelity instruction depends on
   `memory.encoding_noise`:
   - < 0.15: verbatim
   - < 0.45: main points, "paraphrase except for any wording that particularly stuck"
   - < 0.75: gist
   - ≥ 0.75: vague gist, no quotes
3. At noise 0 (perfect memory), no LLM rewrite at all: the observation is stored verbatim.
4. Merging of near-duplicate recent memories (cosine ≥ `merge_threshold`, only when noise > 0).

*Risk:* the phrase "wording that particularly stuck" at medium noise is a memory-fidelity statement, not an
instruction to invent. It still affects how often exact phrases survive, which is precisely the variable the
perfect-memory, baseline and high-noise conditions manipulate.

**D16. Importance** is always the upstream GA poignancy prompt, applied to the *encoded* memory text in every
condition, so memory treatments do not change how importance is scored.

**D17. Forgetting (new).** Upstream never forgets. MemeWorld removes the weakest nodes, by strength
`poignancy/10 × recency`, while an agent is over `memory.memory_capacity`.

**D18. Keywords and triples.** GA derives subject–predicate–object and keywords with an LLM call
(`generate_event_triple`). MemeWorld uses heuristics instead (agent name, source verb, main other person or place;
capitalized and content words). These only feed GA's keyword indices, which retrieval does not use.

**D19. Node ids.** Upstream derives node ids from `len(id_to_node)`, which breaks once nodes can be removed.
MemeWorld re-keys them to monotonic `<agent>:m<n>` ids. It also replaces parentheses, because upstream `add_event`
rewrites any description containing "(".

**D20. Stochastic retrieval (`backend/memory/retrieval.py`).**
- Recency is time-based: `exp(−decay_rate·hours since last access)`. Upstream uses rank-based `0.99^i` over nodes
  sorted oldest-first, which gives the *oldest* node the highest recency score; this looks like an upstream bug.
- Selection samples k memories with P ∝ exp(s/τ) via Gumbel-top-k (seeded); τ = 0 reproduces GA's top-k.
- The pool includes *chat* nodes. Upstream `new_retrieve` only searches events and thoughts, but lossy
  conversation memories are the main carrier of social information.
- k ≈ 5 total instead of upstream's 15–50, split across focal points.
- Modules may re-weight scores before sampling.

**D21. Reflection.**
- Focal points: upstream `generate_focal_pt`, plus one rotating constrained question taken from the plan ("What
  patterns has X noticed recently?", "What has X learned about the people around them?", "Have similar situations
  happened … before?").
- Insights: upstream `insight_and_evidence` over *stochastically* retrieved memories (k = 8; upstream uses 30,
  deterministically), with 3 insights instead of 5.
- The trigger threshold is 60 instead of upstream's 150, because our days are shorter.
- No prompt asks for names, labels, slang or conventions.
- Upstream's post-conversation `planning_thought_on_convo` and `memo_on_convo` are replaced by the lossy
  conversation memory (D23).

## 4. Conversation

**D22. Conversation initiation.** A cheap stochastic gate runs before upstream `decide_to_talk`:
`p = base_talk_prob·(0.35 + familiarity)·sociability`, with a per-pair cooldown of 150 minutes and at most 6
conversations per agent per day. Modules can change `p`. The gate exists to keep LLM calls bounded; GA asks the
LLM for every co-located pair.

**D23. Conversation content.** `agent_chat_v2` is followed, with these changes:
- The relationship summary is generated once per participant per conversation (upstream: before every utterance).
- A conversation has at most 4 utterances (upstream: 8); the plan asked for 1–4 turns.
- Retrieval uses GA focal points [relationship summary, partner's activity, last lines], plus the private trigger
  observation when the conversation was started by a reaction.
- `curr_context` keeps GA's sentence for the initiator. The responder gets a responder-phrased sentence (upstream
  reuses the initiator phrasing for both).
- Appended context lines: a relationship statement, the speaker's own private observation (if any), and
  experimental-module lines.
- A REACT/TALK decision may supply the opening line.
- Bystanders in the same room overhear each utterance with probability 0.3 (×0.6 if busy). Only actual hearers get
  exposure records and memories.
- Afterwards, each participant encodes one lossy memory of the exchange as a GA `chat` node. Overhearers encode
  only what they heard. The node's `filling` (verbatim transcript in GA) is left empty, so exact wording survives
  only through the encoded memory.

## 5. Latent events (world layer)

**D24. Scenario scripts.** Latent event families are hand-written scenario scripts: beats with facts, salience,
visibility and roles, with slot fillers for surface variety. E1, E2 and E4 each have 7 training surfaces and 3 held-out
surfaces (E3 has 2 training and 1 held-out template, each with large slot pools). From
`latent_events.holdout_from_day` (default: day 3) only held-out surfaces are used; this is the generalization
test. An optional LLM surface generator exists (`latent_events.generator: llm`) but is off by default.
*Risk:* a finite template bank means surfaces repeat within a run, which may encourage phrase copying from the
world's own wording. The analyzer penalizes n-grams that appear verbatim in world text (D29).
*Superseded in v2* by the pre-generated world script (D51), 44 skins with one shared shape (D52) and holdout by
skin (D55). `holdout_from_day` and the LLM surface generator are removed; the v1 bank lives on in
`legacy_scenarios_v1.py` for old runs.

**D25. Forced movement.** Event beats force the protagonist, and involved secondary agents, to the beat's location
(event-triggered routine deviation). NPC roles ("a TA", "a student nobody seemed to know") fill roles when no
suitable agent is free.
*v2:* locations are resolved when the world script is generated, every mover of a beat is forced there, and
NPCs are used only when no suitable agent is free (D51, D54).

**D26. Event provenance.** The simulator tags each memory with the world events it came from (sidecar
`memory_meta.json`, never in a node or prompt). Conversation memories inherit the events of the memories the
speakers retrieved while talking. *Risk:* this over-attributes, because a retrieved memory may not actually be
talked about. Latent-event precision and recall are therefore approximate.
*Superseded in v2 for grounding* by typed provenance (D66). This retrieval-based provenance
(`retrieved_event_ids`) now feeds only the legacy evaluation.

## 6. Experimental modules (`backend/modules/`)

**D27. Emotion.**
- Appraisal uses a small valence lexicon (no LLM call).
- Effects: importance += round(gain × arousal × 4); a mood line is added to prompts ("Maya is feeling upbeat.");
  a visible partner mood is added in conversation; emotional contagion (0.2) happens between conversation
  partners.

**D28. Social reward.** `R_i = λ·Δvalence_j`, with the emotion module providing valence. It is implemented as:
1. The treatment statement in chat, reaction and decide-to-talk prompts: the agent "values making the people
   around them feel better".
2. A talk-probability boost toward people who seem down.
3. An importance boost for conversation memories that earned positive reward.

It never rewards phrases, spreading or adoption. **Prestige** defaults to normalized degree centrality when no
scores are configured. **Conformity** boosts memories corroborated by ≥ 2 distinct conversation partners, judged
from the agent's *own* chat memories (log(1+n) boost).
*v2:* social reward no longer needs or switches on the emotion module, and prestige gets `scores_source`
(D64).

## 7. Observer / analyzer (`backend/analysis/`)

**D29. Candidates.**
- Extraction: 1–4-grams with ≥ 2 speakers and ≥ 3 uses. N-grams that start or end with a stopword, or that
  consist only of names or places, are excluded.
- Score: log uses × (1 + log speakers) × novelty (out of `/usr/share/dict/words`) × nickname pattern × quoting,
  times 0.35 when the phrase occurs verbatim in world-generated text ("factual repetition").
- Near-duplicates are removed by subsumption, and variants are grouped by lexical, stem and embedding similarity.
- An LLM classifier (top 20) and one LLM discovery pass add labels and candidates.

**D30. Transmission.** Every exposure of B to m before B's first use is a candidate source. Its weight is
`exp(−Δh/12)`, ×2 if B's own encoded memory of that exposure kept the wording. Weights are normalized against an
"independent invention" prior of 0.15. Depth is the longest chain through each adopter's most confident parent.
An edge is cross-group when source and target share no group.

**D31. Semantics.** Context is the utterance ±1 line, embedded with the expression masked out. The analyzer
reports coherence (mean pairwise cosine), within- and between-group similarity, and per-day centroid drift.

**D32. Lineage.** Confidence = 0.45·lexical + 0.25·context + 0.15·temporal + 0.15·exposure overlap. An edge is
asserted only at ≥ 0.55 for candidates and ≥ 0.5 for variants; weaker links are kept with their scores.

**D33. Probes.** The top 5 candidates are probed. For each agent, the analyzer loads a copy of the final GA-format
memory, retrieves without touching access times, and asks the plan's meaning question. The generalization probe
is a 4-way multiple choice over held-out surfaces, one per family. Answers go only to `analysis.json`.

**D34. Evaluation.**
- Precision is the fraction of usages linked to the modal latent type. Lift is precision over the base rate of
  that type across all utterances.
- Recall is the fraction of later instances of that type that appear in any usage's provenance.
- Alignment is F1, or precision when recall is undefined.
- Generalization is the spontaneous-use rate on held-out instances plus probe-match accuracy.
- Drift is precision in the first half of usages versus the second half.
- Status: fading = no use in the last half day; established = ≥ max(3, N/2) users; spreading = a new user
  appeared in the last day; emerging otherwise.

Only linguistic conventions are analyzed. Behavioral conventions (for example, copied actions) are not detected.

*v2:* this block is kept in `analysis.json` as the labelled **legacy** evaluation. Spread is measured by the
emergence test (D67) and meaning by stratified grounding (D68).

## 8. Frontend (`frontend/`)

**D35. Frontend stack.** GA's Django + Phaser/Tiled frontend is replaced by a dependency-free canvas and SVG page
served by FastAPI. GA's character sprites are reused. Replay runs client-side from `frames.jsonl`. Normal demo
mode is enforced *server-side*: responses without `?debug=1` have latent types, scenario names, narratives and
event provenance stripped out.

## 8b. Changes made after the first real runs

**D36. Emotion module wording.** In the first social-reward run, arousal accumulated additively and saturated,
so "X is feeling a bit restless" was injected into ~90% of prompts. Agents then repeated it (a treatment
artifact). Now:
- arousal moves toward each event's intensity
- mood lines are added only for clearly positive or negative valence
- neutral states add no words

**D37. Observer word-frequency prior.** Candidate extraction now uses `wordfreq` Zipf frequencies:
- single words with Zipf ≥ 3.6 are dropped
- phrases made only of frequent words are down-weighted
- the LLM classifier covers the top 30 candidates instead of 20

This is observer-side only; agents are unaffected.

**D38. GA-format memory reload** uses a separate temporary id namespace. Forgetting leaves gaps in node ids, and
without it reloading could collide with them.

**D39. Event-script ontology fix (world-provided labels and narrator causation).** A review of the event
scripts found two ways the *world* was doing work the experiment attributes to *agents*:
- **Ready-made labels.** Some facts contained nicknames: "because of the whole backpack mix-up",
  "'extra crispy' cookies", "reply-all storm", "apology coffee". Two of these later showed up as "event
  nicknames" in the first report. They were world wording being repeated, not coinages.
- **Omniscient causal narration visible to any bystander.** For example: "Because {P} ate alone, {P} never
  heard...", "Hungry and distracted, {P}...", "The wrong coffee turned out to be decaf, and {S} fell
  asleep". This states the hidden causal chain outright.

The fix:
- Labels are replaced with plain descriptions.
- Each internal or remote causal link is split into an observable outcome (visible to all) and a `cause`
  fact visible only to the people involved (P or PS). Bystanders now see incidents. Anyone else learns the
  chain only by being told.
- Causal links that are physically observable in the same room, such as samples left out while P is
  locked out of the lab, are kept.

Families, beats, timing and holdout splits are unchanged. Runs before and after the fix are not directly
comparable; the controls are prefixed `c2_`. Old scripts: git history / `latent_events.v1.py` in the session
scratchpad.

**D40. Permutation null for latent alignment (observer).** Alignment/precision picks the best latent type
post hoc, so it is high by chance when a candidate has few linked usages ("definitely" scored 0.88). Each
candidate now also gets `evaluation.permutation`:
- latent-type labels are shuffled across the run's events 500 times, keeping type counts and usage→event
  links fixed
- best-type precision is recomputed each time
- the result is a p-value

`compare` reports `n_aligned_p05`, the candidates beating the null at p<.05; about 5% are expected by chance.
Re-scoring the five pre-fix Haiku runs gives **0 of about 208 candidates** at p<.05. The earlier
alignment/lift numbers were chance-level.
*Superseded in v2* as the grounding test by the stratified permutation null with BH-FDR (D68). The legacy
null remains, with best-type precision now a weighted share.

**D41. Observer model held fixed; agent model as an experimental factor.** `analyze --model` overrides the
analyzer's model. The controls vary the agents' model (Haiku vs Sonnet via the claude CLI) and analyze
every run with Sonnet, so classifier differences cannot masquerade as agent effects. New negative-control
condition `configs/no_events.yaml` (event_rate 0): conventions found there estimate the base rate from
chatter alone, and the classifier's false-positive rate for "event-driven" culture.

**D42. Observer-specific viewpoints (perception).** Before this change, every agent who noticed a fact got the
world's exact sentence, so differences came only from which facts got through. Now:
- Perception tags each noticed fact with a **vantage**: participant (it involves the agent), near (same room),
  distracted (same room but mid-conversation), or far (another room in the same building).
- It also records which of the people involved the agent **recognizes** (familiarity ≥ 0.3).
- A renderer (`backend/agents/viewpoint.py`, prompt `viewpoint_v1.txt`, one LLM call per observation, run in
  the per-agent phase) rewrites each fact as that agent perceived it:
  - participants get it as their own experience;
  - far observers only glimpse it or hear it from a distance;
  - distracted observers half-notice it;
  - people the agent doesn't know are described by appearance.
- The renderer may only omit, blur or reframe. It is told not to add events, causes or motives, and to return
  an empty string for anything that couldn't be perceived from where the agent is (that fact is then dropped).
- A hard regex guard removes the names of unrecognized people.
- The renderer sees only the facts this agent noticed: never the event, its hidden family, or another agent's view.
- The canonical text stays on the fact as `world_text` (simulator-only), and each rendering is traced as a
  `viewpoint` record.
- The observer's world-wording penalty now counts perceived renderings as world-provided text too.

This goes beyond GA, whose perception is a shared event triple.

**D43. Fix: participants remembered their own actions as a stranger's.** Every perception was encoded as
"saw", so for example Jordan's memory read "Jordan saw *someone* walking backwards up the stairs". Now:
- Observations containing facts that involve the agent are encoded as "experienced first-hand".
- They get a note that these facts describe what the agent did or what happened to them.
- The verbatim (noise 0) path uses "experienced:".
- The prompt is the new `encode_memory_v2.txt`; v1 is kept so older runs stay reproducible.

**D44. Stochastic memory reconstruction ("memory lens").** Through the CLI the model is close to
deterministic: an identical encoding prompt gave the same memory 3 times out of 4, and temperature cannot be
set (D1). So each LLM encoding now samples a lens from the agent's seeded RNG. It is recorded in the trace and
reproduced exactly on replay:
- **focus**, one of 7: people, what was said or done, the scene, how it felt, the oddest detail, what it means
  for their plans, who else was around;
- **length**, one of 3;
- **style**, one of 4;
- **fidelity jitter**: effective noise = noise + N(0, 0.2);
- **uncertainty** with p = 0.3;
- **one misremembered minor detail** with p = 0.5 × noise;
- **association** with p = 0.35: one related older memory, drawn by stochastic retrieval, is shown as
  "it brought to mind…", so it can colour the new memory.

Mood words are deliberately not sampled; see D36 for how injected mood words became fake memes. The requested
temperature is now 1.0 (it only has an effect on the API backend). `memory.encoding_variability` scales all of
this (0 = off).

Measured on one live Sonnet observation, 6 encodings:
- lens off: 5 of 6 distinct, mean pairwise word overlap 0.62;
- lens on: 6 of 6 distinct, overlap 0.39, including one distortion ("down the stairs").

**D45. Lower attention and a crowd penalty.**
- `base_attention` goes from 0.75 to 0.45, and `familiarity_weight` from 0.35 to 0.25.
- New `crowd_factor` 0.92: attention × 0.92 for each other agent in the building beyond 2.
- Reason: at crowded meals all 8 agents noticed nearly every fact, with attention 0.5–0.9.
- Participants still notice their own facts with probability 1.

Together, D42–D45 change the world model: runs after them are the `c3_` series and are not comparable to
`c2_` or to the seed-42 runs.

### Changes after the pipeline diagnosis (D46–D50)

The diagnosis of runs through c2 found the pipeline breaking in three places:
- Events were referred to too few times to need a name.
- Almost no agent ever connected two incidents: 0–6 reflections and at most 19 utterances per run mention two
  different events.
- The exact-wording channel was weak, and retrieval was crowded by static relationship facts and routine
  sightings.

The changes below target those points. None of them tells agents to invent words, names the hidden
structure, or uses analyzer output.

**D46. Clustered recurrence (world).** At the start of a run, each hidden family is assigned a *home* friend
group from the population's groups. The assignment uses its own RNG stream, is written to the manifest as
`family_home_groups`, and is stripped from the demo API. With `latent_events.clustering` = 0.7, an instance's
protagonist, and where possible its related person, are cast from that group. So the same kind of thing keeps
happening within one circle, which gives that circle both the repeated occasion and the audience for talking
about it. Surfaces, timing and families are unchanged.
*Superseded in v2 by D54.* Family and group were confounded, so grounding could not tell family-specific talk
from group-specific talk. `latent_events.clustering` and `family_home_groups` are removed; explicit circles
and assignment modes replace them.

**D47. Daily catch-up conversations (conversation).** From 17:00, close pairs (familiarity ≥ 0.6) who are in
the same room get one evening catch-up per day, with probability 0.8. It skips the stochastic gate, runs up to 6
utterances, and gets two additions:
- a context line "X and Y are catching up on how things have been going lately";
- an extra retrieval focal point, "what has happened lately that stood out".

No topic or event is supplied, so what comes up is whatever the agents retrieve. This is the recurring shared
context where past incidents get brought up again.

**D48. Spontaneous reminding (memory).** After encoding a salient perception (≥ 0.45), the agent is shown about
7 of its own earlier experiences:
- 4 from stochastic retrieval on what it just perceived;
- 3 salient older ones sampled by importance, so structurally similar but lexically different incidents can
  surface despite the lexical embedder;
- only memories of seeing or hearing something, older than 90 minutes.

It is asked whether the new experience reminds it of one of them ("often nothing comes to mind" is stated). A
positive answer becomes a GA thought node with both memories as evidence ("… It reminded X of an earlier time:
… What felt alike: …"). This is the step where two incidents can become connected; before, nothing performed
it. The prompt is `reminding_v1.txt`, there is one LLM call per salient perception, and each call is traced as
`reminding`. This goes beyond GA, which only links memories through reflection.

**D49. Verbatim stickiness of distinctive wording (memory).** When an agent encodes a conversation or something
overheard:
- Up to 2 distinctive phrases are picked from what was said: 2–3 word spans around a rare word
  (wordfreq Zipf ≤ 3.6), with no person names.
- Each sticks with p = 0.3 + 0.2 × (number of the agent's existing memories that already contain it), capped at
  0.9. Wording the agent has met before sticks more readily, with no extra state.
- A stuck phrase must be kept word for word in quotes. If the model drops it anyway, it is appended to the
  memory.

This counterbalances D44's paraphrasing: gist memory stays, while a small channel for exact wording stays open.
It uses the same frequency resource as the observer (wordfreq), but independently: nothing from the analysis
reaches agents.

**D50. Retrieval source weights.** `retrieval.source_weights: {seed: 0.5, ambient: 0.5}` halves the retrieval
score of seed relationship facts, which duplicate the relationship lines already in every prompt, and of ambient
routine sightings ("X is eating lunch"). Together these made up 50–70% of what speakers recalled before
speaking. Ambient memories now carry their own meta `source_type` ("ambient"). An empty map restores GA
behaviour.

*v2:* D47–D50 are now **off by default**, and each is switched on by one NEED, LINK or WORDING key (D72).
The default `source_weights` is `{}`. D59 extends D48, and D58 fixes D49.

### Ontology v2 (D51+)

v2 ([ONTOLOGY_V2.md](ONTOLOGY_V2.md)) reorganises the simulator around one working hypothesis: a shared
expression for a recurring pattern forms only if three bottlenecks are passed. **NEED**: the same people
repeatedly need to refer to the same thing. **LINK**: someone connects separate instances as the same kind of
thing. **WORDING**: a specific wording survives lossy memory and is reused. Every mechanism has its own switch,
the world is identical across conditions, and the observer measures *emergence* (spread through exposure)
separately from *grounding* (tracking the hidden structure beyond chance). The hard constraints are unchanged and
enforced by tests. The world generator, the random streams and the defaults all changed, so v2 runs are not
comparable to the seed-42, `c2_` or `c3_` runs.

**Research direction (2026-09-19).** After v2 was built, the research direction changed
([memo](research/MEMO_2026-09-19_research_landscape.md)). The primary first study is now the memo's question:
when a community's circumstances change, how do its existing ideas acquire new meanings, and when does cultural
memory help or obstruct that adaptation? (For example, do persistent shared records preserve meanings across
member turnover?) It will be specified in `docs/ONTOLOGY_V3.md`. The v2 bottleneck factorial becomes
**calibration** and a secondary study. It checks that the observer detects a convention that really spreads
(planted control), finds no grounding without structure (scrambled/none, no events), and shows which mechanisms
actually fire. Compute: Haiku agents and a Sonnet observer, both via the claude CLI.

**D51. World script and common random numbers (world).** All latent-event instances are generated *before* the
run (`simulation/world_script.py`) from streams seeded only by `world_seed` (null → `seed`). The world therefore
never depends on agents, LLM output or which mechanisms are on, and conditions sharing `world_seed` see identical
events. There is one stream per purpose (timing, family, skin, cast, slots, referent, holdout, compose, and
`link` for D56; helper `rngs.seed_rng`), and each candidate tick takes a fixed number of draws from every
stream, so switching one world mechanism never shifts another's draws. "Current" locations resolve at generation
to the actor's *planned* position, through the same `plan_day` call and seed the engine uses. "Busy" comes from
overlapping scheduled instances, not live state, and no instance crosses a day end. The engine writes
`world_script.jsonl` at start, releases instances into `events.jsonl` at their start tick, and forces every
mover of a beat to the scripted place (v1 forced only P, or everyone in pinned beats). `latent_events.script_from`
reuses a saved script; `generator: llm` is removed and raises an error. *Risk:* beat locations still depend on
`seed` through plans (designs set `seed = world_seed`), and runtime deviations such as MOVE reactions and
invitations can take bystanders off plan.

**D52. One structure shape, 44 skins, topic-balanced domains (world).** A family (E1–E4) is now an abstract
causal structure (`structures.py`), and a *skin* is one concrete wording of it (`skins/e1.py`–`e4.py`). All
families share one shape: `n0` (P, public, salience 0.55), `n0_private` (P's inner view, P only, 0.30), `n1`
(the second actor, +2 ticks, 0.50) and `n2` (P, +4 ticks, 0.55). The second actor is a related S, or an
unrelated Q in E3. Each family has one skin per topic domain, 8 train (dining, coursework, lab, transit, gym,
dorm, cafe, clubs) and 3 holdout (library, quad, tech), and only `n0` may be pinned to a building, in the same 7
domains for every family. So neither shape, topic nor place carries family information. `structures.validate_skins`
(run in `tests/test_world_v2.py`) enforces the shape, 8–25 words for every referent × slot combination, no
narrator causal glue across facts (including "unluckily" and "coincid*"), no nickname heads or quoted labels,
and no content bigram shared across families or between a family's holdout and train skins. The skin writers
carried each structure through concrete detail, never a stated link: one shared object (E1), complementary error
pairs with a neutral outcome that is never a gain (E2), the same unusual act in different words (E3), and a
leftover of the failure followed by a plain gain (E4). The v1 bank moved to `legacy_scenarios_v1.py`, and
`LATENT_TYPES` gains E0 (plain mishaps). *Risk:* the 0/2/4-tick shape compresses some outcomes (E4 gains that
would take days), role plausibility is unchecked (a non-lab S can get the lab skin), and tense and naturalness
with every referent still need human review.

**D53. Referents (world, NEED).** A *referent* is a persistent world thing that instances can recur around, for
example "the night shuttle". Referents belong to no family; `referents.py` holds 5 per domain, and `{R}` appears
once in `n0`. When `latent_events.referents.enabled` is false, each instance draws its domain's referent
uniformly, so repeats happen only by chance. When it is enabled, the protagonist's own circle (for the free
pool, one population-wide memory) remembers the referents it has met in each domain and reuses one with
probability `reuse` (0.6). The memory is updated for every instance but consulted only when enabled, so draws
stay aligned across conditions. Instances record `referents: [{id, domain, name, reused}]`.

**D54. Circles and assignment modes (world, NEED; replaces D46).** The population file gets disjoint `circles:`
of at least 3 members, validated against the whole file. A circle that `population_size` cuts below 3 dissolves
into the free pool. homewood8 has `c_lab` (maya, dev, hana) and `c_dorm` (ethan, leo, jordan); priya and sofia
are free. `latent_events.assignment.mode` is one of three:
- `none` (default): P is uniform over non-busy agents;
- `balanced`: each instance gets a circle, round-robin from a `world_seed` offset, and with probability
  `strength` P (and S where possible) is cast from it;
- `home`: families map to circles, rotated by `world_seed % n_circles`, so the mapping is counterbalanced across
  seeds.

Instances record `circle` and `cast_from_home`; the latter is true only if the strength draw succeeded *and* a
member was free. S is a related agent (familiarity ≥ 0.3), from the circle where possible. Q (E3) is never cast
from the circle, whose members are related by construction. NPCs are only a fallback (v1 sometimes used them when
agents were free). *Why:* under D46, family and group were confounded, so grounding could not separate
family-specific from group-specific talk. On the clustered `dev_live_c3`, 14 of 26 tested candidates came out
"grounded". `balanced`, which the NEED factor uses, keeps recurrence inside circles with no family–circle
correlation.

**D55. Structure controls and holdout by skin (world).** `latent_events.structure` is `real`, `scrambled` or
`none`. A composed instance keeps the shape and roles, but takes `n0` (with `n0_private`), `n1` and `n2` from
three different skins of any family that share its holdout flag. Each part gets a referent from its own domain,
and `composed_from` records the sources. Under `scrambled` the label is the same family draw the real condition
makes, so real and scrambled share labels and casting and differ only in content: a positive-null calibration
for grounding. Under `none` every label is `E0`, but the drawn family still picks the circle, so casting matches
`scrambled`. Held-out surfaces are whole skins in the three holdout domains, used with probability
`holdout_frac` (0.25) per instance from day 1. This replaces `holdout_from_day`, so holdout is no longer
confounded with day. Probes use one held-out v2 skin per family, with neutral roles. *Risk:* composed stories
can read incoherently, and that must not be read as structure.

**D56. Link visibility (world, LINK).** `latent_events.link_visibility` (default 0) is the probability that an
instance's `n0_private` fact (`kind: "inner"`) is visible to everyone present instead of P only. It is drawn
from its own `link` stream. This turns D39's rule (inner causes reach only the people involved) into a
manipulated variable; the LINK factor uses 0.3.

**D57. Agent random substreams (agent).** `Agent.stream(name)` is a lazily created generator seeded by
(`seed`, agent id, name). The names are `perceive, ambient, encode, lens, assoc, verbatim, remind, react,
reflect, talk, remark, need, prime`, plus `legacy`, which now backs `agent.rng`. Each mechanism draws only from
its own stream (a test fails if lens and verbatim share one). The lens association has its own `assoc` stream,
because its Gumbel draws scale with the size of the memory pool. The engine passes a named stream at every call
site. Engine-level draws use `seed_rng(seed, "talk", tick)` for social gating, catch-ups and group scheduling,
`seed_rng(seed, "invite", conversation_id)` for invitations, and `seed_rng(world_seed, "topology")` for D62.
Designs set `seed = world_seed`, so agents' non-LLM draws are common across cells too. *Risk:* this aligns random
numbers, not trajectories. Once a mechanism changes what an agent says or remembers, later behaviour diverges.

**D58. Verbatim stickiness fixed; Wording records (agent, WORDING; revises D49).** With
`memory.verbatim.enabled`, three rules change. `exclude_self`: only other people's lines are candidates (in a
pre-v2 run, 17 of 60 stuck phrases were the agent's own), and conversation facts now carry a `speaker` field.
`distinctiveness: lexicon`: the rare anchor word must also be outside the *population lexicon*
(`simulation/lexicon.py`: places, rooms, routine activities, classes, clubs, habits and backgrounds; written to
the manifest), so campus vocabulary is not "distinctive"; `zipf` keeps the old rule. Clean spans: 2–3 tokens
inside one clause, with content words at both edges. Function words, light verbs, hedge or -ly adverbs and
contractions can't sit at an edge, which removes spans like "rewriting all those". Only population names are
excluded, so a capitalised coinage can still stick. Referent names are not excluded, because agents cannot know
which wording came from the world; the observer's `in_world_text` flag handles that. Stuck phrases become
Wording records in `MemoryMeta.wordings` (`phrase, heard_from, utterance_id, tick, time, self_produced`), kept on
the surviving node when an encoding merges. With `render_in_text` (the default), they are also quoted in the
description. *Risk:* some -ly adjectives, such as friendly or costly, can no longer be span edges.

**D59. Reminding v2 and typed links (agent, LINK; extends D48).** Reminding can now also fire after conversation
and overheard memories (`reminding.on_sources`). `min_salience` gates perceptions only, because conversation and
overheard salience are fixed constants. Earlier reminding thoughts can be candidates, so links compound into
chains. The new prompt `reminding_v2.txt` (v1 kept) first asks whether the experience was ordinary, in which case
no link is made. It says that place, people or topic alone are not enough and that usually nothing comes to
mind, and it contains no naming or label wording. With `reminding.reflection_weight: 0.0`, reminding thoughts no
longer feed GA's reflection trigger; one live run had about 198 reflections a day. Every link (reminding, lens
association, merge, reflection evidence) is stored in `MemoryMeta.links` and traced as `memory_link` with
`from_event_ids` and `to_event_ids`, so same-event and cross-event links can be told apart. A merge counts as a
link, because the new experience was taken for the old one. *Cost:* up to one extra LLM call per conversation
participant and per overheard observation.

**D60. Open matters (agent, NEED).** With `need.enabled` (`memory/need.py`), a perception the agent took part in,
or any memory with importance ≥ `min_importance` (6), becomes an open matter in `agent.open_matters`. This is
ordinary agent state whose text is the agent's own memory, not meme state. Strength halves every
`half_life_hours` (24); matters below 0.1, or whose memory is forgotten, are dropped, and a merged re-encoding
refreshes its matter. When the agent talks, the top `max_open` matters become extra retrieval focal points, and
the first adds the line "`<name>` still has on their mind: …" (computed once per speaker per conversation). A
matter whose memory is retrieved into an utterance is multiplied by `discuss_decay` (0.5). Traced as
`open_matter` and `open_matter_focal`. An unresolved incident gets a reason to come up again, and no topic is
named.

**D61. Production priming (agent, WORDING).** With `priming.enabled`, the conversation context gets the line
`Things <name> has heard people say lately: "a", "b".`. The phrases are up to `max_phrases` (3) of the agent's
own Wording records from the last `window_hours` (24), excluding self-produced ones. They are sampled from the
`prime` stream, weighted by retrieval recency summed over hearings, once per speaker per conversation, and traced
as `priming`. The line only reports what the agent heard. It does not ask the agent to reuse or coin anything,
and it reads the agent's own memory sidecar, never observer output.

**D62. Generated topology (social).** `topology.mode: generated` (default `file`) replaces the file's
relationships with a generated network, seeded from `world_seed` so every condition shares it. Agents form
`min(n_circles, N//3)` disjoint circles. Households (same home and room) stay together as `roommate`s, because
their profiles already imply that tie. Within a circle, every pair gets a `friend` tie at exactly the configured
strength; there is no jitter, so the knob is the manipulation. Between circles, pairs become acquaintances with
probability `p_between`, and `n_bridges` agents get one strong tie into another circle. With `shared_meals`, each
circle's dinner replaces its members' own dinners (same venue and time, 60 minutes, inside the evening
group-talk window), so co-location agrees with the ties. Generated circles replace the file's `groups`. The
manifest records topology metrics (density, familiarity-weighted modularity, bridges) in both modes. *Risk:*
profile text still names specific people ("does wet-lab research with Dev") and can contradict generated ties,
and routine jitter means shared meals are only approximately co-located.

**D63. Multi-party conversations (social, NEED).** GA has only dyadic chat. With `conversation.group.enabled`,
the engine holds at most one group conversation per (day, window, venue). It starts with probability `prob`
(0.5) when at least 3 free, awake agents are at the venue, and up to 5 take part; the defaults are Dining
Hall/Main Floor at 12:00–13:30 and 18:00–19:30. `group_conversation.py` is the smallest generalisation of
`run_conversation`. Speakers go round-robin from a random start, up to 8 utterances, and the model's `end` is
honoured only after everyone has spoken. The prompt `group_chat_v1.txt` is GA's `iterative_convo_v1` with the
whole table named. Relationship statements replace GA's per-pair relationship-summary call, which would cost
N(N−1) calls. Past context is the latest chat with anyone at the table from the last 8 hours (upstream checks
the oldest chat node). Listeners are the other participants plus bystanders who overhear; every participant
encodes a memory; no invitation follows. Emotional contagion uses the mean of the others' valence, which is the
old rule for a dyad. *Cost:* up to 8 turns with GA's 3-attempt retry, plus N encodings and reminding checks.

**D64. Independent pressure modules with manipulation checks (controller; replaces the coupling in D28).**
`social_reward` no longer switches on `emotion`. On its own, it estimates each partner's apparent valence by
applying the emotion lexicon to what they say, and writes nothing about moods into prompts. The reward is λ ×
the mean change in partners' valence; with `emotion` on, emotion's valence is used instead.
`prestige_bias.scores_source` is `degree` (default), `random` (a seeded permutation of the degree scores: same
status distribution, no link to network position) or `file`. Scores set with a source other than `file` raise
an error. `ModuleStack` counts an effect whenever a hook's output differs from its input, and
`manipulation_checks()` goes into the manifest. *Risk:* social reward alone gets a sparse signal. In a mock run,
total reward stayed 0 and only the prompt treatment and the talk boost were active; the manipulation check shows
whether it fired.

**D65. Planted-phrase positive control (controller).** `controls.planted_phrase: {agent, habit}` appends a habit
sentence (`Maya has a habit of calling any mess "a full pickle".`) to one agent's GA lifestyle line. It is used
only in the `planted` control cell, to check that the observer detects a convention that really spreads; no
treatment cell uses it. It is a persona habit, not an instruction to coin or spread anything. The population
lexicon is computed *before* planting, and the observer removes planted-only tokens from `in_lexicon` and from
world text, so the control is not discounted as vocabulary or world wording.

**D66. Typed provenance (observer; replaces D26 for grounding).** `analysis/provenance.py` gives each utterance
`referent_event_ids`: events whose text shares at least 2 content tokens with it (not a stopword or a person's
name, Zipf < 5, at least 3 letters). Only text released by the utterance's tick counts: facts, viewpoint
renderings and referent names. The trigger event of a reaction remark, or of a conversation a reaction started,
is added too. It also reports `direct_event_ids` (from retrieved perceptions) and `carried_event_ids` (from
retrieved conversation, overheard, reminding and reflection memories). Grounding uses `referent_event_ids` only.
D26's `retrieved_event_ids` over-attributes, so it now feeds only the legacy evaluation. Referent names count as
world text, so reusing them is down-weighted like other factual repetition (D29). *Risk:* linking by shared
words is coarse. There is no stemming, common campus words can link routine talk to events, and every turn of a
triggered conversation inherits the trigger. The permutation null keeps chance links symmetric across families,
but funnel rates are sensitive to them.

**D67. Emergence test (observer).** `analysis/emergence.py` measures spread separately from meaning. The
originator is the first speaker. An agent is *exposed* if it heard a usage before its own first use, or heard one
and never used it. *Adopters* used it after exposure; *independent* users used it with no prior exposure. A
one-sided Fisher exact test compares the two rates. The rule is `emerged = n_adopters ≥ 2 and n_adopters >
n_independent and not in_lexicon`, where `in_lexicon` means every content word is population vocabulary.
`in_world_text` and `n_adopters_carried` (used in a different conversation from the one where it was heard) are
reported but are not part of the rule. *Risk:* an echo inside the same conversation counts as adoption, and event
wording can "emerge" ("submitted to the wrong" in `dev_live_c3`). Read `emerged` together with `in_world_text`.

**D68. Stratified grounding with BH-FDR (observer; replaces D34's alignment and D40 as the grounding outcome).**
Each usage linked to an event gets a fractional family distribution. The statistic is the weighted share of the
post-hoc best family, which replaces the `any()` hit. The null permutes family labels 1000 times *within
circle × day strata*, keeping every usage's links fixed, with one seeded permutation matrix shared by all
candidates. Benjamini–Hochberg q covers the n-gram candidates with at least 3 linked usages and 2 linked
speakers, a filter that never looks at labels. LLM-discovered candidates get a p-value but stay out of the FDR
family, so `n_grounded` does not depend on the observer LLM. `grounded = q < 0.1 and linked_usages ≥ 3 and
linked_speakers ≥ 2`, and E0 runs get p = 1. A run whose families cluster by circle but whose events carry no
circle (v1) gets a `warning`; v1 grounding is not evidence. Holdout generalization, and probe accuracy against
chance for exposed vs unexposed agents, are also reported. The legacy `evaluation` block remains and now also
uses the weighted share, so re-analysing old runs changes the legacy p-values.

**D69. Funnel, manipulation checks and validity (observer).** The metrics from `scripts/diagnose_funnel.py` moved
into `analysis/funnel.py` and now use `referent_event_ids`; the script is a wrapper. Manipulation checks are
counted from trace records, one per mechanism: viewpoint drops, lens distortions and associations, stuck verbatim
phrases (self vs other), reminding asked and linked (same-event vs cross-event), catch-ups, group conversations,
priming lines, open matters, `cast_from_home` and referent-reuse rates, and module counters. Each mechanism is
labelled `off`, `active` (it fired) or `inactive` (it was on but never fired). An effect, or its absence, is
always shown next to whether the manipulation happened.

**D70. outcomes.json and a fixed observer spec (observer; extends D41).** `pipeline.analyze` writes
`outcomes.json`, a fixed outcome vector that does not depend on the LLM classifier. It holds the run and
condition ids, seeds, `code_version`, `observer`, `n_candidates`, `n_emerged`, `max_emerged_adoption` (over
emerged candidates, the maximum of (1 + adopters) / population), `n_grounded`, `n_grounding_tested`, the emerged
and grounded lists, `funnel`, `manipulation` and `validity`. Counts exclude LLM-discovered candidates. The
classifier verdict is only an annotation (`llm_is_convention`), and probes go to emerged candidates first. The
observer spec is `{backend, model, analysis_version}`; the version is `v2`, and older analyses count as `v1`.
`analyze` and `run --analyze` both use `analysis.observer` unless `--backend` or `--model` is given. `compare`
refuses to pool runs with different specs; `/api/compare` passes `allow_mixed_observers=True`, so the UI still
lists every run with its observer.

**D71. Experiment designs and the `design` CLI (controller).** A design file (`configs/designs/<name>.yaml`) holds
a base config, seeds, optional `days`, a fixed `observer`, an optional `common` overlay, factors in a full
factorial (each level is a config overlay), controls (a plain overlay, or `{levels, set}` for a factorial cell
plus one change) and pre-registered `outcomes`. Loading rejects two factors that set the same key or a prefix of
it, since merge order would silently decide the value, and rejects overlays that set `seed`, `world_seed` or `_`
keys. It normalises YAML's boolean on/off level names and warns about unknown config keys, with hints for removed
v1 keys. Each cell × seed runs in its own subprocess, with `seed = world_seed = s`, into
`<runs_root>/<design>/<cell_id>/s<seed>`. Runs go seed-major, so an interrupted design leaves complete
replicates. A `design_cell.json` sidecar gives six statuses (missing, running, failed, finished, stale,
analyzed). Failed dirs are renamed before a retry, new runs claim their dir atomically, and runs analysed by a
different observer spec are re-analysed, never pooled. `--backend mock` writes to `runs/<design>__mock`. The
runs root is `--runs-root`, else `MEMEWORLD_RUNS_ROOT` (also used by `run`), else the design's `runs_root`, else
`runs/`. `design table` reports per-cell mean ± sd over seeds, with an `inactive` column. The manifest now records
`condition`, `code_version` (git sha, dirty flag, prompt hashes), `world_script_sha256`, `population_lexicon`,
`topology` and `manipulation`. Two designs ship. `bottleneck_factorial` crosses NEED × LINK × WORDING ×
real/scrambled (16 cells), plus `no_events` and `planted` controls, × 3 seeds = 54 two-day runs, with Haiku
agents and a Sonnet observer. `smoke_mock` is used by the tests. *Risk:* `--parallel N` means up to 4·N
concurrent claude CLI calls.

**D72. Mechanisms off by default; baseline = the plain cognitive model (controller).** `configs/default.yaml`
switches off every meme-formation mechanism and every module, and `baseline.yaml` is identical to it. What
remains is GA-style perception, memory, retrieval, reflection, reactions and conversation, plus viewpoints (D42)
and the memory lens (D44). Each mechanism is switched on by exactly one key and writes trace records for its
manipulation check. NEED: `latent_events.referents.enabled`, `latent_events.assignment.mode`,
`conversation.catchup.enabled`, `conversation.group.enabled`, `need.enabled`. LINK: `reminding.enabled`,
`latent_events.link_visibility`. WORDING: `memory.verbatim.enabled`, `priming.enabled`,
`retrieval.source_weights`. Compared with `c3_`, D46 is removed and D47–D50 are off, and `social_reward.yaml`
changes one pressure only (D64). Existing prompt files are never edited. New behaviour gets a new versioned file,
and the manifest hashes every prompt.

**D73. Hidden state in v2 (all layers).** Demo mode also strips the new world fields (`schema, skin,
composed_from, structure_mode, circle(s), cast_from_home, assignment, world_script_sha256`), link event ids, and
the observer's grounding, funnel and provenance fields. A mock check of 248 prompts found no skin key, schema id
or circle id. Agent-side mechanisms read only the agent's own memory and sidecar, or world-side resources
(`simulation/lexicon.py`). `validate_skins` keeps its own copy of the observer's stopword list, because
simulation code must not import the analysis package.

**Known open issues after the v2 build.**
- v2 has run end to end only on the mock backend. A live Haiku smoke run (`runs/dev_live_v2`) was in progress at
  the time of writing.
- The world script depends on `seed` through planned beat locations (D51). To vary `seed` with `world_seed` held
  fixed, both the engine and the world script would need plans seeded by `world_seed`.
- Some E4 skins give S a small invented attribute (a band, a spare ticket), which can contradict S's profile.
- Observer probes, and emergence when the manifest has no lexicon, rebuild profiles from the population file,
  without generated topology or planting.
- `group_conversation.py` imports private helpers from `conversation.py`.

**D-proposed (observer; number to be assigned by the integrator): candidate quality, system wording, three
tiers.** Mock runs showed "conventions" such as "remembers that ethan", "maya chen are roommates" and "labmates
they know each". Those were memory frames and relationship lines the mock LLM had stitched into its dialogue,
and a mock judge then labelled them. Four changes follow:
(1) *Well-formedness* (`analysis/wording.py`, used by `candidates.py` and so by `live.py`). N-grams are counted
only if they are plausible reusable units. Person names (manifest, population file, any case for a full name;
Zipf ≥ 4.5 names such as Miles or Park only when capitalized) are allowed only inside nickname constructions:
"the Leo thing", "pulling a Maya", "classic Priya", "Leo-proof". Also rejected:
- a function word, auxiliary, pronoun, reporting verb or time word at either edge;
- a reporting frame ("remembers that", "thinking about how");
- an auxiliary frame at the end ("is preparing", "are friends");
- bare relation, role or place nouns;
- digits and clock times ("nine-thirty");
- spans across punctuation, quotes or *stage directions*;
- unigrams whose lemma has Zipf ≥ 3.6.
(2) *System wording* (`wording.Infrastructure`). This corpus holds what the system put into agents' heads:
- relationship lines and templates;
- routines and day plans;
- seed and ambient memories;
- memory, reminding, need and priming frames;
- the situational lines of conversation contexts;
- profile text (minus the planted habit), place names, NPC roles and co-op roster, menu and binder text;
- the population lexicon.
A verbatim, name-slotted ("template") or inflection-folded (multi-word only) match gives the new status
`system_wording`, which ranks below `planted` and above `world_wording`. It is down-ranked ×0.3 and blocks
`emerged` (`emergence.in_system_text`). Phrases used mostly inside ≥ 8-token runs copied from the speaker's own
earlier memory or reflection text are flagged (`recited_share`) and down-ranked.
(3) *Tiers* (`analysis/tiers.py`, used the same way in analysis.json, outcomes.json, live and report):
- `candidate`: frequency only;
- `spreading`: ≥ 1 carried adopter, not system or world wording;
- `convention`: emerged, plus a real judge verdict `is_convention=true`, plus not system or world wording. The
  planted phrase can reach it only as the control, and it is never counted.
(4) *Judge provenance* (`judge.is_real_verdict`). Mock or replay verdicts are placeholders.
`llm.is_convention` is null for them and the mock answer is kept under `llm.placeholder`. Judge files say
`placeholder: true`, and summaries say "no real judge has run". An old analysis.json's legacy `llm` block counts
as a prompt-v0 verdict, with its provider/model read from analysis_llm_calls.jsonl. When the observer is mock,
`pipeline_spec` now forces a mock judge. Since `analysis.judge` became a default, `analyze()` on a mock run had
been calling the `claude` CLI. The UI card keeps the lifecycle in `card.status`. `card.is_convention` is null
without a real verdict, and true only for the convention tier. Per-run top-25 lists on the real runs are in
`runs/test_candidate_quality/`.

**D-proposed (observer; same number as the previous entry): the list was still not culture.** A review of all
300 top-25 rows of the twelve archived real runs found the old `remembers that Sofia` junk gone, but what
replaced it was course vocabulary, the actor model's small talk, and paraphrases of routines and events:
roughly 4% of the rows were genuine local expressions, and both expressions a real judge had ever accepted
(`frisbee analogy`, `forty-seven steps`) had fallen outside the top 25. Twelve further changes:
(1) *Units, not windows* (`candidates.not_a_unit`). An n-gram whose occurrences are followed by the same
content word ≥ 75% of the time is a truncation (`coffee sounds` ← "coffee sounds great"), and one preceded by
the same non-determiner function word ≥ 80% of the time is a collocation stub (`least you caught` ← "at least
you caught"). A following preposition or copula does not count (the tray return *to* my table is a unit), and
fewer than three occurrences say nothing. n now runs to 5, so a coined label is published whole ("The
Brooks-Chen Cookie Experiment", not "the brooks-chen cookie").
(2) *System wording on a content-lemma bag* (`Infrastructure.bag_match`). Matching strings let every
paraphrase through; the bag matches when all of a candidate's content lemmas sit in one system segment, with
inflections folded and clipped forms expanded (econ → economics, ml → machine learning, meetup → meet up). A
one-content-word candidate matches only routine/profile/place/NPC/co-op segments.
(3) *World wording on event lemmas* (`emergence.world_lemma_match`). A paraphrase sharing ≥ 2 content lemmas
(one of them not ultra-common) with a single event fact is world wording, and so is an incident label built
on the event's own noun (`backpack situation`); a label that brings a word of its own does not match
(`inbox apocalypse`). `live.mark_incident_talk` then collapses the rest: expressions whose uses (≥ 75%) all
link to one referent event keep one representative, the others are flagged `incident_talk`. Coinages
(meta-marked, quoted, novel, or with high collocation surprise) are never collapsed.
(4) *The actor model's register is the null hypothesis* (new `backend/analysis/register.py` +
`data/register_background.json`). One LLM writes every agent, so its stock wording spreads in every run by
construction. The table holds the n-grams of the twelve archived runs in two layers (≥ 2 speakers, and merely
present), looked up **leave-one-out** so a run never counts towards its own background. A phrase two other
runs share with several speakers, or three other runs contain at all, plus a short stop-list of openers and
closers, is `ordinary`: it can never be `emerged` and never reaches the convention tier (`in_register`,
`tiers.classify_status(ordinary=)`, `tier_of(ordinary=)`). This is what keeps `conditional probability`,
`coffee sounds`, `juggling` and `lifesaver` out; measured on the corpus, every reviewer-identified genuine
coinage has background 0.
(5) *Ranking by collocation surprise, not word rarity.* The old prior multiplied by (5.5 − mean word Zipf),
which gave technical single words the ceiling and cut coinages of everyday words to 0.2× (`style points` was
not even in the 60-item pool). `CandidateExtractor.surprise` scores the combination — the phrase's rate in
the run against the product of its words' general-English rates — shrunk towards no-information below six
occurrences. Novelty is judged on the LEMMA (`memorizing`, `napkins`, `recopied` are ordinary words; the old
test asked /usr/share/dict/words, which has no inflected forms, so half the vocabulary got the bonus).
Metalinguistic marking (quoted, or a ratification in the same or the next turn: "is sending me", "we're
calling it", "I'm stealing that") and use across conversations add explicit boosts; `live.record` then
multiplies by the run's own transmission evidence (carried adopters).
(6) *Word classes.* Added: deictic time tails (earlier/lately/recently/…); spelled-out quantities and measure
phrases (`fifteen minutes`, `sixty bucks`) while an idiosyncratic exact number stays (`forty-seven steps`);
clauses with a finite copula or modal inside (`pasta is decent`); personal (non-possessive) pronouns
(`giving you the most trouble`); elongations (`pfff`). Recovered: the label construction `<content> thing`
(`the autopilot thing`, `Maya's frisbee thing`) — the most productive local-referent frame in the corpus,
previously killed because `thing` is a stop word; an evaluative adjective opening a longer coinage (`great
coffee catastrophe`); nickname forms with a head noun (`classic Ethan move`, `the Brooks-Chen cookie
experiment`); `-y` inflections fold before the rarity test (`funniest` → funny). `wording.run_names` no
longer takes NPC descriptor phrases ("a lab technician") as person names — they are `npc` system wording, so
`lab slot` and `dorm room` stop being rejected as names.
(7) *Grouping.* Lemma-identical forms and single-word morphological variants merge unconditionally (the old
rule demanded 30% shared utterances, which disjoint conversations never have). Everything else needs a shared
head plus shared content and half the uses of the bigger variant, measured against that variant rather than
the growing group — so a shared modifier cannot fuse `hike sounds` + `coffee sounds` + `sounds perfect`, and
a containment chain cannot swallow half a run. The canonical form is the best-scoring variant that is NOT
system wording, the group's score is its BEST variant (productive variation is evidence a coinage is alive,
not a reason to average it down), and its wording flags come from the share of USES that match — so junk
cannot hide inside a clean group, and one routine-wording variant can no longer rename and disqualify a
genuine one (which is what buried `frisbee analogy` inside `frisbee at the quad`).
(8) *Buckets* (`candidates.bucket_of`, `snapshot.bucket_counts`). The pool is ordered by bucket first:
`expression` (could be culture) > `personal` (one speaker, or one exchange with no marking) > `ordinary`
(model register) > `wording` (system or world). Nothing leaves the pool and no count changes; only the order
does, so the head of the list the UI calls culture holds only candidates for culture. The seven status chips
and the three tiers are unchanged; `ordinary` is an extra chip in `flags` and a `tier_reasons` stop.
Measured on the twelve runs: genuine local expressions in the top 10 rose from ~4% to ~16%, no system- or
world-wording row is left in any run's top 10, and both judge-confirmed conventions are back (`forty-seven
steps` #1, `frisbee analogy` #7).

## 9. Things the plan asked for that are simplified

**D42. Research-linked commons mode.** A separate `world.mode: commons` adds a pure
state-transition world with consequential sensor projects, private measurements,
finite replenished resources, exclusive equipment, multi-tick work and travel,
explicit local speech, versioned shared records and fresh participant identities.
The legacy mode remains available. The first factorial design targets RQ2 in the
research review: shared records crossed with stable/changed operating conditions,
with matched turnover and release calendars. See
[the implementation contract](COMMONS_IMPLEMENTATION.md) for question coverage.

**D43. Ground truth and records.** Commons agent input uses an explicit local-view
allowlist. The correct calibration mapping, experimental schedule and observer
snapshot never enter prompts. Record contents require an explicit read; available
catalog metadata and actually encountered text are distinct. Writing or revising a
record does not change physical law or automatically enforce an instruction.

**D44. Time, conflict resolution and inheritance.** All free agents choose from one
tick snapshot. Equipment conflicts use seed/tick/actor priority, independent of
LLM calls and iteration order. Work consumes working-time ticks and survives day
boundaries. Departures cancel pending operations without refunding consumed
supplies, drop carried kits at the departure position, and leave records intact.
New identities start without predecessor memories or relationships. Static traits
and group opportunities can be matched; this is not a culture-free population.

**D45. Validity and analysis boundary.** Commons provider failures invalidate the
run, including failures caught by an upstream retry wrapper. Strict replay has no
mock fallback, and existing recordings cannot be overwritten. The CLI working
directory is the platform temporary directory rather than a Unix-only path.
Commons analysis reports descriptive behavior and access, not E1--E4 alignment or
semantic change. The mock policy is solely a software fixture. Daily snapshots
are inspection artifacts; no matched-state history-swap or resume claim is made.

The following simplifications refer to the original latent-event mode:

- "Merge similar memories" is merge-into-existing (keep the old wording, raise importance), not an LLM merge.
- Invitations are probabilistic (D11), not negotiated in dialogue.
- The LLM surface generator for latent events was removed in v2 (D51). Surfaces are the 44 hand-written
  skins (D52).
- Semantic analysis uses the local embedder (D2), so "meaning" similarity is approximate.
- Model temperature is not controllable with the CLI backend (D1).
