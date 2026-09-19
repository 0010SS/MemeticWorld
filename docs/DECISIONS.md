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

**D25. Forced movement.** Event beats force the protagonist, and involved secondary agents, to the beat's location
(event-triggered routine deviation). NPC roles ("a TA", "a student nobody seemed to know") fill roles when no
suitable agent is free.

**D26. Event provenance.** The simulator tags each memory with the world events it came from (sidecar
`memory_meta.json`, never in a node or prompt). Conversation memories inherit the events of the memories the
speakers retrieved while talking. *Risk:* this over-attributes, because a retrieved memory may not actually be
talked about. Latent-event precision and recall are therefore approximate.

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

**D41. Observer model held fixed; agent model as an experimental factor.** `analyze --model` overrides the
analyzer's model. The controls vary the agents' model (Haiku vs Sonnet via the claude CLI) and analyze
every run with Sonnet, so classifier differences cannot masquerade as agent effects. New negative-control
condition `configs/no_events.yaml` (event_rate 0): conventions found there estimate the base rate from
chatter alone, and the classifier's false-positive rate for "event-driven" culture.

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
- The LLM surface generator for latent events exists but is untested at scale (off by default).
- Semantic analysis uses the local embedder (D2), so "meaning" similarity is approximate.
- Model temperature is not controllable with the CLI backend (D1).
