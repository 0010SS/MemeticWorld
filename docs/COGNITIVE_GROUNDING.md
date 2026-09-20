# Cognitive grounding of the MemeWorld design

For every mechanism in the simulator: the cognitive-science principle it borrows, what the code actually
does (with `file:line`), how faithful that is, what we could borrow next, and what each principle
predicts that our observer could measure.

Companion to [COGNITIVE_FRAMING.md](COGNITIVE_FRAMING.md), which states the framing for a paper. This
document is the audit underneath it: where the framing says "reconstructive memory", this says which
lines do that, how far they are from Bartlett, and how we would tell.

Decision numbers (D14, D42, ...) refer to [DECISIONS.md](DECISIONS.md). Ontology references are
[ONTOLOGY_V2.md](ONTOLOGY_V2.md) and [ONTOLOGY_V3.md](ONTOLOGY_V3.md).

> **Citation status.** Classics were compiled from memory and spot-checked. A few entries are marked
> **[unverified]** where the page range, chapter or article number is uncertain; the claim attributed to
> the work is the part we are confident about. Verify every citation before submission.

> **Measurement status.** Counts in this document come from the 128 archived runs under `runs/` that
> carry a `trace.jsonl` (67,224 `memory_encoded` records, 29,174 `utterance` records, 7,950 viewpoint
> renderings). Most of those runs use the **mock** LLM backend; only ~19 use `claude_cli`. Mock and real
> runs must never be pooled for anything about generated text. Counts of *engine-side* events
> (draws, drops, merges, forgets) are safe to pool, and are what the numbers below report.

---

## 0. Stance

**We are not claiming that LLM agents are cognitive models.** No result here is evidence about human
psychology. We borrow mechanisms from cognitive science as **design constraints** on a simulator: a way
of deciding what an agent may and may not have access to, and of making those limits explicit,
parameterised and loggable instead of implicit in a prompt.

Three commitments follow.

1. **Mechanisms are constraints, not claims.** When we implement perceptual load, the claim is "this run
   had a load parameter of 0.92 per co-present agent, and here is what it changed", not "agents
   experience perceptual load". A mechanism earns its place by changing a measured quantity, which is
   why every module keeps a manipulation check (`backend/modules/base.py:32-35`, `:156-158`).

2. **The LLM's own priors are a confound everywhere, and usually in the direction of the effect we want.**
   A pretrained model already knows what a bystander could plausibly see, how gossip spreads, that
   people copy each other's phrases, and that a dirty lens is fixed by cleaning it. So *plausible output
   is not evidence that our mechanism worked*. The discriminating evidence is almost always a
   **loss** — information that fails to arrive, an option that is not taken — because loss is what the
   prior does not supply. Each section ends with the confounds specific to that subsystem.

3. **Fidelity is graded and stated.** Every design element is labelled:

   | Label | Meaning |
   |---|---|
   | **faithful** | the mechanism has the shape the literature gives it |
   | **simplified** | the right direction, fewer factors, or a cruder functional form |
   | **divergent** | implemented, but does something the literature says it does not |
   | **absent** | named in our framing, but no code performs it |

   Labelling something **divergent** is not an accusation; several divergences are deliberate
   experimental-design choices (`memory.verbatim.exclude_self`, the reminding thumb on the scale). What
   matters is that they are not described as fidelity.

---

## 1. One page: design element → principle → fidelity → prediction

| # | Design element (`file:line`) | Principle | Fidelity | Testable prediction (observer metric) |
|---|---|---|---|---|
| 1 | `attention_prob` mixes salience, distance, busy, crowd, familiarity (`agents/perception.py:57-76`) | Priority map / guided search (Wolfe & Horowitz 2017) | simplified | Decomposed weights widen the spread of `p_attend` per fact and lower `witnesses_per_event_mean` (`analysis/funnel.py:84`) |
| 2 | Participation sets p = 1.0 (`agents/perception.py:59-60`) | Self-reference; actor–observer asymmetry | simplified | Coinages come disproportionately from bystanders, not participants (`role` in `live.adopters_detail:371`) |
| 3 | Salience is a world constant (`simulation/structures.py:27-32`) | Bayesian surprise; novelty (Itti & Baldi 2009) | divergent | `p_attend` declines with within-family instance index; currently flat |
| 4 | Spatial gating 1.0 / 0.35 / 0 (`agents/perception.py:62-69`) | Spotlight / zoom lens; modality | simplified | Wider auditory field raises `witnesses_per_event_mean`, lowers `independent_rate` (`emergence.py:255-262`) |
| 5 | `busy_factor` 0.6 (`agents/perception.py:70-71`) | Inattentional blindness (Simons & Chabris 1999) | simplified | `p_attend` drops sharply when the observer is in conversation; only 32 of 7,950 renderings are `distracted` |
| 6 | `crowd_factor` 0.92 per co-present agent (`agents/perception.py:73`) | Perceptual load (Lavie 1995) | divergent | Familiarity should matter *more* under load; our code makes it the only surviving channel |
| 7 | `familiarity_weight` +0.25, additive (`agents/perception.py:74-75`) | Cocktail-party effect (Cherry 1953) | simplified | A content-familiarity channel turns adoption curves S-shaped (`trends.meme_series:80`) |
| 8 | Recognition threshold 0.3 + regex name guard (`agents/viewpoint.py:67-72`) | Familiar/unfamiliar face dissociation (Bruce & Young 1986) | faithful | Misrecognition would inflate `n_independent` (`emergence.py:239-256`) |
| 9 | LLM viewpoint rendering per vantage (`agents/viewpoint.py:75-106`) | Situated construal; egocentric anchoring | divergent | Information should fall monotonically with vantage; **already falsified** for `distracted` |
| 10 | Independent per-agent attention draws (`simulation/engine.py:389-394`) | Joint attention; common ground (Tomasello 1995; Clark 1996) | absent | Co-witnessed events yield more carried adopters per witness (`emergence.carried_rate`) |
| 11 | Overhearing is i.i.d. verbatim access (`agents/conversation.py:186, :247-248`) | Addressees vs overhearers (Schober & Clark 1989) | divergent | Overhear-only adopters copy form without referent (`analysis/meaning.py`) |
| 12 | LLM re-encoding into the agent's words (`memory/encoder.py:335-388`) | Reconstructive memory (Bartlett 1932) | simplified | Within-agent memories of one family converge across days |
| 13 | `encoding_noise` as one gist↔verbatim dial (`memory/encoder.py:55-65`) | Fuzzy-trace theory (Reyna & Brainerd 1995) | divergent | Verbatim reuse decays faster with lag than topic reuse (`funnel.py:189-194`) |
| 14 | Salience-blind pre-filter drop (`memory/encoder.py:269-280`) | Content biases in transmission chains | simplified | Person-involving facts survive encoding more than object facts at equal salience (`encoding_ops`) |
| 15 | Memory lens: focus drawn uniformly from 7 strings (`memory/encoder.py:68-69, :80`) | Perspective at encoding (Pichert & Anderson 1977) | divergent | Persona-weighted foci make focus-match predict adoption; today it should not |
| 16 | Lens uncertainty flag, read by nothing (`memory/encoder.py:82, :87`) | Metamemory / source monitoring | divergent | Confidence gating lowers R for phrases from distorted memories (`trends.meme_series:80`) |
| 17 | Near-duplicate merge at cosine ≥ 0.92 (`memory/encoder.py:400-426`) | Schema blending; interference | absent in practice | 302 merges in 67,224 encodings (0.45%) |
| 18 | Capacity forgetting above 300 nodes (`memory/encoder.py:474-484`) | Adaptive forgetting (Anderson & Schooler 1991) | absent in practice | 25 forgets in 67,224 encodings (0.04%); no bottleneck exists |
| 19 | Exponential recency decay (`memory/store.py:175-177`) | Forgetting curve; power-law retention (Wixted & Ebbesen 1991) | divergent | Power-law decay allows old memories to resurface; exponential cannot |
| 20 | Composite retrieval score, min-max normalised (`memory/retrieval.py:92-97`) | ACT-R activation / rational analysis | simplified | Score should predict next-window re-retrieval; normalisation destroys the scale |
| 21 | Retrieval never fails; top match is always 1.0 (`memory/retrieval.py:93-95, :108`) | Retrieval threshold; SAM stopping rule | absent | Raw cosine of "relevant" retrievals is bimodal with mass near zero |
| 22 | `touch=True` resets `last_accessed` (`memory/retrieval.py:115-117`) | Testing effect (Roediger & Karpicke 2006) | simplified | Spaced beats massed re-mention for meme survival (`trends` half-life) |
| 23 | Reminding: related + salient candidates (`memory/reminding.py:40-61`) | Reminding; surface-driven analogical access (Gentner et al. 1993) | divergent (deliberate) | Cross-event links should be surface-driven; ours are not (`funnel.reminding_same_family_pct`) |
| 24 | Verbatim stickiness by global Zipf (`memory/encoder.py:191-221, :253-263`) | Von Restorff; Hunt's relational distinctiveness | simplified | Local distinctiveness predicts an inverted-U in p(stick) vs prior exposure |
| 25 | Priming line lists phrases without the speaker (`memory/wording.py:65-78`) | Interactive alignment; conceptual pacts (Brennan & Clark 1996) | simplified | Dyad-internal reuse exceeds cross-dyad reuse (`funnel.next_turn_echo_pct`) |
| 26 | Source metadata never reaches agents (`memory/store.py:121-132`) | Source monitoring; cryptomnesia | absent | Sleeper effect: low-credibility phrases used more after attribution fades |
| 27 | Generated topology: dense circles + bridges (`agents/topology.py:85-120`) | Weak ties; complex contagion (Centola & Macy 2007) | faithful | Conventions cross bridges later and less often than within circles |
| 28 | Prestige from static degree centrality (`modules/prestige.py:21-25, :59-61`) | Prestige-biased social learning (Henrich & Gil-White 2001) | simplified | Endogenous prestige gives rising Gini of adoption sources (`live.transmission_tree:411`) |
| 29 | Conformity boost = log(distinct corroborating peers) (`modules/conformity.py:23-42`) | Conformist transmission (Boyd & Richerson 1985) | faithful | Multi-source exposure predicts adoption above single-source |
| 30 | Hidden regime flip with counterbalanced mapping (`simulation/regimes.py:20, content/laser_alpha.py:13`) | Latent-cause inference; non-stationary structure learning | faithful | Choice accuracy tracks the *current* regime with a lag; prior-driven agents show a mapping × regime interaction |
| 31 | The co-op binder as shared external record (`simulation/records.py`) | Distributed cognition; cognitive artifacts (Hutchins 1995) | faithful | `transitions: wipe` removes the convention iff it lived in the artifact, not in heads |
| 32 | Private post-run probes (`analysis/probes.py`) | Public use vs private representation | faithful | Use without meaning: high adoption, chance-level probe accuracy (`meaning.acc_cur:46`, `sri:67`) |

---

## 2. Perception

Code: `backend/agents/perception.py`, `backend/agents/viewpoint.py`, `backend/prompts/viewpoint_v1.txt`,
`backend/simulation/engine.py:380-430` (release and ambient), `configs/default.yaml:101-108`.

### 2.1 The priority computation

`attention_prob` (`agents/perception.py:57-76`) returns one scalar per (observer, fact):

```
p = base_attention * (0.5 + 0.5*salience)      # perception.py:67
  * other_arena_factor if across the arena     # :68-69
  * busy_factor if in a conversation           # :70-71
  * crowd_factor ** (co-present - 2)           # :73
  + familiarity_weight * max familiarity        # :74-75
```

**Principle.** A *priority map*: attention is allocated by combining a small set of weighted signals
into one ranking over competing items (Itti & Koch 2001; Fecteau & Munoz 2006; Bisley & Goldberg 2010).
Wolfe & Horowitz (2017) list five guiding factors: bottom-up salience, top-down goal guidance, scene
structure, selection history, and value.

**Fidelity: simplified.** Two of the five factors are present — salience via `fact["salience"]`, scene
structure via the arena gate (`:62-65`). Goal guidance, selection history and value are absent. The
combination rule is ad hoc: multiplicative for stimulus and load, then an **additive** social bonus
(`:74-75`) that escapes every multiplier, with no normalisation; the only clamp is downstream
(`modules/base.py:86-92`). And facts compete with nothing — each gets an independent Bernoulli draw
(`:85-87`), so there is no limited resource being allocated.

**Borrowing (M).** Restate as explicit weighted log-odds with `perception.weights.{salience, goal,
history, social, load}`, keeping the current numbers as a named preset so old runs replay. Log per-term
contributions next to `p_attend` in the `observation` record so the observer can decompose why a fact
got through. Every downstream meme mechanism starts from who witnessed what, and that is currently
undecomposable.

**Prediction.** Making the social term load-sensitive widens the spread of `p_attend` across observers
of the same fact and lowers `witnesses_per_event_mean` (`analysis/funnel.py:84`). Baseline exists:
3,351 non-participant draws, mean 0.463, median 0.467, max 0.951.

### 2.2 Participation, vantage and the viewpoint renderer

Participation returns 1.0 unconditionally (`perception.py:59-60`), and `vantage` (`:41-48`) assigns
`participant | far | distracted | near`. Each perceived fact is then re-described by an LLM call
(`viewpoint.py:75-106`, temperature 0.9, `max_tokens` 300) with one clause per vantage
(`viewpoint.py:31-32` and `prompts/viewpoint_v1.txt`).

**Principles.** Self-reference and the field-vs-observer perspective on the same episode (Rogers, Kuiper
& Kirker 1977; Nigro & Neisser 1983); actor–observer asymmetry in construal (Jones & Nisbett 1971
[unverified imprint]; Malle 2006 for the qualifications); situated construal and egocentric anchoring in
perspective taking (Schober 1993; Keysar et al. 2000; Epley et al. 2004).

**Fidelity: divergent — the design is faithful, the measured behaviour is not.**

- Participation dominates: **5,684 of 7,950** archived renderings are `vantage='participant'`
  (`near` 1,928, `far` 306, `distracted` 32).
- The perspective difference is a prompt tag only. There is no actor–observer *content* asymmetry
  (actors citing circumstances, observers citing dispositions). Participants also get their own private
  `inner` fact at perfect fidelity (`simulation/structures.py:29`, `simulation/world_script.py:348`).
- **The renderer never drops anything.** The "return an empty string if it could not be perceived"
  instruction (`viewpoint.py:97-98`, final clause of the prompt) fired **0 times in 7,950 renderings
  across 128 runs**; `viewpoint.drop_rate` (`analysis/funnel.py:320`) is identically 0. There were
  also 0 fallbacks, so this is not a parsing artefact.
- In the real-LLM runs the rendering *lengthens* the fact (mean length ratio 1.16 participant, 1.62
  near) and retains only 0.56 / 0.26 of the world fact's tokens. It elaborates rather than loses.
- Fidelity is not monotonic in vantage: `distracted` renderings sit *above* `near` on Jaccard with
  `world_text` (0.978 vs 0.922), and are textually indistinguishable from participant renderings.

**Borrowings.**
- **(M)** Replace the empty-string instruction with a required categorical field the model will emit:
  `{"f0": {"perceived": "...", "confidence": "clear|partial|none"}}`, dropping on `none`.
  Instruction-following models comply with a required enum where they resist producing nothing.
- **(M)** Impose `perception.viewpoint_max_len_ratio` per vantage (1.0 participant, 0.6 near, 0.4
  distracted/far), enforced by truncation, not asked for in the prompt.
- **(S)** `perception.participant_p` (default 0.95) so a participant can miss their own slip.
- **(S)** One clause per vantage giving the construal gap: participants mention circumstances ("the
  tray was already wet"), bystanders mention the person ("she was careless with the tray"). This
  manufactures, at the perception layer, exactly the gap that gives a bystander something worth naming.

**Predictions.** Situated construal predicts Jaccard(world_text, perceived) monotone in vantage
distance and `drop_rate > 0` for far/distracted; both fields are in every `viewpoint` record
(`viewpoint.py:103-104`), so this is testable today — and the monotonicity is **already falsified**.
Actor–observer asymmetry predicts originators are disproportionately non-participants: cross `role ==
'originator'` in `live.adopters_detail` (`analysis/live.py:371-405`) with the event's `involves` set.

### 2.3 Load, crowding and familiarity

`crowd_size` counts everyone in the **building** (`perception.py:51-54`), not the observer's arena and
not the number of competing facts. The decay is unbounded geometric: 16 agents in a dining hall gives
0.92^13 = 0.34. Because the familiarity bonus is additive and applied last, in a crowd the social term
dominates: at familiarity 0.8 the stimulus term falls to ~0.12 while the social bonus stays at 0.20.

**Principle.** Perceptual load theory (Lavie 1995, 2005; Murphy, Groeger & Greene 2016 [unverified
pages]): under high load, irrelevant stimuli are not processed at all, so high-priority items are
*more* dominant, not less.

**Fidelity: divergent.** Our ranking is qualitatively wrong at exactly the crowded meals where most talk
happens. Load kills the stimulus channel and leaves the social channel untouched — the inverse of the
theory.

**Borrowing (S).** Three one-line changes: `perception.load_scope: arena|location`;
`perception.load_includes_facts: true`; apply the load multiplier to the social (and prestige) term.
Cheapest fix with the largest correctness gain in this subsystem.

**Prediction.** Load theory predicts a crossover: the partial correlation of `p_attend` with
max-familiarity should be flat at low load and steep at high load. Today the code predicts the
opposite. Crowd size per tick is reconstructible from `frames.jsonl` positions, so this is computable
on the archive now.

`busy_factor` (0.6) is a mild discount where inattentional blindness is near-total for *unexpected*
stimuli (Simons & Chabris 1999; Mack & Rock 1998; Most et al. 2005 [unverified pages]); it also does
not touch the additive familiarity term, and it is near-inert in practice (32 of 7,950 renderings).
Borrowing (S): `perception.busy_factor_unexpected ≈ 0.15` for facts with no prior expectation, keep 0.6
for expected ones, and hold `state.in_conversation` across the whole conversation window
(`simulation/engine.py:356-361` currently sets it only for the carry-over case) so the condition occurs
often enough to measure.

`familiarity_weight` is right in spirit (Cherry 1953; Moray 1959; Conway, Cowan & Bunting 2001) but
concerns only **who** is involved, never **what** is said. Borrowing (M): add
`perception.known_phrase_weight` for facts containing a phrase already in the agent's verbatim store
(`memory.verbatim`, `configs/default.yaml:54-64`). This closes the perception–adoption loop: adopting a
phrase makes you notice it more readily, which is the mechanism behind accelerating spread, and it is
the only proposed borrowing that feeds perception *from* culture.

### 2.4 What is missing entirely

| Gap | Principle | Why it matters here |
|---|---|---|
| **Joint attention.** Attention draws are deliberately independent per (agent, beat) (`engine.py:389-394`), and nothing marks a memory as co-witnessed. | Joint and shared attention as the substrate of reference (Tomasello & Farrar 1986; Tomasello et al. 2005; Shteynberg 2015); common ground (Clark 1996) | Naming presupposes the expectation of a shared referent. We have removed the only cue for it. This is the pipeline's weakest joint. |
| **Surprise / habituation.** Salience is a world constant; the only history is an exact-string skip over the last 6 ambient memories (`engine.py:418-419`, D14). | Bayesian surprise (Itti & Baldi 2009); habituation of the orienting response (Sokolov 1963; Rankin et al. 2009) | The fourth instance of an event family is as attention-grabbing as the first, which removes the perceptual window a new name would be coined in. |
| **Goal / interest guidance.** `profile.interests` (`agents/profile.py:58`) reaches only the persona blob; perception reads the profile solely for familiarity. | Endogenous vs exogenous attention (Corbetta & Shulman 2002); contingent capture (Folk et al. 1992) | Two agents in the same room have identical attention for every fact. This removes the natural engine of persona-linked and group-linked divergence. |
| **Emotion → attention.** `modify_attention` exists and is called (`perception.py:76`), but the emotion module implements only `modify_memory` (`modules/emotion.py:95-99`). | Arousal-biased competition (Mather & Sutherland 2011); Easterbrook 1959 | Removes the perceptual half of the best-evidenced content bias in cultural transmission (Heath, Bell & Sternberg 2001). |
| **Capacity.** No budget on facts per observer-tick; gains sum and are clamped only at the end (`modules/base.py:86-92`). | Normalisation and competition for a limited resource (Reynolds & Heeger 2009; Desimone & Duncan 1995) | Once two social boosts stack, load and distance stop mattering at all. |

Two channels also bypass the documented knobs and should be unified (S):

- **Reaction remarks** broadcast at p = 0.9 in the arena and 0.2 across the building
  (`engine.py:485`), three times the configured `perception.overhear_prob` of 0.3, at a hard-coded
  salience of 0.5 (`engine.py:506`) vs 0.4 for conversation turns (`agents/conversation.py:248`). These
  constants are in the engine, not in config, so a perception sweep leaves the **widest transmission
  channel in the simulation** untouched.
- **Ambient sightings** fire at p = 0.5 per co-located agent per tick (`engine.py:421`) with fixed
  importance 1-2 and no lens, and are the largest single class in the memory store (see §3.7).

### 2.5 LLM confounds (perception)

1. **The renderer elaborates instead of degrading.** Real-LLM renderings are 1.2-1.6× *longer* than the
   world fact. "Partial perception" is currently a paraphrase step, and the observer's world-wording
   penalty partly measures LLM verbosity.
2. **"Output nothing" instructions do not work.** 0 of 7,950. Any perceptual filter expressed as
   "return an empty string" silently does not exist; it must be a required enum or an engine-side rule.
3. **Mock runs make the renderer a near-identity** (Jaccard 1.00 participant, 0.96 near across 108 mock
   runs). Any vantage effect reported from mock runs is an artefact of the mock.
4. **Graded states expressed as adjectives are not realised.** `distracted` renderings are textually
   indistinguishable from participant renderings. Degradation must be mechanical.
5. **Perspective taking is already in the prior.** The model will write plausible bystander text whether
   or not our vantage model does anything. Plausibility is not evidence; information **loss** is.
6. **Name realism is the regex, not the model** (`viewpoint.py:67-72`). Credit it to the guard.
7. **Validate attention manipulations on engine-side numbers first** (`p_attend`, witness counts,
   `exposure` records) before reading anything off utterances.
8. **The register problem applies to construal.** One model writes every observer, so a shared way of
   describing a scene recurs across independent runs. Viewpoint text is world-provided
   (`analysis/rundata.py:239`) and must stay inside `register.py`'s cross-run penalty.

---

## 3. Memory: encoding and storage

Code: `backend/memory/encoder.py`, `backend/memory/store.py`, `backend/memory/wording.py`,
`backend/memory/need.py`, `backend/prompts/encode_memory_v2.txt`, `configs/default.yaml:47-69`.

### 3.1 Reconstructive encoding

Every observation is rewritten by an LLM into the agent's own words (`encoder.py:335-388`), conditioned
on the persona ISS (`:376`), relationship lines (`:357`), a fidelity instruction (`:55-65`), and a
sampled "memory lens" (`:75-92`).

**Principle.** Reconstructive memory: recall is shaped by prior knowledge structures, not read out
(Bartlett 1932; Brewer & Treyens 1981; Gilboa & Marlatte 2017).

**Fidelity: simplified.** Two gaps. (a) The schema doing the reconstruction is the **model's generic
prior** plus a one-paragraph persona — nothing from the agent's own history enters the encoding prompt
except one optionally sampled association (`:367-372`). Bartlett's drift is toward the rememberer's own
conventionalised schema; ours is toward the model's register. (b) **Encoding happens once and the text
is then immutable.** A grep over `backend/` finds no write to `node.description` anywhere; only
`poignancy` and `last_accessed` are ever mutated (`encoder.py:409-410`, `retrieval.py:117`). Bartlett's
phenomenon is precisely what happens on *repeated* reproduction.

**Borrowing (M).** `memory.schema_context: {enabled: true, k: 3}`: retrieve the agent's k most relevant
own memories with the new observation as cue (`retrieve(..., touch=False)`, already used at
`encoder.py:98-99`) and pass them as "what {first} has come to expect in situations like this",
replacing the single uniformly-drawn association. Repeated exposure to one hidden family would then
make each new encoding more like the previous ones — and that convergence is the compression that makes
a shared shorthand possible.

**Prediction.** Within-agent embedding cosine between memories of the same latent family rises across
days while cross-family cosine does not. Computable now from `memory_encoded.text` grouped by
`originating_event_ids → latent_type` (`analysis/grounding.py`); the baseline question is whether
convergence already happens without schema context.

### 3.2 The single noise dial vs fuzzy-trace theory

`memory.encoding_noise` (`configs/default.yaml:48`) is binned into four fidelity instructions
(`encoder.py:55-65`).

**Principle.** Fuzzy-trace theory: verbatim and gist traces are encoded in parallel and the verbatim
trace decays faster (Sachs 1967; Reyna & Brainerd 1995; Brainerd & Reyna 2002).

**Fidelity: divergent.** FTT's load-bearing claim is that the two traces are **parallel and
independently timed**. We collapse them into one scalar applied once, producing a single frozen text. A
memory encoded verbatim at t=0 is still verbatim three days later; a gist memory never had a verbatim
trace to lose. That removes the dynamic that produces late-emerging gist-consistent false recognition
and that drives levelling in serial reproduction. COGNITIVE_FRAMING.md:299 reports gist-over-verbatim as
a **result**, but our mechanism cannot produce the time course that finding is about.

**Borrowing (M).** Store two fields per node: `gist` (the current description) and `verbatim` (the raw
observation, already computed as `raw` at `encoder.py:343` and already traced as the `observation` field
at `:443`). Add `memory.verbatim_half_life_hours` (~6); retrieval returns the verbatim string only while
that trace is alive. Quoting in conversation then becomes gated by an actual trace instead of a prompt
instruction.

**Prediction.** FTT predicts an **interaction**: exact-wording reuse decays much faster with
hearing→reuse lag than topic reuse does. Metric: `bigrams_retained_verbatim_pct` /
`reuse_given_retained_pct` (`analysis/funnel.py:189-194`) binned by lag in ticks, against
`mentions_per_discussed_event_med` / `afterlife_hours_med` (`funnel.py:62-66`). Utterance ticks give the
lag, so this is testable today.

### 3.3 Selective encoding and entity generalisation

The pre-filter drops a fact with p = noise × (1 − salience) (`encoder.py:269-280`), and replaces an
unfamiliar person's name with "a student"/"they" (`:281-294`) when relationship familiarity < 0.3.
Archive: **2,536 `generalize_entity` ops and 411 `drop_fact` ops** across all runs.

**Principles.** Levelling in serial reproduction (Allport & Postman 1947); levels of processing (Craik &
Lockhart 1972); content biases in transmission chains (Bebbington, MacLeod, Ellison & Fay 2017).

**Fidelity: simplified.** Drop probability is **content-blind** — `salience` is assigned by the world,
not by the agent's interests, goals or valence appraisal. The transmission-chain literature is emphatic
that survival is predicted by *content* (social, negative, threat-relevant, stereotype-consistent,
minimally counterintuitive), not by an environment-supplied scalar. And `encoder.py:274` guarantees the
single most-salient fact is never dropped, so a chain can never lose its own main point — which real
rumour chains routinely do. Generalisation only ever goes vaguer: Allport & Postman's chains also
**sharpen and assimilate**, replacing a vague role with a stereotype-consistent *specific*, sometimes
the wrong one. Our memory distortion can therefore never create a misattributed referent.

**Borrowings.**
- **(S)** `memory.encoding_bias: {social: 1.4, negative: 1.3, self_relevant: 1.5}` as retention
  multipliers inside `_prefilter`, driven by fields the fact already carries (`involves`, `kind`) plus
  the emotion module's valence. No LLM call. Make the guaranteed-keep configurable.
- **(S)** A `substitute_entity` op beside `generalize_entity`: with `memory.entity_substitution: 0.1`,
  replace the unfamiliar person with a *familiar* one, logged as its own op so the observer can count
  misattributions rather than scoring them as hallucinations.

**Prediction.** Content bias predicts facts involving other people survive encoding at a higher rate
than object/place facts at equal world salience. Metric: join each `observation` record's fact list to
the `drop_fact` entries in `encoding_ops` on the matching `memory_encoded` record. Fully testable now.

### 3.4 The memory lens (D44)

`sample_lens` (`encoder.py:75-92`) draws per encoding: a **focus** from 7 fixed strings, a length, a
style, fidelity jitter, an **uncertainty** flag (p = 0.3 × variability), a **distortion** of one minor
detail (p = 0.5 × variability × noise), and an **association** flag (p = 0.35 × variability).

| Element | Principle | Fidelity | Key difference |
|---|---|---|---|
| Focus (`:68-69, :80`) | Perspective at encoding (Pichert & Anderson 1977, 1978) | divergent | Pichert & Anderson's perspective is a **stable assigned stance** that biases encoding systematically, with later recovery on a perspective shift. Ours is i.i.d. uniform over 7 strings, uncorrelated with persona or role: variance without agent-specific bias. Two agents differ by noise, not stance. |
| Length / style / jitter (`:79, :81, :84`) | Encoding variability (Estes 1955; Martin 1968 [unverified]) | simplified | In the literature variability predicts a **retention benefit** via multiple retrieval routes. Here it is output diversity only; each observation is encoded once. D44 is explicit (`DECISIONS.md:375-377`) that this exists to compensate for a near-deterministic CLI backend. It is a simulation device, not a memory model, and should be described as one. |
| Uncertainty (`:82, :87`) | Metamemory; source monitoring (Johnson, Hashtroudi & Lindsay 1993; Nelson & Narens 1990 [unverified]) | divergent | The flag hedges the text and **nothing downstream reads it**. No confidence field on the node or `MemoryMeta` (`store.py:121-132`); retrieval uses only recency, relevance, importance (`retrieval.py:92-96`). In the literature low confidence is the brake that stops people asserting — we kept the distortion and removed the brake. |
| Distortion (`:83, :88-90`) | Misinformation effect (Loftus & Palmer 1974; Loftus 2005; Schacter 1999) | divergent | The misinformation effect is **post-event and other-sourced**: a memory changes because someone else's later account contradicted it. Ours is a self-generated perturbation at encoding, confined to a peripheral detail. It cannot produce (a) one person's account reshaping another's memory, or (b) two witnesses with incompatible versions that must be reconciled in talk. |
| Association (`:95-102, :367-372`) | Reconsolidation (Nader, Schafe & LeDoux 2000; Hupbach et al. 2007); intrusions (Roediger & McDermott 1995) | divergent | **The direction is reversed.** In Hupbach et al. the reminder reactivates the *old* memory, which absorbs the new material. Here the old node is explicitly read-only (`touch=False`, `:99`) and the *new* memory absorbs the old. The candidate is also drawn uniformly from the top 5 (`:102`), so association strength does not govern which memory intrudes. |

**Borrowings.**
- **(S)** Derive the focus distribution from the profile (`profile.attention_style` weights over `FOCI`,
  sampled at `:80`). Cheapest available change that could produce group-specific memory content, and it
  costs no LLM calls.
- **(M)** `MemoryMeta.confidence` set from the lens, a `retrieval.confidence_weight` term
  (`retrieval.py:96`), and a conversation-prompt line so low-confidence memories are voiced as hedged
  reports. Closes the misinformation loop with the gate the theory says exists.
- **(L)** A post-event misinformation path: when an agent hears someone describe an event it also
  witnessed, the heard version may overwrite a detail in the **existing** node — a true edit of stored
  text, which nothing in `backend/` currently does. `memory.misinformation: {enabled, p}` plus a
  `memory_revised` trace type. Symmetrise the association the same way: mark the **old** node for
  revision, and weight the draw at `:102` by retrieval score. This is the missing route from "several
  witnesses" to "one agreed version" to "one name".

**Prediction.** With post-event misinformation on, witnesses of the same event converge in memory
content over days, and the converged version tracks the most-talked-about account rather than the modal
perception. Metric: pairwise cosine over memories sharing an `originating_event_ids` entry as a function
of day, against talk volume (`funnel.py:53-66`). The baseline — that witnesses currently do **not**
converge — is testable today.

### 3.5 Verbatim stickiness (D49/D58)

`distinctive_phrases` (`encoder.py:191-221`) selects 2-3 word spans around a rare content word
(wordfreq Zipf ≤ 3.6), within one clause, content words at both edges, no person names, and — in
`lexicon` mode — not campus vocabulary (`simulation/lexicon.py`). `sticky_phrases` (`:235-266`) then
keeps a phrase with p = base + gain × (number of existing memories containing it), capped at 0.9.
Archive: **442 `wording` records** across all runs.

**Principles.** Von Restorff (1933) isolation effect and Hunt's (1995) correction that distinctiveness
is **relational**, not an item property; memorability of phrasing (Danescu-Niculescu-Mizil et al. 2012);
base-level activation as a power-law function of practice history (Anderson & Schooler 1991; Anderson et
al. 1998 [unverified pages]).

**Fidelity: simplified.** Distinctiveness is global-corpus rarity, so a phrase is distinctive in exactly
the same way for every agent regardless of what that agent has been hearing all week. The familiarity
gain is a **linear count with no time term**: `seen` at `:261` counts current memories containing the
string, so a hearing a week ago weighs as much as one an hour ago, and stickiness can only ever grow.
That is a rich-get-richer rule that manufactures the winner-take-all dynamic we are supposed to be
*observing*.

**Borrowings.**
- **(M)** `memory.verbatim.distinctiveness: local` — score surprisal against the agent's own recent
  memory text, which the code already builds at `encoder.py:253`: `z_local = z_global − λ·log(1 + own
  count)`. A phrase stops being distinctive once you have heard it a lot.
- **(S)** `memory.verbatim.activation: actr` — replace `seen` with a recency-weighted power-law sum over
  `MemoryMeta.wordings` timestamps, which already carry `time` (`wording.py:27`).

**Prediction.** Local distinctiveness predicts an **inverted-U** in p(stick) against prior exposure,
where the global rule predicts monotone increase. Metric: p(stick | n prior hearings) from `wording`
records against `exposure_stats` (`trends.py:66`). Testable now, though thin (16-58 wording records per
run).

**`exclude_self` (`encoder.py:249-252`, D58) is the opposite sign to the literature.** Generation and
production effects say self-generated and spoken-aloud material is remembered *better* (Slamecka & Graf
1978; MacLeod et al. 2010; MacLeod & Bodner 2017). We set that weight to zero because pre-v2, 17 of 60
stuck phrases were the agent's own, and self-priming lets one agent bootstrap a catchphrase alone. That
is a legitimate experimental-design choice but it biases us against idiolect-driven coining. Borrowing
(S): keep `exclude_self: true` as the default arm, expose `memory.verbatim.self_weight` as an explicit
factor with its own condition file, and report the contrast — the observer already separates
`stuck_self` from `stuck_other` (`funnel.py:297-299`).

### 3.6 Consolidation, interference and forgetting — the largest gap

| Mechanism | Code | Archive result |
|---|---|---|
| Near-duplicate merge, cosine ≥ 0.92 | `encoder.py:400-426`, `configs/default.yaml:51` | **302 merges / 67,224 encodings (0.45%)** |
| Capacity forgetting above 300 nodes | `encoder.py:474-484`, `configs/default.yaml:50` | **25 forgets / 67,224 encodings (0.04%)** |
| Exponential recency decay | `store.py:175-177` | used at `retrieval.py:92`, `wording.py:55` |

**Merge is effectively inert, and when it does fire it is total replacement, not blending.** The new
text is discarded outright (logged as `dropped_text`, `:419`) and only poignancy is maxed (`:409`).
Human blending of repeated similar episodes produces a composite (Bower, Black & Turner 1979; Zacks et
al. 2007 [unverified pages]); the older episode does not simply win. The candidate window is also the 40
most recent nodes of the same kind (`:402`), a recency cutoff with no principled basis. The
interference/assimilation mechanism claimed in COGNITIVE_FRAMING.md §4.4 is **not operating**.
Borrowing (M): `memory.merge_threshold: 0.86` plus `memory.merge_mode: blend|replace`, where blend
re-encodes the surviving node's text from both descriptions. Blending repeated similar episodes into one
schematised memory is the core mechanism by which "the third time the printer ate someone's thesis"
becomes a category, and categories are what get names.

**Forgetting never runs.** Peak store size is ~100 nodes per agent against `memory_capacity: 300`. Only
`high_noise_s42` (capacity 120) forgot anything. So at default settings MemeWorld agents have **perfect
retention**, the only loss in the system is what never got encoded, and `perfect_memory.yaml` vs
`default.yaml` is **not the contrast it claims to be**. This directly undercuts the framing: iterated
learning attributes the emergence of compressed, structured codes to a **transmission bottleneck**
(Kirby, Cornish & Smith 2008), and we have no bottleneck downstream of encoding. Borrowing (S): either
set `memory_capacity` so it binds at our run lengths (~60-80 for a 3-day run) and declare it an
experimental factor, or replace the capacity cliff with per-tick probabilistic forgetting driven by the
retention function. **Highest value per line changed in the whole audit.**

**Exponential decay has the wrong form.** Every retention function that survives Rubin & Wenzel's (1996)
survey of 210 datasets decelerates; simple exponential does not. At `decay_rate: 0.1` the recency score
halves every ~6.9 simulated hours and is ~1e-3 after two days, so after min-max normalisation inside
retrieval (`retrieval.py:93`) all older memories land on one indistinguishable floor: yesterday and
three days ago score the same. For a study about whether expressions **persist**, that is load-bearing.
Borrowing (S): `memory.retention: power|exponential` with (1 + hours)^−d, d ≈ 0.5, implemented once in
`recency_score` and reused by `recent_wordings`. One function, two call sites.

**Importance is fixed at encoding** (`encoder.py:392`) and rises only on a merge (`:409`), which never
fires. The literature's key finding is time-dependent: emotional material is not better remembered
immediately, it becomes *relatively* better remembered as neutral material decays (McGaugh 2004; Sharot
& Phelps 2004 [unverified]; Yonelinas & Ritchey 2015 [unverified pages]). Our uniform decay gives
emotional and neutral memories identical decay rates. Borrowing (S):
`memory.decay_rate_by_importance`, one line inside `recency_score`. This reproduces the emotional-
selection result that the meme literature (Heath, Bell & Sternberg 2001) identifies as a driver of what
spreads, and composes with the power-law change.

### 3.7 Ambient memories and NEED

**Ambient sightings bypass the entire encoding pipeline.** `add_simple_event` (`encoder.py:455-471`)
stores routine sightings **verbatim** at fixed salience 0.1, no lens, no merge, no distortion, no LLM —
and they are the largest class in the store (254 of 566 encodings, 45%, in `runs/test_map_ui_v2allon`).
Script theory predicts the opposite: routine, script-consistent events are reconstructed from the script
rather than stored episodically, with high false recognition of script-typical actions (Schank & Abelson
1977; Bower, Black & Turner 1979). This also explains why `retrieval.source_weights: {ambient: 0.5}` was
needed as a patch (D50, `DECISIONS.md:463-468`) — it treats a retrieval-time symptom of an
encoding-level design choice. Borrowing (M): `memory.ambient: {mode: script|episode}`, folding routine
sightings into a per-agent statistics table the prompt reads as background knowledge, and letting only
script-**violating** sightings become episodes. This removes ~45% of the store and makes
`memory_capacity` actually bind.

**NEED / open matters is the cleanest mapping in the subsystem.** `memory/need.py:31-33, 36-58, 78-88`
gives unresolved experiences an exponential half-life accessibility, refreshed on re-encoding, decayed
on discussion, with top-k matters as extra retrieval focal points — a good implementation of the
Zeigarnik effect (Zeigarnik 1927; Klinger 1975 [unverified pages]; Masicampo & Baumeister 2011).
**Fidelity: faithful**, with one divergence: "resolution" is operationalised as *having talked about
it* (`discuss_decay`), whereas the literature is about goal completion or a concrete plan. Retelling can
*sustain* a concern rather than close it. It is also off in `v3_base` (`configs/v3_base.yaml:66`).
Borrowing (S): add `need.resolve_decay` applied only when the agent perceives or is told an outcome, and
turn NEED on — it is our most direct model of why the same incident keeps coming up, which is the
precondition for it acquiring a name.

**Source metadata never reaches agents.** `MemoryMeta.speakers` / `source_ids` / `wordings[].heard_from`
exist but are explicitly "NEVER shown to agents" (`store.py:123`). Whatever source information survives
does so accidentally inside the memory text, and the pre-filter can strip the name anyway
(`encoder.py:290`). Consequence: agents cannot prefer a phrase because a high-prestige person used it,
cannot discount an unreliable source, and cannot misattribute. **Prestige bias, credibility-weighted
transmission and cryptomnesia are all unreachable from the current memory representation** (Johnson et
al. 1993; Brown & Murphy 1989 [unverified pages]). Borrowing (M): an agent-visible `source_belief`
initialised from `heard_from`, allowed to decay to "someone" and drift toward a more typical speaker.

### 3.8 LLM confounds (memory)

1. **The model's summarisation prior already produces gist-level, schema-normalised paraphrase.**
   `encoding_noise: 0` is the only path that skips the LLM rewrite (`encoder.py:354`), so every noise > 0
   arm confounds our lossy-encoding manipulation with the default summariser. **Required control:** an
   arm with noise > 0 and a fidelity instruction that says "reproduce exactly".
2. **Instructed distortion is not memory distortion.** The lens *tells* the model to misremember
   (`:88-90`) and to hedge (`:87`). The model complies, and complies legibly. Any false-memory result is
   a compliance result until shown otherwise.
3. **Verbatim stickiness is partly a string operation.** `render_in_text` appends "keep them word for
   word, in quotes" (`:365-366`), and if the model drops the phrase the code appends it (`:386-388`). So
   `bigrams_retained_verbatim_pct` has a floor set by our own post-processing.
4. **Priming lines are direct instructions to a model with a very strong copy prior** (`wording.py:77`).
   A no-priming arm is not enough; a **scrambled-priming** arm (quote phrases the agent never heard) is
   needed to separate "this is in mind" from "these tokens are in context".
5. **Content biases are already in the model.** Acerbi & Stubbersfield (2023, *PNAS* 120(44):
   e2313790120) showed LLMs reproduce human transmission-chain content biases with no memory mechanism
   at all. Run the ablation (mechanism off) before attributing any survival bias to our mechanism.
6. **Encoding variability is imposed, not sampled from a process** (`DECISIONS.md:375-377`). Never
   describe it as stochastic reconstruction by the agent.
7. **The encoding prompt hands the model social context a rememberer might not have:**
   `relationship_line` for every nameable person (`:357`).
8. **The output format fights D43.** `encode_memory_v2.txt` requires third-person prose starting with
   the persona's name, while D43 frames the same memories as first-hand experience. The resulting
   register is generic narrative prose, not idiosyncratic inner speech — which suppresses exactly the
   personal phrasing variation a study of coining depends on.
9. **Importance is rated by the same model that just wrote the memory** (`:392`), so poignancy
   correlates with the model's own rhetorical emphasis. That circularity propagates into retrieval
   (`retrieval.py:96`), the forgetting rank (`:481`) and NEED gating (`need.py:42`).
10. **Agent-side and observer-side share a frequency model.** wordfreq Zipf is used both at
    `encoder.py:134-140` and by the observer's scorer (`analysis/wording.py:183`). No data flows between
    them, but state the non-independence when reporting how many detected candidates were also sticky.

---

## 4. Retrieval and reminding

Code: `backend/memory/retrieval.py`, `backend/memory/reminding.py`, `backend/memory/reflection.py`,
`configs/default.yaml:71-92`.

### 4.1 The activation function

`score = beta·recency + alpha·relevance + gamma·importance` over min-max normalised components
(`retrieval.py:92-97`; alpha 3.0, beta 0.5, gamma 2.0, top_k 5).

**Principle.** ACT-R declarative activation / rational analysis: availability tracks need probability,
estimated from recency, frequency and contextual cues (Anderson & Schooler 1991; Anderson & Milson 1989;
Anderson et al. 2004; Kahana 2020 for the modern survey).

**Fidelity: simplified**, in three specific ways.

1. ACT-R activation is a **log-odds quantity on an absolute scale**; ours is a weighted sum of three
   components each min-max rescaled to [0,1] across the current pool, so activation is purely
   rank-relative and has no interpretable units.
2. **ACT-R has no importance term at all** — need probability is carried by recency, frequency and cue
   association. Our γ = 2.0 importance term is an LLM 1-10 poignancy rating and is the second-heaviest
   weight, so a large share of "activation" is a model prior about narrative noteworthiness.
3. The weights are inherited GA hyperparameters, not fitted to any environmental statistic, which is the
   whole point of Anderson & Schooler.

**Borrowing (M).** `retrieval.form: additive | actr`. Under `actr`, compute base-level activation plus
cue association and drop the min-max step, so absolute activation exists; keep γ as an explicitly
labelled affective-salience term rather than folding it into "the rational model". Also log the **raw**
pre-normalisation components in `RetrievalResult.scores` (`retrieval.py:118-121`), which currently emits
only normalised values.

### 4.2 Recency, and the missing practice effect

`recency = exp(−decay_rate × hours since last_accessed)` (`store.py:175-177`, used at `retrieval.py:92`).

**Principle.** ACT-R base-level learning, B = ln(Σ t_j^−d): every past access contributes a decaying
trace, so activation encodes both recency **and frequency**, reproducing the forgetting curve and the
spacing effect together (Anderson & Lebiere 1998; Pavlik & Anderson 2005).

**Fidelity: divergent**, on two counts. (a) Exponential rather than power-law, so the long tail is far
too thin — a memory untouched for 48 h scores 0.008 and is effectively unreachable, whereas power-law
retention keeps old material weakly available, which is exactly the regime in which an old incident can
resurface and become a meme. (b) **Only the last access enters the formula.** A memory retrieved forty
times and one retrieved once are indistinguishable. There is therefore no practice effect, no spacing
effect, and no mechanism by which repeated rehearsal makes a memory progressively more retrievable.
**Frequency-driven compounding is the main engine of cultural spread, and the retrieval layer has
none.**

**Borrowing (M).** Store an access-time list on each node (append at `retrieval.py:115-117` instead of
overwriting) and add `memory.base_level: {form: exp|actr, d: 0.5, cap_accesses: 32}`. Under `actr`,
`recency_score` becomes ln(Σ (hours_j + 0.5)^−d), rescaled.

**Prediction.** Base-level learning makes retrieval probability **superlinear** in cumulative mention
count, producing a heavier-tailed distribution of per-meme share-of-talk and higher R in the first days
(`trends.meme_series:80`, R over time at `trends.compute:147`). Needs a new run.

### 4.3 Retrieval-as-learning, without the competitive half

`touch=True` resets `last_accessed` on every retrieved node (`retrieval.py:115-117`), and probes and
reminding correctly pass `touch=False` (`reminding.py:51`, `analysis/probes.py:126`,
`analysis/battery/runner.py:310`) — good hygiene, and worth keeping as an explicit invariant: **the
observer must not perturb the system it measures.**

**Fidelity: simplified.** Strengthening is all-or-none: recency snaps back to 1.0 regardless of how
strong the retrieval was or how often it has happened; poignancy is untouched, so the importance term
never reflects rehearsal. And **retrieval-induced forgetting is entirely absent**: retrieving one memory
does not push its unretrieved neighbours back (Anderson, Bjork & Bjork 1994). RIF is what makes a
group's repeated retelling of one version actively erase the alternatives — the sharpest available
mechanism for producing a single shared account and therefore a single shared name.

**Borrowings.** **(S)** After `chosen` is selected (`retrieval.py:114`), push back high-relevance
non-selected nodes for the same focal point: `n.last_accessed -= timedelta(hours=rif_hours)`, gated by
`retrieval.rif_hours: 0` (default off). Five lines, no new state. **(S)** `memory.retrieval_gain` that
raises poignancy slightly when a node is both retrieved **and cited** in the produced utterance —
"appeared in the prompt" is not retrieval success.

**Predictions.** Testing effect: spaced beats massed re-mention for long-run survival (peak / half-life
in `trends.py`); needs a new run, since no access history is stored. RIF: agents who talk about an event
more should retain fewer competing versions of neighbouring events — within-agent retrieved-memory
diversity falls as talk volume rises. The baseline (no such relation) is testable now from the
`retrieved` lists in reflection and conversation traces.

### 4.4 Retrieval never fails

Min-max normalisation rescales the best-matching node in the pool to relevance 1.0 **even at a raw
cosine of 0.02** (`retrieval.py:93-95`), and `kk = min(k, len(ids))` (`:108`) guarantees a full context
set.

**Principle.** ACT-R's retrieval threshold τ: a chunk is retrieved only if activation exceeds it,
otherwise nothing comes to mind. Retrieval failure is a first-class outcome in every serious memory
model — SAM's stopping rule (Raaijmakers & Shiffrin 1981), REM's likelihood criterion (Shiffrin &
Steyvers 1997).

**Fidelity: absent.** Agents always walk into a conversation with five "relevant" memories, which
manufactures topical continuity a person would not have. Worse for our purposes: **the observer cannot
detect this**, because `RetrievalResult.scores` stores post-normalisation `rel` (`:119-120`), so no
existing trace records how good the match actually was. Spurious low-quality retrievals are an invisible
source of cross-incident association, and cross-incident association is our headline outcome.

**Borrowing (S).** (1) Log `rel_raw` alongside `rel` — two lines, no behavioural change, and it makes
the whole subsystem auditable. (2) `retrieval.min_relevance: 0.0` applied to the raw cosine before
normalisation, so retrieval can legitimately return fewer than k, or nothing; the conversation prompt
already tolerates short lists (`conversation.py:171-173`).

**Prediction.** Encoding specificity (Tulving & Thomson 1973) predicts that when the cue shares little
with anything encoded, retrieval should fail: the distribution of raw cosine for retrieved nodes should
be strongly bimodal with a large mass near zero that we are silently promoting. Measurable immediately
after the `rel_raw` change; **not** testable on current traces, which contain only normalised values.

### 4.5 Relevance is lexical — faithful by accident

Relevance is cosine similarity between focal-point and node embeddings, using a hashed bag of
unigrams, bigrams and character trigrams (`retrieval.py:95`; `llm/embeddings.py:33-66`).

**Principle.** Cue-dependent retrieval and encoding specificity (Tulving & Thomson 1973); and, for
analogy specifically, MAC/FAC: **access is dominated by surface similarity** while inferential soundness
is governed by relational structure (Gentner, Rattermann & Forbus 1993; Forbus, Gentner & Law 1995; Gick
& Holyoak 1980).

**Fidelity: faithful**, and worth saying so plainly. A hashed lexical n-gram matcher is a close
functional analogue of the MAC stage: cheap, non-structural, surface-driven. The human finding is that
people **rarely** retrieve structurally similar but surface-different cases spontaneously, and our
embeddings reproduce that limitation for free. COGNITIVE_FRAMING.md:358 treats lexical embeddings as a
weakness; for the analogical-access question they are the correct model, and the real gap is that we
have no **FAC** stage — no structural matcher that evaluates a retrieved candidate.

### 4.6 Reminding (D48/D59)

`maybe_remind` (`reminding.py:79-134`) fires after a salient perception, conversation or overhearing,
shows the agent `n_related` retrieved candidates plus `n_salient` older ones sampled by poignancy²
(`:55-57`), and asks whether one feels alike. A yes stores a thought linking both memories and a typed
`Link` (`:113-126`). Archive: **2,271 remindings asked, 818 linked (36%), 168 chained.**

**Principle.** Reminding as the mechanism of learning from experience (Schank 1982; Ross 1984).

**Fidelity: divergent, deliberately.** Including important older memories *regardless of similarity*
gives surface-different analogues far more access than human reminding does. D48 and
COGNITIVE_FRAMING.md:218-221 state this as a thumb on the scale, to test whether the **downstream**
processes produce conventions once connections exist. Keep the honesty; also keep the ablation. Two
smaller points: the candidate list is shuffled (`:60`) but the salient draw is by poignancy², an
arbitrary exponent that should be a config key; and prompt v2 was rewritten because v1 said yes 56 of 68
times (`reminding.py:16`) — a compliance problem, not a cognitive one.

**Borrowing (M).** Add a **FAC stage**: after a yes, a second cheap check that the two episodes share a
*relational* pattern (same role structure, same outcome polarity) rather than a topic, gated by
`reminding.structural_check`. Today `what_felt_alike` is free text with no structure test, so a
surface-driven yes and a relational yes are indistinguishable in the trace.

**Prediction.** Gentner et al. predict cross-event links should be **surface-driven**: same-domain,
same-referent pairs far above chance, same-hidden-family pairs at chance. The observer already computes
exactly this contrast: `reminding_same_family_pct` vs `reminding_same_family_chance_pct`
(`funnel.py:120-121`). Testable now, on 818 realised links.

### 4.7 Reflection

`reflect` (`reflection.py:28-61`) triggers on the GA importance counter, generates focal points, appends
one rotating pattern question (`:16-20`), retrieves 8 memories per focal point and writes insights with
cited evidence, each citation recorded as a typed link (`:57-58`).

**Principle.** Offline abstraction and schema induction: general structure extracted from episodes
(McClelland, McNaughton & O'Reilly 1995; Gick & Holyoak 1983).

**Fidelity: simplified.** The schedule is right and the "never ask for a label" constraint is correctly
enforced. But reflection is purely individual: COGNITIVE_FRAMING.md:298 already records the measured
consequence — reflections describe patterns within a single incident and are rarely voiced. Borrowing
(S): raise the retrieval `k` for the pattern question specifically, or seed it with two memories drawn
from **different** events, so comparison has something to compare. Schema induction in Gick & Holyoak
requires *two* analogues present at once; our focal-point retrieval does not guarantee that.

---

## 5. Language and convention

Code: `backend/memory/wording.py`, `backend/memory/encoder.py:191-266`, `backend/simulation/lexicon.py`,
`backend/analysis/candidates.py`, `backend/analysis/emergence.py`, `backend/analysis/register.py`,
`backend/analysis/trends.py`, `configs/default.yaml:54-69`.

### 5.1 What counts as a convention

`emergence_for` (`analysis/emergence.py:220-268`) separates, per expression: the **originator**,
**carried adopters** (used it in a *different* exchange after hearing it elsewhere), **echo-only**
adopters (repeated it in the same exchange), and **independents** (used it with no prior exposure that
could have caused it), then runs a Fisher exact test of carried adoption against independent invention.

**Principle.** Convention in Lewis's (1969) sense — a shared regularity with local meaning — combined
with the transmission-chain requirement that adoption be *caused* by exposure (Mesoudi & Whiten 2008).

**Fidelity: faithful**, and one of the strongest parts of the design. The echo/carried distinction is
exactly the right operationalisation: repeating a phrase back within one conversation is alignment, not
transmission. Four independent "this is not culture" corpora gate the result
(`candidates.py:1-40`): system wording, world wording, the population lexicon
(`simulation/lexicon.py:34-54`), and — importantly — **the actor model's own register**
(`analysis/register.py`), where a phrase several speakers also use in *other, independent runs* is
treated as how the model writes students, not as something this campus coined. That is the correct
control for the deepest LLM confound in the project, and it is already built.

### 5.2 Distinctiveness, stickiness and priming

Covered mechanistically in §3.5. From the language side, three points matter.

**Priming has no speaker.** `priming_line` (`wording.py:65-78`) surfaces bare quoted strings —
"Things {first} has heard people say lately: ..." — even though `heard_from` is stored (`:26`).

**Principles.** Lexical and structural priming and interactive alignment (Bock 1986; Pickering & Garrod
2004); **conceptual pacts** — partner-specific lexical entrainment (Brennan & Clark 1996).

**Fidelity: simplified**, with one decisive consequence. (a) Memory-mediated and phrase-level only;
structural priming is largely syntactic and much shorter-lived. (b) `window_hours: 24` with a summed
exponential weight (`:50-55`) makes this a day-scale effect, whereas dialogue alignment is strongest
turn-to-turn. (c) Because the speaker is dropped, priming **cannot be partner-specific or
prestige-specific**, so Brennan & Clark's conceptual pacts — the mechanism that makes a pair or clique
share a term outsiders do not — are structurally impossible in our design.

**Borrowing (M).** Carry `heard_from` into the priming line and add a short-window partner term
(`priming.partner_window_minutes`), so alignment is with *this interlocutor* rather than with the
population at large. That is precisely the difference between a group dialect and a population-wide
fashion, and telling those apart is the point of the study.

**Prediction.** Conceptual pacts predict higher reuse within the dyad that established a phrase than
across dyads, controlling for exposure. Metric: `next_turn_echo_pct` and `reuse_given_retained_pct`
(`funnel.py:189-194`) split by whether speaker and hearer previously used the phrase together, plus the
transmission tree (`live.py:411-433`). Testable now.

### 5.3 Repeated reference and shortening

**Principle.** Repeated reference to the same referent shortens and standardises descriptions into a
settled form (Clark & Wilkes-Gibbs 1986; Garrod & Doherty 1994); in iterated learning, a transmission
bottleneck drives the same compression (Kirby, Cornish & Smith 2008).

**Fidelity: absent as a mechanism; present only as a hoped-for outcome.** Nothing in the pipeline
shortens a referring expression as a function of how often it has been used. The world side supports it
— referents are persistent, family-neutral things instances recur around
(`simulation/referents.py:14-38`, D53) with `latent_events.referents.reuse: 0.6` — but the agent side
has no length pressure. The measured failure is already on record: 2-5 mentions by 2-3 speakers per
incident, talk over within 2-21 hours (COGNITIVE_FRAMING.md:296). **Nothing was referred to often enough
to need a name.**

**Borrowings.** **(S)** Turn on `latent_events.referents.enabled` and
`latent_events.assignment.mode: home` so the same circle keeps meeting the same referent — this is the
NEED bundle, and it is off in the baseline by design (D72). **(S)** Make the bottleneck real (§3.6): a
binding `memory_capacity` is the iterated-learning pressure toward short, schematic forms.

**Prediction.** A real bottleneck predicts shorter, more schematic surviving expressions. Metric: mean
candidate phrase length and schematicity (`analysis/candidates.py`) plus survival curves — peak and
half-life — from `meme_series` (`trends.py:80`). **Not testable now**: there is zero variance in
forgetting to condition on.

### 5.4 Labels and relational categories

**Principle.** Labels are not just read-outs of categories; a common label helps learners form and share
a relational category (Gentner 2003; Lupyan, Rakison & McClelland 2007), and relational categories are
harder to acquire than feature-based ones (Gentner & Kurtz 2005).

**Fidelity: correctly withheld.** We never ask an agent to name anything, and tests enforce that (D-layer
rules; Orne 1962 on demand characteristics). The consequence is that the label→category direction is
unavailable to our agents by construction, which is a real asymmetry to state: the literature says
labels *help*, and our design forbids the help. That is the right call for the research question — we
are asking whether a label arises, not whether a given label helps — but it means a null result is
partly a design consequence, and should be reported as such.

**Borrowing (M).** A "label offered" arm as an explicit positive control alongside
`controls.planted_phrase` (`configs/default.yaml:180`, D65): one agent is given a habit of using a
phrase for one *relational* family, and the observer measures whether the family becomes more
identifiable in private probes (`analysis/meaning.py:46, :67`) for agents exposed to it. That tests
Lupyan's claim inside our world and gives the null a comparison.

### 5.5 LLM confounds (language)

1. **The register problem is the central one, and it is handled — keep it that way.**
   `register.Background` (`analysis/register.py:60-105`) counts cross-run recurrence. Any candidate that
   several speakers also produce in independent runs is the model's stock phrasing. Never report a
   candidate without its `in_register` flag (`emergence.py:263`).
2. **Two opposite pressures, both artefactual.** The model's helpful, varied register discourages
   repetition and catchphrases (works *against* conventions, COGNITIVE_FRAMING.md:348), while its
   in-context copy prior makes it echo whatever is in the prompt (works *for* them, §3.8.4). They do not
   cancel; they apply at different scales — the copy prior within a conversation, the variety prior
   across days. Report both.
3. **Metalinguistic marking is cheap for an LLM.** `candidates.py` boosts quoted forms and "we're
   calling it" constructions. Models produce these fluently on request-free prompts, so a high
   `nickname_phrases` count (`funnel.py:132-133`) is weak evidence on its own; require the emergence
   test (carried adopters ≥ 2, exceeding independents) as the gate.
4. **Agent-side and observer-side share wordfreq** (§3.8.10). State the non-independence.
5. **Convention formation is known to occur in LLM populations when the task demands coordination**
   (Ashery, Aiello & Baronchelli 2025 [unverified venue]). Our design demands nothing, which is the
   point of the contrast — but it also means "LLMs can do this" is never in question, only "do they do
   it unprompted, under these constraints".

---

## 6. Social structure and transmission

Code: `backend/agents/topology.py`, `backend/agents/conversation.py`,
`backend/agents/group_conversation.py`, `backend/modules/*.py`, `configs/default.yaml:36-45, 110-129,
167-177`.

### 6.1 Network structure

`topology.generate` (`agents/topology.py:85-120`) partitions the population into disjoint circles built
from households, sets dense within-circle ties (`familiarity_within: 0.75`), sparse random between-ties
(`p_between: 0.15`), adds `n_bridges` agents with one strong tie into another circle (`:100-118`), and —
with `shared_meals` — gives each circle a staggered dinner slot so who is co-located agrees with who is
tied (`:67-82`).

**Principles.** Strength of weak ties (Granovetter 1973); **complex contagion** — behaviours and
conventions needing reinforcement from multiple contacts spread *worse* over long ties than simple
contagions do (Centola & Macy 2007; Centola 2010); closed-group convention formation (Garrod & Doherty
1994; Centola & Baronchelli 2015).

**Fidelity: faithful.** This is a textbook complex-contagion topology, deliberately built as an
experimental variable, and the co-location/tie alignment is a real strength — in most agent models
network and space disagree.

**Prediction.** Complex contagion predicts a convention saturates its home circle before crossing a
bridge, and crossings concentrate on bridge agents. Metric: adopter circle membership over time from
`live.adopters_detail` (`:371-405`) crossed with `grounding.circle_membership` (`grounding.py:117`), and
bridge identity from the manifest's topology record. Testable now on any run with
`topology.mode: generated`.

### 6.2 Who talks to whom

`talk_gate` (`conversation.py:42-55`): `p = base_talk_prob × (0.35 + familiarity) × social_trait`,
with a per-pair cooldown and a daily cap; then GA's `decide_to_talk` prompt. Group talk
(`group_conversation.py`) generalises this to a table of 3-5 at shared venues, and evening catch-ups
(D47, `configs/default.yaml:116-121`) fire for close pairs after 17:00.

**Principles.** Homophily and tie strength in interaction rates; conversation as the vehicle of cultural
transmission, with gossip about third parties a large share of talk (Dunbar 2004); sharing of emotional
and unexpected events (Rimé 2009).

**Fidelity: simplified.** Rate scales with familiarity, which is right. Two gaps: (a) **topic choice is
not socially structured** — nothing makes third-party gossip more likely than logistics, and the
measured consequence is 31-53% of utterances about schedules (COGNITIVE_FRAMING.md:301); (b) the
conversation length budget (`max_utterances: 4`) is far below what repeated reference needs, and the
catch-up budget of 6 was added precisely for that reason.

**Borrowing (S).** A weak `conversation.topic_prior` favouring focal points that involve **other
people** over focal points that involve schedules — implemented as a reweighting of the focal-point list
at `conversation.py:162-169`, not as an instruction in the prompt. Rimé and Dunbar both predict this,
and it directly targets the measured logistics dominance.

### 6.3 Overhearing

`overhear_prob` × `busy_factor` per bystander (`conversation.py:37-39`), one i.i.d. draw per utterance
(`:186`), and the overheard fact is the speaker's **exact words** at salience 0.4 (`:247-248`). Overheard
observations are encoded **without** a viewpoint pass (`engine.py:693-707`).

**Principle.** Overhearers understand less than addressees even with identical acoustic access, because
reference is grounded collaboratively *for the addressee* (Schober & Clark 1989; Clark & Wilkes-Gibbs
1986; Clark & Brennan 1991).

**Fidelity: divergent.** We model acoustic **access** and nothing else. The pipeline degrades what you
*see* but not what you *overhear*, which is backwards. Access is also i.i.d. per turn, whereas real
eavesdropping is bursty — you catch a stretch of a conversation, not scattered sentences.

**Borrowings.** **(M)** Route overheard observations through `render_viewpoint` with a new `overhearer`
tag in `prompts/viewpoint_v1.txt` ("heard a fragment of a conversation you are not part of; you do not
know what or whom they are talking about"), config `perception.overhear_viewpoint: true`. **(S)** Make
overhearing contiguous: one draw to enter the channel plus a continuation probability.

**Prediction.** Overhear-only adopters should reproduce a phrase's **form** but not its **referent**.
Metric: `role` in `live.adopters_detail` and the exposure channel (`exposure.listener_ids` vs
conversation participants) against per-use referent agreement from `analysis/meaning.py`. The channel
labels and usage records already exist, so this is testable now.

### 6.4 Social-learning biases (the modules)

| Module | Principle | Fidelity | Key difference |
|---|---|---|---|
| **Prestige** (`modules/prestige.py:21-25, :59-66`) | Prestige-biased social learning: learners attend to models **others** attend to (Henrich & Gil-White 2001; Chudek et al. 2012; Jimenez & Mesoudi 2019 [unverified article no.]) | simplified | Prestige is **exogenous and static** — degree centrality or a hand-supplied table, never updated by who actually got attended to. In Chudek et al. prestige is second-order and *conferred by observed bystander attention*. As implemented it is a popularity prior. Also additive and load-immune (§2.3), so with familiarity it saturates p at 1.0. **0 of 128 archived runs enable it.** |
| **Conformity** (`modules/conformity.py:23-42`) | Conformist transmission; multi-source reinforcement (Boyd & Richerson 1985; Morgan et al. 2012) | faithful | `score += strength·log(distinct corroborating peers)` with peers taken from the agent's *own* chat memories — correctly agent-side, correctly super-linear-then-saturating. The gap is that corroboration uses embedding similarity (cos ≥ 0.35), so "several people said similar things" is lexical, not propositional. |
| **Emotion** (`modules/emotion.py:95-99`) | Emotional selection of content (Heath, Bell & Sternberg 2001); arousal and sharing (Berger & Milkman 2012); arousal-biased competition (Mather & Sutherland 2011) | simplified / partly absent | Enters at encoding (importance) only, never at selection. The `modify_attention` seam exists and is called (`perception.py:76`) and the module does not implement it. |
| **Social reward** (`modules/social_reward.py`) | Disclosure and social connection as intrinsically rewarding (Tamir & Mitchell 2012) | simplified | Reward is partner's Δvalence, routed through memory importance. D36 records the artefact that motivated the current design: injecting a mood word into ~90% of prompts made "restless" the most repeated word in the population. The fix — module outputs enter prompts only when clearly non-neutral — is good practice and should be a general rule for any future module. |

**Borrowings.** **(L)** Endogenous prestige: `module_params.prestige_bias.scores_source: dynamic` with a
`window_hours`, computing prestige from how often others attended to an agent's facts and retold their
utterances in the last window — both available at runtime from `observation` and `exposure` records.
Make the gain multiplicative so load can suppress it. **(S)** Implement `Emotion.modify_attention` as
arousal-biased competition rather than a flat gain: for facts above the agent's median salience in the
beat, p' = p^(1/(1+k·arousal)); below it, p' = p^(1+k·arousal). One method on a class that already
tracks arousal.

**Predictions.** Endogenous prestige predicts rich-get-richer: the Gini of "who is the source of carried
adoptions" rises across the run and transmission trees get deeper and narrower
(`live.transmission_tree:411-433` node degrees, `adopters_detail` source counts). ABC predicts
*narrowing*, not amplification: witness counts rise for the most salient fact of a beat and fall for its
minor facts. Neither is testable now — prestige has never been enabled, and the 3 runs with emotion on
do not touch attention.

### 6.5 LLM confounds (social)

1. **Social-script priors produce the effect regardless of our weights.** The model knows that people
   defer to popular people and copy their friends. Validate every module on engine-side numbers
   (`p_attend`, retrieval scores, exposure counts) **before** reading anything off utterances.
2. **Module prompt lines are stimuli.** D36's lesson generalises: anything a module writes into a prompt
   can itself become the most-repeated phrase in the run. The `manipulation_check` machinery
   (`modules/base.py:32-35`) plus the observer's check against treatment wording is the right guard; use
   it for every new module.
3. **A module that is enabled but never fires is worse than one that is off**, because it appears in the
   condition label. `manipulation_check()["active"]` exists precisely for this and should gate any
   claim about a condition.

---

## 7. Task and meaning (the v3 co-op)

Code: `backend/simulation/workshop.py`, `backend/simulation/regimes.py`,
`backend/simulation/content/laser_alpha.py`, `backend/simulation/records.py`,
`backend/agents/work.py`, `backend/agents/coop_talk.py`, `backend/simulation/coop_world.py`,
`backend/analysis/probes.py`, `backend/analysis/meaning.py`, `configs/default.yaml:183-219`.

This is the newest subsystem and, cognitively, the best-specified: v3 gives agents a **task with a
hidden causal structure** instead of incidents to gossip about.

### 7.1 Hidden classes and a counterbalanced causal mapping

A job faults with probability `p_fault`; each fault has a hidden class K1/K2/K3 with a hidden cause
(LENS / DAMP / BELT) that determines which of six menu actions actually fixes it
(`workshop.py:111-114`, `:212`; `content/laser_alpha.py:9-13, :16-22`). The mapping from **regime** to
**cause** is `{"M1": {"A": "LENS", "B": "DAMP"}, "M2": {"A": "DAMP", "B": "LENS"}}`
(`content/laser_alpha.py:13`), with the mapping chosen by world-seed parity
(`regimes.py:32`), and K2's cause (BELT) stable across regimes (`laser_alpha.py:10`).

**Principles.** Causal structure learning from sparse evidence (Griffiths & Tenenbaum 2005/2009);
latent-cause inference under non-stationarity (Gershman, Blei & Niv 2010); category learning where the
diagnostic feature is relational rather than perceptual (Gentner & Kurtz 2005); explore/exploit under
uncertainty (Daw et al. 2006 [unverified]).

**Fidelity: faithful, and the counterbalance is the strongest methodological device in the repo.** If
agents solve the task from pretrained world knowledge ("a bad cut means a dirty lens"), they are right
in half the (mapping, regime) cells and wrong in the other half, and the **mapping × regime
interaction** in choice accuracy separates prior-driven from experience-driven behaviour. That is a real
test of the central LLM confound, not a hedge about it. `workshop.causal: scrambled` (`:56-57, :321,
:329`) supplies the second control by breaking symptom→cause coherence altogether.

**Prediction.** A pure-prior agent shows a large main effect of *which action the world calls "lens"*
and no effect of regime; an experience-driven agent shows accuracy tracking the current regime with a
lag after each flip, and the lag length is the learning-rate estimate. Both are computable from
`job_truth` records against decision records once a v3 study-1 run exists. **Not testable now** — no
archived run has `workshop.enabled`.

### 7.2 Regime change as a latent-cause problem

`regimes.schedule` flips the active regime on a given day (`regimes.py:37-50`), and the only
agent-visible signal is a **cue text** released at the Stockroom (`regimes.py:7-10`). Nothing names the
regime.

**Principle.** Change-point detection and latent-cause inference: learners must decide whether a run of
failures means *the rule changed* or *I got unlucky* (Gershman, Blei & Niv 2010; Gershman, Radulescu,
Norman & Niv 2014 [unverified]; Gallistel et al. 2001 [unverified]).

**Fidelity: faithful.** This is a clean change-point design with a weak, ambiguous cue — exactly the
regime in which humans over- and under-segment. It also creates the situation this project cares about
most: **a new thing that needs referring to, at a moment when nobody yet has a word for it.** If a
convention is ever going to be coined for a relational category in MemeWorld, the regime flip is the
most likely birthplace.

**Borrowing (S).** Log a per-agent running estimate of "what works now" from the decision trace so the
observer can date each agent's *subjective* change-point and compare it to the true flip day. Dispersion
in subjective change-points across agents is what would make talk about the change useful — and
therefore what would make a shared term worth having.

### 7.3 The binder: external, shared, and wipeable

One binder hangs by the machine (`simulation/records.py`): a front page that is rewritten wholesale
(every revision kept) and an append-only signed log. **Every word in it was written by an agent**; the
world adds only author and timestamp. Reads produce observations encoded as ordinary memories
(`encoder.py:46`, `VERB["record"] = "read in the co-op binder"`). The manipulation is
`records.transitions: [{day, mode: keep|wipe}]`, with a matched world fact in both arms
(`records.py:31-37`).

**Principles.** Distributed cognition and cognitive artifacts (Hutchins 1995; Norman 1991); transactive
memory systems (Wegner 1987); external scaffolding of memory (Clark & Chalmers 1998 [unverified for the
"extended mind" framing specifically]).

**Fidelity: faithful**, and the `keep`/`wipe` contrast is a genuinely good experiment: it asks **where a
convention lives**. If a shared expression survives a wipe, it lived in heads; if it dies, it lived in
the artifact. Matching the world fact across arms is exactly right, because otherwise the wipe arm gets
an extra salient event.

**Prediction.** Distributed-cognition accounts predict a wipe removes the artifact-borne conventions and
leaves the head-borne ones. Metric: candidate survival and `meme_series` share-of-talk
(`trends.py:80`) before vs after the transition day, split by whether the expression's uses trace to
`record` source memories (`memory_encoded.record_ids`, `encoder.py:438`). Not testable now — no
archived run has `records.enabled`.

**Borrowing (M).** `records.retrieval: relevant` already exists as an alternative to `recent`
(`configs/default.yaml:212`). Add a **transactive** variant: what an agent expects to find in the binder
depends on who wrote it, which requires the agent-visible `source_belief` from §3.7. Wegner's system is
about knowing *who knows what*, and we currently give agents no representation of that at all.

### 7.4 Private probes: use vs meaning

`analysis/probes.py` loads a **copy** of each agent's final memory stream, retrieves without touching
access times (`probes.py:126`), and asks what an expression means and which of four incidents it fits.
Answers never enter any memory. `analysis/meaning.py` scores accuracy (`:46`), three-way agreement
(`:51`), a semantic-relatedness index (`:67`) and answer persistence across checkpoints (`:93-101`).

**Principle.** Public use and private representation dissociate: people use words correctly before they
can define them, and the classic test is production vs comprehension/definition.

**Fidelity: faithful**, and this is the design's strongest claim to be measuring *meaning* rather than
*string spread*. The `touch=False` discipline is essential and correctly implemented.

**Prediction.** The interesting result is a **dissociation**: an expression with high carried adoption
(`emergence.carried_rate`) but chance-level probe accuracy (`meaning.acc_cur:46`) is a phrase that
spread without a shared referent — which is a real finding about convention formation, not a null.
Partly testable now on archived runs that have probes.

### 7.5 LLM confounds (task)

1. **World knowledge is the dominant confound, and v3 has the right controls** — `regimes.mapping` and
   `workshop.causal: scrambled`. Run both arms before reporting any learning result.
2. **The six-option menu uses natural, meaningful action names** (`laser_alpha.py:16-22`), so the model
   can reason about plausibility without any evidence. The counterbalance handles this, but the
   *magnitude* of the prior effect should be reported as a headline number, not a footnote.
3. **Binder text is model-written.** Everything in the binder is LLM output, so the artifact inherits
   the model's register wholesale. Register controls (`analysis/register.py`) must be applied to binder
   text as well as speech.
4. **A model asked to decide will decide.** There is no "I don't know" pressure; the "stop and leave the
   job for later" option (`laser_alpha.py:21`) is the only abstention, and its base rate should be
   monitored as a confidence proxy.

---

## 8. What to borrow next

Ranked by value for the research question (does a shared, meaning-bearing expression emerge, and does it
track relational structure?) divided by effort.

| # | Borrowing | Effort | Mechanism sketch | Prediction it unlocks |
|---|---|---|---|---|
| 1 | **Make forgetting bind.** Today 25 forgets in 67,224 encodings; agents have perfect retention and `perfect_memory.yaml` is not a real contrast. | **S** | `memory_capacity: 60-80` for 3-day runs as an explicit factor, or per-tick probabilistic forgetting from the retention function in place of the cliff at `encoder.py:474-484`. Add a `memory_capacity_sweep` condition set. | Iterated-learning compression: surviving expressions get shorter and more schematic (`candidates.py` length, `trends.meme_series:80` half-life). Without a post-encoding bottleneck our framing's central argument has no mechanism. |
| 2 | **Make the viewpoint renderer lose information.** 0 of 7,950 renderings dropped; real-LLM renderings are 1.2-1.6× longer than the world fact. | **M** | Required enum per item in `viewpoint.py:89-99` — `{"perceived": ..., "confidence": "clear\|partial\|none"}`, drop on `none` — plus `perception.viewpoint_max_len_ratio` per vantage, enforced by truncation. | Information monotone in vantage distance, `drop_rate > 0` for far/distracted (`funnel.py:320`). Turns D42 from a paraphrase step into a perceptual filter. |
| 3 | **Record co-witnesses and surface them.** Attention is independent across agents by construction (`engine.py:389-394`), so no agent ever has grounds to expect a shared referent. | **M** | `witnessed_with: [ids]` on each kept fact (`perception.py:90-91`, needs a two-pass loop in `_perceive`), one line into the viewpoint and encode prompts ("Maya and Leo saw this too"); optionally `perception.joint_attention_gain` for a second draw scaled by how many co-present agents noticed. | Co-witnessed events yield more carried adopters per witness than solo-witnessed ones (`emergence.carried_rate`, witness sets from `funnel.py:75-84`). Naming without common ground is the pipeline's weakest joint. |
| 4 | **Mutable memories: post-event misinformation + symmetric reconsolidation.** Nothing in `backend/` ever rewrites `node.description`. | **L** | `memory.misinformation: {enabled, p}`; a heard account may overwrite a detail in an existing witness memory; an association marks the **old** node for revision (reversing `encoder.py:99`); new `memory_revised` trace type; weight the association draw at `:102` by retrieval score. | Witnesses of one event converge in memory content over days, toward the most-talked-about account (pairwise cosine over shared `originating_event_ids` by day). The missing route from "several witnesses" to "one agreed version" to "one name". |
| 5 | **Fix the load model.** Crowding is counted building-wide and kills only the stimulus channel, inverting Lavie (1995) at exactly the crowded meals where most talk happens. | **S** | `perception.load_scope: arena`, `perception.load_includes_facts: true`, and apply the load multiplier to the additive familiarity and prestige terms (`perception.py:51-54, :73-75`). | Crossover: `p_attend`–familiarity correlation flat at low load, steep at high load. Computable on archived runs today. |
| 6 | **Two-trace memory with a decaying verbatim trace.** | **M** | Store `gist` + `verbatim` per node (the raw text already exists at `encoder.py:343` and is already traced at `:443`); `memory.verbatim_half_life_hours: 6`; retrieval returns verbatim only while alive. | The FTT **interaction**: exact-wording reuse decays faster with lag than topic reuse (`funnel.py:189-194` binned by lag). Makes `encoding_noise` a theory, not a dial. |
| 7 | **Base-level activation and a retrieval threshold.** Retrieval currently cannot fail, and a memory retrieved forty times equals one retrieved once. | **M** | Append access times at `retrieval.py:115-117`; `memory.base_level: {form: actr, d: 0.5}`; log `rel_raw` next to `rel` (`:119-121`); `retrieval.min_relevance` applied before normalisation. | Retrieval probability superlinear in cumulative mentions (heavier-tailed share-of-talk, higher early R); and the raw-cosine histogram exposes how many "relevant" retrievals are noise. |
| 8 | **Partner-specific priming with an agent-visible source belief.** `heard_from` is stored (`wording.py:26`) and dropped from the prompt (`:77`). | **M** | Carry the speaker into the priming line; add `priming.partner_window_minutes`; add `MemoryMeta.source_belief` that can fade to "someone" and drift. | Conceptual pacts: dyad-internal reuse exceeds cross-dyad reuse (`funnel.next_turn_echo_pct` split by prior joint use). Unlocks prestige bias, credibility weighting and cryptomnesia, none of which the current memory representation can express. |
| 9 | **Give attention a selection history and a surprise term.** | **M** | `perception.novelty_gain` (salience scaled by embedding distance from recent memory, using the existing embedder at `configs/default.yaml:32-34`) and `perception.habituation_rate` applied to **event** facts, not just ambient routine (generalising `engine.py:418-419` from string equality to embedding neighbourhood). | `p_attend` and utterance counts decline across successive instances of a family; the naming window narrows to the first two instances. The flat null is checkable on archived traces today. |
| 10 | **Route overheard speech through the viewpoint renderer, and make overhearing contiguous.** | **M** | `perception.overhear_viewpoint: true` with an `overhearer` tag in `prompts/viewpoint_v1.txt`; replace the per-turn draw (`conversation.py:186`) with enter + continuation probabilities. | Overhear-only adopters copy form without referent (role × referent agreement from `analysis/meaning.py`). Currently our main lateral transmission channel is lossless, contradicting Schober & Clark (1989). |

Also cheap and worth doing, below the top ten: lift the reaction-remark constants into config
(`engine.py:485, :506`) so perception sweeps reach the widest channel; persona-weighted lens foci
(`encoder.py:80`); `retrieval.rif_hours` (five lines); `memory.retention: power`; content-biased
encoding retention; `memory.decay_rate_by_importance`; `need.resolve_decay` and turning NEED on.

---

## 9. Predictions we can already test with existing runs

These need no code change — only analysis of `runs/*/trace.jsonl`. Baselines in brackets are from the
128-run archive; where generated text is involved, restrict to the ~19 `claude_cli` runs.

| Prediction | Principle | Observer metric | Status in the archive |
|---|---|---|---|
| Information falls monotonically with vantage distance | Situated construal | Jaccard(`world_text`, `perceived`) by `vantage` in `viewpoint` records (`viewpoint.py:103-104`) | **Already falsified** for `distracted` (0.978 > `near` 0.922); `drop_rate` = 0/7,950 |
| `p_attend`–familiarity coupling steepens with load | Perceptual load (Lavie 1995) | Partial correlation of `p_attend` with max familiarity, binned by crowd size from `frames.jsonl` | Computable; code predicts the wrong sign |
| `p_attend` declines across instances of one family | Bayesian surprise / habituation | `p_attend` by within-family instance index (`observation` × `events.jsonl`) | Flat null checkable; n = 3,351 non-participant draws |
| Facts released while the observer is in conversation get lower `p_attend` | Inattentional blindness | `p_attend` split by the observer's `conversation` field in `frames.jsonl` | Testable but thin: 32 `distracted` renderings |
| Witness membership is independent of interests given co-presence | Goal-directed attention (Corbetta & Shulman 2002) | `observation` records × population profile interests | Today's null is checkable; a violation would mean interests leak in via the persona |
| Person-involving facts survive encoding more than object facts at equal salience | Content biases in transmission | `drop_fact` entries in `encoding_ops` joined to the `observation` fact list | Fully testable; 411 drops, 2,536 generalisations |
| Self-involving memories are retrieved more per unit importance | Self-reference effect | `retrieved` / `retrieval_scores` (`reflection.py:54-56`) joined on `memory_encoded.self_experience` (`encoder.py:444`) | Testable; `self_experience` is traced on every encoding |
| Lens `noise` should **not** predict later retrieval count | Encoding variability is output diversity only | `lens.noise` (`encoder.py:443`) vs per-node retrieval count | Testable; a strong correlation would mean "variability" is acting as a hidden importance proxy |
| Focus match should **not** predict adoption (focus is uniform) | Sanity check on the lens | `lens.focus` × time-to-adoption (`trends.adoptions:47`) | Testable; a positive result falsifies the uniform-sampling assumption |
| Witnesses of one event do **not** currently converge in memory | Baseline for misinformation | Pairwise cosine over memories sharing `originating_event_ids`, by day | Testable; this is the null the borrowing would overturn |
| Exact-wording reuse decays faster with lag than topic reuse | Fuzzy-trace theory | `bigrams_retained_verbatim_pct`, `reuse_given_retained_pct` (`funnel.py:189-194`) binned by lag; vs `afterlife_hours_med` (`:62-66`) | Testable; utterance ticks give the lag |
| Reminding links are surface-driven, not family-driven | MAC/FAC (Gentner et al. 1993) | `reminding_same_family_pct` vs `reminding_same_family_chance_pct` (`funnel.py:120-121`) | Testable on 818 realised links across the archive |
| R for phrases from distorted memories is **not** lower (no confidence gate) | Metamemory as a transmission brake | `meme_series` R (`trends.py:80`) split by `lens.distort` on the carrier memory | Testable; a clean manipulation check — the prediction should fail |
| Dyad-internal reuse exceeds cross-dyad reuse | Conceptual pacts | `next_turn_echo_pct` / `reuse_given_retained_pct` split by prior joint use; `live.transmission_tree:411` | Testable |
| A disproportionate share of first exposures traces to reaction remarks | Channel audit | `source` field on `utterance` records (`engine.py:494`) × `first_exposure_tick` in `adopters_detail` | Testable; remarks broadcast at 3× the configured overhear rate |
| Facts per `observation` are unbounded on crowded ticks | Missing capacity limit | `len(facts)` per `observation` vs facts released that tick (`event_beat`) | Testable; the independent-coin model predicts a binomial spread with no ceiling |
| Conventions saturate a circle before crossing a bridge | Complex contagion (Centola & Macy 2007) | Adopter circle membership over time (`adopters_detail:371` × `grounding.circle_membership:117`) | Testable on any `topology.mode: generated` run |
| Use without meaning: high carried adoption, chance probe accuracy | Public use vs private representation | `emergence.carried_rate` × `meaning.acc_cur:46`, `meaning.sri:67` | Partly testable on runs that have probes |

Predictions that need a **new run**: joint attention, misrecognition, modality-typed audibility,
arousal-biased competition, endogenous prestige, base-level activation, retrieval threshold (needs
`rel_raw` logging first), schema blending, the forgetting bottleneck, and everything in §7 (no archived
run has `workshop.enabled` or `records.enabled`).

---

## 10. Reading list

Grouped by subsystem. **[unverified]** marks an entry whose page range, chapter, article number or venue
we could not confirm; the substantive claim attributed to it is the part we are confident about. Works
already listed in COGNITIVE_FRAMING.md's reference section are not repeated here unless this document
leans on a different part of them.

**Perception and attention.**
Itti, L., & Koch, C. (2001). Computational modelling of visual attention. *Nature Reviews Neuroscience, 2*, 194-203. ·
Wolfe, J. M., & Horowitz, T. S. (2017). Five factors that guide attention in visual search. *Nature Human Behaviour, 1*, 0058. ·
Fecteau, J. H., & Munoz, D. P. (2006). Salience, relevance, and firing: a priority map for target selection. *TiCS, 10*(8), 382-390. ·
Bisley, J. W., & Goldberg, M. E. (2010). Attention, intention, and priority in the parietal lobe. *Annual Review of Neuroscience, 33*, 1-21. ·
Lavie, N. (1995). Perceptual load as a necessary condition for selective attention. *JEP:HPP, 21*(3), 451-468. ·
Lavie, N. (2005). Distracted and confused? Selective attention under load. *TiCS, 9*(2), 75-82. ·
Murphy, G., Groeger, J. A., & Greene, C. M. (2016). Twenty years of load theory. *Psychonomic Bulletin & Review, 23*(5), 1316-1340. **[unverified pages]** ·
Simons, D. J., & Chabris, C. F. (1999). Gorillas in our midst. *Perception, 28*(9), 1059-1074. ·
Most, S. B., et al. (2005). What you see is what you set. *Psychological Review, 112*(1), 217-242. **[unverified pages]** ·
Posner, M. I. (1980). Orienting of attention. *QJEP, 32*(1), 3-25. ·
Eriksen, C. W., & St James, J. D. (1986). Zoom lens model. *Perception & Psychophysics, 40*(4), 225-240. ·
Corbetta, M., & Shulman, G. L. (2002). Control of goal-directed and stimulus-driven attention. *Nature Reviews Neuroscience, 3*, 201-215. ·
Folk, C. L., Remington, R. W., & Johnston, J. C. (1992). Involuntary covert orienting is contingent on attentional control settings. *JEP:HPP, 18*(4), 1030-1044. ·
Reynolds, J. H., & Heeger, D. J. (2009). The normalization model of attention. *Neuron, 61*(2), 168-185. ·
Desimone, R., & Duncan, J. (1995). Neural mechanisms of selective visual attention. *Annual Review of Neuroscience, 18*, 193-222. ·
Itti, L., & Baldi, P. (2009). Bayesian surprise attracts human attention. *Vision Research, 49*(10), 1295-1306. ·
Mather, M., & Sutherland, M. R. (2011). Arousal-biased competition in perception and memory. *Perspectives on Psychological Science, 6*(2), 114-133. ·
Tomasello, M., & Farrar, M. J. (1986). Joint attention and early language. *Child Development, 57*(6), 1454-1463. ·
Shteynberg, G. (2015). Shared attention. *Perspectives on Psychological Science, 10*(5), 579-590. ·
Rankin, C. H., et al. (2009). Habituation revisited. *Neurobiology of Learning and Memory, 92*(2), 135-138.

**Memory: encoding, distortion, forgetting.**
Bartlett, F. C. (1932). *Remembering*. Cambridge University Press. ·
Brewer, W. F., & Treyens, J. C. (1981). Role of schemata in memory for places. *Cognitive Psychology, 13*(2), 207-230. ·
Gilboa, A., & Marlatte, H. (2017). Neurobiology of schemas and schema-mediated memory. *TiCS, 21*(8), 618-631. ·
Reyna, V. F., & Brainerd, C. J. (1995). Fuzzy-trace theory: an interim synthesis. *Learning and Individual Differences, 7*(1), 1-75. ·
Brainerd, C. J., & Reyna, V. F. (2002). Fuzzy-trace theory and false memory. *Current Directions, 11*(5), 164-169. ·
Allport, G. W., & Postman, L. (1947). *The Psychology of Rumor*. Holt. ·
Bebbington, K., MacLeod, C., Ellison, T. M., & Fay, N. (2017). The sky is falling: negativity bias in social transmission. *Evolution and Human Behavior, 38*(1), 92-101. ·
Hupbach, A., Gomez, R., Hardt, O., & Nadel, L. (2007). Reconsolidation of episodic memories. *Learning & Memory, 14*(1-2), 47-53. ·
Nader, K., Schafe, G. E., & LeDoux, J. E. (2000). Fear memories require protein synthesis. *Nature, 406*, 722-726. ·
Roediger, H. L., & McDermott, K. B. (1995). Creating false memories. *JEP:LMC, 21*(4), 803-814. ·
Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring. *Psychological Bulletin, 114*(1), 3-28. ·
Mitchell, K. J., & Johnson, M. K. (2009). Source monitoring 15 years later. *Psychological Bulletin, 135*(4), 638-677. **[unverified]** ·
Brown, A. S., & Murphy, D. R. (1989). Cryptomnesia. *JEP:LMC, 15*(3), 432-442. **[unverified pages]** ·
Hunt, R. R. (1995). The subtlety of distinctiveness: what von Restorff really did. *Psychonomic Bulletin & Review, 2*(1), 105-112. ·
Slamecka, N. J., & Graf, P. (1978). The generation effect. *JEP:HLM, 4*(6), 592-604. ·
MacLeod, C. M., et al. (2010). The production effect. *JEP:LMC, 36*(3), 671-685. ·
Wixted, J. T., & Ebbesen, E. B. (1991). On the form of forgetting. *Psychological Science, 2*(6), 409-415. ·
Rubin, D. C., & Wenzel, A. E. (1996). One hundred years of forgetting. *Psychological Review, 103*(4), 734-760. ·
Nørby, S. (2015). Why forget? On the adaptive value of memory loss. *Perspectives on Psychological Science, 10*(5), 551-578. **[unverified]** ·
Bower, G. H., Black, J. B., & Turner, T. J. (1979). Scripts in memory for text. *Cognitive Psychology, 11*(2), 177-220. ·
Schank, R. C., & Abelson, R. P. (1977). *Scripts, Plans, Goals and Understanding*. Erlbaum. ·
Zacks, J. M., et al. (2007). Event perception: a mind-brain perspective. *Psychological Bulletin, 133*(2), 273-293. **[unverified pages]** ·
Zeigarnik, B. (1927). Über das Behalten von erledigten und unerledigten Handlungen. *Psychologische Forschung, 9*, 1-85. ·
Masicampo, E. J., & Baumeister, R. F. (2011). Consider it done! *JPSP, 101*(4), 667-683. ·
Yonelinas, A. P., & Ritchey, M. (2015). The slow forgetting of emotional episodic memories. *TiCS, 19*(5), 259-267. **[unverified pages]**

**Retrieval, activation and analogical access.**
Anderson, J. R., & Schooler, L. J. (1991). Reflections of the environment in memory. *Psychological Science, 2*(6), 396-408. ·
Anderson, J. R., & Milson, R. (1989). Human memory: an adaptive perspective. *Psychological Review, 96*(4), 703-719. ·
Anderson, J. R., & Lebiere, C. (1998). *The Atomic Components of Thought*. Erlbaum (base-level learning). ·
Pavlik, P. I., & Anderson, J. R. (2005). Practice and forgetting effects on vocabulary memory. *Cognitive Science, 29*(4), 559-586. ·
Kahana, M. J. (2020). Computational models of memory search. *Annual Review of Psychology, 71*, 107-138. ·
Raaijmakers, J. G. W., & Shiffrin, R. M. (1981). Search of associative memory. *Psychological Review, 88*(2), 93-134. ·
Shiffrin, R. M., & Steyvers, M. (1997). A model of recognition memory: REM. *Psychonomic Bulletin & Review, 4*(2), 145-166. ·
Tulving, E., & Thomson, D. M. (1973). Encoding specificity. *Psychological Review, 80*(5), 352-373. ·
Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning. *Psychological Science, 17*(3), 249-255. ·
Anderson, M. C., Bjork, R. A., & Bjork, E. L. (1994). Remembering can cause forgetting. *JEP:LMC, 20*(5), 1063-1087. ·
Antony, J. W., et al. (2017). Retrieval as a fast route to memory consolidation. *TiCS, 21*(8), 573-576. ·
Gentner, D., Rattermann, M. J., & Forbus, K. D. (1993). The roles of similarity in transfer. *Cognitive Psychology, 25*(4), 524-575. ·
Forbus, K. D., Gentner, D., & Law, K. (1995). MAC/FAC: a model of similarity-based retrieval. *Cognitive Science, 19*(2), 141-205. ·
Schank, R. C. (1982). *Dynamic Memory*. Cambridge University Press. ·
Ross, B. H. (1984). Remindings and their effects in learning a cognitive skill. *Cognitive Psychology, 16*, 371-416.

**Language, dialogue and convention.**
Clark, H. H., & Wilkes-Gibbs, D. (1986). Referring as a collaborative process. *Cognition, 22*(1), 1-39. ·
Brennan, S. E., & Clark, H. H. (1996). Conceptual pacts and lexical choice. *JEP:LMC, 22*(6), 1482-1493. ·
Clark, H. H., & Brennan, S. E. (1991). Grounding in communication. In Resnick et al. (Eds.), *Perspectives on Socially Shared Cognition*. APA. ·
Schober, M. F., & Clark, H. H. (1989). Understanding by addressees and overhearers. *Cognitive Psychology, 21*, 211-232. ·
Schober, M. F. (1993). Spatial perspective-taking in conversation. *Cognition, 47*(1), 1-24. ·
Keysar, B., et al. (2000). Taking perspective in conversation. *Psychological Science, 11*(1), 32-38. ·
Epley, N., et al. (2004). Perspective taking as egocentric anchoring and adjustment. *JPSP, 87*(3), 327-339. ·
Bock, J. K. (1986). Syntactic persistence in language production. *Cognitive Psychology, 18*(3), 355-387. ·
Pickering, M. J., & Garrod, S. (2004). Toward a mechanistic psychology of dialogue. *BBS, 27*(2), 169-226. ·
Garrod, S., & Doherty, G. (1994). Conversation, co-ordination and convention. *Cognition, 53*, 181-215. ·
Kirby, S., Cornish, H., & Smith, K. (2008). Cumulative cultural evolution in the laboratory. *PNAS, 105*, 10681-10686. ·
Lupyan, G., Rakison, D. H., & McClelland, J. L. (2007). Language is not just for talking. *Psychological Science, 18*, 1077-1083. ·
Gentner, D., & Kurtz, K. J. (2005). Relational categories. In Ahn et al. (Eds.), *Categorization Inside and Outside the Laboratory*. APA.

**Social transmission and cultural evolution.**
Henrich, J., & Gil-White, F. J. (2001). The evolution of prestige. *Evolution and Human Behavior, 22*(3), 165-196. ·
Chudek, M., Heller, S., Birch, S., & Henrich, J. (2012). Prestige-biased cultural learning. *Evolution and Human Behavior, 33*(1), 46-56. ·
Jimenez, A. V., & Mesoudi, A. (2019). Prestige-biased social learning: current evidence and outstanding questions. *Humanities and Social Sciences Communications, 5*. **[unverified article number]** ·
Boyd, R., & Richerson, P. J. (1985). *Culture and the Evolutionary Process*. University of Chicago Press. ·
Granovetter, M. S. (1973). The strength of weak ties. *American Journal of Sociology, 78*(6), 1360-1380. ·
Centola, D., & Macy, M. (2007). Complex contagions and the weakness of long ties. *AJS, 113*, 702-734. ·
Centola, D., & Baronchelli, A. (2015). The spontaneous emergence of conventions. *PNAS, 112*, 1989-1994. ·
Heath, C., Bell, C., & Sternberg, E. (2001). Emotional selection in memes. *JPSP, 81*(6), 1028-1041. ·
Berger, J., & Milkman, K. L. (2012). What makes online content viral? *JMR, 49*, 192-205. ·
Norenzayan, A., Atran, S., Faulkner, J., & Schaller, M. (2006). Memory and mystery. *Cognitive Science, 30*(3), 531-553. ·
Dunbar, R. I. M. (2004). Gossip in evolutionary perspective. *Review of General Psychology, 8*, 100-110. ·
Rimé, B. (2009). Emotion elicits the social sharing of emotion. *Emotion Review, 1*, 60-85.

**Task, causal structure and external cognition.**
Griffiths, T. L., & Tenenbaum, J. B. (2009). Theory-based causal induction. *Psychological Review, 116*(4), 661-716. **[unverified pages]** ·
Gershman, S. J., Blei, D. M., & Niv, Y. (2010). Context, learning, and extinction. *Psychological Review, 117*(1), 197-209. ·
Gershman, S. J., Radulescu, A., Norman, K. A., & Niv, Y. (2014). Statistical computations underlying the dynamics of memory updating. *PLoS Computational Biology, 10*(11). **[unverified]** ·
Daw, N. D., O'Doherty, J. P., Dayan, P., Seymour, B., & Dolan, R. J. (2006). Cortical substrates for exploratory decisions in humans. *Nature, 441*, 876-879. **[unverified]** ·
Hutchins, E. (1995). *Cognition in the Wild*. MIT Press. ·
Norman, D. A. (1991). Cognitive artifacts. In J. M. Carroll (Ed.), *Designing Interaction*. Cambridge University Press. ·
Wegner, D. M. (1987). Transactive memory. In Mullen & Goethals (Eds.), *Theories of Group Behavior*. Springer. ·
Clark, A., & Chalmers, D. (1998). The extended mind. *Analysis, 58*(1), 7-19. **[unverified for the specific framing used here]**

**LLMs as subjects and as confounds.**
Acerbi, A., & Stubbersfield, J. M. (2023). Large language models show human-like content biases in transmission chain experiments. *PNAS, 120*(44), e2313790120. ·
Binz, M., & Schulz, E. (2023). Using cognitive psychology to understand GPT-3. *PNAS, 120*(6). ·
Dillion, D., Tandon, N., Gu, Y., & Gray, K. (2023). Can AI language models replace human participants? *TiCS, 27*, 597-600. ·
Ashery, A. F., Aiello, L. M., & Baronchelli, A. (2025). Emergent social conventions and collective bias in LLM populations. *Science Advances*. **[unverified venue]** ·
Park, J. S., et al. (2023). Generative agents: interactive simulacra of human behavior. *UIST*. ·
Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2024). Cognitive architectures for language agents. *TMLR*. ·
Orne, M. T. (1962). On the social psychology of the psychological experiment. *American Psychologist, 17*, 776-783.
