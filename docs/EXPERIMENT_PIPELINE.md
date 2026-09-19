# Memetics experiment pipeline

This pipeline runs the existing agent society, observes the ideas that develop, and supports both planned comparisons and questions asked after seeing the history. It does not give agents a meme inventory, target vocabulary, definition of success, or researcher interpretations. The world and agent decision rules remain the experimental environment.

The main experiment library targets the **campus society**: ordinary routines, relationships, conversation, memory and reflection. Population is configurable; the default is the 16-person campus, and the supplied 500-person population is available for larger studies. The observer lives in `backend/research` and is selected by `analysis.pipeline: memetics`. The older workshop studies are outside the main library.

## What the complete workflow does

1. Expand an editable experimental design into conditions, world seeds, and optional agent replicas.
2. Run each society in its own process, recording the resolved configuration, simulation code hashes, model calls, event trace, frames, and committed tick digests.
3. Index the committed history. Distinguish speech, authored records, private representations, initial persona material, and environmental evidence.
4. Ask an external LLM to inspect chronological windows of **all eligible textual events**, with retrieved earlier concept histories and original evidence. There is no rare-phrase gate, minimum popularity requirement, predetermined list of ideas, or requirement that an idea improve task performance.
5. Validate source IDs, exact quotations, and temporal ordering. Save provisional concept histories, contextual interpretations, changes, relationships, other observations, uncertainty, and revisions.
6. Calculate circulation, persistence, sense distributions, expressed stances, and possible exposure-to-reuse paths from the validated annotations and simulation receipts.
7. Investigate each design's motivating questions within each completed society, then synthesize across societies while retaining original, namespaced evidence citations. Researchers can add any subsequent question without changing the world.
8. Produce per-society and cross-condition reports, uncertainty intervals, SVG figures, CSV tables, Markdown, JSON, and an interactive research interface.

The observation pass is independent of the question list. An experiment about memory can still reveal affiliation, disagreement, metaphor, categorization, or a development the researchers did not anticipate.

## Launching a study

All commands below are for **Windows CMD**, from the repository root. They assume the project's dependencies are installed in `.venv`; use your existing Python environment if it has a different name.

```cmd
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment list
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment expand memory
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment run memory --dry-run
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment run memory --parallel 2
```

`experiment run` executes the full selected study, including discovery, the saved questions, cross-society interpretation, and reports. It skips compatible completed work when repeated. Paused campus cells continue from a verified recorded prefix when the experiment is run again; simulation and observer failures remain visible; a failed call never becomes a finding of no cultural change. A nonzero exit status means the selected pipeline did not complete.

The supplied campus designs use the GPT API. Paste your key into `OPENAI_API_KEY=` in `.env`; this local file is ignored by Git. Optional `OPENAI_MODEL` and `OPENAI_OBSERVER_MODEL` fields select models; their resolved names are recorded with the run, while the key stays outside artifacts. Environment variables take precedence over `.env`. Live calls require a valid key with access to the selected models. Inspect the expansion before launching: these are full studies, not short demonstration runs, and observer/inquiry calls add to simulation usage. The interface displays the number of societies and whether the selected providers are live. Studies are launched individually.

Mock execution is available for checking software and configuration:

```cmd
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment run emergence --backend mock --parallel 2
```

**Mock mode performs no semantic judgment.** It produces empty discovery outputs and explicit mock labels in the interface and reports. Those results cannot support conclusions about cultural emergence, absence, or drift. Synthetic semantic fixtures used in automated tests are also software checks, not evidence about LLM societies.

Useful controls:

```cmd
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment status memory
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment report memory
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment run memory --seed 11 --only encoding-lossy
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment run memory --no-questions
```

`--seed` and `--only` select a subset for execution, recovery, or a particular investigation. Reports retain the other design cells as missing. `--no-questions` skips question-specific LLM inquiries; the complete discovery pass, measurements, and reports still run. It does not constrain what the observer can discover.

Use `--runs-root` or `MEMEWORLD_RUNS_ROOT` to select a different artifact directory. A backend override that changes the agent backend uses a separate design output directory, such as `memetics_memory__mock`, so mock runs do not satisfy a live study's completion checks.

## Campus experiments

All main-library designs use the campus engine. Most default to five world seeds, 30 simulated days and 16 people. `naming` uses 500 people, three days and the existing cohort-specific initial memories. The default campus has no injected background and no scripted latent events. Enable latent events explicitly in a custom design if needed.

| ID | Active treatments |
|---|---|
| `emergence` | Independent campus societies under stable conditions |
| `memory` | Encoding noise × verbatim wording retention |
| `abstraction` | Reminding × reflection |
| `communities` | Generated cross-group ties × bridge count |
| `communication` | Dyadic exchange length × friend catch-ups × group conversation |
| `influence` | Prestige weighting × conformity |
| `social_meaning` | Social reward × emotion |
| `contingency` | Agent model × independently seeded agent replicas |
| `long_horizon` | 30-day and 90-day campus histories |
| `shared_background` | No introduction versus a supplied Markdown world introduction |
| `naming` | Preferred-name memories versus explicit knowledge of both dining-hall names |

The common reference is [`configs/memetics.yaml`](../configs/memetics.yaml). The library contains 11 campus studies. Existing workshop-specific record, turnover and regime designs remain separate files and are not offered in the campus launcher.

## Shared Markdown introduction and population

The introduction is supplied to **every agent in identity prompts throughout the run**, including the upstream conversation prompts. This is persistent background context, rather than an episodic memory subject to forgetting. Agents receive ordinary Markdown text; no special meme inventory or forced adoption rule is added. The observer treats the introduction and all seed memories as **initial material**, separately from public invention, speech and transmission.

Edit [`configs/backgrounds/campus.md`](../configs/backgrounds/campus.md), or create your own UTF-8 Markdown file. Its text and SHA-256 are frozen into the resolved configuration, and each run writes `shared_background.md` plus per-agent supply receipts. Replay uses the saved text even if the source Markdown changes or disappears. A changed introduction produces an incompatible condition instead of silently reusing an old simulation.

```cmd
REM Complete offline verification, including questions and cross-society reports
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment run configs\designs\memetics_smoke.yaml --parallel 2

REM Compare supplied background with an empty-background control
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment run shared_background --background configs\backgrounds\campus.md --backend mock --days 1 --population-size 8 --seed 11

REM Main campus study with a different population, inspected before a live run
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment expand emergence --population configs\population\homewood500.yaml --population-size 500 --days 30
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment check emergence --population configs\population\homewood500.yaml --population-size 500 --days 30
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment run emergence --population configs\population\homewood500.yaml --population-size 500 --days 30 --parallel 2

REM One society with a supplied intro
.venv\Scripts\python.exe -X utf8 -m backend.cli run --config configs\memetics.yaml --background configs\backgrounds\campus.md --set llm.backend=mock analysis.observer.backend=mock simulation_days=1 --analyze
```

`--population` alone uses the entire selected population file. `--population-size` selects its first N people; it is **not a representative or cohort-balanced sample**. In particular, a small prefix of the 500-person naming population cannot stand in for its full 20/80 split. `--days` overrides every cell's horizon for a pilot, including a long-horizon comparison; use the original horizons for the actual comparison.

Customization is supported by `expand`, `check`, `run`, `status`, `report` and `synthesize`. Reuse the same options to address the same study. A custom design is saved under `runs/.designs/` with a stable content-derived name, so all workers use the same design and its results stay separate. Factor levels take precedence over common defaults; in `shared_background`, `--background` replaces only the supplied introduction, preserving the absent control. `--agent-model` sets the common agent model, while `--observer-model` sets the common observer. API backends require full provider model IDs, rather than CLI aliases.

`experiment check` validates local population, duration and input files, provider availability and model configuration. It makes no paid calls and does not verify remote authentication. An empty GPT key fails before launching societies, with a message pointing to `.env`. The Research interface offers the same population, duration, model and Markdown fields and previews the corresponding run status.

## Creating or changing experiments

Copy a design and edit its factors, schedules, seeds, questions or measurement columns. The controller checks that different factors do not write overlapping configuration keys. `common` applies before the selected factor overlays. `controls` can express additional comparison conditions using the existing design format.

```cmd
copy configs\designs\memetics\memory.yaml configs\designs\my_memory_study.yaml
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment expand configs\designs\my_memory_study.yaml
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment run configs\designs\my_memory_study.yaml --parallel 2
```

Change `name` in the copy so its recordings have a distinct output directory. Change `questions` freely. A question is a research lens, not a demand that a phenomenon appear. To modify one society outside a factorial design, use the existing `run --config ... --set ... --analyze` command with `configs/memetics.yaml`.

`seeds` are world randomization blocks. With `replicates: 1`, the existing seed behavior is preserved. With multiple replicas, the world seed remains the same within a block and the agent seed varies deterministically by replica. Replicas are averaged within a world block before cross-society uncertainty calculations. This avoids counting matched worlds or multiple agent realizations as independent world samples.

Changing a saved design does not change its existing simulated history. Use a new design name for a different condition definition, or preserve a clearly separated output root. Observer compatibility is checked before comparison, and alternative observer versions should be compared or audited separately.

The controller also compares resolved simulation settings and population/initial-memory file hashes: an existing recording with different world/agent settings is marked `incompatible`, preserved, and rerun when requested. Changing only observer settings or research questions does not require a new simulation.

## Open observation and additional questions

An existing recorded society can be observed without rerunning its agents:

```cmd
.venv\Scripts\python.exe -X utf8 -m backend.cli observe runs\MY_RUN --backend openai --model gpt-5.4
.venv\Scripts\python.exe -X utf8 -m backend.cli inquire runs\MY_RUN "How did the meaning of responsibility change, and which agents disagreed?"
.venv\Scripts\python.exe -X utf8 -m backend.cli inquire runs\MY_RUN "What did newcomers reinterpret?" --start 600 --end 1199
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment synthesize inheritance "Which ideas outlived the departure of their earliest speakers?"
```

Other inquiry options are `--agent AGENT_ID`, `--public-only`, `--limit N`, and observer `--backend` / `--model`. Start/end ticks are inclusive. The question, scope, retrieval plan, selected evidence, computations, model configuration and answer are saved together.

Discovery and inquiry have different coverage. Discovery traverses every eligible textual event. An inquiry retrieves a bounded set using LLM-expanded search phrases and links to discovered histories, then computes measurements over all in-scope annotated occurrences of the retrieved histories. It discloses how many events were eligible and supplied. A question-specific answer is therefore not proof that no counterexample exists elsewhere in the archive.

Time-filtered inquiries are **retrospective**: their concept identities may have been constructed with later history even though their source evidence is filtered. For a genuine as-of analysis, observe the committed prefix while the simulation is running:

```cmd
.venv\Scripts\python.exe -X utf8 -m backend.cli observe runs\MY_RUN --watch --interval 30
```

Each distinct committed input and observer specification produces a saved analysis snapshot. Repeated observation reuses cached LLM responses where the exact request matches. `--no-publish` saves an alternative observation without replacing the run's current analysis. This is useful with a different observer model. A complete observation of a running prefix is not a finished society; experiment comparisons require finished, compatible runs.

Use `observe ... --fresh` to record another complete judgment pass with a separate cache, including when recovering from consistently invalid cached responses. Previous snapshots remain available. Invalid discovery responses receive up to two correction requests; all failures are recorded, and an unsuccessful correction leaves the observation failed rather than reporting zero findings.

## What the judge decides—and what software calculates

The judge proposes provisional concept histories and contextual meanings. It can recognize different wording as related, distinguish multiple meanings of one expression, connect or differentiate concepts, and describe changes in content, implications, social function or stance. Those descriptions are open text, not a closed list of cultural types.

Every annotated occurrence has an exact source quotation and an evidence ID. Speaker, tick, channel and conversation are copied from the recording, not inferred by the judge. A claimed change needs interpreted source evidence before and after, with valid temporal ordering. Separate conversations at the same tick have no assumed order. Invalid evidence links or nonverbatim quotations fail the observation rather than silently entering metrics.

The observer distinguishes the following:

- **Initial material:** persona background/habits, supplied seed memories and shared Markdown background, available from the person's arrival. A first locally observed expression is not automatically an invention independent of model pretraining or initialization.
- **Private representation:** memory, reflection, viewpoint or decision text. It is not counted as public circulation.
- **Public use:** speech or an authored record containing the interpreted idea. Public use can express rejection, quotation, irony or uncertainty; it is not automatically endorsement.
- **Exposure:** a logged listener or record-reading receipt. Exposure is not evidence of understanding or adoption.
- **Possible reuse after exposure:** first public use following a logged exposure to a related annotated occurrence. All candidate sources remain visible; this is not causal source identification.

The deterministic measurements are:

| Measurement | Meaning and denominator |
|---|---|
| Concept histories | Observer-proposed histories, including private and single-occurrence histories. The count depends on observer granularity. |
| Public histories | Histories with speech or authored-record occurrences. |
| Multi-agent histories | Histories with public uses by at least two agents. This is a descriptive count, not the project's definition of a meme. |
| Judged changes | Source-linked before/after interpretations, retaining their stated uncertainty. |
| Public users and reach | Distinct public users; daily reach divides observed users by agents active that day. |
| Observed persistence | Span from first to last public use, in simulated days. It is not an estimate of extinction or unobserved continuity. |
| Senses and forms | Observer-distinguished interpretations and the quoted formulations in which they were observed. |
| Agent meaning distributions | Within an agent/day, annotated public uses are normalized across senses. Silence leaves the distribution missing. |
| Collective meaning distributions | Average of those normalized distributions across observed agents, giving each agent equal weight. |
| Pairwise agreement | Mean dot product of the observed agents' sense distributions. Missing when fewer than two agents are observed. |
| Within-agent distribution change | Jensen–Shannon distance from the agent's previous observed daily distribution. No evidence is inserted for intervening silent days. |
| Possible transmission | Logged exposure followed by first public use; identifies same-exchange echoes and later exchanges, source alternatives, and observed sense continuity. |
| Stance and function | Free-text interpretations attached to uses, so prevalence can be inspected alongside support, rejection or other pragmatic functions. |

Reports also expose realized event counts, active populations, existing environment manipulation checks and agent-call errors. These descriptive records help check whether the configured communication, record, turnover and pressure mechanisms operated in the actual history.

Group distributions use the group membership recorded in the run manifest. For turnover/cohort questions, inspect arrival evidence and agent histories rather than assuming static social groups are newcomer cohorts. Counts describe the discovered repertoire, not all concepts an agent could possess.

The observer never needs to label a concept as a meme before tracking it. Researchers can inspect histories under several interpretations of the meme concept without changing how the society runs.

## Comparisons and uncertainty

Reports retain every expected run with its status. They average agent replicas within world-seed blocks, then summarize conditions across complete world blocks. A block with a missing replica is excluded for that measurement and its missing run remains visible.

The report enumerates comparisons differing on exactly one factor, matching world seeds. Differences are **B minus A**. The 95% intervals resample independent world blocks, using 2,000 deterministic bootstrap draws. Fewer than two observed blocks produce no interval. Missing measurements are not replaced with zero.

These intervals are descriptive/exploratory and not adjusted for multiple comparisons. They do not measure judge uncertainty. Five world seeds are configuration defaults, not a claim of adequate power for every effect. Model replicas with shared worlds, different points on one history and multiple utterances from one society do not become extra independent world samples.

Cross-society inquiry uses namespaced original evidence IDs. Similar concept labels do not establish equivalence. Its synthesis must consider divergent interpretations, counterexamples and missing societies. The individual society reports remain available for checking the synthesis.

## Auditing the observer

```cmd
.venv\Scripts\python.exe -X utf8 -m backend.cli audit runs\MY_RUN --backend openai --model gpt-5.4 --sample 40 --seed 7
.venv\Scripts\python.exe -X utf8 -m backend.cli observe runs\MY_RUN --backend claude_cli --model haiku --no-publish
```

The audit samples annotated occurrences and changes reproducibly and asks for an independent assessment against the original evidence. It records supported, contested or insufficient interpretations and explanations. Use a different observer model when appropriate; using the same model is another assessment, not independent model validation.

Audits of positive annotations do **not** estimate discovery recall. Researchers should also inspect randomly selected source episodes, including episodes with no detected concepts, and compare alternative observer snapshots. Neither model agreement nor software passing its tests proves the semantic judgments valid. Saved evidence and exports support colleague review without introducing a human player into the simulated world.

## Pausing, extending, and recovery

```cmd
.venv\Scripts\python.exe -X utf8 -m backend.cli pause runs\MY_RUN
.venv\Scripts\python.exe -X utf8 -m backend.cli continue runs\MY_RUN --out runs\MY_RUN_extended --days 90 --analyze
```

Pause requests take effect at a committed tick boundary. `continue` reexecutes the recorded prefix using cached model responses, verifies the per-tick digests, then makes new calls after the boundary. `--days` is the **total horizon**, including the prefix. The original recording is retained and the continuation uses a separate directory.

Verified continuation requires unchanged simulation code and prompt hashes and is implemented for the campus/co-op engine. It is not available for the older separate commons runtime. Extending a horizon does not invent new scheduled interventions: ensure meetings, turnover and circumstance schedules cover the intended horizon in the original configuration.

Rerunning a study recovers completed compatible cells and retries failed simulation or observation work. Failed simulation attempts are retained in renamed directories. Paused societies remain visible as paused; the explicit continuation command produces a separately recorded society. If incorporating that continuation into a custom comparison, retain its shared-prefix relationship and do not treat it as an independent replicate.

## Interface and sharing

```cmd
.venv\Scripts\python.exe -X utf8 -m backend.cli serve --port 8765
start http://127.0.0.1:8765/?view=research
```

The **Research** tab provides the study library, condition overlays, full-study launch, execution logs, run observation, saved snapshots, open inquiries, audits and continuation controls. The **Culture** tab shows the new cultural-history browser when the selected run has a memetics analysis.

The history browser provides search, public-use timelines, agent-weighted sense trajectories, an agent-by-day interpretation matrix, possible transmission networks, before/after changes, related concepts and source quotations. Source buttons open original evidence; replay buttons connect a finding to the recorded world. Blank interpretation periods remain missing, not apparent consensus or disappearance.

Research output is also usable without the server:

```cmd
.venv\Scripts\python.exe -X utf8 -m backend.cli export runs\MY_RUN
```

Each analysis contains `report.html`, `report.md`, `analysis.json`, `evidence.json`, `occurrences.csv` and `figures/*.svg`. Open the HTML directly; print/save PDF from the report. To share a per-run report with its Markdown and figures, send its complete analysis directory. To retain links from an experiment report to individual societies and inquiries, share the experiment directory with its directory structure intact. JSON/CSV provide the underlying measurements, and SVG figures can be imported into presentation tools.

## Artifacts and provenance

```text
runs/<design>/<condition>/s<world_seed>[_r<agent_replica>]/
  config.resolved.yaml       Complete simulation and observer configuration
  manifest.json             Status, agents, environment and code provenance
  trace.jsonl               Original events
  frames.jsonl              World replay and active population
  digests.jsonl             Committed tick boundaries and replay checks
  llm_calls.jsonl            Agent call recording
  index/evidence.sqlite     Rebuildable research index
  analyses/<analysis_id>/   Versioned observations, interpretations and exports
  analyses/_cache/          Separate observer call recordings
  inquiries/<question_id>/  Question, retrieval plan, evidence, calculations, answer
  audits/<audit_id>/        Sampled annotations and independent assessments
  analysis.json             Current complete published observation
  outcomes.json             Current compatible measurement summary
runs/<design>/
  pipeline.json             Stage, selected runs, failures and completion status
  report.json               Latest comparison snapshot
  reports/<report_id>/      Portable comparison report, CSV and SVG figures
  inquiries/<question_id>/  Cross-society interpretations and original citations
runs/.research_jobs/        Background interface job states and logs
```

IDs derive from the input fingerprint and observer configuration, including prompt and implementation hashes. Changes in the analysis code or requested observer make older outputs stale for new comparisons. Different saved snapshots are retained. The index consumes only newline-complete events at or before a committed tick. Private evidence may enter the external observer, but hidden world-truth event types do not enter its semantic packets.

The runtime never reads research interpretations. Automated tests check that observation leaves simulation artifacts unchanged, rejects unsupported quotations and future evidence, respects exposure ordering, separates private/public channels, and completes the command-line and API workflow using offline fixtures.

## Practical limits when interpreting a study

- LLM concepts and sense boundaries are hypotheses about the recording. Description revisions and alternative observations are preserved; no claim of observer-invariant concept identity is made.
- Discovery visits the full eligible textual corpus, but a bounded context and retrieved historical anchors can miss continuity or create duplicate histories. Coverage of source windows is not semantic recall.
- Inquiry retrieval is bounded and partly based on lexical matching, supplemented by LLM search expansion and concept anchors. It can miss relevant paraphrases or counterexamples; the selected evidence is inspectable.
- Public language is not direct access to belief. Logged exposure is not proof of uptake, and reuse is not endorsement.
- Absence at the end of a run is censored. It does not prove extinction, forgetting, or permanent cultural replacement.
- The presets use existing environment controls. Interpret realized exposure and actual events, especially where opportunities compete in the current scheduler.
- Independent world blocks support condition comparisons. Within-run associations and source paths alone do not establish causal mechanisms.
- Claude CLI does not expose all requested decoding controls; recorded requested temperature/token settings are not a guarantee of provider-enforced decoding equivalence. Preserve the provider/model settings when comparing observers.

These limits are part of the evidence model. They do not restrict which ideas may emerge or which subsequent questions the researchers can investigate.

The GPT backend uses the stateless [OpenAI Responses API](https://developers.openai.com/api/docs/guides/text), with no tools or shared provider conversation state. The configured GPT-5.4 defaults use no-reasoning mode for short simulation responses. Incomplete or empty responses fail instead of becoming agent speech.
