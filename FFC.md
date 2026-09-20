# FFC vs Hopkins Cafe — how to run it

**Audience:** whoever is running this experiment on another machine, without the context of the session
that set it up.

This is a self-contained, ready-to-run experiment. Everything it needs is committed. The one thing you must
not skip is the 30-minute check in §5 — it tells you whether the run is worth finishing.

---

## 1. What this measures

100 agents on the Homewood campus. Every agent starts with a memory of what *they* call the dining hall
beside the freshman residences:

| seeded name | agents |
|---|---:|
| `FFC` | 80 |
| `Hopkins Cafe` | 20 |

Nobody is told the other name exists. The question is which name the population converges on, if either,
and whether the 20% minority can hold or spread its term.

**Why the 80/20 split.** It is near the committed-minority threshold where a convention tips in the
naming-game literature (Ashery et al., *Science Advances* 2025; Centola's critical-mass work). A clean
majority win is the boring outcome; minority survival or spread is the interesting one.

**How it fits the wider project.** This is now the *floor case* for the main study (a cohort of injected
memes, `configs/homewood100_memes.yaml`). A proper noun for a fixed building can only show two of the eight
outcomes in the project's taxonomy — diffusion and formal variation. It cannot broaden or narrow its
meaning, because its referent is one building. Running it under the same probes as the class-applying memes
turns "names can't drift semantically" from an argument into a measured result. So a *null* here is
informative, not a failure.

---

## 2. Before you start

- **Plug the machine in and keep the lid open.** Clamshell sleep overrides `caffeinate` and has silently
  paused runs on this project. A closed lid will stall the run mid-way with no error.
- Expect **~90–120 minutes** for 2 days at 32 workers, and roughly **32,000 `claude` CLI calls**.
- Agents run on **Haiku** via the local `claude` CLI — no API key needed, but you must be logged in
  (`claude` working from your shell).
- Needs ~2 GB free disk for the run artifacts.

Check your setup:

```bash
claude -p --model haiku --output-format json \
  --settings '{"alwaysThinkingEnabled": false}' "reply with OK"
```

---

## 3. Run it

```bash
cd /path/to/MemeWorld

.venv/bin/python -m backend.cli run \
  --config configs/homewood100_naming_wording.yaml \
  --set llm.max_workers=32 simulation_days=2 \
  --out runs/ffc_wording
```

### Two overrides you may need

**`day_end`.** The population was regenerated so all 100 agents eat, and **108 of their 240 dining steps
(45%) fall at or after 17:00** — those are dinner. If your copy of `configs/homewood100_naming.yaml` still
reads `day_end: "17:00"`, every one of those dinner meals is outside the simulated day and you lose half the
meal occasions, which are the main reason the hall gets talked about at all. Check it, and if it is still
17:00, add `day_end=19:30` to the `--set` list.

**`--set` silently ignores unknown keys.** There is no error for a typo — `sim.days=1` is accepted and does
nothing. **Always verify the first log line**: `[Day 1 08:30] tick 1/N`. For 2 days you want `N=68` at a
17:00 end, or `N=88` at 19:30. If N is wrong, your override did not take.

---

## 4. Two cells (optional)

`homewood100_naming_wording.yaml` is the **wording cell** — the one to run first, and the baseline for
everything else. It enables verbatim stickiness, priming, retrieval source weights and the meal-talk cue:
the mechanisms that give a name a path into speech at all.

`homewood100_naming.yaml` is the **plain cell** — identical population and seeding, all of those mechanisms
off. Run it only if you want the contrast. It is expected to produce fewer name uses; that is the point of
it, not a bug.

---

## 5. The 30-minute check — do not skip this

The previous attempt at this experiment produced **2 uses of "Hopkins Cafe" and 0 of "FFC" across 843
utterances**, because the world was handing agents the string "Dining Hall" in 72% of chat prompts and they
simply copied it. That is fixed (see §7), but verify it on *your* run rather than trusting it.

Once the log reaches roughly tick 12 (the 11:30–13:30 lunch window — the first time a meal comes up):

```bash
.venv/bin/python - <<'PY'
import json, collections
RUN = "runs/ffc_wording"
utt = 0; names = collections.Counter(); spk = collections.defaultdict(set)
for line in open(f"{RUN}/trace.jsonl"):
    try: r = json.loads(line)
    except Exception: continue
    if r.get("type") != "utterance": continue
    utt += 1
    t = r.get("text") or ""
    s = r.get("speaker") or r.get("agent")
    for n in ("FFC", "Fresh Food", "Hopkins Cafe"):
        if n in t:
            names[n] += 1; spk[n].add(s)
print(f"utterances: {utt}")
for n, c in names.most_common():
    print(f"  {n:14} {c:4d} uses, {len(spk[n]):3d} distinct speakers")
if not names:
    print("  no seeded name used yet")
# the fix that makes this experiment possible — must stay at 0
leak = sum(1 for l in open(f"{RUN}/llm_calls.jsonl") if "Dining Hall" in l)
tot  = sum(1 for _ in open(f"{RUN}/llm_calls.jsonl"))
print(f'"Dining Hall" in {leak}/{tot} prompts  (must be 0)')
PY
```

**How to read it**

| result | meaning |
|---|---|
| `"Dining Hall"` count > 0 | **Stop the run.** The situated-reference layer is not active; agents are being handed the answer and the result would be meaningless. Check `world.reference_mode: situated` is set. |
| names appearing, both present | Healthy. Let it finish. |
| 0 names after the lunch window has fully passed | Let it run to the end anyway, but say so when you report — it means the phrase never reached speech, which is a finding about the mechanism, not about which name wins. |

---

## 6. Analyse

```bash
.venv/bin/python scripts/analyze_dining_names.py runs/ffc_wording
```

Key fields:

- `overall.lexical_mentions` — raw counts per name.
- `unique_speakers` — how many distinct agents used each. **More informative than raw counts**, which one
  chatty agent can dominate.
- `by_seeded_cohort` — did the 20 Hopkins-seeded agents hold their term, or switch?
- `crossovers` — the first time an agent used the name it was *not* seeded with, paired with an earlier
  utterance it could have heard it in. This is the transmission evidence.
- `convergence.entropy_bits_by_day` — 1 bit = both names used equally, 0 = one has won, `null` = neither
  was used.
- `dining_references` — how often the hall was referred to *at all*, including by plain description. If this
  is near zero, nothing else in the output means anything.

The script prints its own definitions and limitations. Read them; the counts are lexical mentions, not
validated claims of adoption.

---

## 7. What was fixed, and what to be sceptical about

**Fixed — situated reference (decision D75).** The world now describes places to agents instead of naming
them: *"the dining hall beside the freshman residences"*, never `"Dining Hall"`. Measured on a matched mock
day, prompts carrying a canonical campus label went **95.4% → 0.0%**, and `"Dining Hall"` in chat prompts
went **64.3% → 0.0%**. Canonical mode is byte-identical to before, so older runs still reproduce. The seed
memories were rewritten to use the world's own words for the hall, so the seeded name and the percept point
at the same referent.

**Checked — the model has no thumb on the scale.** Probing fresh Haiku with no simulation context: it has
no preference between the two names; it does not know what "FFC" means (6/6); and it will not produce
either name unprompted (8/8). So any occurrence in a run is seeding or transmission, never the model
reaching for something it already knew. `scripts/probe_priors.py` reproduces this.

**Be sceptical about:**

- **Low reference rate.** Agents referred to the hall ~0.16 times per agent-day in mock. If the hall is
  rarely discussed, neither name gets exercised and a null result says nothing about naming.
- **Counts vs adoption.** A quoted or explanatory mention is not adoption. Prefer `unique_speakers` and
  `crossovers`.
- **One trajectory.** This is a single seed. Do not read a narrow margin as a win. `backend/experiment/`
  has a multi-seed runner if you want replication; it costs YAML, not code.
- **Drift vs transmission.** An 80/20 split can converge on the majority name by drift alone. The
  `crossovers` field is what distinguishes them.

---

## 8. What to report back

1. The 30-minute check output.
2. The full `analyze_dining_names.py` JSON.
3. `runs/ffc_wording/manifest.json` (config, code version, population, world-script hash — this is what
   makes the run reproducible).
4. Total wall-clock and call count from the last log line.
5. Anything that looked wrong, including a run you stopped early and why.

Keep the whole `runs/ffc_wording/` directory if you can — it is a few GB but it is the auditable record.
