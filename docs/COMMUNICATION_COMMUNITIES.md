# Injected memes: four communication/community conditions

`communication_communities` inherits the existing injection study via
`configs/homewood100_memes_communication.yaml`. The simulation and analysis mechanisms are unchanged.
The original `homewood100_memes.yaml` remains the source of inherited settings.

Each expression is a frequent, persistent carrier habit. Registry IDs remain stable:

| Registry ID | Previous label | Replacement | Supplied campus meaning |
|---|---|---|---|
| tray_washer | teal tray | seven six | Deliberately nonsensical catchphrase for recurring jokes and casual banter; this exact order is intentional |
| hood_sash | ochre hood | rizz | Charm and skill at flirting or making a romantic impression |
| running_warm | indigo case | lock in | Committing to focused effort and encouraging others to concentrate |
| tuesday_list | mauve list | aura farming | Deliberately appearing cool or impressive, praised or teased as a performance |

The habit lines explicitly say "frequently" and give relevant occasions and usage cues. This prompts
frequent use by the carriers; it is not a hard quota or a guarantee of particular model outputs.
The old object meanings, their staged incidents, and the associated probe situations are removed.
All four entries are ungrounded (no staged origin incident) and broad (usable across social situations).
Their observer probes now cover the intended social uses; this is no longer the original grounding-by-breadth study.
Each phrase is supplied as a persistent habit to five agents using the existing disjoint `spread`
allocation. Familiar wording may arise independently; the existing exposure/uptake evidence matters.
The world vocabulary audit is derived from the replacement words rather than the previous colours.

## Conditions

100 agents from Homewood100, seven days, 08:30–19:30, simulation seed 42; four runs total.
Both agents and observer use Codex CLI / GPT-5.6-Luna. The existing 12 model workers are inherited.
Every condition generates four communities, adds zero strong bridge agents, and enables shared meals.

| Condition | Cell ID | Cross-group tie probability | Ordinary talk gate | Initiation threshold |
|---|---|---:|---:|---:|
| A: Quiet, separated | ties-separated__talk-low | 0.00 | 0.10 | 2 |
| B: Talkative, separated | ties-separated__talk-high | 0.00 | 0.70 | 8 |
| C: Quiet, connected | ties-connected__talk-low | 0.40 | 0.10 | 2 |
| D: Talkative, connected | ties-connected__talk-high | 0.40 | 0.70 | 8 |

The existing threshold gates ordinary initiation, not every participation. Catch-ups, responding,
group talk, and clarification keep their original behavior. Zero cross-group ties does not prevent
stranger encounters or overhearing. The `spread` allocation depends on topology, so carrier IDs can
differ between connectivity conditions even with the same simulation seed.

Memory rules are unchanged. Under the existing distinctive-phrase filter, `aura farming` is eligible
for exact wording retention; `rizz` can be part of a longer retained span. `seven six` and `lock in`
do not qualify for special verbatim retention in the checked neutral sentence. Injection and normal
memory encoding still operate; this study does not force all supplied phrases to stick equally.
The existing registry matcher recognizes all four exact labels independently of that memory filter.

## Windows CMD

Check the preset (no model requests):

```cmd
cd /d C:\Users\pigby\MemeticWorld
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment check communication_communities --runs-root runs\injected-communities
```

Run all four sequentially, with the existing analysis and reports:

```cmd
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment run communication_communities --runs-root runs\injected-communities --parallel 1 --no-questions
```

To run one condition, append `--only` followed by its cell ID from the table. Do not launch overlapping
selections concurrently. `--no-questions` skips optional question-specific inquiries; it does not disable
the inherited analysis. The seven-day horizon is retained; this variant has no staged incident/repair manipulation.

```cmd
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment status communication_communities --runs-root runs\injected-communities
start "" "http://127.0.0.1:8766/?view=research"
```

Refresh the frontend and choose recordings under `injected-communities/`. The existing registry/live
views and trends consume the supplied labels. Restart the server if its experiment catalog is stale.
