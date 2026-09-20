# Current campus studies

The selected studies use `configs/campus100.yaml`: 100 people from the shuffled Homewood 500
roster, three days, 11:00–20:00, 15-minute ticks, and GPT-5 nano for agents and observation.
Population and duration remain configurable. These are the first 100 profiles, not a stratified sample.

All four studies enable speaking-time priming: up to three retained phrases heard from others
within the last 24 simulated hours can appear in the speaker's context. Exact wording retention
is also enabled: distinctive heard phrases survive encoding probabilistically (30% initially,
plus 20 percentage points per existing memory containing the phrase, capped at 90%; at most
two phrases per encoding). These mechanisms use experiences within the run, not supplied slogans.

| Study | Conditions | Runs with the default seed 11 |
|---|---|---:|
| emergence | Reference campus | 1 |
| communities | Sparse/dense cross-group ties × zero/two bridge agents | 4 |
| influence | Prestige off/network/reassigned × conformity off/on | 6 |
| social_meaning | Social reward off/on × emotion off/on | 4 |

The reference and all four selected studies have no shared Markdown, no planted phrase,
and no additional initial-memory file. Ordinary relationship memories are still initialized.
`homewood500.yaml` contains profiles without FFC/Hopkins Cafe naming instructions.
The separate `homewood500_naming.yaml` explicitly loads `homewood500_initial_memories.yaml`;
that naming study remains available separately. The map's building labels are display metadata,
not agent instructions. Models can still draw on their pretrained knowledge.

Paste the API key into the ignored `.env` file. These four designs explicitly select `gpt-5-nano`,
so older optional model values in `.env` do not override them. Use `--agent-model` and
`--observer-model` to change models. The adapter uses minimal reasoning and low verbosity;
2,048 additional output-budget tokens allow room for reasoning beyond short cognitive responses.
This is a token ceiling, not a fixed charge. Exhausting it stops the run instead of repeating
the same inadequate request through long retry pauses. Live throughput remains unmeasured.

Run one selected study at a time in Windows CMD:

```cmd
cd /d C:\Users\pigby\MemeticWorld
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment run emergence --no-questions
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment run communities --parallel 2 --no-questions
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment run influence --parallel 2 --no-questions
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment run social_meaning --parallel 2 --no-questions
```

All four commands together launch 15 societies. `--no-questions` skips optional inquiries;
observation and reports still run. To add replication, edit the designs' `seeds` lists.
`--seed` filters existing seeds; it does not add new seeds to a design.

The Research frontend groups these four under **Your selected studies** and shows actual population,
duration, model and starting material. Select **Configured live providers** to use GPT;
the initial Mock selection is only an offline preview. Campus shows recorded movement and
conversations; Research shows concept histories, interpretations and source evidence after observation.

```cmd
.venv\Scripts\python.exe -X utf8 -m backend.cli serve --port 8765
```

In another CMD window:

```cmd
start http://127.0.0.1:8765/?view=research
```
