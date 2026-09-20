# Run campus experiments using Codex credits

`codex_cli` uses `codex exec` with ChatGPT sign-in for both agents and observation.
It never uses the project's OpenAI API key or falls back to API billing. Your existing
Codex allowance/credits and account model availability apply; this is not unlimited usage.

The default for this provider is `gpt-5.6-luna`, with low reasoning and low verbosity.
As of September 19, 2026, the official credit table lists Luna at 5 input / 0.5 cached
input / 30 output credits per million tokens, versus GPT-5.5 at 125 / 12.5 / 750.
Thus Luna has 25 times lower listed token rates; total run consumption also depends
on token counts. See [official pricing](https://learn.chatgpt.com/docs/pricing).
Use `codex` then `/model` to see models available to your account. API-only models
such as the prior GPT-5-nano preset must not be assumed available through ChatGPT auth.

## Launch from Windows CMD

```cmd
cd /d C:\Users\pigby\MemeticWorld
codex login
codex login status
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment check emergence --backend codex_cli --days 2 --runs-root runs\codex
.venv\Scripts\python.exe -X utf8 -m backend.cli experiment run emergence --backend codex_cli --days 2 --runs-root runs\codex --no-questions
```

Login must report ChatGPT authentication. The check is local and makes no model request;
the run consumes credits. Replace `emergence` with `communities`, `influence`, or
`social_meaning` to run that study. Start with one society at a time. To override both models,
add `--agent-model gpt-5.5 --observer-model gpt-5.5` to the check and run commands.

Population, clocks, memories, priming, recall weights, and experimental conditions
remain those of the selected study (100 people, two days in the command above).
Model/provider choices are frozen into a custom design so subprocesses and reports agree,
and results cannot collide with the API run. Changing the provider/model is a new experimental
condition; do not pool it unlabelled with GPT-5-nano results.

For the newly selected Homewood100 population without supplied naming memories, add
`--population configs/population/homewood100.yaml --population-size 100` to both commands.

## Frontend

Restart the frontend server after installing this change. From CMD:

```cmd
.venv\Scripts\python.exe -X utf8 -m backend.cli serve --port 8766
```

Open the page from another CMD window:

```cmd
start "" "http://127.0.0.1:8766/?view=research"
```

Select **Codex CLI · ChatGPT credits** in Execution. Leave model overrides empty for Luna,
set Days per society to 2, and check the preview before launching. The separate Observer
selector supports Codex too. Runs launched from CMD appear under `codex/` in the recorded
society selector after refresh. Select Campus for playback; reload for newly saved frames.
Research and Culture display the observation results when analysis finishes.

## Execution details and limitations

Each completion has a new ephemeral session and an empty temporary working directory.
The experiment's system instructions replace the coding persona. Project documents,
personal memories, plugins, apps, shell, browsing, and collaboration tools are disabled;
responses with tool/action events are rejected. No conversational state passes between agents.
The existing trace and record/replay cache store final answers, never CLI diagnostics as speech.

Codex controls token usage internally: the pipeline's response budget is supplied as an
instruction, not a hard output cap, and temperature is not configurable through this adapter.
This means the CLI is not a numerically identical substitute for direct API sampling.
Account limits, authentication failures, and timeouts stop the run instead of silently changing
providers or entering the simulator as dialogue. Check the terminal/log before resuming.
Recoverable CLI error/reconnect notifications do not abort a successful completed turn.
An explicit failed turn, nonzero exit, missing final answer, or tool/action event still rejects
the response. HTTP status classification excludes request IDs and model replies.
CLI process startup and quota limits may make this slower than API execution; no 100-person
Codex runtime estimate has been measured.

After updating backend code, start a fresh recording with a new `--runs-root`, such as
`runs\codex-fixed`. Continuation requires matching source fingerprints; it must not mix
responses recorded before and after an adapter change. The previous recording is preserved.

The server and experiment must use the same CODEX_HOME as your logged-in CLI.
On Windows, discovery checks PATH and then the newest matching Codex binary bundled with
the VS Code / VS Code Insiders OpenAI extension, so ordinary CMD windows do not need the
extension directory on PATH. An optional `CODEX_CLI_PATH` in `.env` takes precedence.
Credentials remain managed by Codex; no tokens are copied into the experiment config.
See [official non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode).
