# MemeticWorld

**Can AI agents invent, spread, and converge on memes without being asked to?**

MemeticWorld drops eight LLM-driven students onto a small, Homewood-style campus. They follow
class schedules, bump into each other at lunch, witness small absurd incidents (a dropped tray,
a squirrel stealing a bagel, the lab printer grinding away), talk, and remember. Nobody tells
them to be funny, coin phrases, or make memes. Every word is logged, and an analysis layer
traces which phrases one agent coins and others pick up.

```
ordinary interaction → novel expression → reuse → spread → shared convention
```

## What you see

- **Campus map**: agents walking between buildings, speech bubbles, live incidents.
- **Timeline**: scrub or replay the run; the strip shows how much talking happened each tick.
- **Phrases**: detected candidates ranked by evidence, with originator, adopters and a per-agent
  meter; "competing names" for the same incident shows convergence.
- **Who got it from whom**: a swimlane cascade per phrase. Arrows run from the use that exposed
  an agent to that agent's first reuse in a new conversation. Bold arrows mean the adopter's
  prompt literally contained a memory quoting the phrase.
- **Agent inspector**: exactly what an agent saw, which memories it retrieved, and what the
  model answered.
- **Full vs control**: the same seed rerun with agents that remember *that* they talked but not
  *what* was said. Phrases that spread there too are model habits, not culture.

## How it works

Each 15-minute tick: `observe → retrieve memory → decide → move/talk → store memory`.

- Agents see only their own location. Memory retrieval scores
  `0.6·similarity + 0.3·recency + 0.1·importance` and returns the top 5.
- One small LLM call per decision (`MOVE / TALK / REACT / CONTINUE_ACTIVITY / IDLE`), validated
  with pydantic. Conversations run 1–4 turns, one call per speaker, so no agent ever sees
  another's memories.
- Speech is remembered verbatim. That quote is the only way a phrase can travel.
- The detector separates **adoption** (reuse in a new conversation after hearing it) from
  **echo** (repeating it back on the spot) and from **independent coinage**. It confirms
  transmission from logged retrieval, and demotes anything that also spreads in the control run.

## Quick start

No API key is needed to try it: without one, the backend uses an offline mock LLM, which is
good for the UI and not for science.

```bash
cp .env.example .env            # add LLM_API_KEY / LLM_MODEL for real runs

cd backend
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/uvicorn app.main:app --reload --port 8000

cd ../frontend
npm install && npm run dev      # open http://localhost:3000 → New run
```

Headless: `cd backend && .venv/bin/python -m app.cli run --days 2 --control`, then
`python -m app.cli memes <run_id>`.

Any OpenAI-compatible API works (OpenAI, OpenRouter, Ollama, vLLM, LM Studio); see
`.env.example`. Tests: `cd backend && .venv/bin/python -m pytest -q`.

## Stack

Next.js 16 + TypeScript + Tailwind (SVG charts) · FastAPI + SQLite · REST + Server-Sent Events.
See [CLAUDE.md](CLAUDE.md) for the architecture, data model, API, and contributor rules.

## Demo script (3 minutes)

1. Start a full run with a control (or load a recorded one). Hit **Play** and point out agents
   converging on the Dining Hall at lunch.
2. An incident fires (⚡). Open **Conversations** and watch someone describe it in their own words.
3. Open **Phrases**, pick the top *Strong* phrase, and walk the cascade: origin → confirmed
   adoption → second-hand adoption.
4. Click an adopter, open **Latest LLM call → Memories it retrieved**, and show the quote that
   carried the phrase.
5. **Full vs control**: the same world without speech memory produces far fewer adoptions.

## Honest limitations

- Eight agents over a few simulated days is small. Treat results as case studies, not statistics.
- All agents share one model, so shared priors are the main confound. That's why the control
  run exists, and why phrases made only of words the world supplies are ignored.
- The n-gram detector is heuristic. Paraphrased variants are grouped by incident, not by meaning.
