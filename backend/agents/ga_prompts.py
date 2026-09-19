"""Thin, robust wrappers around upstream Generative Agents prompts.

Where the upstream `run_gpt_prompt_*` function parses modern chat-model output
reliably we call it directly (relationship summary, iterative chat utterance).
Where its validators are brittle (they were written for text-davinci-003) we
reuse the upstream *template file* and upstream `generate_prompt` /
`ChatGPT_safe_generate_response` / `safe_generate_response` helpers, but supply
a tolerant parser. Prompt wording is always upstream's.
"""
from __future__ import annotations

import json
import re

from backend import ga_compat
from backend.llm.client import llm_purpose

ga = ga_compat.load()
T = ga_compat.template_path


def _int_1_10(x):
    m = re.search(r"\d+", str(x))
    if not m:
        raise ValueError(x)
    return max(1, min(10, int(m.group())))


def poignancy(agent, description: str, kind: str = "event") -> int:
    """GA poignancy_{event,chat,thought}_v1 (v3_ChatGPT) with a tolerant integer parser."""
    tpl = {"event": "v3_ChatGPT/poignancy_event_v1.txt", "chat": "v3_ChatGPT/poignancy_chat_v1.txt",
           "thought": "v3_ChatGPT/poignancy_thought_v1.txt"}[kind]
    prompt = ga.gs.generate_prompt([agent.scratch.name, agent.scratch.get_str_iss(),
                                    agent.scratch.name, description], T(tpl))
    with llm_purpose(f"poignancy_{kind}", agent.id):
        out = ga.gs.ChatGPT_safe_generate_response(
            prompt, "5", "The output should ONLY contain ONE integer value on the scale of 1 to 10.",
            2, 4, lambda r, prompt="": bool(_int_1_10(r)), lambda r, prompt="": _int_1_10(r))
    return out if isinstance(out, int) else 4


def decide_to_talk(agent, target, retrieved_events: list, retrieved_thoughts: list) -> tuple[bool, str]:
    """Upstream v2/decide_to_talk_v2.txt; prompt inputs built exactly like upstream
    run_gpt_prompt_decide_to_talk.create_prompt_input."""
    last_chat = agent.a_mem.get_last_chat(target.name)
    last_time, last_about = "", ""
    if last_chat:
        last_time = last_chat.created.strftime("%B %d, %Y, %H:%M:%S")
        last_about = last_chat.description
    context = "".join(f"{n.description}. " for n in retrieved_events) + "\n"
    context += "".join(f"{n.description}. " for n in retrieved_thoughts)
    curr_time = agent.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
    init_p = f"{agent.name} is already {agent.scratch.act_description}"
    targ_p = f"{target.name} is already {target.scratch.act_description}"
    extra = agent.mods.modify_prompt(agent, "decide_to_talk", [], target=target)
    if extra:
        targ_p += "\n" + "\n".join(extra)
    prompt = ga.gs.generate_prompt([context, curr_time, agent.name, target.name, last_time, last_about,
                                    init_p, targ_p, agent.name, target.name],
                                   T("v2/decide_to_talk_v2.txt"))

    def parse(r):
        tail = r.strip().lower().split("\n")[-3:]
        for line in reversed(tail):
            m = re.findall(r"\b(yes|no)\b", line)
            if m:
                return m[-1]
        raise ValueError(r)

    with llm_purpose("decide_to_talk", agent.id):
        out = ga.gs.safe_generate_response(
            prompt, {"engine": "text-davinci-003", "max_tokens": 120, "temperature": 0, "top_p": 1,
                     "stream": False, "frequency_penalty": 0, "presence_penalty": 0, "stop": None},
            2, "no", lambda r, prompt="": _ok(parse, r), lambda r, prompt="": parse(r))
    return out == "yes", prompt


def _ok(fn, r):
    try:
        fn(r)
        return True
    except Exception:  # noqa: BLE001
        return False


def _numbered(r: str, prefix: str) -> list[str]:
    """Items of a numbered list. Upstream prompts end with '1)' / '1.' for davinci-style
    completion; chat models often restart the list and add a preamble, so prefer explicitly
    numbered lines and only fall back to treating the first line as item 1."""
    lines = [l for l in r.strip().split("\n") if l.strip()]
    num = [l for l in lines if re.match(r"^\s*\**\s*\d+\s*[\).]", l)]
    if num:
        items = [re.sub(r"^\s*\**\s*\d+\s*[\).]\s*", "", l) for l in num]
        head = lines[0]
        if head is not num[0] and not head.rstrip().rstrip(".").endswith(":") and \
                not re.match(r"^\s*(based on|here (are|is))", head, re.I):
            items.insert(0, head.strip())  # davinci-style continuation of the prompt's "1."
        return items
    first = [re.sub(r"^\s*\d+\s*[\).]\s*", "", (prefix + lines[0]))] if lines else []
    return first + [l for l in lines[1:] if not l.lstrip().startswith(("-", "*"))]


def focal_points(agent, statements: str, n: int) -> list[str]:
    """Upstream v2/generate_focal_pt_v1.txt (completion style, '1)' numbered list)."""
    prompt = ga.gs.generate_prompt([statements, str(n)], T("v2/generate_focal_pt_v1.txt"))

    def parse(r):
        qs = [re.sub(r"\*\*", "", x).strip() for x in _numbered(r, "1) ")]
        qs = [q for q in qs if len(q) > 5 and not q.endswith(":")]
        if not qs:
            raise ValueError(r)
        return qs[:n]

    with llm_purpose("reflection_focal_points", agent.id):
        return ga.gs.safe_generate_response(
            prompt, {"engine": "text-davinci-003", "max_tokens": 150, "temperature": 0, "top_p": 1,
                     "stream": False, "frequency_penalty": 0, "presence_penalty": 0, "stop": None},
            2, [], lambda r, prompt="": _ok(parse, r), lambda r, prompt="": parse(r))


def insights(agent, nodes: list, n: int) -> dict[str, list[str]]:
    """Upstream v2/insight_and_evidence_v1.txt -> {insight: [evidence node ids]}."""
    statements = "".join(f"{i}. {node.embedding_key}\n" for i, node in enumerate(nodes))
    prompt = ga.gs.generate_prompt([statements, str(n)], T("v2/insight_and_evidence_v1.txt"))

    def parse(r):
        out = {}
        for line in _numbered(r, "1. "):
            line = line.replace("**", "").strip()
            if line.endswith(":"):
                continue
            if "(because of" in line:
                thought, ev = line.split("(because of", 1)
                idx = [int(x) for x in re.findall(r"\d+", ev.split(")")[0])]
            else:
                thought, idx = line, []
            thought = thought.strip().rstrip(".").strip()
            if len(thought) > 8:
                out[thought + "."] = [nodes[i].node_id for i in idx if 0 <= i < len(nodes)]
        if not out:
            raise ValueError(r)
        return dict(list(out.items())[:n])

    with llm_purpose("reflection_insights", agent.id):
        return ga.gs.safe_generate_response(
            prompt, {"engine": "text-davinci-003", "max_tokens": 200, "temperature": 0.5, "top_p": 1,
                     "stream": False, "frequency_penalty": 0, "presence_penalty": 0, "stop": None},
            2, {}, lambda r, prompt="": _ok(parse, r), lambda r, prompt="": parse(r))


def relationship_summary(agent, target, retrieved: dict) -> str:
    """Upstream converse.generate_summarize_agent_relationship (called directly)."""
    with llm_purpose("chat_relationship_summary", agent.id):
        try:
            out = ga.converse.generate_summarize_agent_relationship(agent, target, retrieved)
        except Exception:  # noqa: BLE001
            out = ""
    if not out or out == "..." or out.startswith("LLM_ERROR"):
        r = agent.profile.rel(target.id)
        out = f"{agent.name} and {target.name} are {r.relation_type}s."
    return out


def chat_utterance(maze, agent, target, retrieved: dict, curr_context: str, curr_chat: list):
    """Upstream run_gpt_generate_iterative_chat_utt (v3_ChatGPT/iterative_convo_v1.txt), called directly."""
    with llm_purpose("chat_utterance", agent.id):
        try:
            out = ga.rgp.run_gpt_generate_iterative_chat_utt(maze, agent, target, retrieved,
                                                             curr_context, curr_chat)[0]
            utt, end = out["utterance"], out["end"]
        except Exception:  # noqa: BLE001
            utt, end = "...", True
    utt = str(utt).strip().strip('"').strip()
    if not utt or utt == "..." or utt.startswith("LLM_ERROR"):
        return None, True
    return utt, bool(end)


def as_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None
