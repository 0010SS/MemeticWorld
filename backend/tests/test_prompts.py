"""Guards the core research rule: agents are never told to make memes, slang, or conventions.

If this fails, do not weaken the list. Rephrase the prompt/persona/event instead.
"""
import re

from app.llm import prompts
from app.simulation.agent import Agent
from app.simulation.campus import LOCATIONS
from app.simulation.events import TEMPLATES
from app.simulation.personas import PERSONAS

BANNED = [
    r"\bmemes?\b", r"\bmemetic", r"\bslang\b", r"\bviral\b", r"\bcatch ?phrases?\b", r"\binside jokes?\b",
    r"\brunning jokes?\b", r"\bnicknam", r"\bcoin(s|ed|ing)?\b", r"\binvent", r"\btrend", r"\bconventions?\b",
    r"\blingo\b", r"\bjargon\b", r"\bneologism", r"\bmake up (words|names|phrases)\b", r"\bcallbacks?\b",
    r"\bnew (words|phrases|expressions)\b", r"\bcreative (language|wording)\b", r"\bcatchy\b",
]


def _all_agent_facing_text() -> list[str]:
    texts = [prompts.SYSTEM, prompts.DECISION, prompts.REPLY]
    for persona in PERSONAS:
        agent = Agent.from_persona(persona)
        texts.append(agent.system_prompt())
        texts += [label for label, _ in persona["relationships"].values()]
        texts += [activity for _, _, activity in persona["schedule"]]
    texts += [t.text for t in TEMPLATES]
    texts += [loc["description"] for loc in LOCATIONS.values()]
    observation = prompts.render_observation(
        time_label="Day 1, 12:00 PM", location="Quad", location_description=LOCATIONS["Quad"]["description"],
        activity="sitting", scheduled=("Library", "studying"), nearby=[("Maya", "your roommate", "eating")],
        happenings=["Something happens."], memories=[("Day 1, 9:00 AM", 'Maya said to me: "hello"')])
    texts.append(prompts.render_decision(observation, ["Dorm", "Library"], ["Maya"]))
    texts.append(prompts.render_reply(observation, "Maya", "your roommate", [], [("Maya", "hello")]))
    return texts


def test_no_meme_instructions_anywhere_agents_can_see():
    for text in _all_agent_facing_text():
        for pattern in BANNED:
            match = re.search(pattern, text, re.IGNORECASE)
            assert match is None, f"banned term {match.group(0)!r} in agent-facing text: {text[:120]!r}"


def test_prompts_contain_no_example_utterances():
    # Example lines in the JSON spec would be copied by every agent and look like a fake meme.
    for template in (prompts.DECISION, prompts.REPLY):
        assert "e.g." not in template.lower() and "for example" not in template.lower()
        assert not re.search(r'"utterance"\s*:\s*"', template)
