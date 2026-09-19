"""Deterministic offline stand-in for an LLM (tests, frontend development, CI).

Responses are produced per call purpose and are a pure function of the prompt,
so runs with the mock backend are reproducible. The mock does NOT inject any
catch-phrases; utterances are stitched from the speaker's retrieved memories,
so any repeated wording can only come from actual memory/exposure dynamics.
"""
from __future__ import annotations

import hashlib
import json
import re

from backend.llm.client import Backend, current_purpose


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def _section(prompt: str, start: str, end: str | None = None) -> str:
    i = prompt.find(start)
    if i < 0:
        return ""
    s = prompt[i + len(start):]
    if end:
        j = s.find(end)
        if j >= 0:
            s = s[:j]
    return s


def _name(prompt: str) -> str:
    m = re.search(r"Name: ([A-Z][a-z]+ [A-Z][a-z]+)", prompt)
    return m.group(1) if m else "Someone"


class MockBackend(Backend):
    name = "mock"
    model = "mock"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def generate(self, prompt, system, max_tokens, temperature):
        p = current_purpose()
        h = _h(f"{self.seed}|{prompt}")
        if p == "commons_action":
            from backend.llm.commons_mock import respond
            return respond(prompt)
        if p.startswith("poignancy"):
            base = 2 + h % 4
            if re.search(r"missed|wrong|failed|spilled|locked|late|lost|prize|praised|offered", prompt.split("Rate")[0][-400:]):
                base += 3
            return json.dumps({"output": str(min(9, base))})
        if p == "reminding":
            n = len(re.findall(r"^\d+\. ", prompt.split("earlier things")[-1], re.M))
            if n and h % 3 == 0:
                return json.dumps({"reminded_of": 1 + h % n, "what_felt_alike": "it went sideways the same way"})
            return json.dumps({"reminded_of": None, "what_felt_alike": None})
        if p == "viewpoint":
            items = re.findall(r"^(f\d+) \[([^\]]+)\] (.+)$", prompt, re.M)
            out = {}
            for k, tag, text in items:
                if tag.startswith("different room") and h % 3 == 0:
                    out[k] = ""
                elif tag.startswith("different room"):
                    out[k] = "From the next room, it looked like " + text[0].lower() + text[1:]
                else:
                    out[k] = text
            return json.dumps(out)
        if p == "decide_to_talk":
            return f" They are nearby and know each other.\nAnswer in yes or no: {'yes' if h % 10 < 6 else 'no'}"
        if p == "reflection_focal_points":
            return "What has been happening lately?\n2) Who has been having a rough time?\n3) What keeps going wrong?"
        if p == "reflection_insights":
            stm = re.findall(r"^\d+\. (.+)$", prompt, re.M)
            if not stm:
                return " Things have been busy (because of 0)"
            a = stm[h % len(stm)]
            b = stm[(h // 7) % len(stm)]
            return (f" It seems that {a[:80].rstrip('.').lower()} (because of {h % len(stm)})\n"
                    f"2. Recently, {b[:70].rstrip('.').lower()} (because of {(h // 7) % len(stm)})")
        if p == "chat_relationship_summary":
            names = re.findall(r"summarize ([A-Z][a-z]+ [A-Z][a-z]+) and ([A-Z][a-z]+ [A-Z][a-z]+)'s", prompt)
            a, b = names[0] if names else ("They", "each other")
            return json.dumps({"output": f"{a} and {b} know each other from campus and chat now and then."})
        if p == "chat_utterance":
            me = _name(prompt)
            mem = [l[2:].strip() for l in _section(prompt, "head:", "PART 2").splitlines() if l.startswith("- ")]
            convo = _section(prompt, "conversation so far:", "---").strip()
            n_lines = 0 if convo.startswith("[The conversation") else len(convo.splitlines())
            if mem:
                m = mem[h % len(mem)]
                m = re.sub(r"^[A-Z][a-z]+ [A-Z][a-z]+ (saw|heard in a conversation|overheard|saw:)[: ]*", "", m)
                frag = " ".join(m.split()[:14]).rstrip(".,")
                opts = [f"Did you hear? {frag}.", f"Honestly, {frag[0].lower() + frag[1:]}.",
                        f"I keep thinking about how {frag[0].lower() + frag[1:]}.", f"Yeah... {frag}."]
                utt = opts[h % len(opts)]
            else:
                utt = "Hey, how's your day going?"
            end = n_lines >= 2 and h % 3 == 0
            return json.dumps({me: utt, f"Did the conversation end with {me}'s utterance?": end})
        if p == "group_chat_utterance":   # multi-party talk (agents/group_conversation.py, group_chat_v1.txt)
            mem = [l[2:].strip() for l in _section(prompt, "head:", "PART 2").splitlines() if l.startswith("- ")]
            convo = _section(prompt, "conversation so far:", "---").strip()
            n_lines = 0 if convo.startswith("[The conversation") else len(convo.splitlines())
            m = re.sub(r"^[A-Z][a-z]+ [A-Z][a-z]+ (saw|heard in a conversation|overheard|remembers that)[: ]*", "",
                       mem[h % len(mem)]) if mem else ""
            frag = " ".join(m.split()[:12]).rstrip(".,")
            utt = f"So, {frag[0].lower() + frag[1:]}." if frag else "How's everyone's day going?"
            return json.dumps({"utterance": utt, "end": n_lines >= 3 and h % 3 == 0})
        if p == "encode_memory":
            name = re.search(r"brief description of ([A-Z][a-z]+ [A-Z][a-z]+)", prompt)
            name = name.group(1) if name else "Someone"
            obs = _section(prompt, "just ", "People do not").split(":", 1)[-1].strip()
            words = " ".join(obs.split())
            k = 30 if "every detail" in prompt else 18 if "main points" in prompt else 12
            return f"{name} remembers that " + " ".join(words.split()[:k]).rstrip(".,") + "."
        if p == "react_decision":
            people = _section(prompt, "People nearby:", "\n").strip()
            noticed = _section(prompt, "just noticed:\n", "\n\n").strip()
            names = [x.strip() for x in people.split(",") if x.strip() and x.strip() != "nobody"]
            r = h % 10
            if r < 4 or not noticed:
                return json.dumps({"action": "CONTINUE", "target": None, "utterance": None, "reason": "busy"})
            snippet = " ".join(noticed.split()[:8]).rstrip(".,")
            if r < 7:
                return json.dumps({"action": "REACT", "target": None, "utterance": f"Whoa, {snippet}!", "reason": "surprised"})
            if names:
                return json.dumps({"action": "TALK", "target": names[h % len(names)],
                                   "utterance": f"Did you see that? {snippet}.", "reason": "share"})
            return json.dumps({"action": "CONTINUE", "target": None, "utterance": None, "reason": "alone"})
        if p == "probe_meaning":
            return "I am not sure; I think I heard it in passing."
        if p == "probe_match":
            return "ABCD"[h % 4]
        if p == "analysis_classifier":
            return json.dumps({"is_convention": h % 3 == 0, "gloss": "a locally used expression", "confidence": 0.5})
        if p == "analysis_discovery":
            return json.dumps({"expressions": []})
        if p == "world_event_generator":
            return "{}"
        return "OK"
