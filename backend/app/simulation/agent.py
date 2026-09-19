"""Agent state and the validated shapes of what the model is allowed to answer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

from app.llm import prompts

Action = Literal["MOVE", "TALK", "CONTINUE_ACTIVITY", "REACT", "IDLE"]
ACTION_ALIASES = {"CONTINUE": "CONTINUE_ACTIVITY", "SPEAK": "TALK", "SAY": "TALK", "WALK": "MOVE", "GO": "MOVE"}
MAX_UTTERANCE_CHARS = 320


def _clean_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "null", "none", "n/a"}:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text or None


class Decision(BaseModel):
    action: Action
    target: str | None = None
    activity: str | None = None
    utterance: str | None = None
    reason: str | None = None
    importance: int = 3

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value):
        action = str(value).strip().upper().replace(" ", "_")
        return ACTION_ALIASES.get(action, action)

    @field_validator("target", "activity", "utterance", "reason", mode="before")
    @classmethod
    def _normalize_text(cls, value):
        return _clean_text(value)

    @field_validator("utterance")
    @classmethod
    def _truncate(cls, value):
        return value[:MAX_UTTERANCE_CHARS] if value else value

    @field_validator("importance", mode="before")
    @classmethod
    def _clamp_importance(cls, value):
        try:
            return min(max(int(float(value)), 1), 10)
        except (TypeError, ValueError):
            return 3

    @model_validator(mode="after")
    def _check_consistency(self):
        if self.action in ("TALK", "REACT") and not self.utterance:
            raise ValueError(f"{self.action} needs an utterance")
        if self.action in ("MOVE", "TALK") and not self.target:
            raise ValueError(f"{self.action} needs a target")
        return self


class Reply(BaseModel):
    utterance: str | None = None
    end_conversation: bool = False

    @field_validator("utterance", mode="before")
    @classmethod
    def _normalize_text(cls, value):
        text = _clean_text(value)
        return text[:MAX_UTTERANCE_CHARS] if text else None

    @field_validator("end_conversation", mode="before")
    @classmethod
    def _parse_bool(cls, value):
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return bool(value)


@dataclass
class Agent:
    id: str
    name: str
    role: str
    personality: str
    traits: dict[str, float]
    interests: list[str]
    relationships: dict[str, list]  # other agent id -> [label from this agent's view, closeness 0-1]
    schedule: list[list[str]]       # [start "HH:MM", location, activity]
    current_location: str = "Dorm"
    current_activity: str = ""
    current_goal: str = ""
    linger_ticks: int = 0           # consecutive ticks spent away from the scheduled location
    last_seen: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_persona(cls, persona: dict) -> "Agent":
        return cls(
            id=persona["id"], name=persona["name"], role=persona["role"], personality=persona["personality"],
            traits=dict(persona["traits"]), interests=list(persona["interests"]),
            relationships={k: list(v) for k, v in persona["relationships"].items()},
            schedule=[list(s) for s in persona["schedule"]],
        )

    def system_prompt(self) -> str:
        return prompts.render_system(self.name, self.role, self.personality, self.interests)

    def closeness(self, other_id: str) -> float:
        return float(self.relationships.get(other_id, [None, 0.1])[1])

    def relationship_to(self, other: "Agent") -> str:
        label = self.relationships.get(other.id, [None, 0.1])[0]
        return prompts.describe_relationship(label, self.closeness(other.id))

    def bump_closeness(self, other_id: str, delta: float) -> None:
        label, closeness = self.relationships.get(other_id, [None, 0.1])
        self.relationships[other_id] = [label, round(min(1.0, closeness + delta), 3)]

    def snapshot(self) -> dict:
        return {"location": self.current_location, "activity": self.current_activity}
