"""World events: small, recurring, slightly absurd incidents that agents can witness.

Shared referents are what inside jokes are about, so the world supplies them; the agents
supply the language. Event text is plain description. Don't hand agents a ready-made
catchphrase here: the meme detector ignores phrases made only of words the world supplies,
so a quoted event line can never count as an invention.

Each template may fire at most once per day, at a random tick inside its time window,
with probability `daily_prob`. `topic` is a short hyphenated keyword used by the mock LLM
and the analysis UI; the real model never sees it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from app.simulation.scheduler import DAY_START, Clock, parse_hhmm


@dataclass(frozen=True)
class EventTemplate:
    key: str
    location: str
    text: str  # may contain {subject}: a random agent present at the location
    topic: str
    window: tuple[str, str]
    daily_prob: float
    duration: int = 2  # ticks the event stays visible
    needs_subject: bool = False


@dataclass
class WorldEvent:
    key: str
    location: str
    text: str
    topic: str
    start_tick: int
    end_tick: int
    subject_id: str | None = None
    event_id: int | None = None  # row id in the events log
    seen_by: set[str] = field(default_factory=set)


TEMPLATES: list[EventTemplate] = [
    EventTemplate("tray_drop", "Dining Hall",
                  "{subject} drops a full lunch tray. Plates clatter across the floor and the whole room goes quiet for a second.",
                  "tray", ("12:00", "13:30"), 0.65, duration=2, needs_subject=True),
    EventTemplate("softserve_broken", "Dining Hall",
                  "The soft-serve machine has a handwritten OUT OF ORDER sign taped to it again.",
                  "soft-serve", ("11:30", "19:00"), 0.75, duration=10),
    EventTemplate("squirrel_bagel", "Quad",
                  "A very bold squirrel snatches half a bagel out of someone's hand and sits on the fountain eating it.",
                  "squirrel", ("10:00", "17:30"), 0.7),
    EventTemplate("tour_group", "Quad",
                  "A campus tour group shuffles past. The guide, walking backwards, nearly trips into the fountain.",
                  "tour-guide", ("10:00", "16:00"), 0.5),
    EventTemplate("projector_freeze", "Classroom",
                  "The projector freezes on the same slide again, and the professor gives up and starts drawing on the whiteboard.",
                  "projector", ("09:00", "16:00"), 0.6, duration=3),
    EventTemplate("fire_alarm", "Library",
                  "The fire alarm goes off for no apparent reason and everyone has to shuffle outside for ten minutes.",
                  "fire-alarm", ("11:00", "20:00"), 0.25),
    EventTemplate("loud_laptop", "Library",
                  "Someone's laptop suddenly blasts a video at full volume on the silent study floor.",
                  "laptop", ("10:00", "20:00"), 0.4, duration=1),
    EventTemplate("printer_jam", "Research Lab",
                  "The ancient lab printer jams, grinds loudly for a full minute, then prints one blank page.",
                  "printer", ("08:00", "21:00"), 0.7),
    EventTemplate("flat_results", "Research Lab",
                  "The overnight experiment results come in and the graph is completely flat.",
                  "flat-graph", ("08:00", "12:00"), 0.3, duration=3),
    EventTemplate("gym_loop", "Gym",
                  "The gym speakers get stuck looping the same eight seconds of a pop song.",
                  "playlist", ("08:00", "20:00"), 0.6, duration=3),
    EventTemplate("burnt_popcorn", "Dorm",
                  "The hallway smells like burnt popcorn, and someone has taped a long passive-aggressive note to the microwave.",
                  "popcorn", ("17:00", "21:45"), 0.6, duration=4),
    EventTemplate("wifi_down", "Dorm",
                  "The dorm Wi-Fi goes down and everyone drifts into the hallway holding their laptops up like antennas.",
                  "wifi", ("08:00", "21:45"), 0.3, duration=2),
    EventTemplate("sudden_rain", "Quad",
                  "It suddenly starts pouring and everyone on the quad sprints for cover.",
                  "rain", ("10:00", "19:00"), 0.2, duration=1),
]


class EventScheduler:
    """Plans which templates fire each day and materializes them into WorldEvents."""

    def __init__(self, clock: Clock, rng: random.Random, templates: list[EventTemplate] = TEMPLATES):
        self.clock = clock
        self.rng = rng
        self.templates = templates
        self.planned: dict[int, list[EventTemplate]] = {}
        self.window_end: dict[str, int] = {}  # template key -> last tick it may fire today
        self.active_events: list[WorldEvent] = []

    def plan_day(self, first_tick: int) -> None:
        self.planned.clear()
        tpd = self.clock.ticks_per_day
        for template in self.templates:
            if self.rng.random() >= template.daily_prob:
                continue
            start = self._tick_for(first_tick, template.window[0])
            end = min(self._tick_for(first_tick, template.window[1]), first_tick + tpd - 1)
            if end < start:
                continue
            self.planned.setdefault(self.rng.randint(start, end), []).append(template)
            self.window_end[template.key] = end

    def _tick_for(self, first_tick: int, hhmm: str) -> int:
        return first_tick + max(0, (parse_hhmm(hhmm) - DAY_START) // self.clock.tick_minutes)

    def spawn(self, tick: int, agents_at: dict[str, list[tuple[str, str]]]) -> list[WorldEvent]:
        """Fire events planned for this tick. `agents_at` maps location -> [(agent_id, name)]."""
        fired: list[WorldEvent] = []
        for template in self.planned.pop(tick, []):
            subject_id, text = None, template.text
            if template.needs_subject:
                present = agents_at.get(template.location, [])
                if not present:
                    if tick < self.window_end.get(template.key, tick):  # nobody here yet; try next tick
                        self.planned.setdefault(tick + 1, []).append(template)
                    continue
                subject_id, subject_name = self.rng.choice(present)
                text = template.text.format(subject=subject_name)
            event = WorldEvent(template.key, template.location, text, template.topic,
                               tick, tick + template.duration - 1, subject_id)
            fired.append(event)
            self.active_events.append(event)
        return fired

    def active(self, tick: int) -> list[WorldEvent]:
        self.active_events = [e for e in self.active_events if e.end_tick >= tick]
        return [e for e in self.active_events if e.start_tick <= tick]
