"""Campus world: a semantic location graph plus simulated time.

Locations are GA "sectors"; each has "arenas" (rooms). Agents in the same
location *and* arena are co-located for conversation; events are visible in
the whole location but attention drops for other arenas.
"""
from __future__ import annotations

import datetime as dt
from collections import deque

WORLD_NAME = "the Homewood campus"

WORLD_GRAPH = {
    "Dorm": ["Dining Hall", "Quad"],
    "Dining Hall": ["Dorm", "Quad", "Classroom"],
    "Classroom": ["Dining Hall", "Library", "Quad"],
    "Library": ["Classroom", "Quad", "Research Lab", "Cafe"],
    "Research Lab": ["Library", "Quad"],
    "Gym": ["Quad"],
    "Cafe": ["Quad", "Library"],
    "Quad": ["Dorm", "Dining Hall", "Classroom", "Library", "Research Lab", "Gym", "Cafe"],
}

ARENAS = {
    "Dorm": ["Room 214", "Room 310", "Room 118", "Room 105", "Room 402", "Grad Apartment", "Lounge"],
    "Dining Hall": ["Main Floor"],
    "Classroom": ["Lecture Hall", "Seminar Room"],
    "Library": ["Study Tables", "Quiet Floor"],
    "Research Lab": ["Dry Lab", "Wet Lab", "Makerspace"],
    "Gym": ["Main Floor"],
    "Cafe": ["Counter"],
    "Quad": ["Lawn"],
}

# Display-only labels for the frontend map (agents only ever see the generic names).
HOMEWOOD_LABELS = {
    "Dorm": "AMR / Wolman",
    "Dining Hall": "FFC",
    "Classroom": "Hodson / Gilman",
    "Library": "MSE Library",
    "Research Lab": "Hackerman",
    "Gym": "Rec Center",
    "Cafe": "Levering Cafe",
    "Quad": "Keyser Quad",
}

# Normalised map coordinates for the frontend (0..1).
MAP_POS = {
    "Dorm": (0.19, 0.2), "Dining Hall": (0.13, 0.66), "Classroom": (0.42, 0.78),
    "Library": (0.62, 0.52), "Research Lab": (0.86, 0.3), "Gym": (0.5, 0.1),
    "Cafe": (0.84, 0.74), "Quad": (0.44, 0.44),
}


def default_arena(location: str) -> str:
    return ARENAS[location][0]


def shortest_path(a: str, b: str) -> list[str]:
    if a == b:
        return [a]
    prev = {a: None}
    q = deque([a])
    while q:
        cur = q.popleft()
        for nb in WORLD_GRAPH[cur]:
            if nb not in prev:
                prev[nb] = cur
                if nb == b:
                    path = [b]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return path[::-1]
                q.append(nb)
    return [a, b]


def _minutes(v) -> int:
    """'07:30' -> 450. YAML 1.1 parses unquoted 07:30 as sexagesimal 450, which is already minutes."""
    if isinstance(v, int):
        return v
    h, m = map(int, str(v).split(":"))
    return h * 60 + m


class Clock:
    def __init__(self, cfg: dict):
        self.start_date = dt.datetime.strptime(cfg["start_date"], "%Y-%m-%d")
        self.day_start = dt.timedelta(minutes=_minutes(cfg["day_start"]))
        self.day_end = dt.timedelta(minutes=_minutes(cfg["day_end"]))
        self.tick_minutes = int(cfg["tick_minutes"])
        self.days = int(cfg["simulation_days"])
        self.ticks_per_day = int((self.day_end - self.day_start).total_seconds() // 60 // self.tick_minutes)

    @property
    def total_ticks(self) -> int:
        return self.ticks_per_day * self.days

    def time_of(self, tick: int) -> dt.datetime:
        day, k = divmod(tick, self.ticks_per_day)
        return self.start_date + dt.timedelta(days=day) + self.day_start + dt.timedelta(minutes=k * self.tick_minutes)

    def day_of(self, tick: int) -> int:
        return tick // self.ticks_per_day + 1

    def label(self, tick: int) -> str:
        t = self.time_of(tick)
        return f"Day {self.day_of(tick)} {t.strftime('%H:%M')}"
