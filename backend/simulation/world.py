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
    # The first eight places and their original edges are unchanged. Edges to the campus places added later are
    # appended at the END of each neighbour list, so every shortest path between two original places is the same
    # as before (BFS still reaches the old intermediate first). All edges are reciprocal; the Quad stays the hub.
    "Dorm": ["Dining Hall", "Quad", "Athletic Center"],
    "Dining Hall": ["Dorm", "Quad", "Classroom", "Science Hall"],
    "Classroom": ["Dining Hall", "Library", "Quad", "Student Center"],
    "Library": ["Classroom", "Quad", "Research Lab", "Cafe", "Museum", "Theater"],
    "Research Lab": ["Library", "Quad", "Engineering Hall", "Auditorium", "Admin Building"],
    "Gym": ["Quad", "Science Hall", "Athletic Center"],
    "Cafe": ["Quad", "Library", "Student Center", "Admin Building"],
    "Quad": ["Dorm", "Dining Hall", "Classroom", "Library", "Research Lab", "Gym", "Cafe",
             "Student Center", "Science Hall", "Museum", "Theater"],
    # campus places (real walking neighbours on the Homewood map)
    "Student Center": ["Cafe", "Classroom", "Quad", "Admin Building"],      # Glass Pavilion, on Levering
    "Auditorium": ["Research Lab", "Engineering Hall", "Shuttle Stop"],       # Shriver Hall
    "Engineering Hall": ["Research Lab", "Auditorium"],                      # Malone Hall, south of Hackerman
    "Science Hall": ["Quad", "Gym", "Dining Hall"],                          # Mudd Hall
    "Museum": ["Quad", "Library", "Apartments"],                             # Homewood Museum
    "Theater": ["Quad", "Library", "Shuttle Stop"],                          # Merrick Barn, behind Brody
    "Apartments": ["Museum", "Shuttle Stop"],                                # The Charles, on N Charles St
    "Athletic Center": ["Gym", "Dorm"],                                      # White Athletic Center
    "Admin Building": ["Cafe", "Student Center", "Research Lab"],            # Garland Hall
    "Shuttle Stop": ["Auditorium", "Theater", "Apartments"],                 # N Charles St at the Gatehouse
}

ARENAS = {
    "Dorm": ["Room 214", "Room 310", "Room 118", "Room 105", "Room 402", "Grad Apartment", "Lounge"],
    "Dining Hall": ["Main Floor"],
    "Classroom": ["Lecture Hall", "Seminar Room"],
    "Library": ["Study Tables", "Quiet Floor"],
    "Research Lab": ["Dry Lab", "Wet Lab", "Makerspace", "Stockroom"],   # Stockroom: v3 (appended)
    "Gym": ["Main Floor"],
    "Cafe": ["Counter"],
    "Quad": ["Lawn"],
    # campus places (appended; ARENAS[loc][0] is the default arena)
    "Student Center": ["Common Room", "Club Room"],
    "Auditorium": ["Main Hall", "Lobby"],
    "Engineering Hall": ["Computer Lab", "Study Lounge"],
    "Science Hall": ["Teaching Lab", "Lecture Room"],
    "Museum": ["Gallery", "Garden"],
    "Theater": ["Stage", "Green Room"],
    "Apartments": ["Kitchen", "Study Room"],
    "Athletic Center": ["Field House", "Track"],
    "Admin Building": ["Front Desk", "Mailroom"],
    "Shuttle Stop": ["Bench"],
}

# Places anyone on campus knows and may drop by (every profile's known_locations; the default errand places).
PUBLIC_PLACES = ["Student Center", "Auditorium", "Engineering Hall", "Science Hall", "Museum", "Theater",
                 "Athletic Center", "Admin Building", "Shuttle Stop", "Cafe", "Library", "Quad"]

# Display-only labels for the frontend map (agents only ever see the generic names).
HOMEWOOD_LABELS = {   # the real Homewood buildings each place is drawn as (frontend/homewood_map.json)
    "Dorm": "AMR II",
    "Dining Hall": "Hopkins Cafe (FFC)",
    "Classroom": "Gilman Hall",
    "Library": "MSE Library / Brody",
    "Research Lab": "Hackerman Hall",
    "Gym": "O'Connor Rec Center",
    "Cafe": "Levering Cafe",
    "Quad": "Keyser Quad",
    "Student Center": "Glass Pavilion",
    "Auditorium": "Shriver Hall",
    "Engineering Hall": "Malone Hall",
    "Science Hall": "Mudd Hall",
    "Museum": "Homewood Museum",
    "Theater": "Merrick Barn",
    "Apartments": "The Charles",
    "Athletic Center": "White Athletic Center",
    "Admin Building": "Garland Hall",
    "Shuttle Stop": "Gatehouse stop, N Charles St",
}

# Normalised map coordinates for the frontend (0..1): a display-only schematic (the tile map in
# frontend/homewood_map.json has the real positions). New places sit next to their graph neighbours.
MAP_POS = {
    "Dorm": (0.19, 0.2), "Dining Hall": (0.13, 0.66), "Classroom": (0.42, 0.78),
    "Library": (0.62, 0.52), "Research Lab": (0.86, 0.3), "Gym": (0.5, 0.1),
    "Cafe": (0.84, 0.74), "Quad": (0.44, 0.44),
    "Student Center": (0.63, 0.7), "Auditorium": (0.76, 0.12), "Engineering Hall": (0.95, 0.14),
    "Science Hall": (0.3, 0.32), "Museum": (0.76, 0.52), "Theater": (0.6, 0.35),
    "Apartments": (0.88, 0.44), "Athletic Center": (0.32, 0.06), "Admin Building": (0.95, 0.6),
    "Shuttle Stop": (0.72, 0.26),
}


def default_arena(location: str) -> str:
    return ARENAS[location][0]


def shortest_path(a: str, b: str) -> list[str]:
    if a == b:
        return [a]
    if a not in WORLD_GRAPH or b not in WORLD_GRAPH:   # v3 "Away" sentinel for members not on the roster
        return [b]
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
