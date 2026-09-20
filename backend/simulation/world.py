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
    "Museum": ["Quad", "Library", "Apartments", "Nolans"],                   # Homewood Museum
    "Theater": ["Quad", "Library", "Shuttle Stop"],                          # Merrick Barn, behind Brody
    "Apartments": ["Museum", "Shuttle Stop", "Nolans"],                      # The Charles, on N Charles St
    "Athletic Center": ["Gym", "Dorm"],                                      # White Athletic Center
    "Admin Building": ["Cafe", "Student Center", "Research Lab"],            # Garland Hall
    "Shuttle Stop": ["Auditorium", "Theater", "Apartments"],                 # N Charles St at the Gatehouse
    # The second dining hall (D76). It is the UNEXPOSED control venue for a dining-sited incident: people who
    # eat here have to HEAR about what happened, they cannot see it. Its walking neighbours are the real
    # east-side ones (Scott-Bates Commons sits beside The Charles, up N Charles St from the Museum), and the
    # reciprocal edges are appended to the END of those two lists, so no pre-existing shortest path moves.
    "Nolans": ["Apartments", "Museum"],                                      # Nolan's, Scott-Bates Commons
}

ARENAS = {
    "Dorm": ["Room 214", "Room 310", "Room 118", "Room 105", "Room 402", "Grad Apartment", "Lounge",
             "Room 216", "Room 120", "Room 107", "Room 312", "Room 404"],
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
    "Nolans": ["Servery"],      # one room, like the other hall: the two venues differ in where they are, not in kind
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
    "Nolans": "Nolan's on 33rd",   # the dining hall in Scott-Bates Commons; the map draws its western wing
}

# --- situated reference (world.reference_mode: situated) --------------------------------------------------
# What the world SHOWS an agent standing here, instead of the place's canonical name. A description, never a
# proper name: no canonical key of WORLD_GRAPH/ARENAS and no HOMEWOOD_LABELS building, so the only names in an
# agent's head are the ones it was seeded with or heard from someone else (naming is the agents' business).
# Written from this map's own geography: WORLD_GRAPH adjacency for "beside/across from", MAP_POS for the
# compass words. Lower case and article-carrying, because they are common nouns, not labels.
#
# The one common noun that survives is "dining hall" (see backend/simulation/reference.py): a seed memory about
# a dining hall has to use the same common noun as the percept, or it never attaches to the referent at all. It
# is not the canonical label "Dining Hall" and the name analyzer scores it `no_target_alias`. Since D76 there
# are TWO halls, so the noun alone no longer picks one out: each description names the hall's own side of
# campus, in the same shape, and it is the speaker -- not the world -- who says which hall is meant.
PLACE_DESCRIPTIONS = {
    "Dorm": "the freshman residence halls to the west",
    "Dining Hall": "the dining hall beside the freshman residences",
    "Classroom": "the teaching building south of the green",
    "Library": "the quiet building full of books",
    "Research Lab": "the research building on the north-east edge",
    "Gym": "the workout building at the north end",
    "Cafe": "the small coffee place to the south-east",
    "Quad": "the open green at the campus centre",
    "Student Center": "the student hangout building to the south-east",
    "Auditorium": "the big event hall to the north-east",
    "Engineering Hall": "the engineering building at the far north-east",
    "Science Hall": "the science teaching building to the north-west",
    "Museum": "the old exhibit house to the east",
    "Theater": "the old playhouse just north-east",
    "Apartments": "the graduate flats on the far east",
    "Athletic Center": "the sports complex at the far north",
    "Admin Building": "the administration offices to the far east",
    "Shuttle Stop": "the shuttle pickup at the north-east gate",
    # The control hall. Word for word the same shape as the other hall's description -- "the dining hall" plus
    # whose side of campus it is on -- so nothing in the world's own wording marks one of them as the main one
    # or the other one; which hall a story is about has to come from what the speaker says. It locates the
    # hall by the graduate flats' own DESCRIPTION, never by the canonical key "Apartments", and it carries no
    # comma, because `reference.options()` joins the MOVE list with ", " and a comma inside one option splits
    # it into two (tests/test_reference.py checks both).
    "Nolans": "the dining hall by the graduate flats",
}

# Rooms, described the same way (keyed by place, because arena names repeat: "Main Floor" is both a serving
# floor and a gym floor). Every ARENAS entry has one.
ARENA_DESCRIPTIONS = {
    "Dorm": {"Room 214": "a single bedroom", "Room 310": "a single bedroom", "Room 118": "a single bedroom",
             "Room 105": "a single bedroom", "Room 402": "a single bedroom", "Room 216": "a single bedroom",
             "Room 120": "a single bedroom", "Room 107": "a single bedroom", "Room 312": "a single bedroom",
             "Room 404": "a single bedroom", "Grad Apartment": "a graduate flat upstairs",
             "Lounge": "the shared sitting area"},
    "Dining Hall": {"Main Floor": "the serving floor"},
    "Classroom": {"Lecture Hall": "the big tiered teaching room", "Seminar Room": "the small seminar space"},
    "Library": {"Study Tables": "the shared work tables", "Quiet Floor": "the silent upper level"},
    "Research Lab": {"Dry Lab": "the computer side", "Wet Lab": "the wet-work side",
                     "Makerspace": "the build space", "Stockroom": "the supply closet"},
    "Gym": {"Main Floor": "the workout floor"},
    "Cafe": {"Counter": "the order line"},
    "Quad": {"Lawn": "the grass"},
    "Student Center": {"Common Room": "the shared sitting area", "Club Room": "the club meeting space"},
    "Auditorium": {"Main Hall": "the seating hall", "Lobby": "the entrance area"},
    "Engineering Hall": {"Computer Lab": "the machine room", "Study Lounge": "the quiet corner"},
    "Science Hall": {"Teaching Lab": "the student work room", "Lecture Room": "the tiered teaching room"},
    "Museum": {"Gallery": "the display rooms", "Garden": "the walled grounds"},
    "Theater": {"Stage": "the performance floor", "Green Room": "the backstage room"},
    "Apartments": {"Kitchen": "the shared cooking area", "Study Room": "the quiet work room"},
    "Athletic Center": {"Field House": "the indoor court", "Track": "the running oval"},
    "Admin Building": {"Front Desk": "the reception window", "Mailroom": "the package room"},
    "Shuttle Stop": {"Bench": "the waiting spot"},
    # Word for word what the exposed hall's room is called. The two halls are matched on everything the world
    # says about them except where they are, so a difference between them is an exposure difference.
    "Nolans": {"Servery": "the serving floor"},
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
    "Shuttle Stop": (0.72, 0.26), "Nolans": (0.91, 0.52),
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
