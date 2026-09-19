"""The campus: a small graph of semantic locations plus a 2D layout for the frontend map.

Agents move one edge per tick. The Quad is the hub (like Homewood's Keyser Quad), and the
outer buildings form a ring, so walking anywhere tends to take you through shared space.
Map coordinates are in a 1000 x 640 SVG viewBox; the backend never uses them.
"""
from __future__ import annotations

LOCATIONS: dict[str, dict] = {
    "Dorm": {
        "description": "a brick residence hall with a shared common room and a temperamental microwave",
        "rect": [40, 30, 230, 170],
    },
    "Dining Hall": {
        "description": "a busy dining hall with long tables and a soft-serve machine",
        "rect": [385, 20, 230, 170],
    },
    "Gym": {
        "description": "the rec center, with weights, a track and a climbing wall",
        "rect": [730, 30, 230, 170],
    },
    "Classroom": {
        "description": "a lecture hall with tiered seats and an aging projector",
        "rect": [40, 250, 230, 170],
    },
    "Quad": {
        "description": "the open grassy quad at the center of campus, with benches, a fountain and a coffee cart",
        "rect": [375, 235, 250, 180],
    },
    "Research Lab": {
        "description": "a cramped research lab full of equipment and one ancient printer",
        "rect": [730, 250, 230, 170],
    },
    "Library": {
        "description": "a big library with a silent study floor and group study rooms",
        "rect": [385, 450, 230, 170],
    },
}

EDGES: list[tuple[str, str]] = [
    ("Quad", "Dorm"), ("Quad", "Dining Hall"), ("Quad", "Gym"), ("Quad", "Classroom"),
    ("Quad", "Research Lab"), ("Quad", "Library"),
    ("Dorm", "Dining Hall"), ("Dining Hall", "Gym"), ("Dorm", "Classroom"),
    ("Gym", "Research Lab"), ("Classroom", "Library"), ("Research Lab", "Library"),
]


def build_graph() -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {name: [] for name in LOCATIONS}
    for a, b in EDGES:
        graph[a].append(b)
        graph[b].append(a)
    return graph


GRAPH = build_graph()


def resolve_location(name: str | None) -> str | None:
    """Map a model-provided place name onto a real location (case/whitespace tolerant)."""
    if not name:
        return None
    wanted = name.strip().lower().removeprefix("the ")
    for location in LOCATIONS:
        if location.lower() == wanted:
            return location
    for location in LOCATIONS:
        if location.lower() in wanted or wanted in location.lower():
            return location
    return None


def world_layout() -> dict:
    """Everything the frontend needs to draw the map."""
    return {
        "locations": [{"name": n, "description": d["description"], "rect": d["rect"]} for n, d in LOCATIONS.items()],
        "edges": [list(e) for e in EDGES],
        "viewBox": [0, 0, 1000, 640],
    }
