"""Simulation clock, schedule lookup, and graph pathfinding.

Time is discrete: one tick = `tick_minutes` of campus time. Each simulated day runs from
DAY_START to DAY_END; nights are skipped (everyone is back at their first scheduled spot).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

DAY_START = 8 * 60  # 08:00
DAY_END = 22 * 60   # 22:00


@dataclass(frozen=True)
class Clock:
    tick_minutes: int = 15

    @property
    def ticks_per_day(self) -> int:
        return (DAY_END - DAY_START) // self.tick_minutes

    def total_ticks(self, days: int) -> int:
        return self.ticks_per_day * days

    def day_and_minute(self, tick: int) -> tuple[int, int]:
        """(1-based day, minute of day) for a tick."""
        day, offset = divmod(tick, self.ticks_per_day)
        return day + 1, DAY_START + offset * self.tick_minutes

    def is_day_start(self, tick: int) -> bool:
        return tick % self.ticks_per_day == 0

    def label(self, tick: int) -> str:
        day, minute = self.day_and_minute(tick)
        return f"Day {day}, {format_minute(minute)}"


def parse_hhmm(value: str) -> int:
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def format_minute(minute: int) -> str:
    hours, mins = divmod(minute, 60)
    suffix = "AM" if hours < 12 else "PM"
    return f"{(hours - 1) % 12 + 1}:{mins:02d} {suffix}"


def scheduled_slot(schedule: list[tuple[str, str, str]], minute: int) -> tuple[str, str]:
    """(location, activity) the schedule asks for at `minute`. Before the first entry -> first entry."""
    current = schedule[0]
    for entry in schedule:
        if parse_hhmm(entry[0]) <= minute:
            current = entry
        else:
            break
    return current[1], current[2]


def next_hop(graph: dict[str, list[str]], start: str, goal: str) -> str | None:
    """First step on a shortest path from start to goal (None if already there / unreachable)."""
    if start == goal:
        return None
    parents: dict[str, str] = {start: start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor in parents:
                continue
            parents[neighbor] = node
            if neighbor == goal:
                step = neighbor
                while parents[step] != start:
                    step = parents[step]
                return step
            queue.append(neighbor)
    return None
