"""Consequential campus world. Pure state transitions; no LLM or observer imports.

Only view() and explicitly addressed event text may enter agent prompts. snapshot()
is a researcher view and includes information unavailable to individual residents.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass, field

from backend.simulation.world import WORLD_GRAPH, shortest_path


ACTIONS = {"WAIT", "MOVE", "TAKE", "DROP", "ASSEMBLE", "CALIBRATE", "TEST",
           "DELIVER", "SAY", "WRITE", "READ", "REVISE"}
ROOMS = {"Research Lab": "Makerspace", "Library": "Study Tables", "Quad": "Lawn", "Cafe": "Counter"}


def keyed_number(seed: int, *parts) -> int:
    """A separate deterministic draw per purpose; no shared mutable RNG stream."""
    raw = "|".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big")


@dataclass
class Resident:
    id: str
    name: str
    location: str = "Quad"
    carrying: str | None = None
    active: bool = True
    joined: int = 0
    departed: int | None = None


@dataclass
class Kit:
    id: str
    project: str
    location: str = "Research Lab"
    holder: str | None = None
    assembled: bool = False
    calibration: str | None = None
    delivered: bool = False


@dataclass
class Project:
    id: str
    title: str
    site: str
    kit: str
    released: int
    due: int
    completed: int | None = None
    attempts: int = 0


@dataclass
class Record:
    id: str
    versions: list[dict] = field(default_factory=list)


@dataclass
class Operation:
    id: str
    actor: str
    intention: dict
    started: int
    due: int
    locks: list[str] = field(default_factory=list)


class InvalidAction(ValueError):
    pass


class CommonsWorld:
    schema_version = 1

    def __init__(self, cfg: dict, names: dict[str, str], ticks_per_day: int):
        self.cfg = copy.deepcopy(cfg)
        self.seed = int(cfg.get("seed", 42))
        self.ticks_per_day = ticks_per_day
        if ticks_per_day < 1:
            raise ValueError("The working day must contain at least one tick")
        self.records_enabled = bool(cfg.get("records_enabled", True))
        self.change_enabled = bool(cfg.get("change_enabled", True))
        self.change_day = int(cfg.get("change_day", 5))
        self.turnover_day = int(cfg.get("turnover_day", 3))
        if self.change_day < 2 or self.turnover_day < 2:
            raise ValueError("Change and turnover days must be at least 2")
        self.change_tick = (self.change_day - 1) * ticks_per_day
        self.turnover_tick = (self.turnover_day - 1) * ticks_per_day
        self.newcomers = copy.deepcopy(cfg.get("newcomers", []))
        seen = set(names)
        replaced = set()
        for n in self.newcomers:
            if n["replaces"] not in names or n["replaces"] in replaced or n["id"] in seen:
                raise ValueError("Newcomers need unique ids and distinct existing predecessors")
            if len(n["name"].split()) < 2:
                raise ValueError("Newcomer names must include first and last names")
            seen.add(n["id"])
            replaced.add(n["replaces"])
        self.residents = {aid: Resident(aid, name) for aid, name in names.items()}
        self.kits: dict[str, Kit] = {}
        self.projects: dict[str, Project] = {}
        self.records: dict[str, Record] = {}
        self.operations: dict[str, Operation] = {}
        self.locks: dict[str, str] = {}
        self.read_versions: dict[str, set[tuple[str, int]]] = {aid: set() for aid in names}
        self.supplies = {"components": int(cfg.get("initial_components", 16)),
                         "test_supplies": int(cfg.get("initial_test_supplies", 24))}
        self.daily_supply = {"components": int(cfg.get("daily_components", 8)),
                             "test_supplies": int(cfg.get("daily_test_supplies", 16))}
        if min(*self.supplies.values(), *self.daily_supply.values()) < 0:
            raise ValueError("Supplies must be nonnegative")
        self.consumed = {k: 0 for k in self.supplies}
        self.tick = -1
        self.outdoor_condition = "dry"
        self._bench_mode = "AB"[keyed_number(self.seed, "physical_mapping") % 2]
        self._counter = 0
        self._record_counter = 0
        self._events: list[dict] = []
        self._submitted: set[str] = set()

    def _emit(self, kind: str, *, actor=None, text="", visible_to=(), **fields) -> dict:
        self._counter += 1
        event = {"event_key": f"cw{self._counter:06d}", "kind": kind, "tick": self.tick,
                 "actor": actor, "text": text, "visible_to": sorted(set(visible_to)), **fields}
        self._events.append(event)
        return event

    def drain_events(self) -> list[dict]:
        events, self._events = self._events, []
        return events

    def active_ids(self) -> list[str]:
        return sorted(aid for aid, r in self.residents.items() if r.active)

    def _nearby(self, location: str) -> list[str]:
        return [aid for aid in self.active_ids() if self.residents[aid].location == location
                and not self._traveling(aid)]

    def _traveling(self, aid: str) -> bool:
        op = self.operations.get(aid)
        return bool(op and op.intention["action"] == "MOVE")

    def advance(self, tick: int):
        if tick != self.tick + 1:
            raise ValueError("Advance exactly one tick at a time")
        self.tick = tick
        self._submitted.clear()
        if tick % self.ticks_per_day == 0:
            day = tick // self.ticks_per_day + 1
            if tick:
                for resource, amount in self.daily_supply.items():
                    self.supplies[resource] += amount
                self._emit("supply", amounts=dict(self.daily_supply),
                           text="The scheduled workshop supplies arrived.",
                           visible_to=self._nearby("Research Lab"))
            for index, site in enumerate(("Library", "Quad")):
                pid = f"project-{day:02d}-{index + 1}"
                kid = f"kit-{day:02d}-{index + 1}"
                title = f"{'Indoor' if site == 'Library' else 'Outdoor'} monitor {day}"
                project = Project(pid, title, site, kid, tick, tick + self.ticks_per_day - 1)
                self.projects[pid], self.kits[kid] = project, Kit(kid, pid)
                self._emit("project_released", project=asdict(project),
                           text=f"Request posted: {title}, using {kid} at {site}; tolerance is 1 unit.",
                           visible_to=self._nearby("Quad"))
        if tick == self.change_tick:
            if self.change_enabled:
                self.outdoor_condition = "humid"
            # This treatment event is observer-only. Local conditions are separately observable.
            self._emit("intervention", enabled=self.change_enabled,
                       condition=self.outdoor_condition)
            if self.change_enabled:
                self._emit("weather_observed", text="The outdoor air is now humid.",
                           visible_to=self._nearby("Quad"))
        if tick == self.turnover_tick:
            for spec in self.newcomers:
                self._replace(spec)
        due = sorted((o for o in self.operations.values() if o.due <= tick), key=lambda o: o.id)
        for op in due:
            self._complete(op)
            self._release(op)

    def _replace(self, spec: dict):
        old = self.residents[spec["replaces"]]
        op = self.operations.get(old.id)
        if op:
            self._emit("action_canceled", actor=old.id, operation=op.id,
                       reason="departure", action=op.intention["action"])
            self._release(op)
        if old.carrying:
            kit = self.kits[old.carrying]
            kit.holder, kit.location = None, old.location
            old.carrying = None
        old.active, old.departed = False, self.tick
        new = Resident(spec["id"], spec["name"], joined=self.tick)
        self.residents[new.id] = new
        self.read_versions[new.id] = set()
        self._emit("membership", departed=old.id, arrived=new.id, name=new.name,
                   text=f"{old.name} has left the cooperative. {new.name} has joined.",
                   visible_to=self._nearby("Quad"))

    def submit(self, intentions: dict[str, dict]):
        """Resolve a batch selected from the same pre-action observations."""
        order = sorted(intentions, key=lambda aid: (keyed_number(self.seed, "priority", self.tick, aid), aid))
        for aid in order:
            if aid not in self.residents or not self.residents[aid].active:
                raise ValueError(f"Unknown or inactive actor: {aid}")
            if aid in self.operations:
                raise ValueError(f"Actor is already occupied: {aid}")
            if aid in self._submitted:
                raise ValueError(f"Actor already chose an intention this tick: {aid}")
            self._submitted.add(aid)
            try:
                self._start(aid, intentions[aid])
            except InvalidAction as exc:
                self._emit("action_rejected", actor=aid, intention=copy.deepcopy(intentions[aid]),
                           text=str(exc), visible_to=[aid])

    def _text(self, value, label, limit):
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise InvalidAction(f"{label} must contain 1 to {limit} characters.")
        return value.strip()

    def _kit(self, aid: str, intention: dict) -> Kit:
        kid = intention.get("kit")
        if not isinstance(kid, str) or kid not in self.kits:
            raise InvalidAction("Choose a visible kit id.")
        k, r = self.kits[kid], self.residents[aid]
        if k.location != r.location or k.holder not in (None, aid) or k.delivered:
            raise InvalidAction("That kit is not available to you here.")
        return k

    def _record(self, aid: str, d: dict) -> tuple[Record, int]:
        if not self.records_enabled or self.residents[aid].location != "Library":
            raise InvalidAction("Shared records are accessible only at an available Library archive.")
        rid, version = d.get("record"), d.get("version")
        if not isinstance(rid, str) or rid not in self.records:
            raise InvalidAction("Choose a record from the catalog.")
        rec = self.records[rid]
        if type(version) is not int or not 1 <= version <= len(rec.versions):
            raise InvalidAction("Specify an existing numeric version from the catalog.")
        return rec, version

    def _start(self, aid: str, raw: dict):
        if not isinstance(raw, dict) or not isinstance(raw.get("action"), str):
            raise InvalidAction("Supply an action object with an action name.")
        d = copy.deepcopy(raw)
        action = d["action"] = d["action"].upper()
        if action not in ACTIONS:
            raise InvalidAction("That action is not supported.")
        r = self.residents[aid]
        locks, costs, duration = [], {}, 1
        if action == "MOVE":
            destination = d.get("destination")
            if not isinstance(destination, str) or destination not in WORLD_GRAPH or destination == r.location:
                raise InvalidAction("Choose a different campus location.")
            d["path"] = shortest_path(r.location, destination)
            duration = len(d["path"]) - 1
        elif action in {"TAKE", "DROP", "ASSEMBLE", "CALIBRATE", "TEST", "DELIVER"}:
            k = self._kit(aid, d)
            locks.append("kit:" + k.id)
            if action == "TAKE":
                if r.carrying or k.holder:
                    raise InvalidAction("You can carry one unclaimed kit at a time.")
            elif action == "DROP":
                if r.carrying != k.id:
                    raise InvalidAction("You must be carrying that kit to set it down.")
            elif action == "ASSEMBLE":
                if r.location != "Research Lab" or k.assembled:
                    raise InvalidAction("Assemble an unfinished kit at the Research Lab.")
                locks.append("station:assembly")
                costs, duration = {"components": 2}, 2
            elif action == "CALIBRATE":
                if r.location != "Research Lab" or not k.assembled or d.get("setting") not in ("A", "B"):
                    raise InvalidAction("Calibrate an assembled kit at the Research Lab using setting A or B.")
                locks.append("station:assembly")
                costs = {"components": 1}
            elif action == "TEST":
                test = d.get("test")
                if not k.assembled or k.calibration is None or test not in ("bench", "field"):
                    raise InvalidAction("Test a calibrated kit, specifying bench or field.")
                site = "Research Lab" if test == "bench" else self.projects[k.project].site
                if r.location != site:
                    raise InvalidAction(f"This test must be performed at {site}.")
                if test == "bench":
                    locks.append("station:testing")
                costs, duration = {"test_supplies": 1}, 2
                # Field consumables travel with kits; their stock is an explicit shared supply.
            elif action == "DELIVER":
                if not k.assembled or k.calibration is None or r.location != self.projects[k.project].site:
                    raise InvalidAction("Deliver a calibrated kit at its requested operating site.")
        elif action == "SAY":
            d["text"] = self._text(d.get("text"), "Speech", 1200)
            target = d.get("target")
            nearby = [other for other in self._nearby(r.location) if other != aid]
            if target is not None and (not isinstance(target, str) or target not in nearby):
                raise InvalidAction("Address someone present, or omit target to speak locally.")
            d["listeners"] = [target] if target else nearby
        elif action == "WRITE":
            if not self.records_enabled or r.location != "Library":
                raise InvalidAction("Writing shared records requires an available Library archive.")
            d["title"] = self._text(d.get("title"), "Title", 100)
            d["text"] = self._text(d.get("text"), "Record", 2400)
        elif action in {"READ", "REVISE"}:
            rec, version = self._record(aid, d)
            if action == "REVISE":
                if (rec.id, version) not in self.read_versions[aid]:
                    raise InvalidAction("Read the version you intend to revise first.")
                if version != len(rec.versions):
                    raise InvalidAction("The record has changed; read its current version before revising.")
                d["title"] = self._text(d.get("title"), "Title", 100)
                d["text"] = self._text(d.get("text"), "Record", 2400)
                locks.append("record:" + rec.id)
        for resource in locks:
            if resource in self.locks:
                raise InvalidAction(f"{resource} is currently in use; try another activity or wait.")
        for resource, amount in costs.items():
            if self.supplies[resource] < amount:
                raise InvalidAction(f"Insufficient {resource}; supplies replenish at the next working day.")
        for resource, amount in costs.items():
            self.supplies[resource] -= amount
            self.consumed[resource] += amount
        event = self._emit("action_started", actor=aid, intention=d, duration=duration, costs=costs,
                           location=r.location)
        op = Operation(event["event_key"], aid, d, self.tick, self.tick + duration, locks)
        self.operations[aid] = op
        for resource in locks:
            self.locks[resource] = op.id

    def _release(self, op: Operation):
        for resource in op.locks:
            if self.locks.get(resource) == op.id:
                self.locks.pop(resource)
        self.operations.pop(op.actor, None)

    def _error(self, kit: Kit, site: str) -> float:
        required = self._bench_mode
        if site == "Quad" and self.outdoor_condition == "humid":
            required = "B" if required == "A" else "A"
        return 0.2 if kit.assembled and kit.calibration == required else 4.0

    def _complete(self, op: Operation):
        d, aid = op.intention, op.actor
        action, r = d["action"], self.residents[aid]
        kid = d.get("kit")
        k = self.kits.get(kid) if isinstance(kid, str) else None
        text = ""
        if action == "MOVE":
            r.location = d["destination"]
            if r.carrying:
                self.kits[r.carrying].location = r.location
            text = f"You arrived at {r.location}."
        elif action == "TAKE":
            r.carrying, k.holder = k.id, aid
            text = f"You picked up {k.id}."
        elif action == "DROP":
            r.carrying, k.holder = None, None
            text = f"You set down {k.id}."
        elif action == "ASSEMBLE":
            k.assembled = True
            text = f"You assembled {k.id}. It still needs calibration."
        elif action == "CALIBRATE":
            k.calibration = d["setting"]
            text = f"You calibrated {k.id} to setting {k.calibration}."
        elif action == "TEST":
            site = "Research Lab" if d["test"] == "bench" else self.projects[k.project].site
            error = self._error(k, site)
            text = (f"Measurement: {k.id}, {d['test']} test at {site}, setting {k.calibration}, "
                    f"error {error} units; tolerance 1 unit.")
            self._emit("test_result", actor=aid, operation=op.id, kit=k.id, test=d["test"], site=site,
                       setting=k.calibration, error=error, passed=error <= 1,
                       text=text, visible_to=[aid])
            text = ""
        elif action == "DELIVER":
            project = self.projects[k.project]
            error = self._error(k, project.site)
            passed = error <= 1
            project.attempts += 1
            if passed:
                project.completed, k.delivered, k.holder = self.tick, True, None
                if r.carrying == k.id:
                    r.carrying = None
            text = (f"Delivery: {k.id} at {project.site}, setting {k.calibration}, error {error} units. "
                    + ("The request is complete." if passed else "The request remains open for rework."))
            self._emit("delivery", actor=aid, operation=op.id, project=project.id, kit=k.id,
                       site=project.site, passed=passed, error=error, setting=k.calibration,
                       released=project.released, due=project.due, text=text,
                       visible_to=self._nearby(r.location))
            text = ""
        elif action == "SAY":
            # The utterance was spoken during the reserved interval; audience is pinned at its start.
            self._emit("speech", actor=aid, operation=op.id, text=d["text"],
                       visible_to=d["listeners"], location=r.location)
        elif action in {"WRITE", "REVISE"}:
            if action == "WRITE":
                self._record_counter += 1
                rec = Record(f"record-{self._record_counter:04d}")
                self.records[rec.id] = rec
            else:
                rec = self.records[d["record"]]
            version = {"version": len(rec.versions) + 1, "author": aid, "tick": self.tick,
                       "title": d["title"], "text": d["text"],
                       "parent_version": d.get("version") if action == "REVISE" else None}
            rec.versions.append(version)
            self._emit("record_written", actor=aid, operation=op.id, record=rec.id, version=version)
            text = f"You saved {rec.id} version {version['version']}: {d['title']}."
        elif action == "READ":
            rec, version = self.records[d["record"]], d["version"]
            v = rec.versions[version - 1]
            self.read_versions[aid].add((rec.id, version))
            text = (f"Read {rec.id} version {version}, by {self.residents[v['author']].name}, "
                    f"written at tick {v['tick']}. {v['title']}: {v['text']}")
            self._emit("record_read", actor=aid, operation=op.id, record=rec.id, version=version,
                       author=v["author"], text=text, visible_to=[aid])
            text = ""
        self._emit("action_completed", actor=aid, operation=op.id, action=action,
                   intention=d, location=r.location, text=text, visible_to=[aid] if text else [])

    def view(self, aid: str) -> dict:
        """Allowlisted local observations. Never derive this by stripping snapshot()."""
        r = self.residents[aid]
        local = r.location
        records = []
        if self.records_enabled and local == "Library":
            for rec in self.records.values():
                v = rec.versions[-1]
                records.append({"id": rec.id, "title": v["title"], "version": v["version"],
                                "author": self.residents[v["author"]].name})
        projects = [asdict(p) for p in self.projects.values() if p.completed is None]
        return {
            "tick": self.tick, "id": aid, "location": local, "carrying": r.carrying,
            "destinations": list(WORLD_GRAPH),
            "people": [{"id": other, "name": self.residents[other].name,
                        "activity": self.operations[other].intention["action"]
                        if other in self.operations else "available"}
                       for other in self._nearby(local) if other != aid],
            "kits": [asdict(k) for k in self.kits.values() if k.location == local and not k.delivered
                     and not (k.holder and self._traveling(k.holder))],
            "requests": projects if local in ("Research Lab", "Library", "Quad") else [],
            "supplies": dict(self.supplies) if local == "Research Lab" else None,
            "catalog": records, "archive_available": self.records_enabled,
            "conditions": (self.outdoor_condition if local == "Quad" else "indoors"),
            "stations": {s.split(":", 1)[1]: self.locks.get(s) is not None
                         for s in ("station:assembly", "station:testing")} if local == "Research Lab" else {},
        }

    def snapshot(self) -> dict:
        """Observer-only state. Records include all versions; no prompt uses this."""
        return {"schema_version": self.schema_version, "tick": self.tick,
                "residents": {k: asdict(v) for k, v in self.residents.items()},
                "kits": {k: asdict(v) for k, v in self.kits.items()},
                "projects": {k: asdict(v) for k, v in self.projects.items()},
                "records": {k: asdict(v) for k, v in self.records.items()},
                "operations": {k: asdict(v) for k, v in self.operations.items()},
                "supplies": dict(self.supplies), "consumed": dict(self.consumed),
                "outdoor_condition": self.outdoor_condition,
                "records_enabled": self.records_enabled,
                "research": {"bench_mode": self._bench_mode, "change_tick": self.change_tick,
                             "turnover_tick": self.turnover_tick, "change_enabled": self.change_enabled}}
