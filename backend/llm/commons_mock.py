"""Hand-written smoke policy, using only the same prompt available to an LLM.

This exercises software paths; it is not evidence for any cultural hypothesis.
There is no access to world state, treatment schedules, or correct calibration.
"""
from __future__ import annotations

import hashlib
import json
import re


MEASUREMENT = re.compile(
    r"Measurement: (kit-[\d-]+), (bench|field) test at (Library|Quad|Research Lab), "
    r"setting ([AB]), error ([\d.]+) units")


def respond(prompt: str) -> str:
    context = json.loads(prompt.split("LOCAL_CONTEXT_JSON\n", 1)[1])
    v = context["view"]
    text = "\n".join(context["memories"] + context["observations"])
    measurements = list(MEASUREMENT.finditer(text))
    salt = int(hashlib.sha256(v["id"].encode()).hexdigest()[:8], 16)

    def action(name, **fields):
        return json.dumps({"action": name, **fields, "reason": "offline software exercise"})

    own = next((k for k in v["kits"] if k["id"] == v["carrying"]), None)
    if own:
        kid = own["id"]
        request = next(p for p in v["requests"] if p["id"] == own["project"])
        if not own["assembled"]:
            if v["location"] != "Research Lab":
                return action("MOVE", destination="Research Lab")
            return action("WAIT") if v["stations"]["assembly"] else action("ASSEMBLE", kit=kid)
        field = [m for m in measurements if m[1] == kid and m[2] == "field" and m[4] == own["calibration"]]
        failed = field and float(field[-1][5]) > 1
        if own["calibration"] is None or failed:
            if v["location"] != "Research Lab":
                return action("MOVE", destination="Research Lab")
            if v["stations"]["assembly"]:
                return action("WAIT")
            if failed:
                setting = "B" if own["calibration"] == "A" else "A"
            else:
                useful = [m for m in measurements if m[2] == "field" and m[3] == request["site"] and float(m[5]) <= 1]
                setting = useful[-1][4] if useful else "AB"[salt % 2]
            return action("CALIBRATE", kit=kid, setting=setting)
        if v["location"] != request["site"]:
            return action("MOVE", destination=request["site"])
        if field and float(field[-1][5]) <= 1:
            return action("DELIVER", kit=kid)
        return action("TEST", kit=kid, test="field")

    if v["archive_available"] and v["location"] == "Library":
        titles = {r["title"] for r in v["catalog"]}
        for m in reversed(measurements):
            title = "Observation " + m[1] + " " + m[4]
            if title not in titles:
                return action("WRITE", title=title, text=m[0] + "; tolerance 1 unit. Check the operating site.")
        for r in reversed(v["catalog"]):
            if f"Read {r['id']} version {r['version']}" not in text:
                return action("READ", record=r["id"], version=r["version"])
    if v["people"] and measurements and v["tick"] % 11 == salt % 11:
        return action("SAY", text=measurements[-1][0] + "; tolerance 1 unit.")
    if v["location"] != "Research Lab":
        return action("MOVE", destination="Research Lab")
    available = [k for k in v["kits"] if k["holder"] is None]
    if available:
        return action("TAKE", kit=available[salt % len(available)]["id"])
    if v["archive_available"]:
        return action("MOVE", destination="Library")
    return action("WAIT")
