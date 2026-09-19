"""Hidden latent event families and their surface generators.

SIMULATOR-ONLY. The family ids (E1..E4), their descriptions and the scenario
keys never reach an agent: agents only receive individual *facts* (concrete
surface sentences) through partial perception (backend/agents/perception.py).

A scenario is a small script of beats. Each beat happens `offset` ticks after
the instance starts, at a location ("current" = wherever the protagonist is,
"current:Q" = wherever role Q is), and contains facts:
    (text, salience, visibility, kind)
visibility: "all" (anyone present may notice), "P"/"S"/"PS" (only those roles).
Roles: P protagonist (always an agent); S a related person (agent if one is
available, else an NPC); Q an unrelated person (agent with low familiarity to
P, else an NPC).

`holdout=True` scenarios are never used before `latent_events.holdout_from_day`;
they provide surface-novel instances for the generalization test.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field

LATENT_TYPES = {
    "E1": {"name": "cascade_failure", "description": "a small initial mistake creates a chain of otherwise unrelated failures"},
    "E2": {"name": "cancelling_mistakes", "description": "two independent mistakes accidentally cancel out and nothing goes wrong"},
    "E3": {"name": "independent_coincidence", "description": "unrelated people independently do the same unusual thing"},
    "E4": {"name": "beneficial_failure", "description": "an apparent failure causes a beneficial outcome"},
}

NPC_S = ["a classmate", "a friend from another dorm", "a TA", "a lab technician"]
NPC_Q = "a student nobody seemed to know"


def F(text, sal=0.5, vis="all", kind="action"):
    return {"text": text, "salience": sal, "vis": vis, "kind": kind}


def B(offset, location, *facts, arena=None):
    return {"offset": offset, "location": location, "arena": arena, "facts": list(facts)}


SCENARIOS: list[dict] = [
    # ------------------------------------------------------------------ E1 --
    {"key": "room_mixup", "family": "E1", "holdout": False,
     "slots": {"X": ["linear algebra", "organic chemistry", "microeconomics", "physics", "statistics"]},
     "beats": [
         B(0, "current", F("{S} forwarded {P} an old message with last week's room number for the {X} review session.", .35, "PS", "cause"),
           F("{P} rushed off to the wrong building for the {X} review session.", .5)),
         B(1, "Classroom", F("{P} burst into the Seminar Room looking for the {X} review session, but a different class was meeting there.", .65), arena="Seminar Room"),
         B(2, "Dining Hall", F("{P} got to the dining hall just as it closed and left without eating.", .5)),
         B(4, "Library", F("{P} left a laptop charger at the library and had to go back for it, missing a meeting.", .5),
           F("{P} had not eaten since the wasted trip to the review session and could not concentrate.", .3, "P", "cause")),
     ]},
    {"key": "backpack_swap", "family": "E1", "holdout": False,
     "slots": {"X": ["lab notebook", "calculus textbook", "gym clothes", "sketchbook", "bag of oranges"]},
     "beats": [
         B(0, "current", F("{P} grabbed a backpack that looked exactly like theirs and walked off with it.", .55)),
         B(1, "Library", F("{P} opened the backpack at the library and found someone else's {X} instead of their own notes.", .6)),
         B(2, "Cafe", F("While {P} was trying to return the backpack, {S} waited at the cafe for forty minutes for {P}.", .55)),
         B(3, "Research Lab", F("{P} missed a scheduled meeting in the lab.", .5),
           F("{P} had spent the afternoon returning the other backpack and forgot about the lab meeting.", .3, "P", "cause")),
     ]},
    {"key": "alarm_pm", "family": "E1", "holdout": False, "slots": {},
     "beats": [
         B(0, "current", F("{P} realized their alarm had been set for PM instead of AM.", .4, "P", "cause")),
         B(1, "Classroom", F("{P} ran into the Lecture Hall halfway through, out of breath, and missed the pop quiz.", .65), arena="Lecture Hall"),
         B(2, "Cafe", F("{P} hurried through the cafe and spilled a full coffee on {S}'s notes.", .6)),
         B(3, "Library", F("{P} spent the afternoon at the library recopying {S}'s ruined notes instead of doing their own work.", .5)),
     ]},
    {"key": "reply_all", "family": "E1", "holdout": False,
     "slots": {"X": ["a lost water bottle", "a group project", "a sublet listing", "a bake sale"]},
     "beats": [
         B(0, "current", F("{P} accidentally sent a message about {X} to the entire department mailing list.", .55)),
         B(1, "current", F("{P}'s phone kept buzzing with reply-all messages about the misdirected email.", .45)),
         B(2, "Library", F("{P} submitted a problem set to the wrong course.", .55),
           F("{P} had been answering replies about the misdirected message all morning and picked the wrong course by accident.", .3, "P", "cause")),
         B(4, "Dining Hall", F("{S} told {P} that the grader had marked {P}'s problem set as missing.", .5, "PS")),
     ]},
    {"key": "id_card", "family": "E1", "holdout": False, "slots": {},
     "beats": [
         B(0, "Research Lab", F("{P} couldn't swipe into the Research Lab because their ID card was in another jacket.", .55), arena="Wet Lab"),
         B(1, "Research Lab", F("Because {P} was stuck outside, the samples {P} was supposed to move to the freezer sat out too long.", .6), arena="Wet Lab"),
         B(2, "Research Lab", F("{S} had to redo the morning's prep because of the ruined samples.", .55), arena="Wet Lab"),
         B(3, "Dining Hall", F("{P} skipped dinner to help redo the prep and was grumpy all evening.", .45)),
     ]},
    {"key": "calendar_date", "family": "E1", "holdout": False,
     "slots": {"X": ["intramural signup", "club photo", "fitness assessment"]},
     "beats": [
         B(0, "current", F("{P} had typed the wrong date into their calendar for the {X}.", .3, "P", "cause")),
         B(1, "Gym", F("{P} showed up at the gym for the {X} a day early and found the room empty.", .55)),
         B(2, "Quad", F("Walking back across the quad, {P} stepped into the sprinklers and soaked their shoes.", .5)),
         B(3, "Classroom", F("{P} sat through the next class in wet socks and couldn't focus on anything.", .45)),
     ]},
    {"key": "wrong_shuttle", "family": "E1", "holdout": False, "slots": {},
     "beats": [
         B(0, "current", F("{P} misread the shuttle schedule and caught one going the wrong way.", .45)),
         B(2, "Quad", F("{P} came back to campus an hour late, looking frazzled.", .5)),
         B(3, "Dining Hall", F("{P} arrived after the hot food ran out and ended up eating dry cereal.", .45)),
         B(4, "Library", F("With no time left, {P} rushed an assignment at the library and forgot to attach the file.", .5)),
     ]},
    # E1 held-out surfaces
    {"key": "incubator_timer", "family": "E1", "holdout": True, "slots": {},
     "beats": [
         B(0, "Research Lab", F("{P} set the lab incubator timer to 12 minutes instead of 12 hours.", .5), arena="Wet Lab"),
         B(1, "Research Lab", F("The cultures came out unusable, and {S} had to throw away a week of work.", .65), arena="Wet Lab"),
         B(2, "Cafe", F("{P} bought {S} a coffee at the cafe to apologize but grabbed the wrong order.", .5)),
         B(3, "Quad", F("{S} fell asleep during the afternoon seminar.", .5),
           F("The coffee {P} had bought {S} turned out to be decaf.", .3, "PS", "cause")),
     ]},
    {"key": "locker_combo", "family": "E1", "holdout": True, "slots": {},
     "beats": [
         B(0, "Gym", F("{P} had written the wrong locker combination on a sticky note and couldn't open the gym locker.", .55)),
         B(1, "Classroom", F("Still in workout clothes, {P} wasn't allowed into the lab section without closed shoes.", .55)),
         B(2, "Library", F("{P} had to book a make-up lab slot, which landed right on {S}'s birthday dinner.", .5)),
     ]},
    {"key": "grocery_upload", "family": "E1", "holdout": True, "slots": {},
     "beats": [
         B(0, "current", F("{P} uploaded a grocery list instead of an essay to the course portal.", .5)),
         B(2, "Classroom", F("The professor showed {P}'s grocery list to the class as an example of a formatting problem, and everyone laughed.", .65)),
         B(3, "Dining Hall", F("{P} was too embarrassed to sit with the usual group at dinner and ate alone.", .45)),
         B(4, "Library", F("{P} missed the study group session, which had moved to a new time.", .5),
           F("The new study group time had been announced at the dinner table {P} skipped.", .3, "P", "cause")),
     ]},
    # ------------------------------------------------------------------ E2 --
    {"key": "quiz_time", "family": "E2", "holdout": False, "slots": {"X": ["statistics", "chemistry", "economics"]},
     "beats": [
         B(0, "Classroom", F("{P} walked into the Lecture Hall forty minutes late for the {X} quiz.", .55),
           F("The instructor had posted the wrong start time, so the {X} quiz was only just starting when {P} walked in.", .55), arena="Lecture Hall"),
         B(1, "Classroom", F("{P} took the quiz as if nothing had gone wrong.", .35), arena="Lecture Hall"),
     ]},
    {"key": "drink_swap", "family": "E2", "holdout": False, "slots": {"X": ["an iced latte", "a chai", "a hot chocolate"]},
     "beats": [
         B(0, "Cafe", F("{P} ordered the wrong drink by mistake.", .45), F("{S} was handed the wrong drink too, which turned out to be {X}.", .45),
           F("{P} and {S} swapped cups and each ended up with exactly what they had wanted.", .55)),
     ]},
    {"key": "old_prompt", "family": "E2", "holdout": False, "slots": {},
     "beats": [
         B(0, "Classroom", F("{P} printed an essay written for last year's prompt by mistake.", .5),
           F("The professor had accidentally assigned last year's prompt as well.", .5), arena="Seminar Room"),
         B(1, "Classroom", F("{P}'s essay was accepted without a single comment.", .4), arena="Seminar Room"),
     ]},
    {"key": "key_swap", "family": "E2", "holdout": False, "slots": {},
     "beats": [
         B(0, "Research Lab", F("{P} grabbed {S}'s lab key instead of their own.", .45),
           F("The lab's lock had been changed overnight and only {S}'s key had been updated.", .5),
           F("{P} walked right into the lab while everyone else was locked out.", .55), arena="Dry Lab"),
     ]},
    {"key": "yoga_room", "family": "E2", "holdout": False, "slots": {},
     "beats": [
         B(0, "Gym", F("{P} went to the wrong room at the gym for the yoga class.", .45),
           F("The yoga class had been moved to that very room that morning.", .45),
           F("{P} got the best spot in the class.", .4)),
     ]},
    {"key": "double_booking", "family": "E2", "holdout": False, "slots": {},
     "beats": [
         B(0, "Library", F("{P} accidentally booked the group study room for the wrong day.", .45),
           F("{S} had also booked it for the wrong day -- the very same day.", .5),
           F("{P} and {S} ended up studying together and got a lot done.", .5)),
     ]},
    {"key": "meal_card", "family": "E2", "holdout": False, "slots": {},
     "beats": [
         B(0, "Dining Hall", F("{P} forgot their meal card at home.", .4),
           F("The card reader at the dining hall was broken, so lunch was free for everyone.", .55),
           F("{P} ate lunch without anyone ever asking for a card.", .4)),
     ]},
    # E2 held-out surfaces
    {"key": "salt_cookies", "family": "E2", "holdout": True, "slots": {},
     "beats": [
         B(0, "Quad", F("{P} had doubled the salt in a batch of cookies for the bake sale.", .5),
           F("{S} had left the salt out of their batch completely.", .5),
           F("Mixed together on one tray, the two batches tasted perfect and sold out.", .55)),
     ]},
    {"key": "flyer_room", "family": "E2", "holdout": True, "slots": {},
     "beats": [
         B(0, "Classroom", F("{P} wrote the wrong room number on the club flyer.", .45),
           F("The building's room signs had been swapped during renovations.", .45),
           F("Everyone who followed {P}'s flyer ended up in exactly the right room.", .55), arena="Seminar Room"),
     ]},
    {"key": "portal_link", "family": "E2", "holdout": True, "slots": {},
     "beats": [
         B(0, "Library", F("{P} submitted homework to the wrong course page.", .45),
           F("The TA had linked that same wrong page as the official submission site.", .45),
           F("{P}'s homework was graded on time like everyone else's.", .4)),
     ]},
    # ------------------------------------------------------------------ E3 --
    {"key": "same_odd_act", "family": "E3", "holdout": False,
     "slots": {"X": ["carrying a small potted cactus around", "wearing a bike helmet indoors",
                     "eating cereal with orange juice", "wearing one red shoe and one green shoe",
                     "taking notes on paper napkins", "wearing sunglasses indoors all day",
                     "talking to a rubber duck while working", "doing lunges between every errand",
                     "drinking tea out of a mason jar", "wearing a paper crown"],
               "R1": ["just for fun", "a dare from a friend", "for good luck before an exam", "part of a bet"],
               "R2": ["a new habit experiment", "for no particular reason", "something their roommate suggested", "to stay awake"]},
     "beats": [
         B(0, "current", F("{P} was {X}.", .65), F("{P} said it was {R1}.", .35)),
         B(3, "current:Q", F("{Q} was {X}.", .65), F("{Q} said it was {R2}.", .35)),
     ]},
    {"key": "same_odd_act_three", "family": "E3", "holdout": False,
     "slots": {"X": ["humming the same old cartoon theme song over and over", "carrying an umbrella on a sunny day",
                     "eating a raw onion like an apple", "wearing a winter scarf in warm weather",
                     "counting their steps out loud"],
               "R1": ["just for fun", "a dare", "for luck"], "R2": ["for no reason", "a bet", "a habit"]},
     "beats": [
         B(0, "current", F("{P} was {X}.", .6), F("{P} said it was {R1}.", .3)),
         B(2, "current:Q", F("{Q} was {X}.", .6), F("{Q} said it was {R2}.", .3)),
         B(4, "Quad", F("A student nobody seemed to know was also {X} in the middle of the quad.", .55)),
     ]},
    # E3 held-out surfaces
    {"key": "same_odd_act_holdout", "family": "E3", "holdout": True,
     "slots": {"X": ["carrying a whole watermelon around", "wearing a superhero cape",
                     "drawing tiny dinosaurs on their hand", "eating soup with chopsticks",
                     "walking backwards up the stairs"],
               "R1": ["just for fun", "a dare"], "R2": ["for no reason at all", "a bet"]},
     "beats": [
         B(0, "current", F("{P} was {X}.", .65), F("{P} said it was {R1}.", .35)),
         B(3, "current:Q", F("{Q} was {X}.", .65), F("{Q} said it was {R2}.", .35)),
     ]},
    # ------------------------------------------------------------------ E4 --
    {"key": "failed_run", "family": "E4", "holdout": False, "slots": {},
     "beats": [
         B(0, "Research Lab", F("{P}'s experiment in the lab failed completely.", .6), arena="Wet Lab"),
         B(1, "Research Lab", F("While cleaning up the failed run, {P} noticed a strange pattern that {S} called the most interesting result of the month.", .6), arena="Wet Lab"),
     ]},
    {"key": "missed_shuttle", "family": "E4", "holdout": False, "slots": {},
     "beats": [
         B(0, "current", F("{P} missed the shuttle and had to walk.", .45)),
         B(1, "Quad", F("On the walk, {P} ran into a professor who offered {P} a spot on a research project.", .6)),
     ]},
    {"key": "locked_out_notes", "family": "E4", "holdout": False, "slots": {},
     "beats": [
         B(0, "Dorm", F("{P} got locked out of their room.", .5), arena="Lounge"),
         B(1, "Dorm", F("While waiting in the lounge, {P} met {S}, who had exactly the notes {P} needed for the exam.", .55), arena="Lounge"),
     ]},
    {"key": "coffee_notes", "family": "E4", "holdout": False, "slots": {},
     "beats": [
         B(0, "Cafe", F("{P} spilled coffee all over the only copy of their notes.", .55)),
         B(2, "Library", F("{P} rewrote the notes from memory and suddenly understood the material.", .45, "P")),
         B(4, "Classroom", F("{P} got the highest score in the class on the next quiz.", .55)),
     ]},
    {"key": "dead_laptop", "family": "E4", "holdout": False, "slots": {},
     "beats": [
         B(0, "Library", F("{P}'s laptop died in the middle of a study session.", .5)),
         B(2, "Library", F("Without the laptop, {P} finished all of the reading in record time.", .45)),
     ]},
    {"key": "cut_from_game", "family": "E4", "holdout": False, "slots": {},
     "beats": [
         B(0, "Gym", F("{P} was left out of the pickup game at the gym.", .5)),
         B(1, "Quad", F("{P} joined a chess game on the quad instead and won a small tournament prize.", .55)),
     ]},
    {"key": "wrong_dish", "family": "E4", "holdout": False, "slots": {"X": ["a spicy tofu bowl", "a lentil curry", "a beet salad"]},
     "beats": [
         B(0, "Dining Hall", F("The dining hall got {P}'s order wrong and gave them {X}.", .45),
           F("{P} tried {X} anyway and declared it the best thing on the menu.", .5)),
         B(2, "Dining Hall", F("{S} ordered {X} after hearing {P} rave about it.", .4)),
     ]},
    # E4 held-out surfaces
    {"key": "burnt_cookies", "family": "E4", "holdout": True, "slots": {},
     "beats": [
         B(0, "Dorm", F("{P} burned a batch of cookies in the lounge oven.", .5), arena="Lounge"),
         B(2, "Quad", F("{P}'s burned cookies sold out first at the bake sale.", .55)),
     ]},
    {"key": "flat_tire", "family": "E4", "holdout": True, "slots": {},
     "beats": [
         B(0, "current", F("{P}'s bike got a flat tire.", .45)),
         B(1, "Quad", F("Walking the bike across the quad, {P} found a lost wallet; its owner, a TA, offered {P} extra help before the exam.", .6)),
     ]},
    {"key": "deleted_draft", "family": "E4", "holdout": True, "slots": {},
     "beats": [
         B(0, "Library", F("{P} accidentally deleted the draft of a paper.", .55)),
         B(3, "Library", F("{P} rewrote the paper from scratch, and the new version was much better.", .45)),
         B(5, "Classroom", F("The professor praised {P}'s paper in front of the whole class.", .55)),
     ]},
]


@dataclass
class EventInstance:
    id: str
    latent_type: str
    scenario: str
    holdout: bool
    start_tick: int
    roles: dict                    # role -> {"agent": id|None, "name": display name}
    beats: list                    # concrete beats (absolute tick, location, arena, facts)
    generator: str = "template"
    fired: set = field(default_factory=set)

    def ground_truth(self) -> dict:
        return {"id": self.id, "latent_type": self.latent_type, "scenario": self.scenario,
                "holdout": self.holdout, "start_tick": self.start_tick, "roles": self.roles,
                "beats": self.beats, "generator": self.generator,
                "narrative": " ".join(f["text"] for b in self.beats for f in b["facts"])}


def _fill(text: str, subst: dict) -> str:
    out = text
    for k, v in subst.items():
        out = out.replace("{" + k + "}", v)
    return out[0].upper() + out[1:] if out else out


def scenarios_for(family: str, holdout: bool) -> list[dict]:
    return [s for s in SCENARIOS if s["family"] == family and s["holdout"] == holdout]


def instantiate(scn: dict, event_id: str, tick: int, rng, agents: dict, busy: set, tpd: int) -> EventInstance | None:
    """Bind roles to concrete agents / NPCs and resolve beat ticks.

    `agents`: id -> Agent; `busy`: agents already in an active event; `tpd`: ticks per day.
    """
    free = sorted(a for a in agents if a not in busy)
    if not free:
        return None
    day_tick = tick % tpd
    max_off = max(b["offset"] for b in scn["beats"])
    if day_tick + max_off >= tpd - 1:
        return None
    p = free[int(rng.integers(len(free)))]
    prof = agents[p].profile
    roles = {"P": {"agent": p, "name": prof.first_name}}
    text_all = json.dumps(scn["beats"])
    if "{S}" in text_all:
        rel = sorted(o for o in free if o != p and prof.rel(o).familiarity >= 0.3)
        if rel and rng.random() < 0.75:
            s = rel[int(rng.integers(len(rel)))]
            roles["S"] = {"agent": s, "name": agents[s].profile.first_name}
        else:
            roles["S"] = {"agent": None, "name": NPC_S[int(rng.integers(len(NPC_S)))]}
    if "{Q}" in text_all:
        unrel = sorted(o for o in free if o != p and prof.rel(o).familiarity < 0.3)
        if unrel and rng.random() < 0.85:
            q = unrel[int(rng.integers(len(unrel)))]
            roles["Q"] = {"agent": q, "name": agents[q].profile.first_name}
        else:
            roles["Q"] = {"agent": None, "name": NPC_Q}
    subst = {r: v["name"] for r, v in roles.items()}
    for slot, options in (scn.get("slots") or {}).items():
        subst[slot] = options[int(rng.integers(len(options)))]
    beats = []
    for bi, b in enumerate(scn["beats"]):
        facts = []
        for fi, f in enumerate(b["facts"]):
            involves = [roles[r]["agent"] for r in re.findall(r"\{([PSQ])\}", f["text"])
                        if r in roles and roles[r]["agent"]]
            vis = f["vis"]
            if vis != "all":
                vis = [roles[r]["agent"] for r in vis if r in roles and roles[r]["agent"]]
            facts.append({"id": f"{event_id}.b{bi}.f{fi}", "text": _fill(f["text"], subst),
                          "salience": f["salience"], "visibility": vis, "kind": f["kind"],
                          "involves": sorted(set(involves))})
        beats.append({"idx": bi, "tick": tick + b["offset"], "location": b["location"],
                      "arena": b.get("arena"), "facts": facts,
                      "movers": sorted({a for f in facts for a in f["involves"]})})
    return EventInstance(id=event_id, latent_type=scn["family"], scenario=scn["key"],
                         holdout=scn["holdout"], start_tick=tick, roles=roles, beats=beats)


LLM_GEN_PROMPT = """You write short, concrete, everyday incidents for a simulated college campus.
Hidden structure to realize: {desc}.
Here is an example script in JSON:
{example}

Write a NEW script with the same JSON schema that realizes the same hidden structure with a
completely different surface situation (different objects, places and wording). Use only these
locations: Dorm, Dining Hall, Classroom, Library, Research Lab, Gym, Cafe, Quad, or "current".
Keep the placeholders {{P}}, {{S}}, {{Q}} for people. Never state the hidden structure itself.
Output only the JSON object."""


def llm_scenario(family: str, rng, llm) -> dict | None:
    """Optional LLM surface generator (latent_events.generator: llm)."""
    base = scenarios_for(family, False)
    ex = copy.deepcopy(base[int(rng.integers(len(base)))])
    ex.pop("holdout", None)
    prompt = LLM_GEN_PROMPT.format(desc=LATENT_TYPES[family]["description"], example=json.dumps(ex, indent=1))
    from backend.agents.ga_prompts import as_json
    from backend.llm.client import llm_purpose
    with llm_purpose("world_event_generator"):
        data = as_json(llm.complete(prompt, max_tokens=900, temperature=1.0))
    try:
        valid_locs = {"current", "current:Q", "Dorm", "Dining Hall", "Classroom", "Library",
                      "Research Lab", "Gym", "Cafe", "Quad"}
        beats = []
        for b in data["beats"]:
            if b["location"] not in valid_locs:
                return None
            beats.append(B(int(b["offset"]), b["location"],
                           *[F(f["text"], float(f.get("salience", .5)), f.get("vis", "all"), f.get("kind", "action"))
                             for f in b["facts"]]))
        return {"key": f"llm_{family}", "family": family, "holdout": False,
                "slots": data.get("slots") or {}, "beats": beats}
    except Exception:  # noqa: BLE001
        return None
