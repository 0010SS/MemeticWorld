"""Referents (ontology v2 §1.4): persistent, FAMILY-NEUTRAL world things that event instances happen
around and that people can keep referring to ("the night shuttle", "the lounge oven").

Categories are topic DOMAINS. Every family has exactly one train skin per train domain and one
holdout skin per holdout domain, so domain (topic) carries no information about family, and every
referent can occur in instances of every family. A skin's {R} slot takes a name from its domain's
pool; names are written to fit any skin of that domain (subject or object position).
"""
from __future__ import annotations

TRAIN_DOMAINS = ["dining", "coursework", "lab", "transit", "gym", "dorm", "cafe", "clubs"]
HOLDOUT_DOMAINS = ["library", "quad", "tech"]

CATEGORIES: dict[str, list[str]] = {
    "dining": ["the dining hall card reader", "the late-night grill station", "the salad bar",
               "the drink fountain", "the tray return belt"],
    "coursework": ["the course portal", "the homework dropbox", "the online quiz system",
                   "the syllabus page", "the class discussion board"],
    "lab": ["the minus-80 freezer", "the old centrifuge", "the fume hood by the window",
            "the shared lab scale", "the plate reader"],
    "transit": ["the Blue Line shuttle", "the shuttle tracker app", "the Charles Street shuttle stop",
                "the night shuttle", "the bike rack by the gate"],
    "gym": ["the rec center entry scanner", "the upstairs basketball court", "the climbing wall",
            "the pool locker room", "the squat rack by the mirrors"],
    "dorm": ["the laundry room dryers", "the lounge oven", "the dorm mailroom",
             "the fourth-floor elevator", "the hallway fire door"],
    "cafe": ["the cafe espresso machine", "the mobile order screen", "the pastry case",
             "the cafe tip jar", "the oat milk fridge"],
    "clubs": ["the club mailing list", "the room booking system", "the events bulletin board",
              "the club group chat", "the student activities office"],
    # held-out domains (only used by holdout skins)
    "library": ["the third-floor study room", "the library self-checkout", "the quiet floor printer",
                "the reserve desk", "the study room booking tablet"],
    "quad": ["the quad sprinklers", "the lawn chess table", "the fountain by the steps",
             "the quad picnic benches", "the outdoor speaker pole"],
    "tech": ["the second-floor printer", "the campus wifi", "the lecture hall projector",
             "the charging lockers", "the vending machine by the stairs"],
}

assert set(CATEGORIES) == set(TRAIN_DOMAINS) | set(HOLDOUT_DOMAINS)
