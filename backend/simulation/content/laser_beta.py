"""Content pack `laser_beta` (ontology v3 §1.2): the pre-registered alternative (gate G1).

Causes AIR (air-assist nozzle clogged), WARP (warped sheet) and BELT. K2 (belt), K0 and the oddities
reuse laser_alpha's surfaces. Hidden ids are keys only and never appear in any text.
"""
from backend.simulation.content import laser_alpha as _A

NAME = "laser_beta"

CAUSES = ("AIR", "WARP", "BELT")
STABLE = "BELT"
FIX = {"AIR": "air", "WARP": "pins", "BELT": "belt"}
MAPPINGS = {"M1": {"A": "AIR", "B": "WARP"}, "M2": {"A": "WARP", "B": "AIR"}}

ACTIONS = {
    "rerun": "re-run the sheet as it is",
    "air": "clear the air-assist nozzle, then re-run",
    "pins": "flatten the sheet with hold-down pins, then re-run",
    "belt": "tighten the drive belt, then re-run",
    "slow": "slow the cutting speed down, then re-run",
    "stop": "stop and leave the job for later",
}
ACTION_PAST = {
    "rerun": "re-ran the sheet as it was",
    "air": "cleared the air-assist nozzle and re-ran the sheet",
    "pins": "flattened the sheet with hold-down pins and re-ran it",
    "belt": "tightened the drive belt and re-ran the sheet",
    "slow": "slowed the cutting speed down and re-ran the sheet",
}

SURFACES = {
    "K1": [
        "In the middle of the sheet the cuts never broke through, and a gray soot rimmed every outline there.",
        "The center pieces stayed attached by thin webs, their borders smudged with dark residue.",
        "Shapes near the middle only half separated, and the kerf there was coated in grimy black.",
        "Around the sheet's midsection the beam failed to finish, leaving sooty streaks beside the lines.",
        "The central parts refused to pop out, and the cut walls were stained a smoky gray.",
        "Only the outer pieces came loose; the inner ones hung on, ringed by charcoal-colored smears.",
        "The beam went shallow across the middle, and dirty brown deposits lined those half-finished cuts.",
        "Most pieces in the middle band clung on, with dusky film along each unfinished edge.",
    ],
    "K2": list(_A.SURFACES["K2"]),
    "K3": {
        "M1": [
            "The sheet rocked on the bed when touched, one corner lifted a finger's width.",
            "A corner of the acrylic curled up during the job, so it teetered under the head.",
            "The blank sat with a visible bow, its ends raised off the honeycomb.",
            "One end of the material arched upward, and it slid slightly when the head passed.",
            "A gap showed beneath the plastic's middle when viewed from the side of the machine.",
            "The acrylic rose in a gentle hump at the center, and pressing it made it spring back.",
            "Its corners lifted off the bed like a curling page, and they tapped against the head.",
            "The stock bowed up at both ends and shifted a little as the cut went along.",
        ],
        "M2": [
            "Small flames flickered along the cut, and smoke lingered over the bed afterward.",
            "Tiny flare-ups followed the beam, and a haze of smoke hung inside the lid.",
            "Orange sparks jumped at each corner, and the window fogged with smoke.",
            "Flames licked up behind the beam more than once, and the space under the lid turned murky.",
            "The cut threw off bright flickers, and a gray cloud stayed trapped under the cover.",
            "Little fires kept catching along the lines, and the smoke took ages to thin out.",
            "Flickering flames trailed the head, and the chamber filled with smoke.",
            "Each pass lit small blazes on the surface, and smoke drifted up in thick curls.",
        ],
    },
    "K0": list(_A.SURFACES["K0"]),
}

ODDITIES = list(_A.ODDITIES)

SUCCESS = "the cut went all the way through this time"
FAIL = {
    "K1": "the middle still didn't cut through",
    "K2": "the lines near the right edge still came out doubled",
    "K3": {"M1": "the sheet still sat unevenly on the bed",
           "M2": "small flames flickered along the cut again"},
}
