"""E3 surface skins: independent coincidence (ontology v2 §1.2). SIMULATOR-ONLY.

Structure: P does an unusual, specific thing (n0); an UNRELATED person {Q}, with no connection to P,
does the very same unusual thing (n1); later P's oddity is noticed or commented on again (n2), still
with no causal link between P and Q. n0_private is P's own reason, which Q never shares.

Writing rules these skins follow:
- n0 and n1 describe the same act in different words (ladle / kitchen ladle, ski / snow goggles),
  so the match is inferable rather than copy-pasted, and no fact says the two are alike.
- n1 and n2 are location-neutral: every beat happens wherever P planned to be
  (Q is brought there for n1).
  n0 is pinned only when its wording needs the domain's building.
- A fact names only the people it involves: {Q} (never {P}) in n1, {P} (never {Q}) in n2.
- Vocabulary is deliberately odd-object specific (quills, canoe paddles, headlamps) so content
  bigrams stay unique to this family; holdout skins share no content bigram with train skins.
"""
from __future__ import annotations

SKINS: list[dict] = [
    # ------------------------------------------------------------------ train --
    {"key": "e3_dining", "family": "E3", "domain": "dining", "holdout": False,
     "slots": {"X": ["cornflakes", "granola", "puffed rice"]},
     "locations": {"n0": "Dining Hall"},
     "facts": {
         "n0": "{P} lingered by {R} eating a heaping bowl of {X} with a hefty steel soup ladle.",
         "n0_private": "{P} had sprained a thumb bouldering and found the ladle's thick handle far easier to grip.",
         "n1": "{Q} was spooning breakfast cereal out of a mug with a long kitchen ladle, perfectly calm about it.",
         "n2": "A sophomore from down the hall pretended to shovel soup with an imaginary ladle, grinning at {P}.",
     }},
    {"key": "e3_coursework", "family": "E3", "domain": "coursework", "holdout": False,
     "facts": {
         "n0": "{P} opened {R} on a laptop perched atop a folded-out ironing board and typed standing up.",
         "n0_private": "A chiropractor had told {P} to avoid chairs for a week after a nasty back spasm.",
         "n1": "{Q} was answering emails at a laptop balanced on an ironing board, treating it like a proper desk.",
         "n2": "A TA stopped {P} to ask, amused, whether the ironing board had helped with the homework.",
     }},
    {"key": "e3_lab", "family": "E3", "domain": "lab", "holdout": False,
     "locations": {"n0": "Research Lab"},
     "facts": {
         "n0": "{P} hovered by {R} with a sprig of spearmint wedged behind one ear.",
         "n0_private": "{P} found the smell of the bacterial broth nauseating and hoped the herb would mask it.",
         "n1": "{Q} went about the day with a little green mint leaf poking out from behind an ear, like a pencil.",
         "n2": "A friend pulled a limp green sprig from behind {P}'s ear and asked, laughing, what it was for.",
     }},
    {"key": "e3_transit", "family": "E3", "domain": "transit", "holdout": False,
     "facts": {
         "n0": "{P} showed up grumbling about {R}, a wooden canoe paddle propped over one shoulder.",
         "n0_private": "{P} had bought the paddle at a yard sale for a kayak trip still weeks away.",
         "n1": "{Q} ambled along with a long canoe paddle resting on one shoulder, nowhere near any water.",
         "n2": "A stranger who had noticed the paddle asked {P}, grinning, where the nearest river was.",
     }},
    {"key": "e3_gym", "family": "E3", "domain": "gym", "holdout": False,
     "slots": {"X": ["orange", "purple", "lime-green"]},
     "locations": {"n0": "Gym"},
     "facts": {
         "n0": "{P} showed up at {R} with {X} ski goggles strapped firmly over both eyes.",
         "n0_private": "{P} had lost a bet with a cousin and had to wear the goggles until midnight.",
         "n1": "{Q} was seen in {X} snow goggles, lenses down, looking dead serious with no snow anywhere.",
         "n2": "Within earshot of {P}, two sophomores argued over why {P} had exercised in ski goggles.",
     }},
    {"key": "e3_dorm", "family": "E3", "domain": "dorm", "holdout": False,
     "slots": {"X": ["red", "neon yellow", "sparkly pink"]},
     "locations": {"n0": "Dorm"},
     "facts": {
         "n0": "{P} paced slowly past {R}, pulling a stuffed toy tortoise along on a {X} leash.",
         "n0_private": "{P}'s niece had mailed the toy tortoise with a crayon note demanding daily walks.",
         "n1": "{Q} trailed a plush turtle behind on a string leash, pausing now and then to let it catch up.",
         "n2": "An upperclassman {P} did not recognize asked, deadpan, how the tortoise was settling into campus.",
     }},
    {"key": "e3_cafe", "family": "E3", "domain": "cafe", "holdout": False,
     "locations": {"n0": "Cafe"},
     "facts": {
         "n0": "{P} perched on a stool beside {R}, scratching notes into a journal with a real feather quill.",
         "n0_private": "{P} had overpaid for the quill at a renaissance fair and was determined to get some use out of it.",
         "n1": "{Q} was writing a postcard with a goose-feather pen, dipping the nib into a tiny inkpot between words.",
         "n2": "A classmate squinted at the ink smudges on {P}'s fingertips and asked where people even buy quills.",
     }},
    {"key": "e3_clubs", "family": "E3", "domain": "clubs", "holdout": False,
     "slots": {"X": ["silver", "gold"]},
     "facts": {
         "n0": "{P} checked {R} from under a tall pointed wizard hat speckled with {X} stars.",
         "n0_private": "{P} wanted to see how long it took strangers to mention the hat, and was keeping count.",
         "n1": "{Q} strolled by in a purple conical hat dotted with little moons, the kind a stage magician might wear.",
         "n2": "A junior {P} knew from orientation asked, grinning, whether the pointy hat came with any actual spells.",
     }},
    # ---------------------------------------------------------------- holdout --
    {"key": "e3_library", "family": "E3", "domain": "library", "holdout": True,
     "locations": {"n0": "Library"},
     "facts": {
         "n0": "{P} lingered at {R}, reading a thick paperback through an oversized magnifying glass.",
         "n0_private": "{P} was combing an antique edition for misprints, a favor to an uncle who collects them.",
         "n1": "{Q} was hunched over a novel, peering at each line through a handheld magnifier like a detective.",
         "n2": "A passerby held an imaginary magnifying glass up to one eye, squinting theatrically at {P}.",
     }},
    {"key": "e3_quad", "family": "E3", "domain": "quad", "holdout": True,
     "locations": {"n0": "Quad"},
     "facts": {
         "n0": "{P} settled cross-legged near {R}, reading a newspaper aloud to a stubby potted cactus.",
         "n0_private": "{P} was running a private test of whether cacti grow faster when someone talks to them.",
         "n1": "{Q} was reciting poetry to a spiky succulent in a clay pot, holding the book up for it to see.",
         "n2": "A visiting grandparent chuckled and asked {P} if the cactus had any opinions about the headlines.",
     }},
    {"key": "e3_tech", "family": "E3", "domain": "tech", "holdout": True,
     "facts": {
         "n0": "{P} ranted to a friend about {R}, a camping headlamp glowing on top of a knit beanie.",
         "n0_private": "{P} had a caving trip that weekend and was testing how long the headlamp battery lasted.",
         "n1": "{Q} read a paperback by the beam of a strapped-on headlamp, despite plenty of light nearby.",
         "n2": "Somebody shielded their eyes from {P}'s headlamp and asked if the power had gone out.",
     }},
]
