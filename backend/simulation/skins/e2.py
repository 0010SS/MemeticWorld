"""E2 surface skins: cancelling mistakes (ontology v2 §1.2). SIMULATOR-ONLY.

Structure: P makes a mistake (n0); independently, a related person {S} makes a second mistake (n1);
the two errors offset each other and nothing goes wrong for P (n2). n0_private is P's inner view of
the first mistake (what P misread, never checked or failed to notice), never a label.

Writing rules these skins follow:
- Each pair of errors is complementary (a surplus meets a deficit, a delay meets a delay, a flip
  meets a flip, double the dose meets half the strength), so an attentive observer who sees all four
  facts can infer the offset; no fact says the errors cancelled, and there is no causal glue.
- n2 is a plain neutral outcome (nothing lost, on time, fits, reads fine), not a gain, to keep E2
  apart from E4. No weekday names or clock times, since an instance can start at any tick.
- n1 and n2 are location-neutral (every beat runs wherever P planned to be;
  S is brought there for n1).
  n0 is pinned only when its wording needs the domain's building, so pins follow the domain, never
  the family, and the instance shape stays family-neutral.
- A fact names only the people it involves: {S} (never {P}) in n1, {P} (never {S}) in n2; the other
  person in the offset is implied (a friend, a study partner), so involvement stays shape-only.
- Vocabulary is deliberately object-specific (stuck-together takeout boxes, salt stock at half
  strength, a shrunken cardigan) so content bigrams stay unique to this family; holdout skins share
  no content bigram with train skins.
"""
from __future__ import annotations

SKINS: list[dict] = [
    # ------------------------------------------------------------------ train --
    {"key": "e2_dining", "family": "E2", "domain": "dining", "holdout": False,
     "slots": {"X": ["pad thai", "vegetable dumplings", "bibimbap"]},
     "locations": {"n0": "Dining Hall"},
     "facts": {
         "n0": "{P} hurried in past {R} hoping to eat and found every serving line already shuttered.",
         "n0_private": "{P} had been going by an old list of meal hours scribbled inside a notebook from last semester.",
         "n1": "{S} had walked off earlier with two stuck-together takeout boxes of {X} instead of one and carried the spare around ever since.",
         "n2": "{P} polished off the spare takeout box of {X} and felt comfortably full for the rest of the day.",
     }},
    {"key": "e2_coursework", "family": "E2", "domain": "coursework", "holdout": False,
     "facts": {
         "n0": "{P} copied the problem list from {R} and solved only the even-numbered ones, though a classmate had agreed to take those.",
         "n0_private": "{P} had jotted their half of the split down backwards on a napkin during a noisy lunch.",
         "n1": "{S} spent hours on the odd-numbered problems, forgetting the agreed split had handed that half to someone else.",
         "n2": "{P} compared answers with a study partner and found every problem on the sheet covered exactly once.",
     }},
    {"key": "e2_lab", "family": "E2", "domain": "lab", "holdout": False,
     "locations": {"n0": "Research Lab"},
     "facts": {
         "n0": "{P} mixed a buffer next to {R}, pouring in twice the salt stock the recipe called for.",
         "n0_private": "{P} had read the amount off the line for a double batch without noticing.",
         "n1": "{S} had prepared the bench's salt stock at half strength a week earlier and never corrected the label.",
         "n2": "{P} scrolled through photos of the finished gel, every band crisp and sharp on the first run.",
     }},
    {"key": "e2_transit", "family": "E2", "domain": "transit", "holdout": False,
     "facts": {
         "n0": "{P} got absorbed in a complaint about {R} and rushed out an hour after the tournament carpool was due.",
         "n0_private": "{P} had silenced the phone while typing and never switched the ringer back on.",
         "n1": "{S}, driving the tournament carpool, had typed the wrong hour into a phone reminder and pulled up nearly an hour behind.",
         "n2": "{P} climbed into the late carpool hatchback with a travel mug, and the group reached the tournament before the opening round.",
     }},
    {"key": "e2_gym", "family": "E2", "domain": "gym", "holdout": False,
     "locations": {"n0": "Gym"},
     "facts": {
         "n0": "{P} trotted past {R} down to the basement court, though pickup volleyball always met upstairs.",
         "n0_private": "{P} was picturing last spring's schedule, back when the volleyball net still hung downstairs.",
         "n1": "{S}, who ran pickup volleyball, had mis-clicked a reservation form and booked the basement court instead of the usual space upstairs.",
         "n2": "{P} played the full volleyball session with the regulars and headed back sweaty and cheerful.",
     }},
    {"key": "e2_dorm", "family": "E2", "domain": "dorm", "holdout": False,
     "locations": {"n0": "Dorm"},
     "facts": {
         "n0": "{P} tossed a wool cardigan into the hottest wash cycle, then chatted with a neighbor beside {R}.",
         "n0_private": "{P} had never checked the care tag sewn inside the collar.",
         "n1": "{S} had picked out that cardigan as a going-away gift, two sizes larger than its recipient normally wore.",
         "n2": "{P} showed up in the shrunken wool cardigan, and it fit snugly across the shoulders.",
     }},
    {"key": "e2_cafe", "family": "E2", "domain": "cafe", "holdout": False,
     "slots": {"X": ["pesto panini", "falafel wrap", "brie baguette"]},
     "locations": {"n0": "Cafe"},
     "facts": {
         "n0": "{P} double-tapped the order button for a {X} while standing by {R} and paid for two.",
         "n0_private": "{P} had been reading a message from home and never glanced at the receipt total.",
         "n1": "{S} showed up hungry and penniless, having left a wallet on a desk in the lecture hall.",
         "n2": "{P} gave the second {X} to a hungry friend, and not a crumb of it went to waste.",
     }},
    {"key": "e2_clubs", "family": "E2", "domain": "clubs", "holdout": False,
     "facts": {
         "n0": "{P} printed thirty club posters with the text running upside down, after an hour spent dealing with {R}.",
         "n0_private": "{P} had flipped the page orientation setting and never looked at the first copy.",
         "n1": "{S} taped the stack of posters along the corridor by the lecture hall without looking, flipping each of them upside down.",
         "n2": "{P} passed the posters later and found every headline reading perfectly, top edge up.",
     }},
    # ---------------------------------------------------------------- holdout --
    {"key": "e2_library", "family": "E2", "domain": "library", "holdout": True,
     "locations": {"n0": "Library"},
     "facts": {
         "n0": "{P} photocopied chapter six of an anatomy textbook near {R}, though the request had said chapter five.",
         "n0_private": "{P} had misheard the chapter number over a crackly speakerphone.",
         "n1": "{S} had asked for chapter five, though the upcoming quiz covered only chapter six.",
         "n2": "{P} handed over the copied chapter, and the anatomy quiz that week drew every question from those pages.",
     }},
    {"key": "e2_quad", "family": "E2", "domain": "quad", "holdout": True,
     "slots": {"X": ["a jug of lemonade", "a bag of clementines", "a tin of cookies"]},
     "locations": {"n0": "Quad"},
     "facts": {
         "n0": "{P} spread a blanket near {R} and texted a friend directions, typing left where the path actually bends right.",
         "n0_private": "{P} had pictured the route as seen from the far end of campus, facing the other way.",
         "n1": "{S}, who often mixed up left and right, took the opposite turn from the one in the text.",
         "n2": "{P} spent the picnic sharing {X} with a friend who had arrived right on schedule.",
     }},
    {"key": "e2_tech", "family": "E2", "domain": "tech", "holdout": True,
     "facts": {
         "n0": "{P}, worn out after a long battle with {R}, left the laptop clock running an hour fast.",
         "n0_private": "{P} had tapped a neighboring time zone in a long settings list without a second glance.",
         "n1": "{S} sent out the tutoring call invitation with a start listed one hour later than the real one.",
         "n2": "{P} clicked into the tutoring call at the moment it began, while the others were still unmuting.",
     }},
]
