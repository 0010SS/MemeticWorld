"""E1 surface skins: cascade failure (ontology v2 §1.2). SIMULATOR-ONLY.

Structure: P makes a small slip involving the referent (n0); a consequence of that slip lands on a
related person {S} (n1); a further, otherwise unrelated failure then follows for P while P deals
with the fallout (n2). n0_private is what P did not notice, or why the slip happened.

Writing rules these skins follow:
- No fact states a link: no "because", "as a result", "which meant", "so". The chain is carried by
  shared concrete objects (a borrowed key lanyard, a shoebox of slides, a stale screenshot), and n2 only
  mentions what P was busy with when the new failure happened.
- n1 and n2 are location-neutral (a fee, a notice, a loss, arriving somewhere late), since every beat
  runs wherever P planned to be (S is brought there for n1). n0 is pinned only when its wording needs
  the domain's building.
- A fact names only the people it involves: {S} (never {P}) in n1, {P} (never {S}) in n2, so involvement
  and presence come from the shared shape, not from the wording (validate_skins).
- Vocabulary is deliberately object-specific (sharps bin, lockout fee, courier sticker) so content
  bigrams stay unique to this family; holdout skins share no content bigram with train skins.
"""
from __future__ import annotations

SKINS: list[dict] = [
    # ------------------------------------------------------------------ train --
    {"key": "e1_dining", "family": "E1", "domain": "dining", "holdout": False,
     "slots": {"X": ["a stack of pancakes and a smoothie", "a waffle plate and hot cocoa", "a burrito and two yogurt cups"]},
     "locations": {"n0": "Dining Hall"},
     "facts": {
         "n0": "{P} rested a borrowed key lanyard atop the condiment caddy beside {R} and left the building without it.",
         "n0_private": "{P} was balancing {X} and gave the caddy no backward glance.",
         "n1": "{S} had to pay a lockout fee to get a housing officer to open a bolted bedroom door.",
         "n2": "{P} turned in a calculus worksheet half blank, hands still reeking of lemon detergent from rummaging through dish-room bins.",
     }},
    {"key": "e1_coursework", "family": "E1", "domain": "coursework", "holdout": False,
     "slots": {"X": ["glaciers", "tectonic plates", "sand dunes"], "Y": ["volcanoes", "ocean currents", "karst caves"]},
     "facts": {
         "n0": "{P} circulated a stale screenshot of {R} to a handful of classmates prepping for a quiz.",
         "n0_private": "{P} had fished the image out of a cluttered gallery and was oblivious to the faded timestamp at its top.",
         "n1": "{S} crammed the section on {X} from the screenshot and then faced a quiz devoted entirely to {Y}.",
         "n2": "{P} answered a pile of irritated replies for an hour and let a work-study timesheet go unsigned past its cutoff.",
     }},
    {"key": "e1_lab", "family": "E1", "domain": "lab", "holdout": False,
     "locations": {"n0": "Research Lab"},
     "facts": {
         "n0": "{P} dumped a shoebox of stained microscope slides from the countertop next to {R} into a sharps bin.",
         "n0_private": "{P} mistook the shoebox for leftovers from a departed rotation student and glossed over the initials on its lid.",
         "n1": "{S} lost four days of careful cell counts when a shoebox of stained slides vanished from the countertop.",
         "n2": "{P} pitched in to restart the cell counts and walked out to a snipped cable where a scooter had been chained.",
     }},
    {"key": "e1_transit", "family": "E1", "domain": "transit", "holdout": False,
     "slots": {"X": ["highway guardrail", "canal towpath", "old rail trail"]},
     "facts": {
         "n0": "{P} offered an outdated tip about {R} to a few friends bound for a minor-league ballgame.",
         "n0_private": "{P} was going off a transportation notice from two semesters earlier, unaware a revised version was pinned directly beneath it.",
         "n1": "{S} arrived windblown and flustered after a two-mile trudge along the {X}.",
         "n2": "{P} kept thumbing out apologies and lost a page and a half of an unsaved essay when a frozen word processor crashed.",
     }},
    {"key": "e1_gym", "family": "E1", "domain": "gym", "holdout": False,
     "slots": {"X": ["soccer", "rugby", "lacrosse"]},
     "locations": {"n0": "Gym"},
     "facts": {
         "n0": "{P} scooped up a pair of look-alike cleats from a cubby beside {R} and lugged them home.",
         "n0_private": "{P} had stashed an identical pair in a neighboring cubby earlier and overlooked the initials scrawled across the tongues.",
         "n1": "{S} spent an intramural {X} match benched in borrowed sandals after combing every locker for a vanished pair of cleats.",
         "n2": "{P} jogged the look-alike cleats back to their owner and let a transit pass slip unseen from a hoodie pocket en route.",
     }},
    {"key": "e1_dorm", "family": "E1", "domain": "dorm", "holdout": False,
     "locations": {"n0": "Dorm"},
     "facts": {
         "n0": "{P} shoved a battered parcel that had been sitting beside {R} down the recycling chute.",
         "n0_private": "{P} assumed the parcel was discarded packaging and overlooked a courier sticker bearing a downstairs resident's name.",
         "n1": "{S} went into a probability exam without a newly ordered calculator that a courier email had marked as delivered.",
         "n2": "{P} rummaged in a recycling dumpster for an hour and was a no-show for a shift tryout at the campus bookstore.",
     }},
    {"key": "e1_cafe", "family": "E1", "domain": "cafe", "holdout": False,
     "slots": {"X": ["chai", "hot cider", "maple cortado"]},
     "locations": {"n0": "Cafe"},
     "facts": {
         "n0": "{P} sloshed a steaming {X} into an unzipped satchel hanging off a barstool near {R}.",
         "n0_private": "{P} was craning to hear the barista call names over the grinder noise and gave no heed to the barstool below.",
         "n1": "{S} unpacked a laptop from a damp satchel and found half its keys sugary and refusing to press.",
         "n2": "{P} loaned out a personal laptop for the week and was unable to upload a statics worksheet on time.",
     }},
    {"key": "e1_clubs", "family": "E1", "domain": "clubs", "holdout": False,
     "facts": {
         "n0": "{P} relayed a four-digit room code from {R} to the other officers, with two digits reversed.",
         "n0_private": "{P} recited the code from memory while juggling an armload of flyers and skipped checking the original listing.",
         "n1": "{S} hauled a crate of donated paperbacks up four flights to an unlit, deserted room and waited there alone.",
         "n2": "{P} raced between two buildings rounding up stray members and never filed a club treasury reimbursement before it lapsed.",
     }},
    # ---------------------------------------------------------------- holdout --
    {"key": "e1_library", "family": "E1", "domain": "library", "holdout": True,
     "locations": {"n0": "Library"},
     "facts": {
         "n0": "{P} parked a checked-out bundle of sheet music on a reshelving cart beside {R} and wandered off.",
         "n0_private": "{P} figured the cart doubled as a returns drop and took zero notice of a staff-only placard dangling off one end.",
         "n1": "{S} got an automated notice of a sixty-dollar replacement charge and a freeze on borrowing privileges.",
         "n2": "{P} traded terse emails with circulation staff and let an interlibrary loan quietly expire.",
     }},
    {"key": "e1_quad", "family": "E1", "domain": "quad", "holdout": True,
     "slots": {"X": ["sliced melon", "cut pineapple", "red grapes"]},
     "locations": {"n0": "Quad"},
     "facts": {
         "n0": "{P} unlatched a cooler of {X} next to {R} and left the lid flung wide open.",
         "n0_private": "{P} got roped into a round of spikeball and forgot all about the gaping cooler.",
         "n1": "{S} had to toss out a whole cooler of sun-warmed {X} set aside for a club potluck.",
         "n2": "{P} dashed off for more {X} and lost a rented camera tripod left unattended on the lawn.",
     }},
    {"key": "e1_tech", "family": "E1", "domain": "tech", "holdout": True,
     "facts": {
         "n0": "{P} told a cluster of classmates, with breezy confidence, that {R} had been repaired.",
         "n0_private": "{P} had glimpsed a technician's van idling by the curb and simply assumed any repair was complete.",
         "n1": "{S} counted on {R}, wasted most of an hour, and reached a seminar late and red-faced.",
         "n2": "{P} endured a long, prickly rant about the wasted hour and watched a study-abroad deposit window slam shut.",
     }},
]
