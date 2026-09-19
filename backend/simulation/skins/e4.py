"""E4 surface skins: beneficial failure (ontology v2 §1.2). SIMULATOR-ONLY.

Structure: P fails at something (n0); the failure itself brings P into contact with a related person
{S} or with an opportunity (n1); P ends up better off (n2). n0_private is P's inner view of the
failure (what P had hoped for, how it felt), never a label.

Writing rules these skins follow:
- No fact says the failure helped: no "luckily", "turned out", "thanks to", "because", "as a result".
  n1 points back to a concrete leftover of n0 (the spill, the cloudy tubes, the unused ticket), and n2
  states a plain gain (pay, a prize, a better seat), so the chain is inferable but never narrated.
- n1 and n2 are location-neutral (a message, a later chat, a report of what happened), since every beat
  runs wherever P planned to be (S is brought there for n1). n0 is pinned only when its wording needs
  the domain's building.
- A fact names only the people it involves: {S} (never {P}) in n1, {P} (never {S}) in n2. The contact
  with P in n1 is carried by the leftover (the dropped trays, the unused ticket, 'its owner').
- Vocabulary is deliberately object-specific (catering crew, pink colony, museum archive) so content
  bigrams stay unique to this family; holdout skins share no content bigram with train skins.
"""
from __future__ import annotations

SKINS: list[dict] = [
    # ------------------------------------------------------------------ train --
    {"key": "e4_dining", "family": "E4", "domain": "dining", "holdout": False,
     "slots": {"X": ["sesame noodles", "chickpea stew", "mushroom risotto"]},
     "locations": {"n0": "Dining Hall"},
     "facts": {
         "n0": "{P} tried carrying two stacked trays of {X} and dumped both onto the tiles beside {R}.",
         "n0_private": "{P} had hoped to fetch a roommate's dinner in a single haul and wanted to sink through the floor.",
         "n1": "{S} had watched the spill from a corner booth and, helping stack the dropped trays, mentioned the catering crew was short on weekend servers.",
         "n2": "{P} served at two catered faculty dinners that weekend and pocketed enough in tips to cover a month of groceries.",
     }},
    {"key": "e4_coursework", "family": "E4", "domain": "coursework", "holdout": False,
     "slots": {"X": ["treaty", "novella", "court ruling"]},
     "facts": {
         "n0": "{P} misread the prompt posted on {R} and produced seven dense pages dissecting the wrong {X}.",
         "n0_private": "{P} had skimmed the instructions while brushing teeth and braced for a failing mark on the essay.",
         "n1": "{S} came across the seven pages on the unassigned {X} and urged their author to enter the history department's essay contest.",
         "n2": "{P} collected a two-hundred-dollar check and a framed certificate at the history department's spring essay awards.",
     }},
    {"key": "e4_lab", "family": "E4", "domain": "lab", "holdout": False,
     "slots": {"X": ["yeast cultures", "enzyme aliquots", "bacterial stocks"]},
     "locations": {"n0": "Research Lab"},
     "facts": {
         "n0": "{P} left a rack of {X} on the bench beside {R} overnight, and every tube went cloudy.",
         "n0_private": "{P} had meant to move the rack into the four-degree room before dinner and dreaded explaining the loss to the postdoc.",
         "n1": "{S} was shown a photo of the cloudy tubes and pointed out a pink colony nobody in the lab had catalogued.",
         "n2": "{P} presented the pink colony at the departmental poster fair and won the audience ribbon.",
     }},
    {"key": "e4_transit", "family": "E4", "domain": "transit", "holdout": False,
     "slots": {"X": ["folk concert", "comedy show", "string quartet recital"]},
     "facts": {
         "n0": "{P} trudged back onto campus with an unused ticket to a {X} downtown, muttering about {R}.",
         "n0_private": "{P} had circled the {X} on a paper calendar back in August and sulked all the way up Charles Street.",
         "n1": "{S} noticed the crumpled, unused ticket and offered its owner an unclaimed seat at the sold-out Sunday {X}.",
         "n2": "{P} watched the Sunday {X} from the third row, then chatted with the performers over drinks backstage.",
     }},
    {"key": "e4_gym", "family": "E4", "domain": "gym", "holdout": False,
     "slots": {"X": ["loafers", "flip-flops", "hiking boots"]},
     "locations": {"n0": "Gym"},
     "facts": {
         "n0": "{P} arrived in {X}, and the attendant on duty waved {P} away from {R}.",
         "n0_private": "{P} had promised a sibling to exercise every day this month and hated snapping the streak on day four.",
         "n1": "{S} heard the griping about the footwear rule and offered a vacant slot in a jiu-jitsu trial class.",
         "n2": "{P} left the jiu-jitsu trial with a free semester of lessons and a new sparring partner.",
     }},
    {"key": "e4_dorm", "family": "E4", "domain": "dorm", "holdout": False,
     "slots": {"X": ["internship application", "fellowship essay", "scholarship form"]},
     "locations": {"n0": "Dorm"},
     "facts": {
         "n0": "{P} squandered the evening wrestling with {R}, and the {X} due at midnight went unsubmitted.",
         "n0_private": "{P} had polished the opening paragraph for weeks and sat on the stairwell steps feeling hollow.",
         "n1": "{S} listened to a long vent about the unsent {X} and forwarded a posting for a museum archive job that paid double.",
         "n2": "{P} landed a well-paid post cataloguing antique maps in the museum archive and started that Friday.",
     }},
    {"key": "e4_cafe", "family": "E4", "domain": "cafe", "holdout": False,
     "locations": {"n0": "Cafe"},
     "facts": {
         "n0": "{P} left a spiral sketchbook on the ledge by {R} and walked out without it.",
         "n0_private": "{P} had filled the sketchbook with secret portraits of strangers and panicked at the idea of anyone flipping through it.",
         "n1": "{S} found the spiral sketchbook, recognized the drawings, and messaged its owner about commissioning a poster for {S}'s band.",
         "n2": "{P} sold a gig poster design to a local band, and copies soon lined shop windows across Charles Village.",
     }},
    {"key": "e4_clubs", "family": "E4", "domain": "clubs", "holdout": False,
     "slots": {"X": ["sea shanty sing-along", "origami workshop", "crossword meetup"]},
     "facts": {
         "n0": "{P} scheduled a {X} through {R}, and every seat in the reserved room stayed empty.",
         "n0_private": "{P} had hand-lettered a stack of welcome badges and stared at the doorway, certain the idea had flopped.",
         "n1": "{S} heard about the empty {X} and invited its organizer to lead one at the packed Friday mixer {S} was hosting.",
         "n2": "{P} led a {X} for a roomful of sixty at the Friday mixer and collected twelve signatures for a new club roster.",
     }},
    # ---------------------------------------------------------------- holdout --
    {"key": "e4_library", "family": "E4", "domain": "library", "holdout": True,
     "slots": {"X": ["field guide to mosses", "Portuguese grammar", "atlas of Chesapeake shipwrecks"]},
     "locations": {"n0": "Library"},
     "facts": {
         "n0": "{P} combed the shelves around {R} for an out-of-print {X} and came up empty-handed.",
         "n0_private": "{P} had promised a grandmother a scan of the book's frontispiece and pictured her disappointed voice on Sunday.",
         "n1": "{S} overheard a lament about the missing {X} and lent out a personal copy crammed with pencilled annotations.",
         "n2": "{P} built a seminar presentation around the pencilled margin notes and was invited to repeat it at a faculty colloquium.",
     }},
    {"key": "e4_quad", "family": "E4", "domain": "quad", "holdout": True,
     "slots": {"X": ["hand-stitched bookmarks", "beeswax candles", "tie-dyed tote bags"]},
     "locations": {"n0": "Quad"},
     "facts": {
         "n0": "{P} set up a folding stand beside {R} to sell {X} and sold nothing all afternoon.",
         "n0_private": "{P} had inked price tags by lamplight and pictured a cash box stuffed with bills by noon.",
         "n1": "{S} saw a post about the unsold {X} and offered to buy the entire stock as favors for a cousin's wedding.",
         "n2": "{P} shipped eighty {X} to a wedding in Annapolis and cleared enough for a month's rent.",
     }},
    {"key": "e4_tech", "family": "E4", "domain": "tech", "holdout": True,
     "slots": {"X": ["teaching fellowship", "newspaper editor", "radio host"]},
     "facts": {
         "n0": "{P} lost a fight with {R} and reached the {X} interview after the panel had packed up.",
         "n0_private": "{P} had ironed a shirt for the interview at dawn and rehearsed answers nobody would hear.",
         "n1": "{S} spotted the interview candidate slumped outside the empty panel room and described an opening for paid guides at the campus observatory.",
         "n2": "{P} spent the evening steering a telescope toward Saturn for a crowd of visitors, on the observatory payroll.",
     }},
]
