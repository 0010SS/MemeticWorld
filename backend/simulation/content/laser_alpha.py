"""Content pack `laser_alpha` (ontology v3 §1.2). WORLD TEXT, SIMULATOR-ONLY ids.

Everything agents may perceive is plain past tense. Hidden ids (classes K0-K3, causes, mappings) are
dictionary keys only and never appear in any text. Surfaces never name a cause or a fix; lexical
hygiene (content bigrams, nicknames, quotes) is enforced by tests/test_v3_world.py.
"""
NAME = "laser_alpha"

CAUSES = ("LENS", "DAMP", "BELT")
STABLE = "BELT"                                   # K2's cause in every regime
FIX = {"LENS": "lens", "DAMP": "dry", "BELT": "belt"}
# mapping id -> {regime: K1's cause}; K3 (regime B only) is caused by the B-cause
MAPPINGS = {"M1": {"A": "LENS", "B": "DAMP"}, "M2": {"A": "DAMP", "B": "LENS"}}

# the six-option menu (same wording in the world, the decision prompt and the battery)
ACTIONS = {
    "rerun": "re-run the sheet as it is",
    "lens": "clean the focus lens, then re-run",
    "dry": "dry the sheets on the heated rack for 15 minutes, then re-run",
    "belt": "tighten the drive belt, then re-run",
    "slow": "slow the cutting speed down, then re-run",
    "stop": "stop and leave the job for later",
}
ACTION_PAST = {
    "rerun": "re-ran the sheet as it was",
    "lens": "cleaned the focus lens and re-ran the sheet",
    "dry": "dried the sheets on the heated rack for 15 minutes and re-ran the sheet",
    "belt": "tightened the drive belt and re-ran the sheet",
    "slow": "slowed the cutting speed down and re-ran the sheet",
}

# class -> 8 surfaces (K3: per mapping)
SURFACES = {
    "K1": [
        "The first sheet came out with the left third still attached, and the edges there were the color of "
        "weak tea.",
        "Halfway through, the beam stopped biting on the left; those pieces had tan rims.",
        "On the left side the shapes never dropped free, and their outlines had turned a toasted brown.",
        "The parts on the left clung to the sheet by uncut strips, with scorched-looking borders along them.",
        "Only the right two thirds cut cleanly; toward the left the lines faded out and the rims looked singed.",
        "Pieces from the left end had to be snapped out by hand, and their edges were caramel colored.",
        "The beam lost its bite past the middle heading left, leaving half-cut outlines ringed in brown.",
        "Every shape on the left half stayed stuck in the acrylic, its kerf a dull amber.",
    ],
    "K2": [
        "Near the right edge every line came out doubled, as if the head had wobbled.",
        "Toward the right side of the sheet, straight lines turned into wavy ones, like a shaky hand drew them.",
        "Letters cut near the right margin had a ghost outline beside each stroke, slightly offset.",
        "Circles on the right came out lumpy and zigzagged instead of smooth.",
        "On the right-hand pieces each cut ran twice, a hair apart, so the edges looked fuzzy.",
        "The engraving on the right wandered, every edge rippling as though the sheet had been shaken.",
        "Along the right border the kerf split into two faint tracks instead of one clean groove.",
        "Out toward the right, straight edges came out with small wiggles, as though traced during a bumpy ride.",
    ],
    "K3": {
        "M1": [
            "During the cut there was a faint crackling, and tiny bubbles lined the edges.",
            "The sheet popped and sputtered under the beam, leaving little blisters along each cut.",
            "A soft crackle came from the bed while it cut, and the edges had frothy specks.",
            "Every cut edge came out pitted with small bubbles, and the machine made a snapping sound.",
            "There was a fizzing noise as the beam went, and the edges came out foamy.",
            "The cuts looked like they had boiled: rows of tiny blisters and a steady crackling.",
            "It crackled the whole way around each shape, and the edges were dotted with bubbles.",
            "Small popping sounds came from the sheet, and its cut rims were full of little voids.",
        ],
        "M2": [
            "The lines came out wider than the file said, and a faint haze clouded the surface.",
            "Every kerf was broad and blurry, and a milky film covered the sheet around it.",
            "The cut lines were fat and soft-edged, with a cloudy smear across the acrylic.",
            "Slots meant to fit tightly came out loose because every opening was too wide, and the face "
            "looked frosted.",
            "The engraving came out thick and smudgy, and the clear sheet had gone hazy all over.",
            "Each line was about twice its usual width, and a dull fog sat on the plastic.",
            "The outlines spread wider than drawn, and the surface turned faintly cloudy around them.",
            "Text came out bloated with blurry strokes, and a whitish mist dulled the finish.",
        ],
    },
    "K0": [
        "Every piece dropped out of the sheet cleanly, with crisp edges all the way around.",
        "The job came off the bed complete, each shape sharp and fully separated.",
        "All the parts cut through on the first pass and lifted out without any help.",
        "The finished pieces had smooth, even edges and matched the file exactly.",
        "Nothing stuck; the sheet came away as a clean frame with every shape popped free.",
        "The cut went all the way through everywhere, and the lines were neat and exact.",
        "Each outline came out true to the drawing, with tidy edges and no marks.",
        "The run finished without trouble; the parts separated easily and looked just as planned.",
    ],
}

# irrelevant oddities for clean jobs (p = workshop.oddity_prob)
ODDITIES = [
    "The exhaust fan sounded louder than usual.",
    "The lid's hinge squeaked each time someone opened it.",
    "A strip of masking tape was stuck to the side of the machine.",
    "The room smelled faintly of coffee from a mug by the door.",
]

# outcome clauses: "{P} {ACTION_PAST}; {result}."
SUCCESS = "the cut went all the way through this time"
FAIL = {
    "K1": "the left side still didn't cut through",
    "K2": "the lines near the right edge still came out doubled",
    "K3": {"M1": "it crackled again and the edges still came out bubbly",
           "M2": "the lines still came out wide and cloudy"},
}
