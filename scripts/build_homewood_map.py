#!/usr/bin/env python3
"""Build the 1:1 pixel-art Homewood campus map for the MemeWorld frontend.

Output: frontend/homewood_map.json (run-length encoded layers), frontend/homewood_extra.png (generated
autotiles for streets, sidewalks, brick walkways, grey footprints, water, turf) and frontend/homewood_thumb.png.

The map is a Smallville-style (Generative Agents) 32 px tile map at true scale: 4 m per tile, north up, built
from the OpenStreetMap features in data/homewood_osm.json (see scripts/homewood_geo.py for the extent and the
rasterisation). Every tile and every piece of furniture comes from the tilesets vendored in
third_party/generative_agents (CuteRPG World, Room Builder, Modern Interiors), and furniture is copied as patches
from Smallville's own Tiled map. The places of the experiment (backend/simulation/world.py ARENAS) are furnished
buildings drawn at their real footprints (the Quad is Keyser Quad's lawn, the Shuttle Stop a paved verge on N Charles
St by the Gatehouse, the Museum's Garden its lawn); every other building is a grey footprint with its name; roads,
footpaths, lawns, fields, woods and Stony Run are drawn from the map data. Every arena gets >= 8 standing spots
reachable over the walk-cost grid. The build is deterministic (fixed seeds, no hash()).

    python3 scripts/build_homewood_map.py [--preview out.png] [--grid]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from homewood_geo import Geo, W, H, M_PER_TILE, PLACE_LABELS, proj, inb, cells_polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VILLE = ROOT / "third_party/generative_agents/environment/frontend_server/static_dirs/assets/the_ville"
OUT_JSON = ROOT / "frontend/homewood_map.json"
EXTRA_PNG = ROOT / "frontend/homewood_extra.png"
THUMB_PNG = ROOT / "frontend/homewood_thumb.png"
T = 32

# ----------------------------------------------------------------------------- Smallville source map
ville = json.load(open(VILLE / "visuals/the_ville_jan7.json"))
VW = ville["width"]
VL = {l["name"]: l["data"] for l in ville["layers"]}
VMAP = {"ground": "Exterior Ground", "deco1": "Exterior Decoration L1", "deco2": "Exterior Decoration L2",
        "floor": "Interior Ground", "wall": "Wall", "furn1": "Interior Furniture L1",
        "furn2": "Interior Furniture L2 ", "fg1": "Foreground L1", "fg2": "Foreground L2"}


def vget(layer, x, y):
    return VL[layer][y * VW + x] & 0x0FFFFFFF


# ----------------------------------------------------------------------------- output grid
LAYERS = ["bottom", "ground", "deco1", "deco2", "floor", "wall", "furn1", "furn2", "fg1", "fg2"]
grid = {n: [0] * (W * H) for n in LAYERS}
solid = [False] * (W * H)
cost = [3] * (W * H)          # walking cost: paths 1, lawns 2, grass 3, floors 1, blocked 0
kind = [""] * (W * H)


def put(layer, x, y, gid):
    if inb(x, y) and gid:
        grid[layer][y * W + x] = gid


def get(layer, x, y):
    return grid[layer][y * W + x] if inb(x, y) else 0


def block(x, y, v=True):
    if inb(x, y):
        solid[y * W + x] = v


def setcost(x, y, c, k=None):
    if inb(x, y):
        cost[y * W + x] = c
        if k:
            kind[y * W + x] = k


def stamp(x, y, sx, sy, w, h, layers=("furn1", "furn2", "fg1", "fg2"), collide=True):
    """Copy a w x h patch of Smallville's map (top-left sx,sy) to (x,y)."""
    for dy in range(h):
        for dx in range(w):
            for L in layers:
                g = vget(VMAP[L], sx + dx, sy + dy)
                if not g:
                    continue
                if L in ("fg1", "fg2", "furn1", "furn2") and 769 <= g < 9053:
                    continue  # Room Builder wall pieces that happen to sit in the source patch
                put(L, x + dx, y + dy, g)
            if collide and vget("Collisions", sx + dx, sy + dy):
                block(x + dx, y + dy)


# ----------------------------------------------------------------------------- terrains
GRASS = 2
LAWN = dict(c=226, n=210, s=242, w=225, e=227, nw=196, ne=197, sw=212, se=213, inw=228, ine=229, isw=244, ise=245)
SAND = dict(c=98, n=82, s=114, w=97, e=99, nw=68, ne=69, sw=84, se=85, inw=100, ine=101, isw=116, ise=117)
EXTRA = 40000            # firstgid of the generated tileset; 16 tiles per row, index = row*16 + mask
XROW = {"road": 0, "sidewalk": 1, "brick": 2, "gray_campus": 3, "gray_other": 4, "water": 5, "turf": 6, "parking": 7}


def autotile(cells, t, layer="ground"):
    for (x, y) in cells:
        n, s, w, e = (x, y - 1) in cells, (x, y + 1) in cells, (x - 1, y) in cells, (x + 1, y) in cells
        nw, ne, sw, se = (x - 1, y - 1) in cells, (x + 1, y - 1) in cells, (x - 1, y + 1) in cells, (x + 1, y + 1) in cells
        if not n and not w: g = t["nw"]
        elif not n and not e: g = t["ne"]
        elif not s and not w: g = t["sw"]
        elif not s and not e: g = t["se"]
        elif not n: g = t["n"]
        elif not s: g = t["s"]
        elif not w: g = t["w"]
        elif not e: g = t["e"]
        elif not nw: g = t["inw"]
        elif not ne: g = t["ine"]
        elif not sw: g = t["isw"]
        elif not se: g = t["ise"]
        else: g = t["c"]
        put(layer, x, y, g)


def autotile16(cells, row, layer="ground"):
    """Generated 16-variant autotiles: mask bit0 = no north neighbour, bit1 east, bit2 south, bit3 west."""
    base = EXTRA + XROW[row] * 16
    for (x, y) in cells:
        m = (0 if (x, y - 1) in cells else 1) | (0 if (x + 1, y) in cells else 2) | (0 if (x, y + 1) in cells else 4) | (0 if (x - 1, y) in cells else 8)
        put(layer, x, y, base + m)


def rect(x0, y0, x1, y1):
    return {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}


# ----------------------------------------------------------------------------- walls (Room Builder 9-slice)
OFF = dict(sideA=-1, sideB=2, cap=0, cap2=1, face=76, face2=77, tc1=-153, tc2=-150, c2a=-77, c2b=-74,
           jambL=73, jambL2=149, jambR=-155, jambR2=-79, botL=230, botR=227, bot=228, bot2=229)
VARIANTS_OK = {6668, 8804, 5088}


def paint_walls(floor, cap, gaps, mask, rnd):
    F = lambda x, y: (x, y) in floor
    G = lambda x, y: (x, y) in gaps
    P = lambda k: cap + OFF[k]
    var = cap in VARIANTS_OK
    out = {}
    for (x, y) in mask:
        if F(x, y) or G(x, y):
            continue
        N, S, S2 = F(x, y - 1), F(x, y + 1), F(x, y + 2)
        E, Wf = F(x + 1, y), F(x - 1, y)
        leftcell = (Wf or F(x + 2, y)) and not E
        rightcell = (E or F(x - 2, y)) and not Wf
        g = None
        if G(x, y + 1) and not N and not S:              # a vertical wall ending above a side doorway
            g = P("botL") if leftcell else P("botR") if rightcell else None
        elif G(x, y - 1) and not N and not S:            # a vertical wall starting below a side doorway
            g = P("tc2") if leftcell else P("tc1") if rightcell else None
        if g is None:
            if S:                                        # face row (directly above a floor row)
                if G(x + 1, y): g = P("jambL2")
                elif G(x - 1, y): g = P("jambR2")
                else: g = P("face2") if (var and rnd.random() < 0.3) else P("face")
            elif S2:                                     # cap row, also the band between stacked rooms
                if G(x + 1, y + 1): g = P("jambL")
                elif G(x - 1, y + 1): g = P("jambR")
                else: g = P("cap2") if (var and rnd.random() < 0.3) else P("cap")
            elif N:                                      # bottom wall row
                if G(x + 1, y): g = P("botL")
                elif G(x - 1, y): g = P("botR")
                else: g = P("bot2") if (var and rnd.random() < 0.25) else P("bot")
            elif E: g = P("sideA")
            elif Wf: g = P("sideB")
            elif F(x + 2, y): g = P("sideB")
            elif F(x - 2, y): g = P("sideA")
            elif F(x + 1, y + 2): g = P("tc1")
            elif F(x + 2, y + 2): g = P("tc2")
            elif F(x - 1, y + 2): g = P("tc2")
            elif F(x - 2, y + 2): g = P("tc1")
            elif F(x + 1, y + 1): g = P("c2a")
            elif F(x + 2, y + 1): g = P("c2b")
            elif F(x - 1, y + 1): g = P("c2b")
            elif F(x - 2, y + 1): g = P("c2a")
            elif F(x + 2, y - 1) and not F(x + 1, y - 1): g = P("botL")
            elif F(x + 1, y - 1): g = P("botR")
            elif F(x - 1, y - 1): g = P("botL")
            elif F(x - 2, y - 1): g = P("botR")
        if g:
            out[(x, y)] = g
    return out


FLOORS = {
    "wood": [490],
    "beige": [4874, 4875, 4876, 4877, 4950, 4951, 4952, 4953, 5026, 5027, 5028, 5029, 5102, 5103, 5104, 5105,
              5178, 5179, 5180, 5181, 5254, 5255, 5256, 5257],
    "teal": [4880, 4881, 4882, 4883, 4956, 4957, 4958, 4959, 5032, 5033, 5034, 5035, 5108, 5109, 5110, 5111,
             5184, 5185, 5186, 5187, 5260, 5261, 5262, 5263],
    "peach": [4898, 4899, 4900, 4901, 4974, 4975, 4976, 4977, 5050, 5051, 5052, 5053, 5126, 5127, 5128, 5129,
              5202, 5203, 5204, 5205, 5278, 5279, 5280, 5281],
    "stone": [5348, 5349, 5350, 5351, 5424, 5425, 5426, 5427, 5500, 5501, 5502, 5503, 5576, 5577, 5578, 5579,
              5652, 5653, 5654, 5655, 5728, 5729, 5730, 5731],
}

# ----------------------------------------------------------------------------- furniture catalogue (Smallville coords)
STAMPS = {
    "chalkboard": (108, 18, 2, 2), "poster_shelf": (112, 18, 2, 2), "bulletin": (114, 18, 2, 2),
    "class_desk": (111, 21, 2, 2), "teacher_desk": (108, 23, 2, 2),
    "armchair": (119, 19, 2, 2), "bigshelf": (121, 18, 4, 3), "table4": (119, 22, 4, 4), "tallshelf": (124, 25, 1, 3),
    "fridge": (72, 17, 2, 3), "rack": (74, 17, 2, 2), "sink": (76, 18, 2, 1), "counter8": (76, 19, 8, 2),
    "stool_red": (78, 21, 1, 1), "stool_brown": (80, 21, 1, 1), "stool_beige": (82, 21, 1, 1),
    "cafe_table4": (76, 22, 4, 4), "cafe_table2": (72, 23, 3, 2), "piano": (82, 23, 2, 3),
    "kitchen_sink": (119, 44, 3, 2), "fridge2": (122, 43, 2, 3), "rug": (117, 45, 1, 2), "round_table": (119, 46, 4, 4),
    "sofa": (124, 46, 1, 3), "pool_table": (118, 52, 2, 3),
    "bed_blue": (126, 45, 2, 3), "corkboard": (128, 44, 2, 1), "console_desk": (128, 45, 2, 2), "dresser": (131, 44, 2, 2),
    "desk_chair": (131, 46, 2, 2), "computer_desk": (126, 52, 2, 3), "dresser2": (122, 52, 2, 2),
    "toilet": (130, 52, 1, 2), "bath_sink": (131, 52, 1, 2), "shower": (132, 53, 2, 2), "bathtub": (130, 56, 2, 2),
    "gym_machine": (110, 55, 2, 3), "bike": (111, 60, 1, 2), "dumbbells": (108, 62, 2, 1),
    "orange_shelf": (114, 55, 2, 3), "bed_yellow": (118, 59, 2, 3), "washer": (106, 49, 2, 2), "desk_lamp": (109, 49, 2, 2),
    "store_shelf": (58, 41, 4, 3), "blue_mat": (62, 43, 1, 1), "glass_counter": (66, 44, 4, 2), "bookshelf3": (58, 46, 3, 3),
    "bookshelf3b": (62, 46, 3, 3), "food_case4": (58, 51, 4, 2), "potted_plants": (76, 42, 3, 2), "bouquet": (79, 42, 2, 2),
    "drink_fridge": (83, 47, 2, 2), "food_case2": (85, 51, 2, 2), "tall_fridge": (76, 44, 1, 2), "snack_rack": (78, 46, 1, 3),
    "lights": (89, 41, 4, 2), "lamps": (92, 41, 2, 2),
    "big_tree": (23, 42, 4, 6), "small_tree": (30, 43, 2, 2), "sign": (36, 42, 2, 2), "doormat": (75, 27, 4, 2),
}


def furn(name, x, y):
    sx, sy, w, h = STAMPS[name]
    stamp(x, y, sx, sy, w, h)


def tree(name, x, y):
    sx, sy, w, h = STAMPS[name]
    stamp(x, y, sx, sy, w, h, layers=("deco1", "deco2", "fg1", "fg2"), collide=False)
    for dx in range(w):
        block(x + dx, y + h - 1)
        if h >= 4:
            block(x + dx, y + h - 2)


# ----------------------------------------------------------------------------- buildings (real footprints, thin walls)
PLACES = {}
THIN = dict(cap=0, cap2=1, bot=228, bot2=229, sideL=-1, sideR=2, tl=-153, tr=-150, bl=230, br=227, fill=76)
BEDROOMS = {"Room 214", "Room 216", "Room 310", "Room 312", "Room 118", "Room 120", "Room 105", "Room 107", "Room 402",
            "Room 404", "Grad Apartment"}
PATHS = ("path", "plaza", "sidewalk")


def nb4(c):
    x, y = c
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def erode(mask):
    return {c for c in mask if all(n in mask for n in nb4(c))}


def bbox(cells):
    xs = [x for x, _ in cells]; ys = [y for _, y in cells]
    return [min(xs), min(ys), max(xs), max(ys)]


def largest_rect(cells):
    """Largest axis-aligned rectangle inside a set of cells (histogram method)."""
    if not cells:
        return None
    x0, y0, x1, y1 = bbox(cells)
    w = x1 - x0 + 1
    h = [0] * w
    best = (0, None)
    for y in range(y0, y1 + 1):
        for i in range(w):
            h[i] = h[i] + 1 if (x0 + i, y) in cells else 0
        stack = []
        for i in range(w + 1):
            cur = h[i] if i < w else 0
            start = i
            while stack and stack[-1][1] >= cur:
                s, hh = stack.pop()
                if hh * (i - s) > best[0]:
                    best = (hh * (i - s), (x0 + s, y - hh + 1, x0 + i - 1, y))
                start = s
            stack.append((start, cur))
    return best[1]


KIND_FLOOR = {"bedroom": "beige", "lounge": "wood", "hall": "wood", "dining": "peach", "cafe": "wood", "lecture": "wood",
              "seminar": "beige", "library_quiet": "wood", "library_study": "beige", "makerspace": "stone", "drylab": "stone",
              "wetlab": "teal", "stockroom": "stone", "gym": "wood",
              # campus places
              "clubroom": "beige", "auditorium": "wood", "lobby": "stone", "computerlab": "teal", "studylounge": "beige",
              "gallery": "wood", "stage": "wood", "greenroom": "peach", "kitchen": "teal", "studyroom": "beige",
              "fieldhouse": "wood", "track": "peach", "frontdesk": "beige", "mailroom": "stone"}


def furnish(kind_, cells, rect_, avoid, rnd, min_free=0):
    """Place furniture patches for a room type inside the room's floor cells; wall-mounted pieces sit on the wall row above.
    `min_free` (campus places) keeps at least that many furniture-free floor cells, so small rooms keep standing spots."""
    x0, y0, x1, y1 = rect_
    w, h = x1 - x0 + 1, y1 - y0 + 1
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

    def ok(x, y, pw, ph, wall_top=False):
        for dx in range(pw):
            for dy in range(ph):
                c = (x + dx, y + dy)
                if c in avoid:
                    return False
                if dy == 0 and wall_top:
                    if c in cells or not get("wall", *c) or (c[0], c[1] + 1) not in cells or get("furn1", *c):
                        return False
                    continue
                if c not in cells or solid[c[1] * W + c[0]] or get("furn1", *c) or get("furn2", *c):
                    return False
        return True

    def free_cells():
        return sum(1 for c in cells if not solid[c[1] * W + c[0]] and not get("furn1", *c) and not get("furn2", *c))

    def place(name, x, y, wall_top=False):
        sx, sy, pw, ph = STAMPS[name]
        if min_free and free_cells() - pw * (ph - (1 if wall_top else 0)) < min_free:
            return False
        if ok(x, y, pw, ph, wall_top):
            furn(name, x, y)
            return True
        return False

    def grid(name, gx0, gy0, gx1, gy1, stepx, stepy, wall_top=False):
        n = 0
        for gy in range(gy0, gy1 + 1, stepy):
            for gx in range(gx0, gx1 + 1, stepx):
                n += place(name, gx, gy, wall_top)
        return n

    def wall_row(items, gx0, gx1):
        """Lay wall-mounted pieces left to right along the top wall."""
        x = gx0
        for name in items:
            if x > gx1:
                break
            pw = STAMPS[name][2]
            if place(name, x, wt, True):
                x += pw
            else:
                x += 1

    wt = y0 - 1
    if kind_ == "bedroom":
        place(rnd.choice(["bed_blue", "bed_blue", "bed_yellow"]), x0, y0)
        place("corkboard", x0 + 1, wt, True) or place("corkboard", x0, wt, True)
        if w >= 4:
            place("desk_chair", x1 - 1, y0) or place("console_desk", x1 - 1, y0)
        if h >= 6:
            place("dresser", x0, y1 - 1) or place("dresser2", x0, y1 - 1)
        if w >= 4 and h >= 6:
            place("rug", x1, y0 + 3)
    elif kind_ == "lounge":
        wall_row(["fridge2", "kitchen_sink", "poster_shelf", "potted_plants"], x0, x1)
        place("round_table", cx - 2, cy - 1) or place("round_table", x0 + 1, y0 + 2) or place("cafe_table2", x0, y0 + 2)
        place("sofa", x1, y0 + 1)
        place("pool_table", x0, y1 - 2)
        place("armchair", x1 - 1, y1 - 1)
        place("rug", cx + 2, y1 - 1)
    elif kind_ == "hall":
        for c in sorted(cells)[::41]:
            place("rug", c[0], c[1])
        for gx in range(x0 + 2, x1 - 2, 14):
            place("potted_plants", gx, wt, True)
    elif kind_ == "dining":
        wall_row(["fridge", "rack", "sink", "fridge", "rack", "potted_plants"], x0, x1)
        if not place("counter8", x0, y0 + 3):
            grid("glass_counter", x0, y0 + 3, x1 - 3, y0 + 3, 5, 1)
        for gx in range(x0 + 1, x1, 2):
            place(rnd.choice(["stool_red", "stool_brown", "stool_beige"]), gx, y0 + 5)
        grid("cafe_table4", x0, y0 + 7, x1 - 3, y1 - 3, 5, 5)
        grid("cafe_table2", x0, y0 + 7, x1 - 2, y1 - 1, 4, 3)
    elif kind_ == "cafe":
        wall_row(["fridge", "rack", "sink", "bouquet", "bookshelf3", "potted_plants"], x0, x1)
        grid("glass_counter", x0, y0 + 3, x1 - 3, y0 + 3, 5, 1)
        for gx in range(x0 + 1, x1, 2):
            place(rnd.choice(["stool_red", "stool_brown", "stool_beige"]), gx, y0 + 5)
        grid("cafe_table4", x0, y0 + 7, x1 - 3, y1 - 3, 5, 5)
        grid("cafe_table2", x0, y0 + 7, x1 - 2, y1 - 1, 4, 3)
        place("piano", x1 - 1, y1 - 2)
    elif kind_ == "lecture":
        wall_row(["chalkboard", "chalkboard", "poster_shelf", "chalkboard", "bulletin", "lights", "chalkboard", "chalkboard"], x0, x1)
        place("teacher_desk", x0, y0 + 1)
        place("potted_plants", x1 - 2, wt, True)
        grid("class_desk", x0 + 1, y0 + 3, x1 - 1, y1 - 1, 3, 3)
    elif kind_ == "seminar":
        wall_row(["chalkboard", "bulletin", "bookshelf3", "poster_shelf"], x0, x1)
        place("table4", cx - 2, y0 + 2)
        if h >= 12:
            place("table4", cx - 2, y0 + 8)
        place("armchair", x0, y1 - 1); place("armchair", x1 - 1, y1 - 1)
        place("potted_plants", x1 - 2, wt, True)
    elif kind_ == "library_quiet":
        wall_row(["bigshelf", "bigshelf", "lamps", "bigshelf", "bigshelf"], x0, x1)
        for gy in range(y0 + 3, y1 - 2, 4):
            place("tallshelf", x1, gy)
        grid("table4", x0, y0 + 3, x1 - 4, y1 - 3, 5, 5)
        for gy in range(y0 + 4, y1 - 1, 5):
            place("armchair", x1 - 2, gy)
    elif kind_ == "library_study":
        wall_row(["bookshelf3", "bookshelf3", "bookshelf3", "potted_plants", "bookshelf3"], x0, x1)
        place("computer_desk", x0, y0 + 2); place("computer_desk", x0 + 2, y0 + 2)
        grid("table4", x0, y0 + 6, x1 - 4, y1 - 3, 5, 5)
        place("armchair", x1 - 1, y0 + 3); place("armchair", x1 - 1, y0 + 6)
        place("cafe_table2", x1 - 2, y1 - 1)
    elif kind_ == "makerspace":
        wall_row(["store_shelf", "orange_shelf", "snack_rack", "store_shelf"], x0, x1)
        grid("glass_counter", x0, y0 + 3, x1 - 3, y1 - 1, 5, 3)
        place("washer", x1 - 1, y1 - 1); place("tall_fridge", x1, y0 + 2)
    elif kind_ == "drylab":
        wall_row(["poster_shelf", "bookshelf3", "lights", "bookshelf3"], x0, x1)
        grid("computer_desk", x0, y0 + 2, x0, y1 - 2, 1, 4)
        grid("computer_desk", x1 - 1, y0 + 2, x1 - 1, y1 - 2, 1, 4)
        place("desk_lamp", cx - 1, cy); place("blue_mat", cx, y1)
    elif kind_ == "wetlab":
        wall_row(["drink_fridge", "sink", "drink_fridge", "tall_fridge", "snack_rack"], x0, x1)
        if not grid("food_case4", x0 + 1, y0 + 3, x1 - 4, y1 - 2, 6, 3):
            grid("food_case2", x0 + 1, y0 + 3, x1 - 1, y1 - 1, 4, 3)
        grid("food_case2", x0 + 1, y0 + 3, x1 - 1, y1 - 1, 4, 3)
        place("washer", x1 - 1, y1 - 1)
    elif kind_ == "stockroom":
        wall_row(["store_shelf", "orange_shelf", "snack_rack", "store_shelf"], x0, x1)
        grid("bookshelf3", x0, y0 + 3, x1 - 2, y1 - 2, 4, 4)
        place("drink_fridge", x1 - 1, y1 - 1); place("washer", x0, y1 - 1)
    elif kind_ == "gym":
        wall_row(["gym_machine", "gym_machine", "gym_machine", "gym_machine", "potted_plants", "gym_machine", "gym_machine"], x0, x1)
        grid("bike", x0 + 1, y0 + 4, x1 - 1, y0 + 4, 2, 1)
        grid("dumbbells", x0, y0 + 8, x1 - 1, y0 + 8, 3, 1)
        grid("gym_machine", x0 + 2, y0 + 11, x1 - 2, y0 + 11, 4, 1)
        grid("blue_mat", x0 + 1, y1 - 3, x1 - 1, y1 - 3, 2, 1)
        place("glass_counter", x1 - 4, y1 - 1); place("washer", x0, y1 - 1)
    # ---- campus places
    elif kind_ == "clubroom":
        wall_row(["bulletin", "poster_shelf", "bulletin", "corkboard", "bookshelf3", "bulletin", "potted_plants"], x0, x1)
        grid("table4", x0 + 1, y0 + 1, x1 - 4, y1 - 3, 6, 5)
        place("armchair", x1 - 1, y1 - 1)
    elif kind_ == "auditorium":
        wall_row(["lights", "lamps", "lights", "lamps", "lights", "lamps"], x0, x1)
        place("piano", x0, y0) or place("piano", x1 - 1, y0)
        for c in sorted(c for c in cells if y0 + 1 <= c[1] <= y0 + 2)[::5]:
            place("rug", c[0], c[1])
        for gy in range(y0 + 4, y1 + 1, 2):                    # rows of seats with a centre aisle
            for gx in range(x0, x1 + 1):
                if abs(gx - cx) > 0:
                    place(("stool_red", "stool_red", "stool_brown")[(gx + gy) % 3], gx, gy)
    elif kind_ == "lobby":
        wall_row(["potted_plants", "bulletin", "poster_shelf", "potted_plants", "bulletin", "lamps", "potted_plants"], x0, x1)
        place("glass_counter", cx - 2, y0 + 2) or place("glass_counter", x0 + 1, y0 + 2)
        place("armchair", x0, y1 - 1); place("armchair", x1 - 1, y1 - 1)
        for c in sorted(cells)[::37]:
            place("rug", c[0], c[1])
    elif kind_ == "computerlab":
        wall_row(["poster_shelf", "lights", "bulletin", "lights", "poster_shelf", "lights"], x0, x1)
        grid("computer_desk", x0, y0 + 1, x1 - 1, y1 - 2, 3, 4)
    elif kind_ == "studylounge":
        wall_row(["bookshelf3", "potted_plants", "bookshelf3", "lamps", "bookshelf3"], x0, x1)
        grid("table4", x0 + 1, y0 + 2, x1 - 4, y1 - 3, 6, 5)
        place("sofa", x1, y0 + 1); place("armchair", x0, y1 - 1); place("cafe_table2", x1 - 2, y1 - 1)
    elif kind_ == "gallery":
        wall_row(["poster_shelf", "lamps", "bouquet", "poster_shelf", "lamps", "poster_shelf", "bouquet"], x0, x1)
        for c in sorted(cells)[::17]:
            place(("bouquet", "potted_plants", "rug")[(c[0] + c[1]) % 3], c[0], c[1])
    elif kind_ == "stage":
        wall_row(["lights", "lamps", "lights"], x0, x1)
        for c in sorted(cells, key=lambda c: (c[1], c[0])):
            if place("piano", c[0], c[1]):
                break
        for c in sorted(cells)[::7]:
            place("rug", c[0], c[1])
    elif kind_ == "greenroom":
        for name in ("dresser", "sofa", "rack", "armchair", "bath_sink"):
            for c in sorted(cells, key=lambda c: (-c[1], c[0])):
                if place(name, c[0], c[1]):
                    break
    elif kind_ == "kitchen":
        wall_row(["fridge", "kitchen_sink", "rack", "sink", "fridge2", "potted_plants"], x0, x1)
        place("round_table", cx - 2, cy - 1) or place("cafe_table2", x0 + 1, y0 + 3)
        place("cafe_table2", x0, y1 - 1)
    elif kind_ == "studyroom":
        wall_row(["bookshelf3", "bookshelf3", "lamps", "bookshelf3", "potted_plants"], x0, x1)
        grid("table4", x0 + 1, y0 + 2, x1 - 4, y1 - 3, 6, 5)
        place("computer_desk", x1 - 1, y0) or place("desk_lamp", x1 - 1, y0)
    elif kind_ == "fieldhouse":
        grid("blue_mat", x0 + 3, y0 + 3, x1 - 3, y1 - 3, 7, 7)
        grid("gym_machine", x0 + 6, y0 + 6, x1 - 6, y1 - 6, 14, 14)
        grid("dumbbells", x0 + 10, y0 + 10, x1 - 10, y1 - 10, 14, 14)
    elif kind_ == "track":
        for c in sorted(cells)[::29]:
            place("potted_plants", c[0], c[1])
    elif kind_ == "frontdesk":
        wall_row(["bulletin", "potted_plants", "bookshelf3", "poster_shelf", "potted_plants", "bulletin"], x0, x1)
        place("glass_counter", cx - 2, y0 + 3) or place("glass_counter", x0, y0 + 3)
        for gx in range(x0, x1, 3):
            place("armchair", gx, y1 - 1)
    elif kind_ == "mailroom":
        wall_row(["store_shelf", "orange_shelf", "store_shelf", "orange_shelf", "store_shelf"], x0, x1)
        grid("bookshelf3", x0, y0 + 3, x1 - 2, y1 - 4, 5, 5)
        place("glass_counter", x0, y1 - 1)


def make_building(name, label, mask, cap, floor_name, rooms, kinds, rnd, arena_floor=None, extra_walk=None, min_free=0):
    """Draw a place on its real footprint: 1-tile walls along the footprint edge, rooms (rect lists, clipped to the
    interior) separated by 1-tile partitions, doors picked automatically, an entrance where the real walkways
    reach the building, furniture per room type."""
    inner = erode(mask)
    room_cells = {a: set().union(*[rect(*r) & inner for r in rs]) for a, rs in rooms.items()}
    owner = {c: a for a, cs in room_cells.items() for c in cs}
    partition = set()
    for c in inner:
        if c not in owner and len({owner[n] for n in nb4(c) if n in owner}) >= 2:
            partition.add(c)
    floor = inner - partition
    # doors through partitions: the middle of every shared edge, two cells wide when possible
    gaps = set()
    pairs = {}
    for c in partition:
        touch = tuple(sorted({owner[n] for n in nb4(c) if n in owner}))
        for i in range(len(touch)):
            for j in range(i + 1, len(touch)):
                pairs.setdefault((touch[i], touch[j]), []).append(c)
    for cs in pairs.values():
        cs.sort()
        mid = cs[len(cs) // 2]
        gaps.add(mid)
        for n in nb4(mid):
            if n in cs:
                gaps.add(n); break
    # entrance: a footprint-edge cell with floor inside and a walkway outside (never straight into a bedroom)
    boundary = mask - inner
    cands = []
    for w in boundary:
        ins = [n for n in nb4(w) if n in floor and owner.get(n) not in BEDROOMS]
        outs = [n for n in nb4(w) if n not in mask and inb(*n) and kind[n[1] * W + n[0]] in PATHS]
        if ins and outs:
            score = sum(1 for dx in range(-2, 3) for dy in range(-2, 3) if inb(w[0] + dx, w[1] + dy) and kind[(w[1] + dy) * W + w[0] + dx] in PATHS)
            cands.append((score, w, ins[0], outs[0]))
    if not cands:
        # no walkway touches the building: take the edge cell nearest to a walkway and lay a short path to it
        from collections import deque
        best = None
        for w in boundary:
            ins = [n for n in nb4(w) if n in floor and owner.get(n) not in BEDROOMS]
            outs = [n for n in nb4(w) if n not in mask and inb(*n) and not solid[n[1] * W + n[0]]]
            if not ins or not outs:
                continue
            seen = {outs[0]}; dq = deque([(outs[0], [outs[0]])])
            while dq:
                c, path = dq.popleft()
                if kind[c[1] * W + c[0]] in PATHS:
                    if best is None or len(path) < len(best[0]):
                        best = (path, w, ins[0], outs[0])
                    break
                if len(path) > 14:
                    continue
                for n in nb4(c):
                    if n not in seen and inb(*n) and n not in mask and not solid[n[1] * W + n[0]]:
                        seen.add(n); dq.append((n, path + [n]))
        if best:
            path, w, i0, o0 = best
            if extra_walk is not None:
                extra_walk.update(path)
            cands.append((0, w, i0, o0))
    cands.sort(key=lambda c: -c[0])
    entrance = []
    if cands:
        _, w, i0, o0 = cands[0]
        entrance.append(w)
        for n in nb4(w):
            if n in boundary and any(m in floor for m in nb4(n)) and any(m not in mask and inb(*m) for m in nb4(n)):
                entrance.append(n); break
    gaps |= set(entrance)
    # connectivity: every room must be reachable from the entrance
    from collections import deque
    start = next((n for n in nb4(entrance[0]) if n in floor), next(iter(floor))) if entrance else next(iter(floor))
    for _ in range(12):
        seen = {start}; dq = deque([start])
        while dq:
            c = dq.popleft()
            for n in nb4(c):
                if n not in seen and (n in floor or n in gaps):
                    seen.add(n); dq.append(n)
        missing = [a for a, cs in room_cells.items() if cs and not (cs & seen)]
        if not missing:
            break
        opened = False
        for a in missing:
            for c in partition:
                if c in gaps:
                    continue
                ns = set(nb4(c))
                if ns & room_cells[a] and any(n in seen for n in ns):
                    gaps.add(c); opened = True; break
            if opened:
                break
        if not opened:
            break
    # paint floors
    for c in floor | gaps:
        a = owner.get(c)
        pal = FLOORS[(arena_floor or {}).get(a) or KIND_FLOOR.get(kinds.get(a, ""), floor_name)]
        put("floor", c[0], c[1], rnd.choice(pal)); setcost(c[0], c[1], 1, "floor")
    # walls (1 tile): choose the Room Builder piece from the floor neighbourhood
    F = lambda c: c in floor or c in gaps
    var = cap in VARIANTS_OK
    for w in mask - floor - gaps:
        x, y = w
        n, s, e, wf = F((x, y - 1)), F((x, y + 1)), F((x + 1, y)), F((x - 1, y))
        if s: g = cap + (THIN["cap2"] if var and rnd.random() < .3 else THIN["cap"])
        elif n: g = cap + (THIN["bot2"] if var and rnd.random() < .25 else THIN["bot"])
        elif e: g = cap + THIN["sideL"]
        elif wf: g = cap + THIN["sideR"]
        elif F((x + 1, y + 1)): g = cap + THIN["tl"]
        elif F((x - 1, y + 1)): g = cap + THIN["tr"]
        elif F((x + 1, y - 1)): g = cap + THIN["bl"]
        elif F((x - 1, y - 1)): g = cap + THIN["br"]
        else: g = cap + THIN["fill"]
        put("wall", x, y, g); block(x, y)
        if s or n:
            put("fg1", x, y, g)
    # cobbles outside the entrance
    for w in entrance:
        for o in nb4(w):
            if o not in mask and inb(*o) and not solid[o[1] * W + o[0]]:
                put("ground", o[0], o[1], 9083)
                o2 = (2 * o[0] - w[0], 2 * o[1] - w[1])
                if inb(*o2) and o2 not in mask and not solid[o2[1] * W + o2[0]]:
                    put("ground", o2[0], o2[1], 9080)
                break
    # furniture, keeping the doorways clear
    avoid = {(g[0] + dx, g[1] + dy) for g in gaps for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    for a, cs in room_cells.items():
        if cs:
            furnish(kinds.get(a, "hall"), cs, bbox(cs), avoid, rnd, min_free)
    PLACES[name] = dict(name=name, label=label, kind="building", box=bbox(mask), boxes=[bbox(mask)], entrance=sorted(entrance),
                        arenas={a: dict(rect=bbox(cs), _cells=sorted(cs)) for a, cs in room_cells.items() if cs})


def place_buildings(geo, rnd, extra_walk):
    P = geo.places
    caps = {"Gym": 5088, "Dining Hall": 6668, "Dorm": 5604, "Classroom": 5613, "Library": 5088, "Cafe": 8804, "Research Lab": 6668}
    floors = {"Gym": "wood", "Dining Hall": "peach", "Dorm": "beige", "Classroom": "wood", "Library": "wood", "Cafe": "wood", "Research Lab": "stone"}

    def single(place, arena, kind_):
        mask = P[place]["cells"]
        make_building(place, PLACE_LABELS[place], mask, caps[place], floors[place], {arena: [largest_rect(erode(mask))]}, {arena: kind_}, rnd, extra_walk=extra_walk)

    single("Gym", "Main Floor", "gym")
    single("Dining Hall", "Main Floor", "dining")
    single("Cafe", "Counter", "cafe")
    # Hackerman: four rooms stacked north to south (the Stockroom is the v3 arena)
    mask = P["Research Lab"]["cells"]; ix0, iy0, ix1, iy1 = bbox(erode(mask)); h = iy1 - iy0 + 1
    q = h // 4
    rooms = {"Makerspace": [(ix0, iy0, ix1, iy0 + q - 1)], "Dry Lab": [(ix0, iy0 + q + 1, ix1, iy0 + 2 * q - 1)],
             "Wet Lab": [(ix0, iy0 + 2 * q + 1, ix1, iy0 + 3 * q - 1)], "Stockroom": [(ix0, iy0 + 3 * q + 1, ix1, iy1)]}
    make_building("Research Lab", PLACE_LABELS["Research Lab"], mask, caps["Research Lab"], floors["Research Lab"], rooms,
                  {"Makerspace": "makerspace", "Dry Lab": "drylab", "Wet Lab": "wetlab", "Stockroom": "stockroom"}, rnd, extra_walk=extra_walk)
    # Gilman: lecture hall west, seminar room east, a hallway along the south
    mask = P["Classroom"]["cells"]; ix0, iy0, ix1, iy1 = bbox(erode(mask)); mx = (ix0 + ix1) // 2
    rooms = {"Lecture Hall": [(ix0, iy0, mx - 1, iy1 - 7)], "Seminar Room": [(mx + 1, iy0, ix1, iy1 - 7)], "Hallway": [(ix0, iy1 - 5, ix1, iy1)]}
    make_building("Classroom", PLACE_LABELS["Classroom"], mask, caps["Classroom"], floors["Classroom"], rooms,
                  {"Lecture Hall": "lecture", "Seminar Room": "seminar", "Hallway": "hall"}, rnd, extra_walk=extra_walk)
    # Library: quiet floor in the MSE wing, study tables in Brody
    mask = P["Library"]["cells"]
    mse = geo.building_named("Milton S. Eisenhower Library")[1]["cells"]; brody = geo.building_named("Brody Learning Commons")[1]["cells"]
    rooms = {"Quiet Floor": [largest_rect(erode(mse))], "Study Tables": [largest_rect(erode(brody))]}
    make_building("Library", PLACE_LABELS["Library"], mask, caps["Library"], floors["Library"], rooms,
                  {"Quiet Floor": "library_quiet", "Study Tables": "library_study"}, rnd, {"Study Tables": "beige"}, extra_walk=extra_walk)
    # AMR II: an H. Bedrooms line the two wings (a corridor on the courtyard side), the lounge and the grad apartment
    # are the north blocks, and the central bar is the hallway. Everything is derived from the footprint itself.
    mask = P["Dorm"]["cells"]
    rooms, kinds = dorm_rooms(mask)
    make_building("Dorm", PLACE_LABELS["Dorm"], mask, caps["Dorm"], floors["Dorm"], rooms, kinds, rnd, {"Lounge": "wood", "Hallway": "wood"}, extra_walk=extra_walk)


def runs(cells):
    """A cell set as 1-row rects (make_building takes rooms as rect lists)."""
    rows = {}
    for x, y in sorted(cells):
        rows.setdefault(y, []).append(x)
    out = []
    for y, xs in sorted(rows.items()):
        a = b = xs[0]
        for x in xs[1:]:
            if x == b + 1:
                b = x
            else:
                out.append((a, y, b, y)); a = b = x
        out.append((a, y, b, y))
    return out


def cut(cells, key, t):
    """Split cells by a line key(c) == t (the line itself becomes the partition wall)."""
    return {c for c in cells if key(c) < t}, {c for c in cells if key(c) > t}


def median_cut(cells, key):
    vals = sorted(key(c) for c in cells)
    return vals[len(vals) // 2]


def full_rows(mask, frac=0.8):
    x0, _, x1, _ = bbox(mask)
    rows = {}
    for (x, y) in mask:
        rows[y] = rows.get(y, 0) + 1
    return sorted(y for y, n in rows.items() if n >= frac * (x1 - x0 + 1))


def largest_component(cells):
    from collections import deque
    left, best = set(cells), set()
    while left:
        s0 = min(left); comp = {s0}; dq = deque([s0]); left.discard(s0)
        while dq:
            c = dq.popleft()
            for n in nb4(c):
                if n in left:
                    left.discard(n); comp.add(n); dq.append(n)
        if len(comp) > len(best):
            best = comp
    return best


CAMPUS_MIN_FREE = 14        # furniture-free floor cells kept in every campus-place room (standing spots)


def campus_rooms(geo):
    """place -> (cap, floor, {arena: cells}, {arena: kind}): the campus places' rooms, derived from each real footprint.
    Arena names are backend/simulation/world.py ARENAS."""
    P = geo.places
    out = {}

    def inner_of(place):
        return erode(P[place]["cells"])

    # Glass Pavilion: the common room at the north end, the club room at the south end (next to Levering)
    inn = inner_of("Student Center"); _, iy0, _, iy1 = bbox(inn)
    a, b = cut(inn, lambda c: c[1], iy0 + (iy1 - iy0 + 1) * 11 // 20)
    out["Student Center"] = (8804, "wood", {"Common Room": a, "Club Room": b}, {"Common Room": "lounge", "Club Room": "clubroom"})
    # Shriver Hall: the lobby is the east-west bar, the hall the block south of it
    mask = P["Auditorium"]["cells"]; inn = erode(mask)
    a, b = cut(inn, lambda c: c[1], max(full_rows(mask)))
    out["Auditorium"] = (5613, "wood", {"Main Hall": b, "Lobby": a}, {"Main Hall": "auditorium", "Lobby": "lobby"})
    # Malone Hall: computer lab north, study lounge south
    inn = inner_of("Engineering Hall"); _, iy0, _, iy1 = bbox(inn)
    a, b = cut(inn, lambda c: c[1], (iy0 + iy1) // 2)
    out["Engineering Hall"] = (6668, "stone", {"Computer Lab": a, "Study Lounge": b}, {"Computer Lab": "computerlab", "Study Lounge": "studylounge"})
    # Mudd Hall: the teaching lab in the north wing, the lecture room in the main block
    mask = P["Science Hall"]["cells"]; inn = erode(mask)
    a, b = cut(inn, lambda c: c[1], min(full_rows(mask)) - 1)
    out["Science Hall"] = (5088, "stone", {"Teaching Lab": a, "Lecture Room": b}, {"Teaching Lab": "wetlab", "Lecture Room": "lecture"})
    # Homewood Museum: the house is the gallery (its garden is outdoors, see museum_garden)
    inn = largest_component(inner_of("Museum"))
    out["Museum"] = (5604, "wood", {"Gallery": inn}, {"Gallery": "gallery"})
    # Merrick Barn (a 45-degree footprint): stage and green room either side of a diagonal partition
    inn = inner_of("Theater"); key = lambda c: c[0] + c[1]
    a, b = cut(inn, key, median_cut(inn, key))
    out["Theater"] = (8804, "wood", {"Stage": a, "Green Room": b}, {"Stage": "stage", "Green Room": "greenroom"})
    # The Charles: kitchen west, study room east
    inn = inner_of("Apartments"); ix0, _, ix1, _ = bbox(inn)
    a, b = cut(inn, lambda c: c[0], (ix0 + ix1) // 2)
    out["Apartments"] = (5604, "beige", {"Kitchen": a, "Study Room": b}, {"Kitchen": "kitchen", "Study Room": "studyroom"})
    # White Athletic Center: the long north-west arm is the track, the main hall the field house
    mask = P["Athletic Center"]["cells"]; inn = erode(mask)
    rows = {}
    for (x, y) in mask:
        rows[y] = rows.get(y, 0) + 1
    arm_end = min(y for y, n in sorted(rows.items()) if n >= 12 and y > min(rows) + 5)
    a, b = cut(inn, lambda c: c[1], arm_end)
    out["Athletic Center"] = (5088, "wood", {"Field House": largest_component(b), "Track": largest_component(a)},
                              {"Field House": "fieldhouse", "Track": "track"})
    # Garland Hall: front desk west (by the walkway), mailroom east
    inn = inner_of("Admin Building"); ix0, _, ix1, _ = bbox(inn)
    a, b = cut(inn, lambda c: c[0], ix0 + (ix1 - ix0 + 1) * 11 // 20)
    out["Admin Building"] = (5613, "beige", {"Front Desk": a, "Mailroom": b}, {"Front Desk": "frontdesk", "Mailroom": "mailroom"})
    return out


def place_campus(geo, extra_walk):
    """The campus places, each with its own rnd (independent of the original eight and of each other)."""
    for place, (cap, floor_name, rooms, kinds) in campus_rooms(geo).items():
        rnd = random.Random(f"homewood:{place}")
        make_building(place, PLACE_LABELS[place], geo.places[place]["cells"], cap, floor_name,
                      {a: runs(cs) for a, cs in rooms.items()}, kinds, rnd, extra_walk=extra_walk, min_free=CAMPUS_MIN_FREE)
    fh = PLACES["Athletic Center"]["arenas"].get("Field House")
    if fh:                                             # the field house floor is turf with white lines
        autotile16(set(map(tuple, fh["_cells"])), "turf", layer="floor")


def museum_garden(geo, rnd):
    """The Museum's Garden arena: the lawn around Homewood House (within 6 tiles of the house, the largest open
    stretch), laid as a tended lawn with flower beds and a parasol."""
    house = geo.places["Museum"]["cells"]
    near = set()
    for (x, y) in house:
        for dx in range(-6, 7):
            for dy in range(-6, 7):
                c = (x + dx, y + dy)
                if inb(*c) and c not in house and not solid[c[1] * W + c[0]] and kind[c[1] * W + c[0]] in ("", "lawn") \
                        and not get("wall", *c) and not get("floor", *c) and not get("ground", *c) in (9080, 9083):
                    near.add(c)
    garden = largest_component(near)
    autotile(garden, LAWN)
    for c in garden:
        setcost(*c, 2, "garden")
    FLOWERS = [55, 56, 71, 72, 87, 88, 103, 104, 119, 120, 121, 122, 123]
    for c in sorted(garden):
        if rnd.random() < 0.18:
            put("deco1", c[0], c[1], rnd.choice(FLOWERS))
    for c in sorted(garden, key=lambda c: (c[1], c[0])):
        if all((c[0] + dx, c[1] + dy) in garden for dx in range(4) for dy in range(3)):
            stamp(c[0], c[1], 70, 77, 4, 3, layers=("furn1", "furn2", "fg1", "fg2"), collide=False)   # parasol and chair
            break
    return garden


def shuttle_stop(geo):
    """The Shuttle Stop: a paved verge between the Gatehouse and the N Charles St sidewalk, with a bench and a
    timetable board."""
    pad = set(geo.places["Shuttle Stop"]["cells"])
    autotile16(pad, "sidewalk")
    for c in pad:
        setcost(*c, 1, "plaza")
    x0, y0, x1, y1 = bbox(pad)
    my = (y0 + y1) // 2
    for bx in (x0 + 1, x0 + 1):                        # bench (CuteRPG Village B), facing the street
        if (bx, my) in pad and (bx + 1, my) in pad:
            put("deco2", bx, my, 9160); put("deco2", bx + 1, my, 9161); block(bx, my); block(bx + 1, my)
            break
    if all((x0 + dx, y0 + 1 + dy) in pad for dx in range(2) for dy in range(2)):    # timetable board
        put("deco2", x0, y0 + 1, 9093); put("deco2", x0 + 1, y0 + 1, 9094)
        put("fg1", x0, y0 + 1, 9093); put("fg1", x0 + 1, y0 + 1, 9094)
        put("deco2", x0, y0 + 2, 9109); put("deco2", x0 + 1, y0 + 2, 9110)
        block(x0, y0 + 2); block(x0 + 1, y0 + 2)
    PLACES["Shuttle Stop"] = dict(name="Shuttle Stop", label=PLACE_LABELS["Shuttle Stop"], kind="stop", box=bbox(pad),
                                  boxes=[bbox(pad)], entrance=[], arenas={"Bench": dict(rect=bbox(pad), _cells=sorted(pad))})


def reachable_from(start):
    from collections import deque
    seen = [False] * (W * H); dq = deque([start]); seen[start[1] * W + start[0]] = True
    while dq:
        x, y = dq.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if inb(nx, ny) and not seen[ny * W + nx] and not solid[ny * W + nx] and cost[ny * W + nx] > 0:
                seen[ny * W + nx] = True; dq.append((nx, ny))
    return seen


def connect_places(quad):
    """A place cut off from the campus by roads (e.g. The Charles, across N Charles St, where the mapped crossing
    stops short of the kerb) gets a crosswalk: the cheapest walk from the place to the reachable network, with road
    cells allowed at a high cost, and every road cell on it paved as a crossing."""
    import heapq
    seen = reachable_from(next(iter(sorted(quad))))
    for pl in PLACES.values():
        src = {tuple(c) for ar in pl["arenas"].values() for c in ar.get("_cells", []) if not solid[c[1] * W + c[0]]}
        if not src or any(seen[y * W + x] for (x, y) in src):
            continue
        dist, prev, pq = {}, {}, []
        for c in sorted(src):
            dist[c] = 0; heapq.heappush(pq, (0, c))
        goal = None
        while pq:
            d, c = heapq.heappop(pq)
            if d > dist.get(c, 1e18):
                continue
            if seen[c[1] * W + c[0]]:
                goal = c; break
            for n in nb4(c):
                if not inb(*n):
                    continue
                i = n[1] * W + n[0]
                if kind[i] == "road":
                    step = 6
                elif solid[i] or cost[i] == 0:
                    continue
                else:
                    step = cost[i]
                if d + step < dist.get(n, 1e18):
                    dist[n] = d + step; prev[n] = c; heapq.heappush(pq, (d + step, n))
        if goal is None:
            print(f"  WARNING {pl['name']}: no crossing found")
            continue
        crossing, c = set(), goal
        while c in prev:
            if kind[c[1] * W + c[0]] == "road":
                crossing.add(c)
            c = prev[c]
        for (x, y) in crossing:
            block(x, y, False); setcost(x, y, 1, "path")
        autotile16(crossing, "sidewalk")
        print(f"  {pl['name']}: laid a {len(crossing)}-tile crossing")
        seen = reachable_from(next(iter(sorted(quad))))


def dorm_rooms(mask):
    x0, y0, x1, y1 = bbox(mask)
    width = x1 - x0 + 1
    rows = {}
    for (x, y) in mask:
        rows.setdefault(y, []).append(x)
    bar = [y for y, xs in rows.items() if len(xs) >= 0.8 * width]      # rows that are filled almost edge to edge
    by0, by1 = min(bar), max(bar)
    inner = erode(mask)
    mid = (x0 + x1) / 2
    probe = by1 + 3
    wl = sorted(x for (x, y) in inner if y == probe and x < mid)
    wr = sorted(x for (x, y) in inner if y == probe and x > mid)
    wing_bottom = max(y for _, y in inner)
    rooms = {"Hallway": [(min(x for x, y in inner if by0 < y < by1), by0 + 1, max(x for x, y in inner if by0 < y < by1), by1 - 1)]}
    kinds = {"Hallway": "hall"}
    if wl:
        rooms["Hallway"].append((wl[-1], by1, wl[-1], wing_bottom))                 # west corridor, courtyard side
        names = ["Room 310", "Room 118", "Room 312", "Room 120"]      # 312/120: 16-agent population
        for k, nm in enumerate(names):
            ry0 = by1 + 1 + 5 * k
            if ry0 + 3 <= wing_bottom:
                rooms[nm] = [(wl[0], ry0, wl[-1] - 1, ry0 + 3)]; kinds[nm] = "bedroom"
    if wr:
        rooms["Hallway"].append((wr[0], by1, wr[0], wing_bottom))                   # east corridor
        for k, nm in enumerate(["Room 105", "Room 402", "Room 107", "Room 404"]):   # 107/404: 16-agent population
            ry0 = by1 + 1 + 5 * k
            if ry0 + 3 <= wing_bottom:
                rooms[nm] = [(wr[0] + 1, ry0, wr[-1], ry0 + 3)]; kinds[nm] = "bedroom"
    above = {c for c in inner if c[1] < by0}
    blocks = []
    for name, lo, hi in (("Room 214", x0, x0 + width / 6), ("Room 216", x0 + width / 6, x0 + width / 3),
                         ("Lounge", x0 + width / 3, x0 + 2 * width / 3), ("Grad Apartment", x0 + 2 * width / 3, x1 + 1)):
        cs = {c for c in above if lo <= c[0] < hi}
        r = largest_rect(cs)
        if r:
            rooms[name] = [r]; kinds[name] = "lounge" if name == "Lounge" else "bedroom"
    # rooms added with the 16-agent population that had no fixed slot: carve each from the largest free interior
    # rectangle left (not the hallway, one cell of wall clearance around every room already placed)
    def cells_of(rs):
        return {(x, y) for (a, b, c, d) in rs for x in range(a, c + 1) for y in range(b, d + 1)}
    for name in ("Room 216", "Room 120", "Room 404"):
        if name in rooms:
            continue
        taken = cells_of(rooms["Hallway"])
        for nm, rs in rooms.items():
            if nm != "Hallway":
                taken |= {(x + dx, y + dy) for (x, y) in cells_of(rs) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        r = largest_rect(set(inner) - taken)
        if r and (r[2] - r[0] + 1) * (r[3] - r[1] + 1) >= 4:
            rooms[name] = [r]; kinds[name] = "bedroom"
        elif "Lounge" in rooms and rooms["Lounge"][0][2] - rooms["Lounge"][0][0] >= 6:
            a, b, c, d = rooms["Lounge"][0]                   # no free space left: split the wide lounge
            m = (a + c) // 2
            rooms["Lounge"] = [(a, b, m - 1, d)]
            rooms[name] = [(m + 1, b, c, d)]; kinds[name] = "bedroom"
    return rooms, kinds


def build(seed=7):
    rnd = random.Random(seed)
    geo = Geo()
    for i in range(W * H):
        grid["bottom"][i] = GRASS
    cls = geo.cls

    def cells_of(name):
        return {(i % W, i // W) for i, c in enumerate(cls) if c == name}

    # ---- terrain from the map data
    lawn = cells_of("lawn")
    autotile(lawn, LAWN)
    for c in lawn:
        setcost(*c, 2, "lawn")
    pitch = cells_of("pitch")
    autotile16(pitch, "turf")
    for c in pitch:
        setcost(*c, 2, "lawn")
    wood = cells_of("wood")
    for c in wood:
        setcost(*c, 3, "wood")
    water = cells_of("water")
    autotile16(water, "water")
    for c in water:
        setcost(*c, 0, "water"); block(*c)
    parking = cells_of("parking")
    autotile16(parking, "parking")
    for c in parking:
        setcost(*c, 2, "parking")
    road = cells_of("road")
    autotile16(road, "road")
    for c in road:
        setcost(*c, 0, "road"); block(*c)
    plaza = cells_of("plaza")
    autotile(plaza, SAND)
    for c in plaza:
        setcost(*c, 1, "plaza")
    walk = cells_of("walk") | cells_of("steps") | cells_of("track")
    sidewalk = cells_of("sidewalk")
    trail = cells_of("trail")
    autotile16(sidewalk, "sidewalk")
    autotile(trail, SAND)
    autotile16(walk, "brick")
    for c in walk | sidewalk | trail:
        setcost(*c, 1, "path")
    # ---- grey footprints for every building outside the experiment
    context = []
    for bi, b in enumerate(geo.buildings):
        if "place" in b:
            continue
        cells = {c for c in b["cells"] if cls[c[1] * W + c[0]] == "building" and geo.bid[c[1] * W + c[0]] == bi}
        autotile16(cells, "gray_campus" if b["campus"] else "gray_other")
        for c in cells:
            block(*c); setcost(*c, 0, "building")
        if b["name"] and (b["campus"] or len(cells) >= 30):
            cx, cy = b["centroid"]
            context.append(dict(name=b["name"], x=round(cx, 1), y=round(cy, 1), kind="campus" if b["campus"] else "other"))
    for name, f in geo.names.items():
        t = f["tags"]
        if f["type"] in ("polygon", "multipolygon") and (t.get("leisure") in ("park", "pitch", "garden") or t.get("natural") == "wood") \
                and name != "Keyser Quad":
            cells = cells_polygon(geo._rings(f))
            if len(cells) >= 40:
                cx, cy = geo._centroid(cells)
                context.append(dict(name=name, x=round(cx, 1), y=round(cy, 1), kind="green"))
    best = {}
    for f in geo.features:
        t = f["tags"]
        if f["type"] == "line" and t.get("highway") in ("primary", "secondary", "tertiary", "residential", "unclassified") and t.get("name"):
            pts = [proj(*c) for c in f["coords"]]
            L = sum(((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2) ** .5 for i in range(len(pts) - 1))
            if L > best.get(t["name"], (0,))[0]:
                best[t["name"]] = (L, pts[len(pts) // 2])
    for name, (L, (cx, cy)) in best.items():
        if L >= 15 and 0 < cx < W and 0 < cy < H:
            context.append(dict(name=name, x=round(cx, 1), y=round(cy, 1), kind="road"))

    # ---- the places on their real footprints
    extra_walk = set()
    place_buildings(geo, rnd, extra_walk)
    place_campus(geo, extra_walk)
    if extra_walk:
        walk |= extra_walk
        autotile16(walk, "brick")
        for c in extra_walk:
            setcost(*c, 1, "path")
    quad = P["Quad"]["cells"] if (P := geo.places) else set()
    PLACES["Quad"] = dict(name="Quad", label=PLACE_LABELS["Quad"], kind="lawn", box=P["Quad"]["boxes"][0], boxes=P["Quad"]["boxes"],
                          entrance=[], arenas={"Lawn": dict(rect=P["Quad"]["boxes"][0])})
    garden = museum_garden(geo, random.Random("homewood:Museum garden"))
    mu = PLACES["Museum"]
    mu["arenas"]["Garden"] = dict(rect=bbox(garden), _cells=sorted(garden), _mind=2)
    mu["boxes"] = [mu["box"], bbox(garden)]
    mu["box"] = bbox(set(garden) | {(mu["box"][0], mu["box"][1]), (mu["box"][2], mu["box"][3])})
    shuttle_stop(geo)

    # ---- trees, tufts, flowers
    def free_rect(x, y, w, h, allow=("", "lawn", "wood")):
        for dx in range(w):
            for dy in range(h):
                cx, cy = x + dx, y + dy
                if not inb(cx, cy) or solid[cy * W + cx] or kind[cy * W + cx] not in allow:
                    return False
                if get("deco1", cx, cy) or get("deco2", cx, cy) or get("wall", cx, cy) or get("floor", cx, cy):
                    return False
        return True

    for gy in range(0, H, 2):                      # packed rows of trees, the way Smallville draws its forests
        for gx in range(0, W, 2):
            if (gx, gy) in wood and free_rect(gx, gy, 2, 2, ("wood",)):
                if gx % 12 == 0 and gy % 12 == 0 and free_rect(gx, gy, 4, 6, ("wood",)):
                    tree("big_tree", gx, gy)
                else:
                    tree("small_tree", gx, gy)
    for (x, y) in geo.trees:
        if free_rect(x, y, 2, 2):
            tree("small_tree", x, y)
    for (x, y) in sorted(lawn):
        if rnd.random() < 0.012 and free_rect(x - 1, y - 1, 4, 4, ("lawn",)):
            tree("small_tree", x, y)
    FLOWERS = [55, 56, 71, 72, 87, 88, 103, 104, 119, 120, 121, 122, 123]
    TUFTS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    for _ in range(W * H // 24):
        x, y = rnd.randrange(W), rnd.randrange(H)
        k = kind[y * W + x]
        if free_rect(x, y, 1, 1) and k in ("", "wood"):
            put("deco1", x, y, rnd.choice(TUFTS if rnd.random() < .88 else FLOWERS))
    for _ in range(W * H // 90):
        x, y = rnd.randrange(W), rnd.randrange(H)
        if free_rect(x, y, 1, 1, ("lawn",)):
            put("deco1", x, y, rnd.choice(FLOWERS if rnd.random() < .3 else TUFTS))
    connect_places(quad)
    # ---- agent spots
    for pl in PLACES.values():
        for a, ar in pl["arenas"].items():
            x0, y0, x1, y1 = ar["rect"]
            if pl["kind"] == "lawn":
                cells = [(x, y) for (x, y) in quad if not solid[y * W + x] and kind[y * W + x] == "lawn"]
            else:
                cells = [tuple(c) for c in ar.pop("_cells") if not solid[c[1] * W + c[0]]]
            r2 = random.Random(zlib.crc32(f"{pl['name']}|{a}".encode()) & 0xFFFF)
            r2.shuffle(cells)
            spots, mind = [], ar.pop("_mind", 3 if pl["kind"] == "lawn" else 1)
            for c in cells:
                if all(max(abs(c[0] - s[0]), abs(c[1] - s[1])) > mind for s in spots):
                    spots.append(c)
                if len(spots) >= 12:
                    break
            for c in cells:
                if len(spots) >= 8:
                    break
                if c not in spots:
                    spots.append(c)
            ar["spots"] = spots
    seen = reachable_from(next(iter(sorted(quad))))
    for pl in PLACES.values():
        for a, ar in pl["arenas"].items():
            ok = [c for c in ar["spots"] if seen[c[1] * W + c[0]]]
            if len(ok) != len(ar["spots"]):
                print(f"  {pl['name']} / {a}: dropped {len(ar['spots']) - len(ok)} unreachable spots")
            ar["spots"] = ok
            if len(ok) < 4:
                print(f"  WARNING {pl['name']} / {a}: only {len(ok)} reachable spots")
    return context


# ----------------------------------------------------------------------------- extra tileset (generated)
def write_extra_tileset():
    from PIL import Image, ImageDraw
    field = Image.open(VILLE / "visuals/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Field_B.png").convert("RGBA")

    def ftile(gid):
        l = gid - 1
        return field.crop(((l % 16) * T, (l // 16) * T, (l % 16) * T + T, (l // 16) * T + T))

    rows = len(XROW)
    im = Image.new("RGBA", (16 * T, rows * T), (0, 0, 0, 0))
    rnd = random.Random(3)

    def flat(col, speckle=None, n=30):
        t = Image.new("RGBA", (T, T), col)
        d = ImageDraw.Draw(t)
        if speckle:
            for _ in range(n):
                d.point((rnd.randrange(T), rnd.randrange(T)), fill=speckle)
        return t

    def edged(base, edge_col, width=2, mask=0):
        t = base.copy(); d = ImageDraw.Draw(t)
        if mask & 1: d.rectangle([0, 0, T - 1, width - 1], fill=edge_col)
        if mask & 2: d.rectangle([T - width, 0, T - 1, T - 1], fill=edge_col)
        if mask & 4: d.rectangle([0, T - width, T - 1, T - 1], fill=edge_col)
        if mask & 8: d.rectangle([0, 0, width - 1, T - 1], fill=edge_col)
        return t

    asphalt = flat((78, 80, 86, 255), (88, 90, 96, 255))
    sidewalk = flat((204, 200, 188, 255), (192, 188, 176, 255), 12)
    gray_c = flat((150, 150, 150, 255), (158, 158, 158, 255), 10)
    gray_o = flat((192, 192, 190, 255), (200, 200, 198, 255), 10)
    water = flat((109, 170, 220, 255), (150, 200, 240, 255), 24)
    turf = Image.new("RGBA", (T, T), (126, 200, 92, 255))
    ImageDraw.Draw(turf).rectangle([0, 8, T - 1, 15], fill=(118, 190, 86, 255)); ImageDraw.Draw(turf).rectangle([0, 24, T - 1, 31], fill=(118, 190, 86, 255))
    parking = flat((132, 132, 136, 255), (140, 140, 144, 255))
    centre = ftile(162); edges = {1: ftile(146), 2: ftile(163), 4: ftile(178), 8: ftile(161)}
    for m in range(16):
        im.paste(edged(asphalt, (170, 170, 176, 255), 2, m), (m * T, XROW["road"] * T))
        im.paste(edged(sidewalk, (168, 164, 150, 255), 1, m), (m * T, XROW["sidewalk"] * T))
        b = centre.copy()
        for bit, e in edges.items():
            if m & bit:
                box = {1: (0, 0, T, 4), 2: (T - 4, 0, T, T), 4: (0, T - 4, T, T), 8: (0, 0, 4, T)}[bit]
                b.paste(e.crop(box), box[:2])
        im.paste(b, (m * T, XROW["brick"] * T))
        im.paste(edged(gray_c, (72, 72, 72, 255), 2, m), (m * T, XROW["gray_campus"] * T))
        im.paste(edged(gray_o, (110, 110, 110, 255), 2, m), (m * T, XROW["gray_other"] * T))
        im.paste(edged(water, (190, 220, 245, 255), 2, m), (m * T, XROW["water"] * T))
        im.paste(edged(turf, (240, 240, 240, 255), 2, m), (m * T, XROW["turf"] * T))
        im.paste(edged(parking, (170, 170, 176, 255), 2, m), (m * T, XROW["parking"] * T))
    im.save(EXTRA_PNG)
    return rows


def tilesets_json(extra_rows):
    out = []
    for ts in sorted(ville["tilesets"], key=lambda t: t["firstgid"]):
        out.append({"name": ts["name"], "firstgid": ts["firstgid"], "columns": ts["columns"], "tilecount": ts["tilecount"],
                    "image": "/ga_assets/the_ville/visuals/" + ts["image"]})
    out.append({"name": "homewood_extra", "firstgid": EXTRA, "columns": 16, "tilecount": 16 * extra_rows, "image": "/static/homewood_extra.png"})
    return out


def used_tilesets(ts_list):
    used = set()
    for L in LAYERS:
        for g in grid[L]:
            if g:
                for ts in reversed(ts_list):
                    if g >= ts["firstgid"]:
                        used.add(ts["name"]); break
    return [t for t in ts_list if t["name"] in used]


def rle(values):
    out = []
    prev, run = values[0], 0
    for v in values:
        if v == prev:
            run += 1
        else:
            out += [prev, run]; prev, run = v, 1
    out += [prev, run]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="render a PNG of the map (16 px per tile)")
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()
    context = build()
    rows = write_extra_tileset()
    ts = used_tilesets(tilesets_json(rows))
    data = {"tilewidth": T, "width": W, "height": H, "m_per_tile": M_PER_TILE, "encoding": "rle",
            "tilesets": ts, "layers": {L: rle(grid[L]) for L in LAYERS}, "fg_layers": ["fg1", "fg2"],
            "cost": rle([0 if solid[i] else cost[i] for i in range(W * H)]),
            "places": PLACES, "context": context, "thumb": "/static/homewood_thumb.png",
            "credits": "Tiles: CuteRPG World (PixyMoon), Room Builder / Modern Interiors (LimeZu) via the Generative Agents repo; "
                       "map data (c) OpenStreetMap contributors (ODbL), Johns Hopkins Homewood campus."}
    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT_JSON} ({OUT_JSON.stat().st_size // 1024} KB), {W}x{H} tiles, {len(ts)} tilesets, "
          f"{sum(1 for s in solid if s)} solid cells, {len(context)} context labels")
    render(args.preview, args.grid, thumb=True)


def render(path, with_grid=False, thumb=False, scale=8):
    from PIL import Image, ImageDraw
    imgs, cache = {}, {}
    ts_list = sorted(tilesets_json(len(XROW)), key=lambda t: t["firstgid"])

    def tile_img(gid):
        if gid in cache:
            return cache[gid]
        for ts in reversed(ts_list):
            if gid >= ts["firstgid"]:
                if ts["name"] not in imgs:
                    p = EXTRA_PNG if ts["name"] == "homewood_extra" else VILLE / "visuals" / ts["image"].split("/visuals/")[1]
                    imgs[ts["name"]] = Image.open(p).convert("RGBA")
                im = imgs[ts["name"]]; l = gid - ts["firstgid"]; c = ts["columns"]
                t = im.crop(((l % c) * T, (l // c) * T, (l % c) * T + T, (l // c) * T + T))
                if scale != T:
                    t = t.resize((scale, scale), Image.NEAREST)
                cache[gid] = t
                return t
    out = Image.new("RGBA", (W * scale, H * scale), (0, 0, 0, 255))
    for L in LAYERS:
        for i, g in enumerate(grid[L]):
            if g:
                out.alpha_composite(tile_img(g), ((i % W) * scale, (i // W) * scale))
    if thumb:
        out.resize((W * 2, H * 2), Image.LANCZOS).save(THUMB_PNG) if scale != 2 else out.save(THUMB_PNG)
        print("thumb", THUMB_PNG)
    if not path:
        return
    d = ImageDraw.Draw(out)
    for pl in PLACES.values():
        x0, y0, x1, y1 = pl["box"]
        d.text((x0 * scale + 4, y0 * scale - 14), f"{pl['name']} — {pl['label']}", fill=(255, 255, 255, 255))
        for a, ar in pl["arenas"].items():
            for (sx, sy) in ar.get("spots", [])[:8]:
                d.rectangle([sx * scale + 5, sy * scale + 5, sx * scale + 10, sy * scale + 10], outline=(255, 0, 255, 255))
    if with_grid:
        for x in range(0, W, 25):
            d.line([(x * scale, 0), (x * scale, H * scale)], fill=(255, 0, 0, 90)); d.text((x * scale + 2, 2), str(x), fill=(255, 255, 0, 255))
        for y in range(0, H, 25):
            d.line([(0, y * scale), (W * scale, y * scale)], fill=(255, 0, 0, 90)); d.text((2, y * scale + 2), str(y), fill=(255, 255, 0, 255))
    out.save(path)
    print("preview", path, out.size)


if __name__ == "__main__":
    main()
