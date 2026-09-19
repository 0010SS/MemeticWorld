"""Real-scale geometry of the Homewood campus for the pixel map.

Projects the OpenStreetMap features in data/homewood_osm.json onto a north-up tile grid (4 m per tile, no
compression) and rasterises them into per-cell classes: lawns, sports fields, woods, water, parking, roads,
walkways, plazas, buildings, and the eight experiment places. Used by build_homewood_map.py (tiles) and by the
--outline schematic.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/homewood_osm.json"

# ---------------------------------------------------------------- extent and projection (WGS84 -> tiles)
LAT_N, LAT_S, LON_W, LON_E = 39.3352, 39.3248, -76.6268, -76.6156
M_PER_TILE = 2.0
_LAT_MID = (LAT_N + LAT_S) / 2
M_PER_DEG_LON = 111320 * math.cos(math.radians(_LAT_MID))
M_PER_DEG_LAT = 110574
W = math.ceil((LON_E - LON_W) * M_PER_DEG_LON / M_PER_TILE)
H = math.ceil((LAT_N - LAT_S) * M_PER_DEG_LAT / M_PER_TILE)


def proj(lon: float, lat: float) -> tuple[float, float]:
    return (lon - LON_W) * M_PER_DEG_LON / M_PER_TILE, (LAT_N - lat) * M_PER_DEG_LAT / M_PER_TILE


def inb(x, y):
    return 0 <= x < W and 0 <= y < H


# ---------------------------------------------------------------- pixel-art straightening
def simplify(pts, tol):
    """Douglas-Peucker: drop vertices that deviate less than `tol` tiles from the line."""
    if len(pts) < 3:
        return list(pts)
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        ax, ay = pts[i]; bx, by = pts[j]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        maxd, idx = -1.0, -1
        for k in range(i + 1, j):
            px, py = pts[k]
            if L2 == 0:
                d = math.hypot(px - ax, py - ay)
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
                d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
            if d > maxd:
                maxd, idx = d, k
        if maxd > tol:
            keep[idx] = True
            stack.append((i, idx)); stack.append((idx, j))
    return [p for p, k in zip(pts, keep) if k]


def octi(pts, snap=1.0):
    """Re-route a polyline with 0/45/90-degree pieces only. A segment within `snap` tiles of an axis or a diagonal is
    snapped onto it; anything else becomes straight-diagonal-straight with the original endpoint kept, so
    networks stay connected and positions stay put."""
    out = [tuple(pts[0])]
    cur = tuple(pts[0])
    for i in range(1, len(pts)):
        tx, ty = pts[i]
        dx, dy = tx - cur[0], ty - cur[1]
        adx, ady = abs(dx), abs(dy)
        if adx < 1e-9 and ady < 1e-9:
            continue
        nxt = None
        if ady <= snap and adx >= 2 * ady:
            nxt = (tx, cur[1])
        elif adx <= snap and ady >= 2 * adx:
            nxt = (cur[0], ty)
        elif abs(adx - ady) <= snap:
            L = max(adx, ady)
            nxt = (cur[0] + math.copysign(L, dx), cur[1] + math.copysign(L, dy))
        if nxt is not None:
            out.append(nxt); cur = nxt
            continue
        if adx > ady:
            d, st = ady, (adx - ady) / 2
            p1 = (cur[0] + math.copysign(st, dx), cur[1])
        else:
            d, st = adx, (ady - adx) / 2
            p1 = (cur[0], cur[1] + math.copysign(st, dy))
        p2 = (p1[0] + math.copysign(d, dx), p1[1] + math.copysign(d, dy))
        out += [p1, p2, (tx, ty)]
        cur = (tx, ty)
    return out


def dominant_angle(ring):
    """Edge-length-weighted orientation of an outline modulo 90 degrees, and how coherent it is (0..1)."""
    import cmath
    acc, total = 0j, 0.0
    for i in range(len(ring) - 1):
        (ax, ay), (bx, by) = ring[i], ring[i + 1]
        L = math.hypot(bx - ax, by - ay)
        if L < 0.5:
            continue
        acc += L * cmath.exp(4j * math.atan2(by - ay, bx - ax))
        total += L
    if total == 0:
        return 0.0, 0.0
    return cmath.phase(acc) / 4, abs(acc) / total


def rotate_ring(ring, theta):
    cx = sum(p[0] for p in ring) / len(ring); cy = sum(p[1] for p in ring) / len(ring)
    c, s = math.cos(theta), math.sin(theta)
    return [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in ring]


def straighten_ring(ring, tol=1.0, snap=2.0):
    """Outlines a few degrees off the grid (or off 45 degrees) are rotated about their centre onto it, then
    simplified and routed with 0/45/90-degree edges. Positions and sizes are kept."""
    th, coh = dominant_angle(ring)
    deg = math.degrees(th)
    if coh > 0.55:
        if abs(deg) <= 8:
            ring = rotate_ring(ring, -th)
        elif abs(abs(deg) - 45) <= 8:
            ring = rotate_ring(ring, math.copysign(math.pi / 4, deg) - th)
    r = simplify(ring, tol)
    if len(r) < 4:
        return ring
    r = octi(r, snap)
    if r[0] != r[-1]:
        r.append(r[0])
    return r


# ---------------------------------------------------------------- rasterisation
def cells_polygon(rings: list[list[tuple[float, float]]]) -> set[tuple[int, int]]:
    """Cells whose centre is inside the (multi)polygon, even-odd rule over all rings."""
    xs = [p[0] for r in rings for p in r]
    ys = [p[1] for r in rings for p in r]
    if not xs:
        return set()
    out = set()
    x0, x1 = max(0, int(math.floor(min(xs)))), min(W - 1, int(math.ceil(max(xs))))
    y0, y1 = max(0, int(math.floor(min(ys)))), min(H - 1, int(math.ceil(max(ys))))
    # edge list
    edges = []
    for r in rings:
        for i in range(len(r) - 1):
            (ax, ay), (bx, by) = r[i], r[i + 1]
            if ay != by:
                edges.append((ax, ay, bx, by))
    for y in range(y0, y1 + 1):
        py = y + 0.5
        xs_cross = []
        for ax, ay, bx, by in edges:
            if (ay > py) != (by > py):
                xs_cross.append(ax + (py - ay) * (bx - ax) / (by - ay))
        xs_cross.sort()
        for i in range(0, len(xs_cross) - 1, 2):
            a, b = xs_cross[i], xs_cross[i + 1]
            for x in range(max(x0, int(math.ceil(a - 0.5))), min(x1, int(math.floor(b - 0.5))) + 1):
                out.add((x, y))
    return out


def cells_line(pts: list[tuple[float, float]], width: float) -> set[tuple[int, int]]:
    """Cells within width/2 of the polyline (4-connected for thin lines)."""
    out = set()
    r = width / 2
    for i in range(len(pts) - 1):
        (ax, ay), (bx, by) = pts[i], pts[i + 1]
        L = math.hypot(bx - ax, by - ay)
        if L == 0:
            continue
        # sample along the segment so thin lines stay connected
        n = max(1, int(L / 0.35))
        prev = None
        for k in range(n + 1):
            t = k / n
            px, py = ax + (bx - ax) * t, ay + (by - ay) * t
            c = (int(math.floor(px)), int(math.floor(py)))
            if inb(*c):
                out.add(c)
                if prev and prev != c and abs(prev[0] - c[0]) == 1 and abs(prev[1] - c[1]) == 1 and inb(c[0], prev[1]):
                    out.add((c[0], prev[1]))    # keep diagonal steps 4-connected
            prev = c
        if width > 1:
            x0, x1 = int(math.floor(min(ax, bx) - r)), int(math.ceil(max(ax, bx) + r))
            y0, y1 = int(math.floor(min(ay, by) - r)), int(math.ceil(max(ay, by) + r))
            for y in range(max(0, y0), min(H - 1, y1) + 1):
                for x in range(max(0, x0), min(W - 1, x1) + 1):
                    cx, cy = x + 0.5, y + 0.5
                    t = max(0.0, min(1.0, ((cx - ax) * (bx - ax) + (cy - ay) * (by - ay)) / (L * L)))
                    if math.hypot(cx - (ax + (bx - ax) * t), cy - (ay + (by - ay) * t)) <= r:
                        out.add((x, y))
    return out


# ---------------------------------------------------------------- feature classes
# class -> painting priority (higher paints later / on top)
PRIORITY = {"lawn": 1, "pitch": 2, "track": 2, "wood": 3, "parking": 4, "sidewalk": 4.5, "water": 5, "plaza": 6, "road": 7,
            "walk": 8, "steps": 8, "trail": 8, "building": 9, "place": 10}
ROAD_WIDTH = {"primary": 10, "secondary": 8, "tertiary": 6, "residential": 5, "unclassified": 5, "living_street": 5,
              "service": 3, "construction": 3}      # tiles (2 m each)


def classify(tags: dict):
    """-> (class, line_width or None) or None if the feature is not drawn."""
    hw = tags.get("highway")
    if "building" in tags:
        return ("building", None)
    if hw:
        if tags.get("layer", "0").startswith("-") or tags.get("tunnel"):
            return None              # underground: garage aisles, tunnels
        if hw in ROAD_WIDTH:
            return ("road", ROAD_WIDTH[hw])
        if hw == "pedestrian":
            return ("plaza", 4)
        if hw == "steps":
            return ("steps", 2)
        if hw == "footway" and tags.get("footway") == "sidewalk":
            return None                  # sidewalks are generated as a clean band along every road
        if hw == "path":
            return ("trail", 2)
        if hw in ("footway", "cycleway", "corridor", "track"):
            return ("walk", 2)
        return None
    if tags.get("natural") == "water" or tags.get("waterway") in ("stream", "river"):
        return ("water", 3)
    if tags.get("waterway") in ("drain", "ditch"):
        return None                      # culverted drains: not visible on the ground
    if tags.get("natural") == "wood" or tags.get("landuse") == "forest":
        return ("wood", None)
    if tags.get("leisure") == "pitch":
        return ("pitch", None)
    if tags.get("leisure") == "track":
        return ("track", None)
    if tags.get("amenity") == "parking":
        if tags.get("layer", "0").startswith("-") or tags.get("parking") in ("underground", "multi-storey") or tags.get("location") == "underground":
            return None              # e.g. the garage under Decker Quad: the quad above it is grass
        return ("parking", None)
    if tags.get("leisure") in ("park", "garden", "playground", "recreation_ground", "dog_park") or \
            tags.get("landuse") in ("grass", "recreation_ground", "village_green", "meadow", "cemetery", "greenfield"):
        return ("lawn", None)
    return None


# the experiment's places and the real buildings they are drawn as
PLACE_BUILDINGS = {
    "Gym": ["O'Connor Recreation Center"],
    "Dining Hall": ["AMR III Building C"],          # Hopkins Cafe / Fresh Food Cafe, under AMR III
    "Dorm": ["AMR II"],
    "Classroom": ["Gilman Hall"],
    "Library": ["Milton S. Eisenhower Library", "Brody Learning Commons"],
    "Cafe": ["Levering Hall"],
    "Research Lab": ["Hackerman Hall"],
    "Quad": ["Keyser Quad"],
}
PLACE_LABELS = {"Gym": "O'Connor Rec Center", "Dining Hall": "Hopkins Cafe (FFC)", "Dorm": "AMR II",
                "Classroom": "Gilman Hall", "Library": "MSE Library / Brody", "Cafe": "Levering Cafe",
                "Research Lab": "Hackerman Hall", "Quad": "Keyser Quad"}


class Geo:
    """Rasterised campus: cls[i] class name or '', bid[i] building index, plus building/place metadata."""

    def __init__(self):
        d = json.load(open(DATA))
        self.features = d["features"]
        self.cls = [""] * (W * H)
        self.bid = [-1] * (W * H)
        self.buildings = []      # dicts: name, tags, cells, box, centroid, campus(bool)
        self.places = {}         # place -> dict(boxes=[(x0,y0,x1,y1)], cells=set(), label=...)
        self.trees = []          # (x, y) of mapped single trees
        self.names = {}          # name -> feature for lookup
        self._rasterise()

    def _rings(self, f):
        tol = 1.0 if "building" in f["tags"] else 2.0
        if f["type"] == "polygon":
            return [straighten_ring([proj(*p) for p in f["coords"]], tol)]
        if f["type"] == "multipolygon":
            return [straighten_ring([proj(*p) for p in r], tol) for r in f["outer"] + f["inner"]]
        return None

    def _paint(self, cells, name, bidx=-1):
        pr = PRIORITY[name]
        for (x, y) in cells:
            i = y * W + x
            cur = self.cls[i]
            if not cur or PRIORITY[cur] <= pr:
                self.cls[i] = name
                self.bid[i] = bidx

    def _rasterise(self):
        drawn = []
        underground = set()
        for f in self.features:
            t = f["tags"]
            if f["type"] in ("polygon", "multipolygon") and t.get("amenity") == "parking" and \
                    (t.get("layer", "0").startswith("-") or t.get("parking") in ("underground", "multi-storey") or t.get("location") == "underground"):
                underground |= cells_polygon(self._rings(f))
        for f in self.features:
            t = f["tags"]
            if t.get("name"):
                self.names[t["name"]] = f
            if f["type"] == "point":
                if t.get("natural") == "tree":
                    x, y = proj(*f["coords"])
                    if inb(int(x), int(y)):
                        self.trees.append((int(x), int(y)))
                continue
            c = classify(t)
            if not c:
                continue
            cname, width = c
            if f["type"] == "line":
                if cname == "building":
                    continue
                pts = [proj(*p) for p in f["coords"]]
                if cname in ("walk", "steps", "trail") and len(pts) > 1 and \
                        sum(math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(len(pts) - 1)) < 5:
                    continue          # tiny stubs only add noise
                tol = 4.0 if cname == "water" else 3.5 if width > 2 else 2.5
                cells = cells_line(octi(simplify(pts, tol), 2.0), width)
                if t.get("highway") == "service" and len(cells & underground) > len(cells) * 0.5:
                    continue          # garage aisles under a quad
            else:
                rings = self._rings(f)
                cells = cells_polygon(rings)
                if cname in ("road", "water", "plaza") and not cells:
                    continue
                if cname == "lawn" and not t.get("name") and len(cells) < 150:
                    continue          # small unnamed grass and garden slivers: plain grass, no outline
            if cname == "building":
                if t.get("location") == "underground" and t.get("name") not in sum(PLACE_BUILDINGS.values(), []):
                    continue
                if t.get("building") == "roof":
                    continue
                if len(cells) < 2:
                    continue
                campus = t.get("building") in ("university", "dormitory", "library") or t.get("name") in ("AMR III Building C",)
                self.buildings.append(dict(name=t.get("name", ""), tags=t, cells=cells, campus=campus,
                                           box=self._box(cells), centroid=self._centroid(cells)))
                drawn.append((PRIORITY["building"], cells, "building", len(self.buildings) - 1))
            else:
                drawn.append((PRIORITY[cname], cells, cname, -1))
        for pr, cells, cname, bidx in sorted(drawn, key=lambda d: d[0]):
            self._paint(cells, cname, bidx)
        road = {(i % W, i // W) for i, c in enumerate(self.cls) if c == "road"}
        band = set()
        for (x, y) in road:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (x + dx, y + dy)
                    if n not in road and inb(*n):
                        band.add(n)
        self._paint(band, "sidewalk")
        self._places()

    @staticmethod
    def _box(cells):
        xs = [x for x, _ in cells]
        ys = [y for _, y in cells]
        return [min(xs), min(ys), max(xs), max(ys)]

    @staticmethod
    def _centroid(cells):
        return (sum(x for x, _ in cells) / len(cells), sum(y for _, y in cells) / len(cells))

    def building_named(self, name):
        for i, b in enumerate(self.buildings):
            if b["name"] == name:
                return i, b
        raise KeyError(name)

    def _places(self):
        for place, names in PLACE_BUILDINGS.items():
            if place == "Quad":
                f = self.names["Keyser Quad"]
                cells = cells_polygon(self._rings(f))
                self.places[place] = dict(label=PLACE_LABELS[place], kind="lawn", boxes=[self._box(cells)], cells=cells)
                continue
            cells = set()
            for n in names:
                bi, b = self.building_named(n)
                cells |= b["cells"]
                b["place"] = place
            if place == "Library":      # MSE and Brody are one connected building: bridge the gap between them
                mse = self.building_named(names[0])[1]["cells"]; brody = self.building_named(names[1])[1]["cells"]
                my1 = max(y for _, y in mse); by0 = min(y for _, y in brody)
                xs = sorted(x for x, y in mse if y == my1)
                for y in range(my1 + 1, by0):
                    for x in xs:
                        cells.add((x, y))
            self.places[place] = dict(label=PLACE_LABELS[place], kind="building", boxes=[self._box(cells)], cells=cells)
        for place, p in self.places.items():
            if p["kind"] == "building":
                for (x, y) in p["cells"]:
                    i = y * W + x
                    self.cls[i] = "place"
                    self.bid[i] = -1

    def summary(self):
        cnt = defaultdict(int)
        for c in self.cls:
            cnt[c or "grass"] += 1
        return dict(cnt)


if __name__ == "__main__":
    g = Geo()
    print(f"grid {W}x{H} tiles at {M_PER_TILE} m/tile;", g.summary())
    print(f"{len(g.buildings)} buildings, {sum(1 for b in g.buildings if b['campus'])} campus buildings, {len(g.trees)} mapped trees")
    for place, p in g.places.items():
        print(f"  {place:13} {p['label']:22} boxes={p['boxes']}  sizes={[(b[2]-b[0]+1, b[3]-b[1]+1) for b in p['boxes']]}")
