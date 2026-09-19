#!/usr/bin/env python3
"""Render a schematic outline of the real-scale Homewood tile map (no tiles, just classes and labels).

    python3 scripts/homewood_outline.py [out.png]

Grey = buildings outside the experiment (dark grey = campus buildings), blue = the eight experiment places,
red-brown = walkways, dark = roads, greens = lawns / fields / woods. One tile = 4 m; grid every 100 m.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from homewood_geo import Geo, W, H, M_PER_TILE  # noqa: E402

S = 3  # px per tile
COL = {"": (223, 238, 200), "lawn": (184, 221, 138), "pitch": (159, 211, 122), "track": (217, 160, 102), "wood": (127, 176, 105),
       "water": (127, 183, 224), "parking": (216, 212, 204), "plaza": (233, 217, 163), "road": (109, 109, 109),
       "walk": (217, 154, 108), "steps": (199, 138, 90), "sidewalk": (204, 200, 188), "trail": (226, 206, 150), "building": (191, 191, 191), "campus": (150, 150, 150),
       "place": (79, 131, 209)}


def font(size, bold=False):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main(out):
    g = Geo()
    im = Image.new("RGB", (W * S, H * S), COL[""])
    d = ImageDraw.Draw(im)
    for i, c in enumerate(g.cls):
        if not c:
            continue
        x, y = i % W, i // W
        col = COL[c]
        if c == "building" and g.bid[i] >= 0 and g.buildings[g.bid[i]]["campus"]:
            col = COL["campus"]
        d.rectangle([x * S, y * S, x * S + S - 1, y * S + S - 1], fill=col)
    # building outlines
    for i, c in enumerate(g.cls):
        if c not in ("building", "place"):
            continue
        x, y = i % W, i // W
        for dx, dy, side in ((1, 0, "r"), (-1, 0, "l"), (0, 1, "b"), (0, -1, "t")):
            nx, ny = x + dx, y + dy
            j = ny * W + nx
            other = g.cls[j] if 0 <= nx < W and 0 <= ny < H else ""
            if other != c or (c == "building" and g.bid[j] != g.bid[i]):
                X, Y = x * S, y * S
                line = {"r": [X + S - 1, Y, X + S - 1, Y + S - 1], "l": [X, Y, X, Y + S - 1], "b": [X, Y + S - 1, X + S - 1, Y + S - 1],
                        "t": [X, Y, X + S - 1, Y]}[side]
                d.line(line, fill=(70, 70, 70) if c == "building" else (30, 60, 120), width=1)
    # 100 m grid
    for x in range(0, W, 25):
        d.line([x * S, 0, x * S, H * S], fill=(255, 255, 255), width=1)
        d.text((x * S + 2, 2), f"{x * M_PER_TILE:.0f} m", fill=(90, 90, 90), font=font(10))
    for y in range(0, H, 25):
        d.line([0, y * S, W * S, y * S], fill=(255, 255, 255), width=1)
        d.text((2, y * S + 2), f"{y * M_PER_TILE:.0f} m", fill=(90, 90, 90), font=font(10))
    # labels: campus buildings and larger named buildings
    f11, f10, f13 = font(11), font(10), font(13, bold=True)
    for b in g.buildings:
        if not b["name"] or "place" in b:
            continue
        big = len(b["cells"]) >= 30
        if b["campus"] or big:
            cx, cy = b["centroid"]
            txt = b["name"]
            fnt = f11 if b["campus"] else f10
            w = d.textlength(txt, font=fnt)
            d.text((cx * S - w / 2, cy * S - 6), txt, fill=(20, 20, 20) if b["campus"] else (70, 70, 70), font=fnt)
    for name, f in g.names.items():
        t = f["tags"]
        if f["type"] in ("polygon", "multipolygon") and (t.get("leisure") in ("park", "pitch", "garden") or t.get("natural") == "wood") and "Keyser" not in name:
            from homewood_geo import cells_polygon
            rings = g._rings(f)
            cells = cells_polygon(rings)
            if len(cells) < 40:
                continue
            cx, cy = g._centroid(cells)
            w = d.textlength(name, font=f10)
            d.text((cx * S - w / 2, cy * S - 6), name, fill=(40, 90, 40), font=f10)
    # road names (once per name, at the midpoint of the longest way)
    from homewood_geo import proj
    best = {}
    for f in g.features:
        t = f["tags"]
        if f["type"] == "line" and t.get("highway") in ("primary", "secondary", "tertiary", "residential", "unclassified") and t.get("name"):
            pts = [proj(*c) for c in f["coords"]]
            L = sum(((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2) ** .5 for i in range(len(pts) - 1))
            if L > best.get(t["name"], (0,))[0]:
                best[t["name"]] = (L, pts[len(pts) // 2])
    for name, (L, (cx, cy)) in best.items():
        if L < 15 or not (0 < cx < W and 0 < cy < H):
            continue
        w = d.textlength(name, font=f11)
        d.rectangle([cx * S - w / 2 - 2, cy * S - 7, cx * S + w / 2 + 2, cy * S + 7], fill=(60, 60, 60))
        d.text((cx * S - w / 2, cy * S - 6), name, fill=(255, 255, 255), font=f11)
    for place, p in g.places.items():
        x0, y0, x1, y1 = p["boxes"][0]
        txt = f"{place} · {p['label']}"
        w = d.textlength(txt, font=f13)
        bx, by = x0 * S, y0 * S - 18
        if place == "Quad":
            bx, by = x0 * S + 20, (y1 + 1) * S + 4
        d.rectangle([bx - 3, by - 2, bx + w + 3, by + 15], fill=(20, 40, 90))
        d.text((bx, by), txt, fill=(255, 255, 255), font=f13)
        for (a, b_, c_, e) in p["boxes"]:
            d.rectangle([a * S, b_ * S, (c_ + 1) * S - 1, (e + 1) * S - 1], outline=(20, 40, 90), width=2)
    # legend, scale bar, north arrow
    lx, ly = W * S - 250, H * S - 235
    d.rectangle([lx - 8, ly - 8, W * S - 8, H * S - 8], fill=(255, 255, 255), outline=(120, 120, 120))
    items = [("place", "experiment place (8)"), ("campus", "campus building (grey area)"), ("building", "other building"),
             ("road", "road"), ("walk", "walkway / path"), ("plaza", "plaza"), ("parking", "parking"), ("lawn", "lawn / park"),
             ("pitch", "sports field"), ("wood", "woods"), ("water", "Stony Run")]
    for k, (cls, label) in enumerate(items):
        d.rectangle([lx, ly + k * 16, lx + 14, ly + k * 16 + 12], fill=COL[cls], outline=(80, 80, 80))
        d.text((lx + 20, ly + k * 16 - 1), label, fill=(20, 20, 20), font=f11)
    sy = ly + len(items) * 16 + 8
    d.rectangle([lx, sy, lx + 25 * S, sy + 6], fill=(20, 20, 20))
    d.text((lx, sy + 8), f"100 m = 25 tiles ({M_PER_TILE:.0f} m/tile); map {W}x{H} tiles", fill=(20, 20, 20), font=f10)
    d.text((lx + 170, sy - 2), "N ↑", fill=(20, 20, 20), font=f13)
    im.save(out)
    print("outline", out, im.size)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "data/homewood_outline.png"))
