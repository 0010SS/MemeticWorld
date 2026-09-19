"""Export a vector (cartographic) layer of the Homewood campus for frontend/realmap.js.

Reads data/homewood_osm.json, projects with scripts/homewood_geo.proj (same tile frame as frontend/homewood_map.json,
1 unit = 1 tile = 32 px) and writes frontend/homewood_vector.json. Does not touch the tile map.
Usage: .venv/bin/python scripts/export_realmap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from homewood_geo import DATA, H, W, proj, simplify  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend/homewood_vector.json"
TILEMAP = ROOT / "frontend/homewood_map.json"

GREEN = {"grass", "garden", "park", "pitch", "wood", "scrub", "grassland", "recreation_ground", "playground",
         "dog_park", "allotments", "flowerbed", "track", "meadow"}
MAJOR = {"primary", "secondary", "tertiary", "residential", "unclassified", "trunk", "primary_link",
         "secondary_link", "tertiary_link", "living_street"}


def ring(coords, tol=0.35):
    pts = simplify([proj(*c) for c in coords], tol)
    return [[round(x, 1), round(y, 1)] for x, y in pts]


def visible(r):
    return any(-20 <= x <= W + 20 and -20 <= y <= H + 20 for x, y in r)


def classify(t):
    if "building" in t:
        return "building"
    if t.get("amenity") == "parking":
        return "parking"
    if t.get("natural") == "water" or "waterway" in t or t.get("leisure") == "swimming_pool":
        return "water"
    for k in ("leisure", "landuse", "natural"):
        if t.get(k) in GREEN:
            return "green"
    if "highway" in t:
        return "road"
    return None


def main():
    osm = json.loads(DATA.read_text())
    places = json.loads(TILEMAP.read_text())["places"]
    out = {"frame": {"units": "tiles", "tile_px": 32, "width": W, "height": H},
           "attribution": "© OpenStreetMap contributors", "source": osm.get("source"),
           "buildings": [], "greens": [], "water": [], "parking": [], "roads": [], "paths": [], "waterways": [],
           "places": []}
    for f in osm["features"]:
        t, typ = f["tags"], f["type"]
        cls = classify(t)
        if cls is None or typ == "point":
            continue
        if typ == "line":
            r = ring(f["coords"], 0.3)
            if len(r) < 2 or not visible(r):
                continue
            if cls == "water":
                out["waterways"].append(r)
            elif cls == "road":
                hw = t["highway"]
                (out["roads"] if hw in MAJOR else out["paths"]).append(
                    {"k": hw, "p": r, **({"n": t["name"]} if hw in MAJOR and "name" in t else {})})
            continue
        rings = [f["coords"]] if typ == "polygon" else [o for o in f.get("outer", [])]
        for rc in rings:
            r = ring(rc)
            if len(r) < 3 or not visible(r):
                continue
            if cls == "building":
                out["buildings"].append({"p": r, **({"n": t["name"]} if "name" in t else {})})
            elif cls == "road":
                out["paths"].append({"k": "area", "p": r, "area": 1})
            else:
                out["greens" if cls == "green" else cls].append(r)
    for name, pl in places.items():
        boxes = pl.get("boxes") or [pl["box"]]
        out["places"].append({"id": name, "label": pl.get("label", name), "kind": pl.get("kind"), "boxes": boxes})
    s = json.dumps(out, separators=(",", ":"))
    OUT.write_text(s)
    print(f"wrote {OUT} {len(s)/1e6:.2f} MB  buildings={len(out['buildings'])} roads={len(out['roads'])} "
          f"paths={len(out['paths'])} greens={len(out['greens'])} places={len(out['places'])}")


if __name__ == "__main__":
    main()
