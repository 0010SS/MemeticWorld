Vendored (unmodified, sparse) copy of https://github.com/joonspk-research/generative_agents
Commit: fe05a71d3e4ed7d10bf68aa4eda6dd995ec070f4 (2023-08-11), Apache-2.0 (see LICENSE).
Checked out: reverie/backend_server, the character sprite assets, and (added 2026-09-19 for the pixel-art campus)
the Smallville map assets under environment/frontend_server/static_dirs/assets/the_ville:
  visuals/the_ville_jan7.json          the Tiled map (source of the furniture and tree patches we copy)
  visuals/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_*.png   CuteRPG World tilesets (PixyMoon)
  visuals/map_assets/v1/Room_Builder_32x32.png, interiors_pt1-5.png  Room Builder / Modern Interiors (LimeZu)
  visuals/map_assets/blocks/blocks_*.png   the block-id tiles used by the map's hidden layers
  matrix/special_blocks/*.csv               names of Smallville's sectors, arenas and objects
MemeWorld imports the Python modules through backend/ga_compat.py; no upstream file is edited.
scripts/build_homewood_map.py reads the map assets to build frontend/homewood_map.json.
