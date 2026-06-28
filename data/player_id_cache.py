"""
Persistent name -> balldontlie player_id cache.

Why this exists: discover_mlb_probable_pitchers() re-resolves every probable
pitcher's name to a balldontlie player_id via get_mlb_player_search() on
EVERY run. Most MLB starters repeat day after day across a season, so that's
wasted, rate-limit-risking calls for names we've already resolved before.
This file persists that name->id mapping to disk (same JSON-file pattern as
data/cache_history.py) so a GitHub Actions commit step can carry it across
runs, the same way pick history already is.

File: output/player_id_cache.json (same directory cache_history.py writes
output/pick_history.jsonl to, and that the repo's GitHub Action already
commits -- NOT data/, so this rides along with that existing commit step
instead of needing a new one).
Format:
{
  "<normalized name>": {
    "player_id": <int>,
    "name": "<original display name as first resolved>",
    "last_seen": "<YYYY-MM-DD>"
  },
  ...
}

Usage pattern (batch-friendly -- one disk read/write per pipeline run, not
one per pitcher):

    from data.player_id_cache import load_cache, save_cache, lookup, remember

    cache = load_cache()
    pid = lookup(cache, "Logan Webb")
    if pid is None:
        pid = <resolve via balldontlie search>
        remember(cache, "Logan Webb", pid)
    save_cache(cache)   # call once, after the loop, not per pitcher
"""
import json
import os
from datetime import datetime, timezone

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "player_id_cache.json")


def _ensure_dir():
    os.makedirs(os.path.dirname(os.path.abspath(CACHE_PATH)), exist_ok=True)


def _normalize(name):
    """Collapse whitespace/case so 'Logan Webb' and ' logan  webb ' hit the same key."""
    return " ".join(name.strip().lower().split())


def load_cache():
    """Returns the cache dict, or {} if the file doesn't exist yet or is corrupt."""
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [warn] player_id_cache.json unreadable ({e}) -- starting fresh")
        return {}


def save_cache(cache):
    """Atomic write (write to temp file, then rename) so a crash mid-write
    can't leave a half-written, corrupt cache file behind. cache_history.py
    doesn't need this trick since it only ever appends or does a full
    read-then-rewrite of JSONL lines; this file gets fully overwritten on
    every run, so the atomic swap is worth the few extra lines here."""
    _ensure_dir()
    tmp_path = CACHE_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp_path, CACHE_PATH)


def lookup(cache, name):
    """Returns the cached player_id for `name`, or None if not cached.
    Takes an already-loaded cache dict -- doesn't hit disk."""
    entry = cache.get(_normalize(name))
    return entry["player_id"] if entry else None


def remember(cache, name, player_id):
    """Updates `cache` in place with this name->player_id mapping (and
    refreshes last_seen if it was already cached). Doesn't write to disk --
    call save_cache(cache) once after your loop."""
    cache[_normalize(name)] = {
        "player_id": player_id,
        "name": name,
        "last_seen": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


# ---- Single-call convenience wrappers, for call sites that don't already
# hold a loaded cache and just need one lookup/write. Avoid these in a loop
# (they round-trip the file every call) -- use load_cache()/lookup()/
# remember()/save_cache() for batches like discover_mlb_probable_pitchers().

def get_cached_player_id(name):
    return lookup(load_cache(), name)


def set_cached_player_id(name, player_id):
    cache = load_cache()
    remember(cache, name, player_id)
    save_cache(cache)
