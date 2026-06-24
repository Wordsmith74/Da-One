"""
Line movement filter: drops a generated pick if the market line has moved
hard since the pick was generated -- "steam" usually means sharper money
already found the same edge (or news broke) and the original edge is stale
or gone. Betting into a line that already moved away from you is one of the
most common ways a profitable-looking model bleeds money in practice.

Threshold is sport-specific (models/sport_config.py) because MLB totals/props
and WNBA props don't move the same amount for the same reason -- see comments
in sport_config.py for why the numbers differ.
"""
from models.sport_config import MLB, WNBA

_CONFIGS = {"MLB F5": MLB, "MLB Ks": MLB, "WNBA": WNBA}


def apply_line_movement_filter(picks):
    """Returns (final_picks, dropped_picks). Expects each pick to carry
    'pick_time_line' and 'current_line' -- if a real-time current-line refresh
    isn't wired up yet, current_line will equal pick_time_line and nothing
    gets dropped here, which is the correct conservative default (don't
    invent movement that wasn't actually observed)."""
    final = []
    dropped = []
    for p in picks:
        cfg = _CONFIGS.get(p.get("sport"))
        if cfg is None:
            # Unknown sport tag -- fail safe by keeping the pick, but this
            # should never happen if every pick is tagged correctly upstream.
            final.append(p)
            continue

        pick_time_line = p.get("pick_time_line")
        current_line = p.get("current_line")
        if pick_time_line is None or current_line is None or pick_time_line == 0:
            final.append(p)
            continue

        move_pct = abs(current_line - pick_time_line) / abs(pick_time_line) * 100
        if move_pct >= cfg["steam_move_threshold_pct"]:
            p["_dropped_reason"] = f"line moved {move_pct:.1f}% since pick generation (threshold {cfg['steam_move_threshold_pct']}%)"
            dropped.append(p)
        else:
            final.append(p)

    return final, dropped
