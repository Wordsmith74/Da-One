"""
models/line_movement.py

NEWLY AUTHORED. The file uploaded under this name was the same content as
core/intelligence/line_movement.py (get_line_movement_signals -- main.py's
intelligence-layer signal fetcher), not this module. run_pipeline-1.py
imports a different function, apply_line_movement_filter, from this exact
path -- a post-generation safety filter, not a signal source. No version of
that function was found in any upload, so this is a fresh implementation
built from its call site in run_pipeline-1.py and the existing
steam_move_threshold_pct / moneyline_steam_cents constants in
sport_config.py, which were clearly defined for this purpose.

If a real version of this file exists elsewhere, replace this one.
"""
from __future__ import annotations

from models.sport_config import MLB, WNBA


def _config_for_pick(pick: dict) -> dict:
    sport = str(pick.get("sport", "")).upper()
    return MLB if sport.startswith("MLB") else WNBA


def apply_line_movement_filter(picks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Drop any pick where the market has moved hard enough since pick
    generation that the original edge calculation should no longer be
    trusted -- a real edge against a stale line isn't a real edge against
    today's line.

    Reads pick["pick_time_line"] (the line at generation time) and
    pick["current_line"] (the line as of this check). In the current
    single-pass pipeline these are set identically at generation time
    (see process_mlb_f5/process_mlb_k_prop/process_wnba_prop), so this
    filter is a no-op until something upstream re-fetches current odds and
    overwrites current_line before this call -- this function is the hook
    point for that, not a live-odds fetcher itself.

    For market_type == "total" (every market this pipeline currently
    produces), movement is measured as a percentage of the line itself
    against the sport's steam_move_threshold_pct. moneyline_steam_cents is
    defined in sport_config.py for moneyline markets and is wired here for
    when/if a moneyline pick type is added, but isn't exercised by the
    current totals/props-only pipeline.

    Returns (kept_picks, dropped_picks). Each kept pick gets a
    line_move_pct field added (0.0 if nothing to compare); each dropped
    pick gets a line_move_drop_reason field explaining why.
    """
    kept: list[dict] = []
    dropped: list[dict] = []

    for pick in picks:
        cfg = _config_for_pick(pick)
        pick_time_line = pick.get("pick_time_line")
        current_line = pick.get("current_line")

        if pick.get("market_type") == "moneyline":
            pick_time_odds = pick.get("pick_time_odds")
            current_odds = pick.get("current_odds")
            if pick_time_odds is None or current_odds is None:
                pick["line_move_pct"] = 0.0
                kept.append(pick)
                continue
            move_cents = abs(current_odds - pick_time_odds)
            pick["line_move_pct"] = move_cents
            if move_cents >= cfg["moneyline_steam_cents"]:
                pick["line_move_drop_reason"] = (
                    f"moneyline moved {move_cents} cents "
                    f"(>= {cfg['moneyline_steam_cents']} cent threshold) "
                    f"since pick generation"
                )
                dropped.append(pick)
            else:
                kept.append(pick)
            continue

        if pick_time_line is None or current_line is None or pick_time_line == 0:
            pick["line_move_pct"] = 0.0
            kept.append(pick)
            continue

        move_pct = abs(current_line - pick_time_line) / abs(pick_time_line) * 100
        pick["line_move_pct"] = round(move_pct, 2)

        threshold = cfg["steam_move_threshold_pct"]
        if move_pct >= threshold:
            pick["line_move_drop_reason"] = (
                f"line moved {move_pct:.1f}% (>= {threshold}% threshold) "
                f"since pick generation"
            )
            dropped.append(pick)
        else:
            kept.append(pick)

    return kept, dropped
