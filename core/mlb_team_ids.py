"""
core/mlb_team_ids.py

Single canonical source for MLB team-abbreviation -> MLB Stats API team id.
Deliberately dependency-free (stdlib only, no other core.* imports) so it
can be imported from anywhere -- including core/mlb.py, which historical
grading calls directly and needs to stay lightweight/fast, and
core/intelligence/bullpen_intel.py, which pulls in the full intelligence
package (heavier import chain, has its own circular-import footguns).

Previously this table was hand-duplicated in both of those files ("kept in
sync manually" per the old comment in core/mlb.py) -- pure drift risk, since
nothing enforced the two copies matching. Consolidated here instead so
there is exactly one definition to update.
"""
from __future__ import annotations

MLB_TEAM_IDS: dict[str, int] = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KC":  118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD":  135, "SF":  137, "SEA": 136,
    "STL": 138, "TB":  139, "TEX": 140, "TOR": 141, "WSH": 120,
}
