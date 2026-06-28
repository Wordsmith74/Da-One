"""
Canonical name registry.

Why this exists: this pipeline pulls names from THREE different sources that
each format them differently --
  - balldontlie: full names like "Los Angeles Dodgers", abbreviations like "LAD"
  - The Odds API: full names like "Los Angeles Dodgers" (usually matches BDL)
  - ESPN: displayName ("Los Angeles Dodgers"), abbreviation ("LAD"), shortName
    ("LAD @ SD")
  - FanGraphs Guts park-factor table: team names as plain text in a column,
    occasionally "LA Dodgers" or "LAD" depending on the season's table export
  - pybaseball/Savant: player names sometimes "Last, First", sometimes
    "First Last", sometimes with accents stripped/kept inconsistently

A silent mismatch here (e.g. park-factor lookup using `.str.contains(home_name)`
against a table that spells the team differently) doesn't crash -- it just
quietly returns a neutral default and the pick gets generated on wrong inputs.
That's worse than a crash. Every cross-source name lookup in this pipeline
should go through canonical_team() / canonical_park() / canonical_player()
below instead of ad-hoc string matching.

This is a starter registry, not exhaustive -- extend MLB_TEAMS / WNBA_TEAMS
as mismatches are discovered in real runs (the discovery log functions at the
bottom print anything that fails to resolve, specifically so gaps get found
and filed here instead of failing silently downstream).
"""
import re
import unicodedata

# canonical_key -> {full, abbr, aliases (list), park}
MLB_TEAMS = {
    "dodgers":  {"full": "Los Angeles Dodgers", "abbr": "LAD", "park": "Dodger Stadium",
                 "aliases": ["la dodgers", "lad", "los angeles dodgers"]},
    "padres":   {"full": "San Diego Padres", "abbr": "SD", "park": "Petco Park",
                 "aliases": ["san diego", "sd", "sdp"]},
    "astros":   {"full": "Houston Astros", "abbr": "HOU", "park": "Daikin Park",
                 "aliases": ["houston", "hou"]},  # renamed from Minute Maid Park in 2025
    "mariners": {"full": "Seattle Mariners", "abbr": "SEA", "park": "T-Mobile Park",
                 "aliases": ["seattle", "sea"]},
    "yankees":  {"full": "New York Yankees", "abbr": "NYY", "park": "Yankee Stadium",
                 "aliases": ["ny yankees", "nyy", "new york yankees"]},
    "mets":     {"full": "New York Mets", "abbr": "NYM", "park": "Citi Field",
                 "aliases": ["ny mets", "nym", "new york mets"]},
    "red_sox":  {"full": "Boston Red Sox", "abbr": "BOS", "park": "Fenway Park",
                 "aliases": ["boston", "bos", "redsox"]},
    "braves":   {"full": "Atlanta Braves", "abbr": "ATL", "park": "Truist Park",
                 "aliases": ["atlanta", "atl"]},
    "phillies": {"full": "Philadelphia Phillies", "abbr": "PHI", "park": "Citizens Bank Park",
                 "aliases": ["philadelphia", "phi", "phillies"]},
    "orioles":  {"full": "Baltimore Orioles", "abbr": "BAL", "park": "Oriole Park at Camden Yards",
                 "aliases": ["baltimore", "bal"]},
    # NOTE: starter set covering teams referenced in this codebase's examples --
    # extend with the rest of the 30 MLB teams before relying on this for every game.
}

WNBA_TEAMS = {
    "liberty": {"full": "New York Liberty", "abbr": "NY", "park": "Barclays Center",
                "aliases": ["ny liberty", "new york liberty", "ny"]},
    "aces":    {"full": "Las Vegas Aces", "abbr": "LV", "park": "Michelob Ultra Arena",
                "aliases": ["lv aces", "las vegas aces", "lv"]},
    "storm":   {"full": "Seattle Storm", "abbr": "SEA", "park": "Climate Pledge Arena",
                "aliases": ["seattle storm"]},
    "sky":     {"full": "Chicago Sky", "abbr": "CHI", "park": "Wintrust Arena",
                "aliases": ["chicago sky"]},
    "fever":   {"full": "Indiana Fever", "abbr": "IND", "park": "Gainbridge Fieldhouse",
                "aliases": ["indiana fever"]},
    "mercury": {"full": "Phoenix Mercury", "abbr": "PHX", "park": "PHX Arena",
                "aliases": ["phoenix mercury"]},
    "wings":   {"full": "Dallas Wings", "abbr": "DAL", "park": "College Park Center",
                "aliases": ["dallas wings"]},
    "mystics": {"full": "Washington Mystics", "abbr": "WAS", "park": "Entertainment and Sports Arena",
                "aliases": ["washington mystics"]},
    # NOTE: starter set -- extend with remaining WNBA teams as encountered.
}

_UNRESOLVED_LOG = []  # collects anything that failed to resolve, for review after a run


def _norm(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    return s


def _build_alias_index(team_dict):
    index = {}
    for key, info in team_dict.items():
        candidates = set(info["aliases"]) | {info["full"], info["abbr"], key}
        for c in candidates:
            index[_norm(c)] = key
    return index


_MLB_INDEX = _build_alias_index(MLB_TEAMS)
_WNBA_INDEX = _build_alias_index(WNBA_TEAMS)


def canonical_team(name, sport):
    """
    Resolves any spelling/abbreviation variant to one canonical record:
    {key, full, abbr, park}. Returns None (and logs) if unresolved -- callers
    MUST handle None explicitly rather than letting a park-factor/odds lookup
    silently fall through on a raw, un-normalized string.
    """
    sport = sport.lower()
    team_dict = MLB_TEAMS if sport == "mlb" else WNBA_TEAMS if sport == "wnba" else None
    index = _MLB_INDEX if sport == "mlb" else _WNBA_INDEX if sport == "wnba" else None
    if team_dict is None:
        raise ValueError(f"sport must be 'mlb' or 'wnba' -- got {sport!r}")

    key = index.get(_norm(name))
    if key is None:
        _UNRESOLVED_LOG.append({"sport": sport, "input": name})
        return None

    info = team_dict[key]
    return {"key": key, "full": info["full"], "abbr": info["abbr"], "park": info["park"]}


def canonical_park(team_name, sport):
    team = canonical_team(team_name, sport)
    return team["park"] if team else None


def canonical_player(name):
    """
    Normalizes player-name formatting differences across sources:
    'Valdez, Framber' -> 'Framber Valdez', strips accents for matching purposes
    (e.g. pybaseball name-contains lookups), collapses extra whitespace.
    Does NOT strip accents from the display value -- only used internally for
    matching; returns a clean DISPLAY name (accents kept) plus a separate
    match_key (accents stripped, lowercase) for fuzzy lookups.
    """
    if name is None:
        return {"display": None, "match_key": None}
    raw = name.strip()
    if "," in raw:
        last, first = [p.strip() for p in raw.split(",", 1)]
        raw = f"{first} {last}"
    raw = re.sub(r"\s+", " ", raw)
    return {"display": raw, "match_key": _norm(raw)}


def get_unresolved_log():
    """Call after a pipeline run to see every name that failed to canonicalize --
    these are exactly the silent-breakage points the rest of the system is
    designed to avoid. Surface this in run output, don't bury it."""
    return list(_UNRESOLVED_LOG)
    
