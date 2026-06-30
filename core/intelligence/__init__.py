"""
core/intelligence — pro-grade contextual intelligence layer.

Modules
-------
rest_travel   : Rest days + travel burden → edge adjustment
lineup_intel  : Live injury/lineup status → edge adjustment
stat_model    : Pace-adjusted matchup stats → projected value adjustment
clv_tracker   : Closing Line Value snapshot + long-term edge verification
rivalry_intel : Rivalry detection + H2H context → volatility/edge adjustment
venue_intel   : Home/Away/Neutral environment + park factors → edge adjustment
"""

from core.intelligence.rest_travel import get_rest_travel_factor, RestTravelFactor
from core.intelligence.lineup_intel import get_lineup_intel, LineupIntelFactor
from core.intelligence.stat_model import get_stat_model_factor, StatModelFactor
from core.intelligence.clv_tracker import snapshot_odds, update_closing_line, get_clv_summary
from core.intelligence.rivalry_intel import get_rivalry_intel, RivalryFactor
from core.intelligence.venue_intel import get_venue_factor, VenueFactor

__all__ = [
    "get_rest_travel_factor",
    "RestTravelFactor",
    "get_lineup_intel",
    "LineupIntelFactor",
    "get_stat_model_factor",
    "StatModelFactor",
    "snapshot_odds",
    "update_closing_line",
    "get_clv_summary",
    "get_rivalry_intel",
    "RivalryFactor",
    "get_venue_factor",
    "VenueFactor",
]
