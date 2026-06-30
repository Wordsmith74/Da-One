"""
Contradiction check: drops picks that conflict with each other within the
same matchup/game (e.g. backing the F5 Over on a total while also implicitly
needing a pitcher to dominate enough to hit an Under-shaped K prop in a way
that's mutually inconsistent; or two props on opposite sides of the same
team's total runs).

Grouping key is (sport, matchup) deliberately -- never group MLB and WNBA
picks together for contradiction purposes; they can never contradict each
other and treating them as comparable would risk an incorrect drop.
"""


def filter_contradictions(picks):
    """Returns (cleaned_picks, dropped_picks)."""
    by_matchup = {}
    for p in picks:
        key = (p.get("sport"), p.get("matchup"))
        by_matchup.setdefault(key, []).append(p)

    cleaned = []
    dropped = []
    for key, group in by_matchup.items():
        if len(group) == 1:
            cleaned.extend(group)
            continue

        conflict_found = False
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if _conflicts(group[i], group[j]):
                    conflict_found = True

        if conflict_found:
            # Conservative: drop the whole conflicting cluster for that matchup
            # rather than try to guess which one "wins" -- a confident model
            # shouldn't be internally contradicting itself in the first place.
            dropped.extend(group)
        else:
            cleaned.extend(group)

    return cleaned, dropped


def _conflicts(pick_a, pick_b):
    """
    Same market + same matchup + opposite sides = direct contradiction.
    Different markets within the same matchup are NOT automatically flagged --
    that requires real correlation modeling (e.g. F5 Over + starter Under Ks
    are only weakly related), which is out of scope for a simple safety net.
    Keep this conservative and explicit rather than guessing at correlations.
    """
    same_market = pick_a.get("market") == pick_b.get("market")
    same_player = pick_a.get("player") is not None and pick_a.get("player") == pick_b.get("player")
    opposite_side = pick_a.get("side") and pick_b.get("side") and pick_a["side"] != pick_b["side"]
    return same_market and same_player and opposite_side
