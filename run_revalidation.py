"""
run_revalidation.py

CLI entry point for core/revalidation_engine.py -- re-checks every open,
locked pick against current conditions (line movement, injuries, edge
decay) and CONFIRMs / UPGRADEs / DOWNGRADEs / VOIDs it. Never creates new
picks; the published slate is immutable, this only annotates it.

Wired in 2026-08-29 alongside a GitHub Actions cache-based fix for
data/results.db persistence (see .github/workflows/revalidate_picks.yml
and the "Restore/Save data/results.db" steps added to
generate_daily_picks.yml's generate-picks job) -- this engine existed and
worked correctly since it was written, but had no scheduled caller and
no way to see picks logged by a separate, earlier workflow run, so it was
silently inert. See revalidation_engine.py's own module docstring for
what it does and doesn't check (notably: starting-pitcher-scratch
detection is a known no-op stub, not yet wired to a real "who was
probable at publish time" comparison -- everything else in the engine is
fully functional).

Usage:
    python3 run_revalidation.py [--sports MLB,WNBA] [--dry-run]

Exits 0 even when there's nothing to revalidate or the DB is empty/
missing -- this is a best-effort maintenance step and should never fail
the workflow that runs it.
"""
from __future__ import annotations

import argparse
import sys

from core.revalidation_engine import run_revalidation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sports",
        default="MLB,WNBA",
        help="Comma-separated list of sports to revalidate (default: MLB,WNBA).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate and log changes without writing them to the DB or sending alerts.",
    )
    args = parser.parse_args()
    sports = [s.strip().upper() for s in args.sports.split(",") if s.strip()]

    try:
        changes = run_revalidation(sports=sports, dry_run=args.dry_run)
    except Exception as exc:
        # Best-effort: a revalidation failure should never be treated as
        # a reason to fail the workflow -- the published slate is
        # unaffected either way, this only annotates it.
        print(f"[run_revalidation] failed (non-fatal): {exc}", file=sys.stderr)
        return 0

    if not changes:
        print("[run_revalidation] No notable changes -- all open picks confirmed or nothing open.")
        return 0

    print(f"[run_revalidation] {len(changes)} notable change(s):")
    for c in changes:
        subject = c.get("player") or c.get("team", "?")
        print(
            f"  {c['bet_id']}: {subject} {c.get('market','?')} {c.get('direction','?')} "
            f"-> {c['revalidation_status'].upper()} ({c.get('reason','')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
