"""
core/telegram_publisher.py

Publishes today's picks (output/picks.json) to a Telegram group as a
formatted message, via the Telegram Bot HTTP API. This is separate from
core/alerts.py (which is a neutral local-log sink for internal warnings/
errors, not picks) -- this module's job is the actual public-facing
"here are today's picks" broadcast, run after run_pipeline.py has written
output/picks.json.

Requires two env vars (set as GitHub Actions secrets):
    TELEGRAM_BOT_TOKEN  -- from @BotFather
    TELEGRAM_CHAT_ID    -- the target group's chat id (negative number for
                           groups/supergroups, e.g. -1001234567890)

Usage:
    python3 -m core.telegram_publisher [--picks-path output/picks.json]

Exits 0 even when there's nothing to send or credentials are missing --
this is a best-effort broadcast step and should never fail the workflow
that generates/commits picks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_MSG_LEN = 4096  # Telegram hard limit per message

_TIER_EMOJI = {
    "Nuke": "\U0001F680",     # 🚀
    "Diamond": "\U0001F48E",  # 💎
    "Gold": "\U0001F949",     # 🥉-ish gold medal glyph
}
_TIER_ORDER = ["Nuke", "Diamond", "Gold"]


def _fmt_odds(odds: Any) -> str:
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return "N/A"
    return f"+{int(o)}" if o >= 0 else str(int(o))


def _fmt_pick_line(p: dict[str, Any]) -> str:
    matchup = f"{p.get('away_team', '?')} @ {p.get('home_team', '?')}"
    pick_desc = p.get("pick") or f"{p.get('player') or p.get('team', '')} {p.get('market', '')} {p.get('side', '')}".strip()
    odds = _fmt_odds(p.get("pick_time_odds") or p.get("current_odds"))
    edge = p.get("edge_pct")
    conf = p.get("confidence")
    edge_str = f"{edge:.1f}%" if isinstance(edge, (int, float)) else "N/A"
    conf_str = f"{conf:.0f}%" if isinstance(conf, (int, float)) else "N/A"
    return (
        f"<b>{pick_desc}</b> ({odds})\n"
        f"<i>{matchup}</i>\n"
        f"Edge: {edge_str} | Confidence: {conf_str}"
    )


def format_message(payload: dict[str, Any]) -> list[str]:
    """Build one or more Telegram-ready HTML message chunks from picks.json's payload."""
    picks = payload.get("picks") or []
    generated_at = payload.get("generated_at", "")
    date_str = generated_at[:10] if generated_at else ""

    if not picks:
        return [f"\U0001F4ED <b>Da-One Picks — {date_str}</b>\nNo picks cleared the gate today."]

    by_tier: dict[str, list[dict]] = {}
    for p in picks:
        by_tier.setdefault(p.get("tier") or "Other", []).append(p)

    header = f"\U0001F4C8 <b>Da-One Picks — {date_str}</b> ({len(picks)} pick{'s' if len(picks) != 1 else ''})\n"

    blocks = [header]
    tier_keys = [t for t in _TIER_ORDER if t in by_tier] + [t for t in by_tier if t not in _TIER_ORDER]
    for tier in tier_keys:
        emoji = _TIER_EMOJI.get(tier, "\u2022")
        blocks.append(f"\n{emoji} <b>{tier}</b>")
        for p in by_tier[tier]:
            blocks.append(_fmt_pick_line(p))

    # Pack blocks into <=4096-char messages, splitting on block boundaries.
    messages: list[str] = []
    current = ""
    for block in blocks:
        candidate = (current + "\n" + block) if current else block
        if len(candidate) > _MAX_MSG_LEN:
            if current:
                messages.append(current)
            current = block
        else:
            current = candidate
    if current:
        messages.append(current)
    return messages


def send_message(token: str, chat_id: str, text: str) -> bool:
    try:
        resp = requests.post(
            _TELEGRAM_API.format(token=token),
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"[telegram_publisher] send failed ({resp.status_code}): {resp.text[:500]}", file=sys.stderr)
            return False
        return True
    except requests.RequestException as exc:
        print(f"[telegram_publisher] request error: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--picks-path", default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "output", "picks.json"))
    args = parser.parse_args()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[telegram_publisher] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set -- skipping broadcast.")
        return 0

    if not os.path.exists(args.picks_path):
        print(f"[telegram_publisher] {args.picks_path} not found -- skipping broadcast.")
        return 0

    with open(args.picks_path) as f:
        payload = json.load(f)

    messages = format_message(payload)
    all_ok = True
    for msg in messages:
        ok = send_message(token, chat_id, msg)
        all_ok = all_ok and ok

    if not all_ok:
        print("[telegram_publisher] one or more messages failed to send.", file=sys.stderr)
        # Non-fatal: don't fail the workflow over a broadcast hiccup.
    else:
        print(f"[telegram_publisher] sent {len(messages)} message(s) to chat {chat_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
