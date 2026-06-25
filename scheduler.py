"""
scheduler.py — 24/7 autonomous scheduling daemon for DaPickSyndicate.

Replit-native equivalent of a PM2-managed worker process.
Run via the "Engine Scheduler" Replit Workflow; Deploy the app for true
24/7 operation independent of browser sessions or login state.

Daily schedule (America/New_York):
  08:30  run          Full pipeline — recap → 60 s → picks   (Cycle 1)
  09:15  revalidate   Pregame line revalidation               (Cycle 2)
  09:45  revalidate   Final pregame check                     (Cycle 3)
  10:00  picks        Lock picks + final Telegram broadcast
  19:00–23:00         Grade every 30 min (auto-grade settled bets)
  22:00  reconcile    Nightly catch-all for stuck open bets

Error handling:
  • Each job retried MAX_RETRIES times with exponential backoff.
  • After all retries fail → Telegram critical alert.
  • Scheduler loop itself is guarded — never exits on an unexpected error.
  • SIGTERM caught for graceful Replit workflow shutdown.

Logs: scheduler.log (rotating, 5 MB × 3 files) + stdout.
State: data/scheduler_state.json (daily job completion tracker).
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent
LOG_PATH   = ROOT_DIR / "scheduler.log"
STATE_PATH = ROOT_DIR / "data" / "scheduler_state.json"
PYTHON     = sys.executable            # same interpreter that launched us
MAIN       = str(ROOT_DIR / "main.py")
SYNC       = str(ROOT_DIR / "core" / "supabase_sync.py")

# ── Constants ──────────────────────────────────────────────────────────────────
ET            = ZoneInfo("America/New_York")
MAX_RETRIES   = 3
RETRY_BACKOFF = [60, 120, 300]         # seconds: 1 min, 2 min, 5 min
POLL_INTERVAL = 30                     # seconds between schedule ticks
JOB_TIMEOUT   = 900                    # hard cap per job (15 min)
SYNC_TIMEOUT  = 120                    # hard cap for Supabase sync (2 min)
HEARTBEAT_S   = 3600                   # heartbeat log interval (1 hour)

# Modes that write picks/outcomes and should trigger a Supabase sync
_SYNC_MODES = {"run", "picks", "grade", "reconcile", "revalidate"}

# ── Logging ────────────────────────────────────────────────────────────────────
class _ETFormatter(logging.Formatter):
    """Logging formatter that stamps every line in America/New_York (ET)."""
    _tz = ZoneInfo("America/New_York")

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        ct = datetime.fromtimestamp(record.created, tz=self._tz)
        return ct.strftime(datefmt or "%Y-%m-%d %H:%M:%S ET")


def _build_logger() -> logging.Logger:
    log = logging.getLogger("scheduler")
    log.setLevel(logging.DEBUG)
    fmt = _ETFormatter("%(asctime)s [%(levelname)-8s] %(message)s")
    fh = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log

logger = _build_logger()

# ── Schedule definition ────────────────────────────────────────────────────────
# Each entry: (hour_ET, minute_ET, job_name, mode, extra_cli_args)
_MORNING: list[tuple[int, int, str, str, list[str]]] = [
    (6,  0,  "reconcile_am", "reconcile", []),   # catch-all before morning run
    (8,  30, "run",          "run",        []),   # cycle 1 — full pipeline + signal seed
    (9,  15, "picks_cycle2", "picks",      []),   # cycle 2 — signal confirmation
    (10, 0,  "picks_cycle3", "picks",      []),   # cycle 3 — confirmed picks broadcast
]

_GRADE: list[tuple[int, int, str, str, list[str]]] = [
    (h, m, f"grade_{h:02d}{m:02d}", "grade", [])
    for h in range(19, 24)
    for m in (0, 30)
    if not (h == 23 and m == 30)      # cap at 23:00; skip 23:30
]

_EVENING: list[tuple[int, int, str, str, list[str]]] = [
    (22, 0, "reconcile", "reconcile", []),
]

ALL_JOBS: list[tuple[int, int, str, str, list[str]]] = (
    _MORNING + _GRADE + _EVENING
)

# ── State persistence ──────────────────────────────────────────────────────────
def _load_state() -> dict[str, Any]:
    """Load today's state from disk; return fresh dict if stale or missing."""
    try:
        if STATE_PATH.exists():
            raw = json.loads(STATE_PATH.read_text())
            if raw.get("date") == date.today().isoformat():
                return raw
    except Exception:
        pass
    return {"date": date.today().isoformat(), "completed": [], "failed": []}


def _save_state(state: dict[str, Any]) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception as exc:
        logger.warning(f"State save failed: {exc}")

# ── Telegram alert (best-effort, never raises) ─────────────────────────────────
def _alert(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat  = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        logger.debug("Telegram alert skipped — env vars not set")
        return
    try:
        import urllib.request
        payload = json.dumps({
            "chat_id": chat,
            "text": text,
            "parse_mode": "HTML",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
        logger.debug("Telegram alert sent")
    except Exception as exc:
        logger.warning(f"Telegram alert failed (non-fatal): {exc}")

# ── Supabase sync (best-effort, never raises) ──────────────────────────────────
def _sync_to_supabase(job_name: str) -> None:
    """
    Push latest picks/outcomes to Supabase after a successful job.
    Runs as a child process so any import or runtime error is isolated.
    Never raises — sync failures are logged as warnings only.
    """
    if not Path(SYNC).exists():
        logger.debug(f"[SYNC SKIP]  {job_name} — core/supabase_sync.py not found")
        return
    try:
        result = subprocess.run(
            [PYTHON, SYNC],
            cwd=str(ROOT_DIR),
            env={**os.environ},
            timeout=SYNC_TIMEOUT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info(f"[SYNC OK]    {job_name} — Supabase sync complete")
        else:
            logger.warning(
                f"[SYNC WARN]  {job_name} — exit {result.returncode}: "
                f"{result.stderr.strip()[:300]}"
            )
    except subprocess.TimeoutExpired:
        logger.warning(f"[SYNC WARN]  {job_name} — Supabase sync timed out ({SYNC_TIMEOUT}s)")
    except Exception as exc:
        logger.warning(f"[SYNC WARN]  {job_name} — Supabase sync error: {exc}")

# ── Job executor ───────────────────────────────────────────────────────────────
def _run_job(job_name: str, mode: str, extra: list[str]) -> bool:
    """Execute one engine mode as a child process. Returns True on success."""
    cmd = [PYTHON, MAIN, "--mode", mode] + extra
    logger.info(
        f"[JOB START]  {job_name} | cmd: python3 main.py --mode {mode}"
        + (f" {' '.join(extra)}" if extra else "")
    )
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            env={**os.environ},
            timeout=JOB_TIMEOUT,
        )
        elapsed = time.monotonic() - t0
        if result.returncode == 0:
            logger.info(f"[JOB OK]     {job_name} | {elapsed:.0f}s | exit 0")
            return True
        logger.error(
            f"[JOB FAIL]   {job_name} | {elapsed:.0f}s | exit {result.returncode}"
        )
        return False
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        logger.error(f"[JOB TIMEOUT] {job_name} | exceeded {JOB_TIMEOUT}s — killed")
        return False
    except Exception as exc:
        logger.error(f"[JOB ERROR]  {job_name} | {exc}")
        return False


def _run_with_retry(
    job_name: str, mode: str, extra: list[str], state: dict[str, Any]
) -> None:
    """Run a job with retry logic. Sends a Telegram alert if all retries fail."""
    for attempt in range(1, MAX_RETRIES + 1):
        ok = _run_job(job_name, mode, extra)
        if ok:
            state["completed"].append(job_name)
            _save_state(state)
            # Sync to Supabase after any job that writes picks or outcomes
            if mode in _SYNC_MODES:
                _sync_to_supabase(job_name)
            return
        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF[attempt - 1]
            logger.warning(
                f"[RETRY {attempt}/{MAX_RETRIES}] {job_name} — next attempt in {wait}s"
            )
            time.sleep(wait)

    # All retries exhausted
    state["failed"].append(job_name)
    _save_state(state)
    now_str = datetime.now(tz=ET).strftime("%Y-%m-%d %H:%M ET")
    msg = (
        f"🚨 <b>Scheduler alert</b>\n"
        f"Job <code>{job_name}</code> failed after {MAX_RETRIES} attempts.\n"
        f"Mode: <code>--mode {mode}</code>\n"
        f"Time: {now_str}"
    )
    logger.critical(f"[GIVE UP]    {job_name} — all retries exhausted; Telegram alert sent")
    _alert(msg)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _now_et() -> datetime:
    return datetime.now(tz=ET)


def _is_due(h: int, m: int, now: datetime) -> bool:
    """True when ET time is within a 2-minute window after the scheduled hh:mm."""
    due = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return due <= now < due + timedelta(minutes=2)

# ── Main loop ──────────────────────────────────────────────────────────────────
def main() -> None:
    schedule_str = " | ".join(
        f"{h:02d}:{m:02d}→{n}" for h, m, n, *_ in ALL_JOBS
    )
    logger.info("=" * 70)
    logger.info("DaPickSyndicate — Engine Scheduler starting")
    logger.info(f"Python  : {sys.version.split()[0]}")
    logger.info(f"Root    : {ROOT_DIR}")
    logger.info(f"Poll    : every {POLL_INTERVAL}s   |   Max retries: {MAX_RETRIES}")
    logger.info(f"Jobs/day: {len(ALL_JOBS)}")
    logger.info(f"Schedule: {schedule_str}")
    logger.info("=" * 70)

    _alert(
        "🟢 <b>Scheduler started</b>\n"
        f"Jobs today: {len(ALL_JOBS)} | "
        f"First job: 08:30 ET (run)"
    )

    state      = _load_state()
    last_date  = date.today()
    last_hb_ts = time.monotonic()

    while True:
        try:
            now   = _now_et()
            today = now.date()

            # ── Midnight rollover ──────────────────────────────────────────────
            if today != last_date:
                logger.info(
                    f"[ROLLOVER] New day: {today.isoformat()} — resetting job state"
                )
                state     = {"date": today.isoformat(), "completed": [], "failed": []}
                last_date = today
                _save_state(state)

            # ── Hourly heartbeat ───────────────────────────────────────────────
            if time.monotonic() - last_hb_ts >= HEARTBEAT_S:
                logger.info(
                    f"[HEARTBEAT] {now.strftime('%H:%M ET')} | "
                    f"date={today.isoformat()} | "
                    f"completed={len(state['completed'])} "
                    f"failed={len(state['failed'])}"
                )
                last_hb_ts = time.monotonic()

            # ── Fire any due jobs ──────────────────────────────────────────────
            for (h, m, job_name, mode, extra) in ALL_JOBS:
                if job_name in state["completed"] or job_name in state["failed"]:
                    continue    # already ran or gave up today
                if _is_due(h, m, now):
                    _run_with_retry(job_name, mode, extra, state)

        except KeyboardInterrupt:
            logger.info("Scheduler stopped by KeyboardInterrupt")
            _alert("🔴 <b>Scheduler stopped</b> (KeyboardInterrupt)")
            sys.exit(0)

        except Exception as exc:
            # Never let the main loop die — log and recover
            logger.exception(f"[LOOP ERROR] Unexpected exception — recovering: {exc}")
            _alert(f"⚠️ <b>Scheduler loop error</b> (auto-recovering):\n<code>{exc}</code>")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    def _sigterm(*_: Any) -> None:
        logger.info("SIGTERM received — shutting down scheduler")
        _alert("🔴 <b>Scheduler stopped</b> (SIGTERM)")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm)
    main()
