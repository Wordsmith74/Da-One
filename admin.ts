import { Router, type Request, type Response } from "express";
import { spawn } from "node:child_process";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { logger } from "../lib/logger";

const adminRouter = Router();

// Resolve workspace root: API server runs from artifacts/api-server, so go up two levels
const WORKSPACE = path.resolve(process.cwd(), "..", "..");

// Tracks the currently-running background sequence so we don't double-launch
let _sequenceRunning = false;

// ---------------------------------------------------------------------------
// Signal-queue state helper — used by catch-up to resume mid-sequence
// ---------------------------------------------------------------------------
const DB_PATH = path.resolve(process.cwd(), "..", "..", "data", "results.db");

function _getTodaySignalState(today: string): { sentToday: number; signalCount: number; maxCycle: number } {
  try {
    const db = new DatabaseSync(DB_PATH, { open: true });
    const sent = (db.prepare(
      "SELECT COUNT(*) AS n FROM bets WHERE slate_date = ? AND sent_to_group = 1"
    ).get(today) as { n: number } | undefined)?.n ?? 0;
    const row = db.prepare(
      "SELECT COUNT(*) AS n, MAX(cycle_count) AS mx FROM signal_queue WHERE created_date = ?"
    ).get(today) as { n: number; mx: number | null } | undefined;
    db.close();
    return { sentToday: sent, signalCount: row?.n ?? 0, maxCycle: row?.mx ?? 0 };
  } catch {
    return { sentToday: 0, signalCount: 0, maxCycle: 0 };
  }
}

function spawnPython(
  args: string[],
  label: string,
  res: Response,
): void {
  const child = spawn("python3", ["main.py", ...args], {
    cwd: WORKSPACE,
    env: process.env,
  });

  const lines: string[] = [];

  child.stdout.on("data", (chunk: Buffer) => {
    lines.push(chunk.toString());
  });
  child.stderr.on("data", (chunk: Buffer) => {
    lines.push(chunk.toString());
  });

  child.on("close", (code) => {
    const output = lines.join("").trim();
    logger.info({ code }, `Admin ${label} finished`);
    res.json({ exitCode: code, output });
  });

  child.on("error", (err) => {
    logger.error({ err }, `Failed to spawn ${label} process`);
    res.status(500).json({ error: String(err) });
  });
}

// Promisified spawn — resolves with exit code when the child finishes
function spawnAsync(args: string[], label: string): Promise<number> {
  return new Promise((resolve) => {
    const child = spawn("python3", ["main.py", ...args], {
      cwd: WORKSPACE,
      env: process.env,
    });

    child.stdout.on("data", (chunk: Buffer) => {
      process.stdout.write(`[${label}] ${chunk.toString()}`);
    });
    child.stderr.on("data", (chunk: Buffer) => {
      process.stderr.write(`[${label}] ${chunk.toString()}`);
    });
    child.on("close", (code) => {
      logger.info({ code, label }, "Admin async spawn finished");
      resolve(code ?? 0);
    });
    child.on("error", (err) => {
      logger.error({ err, label }, "Admin async spawn error");
      resolve(1);
    });
  });
}

function delay(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// POST /api/admin/send-picks
// One-shot --mode picks (used by legacy callers). Waits for completion.
adminRouter.post("/admin/send-picks", (req: Request, res: Response) => {
  logger.info("Admin trigger: running picks broadcast");
  spawnPython(["--mode", "picks"], "picks broadcast", res);
});

// POST /api/admin/run-grade
// One-shot --mode grade. Waits for completion.
adminRouter.post("/admin/run-grade", (req: Request, res: Response) => {
  logger.info("Admin trigger: running grade cycle");
  spawnPython(["--mode", "grade"], "grade", res);
});

// POST /api/admin/run-picks-full
// Runs the complete 3-cycle confirmation sequence in the background:
//   Cycle 1: --mode run   (recap + seed signals, cy=1)
//   Cycle 2: --mode picks (confirm signals,       cy=2)
//   Cycle 3: --mode picks (final confirm + broadcast, cy=3)
//
// Sleeps between cycles give signals time to age past the min_minutes gate
// (5 min for all sports). With 2-min sleeps and ~4-min cycles, signals seeded
// in cycle 1 are ~8 min old by cycle 3 — comfortably past the 5-min floor.
// Total sequence: ~16 min.
adminRouter.post("/admin/run-picks-full", (req: Request, res: Response) => {
  if (_sequenceRunning) {
    res.status(409).json({ status: "busy", message: "A picks sequence is already running" });
    return;
  }

  _sequenceRunning = true;
  logger.info("Admin trigger: starting full 3-cycle picks sequence");
  res.status(202).json({
    status: "started",
    message: "Full 3-cycle picks sequence started. Check /api/slate in ~16 minutes for results.",
    cycles: ["--mode run", "--mode picks (cycle 2)", "--mode picks (cycle 3, broadcasts)"],
  });

  (async () => {
    try {
      logger.info("Picks sequence — cycle 1: --mode run");
      await spawnAsync(["--mode", "run"], "cycle-1");
      await delay(120_000);   // 2 min — let signals age past the 5-min min_minutes gate

      logger.info("Picks sequence — cycle 2: --mode picks");
      await spawnAsync(["--mode", "picks"], "cycle-2");
      await delay(120_000);   // 2 min — signals now ~8 min old, will clear gate in cycle 3

      logger.info("Picks sequence — cycle 3: --mode picks (broadcast)");
      await spawnAsync(["--mode", "picks"], "cycle-3");

      logger.info("Picks sequence — all 3 cycles complete");
    } catch (err) {
      logger.error({ err }, "Picks sequence error");
    } finally {
      _sequenceRunning = false;
    }
  })();
});

// GET /api/admin/status
// Returns whether a picks sequence is currently running.
adminRouter.get("/admin/status", (_req: Request, res: Response) => {
  res.json({ sequenceRunning: _sequenceRunning });
});

// ---------------------------------------------------------------------------
// Daily scheduler — fires the 3-cycle picks sequence at 8:30 AM ET every day
// ---------------------------------------------------------------------------

// Tracks the ET date (YYYY-MM-DD) of the last morning + evening scheduler runs
let _lastMorningDate  = "";
let _lastEveningDate  = "";

// Tracks grade-cycle runs to prevent double-firing.
// Grade fires every 30 min from 19:00–23:30 ET to settle picks as games finish.
const _gradeFiredSlots = new Set<string>();
let   _gradeRunning    = false;

async function _runGradeCycle(label: string): Promise<void> {
  if (_gradeRunning) {
    logger.info(`${label}: grade already running — skipping`);
    return;
  }
  _gradeRunning = true;
  logger.info(`${label}: running --mode grade`);
  try {
    await spawnAsync(["--mode", "grade"], label);
    logger.info(`${label}: grade complete`);
  } catch (err) {
    logger.error({ err }, `${label}: grade error`);
  } finally {
    _gradeRunning = false;
  }
}

async function _runScheduledSequence(label: string): Promise<void> {
  if (_sequenceRunning) {
    logger.info(`${label}: sequence already running — skipping`);
    return;
  }
  _sequenceRunning = true;
  logger.info(`${label}: starting 3-cycle picks sequence`);

  try {
    logger.info(`${label} — cycle 1: --mode run`);
    await spawnAsync(["--mode", "run"], "sched-cycle-1");
    await delay(120_000);   // 2 min

    logger.info(`${label} — cycle 2: --mode picks`);
    await spawnAsync(["--mode", "picks"], "sched-cycle-2");
    await delay(120_000);   // 2 min — signals ~8 min old, clears 5-min gate

    logger.info(`${label} — cycle 3: --mode picks (broadcast)`);
    await spawnAsync(["--mode", "picks"], "sched-cycle-3");

    logger.info(`${label}: 3-cycle sequence complete`);
  } catch (err) {
    logger.error({ err }, `${label}: sequence error`);
  } finally {
    _sequenceRunning = false;
  }
}

// Abbreviated 2-cycle variant: skips --mode run when signals are already seeded.
// Used by catch-up on restart when today's signal queue is already populated.
async function _runPicksCyclesOnly(label: string): Promise<void> {
  if (_sequenceRunning) {
    logger.info(`${label}: sequence already running — skipping`);
    return;
  }
  _sequenceRunning = true;
  logger.info(`${label}: starting 2-cycle picks-only sequence (signals already seeded)`);

  try {
    logger.info(`${label} — cycle 2: --mode picks`);
    await spawnAsync(["--mode", "picks"], `${label}-cycle-2`);
    await delay(120_000);   // 2 min

    logger.info(`${label} — cycle 3: --mode picks (broadcast)`);
    await spawnAsync(["--mode", "picks"], `${label}-cycle-3`);

    logger.info(`${label}: 2-cycle picks-only sequence complete`);
  } catch (err) {
    logger.error({ err }, `${label}: picks-only sequence error`);
  } finally {
    _sequenceRunning = false;
  }
}

/**
 * Start the daily picks scheduler.
 * Morning run: 08:30 ET (watchdog at 09:00 ET).
 * Evening run:  18:00 ET (watchdog at 18:30 ET) — catches evening lines.
 * Call once from the server entry point after the HTTP server starts listening.
 */
export function startDailyScheduler(): void {
  logger.info("Daily picks scheduler armed — morning 08:30 ET + evening 18:00 ET");

  // ── Startup catch-up ────────────────────────────────────────────────────
  // If the server starts after the scheduled window, fire the missed slot
  // immediately instead of silently skipping the whole day.
  {
    const nowET = new Date(
      new Date().toLocaleString("en-US", { timeZone: "America/New_York" }),
    );
    const h     = nowET.getHours();
    const m     = nowET.getMinutes();
    const today = `${nowET.getFullYear()}-${String(nowET.getMonth() + 1).padStart(2, "0")}-${String(nowET.getDate()).padStart(2, "0")}`;

    // Past 08:30 ET and morning not yet run → catch up
    if ((h > 8 || (h === 8 && m >= 30)) && h < 18 && today !== _lastMorningDate) {
      _lastMorningDate = today;
      const startedAt = `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")} ET`;
      const sigState = _getTodaySignalState(today);
      if (sigState.sentToday > 0) {
        // Picks already broadcast today — nothing to do
        logger.info({ today, sentToday: sigState.sentToday }, "STARTUP CATCH-UP: picks already sent today — skipping morning sequence");
      } else if (sigState.signalCount > 0) {
        // Signals seeded from an earlier run — skip --mode run, go straight to picks cycles
        logger.warn({ today, startedAt, signalCount: sigState.signalCount, maxCycle: sigState.maxCycle },
          "STARTUP CATCH-UP: signals already seeded — resuming with picks-only cycles");
        void _sendStartupCatchUpAlert("morning", today, startedAt);
        void _runPicksCyclesOnly("Morning-CatchUp");
      } else {
        // No signals at all — run full 3-cycle sequence
        logger.warn({ today, startedAt }, "STARTUP CATCH-UP: server started after 08:30 ET — firing morning sequence now");
        void _sendStartupCatchUpAlert("morning", today, startedAt);
        void _runScheduledSequence("Morning-CatchUp");
      }
    }

    // Past 18:00 ET and evening not yet run → catch up
    if (h >= 18 && today !== _lastEveningDate) {
      _lastEveningDate = today;
      const startedAt = `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")} ET`;
      const sigState = _getTodaySignalState(today);
      if (sigState.sentToday > 0 && sigState.signalCount === 0) {
        logger.info({ today, sentToday: sigState.sentToday }, "STARTUP CATCH-UP: picks already sent today — skipping evening sequence");
      } else if (sigState.signalCount > 0 && sigState.sentToday === 0) {
        logger.warn({ today, startedAt, signalCount: sigState.signalCount },
          "STARTUP CATCH-UP: evening signals seeded — resuming with picks-only cycles");
        void _sendStartupCatchUpAlert("evening", today, startedAt);
        void _runPicksCyclesOnly("Evening-CatchUp");
      } else {
        logger.warn({ today, startedAt }, "STARTUP CATCH-UP: server started after 18:00 ET — firing evening sequence now");
        void _sendStartupCatchUpAlert("evening", today, startedAt);
        void _runScheduledSequence("Evening-CatchUp");
      }
    }
  }

  setInterval(() => {
    const nowET = new Date(
      new Date().toLocaleString("en-US", { timeZone: "America/New_York" }),
    );
    const h     = nowET.getHours();
    const m     = nowET.getMinutes();
    const today = `${nowET.getFullYear()}-${String(nowET.getMonth() + 1).padStart(2, "0")}-${String(nowET.getDate()).padStart(2, "0")}`;

    // ── Morning: primary 08:30 ET ─────────────────────────────────────────
    if (h === 8 && m === 30 && today !== _lastMorningDate) {
      _lastMorningDate = today;
      logger.info({ today }, "Morning scheduler: 08:30 ET trigger fired");
      void _runScheduledSequence("Morning");
      return;
    }

    // ── Morning: watchdog 09:00 ET ────────────────────────────────────────
    if (h === 9 && m === 0 && today !== _lastMorningDate) {
      _lastMorningDate = today;
      logger.error({ today }, "WATCHDOG: 08:30 ET morning trigger missed — re-firing now");
      void _sendSchedulerMissAlert("morning", today);
      void _runScheduledSequence("Morning-Watchdog");
      return;
    }

    // ── Evening: primary 18:00 ET ─────────────────────────────────────────
    if (h === 18 && m === 0 && today !== _lastEveningDate) {
      _lastEveningDate = today;
      logger.info({ today }, "Evening scheduler: 18:00 ET trigger fired");
      void _runScheduledSequence("Evening");
      return;
    }

    // ── Evening: watchdog 18:30 ET ────────────────────────────────────────
    if (h === 18 && m === 30 && today !== _lastEveningDate) {
      _lastEveningDate = today;
      logger.error({ today }, "WATCHDOG: 18:00 ET evening trigger missed — re-firing now");
      void _sendSchedulerMissAlert("evening", today);
      void _runScheduledSequence("Evening-Watchdog");
    }

    // ── Auto-grade: every 30 min 19:00–23:30 ET ───────────────────────────
    // Grades settled picks as games finish throughout the evening.
    // Each slot key is "YYYY-MM-DD HH:MM" — fires once per slot.
    if (h >= 19 && (h < 23 || (h === 23 && m <= 30))) {
      const slot = m < 30 ? `${today} ${String(h).padStart(2,"0")}:00` : `${today} ${String(h).padStart(2,"0")}:30`;
      if (!_gradeFiredSlots.has(slot)) {
        _gradeFiredSlots.add(slot);
        logger.info({ slot }, "Auto-grade: firing --mode grade");
        void _runGradeCycle(`AutoGrade-${slot}`);
      }
    }
  }, 60_000); // check every minute
}

/**
 * Run one grade + rebuild cycle immediately at startup.
 * Ensures stale open picks from prior days are settled without waiting for the
 * 19:00 ET window, and brings performance_stats up to date on every server start.
 */
export async function runStartupGrade(): Promise<void> {
  logger.info("Startup grade: running --mode grade to settle any stale open picks");
  try {
    await spawnAsync(["--mode", "grade"], "startup-grade");
    logger.info("Startup grade: complete");
  } catch (err) {
    logger.warn({ err }, "Startup grade: non-fatal error");
  }
}

// ---------------------------------------------------------------------------
// Telegram alert helper for missed-scheduler audit
// ---------------------------------------------------------------------------

async function _sendStartupCatchUpAlert(slot: "morning" | "evening", date: string, startedAt: string): Promise<void> {
  const token  = process.env["TELEGRAM_BOT_TOKEN"];
  const chatId = process.env["TELEGRAM_CHAT_ID"];
  if (!token || !chatId) return;

  const scheduled = slot === "morning" ? "08:30 ET" : "18:00 ET";
  const text =
    `🔄 *STARTUP CATCH-UP* 🔄\n` +
    `Server restarted at *${startedAt}* on ${date} — after the ${scheduled} ${slot} window.\n` +
    `Firing the ${slot} picks sequence now. Picks will arrive ~16 min after server start.\n` +
    `No manual action needed.`;

  try {
    await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chat_id: chatId, text, parse_mode: "Markdown" }),
    });
  } catch (err) {
    logger.warn({ err }, "STARTUP CATCH-UP: failed to send Telegram alert");
  }
}

async function _sendSchedulerMissAlert(slot: "morning" | "evening", date: string): Promise<void> {
  const token  = process.env["TELEGRAM_BOT_TOKEN"];
  const chatId = process.env["TELEGRAM_CHAT_ID"];

  if (!token || !chatId) {
    logger.warn("WATCHDOG: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — cannot send miss alert");
    return;
  }

  const primary  = slot === "morning" ? "08:30 ET" : "18:00 ET";
  const watchdog = slot === "morning" ? "09:00 ET" : "18:30 ET";
  const text =
    `⚠️ *SCHEDULER AUDIT* ⚠️\n` +
    `The ${primary} ${slot} picks trigger was *missed* on ${date}.\n` +
    `Watchdog detected the gap at ${watchdog} and has auto-fired the sequence now.\n` +
    `Picks will arrive ~16 min late. No manual action needed.`;

  try {
    const url = `https://api.telegram.org/bot${token}/sendMessage`;
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chat_id: chatId, text, parse_mode: "Markdown" }),
    });
    if (!resp.ok) {
      logger.warn({ status: resp.status }, "WATCHDOG: Telegram alert returned non-OK status");
    } else {
      logger.info("WATCHDOG: Telegram miss-alert sent successfully");
    }
  } catch (err) {
    logger.error({ err }, "WATCHDOG: Failed to send Telegram miss-alert");
  }
}

export default adminRouter;
