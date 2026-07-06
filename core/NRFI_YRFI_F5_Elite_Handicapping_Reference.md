# Elite NRFI/YRFI & F5 Handicapping Reference

NRFI/YRFI is a bet on the 0.5 first-inning total: **Under 0.5 = NRFI** (no run), **Over 0.5 = YRFI** (yes run). F5 extends the same logic across the first five innings. Both markets isolate starting pitching + top-of-order offense while removing bullpen and late-game variance. Below is a full-depth checklist of what a serious, quant-minded handicapper tracks — organized from core inputs to the more obscure market-structure edges.

---

## 1. Baseline Rates (know your prior)
- League-wide, roughly **72–75% of first innings are scoreless** in a given MLB season — this is your prior before any adjustment, and it's why NRFI is typically priced at -130 to -180.
- Any model output should be sanity-checked against this base rate; large deviations need a real causal reason (weak lineup, ace pitcher, extreme park/weather), not noise.

## 2. Pitcher "First Frame" Metrics (precision over aggregation)
Season-long ERA is noise for this market — you need innings-specific and situational splits:
- **First-inning ERA / WHIP** — isolates first-inning-only performance, since some pitchers are documented "slow starters" (worse first inning, settle in by inning 3) while others are sharper early (fresh velocity, best command before fatigue).
- **First-Batter-Faced (FBF) OBP** — the single highest-correlation event to a YRFI is the leadoff hitter reaching base.
- **K% and BB% (first time through the order)** — use rate stats, not K/9, to measure ball-in-play suppression specifically vs. the top of the lineup.
- **Reliability filter**: only trust first-inning splits for pitchers with a meaningful sample (~50+ career starts). Season debuts, first starts back from injury, or opener/bullpen-game situations should be down-weighted or excluded — splits are unstable on small samples.
- **Velocity trends from warmups / bullpen sessions** — live, same-day signal that overall-season stats can't capture.
- **Between-start bullpen reports** — mechanical or health flags that precede a bad first inning before it shows up in results.

## 3. Advanced Statcast / Contact-Quality Features
For pitchers and opposing hitters, layer in quality-of-contact data beyond outcome stats:
- xBA allowed by pitch type, xSLG allowed, sweet-spot %, blast rate, squared-up %
- Launch angle distribution, exit velocity allowed, weak-contact %, topped %, under %, pulled air-ball %, line-drive %, ground-ball exit velocity
- Expected runs created from contact quality (a more stable signal than actual runs allowed on small samples)

## 4. Pitch Tunneling & Sequencing
- Tunnel differential, release-point variance, pitch deception
- First-pitch selection and two-strike sequencing tendencies
- Velocity separation and horizontal/vertical movement pairing between pitch types
- Historical hitter performance against specific sequences (e.g., four-seam → slider) — sequence context adds predictive signal beyond single-pitch metrics, though it's not sufficient alone.

## 5. Hitter Approach Profiles (beyond OBP)
Swing%, zone swing%, chase%, contact%, zone contact%, whiff%, called-strike%, foul-ball%, pull%, oppo%, spray tendency, ground-ball%, fly-ball%, pop-up%, line-drive%.

## 6. Lineup Construction — The "Top 4" Rule
- For NRFI/YRFI you effectively only care about hitters 1–4 in the order (only they're guaranteed to bat in the first inning under most sequences).
- Use **OBP, ISO, and wRC+ specifically vs. the starter's handedness** (platoon splits, not season totals).
- A missing 1- or 2-hole hitter (rest day, injury) meaningfully lowers early-scoring probability — this is one of the fastest ways to find value not yet priced by the market.
- **Lineup timing**: MLB lineups are typically posted 2–4 hours before first pitch. A late scratch or a hot bat inserted last-minute is a real-time edge before books fully adjust.

## 7. Situational Splits
Bases empty vs. runners on, first inning only, first plate appearance, day/night, home/road, outdoor/indoor, temperature splits, performance vs. fastball-heavy vs. breaking-ball-heavy pitchers, velocity buckets (95+ mph, etc.).

## 8. Baserunning (overlooked for YRFI)
Stolen-base attempt rate, extra-base advancement %, first-to-third %, home-to-first speed, double-play avoidance, leadoff-hitter aggressiveness — all relevant to converting a leadoff baserunner into a run.

## 9. Catcher Effects
Framing by pitch location (low-zone and high-zone framing separately), personal catcher splits, pitcher comfort/history with a given catcher, catcher game-planning tendencies.

## 10. Environmental & Umpire "Force Multipliers"
This is where sharps separate from recreational bettors — environment often outweighs the pitching matchup itself.
- **Umpire zone size**: a "wide zone" umpire (historically high K/game, low BB/game) effectively upgrades a mediocre pitcher for that game. Umpire assignments should be checked as soon as they're published.
- **Umpire volatility/inconsistency**: some umpires produce higher "run contribution" via inconsistent calls, pushing deeper counts and more walks — a lean toward YRFI.
- **Wind direction and speed at first pitch** — at extreme parks, sustained wind blowing out can shift a total by 1.5–2 runs; always verify at first pitch, not at first-pitch-minus-3-hours, since it can shift.
- **Temperature/humidity/density altitude** — heat + low humidity increases ball carry; density altitude is the most complete single measure of expected ball flight if available.
- **Park factors** specific to early-game conditions (roof status, day-game sun angles, etc.).
- **Roof status changes and real-time wind shifts** for dome/retractable-roof parks are a same-day signal, not a static park factor.

## 11. Psychological & Scheduling Factors
Debut nerves, returning from injury, trade-deadline adjustment, contract year, rivalry intensity, clinching/elimination pressure, cross-country travel, circadian-rhythm effects on early getaway-day/day-after-travel starts.

## 12. Team Strategy Tendencies
First-inning steal frequency, sacrifice-bunt tendency, "green light" managers, lineup-optimization quality, platoon usage, first-time-through-the-order aggressiveness.

## 13. Betting Market Intelligence
Professionals model the market itself as an input, not just the game:
- **Opening vs. closing line**, sharp-book vs. recreational-book movement, market disagreement between books, liquidity, steam timing, overnight moves, limit increases, public ticket % vs. public money %, consensus line, de-vigged fair odds.
- **Closing Line Value (CLV)** is the standard measure sharp bettors use to judge whether their process is sound independent of single-game outcomes — e.g., betting NRFI at -115 that closes at -130 is a good process decision regardless of the game result.
- **Reverse line movement**: if public money is on one side but the line moves the other way, that's a signal of sharp money on the other side.
- **Cross-book arbitrage signals** and **closing-line prediction models** — used by the most sophisticated shops to anticipate where the market will settle, not just react to it.

## 14. Projection Calibration (model quality control)
Don't just ask "will NRFI hit?" — measure whether your probabilities are honest: Brier score, log loss, calibration curves, reliability diagrams, probability binning, expected value by edge tier, ROI by confidence bucket.

## 15. Ensemble Modeling
No single "magic stat" wins this market. Serious models combine multiple approaches and weight/stack them:
- Logistic regression, gradient boosting, random forests, Bayesian models, Poisson models, Skellam distributions, Monte Carlo simulations — combined/stacked rather than used individually.

## 16. What Betting Syndicates Track That Casual Bettors Don't
- Live Statcast updates and in-game trend shifts
- Bullpen warm-up activity (relevant to F5 strategy specifically, since F5 is a bet on starter + early offense with bullpen variance removed)
- Last-minute lineup substitutions and batting-order changes
- Umpire assignment changes after initial posting
- Sportsbook limit increases (a tell that a book is comfortable taking more sharp action on a side)
- Injury news before it reaches the wider market
- Proprietary player ratings updated daily, with automated feature re-simulation whenever new data arrives

## 17. F5-Specific Framing
F5 lets you isolate a starter/lineup mismatch while stripping out bullpen risk — useful when you like the starters but the bullpens are a wash or work against your side. The same first-inning/early-inning inputs above (pitcher first-times-through-order data, top-4 lineup quality, umpire, weather) drive F5 totals, just extended across a longer window with slightly more sample and slightly less extreme variance than single-inning NRFI/YRFI.

## 18. Elite Daily Execution Checklist
1. **Filter**: Does the starter have a strong first-inning ERA / first-time-through-the-order profile (not just season ERA)?
2. **Validate**: Are the opposing top-4 hitters currently underperforming vs. this pitcher's specific handedness?
3. **Adjust**: Does the plate umpire's zone tendency match your side (wide zone favors Under/NRFI)?
4. **Confirm lineups**: Check 2–4 hours pre-game for scratches, rest days, or hot-bat insertions.
5. **Check weather/wind at first pitch**, not just at handicapping time — it can shift.
6. **Check line movement**: if you're on the Under and the line is drifting toward the Over, pause — the market may know something (wind, lineup, injury) you haven't seen yet.
7. **Log CLV** on every bet regardless of outcome, to evaluate process over time.

## 19. Common Pitfalls to Avoid
- **Season-long ERA/WHIP as your primary pitcher input** — use first-inning/first-time-through splits instead.
- **Trusting first-inning splits on small samples** — season debuts, injury returns, and openers should be flagged or filtered out of your model rather than trusted at face value.
- **Parlaying multiple NRFI legs**: compounds variance and juice across legs; even with individually +EV legs, parlay EV is typically worse than flat single-game staking at proper unit sizing.
- **Ignoring lineup changes after your initial model run** — a late scratch can flip a game's true probability meaningfully; re-check before close.
- **Assuming name recognition = NRFI safety** — even elite strikeout starters occasionally allow a first-inning run; check the actual first-inning data, not reputation.

---

### Notes on building this into Da-One
Given your current architecture, the fields above map cleanly onto tiers you could formalize similarly to `_ABS_GUARDRAILS` / `_REL_TIERS`: a **first-inning-specific feature tier** (pitcher FBF OBP, first-inning ERA/K%/BB%), a **lineup-quality tier** (top-4 platoon wRC+/OBP/ISO), an **environment tier** (umpire zone size, wind/park factors), and a **market tier** (CLV tracking, reverse line movement flags). The reliability filter (50+ starts, exclude debuts/injury-returns/openers) is a good candidate for a gatekeeper stage similar to your existing stability filter logic.
