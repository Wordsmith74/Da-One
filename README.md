Pipeline fixes -- MLB strikeout calibration data gaps
Four files changed. Full unified diff in CHANGES.diff; full patched files also included so you can drop them straight into the repo.
1. core/devig.py
Added compute_side_agreement_frac(). Uses the per-book devigged fair probabilities devig.py already computes and measures what fraction of reporting book-weight also favors this side (>50%). This is the real computation behind the side_agreement_frac field that pick_history.jsonl / shadow_log.jsonl have always had a column for, but which was never actually wired up -- nothing upstream ever set it, so it silently logged as null on every published pick.
2. run_pipeline.py
_derive_bet_params(): now computes side_agreement_frac via the above and attaches it to the candidate dict.
run_sport_pipeline(): posterior_std / posterior_mean / a derived relative_sigma_pct are now attached to the candidate the moment the stability check passes (previously these existed only as local variables inside the loop and were dropped as soon as the stability gate passed -- they were only ever logged for rejected candidates, never for picks that actually got published).
The published pick dict and its log_candidate(...) call now carry side_agreement_frac, posterior_std, posterior_mean, and relative_sigma_pct through to both pick_history.jsonl and shadow_log.jsonl.
Precomputed markets (moneyline/spread, which bypass the Bayesian engine) explicitly set these to None rather than leaving the key missing -- honest "not applicable to this market type" rather than a silent gap.
3. data/cache_history.py
append_picks() now persists posterior_std, posterior_mean, relative_sigma_pct on every record.
De-duplication fix: the pipeline can run multiple times a day as odds refresh, and previously re-logged the exact same bet as a brand-new row with a new pick_id every single run. append_picks() now checks existing history for a matching (sport, market, player, side, line, matchup) signature already logged on the same calendar day and skips it. In the sample data you gave me, 83 of 93 graded MLB-strikeout rows were exactly this kind of duplicate (only 20 unique bets existed).
4. backtest.py
grade_pending()'s MLB-strikeout matcher only recognized the legacy sport == "MLB Ks" label. The current pipeline logs strikeout picks as sport == "MLB" + market == "pitcher_strikeouts", which never matched -- so backtest.py's own auto-grader could never grade 97 of the 105 strikeout picks in your history file (they got graded some other way, which is why they already had actual_result filled in). Fixed the match to catch both label formats.
Not fixed here, just documented
edge_pct in your existing pick_history.jsonl almost certainly predates a since-applied fix to calibrate_edge() (there's a comment in _derive_bet_params confirming raw, uncalibrated 24-50% edges used to be written directly). That's very likely why losses show higher average edge than wins in the historical data -- it's probably measurement noise from before the fix, not a real signal. No code change needed since it's already fixed going forward, but don't trust edge-based conclusions drawn from picks generated before this patch.
What this gets you
Once this is deployed and a few weeks of picks accumulate, sigma and agreement will actually be present on every graded strikeout pick, and your original 4-field rule (confidence, edge, sigma, agreement) will be fully backtestable for the first time.
