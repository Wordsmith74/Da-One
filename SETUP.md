# Deployment Setup

One-time setup to get the engine running automatically and publishing picks.

## 1. Create the repo
- Create a new **private** GitHub repo (recommended, since this is for friends only).
- Push this entire `sports-engine/` folder to it.
- Make sure `config/.env` (your real keys) is in `.gitignore` -- never commit it.
  `.env.example` is fine to commit, it's just a template.

## 2. Add your API keys as GitHub Secrets
Repo -> Settings -> Secrets and variables -> Actions -> New repository secret.
Add these two (use FRESH keys -- rotate any posted earlier in chat first):
- `BALL_DONT_LIE_KEY`
- `THE_ODDS_API_KEY`

(`PROP_LINE_API_KEY` and `RAPIDAPI_KEY` from earlier drafts of this doc aren't
read by any current code path in data/fetch.py -- skip them unless you add a
feature that needs them.)

These are injected as environment variables during the Actions run -- never stored
in the repo itself.

## 3. Enable GitHub Pages
Repo -> Settings -> Pages -> Source -> select **GitHub Actions** (not "Deploy from branch").
The workflow handles publishing `output/` automatically after each run.

## 4. Enable the workflow
The workflow file is already at `.github/workflows/daily-picks.yml`. Once pushed,
it will:
  - Run daily at 10:00 UTC (edit the `cron` line in the workflow to change timing)
  - Can also be triggered manually: repo -> Actions tab -> "Generate Daily Picks" -> Run workflow
  - Commits the fresh `output/picks.json` back to the repo
  - Deploys `output/` (the picks.json + index.html) to GitHub Pages

## 5. Find your URL
After the first successful run: repo -> Settings -> Pages will show your live URL,
typically `https://<your-username>.github.io/<repo-name>/`.

## Before going live
`USE_LIVE_DATA = True` is already set in `run_pipeline.py` -- the pipeline pulls
real games/odds/rosters/injuries for all three markets (MLB F5, MLB K-prop,
WNBA points). What's NOT done:

- **Nothing here has been live-tested against real API responses** (built in a
  sandbox with no network access). The exact field names assumed for
  balldontlie game/score objects, The Odds API's F5 market key, and the
  RotoWire injury-page markup are documented best guesses, not verified facts.
  **Run it once manually first** (Actions tab -> "Generate Daily Picks" -> Run
  workflow) and read `output/run_log.json` for `[warn]`/`[ERROR]` lines before
  trusting any pick it produces.
- If preflight fails (missing API key), the pipeline now aborts before
  generating any picks rather than silently running partial/mock data.
- Run `models/backtest.py` against real cached history (`data/cache_history.py`)
  to sanity-check the model's constants before trusting any picks it generates.
- Repo being private + GitHub Pages being public-by-URL is a slight mismatch --
  the *code* stays private, but the published picks page is reachable by anyone
  with the link. Fine for a friends group; just don't post the link publicly.
