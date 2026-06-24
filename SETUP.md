# Deployment Setup

One-time setup to get the engine running automatically and publishing picks.

## 1. Create the repo
- Create a new **private** GitHub repo (recommended, since this is for friends only).
- Push this entire `sports-engine/` folder to it.
- Make sure `config/.env` (your real keys) is in `.gitignore` -- never commit it.
  `.env.example` is fine to commit, it's just a template.

## 2. Add your API keys as GitHub Secrets
Repo -> Settings -> Secrets and variables -> Actions -> New repository secret.
Add each of these (use FRESH keys -- rotate the ones posted earlier in chat first):
- `BALL_DONT_LIE_KEY`
- `THE_ODDS_API_KEY`
- `PROP_LINE_API_KEY`
- `RAPIDAPI_KEY`

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
- Swap the `mock_fetch_*()` functions in `run_pipeline.py` for real calls into
  `data/fetch.py` -- right now the pipeline runs on hardcoded demo data.
- Run `models/backtest.py` against real cached history (`data/cache_history.py`)
  to sanity-check the model's constants before trusting any picks it generates.
- Repo being private + GitHub Pages being public-by-URL is a slight mismatch --
  the *code* stays private, but the published picks page is reachable by anyone
  with the link. Fine for a friends group; just don't post the link publicly.
