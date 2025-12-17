1. What you already have (conceptually)

Based on the scripts and logs we’ve been working through, you already have:
	•	Raw game / player data in SQLite via:
	•	player_game_stats (per player per game).
	•	impact_lineup_stints (possession-level stint data, cached).
	•	impact_ratings_* (per-team, per-season player impact outputs).
	•	Derived season aggregates:
	•	export_profiles_from_db.py → player_profiles_from_db_2025-26.csv/json
	•	Impact model & ratings:
	•	export_impact_ratings.py → impact_per_100 + EPM-like features per player.
	•	Gameday features:
	•	export_gameday_profiles.py → gameday_player_profiles.(csv/json)
	•	export_gameday_matchups.py → gameday_matchups_(date).(csv/json)
These already contain:
	•	Team-level aggregated stats (off/def ratings, EPM aggregates, etc.).
	•	Schedule-derived team form (record, recent form, plus/minus).
	•	Player-level season stats + impact metrics.

That’s basically your feature factory. Good news: this is exactly what you want beneath a prediction model.

⸻

2. What you want the model to do

You want two main layers of predictions for each game:

A. Team-level predictions

For each matchup:
	1.	Winner / moneyline probability
	•	P(home_win) (and implicitly P(away_win)).
	2.	Spread / margin
	•	Predict home_margin = home_pts - away_pts.
	3.	Team totals
	•	Predict home_points, away_points.
	4.	Game total
	•	Predict total_points = home_points + away_points.

So at minimum: a multi-output regression + classification per game.

B. Player-level predictions

For each player in each matchup:
	•	Minutes
	•	Rate stats / counting stats:
	•	PTS, REB, AST, 3PM, FGA, FTA, STL, BLK, TOV, etc.
	•	All informed by:
	•	Their season profile (usage, efficiency, impact).
	•	Opponent defensive profile (team defense, DVP-style by position/role).
	•	Context: injury-driven role changes, depth chart, recent minutes, etc.

And you want this continuously retrained, so it naturally improves as new games come in.

⸻

3. Turning your current pipeline into an ML system

Think of the project now as three layers:
	1.	Data & Features – you already have this (DB + daily exports).
	2.	Training dataset builders – new scripts that backfill historical games into ML-ready tables.
	3.	Models + evaluation + inference – train, validate, then use them on today’s gameday exports.

3.1. Training dataset layer

3.1.1 Team-game training table
Create a team_matchup_training dataset with one row per historical game.
	•	Inputs (features):
	•	Everything you produce in gameday_matchups except the explicit targets:
	•	Aggregated EPM / impact features (home/away + diffs).
	•	Offensive / defensive ratings per 100.
	•	Shooting profiles (TS%, eFG%, 3P%, FT%, TOV ratio, usage).
	•	Team form:
	•	home_games_played, home_win_pct, home_recent_win_pct, etc.
	•	Plus all the diff_* features.
	•	Optionally more:
	•	Rest days, back-to-back flags (you can derive from schedule).
	•	Home/away flags (already encoded in home/away).
	•	Targets (labels) per game (from team boxscores):
	•	home_pts, away_pts.
	•	home_margin = home_pts - away_pts.
	•	home_win flag (1 if home_pts > away_pts).
	•	total_points = home_pts + away_pts.

Implementation pattern:
	•	Build a script like src/scripts/build_team_train_dataset.py that:
	1.	Loops over all games in one or more seasons.
	2.	For each game, reconstructs the same features you use in gameday_matchups (but from the pre-game vantage point).
	3.	Joins in actual results from team_game_stats or boxscore tables.
	4.	Writes a training CSV/Parquet: data/model/team_matchup_train_2023-24.parquet, etc.

You’re already computing these features for a single date; the builder just generalizes that to many dates.

3.1.2 Player-game training table
Create player_game_training with one row per (game, player):
	•	Inputs:
	•	From player_profiles_from_db at the pre-game snapshot:
	•	Season per-game stats (PTS, REB, AST, 3PM, etc.).
	•	Efficiency metrics (TS%, eFG%, usage, etc.).
	•	Impact metrics (EPM, impact_per_100, exposure_stint_units).
	•	From gameday_matchups / opponent features:
	•	Opponent team ratings (defensive rating, TS% allowed, etc.).
	•	DVP-style features you’ll compute:
	•	For each team and position, allowed PTS/REB/AST per 36 or per 100 possessions vs that position.
	•	Context:
	•	Recent minutes (rolling average over last N games).
	•	Starting vs bench flag (if available via lineup/rotation; if not, approximate with minutes share).
	•	Team injuries or absence proxies:
	•	E.g., share of team minutes at position, or “on/off” minutes for key stars.
	•	Targets (per game stats from player boxscores):
	•	PTS, REB, AST, 3PM, FGA, FTA, STL, BLK, TOV, MIN, etc.
	•	You can model some as:
	•	per_minute or per_possession rates, with minutes as a separate target or input.

Implementation:
	•	Script src/scripts/build_player_train_dataset.py:
	1.	For each game in your training horizon:
	•	List all players who actually played.
	•	Fetch their pre-game season profile snapshot.
	•	Join with that game’s opponent / matchup features.
	•	Attach the actual boxscore stat line as labels.
	2.	Save to data/model/player_game_train_202x-yy.parquet.

The heavy lifting is feature engineering; your current pipeline already gives you most of what you need.

⸻

3.2. Model layer

Now you can train actual ML models.

3.2.1 Team models
You can structure this as:
	•	Model 1: Predict home_margin (regression).
	•	Model 2: Predict total_points (regression).
	•	Optional Model 3: Predict home_win (classification).
	•	Or derive from the distribution of home_margin.

Recommended model families:
	•	Gradient-boosted trees (XGBoost, LightGBM, or CatBoost):
	•	Handle non-linearities & feature interaction well.
	•	Robust to mixed-scale features.
	•	Multi-output regression:
	•	Use the same feature vector to produce home_pts and away_pts directly, or:
	•	Single-target models with correlated structure (margin and total) and derive others:
	•	home_pts = (total + margin) / 2
	•	away_pts = total - home_pts

You can start simple:
	•	train_team_model.py:
	•	Loads team_matchup_train_*.parquet.
	•	Splits by time (train on earlier games, validate on later ones).
	•	Trains:
	•	A regression model for margin.
	•	A regression model for total_points.
	•	Saves models under models/team_margin.pkl and models/team_total.pkl.

During inference:
	•	Use gameday_matchups_{today}.csv as input.
	•	Predict margin_pred, total_pred, then derive:
	•	home_pts_pred, away_pts_pred.
	•	P(home_win) via logistic model or by approximating the distribution (e.g., assuming normal residuals over margin and using CDF at 0).

3.2.2 Player models
Best structure:
	1.	Minutes model:
	•	Target: MIN or minutes_share (minutes / team minutes).
	•	Inputs:
	•	Season profile + impact stats.
	•	Recent minutes usage.
	•	Role features, e.g., depth chart / position.
	•	Injury proxies (e.g., star teammates missing).
	•	Output: predicted minutes for that game.
	2.	Rates model for per-minute stats:
	•	Targets:
	•	PTS_per_min, REB_per_min, AST_per_min, etc.
	•	Inputs:
	•	Season scoring/usage efficiency.
	•	Opponent defensive and DVP metrics.
	•	Predicted minutes (or minutes bucket).
	•	Later, you can let the model jointly predict minutes and rates, but two-stage is usually easier.
	3.	Combine:
	•	PTS_pred = MIN_pred * PTS_per_min_pred, etc.

Model families:
	•	Tree-based models per-stat, or
	•	A multi-task neural network that outputs all stats together.

Given your current stack is pure Python, tree-based models with scikit-learn or XGBoost will be straightforward and performant.

⸻

3.3. Continuous retraining

You already run:
	•	update_db_today.py after games.
	•	export_profiles_from_db.py
	•	export_impact_ratings.py
	•	export_gameday_profiles.py
	•	export_gameday_matchups.py

Add scripts:
	•	train_team_model.py
	•	train_player_model.py

And fold them into an offline schedule, like:
	•	Nightly / weekly:
	•	Rebuild training datasets with the latest games appended.
	•	Retrain models on a rolling window (e.g., last 2–3 seasons).
	•	Save versioned models:
	•	models/team_margin_2025-12-17.pkl
	•	models/player_stats_2025-12-17.pkl
	•	Log performance: MAE, RMSE, calibration, etc.
	•	Gameday:
	•	Run your existing daily pipeline.
	•	Then run:
	•	predict_team_matchups.py (team-level outputs).
	•	predict_player_stats.py (player-level box score projections).
	•	Export to CSV/JSON and upload to S3 alongside gameday profiles/matchups.

⸻

4. Can the model be “smarter” than this?

Yes, in several ways:

4.1. Model distributions, not just point estimates

Instead of just predicting point values:
	•	Use models that output full distributions:
	•	Quantile regression (predict p10 / p50 / p90 of points).
	•	Probabilistic models (e.g., Gaussian with mean & variance, or even Poisson for counts).
	•	Then you can derive:
	•	Probabilities of going over/under a line.
	•	Tail events (e.g., star scoring 35+).

This is crucial for betting / risk management.

4.2. Lineup-dependent simulations

You already have:
	•	impact_lineup_stints and a possession-impact model.

You can:
	•	Simulate alternative rotations / lineup mixes and see how team efficiency shifts.
	•	Use that to adjust predictions under scenario X (e.g., “star A in foul trouble,” “bench unit plays more,” etc.).

That’s a level beyond typical public models.

4.3. Better DVP & matchup modeling

Go beyond “team allows X PTS to PFs”:
	•	Build features for how a defense plays:
	•	3PA allowed, rim frequency, FT rate allowed, etc.
	•	For each player, encode:
	•	Shot profile (rim vs mid vs 3), drives, post-ups, etc. (if you ever integrate play-by-play or tracking data).
	•	Align player shot profile vs opponent defensive profile = feature set capturing play-style matchup.

Even without detailed tracking, you can approximate with:
	•	3PA / FGA, FTA / FGA, assisted vs unassisted shares, etc.

4.4. Joint modeling of team & players

Your current design is layered:
	•	Team model and player model are separate.

Smarter version:
	•	Constrain them to be consistent:
	•	Sum of predicted player points ≈ predicted team points.
	•	Sum of predicted player minutes ≈ 5 × game duration (or realistic range).
	•	You can do this via:
	•	Post-processing normalization.
	•	Or optimization (small constrained adjustment problem).

This ensures the outputs adhere to basketball reality and not just per-row ML.

⸻

5. Concrete next steps (roadmap)

Here’s how I’d phase it from where you are today:

Phase 1 – Team model prototype
	1.	Implement team training dataset builder:
	•	build_team_train_dataset.py using your existing matchup_features logic.
	•	Targets: margin, total_points, home_win.
	2.	Train first team models:
	•	LightGBM/XGBoost for margin + total.
	•	Time-based train/validation split.
	•	Evaluate MAE/RMSE on:
	•	margin.
	•	total_points.
	3.	Add inference script:
	•	predict_team_matchups.py that takes gameday_matchups_{today}.csv and writes gameday_team_predictions_{today}.csv/json.
	4.	Optionally: hook this script into run_daily_pipeline.py as step 6.

Phase 2 – Player model prototype
	5.	Implement player training dataset builder:
	•	build_player_train_dataset.py creating per-player-per-game rows with:
	•	Pre-game profile features.
	•	Opponent matchups.
	•	Actual boxscore labels.
	6.	Train minutes model:
	•	Single regression model predicting minutes (or minutes share).
	7.	Train stat-rate models:
	•	Predict PTS_per_min, REB_per_min, AST_per_min, etc.
	8.	Create inference script:
	•	predict_player_stats.py:
	•	Input: gameday_player_profiles.json + gameday_matchups.json.
	•	Output: gameday_player_predictions_{today}.json with full projected box scores.

Phase 3 – Smarts & polish
	9.	Distributional outputs:
	•	Convert margin & totals models to quantile regression or distribution outputs.
	10.	DVP & advanced matchup features:

	•	Build DVP tables from historical player game logs.
	•	Integrate as features into player models.

	11.	Consistency constraints:

	•	Add a post-processing step to reconcile:
	•	Sum player stats to team predictions.
	•	Minutes to team totals.

Phase 4 – Continuous retraining + monitoring
	12.	Automation:

	•	Add training scripts to a weekly cron / CI job.
	•	Log performance over time (dashboard later).

	13.	Model registry / versioning:

	•	Save models with metadata:
	•	Training window, features, metrics.