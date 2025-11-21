# Processed Data - Premier League Datasets

This folder contains two processed datasets ready for machine learning:

1. **`team_season_aggregated.csv`** - Team season performance (for standings prediction)
2. **`processed_premier_league_combined.csv`** - Match-by-match results (for match outcome prediction)

---

## Quick Guide

| Goal | Use This File | Why |
|------|---------------|-----|
| **Predict final standings (1-20)** | `team_season_aggregated.csv` | ✅ No data leakage, aggregated season stats |
| **Predict match outcomes (W/D/L)** | `processed_premier_league_combined.csv` | Match-level features |
| **Analyze team performance trends** | `team_season_aggregated.csv` | Season-level aggregates |
| **Study individual matches** | `processed_premier_league_combined.csv` | Match-level details |

---

## Dataset 1: `team_season_aggregated.csv`

**Best for: Predicting final Premier League standings (positions 1-20)**

### What is it?
Each row represents one team's complete season performance. This dataset aggregates all match results into season-level statistics.

### Structure
- **Rows:** ~500 team-seasons (25 seasons × 20 teams)
- **Columns:** 35 features
- **One row per:** Team per season (e.g., "Liverpool 2023-24")

### Key Columns

**Identifiers:**
- `Season` - Season year (e.g., "2023-24")
- `Season_encoded` - Numeric season ID (0, 1, 2...)
- `Team` - Team name (e.g., "Liverpool", "Arsenal")
- `Team_encoded` - Numeric team ID

**Target Variable:**
- `Final_Position` - Final league position (1=Champion, 20=Relegated)
  - Calculated using official Premier League rules:
    1. Most points
    2. Goal difference
    3. Goals scored

**Match Statistics:**
- `Matches_Played` - Total matches (usually 38)
- `Home_Matches`, `Away_Matches` - Home/away split
- `Wins`, `Draws`, `Losses` - Match outcomes
- `Home_Wins`, `Away_Wins` - Venue-specific wins
- `Points` - Total points (3 per win, 1 per draw)
- `Points_Per_Game` - Average points per match

**Goal Statistics:**
- `Goals_Scored`, `Goals_Conceded` - Total goals
- `Goal_Difference` - Goals scored - goals conceded
- `Avg_Goals_Scored`, `Avg_Goals_Conceded` - Per match averages

**Shooting Statistics:**
- `Total_Shots`, `Total_Shots_On_Target` - Season totals
- `Avg_Shots`, `Avg_Shots_On_Target` - Per match averages
- `Shot_Accuracy` - Percentage of shots on target

**Defensive Statistics:**
- `Clean_Sheets` - Matches with zero goals conceded
- `Clean_Sheet_Rate` - Percentage of clean sheets

**Disciplinary:**
- `Yellow_Cards`, `Red_Cards` - Total cards
- `Fouls` - Total fouls committed

**Other:**
- `Corners`, `Avg_Corners` - Corner kicks
- `Win_Rate`, `Home_Win_Rate`, `Away_Win_Rate` - Win percentages

### Example Row
```
Season: 2023-24
Team: Liverpool
Final_Position: 1
Points: 92
Wins: 28
Goal_Difference: +68
Shot_Accuracy: 45.2%
```

### Why Use This for Standings Prediction?
✅ **No data leakage** - Uses only end-of-season aggregates  
✅ **Realistic** - Features directly correlate with league position  
✅ **Clean target** - Final_Position calculated using official PL rules  
✅ **Interpretable** - Easy to understand what drives predictions  

### Usage Example
```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv('team_season_aggregated.csv')

# Select features
X = df[['Wins', 'Goal_Difference', 'Goals_Scored', 'Clean_Sheets', 
        'Shot_Accuracy', 'Points_Per_Game']]
y = df['Final_Position']

# Train model
model = RandomForestRegressor()
model.fit(X, y)
```

---

## Dataset 2: `processed_premier_league_combined.csv`

**Best for: Predicting match outcomes (Home win / Draw / Away win)**

### What is it?
Each row represents one Premier League match with statistics for both teams.

### Structure
- **Rows:** ~10,000 matches (25 seasons × ~380 matches/season)
- **Columns:** 23 features
- **One row per:** Individual match (e.g., "Arsenal vs Chelsea on 2023-10-21")

### Key Columns

**Match Identifiers:**
- `Season` - Season year (e.g., "2023-24")
- `Season_encoded` - Numeric season ID
- `Date` - Match date (YYYY-MM-DD format)

**Teams (Original):**
- `HomeTeam` - Home team name (e.g., "Arsenal")
- `AwayTeam` - Away team name (e.g., "Chelsea")

**Teams (Encoded for ML):**
- `HomeTeam_le` - Home team label-encoded (0-49)
- `AwayTeam_le` - Away team label-encoded (0-49)

**Target Variable:**
- `FTR` (Full Time Result) - Original: H/D/A
  - `H` = Home win
  - `D` = Draw
  - `A` = Away win
- `FTR_encoded` - Encoded for ML: 0/1/2
  - `0` = Away win
  - `1` = Draw
  - `2` = Home win

**Goals:**
- `FTHG` - Full Time Home Goals
- `FTAG` - Full Time Away Goals

**Shots:**
- `HS` - Home Shots
- `AS` - Away Shots
- `HST` - Home Shots on Target
- `AST` - Away Shots on Target

**Fouls:**
- `HF` - Home Fouls
- `AF` - Away Fouls

**Corners:**
- `HC` - Home Corners
- `AC` - Away Corners

**Cards:**
- `HY` - Home Yellow Cards
- `AY` - Away Yellow Cards
- `HR` - Home Red Cards
- `AR` - Away Red Cards

### Example Row
```
Season: 2023-24
Date: 2023-10-21
HomeTeam: Arsenal
AwayTeam: Chelsea
FTHG: 2
FTAG: 1
FTR: H (Home win)
HS: 18, HST: 8
```

### Why Use This for Match Prediction?
✅ **Match-level detail** - Captures individual game dynamics  
✅ **Both teams encoded** - Ready for ML models  
✅ **Rich statistics** - Goals, shots, fouls, corners, cards  
✅ **Temporal data** - Date column for time-based analysis  

### Usage Example
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv('processed_premier_league_combined.csv')

# Select features (15 total: 3 identifiers + 12 match stats)
X = df[['HomeTeam_le', 'AwayTeam_le', 'Season_encoded',
        'HS', 'AS',      # Shots
        'HST', 'AST',    # Shots on target
        'HF', 'AF',      # Fouls
        'HC', 'AC',      # Corners
        'HY', 'AY',      # Yellow cards
        'HR', 'AR']]     # Red cards
y = df['FTR_encoded']  # 0=Away win, 1=Draw, 2=Home win

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Expected performance: ~57-59% accuracy (vs 33% baseline)
# Interpret predictions using original team names
print(f"Match: {df['HomeTeam'][0]} vs {df['AwayTeam'][0]}")
print(f"Result: {df['FTR'][0]}")
```

---

## Key Differences

| Feature | `team_season_aggregated.csv` | `processed_premier_league_combined.csv` |
|---------|----------------------------|----------------------------------------|
| **Granularity** | Season-level (1 row per team per season) | Match-level (1 row per match) |
| **Use Case** | Predict final standings (1-20) | Predict match outcomes (W/D/L) |
| **Rows** | ~500 | ~10,000 |
| **Target** | `Final_Position` (1-20) | `FTR_encoded` (0/1/2) |
| **Features** | Aggregated season stats | 15 total: 3 identifiers + 12 match stats |
| **Performance** | MAE ~2.5 positions | 57-59% accuracy (vs 33% baseline) |
| **Interpretability** | Season performance metrics | Match-specific events |

---

## Data Preprocessing

Both files were generated by `notebooks/data_preprocessing.ipynb`:

### From Raw Data
1. **Load** raw matches from `data/raw/combined/premier_league_combined.csv`
2. **Filter** complete seasons (2000-01 onwards with full statistics)
3. **Clean** remove missing values, drop unused columns
4. **Encode** categorical variables for ML models

### Match-Level Processing → `processed_premier_league_combined.csv`
- Encode teams (HomeTeam_le, AwayTeam_le)
- Encode results (FTR_encoded)
- Keep match statistics (shots, fouls, corners, cards)
- Preserve original columns for interpretation

### Season-Level Aggregation → `team_season_aggregated.csv`
- Group matches by team and season
- Calculate aggregates (total wins, goals, shots, etc.)
- Calculate derived metrics (win rate, shot accuracy, etc.)
- Compute Final_Position using official PL rules
- Encode teams and seasons numerically

---

## Important Notes

### For `team_season_aggregated.csv`:
- ✅ Uses **end-of-season aggregates** - no data leakage
- ✅ Final_Position follows **official Premier League rules**
- ✅ Perfect for **standings prediction models**
- Expected model performance: MAE ~0.2-1.5 positions

### For `processed_premier_league_combined.csv`:
- ✅ Uses **15 features**: 3 identifiers + 12 match statistics
- ✅ Match statistics capture team playing styles and match dynamics
- ✅ Perfect for **match outcome prediction**
- ✅ Use **Date column** for proper train-test splits (temporal)
- Expected model performance: 57-59% accuracy (vs 33% baseline)

---

## File Locations

```
data/processed/
├── team_season_aggregated.csv          # Season-level (500 rows)
├── processed_premier_league_combined.csv   # Match-level (10,000 rows)
└── README.md                            # This file
```

---

## Quick Stats

### `team_season_aggregated.csv`
- **Seasons:** 2000-01 to 2024-25 (25 seasons)
- **Teams per season:** 20
- **Total rows:** ~500
- **Features:** 35 (identifiers + performance metrics)
- **Target:** Final_Position (1-20)

### `processed_premier_league_combined.csv`
- **Seasons:** 2000-01 to 2024-25 (25 seasons)
- **Matches per season:** ~380
- **Total rows:** ~10,000
- **Features:** 15 total (3 identifiers + 12 match statistics)
  - Identifiers: HomeTeam, AwayTeam, Season
  - Match stats: Shots, Shots on Target, Fouls, Corners, Yellow Cards, Red Cards (home & away)
- **Target:** FTR_encoded (0/1/2)
- **Target distribution:** ~46% Home wins, ~26% Draws, ~28% Away wins
- **Best model performance:** 57-59% accuracy with Random Forest/SVM

---

For more details on preprocessing steps, see `notebooks/data_preprocessing.ipynb`.
