# Premier League Combined Dataset

## Overview

This dataset contains match results from the English Premier League spanning 32 seasons from 1993-94 to 2024-25.

## Dataset Information

- **File**: `premier_league_combined.csv`
- **Total Records**: 12,324 matches
- **Seasons Covered**: 32 (1993-94 to 2024-25)
- **Date Range**: August 14, 1993 to May 25, 2025
- **Unique Teams**: 51 different teams across all seasons

## Data Schema

| Column | Type | Description |
|--------|------|-------------|
| Season | string | Season in format YYYY-YY (e.g., "2023-24") |
| SourceFile | string | Original filename from which the data was extracted |
| Date | date | Match date (YYYY-MM-DD format) |
| HomeTeam | string | Home team name |
| AwayTeam | string | Away team name |
| FTHG | integer | Full Time Home Goals |
| FTAG | integer | Full Time Away Goals |
| FTR | string | Full Time Result (H=Home Win, D=Draw, A=Away Win) |
| HTHG | integer | Half Time Home Goals |
| HTAG | integer | Half Time Away Goals |
| HTR | string | Half Time Result (H=Home Win, D=Draw, A=Away Win) |
| Referee | string | Match Referee |
| HS | integer | Home Team Shots |
| AS | integer | Away Team Shots |
| HST | integer | Home Team Shots on Target |
| AST | integer | Away Team Shots on Target |
| HF | integer | Home Team Fouls Committed |
| AF | integer | Away Team Fouls Committed |
| HC | integer | Home Team Corners |
| AC | integer | Away Team Corners |
| HY | integer | Home Team Yellow Cards |
| AY | integer | Away Team Yellow Cards |
| HR | integer | Home Team Red Cards |
| AR | integer | Away Team Red Cards |

## Data Quality Notes

1. **Missing Data**: Earlier seasons (1990s) have less detailed statistics (shots, cards, etc.) compared to recent seasons
2. **Team Names**: Some teams may have slightly different name formats across seasons
3. **Date Format**: All dates have been standardized to YYYY-MM-DD format
4. **Chronological Order**: Data is sorted by Season and Date

## Usage Examples

### Loading the Data

```python
import pandas as pd

# Load the combined dataset
df = pd.read_csv('premier_league_combined.csv')

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])
```

### Basic Analysis

```python
# Matches per season
matches_per_season = df.groupby('Season').size()

# Goals scored
df['TotalGoals'] = df['FTHG'] + df['FTAG']

# Home advantage
home_wins = (df['FTR'] == 'H').mean()
```

## Source

Original data sourced from Football Datasets: <https://github.com/datasets/football-datasets>

## Created

Combined on September 24, 2025 using automated script.

## File Location

- **Combined**: `data/raw/combined/premier_league_combined.csv`
- **Original Files**: `data/raw/uncombined/season-*.csv`
