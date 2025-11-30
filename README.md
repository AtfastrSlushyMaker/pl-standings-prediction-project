# ⚽ Premier League Standings Prediction

A comprehensive machine learning project comparing **5 algorithms** across **3 business objectives** to predict Premier League outcomes using 25+ seasons of historical data (2000-2025).

## 📋 Project Overview

This project implements three distinct business objectives for Premier League prediction:
1. **BO1: Season Ranking Prediction** (Regression): Predict final league positions (1-20)
2. **BO2: Match Outcome Prediction** (Classification): Predict individual match results (Home/Draw/Away)
3. **BO3: Team Tactical Segmentation** (Clustering): Identify playing style patterns

Each business objective uses multiple algorithms with comprehensive hyperparameter tuning and real-world validation.

### 🎯 Key Achievements

- ✅ **3 Business Objectives**: Season rankings + Match outcomes + Team segmentation
- ✅ **5 Algorithms (BO1 & BO2)**: Random Forest, Gradient Boosting, Decision Tree, KNN, SVM
- ✅ **3 Clustering Algorithms (BO3)**: K-Means, GMM, DBSCAN
- ✅ **Comprehensive Tuning**: GridSearchCV with optimized parameter grids
- ✅ **25+ Seasons of Data**: 500+ team-seasons, ~9,500 matches (2000-2025)
- ✅ **Real-World Validation**: 40% accuracy on actual 2025-26 season (120 matches)
- ✅ **Enhanced Features**: 25 features including form, shot accuracy, discipline metrics

### 🏆 Business Objectives

#### **BO1: Season Ranking Prediction** (Regression)
**Goal**: Predict final league position (1-20) for each team

**Dataset**: `team_season_aggregated.csv` (500+ team-seasons from 2000-2025)

**Best Models:**
| Model | MAE | RMSE | R² | ±1 Acc | ±2 Acc | Param Combinations |
|-------|-----|------|-----|--------|--------|-------------------|
| Random Forest | ~2.5 | ~3.2 | ~0.90 | ~60% | ~85% | 162 |
| Gradient Boosting | ~2.6 | ~3.3 | ~0.89 | ~58% | ~83% | 144 |
| Decision Tree | ~2.8 | ~3.5 | ~0.87 | ~55% | ~80% | 80 |

**Key Features**: 
- GridSearchCV with 5-fold CV
- Handles promoted teams using historical PL data (e.g., Sunderland 2016-17)
- Ensemble averaging for 2025-26 forecast

#### **BO2: Match Outcome Prediction** (Multi-class Classification)
**Goal**: Predict match winner (Home Win, Draw, Away Win)

**Dataset**: `processed_premier_league_combined.csv` (~9,500 matches)

**Best Models:**
| Model | CV Accuracy | Test Accuracy | ROC AUC | Param Combinations |
|-------|-------------|---------------|---------|-------------------|
| SVM (RBF) | 57.1% | 57.9% | ~0.75 | 36 |
| Random Forest | 52.6% | 59.2% | ~0.76 | 324 |
| Gradient Boosting | 42.8% | 57.4% | ~0.75 | 144 |

**Real-World Performance (2025-26 Season):**
- **Overall Accuracy**: 40.00% (48/120 matches correct)
- **Home Win**: 59.62% accuracy (31/52)
- **Draw**: 21.05% accuracy (4/19) ⚠️ Needs improvement
- **Away Win**: 26.53% accuracy (13/49) ⚠️ Needs improvement
- **Baseline**: 33% (random guessing for 3-class problem)

**Enhanced Features**: 25 total (up from 15)
- Team identifiers: HomeTeam, AwayTeam, Season
- Match statistics: Shots, Shots on Target, Fouls, Corners, Yellow Cards, Red Cards (home and away)
- **NEW - Form features (L5)**: Last 5 matches wins, goals scored/conceded for home/away
- **NEW - Advanced metrics**: Shot accuracy, discipline index

**Key Insights**: 
- Home wins predicted well (60% accuracy)
- **Draw prediction needs major improvement** (only 21% accuracy)
- Model overpredicts home advantage
- High-confidence predictions sometimes fail dramatically (e.g., Man City vs Tottenham 0-4)
- Validation results saved to `outputs/BO2_match_outcomes/prediction_validation.csv`

#### **BO3: Team Tactical Segmentation** (Clustering)
**Goal**: Identify distinct tactical playing styles

**Dataset**: `team_season_aggregated.csv` (500+ team-seasons)

**Algorithms Compared:**
| Algorithm | Clusters | Silhouette Score | Key Strength |
|-----------|----------|-----------------|--------------|
| K-Means | 5 | ~0.45 | Clear tactical categories |
| GMM | 5 | ~0.43 | Soft cluster boundaries (hybrid styles) |
| DBSCAN | Variable | ~0.38 | Outlier detection |

**Tactical Styles Identified:**
1. **Attacking** - High goals, shots, possession (e.g., Man City, Liverpool)
2. **Defensive** - Solid defense, counter-attack (e.g., Mourinho teams)
3. **Possession** - Ball control, patient build-up (e.g., Arsenal)
4. **High-Press** - Aggressive pressing, high intensity (e.g., Klopp's Liverpool)
5. **Pragmatic** - Balanced, flexible approach (mid-table consistency)

**Key Features**:
- 18 tactical features (shots, fouls, corners, cards, win rates)
- Unique style assignment (greedy matching algorithm)
- Tactical evolution tracking over multiple seasons
- Exports to `outputs/BO3_team_segmentation/team_style_clusters.csv`

**Business Value**: 
- Scouting: Match player profiles to compatible tactical systems
- Coaching: Benchmark tactical approaches vs league patterns
- Strategic planning: Track tactical evolution and identify successful patterns

## 📊 Data Source

The historical match data used in this project is obtained from **Football Datasets**, a comprehensive repository of football-related datasets.

**Source**: [https://github.com/datasets/football-datasets](https://github.com/datasets/football-datasets?tab=readme-ov-file#football-datasets)

The dataset includes:

- ⚽ Match results from multiple Premier League seasons
- 📈 Team statistics (goals scored, goals conceded, wins, draws, losses)
- 📅 Date and venue information
- 🏆 Historical league standings
- 📊 Additional performance metrics and team attributes

## 📁 Project Structure

```text
pl-standings-prediction-project/
│
├── data/                                    # Datasets
│   ├── raw/                                 # Original datasets
│   │   ├── combined/
│   │   │   ├── premier_league_combined.csv # All seasons combined
│   │   │   └── README.md
│   │   └── uncombined/                     # Individual season files
│   │       ├── season-2324.csv
│   │       ├── season-2425.csv
│   │       └── ...
│   │
│   └── processed/                           # Cleaned datasets
│       ├── team_season_aggregated.csv      # For BO1 (season rankings) ⭐
│       ├── processed_premier_league_combined.csv # For BO2 (match outcomes) ⭐
│       ├── 2025-26_match_predictions.csv   # BO2 forecast output
│       └── README.md                        # Dataset documentation
│
├── notebooks/                               # Jupyter notebooks
│   ├── BO1_season_ranking_comparison.ipynb # Business Objective 1 ⭐
│   ├── BO2_match_winner_comparison.ipynb   # Business Objective 2 ⭐
│   ├── BO3_team_segmentation.ipynb         # Business Objective 3 ⭐
│   ├── data_preprocessing.ipynb            # Data cleaning & aggregation
│   ├── exploratory_analysis.ipynb          # Data exploration
│   └── outputs/                             # Generated predictions
│       ├── BO1_season_rankings/
│       │   └── 2025-26_season_forecast.csv
│       ├── BO2_match_outcomes/
│       │   ├── 2025-26_match_predictions.csv
│       │   └── prediction_validation.csv   # Real-world accuracy ⭐
│       └── BO3_team_segmentation/
│           └── team_style_clusters.csv
│
├── docs/                                    # Documentation
│   ├── tableau_comparatif.md               # Comparative analysis table
│   └── conclusion_finale.md                # Final evaluation report
│
├── scripts/                                 # Python automation scripts
│   ├── combine_datasets.py                 # Merge season files
│   ├── validate_predictions.py             # Compare predictions with actual results ⭐
│   └── FBref_Data_Fetching.ipynb          # Advanced stats scraping (experimental)
│
└── README.md                                # Project documentation
```

### 📓 Notebook Descriptions

**Business Objectives (Main Analysis):**
- **`BO1_season_ranking_comparison.ipynb`**: 🏆 Season ranking prediction (Regression)
  - Predicts final league position (1-20) for each team
  - 5 algorithms with GridSearchCV
  - Evaluation: MAE, RMSE, R², accuracy within ±1/±2 positions
  - 2025-26 forecast with ensemble averaging

- **`BO2_match_winner_comparison.ipynb`**: ⚽ Match outcome prediction (Classification)
  - Predicts Home Win, Draw, or Away Win for each match
  - 5 algorithms with GridSearchCV
  - **Enhanced with 25 features** including form and advanced metrics
  - Evaluation: Accuracy, F1, ROC AUC, per-class performance
  - Complete 380-match 2025-26 forecast
  - **Real-world validation**: 40% accuracy on actual 2025-26 season results

- **`BO3_team_segmentation.ipynb`**: 🎯 Tactical playstyle clustering (Unsupervised)
  - Identifies 5 distinct tactical playing styles
  - 3 clustering algorithms: K-Means, GMM, DBSCAN
  - Unique style assignment with greedy matching
  - Tracks tactical evolution across seasons
  - Famous teams prioritized in visualizations

**Supporting Notebooks:**
- **`exploratory_analysis.ipynb`**: 🔍 Data exploration and visualization
- **`data_preprocessing.ipynb`**: 🧹 Data cleaning and feature engineering

**Documentation:**
- **`docs/tableau_comparatif.md`**: Comparative analysis across algorithms
- **`docs/conclusion_finale.md`**: Comprehensive evaluation and insights

## 🛠️ Installation

To run this project, you'll need Python 3.7+ and the following packages. We recommend using a virtual environment.

### 🐍 Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 📦 Install Required Packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter plotly
```

### 🔧 Core Dependencies

- **pandas**: 🐼 Data manipulation and analysis
- **numpy**: 🔢 Numerical computing
- **matplotlib**: 📊 Basic plotting and visualization
- **seaborn**: 🎨 Statistical data visualization
- **scikit-learn**: 🤖 Machine learning algorithms and tools
- **jupyter**: 📓 Interactive notebook environment
- **plotly**: 📈 Interactive visualizations (optional)

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/AtfastrSlushyMaker/pl-standings-prediction-project.git
cd pl-standings-prediction-project
```

2. **Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

3. **Run the business objective notebooks**
```bash
jupyter notebook notebooks/
```

4. **Validate predictions against real results**
```bash
python scripts/validate_predictions.py
```

### 📊 Recommended Workflow

1. **Explore the data**: `exploratory_analysis.ipynb`
2. **Understand preprocessing**: `data_preprocessing.ipynb`
3. **Run BO1**: `BO1_season_ranking_comparison.ipynb` 
   - Season ranking prediction (regression)
   - ~15-20 minutes to complete GridSearchCV
4. **Run BO2**: `BO2_match_winner_comparison.ipynb`
   - Match outcome prediction (classification)
   - ~20-25 minutes to complete GridSearchCV
5. **Run BO3**: `BO3_team_segmentation.ipynb`
   - Tactical playstyle clustering (unsupervised)
   - ~5-10 minutes to complete
6. **Validate predictions**: Run `python scripts/validate_predictions.py`
   - Compares BO2 forecasts with actual 2025-26 season results
   - Downloads real match results from football-data.co.uk
   - Generates accuracy metrics and detailed comparison CSV
7. **Review outputs**: Check generated forecasts in `notebooks/outputs/`

### ⏱️ Execution Time

- **BO1**: ~10-15 minutes (testing 600+ parameter combinations with 5-fold CV)
- **BO2**: ~10-15 minutes (testing 600+ parameter combinations with 5-fold CV)
- Use `n_jobs=-1` in GridSearchCV for parallel processing (already configured)
- **Note**: macOS users may see ResourceTracker warnings - these are harmless and can be ignored


## 📈 Key Results

### Business Objective Performance

✅ **BO1: Season Ranking Prediction**
- **Best Model**: Random Forest (MAE: ~2.5 positions, R²: ~0.90)
- **Key Insight**: ~60% teams predicted within ±1 position, ~85% within ±2
- **2025-26 Forecast**: Complete predicted standings with proper team transitions
  - OUT: Southampton, Leicester, Ipswich (relegated)
  - IN: Leeds, Burnley, Sunderland (promoted)
- **Innovation**: Handles promoted teams using their most recent PL season data
- **Business Value**: Financial planning, strategic goal setting, media predictions

✅ **BO2: Match Outcome Prediction**  
- **Best Model**: Random Forest (Test Accuracy: 59.2%, CV: 52.6%)
- **Runner-up**: SVM RBF (Test Accuracy: 57.9%, CV: 57.1% - best generalization)
- **Real-World Performance**: 40.00% accuracy on 2025-26 season (120 matches played)
  - Home wins: 59.62% accuracy ✅
  - Draws: 21.05% accuracy ❌ (major weakness)
  - Away wins: 26.53% accuracy ⚠️
- **Key Insight**: Enhanced with 25 features (form, shot accuracy, discipline)
- **2025-26 Forecast**: All 380 matches predicted with complete league table
- **Innovation**: Real-world validation against ongoing season
- **Business Value**: Betting analysis, match previews, fan engagement

✅ **BO3: Team Tactical Segmentation**
- **Best Algorithm**: K-Means (Silhouette: ~0.45)
- **Clusters**: 5 distinct tactical styles (Attacking, Defensive, Possession, High-Press, Pragmatic)
- **Key Insight**: Unique style assignment ensures no duplicate labels
- **Innovation**: Tracks tactical evolution across multiple seasons
- **Famous Teams**: Prioritized in visualizations (Man City, Liverpool, Arsenal, etc.)
- **Business Value**: Scouting compatibility, tactical benchmarking, strategic planning

### Hyperparameter Tuning Impact

**GridSearchCV Optimization:**
- **BO1**: 80-162 combinations per model (optimized from thousands)
- **BO2**: 20-324 combinations per model (optimized from thousands)
- **Approach**: Reduced grids by ~94% while maintaining performance
- **Runtime**: Cut from hours to 10-15 minutes per notebook

**Key Findings:**
- Ensemble methods (RF, GB) consistently outperform single models
- Proper hyperparameter tuning improves performance by 10-15%
- Cross-validation prevents overfitting (generalization gap < 5%)
- Class balancing critical for imbalanced classification (BO2)
- Multiprocessing optimization speeds up training significantly

### Important Features

**BO1 (Season Rankings):**
1. Points - Direct performance indicator
2. Goal Difference - Overall team quality
3. Wins - Consistency measure
4. Goals Scored - Offensive strength
5. Win Rate - Normalized performance

**BO2 (Match Outcomes):**
1. Team Identities - Home/away team matchups
2. Shots (Home & Away) - Offensive pressure and quality
3. Shots on Target - Finishing accuracy
4. Fouls - Team aggression and playing style
5. Corners - Territorial dominance
6. Cards (Yellow/Red) - Discipline and referee patterns

## 📚 Documentation

- **French Comparative Analysis**: `docs/tableau_comparatif.md`
- **French Conclusion Report**: `docs/conclusion_finale.md`
- **Algorithm READMEs**: Detailed performance metrics in each algorithm folder
- **Colab Notebooks**: All 6 algorithms executable online

## 🎓 Academic Context

This project was developed as part of a Machine Learning course focusing on:
- **Business Objective Alignment**: Two distinct prediction tasks (BO1, BO2)
- **Comprehensive Model Comparison**: 5 algorithms per objective
- **Hyperparameter Optimization**: GridSearchCV with extensive parameter grids
- **Reproducible Research**: Clear structure, documentation, and version control
- **Real-World Application**: Premier League data with practical business value

### 📊 Datasets Used

**BO1**: `team_season_aggregated.csv` (500+ team-seasons)
- Aggregated end-of-season statistics per team
- 10 features: Wins, Draws, Losses, Goals, Points, Win Rate, Clean Sheets
- Target: Final league position (1-20)

**BO2**: `processed_premier_league_combined.csv` (~9,500 matches)
- Match-level data from 2000-2025
- 25 enhanced features: Team identifiers, match statistics, form (L5), shot accuracy, discipline
- Target: Match outcome (Home Win/Draw/Away Win)

**BO3**: `team_season_aggregated.csv` (500+ team-seasons)
- 18 tactical features: Shots, fouls, corners, cards, win rates, efficiency metrics
- Clustering: 5 tactical playing styles

---
**⚠️ Note**: This project is for educational and research purposes. Predictions should not be used for commercial betting or gambling activities.
