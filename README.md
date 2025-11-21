# ⚽ Premier League Standings Prediction

A comprehensive machine learning project comparing **5 algorithms** across **2 business objectives** to predict Premier League outcomes using 25 seasons of historical data (2000-2025).

## 📋 Project Overview

This project implements two distinct business objectives for Premier League prediction:
1. **Season Ranking Prediction** (Regression): Predict final league positions (1-20)
2. **Match Outcome Prediction** (Classification): Predict individual match results (Home/Draw/Away)

Each business objective uses 5 algorithms with comprehensive GridSearchCV hyperparameter tuning, testing thousands of parameter combinations per model.

### 🎯 Key Achievements

- ✅ **2 Business Objectives**: Season rankings + Match outcomes
- ✅ **5 Algorithms Per BO**: Random Forest, Gradient Boosting, Decision Tree, KNN, SVM
- ✅ **Comprehensive Tuning**: GridSearchCV with optimized parameter grids (100-600 combinations per model)
- ✅ **25 Seasons of Data**: 500+ team-seasons, ~9,500 matches (2000-2025)

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

**Baseline**: 33% (random guessing for 3-class problem)

**Features**: 15 total (3 identifiers + 12 match statistics)
- Team identifiers: HomeTeam, AwayTeam, Season
- Match statistics: Shots, Shots on Target, Fouls, Corners, Yellow Cards, Red Cards (home and away)

**Key Features**: 
- Raw match statistics allow models to learn patterns directly
- League average stats used for 2025-26 forecasting
- All 380 fixtures predicted for 2025-26
- Complete league table simulation from match predictions
- Accounts for relegated/promoted teams with proper encodings

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
│   ├── data_preprocessing.ipynb            # Data cleaning & aggregation
│   └── exploratory_analysis.ipynb          # Data exploration
│
├── docs/                                    # Documentation
│   ├── tableau_comparatif.md               # Comparative analysis table
│   └── conclusion_finale.md                # Final evaluation report
│
├── scripts/                                 # Python automation scripts
│   └── combine_datasets.py                 # Merge season files
│
└── README.md                                # Project documentation
```

### 📓 Notebook Descriptions

**Business Objectives (Main Analysis):**
- **`BO1_season_ranking_comparison.ipynb`**: 🏆 Season ranking prediction (Regression)
  - Predicts final league position (1-20) for each team
  - 5 algorithms with GridSearchCV (1,000-5,000 combinations each)
  - Evaluation: MAE, RMSE, R², accuracy within ±1/±2 positions
  - 2025-26 forecast with ensemble averaging

- **`BO2_match_winner_comparison.ipynb`**: ⚽ Match outcome prediction (Classification)
  - Predicts Home Win, Draw, or Away Win for each match
  - 5 algorithms with GridSearchCV (120-8,640 combinations each)
  - Evaluation: Accuracy, F1, ROC AUC, per-class performance
  - Complete 380-match 2025-26 forecast with predicted league table

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

### 📊 Recommended Workflow

1. **Explore the data**: `exploratory_analysis.ipynb`
2. **Understand preprocessing**: `data_preprocessing.ipynb`
3. **Run BO1**: `BO1_season_ranking_comparison.ipynb` 
   - Season ranking prediction (regression)
   - ~15-20 minutes to complete GridSearchCV
4. **Run BO2**: `BO2_match_winner_comparison.ipynb`
   - Match outcome prediction (classification)
   - ~20-25 minutes to complete GridSearchCV
5. **Review outputs**: Check generated forecasts and visualizations

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
- **Improvement**: 57-59% accuracy vs 33% baseline (76% improvement)
- **Key Insight**: Raw match statistics (shots, fouls, corners, cards) provide crucial context
- **2025-26 Forecast**: All 380 matches predicted with complete league table
  - Includes all 20 teams with correct encodings
  - Uses league average stats for forecasting
  - Simulates entire season from individual match predictions
- **Innovation**: 15-feature model (expanded from 3) with match statistics
- **Business Value**: Betting analysis, match previews, fan engagement

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
- 15 features: Home Team, Away Team, Season + 12 match statistics (shots, shots on target, fouls, corners, yellow cards, red cards)
- Target: Match outcome (Home Win/Draw/Away Win)

---
**⚠️ Note**: This project is for educational and research purposes. Predictions should not be used for commercial betting or gambling activities.
