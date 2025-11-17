# ⚽ Premier League Standings Prediction

A comprehensive machine learning project implementing **6 different algorithms** to predict Premier League final standings using 25 seasons of historical data (2000-2025).

## 📋 Project Overview

This academic project compares six machine learning algorithms for predicting Premier League final positions based on end-of-season statistics. Each algorithm is implemented, optimized, and evaluated against business objectives with full French documentation.

### 🎯 Key Achievements

- ✅ **6 Complete Algorithms**: Random Forest, XGBoost, SVM, KNN, Decision Tree, Gradient Boosting
- ✅ **Best Performance**: MAE 0.20 positions (Random Forest)
- ✅ **100% Relegation Detection**: SVM classifier
- ✅ **25 Seasons of Data**: 500+ team-season observations
- ✅ **Full Documentation**: Comparative analysis and conclusions in French
- ✅ **All Notebooks Executable**: Google Colab compatible

### 🏆 Algorithm Performance Summary

| Algorithm | MAE | R² | Rank | Strength |
|-----------|-----|-----|------|----------|
| **Random Forest** | 0.20 | 0.95 | 🥇 | Best overall accuracy |
| **XGBoost** | 1.12 | 0.95 | 🥈 | Strong regularization |
| **SVM** | 1.23 | High | 🥉 | 100% relegation detection |
| **KNN** | 1.27 | 0.92 | 4 | Similarity-based predictions |
| **Decision Tree** | 1.5-2.5 | 0.85-0.92 | 5 | Highly interpretable |
| **Gradient Boosting** | 1.62 | Good | 6 | Fast training |

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
│       ├── team_season_aggregated.csv      # For standings prediction ⭐
│       ├── processed_premier_league_combined.csv # For match prediction
│       └── README.md                        # Dataset documentation
│
├── notebooks/                               # Jupyter notebooks
│   ├── algorithms/                          # 6 algorithm implementations ✅
│   │   ├── random_forest/                   # Random Forest (MAE: 0.20) 🥇
│   │   │   ├── random_forest.ipynb
│   │   │   └── README.md
│   │   ├── xgboost/                         # XGBoost (MAE: 1.12) 🥈
│   │   │   ├── xgboost.ipynb
│   │   │   └── README.md
│   │   ├── svm/                             # SVM (100% relegation detection) 🥉
│   │   │   ├── svm_model.ipynb
│   │   │   └── README.md
│   │   ├── knn/                             # KNN (MAE: 1.27)
│   │   │   ├── knn.ipynb
│   │   │   └── README.md
│   │   ├── decision_tree/                   # Decision Tree (Interpretable)
│   │   │   ├── decision_tree.ipynb
│   │   │   └── README.md
│   │   └── gradient_boosting/               # Gradient Boosting (MAE: 1.62)
│   │       ├── gradient_boosting.ipynb
│   │       └── README.md
│   │
│   ├── exploratory_analysis.ipynb          # Data exploration
│   ├── data_preprocessing.ipynb            # Data cleaning & aggregation
│   └── model_training.ipynb                # Combined training notebook
│
├── docs/                                    # French documentation
│   ├── tableau_comparatif.md               # Comparative analysis table
│   ├── conclusion_finale.md                # Final evaluation report
│   └── Objectifs-Data-Science-et-Algorithmes.pdf
│
├── scripts/                                 # Python automation scripts
│   └── combine_datasets.py                 # Merge season files
│
└── README.md                                # Project documentation
```

### 📓 Notebook Descriptions

**Core Notebooks:**
- **`exploratory_analysis.ipynb`**: 🔍 Data exploration, visualization, and pattern analysis
- **`data_preprocessing.ipynb`**: 🧹 Data cleaning, feature engineering, and aggregation
- **`model_training.ipynb`**: 🎯 Combined training and comparison

**Algorithm Implementations (6 complete):**
1. **`random_forest/`**: Random Forest Regressor - Best overall (MAE: 0.20)
2. **`xgboost/`**: XGBoost with regularization - Runner-up (MAE: 1.12)
3. **`svm/`**: SVM for relegation detection - Perfect classification (100%)
4. **`knn/`**: K-Nearest Neighbors - Similarity-based (MAE: 1.27)
5. **`decision_tree/`**: Decision Tree - Interpretable rules (MAE: 1.5-2.5)
6. **`gradient_boosting/`**: LightGBM - Fast training (MAE: 1.62)

**Documentation (French):**
- **`docs/tableau_comparatif.md`**: Comparative table with BO, DSO, and performance metrics
- **`docs/conclusion_finale.md`**: Comprehensive evaluation and recommendations

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
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn shap jupyter
```

3. **Run notebooks**
```bash
jupyter notebook notebooks/
```

### 📊 Workflow

1. **`exploratory_analysis.ipynb`**: Understand the data
2. **`data_preprocessing.ipynb`**: Clean and prepare features
3. **Algorithm notebooks**: Train and evaluate each model
   - Start with `random_forest/random_forest.ipynb` (best performer)
   - Compare with other 5 algorithms
4. **Review documentation**: Check `docs/` for comparative analysis


## 📈 Key Results

### Business Objectives Satisfaction

✅ **All 6 algorithms meet their business objectives**

- **Random Forest**: Predicts final standings with exceptional precision (MAE 0.20)
- **XGBoost**: Maximizes performance with strong regularization
- **SVM**: Detects relegation risks with 100% accuracy (ROC AUC 1.0)
- **KNN**: Predicts positions via team similarity (80% within ±2)
- **Decision Tree**: Provides interpretable decision rules for management
- **Gradient Boosting**: Sequential error correction for balanced predictions

### Top Features (All Models)

1. **Goal Difference** - Primary predictor in all 6 models
2. **Points** - Direct indicator of season performance
3. **Wins** - Number of victories
4. **Goals For** - Offensive efficiency
5. **Clean Sheets** - Defensive stability

## 📚 Documentation

- **French Comparative Analysis**: `docs/tableau_comparatif.md`
- **French Conclusion Report**: `docs/conclusion_finale.md`
- **Algorithm READMEs**: Detailed performance metrics in each algorithm folder
- **Colab Notebooks**: All 6 algorithms executable online

## 🎓 Academic Context

This project was developed as part of a Machine Learning course focusing on:
- Machine learning algorithm comparison
- Business objective alignment (BO)
- Data Science objective evaluation (DSO)
- Reproducible research practices
---
**⚠️ Note**: This project is for educational and research purposes. Predictions should not be used for commercial betting or gambling activities.
