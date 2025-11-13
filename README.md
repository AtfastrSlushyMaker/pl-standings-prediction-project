# ⚽ Premier League Standings Prediction

A machine learning project that predicts Premier League standings using historical match data and advanced statistical modeling techniques.

## 📋 Project Overview

This project aims to predict the final Premier League standings for a given season by analyzing historical match data, team performance metrics, and various statistical indicators. Using machine learning algorithms, we build predictive models that can forecast team positions, points totals, and overall league table outcomes based on patterns observed in past seasons.

The project combines data science techniques with football analytics to provide insights into team performance trends and season outcomes, making it valuable for sports analysts, football enthusiasts, and anyone interested in predictive modeling in sports.

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
│   ├── algorithms/                          # Individual algorithm implementations
│   │   └── random_forest/                   # Random Forest model
│   │       ├── random_forest.ipynb          # Model notebook
│   │       └── README.md                    # Results & documentation
│   │   # Future: xgboost/, svm/, decision_tree/, k_means/, dbscan/
│   │
│   ├── exploratory_analysis.ipynb          # Data exploration
│   └── data_preprocessing.ipynb            # Data cleaning & aggregation
│
├── scripts/                                 # Python automation scripts
│   └── combine_datasets.py                 # Merge season files
│
└── README.md                                # Project documentation
```

### 📓 Notebook Descriptions

**Core Notebooks:**
- **`exploratory_analysis.ipynb`**: 🔍 Data exploration, visualization, and pattern analysis
- **`data_preprocessing.ipynb`**: 🧹 Data cleaning, feature engineering, and aggregation (creates both processed datasets)

**Algorithm Notebooks (in `algorithms/` folder):**
- **`random_forest/random_forest.ipynb`**: 🌲 Random Forest model with GridSearchCV and rank correction (MAE: 0.20) ✅ Complete
- **Future**: XGBoost, SVM, Decision Tree, K-Means, DBSCAN - each in dedicated folders

**Organization:**
- Each algorithm has its own folder with notebook + README
- README contains performance metrics and model documentation
- Easy comparison between different approaches

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

Or install from requirements file (if available):

```bash
pip install -r requirements.txt
```

### 🔧 Core Dependencies

- **pandas**: 🐼 Data manipulation and analysis
- **numpy**: 🔢 Numerical computing
- **matplotlib**: 📊 Basic plotting and visualization
- **seaborn**: 🎨 Statistical data visualization
- **scikit-learn**: 🤖 Machine learning algorithms and tools
- **jupyter**: 📓 Interactive notebook environment
- **plotly**: 📈 Interactive visualizations (optional)

## 🚀 Usage

Follow these steps to reproduce the analysis and generate predictions:

### 1. Data Loading and Exploration

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

This notebook will:

- 📥 Load the Premier League dataset
- 🔍 Explore data structure and quality
- 📊 Generate visualizations of team performance trends
- 🔎 Identify key patterns in historical data

### 2. Data Preprocessing

```bash
jupyter notebook notebooks/data_preprocessing.ipynb
```

This step includes:

- 🧹 Data cleaning and handling missing values
- ⚙️ Feature engineering (creating predictive variables)
- 🔄 Data transformation and normalization
- ✂️ Train/test split preparation

### 3. Model Training

```bash
jupyter notebook notebooks/model_training.ipynb
```

Train multiple machine learning models:

- 📈 Linear regression for points prediction
- 🌳 Random Forest for classification
- 🚀 Gradient boosting models
- 🏆 Model comparison and selection

### 4. Generate Predictions

```bash
jupyter notebook notebooks/predictions.ipynb
```

Final step to:

- 🎯 Load best performing model
- 🔮 Generate standings predictions
- 📊 Evaluate model performance
- 📈 Visualize predicted vs actual results


### 🛠️ Development Setup

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. ✏️ Make your changes
4. 🧪 Add tests if applicable
5. 💾 Commit your changes (`git commit -m 'Add amazing feature'`)
6. 📤 Push to the branch (`git push origin feature/amazing-feature`)
7. 🔄 Open a Pull Request

---

**⚠️ Note**: This project is for educational and research purposes. The predictions generated should not be used for commercial betting or gambling activities.
