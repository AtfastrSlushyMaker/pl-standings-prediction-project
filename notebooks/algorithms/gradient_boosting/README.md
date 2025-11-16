# Gradient Boosting (LightGBM) - Premier League Standings Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AtfastrSlushyMaker/pl-standings-prediction-project/blob/main/notebooks/algorithms/gradient_boosting/gradient_boosting.ipynb)

---

## Performance Summary

LightGBM uses gradient boosting to sequentially correct prediction errors:

- **MAE:** 1.62 positions
- **RMSE:** 2.01 positions
- **Within ±1 position:** 58%
- **Within ±2 positions:** 72%
- **Perfect predictions:** 38% (7-8/20 teams)
- **Training speed:** Fast with LightGBM optimizations

The model builds trees sequentially, with each new tree correcting residual errors from previous trees.

---

## Model Configuration

### Optimal Hyperparameters

```python
import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(
    learning_rate=0.05,       # Moderate learning rate
    num_leaves=31,            # Leaf-wise tree growth
    max_depth=-1,             # No depth limit (controlled by num_leaves)
    n_estimators=500,         # With early stopping
    boosting_type='gbdt',     # Gradient Boosting Decision Tree
    random_state=42
)
```

**Key Configuration Notes:**
- **learning_rate:** 0.05 balances training speed and accuracy
- **num_leaves:** 31 allows complex patterns without overfitting
- **Early stopping:** Automatically finds best iteration via validation set
- **Leaf-wise growth:** Faster than XGBoost's level-wise approach

---

## Training Mechanism

### Sequential Error Correction

```
Iteration 1: Predict average position (baseline)
            → Calculate residuals (errors)

Iteration 2: Train tree to predict residuals from iteration 1
            → Update predictions
            → Calculate new residuals

Iteration 3: Train tree to predict residuals from iteration 2
            → Update predictions
            → ...

Iteration N: Early stopping when validation error stops improving (50 rounds)
            → Use best iteration for final model
```

This approach gradually reduces prediction errors through focused corrections.

---

## Feature Importance (SHAP Analysis)

Impact on predictions (absolute mean SHAP values):

| Rank | Feature | Impact | Description |
|------|---------|--------|-------------|
| 1 | **Goal Difference** | 3.5 positions | Strongest predictor |
| 2 | **Points** | 2.1 positions | Season performance |
| 3 | **Wins** | 1.8 positions | Victory count |
| 4 | **Goals For** | 1.2 positions | Offensive output |
| 5 | **Clean Sheets** | 0.9 positions | Defensive stability |

SHAP values show how each feature pushes predictions higher or lower for individual teams.

---

## Performance Analysis

### Error Distribution

| Error Range | Teams | Percentage |
|-------------|-------|------------|
| Perfect (0) | 7-8 | 38% |
| ±1 position | 4-5 | 20% |
| ±2 positions | 3-4 | 14% |
| >±2 positions | 5-6 | 28% |

The model performs well on top and bottom positions but shows more variation in mid-table.

### Performance by Position Range

- **Top 4:** Good identification (MAE ~1.2)
- **Positions 5-10:** Moderate (MAE ~1.8)
- **Positions 11-17:** More challenging (MAE ~2.0)
- **Relegation (18-20):** Good detection (MAE ~1.5)

---

## Strengths

- **Sequential error correction:** Each tree improves on previous mistakes
- **Fast training:** LightGBM's leaf-wise growth is optimized
- **Automatic early stopping:** Finds optimal iteration count
- **Good speed/accuracy balance:** Faster than XGBoost, more accurate than single tree
- **SHAP integration:** Built-in explainability via SHAP values
- **Handles large datasets:** Efficient memory usage

---

## Limitations

- **Requires careful tuning:** learning_rate and num_leaves need balancing
- **Can overfit:** Without proper early stopping or regularization
- **Less accurate than top models:** MAE 1.62 vs 0.20 (Random Forest)
- **Mid-table struggles:** Higher errors in positions 11-17
- **Hyperparameter sensitive:** Small changes can significantly impact performance

---

## Technical Notes

### Early Stopping

```python
# Training with validation set for early stopping
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50)]  # Stop if no improvement for 50 rounds
)

# Best iteration automatically selected
print(f"Best iteration: {lgb_model.best_iteration_}")
```

### LightGBM vs XGBoost

| Feature | LightGBM | XGBoost |
|---------|----------|---------|
| **Tree growth** | Leaf-wise | Level-wise |
| **Training speed** | Faster | Slower |
| **Memory usage** | Lower | Higher |
| **Accuracy** | Similar | Similar |
| **Default parameters** | More aggressive | More conservative |

LightGBM is preferred for large datasets and rapid prototyping.

---

## Dataset Information

### Training Data
- **Seasons:** 2000-01 through 2023-24
- **Samples:** ~480 team-season records
- **Features:** 25 performance metrics
- **Source:** `data/processed/processed_premier_league_combined.csv`

### Test Data
- **Season:** 2024-25
- **Samples:** 20 teams
- **Purpose:** Final validation

---

## Dependencies

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
```

**Installation:**
```python
!pip install lightgbm scikit-learn shap matplotlib seaborn -q
```

**Tested versions:**
- Python 3.8+
- LightGBM 3.0+
- Scikit-learn 1.0+
- SHAP 0.40+

---

## Usage Example

```python
# Load data
df = pd.read_csv('processed_premier_league_combined.csv')
X = df.drop(['Position', 'Team', 'Season'], axis=1)
y = df['Position']

# Train-validation-test split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, shuffle=False
)

# Train model with early stopping
lgb_model = lgb.LGBMRegressor(
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=500,
    random_state=42
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

# Evaluate
y_pred = lgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# SHAP analysis
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

---

## Use Cases

### 1. Automated Prediction Pipelines
LightGBM's speed makes it ideal for:
- Daily or weekly prediction updates
- Real-time dashboards
- Batch processing of multiple scenarios

### 2. Mid-Season Updates
Sequential learning allows:
- Incremental updates as season progresses
- Quick retraining with new match data
- Adaptive predictions

### 3. Feature Engineering Exploration
Fast training enables:
- Testing many feature combinations
- Rapid prototyping of new variables
- A/B testing different data sources

---

## Algorithm Comparison

| Algorithm | MAE | R² | Rank |
|-----------|-----|-----|------|
| Random Forest | 0.20 | 0.95 | 🥇 |
| XGBoost | 1.12 | 0.95 | 🥈 |
| SVM | 1.23 | High | 🥉 |
| KNN | 1.27 | 0.92 | 4 |
| Decision Tree | 1.5-2.5 | 0.85-0.92 | 5 |
| **Gradient Boosting** | **1.62** | **Good** | **6** |

---

## Key Insights

1. **Sequential boosting works:** Each tree improves on previous errors
2. **Early stopping is critical:** Prevents overfitting, finds optimal iteration
3. **Speed matters:** LightGBM's efficiency enables rapid experimentation
4. **Mid-table is hard:** All models struggle with positions 11-17 (high variance)
5. **SHAP adds value:** Explains individual predictions beyond global feature importance

---

## Conclusion

LightGBM provides a fast, efficient approach to Premier League standings prediction:

✅ MAE: 1.62 positions (acceptable accuracy)  
✅ Fast training with leaf-wise growth  
✅ Automatic early stopping  
✅ Good speed/accuracy balance  
✅ Built-in SHAP explainability  
✅ Ideal for automated pipelines and rapid prototyping  

This model serves as a solid option when training speed and regular updates are priorities.


