# Decision Tree - Premier League Standings Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AtfastrSlushyMaker/pl-standings-prediction-project/blob/main/notebooks/algorithms/decision_tree/decision_tree.ipynb)

---

## Performance Summary

Decision Tree provides interpretable predictions for Premier League standings through transparent decision rules:

- **MAE:** 1.5 to 2.5 positions (depending on tree depth)
- **R² Score:** 0.85 to 0.92
- **Within ±1 position:** 55-65%
- **Within ±2 positions:** 75-85%
- **Key strength:** Highly interpretable with clear if-then rules

The model trades some accuracy for interpretability, making it ideal for explaining predictions to non-technical audiences.

---

## Model Configuration

### Optimal Hyperparameters

```python
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(
    max_depth=10-15,          # Balance between accuracy and interpretability
    min_samples_split=10,     # Minimum samples to split a node
    min_samples_leaf=5,       # Minimum samples per leaf
    criterion='squared_error', # Regression criterion (MSE)
    random_state=42
)
```

**Key Configuration Notes:**
- **max_depth:** 10-15 provides good balance (too deep = overfitting, too shallow = underfitting)
- **min_samples_split/leaf:** Prevents overfitting on small groups
- No normalization required (scale-invariant)

---

## Feature Importance

Top features contributing to splits:

| Rank | Feature | Weight | Description |
|------|---------|--------|-------------|
| 1 | **Goal Difference** | 0.45 | Primary splitting criterion |
| 2 | **Points** | 0.22 | Season total points |
| 3 | **Wins** | 0.15 | Number of victories |
| 4 | **Clean Sheets** | 0.08 | Defensive performance |
| 5 | **Goals For** | 0.05 | Offensive output |

---

## Decision Rules Examples

The tree learns interpretable rules like:

```
IF Goal_Difference > 30 AND Wins > 20
    → Predicted Position: Top 4 (Champions League)

IF Goal_Difference < -10 AND Points < 30
    → Predicted Position: 18-20 (Relegation)

IF Points BETWEEN 40 AND 50 AND Win_Rate > 40%
    → Predicted Position: 7-12 (Upper mid-table)

IF Clean_Sheets > 15 AND Goals_Against < 35
    → Predicted Position: Top 6 (European spots)
```

These rules can be extracted and used for manual analysis or reporting.

---

## Strengths

- **Highly interpretable:** Clear if-then rules anyone can understand
- **Visual representation:** Tree can be plotted and explained
- **Handles non-linearities:** Naturally captures complex interactions
- **No preprocessing needed:** No normalization or encoding required
- **Fast prediction:** Simple traversal down the tree
- **Feature importance:** Shows which variables drive decisions

---

## Limitations

- **Prone to overfitting:** Without pruning, can memorize training data
- **Unstable:** Small data changes can create very different trees
- **Less accurate than ensembles:** MAE 1.5-2.5 vs 0.20 (Random Forest)
- **Step-wise predictions:** Cannot capture smooth transitions
- **Biased to dominant features:** May ignore subtle patterns

---

## Technical Notes

### Tree Depth vs Performance

| max_depth | MAE | R² | Interpretation |
|-----------|-----|-----|----------------|
| 5 | 2.8 | 0.78 | Too simple, underfitting |
| 10 | 2.0 | 0.88 | ✅ Good balance |
| 15 | 1.5 | 0.92 | ✅ Optimal |
| None | 0.5 | 0.98 | Overfitting (memorizes training data) |

Depth 10-15 provides the best generalization.

### Visualization

The tree structure can be visualized to understand decision paths:

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, filled=True, max_depth=3)
plt.show()
```

This shows how the model makes decisions for any given team.

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
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
```

**Tested versions:**
- Python 3.8+
- Scikit-learn 1.0+
- Pandas 1.3+

---

## Usage Example

```python
# Load data
df = pd.read_csv('processed_premier_league_combined.csv')
X = df.drop(['Position', 'Team', 'Season'], axis=1)
y = df['Position']

# Train-test split (temporal)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# Train model
dt_model = DecisionTreeRegressor(
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt_model.fit(X_train, y_train)

# Evaluate
y_pred = dt_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualize tree
plot_tree(dt_model, feature_names=X.columns, filled=True, max_depth=3)
plt.show()
```

---

## Use Cases

### 1. Explaining Predictions to Management
Decision trees excel when you need to justify predictions:
- "Liverpool is predicted 1st because their Goal Difference (+45) is in the top tier and they have >25 wins"
- Clear, defensible reasoning for business decisions

### 2. Identifying Key Performance Thresholds
The tree reveals critical breakpoints:
- "Teams with Goal Difference < -15 almost always finish in relegation"
- "Clean Sheets > 18 strongly indicates Top 6 finish"

### 3. Quick Exploratory Analysis
No preprocessing required - just load and run:
- Fast iteration during initial data exploration
- Immediate insights into feature relationships

---

## Algorithm Comparison

| Algorithm | MAE | R² | Rank |
|-----------|-----|-----|------|
| Random Forest | 0.20 | 0.95 | 🥇 |
| XGBoost | 1.12 | 0.95 | 🥈 |
| SVM | 1.23 | High | 🥉 |
| KNN | 1.27 | 0.92 | 4 |
| **Decision Tree** | **1.5-2.5** | **0.85-0.92** | **5** |
| Gradient Boosting | 1.62 | Good | 6 |

---

## Key Insights

1. **Interpretability has a cost:** Decision Tree is less accurate than ensemble methods
2. **Pruning is essential:** Unrestricted depth leads to overfitting
3. **Goal Difference dominates:** Used in >45% of splitting decisions
4. **Threshold-based logic:** Captures natural breakpoints (e.g., 40 points = safety)
5. **Complements other models:** Use for explanation, ensembles for prediction

---

## Conclusion

Decision Tree provides a transparent, interpretable approach to Premier League standings prediction:

✅ MAE: 1.5-2.5 positions (acceptable accuracy)  
✅ Clear if-then rules anyone can understand  
✅ Visual tree representation  
✅ No preprocessing required  
✅ Fast training and prediction  
✅ Excellent for explaining results to non-technical stakeholders  

This model serves as the go-to option when interpretability and explainability are priorities over raw accuracy.


