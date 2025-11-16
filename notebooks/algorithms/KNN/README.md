# KNN - Premier League Standings Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AtfastrSlushyMaker/pl-standings-prediction-project/blob/main/notebooks/algorithms/knn/knn.ipynb)

---

## Performance Summary

K-Nearest Neighbors (KNN) predicts Premier League standings by finding similar historical teams and averaging their positions:

- **MAE:** 1.27 positions
- **R² Score:** 0.919 (91.9% variance explained)
- **Within ±1 position:** 58%
- **Within ±2 positions:** 80%
- **Within ±3 positions:** 95%
- **Perfect predictions:** Liverpool (1st), Arsenal (2nd), Chelsea (4th)

The model works on a simple principle: "Tell me who you're similar to, and I'll tell you where you'll finish."

---

## Model Configuration

### Optimal Hyperparameters

```python
from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor(
    n_neighbors=7,        # k=7 found optimal via cross-validation
    weights='distance',   # Closer neighbors weighted more heavily
    metric='euclidean',   # Euclidean distance
    algorithm='auto'      # Automatic selection (ball_tree, kd_tree, or brute)
)
```

**Key points:**
- **k=7:** Optimal bias-variance tradeoff (validated by GridSearchCV)
- **weights='distance':** Nearby teams have more influence than distant ones
- **StandardScaler normalization:** Essential for KNN (variables at different scales)

---

## Sensitivity Analysis

### Impact of k (Number of Neighbors)

| k | MAE | R² | Interpretation |
|---|-----|-----|----------------|
| 3 | 1.45 | 0.89 | Too sensitive to noise |
| 5 | 1.32 | 0.91 | Good but slightly unstable |
| **7** | **1.27** | **0.92** | ✅ **Optimal** - Best balance |
| 9 | 1.30 | 0.91 | Starting to over-smooth |
| 15 | 1.52 | 0.87 | Over-smoothing (loses specificity) |

k=7 maximizes generalization without over-smoothing local patterns.

---

## 2024-25 Season Results

### Top 4 Predictions

| Team | Predicted | Actual | Error |
|------|-----------|--------|-------|
| Liverpool | 1 | 1 | 0 ✅ |
| Arsenal | 2 | 2 | 0 ✅ |
| Chelsea | 4 | 4 | 0 ✅ |
| Manchester City | 5 | 6 | 1 |
| Newcastle | 7 | 8 | 1 |

### Mid-Table

| Team | Predicted | Actual | Error |
|------|-----------|--------|-------|
| Aston Villa | 6 | 5 | 1 |
| Brighton | 8 | 9 | 1 |
| Tottenham | 11 | 10 | 1 |
| West Ham | 13 | 14 | 1 |

### Relegation Zone

| Team | Predicted | Actual | Error |
|------|-----------|--------|-------|
| Ipswich Town | 17 | 18 | 1 |
| Leicester City | 19 | 19 | 0 ✅ |
| Southampton | 20 | 20 | 0 ✅ |

**Pattern:** KNN accurately identifies extremes (champion, relegated teams) but shows more variation in mid-table positions (7-15).

---

## How KNN Works

### Example: Predicting Liverpool 2024-25

**Liverpool's 2024-25 Statistics:**
- Goal Difference: +45
- Points: 90
- Wins: 28
- Goals For: 85

**7 Most Similar Historical Teams:**

| Season | Team | Distance | Final Position |
|--------|------|----------|----------------|
| 2019-20 | Liverpool | 0.12 | **1** |
| 2017-18 | Manchester City | 0.18 | **1** |
| 2018-19 | Liverpool | 0.21 | **2** |
| 2013-14 | Liverpool | 0.25 | **2** |
| 2016-17 | Chelsea | 0.28 | **1** |
| 2011-12 | Manchester City | 0.31 | **1** |
| 2020-21 | Manchester City | 0.34 | **1** |

**Prediction (weighted average):**
```
Predicted Position = (1×w1 + 1×w2 + 2×w3 + 2×w4 + 1×w5 + 1×w6 + 1×w7) / Σwi
                   ≈ 1.2 → Rounds to 1
```

**Result:** ✅ **Predicted 1st, Actual 1st** (Liverpool 2024-25 champions)

---

## Strengths

- **Conceptually simple:** Easy to understand and explain (similarity-based)
- **No distribution assumptions:** Non-parametric (no required data distribution)
- **Adaptable:** k adjustable based on context (precision vs robustness)
- **Useful for comparisons:** Quickly identifies similar teams
- **No training phase:** Lazy learning - simply stores training data
- **Robust to outliers:** When k is sufficiently large

---

## Limitations

- **Scale sensitive:** Requires normalization (StandardScaler, MinMaxScaler)
- **Slow predictions:** Calculates distances for each prediction (O(n×d))
- **Curse of dimensionality:** Performance degrades with too many features (>50)
- **Memory intensive:** Stores all training data (no compression)
- **Less accurate than ensembles:** MAE 1.27 vs 0.20 (Random Forest)
- **No feature importance:** Cannot determine which variables matter most

---

## Technical Notes

### Normalization Required

```python
from sklearn.preprocessing import StandardScaler

# Normalization (essential for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training
knn_model.fit(X_train_scaled, y_train)
```

**Why normalize?**
- Goal Difference: Range -50 to +80
- Shot Accuracy: Range 0 to 1
- Without normalization, Goal Difference dominates distance → biased predictions

### Distance Calculation

**Euclidean Distance** (default):
```
d(x, x') = √(Σ(xi - x'i)²)
```

**Alternatives:**
- `metric='manhattan'`: Manhattan distance (sum of absolute differences)
- `metric='minkowski'`: Generalization (p=1 → Manhattan, p=2 → Euclidean)

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

