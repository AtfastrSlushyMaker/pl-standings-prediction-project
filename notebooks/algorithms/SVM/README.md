## \# 🧠 SVM -- Premier League Relegation Risk Prediction

## Performance Summary

The SVM models (Regression + Classification) were designed to predict
**Premier League relegation risks** and estimate **final league
positions** using historical team-season statistics.

### **Results Overview**

-   **SVR (Regression) MAE:** \~**1.23** positions\
-   **Relegation-zone MAE:** \~**3.26** positions\
-   **Three-level accuracy (Top-4 / Mid-Table / Relegation):** Up to
    **100%** for relegated teams\
-   **SVC Classification AUC:** **1.000** (perfect discrimination)\
-   **Final Classification Accuracy / F1 / Precision / Recall:**
    **100%**

The combined regression + classification strategy provides both **exact
ranking predictions** and **relegation probability scoring**.

------------------------------------------------------------------------

## Best Model Configuration

### ✔ SVR (Regression)

-   **Kernel:** RBF (non-linear)\
-   **C:** 10\
-   **Gamma:** 'scale'\
-   **Epsilon:** optimized via validation\
-   **Scaling:** StandardScaler applied to all features

### ✔ SVC (Classification)

-   **Kernel:** RBF\
-   **C:** 1\
-   **Probability calibration:** Enabled\
-   **Threshold optimization:** Based on **max F1-score**\
-   **Optimal threshold:** **0.944**

------------------------------------------------------------------------

## Performance Metrics

### 🧮 **SVR -- Final League Position Prediction**

  Metric                    Value
  ------------------------- ------------------------------
  **MAE**                   \~1.23
  **Relegation-Zone MAE**   \~3.26
  **Category Accuracy**     **100%** for relegated teams

------------------------------------------------------------------------

### 🟥 **SVC -- Direct Relegation Classification**

  Metric                  Score
  ----------------------- -----------
  **Accuracy**            100%
  **F1-score**            100%
  **Precision**           100%
  **Recall**              100%
  **ROC AUC**             **1.000**
  **Optimal Threshold**   **0.944**

------------------------------------------------------------------------

## Risk Prediction Output

Teams are assigned a relegation risk level based on SVC probabilities:

  Probability Range   Risk Level
  ------------------- ----------------
  **\> 0.90**         🔴 High Risk
  **0.40--0.90**      🟠 Medium Risk
  **\< 0.40**         🟢 Low Risk

------------------------------------------------------------------------

## Feature Importance

Permutation importance revealed:

1.  **Goals Against (GA)**\
2.  **Goals For (GF)**\
3.  **Goal Difference (GD)**\
4.  **Points and Win Rate**

------------------------------------------------------------------------

## Dataset Information

### Training Data

-   **Seasons:** 2000--2023\
-   **Records:** \~480\
-   **Features:** 25 performance indicators

### Validation Data

-   **Season:** 2024--25


------------------------------------------------------------------------

## Conclusion

The SVM-based system provides:

-   **Reliable relegation risk predictions**\
-   **Accurate regression of final standings**\
-   **Perfect classification of relegated teams**\
-   **Clear interpretability through feature impact**

------------------------------------------------------------------------
