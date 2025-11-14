"""
Support Vector Machine (SVR) – Premier League Standings Prediction
================================================================

This script implements a comprehensive modelling pipeline using a
Support Vector Regression (SVR) algorithm to predict the final league
position (1–20) for clubs in the English Premier League.  It is
designed to mirror the structure of the previously implemented
Random Forest model so that the two approaches can be compared in
terms of predictive performance, interpretability and visualisation.

Key features of this script:

* Data loading from a processed, aggregated team‑season dataset (see
  ``data_preprocessing.ipynb`` to generate ``team_season_aggregated.csv``).
* Preparation of the feature matrix and target vector, including
  removal of identifiers and scaling of numeric predictors.
* A season‑based train–test split to avoid information leakage
  (training on all seasons except the most recent, validating on
  2024‑25).
* Hyperparameter tuning via ``GridSearchCV`` across a range of SVR
  parameters (kernel type, ``C``, ``epsilon``, ``gamma``).  Negative
  mean absolute error is used as the optimisation criterion.
* Evaluation metrics on the training and test sets, including MAE,
  RMSE, R², MAPE and explained variance.  Additional summary of the
  error distribution is provided.
* Visualisations of actual vs predicted positions and error
  distributions for both training and test sets, using Matplotlib.
* Category‑level analysis: predictions are binned into position
  ranges (e.g. Top 4, Europa League) to generate a confusion matrix
  and classification report.
* Multi‑class ROC–AUC curves using an SVC classifier trained on
  position categories.  One‑vs‑rest curves and a micro‑average curve
  are drawn.
* Permutation feature importance to identify which input variables
  influence the SVR predictions the most.  Bar and cumulative plots
  summarise the results.
* Rank correction: continuous predictions are converted into unique
  integer ranks within each season to ensure a valid league table.

This file is intended to be run in a Jupyter environment (e.g. via
``%run svm_model.py``) or as a standalone script.  It will print
progress to the console and produce several plots.  Modify the
``param_grid`` or feature list as needed for experimentation.

Note: this code assumes the existence of the aggregated CSV file
``team_season_aggregated.csv`` in one of the parent ``data/processed``
directories.  If the file cannot be found, a ``FileNotFoundError``
will be raised with instructions to run the data preprocessing step.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVR, SVC


def load_aggregated_data() -> pd.DataFrame:
    """Load the aggregated team‑season dataset.

    Searches a few candidate locations for the ``team_season_aggregated.csv``
    file.  Raises FileNotFoundError if not present.

    Returns
    -------
    df : pandas.DataFrame
        The loaded dataset.
    """
    candidate_paths = [
        Path("data/processed/team_season_aggregated.csv"),
        Path("../data/processed/team_season_aggregated.csv"),
        Path("../../data/processed/team_season_aggregated.csv"),
    ]
    agg_path = next((p for p in candidate_paths if p.exists()), None)
    if agg_path is None:
        raise FileNotFoundError(
            "team_season_aggregated.csv not found! "
            "Please run data_preprocessing.ipynb first to create this file."
        )
    print(f"✅ Loading aggregated dataset: {agg_path}")
    df = pd.read_csv(agg_path)
    print(f"Dataset shape: {df.shape}")
    return df


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Select predictor columns and target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        The aggregated dataset containing team and seasonal statistics.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target vector (final league position).
    feature_names : list[str]
        Names of the predictor variables retained.
    """
    # Candidate feature columns (exclude identifiers, target and non‑predictive fields)
    feature_cols = [
        'Team_encoded', 'Season_encoded',
        'Wins', 'Draws', 'Losses',
        'Goals_Scored', 'Goals_Conceded', 'Goal_Difference',
        'Avg_Goals_Scored', 'Avg_Goals_Conceded',
        'Total_Shots', 'Total_Shots_On_Target', 'Avg_Shots', 'Avg_Shots_On_Target',
        'Shot_Accuracy', 'Clean_Sheets', 'Clean_Sheet_Rate',
        'Yellow_Cards', 'Red_Cards', 'Fouls', 'Corners',
        'Win_Rate', 'Home_Win_Rate', 'Away_Win_Rate', 'Points_Per_Game',
    ]
    available_features = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"⚠️ Missing feature columns: {missing}")
    X = df[available_features].copy()
    y = df['Final_Position'].copy()
    return X, y, available_features


def season_train_test_split(X: pd.DataFrame, y: pd.Series, seasons: pd.Series) -> tuple:
    """Split features and target by season, training on all but the latest season.

    This mirrors the random forest script: the most recent season (2024‑25)
    acts as a hold‑out test set.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target vector.
    seasons : pandas.Series
        Season labels corresponding to each row.

    Returns
    -------
    X_train, X_test, y_train, y_test, train_seasons, test_seasons
    """
    seasons_sorted = sorted(seasons.unique())
    # Choose the last season as test
    test_seasons = [seasons_sorted[-1]]
    train_seasons = [s for s in seasons_sorted if s not in test_seasons]
    train_mask = seasons.isin(train_seasons)
    test_mask = seasons.isin(test_seasons)
    X_train = X[train_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)
    print("\n📊 Train–Test Split:")
    print(f"  Training seasons ({len(train_seasons)}): {train_seasons[0]} → {train_seasons[-1]}")
    print(f"  Test season: {test_seasons[0]}")
    print(f"  Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, train_seasons, test_seasons


def tune_and_train_svr(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """Perform hyperparameter tuning and train an SVR model.

    Uses ``StandardScaler`` to normalise features and tunes parameters via
    5‑fold cross‑validated grid search, optimising negative mean
    absolute error.

    Returns
    -------
    best_model : sklearn estimator
        The best SVR model found.
    grid_search : GridSearchCV
        The fitted grid search object, useful for inspecting CV results.
    scaler : StandardScaler
        Fitted scaler used for transforming data.
    """
    # Build pipeline: scale then SVR
    svr = SVR()
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', svr),
    ])
    # Hyperparameter grid
    param_grid = {
        'svr__kernel': ['rbf', 'linear', 'poly'],
        'svr__C': [0.1, 1, 10, 100],
        'svr__epsilon': [0.1, 0.2, 0.5, 1.0],
        'svr__gamma': ['scale', 'auto'],
        # Polynomial degree only used if kernel='poly'
        'svr__degree': [2, 3],
    }
    # Compute number of combinations (approximate)
    n_combinations = (
        len(param_grid['svr__kernel']) *
        len(param_grid['svr__C']) *
        len(param_grid['svr__epsilon']) *
        len(param_grid['svr__gamma']) *
        len(param_grid['svr__degree'])
    )
    print("\n🔧 Hyperparameter tuning for SVR")
    print(f"  Parameter combinations: {n_combinations}")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=5,
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    duration = time.time() - start_time
    print(f"✅ Grid search completed in {duration/60:.2f} minutes")
    best_model = grid_search.best_estimator_
    print("\n🏆 Best hyperparameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"  Best CV MAE: {-grid_search.best_score_:.4f} positions")
    return best_model, grid_search


def evaluate_model(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """Compute evaluation metrics for training and test sets."""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    results = {}
    for split, y_true, y_pred in [('train', y_train, y_pred_train), ('test', y_test, y_pred_test)]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        evs = explained_variance_score(y_true, y_pred)
        results[f'{split}_mae'] = mae
        results[f'{split}_rmse'] = rmse
        results[f'{split}_r2'] = r2
        results[f'{split}_mape'] = mape
        results[f'{split}_evs'] = evs
    return results, y_pred_train, y_pred_test


def print_evaluation_summary(results: dict):
    """Nicely format evaluation metrics to the console."""
    print("\n📊 Model Performance Summary")
    print("Training set:")
    print(f"  MAE:  {results['train_mae']:.3f} positions")
    print(f"  RMSE: {results['train_rmse']:.3f} positions")
    print(f"  R²:   {results['train_r2']:.4f}")
    print(f"  MAPE: {results['train_mape']:.2f}%")
    print(f"  EVS:  {results['train_evs']:.4f}")
    print("Test set:")
    print(f"  MAE:  {results['test_mae']:.3f} positions")
    print(f"  RMSE: {results['test_rmse']:.3f} positions")
    print(f"  R²:   {results['test_r2']:.4f}")
    print(f"  MAPE: {results['test_mape']:.2f}%")
    print(f"  EVS:  {results['test_evs']:.4f}")
    # Simple overfitting check
    gap = results['train_mae'] - results['test_mae']
    if abs(gap) < 0.5:
        status = "✅ No overfitting (train/test MAE gap < 0.5)"
    elif gap > 0:
        status = f"⚠️ Slight overfitting (train MAE lower by {gap:.2f})"
    else:
        status = f"ℹ️ Better generalisation on test (test MAE lower by {-gap:.2f})"
    print(f"\nOverfitting check: {status}")


def plot_predictions_and_errors(y_train: pd.Series, y_pred_train: np.ndarray,
                               y_test: pd.Series, y_pred_test: np.ndarray) -> None:
    """Create scatter plots and histograms to visualise predictions and errors."""
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    # Training scatter
    axes[0, 0].scatter(y_train, y_pred_train, alpha=0.6, s=80, edgecolors='black', linewidths=0.5)
    axes[0, 0].plot([1, 20], [1, 20], 'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Final Position')
    axes[0, 0].set_ylabel('Predicted Final Position')
    axes[0, 0].set_title('Training Set: Actual vs Predicted')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 21)
    axes[0, 0].set_ylim(0, 21)
    axes[0, 0].invert_xaxis()
    axes[0, 0].invert_yaxis()
    # Test scatter
    axes[0, 1].scatter(y_test, y_pred_test, alpha=0.7, s=80, edgecolors='black', linewidths=0.5, color='orange')
    axes[0, 1].plot([1, 20], [1, 20], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Final Position')
    axes[0, 1].set_ylabel('Predicted Final Position')
    axes[0, 1].set_title('Test Set: Actual vs Predicted')
    axes[0, 1].legend(['Perfect Prediction'], loc='upper left')
    axes[0, 1].set_xlim(0, 21)
    axes[0, 1].set_ylim(0, 21)
    axes[0, 1].invert_xaxis()
    axes[0, 1].invert_yaxis()
    # Error distribution (training)
    errors_train = y_train - y_pred_train
    axes[1, 0].hist(errors_train, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1, 0].axvline(errors_train.mean(), color='green', linestyle='-', linewidth=2,
                       label=f'Mean = {errors_train.mean():.2f}')
    axes[1, 0].set_xlabel('Prediction Error (Actual - Predicted)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Training Error Distribution\nStd Dev = {errors_train.std():.2f}')
    axes[1, 0].legend()
    # Error distribution (test)
    errors_test = y_test - y_pred_test
    axes[1, 1].hist(errors_test, bins=15, color='salmon', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].axvline(errors_test.mean(), color='green', linestyle='-', linewidth=2,
                       label=f'Mean = {errors_test.mean():.2f}')
    axes[1, 1].set_xlabel('Prediction Error (Actual - Predicted)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Test Error Distribution\nStd Dev = {errors_test.std():.2f}')
    axes[1, 1].legend(['Zero Error', 'Mean Error'], loc='upper left')
    plt.tight_layout()
    plt.show()


def categorize_positions(positions: np.ndarray) -> np.ndarray:
    """Map continuous positions to categorical labels."""
    labels = []
    for pos in positions:
        if pos <= 4:
            labels.append('Top 4')
        elif pos <= 6:
            labels.append('Europa')
        elif pos <= 10:
            labels.append('Upper Mid')
        elif pos <= 14:
            labels.append('Lower Mid')
        elif pos <= 17:
            labels.append('Rel. Battle')
        else:
            labels.append('Relegated')
    return np.array(labels)


def position_category_analysis(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """Perform confusion matrix and classification report on position categories."""
    # Convert to integer positions
    y_true_cat = categorize_positions(y_test.values)
    y_pred_cat = categorize_positions(y_pred)
    labels = ['Top 4', 'Europa', 'Upper Mid', 'Lower Mid', 'Rel. Battle', 'Relegated']
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=labels, yticklabels=labels, linewidths=0.5, linecolor='grey')
    plt.xlabel('Predicted Category')
    plt.ylabel('Actual Category')
    plt.title('Confusion Matrix – Position Categories (Test)')
    plt.tight_layout()
    plt.show()
    print("\n📋 Classification Report – Position Categories")
    print(classification_report(y_true_cat, y_pred_cat, target_names=labels, zero_division=0))
    accuracy = (y_true_cat == y_pred_cat).mean() * 100
    print(f"Overall category accuracy: {accuracy:.1f}%")


def roc_auc_analysis(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series,
                    grid_search: GridSearchCV) -> None:
    """Plot ROC curves for position categories using an SVC classifier."""
    print("\n📈 ROC–AUC analysis for position categories")
    # Convert positions to codes 0–5
    def position_to_code(pos):
        if pos <= 4:
            return 0
        elif pos <= 6:
            return 1
        elif pos <= 10:
            return 2
        elif pos <= 14:
            return 3
        elif pos <= 17:
            return 4
        return 5
    y_train_codes = np.array([position_to_code(p) for p in y_train])
    y_test_codes = np.array([position_to_code(p) for p in y_test])
    # Use the same scaler as in the best SVR model
    scaler = grid_search.best_estimator_.named_steps['scaler']
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Train an SVC for probability estimates
    svc_params = {
        'C': grid_search.best_params_.get('svr__C', 10),
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True,
        'random_state': 42,
    }
    svc = SVC(**svc_params)
    svc.fit(X_train_scaled, y_train_codes)
    y_proba = svc.predict_proba(X_test_scaled)
    # Binarise labels
    n_classes = 6
    y_test_bin = label_binarize(y_test_codes, classes=list(range(n_classes)))
    fpr = {}
    tpr = {}
    roc_auc_vals = {}
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    category_names = ['Top 4', 'Europa', 'Upper Mid', 'Lower Mid', 'Rel. Battle', 'Relegated']
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc_vals[i] = auc(fpr[i], tpr[i])
    # Micro‑average
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_auc_vals['micro'] = auc(fpr['micro'], tpr['micro'])
    plt.figure(figsize=(12, 9))
    for i, color, name in zip(range(n_classes), colors, category_names):
        plt.plot(fpr[i], tpr[i], color=color, lw=2.5,
                 label=f'{name} (AUC = {roc_auc_vals[i]:.3f})')
    plt.plot(fpr['micro'], tpr['micro'], color='navy', linestyle='--', lw=2.5,
             label=f'Micro‑average (AUC = {roc_auc_vals["micro"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves – Multi‑class Position Categories (One‑vs‑Rest)')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    # Print AUC scores
    print("\n📊 AUC scores by category:")
    for i, name in enumerate(category_names):
        print(f"  {name:15s}: {roc_auc_vals[i]:.4f}")
    print(f"  Micro‑average    : {roc_auc_vals['micro']:.4f}")
    try:
        ovr_auc = roc_auc_score(y_test_codes, y_proba, multi_class='ovr', average='weighted')
        ovo_auc = roc_auc_score(y_test_codes, y_proba, multi_class='ovo', average='weighted')
        print(f"  Weighted OVR AUC : {ovr_auc:.4f}")
        print(f"  Weighted OVO AUC : {ovo_auc:.4f}")
    except Exception:
        pass


def compute_permutation_importance(model, X_train: pd.DataFrame, y_train: pd.Series,
                                   feature_names: list[str]) -> pd.DataFrame:
    """Compute permutation importance for the SVR model.

    Returns a DataFrame sorted by decreasing importance.
    """
    # Extract estimator from pipeline
    svr = model.named_steps['svr']
    scaler = model.named_steps['scaler']
    X_scaled = scaler.transform(X_train)
    result = permutation_importance(
        svr,
        X_scaled,
        y_train,
        n_repeats=10,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1,
    )
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
    }).sort_values('Importance', ascending=False)
    return importances


def plot_feature_importance(importances: pd.DataFrame, top_n: int = 15) -> None:
    """Plot bar and cumulative charts for feature importance."""
    top_features = importances.head(top_n)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # Bar plot
    axes[0].barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['Feature'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Importance')
    axes[0].set_title(f'Top {top_n} Permutation Importances')
    # Cumulative plot
    cumsum = importances['Importance'].cumsum()
    axes[1].plot(range(1, len(cumsum) + 1), cumsum, linewidth=2.5, color='darkgreen')
    axes[1].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% threshold')
    axes[1].axhline(y=0.99, color='orange', linestyle='--', linewidth=2, label='99% threshold')
    axes[1].set_xlabel('Number of Features')
    axes[1].set_ylabel('Cumulative Importance')
    axes[1].set_title('Cumulative Permutation Importance')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def rank_correct_predictions(y_pred: np.ndarray, seasons: pd.Series) -> np.ndarray:
    """Correct predicted continuous positions to unique integer ranks per season."""
    corrected = np.zeros_like(y_pred, dtype=int)
    unique_seasons = seasons.unique()
    for season in unique_seasons:
        mask = seasons == season
        preds = y_pred[mask]
        # Rank them: lowest prediction => rank 1
        ranks = preds.argsort().argsort() + 1
        corrected[mask] = ranks
    return corrected


def main() -> None:
    # Load and prepare data
    df = load_aggregated_data()
    X, y, feature_names = prepare_features_and_target(df)
    # Split by season
    X_train, X_test, y_train, y_test, train_seasons, test_seasons = season_train_test_split(
        X, y, df['Season']
    )
    # Hyperparameter tuning and training
    best_model, grid_search = tune_and_train_svr(X_train, y_train)
    # Evaluate
    results, y_pred_train, y_pred_test = evaluate_model(best_model, X_train, y_train, X_test, y_test)
    print_evaluation_summary(results)
    # Visualise predictions and errors
    plot_predictions_and_errors(y_train, y_pred_train, y_test, y_pred_test)
    # Position category analysis
    position_category_analysis(y_test, y_pred_test)
    # ROC–AUC curves
    roc_auc_analysis(X_train, y_train, X_test, y_test, grid_search)
    # Permutation importance
    importances = compute_permutation_importance(best_model, X_train, y_train, feature_names)
    print("\n🎯 Top 15 most important features (permutation importance):")
    for _, row in importances.head(15).iterrows():
        print(f"  {row['Feature']:<30s}: {row['Importance']:.6f}")
    plot_feature_importance(importances, top_n=15)
    # Rank correction demonstration
    y_pred_corr = rank_correct_predictions(y_pred_test, df.loc[df['Season'].isin(test_seasons), 'Season'])
    mae_corr = mean_absolute_error(y_test, y_pred_corr)
    within_1 = (np.abs(y_test.values - y_pred_corr) <= 1).sum()
    within_2 = (np.abs(y_test.values - y_pred_corr) <= 2).sum()
    print("\n🔧 Rank-corrected predictions:")
    print(f"  MAE after rank correction: {mae_corr:.3f} positions")
    print(f"  Predictions within ±1 position: {within_1}/{len(y_test)}")
    print(f"  Predictions within ±2 positions: {within_2}/{len(y_test)}")


if __name__ == '__main__':
    # Running as a script
    main()