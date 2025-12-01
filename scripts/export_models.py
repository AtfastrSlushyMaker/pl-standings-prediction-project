"""
Export trained models for backend API deployment.

This script provides instructions to export models from notebooks.
The actual export happens in the notebooks themselves.
"""

import pickle
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

# Create models directory
MODELS_DIR.mkdir(exist_ok=True)

print("="*80)
print("MODEL EXPORT GUIDE")
print("="*80)
print(f"\nModels will be exported to: {MODELS_DIR}")
print("\nTo export models, run the export cells in each notebook:")
print("1. Open BO2_match_winner_comparison.ipynb")
print("2. Run all cells including the new 'Model Export' cells at the end")
print("3. Repeat for BO3_team_segmentation.ipynb")
print("4. Repeat for BO4_players_reccomendation.ipynb")
print("\n" + "="*80)
print("\n" + "="*80)

if __name__ == '__main__':
    print("\nIMPORTANT: Export cells have been added to your notebooks!")
    print("\nInstructions:")
    print("1. Open each notebook (BO2, BO3, BO4)")
    print("2. Run all cells from top to bottom")
    print("3. The export cells at the end will create .pkl files in models/")
    print("\nAfter running all notebooks, your models/ folder will contain:")
    print("  - bo2_match_prediction.pkl")
    print("  - bo3_kmeans_clustering.pkl")
    print("  - bo3_gmm_clustering.pkl")
    print("  - bo4_defender_lgb.pkl")
    print("  - bo4_midfielder_lgb.pkl")
    print("  - bo4_forward_lgb.pkl")
    print("  - bo4_goalkeeper_lgb.pkl")
    print("  - teams.json")
    print("  - team_encoding.json")
    print("\n" + "="*80)
