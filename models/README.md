# Exported Models

Trained ML models exported from notebooks, ready for backend deployment.

## Contents

**Models** (9 files):
- `bo1_season_ranking.pkl` - KNN model for season rankings
- `bo2_match_prediction.pkl` - Random Forest model for match outcomes
- `bo3_kmeans_clustering.pkl` - KMeans model for team clustering
- `bo4_defender_lgb.pkl` - LightGBM model for defender recommendations
- `bo4_midfielder_lgb.pkl` - LightGBM model for midfielder recommendations
- `bo4_forward_lgb.pkl` - LightGBM model for forward recommendations
- `bo4_goalkeeper_lgb.pkl` - LightGBM model for goalkeeper recommendations
- `teams.json` - Team reference data
- `team_encoding.json` - Team encoding mappings

## Format

Each `.pkl` file contains: `{'model': <model>, 'scaler': <scaler>, 'features': [...], 'metadata': {...}}`

Each `.json` file contains reference data for the backend API.


