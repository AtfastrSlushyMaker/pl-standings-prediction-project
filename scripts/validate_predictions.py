"""
Validate 2025-26 Season Predictions Against Actual Results
Fetches real match results from football-data.co.uk and compares with model predictions
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime

def fetch_current_season_results():
    """
    Fetch 2025-26 Premier League results from football-data.co.uk
    """
    print("="*80)
    print("FETCHING CURRENT SEASON RESULTS (2025-26)")
    print("="*80)
    
    # URL for current season data from football-data.co.uk
    url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
    
    try:
        print(f"\nFetching data from: {url}")
        df = pd.read_csv(url)
        print(f"✓ Downloaded {len(df)} matches")
        
        # Filter only completed matches (those with results)
        df_completed = df[df['FTR'].notna()].copy()
        print(f"✓ Found {len(df_completed)} completed matches")
        
        return df_completed
    except Exception as e:
        print(f"✗ Error fetching data: {str(e)}")
        return None

def map_team_names(actual_name):
    """
    Map football-data.co.uk team names to our model's team names
    """
    mapping = {
        'Man United': 'Man United',
        'Man City': 'Man City',
        'Nott\'m Forest': 'Nott\'m Forest',
        'Newcastle': 'Newcastle',
        'Tottenham': 'Tottenham',
        'Arsenal': 'Arsenal',
        'Liverpool': 'Liverpool',
        'Chelsea': 'Chelsea',
        'Brighton': 'Brighton',
        'Aston Villa': 'Aston Villa',
        'Bournemouth': 'Bournemouth',
        'Fulham': 'Fulham',
        'West Ham': 'West Ham',
        'Brentford': 'Brentford',
        'Crystal Palace': 'Crystal Palace',
        'Everton': 'Everton',
        'Leicester': 'Leicester',
        'Ipswich': 'Ipswich',
        'Southampton': 'Southampton',
        'Wolves': 'Wolves'
    }
    return mapping.get(actual_name, actual_name)

def compare_predictions():
    """
    Compare model predictions with actual results
    """
    print("\n" + "="*80)
    print("LOADING PREDICTIONS")
    print("="*80)
    
    # Load predictions
    predictions_path = Path(__file__).parent.parent / 'notebooks' / 'outputs' / 'BO2_match_outcomes' / '2025-26_match_predictions.csv'
    predictions = pd.read_csv(predictions_path)
    print(f"✓ Loaded {len(predictions)} predicted matches")
    
    # Fetch actual results
    actual_results = fetch_current_season_results()
    
    if actual_results is None:
        print("\n✗ Could not fetch actual results")
        return
    
    print("\n" + "="*80)
    print("COMPARING PREDICTIONS WITH ACTUAL RESULTS")
    print("="*80)
    
    # Map team names in actual results
    actual_results['HomeTeam'] = actual_results['HomeTeam'].apply(map_team_names)
    actual_results['AwayTeam'] = actual_results['AwayTeam'].apply(map_team_names)
    
    # Map actual FTR to predicted outcome format
    actual_results['Actual_Outcome'] = actual_results['FTR'].map({
        'H': 'Home Win',
        'D': 'Draw',
        'A': 'Away Win'
    })
    
    # Merge predictions with actual results
    merged = predictions.merge(
        actual_results[['HomeTeam', 'AwayTeam', 'Actual_Outcome', 'FTHG', 'FTAG']],
        on=['HomeTeam', 'AwayTeam'],
        how='inner'
    )
    
    print(f"\n✓ Found {len(merged)} matches that have been played")
    
    if len(merged) == 0:
        print("\n⚠ No matching matches found between predictions and actual results")
        print("   This might be because:")
        print("   - The 2025-26 season hasn't started yet")
        print("   - Team names don't match between datasets")
        return
    
    # Calculate accuracy
    merged['Correct'] = merged['Predicted_Outcome'] == merged['Actual_Outcome']
    
    accuracy = (merged['Correct'].sum() / len(merged)) * 100
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nTotal Matches Played: {len(merged)}")
    print(f"Correct Predictions: {merged['Correct'].sum()}")
    print(f"Incorrect Predictions: {(~merged['Correct']).sum()}")
    print(f"\n🎯 Overall Accuracy: {accuracy:.2f}%")
    
    # Breakdown by outcome type
    print("\n" + "-"*80)
    print("ACCURACY BY OUTCOME TYPE")
    print("-"*80)
    
    for outcome in ['Home Win', 'Draw', 'Away Win']:
        subset = merged[merged['Predicted_Outcome'] == outcome]
        if len(subset) > 0:
            outcome_accuracy = (subset['Correct'].sum() / len(subset)) * 100
            print(f"{outcome:12s}: {outcome_accuracy:5.2f}% ({subset['Correct'].sum()}/{len(subset)} correct)")
    
    # Show some examples
    print("\n" + "-"*80)
    print("SAMPLE PREDICTIONS (First 10 matches)")
    print("-"*80)
    
    display_cols = ['HomeTeam', 'AwayTeam', 'Predicted_Outcome', 'Actual_Outcome', 'FTHG', 'FTAG', 'Correct']
    print(merged[display_cols].head(10).to_string(index=False))
    
    # Show worst predictions (highest confidence but wrong)
    print("\n" + "-"*80)
    print("BIGGEST MISSES (High Confidence but Wrong)")
    print("-"*80)
    
    wrong_predictions = merged[~merged['Correct']].copy()
    if len(wrong_predictions) > 0:
        wrong_predictions = wrong_predictions.sort_values('Confidence', ascending=False)
        display_cols_wrong = ['HomeTeam', 'AwayTeam', 'Predicted_Outcome', 'Actual_Outcome', 'Confidence', 'FTHG', 'FTAG']
        print(wrong_predictions[display_cols_wrong].head(10).to_string(index=False))
    
    # Save comparison results
    output_path = Path(__file__).parent.parent / 'notebooks' / 'outputs' / 'BO2_match_outcomes' / 'prediction_validation.csv'
    merged.to_csv(output_path, index=False)
    print(f"\n✓ Detailed comparison saved to: {output_path}")

if __name__ == "__main__":
    compare_predictions()
