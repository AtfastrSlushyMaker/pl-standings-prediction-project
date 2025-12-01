"""
Preprocessing utilities for backend API.

These functions should match EXACTLY the preprocessing done in notebooks
to ensure consistent predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ============================================================================
# BO1 - Season Ranking Preprocessing
# ============================================================================

def preprocess_season_data(team_stats: Dict) -> pd.DataFrame:
    """
    Preprocess team season statistics for BO1 prediction.
    
    Args:
        team_stats: Dictionary with keys like:
            - team: str
            - wins: int
            - draws: int
            - losses: int
            - goals_scored: int
            - goals_conceded: int
    
    Returns:
        DataFrame with engineered features
    """
    df = pd.DataFrame([team_stats])
    
    # Calculate derived features (match notebook preprocessing)
    df['matches_played'] = df['wins'] + df['draws'] + df['losses']
    df['points'] = df['wins'] * 3 + df['draws']
    df['win_rate'] = df['wins'] / df['matches_played']
    df['points_per_game'] = df['points'] / df['matches_played']
    df['goal_difference'] = df['goals_scored'] - df['goals_conceded']
    df['clean_sheet_rate'] = 0  # Would need match-by-match data
    
    return df


# ============================================================================
# BO2 - Match Prediction Preprocessing
# ============================================================================

def preprocess_match_data(home_team: str, away_team: str, 
                         historical_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Preprocess match data for BO2 prediction.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        historical_data: Recent match history (optional)
    
    Returns:
        DataFrame with engineered features
    """
    # This needs to match your BO2 feature engineering
    # Extract features like:
    # - home_wins_L5, home_goals_L5, home_form_L5
    # - away_wins_L5, away_goals_L5, away_form_L5
    # - head_to_head history
    
    # Placeholder - needs actual implementation
    features = {
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'home_wins_L5': 0,  # Calculate from historical_data
        'home_goals_L5': 0,
        'away_wins_L5': 0,
        'away_goals_L5': 0,
        # ... add all 25 features used in BO2
    }
    
    return pd.DataFrame([features])


def encode_teams(home_team: str, away_team: str, 
                team_encoder) -> Tuple[int, int]:
    """
    Encode team names to integers using trained LabelEncoder.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        team_encoder: Trained LabelEncoder from model
    
    Returns:
        Tuple of (home_encoded, away_encoded)
    """
    try:
        home_encoded = team_encoder.transform([home_team])[0]
        away_encoded = team_encoder.transform([away_team])[0]
        return home_encoded, away_encoded
    except ValueError as e:
        raise ValueError(f"Unknown team name: {e}")


# ============================================================================
# BO3 - Team Clustering Preprocessing
# ============================================================================

def preprocess_team_style_data(team_season_stats: Dict) -> pd.DataFrame:
    """
    Preprocess team statistics for clustering (BO3).
    
    Args:
        team_season_stats: Dictionary with aggregated season stats
    
    Returns:
        DataFrame with the 18 style features
    """
    df = pd.DataFrame([team_season_stats])
    
    # Engineer features (must match BO3 notebook)
    df['Fouls_per_Match'] = df['Fouls'] / df['Matches_Played']
    df['Yellow_per_Match'] = df['Yellow_Cards'] / df['Matches_Played']
    df['Red_per_Match'] = df['Red_Cards'] / df['Matches_Played']
    df['Cards_per_Foul'] = (df['Yellow_Cards'] + df['Red_Cards']) / df['Fouls'].replace(0, np.nan)
    df['Goals_per_Shot'] = df['Avg_Goals_Scored'] / df['Avg_Shots'].replace(0, np.nan)
    df['Corners_per_Shot'] = df['Avg_Corners'] / df['Avg_Shots'].replace(0, np.nan)
    
    # Select the 18 style features used in clustering
    style_features = [
        'Avg_Goals_Scored', 'Avg_Shots', 'Avg_Shots_On_Target', 
        'Shot_Accuracy', 'Goals_per_Shot',
        'Avg_Goals_Conceded', 'Clean_Sheet_Rate',
        'Avg_Corners', 'Corners_per_Shot',
        'Fouls_per_Match', 'Yellow_per_Match', 'Red_per_Match', 'Cards_per_Foul',
        'Win_Rate', 'Home_Win_Rate', 'Away_Win_Rate', 'Points_Per_Game'
    ]
    
    return df[style_features]


# ============================================================================
# BO4 - Player Recommendations Preprocessing
# ============================================================================

def preprocess_player_data(player_stats: Dict, position: str) -> pd.DataFrame:
    """
    Preprocess player statistics for recommendation (BO4).
    
    Args:
        player_stats: Dictionary with player performance metrics
        position: Player position (Defender, Midfielder, Forward, Goalkeeper)
    
    Returns:
        DataFrame with position-specific features
    """
    df = pd.DataFrame([player_stats])
    
    # Engineer per-90 metrics
    df['Goals_per_90'] = (df['Gls'] / df['90s']).replace([np.inf, -np.inf], 0).fillna(0)
    df['Assists_per_90'] = (df['Ast'] / df['90s']).replace([np.inf, -np.inf], 0).fillna(0)
    df['Tackles_per_90'] = (df['Tkl'] / df['90s']).replace([np.inf, -np.inf], 0).fillna(0)
    df['Interceptions_per_90'] = (df['Int'] / df['90s']).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Efficiency ratios
    df['Pass_Completion_pct'] = (df['Cmp'] / df['Att'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    df['Shots_on_Target_pct'] = (df['SoT'] / df['Sh'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Productivity
    df['Productivity_Score'] = ((df['Gls'] + df['Ast']) / df['90s']).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Position-specific features (from BO4 notebook)
    position_features = {
        'Defender': ['Age', 'Min', '90s', 'Tackles_per_90', 'Interceptions_per_90', 
                    'Tkl', 'Int', 'Clr', 'Pass_Completion_pct', 'Goals_per_90', 'Assists_per_90'],
        'Midfielder': ['Age', 'Min', '90s', 'Goals_per_90', 'Assists_per_90', 'Productivity_Score',
                      'Pass_Completion_pct', 'KP', 'PrgP', 'Tackles_per_90', 'Interceptions_per_90'],
        'Forward': ['Age', 'Min', '90s', 'Goals_per_90', 'Assists_per_90', 'Productivity_Score',
                   'Sh', 'SoT', 'Shots_on_Target_pct', 'G/Sh', 'PrgC'],
        'Goalkeeper': ['Age', 'Min', '90s', 'GA90', 'Save%', 'Saves', 'CS%', 'PSxG']
    }
    
    features = position_features.get(position, [])
    return df[features]


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_team_name(team: str, valid_teams: List[str]) -> Tuple[bool, str]:
    """
    Validate and suggest team names.
    
    Args:
        team: Input team name
        valid_teams: List of valid team names
    
    Returns:
        Tuple of (is_valid, suggestion_or_error_message)
    """
    if team in valid_teams:
        return True, team
    
    # Fuzzy match suggestions
    from difflib import get_close_matches
    suggestions = get_close_matches(team, valid_teams, n=3, cutoff=0.6)
    
    if suggestions:
        return False, f"Did you mean: {', '.join(suggestions)}?"
    else:
        return False, f"Unknown team: {team}"


def validate_position(position: str) -> Tuple[bool, str]:
    """
    Validate player position.
    
    Args:
        position: Input position
    
    Returns:
        Tuple of (is_valid, normalized_position)
    """
    valid_positions = ['Defender', 'Midfielder', 'Forward', 'Goalkeeper']
    
    # Normalize input
    position_map = {
        'def': 'Defender',
        'defender': 'Defender',
        'mid': 'Midfielder',
        'midfielder': 'Midfielder',
        'fwd': 'Forward',
        'forward': 'Forward',
        'gk': 'Goalkeeper',
        'goalkeeper': 'Goalkeeper'
    }
    
    normalized = position_map.get(position.lower())
    
    if normalized:
        return True, normalized
    else:
        return False, f"Invalid position. Valid: {', '.join(valid_positions)}"


# ============================================================================
# Response Formatting
# ============================================================================

def format_probabilities(proba_array: np.ndarray, 
                        classes: List[str] = ['Away Win', 'Draw', 'Home Win']) -> Dict:
    """
    Format model probabilities for API response.
    
    Args:
        proba_array: Array of probabilities from model.predict_proba()
        classes: Class labels
    
    Returns:
        Dictionary mapping classes to probabilities
    """
    return {
        cls.lower().replace(' ', '_'): float(prob)
        for cls, prob in zip(classes, proba_array[0])
    }


def get_confidence_level(max_probability: float) -> str:
    """
    Convert probability to confidence level.
    
    Args:
        max_probability: Maximum probability from prediction
    
    Returns:
        Confidence level string
    """
    if max_probability >= 0.7:
        return "High"
    elif max_probability >= 0.5:
        return "Medium"
    else:
        return "Low"
