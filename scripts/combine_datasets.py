#!/usr/bin/env python3
"""
Script to combine all individual season CSV files into a single dataset.

This script reads all season CSV files from the uncombined folder, adds a season column,
and combines them into a single dataset saved in the combined folder.
"""

import pandas as pd
import glob
import os
from pathlib import Path
import re


def extract_season_from_filename(filename):
    """
    Extract season information from filename.

    Args:
        filename (str): The filename containing season info (e.g., 'season-9394.csv')

    Returns:
        str: Season in format 'YYYY-YY' (e.g., '1993-94')
    """
    # Extract the numeric part from filename like 'season-9394.csv'
    match = re.search(r"season-(\d{4})\.csv", filename)
    if match:
        season_code = match.group(1)
        if len(season_code) == 4:
            # Handle formats like '9394' -> '1993-94' or '0001' -> '2000-01'
            year1 = season_code[:2]
            year2 = season_code[2:]

            # Convert to full years
            if int(year1) >= 90:  # Assume 90+ means 1990s
                full_year1 = f"19{year1}"
            else:  # 00-89 means 2000s+
                full_year1 = f"20{year1}"

            if int(year2) >= 90:  # Handle edge case
                full_year2 = f"19{year2}"
            else:
                full_year2 = f"20{year2}"

            return f"{full_year1}-{year2}"

    return "Unknown"


def combine_season_datasets():
    """
    Combine all season CSV files into a single dataset.
    """
    # Define paths
    uncombined_dir = Path(
        "d:/Projects/pl-standings-prediction-project/data/raw/uncombined"
    )
    combined_dir = Path("d:/Projects/pl-standings-prediction-project/data/raw/combined")

    # Ensure output directory exists
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Find all season CSV files
    season_files = glob.glob(str(uncombined_dir / "season-*.csv"))
    season_files.sort()  # Sort to ensure consistent ordering

    print(f"Found {len(season_files)} season files to combine")

    combined_data = []
    total_matches = 0

    for file_path in season_files:
        filename = os.path.basename(file_path)
        season = extract_season_from_filename(filename)

        print(f"Processing {filename} -> Season {season}")

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Add season column
            df["Season"] = season

            # Add original filename for reference
            df["SourceFile"] = filename

            # Append to combined data
            combined_data.append(df)

            print(f"  Added {len(df)} matches from {season}")
            total_matches += len(df)

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

    if combined_data:
        # Combine all dataframes
        print(f"\nCombining {len(combined_data)} season datasets...")
        combined_df = pd.concat(combined_data, ignore_index=True)

        # Reorder columns to put Season and SourceFile at the beginning
        cols = combined_df.columns.tolist()
        cols.remove("Season")
        cols.remove("SourceFile")
        new_order = ["Season", "SourceFile"] + cols
        combined_df = combined_df[new_order]

        # Convert Date column to datetime for better handling
        try:
            combined_df["Date"] = pd.to_datetime(
                combined_df["Date"], format="%d/%m/%y", errors="coerce"
            )
        except Exception as e:
            print(f"Warning: Could not convert dates: {e}")

        # Sort by Season and Date
        combined_df = combined_df.sort_values(["Season", "Date"]).reset_index(drop=True)

        # Save combined dataset
        output_file = combined_dir / "premier_league_combined.csv"
        combined_df.to_csv(output_file, index=False)

        print(
            f"\n✅ Successfully combined {total_matches} matches from {len(season_files)} seasons"
        )
        print(f"📊 Combined dataset saved to: {output_file}")
        print(f"📏 Dataset shape: {combined_df.shape}")

        # Print summary statistics
        print(f"\n📈 Summary:")
        print(f"  • Seasons covered: {combined_df['Season'].nunique()}")
        print(
            f"  • Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}"
        )
        print(f"  • Total matches: {len(combined_df)}")
        print(
            f"  • Unique teams: {len(set(combined_df['HomeTeam'].unique()) | set(combined_df['AwayTeam'].unique()))}"
        )

        # Show first few rows
        print(f"\n🔍 Sample data:")
        print(combined_df.head())

        return combined_df

    else:
        print("❌ No data to combine!")
        return None


if __name__ == "__main__":
    print("🏈 Premier League Dataset Combiner")
    print("=" * 50)

    result = combine_season_datasets()

    if result is not None:
        print("\n🎉 Dataset combination completed successfully!")
    else:
        print("\n💥 Dataset combination failed!")
