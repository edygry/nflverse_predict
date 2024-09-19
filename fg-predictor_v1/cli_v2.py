import sys
import joblib
import polars as pl
from thefuzz import process
from typing import List, Tuple
import numpy as np

def standardize_team_names(df: pl.DataFrame, column: str) -> pl.DataFrame:
    team_name_mapping = {
        "49ers": "San Francisco",
        "Niners": "San Francisco",
        "Bears": "Chicago",
        "Bengals": "Cincinnati",
        "Bills": "Buffalo",
        "Broncos": "Denver",
        "Browns": "Cleveland",
        "Buccaneers": "Tampa Bay",
        "Cardinals": "Arizona",
        "Chargers": "LA Chargers",
        "Chiefs": "Kansas City",
        "Colts": "Indianapolis",
        "Cowboys": "Dallas",
        "Dolphins": "Miami",
        "Eagles": "Philadelphia",
        "Falcons": "Atlanta",
        "Giants": "NY Giants",
        "Jaguars": "Jacksonville",
        "Jets": "NY Jets",
        "Lions": "Detroit",
        "Packers": "Green Bay",
        "Panthers": "Carolina",
        "Patriots": "New England",
        "Raiders": "Las Vegas",
        "Rams": "LA Rams",
        "Ravens": "Baltimore",
        "Redskins": "Washington",
        "Saints": "New Orleans",
        "Seahawks": "Seattle",
        "Steelers": "Pittsburgh",
        "Texans": "Houston",
        "Titans": "Tennessee",
        "Vikings": "Minnesota",
        "Football Team": "Washington",
        "Commanders": "Washington"
    }
    
    return df.with_columns(pl.col(column).replace(team_name_mapping))

def load_model_and_data(model_path: str, data_path: str):
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_names = model_data['feature_names']
    df_prepared = pl.read_parquet(data_path)
    df_prepared = standardize_team_names(df_prepared, "Team")
    return model, df_prepared, feature_names

def predict_game(model, home_team: str, away_team: str, season_data: pl.DataFrame, feature_names: list) -> dict:
    home_stats = season_data.filter(pl.col("Team") == home_team)
    away_stats = season_data.filter(pl.col("Team") == away_team)
    
    if home_stats.is_empty() or away_stats.is_empty():
        return {"error": "One or both teams not found in the data"}
    
    # Use the most recent year's data
    home_stats_recent = home_stats.sort("Year", descending=True).head(1)
    away_stats_recent = away_stats.sort("Year", descending=True).head(1)
    
    # Get league average Opp_FG_Per_Game
    league_avg_opp_fg = season_data.select('Opp_FG_Per_Game').mean().item()
    
    # Get Opp_FG_Per_Game for each team
    home_opp_fg = home_stats_recent.select('Opp_FG_Per_Game').item()
    away_opp_fg = away_stats_recent.select('Opp_FG_Per_Game').item()
    
    # Calculate relative opponent strength
    home_relative_strength = away_opp_fg / league_avg_opp_fg if league_avg_opp_fg else 1
    away_relative_strength = home_opp_fg / league_avg_opp_fg if league_avg_opp_fg else 1
    
    # Prepare feature vectors
    home_features = home_stats_recent.select(feature_names)
    away_features = away_stats_recent.select(feature_names)
    
    # Update Relative_Opp_Strength for prediction
    home_features = home_features.with_columns(pl.lit(home_relative_strength).alias("Relative_Opp_Strength"))
    away_features = away_features.with_columns(pl.lit(away_relative_strength).alias("Relative_Opp_Strength"))
    
    # Make predictions
    home_prediction = model.predict(home_features.to_numpy())[0]
    away_prediction = model.predict(away_features.to_numpy())[0]
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_fg_prediction": home_prediction,
        "away_fg_prediction": away_prediction,
        "total_fg_prediction": home_prediction + away_prediction,
        "home_prob_1.7_plus_fg": float(home_prediction >= 1.7),
        "away_prob_1.7_plus_fg": float(away_prediction >= 1.7),
        "prob_1.7_plus_fg": float(home_prediction + away_prediction >= 1.7)
    }

def get_team_names(df: pl.DataFrame) -> List[str]:
    return df['Team'].unique().to_list()

def find_closest_match(team: str, team_list: List[str]) -> Tuple[str, int]:
    return process.extractOne(team, team_list)

def main():
    if len(sys.argv) != 3:
        print("Usage: python cli.py 'Home Team' 'Away Team'")
        sys.exit(1)

    model_path = "assets/model_assets/new_nfl_fg_model.joblib"
    data_path = "assets/model_assets/new_nfl_fg_data.parquet"

    try:
        model, df_prepared, feature_names = load_model_and_data(model_path, data_path)
    except FileNotFoundError:
        print("Error: Model or data file not found. Please ensure you've run the training script first.")
        sys.exit(1)

    team_list = get_team_names(df_prepared)

    home_team_input = sys.argv[1]
    away_team_input = sys.argv[2]

    home_team_match, home_team_score = find_closest_match(home_team_input, team_list)
    away_team_match, away_team_score = find_closest_match(away_team_input, team_list)

    if home_team_score < 80 or away_team_score < 80:
        print("Warning: One or both team names might be misspelled.")
        print(f"Did you mean '{home_team_match}' for the home team?")
        print(f"Did you mean '{away_team_match}' for the away team?")
        confirm = input("Do you want to proceed with these teams? (y/n): ")
        if confirm.lower() != 'y':
            print("Prediction cancelled.")
            sys.exit(0)

    prediction = predict_game(model, home_team_match, away_team_match, df_prepared, feature_names)

    if "error" in prediction:
        print(f"Error: {prediction['error']}")
    else:
        print(f"\nPrediction for {prediction['home_team']} vs {prediction['away_team']}:")
        print(f"Home team ({prediction['home_team']}):")
        print(f"  Expected field goals: {prediction['home_fg_prediction']:.2f}")
        print(f"  Probability of 1.7 or more field goals: {prediction['home_prob_1.7_plus_fg']:.2%}")
        print(f"Away team ({prediction['away_team']}):")
        print(f"  Expected field goals: {prediction['away_fg_prediction']:.2f}")
        print(f"  Probability of 1.7 or more field goals: {prediction['away_prob_1.7_plus_fg']:.2%}")
        print(f"Total expected field goals: {prediction['total_fg_prediction']:.2f}")
        print(f"Probability of 1.7 or more total field goals: {prediction['prob_1.7_plus_fg']:.2%}")

if __name__ == "__main__":
    main()