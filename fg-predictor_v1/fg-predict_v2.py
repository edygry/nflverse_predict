import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
from typing import List, Tuple
import joblib

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

def load_data(fg_path: str, opp_fg_path: str) -> pl.DataFrame:
    df_fg = pl.read_csv(fg_path)
    df_opp_fg = pl.read_csv(opp_fg_path)
    
    # Standardize team names in both datasets
    df_fg = standardize_team_names(df_fg, "Team")
    df_opp_fg = standardize_team_names(df_opp_fg, "Team")
    
    print("Field goal data sample after standardization:")
    print(df_fg.select(["Team", "Year"]).head())
    print("\nOpponent FG data sample after standardization:")
    print(df_opp_fg.select(["Team", "Year"]).head())
    
    # Ensure 'Year' column is of the same type in both dataframes
    df_fg = df_fg.with_columns(pl.col('Year').cast(pl.Int64))
    df_opp_fg = df_opp_fg.with_columns(pl.col('Year').cast(pl.Int64))
    
    # Merge the datasets
    df_combined = df_fg.join(df_opp_fg, on=["Team", "Year"], how="left")
    
    print(f"\nCombined data shape: {df_combined.shape}")
    print("Sample of combined data:")
    print(df_combined.head())
    
    # Check for null values in Opp_FG_Per_Game after merging
    null_count = df_combined['Opp_FG_Per_Game'].null_count()
    if null_count > 0:
        print(f"\nWarning: {null_count} null values found in Opp_FG_Per_Game after merging.")
        print("Teams with null Opp_FG_Per_Game:")
        print(df_combined.filter(pl.col("Opp_FG_Per_Game").is_null()).select(["Team", "Year"]))
    
    return df_combined

def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    df_engineered = df.with_columns([
        (pl.col("FGM") / pl.col("Att")).fill_null(0).alias("FG_Success_Rate"),
        pl.col("50-59 > A-M").str.split("_").list.get(1).cast(pl.Int64).alias("50-59_Made"),
        pl.col("60+ > A-M").str.split("_").list.get(1).cast(pl.Int64).alias("60+_Made"),
        pl.col("50-59 > A-M").str.split("_").list.get(0).cast(pl.Int64).alias("50-59_Att"),
        pl.col("60+ > A-M").str.split("_").list.get(0).cast(pl.Int64).alias("60+_Att"),
        (pl.col("FGM") / 16).alias("FG_Per_Game")  # Assuming 16 games per season
    ])

    df_engineered = df_engineered.with_columns([
        (pl.col("50-59_Made") + pl.col("60+_Made")).alias("Long_FG_Made"),
        (pl.col("50-59_Att") + pl.col("60+_Att")).alias("Long_FG_Att")
    ])

    df_engineered = df_engineered.with_columns([
        (pl.when(pl.col("Long_FG_Att") > 0)
         .then(pl.col("Long_FG_Made") / pl.col("Long_FG_Att"))
         .otherwise(0)).alias("Long_FG_Success_Rate"),
        (pl.when(pl.col("Att") > 0)
         .then(pl.col("FG Blk") / pl.col("Att"))
         .otherwise(0)).alias("FG_Block_Rate")
    ])

    # Check if Opp_FG_Per_Game is all null
    if df_engineered["Opp_FG_Per_Game"].is_null().all():
        print("Warning: All Opp_FG_Per_Game values are null. Using placeholder values.")
        df_engineered = df_engineered.with_columns([
            pl.lit(0).alias("Opp_FG_Z_Score"),
            pl.lit(1).alias("Relative_Opp_Strength"),
            pl.col("FG %").alias("Adjusted_FG_Percentage")
        ])
    else:
        league_avg_opp_fg = df_engineered["Opp_FG_Per_Game"].mean()
        league_std_opp_fg = df_engineered["Opp_FG_Per_Game"].std()

        print(f"League average Opp_FG_Per_Game: {league_avg_opp_fg}")
        print(f"League std dev Opp_FG_Per_Game: {league_std_opp_fg}")

        df_engineered = df_engineered.with_columns([
            ((pl.col("Opp_FG_Per_Game") - league_avg_opp_fg) / league_std_opp_fg).fill_null(0).alias("Opp_FG_Z_Score"),
            (pl.col("Opp_FG_Per_Game") / league_avg_opp_fg).fill_null(1).alias("Relative_Opp_Strength"),
            (pl.when(pl.col("Opp_FG_Per_Game") > 0)
             .then(pl.col("FG %") * (league_avg_opp_fg / pl.col("Opp_FG_Per_Game")))
             .otherwise(pl.col("FG %"))).alias("Adjusted_FG_Percentage")
        ])

    return df_engineered

def select_features(X: np.ndarray, y: np.ndarray, feature_names: list, k: int = 15) -> Tuple[np.ndarray, list]:
    selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    return X_selected, selected_features

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mae,
        'R2': r2,
    }

def evaluate_model_cv(model: any, X: np.ndarray, y: np.ndarray, cv: int = 5) -> None:
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    print(f"Cross-validation R2 scores: {scores}")
    print(f"Mean R2: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

def analyze_feature_importance(model: any, feature_names: list) -> pl.DataFrame:
    if isinstance(model, RandomForestRegressor):
        importances = model.feature_importances_
    elif isinstance(model, xgb.XGBRegressor):
        importances = model.feature_importances_
    else:
        raise ValueError("Unsupported model type for feature importance analysis")
    
    return pl.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort('importance', descending=True)

def save_model_and_data(model: any, df_prepared: pl.DataFrame, feature_names: list, model_path: str, data_path: str) -> None:
    joblib.dump({'model': model, 'feature_names': feature_names}, model_path)
    df_prepared.write_parquet(data_path)
    print(f"Model and feature names saved to {model_path}")
    print(f"Prepared data saved to {data_path}")

def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    # Print column names and their null counts
    null_counts = df.null_count()
    print("Null counts for each column:")
    print(null_counts)
    
    # Remove rows where FGM (target variable) is null
    df_cleaned = df.filter(pl.col('FGM').is_not_null())
    
    # For other columns, we'll impute missing values
    columns_to_impute = [col for col in df.columns if col != 'FGM' and df[col].dtype in [pl.Float64, pl.Int64]]
    
    for col in columns_to_impute:
        if df[col].dtype in [pl.Float64, pl.Int64]:
            df_cleaned = df_cleaned.with_columns(pl.col(col).fill_null(strategy="mean"))
    
    print(f"Rows before cleaning: {len(df)}")
    print(f"Rows after cleaning: {len(df_cleaned)}")
    
    return df_cleaned

def main():
    fg_path = "assets/nfl_field_goal_stats_multiple_years.csv"
    opp_fg_path = "assets/opponent_fg_stats_multiple_years.csv"
    model_path = "assets/model_assets/new_nfl_fg_model.joblib"
    data_path = "assets/model_assets/new_nfl_fg_data.parquet"
    
    print("Loading and preparing data...")
    df_combined = load_data(fg_path, opp_fg_path)
    
    print("\nEngineering features...")
    df_prepared = engineer_features(df_combined)
    
    print("\nSample of prepared data:")
    print(df_prepared.head())
    
    print("\nColumn names in prepared data:")
    print(df_prepared.columns)
    
    print("\nData types of columns:")
    print(df_prepared.dtypes)
    
    print("\nCleaning data...")
    df_cleaned = clean_data(df_prepared)
    print(f"Cleaned data for {len(df_cleaned)} team-seasons")
    
    print("\nSample of cleaned data:")
    print(df_cleaned.head())
    
    print("\nNull counts after cleaning:")
    print(df_cleaned.null_count())
    
    feature_columns = [
        "Att", "FG %", "FG_Success_Rate", "Long_FG_Success_Rate",
        "FG_Block_Rate", "Opp_FG_Z_Score", "Relative_Opp_Strength", "Adjusted_FG_Percentage"
    ]
    
    X = df_cleaned.select(feature_columns).to_numpy()
    y = df_cleaned['FG_Per_Game'].to_numpy()  # Use FG_Per_Game as target
    
    
    print("\nPerforming feature selection...")
    X_selected, selected_features = select_features(X, y, feature_columns)
    print("Selected features:", selected_features)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    print("\nTraining Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)
    
    print("\nEvaluating Random Forest model performance...")
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print("\nRandom Forest Model Performance Metrics:")
    for metric, value in rf_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nPerforming cross-validation for Random Forest model...")
    evaluate_model_cv(rf_model, X_selected, y)
    
    print("\nAnalyzing Random Forest feature importance...")
    rf_feature_importance = analyze_feature_importance(rf_model, selected_features)
    print("\nTop 10 Most Important Features (Random Forest):")
    print(rf_feature_importance.head(10))
    
    print("\nTraining XGBoost model...")
    xgb_model = train_xgboost(X_train, y_train)
    
    print("\nEvaluating XGBoost model performance...")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
    print("\nXGBoost Model Performance Metrics:")
    for metric, value in xgb_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nPerforming cross-validation for XGBoost model...")
    evaluate_model_cv(xgb_model, X_selected, y)
    
    print("\nAnalyzing XGBoost feature importance...")
    xgb_feature_importance = analyze_feature_importance(xgb_model, selected_features)
    print("\nTop 10 Most Important Features (XGBoost):")
    print(xgb_feature_importance.head(10))
    
    # Choose the best model based on R2 score
    best_model = rf_model if rf_metrics['R2'] > xgb_metrics['R2'] else xgb_model
    best_model_name = "Random Forest" if best_model == rf_model else "XGBoost"
    
    print(f"\nSaving the best model ({best_model_name}) and prepared data...")
    save_model_and_data(best_model, df_cleaned, selected_features, model_path, data_path)
    
    print("\nModel training and evaluation complete.")

if __name__ == "__main__":
    main()