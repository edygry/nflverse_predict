import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any
import joblib

def load_data(fg_path: str, opp_fg_path: str) -> pl.DataFrame:
    df_fg = pl.read_csv(fg_path)
    df_opp_fg = pl.read_csv(opp_fg_path)
    
    # Merge the datasets
    df_combined = df_fg.join(df_opp_fg, on=["Team", "Year"], how="left")
    
    return df_combined

def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    df_engineered = df.with_columns([
        (pl.col("FGM") / pl.col("Att")).alias("FG_Success_Rate"),
        pl.col("50-59 > A-M").str.split("_").list.get(1).cast(pl.Int64).alias("50-59_Made"),
        pl.col("60+ > A-M").str.split("_").list.get(1).cast(pl.Int64).alias("60+_Made"),
        pl.col("50-59 > A-M").str.split("_").list.get(0).cast(pl.Int64).alias("50-59_Att"),
        pl.col("60+ > A-M").str.split("_").list.get(0).cast(pl.Int64).alias("60+_Att"),
    ]).with_columns([
        (pl.col("50-59_Made") + pl.col("60+_Made")).alias("Long_FG_Made"),
        (pl.col("50-59_Att") + pl.col("60+_Att")).alias("Long_FG_Att"),
    ]).with_columns([
        (pl.col("Long_FG_Made") / pl.col("Long_FG_Att")).fill_null(0).alias("Long_FG_Success_Rate"),
        (pl.col("FG Blk") / pl.col("Att")).alias("FG_Block_Rate")
    ])

    # Calculate league average Opp_FG_Per_Game
    league_avg_opp_fg = df_engineered["Opp_FG_Per_Game"].mean()

    # Add a column for relative opponent strength
    df_engineered = df_engineered.with_columns([
        (pl.col("Opp_FG_Per_Game") / league_avg_opp_fg).alias("Relative_Opp_Strength")
    ])

    return df_engineered

def prepare_data_for_modeling(df: pl.DataFrame) -> Dict[str, Any]:
    feature_columns = [
        "Att", "FG %", "20-29 > A-M", "30-39 > A-M",
        "40-49 > A-M", "Lng", "FG_Success_Rate", "Long_FG_Made", 
        "Long_FG_Att", "Long_FG_Success_Rate", "FG_Block_Rate",
        "Relative_Opp_Strength"
    ]
    X = df.select(feature_columns)
    y = df['FGM']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_columns
    }

def split_data(df: pl.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    feature_columns = [
        "Att", "FG %", "20-29 > A-M", "30-39 > A-M",
        "40-49 > A-M", "Lng", "FG_Success_Rate", "Long_FG_Made", 
        "Long_FG_Att", "Long_FG_Success_Rate", "FG_Block_Rate",
        "Opp_FG_Per_Game"
    ]
    X = df.select(feature_columns)
    y = df['FGM']
    
    # Use numpy to create a random mask
    np.random.seed(random_state)
    mask = np.random.rand(len(df)) >= test_size
    
    return {
        'X_train': X.filter(pl.Series(mask)),
        'X_test': X.filter(pl.Series(~mask)),
        'y_train': y.filter(pl.Series(mask)),
        'y_test': y.filter(pl.Series(~mask)),
        'feature_names': feature_columns
    }

def train_model(X_train: pl.DataFrame, y_train: pl.Series) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        tree_method='hist'
    )
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    return model

def evaluate_model(model: xgb.XGBRegressor, X_test: pl.DataFrame, y_test: pl.DataFrame) -> Dict[str, float]:
    y_pred = model.predict(X_test.to_numpy())
    y_true = y_test.to_numpy().ravel()
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mae,
        'R2': r2,
    }

def analyze_feature_importance(model: xgb.XGBRegressor, feature_names: list) -> pl.DataFrame:
    importances = model.feature_importances_
    return pl.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort('importance', descending=True)

def save_model_and_data(model: xgb.XGBRegressor, df_prepared: pl.DataFrame, feature_names: list, model_path: str, data_path: str):
    # Save the model
    joblib.dump({'model': model, 'feature_names': feature_names}, model_path)
    # Save the prepared data, including the metadata columns
    df_prepared.write_parquet(data_path)
    print(f"Model and feature names saved to {model_path}")
    print(f"Prepared data saved to {data_path}")

def main():
    fg_path = "assets/nfl_field_goal_stats_multiple_years.csv"
    opp_fg_path = "assets/opponent_fg_stats_multiple_years.csv"
    model_path = "assets/model_assets/nfl_fg_model.joblib"
    data_path = "assets/model_assets/nfl_fg_data.parquet"
    
    print("Loading and preparing data...")
    df_combined = load_data(fg_path, opp_fg_path)
    df_prepared = engineer_features(df_combined)
    print(f"Prepared data for {len(df_prepared)} team-seasons")
    
    print("\nPreparing data for modeling...")
    data_split = prepare_data_for_modeling(df_prepared)
    
    print("\nTraining XGBoost model...")
    model = train_model(data_split['X_train'], data_split['y_train'])
    
    print("\nEvaluating model performance...")
    metrics = evaluate_model(model, data_split['X_test'], data_split['y_test'])
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nAnalyzing feature importance...")
    feature_importance = analyze_feature_importance(model, data_split['feature_names'])
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    print("\nSaving model and prepared data...")
    save_model_and_data(model, df_prepared, data_split['feature_names'], model_path, data_path)

    print("\nModel and data exported. You can now use these files with the prediction script.")

if __name__ == "__main__":
    main()