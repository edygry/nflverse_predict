import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any
import joblib
import csv

def load_data(file_path: str) -> pl.DataFrame:
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    processed_data = []
    for row in data:
        processed_row = {
            'Team': row['Team'],
            'FGM': int(row['FGM']),
            'Att': int(row['Att']),
            'FG %': float(row['FG %']),
            'Lng': int(row['Lng']),
            'FG Blk': int(row['FG Blk']),
            'Year': int(row['Year'])
        }
        
        for col in ['1-19', '20-29', '30-39', '40-49', '50-59', '60+']:
            att, made = map(int, row[f'{col} > A-M'].split('_'))
            processed_row[f'{col}_Att'] = att
            processed_row[f'{col}_Made'] = made
        
        processed_data.append(processed_row)
    
    return pl.DataFrame(processed_data)

def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (pl.col("FGM") / pl.col("Att")).alias("FG_Success_Rate"),
        (pl.col("50-59_Made") + pl.col("60+_Made")).alias("Long_FG_Made"),
        (pl.col("50-59_Att") + pl.col("60+_Att")).alias("Long_FG_Att"),
        ((pl.col("50-59_Made") + pl.col("60+_Made")) / (pl.col("50-59_Att") + pl.col("60+_Att"))).fill_null(0).alias("Long_FG_Success_Rate"),
        (pl.col("FG Blk") / pl.col("Att")).alias("FG_Block_Rate")
    ])

def prepare_data_for_modeling(file_path: str) -> pl.DataFrame:
    df_raw = load_data(file_path)
    return engineer_features(df_raw)

def split_data(df: pl.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    feature_columns = [
        "Att", "FG %", "20-29_Att", "20-29_Made", "30-39_Att", "30-39_Made",
        "40-49_Att", "40-49_Made", "Long_FG_Att", "Long_FG_Made", "Lng",
        "FG_Success_Rate", "Long_FG_Success_Rate", "FG_Block_Rate"
    ]
    X = df.select(feature_columns)
    y = df['FGM']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
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
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: xgb.XGBRegressor, X_test: pl.DataFrame, y_test: pl.Series) -> Dict[str, float]:
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

def analyze_feature_importance(model: xgb.XGBRegressor, feature_names: list) -> pl.DataFrame:
    importances = model.feature_importances_
    return pl.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort('importance', descending=True)

def save_model_and_data(model: xgb.XGBRegressor, df_prepared: pl.DataFrame, feature_names: list, model_path: str, data_path: str):
    # Save the model
    joblib.dump({'model': model, 'feature_names': feature_names}, model_path)
    # Save the prepared data
    df_prepared.write_parquet(data_path)
    print(f"Model and feature names saved to {model_path}")
    print(f"Prepared data saved to {data_path}")

def main():
    file_path = "assets/nfl_field_goal_stats_multiple_years.csv"  # Update this path
    model_path = "assets/model_assets/nfl_fg_model.joblib"
    data_path = "assets/model_assets/nfl_fg_data.parquet"
    
    print("Preparing data...")
    df_prepared = prepare_data_for_modeling(file_path)
    print(f"Prepared data for {len(df_prepared)} team-seasons")
    
    print("\nSplitting data into train and test sets...")
    data_split = split_data(df_prepared)
    
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
    print("Example usage in another script:")
    print("model, df_prepared = load_model_and_data('nfl_fg_model.joblib', 'nfl_fg_data.parquet')")
    print("prediction = predict_game(model, 'Cleveland Browns', 'Jacksonville Jaguars', df_prepared)")
    print("print(prediction)")

if __name__ == "__main__":
    main()