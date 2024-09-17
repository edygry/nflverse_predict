# nflverse_predict
# NFL Field Goal Prediction

## Overview

This project uses machine learning to predict field goal performance in NFL games. It analyzes historical field goal data to forecast the number of field goals likely to be scored in future matchups.

## Features

- Predicts expected field goals for home and away teams
- Calculates probabilities for teams scoring above a certain threshold
- Uses XGBoost regression model for accurate predictions
- Handles data from various field goal distances and scenarios

## How It Works

1. **Data Processing**: The system ingests historical NFL field goal data, including attempts, successes, and distances.

2. **Feature Engineering**: It creates relevant features from the raw data, such as success rates at different distances and overall kicking performance metrics.

3. **Model Training**: An XGBoost regression model is trained on the processed data to learn patterns in field goal scoring.

4. **Prediction**: Given two teams, the model predicts the expected number of field goals for each team in their matchup.

## Usage

The project consists of two main components:

1. A training script that processes data and trains the model.
2. A prediction script that uses the trained model to make game-specific predictions.

## Requirements

- Python 3.7+
- Libraries: tbd

## Data

The model is trained on historical NFL field goal data (not included in the repository)

## Future Improvements

- Incorporate additional factors like weather conditions and player-specific stats
- Implement a web interface for easy predictions
- Regularly update the model with the latest NFL data

---