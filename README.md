# System Marginal Price Forecasting Project

This repository contains the starting code for the SMP forecasting project. The goal is to develop a comprehensive price forecasting system for Turkey's electricity market using various machine learning techniques.

## Project Structure

- `data/`: Raw and processed data files
- `notebooks/`: Jupyter notebooks for exploration and analysis .
- `src/`: Source code for the project
- `tests/`: Unit tests

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Follow the notebooks in order:
   - Data collection
   - Data exploration
   - Model evaluation

## Project Requirements

See the project documentation for detailed requirements and grading criteria.

## Models to Implement

### Univariate Approaches
- Statistical Models (ARIMA, SARIMA, PROPHET)
- Machine Learning Models (SVR, Random Forest, XGBoost)
- Deep Learning Models (LSTM, Transformer)

### Multivariate Approaches
Implement the same models with additional features:
- Weather data
- Currency exchange rates
- Economic indicators
- Production sources

## Evaluation

Models will be evaluated using:
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
