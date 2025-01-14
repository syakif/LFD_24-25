import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from builders import extractDate

def predict_future_prices(csv_file, prediction_days):
    """
    Simple SVR model to predict future Smfdolar values using all features
    
    Parameters:
    csv_file (str): Path to CSV file
    prediction_days (int): Number of days to predict into the future
    
    Returns:
    DataFrame with predicted values
    """
    # Read the data
    df = pd.read_csv(csv_file)
    
    # Convert date and set as index
    df['Tarih'] = pd.to_datetime(df['Tarih'], format='%d/%m/%Y %H:%M')
    dates = df['Tarih']
    
    # Prepare features (X) and target (y)
    target = 'Smfdolar'
    features = df.columns.difference(['Tarih', target]) #Var olan bütün column datalarını almaya yarayan code
    
    X = df[features]
    y = df[target]
    
    # Clean the data
    # Replace infinite values with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with median of each column
    X = X.fillna(X.median())
    
    # Handle any remaining extreme values by capping them
    for column in X.columns:
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X[column] = X[column].clip(lower=lower_bound, upper=upper_bound)
    
    # Scale the features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    # Train SVR model
    model = SVR(kernel='rbf', C=100, gamma='auto')
    model.fit(X_scaled, y_scaled)
    
    # Prepare data for future predictions
    last_row = X_scaled[-1].reshape(1, -1)
    future_predictions_scaled = []
    
    # Make iterative predictions
    for _ in range(prediction_days):
        # Make prediction
        next_pred_scaled = model.predict(last_row)
        future_predictions_scaled.append(next_pred_scaled[0])
        
        # Update last row (shift features)
        last_row = np.roll(last_row, -1)
        last_row[0, -1] = next_pred_scaled[0]
    
    # Inverse transform predictions
    future_predictions = scaler_y.inverse_transform(
        np.array(future_predictions_scaled).reshape(-1, 1)
    ).ravel()
    
    # Create future dates
    future_dates = pd.date_range(
        start=dates.iloc[-1], 
        periods=prediction_days+1, 
        freq='D'
    )[1:]
    
    # Create DataFrame with predictions
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Smfdolar': future_predictions
    })
    forecast_df.set_index('Date', inplace=True)
    """
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(dates, df[target], label='Historical Data')
    plt.plot(forecast_df.index, forecast_df['Predicted_Smfdolar'], 
             label='Predictions', color='red')
    plt.title('Smfdolar: Historical Data and Future Predictions')
    plt.xlabel('Date')
    plt.ylabel('Smfdolar')
    plt.legend()
    plt.grid(True)
    plt.show()
    """
    return forecast_df
"""
# Example usage
if __name__ == "__main__":
    # Change these values as needed
    csv_file = "smfdb2.csv"
    days_to_predict = 30  # Change this to predict different number of days
    
    predictions = predict_future_prices(csv_file, days_to_predict)
    print("\nPredicted Values:")
    print(predictions)
"""
# Example usage
from builders import extractDate
if __name__ == "__main__":
    # Change these values as needed
    csv_file = "with_weather.csv"
    #csv_file = extractDate(csv_file)
    days_to_predict = 250  # Change this to predict different number of days
    
    predictions = predict_future_prices(csv_file, days_to_predict)
    print("\nPredicted Values:")
    print(predictions)   