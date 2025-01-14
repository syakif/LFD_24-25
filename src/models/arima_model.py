import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def predict_future_prices(csv_file, prediction_days):
    """
    Simple ARIMA model to predict future Smfdolar values
    
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
    df.set_index('Tarih', inplace=True)
    
    # Get Smfdolar series
    data = df['Smfdolar']
    
    # Fit ARIMA model (p,d,q) = (1,1,1)
    model = ARIMA(data, order=(1,1,1))
    model_fit = model.fit()
    
    # Make future predictions
    forecast = model_fit.forecast(steps=prediction_days)
    
    # Create DataFrame with predictions
    forecast_df = pd.DataFrame({
        'Date': pd.date_range(start=data.index[-1], periods=prediction_days+1)[1:],
        'Predicted_Smfdolar': forecast
    })
    forecast_df.set_index('Date', inplace=True)
    
    # Plot the results
    """""
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data, label='Historical Data')
    plt.plot(forecast_df.index, forecast_df['Predicted_Smfdolar'], label='Predictions', color='red')
    plt.title('Smfdolar: Historical Data and Future Predictions')
    plt.xlabel('Date')
    plt.ylabel('Smfdolar')
    plt.legend()
    plt.grid(True)
    plt.show()
    """""
    return forecast_df

# Example usage
if __name__ == "__main__":
    # Change these values as needed
    csv_file = "smfdb.csv"
    days_to_predict = 30  # Change this to predict different number of days
    
    predictions = predict_future_prices(csv_file, days_to_predict)
    print("\nPredicted Values:")
    print(predictions)

