import pandas as pd

def extractDate(file_path):

    # Load the example CSV file
    df = pd.read_csv(file_path)

    # Convert the 'Date' column to datetime
    df['Tarih'] = pd.to_datetime(df['Tarih'], format='%d/%m/%Y %H:%M')

    # Extract datetime features
    df['Year'] = df['Tarih'].dt.year
    df['Month'] = df['Tarih'].dt.month
    df['Day'] = df['Tarih'].dt.day
    df['Hour'] = df['Tarih'].dt.hour

    # Drop the original "Tarih" column
    #df.drop('Tarih', axis=1, inplace=True)

    return df

def addRollingStat(file_path, window_size):

    df = pd.read_csv(file_path)

    df["Rolling Mean"] = df["Value"].rolling(window=window_size).mean()
    df["Rolling Sum"] = df["Value"].rolling(window=window_size).sum()

    return df