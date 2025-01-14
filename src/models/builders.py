import pandas as pd

def extractDate(file_path):

    # Load the example CSV file
    df = pd.read_csv(file_path)
    print(df)
    # Convert the 'Date' column to datetime if it exists
    df['Tarih'] = pd.to_datetime(df['Tarih'], format='%d/%m/%Y %H:%M')

    # Extract datetime features
    df['Year'] = df['Tarih'].dt.year #bu kötü etkiliyor olabilir
    df['Month'] = df['Tarih'].dt.month
    df['Day'] = df['Tarih'].dt.day
    df['Hour'] = df['Tarih'].dt.hour

    # Drop the original "Tarih" column
    #df.drop('Tarih', axis=1, inplace=True)

    df['Tarih'] = df['Tarih'].dt.strftime('%d/%m/%Y %H:%M')

    return df

def addRollingStat(file_path, window_size):

    # Step 1: Read the CSV file
    df = pd.read_csv(file_path)

    # Step 2: Add rolling statistics
    df["Rolling Mean"] = df["Smfdolar"].rolling(window=window_size).mean()
    df["Rolling Sum"] = df["Smfdolar"].rolling(window=window_size).sum()

    return df


"""
#Deneme
file = "with_weather.csv"
df = extractDate(file)

df.to_csv("deneme.csv", index=False)

file2 = "deneme.csv"

df = addRollingStat(file2, 7)

df.to_csv("deneme.csv", index=False)

print(df)
"""