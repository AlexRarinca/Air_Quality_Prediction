import pandas as pd

air_quality_file = 'merged_data_v3.csv'  # The pollutant file
weather_file = 'city_daily_means_uv_sw.csv'      # The weatherfile
output_file = 'merged_data_v4.csv'             # The name of the new file to create

# --- Define separate date formats ---
air_quality_date_format = '%d/%m/%Y'
weather_date_format = '%Y-%m-%d'

# Define the join keys
merge_keys = ['City', 'datetime']

# Define the columns to add from the weather file
columns_to_add = ['City', 'datetime', 'SWdown']


try:
    # Load the datasets
    print(f"Loading air quality data from: {air_quality_file}")
    df_air_quality = pd.read_csv(air_quality_file)
    
    print(f"Loading weather data from: {weather_file}")
    df_weather = pd.read_csv(weather_file)

    print("Cleaning and converting datetime columns...")
    
    
    # Clean and convert the air_quality file
    df_air_quality['City'] = df_air_quality['City'].str.strip()
    df_air_quality['datetime'] = pd.to_datetime(
        df_air_quality['datetime'], 
        format=air_quality_date_format, 
        errors='coerce'
    )
    # Force time to 00:00:00 to ensure dates match
    df_air_quality['datetime'] = df_air_quality['datetime'].dt.normalize()

    # Clean and convert the weather file
    df_weather['City'] = df_weather['City'].str.strip()
    df_weather['datetime'] = pd.to_datetime(
        df_weather['datetime'], 
        format=weather_date_format, 
        errors='coerce'
    )
    # Force time to 00:00:00 to ensure dates match
    df_weather['datetime'] = df_weather['datetime'].dt.normalize()
    
    
    # Drop rows where dates could not be parsed
    df_air_quality = df_air_quality.dropna(subset=['datetime'])
    df_weather = df_weather.dropna(subset=['datetime'])

    
    # Select only the columns we need from the weather data
    df_weather_subset = df_weather[columns_to_add]
    
    # ADDED CHECK FOR DUPLICATES
    print("Checking for and aggregating duplicates in weather data...")
    df_weather_subset = df_weather_subset.groupby(['City', 'datetime']).mean().reset_index()


    print("Merging dataframes...")
    df_merged = pd.merge(
        df_air_quality,
        df_weather_subset,
        on=merge_keys,
        how='left'
    )

    # Save the merged data
    df_merged.to_csv(output_file, index=False)

    print("\n--- Merge Complete ---")
    print(f"Merged data saved to: {output_file}")
    
    matches_found = df_merged['SWdown'].notna().sum()
    total_rows = len(df_merged)
    print(f"Successfully matched {matches_found} out of {total_rows} rows.")
    
    if matches_found == 0:
        print("\nWARNING: No matches found.")
    
    print("\nFirst 5 rows of merged data:")
    print(df_merged.head())

except FileNotFoundError as e:
    print(f"Error: File not found. Make sure '{e.filename}' is uploaded and the name is correct.")
except KeyError as e:
    print(f"Error: Column {e} not found. Please check your CSV file column names.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")