import pandas as pd
import sys

try:
    # Load the dataset, just as the notebook does
    df = pd.read_csv('data/merged_data_v2.csv', low_memory=False)

    # Replicate the *exact* operation from cell 52
    # This function creates an array of integer codes and an array of unique values
    # The 'uniques' array is the key, as its index *is* the integer ID
    codes, uniques = pd.factorize(df['StationId'])

    # Create a new, clean DataFrame from the 'uniques' array
    # The index of this new DataFrame (0, 1, 2...) is the 'stationid_int'
    mapping_df = pd.DataFrame(uniques, columns=['StationId'])

    # Add the index as its own column to make the mapping explicit
    mapping_df['stationid_int'] = mapping_df.index

    # Re-order columns for clarity
    mapping_df = mapping_df[['stationid_int', 'StationId']]

    # Save the complete mapping to a new CSV file
    mapping_df.to_csv('station_id_mapping.csv', index=False)

    print("Successfully created 'station_id_mapping.csv'")
    print("\n--- Preview of Mapping ---")
    print(mapping_df.head(20).to_string())
    print("\n...")
    print(f"\nTotal stations mapped: {len(mapping_df)}")

except FileNotFoundError:
    print("Error: The file 'merged_data_v2.csv' was not found.", file=sys.stderr)
except Exception as e:
    print(f"An error occurred: {e}", file=sys.stderr)