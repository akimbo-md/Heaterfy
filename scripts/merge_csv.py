import os
import pandas as pd

"""
Merges all CSV files in the specified folder into a single CSV file and removes duplicates
This is useful for building the complete dataset

Will search through the entire folder for CSVs
"""
# Adjust this to what you need
folder_path = "C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\spotify_rips"
output_file = "C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\datasets\\heatify_full_dataset.csv"

def merge_csv_files(folder_path, output_file):
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = []
    total_records_before = 0

    print(f"Found {len(all_files)} CSV files to merge.\n")

    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Adding {len(df)} records from {file}")
        total_records_before += len(df)
        dataframes.append(df)

    # Concatenate all dataframes into one big dataframe
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Print total records before removing duplicates
    print(f"\nTotal records before removing duplicates: {total_records_before}")

    # Remove duplicates based on all columns
    merged_df.drop_duplicates(inplace=True)

    # Print the number of removed duplicates
    removed_records = total_records_before - len(merged_df)
    print(f"Removed {removed_records} duplicate records.")

    # Save the cleaned dataframe to a new CSV file
    merged_df.to_csv(output_file, index=False)

    print(f"\nSuccessfully merged and cleaned {len(merged_df)} records into {output_file}")

merge_csv_files(folder_path, output_file)
