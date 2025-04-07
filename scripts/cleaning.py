import pandas as pd
import re

def extract_year(date_str):
    """Extract the first 4 digits (year) from a date string."""
    if pd.isna(date_str):
        return date_str
        
    # Check if it's already a year format
    if isinstance(date_str, (int, float)):
        return int(date_str)
    
    # Convert to string if not already
    date_str = str(date_str)
    
    # Extract the first 4 consecutive digits (year)
    match = re.search(r'\b\d{4}\b', date_str)
    if match:
        return match.group(0)
    
    # If no 4-digit pattern found, try to extract any digits
    digits = re.search(r'\d+', date_str)
    if digits and len(digits.group(0)) >= 4:
        return digits.group(0)[:4]
    
    # Return original if no year format found
    return date_str

# Load the CSV file
print("Loading CSV file...")
file_path = "C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\datasets\\heatify_catalogue.csv"
df = pd.read_csv(file_path)

# Check if 'Release Date' column exists
if 'Release Date' not in df.columns:
    print("Error: 'Release Date' column not found.")
    print("Available columns:")
    print(df.columns.tolist())
    exit(1)

# Display dataset info before cleaning
print(f"\nDataset shape before cleaning: {df.shape}")

# 1. Remove rows where track name and artist name match
print("\nRemoving rows where track name and artist name match...")
initial_count = len(df)
if 'Track Name' in df.columns and 'Artist' in df.columns:
    df = df[df['Track Name'] != df['Artist']]
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rows where track name and artist name matched.")
else:
    print("Warning: 'Track Name' or 'Artist' column not found.")

# 2. Remove specified columns
columns_to_remove = ['Added At', 'Added By', 'Key', 'Mode', 'Time Signature']
print(f"\nRemoving columns: {columns_to_remove}")
existing_columns = [col for col in columns_to_remove if col in df.columns]
if existing_columns:
    df = df.drop(columns=existing_columns)
    print(f"Removed columns: {existing_columns}")
else:
    print("None of the specified columns found in the dataset.")

# Display the first few values before cleaning Release Date
print("\nOriginal Release Date format (first 5 samples):")
print(df['Release Date'].head())

# Process the 'Release Date' column
print("\nCleaning Release Date column...")
df['Release Date'] = df['Release Date'].apply(extract_year)

# Display the first few values after cleaning
print("\nCleaned Release Date format (first 5 samples):")
print(df['Release Date'].head())

# Display dataset info after all cleaning
print(f"\nDataset shape after cleaning: {df.shape}")

# Save the updated CSV
output_path = "C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\datasets\\heatify_catalogue.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved cleaned CSV to: {output_path}")

# Show a summary of the years
if pd.api.types.is_numeric_dtype(df['Release Date']):
    print("\nRelease Year Summary:")
    print(f"Earliest Year: {df['Release Date'].min()}")
    print(f"Latest Year: {df['Release Date'].max()}")
    print("\nYear Distribution:")
    year_counts = df['Release Date'].value_counts().sort_index()
    print(year_counts.head(10))  # Show top 10 years
else:
    print("\nRelease Date contains non-numeric values even after cleaning.")