import pandas as pd
import re

def extract_year(date_str):
    if pd.isna(date_str):
        return date_str
        
    # Return if already a year
    if isinstance(date_str, (int, float)):
        return int(date_str)
    
    # Convert to string
    date_str = str(date_str)
    
    # Find first 4-digit year
    match = re.search(r'\b\d{4}\b', date_str)
    if match:
        return match.group(0)
    
    # Extract any digits if no proper year found
    digits = re.search(r'\d+', date_str)
    if digits and len(digits.group(0)) >= 4:
        return digits.group(0)[:4]
    
    # Return original if no digits found
    return date_str

def clean_dataset(df):
    print("Cleaning dataset...")

    initial_count = len(df)

    # Remove rows where track name matches artist name
    if 'Track Name' in df.columns and 'Artist Name(s)' in df.columns:
        df = df[df['Track Name'] != df['Artist Name(s)']]
        print("Removed tracks where name matches artist name.")

    # Extract year from release date
    if 'Release Date' in df.columns:
        df['Release Date'] = df['Release Date'].apply(extract_year)
        print("Extracted release year.")

    # Drop columns
    columns_to_remove = ['Added At', 'Added By', 'Key', 'Mode', 'Time Signature']
    existing_columns = [col for col in columns_to_remove if col in df.columns]
    if existing_columns:
        df = df.drop(columns=existing_columns)
        print(f"Removed columns: {existing_columns}")

    # Summary
    final_count = len(df)
    if final_count < initial_count:
        print(f"Cleaned {initial_count - final_count} rows.")

    return df