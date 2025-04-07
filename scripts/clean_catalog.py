import os
import pandas as pd
import argparse

def clean_catalog(input_path, output_path=None):
    # Create default output path if not provided
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_clean.csv"
    
    # Load data
    df = pd.read_csv(input_path)
    original_size = len(df)
    print(f"📊 Original catalog size: {original_size} tracks")
    
    # Step 1: Check for missing required columns
    required_columns = ['Track Name', 'Artist Name(s)']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️ ERROR: Catalog is missing required columns: {missing_columns}")
        return df
    
    # Step 2: Remove rows with missing data in key fields
    missing_data = df['Track Name'].isna() | df['Artist Name(s)'].isna() | \
                  (df['Track Name'] == '') | (df['Artist Name(s)'] == '')
    missing_count = missing_data.sum()
    
    if missing_count > 0:
        print(f"⚠️ Found {missing_count} tracks with missing names or artists")
        df = df[~missing_data]
        print(f"✅ Removed tracks with missing data. Remaining: {len(df)}")
    
    # Step 3: Remove exact duplicates (same track name AND same artist)
    df['dedup_key'] = df['Track Name'].str.strip().str.lower() + '|' + df['Artist Name(s)'].str.strip().str.lower()
    
    # Count duplicates
    value_counts = df['dedup_key'].value_counts()
    duplicated_keys = value_counts[value_counts > 1].index
    duplicate_count = sum(value_counts[value_counts > 1]) - len(duplicated_keys)
    
    if duplicate_count > 0:
        print(f"⚠️ Found {duplicate_count} duplicate tracks")
        
        # Sample some duplicates for verification
        if len(duplicated_keys) > 0:
            sample_size = min(5, len(duplicated_keys))
            print(f"\nExamples of duplicated tracks (showing {sample_size} examples):")
            for i, key in enumerate(duplicated_keys[:sample_size]):
                dupes = df[df['dedup_key'] == key]
                print(f"Example {i+1}: '{dupes['Track Name'].iloc[0]}' by {dupes['Artist Name(s)'].iloc[0]} ({len(dupes)} occurrences)")
        
        # Keep first occurrence of each track
        df = df.drop_duplicates(subset=['dedup_key'], keep='first')
        print(f"✅ Removed duplicates. New size: {len(df)}")
    else:
        print("✅ No duplicate tracks found")
    
    # Drop the temporary deduplication key
    df = df.drop(columns=['dedup_key'])
    
    # Step 4: Save the cleaned catalog
    df.to_csv(output_path, index=False)
    print(f"💾 Saved cleaned catalog to: {output_path}")
    print(f"🧹 Total reduction: {original_size - len(df)} tracks ({100 * (original_size - len(df)) / original_size:.1f}%)")
    
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "datasets", "heatify_catalogue_embedded.csv")
    output_path = os.path.join(base_dir, "datasets", "heatify_catalogue_embedded_cleaned.csv")
    
    clean_catalog(dataset_path, output_path)