import pandas as pd
import os
import sys

def merge_datasets():
    """
    Merge datasets to combine genre information with the full dataset.
    1. Keep all tracks from both datasets
    2. Keep all columns from both datasets
    3. For tracks that appear in both datasets, use genre information from the first dataset
    """
    print("Starting dataset merge process...")
    
    # Define file paths
    base_dir = "C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\datasets"
    genre_file_path = os.path.join(base_dir, "heatify_full_dataset_updated_embedded.csv")
    main_file_path = os.path.join(base_dir, "heatify_full_dataset.csv")
    output_path = os.path.join(base_dir, "heatify_catalogue.csv")
    
    # Check if files exist
    for file_path in [genre_file_path, main_file_path]:
        if not os.path.exists(file_path):
            print(f"Error: File does not exist: {file_path}")
            sys.exit(1)
    
    # Load datasets
    print(f"Loading genre dataset: {genre_file_path}")
    genre_df = pd.read_csv(genre_file_path)
    
    print(f"Loading main dataset: {main_file_path}")
    main_df = pd.read_csv(main_file_path)
    
    # Display dataset info
    print(f"\nGenre dataset shape: {genre_df.shape}")
    print(f"Main dataset shape: {main_df.shape}")
    
    # Identify genre-related columns
    genre_columns = []
    for col in genre_df.columns:
        if 'genre' in col.lower() or col == 'Genres':
            genre_columns.append(col)
    
    print(f"Found genre columns: {genre_columns}")
    
    # Standardize the matching columns in both datasets
    matching_cols = ['Track Name', 'Artist Name(s)']
    
    # Check if columns exist
    for col in matching_cols:
        if col not in genre_df.columns:
            print(f"Warning: '{col}' not found in genre dataset")
        if col not in main_df.columns:
            print(f"Warning: '{col}' not found in main dataset")
    
    print(f"\nUsing {matching_cols} for matching tracks between datasets")
    
    # APPROACH: Full Outer Join with genre priority
    
    # 1. First identify all unique columns from both datasets
    all_columns = list(set(main_df.columns) | set(genre_df.columns))
    print(f"\nTotal unique columns across both datasets: {len(all_columns)}")
    
    # 2. Create an empty dataframe with all those columns
    print("\nCreating comprehensive dataframe with all columns...")
    merged_df = pd.DataFrame(columns=all_columns)
    
    # 3. Add all tracks from main dataset
    print("\nAdding all tracks from main dataset...")
    for col in main_df.columns:
        merged_df[col] = main_df[col]
    
    # 4. Create a lookup of existing tracks
    print("\nCreating lookup of existing tracks...")
    track_keys = set(tuple(x) for x in merged_df[matching_cols].dropna().values)
    
    # 5. Update genre information for existing tracks
    print("\nUpdating genre information for existing tracks...")
    match_count = 0
    
    # For each track in genre_df, find matching track in merged_df and update genre columns
    for _, genre_row in genre_df.iterrows():
        # Create key for this row
        try:
            key = tuple(genre_row[matching_cols])
            
            # Skip if any key column is NaN
            if pd.isna(key).any():
                continue
                
            # If this track exists in merged_df
            if key in track_keys:
                match_count += 1
                
                # Find matching rows in merged_df
                match_mask = True
                for col in matching_cols:
                    match_mask = match_mask & (merged_df[col] == genre_row[col])
                
                # Update genre columns for this track
                for genre_col in genre_columns:
                    if genre_col in genre_df.columns:
                        merged_df.loc[match_mask, genre_col] = genre_row[genre_col]
        except:
            # Skip problematic rows
            continue
    
    print(f"Updated genre information for {match_count} existing tracks")
    
    # 6. Add new tracks from genre_df
    print("\nAdding new tracks from genre dataset...")
    
    # Identify tracks in genre_df that don't exist in merged_df
    new_track_mask = ~genre_df[matching_cols].apply(tuple, axis=1).isin(track_keys)
    new_tracks_df = genre_df[new_track_mask].copy()
    
    print(f"Found {len(new_tracks_df)} additional tracks in genre dataset")
    
    # Add the new tracks to the merged dataframe
    merged_df = pd.concat([merged_df, new_tracks_df], ignore_index=True, sort=False)
    
    # Display final results
    print(f"\nFinal dataset shape: {merged_df.shape}")
    
    # Check genre column coverage
    for genre_col in genre_columns:
        if genre_col in merged_df.columns:
            genre_count = merged_df[genre_col].notna().sum()
            print(f"Records with {genre_col} information: {genre_count} ({genre_count/len(merged_df)*100:.2f}%)")
    
    # Save the final merged dataset
    print(f"\nSaving final dataset to: {output_path}")
    merged_df.to_csv(output_path, index=False)
    
    print("Merge completed successfully!")
    return merged_df

if __name__ == "__main__":
    merge_datasets()