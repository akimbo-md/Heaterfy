import os
import pandas as pd
import numpy as np
import datetime

from src.data_cleaning import clean_dataset
from src.embeddings import genre_embeddings
from src.models.classification_model import run_classification_model
from src.models.regression_model import run_regression_model
from src.models.glm_model import run_glm_model
from src.models.clustering_model import run_kmeans_clustering
from src.models.neural_network import run_neural_network
from src.fill_missing_genres import fill_missing_genres
from src.models.cosine_similarity_model import run_cosine_similarity

def main(dataset_path, playlist_path, threshold=0.60, recency_bonus=0.0):
    print(f"Loading tracks from: {dataset_path}")
    print(f"Loading playlist from: {playlist_path}")
    
    # Create output path in the new_playlist directory
    new_playlist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_playlists")
    os.makedirs(new_playlist_dir, exist_ok=True)
    playlist_name = os.path.splitext(os.path.basename(playlist_path))[0]
    output_path = os.path.join(new_playlist_dir, f"{playlist_name}_heaterfy_recommendations.csv")
    
    # Load data
    df = pd.read_csv(dataset_path)
    playlist_df = pd.read_csv(playlist_path)
    
    print(f"Catalog size: {len(df)} tracks")
    print(f"Playlist size: {len(playlist_df)} tracks")
    
    # Genre summary
    if 'Genres' in playlist_df.columns:
        print("\nAnalyzing genres in the playlist...")
        genre_counts = playlist_df['Genres'].dropna().str.split(',').explode().str.strip().value_counts()
        
        if len(genre_counts) > 0:
            print("Top 5 Genres in the Playlist:")
            for genre, count in genre_counts.head(5).items():
                print(f"\t> {genre}: {count} tracks")
        else:
            print("WARN No genres found in the playlist!")
    else:
        print("WARN 'Genres' column not found in the playlist!")
    
    # STEP 0: Check for duplicates in the playlist
    if 'Track Name' in playlist_df.columns and 'Artist Name(s)' in playlist_df.columns:
        # Create unique key for each track
        playlist_df['dedup_key'] = playlist_df['Track Name'].str.strip().str.lower() + '|' + playlist_df['Artist Name(s)'].str.strip().str.lower()
        playlist_dupe_count = len(playlist_df) - playlist_df['dedup_key'].nunique()
        
        if playlist_dupe_count > 0:
            print(f"Found {playlist_dupe_count} duplicate tracks in playlist")
            playlist_df = playlist_df.drop_duplicates(subset=['dedup_key'], keep='first')
            print(f"Removed duplicates. New playlist size: {len(playlist_df)}")
            
        # Drop temp key
        playlist_df = playlist_df.drop(columns=['dedup_key'])
    
    # STEP 1: Match playlist tracks with catalogue and clean
    print("\nChecking for playlist tracks in main catalog...")
    if 'Track Name' in df.columns and 'Artist Name(s)' in df.columns:
        # Create matching keys
        df['match_key'] = df['Track Name'].str.lower() + '|' + df['Artist Name(s)'].str.lower()
        playlist_df['match_key'] = playlist_df['Track Name'].str.lower() + '|' + playlist_df['Artist Name(s)'].str.lower()
        
        # Count matches
        matches = playlist_df['match_key'].isin(df['match_key'].values)
        match_count = matches.sum()
        print(f"Found {match_count}/{len(playlist_df)} playlist tracks in the main catalog")
        
        if match_count > 0 and 'Genres' in df.columns:
            # Get indices of matches in playlist
            matching_playlist_indices = playlist_df[matches].index
            
            # For each match, update genre info from main catalog
            for idx in matching_playlist_indices:
                playlist_key = playlist_df.loc[idx, 'match_key']
                # Find matching row in main catalog
                match_row = df[df['match_key'] == playlist_key].iloc[0]
                
                # Update genre info if missing in playlist but available in catalog
                if 'Genres' not in playlist_df.columns or pd.isna(playlist_df.loc[idx, 'Genres']):
                    if 'Genres' in df.columns and not pd.isna(match_row['Genres']):
                        if 'Genres' not in playlist_df.columns:
                            playlist_df['Genres'] = ""
                        playlist_df.loc[idx, 'Genres'] = match_row['Genres']
                        print(f"Added genres for '{playlist_df.loc[idx, 'Track Name']}' from catalog")
                
                # Copy genre_embedding if available in main catalog
                if 'genre_embedding' in df.columns and not pd.isna(match_row['genre_embedding']):
                    if 'genre_embedding' not in playlist_df.columns:
                        playlist_df['genre_embedding'] = None
                    playlist_df.loc[idx, 'genre_embedding'] = match_row['genre_embedding']
                    print(f"Added genre embedding for '{playlist_df.loc[idx, 'Track Name']}' from catalog")
        
        # Remove temporary keys
        df = df.drop('match_key', axis=1)
        playlist_df = playlist_df.drop('match_key', axis=1)
    
    # STEP 2: Fill in missing genres
    print("\n🔥 PHASE 1: Filling Missing Genres 🔥")
    playlist_df = fill_missing_genres(playlist_df, df)
    
    # STEP 3: Add genre embeddings for playlist
    print("\n🔥 PHASE 2: Computing Genre Embeddings 🔥")
    playlist_df = genre_embeddings(playlist_df)
    
    # STEP 4: Cleaning
    print("\n🔥 PHASE 4: Cleaning the data 🔥")
    df = clean_dataset(df)
    playlist_df = clean_dataset(playlist_df)
    
    # Remove tracks already in the playlist
    if 'Track Name' in df.columns and 'Artist Name(s)' in df.columns:
        # Create key for each song
        df['track_artist_key'] = df['Track Name'].str.lower() + '|' + df['Artist Name(s)'].str.lower()
        playlist_df['track_artist_key'] = playlist_df['Track Name'].str.lower() + '|' + playlist_df['Artist Name(s)'].str.lower()
        playlist_keys = set(playlist_df['track_artist_key'])
        
        # Filter songs already in the playlist
        original_size = len(df)
        df = df[~df['track_artist_key'].isin(playlist_keys)]
        
        # Remove temp key
        df = df.drop(columns=['track_artist_key'])
        playlist_df = playlist_df.drop(columns=['track_artist_key'])
        
        print(f"Removed {original_size - len(df)} tracks already in the playlist")
    else:
        print("Couldn't filter playlist tracks (missing Track Name or Artist Name)")
        
    print(f"Dataset after cleaning: {len(df)} tracks")
    
    # Step 4: INITIAL FILTERING with XGBoost
    print("\n🔥 PHASE 4: Initial Filtering with XGBOOST 🔥")
    filtered_songs, playlist_df, audio_features = run_classification_model(df, playlist_df, threshold)
    print(f"Tracks remaining after filtering: {len(filtered_songs)} ({len(filtered_songs)/len(df)*100:.1f}%)")
    
    # Step 5: Add release year penalty
    print("\nLooking for the playlist era")
    if 'Release Date' in filtered_songs.columns and 'Release Date' in playlist_df.columns:
        print("Extracting release years...")

        print("Sample release dates from playlist:")
        sample_dates = playlist_df['Release Date'].dropna().sample(min(5, len(playlist_df))).tolist()
        for date in sample_dates:
            print(f"  > '{date}'")

        print("Sample release dates from filtered songs:")
        sample_dates = filtered_songs['Release Date'].dropna().sample(min(5, len(filtered_songs))).tolist()
        for date in sample_dates:
            print(f"  > '{date}'")

        def extract_year(date_str):
            if pd.isna(date_str):
                return None
                
            date_str = str(date_str).strip()
            
            try:
                parsed_date = pd.to_datetime(date_str, errors='raise')
                return parsed_date.year
            except:
                try:
                    # 4-digit year
                    if date_str.isdigit() and len(date_str) == 4:
                        year = int(date_str)
                        if 1900 <= year <= 2030:
                            return year
                except:
                    pass
                    
                # Ughhhhhhhhhhhghghhghg
                import re
                year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
                if year_match:
                    return int(year_match.group(0))
                    
            # We failed.
            print(f"ERROR: Could not parse date: '{date_str}'")
            return None

        # Extract the years (fr this time)
        filtered_songs['Release Year'] = filtered_songs['Release Date'].apply(extract_year)
        playlist_df['Release Year'] = playlist_df['Release Date'].apply(extract_year)
        songs_with_years = filtered_songs['Release Year'].notna().sum()
        playlist_with_years = playlist_df['Release Year'].notna().sum()

        if songs_with_years > 0 and playlist_with_years > 0:
            year_counts = filtered_songs['Release Year'].value_counts().sort_index()
            print("\n📊 Year distribution in filtered songs:")
            # Bucket by decade
            decades = {}
            for year, count in year_counts.items():
                if pd.notna(year):
                    decade = f"{int(year) // 10 * 10}s"
                    if decade not in decades:
                        decades[decade] = 0
                    decades[decade] += count
            
            for decade, count in sorted(decades.items()):
                print(f"  > {decade}: {count} tracks ({count/songs_with_years*100:.1f}%)")
            
            # Calculate playlist year statistics
            playlist_years = playlist_df['Release Year'].dropna()
            if len(playlist_years) > 0:
                avg_playlist_year = int(playlist_years.mean())
                min_playlist_year = int(playlist_years.min())
                max_playlist_year = int(playlist_years.max())
                playlist_range = max_playlist_year - min_playlist_year
                
                # Calculate standard deviation to measure time span focus
                std_playlist_year = playlist_years.std()
                
                print(f"Playlist release years: {min_playlist_year}-{max_playlist_year} (avg: {avg_playlist_year}, std: {std_playlist_year:.2f})")
                
                # Check if this is a recent playlist first
                is_recent_playlist = avg_playlist_year >= 2019
                
                # Determine if this is a decade-focused playlist
                decade_playlist = False
                decade_focus = ""
                
                # Only consider decade classification if it's not a recent playlist
                if not is_recent_playlist:
                    # Count tracks by decade
                    decade_counts = {}
                    for year in playlist_years:
                        decade = (year // 10) * 10
                        decade_counts[decade] = decade_counts.get(decade, 0) + 1
                    
                    # Check if any decade has more than 80% of tracks
                    total_tracks = len(playlist_years)
                    for decade, count in decade_counts.items():
                        percentage = count / total_tracks
                        if percentage >= 0.8:
                            decade_playlist = True
                            decade_focus = f"{decade}s"
                            print(f"Detected a {decade_focus} focused playlist! ({count} of {total_tracks} tracks, {percentage*100:.1f}%)")
                            break
                
                # Explicitly report playlist classification
                if is_recent_playlist:
                    print(f"🆕 Detected a recent/modern playlist (avg year: {avg_playlist_year})")
                    
                    # Override decade classification for recent playlists
                    if decade_playlist:
                        print(f"ℹThis playlist would qualify as a {decade_focus} playlist, but recent playlist classification takes precedence")
                        decade_playlist = False
                elif decade_playlist:
                    print(f"This is a {decade_focus} decade-focused playlist")
                else:
                    print(f"This is a mixed-era playlist (average year: {avg_playlist_year})")
                
                # Calculate year distance for each track
                filtered_songs['year_distance'] = abs(filtered_songs['Release Year'] - avg_playlist_year)
                
                # Dynamic penalty based on playlist characteristics
                if decade_playlist:
                    # Heavy penalty for decade playlists
                    max_distance_for_penalty = 10
                    max_penalty = 40.0
                    print(f"Using strict decade-focused penalty (max: {max_penalty}% for tracks {max_distance_for_penalty}+ years from average)")
                elif is_recent_playlist:
                    # Moderate penalty for recent playlists
                    max_distance_for_penalty = 20
                    max_penalty = 25.0
                    print(f"Using modern playlist penalty (max: {max_penalty}% for tracks {max_distance_for_penalty}+ years from average)")
                elif std_playlist_year < 5:
                    # Very specific era playlists
                    max_distance_for_penalty = 15
                    max_penalty = 30.0
                    print(f"Using tight era penalty (max: {max_penalty}% for tracks {max_distance_for_penalty}+ years from average)")
                else:
                    # For more time-diverse playlists
                    max_distance_for_penalty = 25
                    max_penalty = 20.0
                    print(f"Using standard era penalty (max: {max_penalty}% for tracks {max_distance_for_penalty}+ years from average)")
                    
                # Calculate penalty based on distance
                filtered_songs['year_penalty'] = (filtered_songs['year_distance'] / max_distance_for_penalty) * max_penalty
                filtered_songs['year_penalty'] = filtered_songs['year_penalty'].clip(0, max_penalty)
                
                # Add recency bonus for modern playlists
                if is_recent_playlist:
                    current_year = 2025
                    filtered_songs['recency_bonus'] = 0.0
                    
                    # Add stronger bonus to recent tracks for recent playlists
                    recent_mask = (filtered_songs['Release Year'] >= 2019) & (filtered_songs['Release Year'] <= current_year)
                    
                    # Bonus increases from 0 to 15% for newest tracks
                    filtered_songs.loc[recent_mask, 'recency_bonus'] = (filtered_songs.loc[recent_mask, 'Release Year'] - 2019) / (current_year - 2019) * 15.0
                    
                    print(f"🚀 Applied enhanced recency bonus (up to 15%) for tracks from 2019-{current_year}")
                    
                    # Display recency bonus distribution
                    if recent_mask.sum() > 0:
                        print(f"{recent_mask.sum()} tracks eligible for recency bonus ({recent_mask.sum()/len(filtered_songs)*100:.1f}%)")
                        print(f"Average recency bonus: {filtered_songs.loc[recent_mask, 'recency_bonus'].mean():.2f}%")
                else:
                    current_year = 2025
                    filtered_songs['recency_bonus'] = 0.0
                    
                    # Bonus to very recent tracks
                    recent_mask = (filtered_songs['Release Year'] >= 2022) & (filtered_songs['Release Year'] <= current_year)
                    
                    # Smaller bonus for non-recent playlists
                    if recent_mask.sum() > 0:
                        filtered_songs.loc[recent_mask, 'recency_bonus'] = (filtered_songs.loc[recent_mask, 'Release Year'] - 2022) / (current_year - 2022) * 5.0
                        print(f"Applied small recency bonus (up to 5%) for very recent tracks (2022-{current_year})")
                
                # Calculate final year adjustment (penalty minus bonus)
                filtered_songs['year_adjustment'] = filtered_songs['year_penalty'] - filtered_songs['recency_bonus']
                filtered_songs['year_adjustment'] = filtered_songs['year_adjustment'].clip(-10, max_penalty)  # Cap bonus at 10%
                print(f"Year adjustments: Min: {filtered_songs['year_adjustment'].min():.1f}%, Max: {filtered_songs['year_adjustment'].max():.1f}%, Avg: {filtered_songs['year_adjustment'].mean():.1f}%")
                
                # Calculate year_score (for visualization) - 0-100 scale
                max_distance = filtered_songs['year_distance'].max()
                if max_distance > 0:
                    filtered_songs['year_score'] = 100 - (filtered_songs['year_distance'] / max_distance * 100)
                else:
                    filtered_songs['year_score'] = 100
                
                bins = [-10, -5, 0, 5, 10, 20, 30, 40]
                labels = ['Bonus 5-10%', 'Bonus 0-5%', 'No adj.', '0-5%', '5-10%', '10-20%', '20-30%']
                filtered_songs['year_adjustment_bin'] = pd.cut(filtered_songs['year_adjustment'], bins=bins, labels=labels, right=False)
                filtered_songs['year_adjustment_bin'] = pd.cut(filtered_songs['year_adjustment'], bins=bins, labels=labels, right=False)
                
                print("\nYear adjustment distribution:")
                adj_dist = filtered_songs['year_adjustment_bin'].value_counts().sort_index()
                for bin_label, count in adj_dist.items():
                    percentage = (count / len(filtered_songs)) * 100
                    print(f"\t> {bin_label}: {count} tracks ({percentage:.1f}%)")
                
                # Show outliers with max penalty
                max_penalty_tracks = filtered_songs[filtered_songs['year_penalty'] >= (max_penalty - 0.1)]
                if len(max_penalty_tracks) > 0:
                    print(f"\nYo, {len(max_penalty_tracks)} tracks with max penalty (furthest from playlist years):")
                    sample_max = max_penalty_tracks.sample(min(5, len(max_penalty_tracks)))
                    for _, row in sample_max.iterrows():
                        print(f"\t> {row['Track Name']} by {row['Artist Name(s)']} ({row['Release Year']}) - {row['year_distance']} years from average")
                
                # Show tracks with highest bonus
                bonus_tracks = filtered_songs[filtered_songs['recency_bonus'] > 7.5]
                if len(bonus_tracks) > 0:
                    print(f"\n🚀 {len(bonus_tracks)} tracks with high recency bonus:")
                    sample_bonus = bonus_tracks.sample(min(5, len(bonus_tracks)))
                    for _, row in sample_bonus.iterrows():
                        print(f"\t> {row['Track Name']} by {row['Artist Name(s)']} ({row['Release Year']}) - {row['recency_bonus']:.1f}% bonus")
    
    # Step 6: AUDIO FEATURE SCORING with Ridge Regression
    print("\n🔥 PHASE 5: Audio Feature Scoring with Ridge Regression🔥")
    regression_features = [
        "Danceability", "Energy", "Tempo", "Valence", "Liveness", "Acousticness", 
        "Instrumentalness", "Speechiness", "Loudness"
    ]
    # DEBUG: Check which features are available
    available_features = [f for f in regression_features if f in filtered_songs.columns]
    if len(available_features) < len(regression_features):
        print(f"Using only {len(available_features)}/{len(regression_features)} audio features")
        print(f"Missing: {set(regression_features) - set(available_features)}")
    
    filtered_songs = run_regression_model(filtered_songs, playlist_df, available_features)

    print("\n🔥 PHASE 6: Cosine Similarity Scoring🔥")
    filtered_songs = run_cosine_similarity(filtered_songs, playlist_df)
    
    # Step 7: METADATA SCORING with GLM
    print("\n🔥 PHASE 7: Metadata Scoring with GLM🔥")
    filtered_songs = run_glm_model(filtered_songs, playlist_df)

    # Step 8: CLUSTER ANALYSIS with K-means
    print("\n🔥 PHASE 8: Cluster Analysis with k-means🔥")
    filtered_songs, playlist_df, clustering_data = run_kmeans_clustering(filtered_songs, playlist_df)
    
    # 0-100 range
    for score_col in ['fitness_score_normalized', 'glm_score_normalized', 'cluster_score']:
        if score_col in filtered_songs.columns:
            min_val = filtered_songs[score_col].min()
            max_val = filtered_songs[score_col].max()
            
            if max_val > 100 or min_val < 0 or (max_val - min_val < 0.01):
                filtered_songs[score_col] = 100 * (filtered_songs[score_col] - min_val) / (max_val - min_val)
    
    available_scores = [
    'cosine_score_normalized',
    'fitness_score_normalized',
    'glm_score_normalized',
    'cluster_score'
    ]
    available_scores = [score for score in available_scores if score in filtered_songs.columns]
    
    # Step 9: NEURAL NETWORK for final recommendations
    print("\n🔥 PHASE 9: Neural Network 🔥")
    filtered_songs = run_neural_network(filtered_songs, playlist_df, available_scores)

    # Step 9: FINAL SELECTION
    print("\n🔥 PHASE 10: Final Selection 🔥")

    # How much we should consider each model (default settings)
    model_weights = {
        'cosine_score_normalized': 0.6,  # Best perfomer imo
        'fitness_score_normalized': 0.25, # 2nd best
        'nn_score': 0.05,                 # dissapointment...
        'glm_score_normalized': 0.1     # did aight
    }

    # Only use models that produced scores
    available_weights = {k: v for k, v in model_weights.items() if k in filtered_songs.columns}
    total = sum(available_weights.values())
    normalized_weights = {k: v/total for k, v in available_weights.items()}

    print(f"Base score weights: {normalized_weights}")

    # Calculate base score
    filtered_songs['base_score'] = 0
    for score, weight in normalized_weights.items():
        if score in filtered_songs.columns:
            filtered_songs['base_score'] += filtered_songs[score].fillna(0) * weight

    # Apply release year penalty
    if 'year_adjustment' in filtered_songs.columns:
        print("Applying release year adjustments to base scores")
        filtered_songs['base_score'] = filtered_songs['base_score'] * (1 - filtered_songs['year_adjustment']/100)
        print(f"Applied year adjustments ranging from {filtered_songs['year_adjustment'].min():.1f}% to {filtered_songs['year_adjustment'].max():.1f}%")
    elif 'year_penalty' in filtered_songs.columns:
        # Fallback to old method if adjustment isn't calculated
        print("Applying release year penalty to base scores (legacy method)")
        filtered_songs['base_score'] = filtered_songs['base_score'] * (1 - filtered_songs['year_penalty']/100)
    
    # User recency bonus
    if recency_bonus != 0.0:
        # Get current year
        from datetime import datetime  # Make sure this is imported at the top
        current_year = datetime.now().year  # This should work now
        
        # Apply recency bonus to scores if Release Year is available
        if 'Release Year' in filtered_songs.columns:
            # Normalize years to 0-1 scale
            years = filtered_songs['Release Year']
            min_year = years.min()
            max_year = max(years.max(), current_year)
            year_range = max_year - min_year if max_year > min_year else 1
            
            # Calculate normalized year (0-1)
            normalized_years = (years - min_year) / year_range
            
            # Apply recency bonus (positive for newer, negative for older)
            filtered_songs['base_score'] = filtered_songs['base_score'] * (1 + recency_bonus * normalized_years)
            print(f"Applied recency bonus of {recency_bonus:.1f} (positive favors newer tracks)")
    
    # Apply clustering
    if 'cluster' in filtered_songs.columns and 'cluster' in playlist_df.columns:
        print("\nApplying cluster-based diversity selection")
        
        # Calculate cluster distribution in original playlist
        playlist_cluster_dist = playlist_df['cluster'].value_counts(normalize=True).to_dict()
        print("Original playlist cluster distribution:")
        for cluster, percentage in playlist_cluster_dist.items():
            count = playlist_df[playlist_df['cluster'] == cluster].shape[0]
            print(f"\t> Cluster {cluster}: {percentage*100:.1f}% ({count} tracks)")
        
        num_recommendations = 100
        print(f"Generating {num_recommendations} recommendations with similar cluster distribution")
        
        # Calculate how many tracks we should put into each cluster
        cluster_allocations = {}
        for cluster, percentage in playlist_cluster_dist.items():
            cluster_allocations[cluster] = max(1, round(percentage * num_recommendations))
        
        total_allocated = sum(cluster_allocations.values())
        if total_allocated != num_recommendations:
            # Find the cluster with the highest representation to add remainder to
            largest_cluster = max(cluster_allocations, key=cluster_allocations.get)
            cluster_allocations[largest_cluster] += (num_recommendations - total_allocated)
        
        print("Cluster allocations for recommendations:")
        for cluster, count in cluster_allocations.items():
            print(f"\t> Cluster {cluster}: {count} tracks")
        
        # Choose the final tracks for each cluster
        final_recommendations = []
        
        for cluster, count in cluster_allocations.items():
            cluster_tracks = filtered_songs[filtered_songs['cluster'] == cluster]
            
            if len(cluster_tracks) == 0:
                print(f"⚠️ No tracks found in cluster {cluster}! Reallocating slots.")
                continue
                
            # Sort by base score
            top_cluster_tracks = cluster_tracks.sort_values('base_score', ascending=False).head(count)
            
            final_recommendations.append(top_cluster_tracks)
        
        final_filtered_songs = pd.concat(final_recommendations)
        
        # Not enough tracks, just add the highest scorers to the main cluster
        if len(final_filtered_songs) < num_recommendations:
            shortfall = num_recommendations - len(final_filtered_songs)
            print(f"Only selected {len(final_filtered_songs)} tracks from cluster allocation. Adding {shortfall} more high-scoring tracks.")
            selected_ids = set(final_filtered_songs.index)
            
            remaining_tracks = filtered_songs[~filtered_songs.index.isin(selected_ids)]
            remaining_tracks = remaining_tracks.sort_values('base_score', ascending=False)
            
            additional_tracks = remaining_tracks.head(shortfall)
            final_filtered_songs = pd.concat([final_filtered_songs, additional_tracks])
        
        # Sort recommendations by base_score
        final_filtered_songs = final_filtered_songs.sort_values('base_score', ascending=False)
        
        print(f"Generated {len(final_filtered_songs)} diverse recommendations matching playlist cluster distribution")
        filtered_songs = final_filtered_songs
    else:
        # No clusters :(
        print("No cluster information available.")
        filtered_songs = filtered_songs.sort_values('base_score', ascending=False)

    # Print final results summary!!!!
    print("\nFINAL RESULTS!!!")
    print(f"Total tracks recommended: {len(filtered_songs)}")
    print(f"Average Base Score: {filtered_songs['base_score'].mean():.2f}")
    print(f"Score Range: {filtered_songs['base_score'].min():.2f} - {filtered_songs['base_score'].max():.2f}")


    if 'cluster' in filtered_songs.columns:
        final_dist = filtered_songs['cluster'].value_counts(normalize=True).to_dict()
        print("\nFinal recommendation cluster distribution:")
        for cluster, percentage in final_dist.items():
            count = filtered_songs[filtered_songs['cluster'] == cluster].shape[0]
            original_pct = playlist_cluster_dist.get(cluster, 0) * 100
            print(f"\t> Cluster {cluster}: {percentage*100:.1f}% ({count} tracks) | Original: {original_pct:.1f}%")
    
    print("\nModel Performance Summary:")
    for score in available_scores + ['combined_score', 'nn_score']:
        if score in filtered_songs.columns:
            non_zero = filtered_songs[filtered_songs[score] > 0][score]
            zero_count = (filtered_songs[score] == 0).sum()
            
            print(f"\n{score}:")
            if len(non_zero) > 0:
                print(f"  Non-zero scores: {len(non_zero)}/{len(filtered_songs)} tracks ({len(non_zero)/len(filtered_songs)*100:.1f}%)")
                print(f"  Mean (non-zero): {non_zero.mean():.2f}")
                print(f"  Min: {non_zero.min():.2f}, Max: {filtered_songs[score].max():.2f}")
            else:
                print(f"  All scores are zero!")
    
    # Display top recommendations
    print("\n🔥 Top 20 Tracks Overall:")
    display_cols = ['Track Name', 'Artist Name(s)']
    
    # Get top 20 tracks for display
    top_combined = filtered_songs.head(20).copy()
    for col in top_combined.columns:
        if 'score' in col.lower():
            top_combined[col] = top_combined[col].round(1).astype(str) + '%'
    
    # Display the top combined tracks
    print(top_combined[display_cols])
    
    # Add link to tracks
    if 'Track ID' in top_combined.columns:
        print("\nSpotify Links to Top Tracks:")
        for i, (_, row) in enumerate(top_combined.head(5).iterrows()):
            if pd.notna(row['Track ID']):
                print(f"{i+1}. {row['Track Name']} - {row['Artist Name(s)']}:\n   https://open.spotify.com/track/{row['Track ID']}")
                
    # Save the final results
    try:
        # Existing code for saving and returning
        filtered_songs.to_csv(output_path, index=False)
        playlist_output_path = os.path.join(os.path.dirname(output_path), f"{playlist_name}_with_clusters.csv")
        playlist_df.to_csv(playlist_output_path, index=False)
        print(f"\nFinal results saved to: {output_path}")
        
        # Before returning, validate filtered_songs is not empty
        if filtered_songs is None or len(filtered_songs) == 0:
            print("Warning: No recommendations were generated")
            # Return empty DataFrames rather than None
            return pd.DataFrame(), playlist_df, clustering_data
            
        return filtered_songs, playlist_df, clustering_data
    except Exception as e:
        print(f"Error in final processing: {str(e)}")
        # Return empty DataFrame instead of None to avoid NoneType errors
        return pd.DataFrame(), playlist_df, {}

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "datasets", "heaterfy_catalogue_final2.csv")
    playlist_path = os.path.join(base_dir, "spotify_rips", "afterhours.csv")
    
    results = main(dataset_path, playlist_path)