import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import statsmodels.api as sm
import os
import re
import matplotlib.pyplot as plt
from kneed import KneeLocator

def parse_genre_embedding(embedding_str):
    # Yo, let's clean up this genre embedding string and make it usable 🎵
    try:
        cleaned = embedding_str.strip("[]").replace(",", " ")
        return np.fromstring(cleaned, sep=' ')
    except:
        # If it's trash, return zeros 🤷‍♂️
        return np.zeros(16)

def extract_year(date_str):
    """Pull out the year from the release date. Gotta know when the bangers dropped. 🎶"""
    if pd.isna(date_str):
        return date_str
        
    # Already a year? Cool, just return it.
    if isinstance(date_str, (int, float)):
        return int(date_str)
    
    # Convert to string if it's not already
    date_str = str(date_str)
    
    # Find the first 4-digit year in the string
    match = re.search(r'\b\d{4}\b', date_str)
    if match:
        return match.group(0)
    
    # If no proper year, grab any digits and hope for the best 🤞
    digits = re.search(r'\d+', date_str)
    if digits and len(digits.group(0)) >= 4:
        return digits.group(0)[:4]
    
    # If all else fails, just return the original
    return date_str

def clean_dataset(df):
    """Time to clean up this dataset. No room for garbage here. 🧹"""
    print("🧼 Cleaning up the dataset...")

    initial_count = len(df)

    # Ditch rows where track name matches artist name. That's just lazy. 🗑️
    if 'Track Name' in df.columns and 'Artist Name(s)' in df.columns:
        df = df[df['Track Name'] != df['Artist Name(s)']]
        print("🚮 Removed duplicates where track name = artist name. Lazy stuff.")

    # Extract the year from the release date. We need to know when the heat dropped. 🔥
    if 'Release Date' in df.columns:
        df['Release Date'] = df['Release Date'].apply(extract_year)
        print("📅 Pulled out the release year from the 'Release Date' column.")

    # Drop the columns we don't care about. Bye-bye unnecessary stuff. 👋
    columns_to_remove = ['Added At', 'Added By', 'Key', 'Mode', 'Time Signature']
    existing_columns = [col for col in columns_to_remove if col in df.columns]
    if existing_columns:
        df = df.drop(columns=existing_columns)
        print(f"🗑️ Tossed out these columns: {existing_columns}")

    # Let’s see how much junk we cleaned out.
    final_count = len(df)
    if final_count < initial_count:
        print(f"✨ Cleaned up {initial_count - final_count} rows. Looking fresh now.")

    return df

def run_kmeans_clustering(filtered_songs, playlist_df):
    """Let’s group these tracks into clusters. Vibes only. 🎶"""
    print("🔍 Running K-means clustering...")

    # Features we care about for clustering
    genre_features = [f"genre_dim_{i}" for i in range(16)]
    audio_features = [
        'Danceability', 'Energy', 'Loudness', 'Speechiness',
        'Acousticness', 'Instrumentalness', 'Liveness', 'Valence',
        'Tempo'
    ]

    # Only keep features that exist in both datasets
    available_audio_features = [f for f in audio_features if f in filtered_songs.columns and f in playlist_df.columns]
    available_genre_features = [f for f in genre_features if f in filtered_songs.columns and f in playlist_df.columns]
    features = available_audio_features + available_genre_features

    # Fill missing values with zeros. No empty vibes allowed. 🛠️
    playlist_df[features] = playlist_df[features].fillna(0)
    filtered_songs[features] = filtered_songs[features].fillna(0)
    print("✅ Filled in missing values for clustering features.")

    # Standardize the features. Keep it balanced. ⚖️
    scaler = StandardScaler()
    X_playlist_scaled = scaler.fit_transform(playlist_df[features].values)
    X_filtered_scaled = scaler.transform(filtered_songs[features].values)
    print("📏 Standardized the features for clustering.")

    # Find the optimal number of clusters using the elbow method. Let’s not overthink it. 🤔
    wcss = []
    k_range = range(2, min(30, len(playlist_df) + 1))  # Can't have more clusters than playlist songs

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_playlist_scaled)
        wcss.append(kmeans.inertia_)

    # Use KneeLocator to find the sweet spot for k
    try:
        knee = KneeLocator(list(k_range), wcss, curve='convex', direction='decreasing')
        optimal_k = knee.knee
        if optimal_k is None:
            optimal_k = min(5, len(playlist_df) // 2)  # Fallback value
    except Exception:
        # If KneeLocator fails, just wing it with a fallback value
        optimal_k = min(5, len(playlist_df) // 2)

    print(f"🎯 Found the optimal number of clusters: {optimal_k}")

    # Run K-means with the chosen number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    playlist_clusters = kmeans.fit_predict(X_playlist_scaled)
    filtered_clusters = kmeans.predict(X_filtered_scaled)

    # Map numeric clusters to letter labels (A, B, C, etc.). Keep it classy. 🅰️🅱️🅾️
    letter_map = {i: chr(65+i) for i in range(optimal_k)}
    playlist_df['cluster'] = [letter_map[c] for c in playlist_clusters]
    filtered_songs['cluster'] = [letter_map[c] for c in filtered_clusters]

    # Add cluster distribution info. Let’s see how the vibes are spread. 📊
    playlist_cluster_counts = playlist_df['cluster'].value_counts().to_dict()

    # Calculate the percentage of playlist songs in each cluster
    total_playlist_songs = len(playlist_df)
    playlist_cluster_percentages = {cluster: (count / total_playlist_songs) * 100 
                                   for cluster, count in playlist_cluster_counts.items()}

    # Add a "cluster_score" to measure how well each song fits the playlist vibes
    filtered_songs['cluster_score'] = filtered_songs['cluster'].apply(
        lambda c: playlist_cluster_percentages.get(c, 0)
    )

    # Print the cluster distribution. Let’s see the vibe breakdown. 🎧
    print("\n📊 Cluster distribution in playlist:")
    for cluster, percentage in sorted(playlist_cluster_percentages.items()):
        print(f"Cluster {cluster}: {percentage:.2f}% ({playlist_cluster_counts[cluster]} songs)")

    return filtered_songs

def run_classification_model(df, playlist_df, threshold=0.60):
    print("Running Classification Model...")
    
    feature_cols = [
        "Danceability", "Energy", "Tempo", "Valence", "Liveness", "Acousticness", 
        "Instrumentalness", "Speechiness", "Loudness"
    ]
    
    # Parse and expand the genre_embedding column
    df_genre_embeddings = df["genre_embedding"].apply(parse_genre_embedding)
    playlist_genre_embeddings = playlist_df["genre_embedding"].apply(parse_genre_embedding)
    genre_embed_cols = [f"genre_dim_{i}" for i in range(16)]
    
    # Expand the genre_embedding into separate columns with named columns
    df_genre_embeddings = pd.DataFrame(df_genre_embeddings.tolist(), 
                                      index=df.index, 
                                      columns=genre_embed_cols)
                                      
    playlist_genre_embeddings = pd.DataFrame(playlist_genre_embeddings.tolist(), 
                                            index=playlist_df.index,
                                            columns=genre_embed_cols)
    
    # Remove the original genre_embedding column
    df = df.drop(columns=["genre_embedding"]) 
    playlist_df = playlist_df.drop(columns=["genre_embedding"])
    
    # Concatenate the expanded genre_embedding columns with the original dataframes
    df = pd.concat([df, df_genre_embeddings], axis=1)
    playlist_df = pd.concat([playlist_df, playlist_genre_embeddings], axis=1)
    
    # Update the feature columns to include the new genre_embedding columns
    feature_cols.extend(genre_embed_cols)
    
    # Clean column names
    df.columns = df.columns.astype(str)
    playlist_df.columns = playlist_df.columns.astype(str)
    
    # Drop missing values
    df = df.dropna(subset=feature_cols)
    playlist_df = playlist_df.dropna(subset=feature_cols)
    
    # Compute the 'Centroid' of the Playlist
    playlist_vector = playlist_df[feature_cols].mean(axis=0).values
    
    # Normalize features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[feature_cols])
    playlist_vector_scaled = scaler.transform([playlist_vector])
    
    # Similarity Scores (Cosine)
    df["similarity_score"] = cosine_similarity(df_scaled, playlist_vector_scaled).flatten()
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",  
        n_estimators=100,  
        learning_rate=0.05,  
        max_depth=5,  
        subsample=0.8,  
        colsample_bytree=0.8  
    )
    x_train = df[feature_cols].values
    y_train = df["similarity_score"].values
    
    xgb_model.fit(x_train, y_train)
    
    # Predict
    df["fit_probability"] = xgb_model.predict(x_train)
    
    # Filter songs
    above_thresh_df = df[df["fit_probability"] >= threshold].copy()
    
    # Summary
    print(f"Classification Results:")
    print(f"Songs Retained: {len(above_thresh_df)} ({(len(above_thresh_df)/len(df))*100:.2f}%)")
    print(f"Songs Discarded: {len(df) - len(above_thresh_df)}\n")
    
    return above_thresh_df, playlist_df, feature_cols

def run_regression_model(filtered_songs, playlist_df, features):
    print("Running Regression Model...")
    
    # Fill missing values
    playlist_df[features] = playlist_df[features].fillna(0)
    filtered_songs[features] = filtered_songs[features].fillna(0)
    
    playlist_df['fit_score'] = 1  # Songs in the playlist have a perfect fit score
    filtered_songs['fit_score'] = 0  # The rest have a score of 0
    
    # Combine datasets for training
    combined_df = pd.concat([playlist_df, filtered_songs], ignore_index=True)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(combined_df[features])
    y = combined_df['fit_score']
    
    # Train Ridge regression model
    ridge = Ridge(alpha=1.0)  # Regularization strength
    ridge.fit(X, y)
    
    # Predict scores for all songs
    filtered_songs_scaled = scaler.transform(filtered_songs[features])
    filtered_songs['ridge_fitness_score'] = ridge.predict(filtered_songs_scaled)
    
    # Normalize scores to 0-100
    min_score = filtered_songs['ridge_fitness_score'].min()
    max_score = filtered_songs['ridge_fitness_score'].max()
    filtered_songs['fitness_score_normalized'] = ((filtered_songs['ridge_fitness_score'] - min_score) / 
                                                  (max_score - min_score) * 100)
    
    # Print summary
    print("\nFitness Score Summary:")
    print(f"Average Fitness Score: {filtered_songs['fitness_score_normalized'].mean():.2f}")
    print(f"Highest Fitness Score: {filtered_songs['fitness_score_normalized'].max():.2f}")
    print(f"Lowest Fitness Score: {filtered_songs['fitness_score_normalized'].min():.2f}")
    
    # Print top 20 tracks
    print("\nTop 20 Tracks by Ridge Regression:")
    top_20_tracks = filtered_songs.sort_values('fitness_score_normalized', ascending=False).head(20)
    print(top_20_tracks[['Track Name', 'Artist Name(s)', 'fitness_score_normalized']])
    
    return filtered_songs

def run_glm_model(filtered_songs, playlist_df):
    print("Running GLM Metadata Model...")
    
    # Focus on metadata/categorical features
    numerical_features = ["Duration (ms)", "Popularity"]
    categorical_features = ["Artist Name(s)", "Album Name", "Record Label", "Release Date"]
    
    # Filter to columns that exist in both datasets
    actual_numerical = [f for f in numerical_features if f in filtered_songs.columns and f in playlist_df.columns]
    actual_categorical = [f for f in categorical_features if f in filtered_songs.columns and f in playlist_df.columns]
    
    print(f"Using {len(actual_numerical)} numerical features: {actual_numerical}")
    print(f"Using {len(actual_categorical)} categorical features: {actual_categorical}")
    
    # Add binary target for classification
    playlist_df['is_playlist_song'] = 1  # Songs in playlist (positive examples)
    filtered_songs['is_playlist_song'] = 0  # Other songs (negative examples)
    
    # Clean numerical features
    for feature in actual_numerical:
        # Convert to numeric and handle missing values
        filtered_songs[feature] = pd.to_numeric(filtered_songs[feature], errors='coerce')
        playlist_df[feature] = pd.to_numeric(playlist_df[feature], errors='coerce')
        
        # Fill missing with median
        median = pd.concat([filtered_songs[feature], playlist_df[feature]]).median()
        filtered_songs[feature] = filtered_songs[feature].fillna(median)
        playlist_df[feature] = playlist_df[feature].fillna(median)
    
    # Clean categorical features
    for feature in actual_categorical:
        # Convert all to string
        filtered_songs[feature] = filtered_songs[feature].fillna("Unknown").astype(str)
        playlist_df[feature] = playlist_df[feature].fillna("Unknown").astype(str)
        
        # Get values actually in the playlist
        playlist_values = set(playlist_df[feature].unique())
        
        # Mark values either as themselves (if in playlist) or as "Other"
        filtered_songs[feature] = filtered_songs[feature].apply(lambda x: x if x in playlist_values else "Other")
    
    # Standardize numerical features
    scaler = StandardScaler()
    if actual_numerical:
        # Fit scaler on combined data
        all_numerical = pd.concat([filtered_songs[actual_numerical], playlist_df[actual_numerical]])
        scaler.fit(all_numerical)
        
        # Transform both datasets
        filtered_songs[actual_numerical] = scaler.transform(filtered_songs[actual_numerical])
        playlist_df[actual_numerical] = scaler.transform(playlist_df[actual_numerical])
    
    # For better balance, upsample the playlist data
    upsampled_playlist = pd.concat([playlist_df] * 5)  # Repeat playlist 5 times (more weight on playlist)
    
    # Prepare combined dataset for one-hot encoding
    combined_df = pd.concat([filtered_songs, upsampled_playlist], ignore_index=True)
    feature_matrices = []
    
    # Add numerical features
    if actual_numerical:
        feature_matrices.append(combined_df[actual_numerical])
    
    # Create additional metadata match features
    for feature in actual_categorical:
        # One-hot encode
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', min_frequency=2)
        encoded = encoder.fit_transform(combined_df[[feature]])
        feature_names = [f"{feature}_{val}" for val in encoder.categories_[0]]
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=combined_df.index)
        feature_matrices.append(encoded_df)
        
        # Also add a binary feature for "is in playlist"
        playlist_values = set(playlist_df[feature].unique())
        combined_df[f'{feature}_in_playlist'] = combined_df[feature].apply(lambda x: 1 if x in playlist_values else 0)
        feature_matrices.append(combined_df[[f'{feature}_in_playlist']])
    
    # Combine all feature matrices
    X_combined = pd.concat(feature_matrices, axis=1)
    
    # Add constant term for intercept
    X_combined = sm.add_constant(X_combined)
    
    # Split back into training datasets
    X = X_combined 
    y = combined_df['is_playlist_song']
    
    try:
        # Train GLM model with light regularization
        glm_model = sm.GLM(y, X, family=sm.families.Binomial()).fit_regularized(
            alpha=0.01,  # Light regularization to allow categorical features to have effect
            L1_wt=0,     # Pure L2 regularization 
            maxiter=300  # More iterations for convergence
        )
        
        # Generate predictions
        X_pred = X_combined.iloc[:len(filtered_songs)]
        filtered_songs['glm_probability'] = glm_model.predict(X_pred)
        
    except Exception as e:
        print(f"Error in GLM model: {e}")
        print("Falling back to metadata matching approach...")
        
        # Define importance weights for each feature
        feature_weights = {
            "Artist Name(s)": 0.5,
            "Album Name": 0.2,
            "Record Label": 0.2,
            "Release Date": 0.1
        }
        
        match_scores = []
        for _, row in filtered_songs.iterrows():
            weighted_score = 0
            total_weight = 0
            
            for feature in actual_categorical:
                if feature in feature_weights:
                    weight = feature_weights[feature]
                    total_weight += weight
                    
                    # Check if this value appears in the playlist
                    if row[feature] in playlist_df[feature].values:
                        weighted_score += weight
            
            # Normalize by total weight
            if total_weight > 0:
                match_scores.append(weighted_score / total_weight)
            else:
                match_scores.append(0)
        
        filtered_songs['glm_probability'] = np.array(match_scores)
    
    # Normalize GLM scores to 0-100 range
    min_score = filtered_songs['glm_probability'].min()
    max_score = filtered_songs['glm_probability'].max()
    
    # Ensure we have a reasonable range
    if max_score - min_score < 0.01:
        print("Warning: GLM score range too narrow, applying default scaling")
        filtered_songs['glm_score_normalized'] = filtered_songs['glm_probability'] * 100
    else:
        filtered_songs['glm_score_normalized'] = ((filtered_songs['glm_probability'] - min_score) / 
                                                (max_score - min_score) * 100)
    
    # Print GLM model results
    print("\nGLM Model Results:")
    print(f"Average GLM Score: {filtered_songs['glm_score_normalized'].mean():.2f}")
    print(f"Highest GLM Score: {filtered_songs['glm_score_normalized'].max():.2f}")
    print(f"Lowest GLM Score: {filtered_songs['glm_score_normalized'].min():.2f}")
    
    # Print top tracks by GLM
    print("\nTop 20 Tracks by GLM:")
    top_glm_tracks = filtered_songs.sort_values('glm_score_normalized', ascending=False).head(20)
    print(top_glm_tracks[['Track Name', 'Artist Name(s)', 'glm_score_normalized']])
    
    return filtered_songs

def ensure_playlist_embeddings(playlist_df):
    """
    Check if playlist tracks have genre embeddings and generate them if missing.
    Only processes the playlist, not the main dataset.
    """
    print("Checking playlist genre embeddings...")
    
    # Skip if already processed
    if 'genre_embedding' not in playlist_df.columns or playlist_df['genre_embedding'].isna().any():
        print("Missing genre embeddings in playlist. Processing...")
        
        # Check if we have the genres column in the playlist
        if 'Genres' not in playlist_df.columns:
            print("Warning: No 'Genres' column found in playlist. Creating placeholder embeddings.")
            # Add zero vectors as placeholders - model will rely more on audio features
            playlist_df['genre_embedding'] = [np.zeros(16).tolist()] * len(playlist_df)
            return playlist_df
            
        # Load genre embeddings
        try:
            genre_embed_path = 'json/genre_embeddings.json'
            
            if not os.path.exists(genre_embed_path):
                print(f"Warning: Genre embeddings file not found at {genre_embed_path}")
                print("Using zero vectors as fallback.")
                playlist_df['genre_embedding'] = [np.zeros(16).tolist()] * len(playlist_df)
                return playlist_df
                
            with open(genre_embed_path, "r") as f:
                genre_embeddings = json.load(f)
                
            # Convert embeddings to lowercase keys for matching
            genre_embeddings = {k.lower().strip(): np.array(v) for k, v in genre_embeddings.items()}
            
            # Define helper functions for genre processing (simplified versions from embed_genre_mapping.py)
            def get_genre_list(genre_str):
                """Convert comma-separated genre string to list"""
                if isinstance(genre_str, str):
                    return [g.strip().lower() for g in genre_str.split(",")]
                return []
                
            def clean_genre_name(genre):
                """Clean up genre name by removing generic words"""
                generic_words = {"music", "scene", "band"}
                words = genre.lower().strip().split()
                cleaned_words = [word for word in words not in generic_words]
                return " ".join(cleaned_words)
                
            def find_closest_genre(genre, embeddings):
                """Find closest matching genre in embeddings"""
                if not isinstance(genre, str) or not genre.strip():
                    return None
                    
                genre_clean = clean_genre_name(genre)
                genre_with_dash = genre_clean.replace(" ", "-")
                genre_without_dash = genre_clean.replace("-", " ")
                
                # Try exact matches first
                for variant in [genre_clean, genre_with_dash, genre_without_dash]:
                    if variant in embeddings:
                        return variant
                        
                # Try substring matches
                subgenre_matches = [g for g in embeddings if genre_clean in g]
                if subgenre_matches:
                    return subgenre_matches[0]
                    
                return None
                
            def get_track_embedding(genre_list, embeddings):
                """Compute track embedding from genre list"""
                if not isinstance(genre_list, list) or not genre_list:
                    return np.zeros(16)
                    
                valid_embeddings = []
                for genre in genre_list:
                    matched_genre = find_closest_genre(genre, embeddings)
                    if matched_genre and matched_genre in embeddings:
                        valid_embeddings.append(embeddings[matched_genre])
                        
                if not valid_embeddings:
                    return np.zeros(16)
                    
                return np.mean(valid_embeddings, axis=0)
            
            # Process each track in the playlist
            playlist_df['genres_list'] = playlist_df['Genres'].fillna("").apply(get_genre_list)
            
            # Generate embeddings
            embedding_column = []
            for _, row in playlist_df.iterrows():
                embedding = get_track_embedding(row['genres_list'], genre_embeddings)
                embedding_column.append(embedding.tolist())
                
            playlist_df['genre_embedding'] = embedding_column
            
            # Clean up temporary column
            if 'genres_list' in playlist_df.columns:
                playlist_df = playlist_df.drop('genres_list', axis=1)
                
            print(f"Successfully generated embeddings for {len(playlist_df)} playlist tracks")
            
        except Exception as e:
            print(f"Error processing genre embeddings: {e}")
            print("Using zero vectors as fallback.")
            playlist_df['genre_embedding'] = [np.zeros(16).tolist()] * len(playlist_df)
    else:
        print("All playlist tracks have genre embeddings")
        
    return playlist_df

def debug_embeddings(df):
    # Print the first few parsed embeddings to see if they're zeroed
    print("Debugging parsed embeddings:")
    sample = df['genre_embedding'].head(5).tolist()
    for i, emb in enumerate(sample):
        print(f"Row {i} -> {emb}")

def main(dataset_path, playlist_path, threshold=0.60):
    print(f"Loading dataset from: {dataset_path}")
    print(f"Loading playlist from: {playlist_path}")
    
    # Create output path based on input paths
    output_path = os.path.join(os.path.dirname(dataset_path), "heaterfy_output.csv")
    
    # Load data
    df = pd.read_csv(dataset_path)
    playlist_df = pd.read_csv(playlist_path)
    
    print(f"Original dataset size: {len(df)} songs")
    print(f"Playlist size: {len(playlist_df)} songs")
    
    # Clean both datasets
    df = clean_dataset(df)
    playlist_df = clean_dataset(playlist_df)
    
    print(f"Main dataset after cleaning: {len(df)} songs")
    print(f"Playlist after cleaning: {len(playlist_df)} songs")
    
    # Check and fix any missing genre embeddings in the playlist
    playlist_df = ensure_playlist_embeddings(playlist_df)
    
    # Remove songs that are already in the playlist from the dataset
    print("\n--- Removing songs already in playlist ---")
    
    # Create matching keys based on track and artist
    if 'Track Name' in df.columns and 'Artist Name(s)' in df.columns:
        # Create unique identifier for each song
        df['track_artist_key'] = df['Track Name'].str.lower() + '|' + df['Artist Name(s)'].str.lower()
        playlist_df['track_artist_key'] = playlist_df['Track Name'].str.lower() + '|' + playlist_df['Artist Name(s)'].str.lower()
        
        # Get list of keys in the playlist
        playlist_keys = set(playlist_df['track_artist_key'])
        
        # Filter out songs that are already in the playlist
        original_size = len(df)
        df = df[~df['track_artist_key'].isin(playlist_keys)]
        
        # Remove the temporary key column
        df = df.drop(columns=['track_artist_key'])
        playlist_df = playlist_df.drop(columns=['track_artist_key'])
        
        print(f"Removed {original_size - len(df)} songs that are already in the playlist")
        print(f"Dataset size after filtering: {len(df)} songs")
    else:
        print("Warning: Could not filter playlist songs due to missing Track Name or Artist Name columns")
    
    # Step 1: INITIAL FILTERING with XGBoost
    print("\n--- PHASE 1: Initial Filtering ---")
    filtered_songs, playlist_df, audio_features = run_classification_model(df, playlist_df, threshold)
    print(f"Songs remaining after filtering: {len(filtered_songs)} ({len(filtered_songs)/len(df)*100:.1f}%)")
    
    # For rest of the pipeline, we work only with the filtered dataset
    
    # Step 2: AUDIO FEATURE SCORING with Ridge Regression
    print("\n--- PHASE 2: Audio Feature Scoring ---")
    regression_features = [
        "Danceability", "Energy", "Tempo", "Valence", "Liveness", "Acousticness", 
        "Instrumentalness", "Speechiness", "Loudness"
    ]
    # Check which features are available
    available_features = [f for f in regression_features if f in filtered_songs.columns]
    if len(available_features) < len(regression_features):
        print(f"Warning: Using only {len(available_features)}/{len(regression_features)} audio features")
        print(f"Missing: {set(regression_features) - set(available_features)}")
    
    filtered_songs = run_regression_model(filtered_songs, playlist_df, available_features)
    
    # Step 3: METADATA SCORING with GLM
    print("\n--- PHASE 3: Metadata Scoring ---")
    filtered_songs = run_glm_model(filtered_songs, playlist_df)
    
    # Step 4: CLUSTER ANALYSIS with K-means
    print("\n--- PHASE 4: Cluster Analysis ---")
    filtered_songs = run_kmeans_clustering(filtered_songs, playlist_df)
    
    # Ensure all scores are properly normalized to 0-100 range
    for score_col in ['fitness_score_normalized', 'glm_score_normalized', 'cluster_score']:
        if score_col in filtered_songs.columns:
            # Check if normalization is needed
            min_val = filtered_songs[score_col].min()
            max_val = filtered_songs[score_col].max()
            
            if max_val > 100 or min_val < 0 or (max_val - min_val < 0.01):
                print(f"Re-normalizing {score_col} (range: {min_val:.2f}-{max_val:.2f})")
                filtered_songs[score_col] = 100 * (filtered_songs[score_col] - min_val) / (max_val - min_val)
    
    # Step 5: COMBINE SCORES into final recommendation score
    print("\n--- PHASE 5: Final Score Calculation ---")
    
    # Adjust weights based on which models were successfully run
    weights = {}
    available_scores = []
    
    if 'fitness_score_normalized' in filtered_songs.columns:
        weights['fitness_score_normalized'] = 0.5  # Audio features (Ridge)
        available_scores.append('fitness_score_normalized')
    
    if 'glm_score_normalized' in filtered_songs.columns:
        weights['glm_score_normalized'] = 0.3      # Metadata (GLM)
        available_scores.append('glm_score_normalized')
    
    if 'cluster_score' in filtered_songs.columns:
        weights['cluster_score'] = 0.2             # Cluster similarity
        available_scores.append('cluster_score')
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    print(f"Combining scores with weights: {weights}")
    
    # Calculate weighted average
    filtered_songs['combined_score'] = sum(
        filtered_songs[score] * weight for score, weight in weights.items()
    )
    
    # Final sort by combined score
    filtered_songs = filtered_songs.sort_values('combined_score', ascending=False)
    
    # Print final results summary
    print("\n=== FINAL RESULTS ===")
    print(f"Total songs recommended: {len(filtered_songs)}")
    print(f"Average Combined Score: {filtered_songs['combined_score'].mean():.2f}")
    print(f"Score Range: {filtered_songs['combined_score'].min():.2f} - {filtered_songs['combined_score'].max():.2f}")
    
    # Display score distributions
    print("\nScore Distribution:")
    for score in available_scores + ['combined_score']:
        bins = [0, 25, 50, 75, 100]
        hist = pd.cut(filtered_songs[score], bins=bins).value_counts().sort_index()
        print(f"\n{score}:")
        for bin_range, count in hist.items():
            print(f"  {bin_range}: {count} songs ({count/len(filtered_songs)*100:.1f}%)")
    
    # Display top recommendations
    print("\nTop 20 Tracks Overall:")
    top_combined = filtered_songs.head(20)
    
    # Determine which columns to show
    display_cols = ['Track Name', 'Artist Name(s)', 'combined_score']
    display_cols.extend([c for c in available_scores if c in filtered_songs.columns])
    if 'cluster' in filtered_songs.columns:
        display_cols.append('cluster')
    
    print(top_combined[display_cols])
    
    # Save the final results
    filtered_songs.to_csv(output_path, index=False)
    print(f"\nFinal results saved to: {output_path}")
    
    return filtered_songs

if __name__ == "__main__":
    dataset_path = "datasets/heatify_catalogue.csv"  # Replace with actual path
    playlist_path = "spotify_rips/afterhours.csv"  # Replace with actual path
    
    results = main(dataset_path, playlist_path)
    debug_embeddings(df)