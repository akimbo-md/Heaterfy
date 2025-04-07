"""
Generalized Linear Model for metadata scoring
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import statsmodels.api as sm

def run_glm_model(filtered_songs, playlist_df):
    print("Running GLM Metadata Model...")

    # Copy
    filtered_songs['original_artist'] = filtered_songs['Artist Name(s)'].copy()
    filtered_songs['original_album'] = filtered_songs['Album Name'].copy() if 'Album Name' in filtered_songs.columns else None
    
    # This model will focus on metadata and provide a boost in score for matching songs
    numerical_features = ["Duration (ms)", "Popularity"]
    categorical_features = ["Artist Name(s)", "Album Name", "Record Label", "Release Date"]
    
    # Feature check
    actual_numerical = [f for f in numerical_features if f in filtered_songs.columns and f in playlist_df.columns]
    actual_categorical = [f for f in categorical_features if f in filtered_songs.columns and f in playlist_df.columns]
    print(f"Using {len(actual_numerical)} numerical features: {actual_numerical}")
    print(f"Using {len(actual_categorical)} categorical features: {actual_categorical}")
    
    # Playlist songs are PERFECT
    playlist_df['is_playlist_song'] = 1
    filtered_songs['is_playlist_song'] = 0
    
    # Clean numerical features
    for feature in actual_numerical:
        filtered_songs[feature] = pd.to_numeric(filtered_songs[feature], errors='coerce')
        playlist_df[feature] = pd.to_numeric(playlist_df[feature], errors='coerce')
        
        # Fill missing with median
        median = pd.concat([filtered_songs[feature], playlist_df[feature]]).median()
        filtered_songs[feature] = filtered_songs[feature].fillna(median)
        playlist_df[feature] = playlist_df[feature].fillna(median)
    
    # Clean categorical features
    for feature in actual_categorical:
        filtered_songs[feature] = filtered_songs[feature].fillna("Unknown").astype(str)
        playlist_df[feature] = playlist_df[feature].fillna("Unknown").astype(str)
        
        # Get values actually in the playlist
        playlist_values = set(playlist_df[feature].unique())
        
        # Mark values either as themselves if in playlist or ?????
        filtered_songs[feature] = filtered_songs[feature].apply(lambda x: x if x in playlist_values else "????")
    
    # Scale
    scaler = StandardScaler()
    if actual_numerical:
        all_numerical = pd.concat([filtered_songs[actual_numerical], playlist_df[actual_numerical]])
        scaler.fit(all_numerical)
        filtered_songs[actual_numerical] = scaler.transform(filtered_songs[actual_numerical])
        playlist_df[actual_numerical] = scaler.transform(playlist_df[actual_numerical])

    upsampled_playlist = pd.concat([playlist_df] * 5)  # Make the playlist 5x more powerful
    
    # One hot encode
    combined_df = pd.concat([filtered_songs, upsampled_playlist], ignore_index=True)
    feature_matrices = []
    
    # Add numerical features
    if actual_numerical:
        feature_matrices.append(combined_df[actual_numerical])
    
    # Metadata match features
    for feature in actual_categorical:
        # One-hot encode
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', min_frequency=2)
        encoded = encoder.fit_transform(combined_df[[feature]])
        feature_names = [f"{feature}_{val}" for val in encoder.categories_[0]]
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=combined_df.index)
        feature_matrices.append(encoded_df)
        
        # Also add binary feature for songs already in playlist
        playlist_values = set(playlist_df[feature].unique())
        combined_df[f'{feature}_in_playlist'] = combined_df[feature].apply(lambda x: 1 if x in playlist_values else 0)
        feature_matrices.append(combined_df[[f'{feature}_in_playlist']])
    
    # Combine all features
    X_combined = pd.concat(feature_matrices, axis=1)
    X_combined = sm.add_constant(X_combined)
    
    # Spit back into training datasets
    X = X_combined 
    y = combined_df['is_playlist_song']
    
    # TRAINING
    try:
        glm_model = sm.GLM(y, X, family=sm.families.Binomial()).fit_regularized(
            alpha=0.01,
            L1_wt=0, # ridge only
            maxiter=300
        )
        
        # Generate predictions
        X_pred = X_combined.iloc[:len(filtered_songs)]
        filtered_songs['glm_probability'] = glm_model.predict(X_pred)
        
    except Exception as e:
        print(f"Error in GLM model: {e}")
        print("Falling back to metadata matching...")
        
        # Importance bias
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
                    
                    # Check if appears in the playlist
                    if row[feature] in playlist_df[feature].values:
                        weighted_score += weight
            
            # Match by total weight
            if total_weight > 0:
                match_scores.append(weighted_score / total_weight)
            else:
                match_scores.append(0)
        
        filtered_songs['glm_probability'] = np.array(match_scores)
    
    # 0-100 range
    min_score = filtered_songs['glm_probability'].min()
    max_score = filtered_songs['glm_probability'].max()
    if max_score - min_score < 0.01:
        print("GLM score range too narrow, applying default scaling")
        filtered_songs['glm_score_normalized'] = filtered_songs['glm_probability'] * 100
    else:
        filtered_songs['glm_score_normalized'] = ((filtered_songs['glm_probability'] - min_score) / 
                                                (max_score - min_score) * 100)
    
    # Fill missing scores with 0
    filtered_songs['glm_score_normalized'] = filtered_songs['glm_score_normalized'].fillna(0)
    
    # Summary
    print("\nGLM Model Results:")
    print(f"Average GLM Score: {filtered_songs['glm_score_normalized'].mean():.2f}")
    print(f"Highest GLM Score: {filtered_songs['glm_score_normalized'].max():.2f}")
    print(f"Lowest GLM Score: {filtered_songs['glm_score_normalized'].min():.2f}")
    
    print("\nTop 20 Tracks by GLM:")
    top_glm_tracks = filtered_songs.sort_values('glm_score_normalized', ascending=False).head(20)
    print(top_glm_tracks[['Track Name', 'Artist Name(s)', 'glm_score_normalized']])

    filtered_songs['Artist Name(s)'] = filtered_songs['original_artist']
    if 'original_album' in filtered_songs.columns and filtered_songs['original_album'] is not None:
        filtered_songs['Album Name'] = filtered_songs['original_album']
    
    # Clean up temp columns
    filtered_songs = filtered_songs.drop(columns=['original_artist', 'original_album'], errors='ignore')
    
    return filtered_songs