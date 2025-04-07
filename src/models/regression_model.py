"""
Ridge Regression model for audio feature scoring
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def run_regression_model(filtered_songs, playlist_df, features):
    print("Running Ridge Regression Model...")
    
    # Check if dataset is empty
    if len(filtered_songs) == 0:
        print("ERROR: Empty dataset passed to regression model")
        return filtered_songs
    
    # Check if features are available
    if not all(feature in filtered_songs.columns for feature in features):
        print("ERROR: Not all features available in dataset")
        missing = [f for f in features if f not in filtered_songs.columns]
        print(f"Missing features: {missing}")
        return filtered_songs
    
    # Fill missing values with means
    for feature in features:
        if filtered_songs[feature].isna().any():
            mean_value = filtered_songs[feature].mean()
            filtered_songs[feature] = filtered_songs[feature].fillna(mean_value)
            print(f"Filled {filtered_songs[feature].isna().sum()} missing values in {feature}")
    
    # Fill missing values
    playlist_df[features] = playlist_df[features].fillna(0)
    filtered_songs[features] = filtered_songs[features].fillna(0)
    
    playlist_df['fit_score'] = 1  # Songs in the playlist have a PERFECT fit score
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
    
    # summary
    print("\nFitness Score Summary:")
    print(f"Average Fitness Score: {filtered_songs['fitness_score_normalized'].mean():.2f}")
    print(f"Highest Fitness Score: {filtered_songs['fitness_score_normalized'].max():.2f}")
    print(f"Lowest Fitness Score: {filtered_songs['fitness_score_normalized'].min():.2f}")
    
    # Print top tracks
    print("\nTop 20 Tracks by Ridge Regression:")
    top_20_tracks = filtered_songs.sort_values('fitness_score_normalized', ascending=False).head(20)
    print(top_20_tracks[['Track Name', 'Artist Name(s)', 'fitness_score_normalized']])
    
    return filtered_songs