"""
Cosine Similarity Model for Audio Feature Matching
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def run_cosine_similarity(filtered_songs, playlist_df):
    print("Running Cosine Similarity Model...")
    
    features = [
        "Danceability", "Energy", "Tempo", "Valence", "Liveness", "Acousticness", 
        "Instrumentalness", "Speechiness", "Loudness"
    ]
    available_features = [f for f in features if f in filtered_songs.columns and f in playlist_df.columns]
    
    if len(available_features) < 3:
        print(f"Not enough common audio features for cosine similarity: only {len(available_features)} available")
        return filtered_songs
    
    print(f"Using {len(available_features)} audio features: {available_features}")
    
    # Summarize playlist audio features
    print("\n🎵 Playlist Audio Feature Summary:")
    feature_means = playlist_df[available_features].mean()
    for feature, mean_value in feature_means.items():
        if feature == "Loudness":
            trend = "loud" if mean_value > -8 else "quiet" if mean_value > -12 else "medium"
        if feature == "Tempo":
            trend = "fast" if mean_value >= 130 else "slow" if mean_value < 100 else "medium"
        else:
            trend = "high" if mean_value > 0.6 else "low" if mean_value < 0.4 else "neutral"
        print(f"\t> {feature}: {mean_value:.2f} ({trend})")
    
    
    # Fill missing values
    filtered_songs[available_features] = filtered_songs[available_features].fillna(0)
    playlist_df[available_features] = playlist_df[available_features].fillna(0)
    
    # Find playlist centroid
    playlist_centroid = playlist_df[available_features].mean().values.reshape(1, -1)
    
    # Scale
    scaler = StandardScaler()
    scaler.fit(np.vstack([filtered_songs[available_features].values, playlist_centroid]))
    
    filtered_features_scaled = scaler.transform(filtered_songs[available_features])
    playlist_centroid_scaled = scaler.transform(playlist_centroid)
    
    # Compute
    similarities = cosine_similarity(filtered_features_scaled, playlist_centroid_scaled)
    filtered_songs['cosine_similarity'] = similarities.flatten()
    
    # Normalize scores from 0-100
    min_score = filtered_songs['cosine_similarity'].min()
    max_score = filtered_songs['cosine_similarity'].max()
    
    if max_score > min_score:
        filtered_songs['cosine_score_normalized'] = ((filtered_songs['cosine_similarity'] - min_score) / 
                                                (max_score - min_score) * 100)
    else:
        filtered_songs['cosine_score_normalized'] = 50  # Default if all scores are the same
    
    print("\nCosine Similarity Results:")
    print(f"Average Cosine Score: {filtered_songs['cosine_score_normalized'].mean():.2f}")
    print(f"Highest Cosine Score: {filtered_songs['cosine_score_normalized'].max():.2f}")
    print(f"Lowest Cosine Score: {filtered_songs['cosine_score_normalized'].min():.2f}")
    
    print("\nTop 20 Tracks by Cosine Similarity:")
    top_cosine = filtered_songs.sort_values('cosine_score_normalized', ascending=False).head(20)
    print(top_cosine[['Track Name', 'Artist Name(s)', 'cosine_score_normalized']])
    
    return filtered_songs