from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load data
filtered_songs = pd.read_csv("C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\datasets\\trevor_above_threshold_xgb.csv")
playlist_df = pd.read_csv("C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\spotify_rips\\cmpt_455_-_trevors_heaters_embedded.csv")

# We'll only consider audio features in this model
features = [
    "Danceability", "Energy", "Tempo", "Valence", "Liveness", "Acousticness", 
    "Instrumentalness", "Speechiness", "Loudness"
]

# Fill missing values
playlist_df[features] = playlist_df[features].fillna(0)
filtered_songs[features] = filtered_songs[features].fillna(0)

# Create a proxy target variable
playlist_df['fit_score'] = 1  # Songs in the playlist have a perfect fit score
filtered_songs['fit_score'] = 0  # Other songs have a score of 0

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

# Save results
filtered_songs.to_csv("C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\datasets\\trevor_ridge_scored_songs.csv", index=False)

# Print summary
print("\nFitness Score Summary:")
print(f"Average Fitness Score: {filtered_songs['fitness_score_normalized'].mean():.2f}")
print(f"Highest Fitness Score: {filtered_songs['fitness_score_normalized'].max():.2f}")
print(f"Lowest Fitness Score: {filtered_songs['fitness_score_normalized'].min():.2f}")

# Print top 20 tracks
print("\nTop 20 Tracks by Fitness Score:")
top_20_tracks = filtered_songs.sort_values('fitness_score_normalized', ascending=False).head(20)
print(top_20_tracks[['Track Name', 'Artist Name(s)', 'fitness_score_normalized']])

# Print bottom 5 tracks
print("\nBottom 5 Tracks by Fitness Score:")
bottom_5_tracks = filtered_songs.sort_values('fitness_score_normalized', ascending=True).head(5)
print(bottom_5_tracks[['Track Name', 'Artist Name(s)', 'fitness_score_normalized']])