import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

"""Unsupervised Model: Cosine Similarity"""

# Set display options so we can see all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 50)

# Load the filtered songs from the classification model
filtered_songs = pd.read_csv("C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\datasets\\afters_above_threshold_xgb.csv")

# Load the user playlist
playlist_df = pd.read_csv("C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\spotify_rips\\afterhours_embedded.csv")

# Add more?
features = [
    "Danceability", "Energy", "Tempo", "Valence", "Liveness", "Acousticness", 
    "Instrumentalness", "Speechiness", "Loudness"
]

# Check for missing values in features and fill with zeros if any
playlist_df[features] = playlist_df[features].fillna(0)
filtered_songs[features] = filtered_songs[features].fillna(0)

# Compute the centroid of the playlist features
playlist_centroid = playlist_df[features].mean().values.reshape(1, -1)

# Scale
scaler = StandardScaler()
filtered_features_scaled = scaler.fit_transform(filtered_songs[features])
playlist_centroid_scaled = scaler.transform(playlist_centroid)

# Compute similaritys (Cosine Similarity)
similarities = cosine_similarity(filtered_features_scaled, playlist_centroid_scaled)
filtered_songs['cosine_fitness_score'] = similarities.flatten()

# Normalize scores from 0-100
min_score = filtered_songs['cosine_fitness_score'].min()
max_score = filtered_songs['cosine_fitness_score'].max()
score_range = max_score - min_score

filtered_songs['fitness_score_cosine'] = ((filtered_songs['cosine_fitness_score'] - min_score) / 
                                          score_range * 100)

filtered_songs['fitness_score_cosine'] = filtered_songs['fitness_score_cosine'].fillna(0)

# Save results
filtered_songs_scored = filtered_songs.sort_values('fitness_score_cosine', ascending=False)
filtered_songs_scored.to_csv("C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\datasets\\afters_scored_ridge.csv", index=False)

# Output summary
print("\nFitness Score Summary:")
print(f"Average Fitness Score: {filtered_songs['fitness_score_cosine'].mean():.2f}")
print(f"Highest Fitness Score: {filtered_songs['fitness_score_cosine'].max():.2f}")
print(f"Lowest Fitness Score: {filtered_songs['fitness_score_cosine'].min():.2f}")

# Print top tracks
print("\nTop 20 Tracks by Fitness Score:")
columns_to_display = ['Track Name', 'Artist Name(s)', 'fitness_score_cosine']
top_20 = filtered_songs_scored[columns_to_display].head(20)
print(top_20)

# Print bottom 5 tracks by Fitness Score
print("\nBottom 5 Tracks by Fitness Score:")
bottom_5 = filtered_songs_scored[columns_to_display].tail(5)
print(bottom_5)

# Measure correlation between each feature
feature_importance = []
for feature in features:
    correlation = np.corrcoef(filtered_songs[feature], filtered_songs['cosine_fitness_score'])[0, 1]
    feature_importance.append((feature, correlation))

feature_importance_df = pd.DataFrame(feature_importance, columns=['Feature', 'Correlation'])
feature_importance_df = feature_importance_df.sort_values('Correlation', key=abs, ascending=False)

print("\nFeature Importance (by correlation with fitness score):")
print(feature_importance_df.head(8))