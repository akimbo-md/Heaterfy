"""
K-means clustering for grouping similar tracks
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

def run_kmeans_clustering(filtered_songs, playlist_df):
    print("Running K-means clustering...")

    genre_features = [f"genre_dim_{i}" for i in range(16)]
    audio_features = [
        'Danceability', 'Energy', 'Loudness', 'Speechiness',
        'Acousticness', 'Instrumentalness', 'Liveness', 'Valence',
        'Tempo'
    ]
    available_audio_features = [f for f in audio_features if f in filtered_songs.columns and f in playlist_df.columns]
    available_genre_features = [f for f in genre_features if f in filtered_songs.columns and f in playlist_df.columns]
    features = available_audio_features + available_genre_features # debug

    # Fill missing values with zero
    playlist_df[features] = playlist_df[features].fillna(0)
    filtered_songs[features] = filtered_songs[features].fillna(0)
    print("Filled in missing values for clustering features.")

    # Scale
    scaler = StandardScaler()
    X_playlist_scaled = scaler.fit_transform(playlist_df[features].values)
    X_filtered_scaled = scaler.transform(filtered_songs[features].values)
    print("📏 Standardized the features for clustering.")

    # Calculate WCSS for elbow method
    wcss_values = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_playlist_scaled)
        wcss_values.append(kmeans.inertia_)

    # Find optimal k using knee/elbow method
    kneedle = KneeLocator(
        k_range, wcss_values, S=1.0, curve="convex", direction="decreasing"
    )
    optimal_k = kneedle.elbow

    # If no clear elbow is found, use a default of 4
    optimal_k = optimal_k if optimal_k else 4

    print(f"Found the optimal number of clusters: {optimal_k}")

    # Run K-means
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=42)
    playlist_clusters = kmeans_model.fit_predict(X_playlist_scaled)
    filtered_clusters = kmeans_model.predict(X_filtered_scaled)

    # Map clusters to letter labels
    letter_map = {i: chr(65+i) for i in range(optimal_k)}
    playlist_df['cluster'] = [letter_map[c] for c in playlist_clusters]
    filtered_songs['cluster'] = [letter_map[c] for c in filtered_clusters]
    playlist_cluster_counts = playlist_df['cluster'].value_counts().to_dict()

    # Display cluster distribution
    total_playlist_songs = len(playlist_df)
    playlist_cluster_percentages = {cluster: (count / total_playlist_songs) * 100 
                                   for cluster, count in playlist_cluster_counts.items()}

    # Cluster score?
    filtered_songs['cluster_score'] = filtered_songs['cluster'].apply(
        lambda c: playlist_cluster_percentages.get(c, 0)
    )

    # Summary
    print("\nCluster distribution in playlist:")
    for cluster, percentage in sorted(playlist_cluster_percentages.items()):
        print(f"Cluster {cluster}: {percentage:.2f}% ({playlist_cluster_counts[cluster]} tracks)")

    # Also return clustering metadata
    clustering_data = {
        "wcss": wcss_values,
        "optimal_k": optimal_k
    }

    return filtered_songs, playlist_df, clustering_data  # Return both DataFrames and clustering metadata