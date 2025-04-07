import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from kneed import KneeLocator

'''
This program performs k-means clustering by a user-specified number of iterations.
The variable 'k_values' is what users should modify to change 'k'. 
To change the criteria for clustering, modify the 'features' list. 
'''

def parse_genre_embedding(embedding_str):
    try:
        # Replace commas with spaces to ensure a consistent delimiter
        cleaned = embedding_str.strip("[]").replace(",", " ")
        return np.fromstring(cleaned, sep=' ')
    except:
        return np.zeros(16)  # Assuming 16D vector

def main():
    input_file = 'datasets/trevor_glm_filtered_dataset.csv'
    playlist_file = 'spotify_rips/cmpt_455_-_trevors_heaters_embedded.csv'
    df = pd.read_csv(input_file)
    playlist_df = pd.read_csv(playlist_file)

    # # Check for missing values in each column if your getting an error that mentions 'NaNs'
    # print(df.isnull().sum())
    
    # Parse and expand the genre_embedding column for the playlist file
    playlist_genre_embeddings = playlist_df["genre_embedding"].apply(parse_genre_embedding)

    # Create column names for the genre embeddings
    genre_embed_cols = [f"genre_dim_{i}" for i in range(16)]  # Create names like 'genre_dim_0', 'genre_dim_1', etc.

    # Expand the genre_embedding into separate columns with named columns
    playlist_genre_embeddings = pd.DataFrame(playlist_genre_embeddings.tolist(), 
                                             index=playlist_df.index,
                                             columns=genre_embed_cols)

    # Concatenate the expanded genre_embedding columns with the original playlist dataframe
    playlist_df = pd.concat([playlist_df, playlist_genre_embeddings], axis=1)
    
    genre_features = genre_embed_cols

    audio_features = [
        'Danceability',
        'Energy',
        'Key',
        'Loudness',
        'Mode',
        'Speechiness',
        'Acousticness',
        'Instrumentalness',
        'Liveness',
        'Valence',
        'Tempo',
        'Popularity' 
    ]

    features = genre_features + audio_features
    print("Beginning k-means clustering with initial k value of 2...")
    
    # Filter out rows with missing values (or fill them) for the given features
    playlist_df[features] = playlist_df[features].fillna(0)
    df[features] = df[features].fillna(0)

    # Extract the feature vectors that we will actually use for clustering
    X_full = df[features]
    X_playlist = playlist_df[features]

    # Standardize the features that we're using (idk how much this affects the results)
    scaler = StandardScaler()
    X_playlist = scaler.fit_transform(playlist_df[features])
    X_full = scaler.transform(df[features])

    wcss = [] # We will use the values in this list to create an elbow graph
    k_values = range(2,30) # This is what we adjust to cycle through different k-means
   
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=17)
        kmeans.fit(X_playlist)
        wcss.append(kmeans.inertia_) # This is the WCSS for the current iteration of k

    print("Finished going through all of the k values!")

    # Plot the elbow graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()
    
    # Automatically detect the elbow point
    knee = KneeLocator(k_values, wcss, curve='convex', direction='decreasing')
    optimal_k = knee.knee
    if optimal_k is None:
        optimal_k = 5  # fallback value if knee is undetected
    print(f"Optimal number of clusters detected: {optimal_k}")
    
    # Run k-means clustering on the playlist data using the optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=17)
    playlist_clusters = kmeans.fit_predict(X_playlist)
    full_clusters = kmeans.predict(X_full)
    
    # Map numeric cluster labels to letters
    letter_map = {i: chr(65+i) for i in range(optimal_k)}
    playlist_df['new_cluster_label'] = [letter_map[c] for c in playlist_clusters]
    df['new_cluster_label'] = [letter_map[c] for c in full_clusters]
    
    # Save copies
    playlist_out = 'datasets/trevor_clusters.csv'
    full_out = 'datasets/trevor_full_dataset_clusters.csv'
    playlist_df.to_csv(playlist_out, index=False)
    df.to_csv(full_out, index=False)
    
    print("\nCluster assignment (by letter):")
    print("Playlist clusters distribution:")
    print(playlist_df['new_cluster_label'].value_counts().sort_index())
    print("\nFull dataset clusters distribution:")
    print(df['new_cluster_label'].value_counts().sort_index())

    # PCA for visualization
    pca = PCA(n_components=2)
    playlist_pca = pca.fit_transform(X_playlist)
    full_pca = pca.transform(X_full)

    plt.figure(figsize=(10, 6))
    for label in letter_map.values():
        plt.scatter(playlist_pca[playlist_df['new_cluster_label'] == label, 0], 
                    playlist_pca[playlist_df['new_cluster_label'] == label, 1], 
                    label=label)
    plt.title('PCA of Playlist Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

main()