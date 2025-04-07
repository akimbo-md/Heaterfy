import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

# Set display options so we can see all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', 200)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    # Define the features to be used for the network as specified
    features = [
        'Duration (ms)', 'Popularity', 'Danceability', 'Energy', 'Loudness',
        'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo',
        'Release Date', 'Duration (ms)', 'Record Label', 
        'similarity_score', 'fit_probability', 'ridge_fitness_score', 'fitness_score_glm',
        'cluster_label'
    ]
    # Extend with the 16-dimensional genre embeddings (genre_dim_0 ... genre_dim_15)
    features.extend([f"genre_dim_{i}" for i in range(16)])
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, features

def get_playlist_vector(df, playlist_indices, features):
    # Returns a vector with the mean value for each feature of the playlist songs
    return np.mean(df.iloc[playlist_indices][features].values, axis=0)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, input_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_model(df, features):
    input_size = len(features)
    model = NeuralNetwork(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train = torch.tensor(df[features].values, dtype=torch.float32)
    y_train = torch.tensor(df[features].values, dtype=torch.float32)

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}')

    return model

def recommend_songs(model, df, playlist_vector, features, top_n):
    # Produces a 2D array that represents what feature values the best fitting song should theoretically have
    model.eval()
    playlist_vector = torch.tensor(playlist_vector, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        predicted_song_vector = model(playlist_vector).numpy()
    # Checks how similar the predicted_song_vector is to all the other songs in the dataset, and stores each (similarity) value in a list
    similarities = cosine_similarity(predicted_song_vector, df[features].values)[0]
    # Sort similarities descending and get the indices of the top recommendations
    sorted_indices = np.argsort(similarities)[::-1]
    # Returns the top recommended songs based on the largest similarity values 
    return df.iloc[sorted_indices[:top_n]]

def evaluate_recommendation(df, playlist_indices, recommended_song_index, features):
    '''
    To check if a given song would be considered a heater in a given playlist, we can 
    check to see if the song would fall into the same cluster as the songs in the playlist. 
    '''
    kmeans = KMeans(n_clusters=9, random_state=56)
    df['Cluster'] = kmeans.fit_predict(df[features])
    playlist_cluster = df.iloc[playlist_indices[0]]['Cluster']
    recommended_song_cluster = df.iloc[recommended_song_index]['Cluster']
    return playlist_cluster == recommended_song_cluster

def run_neural_network():
    # Load and preprocess data
    full_df, df_features = load_and_preprocess_data('datasets/afters_full_dataset_clusters.csv')
    playlist_df, playlist_features = load_and_preprocess_data('datasets/afters_clusters.csv')

    #print("Our current playlist to add one more heater to: ")
    #print(playlist_df[['Track Name', 'Artist Name(s)']])
    
    # Use playlist file to compute the playlist vector
    playlist_vector = get_playlist_vector(playlist_df, np.arange(len(playlist_df)), playlist_features)

    # Determine how many recommendations to produce: top 20 (or 5 if playlist has less than 20 songs)
    top_n = 5 if len(playlist_df) < 20 else 20

    # Train the model
    model = train_model(full_df, df_features)

    # Recommend songs (top_n recommendations)
    recommended_songs = recommend_songs(model, full_df, playlist_vector, df_features, top_n)
    print("Recommended Songs:")
    print(recommended_songs[['Track Name', 'Artist Name(s)']])
    
    # (Optional) You can still evaluate recommendation quality via clustering if desired.
    # For example:
    # if evaluate_recommendation(full_df, np.arange(len(playlist_df)), recommended_songs.index[0], df_features):
    #     print("The recommended song fits in with the playlist!")
    # else:
    #     print("The recommended song does not fit in with the playlist.")

if __name__ == "__main__":
    run_neural_network()