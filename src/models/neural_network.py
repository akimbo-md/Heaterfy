import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationNN(nn.Module):
    def __init__(self, input_size):
        super(RecommendationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = torch.sigmoid(self.fc4(x))
        return x

def run_neural_network(filtered_songs, playlist_df, available_scores):
    try:
        print("\nRunning Neural Network with Feature Emphasis...")
        
        audio_features = [
            "Danceability", "Energy", "Loudness", "Speechiness", 
            "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo"
        ]
        available_audio_features = [f for f in audio_features 
                                  if f in filtered_songs.columns and f in playlist_df.columns]
        genre_features = [col for col in filtered_songs.columns 
                         if col.startswith('genre_dim_') and col in playlist_df.columns]
        
        # Base features
        features = available_audio_features + genre_features
        
        # Metadata scores
        if 'glm_score_normalized' in filtered_songs.columns:
            filtered_songs['has_metadata_match'] = (filtered_songs['glm_score_normalized'] > 0).astype(int)
            playlist_df['has_metadata_match'] = 1
            features.append('has_metadata_match')
        
        # Clusters
        cluster_features = []
        if 'cluster' in filtered_songs.columns and 'cluster' in playlist_df.columns:
            all_clusters = set(filtered_songs['cluster'].unique()) | set(playlist_df['cluster'].unique())
            for cluster in all_clusters:
                cluster_col = f'cluster_{cluster}'
                filtered_songs[cluster_col] = (filtered_songs['cluster'] == cluster).astype(int)
                playlist_df[cluster_col] = (playlist_df['cluster'] == cluster).astype(int)
                cluster_features.append(cluster_col)
            features.extend(cluster_features)
        
        score_features = []
        
        # Cosine similarity
        if 'cosine_score_normalized' in filtered_songs.columns:
            cosine_avg = filtered_songs['cosine_score_normalized'].mean()
            playlist_df['cosine_score_normalized'] = cosine_avg
            features.append('cosine_score_normalized')
            score_features.append('cosine_score_normalized')
            
            # give it more votes by repeating the column
            for i in range(2, 8):
                col_name = f'cosine_score_normalized_{i}'
                filtered_songs[col_name] = filtered_songs['cosine_score_normalized']
                playlist_df[col_name] = cosine_avg
                features.append(col_name)
                score_features.append(col_name)
            
            print(f"\t> Emphasizing cosine similarity (7x weight)")
        
        # Ridge
        if 'fitness_score_normalized' in filtered_songs.columns:
            fitness_avg = filtered_songs['fitness_score_normalized'].mean()
            playlist_df['fitness_score_normalized'] = fitness_avg
            features.append('fitness_score_normalized')
            score_features.append('fitness_score_normalized')
            
            # do it again for another vote
            col_name = 'fitness_score_normalized_2'
            filtered_songs[col_name] = filtered_songs['fitness_score_normalized']
            playlist_df[col_name] = fitness_avg
            features.append(col_name)
            score_features.append(col_name)
            
            print(f"\t> Emphasizing fitness score (2x weight)")
        
        # GLM
        other_scores = []
        if 'glm_score_normalized' in filtered_songs.columns:
            glm_avg = filtered_songs['glm_score_normalized'].mean()
            playlist_df['glm_score_normalized'] = glm_avg
            features.append('glm_score_normalized')
            score_features.append('glm_score_normalized')
            other_scores.append('glm_score_normalized')
        # Clustering
        if 'cluster_score' in filtered_songs.columns:
            cluster_avg = filtered_songs['cluster_score'].mean()
            playlist_df['cluster_score'] = cluster_avg
            features.append('cluster_score')
            score_features.append('cluster_score')
            other_scores.append('cluster_score')
        
        if other_scores:
            print(f"\t> Including other scores: {', '.join(other_scores)} (1x weight)")
        
        # Print feature information
        print(f"Training neural network with {len(features)} features:")
        print(f"\t> {len(available_audio_features)} audio features")
        print(f"\t> {len(genre_features)} genre embedding dimensions")
        print(f"\t> {len(score_features)} model scores (with emphasis)")
        print(f"\t> {len(cluster_features)} cluster indicators")
        
        # Fill missing values
        for feature in features:
            if feature in filtered_songs.columns:
                filtered_songs[feature] = filtered_songs[feature].fillna(0)
            if feature in playlist_df.columns:
                playlist_df[feature] = playlist_df[feature].fillna(0)
        
        # Scale
        scaler = StandardScaler()
        X_combined = pd.concat([filtered_songs[features], playlist_df[features]])
        scaler.fit(X_combined)
        
        X_filtered_scaled = scaler.transform(filtered_songs[features])
        X_playlist_scaled = scaler.transform(playlist_df[features])
        
        # Tell the network the playlist songs are PERFECT
        playlist_vector = np.mean(X_playlist_scaled, axis=0).reshape(1, -1)
        similarities = cosine_similarity(X_filtered_scaled, playlist_vector).flatten()
        
        # Training params
        X_train = torch.tensor(X_filtered_scaled, dtype=torch.float32)
        y_train = torch.tensor(similarities, dtype=torch.float32).view(-1, 1)
        input_size = len(features)
        model = RecommendationNN(input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        n_epochs = 200
        batch_size = min(64, max(1, len(filtered_songs) // 10))
        indices = np.arange(len(filtered_songs))
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        # Training loop
        for epoch in range(n_epochs):
            model.train()
            np.random.shuffle(indices)
            total_loss = 0
            
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(X_train[batch_idx])
                loss = criterion(outputs, y_train[batch_idx])
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Early stop check
            if total_loss < best_loss:
                best_loss = total_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss:.4f}")
            
            # Stop if no improvement
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}, best loss: {best_loss:.4f}")
                break
        
        # Generate recommendations
        model.eval()
        with torch.no_grad():
            scores = model(X_train).numpy().flatten()
        filtered_songs['nn_score'] = scores
        
        # Summary
        print("\nTop 20 Tracks by Neural Network:")
        top_nn = filtered_songs.sort_values('nn_score', ascending=False).head(20)
        display_cols = ['Track Name', 'Artist Name(s)', 'nn_score']
        print(top_nn[display_cols])
        
        # 0-100
        min_score = filtered_songs['nn_score'].min()
        max_score = filtered_songs['nn_score'].max()
        if max_score > min_score:
            filtered_songs['nn_score'] = ((filtered_songs['nn_score'] - min_score) / 
                                        (max_score - min_score) * 100)
        
        return filtered_songs.sort_values('nn_score', ascending=False)
        
    except Exception as e:
        import traceback
        print(f"Error in neural network training: {str(e)}")
        traceback.print_exc()
