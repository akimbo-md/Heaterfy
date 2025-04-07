import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Full Dataset
df = pd.read_csv("C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\datasets\\heatify_full_dataset_updated_embedded.csv")
# User Playlist
playlist_df = pd.read_csv("C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\spotify_rips\\vinyl_soul_in_the_living_room.csv")

feature_cols = [
    "Danceability", "Energy", "Tempo", "Valence", "Liveness", "Acousticness", 
    "Instrumentalness", "Speechiness", "Loudness"
]

# Parse and expand the genre_embedding column
def parse_genre_embedding(embedding_str):
    try:
        # Replace commas with spaces
        cleaned = embedding_str.strip("[]").replace(",", " ")
        return np.fromstring(cleaned, sep=' ')
    except:
        return np.zeros(16)  # Assuming 16D vector

df_genre_embeddings = df["genre_embedding"].apply(parse_genre_embedding)
playlist_genre_embeddings = playlist_df["genre_embedding"].apply(parse_genre_embedding)

# Create column names for the genre embeddings (since it dont like lists)
genre_embed_cols = [f"genre_dim_{i}" for i in range(16)]  # Create names like 'genre_dim_0', 'genre_dim_1', etc.

# Expand the genre_embedding into separate columns with named columns
df_genre_embeddings = pd.DataFrame(df_genre_embeddings.tolist(), 
                                  index=df.index, 
                                  columns=genre_embed_cols)
                                  
playlist_genre_embeddings = pd.DataFrame(playlist_genre_embeddings.tolist(), 
                                        index=playlist_df.index,
                                        columns=genre_embed_cols)

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
threshold = 0.60 # Can probably lower this tbh
above_thresh_df = df[df["fit_probability"] >= threshold].copy()
below_thresh_df = df[df["fit_probability"] < threshold].copy()

# Save Filtered Songs
above_thresh_df.to_csv("C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\datasets\\aidan_above_threshold_xgb.csv", index=False)
below_thresh_df.to_csv("C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\datasets\\aidan_below_threshold_xgb.csv", index=False)

# Summary
print(f"Songs Retained: {len(above_thresh_df)} ({(len(above_thresh_df)/len(df))*100:.2f}%)")
print(f"Songs Discarded: {len(below_thresh_df)}")