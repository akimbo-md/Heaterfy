"""
Classification model for initial track filtering
Uses XGBoost to remove the TRASH 🗑️
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from ..embeddings import parse_genre_embedding

def run_classification_model(df, playlist_df, threshold=0.60):
    print("Running Classification Model...")
    
    feature_cols = [
        "Danceability", "Energy", "Tempo", "Valence", "Liveness", "Acousticness", 
        "Instrumentalness", "Speechiness", "Loudness"
    ]
    
    # Deal with genre embedding column
    df_genre_embeddings = df["genre_embedding"].apply(parse_genre_embedding)
    playlist_genre_embeddings = playlist_df["genre_embedding"].apply(parse_genre_embedding)
    genre_embed_cols = [f"genre_dim_{i}" for i in range(16)]
    
    # debug
    zeros_count = 0
    for emb in df_genre_embeddings[:10]:
        if np.all(emb == 0):
            zeros_count += 1
    
    # Expand the genre_embedding into separate columns (shouldn't have done this tbh)
    df_genre_embeddings = pd.DataFrame(df_genre_embeddings.tolist(), index=df.index, columns=genre_embed_cols)                                  
    playlist_genre_embeddings = pd.DataFrame(playlist_genre_embeddings.tolist(), index=playlist_df.index,columns=genre_embed_cols)
    
    df = df.drop(columns=["genre_embedding"]) 
    playlist_df = playlist_df.drop(columns=["genre_embedding"])
    df = pd.concat([df, df_genre_embeddings], axis=1)
    playlist_df = pd.concat([playlist_df, playlist_genre_embeddings], axis=1)
    feature_cols.extend(genre_embed_cols)
    # Clean column names
    df.columns = df.columns.astype(str)
    playlist_df.columns = playlist_df.columns.astype(str)
    
    # Drop missing values
    df = df.dropna(subset=feature_cols)
    playlist_df = playlist_df.dropna(subset=feature_cols)
    
    # Compute the centroid
    playlist_vector = playlist_df[feature_cols].mean(axis=0).values
    
    # Scale
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[feature_cols])
    playlist_vector_scaled = scaler.transform([playlist_vector])
    
    # thank you cosine similarity
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
    above_thresh_df = df[df["fit_probability"] >= threshold].copy()
    
    # Summary
    print(f"Classification Results:")
    print(f"Tracks Kept: {len(above_thresh_df)} ({(len(above_thresh_df)/len(df))*100:.2f}%)")
    print(f"Tracks Filtered: {len(df) - len(above_thresh_df)}")
    
    return above_thresh_df, playlist_df, feature_cols