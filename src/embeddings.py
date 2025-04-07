import os
import json
import numpy as np
import pandas as pd

def parse_genre_embedding(embedding_str):
    try:
        if isinstance(embedding_str, list):
            return np.array(embedding_str)
        cleaned = embedding_str.strip("[]").replace(",", " ")
        return np.fromstring(cleaned, sep=' ')
    except:
        return np.zeros(16)

def get_genre_list(genre_str):
    if isinstance(genre_str, str):
        return [g.strip().lower() for g in genre_str.split(",")]
    return []

def find_genre_embedding(genre, embeddings_dict):
    if not isinstance(genre, str) or not genre.strip():
        return None
        
    genre_lower = genre.lower().strip()
    genre_with_dash = genre_lower.replace(" ", "-")
    genre_without_dash = genre_lower.replace("-", " ")
    
    for variant in [genre_lower, genre_with_dash, genre_without_dash]:
        if variant in embeddings_dict:
            return embeddings_dict[variant]
            
    return None

def get_track_embedding(genre_list, embeddings_dict):
    if not isinstance(genre_list, list) or not genre_list:
        return np.zeros(16)
        
    valid_embeddings = []
    missing_genres = []
    
    for genre in genre_list:
        embedding = find_genre_embedding(genre, embeddings_dict)
        if embedding is not None:
            valid_embeddings.append(embedding)
        else:
            missing_genres.append(genre)
            
    if missing_genres and len(missing_genres) == len(genre_list):
        print(f"No embeddings found for: {', '.join(missing_genres)}")
            
    if not valid_embeddings:
        return np.zeros(16)
        
    return np.mean(valid_embeddings, axis=0)

def genre_embeddings(playlist_df):
    print("Checking playlist genre embeddings...")
    
    if 'genre_embedding' in playlist_df.columns and not playlist_df['genre_embedding'].isna().any():
        print("All playlist tracks already have genre embeddings")
        return playlist_df
        
    print("Processing genre embeddings...")
    
    if 'Genres' not in playlist_df.columns:
        print("No 'Genres' column found. Creating zeros.")
        playlist_df['genre_embedding'] = [np.zeros(16).tolist()] * len(playlist_df)
        return playlist_df
        
    try:
        project_root = os.path.dirname(os.path.dirname(__file__))
        embeddings_path = os.path.join(project_root, "json", "genre_embeddings.json")
        
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Genre embeddings file not found")
            
        with open(embeddings_path, "r") as f:
            embeddings_dict = json.load(f)
            
        print(f"Loaded {len(embeddings_dict)} genre embeddings")
        
        embeddings_dict = {k.lower().strip(): np.array(v) for k, v in embeddings_dict.items()}
        
        genres_list = playlist_df['Genres'].fillna("").apply(get_genre_list)
        
        embedding_column = []
        for i, genres in enumerate(genres_list):
            embedding = get_track_embedding(genres, embeddings_dict)
            embedding_column.append(embedding.tolist())
            
        playlist_df['genre_embedding'] = embedding_column
        print(f"Mapped genres for {len(playlist_df)} tracks")
        
    except Exception as e:
        print(f"Error processing embeddings: {str(e)}")
        playlist_df['genre_embedding'] = [np.zeros(16).tolist()] * len(playlist_df)
        
    return playlist_df