import pandas as pd
import numpy as np
import json
import os
from difflib import get_close_matches

"""
This script embeds genres in a dataset using precomputed genre embeddings.
Ensure that genre_embeddings.json and genre_seeds.json are present.

The script performs the following steps:
1. Load the dataset and genre embeddings
2. Clean the genre names
3. Find the closest matching Spotify genre for each genre in the dataset
4. Compute the track embedding using the genre embeddings (average of genre embeddings)
5. Save the updated dataset with genre embeddings
"""

csv_filename = "C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\spotify_rips\\afterhours.csv"

# Load the saved genre embeddings
with open("genre_embeddings.json", "r") as f:
    genre_embeddings = json.load(f)

# Convert embeddings back to NumPy arrays (ensure all keys are lowercase for uniformity)
genre_embeddings = {k.lower().strip().replace("»", ""): np.array(v) for k, v in genre_embeddings.items()}

# Load Spotify's official genre list (from genre_seeds.json)
with open("C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\genre_seeds.json", "r") as f:
    spotify_genres_list = json.load(f)["genres"]

# Convert genre lists to lowercase for uniformity (ensures we reference correctly)
spotify_genres_set = set(g.lower().strip() for g in spotify_genres_list)  # {lowercased_name}

# Common generic words to remove
GENERIC_WORDS = {"music", "scene", "band"}

def clean_genre_name(genre):
    """Removes generic words like 'music' and normalizes dashes."""
    words = genre.lower().strip().replace("»", "").split()
    cleaned_words = [word for word in words if word not in GENERIC_WORDS]
    cleaned_genre = " ".join(cleaned_words)
    return cleaned_genre

def find_closest_spotify_genre(tag):
    """Finds the closest matching Spotify genre, ensuring correct lookup in genre_embeddings.json."""
    if not isinstance(tag, str):
        return None
    
    if not isinstance(tag, str) or tag.strip() == "":  # 🛑 Prevent empty string from being mapped
        print("🛑 Empty genre detected. Returning None.")
        return None

    tag_clean = clean_genre_name(tag)
    tag_with_dash = tag_clean.replace(" ", "-")  # Convert to dashed format
    tag_without_dash = tag_clean.replace("-", " ")  # Convert to space format

    print(f"🔍 Searching match for: '{tag_clean}' (also checking '{tag_with_dash}' & '{tag_without_dash}')")

    # 1️⃣ **Check exact match in genre_embeddings.json**
    for variant in [tag_clean, tag_with_dash, tag_without_dash]:
        if variant in genre_embeddings:
            print(f"✅ Exact embedding match found: '{variant}'")
            return variant

    # 2️⃣ **Check in spotify_genres_set, then verify in genre_embeddings**
    for variant in [tag_clean, tag_with_dash, tag_without_dash]:
        if variant in spotify_genres_set and variant in genre_embeddings:
            print(f"✅ Exact Spotify genre match found in embeddings: '{variant}'")
            return variant

    # 3️⃣ **Find closest subgenre match**
    subgenre_matches = [g for g in genre_embeddings if tag_clean in g]
    if subgenre_matches:
        best_subgenre = subgenre_matches[0]
        print(f"🎯 Using closest subgenre match: '{tag_clean}' → '{best_subgenre}'")
        return best_subgenre

    # 4️⃣ **Fuzzy matching as a last resort**
    closest_matches = get_close_matches(tag_clean, spotify_genres_set, n=1, cutoff=0.4)
    if closest_matches:
        match = closest_matches[0]
        if match in genre_embeddings:
            print(f"🤖 Fuzzy match found: '{tag_clean}' → '{match}'")
            return match
        else:
            print(f"⚠️ Fuzzy match '{match}' found, but embedding is missing")

    print(f"❌ No match found for: '{tag_clean}'")
    return None

def get_genre_list(genre_str):
    """Processes a genre string into a cleaned list of genres."""
    if isinstance(genre_str, str):
        return [g.strip().lower().replace("»", "") for g in genre_str.split(",")]
    return []  # Return an empty list instead of NaN

def get_track_embedding(genre_list):
    """Computes track embedding, ensuring only valid genres are included."""
    
    if not isinstance(genre_list, list):
        genre_list = []
        
    if not isinstance(genre_list, list) or not genre_list:
        print("🛑 No genre embeddings found. Returning zero vector.\n")
        return np.zeros(16)  # 16D zero vector

    embs = []
    print("\n🎵 Processing track genres:", genre_list)

    # Process each genre in the list
    for g in genre_list:
        mapped_genre = find_closest_spotify_genre(g)
        if mapped_genre:
            mapped_genre = mapped_genre.lower().strip()
            if mapped_genre in genre_embeddings:
                print(f"🔗 Using embedding for: '{mapped_genre}'")
                embs.append(genre_embeddings[mapped_genre])
            else:
                print(f"⚠️ Embedding missing for matched genre: '{mapped_genre}'")
        else:
            print(f"⚠️ Skipping genre: '{g}' (no valid match)")
    # No embeddings found
    if not embs:
        print(f"🛑 No genre embeddings found. Returning zero vector.\n")
        return np.zeros(16)  # Ensure it is a 16D zero vector

    track_embedding = np.mean(embs, axis=0)
    print(f"📊 Final track embedding: {track_embedding}")

    return track_embedding


# Load the dataset
df = pd.read_csv(csv_filename)

df['genres_list'] = df['Genres'].fillna("").apply(get_genre_list)
df['genre_embedding'] = df['genres_list'].apply(get_track_embedding)

# Save the updated dataset
base, ext = os.path.splitext(csv_filename)
output_filename = f"{base}_embedded{ext}"
df.to_csv(output_filename, index=False)

print(f"\n✅ Updated dataset saved to {output_filename}")
