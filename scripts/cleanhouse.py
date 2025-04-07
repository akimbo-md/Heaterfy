import pandas as pd
import streamlit as st

"""
This script loads a dataset of songs and allows users to filter songs by genre and artist then CLEAN HOUSE.
no diddy allowed.
"""

# Load the dataset
file_path = "C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\top_10000_1950-now.csv" # Change to the data to be cleaned

df = pd.read_csv(file_path)
print(df.columns)

# Ensure correct column names
genre_column = "Artist Genres" 
artist_column = "Artist Name(s)"

# Check if the columns exist
if genre_column not in df.columns or artist_column not in df.columns:
    raise KeyError(f"Columns '{genre_column}' or '{artist_column}' not found in the DataFrame")

# Convert genre column to list
def parse_genre(genre_str):
    return genre_str.split(',') if isinstance(genre_str, str) else []

df[genre_column] = df[genre_column].apply(parse_genre)

# Extract all unique genres
all_genres = set()
for genres in df[genre_column]:
    all_genres.update(genres)

all_genres = sorted(all_genres)

# Extract all unique artists
all_artists = sorted(df[artist_column].dropna().unique())

# Streamlit UI
st.title("Filter Songs by Genre and Artist")
selected_genres = st.multiselect("Select genres to remove:", all_genres)
selected_artists = st.multiselect("Select artists to remove:", all_artists)

# Apply genre filter
if selected_genres:
    df = df[~df[genre_column].apply(lambda genre_list: any(genre in genre_list for genre in selected_genres))]

# Apply artist filter
if selected_artists:
    df = df[~df[artist_column].isin(selected_artists)]

st.write("Filtered Data:", df)

# Download
st.download_button(
    label="Download Filtered Data as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_songs.csv",
    mime="text/csv"
)